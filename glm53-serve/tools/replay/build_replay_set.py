#!/usr/bin/env python3
"""Build a token-level replay set from recorded GLM-5.3 OpenCode sessions (ATP harness http.jsonl captures).

Each ATP attempt dir holds http.jsonl: one JSON event per line (api_gate_wait_started, request, response, retry,
rate_limit_wait, model_call_completed). A *model call* (model_call_id) is one OpenCode -> LLM call; it may take
several HTTP attempts (429 retries). For every model call that got an HTTP 200 we take
  - the exact chat payload sent to the hub (messages, tools, reasoning_effort, sampling params),
  - usage from the response (prompt_tokens, completion_tokens incl. reasoning, cached_tokens, MTP accept/reject),
  - timestamps (first event of the call, request/response of the successful attempt, model_call_completed),
render the payload with the GLM-5.3 chat template exactly the way vLLM's /v1/chat/completions does
(tool-call arguments JSON-decoded to dicts, reasoning_effort passed to the template, add_generation_prompt=True,
past `reasoning_content` kept), tokenize with add_special_tokens=False, and store the prompt as a delta against
the previous call's prompt (shared prefix length + new token ids), so multi-turn prefix structure is preserved.

Output: one gzipped JSONL file. Line 1 = {"type": "header", ...build/render config + summary...},
then one {"type": "session", ...} line per session (calls in issue order). Token deltas are base64 of
little-endian uint32. A <out>.summary.json with the same summary is written next to it.

Run with any python that has transformers (or tokenizers + jinja2 for --renderer builtin), e.g. the private venv:
  tools/replay/.venv/bin/python build_replay_set.py \
     --source deepswe=/lustre/.../teacher-deepswe-20261030/dataset/glm.jsonl.gz --n-sessions 64 \
     --out data/deepswe_glm_64.jsonl.gz
Read-only on all inputs.
"""
import argparse
import array
import base64
import collections
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

DEFAULT_MODEL_DIR = "/lustre/fsw/portfolios/nemotron/users/hengyuf/models/GLM-5.3"
ROLE_TOKENS = ("<|system|>", "<|user|>", "<|assistant|>", "<|observation|>")
BLOCK = 64  # GLM-5.3 DSA attention backends force KV block size 64 in vLLM (hsg-ref/notes/NOTES.md)


# ----------------------------------------------------------------------------------------------------------------
# token-id packing (shared with replay_load.py)
# ----------------------------------------------------------------------------------------------------------------

def pack_ids(ids):
    a = array.array("I", ids)
    if a.itemsize != 4:
        raise RuntimeError("array('I') itemsize != 4 on this platform")
    if sys.byteorder != "little":
        a.byteswap()
    return base64.b64encode(a.tobytes()).decode("ascii")


def ids_sha1(ids):
    a = array.array("I", ids)
    if sys.byteorder != "little":
        a.byteswap()
    return hashlib.sha1(a.tobytes()).hexdigest()


# ----------------------------------------------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------------------------------------------

class Renderer:
    """Chat template + tokenizer. 'transformers' = tokenizer.apply_chat_template (what vLLM calls);
    'builtin' = the same Jinja environment transformers builds (sandboxed, trim/lstrip blocks, loopcontrols,
    non-escaping tojson) driven directly with jinja2 + tokenizers (no transformers needed)."""

    def __init__(self, model_dir, renderer="auto", chat_template_path=None):
        self.model_dir = model_dir
        tpath = chat_template_path or os.path.join(model_dir, "chat_template.jinja")
        with open(tpath) as f:
            self.template_text = f.read()
        self.template_path = tpath
        self.template_sha256 = hashlib.sha256(self.template_text.encode()).hexdigest()
        with open(os.path.join(model_dir, "tokenizer.json"), "rb") as f:
            self.tokenizer_sha256 = hashlib.sha256(f.read()).hexdigest()
        if renderer == "auto":
            try:
                import transformers  # noqa: F401
                renderer = "transformers"
            except Exception:
                renderer = "builtin"
        self.kind = renderer
        if renderer == "transformers":
            import transformers
            from transformers import AutoTokenizer
            self.version = "transformers " + transformers.__version__
            self.tok = AutoTokenizer.from_pretrained(model_dir)
            self._render = self._render_tf
            self._encode = lambda text: self.tok(text, add_special_tokens=False)["input_ids"]
            vocab = self.tok.get_added_vocab()
        elif renderer == "builtin":
            import jinja2
            import jinja2.ext
            import tokenizers
            from jinja2.sandbox import ImmutableSandboxedEnvironment
            self.version = "builtin jinja2 %s tokenizers %s" % (jinja2.__version__, tokenizers.__version__)
            self.tk = tokenizers.Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))

            def raise_exception(message):
                raise jinja2.exceptions.TemplateError(message)

            def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
                return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
                                  sort_keys=sort_keys)

            def strftime_now(fmt):
                return time.strftime(fmt)

            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                extensions=[jinja2.ext.loopcontrols])
            env.filters["tojson"] = tojson
            env.globals["raise_exception"] = raise_exception
            env.globals["strftime_now"] = strftime_now
            self.jt = env.from_string(self.template_text)
            self._render = self._render_builtin
            self._encode = lambda text: self.tk.encode(text, add_special_tokens=False).ids
            vocab = {t.content: i for i, t in self.tk.get_added_tokens_decoder().items()}
        else:
            raise SystemExit("unknown renderer %r" % renderer)
        self.added_ids = set(vocab.values())
        self.role_ids = set(vocab[t] for t in ROLE_TOKENS if t in vocab)
        self.id_to_added = {i: t for t, i in vocab.items()}

    def _render_tf(self, messages, tools, kwargs):
        return self.tok.apply_chat_template(messages, tools=tools, chat_template=self.template_text,
                                            tokenize=False, add_generation_prompt=True, **kwargs)

    def _render_builtin(self, messages, tools, kwargs):
        return self.jt.render(messages=messages, tools=tools, documents=None, add_generation_prompt=True, **kwargs)

    def render(self, messages, tools, kwargs):
        return self._render(messages, tools, kwargs)

    def encode(self, text):
        return list(self._encode(text))


def vllm_like_messages(messages, history_reasoning="keep"):
    """Mimic vLLM 0.29's chat preprocessing for the message shapes OpenCode/ATP sends:
    openai/chat_completion/protocol.py _normalize_messages_before (reasoning_content -> reasoning),
    chat_utils.py _parse_chat_message_content (assistant reasoning -> reasoning + reasoning_content, tool content
    parts joined with newlines) and _postprocess_messages (tool_call arguments JSON-decoded to dicts; invalid or
    non-object JSON -> {}; empty tool_calls dropped). history_reasoning='drop' removes past reasoning (what-if)."""
    out = []
    for m in messages:
        role = m.get("role")
        nm = {"role": role}
        c = m.get("content")
        if c is None:
            c = ""
        elif isinstance(c, list):
            texts = [p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"]
            if all(isinstance(p, dict) and p.get("type") == "text" for p in c):
                c = "\n".join(texts) if role == "tool" else c
        nm["content"] = c
        if role == "assistant":
            tcs = m.get("tool_calls")
            if isinstance(tcs, list) and tcs:
                ntc = []
                for tc in tcs:
                    tc = dict(tc)
                    fn = dict(tc.get("function") or {})
                    a = fn.get("arguments")
                    if isinstance(a, str):
                        try:
                            a = json.loads(a) if a else {}
                        except Exception:
                            a = {}
                    if not isinstance(a, dict):
                        a = {}
                    fn["arguments"] = a
                    tc["function"] = fn
                    ntc.append(tc)
                nm["tool_calls"] = ntc
            r = m.get("reasoning")          # vLLM keeps `reasoning` and only falls back to `reasoning_content`
            if r is None:
                r = m.get("reasoning_content")
            if history_reasoning == "keep" and r is not None:
                nm["reasoning_content"] = r
                nm["reasoning"] = r
        elif role == "tool":
            if "tool_call_id" in m:
                nm["tool_call_id"] = m["tool_call_id"]
        if isinstance(m.get("name"), str):
            nm["name"] = m["name"]
        out.append(nm)
    return out


# ----------------------------------------------------------------------------------------------------------------
# http.jsonl parsing
# ----------------------------------------------------------------------------------------------------------------

def iter_model_calls(http_path, stats):
    """Yield successful model calls in completion order. Memory: one pending request payload at a time."""
    pending = {}           # (model_call_id, http_attempt) -> request record
    info = {}              # model_call_id -> dict(t_first, n_http, n_429, waits)
    done_ts = {}           # model_call_id -> model_call_completed ts
    ready = []             # successful calls waiting for model_call_completed (normally 0/1)
    with open(http_path, "rb") as f:
        for raw in f:
            try:
                r = json.loads(raw)
            except Exception:
                stats["bad_json_lines"] += 1
                continue
            ev = r.get("event")
            mc = r.get("model_call_id")
            ts = r.get("timestamp")
            if mc is None:
                continue
            d = info.get(mc)
            if d is None:
                d = info[mc] = {"t_first": ts, "n_http": 0, "status": collections.Counter(), "rl_wait_s": 0.0}
            if ts is not None and (d["t_first"] is None or ts < d["t_first"]):
                d["t_first"] = ts
            if ev == "request":
                pending[(mc, r.get("http_attempt"))] = r
                d["n_http"] += 1
            elif ev == "response":
                st = r.get("status")
                d["status"][st] += 1
                q = pending.pop((mc, r.get("http_attempt")), None)
                if st == 200 and q is not None:
                    try:
                        body = json.loads(r["body"]) if isinstance(r.get("body"), str) else r.get("body")
                    except Exception:
                        stats["bad_200_body"] += 1
                        continue
                    if not isinstance(body, dict) or not body.get("choices"):
                        stats["bad_200_body"] += 1
                        continue
                    ready.append(dict(mc=mc, payload=q["payload"], body=body, t_req=q.get("timestamp"),
                                      t_resp=ts, info=d))
            elif ev == "rate_limit_wait":
                d["rl_wait_s"] += float(r.get("wait_seconds") or 0.0)
            elif ev == "model_call_completed":
                done_ts[mc] = ts
                keep = []
                for c in ready:
                    if c["mc"] == mc:
                        c["t_done"] = ts
                        yield c
                    else:
                        keep.append(c)
                ready = keep
    for c in ready:  # successful but no completion event (session killed right after)
        c["t_done"] = c["t_resp"]
        stats["calls_without_completed_event"] += 1
        yield c
    stats["model_calls_seen"] += len(info)
    stats["model_calls_failed"] += sum(1 for d in info.values() if not d["status"].get(200))
    stats["http_429"] += sum(d["status"].get(429, 0) for d in info.values())


# ----------------------------------------------------------------------------------------------------------------
# per-session worker
# ----------------------------------------------------------------------------------------------------------------

_R = None
_R2 = None
_ARGS = None


def _init_worker(args):
    global _R, _R2, _ARGS
    _ARGS = args
    _R = Renderer(args.model_dir, args.renderer, args.chat_template)
    if args.cross_check_template:
        _R2 = Renderer(args.model_dir, _R.kind, args.cross_check_template)


def lcp(a, b):
    n = min(len(a), len(b))
    # coarse jumps then fine
    i = 0
    step = 4096
    while i + step <= n and a[i:i + step] == b[i:i + step]:
        i += step
    while i < n and a[i] == b[i]:
        i += 1
    return i


def process_session(task):
    sel, = task
    args = _ARGS
    R = _R
    t_start = time.time()
    stats = collections.Counter()
    http_path = os.path.join(sel["attempt_dir"], "http.jsonl")
    calls = []
    prev_text = None
    prev_ids = None
    prev_last_is_added = False
    t0 = None
    tmpl_kwargs = dict(args.template_kwargs)
    for k, c in enumerate(iter_model_calls(http_path, stats)):
        pl = c["payload"]
        body = c["body"]
        kw = dict(tmpl_kwargs)
        if "reasoning_effort" in pl and "reasoning_effort" not in kw:
            kw["reasoning_effort"] = pl["reasoning_effort"]
        msgs = vllm_like_messages(pl.get("messages") or [], args.history_reasoning)
        tools = pl.get("tools") or None
        text = R.render(msgs, tools, kw)
        if _R2 is not None:
            text2 = _R2.render(msgs, tools, kw)
            stats["xcheck_calls"] += 1
            if text2 != text:
                stats["xcheck_text_differs"] += 1
        # incremental tokenization: exact when the previous prompt text ends with an added token (the generation
        # prompt '<|assistant|><think>'), because HF tokenizers split on added tokens before pre-tokenization.
        incremental = prev_text is not None and prev_last_is_added and text.startswith(prev_text)
        if incremental:
            ids = prev_ids + R.encode(text[len(prev_text):])
            stats["tok_incremental"] += 1
        else:
            ids = R.encode(text)
            stats["tok_full"] += 1
        if incremental and args.verify_every and (k % args.verify_every == 0):
            full = R.encode(text)
            stats["tok_verified"] += 1
            if full != ids:
                stats["tok_verify_mismatch"] += 1
                ids = full
        if prev_ids is None:
            shared = 0
        elif incremental:
            shared = len(prev_ids)
        else:
            shared = lcp(prev_ids, ids)
        delta = ids[shared:]
        # assistant span of the previous call at the head of this delta (for --prefix-mode splice):
        # tokens before the first role token (<|observation|>, <|user|>, ...).
        splice_len = None
        if prev_ids is not None and shared == len(prev_ids):
            splice_len = len(delta)
            for j, t in enumerate(delta):
                if t in R.role_ids:
                    splice_len = j
                    break
        usage = body.get("usage") or {}
        ctd = usage.get("completion_tokens_details") or {}
        ptd = usage.get("prompt_tokens_details") or {}
        ch = (body.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        rc = msg.get("reasoning_content") or msg.get("reasoning") or ""
        if t0 is None:
            t0 = c["info"]["t_first"]
        out_len = usage.get("completion_tokens")
        rec = {
            "k": k,
            "prompt_len": len(ids),
            "shared_prefix_len": shared,
            "delta_ids": pack_ids(delta),
            "prompt_sha1": ids_sha1(ids),
            "output_len": int(out_len) if out_len is not None else None,
            "reasoning_tokens": ctd.get("reasoning_tokens"),
            "finish_reason": ch.get("finish_reason"),
            "n_tool_calls": len(msg.get("tool_calls") or []),
            "reasoning_chars": len(rc),
            "content_chars": len(msg.get("content") or ""),
            "splice_len": splice_len,
            "t_first": round(c["info"]["t_first"] - t0, 3),
            "t_req": round(c["t_req"] - t0, 3) if c.get("t_req") is not None else None,
            "t_resp": round(c["t_resp"] - t0, 3) if c.get("t_resp") is not None else None,
            "t_done": round(c["t_done"] - t0, 3) if c.get("t_done") is not None else None,
            "think_after_s": None,
            "hub": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": out_len,
                "cached_tokens": ptd.get("cached_tokens"),
                "accepted_prediction_tokens": ctd.get("accepted_prediction_tokens"),
                "rejected_prediction_tokens": ctd.get("rejected_prediction_tokens"),
                "latency_s": round(c["t_resp"] - c["t_req"], 3) if c.get("t_req") and c.get("t_resp") else None,
                "http_attempts": c["info"]["n_http"],
                "rate_limit_wait_s": round(c["info"]["rl_wait_s"], 3),
            },
            "sampling": {k2: pl.get(k2) for k2 in ("temperature", "top_p", "max_tokens") if k2 in pl},
        }
        calls.append(rec)
        prev_text, prev_ids = text, ids
        prev_last_is_added = bool(ids) and ids[-1] in R.added_ids and text.endswith(R.id_to_added.get(ids[-1], "\0"))
        if args.max_calls and len(calls) >= args.max_calls:
            stats["truncated_max_calls"] += 1
            break
    # think time: recorded gap between this call's completion and the next call's first event
    for a, b in zip(calls, calls[1:]):
        if a["t_done"] is not None and b["t_first"] is not None:
            a["think_after_s"] = round(max(b["t_first"] - a["t_done"], 0.0), 3)
            if b["t_first"] < a["t_done"]:
                stats["overlapping_calls"] += 1
    # drop per-call sampling dict when constant across the session (keep in session header)
    samp = None
    if calls:
        samp = calls[0]["sampling"]
        if all(c["sampling"] == samp for c in calls):
            for c in calls:
                del c["sampling"]
    sess = {
        "type": "session",
        "session_id": sel["session_id"],
        "source": sel["source"],
        "task_id": sel.get("task_id"),
        "status": sel.get("status"),
        "attempt_dir": sel["attempt_dir"],
        "meta": sel.get("meta") or {},
        "n_calls": len(calls),
        "sampling": samp,
        "recorded_wall_s": round(calls[-1]["t_done"] - calls[0]["t_first"], 3) if calls else 0.0,
        "calls": calls,
    }
    stats["build_s"] = round(time.time() - t_start, 1)
    return sess, dict(stats)


# ----------------------------------------------------------------------------------------------------------------
# selection + summary
# ----------------------------------------------------------------------------------------------------------------

def read_rows(path):
    op = gzip.open if path.endswith(".gz") else open
    rows = []
    with op(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for big in ("output", "patch", "messages", "input"):
                r.pop(big, None)
            rows.append(r)
    return rows


def select_sessions(source_name, rows, n, seed, statuses, one_per_task_first):
    ok = []
    for r in rows:
        ad = r.get("attempt_dir")
        if not ad or not os.path.exists(os.path.join(ad, "http.jsonl")):
            continue
        if statuses and r.get("status") not in statuses:
            continue
        ok.append(r)
    rng = random.Random("%s:%s" % (seed, source_name))
    ok.sort(key=lambda r: r["attempt_dir"])
    rng.shuffle(ok)
    if one_per_task_first:
        seen, first, rest = set(), [], []
        for r in ok:
            (rest if r.get("task_id") in seen else first).append(r)
            seen.add(r.get("task_id"))
        ok = first + rest
    chosen = ok[:n] if n > 0 else ok
    sel = []
    for r in chosen:
        ad = r["attempt_dir"].rstrip("/")
        parts = ad.split("/")
        sid = "%s:%s" % (source_name, "/".join(parts[-6:]))
        meta = {k: r.get(k) for k in ("slot", "repeat", "language", "category", "campaign", "run", "compactions",
                                       "reward", "f2p", "p2p", "hub_calls_ok", "prompt_tokens", "cached_tokens",
                                       "completion_tokens", "agent_wall_s", "turns") if r.get(k) is not None}
        sel.append({"source": source_name, "attempt_dir": ad, "session_id": sid, "task_id": r.get("task_id"),
                    "status": r.get("status"), "meta": meta})
    return sel, len(ok)


def pct(vals, qs=(0, 10, 50, 90, 99, 100)):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return {}
    out = {}
    for q in qs:
        if q == 0:
            out["min"] = v[0]
        elif q == 100:
            out["max"] = v[-1]
        else:
            i = min(len(v) - 1, max(0, int(math.ceil(q / 100.0 * len(v))) - 1))
            out["p%d" % q] = v[i]
    out["mean"] = round(sum(v) / len(v), 3)
    out["n"] = len(v)
    return out


def summarize(sessions, block=BLOCK):
    calls = [c for s in sessions for c in s["calls"]]
    out = {"sessions": len(sessions), "calls": len(calls)}
    out["calls_per_session"] = pct([s["n_calls"] for s in sessions])
    out["prompt_len"] = pct([c["prompt_len"] for c in calls])
    out["new_tokens_per_call"] = pct([c["prompt_len"] - c["shared_prefix_len"] for c in calls])
    out["output_len"] = pct([c["output_len"] for c in calls])
    out["reasoning_tokens"] = pct([c["reasoning_tokens"] for c in calls])
    out["think_after_s"] = pct([c["think_after_s"] for c in calls])
    out["final_context_len"] = pct([max(c["prompt_len"] + (c["output_len"] or 0) for c in s["calls"])
                                    for s in sessions if s["calls"]])
    out["recorded_session_wall_s"] = pct([s["recorded_wall_s"] for s in sessions])
    out["hub_latency_s"] = pct([c["hub"]["latency_s"] for c in calls])
    out["hub_decode_tok_s_lower_bound"] = pct([c["output_len"] / c["hub"]["latency_s"] for c in calls
                                               if c["hub"]["latency_s"] and c["output_len"] and c["output_len"] >= 200])
    tot_p = sum(c["prompt_len"] for c in calls)
    tot_o = sum(c["output_len"] or 0 for c in calls)
    tot_new = sum(c["prompt_len"] - c["shared_prefix_len"] for c in calls)
    out["totals"] = {"prompt_tokens": tot_p, "output_tokens": tot_o, "new_prompt_tokens_vs_prev_call": tot_new,
                     "think_s": round(sum(c["think_after_s"] or 0 for c in calls), 1),
                     "hub_latency_s": round(sum(c["hub"]["latency_s"] or 0 for c in calls), 1)}
    # think share of recorded session time
    rw = sum(s["recorded_wall_s"] for s in sessions)
    out["think_share_of_recorded_wall"] = round(out["totals"]["think_s"] / rw, 4) if rw else None
    # rendering validation vs hub prompt_tokens
    diffs = collections.Counter()
    for c in calls:
        hp = c["hub"]["prompt_tokens"]
        if hp is not None:
            diffs[hp - c["prompt_len"]] += 1
    top = diffs.most_common(8)
    out["hub_prompt_tokens_minus_rendered"] = {"top": [[d, n] for d, n in top], "n": sum(diffs.values()),
                                               "within_1": sum(n for d, n in diffs.items() if abs(d) <= 1),
                                               "within_16": sum(n for d, n in diffs.items() if abs(d) <= 16)}
    # offline prefix-cache potential: sticky routing, infinite cache, vLLM block granularity (hit <= prompt-1)
    hit_rec = hit_spl = 0
    prev_out = None
    for s in sessions:
        prev_out = None
        for c in s["calls"]:
            P = c["prompt_len"]
            base = c["shared_prefix_len"]
            hit_rec += (min(base, P - 1) // block) * block
            spl = base
            if c["splice_len"] is not None and prev_out:
                spl = base + min(c["splice_len"] + 1, prev_out)
            hit_spl += (min(spl, P - 1) // block) * block
            prev_out = c["output_len"]
    out["offline_prefix_hit_frac"] = {
        "recorded_mode_sticky": round(hit_rec / tot_p, 4) if tot_p else None,
        "splice_mode_sticky": round(hit_spl / tot_p, 4) if tot_p else None,
        "hub_reported_cached_frac": round(sum(c["hub"]["cached_tokens"] or 0 for c in calls) /
                                          max(1, sum(c["hub"]["prompt_tokens"] or 0 for c in calls)), 4),
        "block_size": block,
    }
    # does the hub cache cover the previous call's generated tokens? (P/D or re-render mismatch => no)
    hub_cov = collections.Counter()
    span = collections.Counter()
    for s in sessions:
        prev = None
        for c in s["calls"]:
            hc = c["hub"]["cached_tokens"]
            if prev is not None and c["splice_len"] is not None and prev["output_len"] is not None:
                d = prev["output_len"] - (c["splice_len"] + 1)
                span["exact" if d == 0 else ("within_2" if abs(d) <= 2 else ("output_longer" if d > 0 else "span_longer"))] += 1
            if prev is not None and hc is not None and c["shared_prefix_len"] == prev["prompt_len"]:
                pp = prev["hub"]["prompt_tokens"] or prev["prompt_len"]
                po = prev["output_len"] or 0
                if po < 2 * block:
                    hub_cov["ambiguous(prev_output<2_blocks)"] += 1
                elif hc >= pp + po - 2 * block:
                    hub_cov["covers_prev_prompt_and_output"] += 1
                elif hc >= pp - 1 - block:
                    hub_cov["covers_prev_prompt_only"] += 1
                else:
                    hub_cov["less_than_prev_prompt(miss/evicted/other replica)"] += 1
            prev = c
    out["hub_cache_vs_prev_call"] = dict(hub_cov)
    # re-render identity evidence: recorded completion_tokens(k-1) vs rendered assistant span(k) + stop/role token
    out["prev_output_len_vs_rendered_span_plus_1"] = dict(span)
    acc = sum(c["hub"]["accepted_prediction_tokens"] or 0 for c in calls)
    rej = sum(c["hub"]["rejected_prediction_tokens"] or 0 for c in calls)
    out["hub_spec_decode"] = {"accepted_prediction_tokens": acc, "rejected_prediction_tokens": rej,
                              "draft_acceptance": round(acc / (acc + rej), 4) if acc + rej else None}
    out["finish_reason"] = dict(collections.Counter(c["finish_reason"] for c in calls))
    out["non_prefix_calls"] = sum(1 for s in sessions for c in s["calls"][1:] if c["splice_len"] is None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True,
                    help="NAME=ROWS where ROWS is a .jsonl[.gz] with one row per session carrying attempt_dir "
                         "(+ task_id/status). Repeatable; --n-sessions applies per source.")
    ap.add_argument("--n-sessions", type=int, default=64, help="sessions per source (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--status", default="pass,fail", help="comma list of row statuses to keep ('' = all)")
    ap.add_argument("--no-one-per-task-first", action="store_true",
                    help="plain seeded random sample instead of one session per task first")
    ap.add_argument("--max-calls", type=int, default=0, help="cap calls per session (testing)")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="tokenizer.json + chat_template.jinja")
    ap.add_argument("--chat-template", default=None, help="override template file (default <model-dir>/chat_template.jinja)")
    ap.add_argument("--cross-check-template", default=None,
                    help="second template file; every call is also rendered with it and text differences counted")
    ap.add_argument("--renderer", choices=["auto", "transformers", "builtin"], default="auto")
    ap.add_argument("--history-reasoning", choices=["keep", "drop"], default="keep",
                    help="keep past assistant reasoning in prompts (default; matches the hub's prompt_tokens and "
                         "vLLM 0.29, which renames reasoning_content->reasoning and renders it) or drop it "
                         "(what-if: a client/harness that strips history reasoning)")
    ap.add_argument("--template-kwargs", default="{}", help="extra JSON kwargs for the template, e.g. '{\"clear_thinking\": true}'")
    ap.add_argument("--verify-every", type=int, default=20,
                    help="re-tokenize the full prompt every N calls and compare with the incremental ids (0=never)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True, help="output .jsonl.gz")
    args = ap.parse_args()
    args.template_kwargs = json.loads(args.template_kwargs)
    statuses = set(s for s in args.status.split(",") if s)

    selections = []
    pool_info = {}
    for spec in args.source:
        name, _, path = spec.partition("=")
        rows = read_rows(path)
        sel, avail = select_sessions(name, rows, args.n_sessions, args.seed, statuses, not args.no_one_per_task_first)
        pool_info[name] = {"rows_file": path, "rows": len(rows), "eligible": avail, "selected": len(sel)}
        selections.extend(sel)
        print("source %s: %d rows, %d eligible, %d selected" % (name, len(rows), avail, len(sel)), flush=True)

    r0 = Renderer(args.model_dir, args.renderer, args.chat_template)
    print("renderer:", r0.kind, r0.version, "template", r0.template_path, r0.template_sha256[:16], flush=True)

    t0 = time.time()
    results = [None] * len(selections)
    agg = collections.Counter()
    with mp.get_context("fork").Pool(processes=max(1, args.workers), initializer=_init_worker,
                                     initargs=(args,)) as pool:
        for i, (sess, st) in enumerate(pool.imap(process_session, [(s,) for s in selections])):
            results[i] = sess
            for k, v in st.items():
                if k != "build_s":
                    agg[k] += v
            print("[%4.0fs] %3d/%d %s calls=%d build=%.0fs %s" % (
                time.time() - t0, i + 1, len(selections), sess["session_id"], sess["n_calls"], st.get("build_s", 0),
                {k: v for k, v in st.items() if k in ("tok_verify_mismatch", "xcheck_text_differs", "overlapping_calls",
                                                      "model_calls_failed")}), flush=True)
    sessions = [s for s in results if s and s["n_calls"] > 0]
    summary = {"overall": summarize(sessions)}
    for name in pool_info:
        ss = [s for s in sessions if s["source"] == name]
        if ss and len(pool_info) > 1:
            summary[name] = summarize(ss)
    header = {
        "type": "header",
        "format": "glm53-replay-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "build_args": {k: v for k, v in vars(args).items()},
        "render": {"renderer": r0.kind, "version": r0.version, "model_dir": args.model_dir,
                   "chat_template": r0.template_path, "chat_template_sha256": r0.template_sha256,
                   "tokenizer_sha256": r0.tokenizer_sha256, "add_generation_prompt": True,
                   "add_special_tokens": False, "history_reasoning": args.history_reasoning,
                   "template_kwargs": args.template_kwargs,
                   "per_call_template_kwargs_from_payload": ["reasoning_effort"]},
        "token_encoding": "delta_ids = base64(little-endian uint32); prompt_k = prompt_{k-1}[:shared_prefix_len] + delta",
        "sources": pool_info,
        "build_stats": dict(agg),
        "summary": summary,
        "build_wall_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with gzip.open(tmp, "wt", compresslevel=6) as f:
        f.write(json.dumps(header) + "\n")
        for s in sessions:
            f.write(json.dumps(s, separators=(",", ":")) + "\n")
    os.replace(tmp, args.out)
    sp = args.out[:-len(".jsonl.gz")] if args.out.endswith(".jsonl.gz") else args.out
    with open(sp + ".summary.json", "w") as f:
        json.dump({k: header[k] for k in ("format", "created", "render", "sources", "build_stats", "summary",
                                          "build_wall_s")}, f, indent=1)
    print(json.dumps(summary["overall"], indent=1))
    print("wrote %s (%.1f MB) in %.0fs" % (args.out, os.path.getsize(args.out) / 1e6, time.time() - t0))


if __name__ == "__main__":
    main()
