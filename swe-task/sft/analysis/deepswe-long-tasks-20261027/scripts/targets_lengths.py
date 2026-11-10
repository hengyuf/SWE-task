#!/usr/bin/env python3
"""Per-rollout context-length metrics for the DeepSWE "targets" dataset (sc14 + Kimi K3, 5 rollouts each, 113 tasks).
READ-ONLY on all inputs. Writes only under OUT.

Input: TARGETS/records/<task_id>.jsonl (5 sc14 records then 5 kimi records per task; see TARGETS/sources.json).

Length metric (same quantity as the SFT corpus' full rendered conversation length = prompt + completion of a call):
  ctx(call)     = prompt tokens + completion tokens of one model call of the MAIN session
  peak_ctx      = max ctx over the main session's assistant messages / calls (compaction summaries included)
  n_calls       = main-session assistant messages / calls that are not compaction summaries
  n_compactions = compaction summaries in the main session

Per-call token source, per model (the most accurate one available; metric_method says which was used):
  sc14  record.source.opencode_results_fpath -> OpenCode export.json (eval run base-step0-rep5-out131k);
        ctx = tokens.input + tokens.cache.read + tokens.cache.write + tokens.output  (sc14 tokenizer, vLLM usage).
  kimi  The attempt dirs (export.json / http.jsonl) are not readable (permission denied). The same rows of Baihe's
        rollouts.jsonl.gz (sha256 cf117f52..., identical to record.source.gz_sha256) carry ng_model_call_capture,
        the per-HTTP-call usage of the hosted Kimi API; it was extracted once, unmodified, to
        deepswe-partial-analysis/verify-raw/pass1/calls.jsonl (li = line in the gz = record.source.line).
        ctx = usage.prompt_tokens + usage.completion_tokens (Kimi tokenizer; prompt includes cached tokens).
        Main session: calls are threaded into sessions with the exact chat-format rule
        n_msgs(next) = n_msgs(prev) + 1 + n_tool_calls(prev) (holds for 39,764/39,764 transitions of single-session
        rollouts); a call with n_msgs == 2 starts a new (subagent) session; the session of the first call is main.
        Kimi requests never carry earlier turns' reasoning (hist_reasoning_fields == 0 on every call), so
        full_conv_len_native_est = peak_ctx + reasoning tokens of the earlier main-session calls approximates the
        full rendered conversation incl. every turn's reasoning (Kimi tokenizer).
Extra, both models: items_tok_sc14 = the record's Responses-API output items tokenized with the sc14 tokenizer
  (reasoning / message / function_call name+args / function_call_output), items_tok_est_sc14 = sum + 9,000 for
  system prompt + tools; *_no_reasoning drops reasoning items. Fallback length when no per-call source exists.

usage: targets_lengths.py [--workers 8] [--limit-tasks N] [--tasks a,b,c]
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

TARGETS = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/data/targets/sc14-out131k-kimi"
CALLS = "/lustre/fsw/portfolios/nemotron/users/hengyuf/deepswe-partial-analysis/verify-raw/pass1/calls.jsonl"
CALLS_GZ_SHA = "cf117f52e1d2c7e86698746ccb0fdd8741cf4335e467b9be603371498bbf2146"
OUT = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/analysis/deepswe-long-tasks-20261027"
OUT_ROWS = os.path.join(OUT, "targets-sc14-kimi.jsonl")
TOKENIZER = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/models/swonlysc-step0014/hf/tokenizer.json"
SYS_TOOLS_TOKENS = 9000

_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_file(TOKENIZER)
    return _TOK


# ------------------------------------------------------------------ items (Responses API) -> sc14 token counts
def item_text(it):
    t = it.get("type")
    out = []
    if t == "reasoning":
        for key in ("summary", "content"):
            for s in it.get(key) or []:
                if isinstance(s, dict) and isinstance(s.get("text"), str):
                    out.append(s["text"])
        return "reasoning", out
    if t == "message":
        c = it.get("content")
        if isinstance(c, str):
            out.append(c)
        else:
            for s in c or []:
                if isinstance(s, dict) and isinstance(s.get("text"), str):
                    out.append(s["text"])
        return "message", out
    if t == "function_call":
        a = it.get("arguments")
        return "function_call", [str(it.get("name") or ""), a if isinstance(a, str) else json.dumps(a)]
    if t == "function_call_output":
        o = it.get("output")
        return "function_call_output", [o if isinstance(o, str) else json.dumps(o)]
    return "other", [json.dumps(it)]


def items_tokens(items):
    texts, kinds = [], []
    for it in items or []:
        k, ts = item_text(it)
        for s in ts:
            if s:
                texts.append(s)
                kinds.append(k)
    counts = Counter({"reasoning": 0, "message": 0, "function_call": 0, "function_call_output": 0, "other": 0})
    if texts:
        tok = tokenizer()
        B = 256
        for i in range(0, len(texts), B):
            encs = tok.encode_batch(texts[i:i + B], add_special_tokens=False)
            for k, e in zip(kinds[i:i + B], encs):
                counts[k] += len(e.ids)
    return dict(counts)


# ------------------------------------------------------------------ sc14: OpenCode export.json
def ctx_of(info):
    t = info.get("tokens") or {}
    if not isinstance(t, dict):
        return None
    c = t.get("cache") or {}
    vals = [t.get("input"), c.get("read"), c.get("write"), t.get("output")]
    if all(v is None for v in vals):
        return None
    return sum(int(v or 0) for v in vals)


def export_metrics(ex):
    info0 = ex.get("info") or {}
    main_id = info0.get("id")
    msgs = ex.get("messages") or []
    res = {"n_calls": 0, "n_compactions": 0, "peak_ctx": None, "peak_is_compaction": None,
           "peak_ctx_calls_only": None, "first_ctx": None, "last_ctx": None, "last_finish": None,
           "n_other_session_msgs": 0, "n_zero_token_calls": 0, "sum_output_tokens": 0, "max_call_output_tokens": 0,
           "sum_reasoning_tokens": 0, "compaction_trigger_ctx": [], "compaction_ctx": [], "uncompacted_len_est": None,
           "session_wall_s": None, "finish_counts": {}}
    t_first = t_last = None
    fin = Counter()
    segs = [[]]
    prev = None
    for m in msgs:
        info = m.get("info") or {}
        if main_id and info.get("sessionID") not in (None, main_id):
            res["n_other_session_msgs"] += 1
            continue
        tm = info.get("time") or {}
        for tv in (tm.get("created"), tm.get("completed")):
            if isinstance(tv, (int, float)):
                t_first = tv if t_first is None else min(t_first, tv)
                t_last = tv if t_last is None else max(t_last, tv)
        if info.get("role") != "assistant":
            continue
        is_comp = info.get("mode") == "compaction" or info.get("summary") is True
        ctx = ctx_of(info)
        tok = info.get("tokens") or {}
        if isinstance(tok, dict):
            o = int(tok.get("output") or 0)
            res["sum_output_tokens"] += o
            res["max_call_output_tokens"] = max(res["max_call_output_tokens"], o)
            res["sum_reasoning_tokens"] += int(tok.get("reasoning") or 0)
        if ctx is not None and (res["peak_ctx"] is None or ctx > res["peak_ctx"]):
            res["peak_ctx"] = ctx
            res["peak_is_compaction"] = is_comp
        if is_comp:
            res["n_compactions"] += 1
            res["compaction_trigger_ctx"].append(prev)
            res["compaction_ctx"].append(ctx)
            segs.append([])
            continue
        res["n_calls"] += 1
        f = info.get("finish")
        fin[str(f)] += 1
        res["last_finish"] = f
        if not ctx:
            res["n_zero_token_calls"] += 1  # e.g. the in-flight call of a killed session
            continue
        if res["first_ctx"] is None:
            res["first_ctx"] = ctx
        res["last_ctx"] = ctx
        prev = ctx
        segs[-1].append(ctx)
        if res["peak_ctx_calls_only"] is None or ctx > res["peak_ctx_calls_only"]:
            res["peak_ctx_calls_only"] = ctx
    if t_first is not None and t_last is not None:
        res["session_wall_s"] = round((t_last - t_first) / 1000.0, 1)
    res["finish_counts"] = dict(fin)
    # length the transcript would have had without compaction: first segment's peak + growth of later segments
    if any(segs):
        est = max(segs[0]) if segs[0] else 0
        for s in segs[1:]:
            if s:
                est += max(0, max(s) - s[0])
        res["uncompacted_len_est"] = est
    return res


# ------------------------------------------------------------------ kimi: per-HTTP-call capture
TIME_TOL_S = 1.0  # clock tolerance when requiring a session's previous call to finish before the next one starts


def _new_session(c):
    return {"sys": c["sys"], "calls": [c], "last": c, "closed": False}


def _push(s, c, is_main):
    s["calls"].append(c)
    s["last"] = c
    s["sys"] = c["sys"]
    if not is_main and c["fr"] == "stop":  # a subagent's final answer ends its session
        s["closed"] = True


def thread_sessions(calls):
    """Split one rollout's calls (capture order = completion order) into chat sessions.

    Rule (exact in OpenAI chat format): next n_msgs = n_msgs + 1 (assistant) + n_tool_calls (tool results).
    A call with n_msgs <= 2 (system + user) starts a new session; sessions[0] (first call) is the main session.
    Ties between sessions are broken by (1) the previous call having finished before this one started and
    (2) prompt-token continuity (this prompt >= previous prompt + previous visible output).
    Returns (sessions, flags)."""
    sessions, flags = [], Counter()
    for c in sorted(calls, key=lambda c: c["pos"]):
        n = c["n_msgs"]
        if not sessions:
            sessions.append(_new_session(c))
            continue
        if n is None:
            flags["no_n_msgs"] += 1
            _push(sessions[0], c, True)
            continue
        if n <= 2:
            sessions.append(_new_session(c))
            continue
        open_ = [s for s in sessions if not s["closed"] and s["last"]["n_msgs"] is not None and s["last"]["n_msgs"] < n]
        exact_any = [s for s in open_ if s["last"]["n_msgs"] + 1 + (s["last"]["ntc"] or 0) == n]
        exact = [s for s in exact_any if s["sys"] == c["sys"]]
        same = [s for s in open_ if s["sys"] == c["sys"]]
        if exact:
            pool = exact
        elif exact_any:  # system prompt text changed mid-session (e.g. the date in it rolled over)
            pool = exact_any
            flags["sys_changed"] += 1
        elif same:
            pool = same
            flags["fallback"] += 1
        else:
            flags["orphan"] += 1
            sessions.append(_new_session(c))
            continue
        if len(pool) > 1:
            timed = [s for s in pool if c["st"] is None or s["last"]["ct"] is None
                     or s["last"]["ct"] <= c["st"] + TIME_TOL_S]
            pool = timed or pool

            def gap(s):
                p = s["last"]
                if not c["ti"] or not p["ti"]:
                    return 10 ** 9
                g = c["ti"] - (p["ti"] + (p["to"] or 0) - (p["tr"] or 0))
                return abs(g) + (0 if g >= -1000 else 10 ** 6)
            pool = sorted(pool, key=gap)
            if len(pool) > 1:
                flags["ambiguous"] += 1
                if any(s is sessions[0] for s in pool):
                    flags["ambiguous_main"] += 1
        s = pool[0]
        _push(s, c, s is sessions[0])
    return sessions, flags


def kimi_metrics(calls):
    sessions, flags = thread_sessions(calls)
    main = sessions[0]["calls"]
    res = {"n_calls": len(main), "n_compactions": 0, "peak_ctx": None, "peak_is_compaction": False,
           "peak_ctx_calls_only": None, "first_ctx": None, "last_ctx": None, "last_finish": None,
           "n_zero_token_calls": 0, "sum_output_tokens": 0, "max_call_output_tokens": 0, "sum_reasoning_tokens": 0,
           "full_conv_len_native_est": None, "n_sessions": len(sessions),
           "n_subagent_calls": sum(len(s["calls"]) for s in sessions[1:]),
           "peak_ctx_any_session": None, "main_tool_calls": sum(c["ntc"] or 0 for c in main),
           "thread_flags": dict(flags), "models": dict(Counter(c["model"] for c in main)),
           "max_prompt_tokens": None, "session_wall_s": None, "finish_counts": dict(Counter(str(c["fr"]) for c in main))}
    reasoning_before = 0
    for c in main:
        ti, to, tr = c["ti"], c["to"], c["tr"]
        res["last_finish"] = c["fr"]
        if not ti and not to:
            res["n_zero_token_calls"] += 1  # usage missing (failover backend answered without usage)
            continue
        ctx = int(ti or 0) + int(to or 0)
        res["sum_output_tokens"] += int(to or 0)
        res["sum_reasoning_tokens"] += int(tr or 0)
        res["max_call_output_tokens"] = max(res["max_call_output_tokens"], int(to or 0))
        if res["first_ctx"] is None:
            res["first_ctx"] = ctx
        res["last_ctx"] = ctx
        if res["peak_ctx"] is None or ctx > res["peak_ctx"]:
            res["peak_ctx"] = ctx
            res["full_conv_len_native_est"] = ctx + reasoning_before
        if res["max_prompt_tokens"] is None or int(ti or 0) > res["max_prompt_tokens"]:
            res["max_prompt_tokens"] = int(ti or 0)
        reasoning_before += int(tr or 0)
    res["peak_ctx_calls_only"] = res["peak_ctx"]
    allv = [int(c["ti"] or 0) + int(c["to"] or 0) for s in sessions for c in s["calls"]]
    res["peak_ctx_any_session"] = max(allv) if allv else None
    ts = [x for c in main for x in (c["st"], c["ct"]) if isinstance(x, (int, float))]
    if ts:
        res["session_wall_s"] = round(max(ts) - min(ts), 1)
    return res


# ------------------------------------------------------------------ worker
def do_task(args):
    task_id, kimi_calls_by_li = args
    path = os.path.join(TARGETS, "records", task_id + ".jsonl")
    rows, t0 = [], time.time()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            src = r.get("source") or {}
            items = r.get("output") or []
            patch = r.get("patch") or ""
            slot = r.get("slot")
            row = {
                "task_id": r.get("task_id"), "task_index": r.get("task_index"), "model": r.get("model"), "slot": slot,
                "repeat": int(slot.rsplit("-r", 1)[1]) if slot and "-r" in slot else None,
                "status": r.get("status"), "resolved": r.get("status") == "pass", "reward": r.get("reward"),
                "f2p": r.get("f2p"), "p2p": r.get("p2p"), "agent_error": r.get("agent_error"),
                "verifier_error": r.get("verifier_error"),
                "patch_bytes": len(patch.encode("utf-8")), "empty_patch": not patch.strip(),
                "n_items": len(items), "item_types": dict(Counter(it.get("type") for it in items)),
                "source_line": src.get("line"),
            }
            m = None
            if r.get("model") == "sc14":
                fp = src.get("opencode_results_fpath")
                row["token_source"] = fp
                if fp and os.path.isfile(fp):
                    try:
                        with open(fp) as ef:
                            m = export_metrics(json.load(ef))
                        row["metric_method"] = "opencode_export_per_message_tokens"
                    except Exception as e:  # unreadable export -> fallback below
                        row["export_error"] = repr(e)[:200]
                        m = None
            elif r.get("model") == "kimi":
                li = src.get("line")
                row["token_source"] = f"{CALLS}#li={li}"
                calls = kimi_calls_by_li.get(li)
                if src.get("gz_sha256") != CALLS_GZ_SHA:
                    row["calls_sha_mismatch"] = True
                elif calls:
                    t_ok = all(c["t"] == r.get("task_index") for c in calls)
                    r_ok = all(c["r"] == row["repeat"] for c in calls)
                    row["calls_key_check"] = bool(t_ok and r_ok)
                    m = kimi_metrics(calls)
                    row["metric_method"] = "hosted_api_per_call_usage(ng_model_call_capture)"
                    row["main_tool_calls_vs_items"] = [m["main_tool_calls"], row["item_types"].get("function_call", 0)]
            tk = items_tokens(items)
            row["items_tok_sc14"] = tk
            tot = sum(tk.values())
            row["items_tok_est_sc14"] = tot + SYS_TOOLS_TOKENS
            row["items_tok_est_sc14_no_reasoning"] = tot - tk["reasoning"] + SYS_TOOLS_TOKENS
            if m is None and not items:  # agent never ran (e.g. AgentSandboxError): nothing to measure
                row["metric_method"] = "none(no per-call source, no output items)"
                m = {"peak_ctx": None, "n_calls": None, "n_compactions": None}
            elif m is None:
                row["metric_method"] = "items_tok_sc14+9000(fallback)"
                m = {"peak_ctx": row["items_tok_est_sc14"], "n_calls": None, "n_compactions": None}
            row.update(m)
            row["peak_over_242k"] = (row["peak_ctx"] or 0) > 242144
            row["peak_over_262k"] = (row["peak_ctx"] or 0) > 262144
            rows.append(row)
    return task_id, rows, round(time.time() - t0, 1)


def load_kimi_calls():
    by_t = defaultdict(lambda: defaultdict(list))
    n = 0
    with open(CALLS) as f:
        for line in f:
            r = json.loads(line)
            if r.get("m") != "kimi":
                continue
            req = r.get("req") or {}
            by_t[r.get("t")][r["li"]].append({
                "pos": r["pos"], "t": r.get("t"), "r": r.get("r"), "st": r.get("st"), "ct": r.get("ct"),
                "sys": req.get("sys_sha"), "n_msgs": req.get("n_msgs"), "ntc": r.get("ntc"), "fr": r.get("fr"),
                "model": r.get("model"), "ti": r.get("ti"), "to": r.get("to"), "tr": r.get("tr"),
            })
            n += 1
    return by_t, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--out", default=OUT_ROWS)
    a = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    task_ids = json.load(open(os.path.join(TARGETS, "task_ids.json")))
    if isinstance(task_ids, dict):
        task_ids = task_ids.get("task_ids") or list(task_ids)
    if a.tasks:
        task_ids = [t for t in task_ids if t in set(a.tasks.split(","))]
    idx = {}
    with open(os.path.join(TARGETS, "per_task.csv")) as f:
        next(f)
        for line in f:
            p = line.strip().split(",")
            idx[p[0]] = int(p[1])
    t0 = time.time()
    by_t, n = load_kimi_calls()
    print(f"loaded {n} kimi calls for {sum(len(v) for v in by_t.values())} rollouts in {time.time() - t0:.0f}s", flush=True)
    work = [(t, dict(by_t.get(idx[t], {}))) for t in task_ids]
    rows = []
    with Pool(a.workers) as pool:
        for k, (tid, rr, dt) in enumerate(pool.imap_unordered(do_task, work), 1):
            rows.extend(rr)
            print(f"[{k}/{len(work)}] {tid} {len(rr)} rows {dt}s", flush=True)
    rows.sort(key=lambda x: (x["task_index"], 0 if x["model"] == "sc14" else 1, x["repeat"]))
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    os.replace(tmp, a.out)
    print(f"wrote {len(rows)} rows to {a.out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
