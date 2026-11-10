#!/usr/bin/env python3
"""Per-rollout context-length metrics for DeepSWE v1.1 eval runs. READ-ONLY on all inputs.

For every row of every <run>/shards/shard_K/rollouts.jsonl (one row per rollout) this records
  * outcome fields from the row: reward, model_patch_bytes, failure_reason/kind, opencode flags, f2p/p2p, ...
  * the length metric from the row's OpenCode session export (row["opencode_results_fpath"] -> export.json):
      ctx(msg)      = tokens.input + tokens.cache.read + tokens.cache.write + tokens.output
      peak_ctx      = max ctx over the main session's assistant messages (compaction summaries included)
      n_calls       = assistant messages that are not compaction summaries
      n_compactions = assistant messages with mode == "compaction" or summary is True
    This is the same quantity as the SFT corpus' full rendered conversation length (prompt + completion of a call).
  * fallback when a rollout has no readable export: tokenize the row's response.output items with the sc14
    tokenizer and add 9,000 tokens for system prompt + tools (metric_method says which was used).

response.usage in the row is cumulative over all calls; it is kept only as cum_* fields, never as a length.

Outputs (OUT dir): <run>.jsonl (one row per rollout, sorted by task_index, rollout_index).
usage: evalrun_lengths.py [--workers 8] RUN [RUN ...]
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

RUNS_DIR = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/eval/deepswe-v1.1/runs"
OUT = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/analysis/deepswe-long-tasks-20261027"
TOKENIZER = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/models/swonlysc-step0014/hf/tokenizer.json"
SYS_TOOLS_TOKENS = 9000
COMPACT_AT = 242144  # OpenCode compaction threshold for the 262,144-token student window

_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_file(TOKENIZER)
    return _TOK


def item_texts(items):
    """Text content of Responses-API items (reasoning, message, function_call, function_call_output)."""
    out = []
    for it in items or []:
        t = it.get("type")
        if t == "reasoning":
            for s in it.get("summary") or []:
                if isinstance(s, dict) and isinstance(s.get("text"), str):
                    out.append(s["text"])
            for s in it.get("content") or []:
                if isinstance(s, dict) and isinstance(s.get("text"), str):
                    out.append(s["text"])
        elif t == "message":
            c = it.get("content")
            if isinstance(c, str):
                out.append(c)
            else:
                for s in c or []:
                    if isinstance(s, dict) and isinstance(s.get("text"), str):
                        out.append(s["text"])
        elif t == "function_call":
            out.append(str(it.get("name") or ""))
            a = it.get("arguments")
            out.append(a if isinstance(a, str) else json.dumps(a))
        elif t == "function_call_output":
            o = it.get("output")
            out.append(o if isinstance(o, str) else json.dumps(o))
        else:
            out.append(json.dumps(it))
    return out


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
    """Length metrics of one OpenCode export (main session only)."""
    info0 = ex.get("info") or {}
    main_id = info0.get("id")
    msgs = ex.get("messages") or []
    res = {
        "n_messages": len(msgs), "n_other_session_msgs": 0, "n_user_msgs": 0, "n_assistant_msgs": 0,
        "n_calls": 0, "n_compactions": 0, "peak_ctx": None, "peak_ctx_msg_idx": None, "peak_is_compaction": None,
        "peak_ctx_calls_only": None, "first_ctx": None, "last_ctx": None, "last_finish": None,
        "finish_counts": {}, "n_error_msgs": 0, "errors": [], "sum_output_tokens": 0, "max_call_output_tokens": 0,
        "n_zero_token_assistant_msgs": 0, "compaction_trigger_ctx": [], "uncompacted_len_est": None,
        "session_tokens": info0.get("tokens"), "session_wall_s": None, "n_calls_with_tokens": 0,
    }
    t_first, t_last = None, None
    finishes = Counter()
    errors = Counter()
    segs = [[]]  # ctx of non-compaction calls, split at compaction summaries
    prev_call_ctx = None
    for i, m in enumerate(msgs):
        info = m.get("info") or {}
        if main_id and info.get("sessionID") not in (None, main_id):
            res["n_other_session_msgs"] += 1
            continue
        tm = info.get("time") or {}
        for tv in (tm.get("created"), tm.get("completed")):
            if isinstance(tv, (int, float)):
                t_first = tv if t_first is None else min(t_first, tv)
                t_last = tv if t_last is None else max(t_last, tv)
        role = info.get("role")
        if role == "user":
            res["n_user_msgs"] += 1
            continue
        if role != "assistant":
            continue
        res["n_assistant_msgs"] += 1
        is_comp = info.get("mode") == "compaction" or info.get("summary") is True
        ctx = ctx_of(info)
        tok = info.get("tokens") or {}
        if isinstance(tok, dict) and tok.get("output") is not None:
            res["sum_output_tokens"] += int(tok.get("output") or 0)
            res["max_call_output_tokens"] = max(res["max_call_output_tokens"], int(tok.get("output") or 0))
        if not ctx:
            res["n_zero_token_assistant_msgs"] += 1
        if info.get("error"):
            res["n_error_msgs"] += 1
            e = info["error"]
            if isinstance(e, dict):
                name = e.get("name") or ""
                msg = ((e.get("data") or {}).get("message") if isinstance(e.get("data"), dict) else "") or ""
                errors[f"{name}: {str(msg)[:160]}"] += 1
            else:
                errors[str(e)[:200]] += 1
        fin = info.get("finish")
        finishes[str(fin)] += 1
        if ctx is not None and (res["peak_ctx"] is None or ctx > res["peak_ctx"]):
            res["peak_ctx"] = ctx
            res["peak_ctx_msg_idx"] = i
            res["peak_is_compaction"] = is_comp
        if is_comp:
            res["n_compactions"] += 1
            res["compaction_trigger_ctx"].append(prev_call_ctx)
            segs.append([])
            continue
        res["n_calls"] += 1
        res["last_finish"] = fin
        if ctx:  # zero-token calls (e.g. the in-flight call of a killed session) carry no length information
            res["n_calls_with_tokens"] += 1
            if res["first_ctx"] is None:
                res["first_ctx"] = ctx
            res["last_ctx"] = ctx
            prev_call_ctx = ctx
            segs[-1].append(ctx)
            if res["peak_ctx_calls_only"] is None or ctx > res["peak_ctx_calls_only"]:
                res["peak_ctx_calls_only"] = ctx
    if t_first is not None and t_last is not None:
        res["session_wall_s"] = round((t_last - t_first) / 1000.0, 1)
    res["finish_counts"] = dict(finishes)
    res["errors"] = [f"{k} x{v}" for k, v in errors.most_common(5)]
    # Rough length the transcript would have had without compaction: first segment's peak plus the growth
    # (peak - first call) of every later segment. Equals peak_ctx_calls_only when there was no compaction.
    nonempty = [s for s in segs if s]
    if nonempty:
        est = max(segs[0]) if segs[0] else 0
        for s in segs[1:]:
            if s:
                est += max(0, max(s) - s[0])
        res["uncompacted_len_est"] = est
    return res


def do_unit(args):
    run, shard = args
    path = f"{RUNS_DIR}/{run}/shards/{shard}/rollouts.jsonl"
    rows = []
    t0 = time.time()
    with open(path) as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                rows.append({"run": run, "shard": shard, "line": ln, "row_error": str(e)[:200]})
                continue
            resp = r.get("response") or {}
            usage = resp.get("usage") or {}
            items = resp.get("output") or []
            rec = {
                "run": run, "shard": shard, "line": ln,
                "task_index": r.get("_ng_task_index"), "task_id": r.get("task_id"),
                "rollout_index": r.get("_ng_rollout_index"),
                "reward": r.get("reward"), "resolved": float(r.get("reward") or 0.0) >= 1.0,
                "model_patch_bytes": r.get("model_patch_bytes"),
                "failure_reason": r.get("failure_reason"), "failure_kind": r.get("failure_kind"),
                "mask_sample": r.get("mask_sample"),
                "opencode_finished": r.get("opencode_finished"), "opencode_export_found": r.get("opencode_export_found"),
                "evaluation_completed": r.get("evaluation_completed"), "apply_failed": r.get("apply_failed"),
                "verifier_error": (str(r.get("verifier_error"))[:300] if r.get("verifier_error") else None),
                "f2p": r.get("f2p"), "p2p": r.get("p2p"),
                "f2p_passed": r.get("f2p_passed"), "f2p_total": r.get("f2p_total"),
                "p2p_passed": r.get("p2p_passed"), "p2p_total": r.get("p2p_total"),
                "cum_input_tokens_uncached": usage.get("input_tokens"),
                "cum_cached_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens"),
                "cum_output_tokens": usage.get("output_tokens"),
                "n_output_items": len(items),
                "output_item_types": dict(Counter(it.get("type") for it in items)),
                "opencode_stdout_chars": len(r.get("opencode_run_stdout") or ""),
                "opencode_stderr_tail": (r.get("opencode_run_stderr") or "")[-300:] or None,
                "export_path": r.get("opencode_results_fpath") or None,
            }
            fp = rec["export_path"]
            ex = None
            if fp and os.path.isfile(fp):
                try:
                    with open(fp) as g:
                        ex = json.load(g)
                except Exception as e2:  # noqa: BLE001
                    rec["export_error"] = str(e2)[:200]
            elif fp:
                rec["export_error"] = "export path does not exist"
            else:
                rec["export_error"] = "no opencode_results_fpath"
            rec["export_ok"] = ex is not None
            if ex is not None:
                rec.update(export_metrics(ex))
                rec["metric_method"] = "export_peak_ctx"
                if rec["peak_ctx"] is None:
                    rec["metric_method"] = "export_without_token_counts"
                del ex
            if not rec["export_ok"] or rec.get("peak_ctx") is None:
                texts = item_texts(items)
                if texts:
                    enc = tokenizer().encode_batch(texts, add_special_tokens=False)
                    est = sum(len(e.ids) for e in enc) + SYS_TOOLS_TOKENS
                    rec["est_items_tokens_plus_9000"] = est
                    if rec.get("peak_ctx") is None:
                        rec["peak_ctx"] = est
                        rec["metric_method"] = "est_tokenized_output_items_plus_9000"
                elif rec.get("peak_ctx") is None:
                    rec["metric_method"] = "none_no_export_no_items"
            del r, items, resp
            rows.append(rec)
    return run, shard, rows, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    units = []
    for run in a.runs:
        sd = f"{RUNS_DIR}/{run}/shards"
        for s in sorted(os.listdir(sd), key=lambda x: (len(x), x)):
            if os.path.isdir(f"{sd}/{s}") and os.path.isfile(f"{sd}/{s}/rollouts.jsonl"):
                units.append((run, s))
    # Large shards first so the pool's tail is short.
    units.sort(key=lambda u: -os.path.getsize(f"{RUNS_DIR}/{u[0]}/shards/{u[1]}/rollouts.jsonl"))
    by_run = {run: [] for run in a.runs}
    t0 = time.time()
    with Pool(min(a.workers, 8)) as pool:
        for n, (run, shard, rows, dt) in enumerate(pool.imap_unordered(do_unit, units), 1):
            by_run[run].extend(rows)
            print(f"[{time.time() - t0:7.1f}s] {n}/{len(units)} {run} {shard} rows={len(rows)} ({dt:.1f}s)", flush=True)
    for run, rows in by_run.items():
        rows.sort(key=lambda x: (x.get("task_index") if x.get("task_index") is not None else 10**9,
                                 x.get("rollout_index") if x.get("rollout_index") is not None else 10**9))
        dst = f"{OUT}/{run}.jsonl"
        tmp = f"{dst}.tmp.{os.getpid()}"
        with open(tmp, "w") as fh:
            for x in rows:
                fh.write(json.dumps(x) + "\n")
        os.replace(tmp, dst)
        meth = Counter(x.get("metric_method") for x in rows)
        print(f"wrote {dst}: {len(rows)} rows; methods={dict(meth)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
