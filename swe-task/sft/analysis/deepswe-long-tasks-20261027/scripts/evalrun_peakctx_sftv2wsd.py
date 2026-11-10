#!/usr/bin/env python3
"""Per-rollout context-length metrics for DeepSWE v1.1 eval runs (READ-ONLY on every input).

usage:
  python3 -I evalrun_peakctx_sftv2wsd.py --out-dir OUT [--procs 8] [--report-only] RUN [RUN ...]

Rows: <runs>/<RUN>/shards/shard_*/rollouts.jsonl, deduplicated like tools/aggregate.py (last row per
(_ng_task_index, _ng_rollout_index)). For every row the OpenCode session export (row.opencode_results_fpath,
agent_results/results/<uuid>/export.json) is read once, and over the MAIN session's assistant messages
(messages whose info.sessionID == export info.id) it computes

  peak_ctx       max of tokens.input + tokens.cache.read + tokens.cache.write + tokens.output over ALL
                 assistant messages (compaction summaries included); the headline length metric, comparable
                 to the SFT corpus' full rendered conversation length as long as nothing was compacted
  n_calls        assistant messages that are not compaction summaries
  n_compactions  assistant messages with mode == "compaction" or summary is True

Secondary fields (labelled, never the headline):
  peak_ctx_noncompaction   the same max over non-summary calls only
  est_uncompacted_len      first call total + sum of positive call-to-call increases of the per-call total
                           (non-summary calls with a non-zero total); an ESTIMATE of the rendered length the
                           session would have had without compaction (drops after compaction/pruning ignored)
  segment_peaks            peak per-call total in each compaction segment
  wall_s, out_tokens_sum/max, n_calls_out_ge_131072, error names, last finish, mapping check
  (first user message of the export contains the row's task prompt).

response.usage in rollouts.jsonl is CUMULATIVE over all calls (kept only as usage_cumulative_*).
Rows without an export (sandbox lost before export) get peak_ctx null and len_method "no_export".

Writes OUT/<RUN>.jsonl (one row per rollout, sorted by task index, rollout index) and OUT/<RUN>.summary.json,
and prints a readable summary. --report-only recomputes the summary from an existing OUT/<RUN>.jsonl.
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

RUNS = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/eval/deepswe-v1.1/runs"
TARGETS = ["scriggo-method-declarations", "wasmi-trap-coredumps", "boa-hierarchical-evaluation-cancellation"]
COMPACT_AT = 242144      # OpenCode auto-compaction threshold for max_model_len 262,144 (262,144 - 20,000)
MAX_MODEL_LEN = 262144
OUT_CAP = 131072          # per-call max output tokens of the "out131k" runs


def short(s, n):
    s = "" if s is None else str(s)
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + "..."


def tail(s, n):
    s = "" if s is None else str(s)
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else "..." + s[-n:]


def norm_alnum(s):
    return re.sub(r"[^0-9A-Za-z]", "", s or "")


def unescape_export_text(t):
    s = (t or "").strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        try:
            v = json.loads(s)
            if isinstance(v, str):
                return v
        except Exception:
            pass
    s = re.sub(r"\\[nrt]", " ", s)
    return re.sub(r"\\(.)", r"\1", s)


def user_prompt_text(row):
    for it in (row.get("responses_create_params") or {}).get("input") or []:
        if isinstance(it, dict) and it.get("role") == "user":
            c = it.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return "".join((p.get("text") or "") for p in c if isinstance(p, dict))
    return ""


# ----------------------------------------------------------------------------- phase 1: rollouts.jsonl
def read_shard(shard_file):
    run = shard_file.split("/")[-4]
    shard = shard_file.split("/")[-2]
    rows, n_lines, n_bad = {}, 0, 0
    with open(shard_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            resp = d.get("response") or {}
            usage = resp.get("usage") or {}
            key = (d.get("_ng_task_index"), d.get("_ng_rollout_index"))
            rows[key] = {
                "run": run, "shard": shard,
                "task_index": d.get("_ng_task_index"), "rollout_index": d.get("_ng_rollout_index"),
                "task_id": d.get("task_id"),
                "reward": d.get("reward"),
                "resolved": float(d.get("reward") or 0.0) >= 1.0,
                "model_patch_bytes": d.get("model_patch_bytes"),
                "failure_reason": d.get("failure_reason"),
                "failure_kind": d.get("failure_kind"),
                "opencode_finished": d.get("opencode_finished"),
                "opencode_export_found": d.get("opencode_export_found"),
                "evaluation_completed": d.get("evaluation_completed"),
                "verifier_error": short(d.get("verifier_error"), 160) if d.get("verifier_error") else None,
                "f2p": d.get("f2p"), "p2p": d.get("p2p"),
                "f2p_passed": d.get("f2p_passed"), "f2p_total": d.get("f2p_total"),
                "p2p_passed": d.get("p2p_passed"), "p2p_total": d.get("p2p_total"),
                "opencode_run_stderr_tail": tail(d.get("opencode_run_stderr"), 120) or None,
                "export_path": d.get("opencode_results_fpath") or None,
                "log_dir": d.get("log_dir") or None,
                "n_response_output_items": len(resp.get("output") or []),
                "usage_cumulative_input_tokens": usage.get("input_tokens"),
                "usage_cumulative_cached_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens"),
                "usage_cumulative_output_tokens": usage.get("output_tokens"),
                "_prompt_head": norm_alnum(user_prompt_text(d))[:160],
            }
    return shard_file, list(rows.values()), n_lines, n_bad


# ----------------------------------------------------------------------------- phase 2: export.json
def export_metrics(args):
    path, prompt_head = args
    try:
        with open(path) as f:
            ex = json.load(f)
    except Exception as e:  # noqa: BLE001
        return path, {"export_read_error": short(f"{type(e).__name__}: {e}", 200)}
    info = ex.get("info") or {}
    sid = info.get("id")
    msgs = ex.get("messages") or []
    main = [m for m in msgs if not sid or (m.get("info") or {}).get("sessionID") in (None, sid)]
    peak = peak_nc = None
    peak_call = None
    max_prompt = None
    n_calls = n_comp = n_asst = 0
    totals_nc = []
    seg_peaks = [0]
    out_sum = out_max = n_cap = reas_sum = mism = 0
    n_finish_length = n_hit_limit = 0
    n_task_tool = 0
    running_tools = []
    errs = Counter()
    err_example = None
    last_finish = None
    last_ctx = None
    first_ctx = None
    t_min = t_max = None
    n_user_comp = n_user_comp_overflow = 0
    first_user_text = None
    comp_totals = []

    def upd(t):
        nonlocal t_min, t_max
        if isinstance(t, (int, float)) and t > 0:
            t_min = t if t_min is None else min(t_min, t)
            t_max = t if t_max is None else max(t_max, t)

    for m in main:
        mi = m.get("info") or {}
        tm = mi.get("time") or {}
        upd(tm.get("created"))
        upd(tm.get("completed"))
        parts = m.get("parts") or []
        for p in parts:  # part-level times too: a session can sit in one long tool call after its last message
            pt = p.get("time") or {}
            st = p.get("state") or {}
            stt = (st.get("time") or {}) if isinstance(st, dict) else {}
            for t in (pt.get("start"), pt.get("end"), stt.get("start"), stt.get("end")):
                upd(t)
            if p.get("type") == "tool":
                if p.get("tool") == "task":
                    n_task_tool += 1
                if isinstance(st, dict) and st.get("status") in ("running", "pending"):
                    running_tools.append(p.get("tool"))
        role = mi.get("role")
        if role == "user":
            if first_user_text is None:
                first_user_text = "".join((p.get("text") or "") for p in parts if p.get("type") == "text")
            if any(p.get("type") == "compaction" for p in parts):
                n_user_comp += 1
                if any(p.get("type") == "compaction" and p.get("overflow") for p in parts):
                    n_user_comp_overflow += 1
            continue
        if role != "assistant":
            continue
        n_asst += 1
        tok = mi.get("tokens") or {}
        cache = tok.get("cache") or {}
        tot = (int(tok.get("input") or 0) + int(cache.get("read") or 0) + int(cache.get("write") or 0)
               + int(tok.get("output") or 0))
        if isinstance(tok.get("total"), (int, float)) and int(tok["total"]) != tot:
            mism += 1
        reas_sum += int(tok.get("reasoning") or 0)
        prompt_side = int(tok.get("input") or 0) + int(cache.get("read") or 0) + int(cache.get("write") or 0)
        max_prompt = prompt_side if max_prompt is None else max(max_prompt, prompt_side)
        is_comp_msg = mi.get("mode") == "compaction" or mi.get("summary") is True
        if peak is None or tot > peak:
            peak = tot
            peak_call = {"mode": "compaction" if is_comp_msg else mi.get("mode"), "finish": mi.get("finish"),
                         "output": int(tok.get("output") or 0), "prompt": prompt_side}
        n_finish_length += mi.get("finish") == "length"
        n_hit_limit += tot >= MAX_MODEL_LEN
        err = mi.get("error")
        if isinstance(err, dict):
            name = err.get("name") or "?"
            errs[name] += 1
            if err_example is None:
                msg = (err.get("data") or {}).get("message") if isinstance(err.get("data"), dict) else None
                err_example = short(f"{name}: {msg or ''}", 200)
        if mi.get("mode") == "compaction" or mi.get("summary") is True:
            n_comp += 1
            comp_totals.append(tot)
            seg_peaks.append(0)
            continue
        n_calls += 1
        o = int(tok.get("output") or 0)
        out_sum += o
        out_max = max(out_max, o)
        n_cap += o >= OUT_CAP
        last_finish = mi.get("finish")
        if tot > 0:
            if first_ctx is None:
                first_ctx = tot
            last_ctx = tot
            totals_nc.append(tot)
            seg_peaks[-1] = max(seg_peaks[-1], tot)
            peak_nc = tot if peak_nc is None else max(peak_nc, tot)
    est = None
    if totals_nc:
        est = totals_nc[0] + sum(max(0, b - a) for a, b in zip(totals_nc, totals_nc[1:]))
    mapping_ok = None
    if prompt_head and first_user_text is not None:
        mapping_ok = prompt_head[:100] in norm_alnum(unescape_export_text(first_user_text))
    return path, {
        "len_method": "export_tokens",
        "export_session_id": sid,
        "export_n_messages": len(msgs),
        "export_n_other_session_messages": len(msgs) - len(main),
        "n_assistant_msgs": n_asst,
        "peak_ctx": peak,
        "n_calls": n_calls,
        "n_compactions": n_comp,
        "n_compaction_user_parts": n_user_comp,
        "n_compaction_overflow": n_user_comp_overflow,
        "peak_ctx_noncompaction": peak_nc,
        "peak_call": peak_call,
        "max_prompt_ctx": max_prompt,
        "n_finish_length": n_finish_length,
        "n_calls_at_ctx_limit": n_hit_limit,
        "n_task_subagent_calls": n_task_tool,
        "tools_running_at_end": running_tools,
        "first_ctx": first_ctx,
        "last_ctx": last_ctx,
        "est_uncompacted_len": est,
        "segment_peaks": seg_peaks,
        "compaction_call_totals": comp_totals,
        "out_tokens_sum": out_sum,
        "out_tokens_max": out_max,
        "n_calls_out_ge_131072": n_cap,
        "reasoning_tokens_sum": reas_sum,
        "n_total_field_mismatch": mism,
        "error_counts": dict(errs),
        "error_example": err_example,
        "last_finish": last_finish,
        "wall_s": round((t_max - t_min) / 1000.0, 1) if t_min is not None and t_max is not None else None,
        "mapping_ok": mapping_ok,
    }


# ----------------------------------------------------------------------------- summary helpers
def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def ordinal(n):
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def agent_error_label(r):
    bits = []
    if r.get("len_method") == "no_export":
        bits.append("no export (sandbox lost before export" + (f"; {short(r.get('verifier_error'), 70)}" if r.get("verifier_error") else "") + ")")
    else:
        if not r.get("opencode_finished"):
            bits.append(f"opencode not finished ({r.get('opencode_run_stderr_tail') or 'no stderr'}, 3 h sandbox kill"
                        + (f"; tool still running: {','.join(r['tools_running_at_end'])}" if r.get("tools_running_at_end") else "")
                        + ")")
        if r.get("n_task_subagent_calls"):
            bits.append(f"{r['n_task_subagent_calls']} task-subagent call(s), subagent sessions not in export")
        if r.get("n_finish_length"):
            bits.append(f"{r['n_finish_length']} call(s) finish=length (runaway output)")
        if r.get("error_counts"):
            bits.append("session errors " + ", ".join(f"{k} x{v}" for k, v in r["error_counts"].items()))
        if r.get("verifier_error"):
            bits.append("verifier error " + short(r["verifier_error"], 70))
    if r.get("failure_reason"):
        bits.append("failure_reason " + short(r["failure_reason"], 80))
    return "; ".join(bits) if bits else "none"


def summarize(run, rows):
    measured = [r for r in rows if r.get("peak_ctx") is not None]
    peaks = [r["peak_ctx"] for r in measured]
    calls = [r["n_calls"] for r in measured]
    comps = [r["n_compactions"] for r in measured]
    ests = [r["est_uncompacted_len"] for r in measured if r.get("est_uncompacted_len") is not None]
    by_task = defaultdict(list)
    for r in rows:
        by_task[(r["task_index"], r["task_id"])].append(r)
    task_rows = []
    for (ti, tid), rs in sorted(by_task.items()):
        rs = sorted(rs, key=lambda x: x["rollout_index"])
        mp = [x["peak_ctx"] for x in rs if x.get("peak_ctx") is not None]
        task_rows.append({
            "task_index": ti, "task_id": tid, "n": len(rs), "n_measured": len(mp),
            "resolved": sum(1 for x in rs if x["resolved"]),
            "median_peak_ctx": med(mp),
            "median_est_uncompacted_len": med([x.get("est_uncompacted_len") for x in rs]),
            "median_n_calls": med([x.get("n_calls") for x in rs]),
            "median_n_compactions": med([x.get("n_compactions") for x in rs]),
            "n_with_compaction": sum(1 for x in rs if (x.get("n_compactions") or 0) > 0),
            "n_peak_over_242k": sum(1 for x in mp if x > COMPACT_AT),
            "n_peak_at_262144_cap": sum(1 for x in mp if x >= MAX_MODEL_LEN),
            "n_with_finish_length_call": sum(1 for x in rs if x.get("n_finish_length")),
            "n_with_task_subagent": sum(1 for x in rs if x.get("n_task_subagent_calls")),
            "n_opencode_not_finished": sum(1 for x in rs if not x.get("opencode_finished")),
            "median_max_prompt_ctx": med([x.get("max_prompt_ctx") for x in rs]),
            "median_wall_s": med([x.get("wall_s") for x in rs]),
            "empty_patches": sum(1 for x in rs if (x.get("model_patch_bytes") or 0) == 0),
        })
    ranked = [t for t in task_rows if t["median_peak_ctx"] is not None]
    for t in task_rows:
        if t["median_peak_ctx"] is not None:
            t["rank_by_median_peak_ctx"] = 1 + sum(1 for u in ranked if u["median_peak_ctx"] > t["median_peak_ctx"])
            t["tied_tasks_at_median_peak_ctx"] = sum(1 for u in ranked if u["median_peak_ctx"] == t["median_peak_ctx"])
        if t["median_est_uncompacted_len"] is not None:
            t["rank_by_median_est_uncompacted_len"] = 1 + sum(
                1 for u in task_rows if u["median_est_uncompacted_len"] is not None
                and u["median_est_uncompacted_len"] > t["median_est_uncompacted_len"])
        if t["median_n_calls"] is not None:
            t["rank_by_median_n_calls"] = 1 + sum(
                1 for u in task_rows if u["median_n_calls"] is not None and u["median_n_calls"] > t["median_n_calls"])
    rates = [t["resolved"] / t["n"] for t in task_rows if t["n"]]
    comp_bucket = {}
    for name, cond in (("0", lambda c: c == 0), ("1", lambda c: c == 1), (">=2", lambda c: c >= 2)):
        sub = [r for r in measured if cond(r["n_compactions"])]
        comp_bucket[name] = {"rollouts": len(sub), "resolved": sum(1 for r in sub if r["resolved"]),
                             "resolved_pct": round(100 * sum(1 for r in sub if r["resolved"]) / len(sub), 2) if sub else None}
    try:
        metrics = json.load(open(os.path.join(RUNS, run, "metrics.json")))
    except Exception:  # noqa: BLE001
        metrics = {}
    overall = {
        "rollouts": len(rows),
        "rollouts_measured": len(measured),
        "rollouts_no_export": sum(1 for r in rows if r.get("len_method") == "no_export"),
        "resolved_rollouts": sum(1 for r in rows if r["resolved"]),
        "resolved_pct": round(100 * statistics.mean(rates), 2) if rates else None,
        "metrics_json_resolved@1": metrics.get("resolved@1"),
        "metrics_json_pass@5": metrics.get("pass@5"),
        "empty_patch_pct": round(100 * sum(1 for r in rows if (r.get("model_patch_bytes") or 0) == 0) / len(rows), 2),
        "opencode_not_finished": sum(1 for r in rows if not r.get("opencode_finished")),
        "median_peak_ctx": med(peaks),
        "mean_peak_ctx": round(statistics.mean(peaks)) if peaks else None,
        "p10_peak_ctx": pct(peaks, 0.10), "p25_peak_ctx": pct(peaks, 0.25), "p75_peak_ctx": pct(peaks, 0.75),
        "p90_peak_ctx": pct(peaks, 0.90), "max_peak_ctx": max(peaks) if peaks else None,
        "pct_peak_over_242k": round(100 * sum(1 for p in peaks if p > COMPACT_AT) / len(peaks), 2) if peaks else None,
        "n_peak_over_262144": sum(1 for p in peaks if p > MAX_MODEL_LEN),
        "pct_with_compaction": round(100 * sum(1 for c in comps if c > 0) / len(comps), 2) if comps else None,
        "median_calls": med(calls), "p90_calls": pct(calls, 0.90),
        "median_compactions": med(comps), "max_compactions": max(comps) if comps else None,
        "median_est_uncompacted_len": med(ests), "p90_est_uncompacted_len": pct(ests, 0.90),
        "pct_est_uncompacted_over_262144": round(100 * sum(1 for e in ests if e > MAX_MODEL_LEN) / len(ests), 2) if ests else None,
        "median_wall_s": med([r.get("wall_s") for r in measured]),
        "n_peak_at_262144_cap": sum(1 for p in peaks if p >= MAX_MODEL_LEN),
        "n_rollouts_with_finish_length_call": sum(1 for r in measured if r.get("n_finish_length")),
        "n_peak_call_finish_length": sum(1 for r in measured if (r.get("peak_call") or {}).get("finish") == "length"),
        "median_max_prompt_ctx": med([r.get("max_prompt_ctx") for r in measured]),
        "p90_max_prompt_ctx": pct([r["max_prompt_ctx"] for r in measured if r.get("max_prompt_ctx") is not None], 0.90),
        "n_rollouts_with_task_subagent": sum(1 for r in measured if r.get("n_task_subagent_calls")),
        "n_rollouts_tool_running_at_end": sum(1 for r in measured if r.get("tools_running_at_end")),
        "resolved_by_compactions": comp_bucket,
        "median_peak_ctx_resolved": med([r["peak_ctx"] for r in measured if r["resolved"]]),
        "median_peak_ctx_unresolved": med([r["peak_ctx"] for r in measured if not r["resolved"]]),
        "tasks_ranked_by_median_peak_ctx": len(ranked),
        "sanity": {
            "mapping_ok": sum(1 for r in measured if r.get("mapping_ok") is True),
            "mapping_bad": sum(1 for r in measured if r.get("mapping_ok") is False),
            "mapping_unchecked": sum(1 for r in measured if r.get("mapping_ok") is None),
            "rows_with_total_field_mismatch": sum(1 for r in measured if r.get("n_total_field_mismatch")),
            "rows_with_reasoning_tokens": sum(1 for r in measured if r.get("reasoning_tokens_sum")),
            "rows_with_other_session_msgs": sum(1 for r in measured if r.get("export_n_other_session_messages")),
            "rows_compaction_user_parts_ne_summaries": sum(
                1 for r in measured if r.get("n_compaction_user_parts") != r.get("n_compactions")),
            "export_read_errors": sum(1 for r in rows if r.get("export_read_error")),
        },
    }
    top = sorted(ranked, key=lambda t: -t["median_peak_ctx"])[:15]
    targets = []
    for tid in TARGETS:
        t = next((x for x in task_rows if x["task_id"] == tid), None)
        if t is None:
            continue
        rs = sorted(by_task[(t["task_index"], tid)], key=lambda x: x["rollout_index"])
        lines = []
        for r in rs:
            pk = f"{r['peak_ctx']:,}" if r.get("peak_ctx") is not None else "n/a (no export)"
            est = f"{r['est_uncompacted_len']:,}" if r.get("est_uncompacted_len") is not None else "n/a"
            pc = r.get("peak_call") or {}
            lines.append(
                f"r{r['rollout_index']} {'pass' if r['resolved'] else 'fail'}, peak_ctx {pk}, calls {r.get('n_calls')}, "
                f"compactions {r.get('n_compactions')}, patch {r.get('model_patch_bytes')} B, "
                f"agent error: {agent_error_label(r)} | peak call {pc.get('mode')}/{pc.get('finish')} out {pc.get('output')}, "
                f"max_prompt {r.get('max_prompt_ctx')}, est_uncompacted {est}, wall {r.get('wall_s')} s, "
                f"f2p {r.get('f2p_passed')}/{r.get('f2p_total')}, p2p {r.get('p2p_passed')}/{r.get('p2p_total')}, "
                f"segment_peaks {r.get('segment_peaks')}")
        t = dict(t)
        t["rollout_lines"] = lines
        if t.get("rank_by_median_peak_ctx"):
            k, n_tie = t["rank_by_median_peak_ctx"], t["tied_tasks_at_median_peak_ctx"]
            t["rank_label"] = (f"{ordinal(k)} longest of {len(ranked)} by median peak_ctx"
                               + (f" (tied with {n_tie - 1} other task(s) at {t['median_peak_ctx']:,})" if n_tie > 1 else ""))
        else:
            t["rank_label"] = None
        targets.append(t)
    return {"run": run, "overall": overall, "targets": targets, "top15_by_median_peak_ctx": top, "tasks": task_rows}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    all_rows = {}
    if a.report_only:
        for run in a.runs:
            with open(os.path.join(a.out_dir, f"{run}.jsonl")) as f:
                all_rows[run] = [json.loads(x) for x in f if x.strip()]
    else:
        shard_files = []
        for run in a.runs:
            fs = glob.glob(os.path.join(RUNS, run, "shards", "shard_*", "rollouts.jsonl"))
            shard_files += sorted(fs, key=lambda p: int(p.split("/")[-2].split("_")[1]))
        rows_by_run = defaultdict(dict)
        with Pool(a.procs) as pool:
            for shard_file, rows, n_lines, n_bad in pool.imap(read_shard, shard_files):
                if n_bad:
                    print(f"WARN {shard_file}: {n_bad}/{n_lines} corrupt lines", file=sys.stderr)
                for r in rows:
                    rows_by_run[r["run"]][(r["task_index"], r["rollout_index"])] = r
            jobs = []
            for run, rows in rows_by_run.items():
                for r in rows.values():
                    if r["export_path"]:
                        cum = (r.get("usage_cumulative_input_tokens") or 0) + (r.get("usage_cumulative_cached_tokens") or 0)
                        jobs.append((cum, r["export_path"], r["_prompt_head"]))
            jobs.sort(key=lambda j: -j[0])  # biggest sessions first for load balance
            res = {}
            for i, (path, m) in enumerate(pool.imap_unordered(export_metrics, [(p, h) for _, p, h in jobs], chunksize=2)):
                res[path] = m
                if (i + 1) % 200 == 0:
                    print(f"  exports parsed: {i + 1}/{len(jobs)}", file=sys.stderr, flush=True)
        for run in a.runs:
            out = []
            for key in sorted(rows_by_run[run], key=lambda k: (k[0], k[1])):
                r = rows_by_run[run][key]
                r.pop("_prompt_head", None)
                if r["export_path"] and r["export_path"] in res:
                    r.update(res[r["export_path"]])
                    if r.get("export_read_error"):
                        r["len_method"] = "export_unreadable"
                        r["peak_ctx"] = None
                else:
                    r["len_method"] = "no_export"
                    r["peak_ctx"] = None
                out.append(r)
            tmp = os.path.join(a.out_dir, f".{run}.jsonl.tmp")
            with open(tmp, "w") as f:
                for r in out:
                    f.write(json.dumps(r) + "\n")
            os.replace(tmp, os.path.join(a.out_dir, f"{run}.jsonl"))
            all_rows[run] = out
    for run in a.runs:
        s = summarize(run, all_rows[run])
        tmp = os.path.join(a.out_dir, f".{run}.summary.json.tmp")
        with open(tmp, "w") as f:
            json.dump(s, f, indent=1)
        os.replace(tmp, os.path.join(a.out_dir, f"{run}.summary.json"))
        o = s["overall"]
        print(f"\n=== {run}")
        print(json.dumps({k: v for k, v in o.items()}, indent=None))
        for t in s["targets"]:
            print(f"-- {t['task_id']} (task {t['task_index']}): resolved {t['resolved']}/{t['n']}, median peak_ctx "
                  f"{t['median_peak_ctx']}, {t['rank_label']}; est_uncompacted rank {t.get('rank_by_median_est_uncompacted_len')}, "
                  f"median est {t['median_est_uncompacted_len']}, calls rank {t.get('rank_by_median_n_calls')}, "
                  f"median calls {t['median_n_calls']}, compactions med {t['median_n_compactions']} "
                  f"({t['n_with_compaction']}/{t['n']} with >=1), median wall {t['median_wall_s']} s, "
                  f"peak at 262,144 cap {t['n_peak_at_262144_cap']}/{t['n']}, finish=length calls in "
                  f"{t['n_with_finish_length_call']}/{t['n']}, not finished {t['n_opencode_not_finished']}/{t['n']}, "
                  f"task-subagent {t['n_with_task_subagent']}/{t['n']}, median max_prompt {t['median_max_prompt_ctx']}, "
                  f"empty patches {t['empty_patches']}/{t['n']}")
            for line in t["rollout_lines"]:
                print("   ", line)
        print("-- top 15 tasks by median peak_ctx:")
        for t in s["top15_by_median_peak_ctx"]:
            print(f"   #{t['rank_by_median_peak_ctx']:>3} {t['task_id']:<55} med_peak {t['median_peak_ctx']:>9} "
                  f"med_est {t['median_est_uncompacted_len']} comp {t['n_with_compaction']}/{t['n']} res {t['resolved']}/{t['n']}")


if __name__ == "__main__":
    main()
