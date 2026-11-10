#!/usr/bin/env python3
"""Summarize targets-sc14-kimi.jsonl (written by targets_lengths.py): per-model overall length/accuracy stats,
per-task medians + rank among the 113 tasks, detail for the focus tasks, and calibration between the native
per-call peak and the sc14-tokenized items estimate. Writes targets-sc14-kimi.summary.json and
targets-sc14-kimi.per_task.csv next to the rows file; prints a text report.

usage: targets_summary.py [--focus id1,id2,...]
"""
import argparse
import csv
import json
import os
import statistics
from collections import defaultdict

OUT = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/analysis/deepswe-long-tasks-20261027"
ROWS = os.path.join(OUT, "targets-sc14-kimi.jsonl")
FOCUS = ["scriggo-method-declarations", "wasmi-trap-coredumps", "boa-hierarchical-evaluation-cancellation"]
COMPACT_AT = 242144
WINDOW = 262144


def q(vals, p):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def med(vals):
    v = [x for x in vals if x is not None]
    return statistics.median(v) if v else None


def pct(n, d):
    return round(100.0 * n / d, 1) if d else None


def ranks(per_task, key):
    """Competition rank (1 = longest) of every task by per_task[t][key]; tasks with None are unranked."""
    items = [(t, d[key]) for t, d in per_task.items() if d.get(key) is not None]
    items.sort(key=lambda x: -x[1])
    out, prev, prev_rank = {}, None, 0
    for i, (t, v) in enumerate(items, 1):
        r = prev_rank if v == prev else i
        out[t] = r
        prev, prev_rank = v, r
    return out, len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--focus", default=",".join(FOCUS))
    a = ap.parse_args()
    focus = a.focus.split(",")
    rows = [json.loads(l) for l in open(ROWS)]
    models = ["sc14", "kimi"]
    summary = {"rows_file": ROWS, "n_rows": len(rows), "models": {}, "focus": focus}
    per_task_all = {}
    for m in models:
        R = [r for r in rows if r["model"] == m]
        M = [r for r in R if r.get("peak_ctx") is not None]
        ov = {
            "rollouts": len(R), "rollouts_measured": len(M),
            "metric_methods": dict(sorted({r["metric_method"]: sum(1 for x in R if x["metric_method"] == r["metric_method"]) for r in R}.items())),
            "resolved": sum(r["resolved"] for r in R), "resolved_pct": pct(sum(r["resolved"] for r in R), len(R)),
            "status_counts": {s: sum(1 for r in R if r["status"] == s) for s in sorted({r["status"] for r in R})},
            "empty_patch_pct": pct(sum(r["empty_patch"] for r in R), len(R)),
            "agent_errors": {str(e): sum(1 for r in R if r["agent_error"] == e) for e in sorted({str(r["agent_error"]) for r in R}) if e != "None"},
            "median_calls": med([r.get("n_calls") for r in M]),
            "median_peak_ctx": med([r["peak_ctx"] for r in M]), "p90_peak_ctx": q([r["peak_ctx"] for r in M], 0.9),
            "max_peak_ctx": max(r["peak_ctx"] for r in M),
            "pct_peak_over_242k": pct(sum(r["peak_ctx"] > COMPACT_AT for r in M), len(M)),
            "pct_peak_over_262k": pct(sum(r["peak_ctx"] > WINDOW for r in M), len(M)),
            "pct_with_compaction": pct(sum((r.get("n_compactions") or 0) > 0 for r in M), len(M)),
            "median_items_tok_est_sc14": med([r["items_tok_est_sc14"] for r in M]),
            "p90_items_tok_est_sc14": q([r["items_tok_est_sc14"] for r in M], 0.9),
            "pct_items_tok_est_sc14_over_262k": pct(sum(r["items_tok_est_sc14"] > WINDOW for r in M), len(M)),
            "median_session_wall_min": round(med([r.get("session_wall_s") for r in M]) / 60, 1),
        }
        if m == "sc14":
            ov["median_uncompacted_len_est"] = med([r.get("uncompacted_len_est") for r in M])
            ov["pct_uncompacted_len_est_over_262k"] = pct(sum((r.get("uncompacted_len_est") or 0) > WINDOW for r in M), len(M))
            nc = [r for r in M if not r.get("n_compactions")]
            ov["calib_peak_over_items_no_compaction"] = {
                "n": len(nc), "median": round(med([r["peak_ctx"] / r["items_tok_est_sc14"] for r in nc]), 3),
                "p10": round(q([r["peak_ctx"] / r["items_tok_est_sc14"] for r in nc], 0.1), 3),
                "p90": round(q([r["peak_ctx"] / r["items_tok_est_sc14"] for r in nc], 0.9), 3)}
        else:
            ov["median_full_conv_len_native_est"] = med([r.get("full_conv_len_native_est") for r in M])
            ov["pct_full_conv_len_native_est_over_262k"] = pct(sum((r.get("full_conv_len_native_est") or 0) > WINDOW for r in M), len(M))
            one = [r for r in M if r.get("n_sessions") == 1 and not r.get("n_zero_token_calls")]
            ov["calib_full_conv_native_over_items"] = {
                "n": len(one), "median": round(med([r["full_conv_len_native_est"] / r["items_tok_est_sc14"] for r in one]), 3),
                "p10": round(q([r["full_conv_len_native_est"] / r["items_tok_est_sc14"] for r in one], 0.1), 3),
                "p90": round(q([r["full_conv_len_native_est"] / r["items_tok_est_sc14"] for r in one], 0.9), 3)}
            ov["rollouts_with_subagents"] = sum(1 for r in M if (r.get("n_sessions") or 1) > 1)
            ov["rollouts_subagent_peak_gt_main"] = sum(1 for r in M if (r.get("peak_ctx_any_session") or 0) > (r.get("peak_ctx") or 0))
            ov["rollouts_with_zero_usage_calls"] = sum(1 for r in M if r.get("n_zero_token_calls"))
        # per task
        by = defaultdict(list)
        for r in R:
            by[r["task_id"]].append(r)
        pt = {}
        for t, rr in by.items():
            mm = [r for r in rr if r.get("peak_ctx") is not None]
            pt[t] = {"task_index": rr[0]["task_index"], "n": len(rr), "resolved": sum(r["resolved"] for r in rr),
                     "median_peak_ctx": med([r["peak_ctx"] for r in mm]),
                     "median_items_tok_est_sc14": med([r["items_tok_est_sc14"] for r in mm]),
                     "median_calls": med([r.get("n_calls") for r in mm]),
                     "compacted_rollouts": sum((r.get("n_compactions") or 0) > 0 for r in mm),
                     "median_uncompacted_len_est": med([r.get("uncompacted_len_est") for r in mm]) if m == "sc14" else None,
                     "median_full_conv_len_native_est": med([r.get("full_conv_len_native_est") for r in mm]) if m == "kimi" else None}
        for key in ("median_peak_ctx", "median_items_tok_est_sc14", "median_uncompacted_len_est", "median_full_conv_len_native_est"):
            rk, n = ranks(pt, key)
            for t in pt:
                pt[t]["rank_" + key] = rk.get(t)
            ov["ranked_tasks_" + key] = n
        per_task_all[m] = pt
        # focus detail
        fd = {}
        for t in focus:
            rr = sorted(by.get(t, []), key=lambda r: r["repeat"])
            fd[t] = dict(pt.get(t, {}))
            fd[t]["rollouts"] = [{k: r.get(k) for k in (
                "slot", "status", "reward", "f2p", "p2p", "peak_ctx", "n_calls", "n_compactions", "patch_bytes",
                "agent_error", "last_finish", "session_wall_s", "items_tok_est_sc14", "uncompacted_len_est",
                "full_conv_len_native_est", "peak_ctx_any_session", "n_sessions", "compaction_trigger_ctx",
                "max_call_output_tokens")} for r in rr]
        ov["focus"] = fd
        summary["models"][m] = ov
    # per-task csv (both models side by side)
    tasks = sorted(per_task_all["sc14"], key=lambda t: per_task_all["sc14"][t]["task_index"])
    cols = ["resolved", "median_peak_ctx", "rank_median_peak_ctx", "median_items_tok_est_sc14",
            "rank_median_items_tok_est_sc14", "median_calls", "compacted_rollouts"]
    with open(os.path.join(OUT, "targets-sc14-kimi.per_task.csv.tmp"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "task_index"] + [f"{m}_{c}" for m in models for c in cols]
                   + ["sc14_median_uncompacted_len_est", "sc14_rank_median_uncompacted_len_est",
                      "kimi_median_full_conv_len_native_est", "kimi_rank_median_full_conv_len_native_est"])
        for t in tasks:
            s, k = per_task_all["sc14"][t], per_task_all["kimi"][t]
            w.writerow([t, s["task_index"]] + [d.get(c) for d in (s, k) for c in cols]
                       + [s["median_uncompacted_len_est"], s["rank_median_uncompacted_len_est"],
                          k["median_full_conv_len_native_est"], k["rank_median_full_conv_len_native_est"]])
    os.replace(os.path.join(OUT, "targets-sc14-kimi.per_task.csv.tmp"), os.path.join(OUT, "targets-sc14-kimi.per_task.csv"))
    tmp = os.path.join(OUT, "targets-sc14-kimi.summary.json.tmp")
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=1, sort_keys=False)
    os.replace(tmp, os.path.join(OUT, "targets-sc14-kimi.summary.json"))
    # report
    for m in models:
        ov = summary["models"][m]
        print(f"\n=== {m}: {ov['rollouts']} rollouts ({ov['rollouts_measured']} measured) methods={ov['metric_methods']}")
        print({k: v for k, v in ov.items() if k not in ("focus", "metric_methods")})
        for t in focus:
            d = ov["focus"][t]
            print(f"  -- {t}: resolved {d['resolved']}/{d['n']}, median peak {d['median_peak_ctx']} "
                  f"(rank {d['rank_median_peak_ctx']}/{ov['ranked_tasks_median_peak_ctx']}), median items_est "
                  f"{d['median_items_tok_est_sc14']} (rank {d['rank_median_items_tok_est_sc14']}), median calls {d['median_calls']}, "
                  f"unc {d.get('median_uncompacted_len_est')} (rank {d.get('rank_median_uncompacted_len_est')}), "
                  f"full {d.get('median_full_conv_len_native_est')} (rank {d.get('rank_median_full_conv_len_native_est')})")
            for r in d["rollouts"]:
                print("     ", r)


if __name__ == "__main__":
    main()
