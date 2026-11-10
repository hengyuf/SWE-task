#!/usr/bin/env python3
"""Summarize <OUT>/<run>.jsonl (from evalrun_lengths.py): run-wide length distribution, per-task medians/ranks,
and per-rollout detail for the three long DeepSWE tasks. Reads only files under OUT plus the run's per_task.csv.

Writes <OUT>/<run>.per_task.csv for each run and <OUT>/evalruns_summary_emptythink-5ep-10ep_base-step0.json; prints a readable digest.
usage: evalrun_summary.py RUN [RUN ...]
"""
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict

RUNS_DIR = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/eval/deepswe-v1.1/runs"
OUT = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/analysis/deepswe-long-tasks-20261027"
FOCUS = ["scriggo-method-declarations", "boa-hierarchical-evaluation-cancellation", "wasmi-trap-coredumps"]
COMPACT_AT = 242144
WINDOW = 262144


def pct(xs, q):
    """Linear-interpolated percentile (q in [0,100])."""
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def r0(x):
    return None if x is None else int(round(x))


def ordinal(n):
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def rank_desc(values, key):
    """1 + number of tasks with a strictly larger value (1 = longest)."""
    v = values[key]
    if v is None:
        return None
    return 1 + sum(1 for k, x in values.items() if x is not None and x > v)


def agent_error(x):
    parts = []
    if x.get("failure_reason"):
        parts.append(f"failure_reason={str(x['failure_reason'])[:90]}")
    if x.get("failure_kind"):
        parts.append(f"failure_kind={x['failure_kind']}")
    if not x.get("export_ok") and not x.get("n_output_items"):
        parts.append("no agent transcript (infra)")
    elif not x.get("opencode_finished"):
        last = x.get("last_finish")
        wall = x.get("session_wall_s")
        why = "3h agent timeout" if wall and wall >= 10000 else f"opencode not finished (wall {wall or 0:,.0f} s)"
        parts.append(why + (" (killed mid-call)" if last in (None, "None") else ""))
    if x.get("n_error_msgs"):
        parts.append(f"{x['n_error_msgs']} assistant error msg(s): {'; '.join(x.get('errors') or [])[:120]}")
    fc = x.get("finish_counts") or {}
    if fc.get("length"):
        parts.append(f"{fc['length']} call(s) finish=length")
    if x.get("verifier_error"):
        parts.append(f"verifier_error={str(x['verifier_error'])[:70]}")
    return "; ".join(parts) if parts else "none"


def load(run):
    rows = [json.loads(l) for l in open(f"{OUT}/{run}.jsonl")]
    meta = {}
    with open(f"{RUNS_DIR}/{run}/per_task.csv") as f:
        for t in csv.DictReader(f):
            meta[t["task_id"]] = t
    return rows, meta


def summarize(run):
    rows, meta = load(run)
    n = len(rows)
    keys = Counter((x["task_index"], x["rollout_index"]) for x in rows)
    dup = sum(1 for v in keys.values() if v > 1)
    meas = [x for x in rows if x.get("peak_ctx") is not None and x.get("metric_method") == "export_peak_ctx"]
    meas_any = [x for x in rows if x.get("peak_ctx") is not None]
    pk = [x["peak_ctx"] for x in meas]
    calls = [x["n_calls"] for x in meas]
    comp = [x["n_compactions"] for x in meas]
    unc = [x["uncompacted_len_est"] for x in meas if x.get("uncompacted_len_est") is not None]
    overall = {
        "rollouts": n,
        "rollouts_measured_from_export": len(meas),
        "rollouts_measured_any_method": len(meas_any),
        "metric_methods": dict(Counter(x.get("metric_method") for x in rows)),
        "duplicate_task_rollout_keys": dup,
        "resolved_pct": round(100 * sum(x["resolved"] for x in rows) / n, 2),
        "empty_patch_pct": round(100 * sum((x.get("model_patch_bytes") or 0) == 0 for x in rows) / n, 2),
        "opencode_not_finished_pct": round(100 * sum(not x.get("opencode_finished") for x in rows) / n, 2),
        "not_finished_with_patch": sum(1 for x in rows if not x.get("opencode_finished") and (x.get("model_patch_bytes") or 0) > 0),
        "not_finished_resolved": sum(1 for x in rows if not x.get("opencode_finished") and x["resolved"]),
        "infra_no_transcript": sum(1 for x in rows if not x.get("export_ok") and not x.get("n_output_items")),
        "median_wall_s": r0(statistics.median([x["session_wall_s"] for x in meas if x.get("session_wall_s") is not None])),
        "median_calls": statistics.median(calls),
        "mean_calls": round(statistics.mean(calls), 1),
        "p90_calls": r0(pct(calls, 90)),
        "median_peak_ctx": r0(statistics.median(pk)),
        "mean_peak_ctx": r0(statistics.mean(pk)),
        "p10_peak_ctx": r0(pct(pk, 10)), "p25_peak_ctx": r0(pct(pk, 25)), "p75_peak_ctx": r0(pct(pk, 75)),
        "p90_peak_ctx": r0(pct(pk, 90)), "p95_peak_ctx": r0(pct(pk, 95)), "max_peak_ctx": max(pk),
        "pct_peak_over_242k": round(100 * sum(p > COMPACT_AT for p in pk) / len(pk), 2),
        "pct_peak_at_window_262144": round(100 * sum(p >= WINDOW for p in pk) / len(pk), 2),
        "pct_with_compaction": round(100 * sum(c > 0 for c in comp) / len(comp), 2),
        "compactions_distribution": dict(sorted(Counter(comp).items())),
        "median_uncompacted_len_est": r0(statistics.median(unc)),
        "p90_uncompacted_len_est": r0(pct(unc, 90)),
        "pct_rollouts_with_finish_length": round(100 * sum(bool((x.get("finish_counts") or {}).get("length")) for x in meas) / len(meas), 2),
        "resolved_pct_by_compaction": {
            "no_compaction": round(100 * statistics.mean([x["resolved"] for x in meas if x["n_compactions"] == 0]), 2),
            "with_compaction": round(100 * statistics.mean([x["resolved"] for x in meas if x["n_compactions"] > 0]), 2),
        },
    }
    # per task
    by_task = defaultdict(list)
    for x in rows:
        by_task[x["task_id"]].append(x)
    med_pk, med_calls, med_unc, mean_pk, tot_comp = {}, {}, {}, {}, {}
    for t, xs in by_task.items():
        m = [x for x in xs if x.get("peak_ctx") is not None and x.get("metric_method") == "export_peak_ctx"]
        med_pk[t] = statistics.median([x["peak_ctx"] for x in m]) if m else None
        mean_pk[t] = statistics.mean([x["peak_ctx"] for x in m]) if m else None
        med_calls[t] = statistics.median([x["n_calls"] for x in m]) if m else None
        uu = [x["uncompacted_len_est"] for x in m if x.get("uncompacted_len_est") is not None]
        med_unc[t] = statistics.median(uu) if uu else None
        tot_comp[t] = statistics.mean([x["n_compactions"] for x in m]) if m else None
    ntasks = len(by_task)
    per_task = []
    for t, xs in by_task.items():
        m = [x for x in xs if x.get("peak_ctx") is not None and x.get("metric_method") == "export_peak_ctx"]
        per_task.append({
            "task_id": t, "language": meta.get(t, {}).get("language", ""),
            "rollouts": len(xs), "measured": len(m),
            "resolved": sum(x["resolved"] for x in xs),
            "per_task_csv_resolved": meta.get(t, {}).get("resolved"),
            "median_peak_ctx": r0(med_pk[t]), "mean_peak_ctx": r0(mean_pk[t]),
            "rank_median_peak_ctx": rank_desc(med_pk, t),
            "median_calls": med_calls[t], "rank_median_calls": rank_desc(med_calls, t),
            "median_uncompacted_len_est": r0(med_unc[t]), "rank_median_uncompacted_len_est": rank_desc(med_unc, t),
            "rollouts_with_compaction": sum(1 for x in m if x["n_compactions"] > 0),
            "total_compactions": sum(x["n_compactions"] for x in m),
            "rank_mean_compactions": rank_desc(tot_comp, t),
            "median_wall_s": r0(statistics.median([x["session_wall_s"] for x in m if x.get("session_wall_s") is not None])) if m else None,
            "empty_patches": sum(1 for x in xs if (x.get("model_patch_bytes") or 0) == 0),
            "not_finished": sum(1 for x in xs if not x.get("opencode_finished")),
        })
    per_task.sort(key=lambda d: (d["rank_median_peak_ctx"] or 10**9, d["task_id"]))
    with open(f"{OUT}/{run}.per_task.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_task[0]))
        w.writeheader()
        w.writerows(per_task)
    mismatch = [d["task_id"] for d in per_task if d["per_task_csv_resolved"] not in (None, "") and int(d["per_task_csv_resolved"]) != d["resolved"]]
    # by language
    by_lang = defaultdict(list)
    for x in meas:
        by_lang[meta.get(x["task_id"], {}).get("language", "")].append(x)
    lang = {
        k: {"rollouts": len(v), "median_peak_ctx": r0(statistics.median([x["peak_ctx"] for x in v])),
            "pct_with_compaction": round(100 * sum(x["n_compactions"] > 0 for x in v) / len(v), 1),
            "median_calls": statistics.median([x["n_calls"] for x in v]),
            "resolved_pct": round(100 * statistics.mean([x["resolved"] for x in v]), 1)}
        for k, v in sorted(by_lang.items())
    }
    # focus tasks
    pt = {d["task_id"]: d for d in per_task}
    focus = []
    for t in FOCUS:
        xs = sorted(by_task.get(t, []), key=lambda x: x["rollout_index"])
        d = pt[t]
        lines = []
        for x in xs:
            pkv = x.get("peak_ctx")
            meth = "" if x.get("metric_method") == "export_peak_ctx" else f" [{x.get('metric_method')}]"
            lines.append(
                f"r{x['rollout_index']} {'pass' if x['resolved'] else 'fail'}, peak_ctx {pkv:,}{meth}, calls {x.get('n_calls')}, "
                f"compactions {x.get('n_compactions')}, patch {x.get('model_patch_bytes'):,} B, agent error: {agent_error(x)} "
                f"| wall {(x.get('session_wall_s') or 0):,.0f} s, f2p {x.get('f2p_passed')}/{x.get('f2p_total')}, "
                f"p2p {x.get('p2p_passed')}/{x.get('p2p_total')}, uncompacted-length est ~{(x.get('uncompacted_len_est') or 0):,}"
                if pkv is not None else
                f"r{x['rollout_index']} {'pass' if x['resolved'] else 'fail'}, peak_ctx n/a (no export.json, empty "
                f"response.output: sandbox infra failure, not scored on length), patch {x.get('model_patch_bytes')} B, "
                f"agent error: {agent_error(x)}"
            )
        focus.append({
            "task_id": t, "language": d["language"], "resolved": f"{d['resolved']}/{d['rollouts']}",
            "median_peak_ctx": d["median_peak_ctx"], "mean_peak_ctx": d["mean_peak_ctx"],
            "peak_ctx_rank_among_tasks": f"{ordinal(d['rank_median_peak_ctx'])} longest of {ntasks} by median peak_ctx",
            "median_calls": d["median_calls"], "calls_rank": f"{ordinal(d['rank_median_calls'])} of {ntasks} by median n_calls",
            "median_uncompacted_len_est": d["median_uncompacted_len_est"],
            "uncompacted_rank": f"{ordinal(d['rank_median_uncompacted_len_est'])} of {ntasks} by median uncompacted_len_est",
            "rollouts_with_compaction": d["rollouts_with_compaction"], "total_compactions": d["total_compactions"],
            "compactions_rank": f"{ordinal(d['rank_mean_compactions'])} of {ntasks} by mean n_compactions",
            "median_wall_s": d["median_wall_s"],
            "empty_patches": d["empty_patches"], "not_finished": d["not_finished"],
            "rollouts": lines,
        })
    return {
        "run": run, "overall": overall, "by_language": lang, "focus": focus,
        "top10_tasks_by_median_peak_ctx": [(d["task_id"], d["median_peak_ctx"], d["median_calls"], d["median_uncompacted_len_est"]) for d in per_task[:10]],
        "top10_tasks_by_median_calls": [
            (d["task_id"], d["median_calls"], d["median_peak_ctx"], d["median_uncompacted_len_est"])
            for d in sorted(per_task, key=lambda d: d["rank_median_calls"] or 10**9)[:10]],
        "top10_tasks_by_median_uncompacted_len_est": [
            (d["task_id"], d["median_uncompacted_len_est"], d["median_peak_ctx"], d["median_calls"])
            for d in sorted(per_task, key=lambda d: d["rank_median_uncompacted_len_est"] or 10**9)[:10]],
        "per_task_resolved_mismatch_vs_per_task_csv": mismatch,
        "n_tasks": ntasks,
    }


def main():
    runs = sys.argv[1:]
    res = [summarize(r) for r in runs]
    with open(f"{OUT}/evalruns_summary_emptythink-5ep-10ep_base-step0.json", "w") as f:
        json.dump(res, f, indent=1)
    for s in res:
        print("=" * 100)
        print(s["run"])
        print(json.dumps(s["overall"], indent=1))
        print("by_language", json.dumps(s["by_language"]))
        print("top10 by median peak_ctx:", s["top10_tasks_by_median_peak_ctx"])
        print("top10 by median calls:", s["top10_tasks_by_median_calls"])
        print("top10 by median uncompacted est:", s["top10_tasks_by_median_uncompacted_len_est"])
        print("per_task resolved mismatches:", s["per_task_resolved_mismatch_vs_per_task_csv"])
        for f_ in s["focus"]:
            print("--", {k: v for k, v in f_.items() if k != "rollouts"})
            for l in f_["rollouts"]:
                print("   ", l)


if __name__ == "__main__":
    main()
