#!/usr/bin/env python3
"""Extra views of cache/ (written by traffic_scan.py scan): per-task table of the 50 synthetic tasks (tr2), tool gaps
between calls, campaign-level concurrency timelines, clamp checks. Read-only on everything but this directory.

    /usr/bin/python3 -I -B traffic_extra.py   -> traffic-extra.json, per-task-tr2.tsv
"""
import collections
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
TA_TASKS = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024/tasks.json")


def pct(values, q):
    values = sorted(values)
    if not values:
        return None
    return values[min(len(values) - 1, max(0, int(round(q / 100 * (len(values) - 1)))))]


def dist(values, digits=1):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    r = lambda v: round(v, digits)
    return {"n": len(values), "mean": r(statistics.mean(values)), "p10": r(pct(values, 10)), "p50": r(pct(values, 50)),
            "p90": r(pct(values, 90)), "p99": r(pct(values, 99)), "max": r(max(values))}


def timeline(intervals):
    """Time-weighted distribution of the number of open intervals (start, end)."""
    events = sorted([(s, 1) for s, e in intervals] + [(e, -1) for s, e in intervals])
    level, last, weights = 0, None, collections.Counter()
    for t, d in events:
        if last is not None and t > last and level > 0:
            weights[level] += t - last
        level += d
        last = t
    total = sum(weights.values())
    if not total:
        return {}
    acc, out = 0.0, {}
    for q in (50, 90, 99):
        acc = 0.0
        for level_value in sorted(weights):
            acc += weights[level_value]
            if acc >= total * q / 100:
                out[f"p{q}"] = level_value
                break
    out["max"] = max(weights)
    out["mean_when_busy"] = round(sum(k * v for k, v in weights.items()) / total, 2)
    return out


def main():
    result = {}
    tasks = json.loads(TA_TASKS.read_text())["tasks"]
    order = [t["task_id"] for t in tasks]
    for campaign in ("tr2", "tdw"):
        attempts = [json.loads(p.read_text()) for p in sorted((HERE / "cache" / campaign).glob("*.json"))]
        gaps, rollout_iv, call_iv = [], [], []
        clamp_trunc = over_262k = 0
        per_task = collections.defaultdict(list)
        for a in attempts:
            calls = sorted((c for c in a["calls"] if c.get("status") == 200 and c.get("pt") is not None),
                           key=lambda c: c["t0"])
            if not calls:
                continue
            for x, y in zip(calls, calls[1:]):
                if x.get("kind") == "main" and y.get("kind") == "main":
                    gaps.append(max(0.0, y["t0"] - x["t1"]))
            rollout_iv.append((calls[0]["t0"], calls[-1]["t1"]))
            call_iv += [(c["t0"], c["t1"]) for c in calls]
            clamp_trunc += sum(c["pt"] + (c["out"] or 0) > 262144 for c in calls)
            over_262k += sum(c["pt"] > 262144 for c in calls)
            per_task[a["task_id"]].append({
                "slot": a["slot"], "status": a["status"], "calls": len(calls),
                "max_prompt": max(c["pt"] for c in calls),
                "wall_min": (calls[-1]["t1"] - calls[0]["t0"]) / 60,
                "out_sum": sum(c["out"] or 0 for c in calls)})
        # concurrency per campaign group (tr2: 8 groups ran in parallel at different times -> whole campaign)
        result[campaign] = {
            "inter_call_tool_gap_s": dist(gaps),
            "rollouts_in_flight_timeline": timeline(rollout_iv),
            "llm_calls_in_flight_timeline": timeline(call_iv),
            "calls_where_prompt_plus_output_gt_262144": clamp_trunc,
            "calls_prompt_gt_262144": over_262k,
        }
        if campaign == "tr2":
            rows = ["group\ttask_id\thub_pass_of_5\tslots\tmax_prompt_max\tmax_prompt_median\tcalls_median\t"
                    "wall_min_median\twall_min_max\tout_tokens_median"]
            safe = []
            for index, task_id in enumerate(order):
                recs = per_task.get(task_id, [])
                if not recs:
                    continue
                passes = sum(r["status"] == "pass" for r in recs)
                mp = [r["max_prompt"] for r in recs]
                rows.append("\t".join(map(str, [index % 8, task_id, passes, "".join(
                    "P" if r["status"] == "pass" else "." if r["status"] == "fail" else "x"
                    for r in sorted(recs, key=lambda r: r["slot"])), max(mp), int(statistics.median(mp)),
                    int(statistics.median(r["calls"] for r in recs)),
                    round(statistics.median(r["wall_min"] for r in recs), 1), round(max(r["wall_min"] for r in recs), 1),
                    int(statistics.median(r["out_sum"] for r in recs))])))
                if max(mp) <= 200000:
                    safe.append({"task_id": task_id, "group": index % 8, "hub_pass_of_5": passes, "max_prompt": max(mp),
                                 "wall_min_median": round(statistics.median(r["wall_min"] for r in recs), 1)})
            (HERE / "per-task-tr2.tsv").write_text("\n".join(rows) + "\n")
            result["tr2"]["tasks_all_5_hub_rollouts_max_prompt_le_200k"] = safe
            result["tr2"]["tasks_all_5_le_200k_count"] = len(safe)
            result["tr2"]["tasks_all_5_le_200k_hub_pass_rate"] = round(
                sum(s["hub_pass_of_5"] for s in safe) / (5 * len(safe)), 4) if safe else None
            # hub pass rate split by whether the rollout stayed under the 262,144-window compaction point
            under = [r for recs in per_task.values() for r in recs if r["max_prompt"] <= 242144 and r["status"] != "infra"]
            over = [r for recs in per_task.values() for r in recs if r["max_prompt"] > 242144 and r["status"] != "infra"]
            result["tr2"]["hub_pass_rate_rollouts_max_prompt_le_242144"] = {
                "n": len(under), "pass_rate": round(sum(r["status"] == "pass" for r in under) / len(under), 4)}
            result["tr2"]["hub_pass_rate_rollouts_max_prompt_gt_242144"] = {
                "n": len(over), "pass_rate": round(sum(r["status"] == "pass" for r in over) / len(over), 4)}
    (HERE / "traffic-extra.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
