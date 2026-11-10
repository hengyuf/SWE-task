#!/usr/bin/env python3
"""teacher-rollouts-v2 (2026-10-25): aggregate features/<source>.jsonl into the behaviour comparison (which teacher's
rollouts look most like base Nemotron sc14), plus pass rates and hub cost. Read-only on inputs.

    /usr/bin/python3 -I -B compare.py [--sources sc14 kimi glm deepseek sftv2] [--campaigns full|pilot|all]
      -> comparison.json, comparison.md (tables used in REPORT.md)

Turn-level percentages are pooled over all turns (long rollouts weigh more); "per rollout" values are means/medians
over rollouts. Distances to sc14 (smaller = closer): total-variation distance of the turn-class mix (action with /
without reasoning, no-tool reasoning-only / text / empty), and Kolmogorov-Smirnov distances of the per-turn
reasoning-token (0 for a turn without reasoning), tool-calls-per-turn and turns-per-rollout distributions.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import bisect  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import statistics  # noqa: E402

HERE = Path(__file__).resolve().parent
LABEL = {"sc14": "base Nemotron (sc14)", "sftv2": "v2 SFT (sc14 + kimi SFT)", "kimi": "Kimi-K3", "glm": "GLM-5.3",
         "deepseek": "DeepSeek-V4-Pro"}
CLASSES = ("action+reasoning", "action, no reasoning", "no-tool reasoning-only", "no-tool text", "no-tool empty")


def load(source, campaigns):
    path = HERE / "features" / f"{source}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    if source in ("glm", "deepseek") and campaigns != "all":
        rows = [r for r in rows if ((r.get("campaign") or "").endswith("-pilot")) == (campaigns == "pilot")]
    return rows


def pct(a, b):
    return round(100.0 * a / b, 1) if b else None


def med(xs):
    return statistics.median(xs) if xs else None


def mean(xs):
    return round(statistics.fmean(xs), 2) if xs else None


def quantile(xs, q):
    """Nearest-rank quantile (q in [0, 1]): the ceil(q * n)-th smallest value, rounded to 1 decimal."""
    if not xs:
        return None
    import math
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))], 1)


def ks(a, b):
    """Two-sample Kolmogorov-Smirnov statistic."""
    if not a or not b:
        return None
    a, b = sorted(a), sorted(b)
    points = sorted(set(a) | set(b))
    return round(max(abs(bisect.bisect_right(a, x) / len(a) - bisect.bisect_right(b, x) / len(b)) for x in points), 3)


def turn_classes(r):
    notool_text = r["text_turns"]
    return [r["action_with_reasoning"], r["action_without_reasoning"], r["reasoning_only_turns"], notool_text,
            r["empty_turns"]]


def per_turn_reasoning(rows):
    values = []
    for r in rows:
        values += r["reasoning_tokens"] + [0] * (r["turns"] - r["turns_with_reasoning"])
    return values


def per_turn_calls(rows):
    values = []
    for r in rows:
        for calls, count in (r.get("calls_hist") or {}).items():
            values += [int(calls)] * count
    return values


def summarize(source, rows, costs):
    graded = [r for r in rows if r["status"] in ("pass", "fail")]
    tasks = collections.defaultdict(list)
    for r in graded:
        tasks[r["task_id"]].append(r["status"] == "pass")
    ex = [r for r in rows if r.get("export") and r.get("turns")]
    T = sum(r["turns"] for r in ex)
    A = sum(r["action_turns"] for r in ex)
    N = sum(r["notool_turns"] for r in ex)
    rtok = [x for r in ex for x in r["reasoning_tokens"]]
    rchr = [x for r in ex for x in r["reasoning_chars"]]
    vis = [x for r in ex for x in r["visible_chars"]]
    tools = collections.Counter()
    for r in ex:
        tools.update(r["tools"])
    total_calls = sum(tools.values())
    classes = [sum(turn_classes(r)[i] for r in ex) for i in range(len(CLASSES))]
    commits = [r for r in ex if r["git_commit_calls"]]
    rollout_reasoned = [100 * r["action_with_reasoning"] / r["action_turns"] for r in ex if r["action_turns"]]
    out = {
        "source": source, "label": LABEL.get(source, source), "rollouts": len(rows), "graded": len(graded),
        "infra": len(rows) - len(graded), "with_export": len(ex),
        "pass_at_1": pct(sum(r["status"] == "pass" for r in graded), len(graded)),
        "task_mean_pass": round(100 * statistics.fmean([sum(v) / len(v) for v in tasks.values()]), 1) if tasks else None,
        "pass_at_5": pct(sum(any(v) for v in tasks.values()), len(tasks)), "tasks": len(tasks),
        "turns_per_rollout_mean": mean([r["turns"] for r in ex]), "turns_per_rollout_median": med([r["turns"] for r in ex]),
        "pct_turns_with_reasoning": pct(sum(r["turns_with_reasoning"] for r in ex), T),
        "pct_action_turns_with_reasoning": pct(sum(r["action_with_reasoning"] for r in ex), A),
        "pct_rollouts_all_actions_reasoned": pct(sum(r["all_actions_reasoned"] for r in ex), len(ex)),
        "rollout_mean_pct_action_reasoned": round(statistics.fmean(
            [100 * r["action_with_reasoning"] / r["action_turns"] for r in ex if r["action_turns"]]), 1) if ex else None,
        "longest_unreasoned_action_run_median": med([r["longest_unreasoned_action_run"] for r in ex]),
        # 2026-10-26: spread over rollouts (the median alone hides the tail)
        "longest_unreasoned_action_run_mean": mean([r["longest_unreasoned_action_run"] for r in ex]),
        "longest_unreasoned_action_run_p90": quantile([r["longest_unreasoned_action_run"] for r in ex], 0.9),
        "longest_unreasoned_action_run_max": max((r["longest_unreasoned_action_run"] for r in ex), default=None),
        "rollout_pct_action_reasoned_p10": quantile(rollout_reasoned, 0.1),
        "rollout_pct_action_reasoned_p50": quantile(rollout_reasoned, 0.5),
        "rollout_pct_action_reasoned_p90": quantile(rollout_reasoned, 0.9),
        "rollout_pct_action_reasoned_max": round(max(rollout_reasoned), 1) if rollout_reasoned else None,
        "pct_rollouts_action_reasoned_ge_90": pct(sum(x >= 90 for x in rollout_reasoned), len(rollout_reasoned)),
        "pct_first_action_reasoned": pct(sum(r["first_action_reasoning"] for r in ex), len(ex)),
        "reasoning_tokens_per_reasoning_turn_median": med(rtok), "reasoning_tokens_per_reasoning_turn_mean": mean(rtok),
        "reasoning_chars_per_reasoning_turn_median": med(rchr),
        "reasoning_tokens_per_turn_mean_incl_zero": round(sum(rtok) / T, 1) if T else None,
        "reasoning_tokens_per_rollout_mean": mean([r["reasoning_tokens_total"] for r in ex]),
        "reasoning_tokens_per_rollout_median": med([r["reasoning_tokens_total"] for r in ex]),
        "pct_turns_with_visible_text": pct(sum(len(r["visible_chars"]) for r in ex), T),
        "pct_action_turns_with_visible_text": pct(sum(r["action_with_visible"] for r in ex), A),
        "pct_action_turns_bare": pct(sum(r["action_bare"] for r in ex), A),
        "visible_chars_median": med(vis),
        "tool_calls_per_turn": round(sum(r["tool_calls"] for r in ex) / T, 2) if T else None,
        "tool_calls_per_action_turn": round(sum(r["tool_calls"] for r in ex) / A, 2) if A else None,
        "pct_turns_parallel_calls": pct(sum(r["parallel_turns"] for r in ex), T),
        "tool_calls_per_rollout_mean": mean([r["tool_calls"] for r in ex]),
        "pct_turns_notool": pct(N, T),
        "pct_turns_reasoning_only": pct(sum(r["reasoning_only_turns"] for r in ex), T),
        "pct_turns_text_notool": pct(sum(r["text_turns"] for r in ex), T),
        "pct_turns_empty": pct(sum(r["empty_turns"] for r in ex), T),
        "turn_class_mix": {name: pct(count, T) for name, count in zip(CLASSES, classes)},
        "pct_rollouts_git_commit": pct(len(commits), len(ex)),
        "pct_rollouts_commit_in_last_3_calls": pct(sum(r["last_commit_calls_from_end"] is not None and
                                                       r["last_commit_calls_from_end"] <= 2 for r in ex), len(ex)),
        "pct_rollouts_branch_created": pct(sum(r["branch_created"] for r in ex), len(ex)),
        "pct_rollouts_final_notool_message": pct(sum(r["final_notool"] for r in ex), len(ex)),
        "pct_rollouts_final_summary_200": pct(sum(r["final_visible_chars"] >= 200 for r in ex), len(ex)),
        "final_message_chars_median": med([r["final_visible_chars"] for r in ex if r["final_notool"]]),
        "pct_rollouts_final_reasoning": pct(sum(r["final_reasoning"] for r in ex if r["final_notool"]),
                                            sum(r["final_notool"] for r in ex)),
        "pct_rollouts_todowrite": pct(sum(r["todowrite_calls"] > 0 for r in ex), len(ex)),
        "todowrite_calls_per_rollout_mean": mean([r["todowrite_calls"] for r in ex]),
        "pct_rollouts_task_subagent": pct(sum(r["task_calls"] > 0 for r in ex), len(ex)),
        "task_calls_per_rollout_mean": mean([r["task_calls"] for r in ex]),
        "pct_rollouts_webfetch": pct(sum(r["webfetch_calls"] > 0 for r in ex), len(ex)),
        "webfetch_calls_total": sum(r["webfetch_calls"] for r in ex),
        "pct_rollouts_skill": pct(sum(r["skill_calls"] > 0 for r in ex), len(ex)),
        "compactions_per_rollout_mean": mean([r["compactions"] for r in ex]),
        "pct_rollouts_compacted": pct(sum(r["compactions"] > 0 for r in ex), len(ex)),
        "tool_share": {name: pct(count, total_calls) for name, count in tools.most_common()},
        "agent_wall_min_median": round(med([r["agent_wall_s"] for r in ex if r.get("agent_wall_s")]) / 60, 1)
        if any(r.get("agent_wall_s") for r in ex) else None,
        "patch_empty_pct": pct(sum(not r.get("patch_chars") for r in graded), len(graded)),
    }
    by_tool = collections.defaultdict(lambda: [0, 0])
    openers = collections.Counter()
    for r in ex:
        for name, (yes, total) in (r.get("reasoned_by_first_tool") or {}).items():
            by_tool[name][0] += yes
            by_tool[name][1] += total
        openers.update(r.get("reasoning_openers") or {})
    out["pct_action_reasoned_by_first_tool"] = {name: [pct(yes, total), total] for name, (yes, total) in
                                                sorted(by_tool.items(), key=lambda kv: -kv[1][1])}
    out["top_reasoning_openers"] = openers.most_common(8)
    deciles = [[0, 0] for _ in range(10)]
    for r in ex:
        for d, (yes, total) in enumerate(r.get("reasoned_by_decile") or []):
            deciles[d][0] += yes
            deciles[d][1] += total
    out["pct_action_reasoned_by_decile"] = [pct(yes, total) for yes, total in deciles]
    after = collections.defaultdict(lambda: [0, 0])
    for r in ex:
        for key, (yes, total) in (r.get("reasoned_after_prev") or {}).items():
            after[key][0] += yes
            after[key][1] += total
    out["pct_action_reasoned_after"] = {key: pct(yes, total) for key, (yes, total) in after.items()}
    out["pct_unreasoned_action_after_reasoned"] = pct(sum(r.get("unreasoned_after_reasoned", 0) for r in ex),
                                                      sum(r["action_without_reasoning"] for r in ex))
    if costs:
        out["hub_cost"] = costs
    return out


def cost_summary(source, campaigns):
    """Hub cost per slot: costs/<model>-<campaign>.json (costs.py; all attempts of a slot) for glm/deepseek, and
    costs/kimi-r0.json (kimi_costs.py; final attempt of repeat 0 of every task, LiteLLM-reported cost) for kimi."""
    files = sorted((HERE / "costs").glob(f"{source}-*.json"))
    if source in ("glm", "deepseek") and campaigns != "all":
        files = [f for f in files if f.stem.endswith("-pilot") == (campaigns == "pilot")]
    slots = []
    by_key = collections.defaultdict(float)  # costs.py "by_key" (2026-10-26): reply cost by the hub key NAME of its leg
    slot_keys = collections.Counter()  # slots by the key(s) their replies ran on ("A+B" = split across legs)
    for f in files:
        for item in json.loads(f.read_text()):
            if "slots" in item:  # costs.py campaign record
                slots += [{"cost": s["total"]["cost"], **{k: s["total"][k] for k in ("prompt", "cached", "completion", "calls")}}
                          for s in item["slots"]]
                for s in item["slots"]:
                    split = s["total"].get("by_key") or {}
                    for key, values in split.items():
                        by_key[key] += values["cost"]
                    if split:
                        slot_keys["+".join(sorted(k for k, v in split.items() if v["calls"]))] += 1
            else:  # kimi_costs.py row
                slots.append({k: item[k] for k in ("cost", "prompt", "cached", "completion", "calls")})
    if not slots:
        return None
    totals = [s["cost"] for s in slots]
    prompt = sum(s["prompt"] for s in slots)
    out = {"slots": len(slots), "usd_total": round(sum(totals), 2), "usd_per_slot_mean": round(statistics.fmean(totals), 3),
           "usd_per_slot_median": round(statistics.median(totals), 3),
           "prompt_Mtok_per_slot": round(prompt / len(slots) / 1e6, 2),
           "cached_pct": round(100 * sum(s["cached"] for s in slots) / max(1, prompt), 1),
           "completion_ktok_per_slot": round(sum(s["completion"] for s in slots) / len(slots) / 1e3, 1),
           "calls_per_slot": round(sum(s["calls"] for s in slots) / len(slots), 1)}
    if by_key:
        out["usd_by_key"] = {key: round(value, 2) for key, value in sorted(by_key.items())}
        out["slots_by_key"] = dict(sorted(slot_keys.items()))
    return out


def bootstrap(rows, numerator, denominator, n=2000, seed=20261025):
    """95% interval of a pooled ratio (percent) by resampling tasks with replacement (task = cluster)."""
    import random
    by_task = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("export") and r.get("turns"):
            by_task[r["task_id"]][0] += numerator(r)
            by_task[r["task_id"]][1] += denominator(r)
    values = list(by_task.values())
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        sample = [values[rng.randrange(len(values))] for _ in values]
        den = sum(v[1] for v in sample)
        if den:
            stats.append(100.0 * sum(v[0] for v in sample) / den)
    stats.sort()
    return [round(stats[int(0.025 * len(stats))], 1), round(stats[int(0.975 * len(stats)) - 1], 1)]


CI_METRICS = {
    "pct_action_turns_with_reasoning": (lambda r: r["action_with_reasoning"], lambda r: r["action_turns"]),
    "pct_action_turns_with_visible_text": (lambda r: r["action_with_visible"], lambda r: r["action_turns"]),
    "pct_turns_parallel_calls": (lambda r: r["parallel_turns"], lambda r: r["turns"]),
}


def distances(base, rows):
    bx = [r for r in base if r.get("export") and r.get("turns")]
    rx = [r for r in rows if r.get("export") and r.get("turns")]
    if not bx or not rx:
        return None
    bc = [sum(turn_classes(r)[i] for r in bx) for i in range(len(CLASSES))]
    rc = [sum(turn_classes(r)[i] for r in rx) for i in range(len(CLASSES))]
    tv = 0.5 * sum(abs(b / sum(bc) - c / sum(rc)) for b, c in zip(bc, rc))
    return {"turn_class_tv": round(tv, 3),
            "ks_reasoning_tokens_per_turn": ks(per_turn_reasoning(bx), per_turn_reasoning(rx)),
            "ks_tool_calls_per_turn": ks(per_turn_calls(bx), per_turn_calls(rx)),
            "ks_turns_per_rollout": ks([r["turns"] for r in bx], [r["turns"] for r in rx]),
            "ks_reasoning_tokens_per_rollout": ks([r["reasoning_tokens_total"] for r in bx],
                                                  [r["reasoning_tokens_total"] for r in rx])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=["sc14", "kimi", "glm", "deepseek", "sftv2"])
    parser.add_argument("--campaigns", choices=("full", "pilot", "all"), default="full")
    parser.add_argument("--out", default="comparison")
    parser.add_argument("--tasks", nargs="*", help="restrict every source to these task ids")
    args = parser.parse_args()
    data = {source: load(source, args.campaigns) for source in args.sources}
    if args.tasks:
        data = {source: [r for r in rows if r["task_id"] in set(args.tasks)] for source, rows in data.items()}
    summary = {source: summarize(source, rows, cost_summary(source, args.campaigns)) for source, rows in data.items() if rows}
    for source in summary:
        summary[source]["ci95_by_task"] = {name: bootstrap(data[source], *fns) for name, fns in CI_METRICS.items()}
    base = data.get("sc14") or []
    for source in summary:
        if source != "sc14":
            summary[source]["distance_to_sc14"] = distances(base, data[source])
    (HERE / f"{args.out}.json").write_text(json.dumps(summary, indent=1) + "\n")
    keys = [k for k in next(iter(summary.values())) if not isinstance(next(iter(summary.values()))[k], (dict, list))
            and k not in ("source", "label")]
    lines = ["| metric | " + " | ".join(summary[s]["label"] for s in summary) + " |",
             "|---|" + "---|" * len(summary)]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(str(summary[s].get(k)) for s in summary) + " |")
    for k in ("turn_class_mix",):
        for name in CLASSES:
            lines.append(f"| {k}: {name} | " + " | ".join(str(summary[s][k].get(name)) for s in summary) + " |")
    for name in CI_METRICS:
        lines.append(f"| 95% CI (task bootstrap): {name} | " + " | ".join(
            str(summary[s]["ci95_by_task"].get(name)) for s in summary) + " |")
    dist_keys = ["turn_class_tv", "ks_reasoning_tokens_per_turn", "ks_tool_calls_per_turn", "ks_turns_per_rollout",
                 "ks_reasoning_tokens_per_rollout"]
    for k in dist_keys:
        lines.append(f"| distance to sc14: {k} | " + " | ".join(
            str((summary[s].get("distance_to_sc14") or {}).get(k, "-")) for s in summary) + " |")
    tool_names = ["bash", "read", "edit", "grep", "glob", "write", "todowrite", "task", "webfetch", "skill"]
    for name in tool_names:
        lines.append(f"| tool share %: {name} | " + " | ".join(str(summary[s]["tool_share"].get(name, 0)) for s in summary) + " |")
    for name in ("bash", "read", "edit", "grep", "write", "todowrite", "glob", "task", "webfetch"):
        lines.append(f"| % action turns reasoned, first tool {name} | " + " | ".join(
            str((summary[s]["pct_action_reasoned_by_first_tool"].get(name) or ["-"])[0]) for s in summary) + " |")
    for key in ("prev_reasoned", "prev_unreasoned", "prev_ok", "prev_error"):
        lines.append(f"| % action turns reasoned, {key} | " + " | ".join(
            str(summary[s]["pct_action_reasoned_after"].get(key)) for s in summary) + " |")
    for d in range(10):
        lines.append(f"| % action turns reasoned, rollout decile {d + 1} | " + " | ".join(
            str(summary[s]["pct_action_reasoned_by_decile"][d]) for s in summary) + " |")
    for s in summary:
        lines.append(f"\n{summary[s]['label']} top reasoning openers: {summary[s]['top_reasoning_openers']}")
    for s in summary:
        if summary[s].get("hub_cost"):
            lines.append(f"\n{summary[s]['label']} hub cost: {json.dumps(summary[s]['hub_cost'])}")
    (HERE / f"{args.out}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
