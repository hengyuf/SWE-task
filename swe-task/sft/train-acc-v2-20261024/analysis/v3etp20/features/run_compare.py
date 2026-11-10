#!/usr/bin/env python3
"""train-acc-v2 (2026-10-25): behaviour comparison of base sc14 and the v2 SFT checkpoints (5 / 10 / 20 epochs) on the
50 training tasks, with the teacher-rollouts-v2 aggregation code (TR/compare.py, imported read-only, never changed).

    python3 -I -B run_compare.py  -> TA/analysis/features/comparison.{json,md}

Inputs: TR/features/{sc14,sftv2}.jsonl (computed by TR/features.py) and TA/analysis/features/{sftv2e10,sftv2e20}.jsonl
(run_features.py: the same extractor, reproduces TR/features/sftv2.jsonl byte for byte). Per source: TR compare's
summarize(), its task-bootstrap 95% CIs (CI_METRICS) and distances to sc14; added here: longest unreasoned action
run (mean / p90 / max), and paired task-bootstrap 95% CIs of the difference in % action turns with reasoning vs the
5-epoch model and vs base (resampling the 50 tasks, the same draw for both models).

2026-10-27: + etp5 / etp10 (empty-think-prefix SFT 5 / 10 ep, run_features.py --sources etp5 etp10), the paired CI vs
the raw v2 checkpoint of the same epochs (diff_pct_action_reasoned_vs_same_epoch_raw), and per source: reasoning-token
quantiles per reasoning turn (nearest rank), % reasoning turns under 20 / 50 tokens, hub output tokens per rollout,
agent wall time mean / p90, and rollouts that hit the 3 h deadline (agent_error TimeoutError).

2026-11-04: copy in analysis/v3etp20/features, + v3etp20 (paired CI vs the raw v2 20 ep run of the same epochs).
"""
import sys

sys.dont_write_bytecode = True

import collections  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import random  # noqa: E402
import statistics  # noqa: E402

HERE = Path(__file__).resolve().parent
ORIG = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024/analysis/features")  # 2026-11-04
TR = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-rollouts-v2-20261025")
sys.path.insert(0, str(TR))
import compare as tr_compare  # noqa: E402  (TR/compare.py)

SOURCES = {"sc14": TR / "features/sc14.jsonl", "sftv2": TR / "features/sftv2.jsonl",
           "sftv2e10": ORIG / "sftv2e10.jsonl", "sftv2e20": ORIG / "sftv2e20.jsonl"}
LABEL = {"sc14": "base (sc14)", "sftv2": "v2 SFT 5 ep (iter38)", "sftv2e10": "v2 SFT 10 ep (iter76)",
         "sftv2e20": "v2 SFT 20 ep (iter152)"}
ETP = {  # empty-think-prefix checkpoints, one line each: model -> (label, raw v2 model of the same epochs)
    "etp5": ("v2 empty-think-prefix 5 ep (iter38)", "sftv2"),
    "etp10": ("v2 empty-think-prefix 10 ep (iter76)", "sftv2e10"),
    "etp20": ("v2 empty-think-prefix 20 ep (iter152)", "sftv2e20"),
    "v3etp5": ("v3 empty-think-prefix 5 ep (iter71)", "sftv2"),  # 2026-10-31: SFT-v3 corpus
    "v3etp10": ("v3 empty-think-prefix 10 ep (iter142)", "sftv2e10"),  # 2026-10-31: SFT-v3 corpus
    "v3etp20": ("v3 empty-think-prefix 20 ep (iter284)", "sftv2e20"),  # 2026-11-04: SFT-v3 corpus
}
SOURCES.update({m: (HERE if m == "v3etp20" else ORIG) / f"{m}.jsonl" for m in ETP})  # run_features.py --sources <m> output
# (2026-11-04 copy in analysis/v3etp20/features: v3etp20 from here, the earlier models from ORIG, read-only)
LABEL.update({m: label for m, (label, _) in ETP.items()})
SAME_EPOCH_RAW = {m: raw for m, (_, raw) in ETP.items()}


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def pctl(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def paired_diff_ci(a_rows, b_rows, n=2000, seed=20261025):
    """95% CI of (b - a) in pooled % action turns with reasoning, resampling tasks (same draw for both)."""
    def per_task(rows):
        out = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            if r.get("export") and r.get("turns"):
                out[r["task_id"]][0] += r["action_with_reasoning"]
                out[r["task_id"]][1] += r["action_turns"]
        return out
    a, b = per_task(a_rows), per_task(b_rows)
    tasks = sorted(set(a) & set(b))
    if len(tasks) < 2:
        return None
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        sample = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        an, ad = sum(a[t][0] for t in sample), sum(a[t][1] for t in sample)
        bn, bd = sum(b[t][0] for t in sample), sum(b[t][1] for t in sample)
        if ad and bd:
            stats.append(100.0 * bn / bd - 100.0 * an / ad)
    stats.sort()
    point = (100.0 * sum(b[t][0] for t in tasks) / sum(b[t][1] for t in tasks)
             - 100.0 * sum(a[t][0] for t in tasks) / sum(a[t][1] for t in tasks))
    return {"diff": round(point, 1), "ci95": [round(stats[int(0.025 * len(stats))], 1),
                                              round(stats[int(0.975 * len(stats)) - 1], 1)], "tasks": len(tasks)}


def main():
    data = {source: load(path) for source, path in SOURCES.items()}
    data = {source: rows for source, rows in data.items() if rows}
    summary = {source: tr_compare.summarize(source, rows, None) for source, rows in data.items()}
    for source, rows in data.items():
        s = summary[source]
        s["label"] = LABEL[source]
        s["ci95_by_task"] = {name: tr_compare.bootstrap(rows, *fns) for name, fns in tr_compare.CI_METRICS.items()}
        ex = [r for r in rows if r.get("export") and r.get("turns")]
        runs = [r["longest_unreasoned_action_run"] for r in ex]
        s["longest_unreasoned_action_run_mean"] = round(statistics.fmean(runs), 2) if runs else None
        s["longest_unreasoned_action_run_p90"] = pctl(runs, 0.9)
        s["longest_unreasoned_action_run_max"] = max(runs) if runs else None
        s["tool_calls_per_rollout_median"] = tr_compare.med([r["tool_calls"] for r in ex])
        s["action_turns_per_rollout_mean"] = tr_compare.mean([r["action_turns"] for r in ex])
        s["open_think_turns_total"] = sum(r.get("open_think_turns", 0) for r in ex)
        # 2026-10-27
        rtok = [x for r in ex for x in r["reasoning_tokens"]]
        for q in (0.1, 0.25, 0.75, 0.9, 0.99):
            s[f"reasoning_tokens_per_reasoning_turn_p{int(q * 100)}"] = pctl(rtok, q)
        s["pct_reasoning_turns_lt_20_tokens"] = round(100.0 * sum(x < 20 for x in rtok) / len(rtok), 1) if rtok else None
        s["pct_reasoning_turns_lt_50_tokens"] = round(100.0 * sum(x < 50 for x in rtok) / len(rtok), 1) if rtok else None
        s["hub_output_tokens_per_rollout_mean"] = tr_compare.mean([(r.get("hub_tokens") or {}).get("output", 0) for r in ex])
        walls = [r["agent_wall_s"] / 60 for r in ex if r.get("agent_wall_s")]
        s["agent_wall_min_mean"] = round(statistics.fmean(walls), 1) if walls else None
        s["agent_wall_min_p90"] = round(pctl(walls, 0.9), 1) if walls else None
        s["rollouts_deadline_timeout"] = sum(r.get("agent_error") == "TimeoutError" for r in rows)
        if source != "sc14" and "sc14" in data:
            s["distance_to_sc14"] = tr_compare.distances(data["sc14"], rows)
            s["diff_pct_action_reasoned_vs_sc14"] = paired_diff_ci(data["sc14"], rows)
        if source not in ("sc14", "sftv2") and "sftv2" in data:
            s["diff_pct_action_reasoned_vs_5ep"] = paired_diff_ci(data["sftv2"], rows)
        if SAME_EPOCH_RAW.get(source) in data:
            s["diff_pct_action_reasoned_vs_same_epoch_raw"] = paired_diff_ci(data[SAME_EPOCH_RAW[source]], rows)
    (HERE / "comparison.json").write_text(json.dumps(summary, indent=1) + "\n")
    first = next(iter(summary.values()))
    keys = [k for k in first if not isinstance(first[k], (dict, list)) and k not in ("source", "label")]
    lines = ["| metric | " + " | ".join(summary[s]["label"] for s in summary) + " |", "|---|" + "---|" * len(summary)]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(str(summary[s].get(k)) for s in summary) + " |")
    for name in tr_compare.CLASSES:
        lines.append(f"| turn_class_mix: {name} | " + " | ".join(str(summary[s]["turn_class_mix"].get(name))
                                                                  for s in summary) + " |")
    for name in tr_compare.CI_METRICS:
        lines.append(f"| 95% CI (task bootstrap): {name} | " + " | ".join(
            str(summary[s]["ci95_by_task"].get(name)) for s in summary) + " |")
    for key in ("diff_pct_action_reasoned_vs_sc14", "diff_pct_action_reasoned_vs_5ep",
                "diff_pct_action_reasoned_vs_same_epoch_raw"):
        lines.append(f"| {key} (paired task bootstrap) | " + " | ".join(
            str(summary[s].get(key, "-")) for s in summary) + " |")
    for k in ("turn_class_tv", "ks_reasoning_tokens_per_turn", "ks_tool_calls_per_turn", "ks_turns_per_rollout",
              "ks_reasoning_tokens_per_rollout"):
        lines.append(f"| distance to sc14: {k} | " + " | ".join(
            str((summary[s].get("distance_to_sc14") or {}).get(k, "-")) for s in summary) + " |")
    for name in ("bash", "read", "edit", "grep", "glob", "write", "todowrite", "task", "webfetch"):
        lines.append(f"| tool share %: {name} | " + " | ".join(str(summary[s]["tool_share"].get(name, 0))
                                                              for s in summary) + " |")
    for d in range(10):
        lines.append(f"| % action turns reasoned, rollout decile {d + 1} | " + " | ".join(
            str(summary[s]["pct_action_reasoned_by_decile"][d]) for s in summary) + " |")
    for s in summary:
        lines.append(f"\n{summary[s]['label']} top reasoning openers: {summary[s]['top_reasoning_openers']}")
    (HERE / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
