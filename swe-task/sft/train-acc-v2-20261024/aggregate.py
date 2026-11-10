#!/usr/bin/env python3
"""train-acc-v2 (2026-10-24): summarize results/<model>-group-<K>.json (driver.py) into accuracy.md + summary.json.
(2026-10-25: it wrote REPORT.md before; REPORT.md now holds hand-written analysis and is never overwritten.)

    python3 -I aggregate.py

Per model over the sampled tasks (5 rollouts each): pass@1 = passes / graded rollouts (infra excluded, as the
pipeline's pass_stats), task-mean pass rate, pass@5 (task has >= 1 pass among its graded rollouts), infra count;
and a per-task table sftv2 vs sc14. Stdlib only.
"""
import json
from pathlib import Path

TA = Path(__file__).resolve().parent
MODELS = ("sftv2", "sftv2e10", "sftv2e20", "etp5", "etp10", "etp20", "v3etp5", "v3etp10", "v3etp20", "sc14")
LABEL = {"sftv2": "v2 SFT 5 ep (iter38)", "sftv2e10": "v2 SFT 10 ep (iter76)", "sftv2e20": "v2 SFT 20 ep (iter152)",
         "etp5": "v2 empty-think-prefix 5 ep (iter38)", "etp10": "v2 empty-think-prefix 10 ep (iter76)",
         "etp20": "v2 empty-think-prefix 20 ep (iter152)", "v3etp5": "v3 empty-think-prefix 5 ep (iter71)",
         "v3etp10": "v3 empty-think-prefix 10 ep (iter142)", "v3etp20": "v3 empty-think-prefix 20 ep (iter284)",
         "sc14": "base (sc14)"}


def load():
    found = {model: {} for model in MODELS}
    groups = {model: [] for model in MODELS}
    for path in sorted((TA / "results").glob("*-group-*.json")):
        data = json.loads(path.read_text())
        groups[data["model"]].append(data["group"])
        for task_id, records in data["records"].items():
            found[data["model"]][task_id] = [record["status"] for record in records]
    return found, groups


def stats(per_task):
    passes = sum(s.count("pass") for s in per_task.values())
    graded = sum(s.count("pass") + s.count("fail") for s in per_task.values())
    infra = sum(s.count("infra") for s in per_task.values())
    rates = [s.count("pass") / (s.count("pass") + s.count("fail")) for s in per_task.values()
             if s.count("pass") + s.count("fail")]
    return {"tasks": len(per_task), "rollouts": sum(len(s) for s in per_task.values()), "graded": graded,
            "passes": passes, "infra": infra, "pass_at_1": passes / graded if graded else None,
            "task_mean_pass_rate": sum(rates) / len(rates) if rates else None,
            "pass_at_5": sum(1 for s in per_task.values() if "pass" in s) / len(per_task) if per_task else None,
            "tasks_all_pass": sum(1 for s in per_task.values() if s and all(x == "pass" for x in s))}


def main():
    found, groups = load()
    tasks = json.loads((TA / "tasks.json").read_text())["tasks"]
    summary = {model: dict(stats(found[model]), groups_done=sorted(groups[model])) for model in MODELS}
    pct = lambda x: "-" if x is None else f"{100 * x:.1f}%"
    lines = ["# SFT v2 training accuracy vs base (2026-10-24; 10/20-epoch checkpoints added 2026-10-25; "
             "empty-think-prefix 2026-10-27; SFT-v3 empty-think-prefix 5/10 ep 2026-10-30)", "",
             f"50 tasks sampled (seed 20261024) from the 97 hengyuf tasks of the v2 SFT pool (Baihe's 97: snapshots "
             f"not accessible); 5 rollouts per task per model, the synth pipeline's probe harness and each task's "
             f"grader; every model served with the lenient-think parser.", "",
             "| model | groups done | rollouts | graded | infra | pass@1 | task-mean | pass@5 | tasks 5/5 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for model in MODELS:
        s = summary[model]
        lines.append(f"| {LABEL[model]} | {len(s['groups_done'])}/8 | {s['rollouts']} | {s['graded']} | {s['infra']} | "
                     f"{pct(s['pass_at_1'])} | {pct(s['task_mean_pass_rate'])} | {pct(s['pass_at_5'])} | {s['tasks_all_pass']} |")
    lines += ["", "| task | run | " + " | ".join(LABEL[m] for m in MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
    for task in tasks:
        cell = lambda m: "".join({"pass": "P", "fail": ".", "infra": "x"}[x] for x in found[m].get(task["task_id"], [])) or "-"
        lines.append(f"| {task['task_id']} | {task['run']} | " + " | ".join(cell(m) for m in MODELS) + " |")
    lines += ["", "P = pass, . = fail, x = infra (excluded)."]
    (TA / "accuracy.md").write_text("\n".join(lines) + "\n")
    (TA / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
