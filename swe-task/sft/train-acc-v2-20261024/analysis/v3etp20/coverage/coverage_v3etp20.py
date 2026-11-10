#!/usr/bin/env python3
"""train-acc v3etp20 (2026-11-04): coverage of the 50 train-acc tasks by the SFT corpora, accuracy by coverage, and the
train-acc vs held-out DeepSWE picture. Stdlib only; read-only on every input.

    python3 -I -B coverage_v3etp20.py -> coverage_v3etp20.{json,txt}

Inputs
  TA/tasks.json                                   the 50 tasks (run, task_id)
  corpus/kimi-pass-v2-20261023/selection/kimi_pass_v2.jsonl   SFT-v2 conversations (797)
  corpus/sft-v3-20261029/selection/sft_v3.jsonl               SFT-v3 conversations (1,594; field rollout_batch v2/v4/v5)
  TA/results/<model>-group-*.json                  pass / fail / infra per slot (infra excluded, as aggregate.py)
  DeepSWE classifier all.jsonl per run (565 rollouts = 113 held-out tasks x 5; reward 1 = resolved), the same files
  as degenerate-behaviors-20261031 used; v3etp20's from the DeepSWE v3etp20 analysis directory (read-only).
Coverage of a task = number of conversations of the corpus whose (run, task_id) is the task.
Per coverage bin: pooled pass@1 (passes / graded) per model and the number of tasks; per task: conversations and
passes of each model. Spearman correlation (average ranks) between a task's SFT-v3 conversations and its v3etp20 pass
rate. Train vs held-out: pass@1 on the 50 training tasks vs resolved@1 on DeepSWE, per model.
"""
import collections
import json
from pathlib import Path

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
CORPUS = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus")
A = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/eval/deepswe-v1.1/analysis")
HERE = Path(__file__).resolve().parent
MODELS = ["sc14", "sftv2", "sftv2e10", "sftv2e20", "etp5", "etp10", "etp20", "v3etp5", "v3etp10", "v3etp20"]
DEEPSWE = {
    "sc14": A / "empty-patch-20261018/base-step0-rep5-out131k/all.jsonl",
    "sftv2": A / "lenient-think-20261023/empty-patch/v2-lenient/all.jsonl",
    "sftv2e10": A / "lenient-think-20261023/empty-patch/v2-10ep-lenient/all.jsonl",
    "sftv2e20": A / "lenient-think-20261023/empty-patch/v2-20ep-lenient/all.jsonl",
    "etp5": A / "etp-sft-20261027/classifier/etp5/all.jsonl",
    "etp10": A / "etp-sft-20261027/classifier/etp10/all.jsonl",
    "etp20": A / "etp-sft-20261027/classifier/etp20/all.jsonl",
    "v3etp5": A / "sftv3-etp-20261030/classifier/v3etp5/all.jsonl",
    "v3etp10": A / "sftv3-etp-20261030/classifier/v3etp10/all.jsonl",
    "v3etp20": A / "sftv3-etp20-20261104/classifier/v3etp20/all.jsonl",
}
BINS = [(0, 0, "0"), (1, 4, "1-4"), (5, 8, "5-8"), (9, 12, "9-12"), (13, 999, "13+")]


def results(model):
    out = {}
    for path in sorted(TA.glob(f"results/{model}-group-*.json")):
        for task_id, recs in json.loads(path.read_text())["records"].items():
            out[task_id] = [r["status"] for r in recs]
    return out


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main():
    tasks = json.loads((TA / "tasks.json").read_text())["tasks"]
    keys = {(t["run"], t["task_id"]) for t in tasks}
    v2 = collections.Counter()
    for line in open(CORPUS / "kimi-pass-v2-20261023/selection/kimi_pass_v2.jsonl"):
        r = json.loads(line)
        if (r.get("run"), r.get("task_id")) in keys:
            v2[(r["run"], r["task_id"])] += 1
    v3 = collections.Counter()
    v3_batch = collections.defaultdict(collections.Counter)
    for line in open(CORPUS / "sft-v3-20261029/selection/sft_v3.jsonl"):
        r = json.loads(line)
        k = (r.get("run"), r.get("task_id"))
        if k in keys:
            v3[k] += 1
            v3_batch[k][r.get("rollout_batch")] += 1
    res = {m: results(m) for m in MODELS}
    per_task = []
    for t in tasks:
        k = (t["run"], t["task_id"])
        row = {"task_id": t["task_id"], "run": t["run"], "v2_conversations": v2[k], "v3_conversations": v3[k],
               "v3_by_batch": dict(v3_batch[k]), "kimi_passes_in_sft_tasks_json": t.get("kimi_passes_in_sft")}
        for m in MODELS:
            st = res[m].get(t["task_id"], [])
            row[m] = [st.count("pass"), sum(s in ("pass", "fail") for s in st)]
        per_task.append(row)
    lines = [f"SFT-v2 conversations on the 50 tasks: {sum(v2.values())} (tasks with >= 1: {sum(1 for k in keys if v2[k])}); "
             f"SFT-v3: {sum(v3.values())} (tasks with >= 1: {sum(1 for k in keys if v3[k])}); v3 by batch: "
             f"{dict(sum((v3_batch[k] for k in keys), collections.Counter()))}"]
    bins = {}
    for lo, hi, name in BINS:
        sel = [r for r in per_task if lo <= r["v3_conversations"] <= hi]
        bins[name] = {"tasks": len(sel), **{m: [sum(r[m][0] for r in sel), sum(r[m][1] for r in sel)] for m in MODELS}}
    lines.append("pass@1 (passes/graded) by SFT-v3 conversations per task:")
    lines.append("| v3 conversations | tasks | " + " | ".join(MODELS) + " |")
    lines.append("|---|---|" + "---|" * len(MODELS))
    for name, b in bins.items():
        lines.append(f"| {name} | {b['tasks']} | " + " | ".join(
            f"{100 * b[m][0] / b[m][1]:.1f}% ({b[m][0]}/{b[m][1]})" if b[m][1] else "-" for m in MODELS) + " |")
    rates = [r["v3etp20"][0] / r["v3etp20"][1] if r["v3etp20"][1] else 0.0 for r in per_task]
    gains = [(r["v3etp20"][0] / r["v3etp20"][1] if r["v3etp20"][1] else 0) - (r["v3etp10"][0] / r["v3etp10"][1] if r["v3etp10"][1] else 0)
             for r in per_task]
    conv = [r["v3_conversations"] for r in per_task]
    sp = {"v3_conversations_vs_v3etp20_rate": round(spearman(conv, rates), 3),
          "v3_conversations_vs_v3etp20_minus_v3etp10": round(spearman(conv, gains), 3),
          "v3_conversations_vs_base_rate": round(spearman(conv, [r["sc14"][0] / r["sc14"][1] for r in per_task]), 3)}
    lines.append(f"Spearman over the 50 tasks: {sp}")
    unsolved = {m: sorted(r["task_id"] for r in per_task if r[m][0] == 0) for m in MODELS}
    lines.append(f"tasks never solved by v3etp20 ({len(unsolved['v3etp20'])}): " + ", ".join(
        f"{t} (v3 convs {next(r['v3_conversations'] for r in per_task if r['task_id'] == t)})" for t in unsolved["v3etp20"]))
    never_any = sorted(set.intersection(*(set(v) for v in unsolved.values())))
    lines.append(f"tasks unsolved by all ten runs ({len(never_any)}): {never_any}")
    # train vs held-out
    held = {}
    for m, path in DEEPSWE.items():
        rows = [json.loads(line) for line in open(path) if line.strip()]
        held[m] = {"resolved": sum(1 for r in rows if (r.get("reward") or 0) >= 1), "rollouts": len(rows)}
    lines.append("train-acc pass@1 (50 training tasks x 5, infra excluded) vs DeepSWE resolved@1 (113 held-out tasks x 5):")
    lines.append("| model | train-acc | DeepSWE | gap (pp) |")
    lines.append("|---|---|---|---|")
    train = {}
    for m in MODELS:
        p = sum(r[m][0] for r in per_task)
        g = sum(r[m][1] for r in per_task)
        train[m] = [p, g]
        h = held[m]
        lines.append(f"| {m} | {100 * p / g:.1f}% ({p}/{g}) | {100 * h['resolved'] / h['rollouts']:.2f}% ({h['resolved']}/{h['rollouts']}) | "
                     f"{100 * p / g - 100 * h['resolved'] / h['rollouts']:+.1f} |")
    out = {"per_task": per_task, "bins": bins, "spearman": sp, "unsolved": unsolved, "unsolved_by_all": never_any,
           "train_acc": train, "deepswe": held}
    (HERE / "coverage_v3etp20.json").write_text(json.dumps(out, indent=1) + "\n")
    (HERE / "coverage_v3etp20.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
