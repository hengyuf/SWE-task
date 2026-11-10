#!/usr/bin/env python3
"""Collect phase (2026-11-02): independent headline counts, read-only, to cross-check collect.py's final outputs.
Kimi / DeepSeek from results/<model>-<campaign>.json records (kimi w2-g1 scriggo-method-declarations r0 as a graded
fail, OPERATOR_NOTE 2026-11-01 11:15 / 14:15); GLM from its probe dirs (graded = the first attempt with
evaluation_completed and no generation-validity failure, as ATP's slot_state; pass = reward 1). Base rates recomputed
from the two base per_task.csv files. Prints JSON; writes --out if given (inside TD).

    /usr/bin/python3 -I -B final/independent_counts_20261102.py [--out analysis/independent-counts-20261102.json]
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

TD = Path(__file__).resolve().parent.parent
E = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/eval/deepswe-v1.1")
K1M = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/deepswe-partial-analysis/per_task.csv")
CAMPAIGNS = ("w1-g0", "w1-g1", "w1-g2", "w2-g0", "w2-g1")


def rj(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def pct(a, b):
    return round(100.0 * a / b, 2) if b else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    unc = {r["task_id"] for r in csv.DictReader(open(E / "analysis/base-uncovered-synth-20261030/uncovered_base_k10.csv"))}
    base = collections.Counter()
    tasks = set()
    for run in ("base-step0-rep5-out131k", "base-step0-rep5-out131k-rerun1"):
        for r in csv.DictReader(open(E / "runs" / run / "per_task.csv")):
            base[r["task_id"]] += int(r["resolved"])
            tasks.add(r["task_id"])
    wave = {t: (1 if t in unc else 2) for t in tasks}
    slots = {m: collections.defaultdict(list) for m in ("kimi", "glm", "deepseek")}  # task -> [status]
    for model in ("kimi", "deepseek"):
        for name in CAMPAIGNS:
            data = rj(TD / "results" / f"{model}-{name}.json")
            for task_id, records in data["records"].items():
                for r in records:
                    status = r["status"]
                    if (model, name, task_id, r["slot"]) == ("kimi", "w2-g1", "scriggo-method-declarations", "kimi-r0"):
                        status = "fail"  # GRADED_FAIL_OVERRIDES
                    slots[model][task_id].append(status)
    glm_states = collections.Counter()
    for name in CAMPAIGNS:
        probe = TD / "probes" / "glm" / name
        manifest = rj(probe / "manifest.json")
        for task in manifest["tasks"]:
            for repeat in range(5):
                d = probe / "models" / "glm" / "attempts" / f"{task['_ng_task_index']}-{repeat}"
                attempts = sorted((p for p in d.glob("attempt-*") if p.is_dir()),
                                  key=lambda p: int(p.name.split("-")[1])) if d.exists() else []
                status = "not_started" if not attempts else "ungraded"
                for a in attempts:
                    t = rj(a / "terminal.json", {}) or {}
                    v = rj(a / "generation-validity-v2-r7.json")
                    if t.get("evaluation_completed") is True and (v is None or v.get("failure") is None):
                        status = "pass" if t.get("reward") == 1 else "fail"
                        break
                glm_states[status] += 1
                slots["glm"][task["task_id"]].append(status)
    out = {"tasks": len(tasks), "wave1": sum(w == 1 for w in wave.values()), "glm_slot_states": dict(glm_states)}
    uncovered = sorted(t for t in tasks if wave[t] == 1 and base[t] <= 2)
    out["uncovered_base_le2"] = len(uncovered)
    k1m = {r["task_id"]: int(r["kimi_correct"]) for r in csv.DictReader(open(K1M))}
    for model, per_task in slots.items():
        m = {}
        for w in (None, 1, 2):
            ids = [t for t in tasks if w is None or wave[t] == w]
            st = [s for t in ids for s in per_task.get(t, [])]
            graded = [s for s in st if s in ("pass", "fail")]
            passes = sum(s == "pass" for s in graded)
            infra = sum(s == "infra" for s in st)
            started = [t for t in ids if any(s in ("pass", "fail") for s in per_task.get(t, []))]
            m["all" if w is None else f"w{w}"] = {
                "slots": len(st), "graded": len(graded), "pass": passes, "infra": infra,
                "pass_at_1_excl_infra": pct(passes, len(graded)), "pass_at_1_raw": pct(passes, len(graded) + infra),
                "tasks_graded": len(started),
                "tasks_any_pass": sum(any(s == "pass" for s in per_task[t]) for t in started),
                "tasks_all5_graded": sum(sum(s in ("pass", "fail") for s in per_task[t]) == 5 for t in started)}
        verdict = {}
        for t in uncovered:
            st = per_task.get(t, [])
            p = sum(s == "pass" for s in st)
            left = sum(s not in ("pass", "fail") for s in st) if model == "glm" else 0
            verdict[t] = "yes" if p >= 3 else "undetermined" if p + left >= 3 else "no"
        m["uncovered_yes"] = sorted(t for t, v in verdict.items() if v == "yes")
        m["uncovered_undetermined"] = sorted(t for t, v in verdict.items() if v == "undetermined")
        out[model] = m
    out["uncovered_any_yes"] = sorted(set().union(*(set(out[m]["uncovered_yes"]) for m in slots)))
    out["uncovered_kimi_or_deepseek_yes"] = sorted(set(out["kimi"]["uncovered_yes"]) | set(out["deepseek"]["uncovered_yes"]))
    out["kimi1m"] = {"avg": pct(sum(k1m.values()), 5 * len(k1m)), "tasks_any_pass": sum(v > 0 for v in k1m.values()),
                     "uncovered_yes": sorted(t for t in uncovered if k1m.get(t, 0) >= 3),
                     "w1": pct(sum(k1m[t] for t in tasks if wave[t] == 1), 5 * sum(wave[t] == 1 for t in tasks)),
                     "w2": pct(sum(k1m[t] for t in tasks if wave[t] == 2), 5 * sum(wave[t] == 2 for t in tasks))}
    kimi_rate = {t: sum(s == "pass" for s in slots["kimi"][t]) for t in tasks}
    out["kimi_vs_1m_tasks"] = {"higher_here": sum(kimi_rate[t] > k1m[t] for t in tasks),
                               "lower_here": sum(kimi_rate[t] < k1m[t] for t in tasks),
                               "tied": sum(kimi_rate[t] == k1m[t] for t in tasks)}
    text = json.dumps(out, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
