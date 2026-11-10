#!/usr/bin/env python3
"""train-acc-v2 (2026-10-25): per-task comparison and paired tests between models on the 50 training tasks.

    python3 -I -B paired.py  -> analysis/accuracy/paired.{md,json}

Per task: passes / graded rollouts (infra excluded) of each model. Paired comparisons (A vs B, per task difference in
pass rate, tasks with no graded rollout in either model dropped):
  * pass@1 difference (pooled passes / graded) with a 95% task-bootstrap CI (10,000 resamples of the 50 tasks, the same
    draw for both models);
  * sign-flip permutation test on the per-task pass-rate differences (mean difference; 100,000 random flips, two-sided);
  * tasks better / equal / worse, and the exact two-sided sign test on the better/worse counts;
  * pass@5 (task solved at least once) difference with McNemar's exact test on discordant tasks.

2026-10-27: + the empty-think-prefix checkpoints etp5 (5 ep, iter38) / etp10 (10 ep, iter76), compared with the raw
v2 SFT of the same epochs, raw 20 ep, each other and base. Added per pair (decomposition table, json key
"decomposition"): pass@1 = p * r with p = patched share (graded rollouts with a non-empty model patch) and r = resolved /
patched; the pass@1 difference splits exactly into a patch-rate part (p_B - p_A) * (r_A + r_B) / 2 and a resolve-rate
part (r_B - r_A) * (p_A + p_B) / 2, each with a 95% task-bootstrap CI (10,000 resamples, own RNG with the same seed,
so the earlier rows are unchanged).
"""
import collections
import json
import math
from pathlib import Path
import random

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
OUT = TA / "analysis/accuracy"
LABEL = {"sc14": "base", "sftv2": "5 ep", "sftv2e10": "10 ep", "sftv2e20": "20 ep",
         "etp5": "etp 5 ep",
         "etp10": "etp 10 ep",
         "etp20": "etp 20 ep",
         "v3etp5": "v3 etp 5 ep",  # 2026-10-31: SFT-v3 empty-think-prefix 5 ep (iter71)
         "v3etp10": "v3 etp 10 ep",  # 2026-10-31: SFT-v3 empty-think-prefix 10 ep (iter142)
         }  # the one model registry (one line per model, column order)
MODELS = tuple(LABEL)
SAME_EPOCH_RAW = {"etp5": "sftv2", "etp10": "sftv2e10", "etp20": "sftv2e20"}  # used only for models in LABEL
ETP = [m for m in MODELS if m in SAME_EPOCH_RAW]
PAIRS = ((("sftv2", "sftv2e10"), ("sftv2", "sftv2e20"), ("sftv2e10", "sftv2e20"), ("sc14", "sftv2e10"),
          ("sc14", "sftv2e20"), ("sc14", "sftv2"))
         # 2026-10-27, empty-think-prefix checkpoints: vs the raw v2 of the same epochs, 10 ep etp vs raw 20 ep,
         # consecutive etp checkpoints, and vs base (derived from LABEL)
         + tuple((SAME_EPOCH_RAW[m], m) for m in ETP) + (("sftv2e20", "etp10"),) + tuple(zip(ETP, ETP[1:]))
         + tuple(("sc14", m) for m in ETP))
# 2026-10-31, SFT-v3 empty-think-prefix: vs the v2 etp run of the same epochs and of about the same optimizer steps
# (v3 5 ep = iter71 ~ v2 10 ep = iter76; v3 10 ep = iter142 ~ v2 20 ep = iter152), v3 10 ep vs v3 5 ep, and vs base
PAIRS += (("etp5", "v3etp5"), ("etp10", "v3etp5"), ("etp10", "v3etp10"), ("etp20", "v3etp10"), ("v3etp5", "v3etp10"),
          ("sc14", "v3etp5"), ("sc14", "v3etp10"))


def load():
    found = {m: {} for m in MODELS}
    for path in sorted((TA / "results").glob("*-group-*.json")):
        data = json.loads(path.read_text())
        if data["model"] in found:
            for task_id, records in data["records"].items():
                found[data["model"]][task_id] = [r["status"] for r in records]
    return found


def load_patched():
    """{model: {task: [patch non-empty per slot]}} in the records' order (2026-10-27)."""
    out = {m: {} for m in MODELS}
    for path in sorted((TA / "results").glob("*-group-*.json")):
        data = json.loads(path.read_text())
        if data["model"] in out:
            for task_id, records in data["records"].items():
                out[data["model"]][task_id] = [bool(r.get("patch_chars")) for r in records]
    return out


def decompose(found, patched, a, b, seed=20261025, n=10000):
    """pass@1 = p * r (p = patched / graded, r = passes / patched); exact symmetric split of the B - A difference."""
    rows = []
    for t in found[a]:
        if t not in found[b]:
            continue
        cells = []
        for m in (a, b):
            graded = [(s, pt) for s, pt in zip(found[m][t], patched[m][t]) if s in ("pass", "fail")]
            cells += [sum(s == "pass" for s, _ in graded), sum(pt for _, pt in graded), len(graded)]
        if cells[2] and cells[5]:
            rows.append(cells)

    def stats(sample):
        pa_, qa_, ga_, pb_, qb_, gb_ = (sum(r[i] for r in sample) for i in range(6))
        p_a, p_b = qa_ / ga_, qb_ / gb_
        r_a, r_b = (pa_ / qa_ if qa_ else 0.0), (pb_ / qb_ if qb_ else 0.0)
        return {"p_a": p_a, "p_b": p_b, "r_a": r_a, "r_b": r_b, "d_pass": p_b * r_b - p_a * r_a,
                "d_p": p_b - p_a, "d_r": r_b - r_a, "part_patch": (p_b - p_a) * (r_a + r_b) / 2,
                "part_resolve": (r_b - r_a) * (p_a + p_b) / 2}
    point = stats(rows)
    rng = random.Random(seed)
    boots = collections.defaultdict(list)
    for _ in range(n):
        for key, value in stats([rows[rng.randrange(len(rows))] for _ in rows]).items():
            boots[key].append(value)
    ci = {key: [round(100 * sorted(v)[int(0.025 * n)], 1), round(100 * sorted(v)[int(0.975 * n) - 1], 1)]
          for key, v in boots.items()}
    tot = [sum(r[i] for r in rows) for i in range(6)]
    return {"passes_a": tot[0], "patched_a": tot[1], "graded_a": tot[2], "passes_b": tot[3], "patched_b": tot[4],
            "graded_b": tot[5], **{k: round(100 * v, 1) for k, v in point.items()}, "ci95": ci}


def binom_two_sided(k, n):
    """Exact two-sided sign/McNemar test p-value for k successes of n at p = 0.5."""
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in probs if p <= probs[k] + 1e-12))


def compare(found, a, b, seed=20261025):
    tasks = [t for t in found[a] if t in found[b]]
    rows = []
    for t in tasks:
        pa, ga = found[a][t].count("pass"), sum(s in ("pass", "fail") for s in found[a][t])
        pb, gb = found[b][t].count("pass"), sum(s in ("pass", "fail") for s in found[b][t])
        if ga and gb:
            rows.append((t, pa, ga, pb, gb))
    rng = random.Random(seed)
    point = 100 * (sum(r[3] for r in rows) / sum(r[4] for r in rows) - sum(r[1] for r in rows) / sum(r[2] for r in rows))
    boots = []
    for _ in range(10000):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        boots.append(100 * (sum(r[3] for r in sample) / sum(r[4] for r in sample)
                            - sum(r[1] for r in sample) / sum(r[2] for r in sample)))
    boots.sort()
    diffs = [r[3] / r[4] - r[1] / r[2] for r in rows]
    observed = sum(diffs) / len(diffs)
    extreme = 0
    flips = 100000
    for _ in range(flips):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs) / len(diffs)
        extreme += abs(s) >= abs(observed) - 1e-12
    better = sum(d > 0 for d in diffs)
    worse = sum(d < 0 for d in diffs)
    solved_a = {r[0] for r in rows if r[1] > 0}
    solved_b = {r[0] for r in rows if r[3] > 0}
    only_b, only_a = len(solved_b - solved_a), len(solved_a - solved_b)
    return {"a": a, "b": b, "tasks": len(rows), "pass_at_1_diff_pp": round(point, 1),
            "pass_at_1_diff_ci95": [round(boots[int(0.025 * len(boots))], 1), round(boots[int(0.975 * len(boots)) - 1], 1)],
            "task_mean_diff_pp": round(100 * observed, 1), "signflip_p": (extreme + 1) / (flips + 1),
            "tasks_b_better": better, "tasks_equal": len(rows) - better - worse, "tasks_b_worse": worse,
            "sign_test_p": binom_two_sided(min(better, worse), better + worse),
            "pass_at_5_a": len(solved_a), "pass_at_5_b": len(solved_b), "solved_only_a": only_a, "solved_only_b": only_b,
            "mcnemar_p": binom_two_sided(min(only_a, only_b), only_a + only_b)}


def fmt_p(p):
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def main():
    found = load()
    patched = load_patched()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = json.loads((TA / "tasks.json").read_text())["tasks"]
    present = [m for m in MODELS if found[m]]
    results = [compare(found, a, b) for a, b in PAIRS if a in present and b in present]
    for r in results:
        r["decomposition"] = decompose(found, patched, r["a"], r["b"])
    lines = ["| A | B | tasks | pass@1 B-A (pp) | 95% CI | task-mean B-A | sign-flip p | B better/equal/worse | sign p |"
             " pass@5 A | pass@5 B | McNemar p |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {LABEL[r['a']]} | {LABEL[r['b']]} | {r['tasks']} | {r['pass_at_1_diff_pp']:+.1f} | "
                     f"[{r['pass_at_1_diff_ci95'][0]:+.1f}, {r['pass_at_1_diff_ci95'][1]:+.1f}] | {r['task_mean_diff_pp']:+.1f} | "
                     f"{fmt_p(r['signflip_p'])} | {r['tasks_b_better']}/{r['tasks_equal']}/{r['tasks_b_worse']} | {fmt_p(r['sign_test_p'])} | "
                     f"{r['pass_at_5_a']} | {r['pass_at_5_b']} | {fmt_p(r['mcnemar_p'])} |")
    lines += ["", "pass@1 = patched share p x resolved-among-patched r (graded rollouts); B - A = patch-rate part + resolve-rate "
              "part (pp, 95% task-bootstrap CI):", "",
              "| A | B | p A | p B | r A | r B | pass@1 B-A | patch-rate part | resolve-rate part | r B-A |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        d = r["decomposition"]
        c = d["ci95"]
        lines.append(f"| {LABEL[r['a']]} | {LABEL[r['b']]} | {d['p_a']:.1f}% ({d['patched_a']}/{d['graded_a']}) | "
                     f"{d['p_b']:.1f}% ({d['patched_b']}/{d['graded_b']}) | {d['r_a']:.1f}% ({d['passes_a']}/{d['patched_a']}) | "
                     f"{d['r_b']:.1f}% ({d['passes_b']}/{d['patched_b']}) | {d['d_pass']:+.1f} | "
                     f"{d['part_patch']:+.1f} [{c['part_patch'][0]:+.1f}, {c['part_patch'][1]:+.1f}] | "
                     f"{d['part_resolve']:+.1f} [{c['part_resolve'][0]:+.1f}, {c['part_resolve'][1]:+.1f}] | "
                     f"{d['d_r']:+.1f} [{c['d_r'][0]:+.1f}, {c['d_r'][1]:+.1f}] |")
    lines += ["", "| task | run | " + " | ".join(LABEL[m] for m in present) + " | passes " + "/".join(LABEL[m] for m in present) + " |",
              "|---|---|" + "---|" * (len(present) + 1)]
    for task in tasks:
        cells = ["".join({"pass": "P", "fail": ".", "infra": "x"}[s] for s in found[m].get(task["task_id"], [])) or "-"
                 for m in present]
        counts = "/".join(str(found[m].get(task["task_id"], []).count("pass")) for m in present)
        lines.append(f"| {task['task_id']} | {task['run']} | " + " | ".join(cells) + f" | {counts} |")
    (OUT / "paired.md").write_text("\n".join(lines) + "\n")
    (OUT / "paired.json").write_text(json.dumps(results, indent=1) + "\n")
    print("\n".join(lines[:len(results) + 2]))


if __name__ == "__main__":
    main()
