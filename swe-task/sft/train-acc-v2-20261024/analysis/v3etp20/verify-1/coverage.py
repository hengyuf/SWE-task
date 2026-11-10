#!/usr/bin/env python3
"""verify-1: SFT-v2 / SFT-v3 coverage of the 50 train-acc tasks (selection lists, read-only), accuracy by coverage bin,
Spearman (average ranks, my own implementation), and the train-acc vs DeepSWE pass@1 table from the DeepSWE classifier
rows (reward >= 1 = resolved). Stdlib only.

    python3 -I -B coverage.py -> out/coverage.txt
"""
import collections
import json
from pathlib import Path

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
CORPUS = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus")
A = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/eval/deepswe-v1.1/analysis")
OUT = TA / "analysis/v3etp20/verify-1/out"
MODELS = ["sc14", "sftv2", "sftv2e10", "sftv2e20", "etp5", "etp10", "etp20", "v3etp5", "v3etp10", "v3etp20"]
DS = {"sc14": A / "empty-patch-20261018/base-step0-rep5-out131k/all.jsonl",
      "sftv2": A / "lenient-think-20261023/empty-patch/v2-lenient/all.jsonl",
      "sftv2e10": A / "lenient-think-20261023/empty-patch/v2-10ep-lenient/all.jsonl",
      "sftv2e20": A / "lenient-think-20261023/empty-patch/v2-20ep-lenient/all.jsonl",
      "etp5": A / "etp-sft-20261027/classifier/etp5/all.jsonl", "etp10": A / "etp-sft-20261027/classifier/etp10/all.jsonl",
      "etp20": A / "etp-sft-20261027/classifier/etp20/all.jsonl", "v3etp5": A / "sftv3-etp-20261030/classifier/v3etp5/all.jsonl",
      "v3etp10": A / "sftv3-etp-20261030/classifier/v3etp10/all.jsonl",
      "v3etp20": A / "sftv3-etp20-20261104/classifier/v3etp20/all.jsonl"}


def avg_ranks(xs):
    s = sorted((x, i) for i, x in enumerate(xs))
    r = [0.0] * len(xs)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1][0] == s[i][0]:
            j += 1
        for k in range(i, j + 1):
            r[s[k][1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    return num / den


def main():
    tasks = json.loads((TA / "tasks.json").read_text())["tasks"]
    key = {t["task_id"]: t["run"] for t in tasks}
    v2 = collections.Counter()
    v2_long = collections.Counter()
    for line in open(CORPUS / "kimi-pass-v2-20261023/selection/kimi_pass_v2.jsonl"):
        r = json.loads(line)
        if key.get(r.get("task_id")) == r.get("run"):
            v2[r["task_id"]] += 1
            if int(r.get("token_length") or 0) > 262144:
                v2_long[r["task_id"]] += 1
    v3 = collections.Counter()
    v3b = collections.Counter()
    for line in open(CORPUS / "sft-v3-20261029/selection/sft_v3.jsonl"):
        r = json.loads(line)
        if key.get(r.get("task_id")) == r.get("run"):
            v3[r["task_id"]] += 1
            v3b[r.get("rollout_batch")] += 1
    lines = [f"SFT-v2 convs on the 50 tasks {sum(v2.values())} covering {len(v2)}; >262,144 tokens {dict(v2_long)}",
             f"SFT-v3 convs {sum(v3.values())} covering {len(v3)}; by batch {dict(v3b)}; uncovered {sorted(set(key) - set(v3))}"]
    res = {}
    for m in MODELS:
        d = {}
        for f in sorted(TA.glob(f"results/{m}-group-*.json")):
            for t, rs in json.loads(f.read_text())["records"].items():
                d[t] = (sum(r["status"] == "pass" for r in rs), sum(r["status"] in ("pass", "fail") for r in rs))
        res[m] = d
    bins = [(0, 0), (1, 4), (5, 8), (9, 12)]
    for lo, hi in bins:
        ts = [t for t in key if lo <= v3.get(t, 0) <= hi]
        cells = []
        for m in MODELS:
            p = sum(res[m][t][0] for t in ts)
            g = sum(res[m][t][1] for t in ts)
            cells.append(f"{m} {p}/{g}" + (f" ({100 * p / g:.1f}%)" if g else ""))
        lines.append(f"bin {lo}-{hi}: {len(ts)} tasks | " + "; ".join(cells))
    order = list(key)
    conv = [v3.get(t, 0) for t in order]
    rate = lambda m, t: res[m][t][0] / res[m][t][1]
    sp20 = pearson(avg_ranks(conv), avg_ranks([rate("v3etp20", t) for t in order]))
    spb = pearson(avg_ranks(conv), avg_ranks([rate("sc14", t) for t in order]))
    spd = pearson(avg_ranks(conv), avg_ranks([rate("v3etp20", t) - rate("v3etp10", t) for t in order]))
    lines.append(f"Spearman conv vs v3etp20 rate {sp20:.3f}; vs base rate {spb:.3f}; vs (v3etp20 - v3etp10) {spd:.3f}")
    # train-acc vs DeepSWE
    for m in MODELS:
        rows = [json.loads(l) for l in open(DS[m])]
        res_n = sum((r.get("reward") or 0) >= 1 for r in rows)
        empty = sum(bool(r.get("empty_patch")) for r in rows)
        p = sum(v[0] for v in res[m].values())
        g = sum(v[1] for v in res[m].values())
        lines.append(f"{m}: train-acc {p}/{g} = {100 * p / g:.2f}% | DeepSWE {res_n}/{len(rows)} = {100 * res_n / len(rows):.2f}% "
                     f"(empty patches {empty}) | gap {100 * p / g - 100 * res_n / len(rows):+.2f}")
    (OUT / "coverage.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
