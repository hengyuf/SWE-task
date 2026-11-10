#!/usr/bin/env python3
"""verify-2: SFT-v2 / SFT-v3 conversations per train-acc task (keyed by (run, task_id)), accuracy by coverage bin,
Spearman (scipy) of conversations vs per-task pass rates."""
import json
from collections import Counter
from pathlib import Path

from scipy.stats import spearmanr

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
C = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus")


def rows(p):
    return [json.loads(x) for x in open(p)]


def res(model):
    out = {}
    for f in sorted((TA / "results").glob(f"{model}-group-*.json")):
        for t, recs in json.loads(f.read_text())["records"].items():
            out[t] = [r["status"] for r in recs]
    return out


def main():
    tasks = json.loads((TA / "tasks.json").read_text())["tasks"]
    keys = {(t["run"], t["task_id"]) for t in tasks}
    v2 = rows(C / "kimi-pass-v2-20261023/selection/kimi_pass_v2.jsonl")
    v3 = rows(C / "sft-v3-20261029/selection/sft_v3.jsonl")
    print("corpus sizes v2", len(v2), "v3", len(v3))
    print("v3 fields:", sorted(v3[0].keys()))
    c2 = Counter((r.get("run"), r.get("task_id")) for r in v2)
    c3 = Counter((r.get("run"), r.get("task_id")) for r in v3)
    # also key by task_id only (in case run naming differs)
    c3t = Counter(r.get("task_id") for r in v3)
    on2 = sum(c2[k] for k in keys)
    on3 = sum(c3[k] for k in keys)
    on3t = sum(c3t[t["task_id"]] for t in tasks)
    print("on the 50 tasks: v2", on2, "tasks", sum(c2[k] > 0 for k in keys), "| v3", on3, "tasks", sum(c3[k] > 0 for k in keys),
          "| v3 by task_id only", on3t)
    bf = Counter()
    for r in v3:
        if (r.get("run"), r.get("task_id")) in keys:
            bf[r.get("rollout_batch") or r.get("batch") or "?"] += 1
    print("v3 on tasks by rollout_batch:", dict(bf))
    print("uncovered by v3:", [t["task_id"] for t in tasks if c3[(t["run"], t["task_id"])] == 0])
    models = ["sc14", "etp20", "v3etp5", "v3etp10", "v3etp20"]
    R = {m: res(m) for m in models}
    bins = [(0, 0), (1, 4), (5, 8), (9, 12), (13, 99)]
    for lo, hi in bins:
        ts = [t for t in tasks if lo <= c3[(t["run"], t["task_id"])] <= hi]
        cells = []
        for m in models:
            p = sum(R[m][t["task_id"]].count("pass") for t in ts)
            g = sum(sum(s in ("pass", "fail") for s in R[m][t["task_id"]]) for t in ts)
            cells.append(f"{m} {p}/{g}")
        print(f"bin {lo}-{hi}: tasks {len(ts)} | " + " | ".join(cells))
    conv = [c3[(t["run"], t["task_id"])] for t in tasks]

    def rate(m, t):
        st = R[m][t["task_id"]]
        g = sum(s in ("pass", "fail") for s in st)
        return st.count("pass") / g if g else 0.0
    r20 = [rate("v3etp20", t) for t in tasks]
    r10 = [rate("v3etp10", t) for t in tasks]
    rb = [rate("sc14", t) for t in tasks]
    print("spearman conv vs v3etp20 rate", round(spearmanr(conv, r20).correlation, 3))
    print("spearman conv vs v3etp20-v3etp10", round(spearmanr(conv, [a - b for a, b in zip(r20, r10)]).correlation, 3))
    print("spearman conv vs base rate", round(spearmanr(conv, rb).correlation, 3))
    # v2 conversations dropped from v3 for the two uncovered tasks (token_length)
    for t in tasks:
        if c3[(t["run"], t["task_id"])] == 0:
            tl = [int(r.get("token_length") or 0) for r in v2 if (r.get("run"), r.get("task_id")) == (t["run"], t["task_id"])]
            print("uncovered", t["task_id"], "v2 convs", c2[(t["run"], t["task_id"])], "v2 token_length", tl)


if __name__ == "__main__":
    main()
