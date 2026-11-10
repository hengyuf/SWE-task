#!/usr/bin/env python3
"""train-acc-v2 (2026-10-31): paired pass@1 tests of the SFT-v3 pairs with the model-side infra slots counted as fails
(context overflow HTTP 400 'fatal' slots of raw 5 ep / etp 10 ep / v3 etp 10 ep, and v3 etp 10 ep's self-kill
'needs_review' slot), same estimators and seeds as analysis/accuracy/paired.py compare(). Read-only on TA/results.
    python3 -I -B sensitivity_infra.py -> sensitivity_infra.json
2026-11-04: copy in analysis/v3etp20/compare, + the v3etp20 pairs (the earlier pairs must reproduce analysis/v3etp).
"""
import collections
import json
from pathlib import Path
import random

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
HERE = TA / "analysis/v3etp20/compare"  # 2026-11-04 copy: outputs here
MODEL_SIDE = {("sftv2", "syn-transitions-cand-c03-c910069f", 0), ("etp10", "syn-jest-cand-c05-9e1caaee", 3),
              ("v3etp10", "syn-wasmtime-cand-c04-r1b1-bbb510ca", 0), ("v3etp10", "syn-wasmtime-cand-c04-r1b1-bbb510ca", 1),
              ("v3etp10", "syn-gomacro-cand-c02-r1b0-e8804dd9", 4)}
PAIRS = (("etp5", "v3etp5"), ("etp10", "v3etp5"), ("etp10", "v3etp10"), ("etp20", "v3etp10"), ("v3etp5", "v3etp10"),
         ("sc14", "v3etp5"), ("sc14", "v3etp10"))
# 2026-11-04: + the v3etp20 pairs (v3etp20 itself has no infra slot; only its partners' model-side slots change)
PAIRS += (("v3etp10", "v3etp20"), ("v3etp5", "v3etp20"), ("etp20", "v3etp20"), ("sc14", "v3etp20"), ("etp10", "v3etp20"),
          ("etp5", "v3etp20"), ("sftv2e20", "v3etp20"), ("sftv2e10", "v3etp20"), ("sftv2", "v3etp20"))


def load():
    found = collections.defaultdict(dict)
    for path in sorted(TA.glob("results/*-group-*.json")):
        data = json.loads(path.read_text())
        for task_id, records in data["records"].items():
            found[data["model"]][task_id] = [
                "fail" if r["status"] == "infra" and (data["model"], task_id, int(r["slot"].rsplit("-r", 1)[1])) in MODEL_SIDE
                else r["status"] for r in records]
    return found


def compare(found, a, b, seed=20261025):
    rows = []
    for t in found[a]:
        pa, ga = found[a][t].count("pass"), sum(s in ("pass", "fail") for s in found[a][t])
        pb, gb = found[b][t].count("pass"), sum(s in ("pass", "fail") for s in found[b][t])
        if ga and gb:
            rows.append((pa, ga, pb, gb))
    rng = random.Random(seed)
    point = 100 * (sum(r[2] for r in rows) / sum(r[3] for r in rows) - sum(r[0] for r in rows) / sum(r[1] for r in rows))
    boots = []
    for _ in range(10000):
        s = [rows[rng.randrange(len(rows))] for _ in rows]
        boots.append(100 * (sum(r[2] for r in s) / sum(r[3] for r in s) - sum(r[0] for r in s) / sum(r[1] for r in s)))
    boots.sort()
    diffs = [r[2] / r[3] - r[0] / r[1] for r in rows]
    observed = sum(diffs) / len(diffs)
    extreme = sum(abs(sum(d if rng.random() < 0.5 else -d for d in diffs) / len(diffs)) >= abs(observed) - 1e-12
                  for _ in range(100000))
    return {"a": a, "b": b, "pass_at_1_diff_pp": round(point, 1), "ci95": [round(boots[250], 1), round(boots[9749], 1)],
            "signflip_p": (extreme + 1) / 100001}


def main():
    found = load()
    pooled = {m: {"passes": sum(v.count("pass") for v in found[m].values()),
                  "graded": sum(sum(s in ("pass", "fail") for s in v) for v in found[m].values())} for m in found}
    out = {"pooled": pooled, "pairs": [compare(found, a, b) for a, b in PAIRS]}
    tmp = HERE / ".sensitivity_infra.json.tmp"
    tmp.write_text(json.dumps(out, indent=1) + "\n")
    tmp.replace(HERE / "sensitivity_infra.json")
    print(json.dumps(out["pairs"], indent=0))


if __name__ == "__main__":
    main()
