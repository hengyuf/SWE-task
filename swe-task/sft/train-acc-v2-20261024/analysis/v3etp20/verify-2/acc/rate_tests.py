#!/usr/bin/env python3
"""verify-2: paired task-level tests for per-slot binary flags (own implementation: numpy bootstrap, exact sign-flip).

Used by the other verify-2 scripts; also runs the results-only checks:
  * empty patch (patch_chars == 0) of 250 slots and of graded slots, v3etp20 vs v3etp10 / v3etp5 / etp20 / sc14
  * pass@1 with v3etp10's 3 model-side infra slots counted as fail (sensitivity)
"""
import json
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")


def exact_signflip(diffs):
    fr = [Fraction(d).limit_denominator(10000) for d in diffs]
    L = 1
    for f in fr:
        L = L * f.denominator // math.gcd(L, f.denominator)
    ints = [int(f * L) for f in fr]
    obs = abs(sum(ints))
    dist = {0: 1}
    for v in ints:
        nd = defaultdict(int)
        for s, c in dist.items():
            nd[s + v] += c
            nd[s - v] += c
        dist = nd
    return sum(c for s, c in dist.items() if abs(s) >= obs) / 2 ** len(ints)


def fisher(a, b, c, d):
    n1, n2, k = a + b, c + d, a + c
    n = n1 + n2
    p = lambda x: math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)
    obs = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= obs * (1 + 1e-9)))


def paired_flag(fa, fb, seed=7, nboot=20000):
    """fa, fb: {task: [0/1 per included slot]} -> B - A in pp (pooled per-slot rate), bootstrap CI, exact sign-flip p."""
    tasks = sorted(set(fa) & set(fb))
    tasks = [t for t in tasks if fa[t] and fb[t]]
    ka = np.array([sum(fa[t]) for t in tasks], float)
    na = np.array([len(fa[t]) for t in tasks], float)
    kb = np.array([sum(fb[t]) for t in tasks], float)
    nb = np.array([len(fb[t]) for t in tasks], float)
    point = 100 * (kb.sum() / nb.sum() - ka.sum() / na.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(tasks), size=(nboot, len(tasks)))
    bs = 100 * (kb[idx].sum(1) / nb[idx].sum(1) - ka[idx].sum(1) / na[idx].sum(1))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    diffs = list(kb / nb - ka / na)
    fp = fisher(int(kb.sum()), int(nb.sum() - kb.sum()), int(ka.sum()), int(na.sum() - ka.sum()))
    return {"A": f"{int(ka.sum())}/{int(na.sum())}", "B": f"{int(kb.sum())}/{int(nb.sum())}", "diff": round(point, 1),
            "ci": [round(lo, 1), round(hi, 1)], "signflip_exact": exact_signflip(diffs), "fisher": fp}


def load(model):
    out = {}
    for f in sorted((TA / "results").glob(f"{model}-group-*.json")):
        d = json.loads(f.read_text())
        for t, recs in d["records"].items():
            out[t] = recs
    return out


def main():
    data = {m: load(m) for m in ["sc14", "etp20", "v3etp5", "v3etp10", "v3etp20"]}
    lines = []
    for ref in ["v3etp10", "v3etp5", "etp20", "sc14"]:
        for variant in ("all", "graded"):
            fa, fb = {}, {}
            for t in data["v3etp20"]:
                sel = lambda recs: [r for r in recs if variant == "all" or r["status"] in ("pass", "fail")]
                fa[t] = [int(not r.get("patch_chars")) for r in sel(data[ref][t])]
                fb[t] = [int(not r.get("patch_chars")) for r in sel(data["v3etp20"][t])]
            r = paired_flag(fa, fb)
            lines.append(f"empty patch ({variant}) v3etp20 vs {ref}: {r}")
    # sensitivity: v3etp10 model-side infra (2 context overflows = wasmtime-c04-r1b1 r0/r1, 1 self-kill = gomacro-c02-r1b0 r4)
    infra = [(t, r["slot"], r["agent_error"], r["verifier_error"]) for t, recs in data["v3etp10"].items() for r in recs
             if r["status"] == "infra"]
    lines.append(f"v3etp10 infra slots: {infra}")
    model_side = {("syn-wasmtime-cand-c04-r1b1-bbb510ca", "v3etp10-r0"), ("syn-wasmtime-cand-c04-r1b1-bbb510ca", "v3etp10-r1"),
                  ("syn-gomacro-cand-c02-r1b0-e8804dd9", "v3etp10-r4")}
    fa, fb = {}, {}
    for t in data["v3etp20"]:
        fa[t] = [int(r["status"] == "pass") for r in data["v3etp10"][t]
                 if r["status"] in ("pass", "fail") or (t, r["slot"]) in model_side]
        fb[t] = [int(r["status"] == "pass") for r in data["v3etp20"][t] if r["status"] in ("pass", "fail")]
    lines.append(f"pass@1 v3etp20 vs v3etp10 with v3etp10 model-side infra as fail: {paired_flag(fa, fb)}")
    out = Path(__file__).resolve().parent / "rate_tests.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
