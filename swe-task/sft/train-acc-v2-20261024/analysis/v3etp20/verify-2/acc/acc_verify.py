#!/usr/bin/env python3
"""verify-2 (2026-11-04): independent recomputation of train-acc accuracy numbers and paired tests from results/*.json.

Own code (numpy bootstrap with its own seed; EXACT sign-flip p via a DP over integer-scaled per-task differences;
exact McNemar / sign binomial). Writes acc_verify.{txt,json} next to this script.
"""
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
OUT = Path(__file__).resolve().parent
MODELS = ["sc14", "sftv2", "sftv2e10", "sftv2e20", "etp5", "etp10", "etp20", "v3etp5", "v3etp10", "v3etp20"]


def load():
    data = {m: {} for m in MODELS}
    for m in MODELS:
        files = sorted((TA / "results").glob(f"{m}-group-*.json"))
        assert len(files) == 8, (m, files)
        for f in files:
            d = json.loads(f.read_text())
            assert d["model"] == m
            for t, recs in d["records"].items():
                assert t not in data[m]
                data[m][t] = recs
    return data


def per_model(data, rng):
    out = {}
    for m in MODELS:
        recs = data[m]
        tasks = sorted(recs)
        passes = sum(r["status"] == "pass" for t in tasks for r in recs[t])
        graded = sum(r["status"] in ("pass", "fail") for t in tasks for r in recs[t])
        infra = sum(r["status"] == "infra" for t in tasks for r in recs[t])
        empty_all = sum(not r.get("patch_chars") for t in tasks for r in recs[t])
        empty_graded = sum(not r.get("patch_chars") for t in tasks for r in recs[t] if r["status"] in ("pass", "fail"))
        patched_graded = sum(bool(r.get("patch_chars")) for t in tasks for r in recs[t] if r["status"] in ("pass", "fail"))
        pass_patched = sum(bool(r.get("patch_chars")) for t in tasks for r in recs[t] if r["status"] == "pass")
        pass_empty = sum((not r.get("patch_chars")) for t in tasks for r in recs[t] if r["status"] == "pass")
        p5 = sum(any(r["status"] == "pass" for r in recs[t]) for t in tasks)
        all5 = sum(sum(r["status"] == "pass" for r in recs[t]) == 5 for t in tasks)
        all_graded_pass = sum(
            (sum(r["status"] == "pass" for r in recs[t]) == sum(r["status"] in ("pass", "fail") for r in recs[t]))
            and sum(r["status"] in ("pass", "fail") for r in recs[t]) > 0 for t in tasks)
        tasks_5graded = [t for t in tasks if sum(r["status"] in ("pass", "fail") for r in recs[t]) == 5]
        p5_5graded = sum(any(r["status"] == "pass" for r in recs[t]) for t in tasks_5graded)
        # task bootstrap of pooled pass@1
        P = np.array([sum(r["status"] == "pass" for r in recs[t]) for t in tasks], float)
        G = np.array([sum(r["status"] in ("pass", "fail") for r in recs[t]) for t in tasks], float)
        idx = rng.integers(0, len(tasks), size=(20000, len(tasks)))
        bs = P[idx].sum(1) / G[idx].sum(1) * 100
        lo, hi = np.percentile(bs, [2.5, 97.5])
        # pass@1 counting infra as fail
        out[m] = dict(passes=passes, graded=graded, infra=infra, slots=sum(len(recs[t]) for t in tasks),
                      pass1=100 * passes / graded, ci=[round(lo, 1), round(hi, 1)], pass5=p5, tasks_all5=all5,
                      tasks_all_graded_pass=all_graded_pass, pass5_over_5graded=f"{p5_5graded}/{len(tasks_5graded)}",
                      empty_all=empty_all, empty_graded=empty_graded, patched_graded=patched_graded,
                      pass_patched=pass_patched, pass_empty=pass_empty,
                      pass1_infra_as_fail_all=100 * passes / (passes + (graded - passes) + infra))
    return out


def exact_signflip_p(diffs):
    """Exact two-sided sign-flip p for the mean of per-task differences (all 2^n sign patterns)."""
    fr = [Fraction(d).limit_denominator(1000) for d in diffs]
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
    tot = 2 ** len(ints)
    ext = sum(c for s, c in dist.items() if abs(s) >= obs)
    return ext / tot


def binom2(k, n):
    if n == 0:
        return 1.0
    pr = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in pr if p <= pr[k] * (1 + 1e-9)))


def paired(data, a, b, rng, nboot=20000):
    rows = []
    for t in sorted(data[a]):
        ra, rb = data[a][t], data[b][t]
        ga = [r for r in ra if r["status"] in ("pass", "fail")]
        gb = [r for r in rb if r["status"] in ("pass", "fail")]
        if not ga or not gb:
            continue
        rows.append((t,
                     sum(r["status"] == "pass" for r in ga), len(ga), sum(bool(r.get("patch_chars")) for r in ga),
                     sum(r["status"] == "pass" for r in gb), len(gb), sum(bool(r.get("patch_chars")) for r in gb),
                     sum(r["status"] == "pass" and bool(r.get("patch_chars")) for r in ga),
                     sum(r["status"] == "pass" and bool(r.get("patch_chars")) for r in gb)))
    arr = np.array([r[1:] for r in rows], float)  # pa, ga, qa, pb, gb, qb, ppa, ppb
    pa, ga, qa, pb, gb, qb, ppa, ppb = arr.T

    def stats(s):
        PA, GA, QA, PB, GB, QB = (s[..., i].sum(-1) for i in range(6))
        p_a, p_b = QA / GA, QB / GB
        r_a = np.where(QA > 0, PA / np.maximum(QA, 1), 0.0)
        r_b = np.where(QB > 0, PB / np.maximum(QB, 1), 0.0)
        d = PB / GB - PA / GA
        part_p = (p_b - p_a) * (r_a + r_b) / 2
        part_r = (r_b - r_a) * (p_a + p_b) / 2
        return d, part_p, part_r, p_a, p_b, r_a, r_b

    d0, pp0, pr0, p_a, p_b, r_a, r_b = stats(arr)
    idx = rng.integers(0, len(rows), size=(nboot, len(rows)))
    d, pp_, pr_, *_ = stats(arr[idx])
    ci = lambda x: [round(float(v) * 100, 1) for v in np.percentile(x, [2.5, 97.5])]
    diffs = list(pb / gb - pa / ga)
    better = sum(x > 1e-12 for x in diffs)
    worse = sum(x < -1e-12 for x in diffs)
    sa = {r[0] for r in rows if r[1] > 0}
    sb = {r[0] for r in rows if r[4] > 0}
    oa, ob = len(sa - sb), len(sb - sa)
    return dict(a=a, b=b, tasks=len(rows), A=f"{int(pa.sum())}/{int(ga.sum())}", B=f"{int(pb.sum())}/{int(gb.sum())}",
                diff=round(float(d0) * 100, 1), ci=ci(d), task_mean=round(100 * sum(diffs) / len(diffs), 1),
                signflip_exact=exact_signflip_p(diffs), better=better, equal=len(rows) - better - worse, worse=worse,
                sign_p=binom2(min(better, worse), better + worse), p5a=len(sa), p5b=len(sb), only_a=oa, only_b=ob,
                mcnemar=binom2(min(oa, ob), oa + ob), part_patch=round(float(pp0) * 100, 1), part_patch_ci=ci(pp_),
                part_resolve=round(float(pr0) * 100, 1), part_resolve_ci=ci(pr_),
                p_a=round(float(p_a) * 100, 1), p_b=round(float(p_b) * 100, 1), r_a=round(float(r_a) * 100, 1),
                r_b=round(float(r_b) * 100, 1))


PAIRS = [("v3etp10", "v3etp20"), ("v3etp5", "v3etp20"), ("etp20", "v3etp20"), ("sc14", "v3etp20"),
         ("etp10", "v3etp20"), ("etp5", "v3etp20"), ("sftv2e20", "v3etp20"), ("sftv2e10", "v3etp20"),
         ("sftv2", "v3etp20"), ("etp20", "v3etp10")]


def main():
    data = load()
    rng = np.random.default_rng(424242)
    pm = per_model(data, rng)
    lines = ["model passes/graded pass1 [ci] infra pass5 all5 allgradedpass pass5(5graded) empty_all(empty_graded) "
             "pass/patched_graded pass_with_empty infra_as_fail"]
    for m in MODELS:
        x = pm[m]
        lines.append(f"{m:9s} {x['passes']}/{x['graded']} {x['pass1']:.1f} {x['ci']} infra={x['infra']} p5={x['pass5']} "
                     f"all5={x['tasks_all5']} allgp={x['tasks_all_graded_pass']} p5_5g={x['pass5_over_5graded']} "
                     f"empty={x['empty_all']}({x['empty_graded']}) {x['pass_patched']}/{x['patched_graded']} "
                     f"pass_empty={x['pass_empty']} infra_fail={x['pass1_infra_as_fail_all']:.1f}")
    lines.append("")
    res = []
    for a, b in PAIRS:
        r = paired(data, a, b, rng)
        res.append(r)
        lines.append(f"{b} vs {a}: {r['A']} -> {r['B']} diff {r['diff']:+.1f} {r['ci']} task_mean {r['task_mean']:+.1f} "
                     f"signflip_exact {r['signflip_exact']:.5f} b/e/w {r['better']}/{r['equal']}/{r['worse']} "
                     f"sign_p {r['sign_p']:.4f} p5 {r['p5a']}->{r['p5b']} (onlyA {r['only_a']} onlyB {r['only_b']}) "
                     f"mcnemar {r['mcnemar']:.4f} patch-part {r['part_patch']:+.1f} {r['part_patch_ci']} "
                     f"resolve-part {r['part_resolve']:+.1f} {r['part_resolve_ci']} p {r['p_a']}->{r['p_b']} "
                     f"r {r['r_a']}->{r['r_b']}")
    # per-task changes for v3etp20 vs v3etp10 (and the solved sets)
    lines.append("")
    tasks = sorted(data["v3etp20"])
    for t in tasks:
        row = []
        for m in MODELS:
            st = [r["status"] for r in data[m][t]]
            row.append("".join({"pass": "P", "fail": ".", "infra": "x"}[s] for s in st))
        lines.append(f"{t:50s} " + " ".join(row))
    (OUT / "acc_verify.txt").write_text("\n".join(lines) + "\n")
    (OUT / "acc_verify.json").write_text(json.dumps({"per_model": pm, "pairs": res}, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    print("\n".join(lines[:len(MODELS) + 2 + len(PAIRS)]))


if __name__ == "__main__":
    main()
