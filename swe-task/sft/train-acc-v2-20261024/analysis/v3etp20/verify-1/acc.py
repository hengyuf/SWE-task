#!/usr/bin/env python3
"""verify-1 (independent): accuracy numbers and paired comparisons from results/<model>-group-*.json only.

Written from scratch for the v3etp20 verification. Definitions (as stated in REPORT.md):
  pass@1 = passes / graded (status pass|fail; infra excluded); 95% CI = task bootstrap (resample the 50 tasks).
  pass@5 = tasks with >= 1 pass; tasks 5/5 = tasks with 5 passes.
  patched = graded rollouts with patch_chars > 0; resolved/patched.
  empty patch = slots (of 250) with patch_chars falsy; (among graded).
  Paired B - A: pooled pass@1 difference over tasks graded in both; bootstrap CI (own RNG); sign-flip p on the per-task
  rate differences computed EXACTLY by dynamic programming (not Monte Carlo); better/equal/worse; McNemar exact on
  pass@5 discordant tasks; decomposition into patch-rate part (p_B-p_A)(r_A+r_B)/2 and resolve-rate part (r_B-r_A)(p_A+p_B)/2.
Output: verify-1/out/acc.json and acc.txt
"""
import json
import math
import random
from fractions import Fraction
from pathlib import Path

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
OUT = TA / "analysis/v3etp20/verify-1/out"
MODELS = ["sc14", "sftv2", "sftv2e10", "sftv2e20", "etp5", "etp10", "etp20", "v3etp5", "v3etp10", "v3etp20"]


def load():
    data = {}
    for m in MODELS:
        recs = {}
        for f in sorted(TA.glob(f"results/{m}-group-*.json")):
            d = json.loads(f.read_text())
            assert d["model"] == m
            for t, rs in d["records"].items():
                assert t not in recs
                recs[t] = rs
        data[m] = recs
    return data


def summ(recs, rng, nboot=20000):
    tasks = sorted(recs)
    rows = []
    for t in tasks:
        g = [r for r in recs[t] if r["status"] in ("pass", "fail")]
        p = sum(r["status"] == "pass" for r in g)
        q = sum(bool(r.get("patch_chars")) for r in g)
        rows.append((t, p, len(g), q))
    P = sum(r[1] for r in rows)
    G = sum(r[2] for r in rows)
    Q = sum(r[3] for r in rows)
    boots = []
    for _ in range(nboot):
        s = [rows[rng.randrange(len(rows))] for _ in rows]
        boots.append(sum(r[1] for r in s) / sum(r[2] for r in s))
    boots.sort()
    allslots = [r for t in tasks for r in recs[t]]
    empty_all = sum(not r.get("patch_chars") for r in allslots)
    empty_graded = sum(not r.get("patch_chars") for r in allslots if r["status"] in ("pass", "fail"))
    infra = sum(r["status"] not in ("pass", "fail") for r in allslots)
    return {"passes": P, "graded": G, "pass1": round(100 * P / G, 1),
            "ci": [round(100 * boots[int(0.025 * nboot)], 1), round(100 * boots[int(0.975 * nboot) - 1], 1)],
            "pass5": sum(r[1] > 0 for r in rows), "five_of_five": sum(r[1] == 5 for r in rows),
            "patched_graded": Q, "resolved_of_patched": f"{P}/{Q}", "empty_all": empty_all, "empty_graded": empty_graded,
            "slots": len(allslots), "infra": infra, "tasks": len(rows),
            "timeout_error": sum(r.get("agent_error") == "TimeoutError" for r in allslots)}


def exact_signflip(diffs):
    """Two-sided exact sign-flip p for the mean of diffs (Fractions), P(|S| >= |obs|)."""
    den = 1
    for d in diffs:
        den = den * d.denominator // math.gcd(den, d.denominator)
    ints = [int(d * den) for d in diffs]
    obs = abs(sum(ints))
    dist = {0: 1}
    for v in ints:
        nd = {}
        for s, c in dist.items():
            nd[s + v] = nd.get(s + v, 0) + c
            nd[s - v] = nd.get(s - v, 0) + c
        dist = nd
    tot = 2 ** len(ints)
    ext = sum(c for s, c in dist.items() if abs(s) >= obs)
    return ext / tot


def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in probs if p <= probs[k] * (1 + 1e-9)))


def paired(data, a, b, rng, nboot=20000):
    rows = []
    for t in sorted(data[a]):
        if t not in data[b]:
            continue
        ga = [r for r in data[a][t] if r["status"] in ("pass", "fail")]
        gb = [r for r in data[b][t] if r["status"] in ("pass", "fail")]
        if not ga or not gb:
            continue
        rows.append((t, sum(r["status"] == "pass" for r in ga), len(ga), sum(bool(r.get("patch_chars")) for r in ga),
                     sum(r["status"] == "pass" for r in gb), len(gb), sum(bool(r.get("patch_chars")) for r in gb)))

    def stat(s):
        pa, ga, qa, pb, gb, qb = (sum(r[i] for r in s) for i in range(1, 7))
        d = pb / gb - pa / ga
        p_a, p_b = qa / ga, qb / gb
        r_a, r_b = (pa / qa if qa else 0.0), (pb / qb if qb else 0.0)
        return d, (p_b - p_a) * (r_a + r_b) / 2, (r_b - r_a) * (p_a + p_b) / 2, p_a, p_b, r_a, r_b

    pt = stat(rows)
    bs = [[], [], []]
    for _ in range(nboot):
        s = [rows[rng.randrange(len(rows))] for _ in rows]
        v = stat(s)
        for i in range(3):
            bs[i].append(v[i])
    cis = []
    for i in range(3):
        bs[i].sort()
        cis.append([round(100 * bs[i][int(0.025 * nboot)], 1), round(100 * bs[i][int(0.975 * nboot) - 1], 1)])
    diffs = [Fraction(r[4], r[5]) - Fraction(r[1], r[2]) for r in rows]
    better = sum(d > 0 for d in diffs)
    worse = sum(d < 0 for d in diffs)
    sa = {r[0] for r in rows if r[1] > 0}
    sb = {r[0] for r in rows if r[4] > 0}
    oa, ob = len(sa - sb), len(sb - sa)
    return {"a": a, "b": b, "tasks": len(rows),
            "pass_a": f"{sum(r[1] for r in rows)}/{sum(r[2] for r in rows)}",
            "pass_b": f"{sum(r[4] for r in rows)}/{sum(r[5] for r in rows)}",
            "diff_pp": round(100 * pt[0], 1), "ci": cis[0],
            "task_mean_diff_pp": round(float(100 * sum(diffs) / len(diffs)), 1),
            "signflip_p_exact": exact_signflip(diffs),
            "better_equal_worse": f"{better}/{len(rows) - better - worse}/{worse}",
            "pass5": f"{len(sa)} -> {len(sb)}", "only_a": oa, "only_b": ob,
            "mcnemar_p": binom_two_sided(min(oa, ob), oa + ob),
            "patch_part": round(100 * pt[1], 1), "patch_ci": cis[1],
            "resolve_part": round(100 * pt[2], 1), "resolve_ci": cis[2],
            "p_a": round(100 * pt[3], 1), "p_b": round(100 * pt[4], 1), "r_a": round(100 * pt[5], 1),
            "r_b": round(100 * pt[6], 1)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()
    rng = random.Random(4242)
    res = {"models": {m: summ(data[m], rng) for m in MODELS}}
    pairs = [("v3etp10", "v3etp20"), ("v3etp5", "v3etp20"), ("etp20", "v3etp20"), ("sc14", "v3etp20"),
             ("etp10", "v3etp20"), ("etp5", "v3etp20"), ("sftv2e20", "v3etp20"), ("sftv2e10", "v3etp20"),
             ("sftv2", "v3etp20"), ("etp20", "v3etp10")]
    res["pairs"] = [paired(data, a, b, rng) for a, b in pairs]
    # per-task pass counts for v3etp20 vs others
    tasks = [t["task_id"] for t in json.loads((TA / "tasks.json").read_text())["tasks"]]
    per = {}
    for t in tasks:
        per[t] = {m: "".join({"pass": "P", "fail": ".", "infra": "x"}.get(r["status"], "?") for r in data[m].get(t, []))
                  for m in MODELS}
    res["per_task"] = per
    # pass@5 gains/losses v3etp10 -> v3etp20; tasks never solved
    p5 = {m: {t for t in tasks if "P" in per[t][m]} for m in MODELS}
    res["new_solved_vs_v3etp10"] = sorted(p5["v3etp20"] - p5["v3etp10"])
    res["lost_vs_v3etp10"] = sorted(p5["v3etp10"] - p5["v3etp20"])
    res["never_v3etp20"] = sorted(set(tasks) - p5["v3etp20"])
    res["never_any"] = sorted(set(tasks) - set().union(*p5.values()))
    res["solved_by_runs"] = {t: [m for m in MODELS if t in p5[m]] for t in res["new_solved_vs_v3etp10"]}
    # model-side infra as fail for v3etp10 (the 3 model-side slots) -> pooled
    (OUT / "acc.json").write_text(json.dumps(res, indent=1, default=str))
    lines = []
    for m in MODELS:
        s = res["models"][m]
        lines.append(f"{m:9s} pass@1 {s['pass1']}% ({s['passes']}/{s['graded']}) {s['ci']} pass@5 {s['pass5']}/50 "
                     f"5/5 {s['five_of_five']} resolved/patched {s['resolved_of_patched']} empty {s['empty_all']} "
                     f"({s['empty_graded']}/{s['graded']}) infra {s['infra']} TimeoutError {s['timeout_error']}")
    for p in res["pairs"]:
        lines.append(f"{p['b']} vs {p['a']}: {p['pass_a']} -> {p['pass_b']} {p['diff_pp']:+.1f} {p['ci']} "
                     f"exact sign-flip p {p['signflip_p_exact']:.5f} b/e/w {p['better_equal_worse']} pass@5 {p['pass5']} "
                     f"McNemar {p['mcnemar_p']:.4f} patch {p['patch_part']:+.1f} {p['patch_ci']} resolve "
                     f"{p['resolve_part']:+.1f} {p['resolve_ci']} p {p['p_a']}->{p['p_b']} r {p['r_a']}->{p['r_b']} "
                     f"taskmean {p['task_mean_diff_pp']:+.1f}")
    for k in ("new_solved_vs_v3etp10", "lost_vs_v3etp10", "never_v3etp20", "never_any", "solved_by_runs"):
        lines.append(f"{k}: {res[k]}")
    (OUT / "acc.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
