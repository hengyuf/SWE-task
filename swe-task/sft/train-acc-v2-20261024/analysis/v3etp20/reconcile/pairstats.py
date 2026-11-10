"""reconcile (2026-11-04): paired-by-task statistics shared by the reconcile scripts. Stdlib only.

paired(a, b): a, b = {task: (k, n)} for models A and B over the same 50 tasks (n may differ by task and model).
  point  = pooled rate B - pooled rate A (pp)
  CI     = 95% task bootstrap of the pooled difference (10,000 resamples of the tasks, seed 20261104)
  p      = exact two-sided sign-flip p on the per-task rate differences d_t = k_B/n_B - k_A/n_A (dynamic programming
           over the exact distribution of sum(+-d_t); d_t scaled to integers by the LCM of the denominators)
  better/equal/worse = tasks with d_t > 0 / = 0 / < 0
fisher(a, b, c, d): two-sided Fisher exact on the 2x2 table [[a, b], [c, d]].
"""
import math
import random
from fractions import Fraction


def fisher(a, b, c, d):
    n1, n2, k = a + b, c + d, a + c
    n = n1 + n2

    def p(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)
    obs = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= obs * (1 + 1e-9)))


def signflip_exact(diffs):
    fr = [Fraction(x).limit_denominator(1000) for x in diffs if x != 0]
    if not fr:
        return 1.0
    lcm = 1
    for f in fr:
        lcm = lcm * f.denominator // math.gcd(lcm, f.denominator)
    ints = [abs(int(f * lcm)) for f in fr]
    obs = abs(sum(int(f * lcm) for f in fr))
    dist = {0: 1}
    for v in ints:
        nd = {}
        for s, c in dist.items():
            nd[s + v] = nd.get(s + v, 0) + c
            nd[s - v] = nd.get(s - v, 0) + c
        dist = nd
    tot = 2 ** len(ints)
    return sum(c for s, c in dist.items() if abs(s) >= obs) / tot


def paired(a, b, reps=10000, seed=20261104):
    tasks = sorted(set(a) & set(b))
    ka, na = sum(a[t][0] for t in tasks), sum(a[t][1] for t in tasks)
    kb, nb = sum(b[t][0] for t in tasks), sum(b[t][1] for t in tasks)
    point = 100 * (kb / nb - ka / na)
    rng = random.Random(seed)
    boots = []
    for _ in range(reps):
        s = [rng.choice(tasks) for _ in tasks]
        ka_, na_ = sum(a[t][0] for t in s), sum(a[t][1] for t in s)
        kb_, nb_ = sum(b[t][0] for t in s), sum(b[t][1] for t in s)
        boots.append(100 * (kb_ / nb_ - ka_ / na_))
    boots.sort()
    lo, hi = boots[int(0.025 * reps)], boots[int(0.975 * reps) - 1]
    diffs = [Fraction(b[t][0], b[t][1]) - Fraction(a[t][0], a[t][1]) for t in tasks if a[t][1] and b[t][1]]
    p = signflip_exact(diffs)
    return {"A": f"{ka}/{na}", "B": f"{kb}/{nb}", "diff_pp": round(point, 1), "ci": [round(lo, 1), round(hi, 1)], "p": p,
            "better": sum(1 for d in diffs if d > 0), "equal": sum(1 for d in diffs if d == 0), "worse": sum(1 for d in diffs if d < 0),
            "fisher": fisher(kb, nb - kb, ka, na - ka)}
