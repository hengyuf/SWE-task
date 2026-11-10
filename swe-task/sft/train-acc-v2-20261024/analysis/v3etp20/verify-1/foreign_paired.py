#!/usr/bin/env python3
"""verify-1: paired task test (exact sign-flip, task bootstrap) of rollouts with a spontaneous non-Latin run in
reasoning (Greek runs < 3 chars skipped; foreign.py definition), v3etp20 vs v3etp10 and vs base.
    python3 -I -B foreign_paired.py -> out/foreign_paired.txt"""
import collections, importlib.util, json, math, random, sys
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import foreign  # noqa: E402  (verify-1/foreign.py)
TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
def flags(model):
    keys, ads = [], []
    for f in sorted(TA.glob(f"results/{model}-group-*.json")):
        for t, rs in json.loads(f.read_text())["records"].items():
            for r in rs:
                keys.append(t); ads.append(r["attempt_dir"])
    with Pool(12) as pool:
        res = pool.map(foreign.scan, ads, chunksize=2)
    per = collections.defaultdict(lambda: [0, 0])
    for t, x in zip(keys, res):
        per[t][1] += 1
        per[t][0] += bool(x and x[0])
    return per
def test(a, b, rng):
    tasks = sorted(a)
    diff = lambda s: 100 * (sum(b[t][0] for t in s) / sum(b[t][1] for t in s) - sum(a[t][0] for t in s) / sum(a[t][1] for t in s))
    boots = sorted(diff([tasks[rng.randrange(50)] for _ in range(50)]) for _ in range(20000))
    d = [Fraction(b[t][0], b[t][1]) - Fraction(a[t][0], a[t][1]) for t in tasks]
    den = 1
    for x in d:
        den = den * x.denominator // math.gcd(den, x.denominator)
    ints = [int(x * den) for x in d]
    dist = {0: 1}
    for v in ints:
        nd = collections.defaultdict(int)
        for s, c in dist.items():
            nd[s + v] += c; nd[s - v] += c
        dist = nd
    p = sum(c for s, c in dist.items() if abs(s) >= abs(sum(ints))) / 2 ** 50
    return f"{sum(b[t][0] for t in tasks)}/{sum(b[t][1] for t in tasks)} vs {sum(a[t][0] for t in tasks)}/{sum(a[t][1] for t in tasks)}: {diff(tasks):+.1f} [{boots[500]:+.1f}, {boots[19499]:+.1f}] exact sign-flip p {p:.2g}"
if __name__ == "__main__":
    rng = random.Random(5)
    F = {m: flags(m) for m in ("sc14", "v3etp10", "v3etp20")}
    lines = [f"v3etp20 vs v3etp10: {test(F['v3etp10'], F['v3etp20'], rng)}", f"v3etp20 vs sc14: {test(F['sc14'], F['v3etp20'], rng)}"]
    (HERE / "out/foreign_paired.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
