#!/usr/bin/env python3
"""verify-1: reasoning length per reasoning turn. (a) aggregation check on the analyst's extractor token lists
(features jsonl: reasoning_tokens per reasoning turn; tokenizer not re-run, /usr/bin/python3 has no `tokenizers`);
(b) independent proxy from my own export scan: reasoning characters per reasoning turn; KS distance to base for both.
    python3 -I -B rtok.py -> out/rtok.txt"""
import bisect, gzip, json, statistics
from pathlib import Path
TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
OUT = TA / "analysis/v3etp20/verify-1/out"
TR = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-rollouts-v2-20261025/features")
F = {"sc14": TR / "sc14.jsonl", "etp20": TA / "analysis/features/etp20.jsonl", "v3etp5": TA / "analysis/features/v3etp5.jsonl",
     "v3etp10": TA / "analysis/features/v3etp10.jsonl", "v3etp20": TA / "analysis/v3etp20/features/v3etp20.jsonl"}
def q(xs, p):
    return xs[min(len(xs) - 1, int(p * len(xs)))]
def ks(a, b):
    a, b = sorted(a), sorted(b)
    pts = sorted(set(a) | set(b))
    return max(abs(bisect.bisect_right(a, x) / len(a) - bisect.bisect_right(b, x) / len(b)) for x in pts)
lines = []
tok, chars = {}, {}
for m, p in F.items():
    xs = sorted(t for l in open(p) if l.strip() for t in (json.loads(l).get("reasoning_tokens") or []))
    tok[m] = xs
    ys = sorted(c for l in gzip.open(OUT / f"export-{m}.jsonl.gz", "rt") for c in (json.loads(l).get("reasoning_lens") or []))
    chars[m] = ys
for m in F:
    xs, ys = tok[m], chars[m]
    lines.append(f"{m}: tokens n {len(xs)} median(nearest-rank) {q(xs, .5)} mean {statistics.fmean(xs):.0f} p90 {q(xs, .9)} "
                 f"<50 {100 * sum(x < 50 for x in xs) / len(xs):.1f}% KS-to-base {ks(xs, tok['sc14']):.3f} | chars n {len(ys)} median {q(ys, .5)} "
                 f"mean {statistics.fmean(ys):.0f} p90 {q(ys, .9)} KS-to-base {ks(ys, chars['sc14']):.3f}")
(OUT / "rtok.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
