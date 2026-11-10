#!/usr/bin/env python3
"""verify-2: paired task-level comparisons of per-rollout quantities between models from out/slots_<model>.jsonl
(metrics.py) and out/<model>.jsonl.gz (calls, for the serving-speed counterfactual).

    python3 -I -B compare.py
"""
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "acc"))
from rate_tests import exact_signflip, fisher  # noqa: E402


def slots(model):
    p = HERE / "out" / f"slots_{model}.jsonl"
    return [json.loads(x) for x in open(p)] if p.exists() else None


def paired_mean(A, B, key, keep=lambda r: True, scale=1.0, seed=11, nboot=20000):
    ta, tb = defaultdict(list), defaultdict(list)
    for r in A:
        if keep(r) and r.get(key) is not None:
            ta[r["task"]].append(r[key] * scale)
    for r in B:
        if keep(r) and r.get(key) is not None:
            tb[r["task"]].append(r[key] * scale)
    tasks = sorted(t for t in ta if t in tb)
    sa = np.array([sum(ta[t]) for t in tasks])
    na = np.array([len(ta[t]) for t in tasks], float)
    sb = np.array([sum(tb[t]) for t in tasks])
    nb = np.array([len(tb[t]) for t in tasks], float)
    point = sb.sum() / nb.sum() - sa.sum() / na.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(tasks), size=(nboot, len(tasks)))
    bs = sb[idx].sum(1) / nb[idx].sum(1) - sa[idx].sum(1) / na[idx].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{point:+.1f} [{lo:+.1f}, {hi:+.1f}] (A mean {sa.sum() / na.sum():.1f}, B mean {sb.sum() / nb.sum():.1f}; n {int(na.sum())}/{int(nb.sum())})"


def paired_flag(A, B, f, keep=lambda r: True, seed=13, nboot=20000):
    ta, tb = defaultdict(list), defaultdict(list)
    for r in A:
        if keep(r):
            ta[r["task"]].append(int(bool(f(r))))
    for r in B:
        if keep(r):
            tb[r["task"]].append(int(bool(f(r))))
    tasks = sorted(t for t in ta if t in tb)
    ka = np.array([sum(ta[t]) for t in tasks], float)
    na = np.array([len(ta[t]) for t in tasks], float)
    kb = np.array([sum(tb[t]) for t in tasks], float)
    nb = np.array([len(tb[t]) for t in tasks], float)
    point = 100 * (kb.sum() / nb.sum() - ka.sum() / na.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(tasks), size=(nboot, len(tasks)))
    bs = 100 * (kb[idx].sum(1) / nb[idx].sum(1) - ka[idx].sum(1) / na[idx].sum(1))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = exact_signflip(list(kb / nb - ka / na))
    fp = fisher(int(kb.sum()), int(nb.sum() - kb.sum()), int(ka.sum()), int(na.sum() - ka.sum()))
    return f"{int(ka.sum())}/{int(na.sum())} -> {int(kb.sum())}/{int(nb.sum())}: {point:+.1f} pp [{lo:+.1f}, {hi:+.1f}] sign-flip p {p:.4f} Fisher {fp:.2g}"


def ols(model):
    xs, ys = [], []
    with gzip.open(HERE / "out" / f"{model}.jsonl.gz", "rt") as fh:
        for line in fh:
            s = json.loads(line)
            for c in s["calls"]:
                if c.get("resp") and c.get("t_done") and c.get("t_last_req"):
                    xs.append(c["resp"]["ct"] or 0)
                    ys.append(c["t_done"] - c["t_last_req"])
    x, y = np.array(xs, float), np.array(ys, float)
    slope, icpt = np.polyfit(x, y, 1)
    return slope, icpt


def main():
    S = {m: slots(m) for m in ["v3etp20", "v3etp10", "v3etp5", "etp20", "sc14"]}
    L = []
    B = S["v3etp20"]
    for ref in ["v3etp10", "etp20", "v3etp5", "sc14"]:
        A = S[ref]
        if A is None:
            continue
        L.append(f"== v3etp20 vs {ref}")
        L.append(f"  completion tokens per rollout (k): {paired_mean(A, B, 'ct_sum', scale=1e-3)}")
        L.append(f"  model time per rollout (min): {paired_mean(A, B, 'model_time_s', scale=1 / 60)}")
        L.append(f"  model calls per rollout: {paired_mean(A, B, 'n_calls_200')}")
        L.append(f"  wall per rollout (min): {paired_mean(A, B, 'wall_s', scale=1 / 60)}")
        L.append(f"  wall per rollout (min), slots with a finished-or-killed session only (export-like, excl. no-response infra): "
                 f"{paired_mean(A, B, 'wall_s', keep=lambda r: r['n_calls_200'] > 0, scale=1 / 60)}")
        L.append(f"  deadline kills: {paired_flag(A, B, lambda r: r['finished'] is False and (r['agent_error'] == 'TimeoutError' or (r['wall_s'] or 0) >= 10500))}")
        L.append(f"  >= 1 unreasoned action: {paired_flag(A, B, lambda r: r['unreasoned_main_session'] > 0)}")
        L.append(f"  >= 1 tool-call compaction summary: {paired_flag(A, B, lambda r: any(c.get('tool') for c in r['comps']))}")
        L.append(f"  >= 1 hallucinated tool name: {paired_flag(A, B, lambda r: bool(r['bad_tool_names']))}")
        L.append(f"  empty patch: {paired_flag(A, B, lambda r: r['empty'])}")
        L.append(f"  >= 1 CJK char in reasoning: {paired_flag(A, B, lambda r: r['cjk_calls'] > 0)}")
        fa = [r["comps"][0] for r in A if r["comps"] and r["comps"][0]["resp"]]
        fb = [r["comps"][0] for r in B if r["comps"] and r["comps"][0]["resp"]]
        a1, b1 = sum(c["tool"] for c in fa), sum(c["tool"] for c in fb)
        L.append(f"  first compaction is a tool call: {a1}/{len(fa)} -> {b1}/{len(fb)} Fisher {fisher(b1, len(fb) - b1, a1, len(fa) - a1):.2g}")
        sa_ = [c for r in A for c in r["comps"] if c["resp"]]
        sb_ = [c for r in B for c in r["comps"] if c["resp"]]
        a2, b2 = sum(c["tool"] for c in sa_), sum(c["tool"] for c in sb_)
        L.append(f"  tool-call summaries: {a2}/{len(sa_)} -> {b2}/{len(sb_)} Fisher {fisher(b2, len(sb_) - b2, a2, len(sa_) - a2):.2g}")
        ca = [r for r in A if r["n_comp"] > 0]
        cb = [r for r in B if r["n_comp"] > 0]
        la = sum(any(c["resp"] and not c["instr"] for c in r["comps"]) for r in ca)
        lb = sum(any(c["resp"] and not c["instr"] for c in r["comps"]) for r in cb)
        L.append(f"  compacted sessions with a summary lacking the commit instruction (own regex): {la}/{len(ca)} -> {lb}/{len(cb)} "
                 f"Fisher {fisher(lb, len(cb) - lb, la, len(ca) - la):.2g}")
    # serving-speed counterfactual: v3etp20's mean completion tokens and calls per rollout at v3etp10's OLS speed
    s10, i10 = ols("v3etp10")
    s20, i20 = ols("v3etp20")
    ct20 = np.mean([r["ct_sum"] for r in B])
    n20 = np.mean([r["n_calls_200"] for r in B])
    L.append(f"OLS latency ~ completion tokens: v3etp10 slope {1000 * s10:.2f} s/1k tok, intercept {i10:.2f} s; v3etp20 slope "
             f"{1000 * s20:.2f}, intercept {i20:.2f}")
    L.append(f"v3etp20 model minutes per rollout at v3etp10 speed: {(s10 * ct20 + i10 * n20) / 60:.1f} (own speed: {(s20 * ct20 + i20 * n20) / 60:.1f}; "
             f"actual mean {np.mean([r['model_time_s'] for r in B]) / 60:.1f})")
    (HERE / "out" / "compare.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
