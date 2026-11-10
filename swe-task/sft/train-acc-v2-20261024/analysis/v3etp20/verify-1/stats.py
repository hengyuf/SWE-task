#!/usr/bin/env python3
"""verify-1: paired task tests on rollout-level flags and per-rollout cost, from my own per-slot rows
(out/export-<m>.jsonl.gz, out/http-<m>.jsonl.gz). Exact sign-flip p (DP over the 50 per-task differences), task
bootstrap 95% CI (20,000 resamples, own seed), Fisher exact on slots. Stdlib only.

    python3 -I -B stats.py -> out/stats.{json,txt}
"""
import collections
import gzip
import json
import math
import random
import re
import statistics
from fractions import Fraction
from pathlib import Path

OUT = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024/analysis/v3etp20/verify-1/out")
MODELS = ["sc14", "etp20", "v3etp5", "v3etp10", "v3etp20"]
CJK = re.compile("[぀-ヿ㐀-䶿一-鿿가-힯]")


def fisher(a, b, c, d):
    n1, n2, k = a + b, c + d, a + c
    n = n1 + n2

    def p(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)
    obs = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= obs * (1 + 1e-9)))


def exact_signflip(diffs):
    den = 1
    for d in diffs:
        den = den * d.denominator // math.gcd(den, d.denominator)
    ints = [int(d * den) for d in diffs]
    obs = abs(sum(ints))
    dist = {0: 1}
    for v in ints:
        nd = collections.defaultdict(int)
        for s, c in dist.items():
            nd[s + v] += c
            nd[s - v] += c
        dist = nd
    return sum(c for s, c in dist.items() if abs(s) >= obs) / 2 ** len(ints)


def mc_signflip(diffs, rng, n=200000):
    obs = abs(sum(diffs))
    ext = 0
    for _ in range(n):
        ext += abs(sum(x if rng.random() < 0.5 else -x for x in diffs)) >= obs - 1e-9
    return (ext + 1) / (n + 1)


def ending(r):
    if not r.get("finished"):
        if r["stdout_len"] == 0 and not r.get("export_found"):
            return "sandbox_lost"
        w = wall(r)
        if r.get("agent_error") == "TimeoutError" or (w or 0) >= 10500:
            return "deadline_kill"
        return "not_finished_other"
    if r.get("commits_ok"):
        return "commit_off_head"
    last = r.get("last_turn") or {}
    if last.get("finish") == "length":
        return "runaway" if (last.get("out") or 0) > 0 else "context_overflow"
    if last.get("think_only") or last.get("unclosed"):
        return "headless" if (last.get("r_markup") or last.get("r_special")) else "reasoning_only"
    if last.get("v_markup") or last.get("v_special"):
        return "malformed"
    if r.get("edits_ok"):
        return "uncommitted"
    return "no_edit_or_other"


def wall(r):
    if r.get("resp_created_at") and r.get("first_t"):
        return r["resp_created_at"] - r["first_t"] / 1000.0
    return None


def load(m):
    ex = {(r["task_id"], r["slot"].rsplit("-r", 1)[1]): r for r in (json.loads(l) for l in gzip.open(OUT / f"export-{m}.jsonl.gz", "rt"))}
    ht = {(r["task_id"], r["slot"].rsplit("-r", 1)[1]): r for r in (json.loads(l) for l in gzip.open(OUT / f"http-{m}.jsonl.gz", "rt"))}
    rows = {}
    for k, r in ex.items():
        h = ht[k]
        calls = [c for c in h["calls"] if "non200" not in c]
        empty = not r.get("patch_chars")
        e = ending(r) if empty else None
        tc = any(((c.get("summary") or {}).get("tools", 0) > 0 or (c.get("summary") or {}).get("markup")) for c in r.get("comps") or [])
        rows[k] = {
            "task": k[0], "status": r["status"], "graded": r["status"] in ("pass", "fail"),
            "empty_patch": empty, "edited_uncommitted": e == "uncommitted", "deadline_kill": e == "deadline_kill" or (
                not r.get("finished") and not (r["stdout_len"] == 0 and not r.get("export_found"))
                and (r.get("agent_error") == "TimeoutError" or (wall(r) or 0) >= 10500)),
            "runaway_end": e == "runaway", "stopped_after_compaction_continue": empty and bool(r.get("n_comp")) and bool(r.get("finished")),
            "toolcall_summary": tc, "unreasoned_action": bool(r.get("export")) and r.get("unreasoned_actions", 0) > 0,
            "length_cut": any(c["finish"] == "length" for c in calls),
            "foreign_spont": any(c.get("r_foreign_spont") for c in calls),
            "cjk_spont": any(any((not inctx) and CJK.search(run) for run, inctx in (c.get("r_foreign") or [])) for c in calls),
            "invalid_tool": (r.get("invalid_tool_calls") or 0) > 0,
            "wall_min": (wall(r) / 60) if wall(r) is not None else None,
            "calls": len(calls), "completion_ktok": sum(c["ct"] or 0 for c in calls) / 1000,
            "model_min": sum((c["ts"] - c["req_ts"]) for c in calls if c.get("ts") and c.get("req_ts")) / 60,
            "_calls": [(c["ct"] or 0, (c["ts"] - c["req_ts"]) if c.get("ts") and c.get("req_ts") else None) for c in calls],
        }
    return rows


def paired_flag(ra, rb, flag, rng, n=20000, only_graded=False):
    tasks = sorted({v["task"] for v in ra.values()} & {v["task"] for v in rb.values()})
    cell = {}
    for t in tasks:
        a = [v for v in ra.values() if v["task"] == t and (v["graded"] or not only_graded)]
        b = [v for v in rb.values() if v["task"] == t and (v["graded"] or not only_graded)]
        cell[t] = (sum(bool(v[flag]) for v in a), len(a), sum(bool(v[flag]) for v in b), len(b))
    tasks = [t for t in tasks if cell[t][1] and cell[t][3]]

    def diff(s):
        return 100 * (sum(cell[t][2] for t in s) / sum(cell[t][3] for t in s) - sum(cell[t][0] for t in s) / sum(cell[t][1] for t in s))
    boots = sorted(diff([tasks[rng.randrange(len(tasks))] for _ in tasks]) for _ in range(n))
    diffs = [Fraction(cell[t][2], cell[t][3]) - Fraction(cell[t][0], cell[t][1]) for t in tasks]
    ka, na = sum(cell[t][0] for t in tasks), sum(cell[t][1] for t in tasks)
    kb, nb = sum(cell[t][2] for t in tasks), sum(cell[t][3] for t in tasks)
    return {"a": f"{ka}/{na}", "b": f"{kb}/{nb}", "diff": round(diff(tasks), 1),
            "ci": [round(boots[int(0.025 * n)], 1), round(boots[int(0.975 * n) - 1], 1)],
            "signflip_exact": exact_signflip(diffs), "fisher": fisher(kb, nb - kb, ka, na - ka)}


def paired_metric(ra, rb, metric, rng, n=20000):
    tasks = sorted({v["task"] for v in ra.values()} & {v["task"] for v in rb.values()})
    sums = {}
    for t in tasks:
        a = [v[metric] for v in ra.values() if v["task"] == t and v[metric] is not None and v["wall_min"] is not None]
        b = [v[metric] for v in rb.values() if v["task"] == t and v[metric] is not None and v["wall_min"] is not None]
        sums[t] = (sum(a), len(a), sum(b), len(b))

    def diff(s):
        return sum(sums[t][2] for t in s) / sum(sums[t][3] for t in s) - sum(sums[t][0] for t in s) / sum(sums[t][1] for t in s)
    boots = sorted(diff([tasks[rng.randrange(len(tasks))] for _ in tasks]) for _ in range(n))
    return {"a": round(sum(v[0] for v in sums.values()) / sum(v[1] for v in sums.values()), 2),
            "b": round(sum(v[2] for v in sums.values()) / sum(v[3] for v in sums.values()), 2),
            "diff": round(diff(tasks), 2), "ci": [round(boots[int(0.025 * n)], 1), round(boots[int(0.975 * n) - 1], 1)]}


def ols(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope, my - slope * mx


def main():
    rng = random.Random(777)
    data = {m: load(m) for m in MODELS}
    out = {"flags": {}, "cost": {}}
    lines = []
    flags = ["empty_patch", "edited_uncommitted", "deadline_kill", "runaway_end", "stopped_after_compaction_continue",
             "toolcall_summary", "unreasoned_action", "length_cut", "foreign_spont", "cjk_spont", "invalid_tool"]
    for f in flags:
        out["flags"][f] = {}
        counts = {m: sum(bool(v[f]) for v in data[m].values()) for m in MODELS}
        lines.append(f"{f}: " + ", ".join(f"{m} {counts[m]}/{len(data[m])}" for m in MODELS))
        for ref in ("v3etp10", "v3etp5", "etp20", "sc14"):
            r = paired_flag(data[ref], data["v3etp20"], f, rng)
            out["flags"][f][ref] = r
            lines.append(f"   v3etp20 vs {ref}: {r['b']} vs {r['a']} {r['diff']:+.1f} {r['ci']} exact sign-flip p {r['signflip_exact']:.5f}; Fisher {r['fisher']:.3g}")
    r = paired_flag(data["v3etp10"], data["v3etp20"], "empty_patch", rng, only_graded=True)
    out["flags"]["empty_patch_graded_v3etp10"] = r
    lines.append(f"empty_patch|graded v3etp20 vs v3etp10: {r}")
    for ref in ("v3etp10", "etp20"):
        out["cost"][ref] = {}
        for metric in ("wall_min", "calls", "completion_ktok", "model_min"):
            r = paired_metric(data[ref], data["v3etp20"], metric, rng)
            out["cost"][ref][metric] = r
            lines.append(f"cost v3etp20 vs {ref} {metric}: {r}")
    # counterfactual: v3etp20 model minutes at v3etp10's serving speed (OLS latency ~ completion tokens over v3etp10 calls)
    xs, ys = [], []
    for v in data["v3etp10"].values():
        if v["wall_min"] is None:
            continue
        for ct, lat in v["_calls"]:
            if lat is not None:
                xs.append(ct)
                ys.append(lat)
    slope, icpt = ols(xs, ys)
    rows_b = [v for v in data["v3etp20"].values() if v["wall_min"] is not None]
    cf = sum(slope * v["completion_ktok"] * 1000 + icpt * v["calls"] for v in rows_b) / len(rows_b) / 60
    out["counterfactual_model_min_at_v3etp10_speed"] = round(cf, 2)
    lines.append(f"v3etp20 model min at v3etp10 serving speed: {cf:.2f} (slope {slope * 1000:.2f} s/1k tok, intercept {icpt:.2f} s)")
    # tasks worse v3etp20 vs v3etp10 and tool-call summary rollouts there
    def rate(rows, t):
        g = [v for v in rows.values() if v["task"] == t and v["graded"]]
        return sum(v["status"] == "pass" for v in g) / len(g) if g else None
    tasks = sorted({v["task"] for v in data["v3etp20"].values()})
    worse = [t for t in tasks if rate(data["v3etp20"], t) < rate(data["v3etp10"], t)]
    better = [t for t in tasks if rate(data["v3etp20"], t) > rate(data["v3etp10"], t)]
    tcw = {m: sum(v["toolcall_summary"] for v in data[m].values() if v["task"] in worse) for m in ("v3etp10", "v3etp20")}
    tcb = {m: sum(v["toolcall_summary"] for v in data[m].values() if v["task"] in better) for m in ("v3etp10", "v3etp20")}
    out["worse_tasks"] = {"n": len(worse), "tc_rollouts": tcw, "better_n": len(better), "tc_rollouts_better": tcb}
    lines.append(f"tasks worse {len(worse)}: tool-call-summary rollouts v3etp10 {tcw['v3etp10']} -> v3etp20 {tcw['v3etp20']}; "
                 f"tasks better {len(better)}: {tcb}")
    loss = sorted(((rate(data["v3etp20"], t) - rate(data["v3etp10"], t), t) for t in tasks))[:6]
    lines.append("largest losses: " + "; ".join(f"{t[4:30]} {d * 100:+.0f}pp tc {sum(v['toolcall_summary'] for v in data['v3etp10'].values() if v['task'] == t)}->"
                                                  f"{sum(v['toolcall_summary'] for v in data['v3etp20'].values() if v['task'] == t)}" for d, t in loss))
    # wall nearest-rank median
    for m in MODELS:
        w = sorted(v["wall_min"] for v in data[m].values() if v["wall_min"] is not None)
        c = sorted(v["calls"] for v in data[m].values() if v["wall_min"] is not None)
        lines.append(f"{m}: wall nearest-rank median {w[len(w) // 2]:.1f} p90 {w[min(len(w) - 1, int(0.9 * len(w)))]:.1f} mean "
                     f"{statistics.fmean(w):.1f} n {len(w)}; calls mean {statistics.fmean(c):.1f} nearest-rank median {c[len(c) // 2]} "
                     f">=500 {sum(x >= 500 for x in c)}; completion ktok mean {statistics.fmean(v['completion_ktok'] for v in data[m].values() if v['wall_min'] is not None):.1f}; "
                     f"model min mean {statistics.fmean(v['model_min'] for v in data[m].values() if v['wall_min'] is not None):.1f}")
    (OUT / "stats.json").write_text(json.dumps(out, indent=1))
    (OUT / "stats.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
