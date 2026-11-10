#!/usr/bin/env python3
"""reconcile (2026-11-04): the verifier problems that the export scan can settle (out/export-<model>.jsonl.gz from
r_export_scan.py; pass/fail from results/*.json). Read-only; stdlib only.

    /usr/bin/python3 -I -B r_analyze.py -> out/analyze.txt, out/analyze.json

 C1 compaction summaries per model; tool-call summaries with '## Goal' / the commit instruction in the body (P6)
 C2 per-task change v3etp10 -> v3etp20 (graded slots): rollouts with a tool-call summary in tasks that got worse /
    stayed equal / got better, and in the floor tasks (0 passes in both runs) (P8)
 C3 rollouts with an unreasoned action turn on the export basis (v3etp10 248) and paired tests (P10)
 C4 v3etp20 reasoning turns with tool-call markup (P11)
 C5 v3etp20 length-cut calls: dominant unit of the last 4,000 chars (P4)
 C6 non-Latin runs in reasoning: lens rule (spontaneous = not in earlier user/tool text; Greek < 3 skipped) and two
    stricter variants: 'unquoted' (see quoted())
    and 'fused' (glued to a Latin letter on either side, unquoted) (P12); 'quoted' = inside inline code or a double-quoted
    string on its line, or attached to a quote character
"""
import collections
import gzip
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pairstats import paired, fisher  # noqa: E402

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
MODELS = ["sc14", "etp20", "v3etp5", "v3etp10", "v3etp20"]


def load(m):
    return [json.loads(line) for line in gzip.open(HERE / f"out/export-{m}.jsonl.gz", "rt")]


def per_task(rows, flag, base=lambda r: r["export"]):
    d = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if base(r):
            d[r["task_id"]][1] += 1
            d[r["task_id"]][0] += bool(flag(r))
    return {t: tuple(v) for t, v in d.items()}


def tc(s):
    return s["markup"] or s["tools"]


QUOTES = "\"'`“”‘’«»「」"


def quoted(f):
    """inside inline code (odd backticks before it on its line and one after), inside a double-quoted string (odd
    double quotes before it on its line and one after), or directly attached to a quote character."""
    ll, lr = f.get("line_left", f["left"]), f.get("line_right", f["right"])
    if ll.count("`") % 2 == 1 and "`" in lr:
        return True
    if ll.count('"') % 2 == 1 and '"' in lr:
        return True
    if ll[-1:] in QUOTES or lr[:1] in QUOTES:
        return True
    return False


def fused(f):
    return bool(re.search(r"[A-Za-z]$", f["left"])) or bool(re.match(r"[A-Za-z]", f["right"]))


def main():
    D = {m: load(m) for m in MODELS}
    L, J = [], {}
    # ---- C1
    L.append("== C1 compaction summaries (export)")
    for m in MODELS:
        S = [s for r in D[m] if r["export"] for s in r["summaries"]]
        T = [s for s in S if tc(s)]
        L.append(f"{m:8s} summaries {len(S)} tool-call {len(T)} ({100 * len(T) / len(S):.1f}%) in "
                 f"{sum(1 for r in D[m] if r['export'] and any(tc(s) for s in r['summaries']))} rollouts; "
                 f"tool-call summaries with '## Goal' in the body {sum(s['goal'] for s in T)}, anywhere {sum(s['goal_anywhere'] for s in T)}; "
                 f"with the commit instruction in the body {sum(s['instr'] for s in T)}; with markup in the body (after </think>) "
                 f"{sum(s['markup_in_body'] for s in T)}")
    T20 = [(r["task_id"], r["rep"], s) for r in D["v3etp20"] if r["export"] for s in r["summaries"] if tc(s)]
    L.append("v3etp20 tool-call summaries that contain '## Goal' (body head):")
    for t, rep, s in T20:
        if s["goal"] or s["goal_anywhere"]:
            L.append(f"   {t} r{rep} msg {s['i']} instr {s['instr']} finish {s['finish']} body_len {s['body_len']} head {s['head'][:120]!r}")
    # first instruction loss by kind (lens definition) in v3etp20 to tie '34 via tool-call'
    lost = collections.Counter()
    for r in D["v3etp20"]:
        summ = r.get("summaries") or []
        if not summ:
            continue
        k = next((s for s in summ if not s["instr"]), None)
        lost["never" if k is None else ("toolcall" if tc(k) else "runaway" if k["finish"] == "length" else "empty" if not k["body_len"] else "omitted")] += 1
    L.append(f"v3etp20 first summary without the commit instruction, by kind: {dict(lost)}")
    # ---- C2
    L.append("\n== C2 per task v3etp10 -> v3etp20 (pass rate over graded slots); rollouts with >= 1 tool-call summary (export)")
    res = {}
    for m in ("v3etp10", "v3etp20"):
        for r in D[m]:
            res.setdefault(m, collections.defaultdict(list))[r["task_id"]].append(r)
    groups = collections.defaultdict(lambda: {"tasks": 0, "v3etp10": 0, "v3etp20": 0, "list": []})
    for t in sorted(res["v3etp20"]):
        a = [r for r in res["v3etp10"][t] if r["status"] in ("pass", "fail")]
        b = [r for r in res["v3etp20"][t] if r["status"] in ("pass", "fail")]
        ra = sum(r["status"] == "pass" for r in a) / len(a)
        rb = sum(r["status"] == "pass" for r in b) / len(b)
        g = "worse" if rb < ra else "better" if rb > ra else "equal"
        if g == "equal" and ra == 0:
            g = "equal (0 passes in both)"
        ta = sum(1 for r in res["v3etp10"][t] if r["export"] and any(tc(s) for s in r["summaries"]))
        tb = sum(1 for r in res["v3etp20"][t] if r["export"] and any(tc(s) for s in r["summaries"]))
        groups[g]["tasks"] += 1
        groups[g]["v3etp10"] += ta
        groups[g]["v3etp20"] += tb
        if ta or tb:
            groups[g]["list"].append(f"{t.split('-cand-')[0][4:]}-{t.split('-cand-')[1].rsplit('-', 1)[0]} {ta}->{tb}")
    for g in ("worse", "equal", "equal (0 passes in both)", "better"):
        x = groups[g]
        L.append(f"  {g:26s} tasks {x['tasks']:2d} tool-call-summary rollouts {x['v3etp10']} -> {x['v3etp20']} | {', '.join(x['list'])}")
    J["C2"] = {g: {k: v for k, v in x.items()} for g, x in groups.items()}
    # resolve rate of compacted rollouts by summary state
    for m in ("v3etp10", "v3etp20"):
        comp = [r for r in D[m] if r["export"] and r["summaries"] and r["status"] in ("pass", "fail")]
        bad = [r for r in comp if any(tc(s) for s in r["summaries"])]
        bad2 = [r for r in comp if any(tc(s) or s["finish"] == "length" for s in r["summaries"])]
        fine = [r for r in comp if r not in bad2]
        L.append(f"  {m}: compacted graded {len(comp)}; with a tool-call summary resolve {sum(r['status'] == 'pass' for r in bad)}/{len(bad)}; "
                 f"with a tool-call or length-cut summary {sum(r['status'] == 'pass' for r in bad2)}/{len(bad2)}; "
                 f"fine summaries {sum(r['status'] == 'pass' for r in fine)}/{len(fine)}")
    # ---- C3
    L.append("\n== C3 rollouts with >= 1 unreasoned action turn (export; v3etp10 has 248 exports)")
    U = {m: per_task(D[m], lambda r: r["n_unreasoned"] > 0) for m in MODELS}
    for m in MODELS:
        k = sum(v[0] for v in U[m].values())
        n = sum(v[1] for v in U[m].values())
        turns = sum(r.get("n_unreasoned", 0) for r in D[m])
        acts = sum(r.get("n_action", 0) for r in D[m])
        L.append(f"  {m:8s} rollouts {k}/{n} (every action reasoned {n - k}/{n}); turns {turns}/{acts} ({100 * turns / acts:.4f}%)")
    pr = paired(U["v3etp10"], U["v3etp20"])
    L.append(f"  v3etp20 vs v3etp10, export basis (248 vs 250): {pr}")
    J["C3_export"] = pr
    # 250 basis with the 2 no-export v3etp10 slots counted clean (as stats_tests) -> reproduces the draft
    U250 = {t: (k, 5) for t, (k, n) in U["v3etp10"].items()}
    pr2 = paired(U250, U["v3etp20"])
    L.append(f"  v3etp20 vs v3etp10, 250 basis with the 2 no-export slots counted clean (stats_tests): {pr2}")
    J["C3_250_clean"] = pr2
    L.append(f"  v3etp20 unreasoned turns: " + "; ".join(f"{r['task_id']} r{r['rep']} {u['closed']} {u['text_head']!r}" for r in D['v3etp20'] for u in r.get('unreasoned', [])))
    # ---- C4
    L.append("\n== C4 v3etp20 non-summary reasoning with tool-call markup")
    n = 0
    for r in D["v3etp20"]:
        for x in r.get("markup", []):
            n += 1
            L.append(f"  {r['task_id']} r{r['rep']} msg {x['i']} opener {x['opener']} kinds {x['kinds']} tool parts {x['tools']} finish {x['finish']} "
                     f"closed {x['closed']} status {r['status']} | ...{x['snippet'][100:200]!r}")
    L.append(f"  turns {n} in {sum(1 for r in D['v3etp20'] if r.get('markup'))} rollouts")
    # ---- C5
    L.append("\n== C5 v3etp20 length-cut calls (export finish 'length'): tail zlib and dominant tokens of the last 4,000 chars")
    for r in D["v3etp20"]:
        for x in r.get("length", []):
            L.append(f"  {r['task_id']} r{r['rep']} msg {x['i']} summary {x['summary']} out {x['out']} r_len {x['r_len']} tail_zr {x['tail_zr']} "
                     f"top {x['top']} spaces_in_tail {x['spaces_tail']} last {x['last'][-60:]!r}")
    L.append(f"  length-cut messages {sum(len(r.get('length', [])) for r in D['v3etp20'])} in {sum(1 for r in D['v3etp20'] if r.get('length'))} rollouts")
    # ---- C6
    L.append("\n== C6 non-Latin runs in reasoning (all assistant messages), rollouts")
    variants = {
        "lens (spontaneous)": lambda f: not f["copied"],
        "spontaneous, unquoted": lambda f: not f["copied"] and not quoted(f),
        "spontaneous, fused to a Latin letter, unquoted": lambda f: not f["copied"] and not quoted(f) and fused(f),
    }
    F = {}
    for name, cond in variants.items():
        F[name] = {m: per_task(D[m], lambda r, c=cond: any(c(f) for f in r.get("foreign", []))) for m in MODELS}
        cnt = {m: f"{sum(v[0] for v in F[name][m].values())}/{sum(v[1] for v in F[name][m].values())}" for m in MODELS}
        L.append(f"  {name:48s} {cnt}")
        for ref in ("v3etp10", "v3etp5", "etp20", "sc14"):
            p = paired(F[name][ref], F[name]["v3etp20"])
            L.append(f"      v3etp20 vs {ref}: {p['diff_pp']:+.1f} [{p['ci'][0]:+.1f}, {p['ci'][1]:+.1f}] p={p['p']:.2g} (Fisher {p['fisher']:.2g})")
            J.setdefault("C6", {}).setdefault(name, {})[ref] = p
    # examples: v3etp20 / v3etp10 spontaneous runs dropped by the 'unquoted' rule, and kept by 'fused'
    for m in ("v3etp20", "v3etp10"):
        drop = [(r["task_id"], r["rep"], f) for r in D[m] for f in r.get("foreign", []) if not f["copied"] and quoted(f)]
        keep = [(r["task_id"], r["rep"], f) for r in D[m] for f in r.get("foreign", []) if not f["copied"] and not quoted(f) and fused(f)]
        mid = [(r["task_id"], r["rep"], f) for r in D[m] for f in r.get("foreign", []) if not f["copied"] and not quoted(f) and not fused(f)]
        L.append(f"  {m}: spontaneous runs {sum(1 for r in D[m] for f in r.get('foreign', []) if not f['copied'])}; quoted/code {len(drop)}; "
                 f"unquoted & fused {len(keep)}; unquoted, standalone {len(mid)}")
        for lab, lst in (("quoted/code", drop), ("unquoted standalone", mid), ("fused", keep)):
            for t, rep, f in lst[:8]:
                L.append(f"     [{lab}] {t.split('-cand-')[0][4:]} r{rep}: ...{f['left'][-30:]!r} [{f['run']}] {f['right'][:30]!r}")
    (HERE / "out/analyze.txt").write_text("\n".join(L) + "\n")
    (HERE / "out/analyze.json").write_text(json.dumps(J, indent=1, default=str) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
