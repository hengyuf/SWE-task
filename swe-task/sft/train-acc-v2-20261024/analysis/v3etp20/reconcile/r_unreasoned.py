#!/usr/bin/env python3
"""reconcile (2026-11-04): rollouts with >= 1 unreasoned action turn, v3etp20 vs v3etp10, on the two possible bases.
  export basis   rollouts with a session export (v3etp10 248, v3etp20 250), from out/export-<m>.jsonl.gz
  all slots      v3etp10's 2 no-export infra slots taken from their HTTP journal (out/http_targeted.txt:
                 wasmtime-c03 r1 has 1 action call with no reasoning, ibis-c03-r1b4 r3 none)
Paired task bootstrap + exact sign-flip (pairstats.py). Read-only; stdlib only.

    /usr/bin/python3 -I -B r_unreasoned.py -> out/unreasoned.txt
"""
import collections
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pairstats import paired  # noqa: E402

HTTP_NOEXPORT = {("syn-wasmtime-cand-c03-353ec720", 1): 1, ("syn-ibis-cand-c03-r1b4-3d3b86b5", 3): 0}


def rows(m):
    return [json.loads(x) for x in gzip.open(HERE / f"out/export-{m}.jsonl.gz", "rt")]


def main():
    L = []
    R10, R20 = rows("v3etp10"), rows("v3etp20")
    for basis in ("export", "all slots"):
        tasks = {}
        for name, R in (("v3etp10", R10), ("v3etp20", R20)):
            d = collections.defaultdict(lambda: [0, 0])
            for r in R:
                if r["export"]:
                    d[r["task_id"]][1] += 1
                    d[r["task_id"]][0] += r["n_unreasoned"] > 0
                elif basis == "all slots":
                    d[r["task_id"]][1] += 1
                    d[r["task_id"]][0] += HTTP_NOEXPORT[(r["task_id"], r["rep"])] > 0
            tasks[name] = {t: tuple(v) for t, v in d.items()}
        p = paired(tasks["v3etp10"], tasks["v3etp20"])
        L.append(f"{basis:9s}: v3etp10 {p['A']} -> v3etp20 {p['B']}: {p['diff_pp']:+.1f} pp [{p['ci'][0]:+.1f}, {p['ci'][1]:+.1f}], "
                 f"sign-flip p {p['p']:.3f}, Fisher {p['fisher']:.3f}, tasks better/equal/worse {p['better']}/{p['equal']}/{p['worse']}")
    t10 = sum(r.get("n_unreasoned", 0) for r in R10)
    a10 = sum(r.get("n_action", 0) for r in R10)
    t20 = sum(r.get("n_unreasoned", 0) for r in R20)
    a20 = sum(r.get("n_action", 0) for r in R20)
    L.append(f"turns: v3etp20 {t20}/{a20} ({100 * t20 / a20:.4f}%), v3etp10 {t10}/{a10} ({100 * t10 / a10:.4f}%) -> ratio of counts "
             f"{t20 / t10:.2f}, of rates {(t20 / a20) / (t10 / a10):.2f}; with the HTTP-journal turn of wasmtime-c03 r1: counts {t20 / (t10 + 1):.2f}")
    (HERE / "out/unreasoned.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
