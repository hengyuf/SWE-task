#!/usr/bin/env python3
"""reconcile (2026-11-04): is the 'broken summary -> lower resolve rate' association only task difficulty? Within-task
comparison for v3etp20 (and v3etp10): compacted graded rollouts with >= 1 tool-call summary vs compacted graded rollouts
whose summaries were all fine (no tool call, no length cut), restricted to tasks that have both kinds; pooled counts and
a Mantel-Haenszel common odds ratio over those tasks. From out/export-<m>.jsonl.gz. Read-only; stdlib only.

    /usr/bin/python3 -I -B r_within_task.py -> out/within_task.txt
"""
import collections
import gzip
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    L = []
    for m in ("v3etp10", "v3etp20"):
        by = collections.defaultdict(lambda: {"bad": [0, 0], "fine": [0, 0]})
        for line in gzip.open(HERE / f"out/export-{m}.jsonl.gz", "rt"):
            r = json.loads(line)
            if not r["export"] or not r["summaries"] or r["status"] not in ("pass", "fail"):
                continue
            tc = any(s["markup"] or s["tools"] for s in r["summaries"])
            lc = any(s["finish"] == "length" for s in r["summaries"])
            k = "bad" if tc else ("fine" if not lc else None)
            if k:
                by[r["task_id"]][k][0] += r["status"] == "pass"
                by[r["task_id"]][k][1] += 1
        both = {t: v for t, v in by.items() if v["bad"][1] and v["fine"][1]}
        bp = sum(v["bad"][0] for v in both.values())
        bn = sum(v["bad"][1] for v in both.values())
        fp = sum(v["fine"][0] for v in both.values())
        fn = sum(v["fine"][1] for v in both.values())
        num = den = 0.0
        for v in both.values():
            a, b = v["bad"][0], v["bad"][1] - v["bad"][0]
            c, d = v["fine"][0], v["fine"][1] - v["fine"][0]
            n = a + b + c + d
            num += a * d / n
            den += b * c / n
        orr = num / den if den else float("nan")
        L.append(f"{m}: tasks with both kinds {len(both)}; tool-call-summary rollouts resolve {bp}/{bn}, fine-summary rollouts {fp}/{fn}; "
                 f"Mantel-Haenszel OR (bad vs fine) {orr:.2f}")
        for t, v in sorted(both.items()):
            L.append(f"   {t[4:44]:40s} bad {v['bad'][0]}/{v['bad'][1]} fine {v['fine'][0]}/{v['fine'][1]}")
    (HERE / "out/within_task.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
