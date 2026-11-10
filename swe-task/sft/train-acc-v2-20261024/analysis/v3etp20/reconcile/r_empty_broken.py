#!/usr/bin/env python3
"""reconcile (2026-11-04): empty patches that follow a broken compaction summary (>= 1 summary with tool-call markup /
a tool part, or cut at the length limit), per model and by the classifier's ending code, from out/export-<m>.jsonl.gz
(own scan) and the classifier rows (analysis/{,v3etp20/}empty-patch/<m>/all.jsonl, used only for the ending code).
Read-only; stdlib only.

    /usr/bin/python3 -I -B r_empty_broken.py -> out/empty_broken.txt
"""
import collections
import gzip
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")


def cls_path(m):
    return (TA / "analysis/v3etp20/empty-patch" if m == "v3etp20" else TA / "analysis/empty-patch") / m / "all.jsonl"


def main():
    L = []
    for m in ("sc14", "etp20", "v3etp5", "v3etp10", "v3etp20"):
        rows = {(r["task_id"], r["rep"]): r for r in (json.loads(x) for x in gzip.open(HERE / f"out/export-{m}.jsonl.gz", "rt"))}
        code = {}
        for x in open(cls_path(m)):
            c = json.loads(x)
            code[(c["task_id"], c["ri"])] = (c["code"], c.get("subcode"), c["empty_patch"])
        emp = [k for k, r in rows.items() if not r["patch_chars"]]
        broken = lambda r: r["export"] and any(s["markup"] or s["tools"] or s["finish"] == "length" for s in r["summaries"])
        by = collections.Counter()
        for k in emp:
            c, sub, _ = code[k]
            lab = ("uncommitted" if c == "E1_UNCOMMITTED" else "deadline/3h" if c.startswith("N1") or (c == "N9_OTHER_NF") else
                   "runaway" if sub == "E3a_cap_cut" else "format" if c == "E3_PREMATURE_END" else c)
            by[(lab, bool(broken(rows[k])))] += 1
        nb = sum(1 for k in emp if broken(rows[k]))
        L.append(f"{m:8s} empty patches {len(emp)}; after a broken summary {nb}; other {len(emp) - nb} | "
                 + ", ".join(f"{lab}{' (broken)' if b else ''} {v}" for (lab, b), v in sorted(by.items())))
    (HERE / "out/empty_broken.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
