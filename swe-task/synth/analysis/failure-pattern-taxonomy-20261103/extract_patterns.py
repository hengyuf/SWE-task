#!/usr/bin/env python3
"""Collect every DeepSWE TARGET failure analysis (runs/<run>/analysis/<task>/result.json) of hengyuf's synth runs and of
Baihe's fork, and write one row per failure pattern (patterns.jsonl) plus one row per analysis (analyses.jsonl).
One analysis per target is marked primary: official4 > official3 > official2 > official1 > val1 > other hengyuf runs >
Baihe's fork (newest data first). Read-only on the runs; stdlib only."""
import json, re, sys
from pathlib import Path
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
SOURCES = [("hengyuf", S / "synth/runs"), ("baihe-fork", Path("/lustre/fsw/portfolios/nemotron/users/baiheh/swe-task-hengyu/synth/runs"))]
TASKS = S / "synth/cache/deep-swe/tasks"
OUT = Path(__file__).resolve().parent

def rank(source, run):
    if source != "hengyuf":
        return 9
    for i, prefix in enumerate(("official4", "official3", "official2", "official1", "val1")):
        if run.startswith(prefix):
            return i
    return 8

def language(task):
    toml = TASKS / task / "task.toml"
    try:
        m = re.search(r'^\s*language\s*=\s*"([^"]+)"', toml.read_text(), re.M)
    except OSError:
        return None
    lang = m.group(1) if m else None
    if task == "koota-entity-snapshot-rollback":
        lang = "typescript"  # task.toml says python, but its tests are .ts (pmndrs/koota); see official4 notes
    return lang

analyses, patterns = [], []
for source, root in SOURCES:
    for res in sorted(root.glob("*/analysis/*/result.json")):
        run, task = res.parts[-4], res.parts[-2]
        try:
            r = json.loads(res.read_text())
        except (OSError, ValueError):
            continue
        d = res.parent
        stats = {}
        try:
            stats = json.loads((d / "inputs/stats.json").read_text())
        except (OSError, ValueError):
            pass
        md = d / "failure_analysis.md"
        a = {"source": source, "run": run, "target": task, "language": language(task), "rank": rank(source, run),
             "result_json": str(res), "failure_analysis_md": str(md) if md.exists() else None,
             "md_bytes": md.stat().st_size if md.exists() else None,
             "n_patterns": len(r.get("patterns") or []), "transferable_signature": r.get("transferable_signature"),
             "excluded": len(r.get("excluded") or []), "selection": (stats.get("selection") or {}).get("show"),
             "pass_stats": stats.get("pass_stats")}
        analyses.append(a)
# primary analysis per target
best = {}
for a in analyses:
    k = a["target"]
    if k not in best or (a["rank"], a["run"]) < (best[k]["rank"], best[k]["run"]):
        best[k] = a
for a in analyses:
    a["primary"] = best[a["target"]] is a
for a in analyses:
    r = json.loads(Path(a["result_json"]).read_text())
    student = (a.get("pass_stats") or {}).get("sc14") or {}
    graded = (student.get("pass", 0) or 0) + (student.get("fail", 0) or 0)
    for i, p in enumerate(r.get("patterns") or []):
        patterns.append({"id": f"{a['source']}:{a['run']}:{a['target']}:{i}", "primary": a["primary"], "source": a["source"],
                         "run": a["run"], "target": a["target"], "language": a["language"],
                         "name": p.get("name"), "category": p.get("category"), "description": p.get("description"),
                         "teacher_contrast": p.get("teacher_contrast"), "n_student_slots": len(p.get("student_slots") or []),
                         "student_graded": graded or None, "student_fail": student.get("fail")})
(OUT / "analyses.jsonl").write_text("".join(json.dumps(a) + "\n" for a in analyses))
(OUT / "patterns.jsonl").write_text("".join(json.dumps(p) + "\n" for p in patterns))
prim = [a for a in analyses if a["primary"]]
pp = [p for p in patterns if p["primary"]]
from collections import Counter
print("analyses", len(analyses), "| targets (primary)", len(prim), "| patterns all", len(patterns), "| primary", len(pp))
print("primary by source/run-prefix:", Counter((a["source"], a["run"].split("-")[0]) for a in prim))
print("primary by language:", Counter(a["language"] for a in prim))
print("patterns per primary analysis:", Counter(a["n_patterns"] for a in prim))
print("distinct analyst 'category' strings (primary):", len({p["category"] for p in pp}))
print("pattern text size (primary, chars):", sum(len(json.dumps({k: p[k] for k in ("name","category","description","teacher_contrast")})) for p in pp))
