#!/usr/bin/env python3
"""Old-grader classes of every tier-1/2 GLM-5.3 pass (DESIGN.md "Old-grader rule"): regrade/classes.json.

Inputs: touch_scan.py (scan of model.patch test-path hits for passes on old-grader tasks: tests/grader.py of 13,468
bytes) -> classify.py (A / B) -> classified.json; plus every pass on a current-grader task (class "n/a").
Key of a pass: "<batch>/<campaign>/<task_id>/<slot>" with batch tasksall | sftv3.
  python3 -I -B make_classes.py   (run touch_scan.py and classify.py first, in this dir)
"""
import hashlib, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft")
cl = json.loads((HERE / "classified.json").read_text())
old = {f"{x['batch']}/{x['campaign']}/{x['task_id']}/{x['slot']}": x for x in cl}
passes = {}
for batch, f in (("tasksall", "glm-rollouts-tasksall-20261108/analysis/rollouts.jsonl"),
                 ("sftv3", "glm-rollouts-sftv3-20261107/analysis/final-rollouts.jsonl")):
    for line in (S / f).read_text().splitlines():
        r = json.loads(line)
        if r["status"] != "pass" or (batch == "tasksall" and r["tier"] not in (1, 2)):
            continue
        key = f"{batch}/{r['campaign']}/{r['task_id']}/{r['slot']}"
        x = old.get(key)
        passes[key] = {"grader_kind": "old" if x else "current", "class": x["cls"] if x else "n/a",
                       "language": x["lang"] if x else None, "run": x["run"] if x else None,
                       "why": x["why"] if x else [], "hits": x["hits"] if x else [],
                       "attempt_dir": r["attempt_dir"]}
assert len(old) == sum(1 for p in passes.values() if p["grader_kind"] == "old") == 704
src = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ("touch_scan.py", "classify.py", "make_classes.py")}
out = {"rule": "DESIGN.md Old-grader rule (rev 2)", "tools_sha256": src, "n_passes": len(passes),
       "counts": {k: sum(1 for p in passes.values() if p["class"] == k) for k in ("n/a", "A", "B")}, "passes": passes}
tmp = HERE / ".classes.json.tmp"; tmp.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n"); tmp.replace(HERE / "classes.json")
print(out["n_passes"], out["counts"])
