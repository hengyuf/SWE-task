#!/usr/bin/env python3
"""Read-only: classify old-grader GLM passes (touch_scan.json) by whether the current sanitizing grader could change
the verdict. A = no effective test-path change, or only NEW test-path files that are not conftest/setup-like and whose
basename no non-test hunk of model.patch mentions; B = an existing test file modified/deleted (outside test.patch; JS/TS
add-only edits excluded, the grader keeps them), or a risky new file."""
import collections, json, re
from pathlib import Path
res = json.load(open("touch_scan.json"))
RISKY = re.compile(r"(^|/)(conftest\.py|pytest\.ini|setup\.(py|cfg|js|ts|mjs|cjs)|jest\.(config|setup)[^/]*|vitest\.(config|setup)[^/]*|__init__\.py)$")
def nontest_text(patch, test_paths):
    keep = []
    for b in re.split(r"(?m)^diff --git ", patch)[1:]:
        head = b.split("\n", 1)[0]
        if not any(head.endswith(" b/" + p) for p in test_paths):
            keep.append(b)
    return "\n".join(keep)
out = collections.Counter(); per = []
for x in res:
    eff = [h for h in x["hits"] if not h["additive_kept"] and not h["in_test_patch"]]
    cls, why = "A", []
    if eff:
        att = None
    for h in eff:
        if h["kind"] != "add":
            cls = "B"; why.append(f"{h['kind']}:{h['path']}")
    if cls == "A" and eff:
        S = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/"
        base = {"tasksall": S + "glm-rollouts-tasksall-20261108", "sftv3": S + "glm-rollouts-sftv3-20261107"}[x["batch"]]
        # locate the record's attempt dir again
        recf = base + ("/analysis/rollouts.jsonl" if x["batch"] == "tasksall" else "/analysis/final-rollouts.jsonl")
        per.append(x)
    x["cls"], x["why"] = cls, why
# second pass for A-with-new-files: referenced-by-name / risky names
S = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/"
recs = {}
for b, f in (("tasksall", "glm-rollouts-tasksall-20261108/analysis/rollouts.jsonl"), ("sftv3", "glm-rollouts-sftv3-20261107/analysis/final-rollouts.jsonl")):
    for l in open(S + f):
        r = json.loads(l); recs[(b, r["campaign"], r["task_id"], r["slot"])] = r
for x in res:
    eff = [h for h in x["hits"] if not h["additive_kept"] and not h["in_test_patch"]]
    if x["cls"] != "A" or not eff:
        continue
    r = recs[(x["batch"], x["campaign"], x["task_id"], x["slot"])]
    p = Path(r["attempt_dir"].replace("/projects/nemotron_agents_dev/users/hengyuf/", "/users/hengyuf/")) / "model.patch"
    text = p.read_bytes().decode("utf-8", "surrogateescape")
    nt = nontest_text(text, [h["path"] for h in x["hits"]])
    for h in eff:
        name = h["path"].rsplit("/", 1)[-1]
        stem = name.split(".")[0]
        if RISKY.search(h["path"]):
            x["cls"] = "B"; x["why"].append("risky-new:" + h["path"])
        elif name in nt or (len(stem) > 3 and re.search(r"\b" + re.escape(stem) + r"\b", nt) and "testdata" in h["path"]):
            x["cls"] = "B"; x["why"].append("referenced-new:" + h["path"])
c = collections.Counter((x["batch"], x["cls"]) for x in res)
print(sorted(c.items()))
print("A:", sum(x["cls"] == "A" for x in res), "B:", sum(x["cls"] == "B" for x in res),
      "B tasks:", len({(x["run"], x["task_id"]) for x in res if x["cls"] == "B"}))
print(collections.Counter(w.split(":")[0] for x in res if x["cls"] == "B" for w in x["why"]))
json.dump(res, open("classified.json", "w"), indent=1)
