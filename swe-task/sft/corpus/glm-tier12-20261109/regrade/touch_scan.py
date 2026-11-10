#!/usr/bin/env python3
"""Read-only: for every tier-1/2 GLM pass on an OLD-grader task (tests/grader.py 13,468 bytes, no test-path sanitizer),
list the model.patch paths that the current sanitizing grader would restore/delete (verifier/test_paths.py language
globs; JS/TS non_additive = existing test files with add-only changes are kept). Output: touch_scan.json."""
import collections, json, os, re, sys
from pathlib import Path
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
GR2 = S / "sft/glm-rollouts-tasksall-20261108"; GR = S / "sft/glm-rollouts-sftv3-20261107"
sys.path.insert(0, str(S / "synth/verifier"))
import importlib.util
spec = importlib.util.spec_from_file_location("tp", S / "synth/verifier/test_paths.py"); tp = importlib.util.module_from_spec(spec); spec.loader.exec_module(tp)
TOP100 = S / "top100.jsonl"

def glob_match(pattern, path):
    ps, xs = pattern.strip("/").split("/"), path.split("/")
    import fnmatch
    def m(i, j):
        if i == len(ps): return j == len(xs)
        if ps[i] == "**":
            if i == len(ps) - 1: return j < len(xs)
            return any(m(i + 1, k) for k in range(j, len(xs) + 1))
        return j < len(xs) and fnmatch.fnmatchcase(xs[j], ps[i]) and m(i + 1, j + 1)
    return m(0, 0)

def touched(text):
    out = []  # (path, kind) kind: add/mod/del ; additive flag per path
    blocks = re.split(r"(?m)^diff --git ", text)
    for b in blocks[1:]:
        head = b.split("\n", 1)[0]
        mm = re.match(r'a/(.*) b/(.*)$', head)
        if not mm: out.append((head, "unparsed", False)); continue
        a, bb = mm.group(1), mm.group(2)
        new = "\nnew file mode" in b[:400]; dele = "\ndeleted file mode" in b[:400]
        hunks = [l for l in b.split("\n") if l.startswith("-") and not l.startswith("---")]
        additive = (not new and not dele and a == bb and not hunks and "@@" in b)
        for p in {a, bb}:
            out.append((p, "add" if new else "del" if dele else "mod", additive))
    return out

def is_old(td):
    g = Path(td) / "tests/grader.py"
    return g.exists() and g.stat().st_size == 13468

inv = {}
for l in (S / "sft/tasks-all-20261106/all_tasks.jsonl").read_text().splitlines():
    if l.strip():
        r = json.loads(l); inv[(r["run"], r["task_id"])] = r
recs = []
t2 = {(x["campaign"], x["task_id"]): x for x in json.loads((GR2 / "tasks.json").read_text())["tasks"]}
for l in (GR2 / "analysis/rollouts.jsonl").read_text().splitlines():
    r = json.loads(l)
    if r["tier"] in (1, 2) and r["status"] == "pass":
        x = t2[(r["campaign"], r["task_id"])]
        recs.append(("tasksall", r, x["task_dir"], x.get("repo"), x.get("target"), x["run"]))
t1 = {x["task_id"]: x for x in json.loads((GR / "tasks.json").read_text())["tasks"]}
for l in (GR / "analysis/final-rollouts.jsonl").read_text().splitlines():
    r = json.loads(l)
    if r["status"] == "pass":
        x = t1[r["task_id"]]
        iv = inv.get((x["run"], r["task_id"])) or {}
        recs.append(("sftv3", r, x["task_dir"], iv.get("repo"), iv.get("target"), x["run"]))
res = []
for batch, r, td, repo, target, run in recs:
    if not is_old(td):
        continue
    try:
        lang = tp.repo_language(TOP100, target, repo)
        keys = tp.config_keys(lang)
    except Exception as e:
        lang, keys = None, None
    patch = Path(r["attempt_dir"].replace("/projects/nemotron_agents_dev/users/hengyuf/", "/users/hengyuf/")) / "model.patch"
    text = patch.read_bytes().decode("utf-8", "surrogateescape") if patch.exists() else None
    tpatch = (Path(td) / "tests/test.patch").read_bytes().decode("utf-8", "surrogateescape")
    hidden = {p for p, _, _ in touched(tpatch)}
    hits = []
    if text is not None and keys:
        for p, kind, additive in touched(text):
            if any(glob_match(g, p) for g in keys["test_paths"]):
                kept = keys["test_paths_mode"] == "non_additive" and additive and kind == "mod"
                hits.append({"path": p, "kind": kind, "additive_kept": kept, "in_test_patch": p in hidden})
    eff = [h for h in hits if not h["additive_kept"] and not h["in_test_patch"]]
    res.append({"batch": batch, "campaign": r["campaign"], "task_id": r["task_id"], "slot": r["slot"], "run": run,
                "repo": repo, "lang": lang, "patch_missing": text is None, "hits": hits, "effective_hits": len(eff)})
c = collections.Counter((x["batch"], x["lang"], x["effective_hits"] > 0) for x in res)
print("old-grader passes:", len(res)); [print(k, v) for k, v in sorted(c.items(), key=str)]
print("unknown language:", sum(1 for x in res if x["lang"] is None), "patch missing:", sum(x["patch_missing"] for x in res))
print("passes with effective hits:", sum(1 for x in res if x["effective_hits"]), "tasks:", len({(x["run"], x["task_id"]) for x in res if x["effective_hits"]}))
Path(__file__).with_name("touch_scan.json").write_text(json.dumps(res, indent=1) + "\n")
