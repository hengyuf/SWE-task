#!/usr/bin/env python3
"""Add test-path globs to the sanitizing grader's config of one round's candidates BEFORE that round's probe starts.

Why (2026-10-20, official1-anko round 1): gpython's TestVm/TestPy run every script in vm/tests/, py/tests/ and
stdlib/*/tests/ and grade them as P2P ids, but the Go defaults (**/*_test.go, **/testdata/**) do not cover those
directories and the evolve test.patches add only *_test.go files, so write_tests added no per-task extras. A model's
edit to an existing script (false PASS: round-0 c02 kimi-r3 rewrote vm/tests/call.py) or a failing script it adds
(false FAIL) would reach grading. Like tools/drop_p2p_gates.py, this only touches config.json in cand-*/final/tests/
and, once packaged, task/<id>/tests/, only for tests that carry the sanitizing grader (tests/grader.py ==
synth/verifier/grader.py), only while round-<N>/probe-* does not exist, and never when the task's solution.patch
touches a path the new globs match (that task is reported and left alone). Idempotent; backups <file>.pre-add-test-
paths; log <root>/test-paths-add.jsonl.

    /usr/bin/python3 -I -B tools/add_test_paths.py --run official1-anko --round 1 \
        --glob 'vm/tests/**' --glob 'py/tests/**' --glob 'stdlib/*/tests/**' [--apply]
"""
import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time

S = Path(__file__).resolve().parent.parent
RUNS = S / "synth/runs"
SANITIZER = S / "synth/verifier/grader.py"


def load_grader():
    spec = importlib.util.spec_from_file_location("sanitizing_grader", SANITIZER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--glob", action="append", required=True)
    parser.add_argument("--apply", action="store_true", help="write the changes (default: list them)")
    args = parser.parse_args()
    grader = load_grader()
    sanitizer_sha = sha(SANITIZER)
    roots = [path for path in (RUNS / args.run / "build").glob("*/*") if path.is_dir()]
    if len(roots) != 1:
        raise SystemExit(f"expected one build root under {RUNS / args.run / 'build'}, found {roots}")
    root = roots[0]
    if list((root / f"round-{args.round}").glob("probe-*")):
        print(json.dumps({"run": args.run, "round": args.round, "skipped": "the round's probe already exists"}))
        return 0
    suffix = re.compile(rf"-r{args.round}b\d+$") if args.round else re.compile(r"^cand-c\d+$")
    changes, refused, waiting = [], [], []
    for cand in sorted(root.glob("cand-*")):
        if not suffix.search(cand.name):
            continue
        dirs = [cand / "final/tests"]
        stage = json.loads((cand / "stage.json").read_text()) if (cand / "stage.json").is_file() else {}
        if stage.get("ok") and stage.get("task_dir"):
            dirs.append(Path(stage["task_dir"]).resolve() / "tests")  # /users is a symlink to /projects
        if not (dirs[0] / "config.json").is_file():
            waiting.append(cand.name)  # not gated yet: the loop comes back
            continue
        solution = cand / "final/solution.patch"
        if not solution.is_file() and len(dirs) > 1:
            solution = dirs[1].parent / "solution/solution.patch"
        touched = grader.touched_paths(solution.read_text()) if solution.is_file() else []
        clash = sorted({path for path in touched for glob in args.glob if grader.glob_match(glob, path)})
        if clash:
            refused.append({"cand": cand.name, "solution_paths": clash[:10]})
            continue
        for tests in dirs:
            path = tests / "config.json"
            if not path.is_file():
                continue
            if not (tests / "grader.py").is_file() or sha(tests / "grader.py") != sanitizer_sha:
                refused.append({"cand": cand.name, "file": str(path), "why": "not the sanitizing grader"})
                continue
            text = path.read_text()
            config = json.loads(text)
            current = list(config.get("test_paths") or [])
            add = [glob for glob in args.glob if glob not in current]
            if not add:
                continue
            changes.append({"cand": cand.name, "file": str(path.relative_to(root.resolve()) if path.is_relative_to(root.resolve()) else path), "added": add})
            if args.apply:
                backup = path.with_name(path.name + ".pre-add-test-paths")
                if not backup.exists():
                    backup.write_text(text)
                config["test_paths"] = current + add
                temporary = path.with_name("." + path.name + ".tmp")
                temporary.write_text(json.dumps(config, indent=2) + "\n")
                temporary.replace(path)
    if args.apply and changes:
        with (root / "test-paths-add.jsonl").open("a") as stream:
            for change in changes:
                stream.write(json.dumps(dict(change, at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                             tool="tools/add_test_paths.py")) + "\n")
    print(json.dumps({"run": args.run, "round": args.round, "applied": args.apply, "changes": changes,
                      "refused": refused, "not_gated_yet": waiting}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
