#!/usr/bin/env python3
"""Remove short shell `timeout N` wrappers in front of Go build/test commands from BUILT tasks of one round.

Why (2026-10-18): once build/typecheck steps became [gate] pass-to-pass tests, some authors wrapped cold Go compiles
(MATERIALIZE purges the build cache; verifiers have 2 CPUs) and suites in short `timeout 300s go build ...` commands,
so under probe load a correct solution can fail the gate. Without the wrapper an extreme slowdown hits the verifier's
overall limit (task.toml verifier.timeout_sec) and becomes an ungraded slot, never a false FAIL; `go test -timeout`
flags stay. Only candidates whose stage.json is ok (packaged) are touched, and only while the round's probe has not
been created yet (the probe copies the task dirs when it starts). Idempotent; backups <file>.pre-timeout-fix; log
<build root>/timeout-fix.jsonl.

    /usr/bin/python3 -I -B tools/strip_gate_timeouts.py --run official1-promlabels --round 1 [--apply]
"""
import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path
import re
import time

RUNS = Path(__file__).resolve().parent.parent / "synth/runs"
WRAP = re.compile(r"\btimeout\s+\d+s?\s+(?=go\s+(?:build|test|vet|run)\b)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--apply", action="store_true", help="write the changes (default: list them)")
    args = parser.parse_args()
    roots = [path for path in (RUNS / args.run / "build").glob("*/*") if path.is_dir()]
    if len(roots) != 1:
        raise SystemExit(f"expected one build root under {RUNS / args.run / 'build'}, found {roots}")
    root = roots[0]
    if list((root / f"round-{args.round}").glob("probe-*")):
        print(json.dumps({"run": args.run, "round": args.round, "skipped": "the round's probe already exists"}))
        return 0
    suffix = re.compile(rf"-r{args.round}b\d+$") if args.round else re.compile(r"^cand-c\d+$")
    changes = []
    for cand in sorted(root.glob("cand-*")):
        if not suffix.search(cand.name) or not (cand / "stage.json").is_file():
            continue
        stage = json.loads((cand / "stage.json").read_text())
        built = json.loads((cand / "built.json").read_text()) if (cand / "built.json").is_file() else None
        if not stage.get("ok") or not built:
            continue
        # task_dir is recorded under /users/..., a symlink to the resolved /projects/... root: resolve before relative_to
        for path in (cand / "final/tests/test.sh", Path(stage["task_dir"]).resolve() / "tests/test.sh",
                     cand / f"author-{built['attempt']}/run_tests.sh"):
            text = path.read_text() if path.is_file() else ""
            count = len(WRAP.findall(text))
            if not count:
                continue
            changes.append({"cand": cand.name, "file": str(path.relative_to(root)), "removed": count})
            if args.apply:
                backup = path.with_name(path.name + ".pre-timeout-fix")
                if not backup.exists():
                    backup.write_text(text)
                path.write_text(WRAP.sub("", text))
    if args.apply and changes:
        with (root / "timeout-fix.jsonl").open("a") as stream:
            for change in changes:
                stream.write(json.dumps(dict(change, at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                             tool="tools/strip_gate_timeouts.py")) + "\n")
    print(json.dumps({"run": args.run, "round": args.round, "applied": args.apply, "changes": changes}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
