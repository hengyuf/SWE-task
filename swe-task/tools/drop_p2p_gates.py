#!/usr/bin/env python3
"""Drop lint-style "[gate] ..." entries from the P2P lists of one round's candidates BEFORE that round's probe starts.

Why (2026-10-19): author.md allows gates only for build/typecheck steps, but some round-1 authors in official1-dasel
also gated on `go vet ./...` ("[gate] vet"). vet is lint: it also checks the student's own test files, so a correct
solution can be graded FAIL by a vet finding. Like tools/strip_gate_timeouts.py, this only touches the P2P list of
config.json in cand-*/final/tests/ (package() copies it into the task dir) and, once packaged, task/<id>/tests/, and
only while round-<N>/probe-* does not exist (the probe copies the task dirs when it starts). The RUN TESTS block still
reports the gate; it is just no longer graded. Idempotent; backups <file>.pre-gate-drop; log <root>/gate-drop.jsonl.

    /usr/bin/python3 -I -B tools/drop_p2p_gates.py --run official1-dasel --round 1 --gate vet [--apply]
"""
import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path
import re
import time

RUNS = Path(__file__).resolve().parent.parent / "synth/runs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--gate", action="append", required=True, help="gate name after '[gate] ', e.g. vet")
    parser.add_argument("--apply", action="store_true", help="write the changes (default: list them)")
    args = parser.parse_args()
    roots = [path for path in (RUNS / args.run / "build").glob("*/*") if path.is_dir()]
    if len(roots) != 1:
        raise SystemExit(f"expected one build root under {RUNS / args.run / 'build'}, found {roots}")
    root = roots[0]
    if list((root / f"round-{args.round}").glob("probe-*")):
        print(json.dumps({"run": args.run, "round": args.round, "skipped": "the round's probe already exists"}))
        return 0
    wanted = re.compile(r"\[gate\] (?:%s)$" % "|".join(re.escape(gate) for gate in args.gate))
    suffix = re.compile(rf"-r{args.round}b\d+$") if args.round else re.compile(r"^cand-c\d+$")
    changes = []
    for cand in sorted(root.glob("cand-*")):
        if not suffix.search(cand.name):
            continue
        paths = [cand / "final/tests/config.json"]
        stage = json.loads((cand / "stage.json").read_text()) if (cand / "stage.json").is_file() else {}
        if stage.get("ok") and stage.get("task_dir"):
            paths.append(Path(stage["task_dir"]).resolve() / "tests/config.json")  # /users is a symlink to /projects
        for path in paths:
            if not path.is_file():
                continue
            text = path.read_text()
            config = json.loads(text)
            drop = [test for test in config.get("p2p_node_ids", []) if wanted.search(test)]
            if not drop:
                continue
            changes.append({"cand": cand.name, "file": str(path.relative_to(root)), "removed": drop})
            if args.apply:
                backup = path.with_name(path.name + ".pre-gate-drop")
                if not backup.exists():
                    backup.write_text(text)
                config["p2p_node_ids"] = [test for test in config["p2p_node_ids"] if test not in drop]
                temporary = path.with_name("." + path.name + ".tmp")
                temporary.write_text(json.dumps(config, indent=2) + "\n")
                temporary.replace(path)
    if args.apply and changes:
        with (root / "gate-drop.jsonl").open("a") as stream:
            for change in changes:
                stream.write(json.dumps(dict(change, at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                             tool="tools/drop_p2p_gates.py")) + "\n")
    print(json.dumps({"run": args.run, "round": args.round, "applied": args.apply, "changes": changes}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
