"""Smoke test of the new-build grader (hengyuf 2026-10-20, S/README.md "Grading gap": new builds write the sanitizing
grader). Read-only for runs/: everything it writes lives next to this file.

  scan                         every built candidate (cand-*/stage.json ok) of the FINISHED official1 builds
                               (summary.json; the 5 live builds excluded): reference solution.patch paths that the new
                               gate problem would name (task_builder.restored_paths with the build's Build.grading)
  extra                        the same candidates: the per-task test_paths extra (task_builder.test_patch_globs:
                               directories of test.patch files the language globs miss) and the probe slots whose
                               model.patch touches files that only the extra globs cover (they would now be restored)
  regen  <run> <cand>          rebuild <cand>'s tests/ with task_builder.write_tests into <run>/<task_id>/ (the rest of
                               the packaged task copied), compare with the packaged tests/, run Gym's task loader
                               (task_builder.validate_task)
  grade  <run> <cand> NAME...  grade the scratch task in sandboxes from task.toml's docker_image (sandbox.grade, as
                               tools/grading_repair.py validate does): solution (want 1, sanitizer saw 0 paths), empty
                               (want 0), tamper = solution + a broken model-added test file next to the hidden tests
                               (want 1, the file deleted), tamper-old = the same patch on the PACKAGED task (DeepSWE's
                               grader; the control: want 0)

Run from S/synth with the pipeline venv; --code DIR imports task_builder from DIR (a staged copy) instead of S/synth:
  ../atp/runtime/controller/bin/python3 -B repair/new-build-grader-smoke/smoke.py [--code DIR] scan
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import shutil  # noqa: E402

HERE = Path(__file__).resolve().parent
SYNTH = HERE.parents[1]
LIVE = {"official1-sqlformatter", "official1-dynamodb", "official1-wasmi", "official1-scriggo", "official1-anko"}


def setup(code: Path):
    sys.path.insert(0, str(code))
    import task_builder
    import sandbox
    if Path(task_builder.__file__).resolve().parent != code.resolve():
        raise SystemExit(f"task_builder came from {task_builder.__file__}, not {code}")
    return task_builder, sandbox


def build_of(tb, run: str):
    roots = sorted(p for p in (tb.RUNS / run / "build").glob("*/*") if p.is_dir())
    if len(roots) != 1:
        raise SystemExit(f"{run}: {len(roots)} build roots")
    root = roots[0]
    repo = (tb.read_json(root / "summary.json") or {}).get("repo") or root.name.replace("__", "/", 1)
    b = tb.Build(run, root.parent.name, repo)
    assert b.root.resolve() == root.resolve(), (b.root, root)
    return b


def finished_runs(tb) -> list:
    out = []
    for run_dir in sorted(tb.RUNS.glob("official1-*")):
        roots = [p for p in (run_dir / "build").glob("*/*") if p.is_dir()]
        if run_dir.name not in LIVE and len(roots) == 1 and (roots[0] / "summary.json").is_file():
            out.append(run_dir.name)
    return out


def cmd_scan(tb) -> dict:
    report = {}
    for run in finished_runs(tb):
        b = build_of(tb, run)
        built, hits = 0, []
        for stage_path in sorted(b.root.glob("cand-*/stage.json")):
            stage = tb.read_json(stage_path) or {}
            solution = Path(stage.get("task_dir", "/nonexistent")) / "solution/solution.patch"
            if not stage.get("ok") or not solution.exists():
                continue
            built += 1
            paths = tb.restored_paths(solution.read_text(errors="surrogateescape"), b.grading["test_paths"])
            if paths:
                hits.append({"cand": stage_path.parent.name, "task_id": stage["task_id"], "paths": paths})
        report[run] = {"repo": b.repo, "grading": b.grading, "built": built, "solution_hits": hits}
        print(f"{run:24s} {b.repo:40s} mode {b.grading['test_paths_mode']:12s} built {built:3d} "
              f"solutions touching test paths {len(hits)}" + "".join(f"\n    {h['cand']}: {h['paths']}" for h in hits))
    (HERE / "scan.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def cmd_extra(tb) -> dict:
    grader = tb._load(tb.SANITIZER, "sanitizing_grader")
    report = {}
    for run in finished_runs(tb):
        b = build_of(tb, run)
        defaults, rows = b.grading["test_paths"], []
        for stage_path in sorted(b.root.glob("cand-*/stage.json")):
            stage = tb.read_json(stage_path) or {}
            task = Path(stage.get("task_dir", "/nonexistent"))
            if not stage.get("ok") or not (task / "tests/test.patch").exists():
                continue
            extra, skipped = tb.test_patch_globs((task / "tests/test.patch").read_text(errors="surrogateescape"),
                                                 (task / "solution/solution.patch").read_text(errors="surrogateescape"),
                                                 defaults)
            slots = []
            for patch in sorted(b.root.glob(f"round-*/{stage_path.parent.name}/rollouts/*.patch")):
                hit = [p for p in grader.touched_paths(patch.read_text(errors="surrogateescape"))
                       if any(grader.glob_match(g, p) for g in extra)
                       and not any(grader.glob_match(g, p) for g in defaults)]
                if hit:
                    md = patch.with_suffix(".md")
                    status = md.read_text(errors="replace").split("\n", 1)[0].rsplit(" ", 1)[-1] if md.exists() else "?"
                    slots.append({"slot": patch.stem, "status": status, "paths": hit,
                                  "outside_test_dirs": [p for p in hit
                                                        if not tb.TEST_DIR_NAMES & set(p.split("/")[:-1])]})
            rows.append({"cand": stage_path.parent.name, "task_id": stage["task_id"], "extra": extra,
                         "skipped": skipped, "slots": slots})
        report[run] = rows
        with_extra = [r for r in rows if r["extra"]]
        print(f"{run:24s} built {len(rows):3d} with extra globs {len(with_extra):3d}; slots touching extra-only paths "
              f"{sum(len(r['slots']) for r in rows):3d} (pass {sum(s['status'] == 'pass' for r in rows for s in r['slots'])}"
              f", outside test dirs {sum(bool(s['outside_test_dirs']) for r in rows for s in r['slots'])})")
        globs = sorted({g for r in rows for g in r["extra"]})
        if globs:
            print(f"    extra globs: {globs[:12]}{' ...' if len(globs) > 12 else ''}")
        skipped = sorted({d for r in rows for d in r["skipped"]})
        if skipped:
            print(f"    skipped dirs: {skipped[:12]}{' ...' if len(skipped) > 12 else ''}")
    (HERE / "extra.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def candidate(tb, run: str, cand_name: str):
    if run in LIVE or run not in finished_runs(tb):
        raise SystemExit(f"{run} is not a finished build")
    b = build_of(tb, run)
    cand = b.root / cand_name
    stage, built = tb.read_json(cand / "stage.json") or {}, tb.read_json(cand / "built.json") or {}
    if not stage.get("ok") or not built:
        raise SystemExit(f"{cand} is not a built candidate")
    return b, cand, stage, built


def scratch_task(run: str, task_id: str) -> Path:
    return HERE / run / task_id


def cmd_regen(tb, run: str, cand_name: str) -> dict:
    b, cand, stage, built = candidate(tb, run, cand_name)
    author_dir, source = cand / f"author-{built['attempt']}", Path(stage["task_dir"])
    task = scratch_task(run, stage["task_id"])
    shutil.rmtree(task, ignore_errors=True)  # this script's own earlier output
    shutil.copytree(source, task, ignore=shutil.ignore_patterns("tests", "*.pre-*", "__pycache__"))
    tb.write_tests(task / "tests", b, (author_dir / "run_tests.sh").read_text(), (author_dir / "test.patch").read_text(),
                   built["mat"]["base_commit"], built["gate"]["f2p"], built["gate"]["p2p"],
                   solution=(author_dir / "solution.patch").read_text())
    config, packaged = (json.loads((d / "tests/config.json").read_text()) for d in (task, source))
    extra = {key: config.pop(key) for key in ("test_paths", "test_paths_mode")}
    same = {name: (task / "tests" / name).read_bytes() == (source / "tests" / name).read_bytes()
            for name in ("test.sh", "test.patch")}
    same["config.json without the new keys"] = config == packaged
    same["grader.py is verifier/grader.py"] = (task / "tests/grader.py").read_bytes() == tb.SANITIZER.read_bytes()
    same["packaged grader.py is DeepSWE's"] = (source / "tests/grader.py").read_bytes() == \
        (tb.DEEPSWE / b.target / "tests/grader.py").read_bytes()
    tb.validate_task(task)  # raises GateError when Gym's loader rejects the scratch task
    record = {"run": run, "cand": cand_name, "task_id": stage["task_id"], "source": str(source), "scratch": str(task),
              "config_keys": extra, "build_grading": b.grading, "checks": same, "gym_loader": "accepted",
              "restored_paths_of_solution": tb.restored_paths((task / "solution/solution.patch").read_text(),
                                                              extra["test_paths"])}
    (HERE / run / f"{stage['task_id']}.regen.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return record


def tamper_patch(test_patch: str, tb) -> tuple:
    """A model-added Go test file that does not compile, in the package of the first hidden test file."""
    first = next(p for p in tb.grader_module().patch_paths(test_patch) if p.endswith("_test.go"))
    directory = first.rsplit("/", 1)[0] if "/" in first else ""
    path = f"{directory}/zz_sanitizer_probe_test.go" if directory else "zz_sanitizer_probe_test.go"
    package = None
    for line in test_patch.splitlines():
        if line.startswith("+package ") or line.startswith(" package "):
            package = line[1:].split()[1]
            break
    body = [f"package {package or 'main'}", "", "this line is not Go: the sanitizer must delete this model-added file"]
    return path, (f"diff --git a/{path} b/{path}\nnew file mode 100644\nindex 0000000..1111111\n--- /dev/null\n"
                  f"+++ b/{path}\n@@ -0,0 +1,{len(body)} @@\n" + "".join(f"+{line}\n" for line in body))


async def cmd_grade(tb, sandbox, run: str, cand_name: str, names: list) -> dict:
    b, cand, stage, built = candidate(tb, run, cand_name)
    task, source = scratch_task(run, stage["task_id"]), Path(stage["task_dir"])
    if not (task / "tests/config.json").exists():
        raise SystemExit("run regen first")
    snapshot = tb.task_toml(task)["environment"]["docker_image"]
    solution = (task / "solution/solution.patch").read_text()
    probe_path, extra = tamper_patch((task / "tests/test.patch").read_text(), tb)
    tampered = solution + ("" if solution.endswith("\n") else "\n") + extra
    jobs = {"solution": (task, solution), "empty": (task, ""), "tamper": (task, tampered),
            "tamper-old": (source, tampered)}
    out = HERE / run / f"{stage['task_id']}.grades"

    async def one(name):
        task_dir, patch = jobs[name]
        shutil.rmtree(out / name, ignore_errors=True)  # this script's own earlier output
        try:
            return await sandbox.grade(snapshot, task_dir, patch, run=run, out_dir=out / name)
        except Exception as error:  # noqa: BLE001
            return {"error": f"{type(error).__name__}: {str(error)[-500:]}"}

    results = dict(zip(names, await asyncio.gather(*(one(name) for name in names))))
    summary = {}
    for name, result in results.items():
        log_text = (out / name / "output.log").read_text(errors="replace") if (out / name / "output.log").exists() else ""
        marks = [line.strip() for line in log_text.splitlines() if "test-sanitize:" in line and not line.startswith("+")]
        reward_json = result.get("reward_json") or {}
        summary[name] = {"reward": result.get("reward"), "error": result.get("error"),
                         "f2p": f"{reward_json.get('f2p_passed')}/{reward_json.get('f2p_total')}",
                         "p2p": f"{reward_json.get('p2p_passed')}/{reward_json.get('p2p_total')}",
                         "exit_code": result.get("exit_code"), "timed_out": result.get("timed_out"),
                         "sanitize_lines": marks[:8], "task_dir": str(jobs[name][0])}
    want = {"solution": lambda s: s["reward"] == 1 and any("0 path(s) of model.patch match" in m
                                                              for m in s["sanitize_lines"]),
            "empty": lambda s: s["reward"] == 0,
            "tamper": lambda s: s["reward"] == 1 and any(f'deleted "{probe_path}"' in m for m in s["sanitize_lines"]),
            "tamper-old": lambda s: s["reward"] == 0 and not s["sanitize_lines"]}
    checks = {name: want[name](summary[name]) for name in names}
    record = {"run": run, "cand": cand_name, "task_id": stage["task_id"], "snapshot": snapshot,
              "tamper_path": probe_path, "results": summary, "checks": checks, "all_ok": all(checks.values())}
    path = HERE / run / f"{stage['task_id']}.grade-{'-'.join(names)}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code", type=Path, default=SYNTH)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scan")
    sub.add_parser("extra")
    for name in ("regen", "grade"):
        command = sub.add_parser(name)
        command.add_argument("run")
        command.add_argument("cand")
        if name == "grade":
            command.add_argument("names", nargs="+", choices=["solution", "empty", "tamper", "tamper-old"])
    args = parser.parse_args()
    tb, sandbox = setup(args.code)
    if args.command == "scan":
        cmd_scan(tb)
    elif args.command == "extra":
        cmd_extra(tb)
    elif args.command == "regen":
        cmd_regen(tb, args.run, args.cand)
    else:
        if len(args.names) > 4:
            raise SystemExit("at most 4 sandboxes")
        asyncio.run(cmd_grade(tb, sandbox, args.run, args.cand, args.names))


if __name__ == "__main__":
    main()
