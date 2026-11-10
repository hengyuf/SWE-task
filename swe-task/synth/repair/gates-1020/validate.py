"""Sandbox validation of the 2026-10-20 gate fixes (S/README.md, "-gates-1020" bullets) with the installed code in
S/synth (or --code DIR, e.g. a staged copy). Read-only for runs/: snapshots, probe data and candidate dirs are only read;
everything written lives in repair/gates-1020/validate/. Sandboxes are tagged run=gates-1020-validate; at most 4 at a
time (one process at a time; SEM below).

  (each command takes RUN/CAND pairs, or RUN CAND...)
  oracles  RUN CAND...          stale_oracles' scan on each candidate's C snapshot (built.json): items, hits, problems
  released RUN CAND...          released_copies (per-file + per-package rules) on each candidate's C snapshot
  purge    RUN CAND...          PURGE_RELEASES with PURGE_DRY_RUN=1 on each candidate's C snapshot: what a rebuild would
                                delete (nothing is deleted)
  remat    RUN CAND [--grade]   the NEW MATERIALIZE on the build's image A + the candidate's start.patch -> a scratch C
                                snapshot; released_copies on it; --grade: the packaged final tests on it (empty, reference
                                solution); the scratch snapshot is deleted at the end
Run from S/synth with the pipeline venv:
  ../atp/runtime/controller/bin/python3 -B repair/gates-1020/validate.py [--code DIR] oracles official1-wasmi cand-c04
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "validate"
RUN_TAG = "gates-1020-validate"
SEM = asyncio.Semaphore(4)


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


def candidate(tb, b, cand: str) -> dict:
    """Paths and facts of a built candidate (or of an accepted task's candidate): author dir, C snapshot, globs."""
    cdir = b.root / cand
    built = tb.read_json(cdir / "built.json") or {}
    if not built:
        raise SystemExit(f"{cdir}: no built.json")
    author = cdir / f"author-{built['attempt']}"
    test_patch = (author / "test.patch").read_text()
    solution = (author / "solution.patch").read_text()
    globs = b.grading["test_paths"]
    extra, _ = tb.test_patch_globs(test_patch, solution, globs)
    return {"dir": cdir, "author": author, "snapshot": built["mat"]["snapshot"], "base_commit": built["mat"]["base_commit"],
            "test_patch": test_patch, "solution": solution, "start_patch": (author / "start.patch").read_text(),
            "globs": globs + extra, "stage": tb.read_json(cdir / "stage.json") or {}}


def save(name: str, data) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=sorted) + "\n")
    return path


async def oracles(tb, run: str, cand: str) -> dict:
    b = build_of(tb, run)
    c = candidate(tb, b, cand)
    items = tb.removed_source_items(c["start_patch"], c["globs"])
    result = {"run": run, "cand": cand, "snapshot": c["snapshot"], "items": items, "problems": [], "stale": {}}
    if items:
        async with SEM, tb.owned(c["snapshot"], run=RUN_TAG, purpose="oracles") as box:
            await box.put_text("".join(i + "\n" for i in sorted(items) if tb.IDENTIFIER.fullmatch(i)), "/tmp/oracle-idents")
            await box.put_text("".join(i + "\n" for i in sorted(items) if not tb.IDENTIFIER.fullmatch(i)),
                               "/tmp/oracle-literals")
            started = time.monotonic()
            out = await box.sh(tb.ORACLE_SCAN, timeout=900, check=False)
            result["seconds"] = round(time.monotonic() - started, 1)
        result["complete"] = out.stdout.rstrip().endswith(tb.ARCHIVE_DONE)
        result["hit_lines"] = len(out.stdout.splitlines())
        result["problems"], result["stale"] = tb.oracle_problems(out.stdout, items, c["globs"], c["test_patch"])
        hits = {}
        for line in out.stdout.splitlines():
            tag, _, rest = line.partition("\t")
            path, _, item = rest.partition("\t")
            if tag in ("i", "l"):
                hits.setdefault(item, []).append(path)
        result["item_files"] = {item: sorted(paths)[:8] + ([f"... {len(paths)} files"] if len(paths) > 8 else [])
                                for item, paths in sorted(hits.items())}
    save(f"oracles-{run}-{cand}.json", result)
    print(f"== oracles {run} {cand}: {len(items)} removed items; problems: {result['problems'] or 'none'}")
    return result


async def released(tb, run: str, cand: str) -> dict:
    b = build_of(tb, run)
    c = candidate(tb, b, cand)
    async with SEM, tb.owned(c["snapshot"], run=RUN_TAG, purpose="released") as box:
        report = await tb.released_copies(box, b.repo, c["solution"])
    keep = {k: report[k] for k in ("problems", "incomplete", "names", "seconds", "package_lines", "package_identifiers",
                                   "lines", "identifiers")}
    keep["selection"] = {k: v for k, v in report["selection"].items() if k != "unpack"}
    keep.update(run=run, cand=cand, snapshot=c["snapshot"], solution_tokens=tb.leak_tokens(c["solution"]))
    save(f"released-{run}-{cand}.json", keep)
    best = [(row["root"], f"{row['best']}/{row['total']} need {row['need']}") for row in report.get("package_lines", [])[:3]]
    ids = [(row["root"], f"{row['best']}/{row['total']} need {row['need']}") for row in
           report.get("package_identifiers", [])[:3]]
    print(f"== released {run} {cand}: problems {report['problems'] or 'none'}\n   package lines {best}\n"
          f"   package identifiers {ids}")
    return keep


async def purge(tb, run: str, cand: str) -> dict:
    b = build_of(tb, run)
    c = candidate(tb, b, cand)
    async with SEM, tb.owned(c["snapshot"], run=RUN_TAG, purpose="purge-dry") as box:
        out = await box.sh("export PURGE_DRY_RUN=1\n" + tb.PURGE_RELEASES + "\necho done", timeout=900, check=False)
    lines = (out.stdout + "\n" + out.stderr).splitlines()
    result = {"run": run, "cand": cand, "snapshot": c["snapshot"], "exit": out.exit_code,
              "would_purge": [line.split("released copy ", 1)[1] for line in lines if "would purge released copy" in line],
              "own": [line for line in lines if "own packages" in line]}
    save(f"purge-{run}-{cand}.json", result)
    print(f"== purge (dry) {run} {cand}: {result['own']} would purge {len(result['would_purge'])}: "
          f"{result['would_purge'][:6]}")
    return result


async def remat(tb, sandbox, run: str, cand: str, grade: bool) -> dict:
    b = build_of(tb, run)
    c = candidate(tb, b, cand)
    env = tb.read_json(b.root / "env/stage.json")
    work = OUT / f"remat-{run}-{cand}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    (work / "start.patch").write_text(c["start_patch"])
    result = {"run": run, "cand": cand, "image_a": env["snapshot"], "old_c": c["snapshot"]}
    async with SEM, tb.owned(env["snapshot"], run=RUN_TAG, purpose="remat") as box:
        await box.put(work / "start.patch", "/tmp/start.patch")
        await box.put_text(tb.MATERIALIZE, "/tmp/materialize.sh")
        out = await box.sh("bash /tmp/materialize.sh /tmp/start.patch", check=False, timeout=1800)
        result.update(exit=out.exit_code, base_commit=(out.stdout.strip().splitlines() or [""])[-1],
                      stderr=out.stderr[-6000:])
        if out.exit_code:
            save(f"remat-{run}-{cand}.json", result)
            raise SystemExit(f"materialize failed: {out.stderr[-2000:]}")
        snapshot = await box.snapshot(f"{RUN_TAG}-{cand}")
    result.update(new_c=snapshot, base_commit_same=result["base_commit"] == c["base_commit"])
    save(f"remat-{run}-{cand}.json", result)
    print(f"== remat {run} {cand}: exit {result['exit']} base commit same {result['base_commit_same']} -> {snapshot}")
    print("\n".join(line for line in result["stderr"].splitlines() if "materialize:" in line))
    try:
        async with SEM, tb.owned(snapshot, run=RUN_TAG, purpose="released") as box:
            report = await tb.released_copies(box, b.repo, c["solution"])
            check = await box.sh("ls -d /usr/local/share/.cache/yarn/v6/npm-@sqltools-* 2>/dev/null | head; "
                                 "ls -la /app/docs/node_modules/@sqltools 2>/dev/null | head", check=False)
        result.update(released_problems=report["problems"], released_incomplete=report["incomplete"],
                      package_lines=report.get("package_lines"), cache_listing=check.stdout)
        print(f"   released_copies on the new C: {report['problems'] or 'none'}; listing:\n{check.stdout}")
        if grade:
            task = work / "task"
            shutil.copytree(c["dir"] / "final", task)
            graded = {}
            for name, patch in (("empty", ""), ("solution", c["solution"])):
                async with SEM:
                    r = await sandbox.grade(snapshot, task, patch, run=RUN_TAG, out_dir=work / f"grade-{name}")
                graded[name] = {"reward": r["reward"], "reward_json": r["reward_json"], "exit": r["exit_code"]}
            result["grades"] = graded
            print(f"   grades on the new C: {graded}")
    finally:
        from sandbox_helpers import Client, Config
        async with Client(Config.load(tb.ATP_ENV), timeout=120) as client:
            await client.delete_snapshot(snapshot.removeprefix("snapshot:"))
        result["snapshot_deleted"] = True
        save(f"remat-{run}-{cand}.json", result)
    return result


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code", type=Path, default=HERE.parents[1])
    parser.add_argument("command", choices=["oracles", "released", "purge", "remat"])
    parser.add_argument("items", nargs="+", help="RUN/CAND (or RUN CAND...)")
    parser.add_argument("--grade", action="store_true")
    args = parser.parse_args()
    tb, sandbox = setup(args.code)
    pairs = [tuple(item.split("/", 1)) for item in args.items if "/" in item]
    if not pairs:
        pairs = [(args.items[0], cand) for cand in args.items[1:]]
    if args.command == "remat":
        await remat(tb, sandbox, *pairs[0], args.grade)
        return
    func = {"oracles": oracles, "released": released, "purge": purge}[args.command]
    results = await asyncio.gather(*(func(tb, run, cand) for run, cand in pairs), return_exceptions=True)
    for (run, cand), result in zip(pairs, results):
        if isinstance(result, BaseException):
            print(f"!! {args.command} {run} {cand}: {type(result).__name__}: {str(result)[-500:]}")


if __name__ == "__main__":
    asyncio.run(main())
