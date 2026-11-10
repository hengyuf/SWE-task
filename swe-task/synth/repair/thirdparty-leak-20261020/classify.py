"""Read-only re-run of the final identifier rule (hengyuf 2026-10-20: identifier evidence is what the solution
DECLARES, plus its string literals; no package-based dropping of line or identifier hits) on the leak-scan problems
recorded in runs/official1-*/build/*/*/cand-*/gates-*.json: every problem is classified kept or dropped, every gate
report re-judged without the dropped ones.

A recorded LINE problem ("distinctive lines of", per file or per package) is kept: line evidence is unchanged and
nothing is dropped by package any more. A recorded IDENTIFIER problem is re-judged on the solution.patch that its gate
judged (author-<n>/solution.patch for gates-<n>.json): solution_identifiers of --code gives the source file's evidence
(declared names and string literals). The record does not list the matched items, so: with --copy SHOWN=LOCAL (the
released file available here, e.g. the real wasm-encoder sources) ARCHIVE_SCAN's greps are re-run on that file
(identifiers -w, literals fixed) and identifier_problems decides, assuming the fewest absent items (every hit absent,
nothing else: need 3), so a kept verdict is never lost to the starting state; without a copy it is dropped only when
the file has fewer than 3 evidence items (no window can reach the floor of 3), else "undetermined".
  ../../../atp/runtime/controller/bin/python3 -B classify.py --code <synth dir> [--copy SHOWN=LOCAL ...] [--out F]
Writes nothing but --out."""
import argparse
import glob
import json
import subprocess
import sys
import tempfile
from pathlib import Path

RUNS = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth/runs")


def solution_patch(cand: Path, gate: str):
    """The solution.patch that gates-<n>.json judged (author-<n>, also a renamed author-<n>.<suffix>), or None."""
    attempt = gate.removeprefix("gates-").removesuffix(".json")
    patches = sorted(glob.glob(str(cand / f"author-{attempt}/solution.patch"))) or sorted(
        glob.glob(str(cand / f"author-{attempt}.*/solution.patch")))
    return Path(patches[0]) if patches else None


def local_copy(path: str, copies: dict):
    for shown, local in copies.items():
        if path.startswith(shown):
            candidate = Path(local) / path[len(shown):]
            return candidate if candidate.is_file() else None
    return None


def rescan(tb, path: str, local: Path, items: list, source: str, solution: str) -> tuple:
    """identifier_problems on ARCHIVE_SCAN's greps of one local file for `source`'s evidence items."""
    idents = [item for item in items if tb.IDENTIFIER.fullmatch(item)]
    literals = [item for item in items if not tb.IDENTIFIER.fullmatch(item)]
    hits = []
    with tempfile.TemporaryDirectory() as scratch:
        for flags, patterns in (("-IonbwF", idents), ("-IonbF", literals)):
            if patterns:
                Path(scratch, "p").write_text("".join(item + "\n" for item in patterns))
                output = subprocess.run(["grep", flags, "-f", f"{scratch}/p", "--", str(local)], capture_output=True,
                                        text=True, errors="replace").stdout
                for line in output.splitlines():
                    number, offset, text = line.split(":", 2)
                    found = [text] if flags == "-IonbwF" else [item for item in literals if item in text]
                    hits += [(int(number), item, int(offset) + text.index(item)) for item in found]
    absent = {source: sorted({item for _, item, _ in hits})}  # the fewest absent items: need stays at 3
    problems = tb.identifier_problems({path: hits}, absent, tb.change_windows(solution))
    return problems, sorted({item for _, item, _ in hits})


def explain(tb, problem: str, patch, copies: dict) -> dict:
    if " distinctive lines of " in problem:
        return {"verdict": "kept", "reason": "line problem: line evidence is unchanged, no package-based dropping"}
    path, rest = problem.split(" holds ", 1)
    source = rest.split(" distinctive identifiers of ", 1)[1].rsplit(" within ", 1)[0]
    if patch is None:
        return {"verdict": "undetermined", "reason": "no recorded solution.patch"}
    solution = patch.read_text()
    evidence = tb.solution_identifiers(solution)
    if source == "the solution's code files" or source.startswith("the solution's code files"):
        items = sorted({item for values in evidence.values() for item in values})
    else:
        items = evidence.get(source, [])
    referenced = tb.referenced_identifiers(solution).get(source, [])
    base = {"evidence": items, "referenced": referenced}
    local = local_copy(path, copies)
    if local:
        problems, found = rescan(tb, path, local, items, source, solution)
        verdict = "kept" if problems else "dropped"
        return dict(base, verdict=verdict, found=found, reason=(
            f"re-scanned {local}: {len(found)} of {source}'s {len(items)} evidence items in it ({found}); "
            + ("still a problem" if problems else "below the floor of 3 in every window")))
    if len(items) < 3:
        return dict(base, verdict="dropped", reason=f"{source} has {len(items)} evidence items (< 3): no window "
                                                    "can reach the floor")
    return dict(base, verdict="undetermined", reason=f"{len(items)} evidence items; the released file is not here")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True, help="synth directory whose task_builder to use")
    parser.add_argument("--copy", action="append", default=[], help="SHOWN=LOCAL: a released file tree available here")
    parser.add_argument("--out")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, args.code)
    import task_builder as tb
    copies = dict(item.split("=", 1) for item in args.copy)
    problems, gates = [], []
    for path in sorted(glob.glob(str(RUNS / "official1-*/build/*/*/cand-*/gates-*.json"))):
        run, _, target, repo_dir, cand, name = Path(path).relative_to(RUNS).parts
        recorded = json.loads(Path(path).read_text()).get("problems", [])
        held = [problem for problem in recorded if " holds " in problem and (
            " distinctive lines of " in problem or " distinctive identifiers of " in problem)]
        if not held:
            continue
        patch = solution_patch(Path(path).parent, name)
        dependency = set(tb.dependency_leaks(recorded))
        verdicts = {problem: explain(tb, problem, patch, copies) for problem in held}
        for problem in held:
            problems.append({"run": run, "candidate": cand, "gate": name, "problem": problem,
                             "kind": "lines" if " distinctive lines of " in problem else "identifiers",
                             "dependency_leak": problem in dependency, "in_app": problem.startswith("/app/"),
                             "solution_patch": str(patch.relative_to(RUNS)) if patch else None, **verdicts[problem]})
        remaining = [problem for problem in recorded if verdicts.get(problem, {}).get("verdict") != "dropped"]
        stage = Path(path).parent / "stage.json"
        gates.append({"run": run, "candidate": cand, "gate": name, "problems": len(recorded),
                      "dependency_leaks": len(dependency), "dropped": len(recorded) - len(remaining),
                      "passes_now": not remaining,
                      "stage_ok": json.loads(stage.read_text()).get("ok") if stage.exists() else None})
    for gate in gates:
        if gate["dependency_leaks"] or gate["dropped"]:
            print(f"{gate['run']} {gate['candidate']} {gate['gate']}: {gate['problems']} problems, "
                  f"{gate['dependency_leaks']} dependency leaks, {gate['dropped']} dropped -> "
                  f"{'PASSES under the final rule' if gate['passes_now'] else 'still fails'} (stage ok={gate['stage_ok']})")
    for entry in problems:
        if entry["dependency_leak"] or entry["verdict"] != "kept":
            print(f"  [{entry['verdict']}] {entry['run']} {entry['candidate']} {entry['gate']}: {entry['reason'][:200]}"
                  f" :: {entry['problem'][:150]}")
    for label, group in (("dependency_leaks", [e for e in problems if e["dependency_leak"]]),
                         ("other (not dependency leaks)", [e for e in problems if not e["dependency_leak"]])):
        counts = {verdict: sum(e["verdict"] == verdict for e in group) for verdict in ("kept", "dropped", "undetermined")}
        kinds = {kind: sum(e["kind"] == kind for e in group) for kind in ("lines", "identifiers")}
        print(f"{label} problems: {len(group)} {kinds}, {counts}; in /app: {sum(e['in_app'] for e in group)}")
    if args.out:
        Path(args.out).write_text(json.dumps({"problems": problems, "gates": gates}, indent=1) + "\n")


if __name__ == "__main__":
    main()
