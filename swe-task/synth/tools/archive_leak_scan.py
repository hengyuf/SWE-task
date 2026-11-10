#!/usr/bin/env python3
"""Exposure audit: released copies of a task's removed code in archives and package caches on its starting state.

official1-testem c03 (2026-10-18): kimi found jest-worker 27.5.1/29.7.0 in yarn's global cache (/root/.yarn/berry/cache),
unzipped 29.7.0 and read the removed implementation; the gates' leak scan never reads archives (grep -I) and scans /root
and other cache roots only where a directory name contains the repo name. This tool runs the gates' own check
(task_builder.released_copies) on packaged tasks and reports what it found.

  scan    [--run R ...] [--task-dir DIR ...] [--snapshot S] [--parallel N] [--force] [--dry-run]
          every accepted task of the runs (default: runs/official1-*/accepted/tasks.jsonl), or the packaged task dirs
          given (accepted/<id>, withdrawn/<id>, build/.../cand-*/task/<id>). One no-network sandbox per task from its
          snapshot (provenance.json "snapshot", else the candidate's stage.json, else task.toml docker_image;
          --snapshot overrides for a single task), at most 4 at a time, always deleted. Tasks with a report are
          skipped (--force rescans). --dry-run prints the solution's distinctive lines and identifiers, no sandbox.
  review  --report PATH --judgment exposed|clean|unclear --note TEXT
          record a manual reading of the released file(s) in the report ("manual_review")
  summary [--run R ...]   per run: released copies of the repo's own packages, and every task's verdict

Report: runs/<run>/audit/archive_leaks/<task_id>.json with archives_found, token_problems (leak_tokens/leak_problems,
signal i), identifier_signal (identifiers and string literals the solution adds and the starting state lacks, signal ii),
verdict (exposed: a signal fired; unclear: partial evidence or an incomplete scan; clean) and evidence. Run from S/synth
with the pipeline venv:  ../atp/runtime/controller/bin/python3 -B tools/archive_leak_scan.py scan --run official1-testem
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import asyncio  # noqa: E402
import base64  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import shlex  # noqa: E402
import time  # noqa: E402

SYNTH = Path(__file__).resolve().parents[1]
if str(SYNTH) not in sys.path:
    sys.path.insert(0, str(SYNTH))

from common import RUNS, read_json, task_toml, write_json  # noqa: E402
import sandbox  # noqa: E402
import task_builder  # noqa: E402
from task_builder import LEAK_ARCHIVES, _norm  # noqa: E402

AUDIT = "audit/archive_leaks"
CLEAN = r"LC_ALL=C sed -e 's/\r//g' -e 's/\xc2\x85//g' -e 's/\xe2\x80[\xa8\xa9]//g' -e 's/[\x0b\x0c\x1c-\x1e]//g'"
MAX_PARALLEL = 4  # five builds share the 140-slot sandbox pool
LOCKFILES = r"""set -u
cd /app || exit 3
git ls-files -- yarn.lock '*/yarn.lock' pnpm-lock.yaml '*/pnpm-lock.yaml' package-lock.json '*/package-lock.json' \
    uv.lock '*/uv.lock' poetry.lock '*/poetry.lock' go.sum '*/go.sum' bun.lock '*/bun.lock' 2>/dev/null | head -3000 \
  | while IFS= read -r f; do
  case "$f" in
    *yarn.lock) awk -v f="$f" '
        /^[^ #].*:$/ { d = $0; sub(/^"/, "", d); sub(/[",].*$/, "", d); sub(/:$/, "", d); next }
        /^  resolution: / { r = $0; sub(/^  resolution: "/, "", r); sub(/"$/, "", r); print "L\t" f "\t" r; next }
        /^  version "/ { v = $0; sub(/^  version "/, "", v); sub(/"$/, "", v); n = d; sub(/@[^@]*$/, "", n); print "L\t" f "\t" n "@" v }' "$f" ;;
    *pnpm-lock.yaml) grep -E "^  '?/?@?[^ ':@]+(/[^ ':@]+)?@[0-9][^:' ]*'?:" "$f" | sed -E "s#^  '?/?##; s#'?:.*##" \
        | awk -v f="$f" '{print "L\t" f "\t" $0}' ;;
    *package-lock.json) awk -v f="$f" '
        /^    "(.*\/)?node_modules\/[^"]+": \{/ { n = $0; sub(/^    "(.*\/)?node_modules\//, "", n); sub(/": \{.*/, "", n); next }
        n != "" && /^      "version": "/ { v = $0; sub(/^      "version": "/, "", v); sub(/".*/, "", v); print "L\t" f "\t" n "@" v; n = "" }' "$f" ;;
    *uv.lock|*poetry.lock) awk -v f="$f" '
        function flush() { if (n != "") print "L\t" f "\t" n "@" v "\t" s; n = ""; v = ""; s = "" }
        /^\[\[package\]\]/ { flush() } /^name = "/ { n = $0; sub(/^name = "/, "", n); sub(/".*/, "", n) }
        /^version = "/ { v = $0; sub(/^version = "/, "", v); sub(/".*/, "", v) } /^source = / { s = $0 } END { flush() }' "$f" ;;
    *go.sum) awk -v f="$f" '$2 !~ /\/go\.mod$/ { print "L\t" f "\t" $1 "@" $2 }' "$f" | sort -u ;;
    *bun.lock) grep -oE '"(@[^"/]+/)?[^"@/]+@[0-9][^"]*"' "$f" | tr -d '"' | sort -u | awk -v f="$f" '{print "L\t" f "\t" $0}' ;;
  esac
done | LC_ALL=C sed -e 's/\r//g' -e 's/\xc2\x85//g' -e 's/\xe2\x80[\xa8\xa9]//g'
printf 'E\tdone\n'
"""


def solution_files(solution: str) -> list:
    return re.findall(r"^\+\+\+ b/(\S+)", solution, re.M)


def touched_packages(files: list, released: dict) -> list:
    """The repo's own packages that the solution's files belong to (workspace dir or package name, Python top-level
    package, the repo itself), as normalized names."""
    touched = set()
    for path in files:
        homes = [directory for directory in released["workspace_dirs"] if path.startswith(directory + "/")]
        if homes:
            home = max(homes, key=len)
            touched |= {_norm(released["workspace_dirs"][home]), _norm(home.rsplit("/", 1)[-1])}
        top = path.split("/", 1)[0]
        if _norm(top) in {_norm(name) for name in released["names"]}:
            touched.add(_norm(top))
        modules = [directory for directory in released.get("go_module_dirs", {})
                   if directory == "" or path.startswith(directory + "/")]
        if modules:
            module = released["go_module_dirs"][max(modules, key=len)]
            touched.add(_norm(re.sub(r"/v\d+$", "", module).rsplit("/", 1)[-1]))
    if not touched:
        touched |= {_norm(name) for name in released["names"]}
    return sorted(touched)


def identity(entry: dict) -> tuple:
    """(package, version) of an archive or cache entry, best effort."""
    path, key = entry["path"], entry.get("key", "")
    if key:
        package = key.split("/-/")[0].split("//", 1)[-1].split("/", 1)[-1]
        version = re.search(r"-(\d[^/]*?)\.tgz$", key)
        return package, version.group(1) if version else ""
    base = path.rsplit("/", 1)[-1]
    if match := re.match(r"^(@[^-]+-)?(.+?)-npm-(\d[^-]*(?:-[a-z][^-]*)?)-[0-9a-f]{10}-\d+\.zip$", base):
        scope = match.group(1)
        return (f"{scope[:-1]}/{match.group(2)}" if scope else match.group(2)), match.group(3)
    if match := re.search(r"/cache/download/(.+)/@v/(v[^/]+)\.zip$", path):
        return match.group(1), match.group(2)
    if entry.get("top_level"):
        info = [top for top in entry["top_level"].split(",") if top.endswith((".dist-info", ".egg-info"))]
        if info and (match := re.match(r"^(.+?)-(\d[^-]*)\.(dist|egg)-info$", info[0])):
            return match.group(1), match.group(2)
        return entry["top_level"].split(",")[0], ""
    if match := re.match(r"^(.+?)-(\d[\w.+!-]*?)(-py\d.*|-cp\d.*)?\.(whl|zip|tar\.gz|tgz|crate|jar|egg)$", base):
        return match.group(1), match.group(2)
    return base, ""


def lockfile_releases(output: str, released: dict) -> list:
    """Lockfile entries that resolve one of the repo's own packages (workspace name or directory name, Python project
    or import name) to a published version: evidence that a release is cached."""
    own = {_norm(name) for name in released["names"]}
    found = set()
    for fields in task_builder.parse_tagged(output).get("L", []):
        lockfile, entry = fields[0], fields[1] if len(fields) > 1 else ""
        source = fields[2] if len(fields) > 2 else ""
        if "@workspace:" in entry or "@link:" in entry or "@file:" in entry or "editable" in source or "virtual" in source:
            continue
        match = re.match(r"^(@?[^@]+)@(?:npm:)?([^@:(]+)", entry.replace("@patch:", "@").replace("%3A", ":"))
        if match and _norm(match.group(1)) in own and re.match(r"v?\d", match.group(2)):
            found.add((match.group(1), match.group(2), lockfile))
    return [{"name": name, "version": version, "lockfile": lockfile} for name, version, lockfile in sorted(found)]


def local_path(display: str, scan: dict) -> str:
    """The sandbox path of a reported file ("<archive>!/<member>" -> LEAK_ARCHIVES/<n>/<member>)."""
    if "!/" not in display:
        return display
    source, member = display.split("!/", 1)
    for number, (unpacked, _status) in scan.get("unpacked", {}).items():
        if unpacked == source:
            return f"{LEAK_ARCHIVES}/{number}/{member}"
    return display


async def excerpts(box, report: dict, limit: int = 4) -> list:
    """The windows of the strongest identifier and line matches, read from the sandbox before it is deleted."""
    rows = sorted(report["identifiers"][:limit] + report["lines"][:limit],
                  key=lambda row: -row["best"] / max(1, row["need"]))
    out, seen = [], set()
    for row in rows:
        key = (row["path"], tuple(row["lines"]))
        if key in seen or row["best"] < 2 or len(out) >= limit:
            continue
        seen.add(key)
        first, last = max(1, row["lines"][0] - 15), row["lines"][1] + 15
        last = min(last, first + 299)
        path = shlex.quote(local_path(row["path"], report["scan"]))
        result = await box.sh(f"sed -n '{first},{last}p' {path} | cut -c1-400 | {CLEAN}", cwd="/", timeout=120,
                              check=False)
        entry = {"path": row["path"], "source": row["source"], "lines": [first, last], "text": result.stdout}
        if store := re.match(r"^(.*/store/v\d+)/files/([0-9a-f]{2})/([0-9a-f]+)", row["path"]):
            entry["pnpm_owner"] = await pnpm_owner(box, store.group(1), store.group(2) + store.group(3))
        out.append(entry)
    return out


async def pnpm_owner(box, store: str, hexdigest: str) -> list:
    """The pnpm store index files (named <hash>-<package>@<version>.json in store v10) listing a content file."""
    integrity = "sha512-" + base64.b64encode(bytes.fromhex(hexdigest)).decode()
    result = await box.sh(f"grep -rlF {shlex.quote(integrity)} {shlex.quote(store)}/index 2>/dev/null | head -5",
                          cwd="/", timeout=300, check=False)
    return result.stdout.split()


def verdict(report: dict) -> tuple:
    """("exposed" | "unclear" | "clean", reasons)."""
    problems = [problem for problem in report["problems"] if not problem.startswith("archive leak scan incomplete")]
    if problems:
        return "exposed", problems
    reasons = [f"scan incomplete: {note}" for note in report["incomplete"]]
    failed = [source for source, status in report.get("scan", {}).get("unpacked", {}).values() if status != "0"]
    reasons += [f"could not unpack {source}" for source in failed[:10]]
    reasons += [f"partial identifier match: {row['path']} holds {row['best']}/{row['total']} of {row['source']} "
                f"(need {row['need']})" for row in report["identifiers"] if 3 <= row["best"] < row["need"]][:5]
    reasons += [f"partial line match: {row['path']} holds {row['best']}/{row['total']} lines of {row['source']} "
                f"(need {row['need']})" for row in report["lines"] if 2 <= row["best"] < row["need"]][:5]
    reasons += [f"documentation copy: {row['path']} holds {row['best']}/{row['total']} lines of {row['source']}"
                for row in report["doc_lines"]][:5]
    return ("unclear" if reasons else "clean"), reasons


def evidence(record: dict) -> str:
    own = [entry for entry in record["archives_found"] if entry["own_package"]]
    touched = [entry for entry in own if entry["touched"]]
    parts = [f"{len(record['archives_found'])} archive/cache entries name the repo's packages ({len(own)} are releases "
             f"of its own packages); {len(touched)} are releases of the package(s) the solution touches "
             f"({', '.join(record['touched_packages'][:6])})"]
    if touched:
        parts[-1] += ": " + ", ".join(sorted({f"{e['package']}@{e['version']} ({e['kind']})" for e in touched})[:8])
    parts.append(f"examined on C: {record['examined']}; {len(record['scanned_roots'])} cache roots / package "
                 f"directories scanned as plain files")
    signal = record["identifier_signal"]
    distinct = {source: len(values) for source, values in signal["distinctive"].items()}
    parts.append(f"distinctive identifiers absent from the starting state per solution code file: {distinct or 'none'}")
    if signal["files"]:
        top = signal["files"][0]
        parts.append(f"strongest identifier match: {top['path']} holds {top['best']}/{top['total']} of {top['source']} "
                     f"(need {top['need']}): {', '.join(top['items'][:8])}")
    else:
        parts.append("no released file holds any distinctive identifier")
    if record["token_best"]:
        top = record["token_best"][0]
        parts.append(f"strongest line match: {top['path']} holds {top['best']}/{top['total']} distinctive lines of "
                     f"{top['source']} (need {top['need']})")
    else:
        parts.append("no released file holds any distinctive line")
    parts.append(f"verdict {record['verdict']}" + (f": {record['reasons'][0]}" if record["reasons"] else ""))
    return "; ".join(parts) + "."


def build_record(task_dir: Path, run: str, snapshot: str, repo: str, solution: str, report: dict, locks: str,
                 shown: list, seconds: float) -> dict:
    released = report["names"] or {"names": [], "go_modules": [], "workspace_dirs": {}}
    touched = touched_packages(solution_files(solution), released) if released["names"] else []
    own = {_norm(name) for name in released["names"]} | {_norm(name.split("/")[-1]) for name in released["go_modules"]}
    status = {source: code for source, code in report.get("scan", {}).get("unpacked", {}).values()}
    archives = []
    for entry in report.get("selection", {}).get("archives", []):
        package, version = identity(entry)
        flat = _norm(package)
        archives.append(dict(entry, package=package, version=version, unpacked=status.get(entry["path"]),
                             own_package=flat in own or _norm(package.rsplit("/", 1)[-1]) in own and "/cache/download/"
                             in entry["path"],
                             touched=flat in touched or _norm(package.replace("/", "-").lstrip("@")) in touched))
    present = {source: [item for item in values if item in report.get("scan", {}).get("present", set())]
               for source, values in task_builder.solution_identifiers(solution).items()}
    kind, reasons = verdict(report)
    record = {
        "task_id": task_dir.name, "run": run, "task_dir": str(task_dir), "snapshot": snapshot, "repo": repo,
        "scanned_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "seconds": round(seconds),
        "package_names": released["names"], "go_modules": released["go_modules"], "touched_packages": touched,
        "archives_found": archives, "lockfile_releases": lockfile_releases(locks, released) if released["names"] else [],
        "cache_roots": report.get("selection", {}).get("cache_roots", []),
        "package_dirs": report.get("selection", {}).get("package_dirs", []),
        "scanned_roots": report.get("selection", {}).get("roots", []),
        "tools": report.get("selection", {}).get("tools", []),
        "examined": report.get("selection", {}).get("examined", {}),
        "token_problems": [p for p in report["problems"] if " distinctive lines of " in p],
        "token_best": report["lines"][:10], "doc_line_hits": report["doc_lines"][:10],
        "identifier_signal": {
            "distinctive": report.get("absent", {}), "present_in_start": present,
            "problems": [p for p in report["problems"] if " distinctive identifiers of " in p],
            "files": report["identifiers"][:15]},
        "incomplete": report["incomplete"], "step_seconds": report.get("seconds", {}), "verdict": kind,
        "reasons": reasons, "excerpts": shown}
    record["evidence"] = evidence(record)
    return record


def report_path(task_dir: Path, run: str) -> Path:
    return RUNS / run / AUDIT / f"{task_dir.name}.json"


def run_of(task_dir: Path) -> str:
    parts = task_dir.resolve().parts
    return parts[parts.index("runs") + 1] if "runs" in parts else "adhoc"


def snapshot_of(task_dir: Path) -> str:
    provenance = read_json(task_dir / "provenance.json") or {}
    if provenance.get("snapshot"):
        return provenance["snapshot"]
    for parent in task_dir.parents:  # build/.../cand-X/task/<id>: cand-X/stage.json or built.json
        stage = read_json(parent / "stage.json") or {}
        if stage.get("snapshot"):
            return stage["snapshot"]
        built = read_json(parent / "built.json") or {}
        if built.get("mat", {}).get("snapshot"):
            return built["mat"]["snapshot"]
        if parent.name.startswith("cand-"):
            break
    return task_toml(task_dir)["environment"]["docker_image"]


async def scan_task(task_dir: Path, snapshot: str, gate: asyncio.Semaphore, force: bool) -> dict | None:
    run = run_of(task_dir)
    out = report_path(task_dir, run)
    if out.exists() and not force:
        print(f"skip {task_dir.name}: {out} exists", flush=True)
        return read_json(out)
    solution = (task_dir / "solution/solution.patch").read_text()
    repo = task_toml(task_dir)["metadata"]["repository_url"].rstrip("/").removeprefix("https://github.com/")
    async with gate:
        started = time.time()
        try:
            async with sandbox.owned(snapshot, run=run, purpose="archive-leak-scan") as box:
                report = await task_builder.released_copies(box, repo, solution)
                locks = (await box.sh(LOCKFILES, cwd="/app", timeout=600, check=False)).stdout
                shown = await excerpts(box, report)
        except Exception as error:  # noqa: BLE001 -- recorded, the next scan retries the task
            print(f"ERROR {task_dir.name}: {type(error).__name__}: {str(error)[-400:]}", flush=True)
            write_json(out.with_suffix(".error.json"), {"task_id": task_dir.name, "error": f"{type(error).__name__}: "
                                                        f"{str(error)[-2000:]}", "snapshot": snapshot})
            return None
        seconds = time.time() - started
    record = build_record(task_dir, run, snapshot, repo, solution, report, locks, shown, seconds)
    write_json(out, record)
    out.with_suffix(".error.json").unlink(missing_ok=True)
    print(f"{record['verdict']:8s} {task_dir.name} ({record['seconds']} s): {record['evidence'][:300]}", flush=True)
    return record


def accepted_tasks(runs: list) -> list:
    tasks = []
    for run in runs:
        rows = RUNS / run / "accepted/tasks.jsonl"
        for line in rows.read_text().splitlines() if rows.exists() else []:
            if line.strip():
                tasks.append(RUNS / run / "accepted" / json.loads(line)["task_id"])
    return tasks


def dry_run(task_dir: Path) -> None:
    solution = (task_dir / "solution/solution.patch").read_text()
    print(json.dumps({"task": str(task_dir), "snapshot": snapshot_of(task_dir),
                      "lines": task_builder.leak_tokens(solution),
                      "identifiers": task_builder.solution_identifiers(solution),
                      "windows": task_builder.change_windows(solution)}, indent=2))


def summary(runs: list) -> None:
    for run in runs:
        reports = [read_json(path) for path in sorted((RUNS / run / AUDIT).glob("*.json"))
                   if not path.name.endswith(".error.json")]
        if not reports:
            continue
        releases = {}
        for record in reports:
            for entry in record["archives_found"]:
                if entry["own_package"]:
                    releases.setdefault((entry["package"], entry["version"], entry["kind"]), entry["path"])
        print(f"== {run}: {len(reports)} reports")
        for (package, version, kind), path in sorted(releases.items()):
            print(f"   release {package}@{version} [{kind}] {path}")
        for record in reports:
            review = (record.get("manual_review") or {}).get("judgment")
            print(f"   {record['verdict']:8s}{' -> ' + review if review else '':14s} {record['task_id']}: "
                  f"{(record['reasons'] or ['-'])[0][:160]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("scan")
    sub.add_argument("--run", nargs="*", default=[])
    sub.add_argument("--task-dir", "--path", nargs="*", default=[], type=Path)
    sub.add_argument("--snapshot")
    sub.add_argument("--parallel", type=int, default=MAX_PARALLEL)
    sub.add_argument("--force", action="store_true")
    sub.add_argument("--dry-run", action="store_true")
    sub = commands.add_parser("review")
    sub.add_argument("--report", required=True, type=Path)
    sub.add_argument("--judgment", required=True, choices=("exposed", "clean", "unclear"))
    sub.add_argument("--note", required=True)
    sub = commands.add_parser("summary")
    sub.add_argument("--run", nargs="*", default=[])
    args = parser.parse_args()
    default_runs = sorted(path.parent.parent.name for path in RUNS.glob("official1-*/accepted/tasks.jsonl"))
    if args.command == "review":
        record = read_json(args.report)
        record["manual_review"] = {"judgment": args.judgment, "note": args.note,
                                   "at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
        write_json(args.report, record)
        return
    if args.command == "summary":
        summary(args.run or default_runs)
        return
    tasks = [path.resolve() for path in args.task_dir] or accepted_tasks(args.run or default_runs)
    if args.snapshot and len(tasks) != 1:
        parser.error("--snapshot needs exactly one --task-dir")
    if args.dry_run:
        for task_dir in tasks:
            dry_run(task_dir)
        return
    gate = asyncio.Semaphore(max(1, min(MAX_PARALLEL, args.parallel)))

    async def scan_all():
        return await asyncio.gather(*(scan_task(task_dir, args.snapshot or snapshot_of(task_dir), gate, args.force)
                                      for task_dir in tasks))

    records = asyncio.run(scan_all())
    counts = {}
    for record in records:
        counts[record["verdict"] if record else "error"] = counts.get(record["verdict"] if record else "error", 0) + 1
    print(f"done: {len(tasks)} tasks {counts}")


if __name__ == "__main__":
    main()
