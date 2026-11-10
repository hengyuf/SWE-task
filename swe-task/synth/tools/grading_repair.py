#!/usr/bin/env python3
"""Repair the test-file grading gap of EXISTING tasks (S/README.md, "Known grading gap"; hengyuf's decision 2026-10-19).

The shared DeepSWE grader resets only the files test.patch touches, so a model's edits to OTHER test files reach the
graded run (false FAILs: fastapi c04-r1b2; false PASSes: testem c01 kimi-r4, arcane c02-r1b4 sc14-r4; a redeclared Go
helper; model test files collected by a hidden testMatch; promlabels c05-r1b0 sc14-r1). verifier/grader.py is that
grader plus one marked step: every path model.patch touches that matches config.json's "test_paths" globs goes back to
base_commit before test.patch is applied. Existing tasks get it (and are re-scored by main); NEW builds keep the old
grader (task_builder.write_tests is unchanged).

  plan      [--run R ...]                          read-only: per in-scope run and task, the final graded attempt of
                                                   every probe slot, the test paths its model.patch touches, and which
                                                   slots to regrade -> repair/plan.json
  stage     --run R (--task T ... | --planned)     scratch copy of the task dir with the new grader and "test_paths"
                                                   (repair/tasks/<R>/<T>/new/<T>/), plus the same tests with the OLD
                                                   grader (.../old/<T>/tests/); Gym's task loader must accept the copy
  validate  --run R (--task T ... | --planned)     sandbox from the task snapshot (task.toml docker_image), scratch copy:
            [--tamper]                             reference solution -> 1, empty patch -> 0 (--tamper: also the
                                                   reference plus a model-added file at a test path -> 1 and deleted)
  regrade   --run R (--task T ... | --planned)     grade probe slots' model.patch on the scratch copy with the new and/or
            (--slot S ... | --scheduled)           OLD grader (results in repair/tasks/<R>/<T>/regrade.json; nothing in
            [--grader new|old|both]                the build tree changes; re-scoring is main's decision)
  install   --run R --task T ...                   copy the new grader and the "test_paths" key into
                                                   runs/<R>/accepted/<T>/tests/ (validated tasks only; backups
                                                   <file>.pre-test-sanitize when the replaced grader is DeepSWE's,
                                                   <file>.pre-grader-v3 when it is an earlier sanitizer,
                                                   PRIOR_GRADERS)

"test_paths_mode" (hengyuf 2026-10-20; verifier/grader.py): stage writes "non_additive" for JavaScript/TypeScript
repositories (top100.jsonl language, the source of the globs too: task_mode) and "all" for every other language. In
"non_additive" mode an existing test file that model.patch only ADDS lines to keeps the model's version (jest suites
mock internal modules, and a correct solution extends such a mock factory); "all" restores every test path.
Grader v3 (hengyuf 2026-10-21; verifier/grader.py): the sanitizer also cuts the Cargo [[test]]/[[bench]]/[[example]]/
[[bin]] tables a model added for a test file it deletes (official1-wasmi), and in "non_additive" mode restores a
kept-additive file whose added lines import a deleted or restored test path (official1-dynamodb); new log actions
dropped-manifest-target, restored-manifest-target and dangling-import (sanitize_marks).
Scratch copies: repair/tasks-v3/<R>/<T>/ (TASKS_DIR). repair/tasks/ holds the first repair (2026-10-19: the first
sanitizer, full restore, no "test_paths_mode") and repair/tasks-v2/ the test_paths_mode repairs (2026-10-20/21); neither
is written again. A task whose accepted tests already hold an earlier sanitizer (PRIOR_GRADERS) is staged from them: the
"old" control copy gets DeepSWE's grader and the packaged task's config.json.

Runs (check_run): IN_SCOPE, the 11 runs of the first repair (2026-10-19), and FINISHED_LATER, builds that ran with
DeepSWE's grader and finished afterwards (hengyuf 2026-10-20, "跑完后统一修复": repaired the same way once they finish).
A FINISHED_LATER run is in scope only while its build is finished (unfinished): one build root holding summary.json
(task_builder.build writes it last) and no pending or running Slurm job of the run (squeue: a job named after the run,
e.g. synth-build-<R>, or a job id of runs/<R>/launch.jsonl; a failed query refuses the run). Every other run is
refused: live builds are never touched.

Recovered tasks (batch 3c, 2026-10-20): a task score() left inconclusive (fewer than 3 valid rollouts per model) whose
tools/recover_slots.py record (runs/<R>/audit/recovery/<T>.json) recommends newly_in_range is planned as
"recovered_in_range" (recovery_in_range) and scheduled like an accepted task (guard_rule: the flips that could take it
out of range, i.e. the record's pending_repair slots). Its repair record (runs/<R>/audit/grading_repair/<T>.json,
statuses_after) is what `recover_slots.py record` reads; install still needs an accepted task (main's post-hoc
acceptance, then a re-plan that finds it accepted).

Tasks graded by a sanitizer already (batch 3e, 2026-10-21): a build resumed after the new-build grader switch gates its
remaining candidates with verifier/grader.py and "test_paths" (task_builder.write_tests; official1-anko round 1), so
their probes graded every slot with the sanitizer. plan reads the grader of the scored probe's own copy of the task
(probe_grader: <probe>/tasks/<T>/tests/, which rollouts.probe copies at probe creation and ATP grades from): a known
sanitizer (current or PRIOR_GRADERS) makes the task "skip_sanitized" (sanitized_skip; listed in skipped with that
grader, its test_paths and mode, and "decision": the status it would have been planned with). Such a task is never
staged, regraded or installed: its recorded outcomes are sanitized grades already. Accepted ones are listed in the run
summary's "accepted_sanitized"; test_paths or a mode other than the run's is a run problem (a gap the repair would
close).

Never touches probe data, metrics.json, stage.json, cand-*/ directories, build code or prompts: the tool's own outputs
live under repair/ (plus the two files install replaces in accepted/<T>/tests/). Every write is atomic and keeps the
first version of a replaced file as <file>.pre-test-sanitize; every action is logged to repair/test-sanitize.jsonl.
Sandboxes: at most 6 at a time (--concurrency); sandbox.owned backs off when the shared pool is full.
Run from S/synth with the pipeline venv, e.g.
  ../atp/runtime/controller/bin/python3 -B tools/grading_repair.py plan
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import asyncio  # noqa: E402
import contextlib  # noqa: E402
import datetime  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import pwd  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import uuid  # noqa: E402

SYNTH = Path(__file__).resolve().parents[1]
for _path in (SYNTH, SYNTH / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common import DEEPSWE, RUNS, STUDENT, TEACHER, TOP100, read_json  # noqa: E402
import regrade  # noqa: E402  (reused: failing_tests, regraded_record -- the slot record as score() takes it)
import rollouts  # noqa: E402
import sandbox  # noqa: E402
import task_builder  # noqa: E402

IN_SCOPE = ("official1-narwhals", "official1-bandit", "official1-vitest", "official1-httpx", "official1-cattrs",
            "official1-fastapi", "official1-testem", "official1-promlabels", "official1-statemachine",
            "official1-koota", "official1-arcane")  # the first repair (2026-10-19): finished builds, always in scope
# Builds that ran with DeepSWE's grader and finished later (hengyuf 2026-10-20): in scope only while check_run finds
# them finished (unfinished() is None). Every other official1-* run is a live build: never touched. Batch 2: the first
# seven; batch 3a (2026-10-20 18:4x, backup .pre-batch3a): official1-dynamodb (build finished 18:33); batch 3b
# (2026-10-20 20:0x, backup .pre-batch3b): official1-sqlformatter (build job 20540630 COMPLETED 19:44); batch 3c
# (2026-10-20 23:1x, backup .pre-batch3c): official1-scriggo (build job 20553877 COMPLETED 23:04); batch 3d
# (2026-10-21 07:0x, backup .pre-batch3d): official1-wasmi (build job 20552976 COMPLETED 2026-10-21 06:49); batch 3e
# (2026-10-21 12:3x, backup .pre-batch3e): official1-anko (build job 20580934 COMPLETED 2026-10-21 12:12; its round-1
# tasks were gated and probed with the sanitizer after the build's resume: skip_sanitized).
FINISHED_LATER = ("official1-aiomonitor", "official1-etree", "official1-dasel", "official1-absslices", "official1-ytt",
                  "official1-opa", "official1-promreload", "official1-dynamodb", "official1-sqlformatter",
                  "official1-scriggo", "official1-wasmi", "official1-anko")
RUN_ORDER = IN_SCOPE + FINISHED_LATER  # plan's default runs and plan.json's task order
SLURM_BIN = Path("/cm/shared/apps/slurm/current/bin")
SLURM_CONF = "/cm/shared/apps/slurm/var/etc/cw-dfw-cs-001/slurm.conf"  # login nodes run no slurmd (launch_build.sh)
NEW_GRADER = SYNTH / "verifier/grader.py"
OLD_GRADER = DEEPSWE / "abs-stepped-slices/tests/grader.py"  # DeepSWE's, byte-identical in every task
REPAIR = SYNTH / "repair"
PLAN = REPAIR / "plan.json"
LOG = REPAIR / "test-sanitize.jsonl"
TASKS_DIR = "tasks-v3"  # scratch copies under REPAIR ("tasks" of the first repair and "tasks-v2" stay as they were)
BACKUP = ".pre-test-sanitize"  # first version of a file whose DeepSWE grader install replaced
BACKUP_V2 = ".pre-non-additive"  # the first sanitizer's files that the test_paths_mode grader replaced (2026-10-20)
BACKUP_V3 = ".pre-grader-v3"  # an earlier sanitizer's files that grader v3 replaced (2026-10-21)
# Earlier versions of verifier/grader.py that install may replace in accepted/<T>/tests/ (sha256 -> what it was)
PRIOR_GRADERS = {"1951f9103ddabe2c966be93c2e25e9307f98abd9510c8614aecfc8754d924c28":
                 "first test-path sanitizer (2026-10-19): restores every test path, no test_paths_mode",
                 "3936c324fb5d11e57c3e6d4397814808c6c660ef36ead93b58808adbaa4ac0ee":
                 "test_paths_mode sanitizer (2026-10-20, v2): keeps dangling Cargo target tables and imports"}
MODES = {"JavaScript": "non_additive", "TypeScript": "non_additive"}  # test_paths_mode per top100 language; else "all"
MARK = "[verifier] test-sanitize:"
MAX_SANDBOXES = 6
TESTS_FILES = ("test.sh", "test.patch", "config.json")  # with grader.py: what sandbox.grade / Gym stage
GLOBS = {  # hengyuf's defaults per repository language (top100.jsonl "language" of the build's repository)
    "Python": ["tests/**", "test/**", "**/test_*.py", "**/*_test.py", "**/conftest.py"],
    "Go": ["**/*_test.go", "**/testdata/**"],
    "JavaScript": ["**/*.test.*", "**/*.spec.*", "**/__tests__/**", "test/**", "tests/**"],
    "TypeScript": ["**/*.test.*", "**/*.spec.*", "**/__tests__/**", "test/**", "tests/**"],
    "Rust": ["tests/**"],
}
FAMILY = {"Python": "Python", "Go": "Go", "JavaScript": "JS/TS", "TypeScript": "JS/TS", "Rust": "Rust"}
EXTENSIONS = {".py": "Python", ".pyi": "Python", ".go": "Go", ".rs": "Rust", ".ts": "JS/TS", ".tsx": "JS/TS",
              ".mts": "JS/TS", ".cts": "JS/TS", ".js": "JS/TS", ".jsx": "JS/TS", ".mjs": "JS/TS", ".cjs": "JS/TS"}
# Per-task glob overrides (narrowing) when a reference solution.patch touches a default glob: {task_id: [globs]}.
# plan reports such hits; none of the in-scope tasks needed one on 2026-10-19.
GLOB_OVERRIDES = {}
# Per-run replacements of the language defaults ({run: [globs]}), e.g. after reviewing a run's "glob_gaps" in plan.json
# (official1-narwhals: ibis keeps SQL snapshots and helpers in nested tests/ dirs, which "**/tests/**" would cover and
# no ibis reference solution touches). Each override is the language defaults plus the run's test directories, as new
# builds restore them per task since 2026-10-20 (task_builder.test_patch_globs: test-named dirs test.patch touches).
RUN_GLOBS = {
    # batch 3b (2026-10-20): mtxr/vscode-sqltools keeps the formatter suites with their shared cases
    # (packages/formatter/test/behavesLikeSqlFormatter.ts, run by every dialect suite), golden JSON, hidden helpers and
    # fixtures/ in packages/<package>/test/, which the root-only "test/**" misses (plan glob_gaps: 12 slot patches edit
    # behavesLikeSqlFormatter.ts). No reference solution.patch of the run touches packages/*/test/.
    "official1-sqlformatter": GLOBS["TypeScript"] + ["packages/*/test/**"],
    # batch 3c (2026-10-20): cosmos72/gomacro keeps two fixture sets outside the Go defaults, read by graded P2P tests:
    # ast2/test/ (z_test.go, z_test_data_2.txt, the placeholder empty.go; TestToNodes/z_test_data_2 is P2P in 13 of the
    # 21 planned tasks) and the TestSorter fixtures base/dep/z_test_data_{1,2,3}.txt (TestSorter/z_test_data_N in 8).
    # base/dep itself is source (reference solutions of c04-r1b3 and c05-r1b2 edit base/dep/scope.go), so only its
    # fixture files are listed, never the directory. No slot patch, test.patch or reference solution.patch of the run
    # touches either.
    "official1-scriggo": GLOBS["Go"] + ["ast2/test/**", "base/dep/z_test_data_*.txt"],
    # batch 3d (2026-10-21): bytecodealliance/wasmtime (a Rust workspace) keeps each crate's integration tests in
    # crates/<crate>/tests/ (cargo test targets), which the root-only "tests/**" misses: test.patch adds the hidden
    # tests in crates/wasmtime/tests/ in 5 built tasks (accepted c01-r1b1 and c05-r1b1 among them; the graded
    # `wasmtime` test binaries) and edits crates/wizer/tests/all/ in round-0 c01 and c02 (wasmtime-wizer), and 21 slot
    # patches edit both (plan glob_gaps). No reference solution.patch of the run touches a crates/*/tests/ path (they
    # edit crates/*/src/, crates/wasmtime/Cargo.toml, docs/). Deliberately NOT test paths: crates/debugger/src/lib.rs
    # (c03's test.patch edits its inline `mod test`, c03-r1b4's reference solution edits the file) and
    # crates/test-programs/src/bin/ (guest programs of the test-programs crate: source by name, as
    # task_builder.test_patch_globs treats them; no slot patch edits them).
    "official1-wasmi": GLOBS["Rust"] + ["crates/*/tests/**"],
    # batch 3e (2026-10-21): go-python/gpython's vm.TestVm, py.TestPy and each stdlib package's TestVm run every script
    # in vm/tests/, py/tests/ and stdlib/<module>/tests/ as subtests of graded parents (P2P vm.TestVm/call.py, ...),
    # which the Go defaults miss: round-0 test.patch adds hidden scripts there (vm/tests/hidden_defaults_*.py,
    # py/tests/function.py, stdlib/math/tests/*.py) and 241 slot patches edit or add scripts there (plan glob_gaps:
    # vm/tests 266 paths, py/tests 42). main added the same three directories to every round-1 task's "test_paths"
    # (2026-10-20 19:33-20:17, tools/add_test_paths.py, before the round-1 probe was created at 20:31), so the whole run
    # is graded alike. No reference solution.patch touches them (they edit ast/, compile/, parser/, py/*.go, symtable/,
    # vm/*.go). Deliberately NOT test paths: the generators compile/make_compile_test.py, parser/make_grammar_test.py
    # and symtable/make_symtable_test.py (test.patch edits them, no graded test runs them; source directories, as
    # task_builder.test_patch_globs treats them).
    "official1-anko": GLOBS["Go"] + ["vm/tests/**", "py/tests/**", "stdlib/*/tests/**"],
}
# Directory segments that hold test material in most layouts; plan lists unmatched paths under them ("glob_gaps").
TEST_DIR = re.compile(r"(^|/)(tests?|__tests__|__typetests__|testdata|__snapshots__|snapshots|fixtures?)/")


class RepairError(RuntimeError):
    pass


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NEW = load_module(NEW_GRADER, "test_sanitize_grader")  # its own matcher and patch parser: plan == grader behaviour
OLD = load_module(OLD_GRADER, "deepswe_grader_reference")


# ============================================================================================ files and log

def sha256(data) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def file_sha(path: Path):
    return sha256(Path(path).read_bytes()) if Path(path).exists() else None


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write(path: Path, data, *, backup=True, dry_run=False, suffix=BACKUP) -> dict:
    """Write path atomically (temp file + rename in the same directory); the FIRST version of a file it replaces stays
    as <file><suffix> (default .pre-test-sanitize). Returns what happened (for the log)."""
    path = Path(path)
    data = data.encode() if isinstance(data, str) else data
    kept = path.with_name(path.name + suffix)
    entry = {"path": str(path), "sha256": sha256(data), "replaced": path.exists()}
    if path.exists() and backup:
        entry["backup"] = str(kept)
        entry["backup_existed"] = kept.exists()
    if dry_run:
        return entry
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and backup and not kept.exists():
        partial = kept.with_name(f".{kept.name}.{uuid.uuid4().hex}.tmp")
        shutil.copy2(path, partial)
        os.replace(partial, kept)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(data)
    if path.exists():
        shutil.copymode(path, temporary)
    os.replace(temporary, path)
    return entry


def write_json_atomic(path: Path, value, **kw) -> dict:
    return atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n", **kw)


def log_action(entry: dict, dry_run=False) -> dict:
    entry = {"at": now(), "tool": "tools/grading_repair.py", **entry}
    if dry_run:
        return entry
    REPAIR.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return entry


@contextlib.contextmanager
def locked(name: str, dry_run=False):
    """One tool process per run (or for plan) at a time; the lock files live in repair/."""
    if dry_run:
        yield
        return
    REPAIR.mkdir(parents=True, exist_ok=True)
    with (REPAIR / f".{name}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another tools/grading_repair.py holds {REPAIR / f'.{name}.lock'}") from None
        yield


# ============================================================================================ probe data (read-only)

SUPPRESSED = []  # probe-side cache writes that ATP's readers would have made; plan reports them, nothing is written


@contextlib.contextmanager
def probe_reads_only():
    """rollouts.probe_records -> ATP's slot_state may refresh a generation-validity cache (atomic_json) or record a
    torn JSONL tail (read_jsonl) inside the probe's attempt directories. Probe data must never change here, so both
    writers are replaced for the duration: the returned states are the same, the writes are only recorded."""
    import deepswe_campaign as campaign
    real_atomic, real_jsonl = campaign.atomic_json, campaign.read_jsonl

    def no_write_json(path, value):
        SUPPRESSED.append(str(path))

    def read_jsonl(path):
        path = Path(path)
        rows = []
        lines = path.read_bytes().splitlines(keepends=True)
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL rows must be objects")
                rows.append(row)
            except (ValueError, UnicodeError):
                if index != len(lines) - 1 or line.endswith(b"\n"):
                    raise ValueError(f"Corrupt JSONL row {index + 1} in {path}") from None
                SUPPRESSED.append(str(path.with_name(path.name + ".interrupted-tail")))
        return rows

    campaign.atomic_json, campaign.read_jsonl = no_write_json, read_jsonl
    try:
        yield
    finally:
        campaign.atomic_json, campaign.read_jsonl = real_atomic, real_jsonl


_RECORDS = {}


def probe_records(probe: Path) -> dict:
    probe = Path(probe).resolve()
    if probe not in _RECORDS:
        with probe_reads_only():
            _RECORDS[probe] = rollouts.probe_records(probe)
    return _RECORDS[probe]


def md_header(path: Path):
    """(slot, status) from the first line score() wrote: "# <slot> — <status>"."""
    if not path.exists():
        return None, None
    with path.open(errors="replace") as stream:
        first = stream.readline().strip()
    match = re.match(r"^# (\S+) — (\S+)$", first)
    return (match.group(1), match.group(2)) if match else (None, None)


def earliest(scored: Path, name: str) -> Path:
    """The first scoring's copy of a score() output (regrade.py moves replaced ones aside)."""
    for suffix in (".pre-regrade", ".pre-gate-fix"):
        if (scored / f"{name}{suffix}").exists():
            return scored / f"{name}{suffix}"
    return scored / name


def scored_probe(round_dir: Path, task_id: str, scored: Path):
    """The probe whose records score() rendered into the first scoring's rollouts/ (several probes of a round can hold
    a task: official1-narwhals round-1 kept its abandoned probe-7f97a213)."""
    first = earliest(scored, "rollouts")
    found, problems = [], []
    for probe in sorted(round_dir.glob("probe-*")):
        manifest = read_json(probe / "manifest.json") or {}
        if task_id not in {t.get("task_id") for t in manifest.get("tasks", [])}:
            continue
        records = probe_records(probe).get(task_id) or []
        mismatch = []
        for record in records:
            md, patch = first / f"{record['slot']}.md", first / f"{record['slot']}.patch"
            if not md.exists() or md.read_text(errors="replace").split("\n")[:3] != \
                    rollouts.render(record).split("\n")[:3]:
                mismatch.append(record["slot"])
            elif patch.exists() and patch.read_text() != record["patch"]:
                mismatch.append(record["slot"])
        if records and not mismatch:
            found.append((probe, records))
        else:
            problems.append(f"{probe.name}: {len(mismatch)} slot(s) differ from {first.name}/")
    if len(found) != 1:
        raise RepairError(f"{task_id}: {len(found)} probes match the scored rollouts ({'; '.join(problems)})")
    return found[0]


# ============================================================================================ build tree (read-only)

_FINISHED = set()  # FINISHED_LATER runs check_run found finished (one check per process)


def slurm_jobs(users) -> list:
    """Every unfinished Slurm job of these users ([{id, state, name}]; squeue's default states: pending, running,
    suspended, completing, ...). Raises when squeue fails: no scope decision on partial information."""
    squeue = str(SLURM_BIN / "squeue") if (SLURM_BIN / "squeue").exists() else shutil.which("squeue")
    if not squeue:
        raise RepairError("no squeue")
    env = {"HOME": os.environ.get("HOME", "/"), "USER": os.environ.get("USER", ""), "LANG": "C.UTF-8",
           "SLURM_CONF": os.environ.get("SLURM_CONF") or SLURM_CONF, "PATH": f"{SLURM_BIN}:/usr/bin:/bin"}
    proc = subprocess.run([squeue, "-h", "-u", ",".join(sorted(users)), "-o", "%i|%T|%j"], capture_output=True,
                          text=True, env=env, timeout=120)
    if proc.returncode != 0:
        raise RepairError(f"squeue exit {proc.returncode}: {(proc.stderr or proc.stdout).strip()[-300:]}")
    jobs = []
    for line in proc.stdout.splitlines():
        if line.strip():
            job_id, state, name = (line.split("|", 2) + ["", ""])[:3]
            jobs.append({"id": job_id.strip(), "state": state.strip(), "name": name.strip()})
    return jobs


def launched_job_ids(run: str) -> set:
    """Job ids runs/<R>/launch.jsonl recorded (tools/launch_build.sh, every launch and resume of the build)."""
    path, ids = RUNS / run / "launch.jsonl", set()
    for line in (path.read_text(errors="replace").splitlines() if path.exists() else []):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("job_id"):
            ids.add(str(row["job_id"]))
    return ids


def unfinished(run: str):
    """None when the run's build is finished, else why not: exactly one build root under runs/<R>/build, holding
    summary.json (task_builder.build writes it last; task_builder.cleanup's test), and no unfinished Slurm job of the
    run: a job whose name holds the run name as a whole token (synth-build-<R>, ...) or whose id launch.jsonl recorded,
    queried for the owner of runs/<R> and the current user. The Slurm job this process runs in (SLURM_JOB_ID,
    SLURM_ARRAY_JOB_ID) does not count: a repair job may be named after its run. A resumed build keeps its earlier
    summary.json until it ends, hence both checks; a failed squeue query counts as unfinished."""
    roots = sorted(p for p in (RUNS / run / "build").glob("*/*") if p.is_dir())
    if len(roots) != 1:
        return f"runs/{run}/build holds {len(roots)} builds"
    if not (roots[0] / "summary.json").is_file():
        return f"no {roots[0] / 'summary.json'} (the build has not finished)"
    try:
        users = {pwd.getpwuid(os.getuid()).pw_name, pwd.getpwuid((RUNS / run).stat().st_uid).pw_name}
        jobs = slurm_jobs(users)
    except (OSError, KeyError, subprocess.SubprocessError, RepairError) as error:
        return f"cannot list the Slurm jobs ({type(error).__name__}: {error})"
    launched = launched_job_ids(run)
    own = {os.environ.get(name) for name in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID")} - {None, ""}
    token = re.compile(r"(?<![A-Za-z0-9])" + re.escape(run) + r"(?![A-Za-z0-9])")
    live = [job for job in jobs if not ({job["id"], re.split(r"[_+]", job["id"])[0]} & own)
            and (re.split(r"[_+]", job["id"])[0] in launched or token.search(job["name"]))]
    if live:
        return "Slurm job(s) of the run not finished: " + ", ".join(f"{j['id']} {j['name']} {j['state']}"
                                                                     for j in live)
    return None


def check_run(run: str) -> str:
    """IN_SCOPE runs; FINISHED_LATER runs only while their build is finished (unfinished); nothing else."""
    if run in IN_SCOPE or run in _FINISHED:
        return run
    if run not in FINISHED_LATER:
        raise SystemExit(f"{run} is not one of the in-scope runs: {', '.join(RUN_ORDER)}")
    problem = unfinished(run)
    if problem:
        raise SystemExit(f"{run} is refused while its build may be live: {problem}")
    _FINISHED.add(run)
    return run


def run_rank(run: str) -> int:
    """plan.json's run order (RUN_ORDER; any other run after them)."""
    return RUN_ORDER.index(run) if run in RUN_ORDER else len(RUN_ORDER)


def build_root(run: str) -> Path:
    roots = sorted(p for p in (RUNS / check_run(run) / "build").glob("*/*") if p.is_dir())
    if len(roots) != 1:
        raise RepairError(f"runs/{run}/build holds {len(roots)} builds")
    return roots[0]


def repo_language(target: str, repo: str) -> str:
    rows = [json.loads(line) for line in TOP100.read_text().splitlines() if line.strip()]
    mine = [r for r in rows if r["repo"].lower() == repo.lower()]
    exact = sorted({r["language"] for r in mine if r["task_id"] == target})
    languages = exact or sorted({r["language"] for r in mine})
    if len(languages) != 1:
        raise RepairError(f"top100.jsonl gives {languages} for {repo}")
    return languages[0]


def files_language(*patches: str):
    """The dominant language of the files a task's patches touch (by extension), as a GLOBS family."""
    counts = {}
    for patch in patches:
        for path in NEW.touched_paths(patch):
            family = EXTENSIONS.get(os.path.splitext(path)[1].lower())
            if family:
                counts[family] = counts.get(family, 0) + 1
    return max(counts, key=counts.get) if counts else None


def accepted_ids(run: str) -> list:
    path = RUNS / run / "accepted/tasks.jsonl"
    if not path.exists():
        return []
    return [json.loads(line)["task_id"] for line in path.read_text().splitlines() if line.strip()]


def withdrawn_ids(run: str) -> set:
    folder = RUNS / run / "withdrawn"
    return {p.name for p in folder.iterdir() if p.is_dir() and not p.name.endswith(".workspace")} \
        if folder.is_dir() else set()


def scored_tasks(root: Path) -> list:
    """Every task score() wrote metrics for: (round dir, scored dir, metrics)."""
    out = []
    for metrics_path in sorted(root.glob("round-*/cand-*/metrics.json")):
        metrics = read_json(metrics_path)
        if isinstance(metrics, dict) and metrics.get("task_id"):
            out.append((metrics_path.parent.parent, metrics_path.parent, metrics))
    return out


def classify(metrics: dict, scored: Path, accepted: set, withdrawn: set) -> tuple:
    """(status, why): accepted | near_bar_candidate (decided later) | skip_* statuses."""
    task_id, reason = metrics["task_id"], str(metrics.get("reason") or "")
    if task_id in withdrawn or metrics.get("withdrawn"):
        return "skip_withdrawn", str(metrics.get("withdrawn") or reason)[:300]
    if task_id in accepted:
        if not metrics.get("accept"):
            return "accepted_metrics_disagree", reason
        return "accepted", reason
    if reason.startswith("similarity"):
        return "skip_similarity", reason
    if reason.startswith("inconclusive"):
        return "skip_inconclusive", reason
    invalid = task_builder.seed_check(scored, metrics)
    if invalid:
        return "skip_invalid", invalid
    if reason.startswith("pass rates out of range"):
        return "rates", reason
    return "skip_other", reason


def recovery_in_range(run: str, task_id: str):
    """What tools/recover_slots.py's record of a task (runs/<R>/audit/recovery/<T>.json) says when it recommends
    newly_in_range: the post-hoc recovery of lost slots made a task that score() left inconclusive valid and in range
    (decision view: the sanitizing grader), pending the grading repair of recorded slots whose patches edit test paths
    that test.patch does not reset (pending_repair). None without such a record (missing or unreadable, another
    recommendation, another task or run)."""
    path = RUNS / run / "audit/recovery" / f"{task_id}.json"
    record = read_json(path)
    if not isinstance(record, dict) or record.get("recommendation") != "newly_in_range" \
            or record.get("task_id") != task_id or record.get("run", run) != run:
        return None
    view = record.get("decision_view") or "sanitizing_grader"
    after = (record.get("after") or {}).get(view) or {}
    pending = record.get("pending_repair") or {}
    return {"path": str(path), "sha256": file_sha(path), "at": record.get("at"), "label": record.get("label"),
            "recommendation": record["recommendation"], "decision_view": view,
            "rates_after": {key: after.get(key) for key in ("sc14", "kimi_clean", "valid", "in_range")},
            "pending_repair": list(pending.get("pending") or []),
            "verdict_can_change": pending.get("verdict_can_change")}


SANITIZED = ("current", "prior")  # grader_kind of a sanitizing grader (verifier/grader.py, or a PRIOR_GRADERS one)


def probe_grader(probe: Path, task_dir) -> dict:
    """The grader that graded a task's probe slots: the scored probe's own copy of the task's tests/ (rollouts.probe
    copies every task dir to <probe>/tasks/<name>/ when it creates the probe, and ATP grades from that copy), else the
    packaged task's tests/. kind: grader_kind (deepswe | current | prior | None for an unknown file)."""
    tests = Path(probe) / "tasks" / Path(task_dir).name / "tests"
    if not (tests / "grader.py").is_file():
        tests = Path(task_dir) / "tests"
    sha = file_sha(tests / "grader.py")
    config = read_json(tests / "config.json")
    config = config if isinstance(config, dict) else {}
    return {"tests": str(tests), "grader_sha256": sha, "kind": grader_kind(sha) if sha else None,
            "test_paths": config.get("test_paths"), "test_paths_mode": config.get("test_paths_mode")}


def sanitized_skip(metrics: dict, scored: Path, status: str, graded: dict, globs: list, mode: str):
    """None unless a sanitizing grader graded the task's probe slots (graded: probe_grader); then the task's skipped
    record "skip_sanitized" (batch 3e: official1-anko round 1, gated after the build's resume). Its recorded outcomes
    are sanitized grades already, so nothing is staged, regraded or installed. "decision" keeps the status it would
    have been planned with (accepted | rates | recovered_in_range); "matches_run_globs": its test_paths (in any order)
    and mode (absent: "all", the grader's default) equal the run's."""
    if graded.get("kind") not in SANITIZED:
        return None
    got = graded.get("test_paths_mode") or "all"
    same = sorted(graded.get("test_paths") or []) == sorted(globs) and got == mode
    return {"task_id": metrics["task_id"], "cand": Path(scored).name, "status": "skip_sanitized", "decision": status,
            "why": f"{status}, but its probe graded every slot with a sanitizing grader ({graded['kind']} "
                   f"{graded['grader_sha256'][:12]}, test_paths {graded.get('test_paths')}, mode {got}"
                   f"{'' if same else '; NOT the run globs ' + str(list(globs)) + ', mode ' + mode}): the recorded "
                   "outcomes are sanitized grades, nothing to regrade or install",
            "probe_grader": graded, "matches_run_globs": same}


# ============================================================================================ plan

def task_globs(task_id: str, language: str, run: str = None) -> list:
    return list(GLOB_OVERRIDES.get(task_id) or RUN_GLOBS.get(run) or GLOBS[language])


def glob_gaps(root: Path, globs: list, language: str) -> dict:
    """Paths the scored rollouts' patches touch under test-material directories (TEST_DIR; e2e/ in JS/TS repos) that
    no glob matches: what the sanitizer would NOT reset. Directories a reference solution of the run touches are
    marked (they must stay outside the globs)."""
    solution_dirs = set()
    for stage_path in root.glob("cand-*/stage.json"):
        stage = read_json(stage_path) or {}
        solution = Path(stage.get("task_dir", "/nonexistent")) / "solution/solution.patch"
        if stage.get("ok") and solution.exists():
            solution_dirs |= {p.rsplit("/", 1)[0] for p in NEW.touched_paths(solution.read_text(errors="replace"))}
    counts, slots = {}, 0
    for patch in root.glob("round-*/cand-*/rollouts/*.patch"):
        missed = [p for p in NEW.touched_paths(patch.read_text(errors="replace"))
                  if (TEST_DIR.search(p) or (FAMILY[language] == "JS/TS" and p.startswith("e2e/")))
                  and not matching([p], globs)]
        slots += bool(missed)
        for path in missed:
            directory = path.rsplit("/", 1)[0]
            counts[directory] = counts.get(directory, 0) + 1
    top = sorted(counts.items(), key=lambda item: -item[1])[:12]
    return {"slot_patches": slots, "dirs": [{"dir": d, "paths": n, "reference_solution_touches": d in solution_dirs}
                                             for d, n in top]}


def matching(paths, globs) -> list:
    return [p for p in paths if any(NEW.glob_match(g, p) for g in globs)]


def path_kinds(patch: str) -> dict:
    """{path: kind} per file block of a git diff: modified | added | deleted | renamed-away | renamed-to | copy-source |
    copied-to (an annotation for prioritising; the grader itself treats every kind alike)."""
    kinds = {}
    for block in re.split(r"(?m)^(?=diff --git )", patch):
        if not block.startswith("diff --git "):
            continue
        header = re.split(r"(?m)^(?:@@ |GIT binary patch)", block, maxsplit=1)[0]
        lines = header.split("\n")
        named = {}
        for line in lines:
            for key in ("rename from ", "rename to ", "copy from ", "copy to "):
                if line.startswith(key):
                    named[key.strip()] = NEW.git_unquote(line[len(key):])
        if "rename from" in named:
            kinds.setdefault(named["rename from"], "renamed-away")
            kinds.setdefault(named.get("rename to", ""), "renamed-to")
            continue
        if "copy from" in named:
            kinds.setdefault(named["copy from"], "copy-source")
            kinds.setdefault(named.get("copy to", ""), "copied-to")
            continue
        kind = "added" if any(l.startswith("new file mode") for l in lines) else \
            "deleted" if any(l.startswith("deleted file mode") for l in lines) else "modified"
        for path in NEW.touched_paths(header):
            kinds.setdefault(path, kind)
    kinds.pop("", None)
    return kinds


EXISTING_KINDS = frozenset({"modified", "deleted", "renamed-away"})  # a base_commit test file the model changed
PRIORITIES = ("existing", "added_loaded", "added_collected", "added_tree_checked", "added_unknown",
              "added_uncollected")
UNLIKELY = frozenset({"added_uncollected"})
# whitelisted tests that inspect the whole tree: the [gate] build/typecheck convention (jest's `yarn build:ts`
# typechecks test files too) and repository style checks (pytransitions' tests/test_codestyle.py runs pycodestyle and
# mypy over the tests). They see a model-added test file even when no suite collects it. Deliberately narrow: pylama's
# tests.test_linters.test_pycodestyle or a "jest-runner-tsc" displayName test lint fixtures, not the tree.
TREE_CHECK = re.compile(r"\[gate\]|\btest_codestyle\b|\bTestCodeFormat\b|\bTestCodeStyle\b")


def report_evidence(attempt: Path):
    """The raw suite output of the slot's ORIGINAL grading (the grading/attempt-*/ that rollouts._record read):
    reports/* and run.log from verifier-logs.tar.gz, read in memory. None when there is none."""
    import tarfile
    ctrfs = sorted(Path(attempt).glob("grading/attempt-*/ctrf.json")) if attempt else []
    archive = ctrfs[-1].parent / "verifier-logs.tar.gz" if ctrfs else None
    if not archive or not archive.exists():
        return None
    texts = []
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            name = member.name.lstrip("./")
            if member.isfile() and (name.startswith("reports/") or name == "run.log") and member.size < 64 << 20:
                texts.append(tar.extractfile(member).read().decode("utf-8", "replace"))
    return "\n".join(texts)


def identities(path: str) -> list:
    """How a test file shows up in raw reports: pytest's dotted module, or the (tail of the) path."""
    out = [path]
    if path.endswith(".py"):
        out.append(path[:-3].replace("/", "."))
    parts = path.split("/")
    if len(parts) > 2:
        out.append("/".join(parts[-2:]))
    return out


def seen_in(text: str, path: str) -> bool:
    return any(re.search(r"(?<![\w.\-])" + re.escape(token) + r"(?![\w\-])", text) for token in identities(path))


def slot_priority(kinds: dict, language: str, evidence, witnesses: list, tree_checks: bool):
    """existing: a base_commit test file was edited, deleted or renamed away (the tests it hosts change);
    added_loaded: only new test files, but ones the runner loads with the hidden tests anyway (Go: every *_test.go
    and testdata file of a tested package compiles/reads together; Python: conftest.py, __init__.py);
    added_collected: a new test file shows up in the original grading's raw reports/run log (collected or errored);
    added_tree_checked: not collected, but the task whitelists a whole-tree check (codestyle, lint, typecheck, gate);
    added_unknown: no raw reports, or they never name test files (not even the hidden ones);
    added_uncollected: the raw reports name the hidden test files but none of the model's new ones."""
    if not kinds:
        return None
    if any(kind in EXISTING_KINDS for kind in kinds.values()):
        return "existing"
    if language == "Go" or any(p.rsplit("/", 1)[-1] in ("conftest.py", "__init__.py") for p in kinds):
        return "added_loaded"
    if evidence is not None and any(seen_in(evidence, p) for p in kinds):
        return "added_collected"
    if tree_checks:
        return "added_tree_checked"
    if evidence is None or not any(seen_in(evidence, w) for w in witnesses):
        return "added_unknown"
    return "added_uncollected"


def slot_entry(record: dict, current: str, test_reset: set, globs: list, verdict, language: str,
               witnesses: list, tree_checks: bool) -> dict:
    touched = NEW.touched_paths(record["patch"])
    tests = matching(touched, globs)
    kinds = path_kinds(record["patch"])
    attempt = Path(record["attempt_dir"]) if record.get("attempt_dir") else None
    patch_file = attempt / "model.patch" if attempt else None
    surviving = [p for p in tests if p not in test_reset]
    surviving_kinds = {p: kinds.get(p, "modified") for p in surviving}
    evidence = report_evidence(attempt) if surviving and not any(
        k in EXISTING_KINDS for k in surviving_kinds.values()) and language != "Go" else None
    entry = {"slot": record["slot"], "model": record["model"], "status": current, "probe_status": record["status"],
             "f2p": record["f2p"], "p2p": record["p2p"], "verifier_error": record["verifier_error"],
             "agent_error": record["agent_error"], "attempt_dir": str(attempt) if attempt else None,
             "model_patch": str(patch_file) if patch_file and patch_file.exists() else None,
             "model_patch_matches_record": (patch_file.read_text(errors="surrogateescape") == record["patch"])
             if patch_file and patch_file.exists() else None,
             "patch_sha256": sha256(record["patch"]), "patch_chars": len(record["patch"]),
             "test_paths": tests,
             # what the OLD grader leaves in the graded run: test paths outside the files test.patch resets
             "surviving": surviving, "surviving_kinds": surviving_kinds,
             "priority": slot_priority(surviving_kinds, language, evidence, witnesses, tree_checks)}
    if record["model"] == TEACHER and current == "pass":
        entry["cheated"] = bool((verdict or {}).get("cheated")) if isinstance(verdict, dict) else None
    return entry


def rates(slots: list) -> dict:
    s = [x for x in slots if x["model"] == STUDENT and x["status"] in ("pass", "fail")]
    t = [x for x in slots if x["model"] == TEACHER and x["status"] in ("pass", "fail")]
    clean = sum(x["status"] == "pass" and x.get("cheated") is False for x in t)
    return {"sc14": sum(x["status"] == "pass" for x in s) / len(s) if s else None,
            "kimi_clean": clean / len(t) if t else None, "sc14_valid": len(s), "kimi_valid": len(t),
            "sc14_pass": sum(x["status"] == "pass" for x in s), "kimi_clean_pass": clean,
            "kimi_cheated_pass": sum(x["status"] == "pass" and x.get("cheated") is True for x in t)}


def in_range(sc14, kimi) -> bool:
    """score()'s own test (sc14 <= 0.2 and clean kimi >= 0.6), same float arithmetic."""
    return sc14 is not None and kimi is not None and not (sc14 > 0.2 or kimi < 0.6)


GUARDED = ("accepted", "recovered_in_range")  # plan statuses whose in-range verdict the repair must not lose


def guard_rule(slot: dict, label: str) -> tuple:
    """(why, schedule) for an affected slot of a GUARDED task: the flips that could take it out of range (sc14 FAIL ->
    pass, clean kimi PASS -> fail) are scheduled, the others listed (the 2026-10-19 rule for accepted tasks)."""
    if slot["model"] == STUDENT and slot["status"] == "fail":
        return f"{label}: sc14 FAIL touching test paths (could become a pass)", True
    if slot["model"] == TEACHER and slot["status"] == "pass" and slot.get("cheated") is True:
        return "kimi PASS judged cheated: already excluded, a flip cannot change the clean rate", False
    if slot["model"] == TEACHER and slot["status"] == "pass":
        return f"{label}: kimi clean PASS touching test paths (could become a fail)", True
    if slot["status"] in ("pass", "fail"):
        return f"{slot['model']} {slot['status'].upper()}: a flip can only help the acceptance", False
    return f"{slot['status']} slot (not counted in the rates)", False


def plan_task(run: str, root: Path, round_dir: Path, scored: Path, metrics: dict, status: str, language: str,
              recovery=None) -> dict:
    task_id = metrics["task_id"]
    packaged = Path(metrics["task_dir"])
    accepted_copy = RUNS / run / "accepted" / task_id
    source = accepted_copy if status == "accepted" else packaged
    toml = task_builder.task_toml(source) if (source / "task.toml").exists() else {}
    snapshot = toml.get("environment", {}).get("docker_image")
    globs = task_globs(task_id, language, run)
    test_patch = (source / "tests/test.patch").read_text(errors="surrogateescape")
    solution = (source / "solution/solution.patch").read_text(errors="surrogateescape")
    problems = []
    if snapshot != metrics.get("snapshot"):
        problems.append(f"task.toml docker_image {snapshot} != metrics snapshot {metrics.get('snapshot')}")
    if status == "accepted":
        for name in TESTS_FILES + ("grader.py",):
            if file_sha(accepted_copy / "tests" / name) != file_sha(packaged / "tests" / name):
                problems.append(f"accepted/{task_id}/tests/{name} differs from the packaged {packaged}/tests/{name}")
    if file_sha(source / "tests/grader.py") != file_sha(OLD_GRADER):
        problems.append("tests/grader.py is not DeepSWE's grader")
    solution_hits = matching(NEW.touched_paths(solution), globs)
    if solution_hits:
        problems.append(f"reference solution.patch touches test paths {solution_hits}: narrow the globs "
                        f"(GLOB_OVERRIDES) before staging")
    probe, records = scored_probe(round_dir, task_id, scored)
    current_rollouts = scored / "rollouts"
    audit = read_json(scored / "teacher_audit.json") or {}
    test_reset = set(OLD.patch_paths(test_patch))  # exactly what the OLD grader resets before applying test.patch
    witnesses = matching(NEW.touched_paths(test_patch), globs)  # hidden test files: always in the raw reports
    config = json.loads((source / "tests/config.json").read_text())
    tree_check_ids = [i for i in config.get("p2p_node_ids", []) + config.get("f2p_node_ids", [])
                      if TREE_CHECK.search(str(i))]
    slots = []
    for record in records:
        md_slot, current = md_header(current_rollouts / f"{record['slot']}.md")
        if md_slot != record["slot"]:
            problems.append(f"{record['slot']}: no current rollouts/{record['slot']}.md header")
            current = record["status"]
        slots.append(slot_entry(record, current, test_reset, globs, audit.get(record["slot"]), language,
                                witnesses, bool(tree_check_ids)))
    now_rates = rates(slots)
    for key, field in (("sc14", "student_rate"), ("kimi_clean", "teacher_clean_rate")):
        if now_rates[key] != metrics.get(field):
            problems.append(f"recomputed {key} rate {now_rates[key]} != metrics {field} {metrics.get(field)}")
    affected = [x for x in slots if x["surviving"]]
    scheduled, listed = [], []
    entry = {"run": run, "task_id": task_id, "cand": scored.name, "round": round_dir.name, "status": status,
             "reason": metrics.get("reason"), "similarity": metrics.get("similarity"),
             "task_dir": str(source), "packaged_task_dir": str(packaged), "scored_dir": str(scored),
             "snapshot": snapshot, "language": language, "globs": globs, "probe": str(probe),
             "rates": now_rates, "metrics_rates": {"sc14": metrics.get("student_rate"),
                                                   "kimi_clean": metrics.get("teacher_clean_rate"),
                                                   "valid": metrics.get("valid")},
             "test_patch_paths": sorted(test_reset),
             "test_patch_unmatched": [p for p in NEW.touched_paths(test_patch) if p not in matching([p], globs)],
             "tree_check_ids": tree_check_ids[:10], "affected_slots": len(affected), "notes": []}
    start_patch = source / "environment/start.patch"
    rehomed = task_builder.rehomed_go_tests(start_patch.read_text(errors="surrogateescape"), test_patch) \
        if start_patch.exists() else []
    if rehomed:  # README "Broken test setups" (arcane c04): the clash came from model edits the sanitizer reverts
        entry["notes"].append(f"re-homed Go tests ({len(rehomed)}, e.g. {rehomed[0][2]} from {rehomed[0][0]} into "
                              f"{rehomed[0][1]}): a model test re-added under the same name in the original file "
                              "breaks the package under the old grader; the sanitizer restores that file")

    def put(slot, why, schedule):
        item = dict(slot, why=why)
        (scheduled if schedule else listed).append(item)

    if status in GUARDED:
        for x in affected:
            put(x, *guard_rule(x, "accepted" if status == "accepted" else "recovered in range"))
    else:  # rejected on rates: could flipping every test-touching slot in the helpful direction bring it into range?
        fav_s = [x for x in affected if x["model"] == STUDENT and x["status"] == "pass"]
        fav_t = [x for x in affected if x["model"] == TEACHER and x["status"] == "fail"]
        r = now_rates

        def best(s_flips, t_flips):
            return ((r["sc14_pass"] - len(s_flips)) / r["sc14_valid"] if r["sc14_valid"] else None,
                    (r["kimi_clean_pass"] + len(t_flips)) / r["kimi_valid"] if r["kimi_valid"] else None)

        best_s, best_t = best(fav_s, fav_t)
        likely_s, likely_t = best([x for x in fav_s if x["priority"] not in UNLIKELY],  # drop only flips that the
                                  [x for x in fav_t if x["priority"] not in UNLIKELY])  # raw reports argue against
        entry["best_case"] = {"sc14": best_s, "kimi_clean": best_t, "sc14_flips": len(fav_s),
                              "kimi_flips": len(fav_t), "sc14_likely": likely_s, "kimi_clean_likely": likely_t,
                              "in_range_with_likely_flips_only": in_range(likely_s, likely_t)}
        entry["status"] = "near_bar" if (fav_s or fav_t) and in_range(best_s, best_t) else "rates_out_of_reach"
        if entry["status"] == "near_bar":
            for x in fav_s:
                put(x, "near bar: sc14 PASS touching test paths (could become a fail)", True)
            for x in fav_t:
                put(x, "near bar: kimi FAIL touching test paths (could become a pass)", True)
            for x in affected:
                if x in fav_s or x in fav_t:
                    continue
                if x["status"] in ("pass", "fail") and not (x["model"] == TEACHER and x.get("cheated")):
                    put(x, f"near bar, adverse: {x['model']} {x['status'].upper()} touching test paths could flip "
                           "against the acceptance; regrade it too before accepting", False)
                else:
                    put(x, f"{x['status']} slot{' (cheated pass)' if x.get('cheated') else ''}: cannot change the "
                           "rates", False)
    if recovery:  # recovery_in_range: tools/recover_slots.py made this inconclusive task valid and in range
        entry["recovery"] = recovery
        planned, pending = sorted(x["slot"] for x in scheduled), sorted(recovery["pending_repair"])
        entry["notes"].append(
            f"recovered_in_range: {recovery['path']} recommends newly_in_range ({recovery['decision_view']}: "
            f"{recovery['rates_after']}); scheduled {planned} (its pending_repair {pending}"
            f"{'' if planned == pending else ': DIFFERENT'}). The rates above are score()'s inconclusive ones: "
            "re-score from the recovery record with this repair's statuses (`tools/recover_slots.py record --run R "
            "--task T` reads audit/grading_repair/<T>.json statuses_after); install waits for main's acceptance")
    entry.update(scheduled=scheduled, listed=listed, problems=problems)
    return entry


def solution_scan(run: str, root: Path, language: str) -> list:
    """Every BUILT task of the run (cand-*/stage.json ok): reference solution.patch paths matching its globs."""
    hits = []
    for stage_path in sorted(root.glob("cand-*/stage.json")):
        stage = read_json(stage_path) or {}
        if not stage.get("ok"):
            continue
        solution = Path(stage["task_dir"]) / "solution/solution.patch"
        if not solution.exists():
            continue
        paths = matching(NEW.touched_paths(solution.read_text(errors="surrogateescape")),
                         task_globs(stage["task_id"], language, run))
        if paths:
            hits.append({"task_id": stage["task_id"], "paths": paths})
    return hits


def plan_run(run: str) -> tuple:
    root = build_root(run)
    target, repo = root.parent.name, (read_json(root / "summary.json") or {}).get("repo") \
        or root.name.replace("__", "/", 1)
    language = repo_language(target, repo)
    if language not in GLOBS:
        raise RepairError(f"{run}: no default globs for {language}")
    accepted, withdrawn = set(accepted_ids(run)), withdrawn_ids(run)
    tasks, skipped, counts = [], [], {}
    scored = scored_tasks(root)
    files = {}
    for round_dir, scored_dir, metrics in scored:
        task_dir = Path(metrics["task_dir"])
        if (task_dir / "tests/test.patch").exists():
            family = files_language((task_dir / "tests/test.patch").read_text(errors="surrogateescape"),
                                    (task_dir / "solution/solution.patch").read_text(errors="surrogateescape"))
            files[family] = files.get(family, 0) + 1
        status, why = classify(metrics, scored_dir, accepted, withdrawn)
        recovery = recovery_in_range(run, metrics["task_id"]) if status == "skip_inconclusive" else None
        if recovery:
            status = "recovered_in_range"
        graded = skip = None
        if status in ("accepted", "rates", "recovered_in_range"):  # graded by a sanitizer already? (batch 3e)
            probe, _ = scored_probe(round_dir, metrics["task_id"], scored_dir)
            graded = probe_grader(probe, task_dir)
            skip = sanitized_skip(metrics, scored_dir, status, graded, task_globs(metrics["task_id"], language, run),
                                  task_mode(language))
        if skip:
            counts[skip["status"]] = counts.get(skip["status"], 0) + 1
            skipped.append(skip)
        elif status in ("accepted", "rates", "recovered_in_range"):
            entry = plan_task(run, root, round_dir, scored_dir, metrics, status, language, recovery=recovery)
            if graded["kind"] != "deepswe":  # the repair's premise: DeepSWE's grader graded the recorded slots
                entry["problems"].append(f"its probe graded with an unknown grader {graded['grader_sha256']} "
                                         f"({graded['tests']}), neither DeepSWE's nor a known sanitizer")
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
            if entry["status"] == "rates_out_of_reach":
                skipped.append({"task_id": metrics["task_id"], "cand": scored_dir.name, "status": entry["status"],
                                "why": f"best case sc14 {entry['best_case']['sc14']}, clean kimi "
                                       f"{entry['best_case']['kimi_clean']} ({metrics.get('reason')})",
                                "affected_slots": entry["affected_slots"]})
            else:
                tasks.append(entry)
        else:
            counts[status] = counts.get(status, 0) + 1
            skipped.append({"task_id": metrics["task_id"], "cand": scored_dir.name, "status": status, "why": why})
    sanitized = [s for s in skipped if s["status"] == "skip_sanitized"]
    missing = sorted(accepted - {t["task_id"] for t in tasks if t["status"] == "accepted"}
                     - {s["task_id"] for s in sanitized if s["decision"] == "accepted"})
    problems = [f"accepted/tasks.jsonl rows without a scored accepted task: {missing}"] if missing else []
    problems += [f"{s['task_id']}: its probe's sanitizer had test_paths {s['probe_grader']['test_paths']}, mode "
                 f"{s['probe_grader']['test_paths_mode']}, not the run's {task_globs(s['task_id'], language, run)}, "
                 f"{task_mode(language)}" for s in sanitized if not s["matches_run_globs"]]
    dominant = max(files, key=files.get) if files else None
    if dominant and dominant != FAMILY[language]:
        problems.append(f"top100 language {language} but the tasks' files are mostly {dominant}")
    summary = {"build_root": str(root), "target": target, "repo": repo, "language": language,
               "files_language": files, "globs": task_globs(None, language, run), "counts": counts,
               "glob_gaps": glob_gaps(root, task_globs(None, language, run), language),
               "accepted_tasks": sum(t["status"] == "accepted" for t in tasks),
               "near_bar_rejected": sum(t["status"] == "near_bar" for t in tasks),
               "slots_to_regrade": sum(len(t["scheduled"]) for t in tasks),
               "slots_to_regrade_by_priority": {p: sum(s["priority"] == p for t in tasks for s in t["scheduled"])
                                                for p in PRIORITIES},
               "slots_to_regrade_by_status": {status: sum(len(t["scheduled"]) for t in tasks if t["status"] == status)
                                              for status in ("accepted", "near_bar")},
               "near_bar_in_range_with_likely_flips": sum(t["status"] == "near_bar" and
                                                          t["best_case"]["in_range_with_likely_flips_only"]
                                                          for t in tasks),
               "listed_slots": sum(len(t["listed"]) for t in tasks),
               "solution_glob_hits": solution_scan(run, root, language), "problems": problems}
    recovered = [t for t in tasks if t["status"] == "recovered_in_range"]
    if recovered:  # only then: the summaries of runs without such a task keep their keys
        summary["recovered_in_range"] = len(recovered)
        summary["slots_to_regrade_by_status"]["recovered_in_range"] = sum(len(t["scheduled"]) for t in recovered)
    if sanitized:  # only then (batch 3e): the summaries of runs without such a task keep their keys
        summary["sanitized_tasks"] = len(sanitized)
        summary["accepted_sanitized"] = sorted(s["task_id"] for s in sanitized if s["decision"] == "accepted")
    return summary, tasks, skipped


def cmd_plan(args) -> int:
    runs = [check_run(r) for r in (args.run or RUN_ORDER)]
    with locked("plan", args.dry_run):
        plan = read_json(PLAN) if args.run and PLAN.exists() else None
        out = {"runs": {}, "tasks": [], "skipped": []} if not plan else plan
        for run in runs:
            print(f"== {run}")
            summary, tasks, skipped = plan_run(run)
            out["runs"][run] = summary
            out["tasks"] = [t for t in out["tasks"] if t["run"] != run] + tasks
            out["skipped"] = [s for s in out["skipped"] if s.get("run") != run] + [dict(s, run=run) for s in skipped]
            print(f"  {summary['language']} globs {summary['globs']}; accepted {summary['accepted_tasks']}, near bar "
                  f"{summary['near_bar_rejected']} ({summary['near_bar_in_range_with_likely_flips']} in range with "
                  f"likely flips only), slots to regrade {summary['slots_to_regrade']} (by priority "
                  f"{summary['slots_to_regrade_by_priority']}; by status {summary['slots_to_regrade_by_status']}), "
                  f"listed {summary['listed_slots']}; counts {summary['counts']}")
            if summary.get("recovered_in_range"):
                print(f"  recovered in range (tools/recover_slots.py): {summary['recovered_in_range']} task(s), slots "
                      f"to regrade {summary['slots_to_regrade_by_status']['recovered_in_range']}")
            if summary.get("sanitized_tasks"):
                print(f"  graded by a sanitizer in their probes (skip_sanitized, nothing to repair): "
                      f"{summary['sanitized_tasks']} task(s); accepted among them: {summary['accepted_sanitized']}")
            for note in [f"{t['task_id']}: {n}" for t in tasks for n in t["notes"]]:
                print(f"  note: {note}")
            gaps = summary["glob_gaps"]
            if gaps["slot_patches"]:
                print(f"  glob gaps: {gaps['slot_patches']} slot patch(es) touch unmatched paths under test dirs, e.g. "
                      + ", ".join(f"{d['dir']} ({d['paths']}{', reference solution touches it' if d['reference_solution_touches'] else ''})"
                                  for d in gaps["dirs"][:4]))
            for hit in summary["solution_glob_hits"]:
                print(f"  SOLUTION TOUCHES TEST PATHS: {hit['task_id']}: {hit['paths']}")
            for problem in summary["problems"] + [f"{t['task_id']}: {p}" for t in tasks for p in t["problems"]]:
                print(f"  problem: {problem}")
        out["tasks"].sort(key=lambda t: (run_rank(t["run"]), t["run"], t["status"] != "accepted", t["task_id"]))
        out.update(created_at=now(), tool="tools/grading_repair.py plan",
                   grader={"path": str(NEW_GRADER), "sha256": file_sha(NEW_GRADER)},
                   old_grader={"path": str(OLD_GRADER), "sha256": file_sha(OLD_GRADER)},
                   suppressed_probe_cache_writes=sorted(set(SUPPRESSED)),
                   rules={
                       "slot": "the final graded attempt of every probe slot (rollouts.probe_records = what score() "
                               "used); status = the current rollouts/<slot>.md header (re-scored slots included)",
                       "test_paths": "touched_paths(model.patch) (old and new names) matching the task's globs",
                       "surviving": "test paths outside patch_paths(test.patch), i.e. what the OLD grader left in "
                                    "the graded run; only such slots can grade differently",
                       "accepted": "schedule sc14 FAIL and clean kimi PASS slots with surviving test paths; list "
                                   "the others",
                       "recovered_in_range": "an inconclusive task whose tools/recover_slots.py record recommends "
                                             "newly_in_range (recovery_in_range): scheduled like an accepted task",
                       "near_bar": "a task rejected on rates whose best case (every sc14 PASS with surviving test "
                                   "paths -> FAIL, every kimi FAIL with them -> PASS, cheated passes still "
                                   "excluded) is sc14 <= 0.2 and clean kimi >= 0.6; schedule those slots, list the "
                                   "adverse ones",
                       "skipped": "rejected on similarity, invalid (task_builder.seed_check), inconclusive "
                                  "(unless recovered_in_range), withdrawn; skip_sanitized: a task whose probe "
                                  "graded every slot with a sanitizing grader already (probe_grader)",
                       "priority": "an ordering hint only (every scheduled slot follows the rules above): "
                                   "existing > added_loaded > added_collected > added_tree_checked > added_unknown > "
                                   "added_uncollected (slot_priority); best_case.*_likely leaves out only "
                                   "added_uncollected flips"})
        for run in runs:
            summary = out["runs"][run]
            summary["accepted_tasks"] = sum(t["status"] == "accepted" for t in out["tasks"] if t["run"] == run)
        if args.dry_run:
            print(f"[dry run] would write {PLAN}")
        else:
            entry = write_json_atomic(PLAN, out)
            log_action({"action": "plan", "runs": runs, **entry})
            print(f"wrote {PLAN}")
    total = sum(s["slots_to_regrade"] for s in out["runs"].values())
    print(f"slots to regrade: {total}; suppressed probe cache writes: {len(set(SUPPRESSED))}")
    return 0


# ============================================================================================ stage

def load_plan() -> dict:
    plan = read_json(PLAN)
    if not isinstance(plan, dict) or "tasks" not in plan:
        raise RepairError(f"no plan: run `plan` first ({PLAN})")
    return plan


def plan_entries(plan: dict, run: str, tasks, planned: bool) -> list:
    mine = [t for t in plan["tasks"] if t["run"] == run]
    if planned:
        return mine
    entries = []
    for task_id in tasks or []:
        entry = next((t for t in mine if t["task_id"] == task_id), None)
        if entry is None:
            raise RepairError(f"{task_id} is not an accepted or near-bar task of {run} in {PLAN}")
        entries.append(entry)
    if not entries:
        raise RepairError("name tasks with --task, or --planned for every task of the run in plan.json")
    return entries


def task_home(run: str, task_id: str) -> Path:
    return REPAIR / TASKS_DIR / run / task_id


def staged_dirs(run: str, task_id: str) -> tuple:
    home = task_home(run, task_id)
    return home / "new" / task_id, home / "old" / task_id


def task_mode(language: str) -> str:
    """config.json "test_paths_mode" for a repository language (top100.jsonl, as for the globs): MODES, else "all"."""
    return MODES.get(language, "all")


def sanitized_config(config_text: str, globs: list, mode: str) -> str:
    """config.json with the top-level "test_paths" and "test_paths_mode" keys, in write_json's format (indent 2,
    sorted keys)."""
    config = json.loads(config_text)
    if "test_paths" in config and config["test_paths"] != globs:
        raise RepairError(f"config.json already has other test_paths {config['test_paths']}")
    if config.get("test_paths_mode", mode) != mode:
        raise RepairError(f"config.json already has test_paths_mode {config['test_paths_mode']!r}, not {mode!r}")
    return json.dumps(dict(config, test_paths=list(globs), test_paths_mode=mode), indent=2, sort_keys=True) + "\n"


def without_test_paths(config_text: str) -> dict:
    """config.json without the sanitizer's keys (what the packaged task had)."""
    config = json.loads(config_text)
    config.pop("test_paths", None)
    config.pop("test_paths_mode", None)
    return config


def grader_kind(sha) -> str:
    """deepswe | prior (an earlier sanitizer, PRIOR_GRADERS) | current (verifier/grader.py) | None (unknown)."""
    if sha == file_sha(OLD_GRADER):
        return "deepswe"
    if sha in PRIOR_GRADERS:
        return "prior"
    return "current" if sha == file_sha(NEW_GRADER) else None


def old_control_files(entry: dict, source: Path, installed: str) -> dict:
    """{name: bytes} of the OLD-grader control copy: DeepSWE's grader with the task's own test.sh, test.patch and
    config.json. When the source already holds a sanitizer (an accepted task installed earlier), config.json comes
    from the packaged task, which must equal the source's config without the sanitizer keys."""
    files = {name: (source / "tests" / name).read_bytes() for name in ("test.sh", "test.patch")}
    files["grader.py"] = OLD_GRADER.read_bytes()
    if installed == "deepswe":
        files["config.json"] = (source / "tests/config.json").read_bytes()
        return files
    packaged = Path(entry.get("packaged_task_dir") or "/nonexistent") / "tests"
    if not (packaged / "config.json").exists() or file_sha(packaged / "grader.py") != file_sha(OLD_GRADER):
        raise RepairError(f"{entry['task_id']}: no packaged task with DeepSWE's grader for the old control "
                          f"({packaged})")
    original = (packaged / "config.json").read_bytes()
    if json.loads(original) != without_test_paths((source / "tests/config.json").read_text()):
        raise RepairError(f"{entry['task_id']}: {source}/tests/config.json differs from the packaged one beyond "
                          "the sanitizer keys")
    for name in ("test.sh", "test.patch"):
        if file_sha(packaged / name) != file_sha(source / "tests" / name):
            raise RepairError(f"{entry['task_id']}: {source}/tests/{name} differs from the packaged {name}")
    files["config.json"] = original
    return files


def source_shas(source: Path) -> dict:
    names = [f"tests/{n}" for n in TESTS_FILES + ("grader.py",)] + ["solution/solution.patch", "task.toml",
                                                                     "instruction.md"]
    return {name: file_sha(source / name) for name in names}


def stage_task(run: str, entry: dict, *, restage=False, dry_run=False) -> dict:
    task_id, source, globs = entry["task_id"], Path(entry["task_dir"]), entry["globs"]
    mode = task_mode(entry["language"])
    home = task_home(run, task_id)
    new_dir, old_dir = staged_dirs(run, task_id)
    if entry["problems"]:
        raise RepairError(f"{task_id}: the plan lists problems: {entry['problems']}")
    installed = grader_kind(file_sha(source / "tests/grader.py"))
    if installed is None:
        raise RepairError(f"{source}/tests/grader.py is neither DeepSWE's grader nor a known sanitizer")
    hits = matching(NEW.touched_paths((source / "solution/solution.patch").read_text(errors="surrogateescape")),
                    globs)
    if hits:
        raise RepairError(f"{task_id}: the reference solution touches test paths {hits}: narrow its globs")
    config_text = sanitized_config((source / "tests/config.json").read_text(), globs, mode)
    old_files = old_control_files(entry, source, installed)
    record = {"run": run, "task_id": task_id, "status": entry["status"], "source": str(source),
              "source_sha256": source_shas(source), "source_grader": installed, "snapshot": entry["snapshot"],
              "test_paths": globs, "test_paths_mode": mode, "new_dir": str(new_dir), "old_dir": str(old_dir),
              "grader_sha256": file_sha(NEW_GRADER), "config_sha256": sha256(config_text),
              "old_sha256": {name: sha256(data) for name, data in old_files.items()}}
    previous = read_json(home / "stage.json")
    keys = ("source_sha256", "grader_sha256", "config_sha256", "test_paths", "test_paths_mode", "old_sha256")
    if previous and new_dir.exists() and not restage:
        same = {k: previous.get(k) for k in keys} == {k: record[k] for k in keys}
        if same and staged_ok(run, task_id, previous):
            return dict(previous, unchanged=True)
        raise RepairError(f"{task_id}: staged earlier from other inputs; --restage to replace {home / 'new'}")
    if dry_run:
        return dict(record, dry_run=True)
    home.mkdir(parents=True, exist_ok=True)
    built = {}
    for name in ("new", "old"):
        tmp = home / f".{name}.{uuid.uuid4().hex}.tmp"
        target = tmp / task_id
        if name == "new":
            shutil.copytree(source, target, symlinks=True, ignore=shutil.ignore_patterns(
                "*" + BACKUP, "*" + BACKUP_V2, "*" + BACKUP_V3, "*.pre-early-exit-fix", "*.pre-gate-fix",
                "*.pre-regrade", "__pycache__"))
            shutil.copyfile(NEW_GRADER, target / "tests/grader.py")
            (target / "tests/config.json").write_text(config_text)
            task_builder.validate_task(target)  # Gym's DeepSWE loader (both Gym copies) accepts the scratch copy
        else:
            (target / "tests").mkdir(parents=True)
            for file_name, data in old_files.items():
                (target / "tests" / file_name).write_bytes(data)
        built[name] = tmp
    for name, tmp in built.items():
        final = home / name
        if final.exists():
            kept = final.with_name(final.name + BACKUP)
            if kept.exists():
                shutil.rmtree(final)
            else:
                final.rename(kept)
        tmp.rename(final)
    record["at"] = now()
    write_json_atomic(home / "stage.json", record)
    log_action({"action": "stage", "run": run, "task_id": task_id, "dir": str(home), "test_paths": globs,
                "test_paths_mode": mode, "source_grader": installed, "grader_sha256": record["grader_sha256"],
                "config_sha256": record["config_sha256"]})
    return record


def staged_ok(run: str, task_id: str, stage: dict) -> bool:
    """The scratch copies still hold exactly what stage wrote (nothing edited by hand since)."""
    new_dir, old_dir = staged_dirs(run, task_id)
    source = stage["source_sha256"]
    old = stage.get("old_sha256") or {name: source[f"tests/{name}"] for name in TESTS_FILES + ("grader.py",)}
    return (file_sha(new_dir / "tests/grader.py") == stage["grader_sha256"] == file_sha(NEW_GRADER)
            and file_sha(new_dir / "tests/config.json") == stage["config_sha256"]
            and all(file_sha(new_dir / name) == sha for name, sha in source.items()
                    if name not in ("tests/grader.py", "tests/config.json"))
            and sorted(old) == sorted(TESTS_FILES + ("grader.py",))
            and all(file_sha(old_dir / "tests" / name) == sha for name, sha in old.items()))


def require_stage(run: str, task_id: str) -> dict:
    stage = read_json(task_home(run, task_id) / "stage.json")
    if not stage:
        raise RepairError(f"{task_id}: not staged (run stage first)")
    if not staged_ok(run, task_id, stage):
        raise RepairError(f"{task_id}: the scratch copy changed since stage (or the grader did): stage --restage")
    return stage


def cmd_stage(args) -> int:
    run = check_run(args.run)
    status = 0
    with locked(run, args.dry_run):
        for entry in plan_entries(load_plan(), run, args.task, args.planned):
            try:
                record = stage_task(run, entry, restage=args.restage, dry_run=args.dry_run)
            except (RepairError, OSError) as error:
                print(f"{entry['task_id']}: NOT STAGED: {error}")
                status = 1
                continue
            word = "unchanged" if record.get("unchanged") else "would stage" if record.get("dry_run") else "staged"
            print(f"{entry['task_id']}: {word} ({entry['status']}) {record['new_dir']} test_paths {record['test_paths']}"
                  f" mode {record.get('test_paths_mode')}")
    return status


# ============================================================================================ grading

SANITIZE_LINE = re.compile(re.escape(MARK) + r' (restored|deleted|absent|kept-additive|dangling-import|'
                           r'dropped-manifest-target|restored-manifest-target) ("(?:[^"\\]|\\.)*")')


def sanitize_marks(out_dir: Path) -> dict:
    log_text = (out_dir / "output.log").read_text(errors="replace") if (out_dir / "output.log").exists() else ""
    marks = [line.strip() for line in log_text.splitlines() if MARK in line and not line.lstrip().startswith("+")]
    actions = {}
    for line in marks:
        match = SANITIZE_LINE.search(line)
        if match:
            actions.setdefault(match.group(1), []).append(json.loads(match.group(2)))
    return {"lines": marks[:50], "actions": actions, "applied": "model.patch applied" in log_text}


def grade_summary(result: dict, out_dir: Path, config: dict) -> dict:
    reward_json = result.get("reward_json")
    summary = {"reward": result.get("reward"), "reward_json": reward_json,
               "reward_json_present": isinstance(reward_json, dict),
               "apply_failed": bool((reward_json or {}).get("apply_failed")), "exit_code": result.get("exit_code"),
               "timed_out": result.get("timed_out"), "out_dir": str(out_dir),
               "failing_tests": regrade.failing_tests(result.get("suites") or {}, config),
               "sanitize": sanitize_marks(out_dir)}
    if isinstance(reward_json, dict) and result.get("reward") is not None:
        summary.update(status="pass" if result["reward"] == 1 else "fail",
                       f2p=f"{reward_json.get('f2p_passed')}/{reward_json.get('f2p_total')}",
                       p2p=f"{reward_json.get('p2p_passed')}/{reward_json.get('p2p_total')}")
    else:
        summary.update(status="infra")
    return summary


def tamper_patch(language: str) -> tuple:
    """A model-added file at a test path (it would break the suites if it survived); (path, patch)."""
    path, body = {
        "Python": ("tests/test_zz_sanitizer_probe.py",
                   ['raise RuntimeError("test-sanitize probe: this model-added test file must not be graded")']),
        "Go": ("zz_sanitizer_probe_test.go", ["package zz_sanitizer_probe", "", "this is not Go: the sanitizer "
                                              "must delete this model-added test file"]),
        "Rust": ("tests/zz_sanitizer_probe.rs", ['compile_error!("test-sanitize probe");']),
    }.get(language, ("zz-sanitizer-probe.test.js",
                     ['throw new Error("test-sanitize probe: this model-added test file must not be graded");']))
    lines = "".join(f"+{line}\n" for line in body)
    return path, (f"diff --git a/{path} b/{path}\nnew file mode 100644\nindex 0000000..1111111\n--- /dev/null\n"
                  f"+++ b/{path}\n@@ -0,0 +1,{len(body)} @@\n{lines}")


async def validate_task(run: str, entry: dict, gate, *, tamper=False, dry_run=False) -> dict:
    task_id = entry["task_id"]
    stage = require_stage(run, task_id)
    new_dir, _ = staged_dirs(run, task_id)
    snapshot = task_builder.task_toml(new_dir)["environment"]["docker_image"]
    if snapshot != entry["snapshot"]:
        raise RepairError(f"{task_id}: task.toml docker_image {snapshot} != plan snapshot {entry['snapshot']}")
    solution = (new_dir / "solution/solution.patch").read_text()
    patches = {"solution": solution, "empty": ""}
    probe_path = None
    if tamper:
        probe_path, extra = tamper_patch(entry["language"])
        if not matching([probe_path], entry["globs"]):
            raise RepairError(f"tamper path {probe_path} matches none of {entry['globs']}")
        patches["tamper"] = solution + ("" if solution.endswith("\n") else "\n") + extra
    out = task_home(run, task_id) / "validate"
    if dry_run:
        return {"task_id": task_id, "dry_run": True, "snapshot": snapshot, "grades": list(patches), "out": str(out)}
    config = json.loads((new_dir / "tests/config.json").read_text())

    async def one(name):
        async with gate:
            shutil.rmtree(out / name, ignore_errors=True)  # this tool's own output of an earlier validate
            return await sandbox.grade(snapshot, new_dir, patches[name], run=run, out_dir=out / name)

    names = list(patches)
    results = await asyncio.gather(*(one(name) for name in names), return_exceptions=True)
    graded = {}
    for name, result in zip(names, results):
        graded[name] = {"error": f"{type(result).__name__}: {str(result)[-500:]}"} \
            if isinstance(result, BaseException) else grade_summary(result, out / name, config)
    sol, emp = graded["solution"], graded["empty"]
    checks = {"solution_scores_1": sol.get("reward_json_present") is True and sol.get("reward") == 1
              and sol["sanitize"]["applied"] and any("0 path(s) of model.patch match" in line
                                                     for line in sol["sanitize"]["lines"]),
              "empty_scores_0": emp.get("reward_json_present") is True and emp.get("reward") == 0}
    mode = stage.get("test_paths_mode", "all")
    if mode != "all":  # the grader in the sandbox read the mode (it logs it whenever the sanitizer runs)
        checks["mode_logged"] = any(f"test_paths_mode {mode}:" in line
                                    for line in (sol.get("sanitize") or {}).get("lines", []))
    if tamper:
        tam = graded["tamper"]
        checks["tamper_deleted_and_scores_1"] = tam.get("reward") == 1 and \
            probe_path in (tam.get("sanitize") or {}).get("actions", {}).get("deleted", [])
    record = {"run": run, "task_id": task_id, "snapshot": snapshot, "status": entry["status"],
              "grader_sha256": file_sha(new_dir / "tests/grader.py"),
              "config_sha256": file_sha(new_dir / "tests/config.json"),
              "tests_sha256": {name: file_sha(new_dir / "tests" / name) for name in TESTS_FILES},
              "test_paths": stage["test_paths"], "test_paths_mode": mode, "results": graded, "checks": checks,
              "all_ok": all(checks.values()), "at": now()}
    write_json_atomic(task_home(run, task_id) / "validate.json", record)
    log_action({"action": "validate", "run": run, "task_id": task_id, "test_paths_mode": mode, "checks": checks,
                "all_ok": record["all_ok"], "path": str(task_home(run, task_id) / "validate.json")})
    return record


def gate_for(args):
    if not 1 <= args.concurrency <= MAX_SANDBOXES:
        raise SystemExit(f"--concurrency must be 1..{MAX_SANDBOXES} (sandbox budget per agent)")
    return asyncio.Semaphore(args.concurrency)


async def _validate(args) -> int:
    run = check_run(args.run)
    gate = gate_for(args)
    with locked(run, args.dry_run):
        entries = plan_entries(load_plan(), run, args.task, args.planned)
        results = await asyncio.gather(*(validate_task(run, e, gate, tamper=args.tamper, dry_run=args.dry_run)
                                         for e in entries), return_exceptions=True)
    status = 0
    for entry, result in zip(entries, results):
        if isinstance(result, BaseException):
            print(f"{entry['task_id']}: NOT VALIDATED: {type(result).__name__}: {str(result)[-600:]}")
            status = 1
            continue
        if result.get("dry_run"):
            print(f"{entry['task_id']}: would grade {result['grades']} on {result['snapshot']} -> {result['out']}")
            continue
        print(f"== {entry['task_id']} ({entry['status']}) on {result['snapshot']}")
        for name, graded in result["results"].items():
            reward = graded.get("reward_json") or {}
            print(f"  {name:8s} reward {graded.get('reward')!s:5s} F2P {reward.get('f2p_passed')}/{reward.get('f2p_total')} "
                  f"P2P {reward.get('p2p_passed')}/{reward.get('p2p_total')} exit {graded.get('exit_code')}"
                  f"{' ERROR ' + graded['error'] if 'error' in graded else ''}")
            for line in (graded.get("sanitize") or {}).get("lines", [])[:4]:
                print(f"           {line[:160]}")
        print(f"  checks {result['checks']} -> {'OK' if result['all_ok'] else 'FAILED'}")
        status |= 0 if result["all_ok"] else 1
    return status


def task_slots(run: str, entry: dict, wanted, scheduled: bool) -> list:
    """[(slot, record, plan slot entry or None)] from the scored probe (the final graded attempts)."""
    probe = Path(entry["probe"])
    records = probe_records(probe).get(entry["task_id"]) or []
    by_slot = {r["slot"]: r for r in records}
    planned = {s["slot"]: s for s in entry["scheduled"] + entry["listed"]}
    if scheduled:
        names = [s["slot"] for s in entry["scheduled"]]
    else:
        names = [part.strip() for item in wanted or [] for part in item.split(",") if part.strip()]
        bad = [name for name in names if not re.fullmatch(r"[A-Za-z0-9_.]+-r\d+", name)]
        if bad:
            raise RepairError(f"slots {bad}: use <model>-r<repeat>, e.g. sc14-r3")
    out = []
    for name in names:
        record = by_slot.get(name)
        if record is None:
            raise RepairError(f"{name}: not a slot of {entry['task_id']} in {probe}")
        plan_slot = planned.get(name)
        if plan_slot and plan_slot["patch_sha256"] != sha256(record["patch"]):
            raise RepairError(f"{name}: the probe's patch changed since the plan")
        out.append((name, record, plan_slot))
    return out


async def regrade_task(run: str, entry: dict, slots: list, graders: list, gate, *, dry_run=False) -> dict:
    task_id = entry["task_id"]
    stage = require_stage(run, task_id)
    new_dir, old_dir = staged_dirs(run, task_id)
    snapshot = task_builder.task_toml(new_dir)["environment"]["docker_image"]
    home = task_home(run, task_id)
    if dry_run:
        return {"task_id": task_id, "dry_run": True, "snapshot": snapshot,
                "jobs": [f"{slot}/{grader}" for slot, _, _ in slots for grader in graders]}
    config = json.loads((new_dir / "tests/config.json").read_text())
    current = {slot: md_header(Path(entry["scored_dir"]) / "rollouts" / f"{slot}.md")[1] or record["status"]
               for slot, record, _ in slots}  # the status score() used last (re-scored slots included)

    async def one(slot, record, grader):
        async with gate:
            out_dir = home / "regrade" / slot / grader
            shutil.rmtree(out_dir, ignore_errors=True)  # this tool's own earlier attempt
            return await sandbox.grade(snapshot, new_dir if grader == "new" else old_dir, record["patch"], run=run,
                                       out_dir=out_dir)

    jobs = [(slot, record, grader) for slot, record, _ in slots for grader in graders]
    results = await asyncio.gather(*(one(*job) for job in jobs), return_exceptions=True)
    path = home / "regrade.json"
    state = read_json(path) or {"run": run, "task_id": task_id, "slots": {}}
    for (slot, record, grader), result in zip(jobs, results):
        item = state["slots"].setdefault(slot, {})
        item.update(model=record["model"], attempt_dir=record["attempt_dir"], patch_sha256=sha256(record["patch"]),
                    probe_status=record["status"], current_status=current.get(slot, record["status"]))
        if isinstance(result, BaseException):
            item[grader] = {"error": f"{type(result).__name__}: {str(result)[-500:]}", "at": now()}
            continue
        summary = grade_summary(result, home / "regrade" / slot / grader, config)
        summary.update(at=now(), grader_sha256=file_sha((new_dir if grader == "new" else old_dir) / "tests/grader.py"),
                       config_sha256=file_sha((new_dir if grader == "new" else old_dir) / "tests/config.json"))
        if grader == "new":  # the slot's record as score() would take it (rollouts._record's fields)
            info = {"tool": "tools/grading_repair.py", "grader_sha256": summary["grader_sha256"],
                    "out_dir": summary["out_dir"], "old_status": item["current_status"]}
            new = regrade.regraded_record(record, result, config, info)
            summary["record"] = {k: new.get(k) for k in ("status", "reward", "f2p", "p2p", "failing_tests",
                                                          "verifier_error")}
        item[grader] = summary
        item["changed"] = (item.get("new") or {}).get("status") not in (None, "infra", item["current_status"]) \
            if "new" in item else None
    state.update(snapshot=snapshot, stage_config_sha256=stage["config_sha256"],
                 test_paths_mode=stage.get("test_paths_mode", "all"), at=now())
    write_json_atomic(path, state)
    log_action({"action": "regrade", "run": run, "task_id": task_id, "path": str(path),
                "test_paths_mode": stage.get("test_paths_mode", "all"),
                "slots": {slot: {g: (state["slots"][slot].get(g) or {}).get("status") for g in graders}
                          for slot, _, _ in slots}})
    return state


async def _regrade(args) -> int:
    run = check_run(args.run)
    gate = gate_for(args)
    graders = ["new", "old"] if args.grader == "both" else [args.grader]
    with locked(run, args.dry_run):
        entries = plan_entries(load_plan(), run, args.task, args.planned)
        work = []
        for entry in entries:
            slots = task_slots(run, entry, args.slot, args.scheduled)
            if slots:
                work.append((entry, slots))
        results = await asyncio.gather(*(regrade_task(run, e, s, graders, gate, dry_run=args.dry_run)
                                         for e, s in work), return_exceptions=True)
    status = 0
    for (entry, slots), result in zip(work, results):
        if isinstance(result, BaseException):
            print(f"{entry['task_id']}: NOT REGRADED: {type(result).__name__}: {str(result)[-600:]}")
            status = 1
            continue
        if result.get("dry_run"):
            print(f"{entry['task_id']}: would grade {result['jobs']} on {result['snapshot']}")
            continue
        print(f"== {entry['task_id']} ({entry['status']})")
        for slot, _, _ in slots:
            item = result["slots"][slot]
            parts = []
            for grader in graders:
                graded = item.get(grader) or {}
                parts.append(f"{grader} {graded.get('status', 'ERROR ' + str(graded.get('error'))[:120])} "
                             f"(F2P {graded.get('f2p')}, P2P {graded.get('p2p')})")
            print(f"  {slot}: current {item['current_status']} | " + " | ".join(parts)
                  + (" CHANGED" if item.get("changed") else ""))
            for line in ((item.get("new") or {}).get("sanitize") or {}).get("lines", [])[:6]:
                print(f"           {line[:160]}")
            status |= 1 if any("error" in (item.get(g) or {}) or (item.get(g) or {}).get("status") == "infra"
                               for g in graders) else 0
    return status


# ============================================================================================ install

def install_task(run: str, entry: dict, *, dry_run=False) -> dict:
    task_id = entry["task_id"]
    if entry["status"] != "accepted":
        raise RepairError(f"{task_id} is {entry['status']}, not accepted: install only goes into accepted/")
    if task_id not in accepted_ids(run):
        raise RepairError(f"{task_id} is not in runs/{run}/accepted/tasks.jsonl")
    tests = RUNS / run / "accepted" / task_id / "tests"
    stage = require_stage(run, task_id)
    new_dir, _ = staged_dirs(run, task_id)
    validated = read_json(task_home(run, task_id) / "validate.json") or {}
    if not (validated.get("all_ok") and validated.get("grader_sha256") == file_sha(new_dir / "tests/grader.py")
            and validated.get("config_sha256") == file_sha(new_dir / "tests/config.json")):
        raise RepairError(f"{task_id}: no passing validate.json for the staged files (run validate)")
    grader_bytes, config_bytes = (new_dir / "tests/grader.py").read_bytes(), (new_dir / "tests/config.json").read_bytes()
    if file_sha(tests / "grader.py") == sha256(grader_bytes) and file_sha(tests / "config.json") == sha256(config_bytes):
        return {"task_id": task_id, "unchanged": True}
    for name in ("test.sh", "test.patch"):
        if file_sha(tests / name) != stage["source_sha256"][f"tests/{name}"]:
            raise RepairError(f"{task_id}: accepted tests/{name} changed since stage: stage --restage, validate")
    if without_test_paths((tests / "config.json").read_text()) != without_test_paths(config_bytes.decode()):
        raise RepairError(f"{task_id}: accepted tests/config.json differs from the staged one beyond test_paths")
    replaced = grader_kind(file_sha(tests / "grader.py"))
    if replaced is None:
        raise RepairError(f"{task_id}: accepted tests/grader.py is neither DeepSWE's, an earlier sanitizer nor the "
                          "new grader")
    suffix = BACKUP if replaced == "deepswe" else BACKUP_V3  # the first version replaced stays as <file><suffix>
    writes = [atomic_write(tests / "grader.py", grader_bytes, dry_run=dry_run, suffix=suffix),
              atomic_write(tests / "config.json", config_bytes, dry_run=dry_run, suffix=suffix)]
    log_action({"action": "install", "run": run, "task_id": task_id, "writes": writes, "replaced_grader": replaced,
                "test_paths": stage["test_paths"], "test_paths_mode": stage.get("test_paths_mode", "all"),
                "validate": str(task_home(run, task_id) / "validate.json")}, dry_run)
    return {"task_id": task_id, "writes": writes, "dry_run": dry_run}


def cmd_install(args) -> int:
    run = check_run(args.run)
    status = 0
    with locked(run, args.dry_run):
        for entry in plan_entries(load_plan(), run, args.task, False):
            try:
                result = install_task(run, entry, dry_run=args.dry_run)
            except (RepairError, OSError) as error:
                print(f"{entry['task_id']}: NOT INSTALLED: {error}")
                status = 1
                continue
            if result.get("unchanged"):
                print(f"{entry['task_id']}: already installed")
                continue
            for write in result["writes"]:
                print(f"{entry['task_id']}: {'would write' if args.dry_run else 'wrote'} {write['path']}"
                      f" (backup {write.get('backup')})")
    return status


# ============================================================================================ CLI

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("plan", help="read-only scan -> repair/plan.json")
    sub.add_argument("--run", nargs="+", help="in-scope run names (default: every in-scope run, RUN_ORDER; a "
                                              "FINISHED_LATER run must be finished)")
    sub.add_argument("--dry-run", action="store_true")

    def task_args(sub, planned=True):
        sub.add_argument("--run", required=True)
        sub.add_argument("--task", nargs="+", help="task ids (accepted or near-bar tasks in plan.json)")
        if planned:
            sub.add_argument("--planned", action="store_true", help="every task of the run in plan.json")
        sub.add_argument("--dry-run", action="store_true")

    sub = commands.add_parser("stage", help="scratch copy with the new grader and test_paths")
    task_args(sub)
    sub.add_argument("--restage", action="store_true", help="replace an earlier scratch copy (first one kept)")
    sub = commands.add_parser("validate", help="reference -> 1 and empty -> 0 through the scratch copy (sandbox)")
    task_args(sub)
    sub.add_argument("--tamper", action="store_true", help="also: reference + a model-added test-path file -> 1")
    sub.add_argument("--concurrency", type=int, default=MAX_SANDBOXES)
    sub = commands.add_parser("regrade", help="grade probe slots' model.patch on the scratch copy (sandbox)")
    task_args(sub)
    sub.add_argument("--slot", nargs="+", help="slot names, e.g. sc14-r3 kimi-r0 (the probe's final graded attempts)")
    sub.add_argument("--scheduled", action="store_true", help="the plan's scheduled slots of each task")
    sub.add_argument("--grader", choices=("new", "old", "both"), default="both")
    sub.add_argument("--concurrency", type=int, default=MAX_SANDBOXES)
    sub = commands.add_parser("install", help="new grader + test_paths into accepted/<task>/tests/ (validated)")
    task_args(sub, planned=False)
    args = parser.parse_args(argv)
    if args.command == "regrade" and bool(args.slot) == bool(args.scheduled):
        parser.error("regrade takes --slot ... or --scheduled")
    if args.command == "install" and not args.task:
        parser.error("install takes --task ...")
    try:
        if args.command in ("validate", "regrade"):
            return asyncio.run({"validate": _validate, "regrade": _regrade}[args.command](args))
        return {"plan": cmd_plan, "stage": cmd_stage, "install": cmd_install}[args.command](args)
    except RepairError as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
