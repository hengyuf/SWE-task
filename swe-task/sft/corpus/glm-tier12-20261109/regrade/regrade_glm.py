#!/usr/bin/env python3
"""GLM-5.3 tier-1/2 SFT corpus: re-grade passes on OLD-grader tasks with the CURRENT verifier (DESIGN.md "Old-grader
rule"; commitment: glm-rollouts-tasksall-20261108/README.md "Graders").

704 GLM-5.3 passes were graded by DeepSWE's old tests/grader.py (13,468 bytes, no test-path sanitizer), so the model's
edits to test files were in the graded tree. classes.json (make_classes.py) splits them into class A (no risky
test-path change) and class B (an existing test file modified / deleted outside test.patch, or a risky new file).
This tool grades them again with the current verifier: synth/verifier/grader.py plus config.json "test_paths" /
"test_paths_mode" configured as for new builds / repairs. Each grade runs on the snapshot GLM ran on, in a fresh
no-network sandbox (synth/sandbox.py grade(), Gym's DeepSWE verifier).

  plan      read-only -> plan.json. Selection:
              B             every class-B pass;
              A-sample      60 class-A passes: class-A keys sorted by sha256(key), the first 60 (checks the static
                            rule);
              A-glob-delta  class-A passes whose model.patch touches, outside test.patch, a path that only the task's
                            non-default globs match (grading_repair.RUN_GLOBS, task_builder.test_patch_globs extras).
                            classify.py saw the language globs only. Leave them out with --no-glob-delta.
            Per task: owner, task_dir, language, snapshot (task.toml docker_image == tasks.json image == each pass's
            terminal.json image), reference solution, test_paths / mode, source sha256s.
            Per pass: model.patch (sha256 == terminal.json model_patch_sha256, terminal reward 1), static test-path
            hits.
  stage     -> tasks/<run>/<task_id>/ + stage.json. grading_repair.stage_task's recipe: copy of the whole task dir
            (its backup patterns ignored); tests/grader.py = synth/verifier/grader.py; tests/config.json = the
            original + "test_paths" / "test_paths_mode" (grading_repair.sanitized_config). Gym's DeepSWE loader
            (task_builder.validate_task) must accept the copy. The original task dir is only read.
  validate  per staged task: reference solution -> reward 1 (sanitizer ran, "0 path(s)" matched), empty patch ->
            reward 0, test_paths_mode logged when it is not "all" (grading_repair.validate_task's checks)
            -> results.json "validation"; outputs in validate/<run>/<task_id>/{solution,empty}/.
  regrade   per planned pass: model.patch text on the staged copy -> results.json "results"; outputs in
            out/<safe key>/; "[verifier] test-sanitize:" lines parsed into sanitize_actions.
  status    counts from results.json (read-only).

test_paths = base + extras. base = grading_repair.task_globs: GLOB_OVERRIDES[task] or RUN_GLOBS[run] (official1-anko /
scriggo / sqlformatter / wasmi) or the language defaults (verifier/test_paths.py GLOBS == grading_repair.GLOBS).
extras = task_builder.test_patch_globs(test.patch, solution.patch, base), the directories new builds add.
test_paths_mode = grading_repair.task_mode(language) (== test_paths.config_keys; JavaScript / TypeScript
"non_additive", else "all"). language = the top100.jsonl language of the task's repository (test_paths.repo_language,
as touch_scan.py) and must equal classes.json's.

results.json (temp file + rename after every completed item): {"grader_sha256", "plan_sha256", "validation":
{task_key: {...}}, "results": {pass key: {"reward": 1|0|null, "status": "pass"|"fail"|"infra"|"validation_failed",
"sanitize_actions": [{"action", "path"[, "detail"]}], "exit_code", "timed_out", "out_dir", "graded_at", "class", ...}}}.
- Final (skipped on resume unless --redo): pass / fail graded with the current grader (and, while the task has a
  "staged" record, its config: a later stage failure never downgrades a final result); validation_failed while its
  task's validation is still failed. A task's validation is final when "ok" or "failed" (a later stage problem does
  not downgrade it either; the task's non-final passes then wait as "infra"). "stage_failed" is re-checked against
  stage.json on every validate; stage adopts an existing copy that is exactly what it would write.
- Durability: results.json / stage.json are written temp file + fsync + rename (+ directory fsync).
- infra: no reward.json, or a sandbox error, after MAX_INFRA_RETRIES retries (fresh sandbox each time). Also a pass
  whose task validation is infra or missing. Retried on the next run.
- A task whose validation fails (or which cannot be staged) marks its passes validation_failed; they are not graded.
- results.json is bound to one plan.json (plan_sha256) and one grader (grader_sha256). The tool refuses to mix them.
Task key "<run>/<task_id>"; pass key "<batch>/<campaign>/<task_id>/<slot>" (classes.json).

Sandbox credentials (per task owner, read from files by NAME, never printed). sandbox.owned() calls
Config.load(ATP_ENV). In this process sandbox.Config is replaced by a per-grade lookup (contextvar), so each grade
uses its task owner's credential:
  hengyuf: Config.load(common.ATP_ENV = S/secrets/opensandbox.env). Its OPENSANDBOX_DOMAIN / OPENSANDBOX_API_KEY /
           OPENSANDBOX_TLS_VERIFY must equal the ATP app's S/atp/.env (the CELL3_SANDBOX_API_KEY credential).
  baihe:   Config.load(glm-rollouts-sftv3-20261107/secrets/baihe-opensandbox.env) (BAIHE_SANDBOX_API_KEY as
           OPENSANDBOX_API_KEY). Same checks as glm-rollouts-tasksall-20261108/probe_worker.py: file mode 600, key
           present, DOMAIN / TLS_VERIFY equal to the ATP app's, key different from the ATP app's; protocol and
           use_server_proxy equal to the hengyuf config's.
OPENSANDBOX_* environment variables would override both files (Config.load merges them), so validate and regrade
refuse to run while any is set (variable NAMES are reported, values never). Exception text recorded in results.json
or printed passes through redact(), which cuts out both loaded API keys.
Sandbox metadata run = "glm-regrade-<SLURM_JOB_ID or login>" (<= 40 chars), purpose "grade" (sandbox.grade).
--concurrency (default 16) caps sandboxes per process; validate runs a task's two grades in parallel under the same cap.
One process at a time (flock on .regrade_glm.lock). SIGTERM / SIGINT cancel the run; sandbox.owned deletes in-flight
sandboxes.

Writes only under this directory (plan.json, stage.json, results.json, events.jsonl, tasks/, validate/, out/). The
transient upload / download files of synth/sandbox.py go to its usual private scratch (common.scratch()). Never
touches task dirs, rollout dirs, shared synth / ATP code or secrets. Makes no hub or LLM call.
Run with the synth pipeline interpreter from a private cwd:
  /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp/runtime/controller/bin/python3 -I -B -X utf8 \
      <this dir>/regrade_glm.py plan|stage|validate|regrade|status [options]
(regrade.sbatch runs plan (if no plan.json), stage, validate, regrade and status in one CPU job.)
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import asyncio  # noqa: E402
import collections  # noqa: E402
import contextlib  # noqa: E402
import contextvars  # noqa: E402
import datetime  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import locale  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import time  # noqa: E402
import tomllib  # noqa: E402
import uuid  # noqa: E402

S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
SYNTH = S / "synth"
REGRADE = S / "sft/corpus/glm-tier12-20261109/regrade"
CLASSES = REGRADE / "classes.json"
PLAN = REGRADE / "plan.json"
STAGE = REGRADE / "stage.json"
RESULTS = REGRADE / "results.json"
EVENTS = REGRADE / "events.jsonl"
LOCK = REGRADE / ".regrade_glm.lock"
TASKS = REGRADE / "tasks"            # staged copies: tasks/<run>/<task_id>/
VALIDATE_OUT = REGRADE / "validate"  # validate/<run>/<task_id>/{solution,empty}/
OUT = REGRADE / "out"                # out/<safe key>/
GR2 = S / "sft/glm-rollouts-tasksall-20261108"
GR = S / "sft/glm-rollouts-sftv3-20261107"
TASKS_JSON = {"tasksall": GR2 / "tasks.json", "sftv3": GR / "tasks.json"}
INVENTORY = S / "sft/tasks-all-20261106/all_tasks.jsonl"
TOP100 = S / "top100.jsonl"
GRADING_REPAIR = SYNTH / "tools/grading_repair.py"
TEST_PATHS_PY = SYNTH / "verifier/test_paths.py"
ATP_APP_ENV = S / "atp/.env"                             # the ATP app config (CELL3_SANDBOX_API_KEY)
BAIHE_ENV = GR / "secrets/baihe-opensandbox.env"         # BAIHE_SANDBOX_API_KEY (mode 600)
CREDENTIAL_NAMES = {"hengyuf": "CELL3_SANDBOX_API_KEY", "baihe": "BAIHE_SANDBOX_API_KEY"}
LONG_ROOT = "/lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/hengyuf/"
SHORT_ROOT = "/lustre/fsw/portfolios/nemotron/users/hengyuf/"
OLD_GRADER_BYTES = 13468
SAMPLE_A = 60
MAX_INFRA_RETRIES = 2
RETRY_PAUSE_S = 30
CONCURRENCY_MAX = 32
SOURCE_FILES = ("tests/test.sh", "tests/test.patch", "tests/config.json", "tests/grader.py",
                "solution/solution.patch", "task.toml", "instruction.md")
TESTS_COMPARED = ("tests/test.sh", "tests/test.patch", "tests/config.json", "tests/grader.py")
SANITIZE_COUNT = re.compile(r"test-sanitize: (\d+) path\(s\) of model\.patch match test_paths")
RUN = ("glm-regrade-" + (os.environ.get("SLURM_JOB_ID") or "login"))[:40]
CAP_LINES = 50
CAP_FAILING = 50


class StageError(RuntimeError):
    pass


# ============================================================================================ small helpers

def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def sha256_bytes(data) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def file_sha(path):
    path = Path(path)
    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def norm(path: str) -> str:
    """The /lustre/fsw/portfolios/nemotron/users/hengyuf/ spelling of a path recorded under the projects/ alias."""
    return SHORT_ROOT + path[len(LONG_ROOT):] if path.startswith(LONG_ROOT) else path


def safe_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", key.replace("/", "__"))


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def write_json_atomic(path: Path, value) -> None:
    """Temp file, fsync, rename, directory fsync (review 2026-11-09): without the fsync a node crash or Lustre client
    eviction after the rename could leave results.json / stage.json empty or torn, i.e. every earlier result lost."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=1, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    try:
        directory = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory)
    except OSError:
        pass
    finally:
        os.close(directory)


def event(entry: dict) -> None:
    """Append-only action log (events.jsonl)."""
    with EVENTS.open("a") as stream:
        stream.write(json.dumps({"at": now(), "run": RUN, **entry}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextlib.contextmanager
def locked():
    """One regrade_glm.py process at a time (results.json / stage.json have a single writer)."""
    with LOCK.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another regrade_glm.py holds {LOCK}") from None
        yield


_MACHINERY = None


def machinery():
    """synth/tools/grading_repair.py, imported from its path (it imports common, regrade, rollouts, sandbox,
    task_builder from synth/); nothing is edited. ~1 min on Lustre."""
    global _MACHINERY
    if _MACHINERY is None:
        started = time.time()
        spec = importlib.util.spec_from_file_location("grading_repair", GRADING_REPAIR)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tp = module.task_builder.TEST_PATHS
        if Path(tp.__file__).resolve() != TEST_PATHS_PY.resolve():
            raise SystemExit(f"task_builder.TEST_PATHS is {tp.__file__}, not {TEST_PATHS_PY}")
        if module.GLOBS != tp.GLOBS or module.MODES != tp.MODES:
            raise SystemExit("grading_repair.GLOBS / MODES differ from verifier/test_paths.py")
        if Path(module.NEW_GRADER).resolve() != (SYNTH / "verifier/grader.py").resolve():
            raise SystemExit(f"grading_repair.NEW_GRADER is {module.NEW_GRADER}")
        log(f"machinery: grading_repair + task_builder + sandbox imported in {time.time() - started:.0f} s")
        _MACHINERY = module
    return _MACHINERY


def load_plan():
    data = PLAN.read_bytes()
    return json.loads(data), sha256_bytes(data)


# ============================================================================================ plan (read-only)

def task_globs(m, task_id: str, language: str, run: str) -> tuple:
    """(base globs, where they come from): grading_repair.task_globs."""
    if task_id in m.GLOB_OVERRIDES:
        where = f"grading_repair.GLOB_OVERRIDES[{task_id}]"
    elif run in m.RUN_GLOBS:
        where = f"grading_repair.RUN_GLOBS[{run}]"
    else:
        where = f"test_paths.GLOBS[{language}]"
    return m.task_globs(task_id, language, run), where


def make_task(m, batch: str, t: dict, task_id: str, class_language, inventory: dict, old_sha: str) -> dict:
    problems = []
    run, owner = t["run"], t["owner"]
    task_dir = Path(norm(t["task_dir"]))
    if owner not in CREDENTIAL_NAMES:
        problems.append(f"unknown owner {owner!r}")
    if batch == "tasksall":
        repo, target, listed = t.get("repo"), t.get("target"), t.get("image")
    else:
        row = inventory.get((run, task_id)) or {}
        repo, target = row.get("repo"), row.get("target")
        listed = f"snapshot:{t['snapshot']}" if t.get("snapshot") else None
    tp = m.task_builder.TEST_PATHS
    try:
        language = tp.repo_language(TOP100, target, repo)
    except Exception as error:  # UnknownLanguage or a missing repo/target
        language = None
        problems.append(f"no top100 language for repo {repo!r} target {target!r}: {error}")
    if language != class_language:
        problems.append(f"top100 language {language} != classes.json language {class_language}")
    shas = {name: file_sha(task_dir / name) for name in SOURCE_FILES}
    missing = [name for name, sha in shas.items() if sha is None]
    if missing:
        problems.append(f"missing source files {missing}")
    entry = {"task_key": f"{run}/{task_id}", "run": run, "task_id": task_id, "owner": owner,
             "sandbox_credential": CREDENTIAL_NAMES.get(owner), "batches": [batch], "repo": repo, "target": target,
             "language": language, "task_dir": str(task_dir), "source_sha256": shas,
             "reference_solution": str(task_dir / "solution/solution.patch"),
             "reference_solution_sha256": shas.get("solution/solution.patch"), "listed_image": listed,
             "dir_kind": t.get("dir_kind"), "source": t.get("source"), "problems": problems, "notes": []}
    if missing:
        return entry
    toml = tomllib.loads((task_dir / "task.toml").read_text())
    snapshot = (toml.get("environment") or {}).get("docker_image")
    entry["snapshot"] = snapshot
    if not snapshot:
        problems.append("task.toml has no [environment] docker_image")
    elif listed is None:  # sftv3 make_tasks.py records Baihe's snapshots only; terminal.json image is checked per pass
        entry["notes"].append("tasks.json lists no image for this task (sftv3 hengyuf rows); task.toml docker_image "
                              "is checked against every pass's terminal.json image")
    elif listed != snapshot:
        problems.append(f"tasks.json image {listed} != task.toml docker_image {snapshot}")
    if (toml.get("metadata") or {}).get("task_id") != task_id:
        problems.append(f"task.toml metadata.task_id {(toml.get('metadata') or {}).get('task_id')} != {task_id}")
    grader = task_dir / "tests/grader.py"
    entry["source_grader"] = "deepswe" if shas["tests/grader.py"] == old_sha else None
    if shas["tests/grader.py"] != old_sha or grader.stat().st_size != OLD_GRADER_BYTES:
        problems.append("tests/grader.py is not DeepSWE's grader (the old one)")
    config = json.loads((task_dir / "tests/config.json").read_text())
    if "test_paths" in config or "test_paths_mode" in config:
        problems.append("tests/config.json already has test_paths / test_paths_mode")
    if language not in m.GLOBS:
        problems.append(f"no test-path globs for language {language}")
        return entry
    test_patch = (task_dir / "tests/test.patch").read_text(errors="surrogateescape")
    solution = (task_dir / "solution/solution.patch").read_text(errors="surrogateescape")
    base, where = task_globs(m, task_id, language, run)
    extra, skipped = m.task_builder.test_patch_globs(test_patch, solution, base)
    globs = list(base) + list(extra)
    mode = m.task_mode(language)
    if mode != tp.config_keys(language)["test_paths_mode"]:
        problems.append(f"grading_repair.task_mode {mode} != test_paths.config_keys mode")
    solution_hits = m.matching(m.NEW.touched_paths(solution), globs)
    if solution_hits:
        problems.append(f"the reference solution touches test paths {solution_hits[:10]} (the sanitizer would revert "
                        "them; grading_repair refuses such a task)")
    entry.update(test_paths_base=list(base), test_paths_base_source=where, test_paths_extra=list(extra),
                 test_paths_extra_skipped=skipped, test_paths=globs, test_paths_mode=mode,
                 language_globs=list(m.GLOBS[language]), solution_test_path_hits=solution_hits,
                 test_patch_paths=sorted(set(m.OLD.patch_paths(test_patch))))
    return entry


def static_hits(m, text: str, task: dict) -> dict:
    """What the current sanitizer should touch for this model.patch (prediction; the sandbox log is the truth)."""
    globs = task.get("test_paths")
    if not globs:
        return {}
    touched = m.NEW.touched_paths(text)
    hits = m.matching(touched, globs)
    reset = set(task["test_patch_paths"])
    outside = [p for p in hits if p not in reset]
    additive = set(m.NEW.additive_paths(text)) if task["test_paths_mode"] == "non_additive" else set()
    effective = [p for p in outside if p not in additive]
    delta = [p for p in effective if not m.matching([p], task["language_globs"])]
    return {"test_path_hits": hits, "outside_test_patch": outside,
            "additive_candidates": sorted(additive & set(outside)), "glob_delta": delta}


def cmd_plan(args) -> int:
    if PLAN.exists() and not args.force:
        print(f"{PLAN} exists (results.json is bound to its sha256); --force writes a new one")
        return 2
    m = machinery()
    classes = json.loads(CLASSES.read_bytes())
    passes = classes["passes"]
    ta_rows = json.loads(TASKS_JSON["tasksall"].read_text())["tasks"]
    tv_rows = json.loads(TASKS_JSON["sftv3"].read_text())["tasks"]
    ta = {(t["campaign"], t["task_id"]): t for t in ta_rows}
    tv = {t["task_id"]: t for t in tv_rows}
    if len(ta) != len(ta_rows) or len(tv) != len(tv_rows):
        raise SystemExit("tasks.json keys are not unique")
    inventory = {}
    for line in INVENTORY.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            inventory[(row["run"], row["task_id"])] = row
    old_sha, new_sha = file_sha(m.OLD_GRADER), file_sha(m.NEW_GRADER)
    tasks, task_of, statics, problems = {}, {}, {}, []
    old_keys = sorted(k for k, p in passes.items() if p["grader_kind"] == "old")
    for key in old_keys:
        p = passes[key]
        parts = key.split("/")
        if len(parts) != 4 or parts[0] not in TASKS_JSON:
            raise SystemExit(f"unexpected pass key {key}")
        batch, campaign, task_id, _slot = parts
        t = ta.get((campaign, task_id)) if batch == "tasksall" else tv.get(task_id)
        if t is None:
            raise SystemExit(f"{key}: no {batch} tasks.json row")
        if t["run"] != p["run"]:
            problems.append(f"{key}: tasks.json run {t['run']} != classes.json run {p['run']}")
        task_key = f"{t['run']}/{task_id}"
        if task_key not in tasks:
            tasks[task_key] = make_task(m, batch, t, task_id, p["language"], inventory, old_sha)
        else:
            task = tasks[task_key]
            if batch not in task["batches"]:
                task["batches"].append(batch)
            if task["task_dir"] != norm(t["task_dir"]):
                task["problems"].append(f"{key}: another task_dir {t['task_dir']}")
        task_of[key] = task_key
        patch = Path(norm(p["attempt_dir"])) / "model.patch"
        text = patch.read_bytes().decode("utf-8", "surrogateescape") if patch.is_file() else ""
        statics[key] = static_hits(m, text, tasks[task_key])
    # selection
    sample = sorted((k for k, p in passes.items() if p["class"] == "A"),
                    key=lambda k: (hashlib.sha256(k.encode("utf-8")).hexdigest(), k))[:SAMPLE_A]
    selection = collections.defaultdict(list)
    for key in old_keys:
        if passes[key]["class"] == "B":
            selection[key].append("B")
    for key in sample:
        selection[key].append("A-sample")
    delta_a = sorted(k for k in old_keys if passes[k]["class"] == "A" and statics[k].get("glob_delta"))
    if not args.no_glob_delta:
        for key in delta_a:
            selection[key].append("A-glob-delta")
    plan_passes = {}
    for key in sorted(selection):
        p = passes[key]
        batch, campaign, task_id, slot = key.split("/")
        task = tasks[task_of[key]]
        attempt = Path(norm(p["attempt_dir"]))
        patch = attempt / "model.patch"
        entry = {"key": key, "safe_key": safe_key(key), "batch": batch, "campaign": campaign, "task_id": task_id,
                 "slot": slot, "class": p["class"], "why": p.get("why", []), "selection": selection[key],
                 "language": p["language"], "run": p["run"], "task": task_of[key], "owner": task["owner"],
                 "attempt_dir": str(attempt), "model_patch": str(patch), "static": statics[key], "problems": [],
                 "notes": []}
        problems_here = entry["problems"]
        if not patch.is_file():
            problems_here.append("no model.patch")
        else:
            data = patch.read_bytes()
            entry.update(model_patch_sha256=sha256_bytes(data), model_patch_bytes=len(data))
            try:
                data.decode("utf-8")
            except UnicodeDecodeError:
                problems_here.append("model.patch is not valid UTF-8 (sandbox.grade uploads text)")
            if not data:
                problems_here.append("model.patch is empty")
        terminal = read_json(attempt / "terminal.json") or {}
        entry["terminal"] = {"reward": terminal.get("reward"), "image": terminal.get("image"),
                             "evaluation_completed": terminal.get("evaluation_completed"),
                             "model_patch_sha256": terminal.get("model_patch_sha256"),
                             "f2p": f"{terminal.get('f2p_passed')}/{terminal.get('f2p_total')}",
                             "p2p": f"{terminal.get('p2p_passed')}/{terminal.get('p2p_total')}",
                             "verification_time_s": terminal.get("verification_time_s")}
        if terminal.get("reward") != 1:
            problems_here.append(f"terminal.json reward {terminal.get('reward')} (expected 1: a pass)")
        if terminal.get("image") != task.get("snapshot"):
            problems_here.append(f"terminal.json image {terminal.get('image')} != task snapshot {task.get('snapshot')}")
        if terminal.get("model_patch_sha256") != entry.get("model_patch_sha256"):
            problems_here.append("model.patch sha256 != terminal.json model_patch_sha256")
        # the copy of the task ATP graded from: <probe>/tasks/<task dir name>/ (rollouts.probe)
        probe_dir = attempt.parents[4] if len(attempt.parents) > 4 else None
        copy = probe_dir / "tasks" / Path(task["task_dir"]).name if probe_dir else None
        if copy and copy.is_dir():
            same = {name: file_sha(copy / name) == task["source_sha256"].get(name) for name in TESTS_COMPARED}
            entry["probe_copy"] = {"dir": str(copy), "tests_equal": all(same.values())}
            if not all(same.values()):
                problems_here.append(f"the probe's task copy differs from task_dir in {[n for n, ok in same.items() if not ok]}")
        else:
            entry["probe_copy"] = {"dir": str(copy) if copy else None, "tests_equal": None}
            entry["notes"].append("no probe task copy found (informational)")
        plan_passes[key] = entry
    plan_tasks = {k: tasks[k] for k in sorted({e["task"] for e in plan_passes.values()})}
    for task_key, task in plan_tasks.items():
        task["passes"] = sorted(k for k, e in plan_passes.items() if e["task"] == task_key)
    safe = [e["safe_key"] for e in plan_passes.values()]
    if len(set(safe)) != len(safe):
        raise SystemExit("safe keys collide")

    def counter(items):
        return dict(sorted(collections.Counter(items).items(), key=lambda kv: str(kv[0])))

    entries = list(plan_passes.values())
    counts = {
        "passes": len(entries), "tasks": len(plan_tasks),
        "passes_by_owner": counter(e["owner"] for e in entries),
        "tasks_by_owner": counter(t["owner"] for t in plan_tasks.values()),
        "passes_by_class": counter(e["class"] for e in entries),
        "passes_by_selection": counter("+".join(e["selection"]) for e in entries),
        "passes_by_language": counter(e["language"] for e in entries),
        "tasks_by_language": counter(t["language"] for t in plan_tasks.values()),
        "passes_by_batch": counter(e["batch"] for e in entries),
        "passes_by_owner_class": counter(f"{e['owner']}/{e['class']}" for e in entries),
        "tasks_with_extra_globs": sum(bool(t.get("test_paths_extra")) for t in plan_tasks.values()),
        "tasks_by_base_globs": counter(t.get("test_paths_base_source", "?").split("[")[0] for t in plan_tasks.values()),
        "class_B_tasks": len({e["task"] for e in entries if e["class"] == "B"}),
        "old_grader_passes_scanned": len(old_keys),
        "tasks_with_notes": sum(bool(t["notes"]) for t in plan_tasks.values()),
        "passes_with_notes": sum(bool(e["notes"]) for e in entries),
    }
    all_problems = problems + [f"{k}: {x}" for k, t in plan_tasks.items() for x in t["problems"]] + \
        [f"{k}: {x}" for k, e in plan_passes.items() for x in e["problems"]]
    plan = {
        "tool": "regrade_glm.py plan", "tool_sha256": file_sha(Path(__file__)), "created_at": now(),
        "inputs": {str(path): file_sha(path) for path in (CLASSES, TASKS_JSON["tasksall"], TASKS_JSON["sftv3"],
                                                          INVENTORY, TOP100, GRADING_REPAIR, TEST_PATHS_PY,
                                                          SYNTH / "task_builder.py", SYNTH / "sandbox.py")},
        "grader": {"path": norm(str(m.NEW_GRADER)), "sha256": new_sha},
        "old_grader": {"path": norm(str(m.OLD_GRADER)), "sha256": old_sha},
        "rules": {
            "B": "every class-B pass of classes.json",
            "A-sample": f"class-A keys sorted by sha256(key.encode('utf-8')).hexdigest() (then key), first {SAMPLE_A}",
            "A-glob-delta": "class-A passes whose model.patch touches, outside test.patch (and not as a JS/TS add-only "
                            "edit), a path that the task's test_paths match but the language default globs do not "
                            "(RUN_GLOBS / test_patch_globs extras; classify.py used the language globs only)"
                            + (" -- NOT selected (--no-glob-delta)" if args.no_glob_delta else ""),
            "test_paths": "base = grading_repair.task_globs (GLOB_OVERRIDES / RUN_GLOBS / language GLOBS) + "
                          "task_builder.test_patch_globs(test.patch, solution.patch, base) extras; mode = "
                          "grading_repair.task_mode(language)",
            "snapshot": "task.toml [environment] docker_image (== tasks.json image / snapshot, == terminal.json "
                        "image of every pass)",
            "static": "touched_paths(model.patch) matching test_paths; outside_test_patch: not reset by test.patch; "
                      "glob_delta: of those, matched only by non-default globs (a prediction, the sandbox log is "
                      "the truth)"},
        "counts": counts,
        "class_A_glob_delta": {k: statics[k]["glob_delta"] for k in delta_a},
        "glob_delta_all_classes": {k: statics[k]["glob_delta"] for k in old_keys if statics[k].get("glob_delta")},
        "problems": all_problems,
        "tasks": plan_tasks,
        "passes": plan_passes,
    }
    write_json_atomic(PLAN, plan)
    print(json.dumps(counts, indent=1))
    print(f"class-A glob-delta passes: {len(delta_a)} ({'selected' if not args.no_glob_delta else 'NOT selected'})")
    for problem in all_problems:
        print(f"problem: {problem}")
    print(f"wrote {PLAN} (sha256 {file_sha(PLAN)})")
    return 0


# ============================================================================================ stage

def staged_ok(m, rec: dict) -> bool:
    """The staged copy still holds exactly what stage wrote, with the current grader."""
    staged = Path(rec["staged_dir"])
    if not (file_sha(staged / "tests/grader.py") == rec["grader_sha256"] == file_sha(m.NEW_GRADER)):
        return False
    if file_sha(staged / "tests/config.json") != rec["config_sha256"]:
        return False
    return all(file_sha(staged / name) == sha for name, sha in rec["source_sha256"].items()
               if name not in ("tests/grader.py", "tests/config.json"))


def stage_task(m, entry: dict, previous, *, restage=False) -> dict:
    if entry["problems"]:
        raise StageError(f"plan problems: {entry['problems']}")
    source = Path(entry["task_dir"])
    shas = {name: file_sha(source / name) for name in SOURCE_FILES}
    if shas != entry["source_sha256"]:
        changed = [n for n in SOURCE_FILES if shas.get(n) != entry["source_sha256"].get(n)]
        raise StageError(f"source task changed since the plan: {changed}")
    try:
        config_text = m.sanitized_config((source / "tests/config.json").read_text(), entry["test_paths"],
                                         entry["test_paths_mode"])
    except m.RepairError as error:
        raise StageError(str(error)) from None
    dest = TASKS / entry["run"] / entry["task_id"]
    rec = {"status": "staged", "task_key": entry["task_key"], "run": entry["run"], "task_id": entry["task_id"],
           "owner": entry["owner"], "source": str(source), "source_sha256": shas, "snapshot": entry["snapshot"],
           "test_paths": entry["test_paths"], "test_paths_mode": entry["test_paths_mode"],
           "test_paths_base": entry["test_paths_base"], "test_paths_extra": entry["test_paths_extra"],
           "grader": norm(str(m.NEW_GRADER)), "grader_sha256": file_sha(m.NEW_GRADER),
           "config_sha256": sha256_bytes(config_text), "staged_dir": str(dest)}
    same_keys = ("source_sha256", "grader_sha256", "config_sha256", "test_paths", "test_paths_mode", "snapshot",
                 "staged_dir")
    if dest.exists():
        if previous and previous.get("status") == "staged" and not restage \
                and {k: previous.get(k) for k in same_keys} == {k: rec[k] for k in same_keys} \
                and staged_ok(m, previous):
            return dict(previous, unchanged=True)
        if not restage and not (dest / "tests").is_symlink() and staged_ok(m, rec):
            # review 2026-11-09: the copy holds exactly what this stage writes (current grader, this config, the
            # planned source files) but stage.json has no "staged" record for it: a stage killed between the rename
            # below and the stage.json write, or an earlier transient failure (OSError) that replaced the record.
            # Adopt it; refusing would keep the task stage_failed (its passes validation_failed) until --restage.
            rec.update(at=now(), adopted=True)
            return rec
        if not restage:
            raise StageError(f"{dest} exists but was staged from other inputs or edited since: --restage replaces it")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{entry['task_id']}.{uuid.uuid4().hex}.tmp"
    target = tmp / entry["task_id"]  # Gym's loader checks a parent that holds exactly this one task
    try:
        shutil.copytree(source, target, symlinks=True, ignore=shutil.ignore_patterns(
            "*" + m.BACKUP, "*" + m.BACKUP_V2, "*" + m.BACKUP_V3, "*.pre-early-exit-fix", "*.pre-gate-fix",
            "*.pre-regrade", "__pycache__"))
        # review 2026-11-09: copytree(symlinks=True) keeps symlinks, and copyfile / write_text below follow them: a
        # symlinked tests/ or tests/grader.py / config.json would be written THROUGH into its target (another task
        # dir or shared code). None of the 119 planned task dirs holds a symlink (checked); refuse rather than risk it.
        linked = [name for name in ("tests", "tests/grader.py", "tests/config.json") if (target / name).is_symlink()]
        if linked or not (target / "tests").is_dir():
            raise StageError(f"refusing to write the grader / config.json of {source}: symlinked or missing {linked}")
        shutil.copyfile(m.NEW_GRADER, target / "tests/grader.py")
        (target / "tests/config.json").write_text(config_text)
        try:
            m.task_builder.validate_task(target)  # Gym's DeepSWE task loader (both Gym copies) accepts the copy
        except m.task_builder.GateError as error:
            raise StageError(str(error)) from None
        if dest.exists():
            old = dest.parent / f".{entry['task_id']}.replaced-{uuid.uuid4().hex[:8]}"
            dest.rename(old)
            shutil.rmtree(old)
        target.rename(dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    rec["at"] = now()
    if not staged_ok(m, rec):
        raise StageError("the staged copy does not hold what stage wrote")
    return rec


def select_tasks(plan: dict, wanted) -> list:
    if not wanted:
        return list(plan["tasks"])
    unknown = [t for t in wanted if t not in plan["tasks"]]
    if unknown:
        raise SystemExit(f"not planned tasks (use <run>/<task_id>): {unknown}")
    return list(dict.fromkeys(wanted))


def cmd_stage(args) -> int:
    m = machinery()
    plan, plan_sha = load_plan()
    with locked():
        db = read_json(STAGE) or {"plan_sha256": plan_sha, "tasks": {}}
        if db.get("plan_sha256") != plan_sha:
            raise SystemExit(f"{STAGE} belongs to another plan.json (plan_sha256 {db.get('plan_sha256')})")
        failed = staged = unchanged = 0
        for task_key in select_tasks(plan, args.task):
            entry = plan["tasks"][task_key]
            try:
                rec = stage_task(m, entry, db["tasks"].get(task_key), restage=args.restage)
            except (StageError, OSError) as error:
                rec = {"status": "failed", "task_key": task_key, "error": f"{type(error).__name__}: {error}"[:2000],
                       "at": now()}
                failed += 1
                print(f"{task_key}: NOT STAGED: {rec['error']}")
            else:
                if rec.pop("unchanged", False):
                    unchanged += 1
                    print(f"{task_key}: unchanged ({rec['staged_dir']})")
                    continue
                staged += 1
                print(f"{task_key}: {'adopted (existing copy is exactly this stage)' if rec.get('adopted') else 'staged'}"
                      f" {rec['staged_dir']} test_paths {rec['test_paths']} mode {rec['test_paths_mode']}")
            db["tasks"][task_key] = rec
            write_json_atomic(STAGE, db)
            event({"action": "stage", "task": task_key, "status": rec["status"], "error": rec.get("error"),
                   "config_sha256": rec.get("config_sha256"), "grader_sha256": rec.get("grader_sha256")})
    print(f"stage: {staged} staged, {unchanged} unchanged, {failed} failed")
    return 0


# ============================================================================================ sandbox credentials

_OWNER = contextvars.ContextVar("glm_regrade_sandbox_owner", default=None)
_SECRETS = []  # the loaded sandbox API keys (never printed): redact() removes them from recorded error text


def redact(text) -> str:
    """Error text as recorded in results.json / printed, with any loaded sandbox API key cut out (review 2026-11-09:
    exception strings of the sandbox client go into results.json "attempts")."""
    text = str(text)
    for secret in _SECRETS:
        if secret:
            text = text.replace(secret, "<redacted>")
    return text


def setup_credentials(m, owners) -> None:
    """Make sandbox.owned() use the credential of the task owner of each grade (see the module docstring). Prints
    variable / file NAMES and check outcomes only."""
    sandbox = m.sandbox
    helpers = sys.modules["sandbox_helpers"]
    real = helpers.Config  # the helper's own class (only the name inside synth/sandbox.py is rebound below)
    leaked = sorted(name for name in os.environ if name.startswith("OPENSANDBOX_"))
    if leaked:
        raise SystemExit(f"OPENSANDBOX_* variables are set in this environment ({', '.join(leaked)}); they would "
                         "override the credential files: unset them")
    literal = helpers._literal_env
    atp_env = Path(sandbox.ATP_ENV)
    app, synth = literal(ATP_APP_ENV), literal(atp_env)
    for name in ("OPENSANDBOX_DOMAIN", "OPENSANDBOX_API_KEY", "OPENSANDBOX_TLS_VERIFY"):
        if not app.get(name) and name != "OPENSANDBOX_TLS_VERIFY":
            raise SystemExit(f"{ATP_APP_ENV} has no {name}")
        if app.get(name) != synth.get(name):
            raise SystemExit(f"{atp_env} {name} differs from the ATP app's {ATP_APP_ENV}: refusing")
    configs = {"hengyuf": real.load(atp_env)}
    if "baihe" in owners:
        if not BAIHE_ENV.is_file() or (os.stat(BAIHE_ENV).st_mode & 0o077):
            raise SystemExit(f"{BAIHE_ENV} must exist with mode 600")
        baihe_values = literal(BAIHE_ENV)
        if not baihe_values.get("OPENSANDBOX_API_KEY"):
            raise SystemExit(f"{BAIHE_ENV} has no OPENSANDBOX_API_KEY")
        for name in ("OPENSANDBOX_DOMAIN", "OPENSANDBOX_TLS_VERIFY"):
            if baihe_values.get(name) != app.get(name):
                raise SystemExit(f"{BAIHE_ENV} {name} differs from the ATP app's; refusing a mixed sandbox setup")
        if baihe_values["OPENSANDBOX_API_KEY"] == app.get("OPENSANDBOX_API_KEY"):
            raise SystemExit("the Baihe sandbox credential equals the ATP app's; refusing")
        baihe = real.load(BAIHE_ENV)
        mine = configs["hengyuf"]
        if (baihe.domain, baihe.protocol, baihe.tls_verify, baihe.use_server_proxy) != \
                (mine.domain, mine.protocol, mine.tls_verify, mine.use_server_proxy):
            raise SystemExit(f"{BAIHE_ENV} connection settings differ from {atp_env}'s; refusing")
        configs["baihe"] = baihe
    _SECRETS[:] = [config.api_key for config in configs.values() if config.api_key]

    class OwnerConfig:
        """Stand-in for sandbox_helpers.Config inside synth/sandbox.py: owned() calls Config.load(ATP_ENV)."""

        @staticmethod
        def load(env_file=None):
            if env_file is not None and Path(env_file) != atp_env:
                raise RuntimeError(f"unexpected Config.load({env_file})")
            owner = _OWNER.get()
            if owner not in configs:
                raise RuntimeError(f"no sandbox credential for task owner {owner!r}")
            return configs[owner]

    sandbox.Config = OwnerConfig
    log("sandbox credentials: " + ", ".join(
        f"{owner} = {CREDENTIAL_NAMES[owner]} ({atp_env if owner == 'hengyuf' else BAIHE_ENV})" for owner in configs)
        + f"; checks passed; sandbox run metadata {RUN!r}")


async def grade_as(m, owner: str, source: str, task_dir: Path, patch: str, out_dir: Path) -> dict:
    token = _OWNER.set(owner)
    try:
        return await m.sandbox.grade(source, task_dir, patch, run=RUN, out_dir=out_dir)
    finally:
        _OWNER.reset(token)


# ============================================================================================ grading

def sanitize_details(m, out_dir: Path) -> dict:
    """The grader's "[verifier] test-sanitize:" (and "ERROR: test-sanitize:") lines of output.log, parsed:
    actions in log order ({"action", "path"[, "detail"]}; grading_repair.SANITIZE_LINE), the matched-path count."""
    path = Path(out_dir) / "output.log"
    text = path.read_text(errors="replace") if path.is_file() else ""
    lines = [line.strip() for line in text.splitlines()
             if "[verifier]" in line and "test-sanitize" in line and not line.lstrip().startswith("+")]
    actions, errors, matched, mode_line = [], [], None, None
    for line in lines:
        hit = m.SANITIZE_LINE.search(line)
        if hit:
            item = {"action": hit.group(1), "path": json.loads(hit.group(2))}
            rest = line[hit.end():].strip()
            if rest and hit.group(1) in ("dangling-import", "dropped-manifest-target", "restored-manifest-target"):
                item["detail"] = rest[:300]
            actions.append(item)
            continue
        count = SANITIZE_COUNT.search(line)
        if count:
            matched = int(count.group(1))
        elif "ERROR" in line:
            errors.append(line[:400])
        elif "manifest-unread" in line:
            actions.append({"action": "manifest-unread", "detail": line[:300]})
        elif "test_paths_mode" in line:
            mode_line = line[:300]
    return {"actions": actions, "matched": matched, "mode_line": mode_line, "errors": errors,
            "lines": lines[:CAP_LINES], "applied": "model.patch applied" in text}


async def graded(m, owner: str, source: str, task_dir: Path, patch: str, out_dir: Path, gate, config: dict,
                 label: str) -> tuple:
    """sandbox.grade with up to MAX_INFRA_RETRIES retries (fresh sandbox each) when it raises or leaves no
    reward.json. Returns (grade_summary or None, attempts)."""
    attempts, summary = [], None
    for attempt in range(1, MAX_INFRA_RETRIES + 2):
        if attempt > 1:
            await asyncio.sleep(RETRY_PAUSE_S * (attempt - 1))
        async with gate:
            shutil.rmtree(out_dir, ignore_errors=True)  # this tool's own earlier output of this item
            started = time.time()
            try:
                result = await grade_as(m, owner, source, task_dir, patch, out_dir)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # sandbox create / exec / transfer failures
                summary = None
                attempts.append({"attempt": attempt, "at": now(), "seconds": round(time.time() - started),
                                 "error": f"{type(error).__name__}: {redact(error)[-400:]}"})
                log(f"{label}: attempt {attempt}: {type(error).__name__} (infra)")
                continue
        summary = m.grade_summary(result, out_dir, config)
        attempts.append({"attempt": attempt, "at": now(), "seconds": round(time.time() - started),
                         "status": summary["status"], "exit_code": summary.get("exit_code"),
                         "timed_out": summary.get("timed_out")})
        if summary["status"] in ("pass", "fail"):
            return summary, attempts
        log(f"{label}: attempt {attempt}: no reward.json (exit {summary.get('exit_code')}, timed out "
            f"{summary.get('timed_out')}) (infra)")
    return summary, attempts


def grade_record(m, summary, attempts: list, out_dir: Path) -> dict:
    """The common fields of a validation grade or a pass result."""
    if summary is None:
        return {"reward": None, "status": "infra", "sanitize_actions": [], "exit_code": None, "timed_out": None,
                "out_dir": str(out_dir), "graded_at": now(), "attempts": attempts}
    reward = summary.get("reward")
    reward = int(reward) if summary["status"] in ("pass", "fail") and reward in (0, 1) else None
    details = sanitize_details(m, out_dir)
    failing = summary.get("failing_tests") or []
    return {"reward": reward, "status": summary["status"] if reward is not None else "infra",
            "sanitize_actions": details["actions"], "sanitize_matched": details["matched"],
            "sanitize_mode_line": details["mode_line"], "sanitize_errors": details["errors"],
            "sanitize_lines": details["lines"], "model_patch_applied": details["applied"],
            "exit_code": summary.get("exit_code"), "timed_out": summary.get("timed_out"), "out_dir": str(out_dir),
            "graded_at": now(), "f2p": summary.get("f2p"), "p2p": summary.get("p2p"),
            "apply_failed": summary.get("apply_failed"),
            "failing_tests": failing[:CAP_FAILING] if reward is not None else [],
            "failing_tests_count": len(failing) if reward is not None else None, "attempts": attempts}


def open_results(plan_sha: str, grader_sha: str) -> dict:
    results = read_json(RESULTS)
    if results is None:
        return {"grader_sha256": grader_sha, "plan_sha256": plan_sha, "validation": {}, "results": {},
                "created_at": now()}
    if results.get("plan_sha256") != plan_sha:
        raise SystemExit(f"{RESULTS} belongs to another plan.json ({results.get('plan_sha256')}); move it aside first")
    if results.get("grader_sha256") != grader_sha:
        raise SystemExit(f"{RESULTS} was graded with grader {results.get('grader_sha256')}, synth/verifier/grader.py "
                         f"is now {grader_sha}; move it aside first")
    return results


def save_results(results: dict) -> None:
    statuses = collections.Counter(r.get("status") for r in results["results"].values())
    validation = collections.Counter(v.get("status") for v in results["validation"].values())
    results.update(updated_at=now(), counts={"results": dict(statuses), "validation": dict(validation)})
    write_json_atomic(RESULTS, results)


def stage_records(plan_sha: str) -> dict:
    db = read_json(STAGE) or {"plan_sha256": plan_sha, "tasks": {}}
    if db.get("plan_sha256") != plan_sha:
        raise SystemExit(f"{STAGE} belongs to another plan.json")
    return db["tasks"]


async def validate_task(m, entry: dict, stage_rec: dict, gate) -> dict:
    task_key = entry["task_key"]
    staged = Path(stage_rec["staged_dir"])
    config = json.loads((staged / "tests/config.json").read_text())
    snapshot = tomllib.loads((staged / "task.toml").read_text())["environment"]["docker_image"]
    if snapshot != entry["snapshot"]:
        raise StageError(f"staged task.toml docker_image {snapshot} != plan snapshot {entry['snapshot']}")
    patches = {"solution": (staged / "solution/solution.patch").read_text(), "empty": ""}
    out = VALIDATE_OUT / entry["run"] / entry["task_id"]

    async def one(name):
        summary, attempts = await graded(m, entry["owner"], snapshot, staged, patches[name], out / name, gate,
                                         config, f"validate {task_key} {name}")
        return grade_record(m, summary, attempts, out / name), summary

    (sol, sol_summary), (emp, emp_summary) = await asyncio.gather(one("solution"), one("empty"))
    sol_lines = (sol_summary or {}).get("sanitize", {}).get("lines", [])
    checks = {  # grading_repair.validate_task's checks
        "solution_scores_1": bool(sol_summary) and sol_summary.get("reward_json_present") is True
        and sol["reward"] == 1 and sol_summary["sanitize"]["applied"]
        and any("0 path(s) of model.patch match" in line for line in sol_lines),
        "empty_scores_0": bool(emp_summary) and emp_summary.get("reward_json_present") is True and emp["reward"] == 0}
    mode = stage_rec["test_paths_mode"]
    if mode != "all":
        checks["mode_logged"] = any(f"test_paths_mode {mode}:" in line for line in sol_lines)
    infra = "infra" in (sol["status"], emp["status"])
    status = "ok" if all(checks.values()) else "infra" if infra else "failed"
    return {"status": status, "task_key": task_key, "owner": entry["owner"], "snapshot": snapshot,
            "staged_dir": str(staged), "grader_sha256": stage_rec["grader_sha256"],
            "config_sha256": stage_rec["config_sha256"], "test_paths": stage_rec["test_paths"],
            "test_paths_mode": mode, "checks": checks, "solution": sol, "empty": emp, "graded_at": now()}


def install_signal_handlers():
    loop, task = asyncio.get_running_loop(), asyncio.current_task()

    def stop(name):
        log(f"{name}: cancelling; sandbox.owned deletes the in-flight sandboxes")
        task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        loop.add_signal_handler(sig, stop, sig.name)


async def run_validate(args) -> int:
    m = machinery()
    plan, plan_sha = load_plan()
    with locked():
        stages = stage_records(plan_sha)
        task_keys = select_tasks(plan, args.task)
        setup_credentials(m, {plan["tasks"][k]["owner"] for k in task_keys})
        grader_sha = file_sha(m.NEW_GRADER)
        results = open_results(plan_sha, grader_sha)
        install_signal_handlers()
        gate = asyncio.Semaphore(args.concurrency)
        todo = []
        for task_key in task_keys:
            st, prev = stages.get(task_key), results["validation"].get(task_key)
            if st is None:
                print(f"{task_key}: not staged (run stage first): skipped")
                continue
            if st.get("status") != "staged" or not staged_ok(m, st):
                reason = st.get("error") if st.get("status") != "staged" else \
                    "the staged copy changed since stage (or the grader did): stage --restage"
                if prev and prev.get("status") in ("ok", "failed") and not args.redo \
                        and prev.get("grader_sha256") == grader_sha:
                    # review 2026-11-09: a final validation (this plan, this grader) is not downgraded by a later
                    # stage problem; regrade keeps the task's final results and leaves its other passes "infra"
                    # until the stage is repaired (no staged copy, no grading).
                    print(f"{task_key}: validation {prev['status']} (final) kept; stage problem now: "
                          f"{str(reason)[:300]}")
                    continue
                results["validation"][task_key] = {"status": "stage_failed", "task_key": task_key, "reason": reason,
                                                   "graded_at": now()}
                save_results(results)
                event({"action": "validate", "task": task_key, "status": "stage_failed"})
                print(f"{task_key}: stage_failed: {str(reason)[:300]}")
                continue
            if prev and prev.get("status") in ("ok", "failed") and not args.redo \
                    and prev.get("config_sha256") == st["config_sha256"] and prev.get("grader_sha256") == grader_sha:
                print(f"{task_key}: validation {prev['status']} (final): skipped")
                continue
            todo.append(task_key)
        log(f"validate: {len(todo)} task(s), concurrency {args.concurrency}")

        async def one(task_key):
            entry = plan["tasks"][task_key]
            try:
                rec = await validate_task(m, entry, stages[task_key], gate)
            except (StageError, OSError, ValueError, KeyError) as error:
                rec = {"status": "stage_failed", "task_key": task_key,
                       "reason": f"{type(error).__name__}: {redact(error)}", "graded_at": now()}
            results["validation"][task_key] = rec
            save_results(results)
            event({"action": "validate", "task": task_key, "status": rec["status"], "checks": rec.get("checks"),
                   "solution": (rec.get("solution") or {}).get("reward"), "empty": (rec.get("empty") or {}).get("reward")})
            log(f"validate {task_key} ({entry['owner']}): {rec['status']}; solution reward "
                f"{(rec.get('solution') or {}).get('reward')} empty reward {(rec.get('empty') or {}).get('reward')} "
                f"checks {rec.get('checks')}")

        outcomes = await asyncio.gather(*(one(k) for k in todo), return_exceptions=True)
    errors = [(k, o) for k, o in zip(todo, outcomes) if isinstance(o, BaseException)]
    for task_key, error in errors:
        print(f"{task_key}: ERROR {type(error).__name__}: {redact(error)[-500:]}")
    counts = collections.Counter(results["validation"].get(k, {}).get("status") for k in task_keys)
    print(f"validate: {dict(counts)} over {len(task_keys)} selected task(s)")
    return 1 if errors else 0


def select_keys(plan: dict, keys, tasks) -> list:
    if not keys and not tasks:
        return list(plan["passes"])
    chosen = []
    unknown = [k for k in (keys or []) if k not in plan["passes"]]
    unknown += [t for t in (tasks or []) if t not in plan["tasks"]]
    if unknown:
        raise SystemExit(f"not planned: {unknown}")
    chosen += list(keys or [])
    for task_key in tasks or []:
        chosen += plan["tasks"][task_key]["passes"]
    return list(dict.fromkeys(chosen))


def pass_base(pe: dict) -> dict:
    return {"class": pe["class"], "selection": pe["selection"], "task": pe["task"], "owner": pe["owner"],
            "language": pe["language"], "model_patch_sha256": pe.get("model_patch_sha256"),
            "predicted_outside_test_patch": (pe.get("static") or {}).get("outside_test_patch")}


async def run_regrade(args) -> int:
    m = machinery()
    plan, plan_sha = load_plan()
    with locked():
        stages = stage_records(plan_sha)
        keys = select_keys(plan, args.key, args.task)
        setup_credentials(m, {plan["passes"][k]["owner"] for k in keys})
        grader_sha = file_sha(m.NEW_GRADER)
        results = open_results(plan_sha, grader_sha)
        install_signal_handlers()
        gate = asyncio.Semaphore(args.concurrency)
        todo = []
        for key in keys:
            pe = plan["passes"][key]
            task_key = pe["task"]
            st, v, prev = stages.get(task_key), results["validation"].get(task_key), results["results"].get(key)
            if prev and not args.redo:
                # review 2026-11-09: a pass / fail graded under this plan and grader stays final unless a STAGED copy
                # now carries another config. Before, a stage record that was merely "failed" in a later job (a
                # transient OSError, the source dir edited after the plan) turned it into validation_failed / infra:
                # a crash or hiccup could replace a valid final result with an exclusion. Under one plan the config
                # is a function of the plan (pinned source config.json + test_paths + mode), so no evidence of a
                # change means none happened.
                if prev.get("status") in ("pass", "fail") and prev.get("grader_sha256") == grader_sha \
                        and (not st or st.get("status") != "staged"
                             or prev.get("config_sha256") == st.get("config_sha256")):
                    continue
                if prev.get("status") == "validation_failed" and v and v.get("status") in ("failed", "stage_failed"):
                    continue
            if v and v.get("status") in ("failed", "stage_failed"):
                rec = dict(pass_base(pe), reward=None, status="validation_failed", sanitize_actions=[], exit_code=None,
                           timed_out=None, out_dir=None, graded_at=now(), validation_status=v["status"],
                           reason=f"task validation {v['status']}: " + (str(v.get("reason")) if v.get("reason")
                                                                        else f"checks {v.get('checks')}"))
            elif not v or v.get("status") != "ok" or not st or st.get("status") != "staged" \
                    or v.get("config_sha256") != st.get("config_sha256"):
                rec = dict(pass_base(pe), reward=None, status="infra", sanitize_actions=[], exit_code=None,
                           timed_out=None, out_dir=None, graded_at=now(),
                           validation_status=(v or {}).get("status", "missing"),
                           reason="task not validated with its current staged copy (validation "
                                  f"{(v or {}).get('status', 'missing')}, stage {(st or {}).get('status', 'missing')})"
                                  ": run stage / validate")
            elif pe["problems"]:
                rec = dict(pass_base(pe), reward=None, status="infra", sanitize_actions=[], exit_code=None,
                           timed_out=None, out_dir=None, graded_at=now(), reason=f"plan problems: {pe['problems']}")
            else:
                todo.append(key)
                continue
            results["results"][key] = rec
            save_results(results)
            event({"action": "regrade", "key": key, "status": rec["status"], "reason": rec.get("reason")})
            print(f"{key}: {rec['status']} ({rec.get('reason', '')[:200]})")
        log(f"regrade: {len(todo)} pass(es) to grade of {len(keys)} selected, concurrency {args.concurrency}")

        async def one(key):
            pe = plan["passes"][key]
            st = stages[pe["task"]]
            task = plan["tasks"][pe["task"]]
            data = Path(pe["model_patch"]).read_bytes()
            if sha256_bytes(data) != pe["model_patch_sha256"]:
                rec = dict(pass_base(pe), reward=None, status="infra", sanitize_actions=[], exit_code=None,
                           timed_out=None, out_dir=None, graded_at=now(), reason="model.patch changed since the plan")
            elif not staged_ok(m, st):
                rec = dict(pass_base(pe), reward=None, status="infra", sanitize_actions=[], exit_code=None,
                           timed_out=None, out_dir=None, graded_at=now(),
                           reason="the staged copy changed since stage: stage --restage, validate")
            else:
                staged = Path(st["staged_dir"])
                config = json.loads((staged / "tests/config.json").read_text())
                out_dir = OUT / pe["safe_key"]
                summary, attempts = await graded(m, pe["owner"], task["snapshot"], staged, data.decode("utf-8"),
                                                 out_dir, gate, config, f"regrade {key}")
                rec = dict(pass_base(pe), **grade_record(m, summary, attempts, out_dir),
                           grader_sha256=st["grader_sha256"], config_sha256=st["config_sha256"],
                           snapshot=task["snapshot"])
                predicted = set((pe.get("static") or {}).get("test_path_hits") or [])
                acted = {a["path"] for a in rec["sanitize_actions"] if a.get("action") in
                         ("restored", "deleted", "absent", "kept-additive")}
                if rec["status"] in ("pass", "fail"):
                    rec["sanitize_matches_prediction"] = acted == predicted and rec.get("sanitize_matched") == len(predicted)
            results["results"][key] = rec
            save_results(results)
            event({"action": "regrade", "key": key, "status": rec["status"], "reward": rec.get("reward"),
                   "sanitize": [f"{a['action']} {a.get('path', '')}" for a in rec.get("sanitize_actions", [])][:20],
                   "reason": rec.get("reason")})
            log(f"regrade {key} [{pe['class']}]: {rec['status']} reward {rec.get('reward')} F2P {rec.get('f2p')} "
                f"P2P {rec.get('p2p')}; sanitize " +
                (", ".join(f"{a['action']} {a.get('path', '')}" for a in rec.get("sanitize_actions", [])[:8]) or "-"))

        outcomes = await asyncio.gather(*(one(k) for k in todo), return_exceptions=True)
    errors = [(k, o) for k, o in zip(todo, outcomes) if isinstance(o, BaseException)]
    for key, error in errors:
        print(f"{key}: ERROR {type(error).__name__}: {redact(error)[-500:]}")
    counts = collections.Counter(results["results"].get(k, {}).get("status") for k in keys)
    print(f"regrade: {dict(counts)} over {len(keys)} selected pass(es)")
    return 1 if errors else 0


# ============================================================================================ status

def cmd_status(args) -> int:
    plan, plan_sha = load_plan()
    results = read_json(RESULTS) or {"validation": {}, "results": {}}
    if results.get("plan_sha256") not in (None, plan_sha):
        print(f"WARNING: {RESULTS} belongs to another plan.json")
    validation = results["validation"]
    print(f"plan: {len(plan['passes'])} passes on {len(plan['tasks'])} tasks; results.json grader "
          f"{str(results.get('grader_sha256'))[:12]}, updated {results.get('updated_at')}")
    print("validation:", dict(collections.Counter(v.get("status") for v in validation.values())),
          f"(not validated: {sum(1 for k in plan['tasks'] if k not in validation)})")
    table = collections.Counter()
    for key, pe in plan["passes"].items():
        r = results["results"].get(key)
        table[("+".join(pe["selection"]), pe["owner"], r.get("status") if r else "none")] += 1
    for (selection, owner, status), n in sorted(table.items()):
        print(f"  {selection:24s} {owner:8s} {status:18s} {n}")
    for key, r in sorted(results["results"].items()):
        if r.get("status") in ("fail", "validation_failed", "infra") and args.verbose:
            print(f"{r['status']:18s} {key}: {r.get('reason') or ''} F2P {r.get('f2p')} P2P {r.get('p2p')} sanitize "
                  + ", ".join(f"{a['action']} {a.get('path', '')}" for a in r.get("sanitize_actions", [])[:6]))
    return 0


# ============================================================================================ CLI

def concurrency(value: str) -> int:
    number = int(value)
    if not 1 <= number <= CONCURRENCY_MAX:
        raise argparse.ArgumentTypeError(f"1..{CONCURRENCY_MAX}")
    return number


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("plan", help="read-only -> plan.json")
    sub.add_argument("--force", action="store_true", help="replace an existing plan.json")
    sub.add_argument("--no-glob-delta", action="store_true", help="do not select the class-A glob-delta passes")
    sub = commands.add_parser("stage", help="scratch task copies with the current grader -> stage.json")
    sub.add_argument("--task", nargs="+", help="task keys <run>/<task_id> (default: every planned task)")
    sub.add_argument("--restage", action="store_true", help="replace an existing staged copy")
    sub = commands.add_parser("validate", help="reference solution -> 1, empty patch -> 0 (sandbox)")
    sub.add_argument("--task", nargs="+", help="task keys <run>/<task_id> (default: every planned task)")
    sub.add_argument("--concurrency", type=concurrency, default=16, help="sandboxes at a time (default 16)")
    sub.add_argument("--redo", action="store_true", help="validate again even when a final result exists")
    sub = commands.add_parser("regrade", help="model.patch of planned passes on the staged copies (sandbox)")
    sub.add_argument("--key", nargs="+", help="pass keys <batch>/<campaign>/<task_id>/<slot>")
    sub.add_argument("--task", nargs="+", help="every planned pass of these task keys")
    sub.add_argument("--concurrency", type=concurrency, default=16, help="sandboxes at a time (default 16)")
    sub.add_argument("--redo", action="store_true", help="grade again even when a final result exists")
    sub = commands.add_parser("status", help="counts from results.json")
    sub.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if "utf" not in locale.getpreferredencoding(False).lower().replace("-", ""):
        raise SystemExit("run with -X utf8 (or a UTF-8 locale): sandbox.grade writes patches with the default encoding")
    if Path(__file__).resolve().parent != REGRADE.resolve():
        raise SystemExit(f"{__file__} must live in {REGRADE}")
    if args.command == "plan":
        with locked():
            return cmd_plan(args)
    if args.command == "status":
        return cmd_status(args)
    for directory in (TASKS, VALIDATE_OUT, OUT):
        directory.mkdir(parents=True, exist_ok=True)
    if args.command == "stage":
        return cmd_stage(args)
    runner = run_validate if args.command == "validate" else run_regrade
    try:
        return asyncio.run(runner(args))
    except asyncio.CancelledError:
        log("cancelled: completed items are in results.json; rerun to resume")
        return 143


if __name__ == "__main__":
    sys.exit(main())
