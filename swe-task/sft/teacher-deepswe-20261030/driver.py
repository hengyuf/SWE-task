#!/usr/bin/env python3
"""teacher-deepswe (2026-10-30): one hub-teacher probe campaign = one model (kimi | glm | deepseek) x a subset of the 113
DeepSWE v1.1 tasks x 5 rollouts per task, with OpenCode's context window limited to 262,144 tokens and OpenCode's STOCK
system prompt. User request (hengyuf, 2026-10-30, verbatim): "我想在deepSWE 每一个task 上额外用kimi,Deepseek, GLM
(Deepseek: nvidia/deepseek-ai/deepseek-v4-pro GLM: nvidia/zai-org/glm-5.3 ), 每个模型rollout 5遍. 用一个subagent 来收集他们的
rollouts". A copy of kimi-rollouts-v4's driver (sft/kimi-rollouts-v4-stock256k-20261028 = KR4, read-only) with the same
mechanisms (legs, budget guard, budget-refusal recovery <name>-rec<n>, broken-reply re-runs <name>-brk<n>, results shape)
and these changes: the model is a parameter, the task sets are the DeepSWE waves, one hub key per leg (NVAPI_KEY_NEW_2
again since 2026-10-31 18:15; see below),
hengyuf's sandbox credential is the only one, and every slot also records whether its session ended on an output-cap
runaway (flag only).

    (cd swe-task/synth && atp/runtime/controller/bin/python3 -B <TD>/driver.py --model MODEL --taskset TASKSET
        [--group K] [--repeats N] [--max-concurrency C] [--dry-run])

The synth pipeline's own rollout and grading path (rollouts.probe: ATP campaign, OpenCode 1.17.11 harness, each task's
own DeepSWE grader), as KR4, with these settings (all in memory; settings.json, the synth code, hub_key_active.txt,
S/atp/.env and S/secrets are never edited):
* model alias MODEL = its spec + context_window 262144: kimi and glm = synth/settings.json's specs
  (nvidia/moonshotai/kimi-k3, nvidia/zai-org/glm-5.3, reasoning_effort max); deepseek = teacher-rollouts-v2's in-memory
  spec (nvidia/deepseek-ai/deepseek-v4-pro, reasoning_effort max; not in settings.json). ATP's build_config gives
  OpenCode limit.context = limit.input = 262144 and limit.output = max_output_tokens = 131072, so OpenCode compacts at
  262144 - 20000 = 242,144 prompt tokens; temperature 1.0 / top_p 0.95 are settings.json's. The aliases keep their
  names, so ATP streams kimi's hub replies (ATP_DEEPSWE_UPSTREAM_STREAM) and not glm's / deepseek's, as in KR4 / TR2.
* system prompt: OpenCode's stock prompt (probe_worker.py, TDW_PROMPT=stock; no agent.build.prompt override, so OpenCode
  uses SystemPrompt.provider() = prompt/opencode-1.17.11-default.txt for "dummy_model", the same for every alias).
* hub key by NAME (KEY_PERIODS, by leg start time): NVAPI_KEY_NEW_2 for every leg started from 2026-10-31 18:15
  (OPERATOR_NOTE 2026-10-31 18:15, a user decision relayed by main: "kimi/GLM/DeepSeek 的rollout 用NEW_2 继续跑"; NEW_3
  had reached its $1,000 floor at 10:30 and is never used again by this campaign). Before that: NVAPI_KEY_NEW_3 for the
  legs started from 2026-10-30 21:45 (OPERATOR_NOTE 21:45: "use NEW_3 as the api_key"; floors $1,200 / $1,000) until the
  floor stop of 10-31 10:30-10:40, and NVAPI_KEY_NEW_2 for the setup and the three pilot legs started at 21:34. NEW_2
  also carries the live r6 RL run's rubric judge (same user decision), so the floors protect it: a leg starts only while
  NEW_2 has >= START_FLOOR ($1,700) left and the guard stops it below GUARD_FLOOR ($1,500). No other key, ever, even
  when NEW_2 reaches the floor (no move to any other key unless a later OPERATOR_NOTE entry says so). Every earlier leg
  of a resumed campaign must have used the key of its start time (check_existing, key_of_time). A leg starts only
  while the key has >= START_FLOOR left; a guard re-checks it
  every GUARD_INTERVAL s and stops the leg (SIGTERM to the ATP worker: in-flight rollouts are interrupted and stay
  retryable) once it has < GUARD_FLOOR left. A slot whose final attempt still got a hub budget refusal (ATP makes a hub
  4xx fatal; also the known transient-401-then-false-422 glitch of keys past $5k spend) is re-run once in
  probes/<model>/<name>-rec<n> and replaces the refused slot (KR4's mechanism); so are slots listed for this model and
  campaign in quota-incident-slots.json (none shipped).
* broken-reply re-runs: KR4's rule (hengyuf approved it on 2026-10-28 for the Kimi v4 batch: "这次若遇到被坏回复终止的
  slot, 可以重跑1遍"), applied here on the workflow's instruction (main, 2026-10-30) for every model whose detector was
  validated: analysis/brokenreply-validate-tr2.json "decision" (kimi: KR4's validation; glm / deepseek: 0 false
  positives on teacher-rollouts-v2's 504 endings / 84,065 replies). Once the campaign's slots are terminal (and the
  rec recoveries are done), every slot whose final attempt was ENDED by a broken reply (brokenreply.py = degenerate.py's
  definition: the LAST 200 reply is [PAD] / special-token / empty and has no tool call) is re-run exactly once in
  probes/<model>/<name>-brk<n> (reruns/<model>-<name>.json is the plan, written before the first re-run and checked on
  every later leg). A graded re-run REPLACES the slot (the original stays as "original_attempt"); a re-run that ends
  broken again is kept (never a third attempt); an infra re-run leaves the original in place ("rerun_attempt").
  <name>-brk<n> slots are never re-run (no rec either). A model whose decision is false gets the flag only.
* output-cap runaways (flag only, never re-run): "ended_by_output_cap_runaway" = the last 200 reply has no tool call and
  used the whole output budget (completion_tokens >= 131072 or finish "length"); seen once in TR2 (DeepSeek).
* sandbox: hengyuf's OpenSandbox credential (S/atp/.env CELL3_SANDBOX_API_KEY, ATP's own) for every campaign; Baihe's
  is never used (probe_worker.py has no path for it). The cell is shared with the live r6 RL run.
Tasksets (tasks/waves.json, written once by setup/make_tasks.py; task order = sorted task_id = DeepSWE task_index order;
group K of G = wave_tasks[K::G]):
  pilot       1 rollout of PILOT_TASK (wave 1, Go, near the median Kimi rollout size: analysis/size-and-pilot.json)
  deepswe-w1  the 79 tasks without an accepted synthetic task (uncovered_base_k10.csv), 3 groups (27, 26, 26)
  deepswe-w2  the other 34 tasks, 2 groups (17, 17)
Campaign names: pilot | w1-g<K> | w2-g<K>; Slurm job / exp names tdw-<model>-<name> (submit.sh). Writes
probes/<model>/<name>/, appends one line per leg event to legs.jsonl, and when the campaign is complete
results/<model>-<name>.json (KR4's shape + model alias / id, wave, runaway flags). Resumable: rerun the same command in a
fresh CPU job (the next leg).
Exit codes: 0 complete; 1 incomplete leg / error; 4 the guard stopped the leg; 5 the leg refused to start (NEW_2 budget
unknown or below START_FLOOR).
"""
import argparse
import asyncio
import collections
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time

TD = Path(__file__).resolve().parent
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
SYNTH = S / "synth"
sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != TD]
sys.path.insert(0, str(SYNTH))
import common  # noqa: E402
from common import SETTINGS, log  # noqa: E402
import rollouts  # noqa: E402

BASE_SPECS = {
    "kimi": {"id": "nvidia/moonshotai/kimi-k3", "sampling_overrides": {"reasoning_effort": "max"}},
    "glm": {"id": "nvidia/zai-org/glm-5.3", "sampling_overrides": {"reasoning_effort": "max"}},
    "deepseek": {"id": "nvidia/deepseek-ai/deepseek-v4-pro", "sampling_overrides": {"reasoning_effort": "max"}}}
FROM_SETTINGS = ("kimi", "glm")  # specs that must equal synth/settings.json's; deepseek is TR2's in-memory spec
CONTEXT_WINDOW = 262144
OUTPUT_CAP = 131072
PROMPT = "stock"  # OpenCode's own system prompt (probe_worker.py refuses anything else)
STOCK_PROMPT_FILE = TD / "prompt" / "opencode-1.17.11-default.txt"
STOCK_PROMPT_SHA256 = "962fbf3cb3ec659c9a5244425ee2e7bb141ad4428f489a630a7738566880dc6a"
KEY = "NVAPI_KEY_NEW_2"   # the campaign's hub key for every leg started from 2026-10-31 18:15 (OPERATOR_NOTE 18:15)
START_FLOOR = 1700.0      # USD: a leg starts only while NEW_2 has at least this left (NEW_2 also carries the r6 judge)
GUARD_FLOOR = 1500.0      # USD: a running leg stops once NEW_2 has less than this left
# The campaign's hub key by leg start time: (key NAME, from, until), [from, until) as ledger times (%Y-%m-%dT%H:%M:%S%z)
KEY_PERIODS = (
    ("NVAPI_KEY_NEW_2", None, "2026-10-30T21:45:00-0700"),  # the setup and the three pilot legs of 21:34
    ("NVAPI_KEY_NEW_3", "2026-10-30T21:45:00-0700", "2026-10-31T18:15:00-0700"),  # OPERATOR_NOTE 21:45 .. floor stop
    ("NVAPI_KEY_NEW_2", "2026-10-31T18:15:00-0700", None))  # OPERATOR_NOTE 2026-10-31 18:15 (user decision)
KEY_SWITCHES = (
    {"at": "2026-10-30T21:45:00-0700", "from": "NVAPI_KEY_NEW_2", "to": "NVAPI_KEY_NEW_3", "floors_usd": [1200.0, 1000.0],
     "source": "OPERATOR_NOTE 2026-10-30 21:45 (user decision relayed by main)"},
    {"at": "2026-10-31T18:15:00-0700", "from": "NVAPI_KEY_NEW_3", "to": "NVAPI_KEY_NEW_2", "floors_usd": [1700.0, 1500.0],
     "source": "OPERATOR_NOTE 2026-10-31 18:15 (user decision relayed by main; NEW_3 had reached its $1,000 floor)"})
GUARD_INTERVAL = 300.0    # s between the guard's budget checks (one 4-token GLM call each)
PILOT_TASK = "geo-shapeindex-serialization"
TASKSETS = {"pilot": None, "deepswe-w1": ("w1", 3), "deepswe-w2": ("w2", 2)}
SANDBOX = "hengyuf"
SANDBOX_CREDENTIAL = "CELL3_SANDBOX_API_KEY"
BUDGET_REFUSAL = re.compile(r"(?i)budget[_ ]exceeded|budget has been exceeded|exceededbudget|max budget")
RERUN_REASON = "ended_by_broken_reply"
RERUN_RULE = ("KR4 broken-reply rule (hengyuf approved it on 2026-10-28 for the Kimi v4 batch: 这次若遇到被坏回复终止的 "
              "slot, 可以重跑1遍), applied to this campaign on the workflow's instruction (main, 2026-10-30): one re-run "
              "per slot ended by a broken reply, for models validated in analysis/brokenreply-validate-tr2.json")
LEGS_FILE = "tdw-legs.jsonl"  # per campaign, written by probe_worker.py at every ATP worker start
WAVES = TD / "tasks" / "waves.json"
VALIDATION = TD / "analysis" / "brokenreply-validate-tr2.json"
LABEL_LIMIT = 63          # Kubernetes label value limit (OpenSandbox metadata = k8s labels)
WORKERS = []
MODEL = None              # set by main(); module-level so the helpers read the leg's model
SPEC = None


def _load(name: str):
    """A module of this directory by path (TD is not on sys.path, so nothing here can shadow a synth/ATP module)."""
    spec = importlib.util.spec_from_file_location(f"tdw_{name}", TD / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BROKEN = _load("brokenreply")  # degenerate.py's classify(), applied to the last 200 reply of an attempt


def waves() -> dict:
    return json.loads(WAVES.read_text())


def taskset(name: str, group: int) -> tuple:
    """(tasks [{task_id, task_dir, ...}], campaign name, repeats, wave) of a taskset / group."""
    data = waves()
    by_id = {task["task_id"]: task for task in data["tasks"]}
    assert len(by_id) == 113 and len(data["wave1"]) == 79 and len(data["wave2"]) == 34, data["counts"]
    assert not set(data["wave1"]) & set(data["wave2"]) and set(data["wave1"]) | set(data["wave2"]) == set(by_id)
    if name == "pilot":
        assert group == 0
        return [by_id[PILOT_TASK]], "pilot", 1, by_id[PILOT_TASK]["wave"]
    wave, groups = TASKSETS[name]
    assert 0 <= group < groups, f"{name} has groups 0..{groups - 1}"
    ids = data["wave1"] if wave == "w1" else data["wave2"]
    assert ids == sorted(ids)
    return [by_id[task_id] for task_id in ids[group::groups]], f"{wave}-g{group}", 5, int(wave[1])


def check_task(task: dict) -> None:
    """A task dir still has the files of the snapshot (the pinned DeepSWE dirs are read-only)."""
    task_dir = Path(task["task_dir"])
    text = (task_dir / "task.toml").read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == task["task_toml_sha256"], f"{task_dir}/task.toml changed"
    for required in ("instruction.md", "tests/test.sh", "tests/grader.py", "tests/config.json"):
        assert (task_dir / required).is_file(), (task_dir, required)


def sandbox_labels(name: str, tasks: list, repeats: int) -> dict:
    """The longest value of every OpenSandbox metadata label ATP / Gym put on this campaign's sandboxes (they become
    Kubernetes labels, <= 63 characters; ATP's recovery compares atp-model / atp-attempt verbatim, so a truncated value
    would make cleanup miss the sandbox). Sources: deepswe_stage.build_config sandbox.default_metadata (run = the launch
    dir name "launch-<SLURM_JOB_ID>-<12 hex>", user), Gym's attribution (team = the Slurm account, user, workload =
    deepswe-http from deepswe_stage.runtime_env, run), resources_servers/deepswe (benchmark, deepswe-task =
    task_id[:63], deepswe-phase, nemo_gym_agent = the server name) and ATP's deepswe_capture.attempt_metadata
    (atp-model = the model dir name = the alias, atp-attempt = "<task index>-<repeat>-attempt-<n>"). The campaign name
    itself is not a label value; it is listed with its longest recovery suffixes for completeness."""
    max_attempts = SETTINGS["max_attempts"]
    labels = {
        "nemo-gym.nvidia.com/run": "launch-" + "9" * 10 + "-" + "f" * 12,  # 10-digit job id (now 8)
        "nemo-gym.nvidia.com/user": "hengyuf", "nemo-gym.nvidia.com/team": max(
            ("nemotron_agents_dev", "nemotron_rl_algo"), key=len),
        "nemo-gym.nvidia.com/workload": "deepswe-http", "benchmark": "deepswe-v1-1",
        "deepswe-task": max((task["task_id"] for task in tasks), key=len),
        "deepswe-phase": "verifier", "nemo_gym_agent": "deepswe_opencode_resources_server",
        "atp-model": MODEL,
        "atp-attempt": f"{len(tasks) - 1}-{repeats - 1}-attempt-{max_attempts - 1}",
        "campaign (not a label)": f"{name}-brk{repeats}" if name != "pilot" else f"{name}-rec{repeats}",
        "job name (not a label)": f"tdw-{MODEL}-{name}"}
    return {key: {"value": value, "chars": len(value)} for key, value in labels.items()}


def budget(key: str, tag: str) -> dict:
    """The key's budget via key_budget.py (prints the key NAME and numbers only; refuses keys outside its list)."""
    try:
        result = subprocess.run(["/usr/bin/python3", "-I", "-B", str(TD / "key_budget.py"), "--key-name", key,
                                 "--tag", tag], capture_output=True, text=True, timeout=300)
        line = (result.stdout.strip().splitlines() or ["{}"])[-1]
        record = json.loads(line)
    except (ValueError, subprocess.SubprocessError, OSError) as error:
        record = {"key": key, "error": f"{type(error).__name__}: {str(error)[:200]}"}
    record.setdefault("key", key)
    return record


def known_budget(key: str, tag: str, tries: int = 3) -> dict:
    """budget() retried (60 s apart) while the hub's answer gives no remaining budget."""
    for attempt in range(tries):
        record = budget(key, tag)
        log(f"teacher-deepswe: budget {json.dumps(record)}")
        if record.get("remaining") is not None:
            break
        if attempt + 1 < tries:
            time.sleep(60)
    return record


def key_of_time(at):
    """The campaign's hub key NAME for a leg started at `at` (a ledger time, %Y-%m-%dT%H:%M:%S%z) per KEY_PERIODS; None
    when the time is unreadable."""
    fmt = "%Y-%m-%dT%H:%M:%S%z"
    try:
        when = datetime.datetime.strptime(str(at), fmt)
    except ValueError:
        return None
    for name, since, until in KEY_PERIODS:
        if (since is None or when >= datetime.datetime.strptime(since, fmt)) and \
                (until is None or when < datetime.datetime.strptime(until, fmt)):
            return name
    return None


def choose_key(tag: str) -> tuple:
    """(KEY, its budget record) while KEY has >= START_FLOOR left; refuses otherwise (there is no fallback key)."""
    record = known_budget(KEY, tag)
    if record.get("remaining") is None:
        return None, record
    if record["remaining"] < START_FLOOR:
        return None, record
    return KEY, record


async def _run(argv, log_path, env) -> int:
    """rollouts._run (unchanged behaviour) that also remembers the ATP worker process for the budget guard."""
    with open(log_path, "ab") as stream:
        process = await asyncio.create_subprocess_exec(*map(str, argv), stdout=stream, stderr=stream, env=env)
        if any(str(arg).endswith("probe_worker.py") for arg in argv):
            WORKERS.append(process)
        return await process.wait()


async def guard(key: str, name: str, stop: dict):
    """Stop this leg (SIGTERM to the ATP worker) once `key` has < GUARD_FLOOR left; checks every GUARD_INTERVAL s."""
    await asyncio.sleep(random.uniform(30, GUARD_INTERVAL))
    while True:
        record = await asyncio.to_thread(budget, key, f"guard {MODEL}/{name} job {os.environ.get('SLURM_JOB_ID')}")
        remaining = record.get("remaining")
        if remaining is not None and remaining < GUARD_FLOOR:
            stop.update(reason=f"{key} remaining ${remaining:.2f} < ${GUARD_FLOOR:.0f}", at=time.time(), budget=record)
            log(f"teacher-deepswe: GUARD {stop['reason']}: stopping the leg (in-flight rollouts stay retryable)")
            for process in WORKERS:
                if process.returncode is None:
                    process.send_signal(signal.SIGTERM)
            return
        await asyncio.sleep(GUARD_INTERVAL)


async def probe_with_guard(probe_dir, task_dirs, repeats, key, name):
    stop = {}
    watcher = asyncio.create_task(guard(key, name, stop))
    try:
        return await rollouts.probe(probe_dir, task_dirs, models=(MODEL,), repeats=repeats), stop
    except RuntimeError as error:
        return error, stop
    finally:
        watcher.cancel()


def check_existing(probe_dir: Path) -> None:
    """A campaign created earlier (main, rec or brk) must have this leg's model spec, sandbox and stock prompt."""
    if (probe_dir / "settings.json").exists():
        created = json.loads((probe_dir / "settings.json").read_text())
        assert created["models"] == {MODEL: SPEC}, f"{probe_dir} was created with {created['models']}"
    path = probe_dir / LEGS_FILE
    legs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    assert all(leg["sandbox"] == SANDBOX for leg in legs), f"{probe_dir}: earlier legs used another sandbox"
    assert all(leg.get("prompt") == PROMPT for leg in legs), f"{probe_dir}: earlier legs used another prompt"
    assert all(leg.get("hub_key") == key_of_time(leg.get("at")) for leg in legs), \
        f"{probe_dir}: an earlier leg used another hub key than the key of its start time (KEY_PERIODS)"
    assert all(leg.get("model") == MODEL for leg in legs), f"{probe_dir}: earlier legs ran another model"


def quota_incident_slots(name: str) -> set:
    """{(task_id, slot)} of this model's campaign listed in quota-incident-slots.json (slots in flight when a Lustre
    quota incident hit; ATP leaves them needs_review/active). None shipped; entries need "model" and "campaign"."""
    path = TD / "quota-incident-slots.json"
    if not path.exists():
        return set()
    return {(entry["task_id"], entry["slot"]) for entry in json.loads(path.read_text())["slots"]
            if entry["campaign"] == name and entry.get("model") == MODEL}


def refused_slots(records: dict, name: str = "") -> dict:
    """{task_id: [slot, ...]} of infra slots to re-run once: the final attempt got a hub budget refusal (a fatal 4xx),
    or the slot is listed as a casualty of a quota incident (quota_incident_slots)."""
    out = collections.defaultdict(list)
    listed = quota_incident_slots(name)
    for task_id, slot_records in records.items():
        for record in slot_records:
            if record.get("status") != "infra" or not record.get("attempt_dir"):
                continue
            if (task_id, record["slot"]) in listed:
                out[task_id].append(record["slot"])
                continue
            path = Path(record["attempt_dir"]) / "http.jsonl"
            if not path.exists():
                continue
            with open(path, "rb") as stream:
                for raw in stream:
                    head = raw[:3000]
                    if b'"event": "response"' not in head and b'"event":"response"' not in head:
                        continue
                    event = json.loads(raw)
                    if event.get("status") in (400, 401, 403, 404, 422) and BUDGET_REFUSAL.search(event.get("body") or ""):
                        out[task_id].append(record["slot"])
                        break
    return dict(out)


def compact(record):
    """KR4's compact record (the teacher-rollouts-v2 shape + the broken-ending verdict) + the last reply's stop class /
    finish / completion tokens and the output-cap runaway flag (same tail reader, brokenreply.last_reply_detail)."""
    attempt = record.get("attempt_dir")
    verdict = BROKEN.broken_end(attempt)
    detail = BROKEN.last_reply_detail(attempt) if attempt else {}
    runaway = bool(detail) and not detail.get("tool") and (
        (detail.get("completion_tokens") or 0) >= OUTPUT_CAP or detail.get("finish") == "length")
    return ({key: record.get(key) for key in ("slot", "status", "reward", "f2p", "p2p", "agent_error", "verifier_error",
                                              "attempt_dir")}
            | {"patch_chars": len(record.get("patch") or ""), "ended_by_broken_reply": verdict["ended"],
               "last_reply_kind": verdict["kind"], "last_reply_stop_class": detail.get("stop_class"),
               "last_reply_finish": detail.get("finish"), "last_reply_completion_tokens": detail.get("completion_tokens"),
               "ended_by_output_cap_runaway": runaway})


def stats(records: dict) -> dict:
    flat = [record for slot_records in records.values() for record in slot_records]
    passed = sum(record["status"] == "pass" for record in flat)
    failed = sum(record["status"] == "fail" for record in flat)
    return {"pass": passed, "fail": failed, "infra": len(flat) - passed - failed,
            "rate": round(passed / (passed + failed), 4) if passed + failed else None}


def broken_plan(name: str, records: dict) -> dict:
    """The broken-reply re-run plan of a campaign whose slots are terminal: every slot (in record order) whose final
    record ended on a broken reply, grouped by the task's number of such slots -> campaign <name>-brk<n>, repeat i."""
    broken = {task_id: [record for record in slot_records if record["ended_by_broken_reply"]]
              for task_id, slot_records in records.items()}
    broken = {task_id: found for task_id, found in broken.items() if found}
    slots = []
    for task_id in sorted(broken):
        count = len(broken[task_id])
        for repeat, record in enumerate(broken[task_id]):
            slots.append({"task_id": task_id, "slot": record["slot"], "original_attempt": record["attempt_dir"],
                          "original_status": record["status"], "last_reply_kind": record["last_reply_kind"],
                          "rerun_campaign": f"{name}-brk{count}", "rerun_slot": f"{MODEL}-r{repeat}"})
    return {"model": MODEL, "campaign": name, "rule": RERUN_RULE, "reason": RERUN_REASON, "slots": slots}


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def leg_record(**fields):
    with open(TD / "legs.jsonl", "a") as stream:
        stream.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "job": os.environ.get("SLURM_JOB_ID"),
                                 "model": MODEL, **fields}, ensure_ascii=False) + "\n")


def legs_of(probe_dir: Path) -> list:
    path = probe_dir / LEGS_FILE
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def rerun_decision() -> dict:
    """analysis/brokenreply-validate-tr2.json's per-model decision (the validation's evidence and the reviewer's call)."""
    decision = (json.loads(VALIDATION.read_text()).get("decision") or {}).get(MODEL) or {}
    assert isinstance(decision.get("broken_reply_reruns"), bool), f"no broken-reply decision for {MODEL} in {VALIDATION}"
    return decision


def main():
    global MODEL, SPEC
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(BASE_SPECS), required=True)
    parser.add_argument("--taskset", choices=sorted(TASKSETS), required=True)
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--max-concurrency", type=int, default=8,
                        help="rollouts in flight for this campaign (fixed at campaign creation; at most 8 here)")
    parser.add_argument("--dry-run", action="store_true",
                        help="check and print the campaign (tasks, spec, labels) without hub, sandbox or file writes")
    args = parser.parse_args()
    MODEL = args.model
    SPEC = dict(BASE_SPECS[MODEL], context_window=CONTEXT_WINDOW)
    assert key_of_time(time.strftime("%Y-%m-%dT%H:%M:%S%z")) == KEY, "KEY is not the key of the current KEY_PERIODS entry"
    chosen, name, repeats, wave = taskset(args.taskset, args.group)
    repeats = args.repeats or repeats
    assert chosen, f"no tasks for {args.taskset}"
    assert 1 <= args.max_concurrency <= 8, "at most 8 rollouts in flight per campaign"
    for task in chosen:
        check_task(task)
    if MODEL in FROM_SETTINGS:
        found = SETTINGS["models"].get(MODEL)
        assert found == BASE_SPECS[MODEL], f"settings.json {MODEL} spec changed: {found}"
    else:
        assert MODEL not in SETTINGS["models"], f"settings.json now has a {MODEL} spec: {SETTINGS['models'][MODEL]}"
    assert SETTINGS["max_output_tokens"] == OUTPUT_CAP and SETTINGS["context_window"] == 1048576
    assert SETTINGS["temperature"] == 1.0 and SETTINGS["top_p"] == 0.95
    assert SETTINGS["agent_timeout_seconds"] == 10800
    assert SPEC["sampling_overrides"] == {"reasoning_effort": "max"} and "max_output_tokens" not in SPEC
    assert hashlib.sha256(STOCK_PROMPT_FILE.read_bytes()).hexdigest() == STOCK_PROMPT_SHA256, "stock prompt file changed"
    assert (TD / "probe_worker.py").is_file()
    decision = rerun_decision()
    labels = sandbox_labels(name, chosen, repeats)
    too_long = {key: value for key, value in labels.items() if value["chars"] > LABEL_LIMIT}
    assert not too_long, f"sandbox labels over {LABEL_LIMIT} characters: {too_long}"
    probe_dir = TD / "probes" / MODEL / name
    if args.dry_run:
        print(json.dumps({"model": MODEL, "spec": SPEC, "taskset": args.taskset, "group": args.group,
                          "campaign": name, "wave": wave, "repeats": repeats, "slots": len(chosen) * repeats,
                          "max_concurrency": args.max_concurrency, "probe_dir": str(probe_dir),
                          "results": str(TD / "results" / f"{MODEL}-{name}.json"),
                          "broken_reply_reruns": decision["broken_reply_reruns"], "labels": labels,
                          "tasks": [task["task_id"] for task in chosen]}, ensure_ascii=False))
        return

    SETTINGS["models"][MODEL] = dict(SPEC)  # in memory only: rollouts.probe() writes it into the campaign settings
    probe_dir.parent.mkdir(parents=True, exist_ok=True)
    check_existing(probe_dir)  # a resumed campaign must have been created with these settings
    rollouts.SYNTH = TD  # probe() runs SYNTH / "probe_worker.py": this directory's wrapper
    rollouts._run = _run
    rollouts.PROBE_MAX_CONCURRENCY = args.max_concurrency  # probe_limits() reads it at campaign creation

    job = os.environ.get("SLURM_JOB_ID")
    key, start = choose_key(f"start {MODEL}/{name} job {job}")
    if key is None:
        leg_record(campaign=name, event="refused", reason=f"{KEY} budget unknown or below ${START_FLOOR:.0f}",
                   budget=start)
        log(f"teacher-deepswe: {MODEL}/{name}: not starting the leg ({KEY} remaining {start.get('remaining')}, "
            f"start floor ${START_FLOOR:.0f}; no other key is allowed)")
        raise SystemExit(5)
    os.environ["TDW_HUB_KEY_NAME"] = key  # names only; the wrapper reads the value from ~hengyuf/.env itself
    os.environ["TDW_PROMPT"] = PROMPT
    os.environ["TDW_MODEL"] = MODEL
    log(f"teacher-deepswe: {MODEL} ({SPEC['id']}, context {SPEC['context_window']}, prompt {PROMPT}) {name}: "
        f"{len(chosen)} tasks x {repeats}, hub key {key} (remaining ${start.get('remaining')}), "
        f"sandbox {SANDBOX_CREDENTIAL}, max concurrency {args.max_concurrency}, "
        f"broken-reply re-runs {'on' if decision['broken_reply_reruns'] else 'off (flag only)'}")
    leg_record(campaign=name, event="start", hub_key=key, sandbox_credential=SANDBOX_CREDENTIAL, budget=start,
               max_concurrency=args.max_concurrency, prompt=PROMPT, context_window=SPEC["context_window"],
               model_id=SPEC["id"], wave=wave)
    started = time.time()
    records, stop = asyncio.run(probe_with_guard(probe_dir, [Path(t["task_dir"]) for t in chosen], repeats, key, name))
    if isinstance(records, Exception):
        leg_record(campaign=name, event="incomplete", hub_key=key, guard=stop or None, error=str(records)[:300])
        log(f"teacher-deepswe: {MODEL}/{name} not complete this leg ({records}); guard: {stop or 'not triggered'}")
        raise SystemExit(4 if stop else 1)
    task_dirs = {t["task_id"]: Path(t["task_dir"]) for t in chosen}

    # 1. infra recovery (KR4): budget-refused / quota-incident slots, once, in <name>-rec<n>
    recovery = {}
    refused = refused_slots(records, name)
    if refused:
        log(f"teacher-deepswe: {MODEL}/{name}: budget-refused / quota-incident slots {json.dumps(refused)}; "
            "re-running them once")
        by_count = collections.defaultdict(list)
        for task_id, slots in refused.items():
            by_count[len(slots)].append(task_id)
        for count, task_ids in sorted(by_count.items()):
            rec_name = f"{name}-rec{count}"
            check_existing(TD / "probes" / MODEL / rec_name)
            rec_records, rec_stop = asyncio.run(probe_with_guard(TD / "probes" / MODEL / rec_name,
                                                                 [task_dirs[t] for t in sorted(task_ids)], count, key,
                                                                 rec_name))
            if isinstance(rec_records, Exception):
                leg_record(campaign=rec_name, event="incomplete", hub_key=key, guard=rec_stop or None,
                           error=str(rec_records)[:300])
                raise SystemExit(4 if rec_stop else 1)
            for task_id in task_ids:
                replaced = []
                for slot, record in zip(refused[task_id], rec_records[task_id]):
                    index = next(i for i, r in enumerate(records[task_id]) if r["slot"] == slot)
                    replaced.append({"slot": slot, "refused_attempt": records[task_id][index].get("attempt_dir"),
                                     "recovery_campaign": rec_name, "recovery_slot": record["slot"]})
                    records[task_id][index] = dict(record, slot=slot)
                recovery[task_id] = replaced

    # 2. broken-reply re-runs (KR4's rule; models validated in analysis/brokenreply-validate-tr2.json)
    final = {task_id: [compact(r) for r in slot_records] for task_id, slot_records in records.items()}
    first = {task_id: [dict(r) for r in slot_records] for task_id, slot_records in final.items()}
    plan = broken_plan(name, final)
    if not decision["broken_reply_reruns"]:
        plan = dict(plan, slots=[], flag_only=True)
    plan_path = TD / "reruns" / f"{MODEL}-{name}.json"
    if plan_path.exists():
        recorded = json.loads(plan_path.read_text())
        if recorded["slots"] != plan["slots"]:
            leg_record(campaign=name, event="rerun_plan_mismatch", hub_key=key)
            raise SystemExit(f"{plan_path} differs from the broken-reply re-runs found now; refusing to re-plan")
    elif plan["slots"]:
        write_json_atomic(plan_path, dict(plan, planned_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), job=job))
    reruns = {}
    if plan["slots"]:
        log(f"teacher-deepswe: {MODEL}/{name}: {len(plan['slots'])} slot(s) ended by a broken reply; re-running each once")
        by_campaign = collections.defaultdict(list)
        for entry in plan["slots"]:
            by_campaign[entry["rerun_campaign"]].append(entry)
        for brk_name in sorted(by_campaign, key=lambda n: -len(by_campaign[n])):
            entries = by_campaign[brk_name]
            count = int(brk_name.rsplit("-brk", 1)[1])
            task_ids = sorted({entry["task_id"] for entry in entries})
            brk_dir = TD / "probes" / MODEL / brk_name
            check_existing(brk_dir)
            brk_records, brk_stop = asyncio.run(probe_with_guard(brk_dir, [task_dirs[t] for t in task_ids], count,
                                                                 key, brk_name))
            if isinstance(brk_records, Exception):
                leg_record(campaign=brk_name, event="incomplete", hub_key=key, guard=brk_stop or None,
                           error=str(brk_records)[:300])
                raise SystemExit(4 if brk_stop else 1)
            for entry in entries:
                task_id = entry["task_id"]
                rerun = compact(next(r for r in brk_records[task_id] if r["slot"] == entry["rerun_slot"]))
                index = next(i for i, r in enumerate(final[task_id]) if r["slot"] == entry["slot"])
                original = first[task_id][index]
                graded = rerun["status"] in ("pass", "fail")
                outcome = ("infra" if not graded else "broken_again" if rerun["ended_by_broken_reply"]
                           else rerun["status"])
                reruns.setdefault(task_id, []).append({
                    "slot": entry["slot"], "original_attempt": original["attempt_dir"],
                    "original_status": original["status"], "original_last_reply_kind": original["last_reply_kind"],
                    "rerun_campaign": brk_name, "rerun_slot": rerun["slot"], "rerun_attempt": rerun["attempt_dir"],
                    "rerun_status": rerun["status"], "rerun_ended_by_broken_reply": rerun["ended_by_broken_reply"],
                    "outcome": outcome, "replaced": graded})
                provenance = {"rerun_reason": RERUN_REASON, "rerun_campaign": brk_name, "rerun_slot": rerun["slot"]}
                if graded:  # the re-run replaces the original; the original stays in the record
                    final[task_id][index] = dict(rerun, slot=entry["slot"], original_attempt=original, **provenance)
                else:  # no grade from the re-run: keep the original, record the re-run
                    final[task_id][index] = dict(original, rerun_attempt=rerun, rerun_replaced=False, **provenance)

    end = budget(key, f"end {MODEL}/{name} job {job}")
    legs = legs_of(probe_dir)
    extra = sorted(p.name for p in (TD / "probes" / MODEL).glob(f"{name}-*")
                   if re.fullmatch(re.escape(name) + r"-(rec|brk)\d+", p.name))
    out = TD / "results" / f"{MODEL}-{name}.json"
    settings = json.loads((probe_dir / "settings.json").read_text())
    outcomes = collections.Counter(entry["outcome"] for entries in reruns.values() for entry in entries)
    groups = TASKSETS[args.taskset][1] if TASKSETS[args.taskset] else None
    write_json_atomic(out, {
        "model": MODEL, "model_id": SPEC["id"], "spec": SPEC, "taskset": args.taskset, "campaign": name, "wave": wave,
        "group": args.group if groups else None, "groups": groups,
        "tasks": [t["task_id"] for t in chosen], "repeats": repeats,
        "hub_keys_by_leg": [{"job": leg.get("job"), "hub_key": leg.get("hub_key"), "at": leg.get("at")} for leg in legs],
        "hub_keys_by_campaign": {campaign: [{"job": leg.get("job"), "hub_key": leg.get("hub_key"), "at": leg.get("at")}
                                            for leg in legs_of(TD / "probes" / MODEL / campaign)]
                                 for campaign in [name] + extra},
        "hub_key_policy": {"key": KEY, "start_floor_usd": START_FLOOR, "guard_floor_usd": GUARD_FLOOR,
                           "periods": [{"key": period[0], "from": period[1], "until": period[2]}
                                       for period in KEY_PERIODS],
                           "switches": [dict(switch) for switch in KEY_SWITCHES]},
        "sandbox_credential": SANDBOX_CREDENTIAL,
        "max_concurrency": settings.get("max_concurrency_per_model"),
        "context_window": SPEC["context_window"], "max_output_tokens": OUTPUT_CAP,
        "temperature": settings.get("temperature"), "top_p": settings.get("top_p"), "prompt": PROMPT,
        "system_prompt": {"file": str(STOCK_PROMPT_FILE.relative_to(TD)), "sha256": STOCK_PROMPT_SHA256,
                          "note": "OpenCode 1.17.11 stock prompt, no override (no added sentence)"},
        "job": job, "started_at": started, "budget_start": start, "budget_end": end,
        "recovered_budget_refusals": recovery,
        "broken_reply_reruns_enabled": decision["broken_reply_reruns"],
        "broken_reply_rerun_decision": decision.get("basis"),
        "broken_reply_reruns": reruns, "broken_reply_rerun_rule": RERUN_RULE,
        "broken_reply_rerun_outcomes": dict(outcomes),
        "stats_first_attempts": stats(first), "stats_final": stats(final),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "first_attempt_records": {task_id: [r for r in slot_records] for task_id, slot_records in first.items()},
        "records": final})
    leg_record(campaign=name, event="complete", hub_key=key, stats=stats(final), stats_first_attempts=stats(first),
               recovered=sum(map(len, recovery.values())), broken_reruns=dict(outcomes))
    log(f"teacher-deepswe: {MODEL}/{name} done: first attempts {json.dumps(stats(first))}; final "
        f"{json.dumps(stats(final))}; broken-reply re-runs {json.dumps(dict(outcomes))}")


if __name__ == "__main__":
    main()
