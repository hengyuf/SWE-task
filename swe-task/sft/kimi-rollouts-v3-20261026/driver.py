#!/usr/bin/env python3
"""kimi-rollouts-v3 (2026-10-26): one Kimi-K3 probe campaign = Kimi-K3 x a task subset x N rollouts per task, with
(a) OpenCode's context window limited to 262,144 tokens and (b) one sentence asking for reasoning before every tool call
added to the agent's system prompt (user, 2026-10-26).

    (cd swe-task/synth && atp/runtime/controller/bin/python3 -B <KR>/driver.py --taskset TASKSET [--group K]
        [--repeats N] [--max-concurrency C])

The synth pipeline's own rollout and grading path (rollouts.probe: ATP campaign, OpenCode 1.17.11 harness, each task's
own grader), as teacher-rollouts-v2's driver.py, with these differences (all in memory; settings.json, the synth code,
hub_key_active.txt, S/atp/.env and S/secrets are never edited):
* model "kimi" = settings.json's spec (nvidia/moonshotai/kimi-k3, reasoning_effort max) + context_window 262144: ATP's
  build_config gives OpenCode limit.context = limit.input = 262144 (limit.output stays max_output_tokens = 131072), so
  OpenCode compacts at 262144 - 20000 = 242,144 prompt tokens, as for the local student. The alias stays "kimi", so ATP
  streams the hub replies exactly as for the earlier Kimi rollouts (run_deepswe: ATP_DEEPSWE_UPSTREAM_STREAM).
* system prompt: probe_worker.py (this directory; rollouts.SYNTH is pointed here) adds the sentence via OpenCode's
  agent.build.prompt (see its docstring and prompt/PROMPT.json).
* hub key by NAME, per leg (user, 2026-10-26: NVAPI_KEY_NEW_2 first, NVAPI_KEY_NEW_3 once NEW_2 is used up): a leg
  starts on NEW_2 while NEW_2 has >= SWITCH_FLOOR left, else on NEW_3. A guard re-checks the leg's key every
  GUARD_INTERVAL s and stops the leg (SIGTERM to the ATP worker: in-flight rollouts are interrupted and stay retryable)
  once that key has < GUARD_FLOOR left, so the next leg resumes on NEW_3 instead of burning slots on budget refusals.
  A slot that still failed on a budget refusal (ATP makes a hub 4xx fatal and never retries it) is re-run once in a
  recovery campaign of the same settings (probes/kimi/<name>-rec<n>) and replaces the refused slot in the results.
  The same applies to the slots listed in quota-incident-slots.json (in flight when hengyuf's Lustre user quota filled,
  2026-10-26 ~11:13 PDT; ATP leaves such slots needs_review, which it never retries) that are still infra.
* sandbox credential by NAME: tasksets of Baihe's tasks (baihe*) use BAIHE_SANDBOX_API_KEY (secrets/baihe-opensandbox.env,
  make_baihe_secrets.py) for every sandbox of the campaign; all others use S/atp/.env's CELL3_SANDBOX_API_KEY. A campaign
  never mixes them (checked on resume).
Tasksets (task order fixed; group K of G = tasks[K::G]):
  pilot        1 rollout of PILOT_TASK (one of the 50)                    sandbox hengyuf
  sel50        the 50 train-acc-v2 tasks (TA/tasks.json, seed 20261024)    sandbox hengyuf, 5 groups
  rest47       the other 47 accessible hengyuf v2 tasks (TA/pool.json)    sandbox hengyuf, 5 groups
  baihe-pilot  1 rollout of BAIHE_PILOT_TASK                              sandbox baihe
  ctrl-old     CONTROL ARM (user, 2026-10-26 ~13:40): CTRL_TASKS (4 of the 50) x 5 with the PREVIOUS Kimi settings: no
               sentence (OpenCode's stock prompt; probe_worker KR3_PROMPT=stock) and settings.json's spec unchanged
               (no context_window override: OpenCode limit 1,048,576); key, sandbox, harness, grader as for sel50
  baihe        Baihe's 97 v2 tasks (baihe_v2_tasks.tsv)                    sandbox baihe, 8 groups
Writes probes/kimi/<name>/ (name = pilot | baihe-pilot | <taskset>-g<K>), appends one line per leg to legs.jsonl, and
when the campaign is complete results/kimi-<name>.json (compact per-slot records, the teacher-rollouts-v2 shape).
Resumable: rerun the same command in a fresh CPU job (the next leg).
"""
import argparse
import asyncio
import collections
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time

KR = Path(__file__).resolve().parent
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
TA = S / "sft/train-acc-v2-20261024"
BAIHE_TSV = S / "sft/baihe-snapshot-access-20261026/baihe_v2_tasks.tsv"
SYNTH = S / "synth"
sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != KR]
sys.path.insert(0, str(SYNTH))
import common  # noqa: E402
from common import SETTINGS, log  # noqa: E402
import rollouts  # noqa: E402

MODEL = "kimi"
BASE_SPEC = {"id": "nvidia/moonshotai/kimi-k3", "sampling_overrides": {"reasoning_effort": "max"}}
CONTEXT_WINDOW = 262144
SPEC = dict(BASE_SPEC, context_window=CONTEXT_WINDOW)
KEYS = ("NVAPI_KEY_NEW_2", "NVAPI_KEY_NEW_3")
SWITCH_FLOOR = 100.0    # USD: a leg starts on NEW_3 once NEW_2 has less than this left
GUARD_FLOOR = 40.0      # USD: a running leg stops once its key has less than this left
GUARD_INTERVAL = 300.0  # s between the guard's budget checks (one 4-token GLM call each)
PILOT_TASK = "syn-aiohttp-cand-c05-r1b0-7d9ec1f5"  # near the median kimi rollout size of the 50 (TR pilot_pick.json)
BAIHE_PILOT_TASK = "syn-citty-cand-c01-r1b0-6fa08bef"
# Control tasks: the 4 sel50 tasks with the most new-arm (sel50-g3) slots still to run when the control was launched
# (13:40: 4, 4, 3, 3 pending), so both arms run the same tasks in the same hours; chosen without looking at their
# degenerate-reply counts (selecting on the outcome would bias the comparison).
CTRL_TASKS = ("syn-aiohttp-cand-c05-r1b0-7d9ec1f5", "syn-dacite-cand-c02-r1b4-325c5f9d",
              "syn-vscode-sqltools-cand-c05-r1b3-911416cb", "syn-yj-cand-c01-7e8e179e")
GROUPS = {"sel50": 5, "rest47": 5, "baihe": 8}
SANDBOX_OF = {"pilot": "hengyuf", "sel50": "hengyuf", "rest47": "hengyuf", "baihe-pilot": "baihe", "baihe": "baihe",
              "ctrl-old": "hengyuf"}
SANDBOX_CREDENTIAL = {"hengyuf": "CELL3_SANDBOX_API_KEY", "baihe": "BAIHE_SANDBOX_API_KEY"}
BUDGET_REFUSAL = re.compile(r"(?i)budget[_ ]exceeded|budget has been exceeded|exceededbudget|max budget")
WORKERS = []


def hengyuf_tasks():
    tasks = json.loads((TA / "tasks.json").read_text())["tasks"]
    pool = json.loads((TA / "pool.json").read_text())["tasks"]
    chosen = {task["task_id"] for task in tasks}
    return ([{"task_id": t["task_id"], "task_dir": t["task_dir"]} for t in tasks],
            [{"task_id": t["task_id"], "task_dir": t["task_dir"]} for t in pool if t["task_id"] not in chosen])


def baihe_tasks():
    lines = BAIHE_TSV.read_text().splitlines()
    header = lines[0].split("\t")
    rows = [dict(zip(header, line.split("\t"))) for line in lines[1:] if line.strip()]
    return [{"task_id": row["task_id"], "task_dir": row["task_dir"], "snapshot": row["snapshot"]} for row in rows]


def taskset(name, group):
    sel50, rest47 = hengyuf_tasks()
    assert len(sel50) == 50 and len(rest47) == 47, (len(sel50), len(rest47))
    if name == "pilot":
        return [t for t in sel50 if t["task_id"] == PILOT_TASK], "pilot", 1
    if name == "baihe-pilot":
        return [t for t in baihe_tasks() if t["task_id"] == BAIHE_PILOT_TASK], "baihe-pilot", 1
    if name == "ctrl-old":
        chosen = [t for t in sel50 if t["task_id"] in CTRL_TASKS]
        assert len(chosen) == len(CTRL_TASKS)
        return chosen, "ctrl-old", 5
    tasks = {"sel50": sel50, "rest47": rest47, "baihe": baihe_tasks() if name == "baihe" else None}[name]
    if name == "baihe":
        assert len(tasks) == 97, len(tasks)
    assert 0 <= group < GROUPS[name]
    return tasks[group::GROUPS[name]], f"{name}-g{group}", 5


def budget(key: str, tag: str) -> dict:
    """A key's budget via key_budget.py (prints the key NAME and numbers only)."""
    try:
        result = subprocess.run(["/usr/bin/python3", "-I", "-B", str(KR / "key_budget.py"), "--key-name", key,
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
        log(f"kimi-rollouts-v3: budget {json.dumps(record)}")
        if record.get("remaining") is not None:
            break
        if attempt + 1 < tries:
            time.sleep(60)
    return record


def choose_key(tag: str) -> tuple:
    """(key name, its budget record): NEW_2 while it has >= SWITCH_FLOOR left, else NEW_3 (refuses below SWITCH_FLOOR)."""
    first = known_budget(KEYS[0], tag)
    if first.get("remaining") is None:
        raise SystemExit(f"{KEYS[0]} budget unknown ({first}); not starting the leg")
    if first["remaining"] >= SWITCH_FLOOR:
        return KEYS[0], first
    second = known_budget(KEYS[1], tag)
    if second.get("remaining") is None:
        raise SystemExit(f"{KEYS[1]} budget unknown ({second}); not starting the leg")
    if second["remaining"] < max(GUARD_FLOOR, SWITCH_FLOOR):
        raise SystemExit(f"{KEYS[0]} and {KEYS[1]} both below ${SWITCH_FLOOR:.0f}; not starting the leg")
    return KEYS[1], second


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
        record = await asyncio.to_thread(budget, key, f"guard {name} job {os.environ.get('SLURM_JOB_ID')}")
        remaining = record.get("remaining")
        if remaining is not None and remaining < GUARD_FLOOR:
            stop.update(reason=f"{key} remaining ${remaining:.2f} < ${GUARD_FLOOR:.0f}", at=time.time(), budget=record)
            log(f"kimi-rollouts-v3: GUARD {stop['reason']}: stopping the leg (in-flight rollouts stay retryable)")
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


def quota_incident_slots(name: str) -> set:
    """{(task_id, slot)} of this campaign listed in quota-incident-slots.json (in flight when the Lustre user quota
    filled on 2026-10-26 ~11:13-11:16 PDT; ATP left them needs_review/active)."""
    path = KR / "quota-incident-slots.json"
    if not path.exists():
        return set()
    return {(entry["task_id"], entry["slot"]) for entry in json.loads(path.read_text())["slots"]
            if entry["campaign"] == name}


def refused_slots(records: dict, name: str = "") -> dict:
    """{task_id: [slot, ...]} of infra slots to re-run once: the final attempt got a hub budget refusal (a fatal 4xx),
    or the slot is listed as a casualty of the 2026-10-26 quota incident (quota_incident_slots)."""
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
    return ({key: record.get(key) for key in ("slot", "status", "reward", "f2p", "p2p", "agent_error", "verifier_error",
                                              "attempt_dir")} | {"patch_chars": len(record.get("patch") or "")})


def leg_record(**fields):
    with open(KR / "legs.jsonl", "a") as stream:
        stream.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "job": os.environ.get("SLURM_JOB_ID"),
                                 **fields}) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskset", choices=sorted(SANDBOX_OF), required=True)
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--max-concurrency", type=int, default=8,
                        help="rollouts in flight for this campaign (fixed at campaign creation; pipeline cap 32)")
    args = parser.parse_args()
    chosen, name, repeats = taskset(args.taskset, args.group)
    repeats = args.repeats or repeats
    assert chosen, f"no tasks for {args.taskset}"
    sandbox = SANDBOX_OF[args.taskset]
    probe_dir = KR / "probes" / MODEL / name
    probe_dir.parent.mkdir(parents=True, exist_ok=True)

    found = SETTINGS["models"][MODEL]
    assert found == BASE_SPEC, f"settings.json kimi spec changed: {found}"
    control = args.taskset == "ctrl-old"
    spec = dict(BASE_SPEC) if control else dict(SPEC)  # the control keeps settings.json's spec (1M window)
    SETTINGS["models"][MODEL] = spec  # in memory only: rollouts.probe() writes it into the campaign settings
    assert SETTINGS["max_output_tokens"] == 131072 and SETTINGS["context_window"] == 1048576
    if (probe_dir / "settings.json").exists():  # a resumed campaign must have been created with these settings
        created = json.loads((probe_dir / "settings.json").read_text())
        assert created["models"] == {MODEL: spec}, f"{probe_dir} was created with {created['models']}"
        legs = [json.loads(line) for line in (probe_dir / "kr3-legs.jsonl").read_text().splitlines() if line.strip()] \
            if (probe_dir / "kr3-legs.jsonl").exists() else []
        assert all(leg["sandbox"] == sandbox for leg in legs), f"{probe_dir}: earlier legs used another sandbox"
    assert (KR / "probe_worker.py").is_file()
    rollouts.SYNTH = KR  # probe() runs SYNTH / "probe_worker.py": this directory's wrapper
    rollouts._run = _run
    assert 1 <= args.max_concurrency <= 32
    rollouts.PROBE_MAX_CONCURRENCY = args.max_concurrency  # probe_limits() reads it at campaign creation
    if sandbox == "baihe" and not (KR / "secrets" / "baihe-opensandbox.env").is_file():
        raise SystemExit("secrets/baihe-opensandbox.env missing: run make_baihe_secrets.py")
    for task in chosen:
        assert (Path(task["task_dir"]) / "task.toml").is_file(), task

    job = os.environ.get("SLURM_JOB_ID")
    key, start = choose_key(f"start {name} job {job}")
    os.environ["KR3_HUB_KEY_NAME"] = key  # names only; the wrapper reads the value from ~hengyuf/.env itself
    os.environ["KR3_SANDBOX"] = sandbox
    os.environ["KR3_PROMPT"] = "stock" if control else "sentence"
    log(f"kimi-rollouts-v3: kimi ({spec['id']}, context {spec.get('context_window', SETTINGS['context_window'])}, "
        f"prompt {os.environ['KR3_PROMPT']}) {name}: {len(chosen)} tasks x {repeats}, "
        f"hub key {key} (remaining ${start.get('remaining')}), sandbox {SANDBOX_CREDENTIAL[sandbox]}, "
        f"max concurrency {args.max_concurrency}")
    leg_record(campaign=name, event="start", hub_key=key, sandbox_credential=SANDBOX_CREDENTIAL[sandbox],
               budget=start, max_concurrency=args.max_concurrency)
    started = time.time()
    records, stop = asyncio.run(probe_with_guard(probe_dir, [Path(t["task_dir"]) for t in chosen], repeats, key, name))
    if isinstance(records, Exception):
        leg_record(campaign=name, event="incomplete", hub_key=key, guard=stop or None, error=str(records)[:300])
        log(f"kimi-rollouts-v3: {name} not complete this leg ({records}); guard: {stop or 'not triggered'}")
        raise SystemExit(4 if stop else 1)

    recovery = {}
    refused = refused_slots(records, name)
    if refused:
        log(f"kimi-rollouts-v3: {name}: budget-refused / quota-incident slots {json.dumps(refused)}; re-running them once")
        by_count = collections.defaultdict(list)
        for task_id, slots in refused.items():
            by_count[len(slots)].append(task_id)
        task_dirs = {t["task_id"]: Path(t["task_dir"]) for t in chosen}
        for count, task_ids in sorted(by_count.items()):
            rec_name = f"{name}-rec{count}"
            rec_records, rec_stop = asyncio.run(probe_with_guard(KR / "probes" / MODEL / rec_name,
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

    end = budget(key, f"end {name} job {job}")
    legs = [json.loads(line) for line in (probe_dir / "kr3-legs.jsonl").read_text().splitlines() if line.strip()]
    out = KR / "results" / f"{MODEL}-{name}.json"
    out.parent.mkdir(exist_ok=True)
    tmp = out.with_name("." + out.name + ".tmp")
    settings = json.loads((probe_dir / "settings.json").read_text())
    tmp.write_text(json.dumps({
        "model": MODEL, "model_id": spec["id"], "spec": spec, "taskset": args.taskset,
        "group": args.group if args.taskset in GROUPS else None, "groups": GROUPS.get(args.taskset),
        "tasks": [t["task_id"] for t in chosen], "repeats": repeats,
        "hub_keys_by_leg": [{"job": leg.get("job"), "hub_key": leg.get("hub_key"), "at": leg.get("at")} for leg in legs],
        "sandbox_credential": SANDBOX_CREDENTIAL[sandbox],
        "max_concurrency": settings.get("max_concurrency_per_model"),
        "context_window": spec.get("context_window", SETTINGS["context_window"]), "prompt": os.environ["KR3_PROMPT"],
        "build_prompt_sha256": None if control else
        json.loads((KR / "prompt" / "PROMPT.json").read_text())["build_prompt"]["sha256"],
        "job": job, "started_at": started, "budget_start": start, "budget_end": end, "recovered_budget_refusals": recovery,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "records": {task_id: [compact(r) for r in slot_records] for task_id, slot_records in records.items()}},
        indent=1) + "\n")
    os.replace(tmp, out)
    stats = rollouts.pass_stats([record for slot_records in records.values() for record in slot_records])
    leg_record(campaign=name, event="complete", hub_key=key, stats=stats, recovered=sum(map(len, recovery.values())))
    log(f"kimi-rollouts-v3: {name} done: {json.dumps(stats)}")


if __name__ == "__main__":
    main()
