#!/usr/bin/env python3
"""kimi-rollouts-v5 (2026-10-28): one Kimi-K3 probe campaign = Kimi-K3 x a task subset x 5 rollouts per task, with
OpenCode's context window limited to 262,144 tokens and OpenCode's STOCK system prompt, on the second official batch
(user, 2026-10-28, verbatim: "这组rollout 跑完之后,再给official的第二批(找rank2-5 repo 的那一批) 里的accepted tasks 和标记为strong
的task 每个task 额外做5个kimi 的256 k context rollout"). A copy of kimi-rollouts-v4's driver
(sft/kimi-rollouts-v4-stock256k-20261028, read-only) with the same settings; only the task set (official2-tasks.json,
official2_tasks.py), the hengyuf-only sandbox credential and the kr5- names differ.

    (cd swe-task/synth && atp/runtime/controller/bin/python3 -B <KR>/driver.py --taskset official2 --group K
        [--max-concurrency C] [--dry-run])

The synth pipeline's own rollout and grading path (rollouts.probe: ATP campaign, OpenCode 1.17.11 harness, each task's
own installed grader), with kimi-rollouts-v4's settings (all in memory; settings.json, the synth code,
hub_key_active.txt, S/atp/.env and S/secrets are never edited):
* model "kimi" = settings.json's spec (nvidia/moonshotai/kimi-k3, reasoning_effort max) + context_window 262144: ATP's
  build_config gives OpenCode limit.context = limit.input = 262144 (limit.output stays max_output_tokens = 131072), so
  OpenCode compacts at 262144 - 20000 = 242,144 prompt tokens. The alias stays "kimi", so ATP streams the hub replies
  exactly as for the earlier Kimi rollouts (run_deepswe: ATP_DEEPSWE_UPSTREAM_STREAM).
* system prompt: OpenCode's stock prompt, unchanged (probe_worker.py, KR3_PROMPT=stock: no agent.build.prompt
  override, so OpenCode uses SystemPrompt.provider() = prompt/opencode-1.17.11-default.txt for "dummy_model").
* hub key by NAME, per leg (user, 2026-10-26: NVAPI_KEY_NEW_2 first, NVAPI_KEY_NEW_3 once NEW_2 is used up): a leg
  starts on NEW_2 while NEW_2 has >= SWITCH_FLOOR left, else on NEW_3. A guard re-checks the leg's key every
  GUARD_INTERVAL s and stops the leg (SIGTERM to the ATP worker: in-flight rollouts are interrupted and stay retryable)
  once that key has < GUARD_FLOOR left, so the next leg resumes on NEW_3 instead of burning slots on budget refusals.
  A slot that still failed on a budget refusal (ATP makes a hub 4xx fatal and never retries it) is re-run once in a
  recovery campaign of the same settings (probes/kimi/<name>-rec<n>) and replaces the refused slot in the results.
  The same applies to slots listed in quota-incident-slots.json (none in v5; the mechanism stays for a future Lustre
  quota incident: ATP leaves such slots needs_review, which it never retries) that are still infra.
* broken-reply re-runs (user, 2026-10-28, for v4, verbatim: "这次若遇到被坏回复终止的 slot, 可以重跑1遍"; main applies the
  same rule to this follow-on batch): once the campaign's slots are terminal (and the infra recoveries above are done),
  every slot whose final attempt was ENDED by a broken Kimi reply (brokenreply.py = degenerate.py's definition: the
  LAST 200 reply is [PAD] / special-token / empty and has no tool call, so the session stopped there) is re-run exactly
  once in a recovery campaign of the same settings, key order and sandbox credential: probes/kimi/<name>-brk<n>
  (n = the task's number of such slots; reruns/<name>.json is the plan, written before the first re-run starts and
  checked on every later leg). A graded re-run (pass / fail) REPLACES the original slot in the results; the record
  keeps the original as "original_attempt" with "rerun_reason" = "ended_by_broken_reply". A re-run that ends on a
  broken reply again is kept as it is (never a third attempt); a re-run that ends infra (no grade) leaves the original
  in place, with the re-run kept as "rerun_attempt". Slots that merely contain a broken reply but ended normally are
  not re-run, and a <name>-brk<n> campaign's own slots are never re-run (no budget-refusal recovery either).
* sandbox credential: hengyuf's only. The official2 tasks were built by hengyuf's pipeline, so every sandbox (agent,
  verifier, cleanup) uses S/atp/.env's CELL3_SANDBOX_API_KEY, as v4's hengyuf tasksets (Baihe's credential is not used;
  probe_worker.py has no Baihe path).
Tasksets (task order fixed = official2-tasks.json, sorted by run then task_id; group K of G = tasks[K::G]):
  official2    the 33 official2 tasks (32 accepted + the strong-labelled ones; official2_tasks.py)   4 groups
  pilot        1 rollout of PILOT_TASK (one of the 33), campaign "pilot", as kimi-rollouts-v4's pilot (added
               2026-10-29 for main's pre-launch check; an extra rollout outside the 165)
Every leg first checks that each task dir still has the digests frozen in official2-tasks.json (instruction.md,
task.toml, every tests/ file), so a task changed afterwards (e.g. a re-installed grader) stops the batch instead of
mixing graders between the main and the rec / brk campaigns. --dry-run stops after the pre-flight checks and prints the
campaign settings that rollouts.probe() would write (no hub call, no Slurm, no write).
Writes probes/kimi/<name>/ (name = official2-g<K> | pilot), appends one line per leg to legs.jsonl, and when the campaign is
complete results/kimi-<name>.json (compact per-slot records, the teacher-rollouts-v2 shape, plus the re-run
provenance). Resumable: rerun the same command in a fresh CPU job (the next leg).
"""
import argparse
import asyncio
import collections
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

KR = Path(__file__).resolve().parent
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
TASKS_FILE = KR / "official2-tasks.json"  # official2_tasks.py --write (frozen task list + digests)
TASKS_TOTAL = 33
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
PROMPT = "stock"  # OpenCode's own system prompt (probe_worker.py refuses anything else)
STOCK_PROMPT_FILE = KR / "prompt" / "opencode-1.17.11-default.txt"
STOCK_PROMPT_SHA256 = "962fbf3cb3ec659c9a5244425ee2e7bb141ad4428f489a630a7738566880dc6a"
KEYS = ("NVAPI_KEY_NEW_2", "NVAPI_KEY_NEW_3")
SWITCH_FLOOR = 100.0    # USD: a leg starts on NEW_3 once NEW_2 has less than this left
GUARD_FLOOR = 40.0      # USD: a running leg stops once its key has less than this left
GUARD_INTERVAL = 300.0  # s between the guard's budget checks (one 4-token GLM call each)
REPEATS = 5
PILOT_TASK = "syn-markdown-cand-c04-r1b0-3b72285b"  # official2-obsidian, accepted; build-time Kimi (1M) 5/5 pass,
# median 27.6 min per rollout = the median of the 33 tasks' medians (27.4 min; scratch analysis 2026-10-29)
GROUPS = {"official2": 4}
SANDBOX_OF = {"official2": "hengyuf", "pilot": "hengyuf"}
SANDBOX_CREDENTIAL = {"hengyuf": "CELL3_SANDBOX_API_KEY"}
BUDGET_REFUSAL = re.compile(r"(?i)budget[_ ]exceeded|budget has been exceeded|exceededbudget|max budget")
RERUN_REASON = "ended_by_broken_reply"
RERUN_RULE = ("user 2026-10-28 (for v4): 这次若遇到被坏回复终止的 slot, 可以重跑1遍; applied by main to this follow-on batch "
              "(one re-run per slot ended by a broken reply)")
LEGS_FILE = "kr5-legs.jsonl"  # per campaign, written by probe_worker.py at every ATP worker start
WORKERS = []


def _load(name: str):
    """A module of this directory by path (KR is not on sys.path, so nothing here can shadow a synth/ATP module)."""
    spec = importlib.util.spec_from_file_location(f"kr5_{name}", KR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BROKEN = _load("brokenreply")  # degenerate.py's classify(), applied to the last 200 reply of an attempt


def official2_tasks() -> list:
    """The frozen task list (official2-tasks.json): [{task_id, run, why, task_dir, digests, ...}] in file order."""
    data = json.loads(TASKS_FILE.read_text())
    tasks = data["tasks"]
    assert len(tasks) == TASKS_TOTAL == len({t["task_id"] for t in tasks}), (len(tasks), TASKS_TOTAL)
    assert all(not t.get("problems") for t in tasks), [t["task_id"] for t in tasks if t.get("problems")]
    return tasks


def task_changes(tasks) -> list:
    """Tasks whose dir no longer has the digests frozen in official2-tasks.json (official2_tasks.digests())."""
    tool = _load("official2_tasks")
    changed = []
    for task in tasks:
        try:
            now = tool.digests(Path(task["task_dir"]))
        except OSError as error:
            changed.append({"task_id": task["task_id"], "error": f"{type(error).__name__}: {error}"})
            continue
        if now != task["digests"]:
            changed.append({"task_id": task["task_id"],
                            "files": sorted(k for k in set(now) | set(task["digests"]) if now.get(k) != task["digests"].get(k))})
    return changed


def taskset(name, group):
    tasks = official2_tasks()
    if name == "pilot":
        chosen = [t for t in tasks if t["task_id"] == PILOT_TASK]
        assert len(chosen) == 1, PILOT_TASK
        return chosen, "pilot", 1
    assert name in GROUPS and 0 <= group < GROUPS[name], (name, group)
    return tasks[group::GROUPS[name]], f"{name}-g{group}", REPEATS


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
        log(f"kimi-rollouts-v5: budget {json.dumps(record)}")
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
            log(f"kimi-rollouts-v5: GUARD {stop['reason']}: stopping the leg (in-flight rollouts stay retryable)")
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


def check_existing(probe_dir: Path, sandbox: str) -> None:
    """A campaign created earlier (main, rec or brk) must have this batch's model spec, sandbox and stock prompt."""
    if (probe_dir / "settings.json").exists():
        created = json.loads((probe_dir / "settings.json").read_text())
        assert created["models"] == {MODEL: SPEC}, f"{probe_dir} was created with {created['models']}"
    path = probe_dir / LEGS_FILE
    legs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    assert all(leg["sandbox"] == sandbox for leg in legs), f"{probe_dir}: earlier legs used another sandbox"
    assert all(leg.get("prompt") == PROMPT for leg in legs), f"{probe_dir}: earlier legs used another prompt"


def quota_incident_slots(name: str) -> set:
    """{(task_id, slot)} of this campaign listed in quota-incident-slots.json (slots in flight when a Lustre quota
    incident hit; ATP leaves them needs_review/active). kimi-rollouts-v5 ships no such file."""
    path = KR / "quota-incident-slots.json"
    if not path.exists():
        return set()
    return {(entry["task_id"], entry["slot"]) for entry in json.loads(path.read_text())["slots"]
            if entry["campaign"] == name}


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
    """The teacher-rollouts-v2 compact record + this attempt's broken-ending verdict (brokenreply.py)."""
    verdict = BROKEN.broken_end(record.get("attempt_dir"))
    return ({key: record.get(key) for key in ("slot", "status", "reward", "f2p", "p2p", "agent_error", "verifier_error",
                                              "attempt_dir")}
            | {"patch_chars": len(record.get("patch") or ""), "ended_by_broken_reply": verdict["ended"],
               "last_reply_kind": verdict["kind"]})


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
    return {"campaign": name, "rule": RERUN_RULE, "reason": RERUN_REASON, "slots": slots}


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n")
    os.replace(tmp, path)


def leg_record(**fields):
    with open(KR / "legs.jsonl", "a") as stream:
        stream.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "job": os.environ.get("SLURM_JOB_ID"),
                                 **fields}) + "\n")


def legs_of(probe_dir: Path) -> list:
    path = probe_dir / LEGS_FILE
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskset", choices=sorted(SANDBOX_OF), required=True)
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument("--max-concurrency", type=int, default=8,
                        help="rollouts in flight for this campaign (fixed at campaign creation; pipeline cap 32)")
    parser.add_argument("--dry-run", action="store_true",
                        help="pre-flight checks only: print the campaign settings rollouts.probe() would write "
                             "(no hub call, no Slurm, nothing written)")
    args = parser.parse_args()
    chosen, name, repeats = taskset(args.taskset, args.group)
    assert chosen, f"no tasks for {args.taskset}"
    sandbox = SANDBOX_OF[args.taskset]
    probe_dir = KR / "probes" / MODEL / name

    found = SETTINGS["models"][MODEL]
    assert found == BASE_SPEC, f"settings.json kimi spec changed: {found}"
    spec = dict(SPEC)
    SETTINGS["models"][MODEL] = spec  # in memory only: rollouts.probe() writes it into the campaign settings
    assert SETTINGS["max_output_tokens"] == 131072 and SETTINGS["context_window"] == 1048576
    check_existing(probe_dir, sandbox)  # a resumed campaign must have been created with these settings
    assert (KR / "probe_worker.py").is_file()
    assert hashlib.sha256(STOCK_PROMPT_FILE.read_bytes()).hexdigest() == STOCK_PROMPT_SHA256, "stock prompt file changed"
    rollouts.SYNTH = KR  # probe() runs SYNTH / "probe_worker.py": this directory's wrapper
    rollouts._run = _run
    assert 1 <= args.max_concurrency <= 32
    rollouts.PROBE_MAX_CONCURRENCY = args.max_concurrency  # probe_limits() reads it at campaign creation
    for task in chosen:
        assert (Path(task["task_dir"]) / "task.toml").is_file(), task
    changed = task_changes(chosen)
    if changed:
        if not args.dry_run:
            leg_record(campaign=name, event="task_changed", changed=changed)
        raise SystemExit(f"task dirs differ from official2-tasks.json (frozen digests): {json.dumps(changed)[:2000]}; "
                         "review before running")
    rows = [rollouts.dataset_row(Path(t["task_dir"])) for t in chosen]  # the rows rollouts.probe() writes
    assert [row["task_id"] for row in rows] == [t["task_id"] for t in chosen]
    if args.dry_run:
        maximum, initial = rollouts.probe_limits(len(rows) * repeats)
        settings = dict(SETTINGS, expected_tasks=len(rows), repeats=repeats, max_concurrency_per_model=maximum,
                        initial_concurrency_per_model=initial, models={MODEL: SETTINGS["models"][MODEL]})
        print(json.dumps({"dry_run": True, "campaign": name, "probe_dir": str(probe_dir),
                          "campaign_exists": probe_dir.exists(), "tasks": [t["task_id"] for t in chosen],
                          "repeats": repeats, "slots": len(rows) * repeats,
                          "sandbox": sandbox, "sandbox_credential": SANDBOX_CREDENTIAL[sandbox], "prompt": PROMPT,
                          "system_prompt_sha256": STOCK_PROMPT_SHA256, "worker": str(rollouts.SYNTH / "probe_worker.py"),
                          "rows": [{"task_id": row["task_id"], "image": row["image"],
                                    "verifier_timeout": row["_atp_verifier_timeout"],
                                    "instruction_chars": len(row["responses_create_params"]["input"][0]["content"])}
                                   for row in rows],
                          "settings": settings}, indent=1, sort_keys=True))
        return
    probe_dir.parent.mkdir(parents=True, exist_ok=True)

    job = os.environ.get("SLURM_JOB_ID")
    key, start = choose_key(f"start {name} job {job}")
    os.environ["KR3_HUB_KEY_NAME"] = key  # names only; the wrapper reads the value from ~hengyuf/.env itself
    os.environ["KR3_SANDBOX"] = sandbox
    os.environ["KR3_PROMPT"] = PROMPT
    log(f"kimi-rollouts-v5: kimi ({spec['id']}, context {spec['context_window']}, prompt {PROMPT}) {name}: "
        f"{len(chosen)} tasks x {repeats}, hub key {key} (remaining ${start.get('remaining')}), "
        f"sandbox {SANDBOX_CREDENTIAL[sandbox]}, max concurrency {args.max_concurrency}")
    leg_record(campaign=name, event="start", hub_key=key, sandbox_credential=SANDBOX_CREDENTIAL[sandbox],
               budget=start, max_concurrency=args.max_concurrency, prompt=PROMPT,
               context_window=spec["context_window"])
    started = time.time()
    records, stop = asyncio.run(probe_with_guard(probe_dir, [Path(t["task_dir"]) for t in chosen], repeats, key, name))
    if isinstance(records, Exception):
        leg_record(campaign=name, event="incomplete", hub_key=key, guard=stop or None, error=str(records)[:300])
        log(f"kimi-rollouts-v5: {name} not complete this leg ({records}); guard: {stop or 'not triggered'}")
        raise SystemExit(4 if stop else 1)
    task_dirs = {t["task_id"]: Path(t["task_dir"]) for t in chosen}

    # 1. infra recovery (as v3): budget-refused / quota-incident slots, once, in <name>-rec<n>
    recovery = {}
    refused = refused_slots(records, name)
    if refused:
        log(f"kimi-rollouts-v5: {name}: budget-refused / quota-incident slots {json.dumps(refused)}; re-running them once")
        by_count = collections.defaultdict(list)
        for task_id, slots in refused.items():
            by_count[len(slots)].append(task_id)
        for count, task_ids in sorted(by_count.items()):
            rec_name = f"{name}-rec{count}"
            check_existing(KR / "probes" / MODEL / rec_name, sandbox)
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

    # 2. broken-reply re-runs (user, 2026-10-28): one re-run per slot ENDED by a broken reply, in <name>-brk<n>
    final = {task_id: [compact(r) for r in slot_records] for task_id, slot_records in records.items()}
    first = {task_id: [dict(r) for r in slot_records] for task_id, slot_records in final.items()}
    plan = broken_plan(name, final)
    plan_path = KR / "reruns" / f"{name}.json"
    if plan_path.exists():
        recorded = json.loads(plan_path.read_text())
        if recorded["slots"] != plan["slots"]:
            leg_record(campaign=name, event="rerun_plan_mismatch", hub_key=key)
            raise SystemExit(f"{plan_path} differs from the broken-reply re-runs found now; refusing to re-plan")
    elif plan["slots"]:
        write_json_atomic(plan_path, dict(plan, planned_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), job=job))
    reruns = {}
    if plan["slots"]:
        log(f"kimi-rollouts-v5: {name}: {len(plan['slots'])} slot(s) ended by a broken reply; re-running each once")
        by_campaign = collections.defaultdict(list)
        for entry in plan["slots"]:
            by_campaign[entry["rerun_campaign"]].append(entry)
        for brk_name in sorted(by_campaign, key=lambda n: -len(by_campaign[n])):
            entries = by_campaign[brk_name]
            count = int(brk_name.rsplit("-brk", 1)[1])
            task_ids = sorted({entry["task_id"] for entry in entries})
            brk_dir = KR / "probes" / MODEL / brk_name
            check_existing(brk_dir, sandbox)
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

    end = budget(key, f"end {name} job {job}")
    legs = legs_of(probe_dir)
    extra = sorted(p.name for p in (KR / "probes" / MODEL).glob(f"{name}-*")
                   if re.fullmatch(re.escape(name) + r"-(rec|brk)\d+", p.name))
    out = KR / "results" / f"{MODEL}-{name}.json"
    settings = json.loads((probe_dir / "settings.json").read_text())
    outcomes = collections.Counter(entry["outcome"] for entries in reruns.values() for entry in entries)
    write_json_atomic(out, {
        "model": MODEL, "model_id": spec["id"], "spec": spec, "taskset": args.taskset,
        "group": args.group if args.taskset in GROUPS else None, "groups": GROUPS.get(args.taskset),
        "tasks": [t["task_id"] for t in chosen], "repeats": repeats,
        "task_list": {"file": TASKS_FILE.name, "sha256": hashlib.sha256(TASKS_FILE.read_bytes()).hexdigest(),
                      "why": {t["task_id"]: t["why"] for t in chosen},
                      "task_dirs": {t["task_id"]: t["task_dir"] for t in chosen}},
        "hub_keys_by_leg": [{"job": leg.get("job"), "hub_key": leg.get("hub_key"), "at": leg.get("at")} for leg in legs],
        "hub_keys_by_campaign": {campaign: [{"job": leg.get("job"), "hub_key": leg.get("hub_key"), "at": leg.get("at")}
                                            for leg in legs_of(KR / "probes" / MODEL / campaign)]
                                 for campaign in [name] + extra},
        "sandbox_credential": SANDBOX_CREDENTIAL[sandbox],
        "max_concurrency": settings.get("max_concurrency_per_model"),
        "context_window": spec["context_window"], "prompt": PROMPT,
        "system_prompt": {"file": str(STOCK_PROMPT_FILE.relative_to(KR)), "sha256": STOCK_PROMPT_SHA256,
                          "note": "OpenCode 1.17.11 stock prompt, no override (no added sentence)"},
        "job": job, "started_at": started, "budget_start": start, "budget_end": end,
        "recovered_budget_refusals": recovery,
        "broken_reply_reruns": reruns, "broken_reply_rerun_rule": RERUN_RULE,
        "broken_reply_rerun_outcomes": dict(outcomes),
        "stats_first_attempts": stats(first), "stats_final": stats(final),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "first_attempt_records": {task_id: [r for r in slot_records] for task_id, slot_records in first.items()},
        "records": final})
    leg_record(campaign=name, event="complete", hub_key=key, stats=stats(final), stats_first_attempts=stats(first),
               recovered=sum(map(len, recovery.values())), broken_reruns=dict(outcomes))
    log(f"kimi-rollouts-v5: {name} done: first attempts {json.dumps(stats(first))}; final {json.dumps(stats(final))}; "
        f"broken-reply re-runs {json.dumps(dict(outcomes))}")


if __name__ == "__main__":
    main()
