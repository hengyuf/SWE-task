#!/usr/bin/env python3
"""teacher-rollouts-v2 (2026-10-25): one probe campaign = one hub teacher x a task subset x N rollouts per task.

    (cd swe-task/synth && atp/runtime/controller/bin/python3 -B <TR>/driver.py --model glm|deepseek \
        --taskset pilot|full --key-names NAME[,NAME ...] [--group K --groups 8] [--repeats N] [--min-remaining USD])

The synth pipeline's own rollout and grading path (rollouts.probe: ATP campaign, OpenCode harness, each task's own
grader), exactly as the pipeline probes kimi when it builds a task (and as train-acc-v2's driver.py, minus the local
fleet: hub models need no router or workers). Differences from a build probe:
* models: "glm" = settings.json's spec (nvidia/zai-org/glm-5.3, reasoning_effort max); "deepseek" is added IN MEMORY
  (nvidia/deepseek-ai/deepseek-v4-pro, reasoning_effort max); settings.json is never edited;
* hub key, by NAME (2026-10-26): --key-names is an ordered list; this leg uses the first key whose remaining budget is
  >= --min-remaining (the budget gate checks the key the leg uses; a key below the floor is "used up" and the next one
  is tried; an unknown budget stops the leg). submit.sh passes NVAPI_KEY_NEW_2,NVAPI_KEY_NEW_3 (user, 2026-10-26
  ~00:45: NEW_2 first, NEW_3 once NEW_2 is used up). Every leg started before 2026-10-26 01:00 used NVAPI_KEY_NEW_1
  (hard-coded then; user 2026-10-25). The key reaches this directory's probe_worker.py wrapper as $TR_HUB_KEY_NAME;
  rollouts.probe() runs SYNTH/probe_worker.py, so rollouts.SYNTH is pointed here (probe() uses SYNTH for nothing else);
* ledger: each leg appends a start record (job, model, campaign, key, budget) and an end record to legs.jsonl; the
  results file lists the key of every leg of the campaign (legs.jsonl was backfilled from logs/ for the NEW_1 legs).
Tasks: the 50 train-acc-v2 tasks (TA/tasks.json). pilot = PILOT_TASKS (2 tasks), full = tasks[group::groups].
Writes probes/<model>/<name>/ (the campaign; name = pilot | group-<K>) and results/<model>-<name>.json (compact
per-slot records, the train-acc-v2 shape). Resumable: rerun the same command in a fresh CPU job.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TR = Path(__file__).resolve().parent
TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
SYNTH = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth")
sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != TR]
sys.path.insert(0, str(SYNTH))
import common  # noqa: E402
from common import SETTINGS, log  # noqa: E402
import rollouts  # noqa: E402

KEY_NAMES = ("NVAPI_KEY_NEW_1", "NVAPI_KEY_NEW_2", "NVAPI_KEY_NEW_3")  # names probe_worker.py accepts
LEDGER = TR / "legs.jsonl"
HUB_MODELS = {"glm": {"id": "nvidia/zai-org/glm-5.3", "sampling_overrides": {"reasoning_effort": "max"}},
              "deepseek": {"id": "nvidia/deepseek-ai/deepseek-v4-pro", "sampling_overrides": {"reasoning_effort": "max"}}}
# Near the median kimi rollout size of the 50 tasks (main-session prompt tokens; pilot_pick.json), two languages.
PILOT_TASKS = ("syn-aiohttp-cand-c05-r1b0-7d9ec1f5", "syn-argo-cd-cand-c05-cc3e1272")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def budget(key_name: str, min_remaining: float, tag: str) -> tuple:
    """(exit code, record) of key_budget.py for one key NAME (it prints the key NAME and numbers only)."""
    result = subprocess.run(["/usr/bin/python3", "-I", "-B", str(TR / "key_budget.py"), "--key-name", key_name,
                             "--min-remaining", str(min_remaining), "--tag", tag],
                            capture_output=True, text=True, timeout=300)
    line = (result.stdout.strip().splitlines() or ["{}"])[-1]
    try:
        record = json.loads(line)
    except ValueError:
        record = {"error": line[:300]}
    log(f"teacher-rollouts: budget {json.dumps(record)} (exit {result.returncode})")
    return result.returncode, record


def choose_key(key_names: list, min_remaining: float, tag: str) -> tuple:
    """(name, budget record) of the first key with >= min_remaining left; SystemExit if none or a budget is unknown."""
    for key_name in key_names:
        if not common.env_value(common.USER_ENV, key_name):
            raise SystemExit(f"no {key_name} in {common.USER_ENV}")
        code, record = budget(key_name, min_remaining, tag)
        if code == 3:
            log(f"teacher-rollouts: {key_name} remaining {record.get('remaining')} < {min_remaining}: trying the next key")
            continue
        if code != 0 or record.get("remaining") is None:
            raise SystemExit(f"{key_name} budget unknown ({record}); not starting the campaign")
        return key_name, record
    raise SystemExit(f"every key of {key_names} has < {min_remaining} left: not starting the campaign")


def ledger(record: dict) -> None:
    with open(LEDGER, "a") as stream:
        stream.write(json.dumps(record) + "\n")


def campaign_legs(model: str, name: str) -> list:
    """Start records of every leg of this campaign in legs.jsonl, oldest first."""
    legs = []
    try:
        lines = LEDGER.read_text().splitlines()
    except OSError:
        return legs
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("event") == "start" and entry.get("model") == model and entry.get("campaign") == name:
            legs.append({key: entry.get(key) for key in ("job", "key", "at", "source")})
    return legs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(HUB_MODELS), required=True)
    parser.add_argument("--taskset", choices=("pilot", "full"), required=True)
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--min-remaining", type=float, default=0.0)
    parser.add_argument("--key-names", required=True,
                        help="ordered hub key NAMES (comma list); the leg uses the first with >= --min-remaining left")
    parser.add_argument("--max-concurrency", type=int, default=rollouts.PROBE_MAX_CONCURRENCY,
                        help="rollouts in flight per campaign (fixed at campaign creation; pipeline cap 32)")
    args = parser.parse_args()
    key_names = [key_name.strip() for key_name in args.key_names.split(",") if key_name.strip()]
    if not key_names or any(key_name not in KEY_NAMES for key_name in key_names):
        raise SystemExit(f"--key-names {args.key_names!r}: every name must be one of {KEY_NAMES}")
    tasks = json.loads((TA / "tasks.json").read_text())["tasks"]
    if args.taskset == "pilot":
        chosen, name, repeats = [task for task in tasks if task["task_id"] in PILOT_TASKS], "pilot", args.repeats or 1
        assert len(chosen) == len(PILOT_TASKS), "pilot tasks missing from tasks.json"
    else:
        chosen, name, repeats = tasks[args.group::args.groups], f"group-{args.group}", args.repeats or 5
    probe_dir = TR / "probes" / args.model / name
    probe_dir.parent.mkdir(parents=True, exist_ok=True)

    spec = SETTINGS["models"].setdefault(args.model, HUB_MODELS[args.model])
    assert spec == HUB_MODELS[args.model], f"unexpected spec for {args.model}: {spec}"
    assert not spec.get("local") and spec["sampling_overrides"]["reasoning_effort"] == "max"
    assert (TR / "probe_worker.py").is_file()
    rollouts.SYNTH = TR  # probe() runs SYNTH / "probe_worker.py": the key-by-name wrapper
    # probe_limits() reads this module global at campaign creation: max = min(slots, cap), initial = min(16, max).
    # Lower caps keep the hub models' shared per-model token limits (GLM is also the r4 RL judge) unsaturated.
    assert 1 <= args.max_concurrency <= 32
    rollouts.PROBE_MAX_CONCURRENCY = args.max_concurrency
    job = os.environ.get("SLURM_JOB_ID")
    key_name, start = choose_key(key_names, args.min_remaining, f"start {args.model}/{name} job {job}")
    os.environ["TR_HUB_KEY_NAME"] = key_name  # the wrapper reads (and drops) it; a NAME, never a key value
    leg = {"model": args.model, "campaign": name, "job": job, "key": key_name}
    ledger(dict(leg, event="start", at=now(), source="driver", budget_start=start))

    log(f"teacher-rollouts: {args.model} ({spec['id']}) {name}: {len(chosen)} tasks x {repeats}, key {key_name}, "
        f"remaining ${start.get('remaining')}, max concurrency {args.max_concurrency}")
    started = time.time()
    try:
        records = asyncio.run(rollouts.probe(probe_dir, [Path(task["task_dir"]) for task in chosen],
                                             models=(args.model,), repeats=repeats))
    except BaseException as error:
        ledger(dict(leg, event="end", at=now(), complete=False, error=f"{type(error).__name__}: {error}"[:300]))
        raise
    _, end = budget(key_name, 0, f"end {args.model}/{name} job {job}")  # informational: never blocks the results
    ledger(dict(leg, event="end", at=now(), complete=True, budget_end=end))
    legs = campaign_legs(args.model, name)
    compact = {task_id: [{key: record.get(key) for key in ("slot", "status", "reward", "f2p", "p2p", "agent_error",
                                                            "verifier_error", "attempt_dir")}
                         | {"patch_chars": len(record.get("patch") or "")}
                         for record in slot_records] for task_id, slot_records in records.items()}
    out = TR / "results" / f"{args.model}-{name}.json"
    out.parent.mkdir(exist_ok=True)
    tmp = out.with_name("." + out.name + ".tmp")
    tmp.write_text(json.dumps({"model": args.model, "model_id": spec["id"], "spec": spec, "taskset": args.taskset,
                               "group": args.group if args.taskset == "full" else None, "groups": args.groups,
                               "tasks": [task["task_id"] for task in chosen], "repeats": repeats, "key": key_name,
                               "keys": sorted({entry["key"] for entry in legs if entry.get("key")} | {key_name}),
                               "legs": legs, "max_concurrency": args.max_concurrency,
                               "job": job, "started_at": started,
                               "budget_start": start, "budget_end": end,
                               "finished_at": now(), "records": compact}, indent=1) + "\n")
    os.replace(tmp, out)
    stats = rollouts.pass_stats([record for slot_records in records.values() for record in slot_records])
    log(f"teacher-rollouts: {args.model} {name} done: {json.dumps(stats)}")


if __name__ == "__main__":
    main()
