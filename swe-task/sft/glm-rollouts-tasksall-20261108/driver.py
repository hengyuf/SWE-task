#!/usr/bin/env python3
"""glm-rollouts-tasksall (2026-11-08): one campaign leg = self-hosted GLM-5.3 x one campaign of tasks.json x 5 rollouts.

User (hengyuf, 2026-11-08, verbatim): "... Then do the rollout for GLM 5.3 on all the tasks in
swe-task/sft/tasks-all-20261106/, you don't need to redo the rollout for the tasks that already has the hosted GLM 5.3
rollouts. Prioritize the task that are accepted or marked as strong (first do the accepted/strong tasks, then do
partial, then weak, then none or without any labels)."

    (cd swe-task/synth && atp/runtime/controller/bin/python3 -B <GR2>/driver.py --campaign NAME
        --base-url http://127.0.0.1:<router port>/v1 [--dry-run])

Same rollout setup as glm-rollouts-sftv3-20261107 (whose driver.py this follows): the synth pipeline's rollout and
grading path (rollouts.probe: ATP campaign, OpenCode 1.17.11 with its STOCK system prompt, each task's own grader);
model alias "glmlocal" (in memory only): id "glm-5.3", base_url ${oc.env:GLM53_BASE_URL} = this leg's router
(router.py over the glm53-serve-ta-* GPU jobs), api_key "EMPTY", reasoning_effort "max", temperature 1.0 / top_p 0.95
(synth settings), context_window 262,144 (OpenCode compacts at 242,144: compaction allowed) and max_output_tokens
262,144 (the "256k output limit"; vLLM clamps it to the context left). No hub key anywhere (probe_worker.py).
* tasks.json (make_tasks.py) = the frozen targets + the campaign plan (tier-major order; per campaign: tasks, sandbox
  owner, repeats 5, max concurrency). Digests are re-checked every leg.
* hengyuf tasks run on CELL3_SANDBOX_API_KEY (S/atp/.env), Baihe's on BAIHE_SANDBOX_API_KEY
  (glm-rollouts-sftv3-20261107/secrets/baihe-opensandbox.env, mode 600).
* A campaign starts only when released (GO-<campaign>, created by keeper.py in priority order).
* Slots whose OpenCode run was ended by a sandbox-side failure are re-run once in <campaign>-rec<n> (the re-run
  replaces the slot; provenance in the results file), as in glm-rollouts-sftv3.
Writes probes/glmlocal/<campaign>/ (ATP campaign), legs.jsonl, and when complete results/glmlocal-<campaign>.json.
Resumable: rerun the same command in a fresh CPU job (the next leg).
"""
import argparse
import asyncio
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.request

GR = Path(__file__).resolve().parent
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
SYNTH = S / "synth"
sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != GR]
sys.path.insert(0, str(SYNTH))
import common  # noqa: E402,F401
from common import SETTINGS, log  # noqa: E402
import rollouts  # noqa: E402

ALIAS = "glmlocal"
SPEC = {"id": "glm-5.3", "base_url": "${oc.env:GLM53_BASE_URL}", "api_key": "EMPTY",
        "context_window": 262144, "max_output_tokens": 262144, "sampling_overrides": {"reasoning_effort": "max"}}
PROMPT = "stock"
TASKS_FILE = GR / "tasks.json"
SANDBOX_CREDENTIAL = {"hengyuf": "CELL3_SANDBOX_API_KEY", "baihe": "BAIHE_SANDBOX_API_KEY"}
BAIHE_SECRETS = S / "sft/glm-rollouts-sftv3-20261107/secrets/baihe-opensandbox.env"
HUB_MARKERS = ("nvidia.com", "inference-api")
# Sandbox-side failures that end OpenCode before it finishes (SandboxPtyError etc.): ATP grades the half-done workspace
# as a fail; such a slot is re-run ONCE in <name>-rec<n> and the re-run replaces it ("original_attempt" kept).
INFRA_RERUN = re.compile(r"Sandbox")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digests(task_dir: Path) -> dict:
    out = {"instruction.md": sha256(task_dir / "instruction.md"), "task.toml": sha256(task_dir / "task.toml")}
    for path in sorted(p for p in (task_dir / "tests").rglob("*") if p.is_file()):
        out[str(path.relative_to(task_dir))] = sha256(path)
    return out


def load_plan():
    data = json.loads(TASKS_FILE.read_text())
    tasks = {t["key"]: t for t in data["tasks"]}
    assert len(tasks) == len(data["tasks"]) == data["summary"]["tasks"]
    campaigns = {c["name"]: c for c in data["campaigns"]}
    return tasks, campaigns


def campaign_tasks(name):
    tasks, campaigns = load_plan()
    if name not in campaigns:
        raise SystemExit(f"unknown campaign {name!r}")
    campaign = campaigns[name]
    chosen = [tasks[key] for key in campaign["tasks"]]
    ids = [t["task_id"] for t in chosen]
    assert len(ids) == len(set(ids)), f"{name}: repeated task id"
    assert all(t["owner"] == campaign["sandbox"] for t in chosen), f"{name}: mixed sandbox owners"
    return campaign, chosen


def router_health(base_url: str) -> dict:
    url = base_url.rstrip("/")
    url = (url[:-3] if url.endswith("/v1") else url) + "/health"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=20) as response:
        return json.load(response)


def wait_for_backend(base_url: str, max_wait_s: float) -> dict:
    """Block until the router reports >= 1 healthy backend (no sandbox is created before a model can answer)."""
    started = time.time()
    last = None
    while True:
        try:
            last = router_health(base_url)
            if last.get("healthy_backends", 0) >= 1:
                return last
        except (OSError, ValueError) as error:
            last = {"error": str(error)[:200]}
        if time.time() - started > max_wait_s:
            raise SystemExit(f"no healthy GLM backend behind {base_url} after {max_wait_s:.0f} s ({last})")
        if int(time.time() - started) % 600 < 31:
            log(f"glm-rollouts-tasksall: waiting for a GLM backend behind {base_url} ({last})")
        time.sleep(30)


def wait_for_go(name: str, max_wait_s: float) -> None:
    """A campaign starts only when released: GR/GO-<campaign> (keeper.py, priority order; no global GO here)."""
    started = time.time()
    while not (GR / f"GO-{name}").exists():
        if time.time() - started > max_wait_s:
            raise SystemExit(f"no {GR / ('GO-' + name)} after {max_wait_s:.0f} s")
        if int(time.time() - started) % 900 < 61:
            log(f"glm-rollouts-tasksall: waiting for {GR / ('GO-' + name)}")
        time.sleep(60)


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n")
    os.replace(tmp, path)


def leg_record(**fields):
    with open(GR / "legs.jsonl", "a") as stream:
        stream.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "job": os.environ.get("SLURM_JOB_ID"),
                                 **fields}) + "\n")


def compact(record):
    return ({key: record.get(key) for key in ("slot", "status", "reward", "f2p", "p2p", "agent_error", "verifier_error",
                                              "attempt_dir")}
            | {"patch_chars": len(record.get("patch") or "")})


# ATP staging downloads the OpenCode and ripgrep releases from GitHub in every leg; on 2026-11-08 23:22-23:39 GitHub
# answered 504 and 10 legs died in staging. Such a failure is retried in the same leg (wait, wipe this job's ATP
# scratch: stage() refuses a scratch whose staging failed) instead of burning keeper legs.
STAGING_NET_ERROR = re.compile(r"HTTP Error 5\d\d|URLError|timed out|Temporary failure in name resolution|"
                               r"Connection reset|RemoteDisconnected|IncompleteRead")


# ATP staging downloads the OpenCode and ripgrep releases from GitHub in every leg (flaky since 2026-11-08 23:22):
# rollouts.probe() runs ATP's job-local deepswe_stage.py through stage_wrapper.py, which serves exactly those two
# URLs from release-cache/ (sha256-verified, byte-identical to earlier GitHub downloads) and changes nothing else.
STAGE_WRAPPER = GR / "stage_wrapper.py"
_rollouts_run = rollouts._run


async def _run_with_stage_wrapper(argv, log_path, env):
    argv = list(argv)
    if len(argv) >= 3 and str(argv[0]) == "/usr/bin/python3" and str(argv[2]).endswith("scripts/deepswe_stage.py"):
        argv = [argv[0], argv[1], STAGE_WRAPPER, *argv[2:]]
    return await _rollouts_run(argv, log_path, env)


rollouts._run = _run_with_stage_wrapper


def probe(probe_dir: Path, task_dirs, *, repeats, tries=6, wait_s=300) -> dict:
    """rollouts.probe(probe_dir, task_dirs, models=(ALIAS,), repeats=repeats), retrying network-caused staging
    failures up to `tries` times (only the stage.log lines this attempt appended are inspected)."""
    stage_log = Path(probe_dir) / "stage.log"
    for attempt in range(1, tries + 1):
        offset = stage_log.stat().st_size if stage_log.exists() else 0
        try:
            return asyncio.run(rollouts.probe(probe_dir, task_dirs, models=(ALIAS,), repeats=repeats))
        except RuntimeError as error:
            if "ATP staging failed" not in str(error) or attempt == tries:
                raise
            try:
                with open(stage_log, "rb") as stream:
                    stream.seek(offset)
                    appended = stream.read().decode("utf-8", "replace")
            except OSError:
                appended = ""
            match = STAGING_NET_ERROR.search(appended)
            if not match:
                raise
            log(f"glm-rollouts-tasksall: {Path(probe_dir).name}: ATP staging failed on a network error "
                f"({match.group(0)}; try {attempt}/{tries}); retrying in {wait_s} s with a fresh ATP scratch")
            leg_record(campaign=Path(probe_dir).name, event="staging_retry", attempt=attempt, error=match.group(0))
            time.sleep(wait_s)
            shutil.rmtree(rollouts.scratch() / "atp", ignore_errors=True)


def unfinished_infra(record) -> str | None:
    """The sandbox error type when OpenCode did not finish because of a sandbox-side failure, else None."""
    attempt = record.get("attempt_dir")
    if not attempt:
        return None
    try:
        result = json.loads((Path(attempt) / "native-result.json").read_text())
    except (OSError, ValueError):
        return None
    error = result.get("opencode_run_error_type")
    if result.get("opencode_finished") is False and error and INFRA_RERUN.search(str(error)):
        return str(error)
    return None


def recover(name: str, records: dict, task_dirs: dict) -> dict:
    """Re-run once every slot whose final attempt was ended by a sandbox-side failure; returns {task_id: [entry]}."""
    todo = collections.defaultdict(list)
    for task_id, slot_records in records.items():
        for record in slot_records:
            error = unfinished_infra(record)
            if error:
                todo[task_id].append((record["slot"], error))
    recovered = {}
    by_count = collections.defaultdict(list)
    for task_id, slots in todo.items():
        by_count[len(slots)].append(task_id)
    for count, task_ids in sorted(by_count.items()):
        rec_name = f"{name}-rec{count}"
        rec_dir = GR / "probes" / ALIAS / rec_name
        log(f"glm-rollouts-tasksall: {name}: re-running {sum(len(todo[t]) for t in task_ids)} sandbox-failed slot(s) "
            f"once in {rec_name}: {json.dumps({t: todo[t] for t in task_ids})[:1500]}")
        leg_record(campaign=rec_name, event="start", reason="sandbox-side failures", slots={t: todo[t] for t in task_ids})
        rec_records = probe(rec_dir, [task_dirs[t] for t in sorted(task_ids)], repeats=count)
        for task_id in task_ids:
            for (slot, error), rec in zip(todo[task_id], rec_records[task_id]):
                index = next(i for i, r in enumerate(records[task_id]) if r["slot"] == slot)
                original = records[task_id][index]
                records[task_id][index] = dict(rec, slot=slot)
                recovered.setdefault(task_id, []).append({
                    "slot": slot, "original_attempt": original.get("attempt_dir"), "original_status": original.get("status"),
                    "original_error": error, "recovery_campaign": rec_name, "recovery_slot": rec["slot"],
                    "recovery_attempt": rec.get("attempt_dir"), "recovery_status": rec.get("status"),
                    "recovery_error": unfinished_infra(rec)})
    return recovered


def stats(records: dict) -> dict:
    flat = [record for slot_records in records.values() for record in slot_records]
    passed = sum(record["status"] == "pass" for record in flat)
    failed = sum(record["status"] == "fail" for record in flat)
    return {"pass": passed, "fail": failed, "infra": len(flat) - passed - failed,
            "rate": round(passed / (passed + failed), 4) if passed + failed else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--base-url", required=True, help="this leg's router, http://127.0.0.1:<port>/v1")
    parser.add_argument("--max-wait-backend", type=float, default=12 * 3600)
    parser.add_argument("--max-wait-go", type=float, default=20 * 3600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(marker in args.base_url for marker in HUB_MARKERS):
        raise SystemExit("--base-url must be our own router, never the NVIDIA hub")
    campaign, chosen = campaign_tasks(args.campaign)
    name, repeats, sandbox = campaign["name"], campaign["repeats"], campaign["sandbox"]
    max_concurrency = campaign["max_concurrency"]
    probe_dir = GR / "probes" / ALIAS / name
    changed = [t["key"] for t in chosen if digests(Path(t["task_dir"])) != t["digests"]]
    if changed:
        raise SystemExit(f"task files changed since tasks.json was frozen: {changed}")
    plan = {"alias": ALIAS, "spec": SPEC, "campaign": str(probe_dir), "name": name, "tier": campaign["tier"],
            "tier_name": campaign["tier_name"], "tasks": [t["key"] for t in chosen], "repeats": repeats,
            "slots": len(chosen) * repeats, "sandbox_credential": SANDBOX_CREDENTIAL[sandbox],
            "max_concurrency": max_concurrency}
    if args.dry_run:
        print(json.dumps(plan, indent=1))
        return
    assert 1 <= max_concurrency <= 32
    if (probe_dir / "settings.json").exists():  # a resumed campaign must have been created with this spec
        created = json.loads((probe_dir / "settings.json").read_text())["models"]
        assert created == {ALIAS: SPEC}, f"{probe_dir} was created with {created}"
    legs = probe_dir / "gr-legs.jsonl"
    if legs.exists():
        earlier = [json.loads(line) for line in legs.read_text().splitlines() if line.strip()]
        assert all(leg["sandbox"] == sandbox for leg in earlier), f"{probe_dir}: earlier legs used another sandbox"
    if sandbox == "baihe" and not BAIHE_SECRETS.is_file():
        raise SystemExit(f"{BAIHE_SECRETS} missing")
    assert SETTINGS["temperature"] == 1.0 and SETTINGS["top_p"] == 0.95

    os.environ["GLM53_BASE_URL"] = args.base_url  # inherited by ATP staging, probe_worker and Gym (${oc.env:...})
    os.environ["GR_SANDBOX"] = sandbox
    os.environ["GR_PROMPT"] = PROMPT
    wait_for_go(name, args.max_wait_go)
    health = wait_for_backend(args.base_url, args.max_wait_backend)
    SETTINGS["models"][ALIAS] = dict(SPEC)        # in memory only: rollouts.probe() writes it into the campaign
    rollouts.SYNTH = GR                            # probe() runs GR/probe_worker.py
    rollouts.PROBE_MAX_CONCURRENCY = max_concurrency
    probe_dir.parent.mkdir(parents=True, exist_ok=True)
    log(f"glm-rollouts-tasksall: {name} (tier {campaign['tier']} {campaign['tier_name']}): {len(chosen)} tasks x "
        f"{repeats} via {args.base_url} (router {health}), sandbox {SANDBOX_CREDENTIAL[sandbox]}, max concurrency "
        f"{max_concurrency}")
    leg_record(campaign=name, event="start", sandbox_credential=SANDBOX_CREDENTIAL[sandbox],
               max_concurrency=max_concurrency, base_url=args.base_url, router=health)
    started = time.time()
    task_dirs = {t["task_id"]: Path(t["task_dir"]) for t in chosen}
    try:
        records = probe(probe_dir, [Path(t["task_dir"]) for t in chosen], repeats=repeats)
    except RuntimeError as error:
        leg_record(campaign=name, event="incomplete", error=str(error)[:300])
        log(f"glm-rollouts-tasksall: {name} not complete this leg ({error})")
        raise SystemExit(1)
    first = {task_id: [compact(r) for r in slot_records] for task_id, slot_records in records.items()}
    try:
        recovered = recover(name, records, task_dirs)
    except RuntimeError as error:
        leg_record(campaign=name, event="recovery_incomplete", error=str(error)[:300])
        log(f"glm-rollouts-tasksall: {name} recovery not complete this leg ({error})")
        raise SystemExit(1)
    final = {task_id: [compact(r) for r in slot_records] for task_id, slot_records in records.items()}
    settings = json.loads((probe_dir / "settings.json").read_text())
    keys = {t["task_id"]: t["key"] for t in chosen}
    out = GR / "results" / f"{ALIAS}-{name}.json"
    write_json_atomic(out, dict(plan, started_at=started, finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                job=os.environ.get("SLURM_JOB_ID"), prompt=PROMPT, task_keys=keys,
                                max_concurrency_created=settings.get("max_concurrency_per_model"),
                                stats=stats(final), stats_first_attempts=stats(first),
                                sandbox_failure_reruns=recovered, sandbox_failure_rule=INFRA_RERUN.pattern,
                                first_attempt_records=first, records=final))
    leg_record(campaign=name, event="complete", stats=stats(final), stats_first_attempts=stats(first),
               sandbox_failure_reruns=sum(map(len, recovered.values())))
    log(f"glm-rollouts-tasksall: {name} done: {json.dumps(stats(final))}")


if __name__ == "__main__":
    main()
