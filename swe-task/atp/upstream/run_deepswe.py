#!/usr/bin/env python3
"""Create, submit, run, and inspect the hosted DeepSWE campaign."""

import argparse
import asyncio
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import uuid

from deepswe_http import redact


APP = Path(os.environ.get("ATP_APP", str(Path(__file__).resolve().parents[1]))).resolve()
ACCOUNTS = (("nemotron_agents_dev", 0), ("nemotron_rl_algo", 5))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w") as stream:
        json.dump(redact(value), stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def gym_command(staged, config, launch, settings, concurrency):
    return [str(staged["gym_cli"]), "eval", "run", "--config", str(config),
            "--split", "benchmark", "--num-repeats", str(settings["repeats"]),
            "--concurrency", str(concurrency), "--temperature", str(settings["temperature"]),
            "--top-p", str(settings["top_p"]), "--max-output-tokens", str(settings["max_output_tokens"]),
            "--output", str(launch / "rollouts.jsonl"), "++route_failures_to_sidecar=true"]


def shard_concurrency(shard, manifest):
    """Native queue ceiling; adaptive admission owns the smaller live budget."""
    total = manifest["settings"].get("max_concurrency_per_model", 2048)
    count = manifest["shards"]
    if type(total) is not int or total < count:
        raise ValueError(f"Maximum global concurrency must be an integer >= {count}")
    slots = sum(task["_ng_task_index"] % count == shard for task in manifest["tasks"]) * manifest["settings"]["repeats"]
    return max(1, min(slots, total // count + int(shard < total % count)))


def job_end_time(environment, run=subprocess.run):
    if environment.get("SLURM_JOB_END_TIME"):
        end = float(environment["SLURM_JOB_END_TIME"])
    else:
        job = environment.get("SLURM_JOB_ID")
        if not job:
            raise ValueError("DeepSWE workers must run inside a CPU Slurm allocation")
        result = run(["scontrol", "show", "job", "--oneliner", job], capture_output=True,
                     text=True, check=True, timeout=30)
        match = re.search(r"(?:^|\s)EndTime=(\S+)", result.stdout)
        if not match or match.group(1) in {"Unknown", "N/A", "None"}:
            raise ValueError("Slurm did not provide a finite job end time")
        end = datetime.fromisoformat(match.group(1)).timestamp()
    if not math.isfinite(end) or end <= 0:
        raise ValueError("Slurm did not provide a finite positive job end time")
    return end


def redact_chunk(buffer, secrets, final=False):
    """Keep enough tail to redact credentials split across stdout chunks."""
    end = len(buffer) if final else max(0, len(buffer) - max(map(len, secrets), default=1) + 1)
    for secret in secrets:
        start = buffer.find(secret)
        while start >= 0 and start < end:
            if start + len(secret) > end:
                end = start
                break
            start = buffer.find(secret, start + 1)
    output, remainder = buffer[:end], buffer[end:]
    for secret in secrets:
        output = output.replace(secret, b"[REDACTED]")
    return output, remainder


async def drain_output(process, log_path, environment, label):
    secrets = sorted({value.encode() for key, value in environment.items() if value and
                      key.upper().endswith(("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD"))},
                     key=len, reverse=True)
    pending = b""
    with log_path.open("ab") as log:
        def write(chunk):
            log.write(chunk)
            log.flush()
            os.fsync(log.fileno())
            print(f"[{label}] " + chunk.decode("utf-8", errors="replace"), end="", flush=True)

        while True:
            chunk = await process.stdout.read(65536)
            pending += chunk
            clean, pending = redact_chunk(pending, secrets, final=not chunk)
            if clean:
                await asyncio.to_thread(write, clean)
            if not chunk:
                break


async def process_exited(process):
    # Process.wait() can wait for inherited pipe EOF after the leader has exited.
    while process.returncode is None:
        await asyncio.sleep(0.1)
    return process.returncode


async def stop_process(process):
    for sig, seconds in ((signal.SIGINT, 20), (signal.SIGTERM, 10), (signal.SIGKILL, 10)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            await asyncio.wait_for(process_exited(process), 5)
            return
        deadline = asyncio.get_running_loop().time() + seconds
        while asyncio.get_running_loop().time() < deadline:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                await asyncio.wait_for(process_exited(process), 5)
                return
            await asyncio.sleep(0.1)
    raise RuntimeError("Gym process group still exists after forced shutdown")


async def finish_output(output, launch, timeout=10):
    """Do not let an inherited pipe prevent remote artifact recovery."""
    try:
        await asyncio.wait_for(asyncio.shield(output), timeout)
    except asyncio.TimeoutError:
        save_json(launch / "console-capture.json", {"complete": False, "finished_at": time.time(),
                  "reason": "Console pipe did not reach EOF after process shutdown"})
        output.cancel()
        await asyncio.gather(output, return_exceptions=True)
    except Exception as error:
        save_json(launch / "console-capture.json", {"complete": False, "finished_at": time.time(),
                  "error_type": type(error).__name__, "error": str(error)})
        raise
    else:
        save_json(launch / "console-capture.json", {"complete": True, "finished_at": time.time()})


def mark_stopped(launch):
    stopped = set(json.loads(os.environ.get("ATP_DEEPSWE_STOPPED_LAUNCH_DIRS", "[]")))
    stopped.add(str(launch))
    os.environ["ATP_DEEPSWE_STOPPED_LAUNCH_DIRS"] = json.dumps(sorted(stopped))
    save_json(launch / "stopped.json", {"confirmed_at": time.time(), "launch": str(launch)})


def resolved_config(config):
    from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
    from omegaconf import OmegaConf

    parsed = GlobalConfigDictParser().parse(GlobalConfigDictParserConfig(
        initial_global_config_dict=OmegaConf.create({"config_paths": [str(config)]}),
        skip_load_from_cli=True, skip_load_from_dotenv=True, offline=True))
    return OmegaConf.to_container(parsed, resolve=True)


async def collect(staged, config, launch, environment, settings, concurrency, stopped, label):
    command = gym_command(staged, config, launch, settings, concurrency)
    save_json(launch / "launch.json", {"command": command, "started_at": time.time(),
              "slurm_job_id": environment["SLURM_JOB_ID"], "hostname": socket.gethostname(),
              "source": environment.get("ATP_DEEPSWE_SOURCE"),
              "model": environment["ATP_DEEPSWE_MODEL_ALIAS"], "shard": int(environment["ATP_DEEPSWE_SHARD_ID"])})
    process = await asyncio.create_subprocess_exec(*command, cwd=staged["gym"], env=environment,
              stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
    output = asyncio.create_task(drain_output(process, launch / "gym.log", environment, label))
    done = asyncio.create_task(process_exited(process))
    interrupted = asyncio.create_task(stopped.wait())
    try:
        completed, _ = await asyncio.wait((done, interrupted, output), return_when=asyncio.FIRST_COMPLETED)
        if output in completed and output.exception() is None and done not in completed and interrupted not in completed:
            completed, _ = await asyncio.wait((done, interrupted), return_when=asyncio.FIRST_COMPLETED)
        if interrupted in completed or (output in completed and output.exception() is not None):
            await stop_process(process)
        else:
            await done
        save_json(launch / "process.json", {"exit_code": process.returncode,
                  "interrupted": stopped.is_set(), "finished_at": time.time(), "cleanup": "pending"})
        return process.returncode
    finally:
        interrupted.cancel()
        try:
            save_json(launch / "cleanup.json", {"phase": "stopping_process_group", "started_at": time.time()})
            await stop_process(process)
            mark_stopped(launch)
            save_json(launch / "cleanup.json", {"phase": "draining_console", "updated_at": time.time()})
            await finish_output(output, launch)
            save_json(launch / "cleanup.json", {"phase": "process_cleanup_complete", "finished_at": time.time()})
            save_json(launch / "process.json", {"exit_code": process.returncode,
                      "interrupted": stopped.is_set(), "finished_at": time.time(), "cleanup": "complete"})
        except BaseException as error:
            save_json(launch / "cleanup.json", {"phase": "failed", "finished_at": time.time(),
                      "error_type": type(error).__name__, "error": str(error)})
            if not output.done():
                output.cancel()
            raise
        finally:
            done.cancel()
            await asyncio.gather(done, interrupted, output, return_exceptions=True)


async def recover_model(campaign, campaign_path, model, shard, config, launch, end, grace, stopped):
    """Recover concurrently and durably; the shutdown grace is not a runtime cap."""
    deadline = end - (10 if stopped.is_set() else grace)
    if stopped.is_set():
        deadline = min(deadline, time.time() + max(1, grace - 40))
    for attempt in range(3):
        allowance = deadline - time.time()
        if allowance <= 0:
            return False
        recovery = asyncio.create_task(campaign.recover_incomplete(campaign_path, model, shard, config))
        stop = asyncio.create_task(stopped.wait())
        try:
            done, _ = await asyncio.wait((recovery, stop), timeout=allowance,
                                         return_when=asyncio.FIRST_COMPLETED)
            if recovery in done:
                result = recovery.result()
            elif stop in done:
                # A termination signal can arrive during a long recovery pass.
                deadline = min(deadline, end - 10, time.time() + max(1, grace - 40))
                result = await asyncio.wait_for(recovery, max(0, deadline - time.time()))
            else:
                raise asyncio.TimeoutError
        except asyncio.TimeoutError:
            save_json(launch / "recovery-timeout.json", {"finished_at": time.time(),
                      "deadline": deadline, "resumable": True})
            return False
        finally:
            for task in (recovery, stop):
                if not task.done():
                    task.cancel()
            await asyncio.gather(recovery, stop, return_exceptions=True)
        save_json(launch / f"recovery-{attempt}.json", result)
        if not result["remaining"]:
            return True
        # Completed cleanup markers survive these retries and future workers.
        if stopped.is_set() or attempt == 2:
            return False
        try:
            await asyncio.wait_for(stopped.wait(), min(30, max(0, deadline - time.time())))
        except asyncio.TimeoutError:
            pass
        if stopped.is_set():
            return False
    return False


async def run_model(campaign, staged, campaign_path, model, shard, end, stopped):
    settings = staged["settings"]
    grace = int(settings["shutdown_grace_seconds"])
    manifest = campaign.load_manifest(campaign_path)
    model_dir = campaign_path / "models" / model / "shards" / str(shard)
    model_dir.mkdir(parents=True, exist_ok=True)
    def completed_slots():
        return sum(campaign.slot_state(campaign_path, model, task["_ng_task_index"], repeat)["status"] == "completed"
                   for task in manifest["tasks"] if task["_ng_task_index"] % manifest["shards"] == shard
                   for repeat in range(settings["repeats"]))

    stalled_launches = 0
    with campaign.shard_lock(campaign_path, model, shard):
        while True:
            if stopped.is_set() or time.time() >= end - grace:
                break
            concurrency = shard_concurrency(shard, manifest)
            launch = model_dir / ("launch-" + os.environ["SLURM_JOB_ID"] + "-" + uuid.uuid4().hex[:12])
            launch.mkdir()
            dataset = campaign.write_shard(campaign_path, shard, launch / "tasks.jsonl")
            from deepswe_stage import build_config
            config = build_config(staged, campaign_path, launch, model, dataset, concurrency=concurrency)
            full_config = resolved_config(config)
            if not await recover_model(campaign, campaign_path, model, shard, full_config,
                                       launch, end, grace, stopped):
                break
            if stopped.is_set() or not campaign.pending_slots(campaign_path, model, shard):
                break
            from deepswe_admission import reset_shard_health
            reset_shard_health(campaign_path, model, shard)
            completed_before = completed_slots()
            environment = dict(staged["env"], ATP_DEEPSWE_CAMPAIGN=str(campaign_path),
                ATP_DEEPSWE_LAUNCH_DIR=str(launch), ATP_DEEPSWE_MODEL_ALIAS=model,
                ATP_DEEPSWE_SHARD_ID=str(shard), ATP_DEEPSWE_JOB_DEADLINE=str(end - grace),
                ATP_DEEPSWE_API_CONCURRENCY=str(concurrency),
                ATP_DEEPSWE_AGENT_TIMEOUT=str(settings["agent_timeout_seconds"]),
                ATP_DEEPSWE_MAX_OUTPUT_TOKENS=str(settings["max_output_tokens"]),
                ATP_DEEPSWE_OPENCODE_BINARY=str(staged["opencode_binary"]),
                NEMO_GYM_RUN_ID=launch.name)
            environment.setdefault("ATP_DEEPSWE_UPSTREAM_STREAM", "1" if model == "kimi" else "0")
            environment.pop("NEMO_GYM_VLLM_TRANSPORT_LOG", None)
            # Ray's Unix socket names must fit Linux's 107-byte path limit.
            ray_tmp = Path(f"/tmp/atpr-{os.getuid()}-{uuid.uuid4().hex[:12]}")
            ray_tmp.mkdir(mode=0o700)
            environment["RAY_TMPDIR"] = str(ray_tmp)
            try:
                exit_code = await collect(staged, config, launch, environment, settings, concurrency, stopped, f"{model}/{shard}")
            except Exception as error:
                exit_code = None
                save_json(launch / "collector-error.json", {"error_type": type(error).__name__, "error": str(error)})
            finally:
                try:
                    recovered = await recover_model(campaign, campaign_path, model, shard, full_config,
                                                    launch, end, grace, stopped)
                except Exception as error:
                    save_json(launch / "recovery-error.json", {"type": type(error).__name__, "error": str(error)})
                    raise
            if not recovered:
                break
            admission = launch / "admission-closed.json"
            if admission.exists() and "Insufficient allocation time" in json.loads(admission.read_text()).get("reason", ""):
                break
            progress = completed_slots() - completed_before
            stalled_launches = 0 if progress > 0 else stalled_launches + 1
            save_json(launch / "collector-outcome.json", {"exit_code": exit_code, "valid_completed_slots": progress,
                      "consecutive_stalled_launches": stalled_launches})
            if stalled_launches >= 3:
                break
            if stalled_launches and campaign.pending_slots(campaign_path, model, shard):
                try:
                    await asyncio.wait_for(stopped.wait(), min(30 * stalled_launches, max(0, end - grace - time.time())))
                except asyncio.TimeoutError:
                    pass
        remaining = sum(campaign.slot_state(campaign_path, model, task["_ng_task_index"], repeat)["status"] != "completed"
                        for task in manifest["tasks"] if task["_ng_task_index"] % manifest["shards"] == shard
                        for repeat in range(settings["repeats"]))
        save_json(model_dir / "worker.json", {"remaining_slots": remaining,
                  "interrupted": stopped.is_set(), "finished_at": time.time(),
                  "slurm_job_id": os.environ["SLURM_JOB_ID"]})
        return remaining


async def worker(arguments):
    import deepswe_campaign as campaign
    from deepswe_source import select_source
    from deepswe_stage import stage

    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Run worker inside a CPU Slurm allocation")
    if any(os.environ.get(key, "0") not in {"", "0"} for key in ("SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE")):
        raise ValueError("Hosted DeepSWE workers require no GPU allocation")
    campaign_path = arguments.campaign.resolve()
    manifest = campaign.load_manifest(campaign_path)
    if not 0 <= arguments.shard_id < manifest["shards"]:
        raise ValueError("Shard ID is outside the campaign")
    source = select_source(campaign_path, arguments.source)
    for module_path in [Path(__file__), *(Path(sys.modules[name].__file__) for name in
                        ("deepswe_campaign", "deepswe_http", "deepswe_source", "deepswe_stage"))]:
        if module_path.read_bytes() != (source / "scripts" / module_path.name).read_bytes():
            raise ValueError("Worker code differs from the campaign snapshot; bootstrap its source into fresh scratch")
    os.environ["ATP_DEEPSWE_SOURCE"] = str(source)
    if arguments.recovery_allowance:
        from deepswe_recovery import load_allowance
        load_allowance(campaign_path, arguments.recovery_allowance, source)
        os.environ["ATP_DEEPSWE_RECOVERY_ALLOWANCE"] = str(arguments.recovery_allowance.resolve())
    else:
        os.environ.pop("ATP_DEEPSWE_RECOVERY_ALLOWANCE", None)
    scratch = Path(os.environ.get("ATP_DEEPSWE_SCRATCH", "/raid/scratch/atp-deepswe-" + os.environ["SLURM_JOB_ID"]))
    staged = stage(arguments.app.resolve(), scratch)
    staged["settings"] = manifest["settings"]
    os.environ.update(staged["env"])
    os.environ["ATP_DEEPSWE_CAMPAIGN"] = str(campaign_path)
    end = job_end_time(os.environ)
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stopped.set)
    timer = loop.call_later(max(0, end - time.time() - manifest["settings"]["shutdown_grace_seconds"]), stopped.set)
    try:
        results = await asyncio.gather(*(run_model(campaign, staged, campaign_path, model,
                    arguments.shard_id, end, stopped) for model in manifest["settings"]["models"]),
                    return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                print("DeepSWE worker failed: " + redact(str(result)), file=sys.stderr)
        return 130 if stopped.is_set() else int(any(isinstance(result, BaseException) or result for result in results))
    finally:
        timer.cancel()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


def submit(campaign_path, app, walltime, run=subprocess.run, resume=False, source=None, recovery_allowance=None):
    import deepswe_campaign as campaign
    from deepswe_source import freeze_source, select_source

    manifest = campaign.load_manifest(campaign_path)
    if manifest["shards"] != 10:
        raise ValueError("The two five-node submissions require exactly ten campaign shards")
    if not re.fullmatch(r"(?:[0-9]+-)?[0-9]+:[0-5][0-9]:[0-5][0-9]", walltime):
        raise ValueError("Job time must be HH:MM:SS or D-HH:MM:SS")
    requested_source = source
    source = select_source(campaign_path, source) if source is not None else freeze_source(app, campaign_path)
    # Submission may run inside another allocation. Its CPU masks, step options,
    # and end time must not leak into the new jobs; Slurm sets their own context.
    # Preserve the cluster connection configuration required by the site client.
    submission_environment = {key: value for key, value in os.environ.items()
                              if key in {"SLURM_CONF", "SLURM_CONF_SERVER"}
                              or not key.startswith(("SLURM_", "SLURMD_", "SRUN_", "SBATCH_"))}
    (campaign_path / "slurm").mkdir(parents=True, exist_ok=True)
    ledger = campaign_path / "submission.json"
    with (campaign_path / "submission.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if ledger.exists():
            if not resume:
                raise ValueError("Submission already recorded; use submit --resume after its jobs finish")
            previous = json.loads(ledger.read_text())
            if any(record["state"] != "submitted" or not record.get("job_id") for record in previous["jobs"]):
                raise ValueError("Previous submission has an unknown outcome; reconcile Slurm before resubmission")
            terminal = {"BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT", "REVOKED"}
            for record in previous["jobs"]:
                result = run(["sacct", "-X", "--noheader", "--parsable2", "-j", record["job_id"],
                              "--format=JobIDRaw,State"], capture_output=True, text=True, check=True, timeout=30)
                states = [line.split("|", 2)[1].split()[0].rstrip("+") for line in result.stdout.splitlines()
                          if line.split("|", 1)[0] == record["job_id"] and "|" in line]
                if not states or any(state not in terminal for state in states):
                    raise ValueError(f"Previous Slurm job {record['job_id']} is live or has no confirmed terminal state")
            if requested_source is None:
                resume_source = previous.get("resume_source", previous.get("source"))
                if resume_source is not None:
                    source = select_source(campaign_path, Path(resume_source))
            if recovery_allowance is None and previous.get("recovery_allowance"):
                recovery_allowance = Path(previous["recovery_allowance"])
            save_json(campaign_path / ("submission-" + uuid.uuid4().hex + ".json"), previous)
        elif resume:
            raise ValueError("No prior submission exists; omit --resume")
        if recovery_allowance is not None:
            from deepswe_recovery import load_allowance
            from deepswe_admission import reset_admission
            allowance = load_allowance(campaign_path, recovery_allowance, source)
            # Only reached after every old writer has a confirmed terminal state.
            reset_admission(campaign_path, allowance["id"])
        submission_environment.pop("ATP_DEEPSWE_RECOVERY_ALLOWANCE", None)
        records = [{"account": account, "shard_offset": offset, "state": "not_sent",
                    "command": ["sbatch", "--parsable", "--account=" + account, "--time=" + walltime,
                                "--chdir=" + str(app),
                                "--output=" + str(campaign_path / "slurm/deepswe-%j.out"),
                                "--error=" + str(campaign_path / "slurm/deepswe-%j.err"),
                                str(source / "deepswe.sbatch"),
                                "--app", str(app), "--campaign", str(campaign_path), "--source", str(source),
                                "--shard-offset", str(offset)]} for account, offset in ACCOUNTS]
        intent = {"created_at": time.time(), "source": str(source), "jobs": records}
        if recovery_allowance is not None:
            intent["recovery_allowance"] = str(Path(recovery_allowance).resolve())
            for record in records:
                record["command"].extend(["--recovery-allowance", intent["recovery_allowance"]])
        save_json(ledger, intent)
        for record in records:
            record.update(state="submitting", sent_at=time.time())
            save_json(ledger, intent)
            try:
                response = run(record["command"], cwd=app, env=submission_environment,
                               capture_output=True, text=True, timeout=60)
            except (OSError, subprocess.TimeoutExpired) as error:
                record.update(state="unknown", error=type(error).__name__)
                save_json(ledger, intent)
                raise RuntimeError("sbatch outcome is unknown; do not retry until Slurm reconciliation") from error
            output = response.stdout.strip()
            if response.returncode or not re.fullmatch(r"[0-9]+(?:;[^\s;]+)?", output):
                record.update(state="unknown", exit_code=response.returncode,
                              stdout=output, stderr=response.stderr)
                save_json(ledger, intent)
                raise RuntimeError("sbatch did not confirm a job ID; inspect submission.json and reconcile Slurm")
            record.update(state="submitted", job_id=output.split(";", 1)[0], confirmed_at=time.time())
            save_json(ledger, intent)
        return intent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--app", type=Path, default=APP)
    create.add_argument("--campaign", type=Path, required=True)
    create.add_argument("--dataset", type=Path, required=True)
    create.add_argument("--settings", type=Path)
    create.add_argument("--shards", type=int, default=10)
    create.add_argument("--task-limit", type=int)
    create.add_argument("--repeats", type=int)
    create.add_argument("--agent-timeout", type=int)
    create.add_argument("--concurrency", type=int, help="Initial global concurrency per model")
    work = commands.add_parser("worker")
    work.add_argument("--app", type=Path, default=APP)
    work.add_argument("--campaign", type=Path, required=True)
    work.add_argument("--shard-id", type=int, required=True)
    work.add_argument("--source", type=Path, help="Verified original or named campaign source revision")
    work.add_argument("--recovery-allowance", type=Path)
    launch = commands.add_parser("submit")
    launch.add_argument("--app", type=Path, default=APP)
    launch.add_argument("--campaign", type=Path, required=True)
    launch.add_argument("--time", default="24:00:00")
    launch.add_argument("--resume", action="store_true", help="Submit again only after all prior jobs are terminal")
    launch.add_argument("--source", type=Path, help="Verified original or named campaign source revision")
    launch.add_argument("--recovery-allowance", type=Path)
    recovery = commands.add_parser("prepare-recovery")
    recovery.add_argument("--campaign", type=Path, required=True)
    recovery.add_argument("--source", type=Path, required=True)
    freeze = commands.add_parser("freeze-source")
    freeze.add_argument("--app", type=Path, default=APP)
    freeze.add_argument("--campaign", type=Path, required=True)
    freeze.add_argument("--revision", required=True)
    concurrency = commands.add_parser("set-concurrency")
    concurrency.add_argument("--campaign", type=Path, required=True)
    concurrency.add_argument("--value", type=int, required=True,
                             help="Current adaptive budget per model; pressure can reduce it")
    status = commands.add_parser("status")
    status.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.command == "worker":
        return asyncio.run(worker(args))
    if args.command == "prepare-recovery":
        from deepswe_recovery import prepare_allowance
        print(prepare_allowance(args.campaign, args.source))
        return 0
    if args.command == "freeze-source":
        from deepswe_source import freeze_source
        print(freeze_source(args.app, args.campaign, revision=args.revision))
        return 0
    if args.command == "set-concurrency":
        from deepswe_admission import set_concurrency
        print(json.dumps(set_concurrency(args.campaign, args.value), indent=2))
        return 0
    if args.command == "submit":
        result = submit(args.campaign.resolve(), args.app.resolve(), args.time, resume=args.resume,
                        source=args.source, recovery_allowance=args.recovery_allowance)
    else:
        import deepswe_campaign as campaign
        if args.command == "create":
            settings = json.loads((args.settings or args.app / "configs/deepswe.json").read_text())
            settings.setdefault("max_concurrency_per_model", 2048)
            if args.agent_timeout is not None:
                if args.agent_timeout <= 0:
                    raise ValueError("Agent timeout must be positive")
                settings["agent_timeout_seconds"] = args.agent_timeout
            if args.concurrency is not None:
                if args.concurrency < args.shards:
                    raise ValueError("Global model concurrency must be >= shard count")
                settings["initial_concurrency_per_model"] = args.concurrency
            initial, maximum = settings["initial_concurrency_per_model"], settings["max_concurrency_per_model"]
            if (type(initial) is not int or type(maximum) is not int
                    or not args.shards <= initial <= maximum <= 2048):
                raise ValueError("Concurrency must satisfy shard count <= initial <= maximum <= 2048")
            result = campaign.create_campaign(args.campaign.resolve(), args.dataset.resolve(), settings,
                       shards=args.shards, task_limit=args.task_limit, repeats=args.repeats)
            from deepswe_source import freeze_source
            freeze_source(args.app.resolve(), args.campaign.resolve())
        else:
            ledger = campaign.read_json(args.campaign / "submission.json", {})
            if ledger.get("recovery_allowance"):
                from deepswe_recovery import load_allowance
                load_allowance(args.campaign, ledger["recovery_allowance"], ledger.get("source"))
                os.environ["ATP_DEEPSWE_RECOVERY_ALLOWANCE"] = ledger["recovery_allowance"]
            result = campaign.summarize(args.campaign.resolve())
    print(json.dumps(redact(result), indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print("DeepSWE refused or failed: " + redact(str(error)), file=sys.stderr)
        raise SystemExit(1)
