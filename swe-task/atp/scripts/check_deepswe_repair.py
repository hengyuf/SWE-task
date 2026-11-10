#!/usr/bin/env python3
"""Live CPU-job check of independent process control and deleted-PTY cleanup; no model calls."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import shlex
from types import SimpleNamespace
import time

import deepswe_capture as capture
from run_deepswe import resolved_config


PAYLOAD = b"PTY-repair\x00\xff" * 8192
TAIL = b"PTY-tail\x00\xfe" * 8192


def probe_command(*, termination_tail=False):
    program = (
        "import os,signal,sqlite3,sys,time;from pathlib import Path;"
        f"root=Path({capture.REMOTE_DATA!r})/'opencode';root.mkdir(parents=True,exist_ok=True);"
        "db=sqlite3.connect(root/'opencode.db');db.execute('create table probe(value text)');"
        "db.execute(\"insert into probe values ('durable')\");db.commit();db.close();"
        "payload=b'PTY-repair\\x00\\xff'*8192\n"
    )
    if termination_tail:
        program += ("def stopped(*args):\n"
                    " tail=b'PTY-tail\\x00\\xfe'*8192;os.write(1,tail);os.write(2,tail);sys.exit(0)\n"
                    "signal.signal(signal.SIGTERM,stopped)\n")
    program += ("os.write(1,payload);os.write(2,payload);"
                f"Path({capture.REMOTE_CAPTURE + '/probe-ready'!r}).write_text('ready');time.sleep(180)")
    return "python3 -c " + shlex.quote(program)


async def run_check(arguments):
    from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, create_provider
    from nemo_gym.sandbox.config import resolve_provider_config

    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(ATP_DEEPSWE_CAMPAIGN=str(output), ATP_DEEPSWE_MODEL_ALIAS="repair")
    config = resolved_config(arguments.config)
    owned = []
    result = {"passed": False, "purpose": "pty_repair_probe_excluded_from_evaluation", "cases": []}

    async def create_case(index):
        directory = output / f"models/repair/attempts/{index}-0/attempt-0"
        directory.mkdir(parents=True)
        body = {"_atp_attempt_dir": str(directory), "_atp_deadline": time.time() + arguments.timeout}
        capture._write_json(directory / "started.json", {
            "purpose": result["purpose"], "started_at": time.time(), "deadline": body["_atp_deadline"],
            "job_id": os.environ["SLURM_JOB_ID"],
        })
        context = capture._ATTEMPT.set(capture._context(body))
        try:
            sandbox = AsyncSandbox(create_provider(resolve_provider_config("sandbox", config)))
            await sandbox.start(SandboxSpec(image=arguments.image, ttl_s=900, ready_timeout_s=180,
                resources={"cpu": 1, "memory_mib": 1024}, metadata=capture.attempt_metadata(),
                provider_options={"network_policy": {"defaultAction": "deny", "egress": []}}))
            descriptor = await sandbox.serialize()
            owned.append((sandbox, descriptor, directory))
            await capture.register_sandbox(sandbox, "agent")
        finally:
            capture._ATTEMPT.reset(context)
        return sandbox, directory, body

    async def exercise():
        sandbox, directory, body = await create_case(0)
        created = asyncio.Event()
        original = None

        async def create(**kwargs):
            nonlocal original
            original = await sandbox.pty.create(**kwargs)
            created.set()
            return original

        proxy = SimpleNamespace(pty=SimpleNamespace(create=create),
                                exec=sandbox.exec, download=sandbox.download)

        async def request_json():
            return body

        running = asyncio.create_task(capture.run_opencode(proxy, probe_command(),
                                      SimpleNamespace(json=request_json), arguments.timeout))
        try:
            await asyncio.wait_for(created.wait(), timeout=60)

            async def wait_for_payload():
                while any(not (directory / name).exists() or (directory / name).stat().st_size < len(PAYLOAD)
                          for name in ("stdout.log", "stderr.log")):
                    if running.done():
                        await running
                        raise RuntimeError("Probe command exited before its known output arrived")
                    await asyncio.sleep(0.1)

            await asyncio.wait_for(wait_for_payload(), timeout=60)
            # Fault injection: terminate only the client's pump. The remote
            # command remains alive, while its original exit future is poisoned.
            original._pump_task.cancel()
            await asyncio.gather(original._pump_task, return_exceptions=True)
            try:
                await running
            except Exception as error:
                if type(error).__name__ != "SandboxPtyError":
                    raise
            else:
                raise RuntimeError("Fault injection did not interrupt the original PTY")
            quiescence = json.loads((directory / "generation-quiescence.json").read_text())
            events = [json.loads(line) for line in (directory / "capture-events.jsonl").read_text().splitlines()]
            if not quiescence.get("confirmed") or not any(event["event"] == "agent_control_recovered" and
                    event.get("control_channel") == "sandbox_exec" for event in events):
                raise RuntimeError("Independent exec control did not confirm process-group termination")
            for name in ("stdout.log", "stderr.log", "remote-stdout.log", "remote-stderr.log"):
                if (directory / name).read_bytes() != PAYLOAD:
                    raise RuntimeError(f"Known probe bytes were not preserved in {name}")
            if (directory / "streams-complete.json").exists():
                raise RuntimeError("Broken original streams must remain explicitly incomplete")
            if not (directory / "remote-streams-complete.json").is_file():
                raise RuntimeError("Broken-PTY recovery did not prove complete remote streams")
            result["cases"].append({"case": "poisoned_exit_future", "passed": True,
                                    "exact_bytes_per_stream": len(PAYLOAD)})
            # Deliberately repeat the production wrapper, not an idempotent
            # probe body. Its second CREATE TABLE would fail if the guard broke.
            wrapped = json.loads((directory / "agent-command.json").read_text())["wrapped_command"]
            replay = await sandbox.exec("bash -c " + shlex.quote(wrapped), timeout_s=15)
            if replay.return_code != 125:
                raise RuntimeError("The production once guard did not refuse command replay")
            await capture.snapshot_agent(sandbox, directory, final=True)
            for name in ("remote-stdout.log", "remote-stderr.log"):
                if (directory / name).read_bytes() != PAYLOAD:
                    raise RuntimeError(f"Replay guard did not preserve {name}")
            result["cases"].append({"case": "command_replay_guard", "passed": True})
        finally:
            if not running.done():
                running.cancel()
            await asyncio.gather(running, return_exceptions=True)

        sandbox, directory, _ = await create_case(1)
        wrapped = (f"mkdir -p {capture.REMOTE_CAPTURE}; " + probe_command() +
                   f" > {capture.REMOTE_CAPTURE}/stdout.log 2> {capture.REMOTE_CAPTURE}/stderr.log")
        session = await sandbox.pty.create(command="bash -c " + shlex.quote(wrapped), pty=False)
        capture._write_json(directory / "pty-session.json", {"session_id": session.session_id})
        capture._write_json(directory / "generation-quiescence.json", {"confirmed": False})
        try:
            async def wait_for_remote_payload():
                while True:
                    check = await sandbox.exec(f"test -f {capture.REMOTE_CAPTURE}/probe-ready", timeout_s=10)
                    if check.return_code == 0:
                        return
                    await asyncio.sleep(0.1)
            await asyncio.wait_for(wait_for_remote_payload(), timeout=60)
        finally:
            await session.close()  # Recovery must work without any attachable PTY.
        report = await capture.recover_attempt(directory, config)
        events = [json.loads(line) for line in (directory / "capture-events.jsonl").read_text().splitlines()]
        if report["remaining_sandboxes"] or not any(event["event"] == "recovery_process_unconfirmed" for event in events):
            raise RuntimeError("Deleted-PTY recovery did not harvest and stop its owned sandbox")
        if (directory / "model.patch").exists() or (directory / "agent-final-snapshot.json").exists():
            raise RuntimeError("Unconfirmed generation was incorrectly finalized")
        for name in ("remote-stdout.log", "remote-stderr.log"):
            if (directory / name).read_bytes() != PAYLOAD:
                raise RuntimeError(f"Deleted-PTY recovery lost known bytes in {name}")
        result["cases"].append({"case": "deleted_pty", "passed": True, "partial_artifacts_preserved": True})

        sandbox, directory, body = await create_case(2)

        async def deadline_request_json():
            return body

        try:
            await capture.run_opencode(sandbox, probe_command(termination_tail=True),
                                       SimpleNamespace(json=deadline_request_json), timeout_s=15)
        except TimeoutError:
            pass
        else:
            raise RuntimeError("Deadline probe did not reach its controlled timeout")
        proof = json.loads((directory / "process-control.json").read_text())
        if not proof.get("wrapper", {}).get("confirmed") or proof.get("wrapper_forced_stop") is not False:
            raise RuntimeError("Deadline did not let the wrapper and writers drain without a forced stop")
        for name in ("remote-stdout.log", "remote-stderr.log"):
            if (directory / name).read_bytes() != PAYLOAD + TAIL:
                raise RuntimeError(f"Deadline lost the producer's final known tail in {name}")
        if not (directory / "remote-streams-complete.json").is_file():
            raise RuntimeError("Deadline capture did not prove complete remote streams")
        result["cases"].append({"case": "deadline_writer_drain", "passed": True,
                                "exact_bytes_per_stream": len(PAYLOAD + TAIL),
                                "final_tail_bytes_per_stream": len(TAIL)})

    try:
        await asyncio.wait_for(exercise(), timeout=arguments.timeout)
        result["checks_passed"] = True
    except BaseException as error:
        result.update(checks_passed=False, error_type=type(error).__name__, error=str(error))
    finally:
        capture._write_json(output / "check-result.json", result)
        cleanup_errors = []
        for sandbox, descriptor, directory in owned:
            if capture._stop_confirmed(directory, descriptor, "agent"):
                continue
            try:
                await capture.snapshot_agent(sandbox, directory, final=False)
                await asyncio.wait_for(sandbox.stop(), timeout=60)
                capture.mark_sandbox_stopped(descriptor, "agent", directory)
            except Exception as error:
                cleanup_errors.append({"sandbox_id": descriptor["sandbox_id"], "error": str(error)})
        result.update(cleanup_errors=cleanup_errors, cleanup_complete=not cleanup_errors,
                      passed=bool(result.get("checks_passed")) and not cleanup_errors, finished_at=time.time())
        capture._write_json(output / "check-result.json", result)
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Existing Gym YAML, resolved using the staged environment")
    parser.add_argument("--image", required=True, help="Existing DeepSWE task image with Python 3")
    parser.add_argument("--output", required=True, type=Path, help="New probe directory outside evaluation attempts")
    parser.add_argument("--timeout", type=float, default=600)
    arguments = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or any(os.environ.get(key, "0") not in {"", "0"}
            for key in ("SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE")):
        raise RuntimeError("Run the repair probe inside a CPU allocation without GPUs")
    return asyncio.run(run_check(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
