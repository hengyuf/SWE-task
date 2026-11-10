#!/usr/bin/env python3
"""Check the sandbox callback and pinned golden verifier in an isolated CPU-job campaign."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import shlex
import signal
import sys
import tempfile
import time

from deepswe_campaign import create_campaign, write_shard
from deepswe_capture import _atomic_copy, recover_attempt
from deepswe_stage import RESOURCE_NAME, _descriptor, build_config, local_path, runtime_env, stage
from run_deepswe import drain_output, finish_output, job_end_time, mark_stopped, resolved_config, save_json, stop_process


def prepare_request(staged, output, launch, deadline):
    return prepare_requests(staged, output, launch, deadline, task_limit=1)[0]


def prepare_requests(staged, output, launch, deadline, *, task_limit=None):
    settings = dict(staged["settings"], models={"super": staged["settings"]["models"]["super"]}, repeats=1)
    manifest = create_campaign(output, staged["dataset"], settings, shards=1, task_limit=task_limit, repeats=1)
    requests, tasks = [], []
    for task in manifest["tasks"]:
        row = dict(task)
        index = row["_ng_task_index"]
        attempt = output / f"models/super/attempts/{index}-0/attempt-0"
        attempt.mkdir(parents=True, exist_ok=False)
        source = staged["tasks_dir"] / row["task_id"] / "solution/solution.patch"
        _atomic_copy(source, attempt / "model.patch")
        row.update(_ng_rollout_index=0, _ng_attempt_index=0, _atp_attempt_dir=str(attempt),
                   _atp_deadline=deadline, _atp_regrade=True, _atp_grade_attempt=0)
        row["response"] = {"output": [], "id": f"golden-check-{index}", "created_at": int(time.time()),
                           "model": "golden-check", "object": "response", "parallel_tool_calls": False,
                           "tool_choice": "auto", "tools": []}
        save_json(attempt / "started.json", {"purpose": "golden_verifier_and_callback_check", "started_at": time.time(),
                  "job_id": os.environ.get("SLURM_JOB_ID"), "launch_dir": str(launch), "deadline": deadline,
                  "task_index": index, "repeat_index": 0, "attempt_index": 0, "model": "super", "request": row})
        save_json(attempt / "request.json", row)
        requests.append((row, attempt))
        tasks.append({"task_id": row["task_id"], "golden_source": str(source), "attempt_dir": str(attempt)})
    save_json(output / "check.json", {"purpose": "preflight_only_excluded_from_evaluation", "tasks": tasks,
                                     "task_count": len(tasks), "created_at": time.time()})
    return requests


def check_result(result, attempt):
    if result.get("evaluation_completed") is not True or result.get("reward") != 1:
        raise RuntimeError("The pinned golden patch did not receive a completed passing verifier result")
    artifacts = json.loads((attempt / "grading/attempt-0/capture.json").read_text())
    if not artifacts.get("artifact_complete"):
        raise RuntimeError("Golden verification finished without all required verifier artifacts")


async def verify_golden_tasks(client, resource_url, requests, concurrency):
    gate = asyncio.Semaphore(concurrency)

    async def verify(row, attempt):
        async with gate:
            outcome = {"task_id": row["task_id"], "passed": False, "started_at": time.time()}
            try:
                async with client.post(resource_url + "/verify", json=row) as response:
                    raw = await response.read()
                    with (attempt / "verify-http-body.bin").open("xb") as stream:
                        stream.write(raw)
                        stream.flush()
                        os.fsync(stream.fileno())
                    metadata = {"status": response.status, "body_path": "verify-http-body.bin"}
                    save_json(attempt / "verify-http.json", metadata)
                    payload = json.loads(raw)
                    save_json(attempt / "verify-http.json", dict(metadata, body=payload))
                    response.raise_for_status()
                check_result(payload, attempt)
                outcome.update(passed=True, reward=payload["reward"], evaluation_completed=True)
            except Exception as error:
                outcome.update(error_type=type(error).__name__, error=str(error))
            except asyncio.CancelledError:
                outcome.update(error_type="CancelledError", error="Golden check interrupted")
                raise
            finally:
                outcome["finished_at"] = time.time()
                save_json(attempt / "check-result.json", outcome)
                print(f"Golden {row['task_id']}: {'PASS' if outcome['passed'] else 'FAIL'}", flush=True)
            return outcome

    return await asyncio.gather(*(verify(row, attempt) for row, attempt in requests))


async def wait_ready(client, base_urls, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.returncode is not None:
            raise RuntimeError(f"Gym exited during preflight startup: {process.returncode}")
        ready = True
        for url in base_urls:
            try:
                async with client.get(url + "/", timeout=5) as response:
                    await response.read()
            except (OSError, TimeoutError):
                ready = False
                break
        if ready:
            return
        await asyncio.sleep(1)
    raise TimeoutError("Gym preflight servers did not become reachable")


async def callback_probe(client, resource_url, model_url, row, full_config, attempt, timeout):
    from nemo_gym.sandbox import AsyncSandbox, create_provider
    from nemo_gym.sandbox.config import resolve_provider_config

    sandbox = None
    try:
        async with client.post(resource_url + "/seed_session", json=row) as response:
            response.raise_for_status()
            seed = await response.json()
        sandbox = await AsyncSandbox.connect(seed["sandbox_descriptor"],
                    provider=create_provider(resolve_provider_config("sandbox", full_config)))
        headers = {"Content-Type": "application/json", "x-atp-attempt-dir": str(attempt),
                   "x-atp-deadline": str(min(row["_atp_deadline"], time.time() + timeout)),
                   "x-atp-rollout-id": "golden-callback-check"}
        body = {"model": "dummy_model", "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 32, "stream": False}
        code = ("import json,urllib.request;" +
                f"request=urllib.request.Request({model_url + '/v1/chat/completions'!r}," +
                f"data=json.dumps({body!r}).encode(),headers={headers!r});" +
                f"response=urllib.request.urlopen(request,timeout={timeout!r});" +
                "data=response.read();print(data.decode());" +
                "assert response.status==200 and json.loads(data).get('choices')")
        result = await sandbox.exec("python3 -c " + shlex.quote(code), timeout_s=timeout + 15)
        save_json(attempt / "callback.json", {"return_code": result.return_code,
                  "stdout": result.stdout, "stderr": result.stderr, "finished_at": time.time()})
        if result.return_code != 0:
            raise RuntimeError("The sandbox could not complete a model call through the local Gym proxy")
    except BaseException as error:
        save_json(attempt / "callback-failure.json", {"error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        if sandbox is not None:
            await sandbox.stop()
            save_json(attempt / "callback-stopped.json", {"stopped_at": time.time(),
                      "sandbox_id": seed["sandbox_descriptor"]["sandbox_id"]})


async def run_check(arguments, staged):
    import aiohttp

    environment = dict(staged["env"])
    output = arguments.output.resolve()
    launch = output / "launch"
    end = job_end_time(environment)
    deadline = end - staged["settings"]["shutdown_grace_seconds"]
    if deadline - time.time() < 120:
        raise RuntimeError("Not enough allocation time remains for a preflight and cleanup")
    environment.update(ATP_DEEPSWE_CAMPAIGN=str(output), ATP_DEEPSWE_MODEL_ALIAS="super",
                       ATP_DEEPSWE_SHARD_ID="0",
                       ATP_DEEPSWE_LAUNCH_DIR=str(launch), ATP_DEEPSWE_API_CONCURRENCY="1",
                       ATP_DEEPSWE_JOB_DEADLINE=str(deadline), NEMO_GYM_RUN_ID=output.name)
    environment["RAY_TMPDIR"] = tempfile.mkdtemp(prefix=f"atpr-{os.getuid()}-", dir="/tmp")
    os.environ.update(environment)
    requests = prepare_requests(staged, output, launch, deadline, task_limit=None if arguments.all_tasks else 1)
    row, attempt = requests[0]
    dataset = write_shard(output, 0, launch / "tasks.jsonl")
    config = build_config(staged, output, launch, "super", dataset, concurrency=1)
    full_config = resolved_config(config)
    model_config = full_config["policy_model"]["responses_api_models"]["vllm_model"]
    resource_config = full_config[RESOURCE_NAME]["resources_servers"]["deepswe"]
    model_url = f"http://{model_config['host']}:{model_config['port']}"
    resource_url = f"http://{resource_config['host']}:{resource_config['port']}"
    command = [str(staged["gym_cli"]), "env", "start", "--config", str(config)]
    save_json(launch / "launch.json", {"command": command, "purpose": "golden_and_callback_preflight",
                                      "started_at": time.time(), "slurm_job_id": environment["SLURM_JOB_ID"]})
    process = await asyncio.create_subprocess_exec(*command, cwd=staged["gym"], env=environment,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
    output_task = asyncio.create_task(drain_output(process, launch / "gym.log", environment, "golden-check"))
    current = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, current.cancel)
    outcome = {"passed": False, "purpose": "preflight_only_excluded_from_evaluation",
               "task_count": len(requests), "golden_only": arguments.golden_only, "concurrency": arguments.concurrency}
    try:
        async with asyncio.timeout(max(1, deadline - time.time())):
            async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True),
                        timeout=aiohttp.ClientTimeout(total=None, sock_connect=15)) as client:
                await wait_ready(client, [resource_url, model_url], process, arguments.startup_timeout)
                if not arguments.golden_only:
                    await callback_probe(client, resource_url, model_url, row, full_config, attempt,
                                         min(arguments.callback_timeout, max(1, deadline - time.time())))
                results = await verify_golden_tasks(client, resource_url, requests, arguments.concurrency)
                outcome.update(passed=all(result["passed"] for result in results), results=results,
                               passed_tasks=sum(result["passed"] for result in results))
    except BaseException as error:
        outcome.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        outcome.update(verification_passed=outcome["passed"], passed=False, cleanup_pending=True, cleanup_complete=False)
        save_json(output / "check-result.json", outcome)
        try:
            await stop_process(process)
            mark_stopped(launch)
            gate = asyncio.Semaphore(arguments.concurrency)

            async def recover(attempt):
                async with gate:
                    try:
                        return await recover_attempt(attempt, full_config)
                    except Exception as error:
                        save_json(attempt / "recovery-error.json", {"error_type": type(error).__name__, "error": str(error)})
                        return {"remaining_sandboxes": [str(attempt)]}

            recovery = await asyncio.wait_for(asyncio.gather(*(recover(attempt) for _, attempt in requests)),
                                             timeout=max(1, min(240, end - time.time() - 10)))
            if any(item["remaining_sandboxes"] for item in recovery):
                outcome.update(passed=False, cleanup_incomplete=True)
                raise RuntimeError("Preflight sandbox cleanup is incomplete; inspect recovery.json")
            await finish_output(output_task, launch, timeout=max(1, min(10, end - time.time() - 5)))
            if not json.loads((launch / "console-capture.json").read_text()).get("complete"):
                raise RuntimeError("Preflight console capture did not reach EOF")
            outcome.update(passed=outcome["verification_passed"], cleanup_complete=True)
        except BaseException as error:
            outcome.update(passed=False, cleanup_error_type=type(error).__name__, cleanup_error=str(error))
            raise
        finally:
            outcome["cleanup_pending"] = False
            outcome["finished_at"] = time.time()
            save_json(output / "check-result.json", outcome)
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.remove_signal_handler(sig)
    return 0 if outcome["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New preflight directory, separate from evaluation")
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--callback-timeout", type=float, default=300)
    parser.add_argument("--golden-only", action="store_true", help="Skip the hosted model callback")
    parser.add_argument("--all-tasks", action="store_true", help="Verify every pinned task's known-correct patch")
    parser.add_argument("--concurrency", type=int, default=32, help="Maximum simultaneous verifier sandboxes")
    parser.add_argument("--reuse-staged", action="store_true", help="Use an existing local runtime without staging again")
    arguments = parser.parse_args()
    if arguments.concurrency < 1 or arguments.concurrency > 113:
        parser.error("--concurrency must be between 1 and 113")
    if arguments.all_tasks and not arguments.golden_only:
        parser.error("--all-tasks requires --golden-only")
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_JOB_PARTITION", "").startswith("cpu"):
        raise RuntimeError("DeepSWE preflight requires a CPU Slurm allocation")
    if any(os.environ.get(key, "0") not in {"", "0"} for key in ("SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE")):
        raise RuntimeError("DeepSWE preflight requires a CPU allocation without GPUs")
    if arguments.reuse_staged:
        scratch = local_path(arguments.scratch)
        ready = json.loads((scratch / "staged.json").read_text())
        app = arguments.app.resolve()
        if ready["app"] != str(app):
            raise ValueError("Prepared runtime belongs to another application")
        staged = _descriptor(app, scratch, ready["settings"], runtime_env(app, scratch))
        for key in ("python", "gym_cli", "dataset"):
            if not staged[key].is_file():
                raise RuntimeError(f"Prepared scratch is incomplete: {key}")
    else:
        staged = stage(arguments.app.resolve(), arguments.scratch)
    if Path(sys.executable).resolve() != staged["python"].resolve():
        os.execve(staged["python"], [str(staged["python"]), str(staged["scratch"] / "scripts/check_deepswe.py"),
                  *sys.argv[1:]], staged["env"])
    return asyncio.run(run_check(arguments, staged))


if __name__ == "__main__":
    raise SystemExit(main())
