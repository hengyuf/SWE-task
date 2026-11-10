#!/usr/bin/env python3
"""Validate one OpenCode callback sandbox and one SWE-bench golden patch."""

import asyncio
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

from atp_runtime import app_root, build_config, make_run_dir, runtime_env


RUN_KEY = "nemo-gym.nvidia.com/run"
USER_KEY = "nemo-gym.nvidia.com/user"


def redacted(value):
    if isinstance(value, str):
        credentials = ("OPENSANDBOX_API_KEY", "HF_TOKEN", "NVIDIA_API_KEY", "WANDB_API_KEY")
        secrets = {os.environ[name] for name in credentials if os.environ.get(name)}
        for secret in sorted(secrets, key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {redacted(key): redacted(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redacted(item) for item in value]
    return value


def normalized_label(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")[:63].strip("._-") or "metadata"


def stop_owned_processes(process):
    for sig, timeout in ((signal.SIGINT, 15), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
        # The shell can exit before its server children; signal the same owned group again.
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Owned Gym process did not exit after cleanup") from exc


async def wait_ready(client, process, urls):
    import httpx

    deadline = time.monotonic() + 180
    pending = set(urls)
    while pending and time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Gym exited before its CPU servers became ready; see gym.log")
        for url in tuple(pending):
            try:
                response = await client.get(url, timeout=3)
                if response.status_code == 200 and response.json().get("status") == "ok":
                    pending.remove(url)
            except (httpx.HTTPError, ValueError, AttributeError):
                pass
        if pending:
            await asyncio.sleep(1)
    if pending:
        raise RuntimeError("CPU Gym server readiness timed out; see gym.log")


async def callback_check(config, row, proxy_url, scope, report):
    from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
    from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata

    sandbox = AsyncSandbox(resolve_provider_config("sandbox", config))
    task_id = row["instance_id"].lower().replace("__", "_1776_")
    image = f"docker.io/swebench/sweb.eval.x86_64.{task_id}:latest"
    spec = SandboxSpec(
        image=image,
        ttl_s=1800,
        ready_timeout_s=1200,
        resources=SandboxResources(cpu=2, memory_mib=16384, disk_gib=30),
        metadata={**resolve_provider_metadata("sandbox", config), **scope, "atp-check": "callback"},
        env={"EXECD_API_GRACE_SHUTDOWN": "50ms"},
    )
    try:
        await sandbox.start(spec)
        descriptor = await sandbox.serialize()
        report["callback"] = {"sandbox_id": descriptor.get("sandbox_id"), "image": image}
        health = await sandbox.exec(
            f"curl --noproxy '*' --fail --silent --show-error --connect-timeout 10 --max-time 30 {quote_shell(proxy_url)}",
            timeout_s=40,
        )
        if health.return_code != 0 or json.loads(health.stdout or "{}").get("status") != "ok":
            raise RuntimeError("OpenSandbox could not reach the Gym model proxy's health route")
        report["callback"]["reachable"] = True
        installed = await sandbox.exec(
            "set -eu; installer=$(mktemp); "
            "curl -fL --connect-timeout 15 --max-time 120 -o \"$installer\" https://opencode.ai/install; "
            "VERSION=1.17.11 bash \"$installer\"; "
            "/root/.opencode/bin/opencode --version",
            timeout_s=600,
            user="root",
        )
        version_lines = (installed.stdout or "").splitlines()
        version = next((line.strip() for line in reversed(version_lines) if line.strip() == "1.17.11"), None)
        if installed.return_code != 0 or version != "1.17.11":
            raise RuntimeError("Pinned OpenCode 1.17.11 installation/version check failed")
        report["callback"]["opencode_version"] = version
    finally:
        await sandbox.stop()
        report.setdefault("callback", {})["stop_completed"] = True


def quote_shell(value):
    import shlex

    return shlex.quote(value)


async def cleanup_owned_sandboxes(config, scope):
    """Match both ownership labels before deleting, following Gym's cleanup helper."""
    import httpx

    connection = config["sandbox"]["opensandbox"]["connection"]
    domain = connection["domain"].rstrip("/")
    base = domain if "://" in domain else f"{connection.get('protocol', 'http')}://{domain}"
    deleted = []
    async with httpx.AsyncClient(
        headers={"OPEN-SANDBOX-API-KEY": connection["api_key"]},
        verify=connection.get("tls_verify", True),
        timeout=30,
        follow_redirects=False,
        trust_env=False,
    ) as client:
        for sweep in range(3):
            matches = []
            for page in range(1, 1001):
                response = await client.get(f"{base}/v1/sandboxes", params={"page": page, "pageSize": 100})
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload.get("items"), list):
                    raise RuntimeError("OpenSandbox returned an invalid sandbox listing")
                for item in payload["items"]:
                    metadata = item.get("metadata") or {}
                    if all(metadata.get(key) == value for key, value in scope.items()):
                        if not isinstance(item.get("id"), str) or not item["id"]:
                            raise RuntimeError("Owned sandbox listing has no valid ID")
                        matches.append(item["id"])
                has_next = payload.get("pagination", {}).get("hasNextPage")
                if has_next is False:
                    break
                if has_next is not True:
                    raise RuntimeError("OpenSandbox returned invalid pagination")
            else:
                raise RuntimeError("OpenSandbox sandbox listing exceeded the page bound")
            if not matches:
                return {"verified_empty": True, "deleted_ids": deleted}
            for sandbox_id in matches:
                response = await client.delete(f"{base}/v1/sandboxes/{quote(sandbox_id, safe='')}")
                if response.status_code != 404:
                    response.raise_for_status()
                deleted.append(sandbox_id)
        raise RuntimeError("Owned OpenSandbox resources remain after cleanup")


async def check(app, run_dir, config_path, environment, report):
    import httpx
    from omegaconf import OmegaConf
    from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig

    parsed = GlobalConfigDictParser().parse(GlobalConfigDictParserConfig(
        initial_global_config_dict=OmegaConf.create({"config_paths": [str(config_path)]}),
        skip_load_from_cli=True,
        skip_load_from_dotenv=True,
        offline=True,
    ))
    config = OmegaConf.to_container(parsed, resolve=True)
    scope = {RUN_KEY: normalized_label(run_dir.name), USER_KEY: normalized_label(environment["NEMO_GYM_USER"])}
    metadata = config["sandbox"].get("default_metadata", {})
    if any(metadata.get(key) != value for key, value in scope.items()):
        raise RuntimeError("Validation config must explicitly stamp this run and user on sandbox.default_metadata")
    proxy = config["policy_model"]["responses_api_models"]["vllm_model"]
    resource = config["swebench_resources_server"]["resources_servers"]["swebench"]
    if not resource.get("is_verifying_golden_patch"):
        raise RuntimeError("CPU validation requires the golden-patch verifier mode")
    proxy_url = f"http://{proxy['host']}:{proxy['port']}/"
    resource_url = f"http://{resource['host']}:{resource['port']}"
    dataset = app / "vendor/Gym/benchmarks/swebench/data/swebench_verified_benchmark.jsonl"
    with dataset.open() as handle:
        row = json.loads(next(handle))
    report.update(instance_id=row["instance_id"], dataset=str(dataset), dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest())
    process = None
    try:
        with (run_dir / "gym.log").open("w") as gym_log:
            process = subprocess.Popen(
                [str(app / "runtime/controller/bin/gym"), "env", "start", "--config", str(config_path)],
                cwd=app / "vendor/Gym", env=environment,
                stdout=gym_log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            report["gym_pid"] = process.pid
            async with httpx.AsyncClient(trust_env=False) as client:
                await wait_ready(client, process, [proxy_url, resource_url + "/"])
                print("CPU Gym services ready; checking remote callback and OpenCode 1.17.11", flush=True)
                await callback_check(config, row, proxy_url, scope, report)
                report["callback_passed"] = True
                report["opencode_version"] = report["callback"]["opencode_version"]
                request = dict(row)
                request["responses_create_params"] = {"input": []}
                request["response"] = {
                    "output": [], "id": "", "created_at": 0, "model": "", "object": "response",
                    "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
                }
                print(f"Callback passed; verifying golden patch for {row['instance_id']}", flush=True)
                response = await client.post(resource_url + "/verify", json=request, timeout=3600)
                response.raise_for_status()
                result = response.json()
                report["golden"] = {key: result.get(key) for key in (
                    "instance_id", "evaluation_completed", "resolved", "reward", "log_dir", "verifier_sandbox_observation",
                )}
                (run_dir / "golden_test_output.txt").write_text(redacted(result.get("test_output", "")))
                if result.get("evaluation_completed") is not True or result.get("resolved") is not True or result.get("reward") != 1:
                    raise RuntimeError("Golden patch did not resolve with a completed evaluation; see check.json and verifier logs")
                report["golden_patch_passed"] = True
    finally:
        try:
            if process is not None:
                await asyncio.to_thread(stop_owned_processes, process)
        finally:
            report["cleanup"] = await asyncio.wait_for(cleanup_owned_sandboxes(config, scope), timeout=120)


def main():
    if len(sys.argv) != 1:
        raise SystemExit("check_cpu.py takes no arguments")
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("CPU validation requires this application's own Slurm allocation")
    app = app_root()
    environment = runtime_env()
    os.environ.clear()
    os.environ.update(environment)
    os.chdir(app / "vendor/Gym")
    run_dir = make_run_dir("check")
    environment["NEMO_GYM_RUN_ID"] = run_dir.name
    os.environ["NEMO_GYM_RUN_ID"] = run_dir.name
    report = {
        "status": "failed", "run_id": run_dir.name, "run_dir": str(run_dir),
        "slurm_job_id": environment["SLURM_JOB_ID"], "preparation_id": None,
        "golden_patch_passed": False, "callback_passed": False, "opencode_version": None,
        "runtime_helper_sha256": hashlib.sha256((app / "scripts/atp_runtime.py").read_bytes()).hexdigest(),
        "opensandbox_origin": {"host": environment["ATP_OS_HOST"], "protocol": environment["ATP_OS_PROTOCOL"],
                               "tls_verify": environment["ATP_OS_TLS_VERIFY"] == "true"},
    }
    try:
        preparation = json.loads((app / "runtime/preparation.json").read_text())
        report["preparation_id"] = preparation.get("preparation_id")
        if preparation.get("status") != "ready" or not report["preparation_id"]:
            raise RuntimeError("Application preparation has not completed successfully")
        config_path = build_config(run_dir, "http://127.0.0.1:9/v1", mode="check")

        async def run():
            loop = asyncio.get_running_loop()
            task = asyncio.current_task()
            loop.add_signal_handler(signal.SIGTERM, task.cancel)
            await check(app, run_dir, config_path, environment, report)

        asyncio.run(run())
        report["status"] = "passed"
    except BaseException as exc:
        report["error"] = redacted(f"{type(exc).__name__}: {exc}")
    finally:
        report_text = json.dumps(redacted(report), indent=2) + "\n"
        (run_dir / "check.json").write_text(report_text)
        marker = app / "runtime/cpu-check.json"
        temporary_marker = marker.with_name(f".cpu-check.{os.getpid()}.json")
        temporary_marker.write_text(report_text)
        temporary_marker.replace(marker)
    print(f"CPU validation {report['status']}: {run_dir / 'check.json'}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
