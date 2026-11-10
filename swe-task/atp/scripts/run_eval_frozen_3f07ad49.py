#!/usr/bin/env python3
"""Run one explicitly allocated Lightning/OpenCode SWE evaluation; never submit jobs."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid


APP = Path(__file__).resolve().parents[1]
MODEL_PYTHON = "/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python"
CONTEXT = 262144


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def app_path(path):
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(APP):
        raise ValueError("Application paths must remain under agent-teacher-pipelines")
    return path


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_ready():
    from atp_runtime import GYM_COMMIT

    prepared = json.loads((APP / "runtime/preparation.json").read_text())
    checked = json.loads((APP / "runtime/cpu-check.json").read_text())
    if prepared.get("status") != "ready" or checked.get("status") != "passed":
        raise ValueError("Preparation and CPU checks must both pass before evaluation")
    if not prepared.get("preparation_id") or checked.get("preparation_id") != prepared["preparation_id"]:
        raise ValueError("CPU check does not match this preparation")
    if not (checked.get("golden_patch_passed") is True and checked.get("callback_passed") is True):
        raise ValueError("CPU golden-patch and sandbox callback checks must pass")
    if checked.get("opencode_version") != "1.17.11":
        raise ValueError("CPU check used an unexpected OpenCode version")
    if checked.get("runtime_helper_sha256") != sha256(APP / "scripts/atp_runtime.py"):
        raise ValueError("Runtime configuration changed after CPU validation")
    if prepared.get("gym_commit") != GYM_COMMIT or (APP / "vendor/Gym.commit").read_text().strip() != GYM_COMMIT:
        raise ValueError("Prepared Gym snapshot does not match its pinned commit")
    for overlay in prepared.get("overlays", []):
        if sha256(app_path(APP / "vendor/Gym" / overlay["target"])) != overlay["patched_sha256"]:
            raise ValueError("Prepared Gym overlay changed")
    for component in ("responses_api_models/vllm_model", "resources_servers/swebench", "responses_api_agents/opencode_sandboxed_agent"):
        venv = APP / "runtime/servers" / component / ".venv"
        if not (venv / "bin/python").is_file() or not (venv / "bin/activate").is_file():
            raise ValueError("Prepared server environment missing: " + component)
    model = app_path(prepared["model"]["path"])
    dataset = app_path(prepared["dataset"]["path"])
    if not (model / "config.json").is_file() or prepared["dataset"]["count"] != 500:
        raise ValueError("Prepared model or 500-task dataset is missing")
    index = model / "model.safetensors.index.json"
    if sha256(index) != prepared["model"]["index_sha256"]:
        raise ValueError("Prepared model index changed")
    shards = prepared["model"]["shards"]
    if not shards or set(json.loads(index.read_text())["weight_map"].values()) != set(shards):
        raise ValueError("Prepared model shard manifest does not match its index")
    for name, size in shards.items():
        shard = (model / name).resolve()
        if not shard.is_relative_to(model) or size <= 0 or shard.stat().st_size != size:
            raise ValueError("Prepared model shard changed: " + name)
    if sha256(dataset) != prepared["dataset"]["sha256"]:
        raise ValueError("Prepared dataset changed; repeat preparation and CPU checks")
    image_stat = Path(prepared["image"]["path"]).stat()
    if (image_stat.st_size, image_stat.st_mtime_ns) != (prepared["image"]["size"], prepared["image"]["mtime_ns"]):
        raise ValueError("Serving image changed after preparation")
    serving = json.loads((APP / "runtime/serving-check.json").read_text())
    if serving.get("status") != "passed" or serving.get("image") != prepared["image"]:
        raise ValueError("Serving dependency overlay must pass CPU checks for the selected image")
    overlay = serving["overlay"]
    if not app_path(overlay["path"]).is_relative_to(APP / "overlays") or sha256(overlay["path"]) != overlay["sha256"]:
        raise ValueError("Serving dependency overlay changed after CPU validation")
    prepared["serving_overlay"] = overlay
    return prepared


def model_command(prepared, run_dir, job_tmp, port, concurrency):
    model = app_path(prepared["model"]["path"])
    cache = "/tmp/atp"
    container_env = {
        "PYTHONPATH": "/opt/atp-serving-openai.zip",
        "TMPDIR": cache + "/tmp", "XDG_CACHE_HOME": cache + "/cache",
        "HF_HOME": cache + "/hf", "HF_MODULES_CACHE": cache + "/hf/modules",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "VLLM_CACHE_ROOT": cache + "/vllm", "TORCHINDUCTOR_CACHE_DIR": cache + "/inductor",
        "TORCH_EXTENSIONS_DIR": cache + "/extensions", "TRITON_CACHE_DIR": cache + "/triton",
        "CUDA_CACHE_PATH": cache + "/cuda", "FLASHINFER_WORKSPACE_BASE": cache + "/flashinfer",
        "FLASHINFER_CACHE_DIR": cache + "/flashinfer-cache", "TVM_FFI_CACHE_DIR": cache + "/tvm",
        "NUMBA_CACHE_DIR": cache + "/numba", "MPLCONFIGDIR": cache + "/matplotlib",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OMP_NUM_THREADS": "8",
        "TOKENIZERS_PARALLELISM": "false",
    }
    return [
        "srun", "--overlap", "--exact", "--nodes=1", "--ntasks=1", "--gpus-per-task=1",
        "--cpus-per-task=8", "--kill-on-bad-exit=1", "--job-name=atp-vllm",
        "--container-image=" + prepared["image"]["path"], "--container-writable",
        "--no-container-entrypoint", "--no-container-mount-home", "--container-workdir=/tmp/atp",
        "--container-mounts=" + str(model) + ":/model:ro+rprivate," + str(job_tmp) + ":/tmp/atp:rw+rprivate,"
        + str(app_path(prepared["serving_overlay"]["path"])) + ":/opt/atp-serving-openai.zip:ro+rprivate",
        "env", *[key + "=" + value for key, value in container_env.items()],
        MODEL_PYTHON, "-m", "vllm.entrypoints.openai.api_server", "--model", "/model",
        "--served-model-name", "lightning-3.5", "--host", "0.0.0.0", "--port", str(port),
        "--tensor-parallel-size", "1", "--dtype", "bfloat16", "--trust-remote-code",
        "--max-model-len", str(CONTEXT), "--max-num-seqs", str(min(concurrency, 8)),
        "--gpu-memory-utilization", "0.9", "--mamba-backend", "flashinfer", "--mamba-ssm-cache-dtype", "float16",
        "--enable-mamba-cache-stochastic-rounding", "--mamba-cache-philox-rounds", "5",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "nemotron_v3", "--generation-config", "vllm",
        "--default-chat-template-kwargs", '{"enable_thinking":true,"force_nonempty_content":true}',
    ]


def gym_command(config, run_dir, args):
    sampling = "{temperature:1.0,top_p:0.95,max_tokens:" + str(args.max_output_tokens) + "}"
    command = [
        str(APP / "runtime/controller/bin/gym"), "eval", "run", "--config", str(config),
        "--split", "benchmark", "--limit", str(args.limit), "--concurrency", str(args.concurrency),
        "--num-repeats", "1", "--temperature", "1.0", "--top-p", "0.95",
        "--max-output-tokens", str(args.max_output_tokens), "--output", str(run_dir / "rollouts.jsonl"),
        "++policy_model.responses_api_models.vllm_model.sampling_overrides=" + sampling,
        "++reuse_existing_data_preparation=true",
        "++swebench_verified_opencode_resources_server.resources_servers.swebench.clear_swebench_debug_logs=false",
        "++route_failures_to_sidecar=true",
    ]
    if args.resume:
        command.append("--resume")
    if getattr(args, "smoke", False):
        # This native benchmark inherits its agent; apply CLI overrides to the base instance.
        agent = "opencode_sandboxed_agent.responses_api_agents.opencode_sandboxed_agent"
        resource = "swebench_verified_opencode_resources_server.resources_servers.swebench"
        command.extend([
            f"++{agent}.sandbox_timeout=600", f"++{resource}.evaluation_timeout=300",
            f"++{agent}.sandbox_config.ready_timeout_s=120", f"++{resource}.sandbox_config.ready_timeout_s=120",
            "++sandbox.opensandbox.create.request_timeout_s=180", "++sandbox.opensandbox.create.timeout_s=180",
            "++sandbox.opensandbox.create.connect_attempt_timeout_s=180", "++sandbox.opensandbox.create.retries=0",
        ])
    return command


def validate_rollouts(dataset, output, limit):
    expected = [json.loads(line)["instance_id"] for line in Path(dataset).read_text().splitlines() if line.strip()][:limit]
    rows, invalid_rows = [], 0
    for line in output.read_text().splitlines() if output.exists() else []:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("instance_id"), str):
                raise ValueError("Malformed rollout")
            rows.append(row)
        except ValueError:
            invalid_rows += 1
    counts = Counter(row["instance_id"] for row in rows)
    invalid_ids = {row["instance_id"] for row in rows if row.get("evaluation_completed") is not True
                   or type(row.get("resolved")) is not bool or row.get("_ng_rollout_index") != 0
                   or row.get("_ng_failure_class") is not None}
    failed = sorted(instance for instance in expected if counts[instance] != 1 or instance in invalid_ids)
    unexpected = sorted(set(counts) - set(expected))
    complete = not (failed or unexpected or invalid_rows) and len(expected) == limit == len(rows) == len(set(expected))
    resolved = sum(row.get("resolved") is True for row in rows if row["instance_id"] in expected)
    unfinished_agents = sorted(row["instance_id"] for row in rows
                               if row.get("opencode_finished") is not True or row.get("opencode_export_found") is not True)
    generated_tokens = sum(((row.get("response") or {}).get("usage") or {}).get("output_tokens", 0) or 0 for row in rows)
    tool_calls = sum(item.get("type") == "function_call" for row in rows
                     for item in (row.get("response") or {}).get("output", []))
    return {"complete": complete, "expected_tasks": limit, "rollout_rows": len(rows),
            "invalid_rows": invalid_rows, "failed_instance_ids": failed, "unexpected_instance_ids": unexpected,
            "resolved_count": resolved, "resolved_mean": resolved / limit if complete else None,
            "unfinished_agent_instance_ids": unfinished_agents, "generated_tokens": generated_tokens,
            "tool_calls": tool_calls,
            "smoke_passed": complete and not unfinished_agents and generated_tokens > 0 and tool_calls > 0,
            "unresolved_instance_ids": sorted(row["instance_id"] for row in rows
                                              if row.get("resolved") is False and row["instance_id"] not in invalid_ids)}


def wait_model(process, url):
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("vLLM exited before readiness; inspect vllm.log")
        try:
            with client.open(url + "/models", timeout=5) as response:
                models = json.load(response)
            if any(model.get("id") == "lightning-3.5" for model in models.get("data", [])):
                return
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(5)
    raise RuntimeError("vLLM readiness timed out; inspect vllm.log")


def stop_child(process):
    if process is None:
        return
    for sig, timeout in ((signal.SIGINT, 20), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
        # The group can still contain child servers after its leader exits.
        if sig == signal.SIGKILL:
            return
        time.sleep(1)


def cleanup_remote(env, log):
    code = """import asyncio, os, sys
sys.path.insert(0, os.environ['ATP_APP'] + '/scripts')
from check_cpu import cleanup_owned_sandboxes, normalized_label, RUN_KEY, USER_KEY
connection = dict(domain=os.environ['ATP_OS_HOST'], protocol=os.environ['ATP_OS_PROTOCOL'],
    api_key=os.environ['OPENSANDBOX_API_KEY'], tls_verify=os.environ['ATP_OS_TLS_VERIFY'] == 'true')
config = {'sandbox': {'opensandbox': {'connection': connection}}}
scope = {RUN_KEY: normalized_label(os.environ['NEMO_GYM_RUN_ID']),
         USER_KEY: normalized_label(os.environ['NEMO_GYM_USER'])}
asyncio.run(asyncio.wait_for(cleanup_owned_sandboxes(config, scope), timeout=60))
"""
    try:
        result = subprocess.run([str(APP / "runtime/controller/bin/python"), "-c", code],
                                cwd=APP / "vendor/Gym", env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=70, start_new_session=True)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", default="1", help="Number of tasks, or all (500)")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Bound one task and require successful agent execution and tool calls")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without creating files or starting processes")
    args = parser.parse_args()
    args.limit = 500 if args.limit == "all" else int(args.limit)
    if not 1 <= args.limit <= 500 or args.concurrency < 1 or not 1 <= args.max_output_tokens < CONTEXT:
        parser.error("limit must be 1..500; concurrency must be positive; token budget must be 1..262143")
    if args.resume and args.run_dir is None:
        parser.error("--resume requires the original --run-dir")
    if args.smoke and args.limit != 1:
        parser.error("--smoke requires --limit 1")
    run_dir = app_path(args.run_dir or APP / "runs" / ("eval-" + os.environ.get("SLURM_JOB_ID", "preview")))
    if not run_dir.is_relative_to(APP / "runs") or run_dir == APP / "runs":
        parser.error("--run-dir must be a subdirectory of APP/runs")
    readiness_error = None
    try:
        prepared = read_ready()
    except (OSError, ValueError, KeyError) as error:
        if not args.dry_run:
            raise
        readiness_error = str(error)
        prepared = {"model": {"path": str(APP / "models/lightning-3.5")}, "image": {
            "path": str(APP.parent / "images/nemo-rl.nightly-09032026.x86_64.sqsh")},
            "serving_overlay": {"path": str(APP / "overlays/serving-openai-2.44.0.zip")}}
    if args.dry_run:
        print(json.dumps({"ready": readiness_error is None, "readiness_error": readiness_error,
                          "run_dir": str(run_dir), "gpu_jobs_submitted": 0}, indent=2))
        print(shlex.join(model_command(prepared, run_dir, Path("/tmp/atp-JOB"), 8000, args.concurrency)))
        print(shlex.join(gym_command(run_dir / "attempt-PREVIEW/gym.yaml", run_dir, args)))
        return 0
    if not os.environ.get("SLURM_JOB_ID") or not any(os.environ.get(key) for key in ("SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_GPUS")):
        parser.error("Run only inside an explicitly submitted GPU allocation using eval.sbatch")
    import atp_runtime
    from gpu_monitor import GpuMonitor

    settings = {"preparation_id": prepared["preparation_id"], "benchmark": "swebench/verified/opencode",
                "model": prepared["model"], "image": prepared["image"], "dataset": prepared["dataset"],
                "serving_overlay": prepared["serving_overlay"],
                "gym_commit": prepared["gym_commit"], "runner_sha256": sha256(__file__),
                "gpu_monitor_sha256": sha256(APP / "scripts/gpu_monitor.py"), "smoke": args.smoke,
                "runtime_helper_sha256": sha256(APP / "scripts/atp_runtime.py"),
                "limit": args.limit, "concurrency": args.concurrency, "num_repeats": 1,
                "max_output_tokens": args.max_output_tokens, "temperature": 1.0, "top_p": 0.95,
                "context": CONTEXT, "tensor_parallel_size": 1, "opencode_version": "1.17.11"}
    if args.resume:
        if json.loads((run_dir / "settings.json").read_text()) != settings:
            raise ValueError("Resume settings or prepared inputs changed; choose a new run directory")
    elif run_dir.exists():
        raise ValueError("Run directory already exists; use --resume or a new directory")
    else:
        run_dir.mkdir(parents=True)
    lock = (run_dir / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.resume:
        save_json(run_dir / "settings.json", settings)
    scratch = Path("/raid") if Path("/raid").is_dir() and os.access("/raid", os.W_OK) else Path("/tmp")
    job_tmp = Path(tempfile.mkdtemp(prefix="atp-" + os.environ["SLURM_JOB_ID"] + "-", dir=scratch))
    try:
        (job_tmp / "tmp").mkdir()
        os.environ["ATP_JOB_TMP"] = str(job_tmp)
        env = atp_runtime.runtime_env()
        env["UV_OFFLINE"] = "1"
        origin = {"host": env["ATP_OS_HOST"], "protocol": env["ATP_OS_PROTOCOL"],
                  "tls_verify": env["ATP_OS_TLS_VERIFY"] == "true"}
        if json.loads((APP / "runtime/cpu-check.json").read_text()).get("opensandbox_origin") != origin:
            raise ValueError("OpenSandbox endpoint changed after CPU validation")
        origin_file = run_dir / "opensandbox-origin.json"
        if args.resume and json.loads(origin_file.read_text()) != origin:
            raise ValueError("OpenSandbox endpoint changed; choose a new run directory")
        if not args.resume:
            save_json(origin_file, origin)
        model_env = dict(env)
        env["CUDA_VISIBLE_DEVICES"] = ""
        attempt = run_dir / ("eval-" + os.environ["SLURM_JOB_ID"] + "-" + uuid.uuid4().hex[:12])
        attempt.mkdir()
        env["NEMO_GYM_RUN_ID"] = attempt.name
        env.setdefault("NEMO_GYM_USER", os.environ.get("USER", "unknown"))
        env["RAY_ADDRESS"] = "local"
        env["RAY_TMPDIR"] = str(job_tmp / "ray")
        with socket.socket() as reserved:
            reserved.bind(("0.0.0.0", 0))
            port = reserved.getsockname()[1]
            model_url = "http://127.0.0.1:" + str(port) + "/v1"
            config = atp_runtime.build_config(attempt, model_url, mode="eval")
        for key in tuple(model_env):
            if any(word in key.upper() for word in ("TOKEN", "API_KEY", "SECRET", "PASSWORD")) or key.startswith("NEMO_GYM"):
                model_env.pop(key)
        model_args = model_command(prepared, run_dir, job_tmp, port, args.concurrency)
        gym_args = gym_command(config, run_dir, args)
        save_json(attempt / "launch.json", {"slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
                  "sandbox_run_id": env["NEMO_GYM_RUN_ID"], "model_command": model_args, "gym_command": gym_args})
    except BaseException:
        shutil.rmtree(job_tmp)
        lock.close()
        raise
    model_process = gym_process = None
    monitor = GpuMonitor(attempt, model_env, idle_seconds=600)
    status = {"status": "starting", "attempt": str(attempt), "sandbox_run_id": env["NEMO_GYM_RUN_ID"]}
    save_json(run_dir / "status.json", status)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    code = 1
    try:
        with (attempt / "vllm.log").open("w") as model_log, (attempt / "gym.log").open("w") as gym_log:
            monitor.start()
            startup_started = time.monotonic()
            model_process = subprocess.Popen(model_args, cwd=APP, env=model_env, stdout=model_log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
            wait_model(model_process, model_url)
            status.update(status="evaluating", model_startup_seconds=time.monotonic() - startup_started)
            save_json(run_dir / "status.json", status)
            gym_process = subprocess.Popen(gym_args, cwd=APP / "vendor/Gym", env=env, stdout=gym_log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            while gym_process.poll() is None:
                monitor.observe(model_url)
                if model_process.poll() is not None:
                    raise RuntimeError("vLLM stopped during evaluation; inspect vllm.log")
                time.sleep(2)
            code = gym_process.returncode
            status["status"] = "completed" if code == 0 else "failed"
    except KeyboardInterrupt:
        code = 130
        status["status"] = "interrupted"
    except Exception as error:
        status.update(status="failed", error=str(error))
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        stop_child(gym_process)
        stop_child(model_process)
        monitor.stop()
        with (attempt / "cleanup.log").open("w") as cleanup_log:
            status["sandbox_cleanup_passed"] = gym_process is None or cleanup_remote(env, cleanup_log)
        summary = validate_rollouts(prepared["dataset"]["path"], run_dir / "rollouts.jsonl", args.limit)
        save_json(run_dir / "summary.json", summary)
        status["evaluation_complete"] = summary["complete"]
        if code == 0 and not summary["complete"]:
            code = 1
            status["status"] = "incomplete"
        if code == 0 and args.smoke and not summary["smoke_passed"]:
            code = 1
            status["status"] = "smoke_failed"
        if not status["sandbox_cleanup_passed"] and code == 0:
            code = 1
            status["status"] = "completed_cleanup_failed"
        status["exit_code"] = code
        save_json(run_dir / "status.json", status)
        shutil.rmtree(job_tmp)
        lock.close()
    print("Evaluation " + status["status"] + ": " + str(run_dir))
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print("Evaluation refused or failed: " + str(error), file=sys.stderr)
        raise SystemExit(1)
