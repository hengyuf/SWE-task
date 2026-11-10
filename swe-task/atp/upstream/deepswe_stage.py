#!/usr/bin/env python3
"""Prepare the pinned DeepSWE HTTP evaluation entirely on job-local disk."""
from __future__ import annotations

import argparse
import getpass
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
from urllib.parse import urlsplit
from urllib.request import urlopen


GYM_COMMIT = "2f11dec3a1c11c0c5f302ab199f97754f7b5c8bd"
DATASET_REVISION = "435ee89ec2f2e2289f33b0da4f992f0b7b7266b9"
RG_VERSION = "15.1.0"
AGENT_NAME = "deepswe_opencode_sandboxed_agent"
RESOURCE_NAME = "deepswe_opencode_resources_server"
PATCHES = ("opensandbox-tls.patch", "deepswe-http.patch", "deepswe-capture.patch", "deepswe-collector.patch")
DOTENV_KEYS = {"OPENSANDBOX_DOMAIN", "OPENSANDBOX_API_KEY", "OPENSANDBOX_TLS_VERIFY",
               "HF_TOKEN", "NVIDIA_API_KEY", "WANDB_API_KEY"}


def read_dotenv(path: Path) -> dict:
    """Read assignments literally: no shell expansion, substitution, or execution."""
    values = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key not in DOTENV_KEYS:
            raise ValueError(f"Unsupported .env assignment on line {number}")
        words = shlex.split(value, comments=True)
        if len(words) > 1:
            raise ValueError(f"Quote the .env value on line {number}")
        values[key] = words[0] if words else ""
    return values


def local_path(path: Path) -> Path:
    path = path.resolve()
    if not any(path.is_relative_to(root) and path != root for root in (Path("/tmp"), Path("/raid/scratch"))):
        raise ValueError("DeepSWE scratch must be a private directory under /raid/scratch or /tmp")
    return path


def runtime_env(app: Path, scratch: Path) -> dict:
    scratch = local_path(scratch)
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("NEMO_GYM_", "UV_", "RAY_", "ENROOT_")) or key in {
            "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "NEMO_GYM_VLLM_TRANSPORT_LOG",
        }:
            env.pop(key, None)
    env.update(read_dotenv(app / ".env"))
    from deepswe_credentials import load_extra_credentials
    load_extra_credentials(app, env)
    for key in ("NVIDIA_API_KEY", "OPENSANDBOX_API_KEY", "OPENSANDBOX_DOMAIN"):
        if not env.get(key):
            raise ValueError(f"Missing {key} in the application .env or process environment")
    endpoint = env["OPENSANDBOX_DOMAIN"]
    endpoint = urlsplit(endpoint if "://" in endpoint else "https://" + endpoint)
    if (endpoint.scheme not in {"http", "https"} or not endpoint.hostname or endpoint.username
            or endpoint.password or endpoint.path not in {"", "/"} or endpoint.query or endpoint.fragment):
        raise ValueError("OPENSANDBOX_DOMAIN must be an HTTP(S) origin or host")
    tls_verify = env.get("OPENSANDBOX_TLS_VERIFY", "true").lower()
    if tls_verify not in {"true", "false"}:
        raise ValueError("OPENSANDBOX_TLS_VERIFY must be true or false")
    env.update(ATP_APP=str(app), ATP_DEEPSWE_SCRATCH=str(scratch), ATP_OS_HOST=endpoint.netloc,
               ATP_OS_PROTOCOL=endpoint.scheme, ATP_OS_TLS_VERIFY=tls_verify,
               NEMO_GYM_USER=getpass.getuser(), NEMO_GYM_TEAM=env.get("SLURM_JOB_ACCOUNT", "nemotron_agents_dev"),
               NEMO_GYM_WORKLOAD="deepswe-http", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="disabled", UV_LINK_MODE="hardlink")
    directories = {
        "UV_CACHE_DIR": "cache/uv", "UV_PYTHON_INSTALL_DIR": "python", "UV_PYTHON_BIN_DIR": "bin",
        "UV_PROJECT_ENVIRONMENT": "controller", "XDG_CACHE_HOME": "cache/xdg", "XDG_CONFIG_HOME": "config",
        "XDG_DATA_HOME": "data", "XDG_STATE_HOME": "state", "TMPDIR": "tmp", "RAY_TMPDIR": "tmp/ray",
        "HF_HOME": "cache/huggingface", "HF_HUB_CACHE": "cache/huggingface/hub",
        "HF_DATASETS_CACHE": "cache/huggingface/datasets", "HF_MODULES_CACHE": "cache/huggingface/modules",
        "PIP_CACHE_DIR": "cache/pip", "TORCH_HOME": "cache/torch", "TRITON_CACHE_DIR": "cache/triton",
        "TORCH_EXTENSIONS_DIR": "cache/torch-extensions", "TORCHINDUCTOR_CACHE_DIR": "cache/torchinductor",
        "VLLM_CACHE_ROOT": "cache/vllm", "CUDA_CACHE_PATH": "cache/cuda", "MPLCONFIGDIR": "cache/matplotlib",
        "WANDB_DIR": "cache/wandb",
    }
    for key, relative in directories.items():
        directory = scratch / relative
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)
    # Recovery and submission still need site Slurm clients after Python/venv
    # isolation. Retain only their discovered directories from the incoming PATH.
    scheduler_dirs = []
    for command in ("sacct", "scontrol", "sbatch", "srun"):
        executable = shutil.which(command, path=env.get("PATH", os.defpath))
        if executable:
            scheduler_dirs.append(str(Path(executable).absolute().parent))
    path_dirs = [str(scratch / "controller/bin"), str(scratch / "bin"), *scheduler_dirs,
                 "/usr/local/bin", *os.defpath.split(os.pathsep)]
    env["PATH"] = os.pathsep.join(dict.fromkeys(path_dirs))
    env["PYTHONPATH"] = os.pathsep.join([str(scratch / "scripts"), str(scratch / "Gym")])
    constraints = scratch / "constraints.txt"
    if constraints.exists():
        env["UV_CONSTRAINT"] = str(constraints)
    return env


def run(command, cwd: Path, env: dict) -> None:
    secrets = [value for key, value in env.items()
               if key.endswith(("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")) and value]
    with subprocess.Popen(list(map(str, command)), cwd=cwd, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        for line in process.stdout:
            for secret in secrets:
                line = line.replace(secret, "[REDACTED]")
            print(line, end="", flush=True)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)


def _descriptor(app: Path, scratch: Path, settings: dict, env: dict) -> dict:
    env["ATP_DEEPSWE_POLICY_BASE_URL"] = settings["base_url"]
    gym = scratch / "Gym"
    return {"app": app, "scratch": scratch, "gym": gym, "python": scratch / "controller/bin/python",
            "gym_cli": scratch / "controller/bin/gym", "dataset": gym / "benchmarks/deepswe/data/deepswe_benchmark.jsonl",
            "tasks_dir": gym / "resources_servers/deepswe/data/cache/tasks", "opencode_binary": scratch / "bin/opencode",
            "rg_binary": scratch / "bin/rg", "source": Path(env.get("ATP_DEEPSWE_SOURCE", app)),
            "settings": settings, "env": env}


def _stage_ripgrep(scratch: Path, env: dict) -> Path:
    """Provide the static ripgrep binary before entering a network-isolated task."""
    binary = scratch / "bin/rg"
    binary.parent.mkdir(parents=True, exist_ok=True)
    candidate = binary
    if not binary.is_file():
        asset = (f"https://github.com/BurntSushi/ripgrep/releases/download/{RG_VERSION}/"
                 f"ripgrep-{RG_VERSION}-x86_64-unknown-linux-musl.tar.gz")
        download = scratch / "ripgrep.tar.gz"
        with urlopen(asset, timeout=120) as response, download.open("wb") as output:
            shutil.copyfileobj(response, output)
        candidate = binary.with_name("rg.tmp")
        with tarfile.open(download) as archive:
            members = [member for member in archive.getmembers() if Path(member.name).name == "rg" and member.isfile()]
            if len(members) != 1:
                raise ValueError("Official ripgrep release must contain exactly one rg executable")
            with archive.extractfile(members[0]) as source, candidate.open("wb") as output:
                shutil.copyfileobj(source, output)
        candidate.chmod(0o700)
    version = subprocess.check_output([str(candidate), "--version"], cwd=scratch, env=env, text=True)
    if version.partition("\n")[0].split()[:2] != ["ripgrep", RG_VERSION]:
        raise ValueError("Downloaded ripgrep executable has the wrong version")
    if candidate != binary:
        candidate.replace(binary)
    return binary


def validate_dataset(dataset: Path, tasks_dir: Path, expected: int = 113) -> list:
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    if len(rows) != expected or len({row["task_id"] for row in rows}) != expected:
        raise ValueError(f"Expected {expected} distinct pinned DeepSWE tasks")
    marker = json.loads((tasks_dir / ".nemo_gym_deepswe.json").read_text())
    if marker["source_revision"] != DATASET_REVISION:
        raise ValueError("DeepSWE verifier assets do not match the pinned source revision")
    if any(not (tasks_dir / row["task_id"] / "task.toml").is_file() for row in rows):
        raise ValueError("DeepSWE dataset is missing verifier task assets")
    return rows


def annotate_verifier_timeouts(dataset: Path, tasks_dir: Path, rows: list) -> None:
    import tomllib

    for row in rows:
        config = tomllib.loads((tasks_dir / row["task_id"] / "task.toml").read_text())
        # Match the pinned Gym task_schema.VerifierConfig default.
        timeout = config.get("verifier", {}).get("timeout_sec", 600.0)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f"Invalid verifier timeout for {row['task_id']}")
        row["_atp_verifier_timeout"] = timeout
    temporary = dataset.with_suffix(".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    temporary.replace(dataset)


def stage(app: Path, scratch: Path) -> dict:
    app, scratch = app.resolve(), local_path(scratch)
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_JOB_PARTITION", "").startswith("cpu"):
        raise RuntimeError("Run DeepSWE preparation inside a CPU Slurm allocation")
    os.umask(0o077)
    source = Path(os.environ.get("ATP_DEEPSWE_SOURCE", app)).resolve()
    if source != app:
        from deepswe_source import verify_source
        verify_source(source)
    settings = json.loads((source / "configs/deepswe.json").read_text())
    if settings["gym_commit"] != GYM_COMMIT or settings["dataset_revision"] != DATASET_REVISION:
        raise ValueError("DeepSWE settings do not match the supported source pins")
    env = runtime_env(app, scratch)
    env["ATP_DEEPSWE_SOURCE"] = str(source)
    env["ATP_DEEPSWE_OPENCODE_BINARY"] = str(scratch / "bin/opencode")
    env["ATP_DEEPSWE_RG_BINARY"] = str(scratch / "bin/rg")
    env["ATP_DEEPSWE_AGENT_TIMEOUT"] = str(settings["agent_timeout_seconds"])
    env["ATP_DEEPSWE_MAX_OUTPUT_TOKENS"] = str(settings["max_output_tokens"])
    staged = _descriptor(app, scratch, settings, env)
    marker = scratch / "staged.json"
    if marker.exists():
        ready = json.loads(marker.read_text())
        if (ready["app"] != str(app) or ready["settings"] != settings
                or ready.get("source", ready["app"]) != str(source)):
            raise ValueError("Scratch belongs to different application inputs; use a new scratch directory")
        for key in ("python", "gym_cli", "dataset", "opencode_binary"):
            if not staged[key].is_file():
                raise RuntimeError(f"Prepared scratch is incomplete: {key}")
        _stage_ripgrep(scratch, env)
        if ready.get("rg_binary") != str(staged["rg_binary"]):
            ready["rg_binary"] = str(staged["rg_binary"])
            temporary = marker.with_suffix(".tmp")
            temporary.write_text(json.dumps(ready, indent=2) + "\n")
            temporary.replace(marker)
        return staged
    shutil.copytree(source / "scripts", scratch / "scripts", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    uv = scratch / "bin/uv"
    if not uv.exists():
        shutil.copy2(app / "runtime/bin/uv", uv)
    run([uv, "python", "install", settings["python_version"]], scratch, env)
    python = subprocess.check_output([str(uv), "python", "find", "--managed-python", settings["python_version"]],
                                     cwd=scratch, env=env, text=True).strip()
    if not Path(python).resolve().is_relative_to(scratch):
        raise RuntimeError("uv selected a Python interpreter outside job-local scratch")
    if sys.version_info < (3, 13):
        run([python, scratch / "scripts/deepswe_stage.py", "--app", app, "--scratch", scratch], scratch, env)
        return stage(app, scratch)
    gym = staged["gym"]
    if gym.exists():
        raise RuntimeError("Previous staging failed; use a fresh job-local scratch directory")
    gym.mkdir()
    with tarfile.open(app / "vendor/Gym.tar") as archive:
        if archive.pax_headers.get("comment") != GYM_COMMIT:
            raise ValueError("Gym archive does not identify the pinned commit")
        members = [member for member in archive.getmembers()
                   if not member.name.startswith((".claude/", ".codex/")) and member.name != "CLAUDE.md"]
        archive.extractall(gym, members=members, filter="data")
    for patch in PATCHES:
        run(["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", source / "overlays" / patch], gym, env)
    run([uv, "sync", "--frozen", "--python", python, "--extra", "sandbox"], gym, env)
    constraints = scratch / "constraints.txt"
    run([uv, "export", "--quiet", "--frozen", "--extra", "sandbox", "--extra", "dev", "--no-emit-project",
         "--no-hashes", "--output-file", constraints], gym, env)
    env["UV_CONSTRAINT"] = str(constraints)
    setup = scratch / "setup"
    config = build_config(staged, setup, setup, next(iter(settings["models"])), staged["dataset"])
    run([staged["gym_cli"], "env", "prefetch", f"+config_paths=[{config}]", "++skip_venv_if_present=false"], gym, env)
    run([staged["gym_cli"], "eval", "prepare", "--config", config], gym, env)
    rows = validate_dataset(staged["dataset"], staged["tasks_dir"], settings["expected_tasks"])
    annotate_verifier_timeouts(staged["dataset"], staged["tasks_dir"], rows)
    # The baseline build also runs on x86_64 sandboxes without AVX2.
    asset = f"https://github.com/anomalyco/opencode/releases/download/v{settings['opencode_version']}/opencode-linux-x64-baseline.tar.gz"
    download = scratch / "opencode.tar.gz"
    with urlopen(asset, timeout=120) as response, download.open("wb") as output:
        shutil.copyfileobj(response, output)
    with tarfile.open(download) as archive:
        members = [member for member in archive.getmembers() if Path(member.name).name == "opencode" and member.isfile()]
        if len(members) != 1:
            raise ValueError("Official OpenCode release must contain exactly one opencode executable")
        with archive.extractfile(members[0]) as source, staged["opencode_binary"].open("wb") as output:
            shutil.copyfileobj(source, output)
    staged["opencode_binary"].chmod(0o700)
    version = subprocess.check_output([str(staged["opencode_binary"]), "--version"], cwd=scratch, env=env, text=True).strip()
    if version != settings["opencode_version"]:
        raise ValueError("Downloaded OpenCode executable has the wrong version")
    _stage_ripgrep(scratch, env)
    public = {key: str(value) if isinstance(value, Path) else value for key, value in staged.items() if key != "env"}
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(public, indent=2) + "\n")
    temporary.replace(marker)
    return staged


def build_config(staged: dict, campaign: Path, launch: Path, model_alias: str,
                 dataset_path: Path, *, concurrency: int = 1) -> Path:
    if concurrency < 1:
        raise ValueError("Model shard concurrency must be positive")
    settings, gym, scratch, env = (staged[key] for key in ("settings", "gym", "scratch", "env"))
    host = socket.gethostbyname(socket.gethostname())
    sockets = []
    try:
        for _ in range(4):
            sock = socket.socket()
            sock.bind((host, 0))
            sockets.append(sock)
        ports = [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()
    launch = launch.resolve()
    launch.mkdir(parents=True, exist_ok=True)
    native_logs = scratch / "logs" / campaign.name / launch.relative_to(campaign.resolve())
    config = {
        "config_paths": [str(gym / relative) for relative in (
            "responses_api_models/vllm_model/configs/vllm_model.yaml",
            "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml", "benchmarks/deepswe/opencode.yaml")],
        "head_server": {"host": host, "port": ports[0]}, "use_absolute_ip": True,
        "python_version": settings["python_version"], "cache_dir": str(scratch / "cache/gym"),
        "results_dir": str(launch / "results"), "uv_cache_dir": env["UV_CACHE_DIR"],
        "uv_venv_dir": str(scratch / "servers"), "nemo_gym_log_dir": str(native_logs),
        "skip_venv_if_present": True, "uv_pip_set_python": True,
        "observability_enabled": True, "model_call_capture_dir": str(launch / "captures"),
        "global_aiohttp_client_request_debug": False,
        "policy_base_url": settings["base_url"], "policy_api_key": "${oc.env:NVIDIA_API_KEY}",
        "policy_model_name": settings["models"][model_alias]["id"],
        "num_samples_in_parallel": concurrency,
        "policy_model": {"responses_api_models": {"vllm_model": {
            "host": host, "port": ports[1], "num_workers": 1, "uses_reasoning_parser": True,
            "uses_interleaved_reasoning": True, "chat_template_kwargs": None,
            "sampling_overrides": {"temperature": settings["temperature"], "top_p": settings["top_p"],
                                   "max_tokens": settings["max_output_tokens"],
                                   **settings["models"][model_alias].get("sampling_overrides", {})},
        }}},
        RESOURCE_NAME: {"resources_servers": {"deepswe": {
            "host": host, "port": ports[2], "num_workers": 1, "tasks_dir": str(staged["tasks_dir"]),
            "expected_task_count": settings["expected_tasks"], "clear_verifier_logs": False,
            "logs_dir": str(launch / "verifier-logs"), "include_model_patch_in_response": True,
        }}},
        AGENT_NAME: {"responses_api_agents": {"opencode_sandboxed_agent": {
            "host": host, "port": ports[3], "num_workers": 4, "token_id_capture": False,
            "opencode_version": settings["opencode_version"], "opencode_max_context_window": settings["context_window"],
            "sandbox_timeout": settings["agent_timeout_seconds"],
            "datasets": [{"name": settings["benchmark"], "type": "benchmark", "jsonl_fpath": str(dataset_path),
                          "prepare_script": "benchmarks/deepswe/prepare.py", "num_repeats": 1}],
        }}},
        "sandbox": {"default_metadata": {"nemo-gym.nvidia.com/run": launch.name,
                                           "nemo-gym.nvidia.com/user": env["NEMO_GYM_USER"]},
                    "opensandbox": {"connection": {"domain": "${oc.env:ATP_OS_HOST}",
                        "protocol": "${oc.env:ATP_OS_PROTOCOL}", "api_key": "${oc.env:OPENSANDBOX_API_KEY}",
                        "tls_verify": env["ATP_OS_TLS_VERIFY"] == "true"}}},
    }
    path = launch / "gym.yaml"
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    arguments = parser.parse_args()
    result = stage(arguments.app, arguments.scratch)
    print(f"DeepSWE runtime ready: {result['scratch'] / 'staged.json'}")
