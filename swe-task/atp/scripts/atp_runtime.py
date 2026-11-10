"""Paths, environment and native Gym configuration owned by this application."""
import datetime
import getpass
import json
import os
from pathlib import Path
import shlex
import socket
import uuid
from urllib.parse import urlsplit

GYM_COMMIT = "2f11dec3a1c11c0c5f302ab199f97754f7b5c8bd"
MODEL_REPO = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
MODEL_REVISION = "a9904d24bcc1d289a1950fa9d2b978c47cf903b9"


def app_root():
    return Path(__file__).resolve().parents[1]


def contained(path):
    path = Path(path).resolve()
    path.relative_to(app_root())
    return path


def runtime_env():
    app = app_root()
    env = dict(os.environ)
    foreign_roots = [app.parent / "pipeline", app.parent / "reasoning-rl-pipelines"]
    foreign_roots += [Path(env[key]).resolve() for key in ("VIRTUAL_ENV", "CONDA_PREFIX") if env.get(key)]
    path_entries = [entry for entry in env.get("PATH", "").split(os.pathsep) if entry
                    and not any(Path(entry).resolve().is_relative_to(root) for root in foreign_roots)]
    for key in list(env):
        if key.startswith(("NEMO_GYM_", "RAY_", "UV_", "ENROOT_")) or key in {
            "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "NRL_FORCE_REBUILD_VENVS",
        }:
            env.pop(key, None)
    # Parse dotenv as data, never execute shell substitutions or commands.
    for line in (app / ".env").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, value = line.partition("=")
        if not sep or not key.replace("_", "a").isalnum():
            raise ValueError("Invalid .env assignment")
        if key not in {"OPENSANDBOX_DOMAIN", "OPENSANDBOX_API_KEY", "OPENSANDBOX_TLS_VERIFY", "HF_TOKEN", "NVIDIA_API_KEY", "WANDB_API_KEY"}:
            raise ValueError(f"Unsupported .env key: {key}")
        words = shlex.split(value, comments=True)
        if len(words) > 1:
            raise ValueError(f"Quote the .env value for {key}")
        env[key] = words[0] if words else ""
    endpoint = env.get("OPENSANDBOX_DOMAIN", "")
    parsed = urlsplit(endpoint if "://" in endpoint else "https://" + endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.path not in {"", "/"} or parsed.query or parsed.username:
        raise ValueError("OPENSANDBOX_DOMAIN must be a host or HTTP(S) origin")
    tls_verify = env.get("OPENSANDBOX_TLS_VERIFY", "true").lower()
    if tls_verify not in {"true", "false"}:
        raise ValueError("OPENSANDBOX_TLS_VERIFY must be true or false")
    env["ATP_OS_TLS_VERIFY"] = tls_verify
    env.update(ATP_APP=str(app), ATP_OS_HOST=parsed.netloc, ATP_OS_PROTOCOL=parsed.scheme,
               NEMO_GYM_USER=getpass.getuser(), NEMO_GYM_TEAM="nemotron_agents_dev",
               NEMO_GYM_WORKLOAD="agent-teacher-swe", PYTHONNOUSERSITE="1",
               PYTHONDONTWRITEBYTECODE="1", RAY_ADDRESS="local", HF_HUB_DISABLE_TELEMETRY="1",
               WANDB_MODE="disabled", UV_LINK_MODE="hardlink")
    paths = {
        "HF_HOME": "cache/huggingface", "HF_HUB_CACHE": "cache/huggingface/hub",
        "HF_DATASETS_CACHE": "cache/huggingface/datasets", "UV_CACHE_DIR": "cache/uv",
        "UV_PYTHON_INSTALL_DIR": "runtime/python", "UV_PYTHON_BIN_DIR": "runtime/bin",
        "UV_PROJECT_ENVIRONMENT": "runtime/controller", "XDG_CACHE_HOME": "cache/xdg",
        "XDG_CONFIG_HOME": "runtime/config", "XDG_DATA_HOME": "runtime/data",
        "XDG_STATE_HOME": "runtime/state", "PIP_CACHE_DIR": "cache/pip",
        "TRITON_CACHE_DIR": "cache/triton", "TORCH_HOME": "cache/torch",
        "VLLM_CACHE_ROOT": "cache/vllm", "CUDA_CACHE_PATH": "cache/cuda",
        "HF_MODULES_CACHE": "cache/huggingface/modules", "TORCH_EXTENSIONS_DIR": "cache/torch-extensions",
        "TORCHINDUCTOR_CACHE_DIR": "cache/torchinductor", "FLASHINFER_CACHE_DIR": "cache/flashinfer",
        "FLASHINFER_WORKSPACE_BASE": "cache/flashinfer-workspace",
        "MPLCONFIGDIR": "cache/matplotlib", "WANDB_DIR": "runs/wandb",
        "ENROOT_CACHE_PATH": "cache/enroot", "ENROOT_CONFIG_PATH": "runtime/enroot-config",
    }
    for key, relative in paths.items():
        path = contained(app / relative)
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    tmp = Path(env.get("ATP_JOB_TMP", str(app / "runtime/tmp"))).resolve()
    if not tmp.is_relative_to(app) and not (env.get("SLURM_JOB_ID") and tmp.parent in {Path("/tmp"), Path("/raid")} and tmp.name.startswith("atp-" + env["SLURM_JOB_ID"] + "-")):
        raise ValueError("Temporary directory must belong to this application/job")
    tmp.mkdir(parents=True, exist_ok=True)
    for key, child in {"TMPDIR": "", "RAY_TMPDIR": "ray", "ENROOT_RUNTIME_PATH": "enroot-runtime", "ENROOT_DATA_PATH": "enroot-data", "ENROOT_TEMP_PATH": "enroot-tmp"}.items():
        path = tmp / child
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env["PATH"] = os.pathsep.join([str(app / "runtime/controller/bin"), str(app / "runtime/bin"), *path_entries])
    return env


def make_run_dir(kind):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = contained(app_root() / "runs" / f"{kind}-{stamp}-{uuid.uuid4().hex[:8]}")
    run.mkdir(parents=True)
    return run


def build_config(run_dir, model_url, mode="eval"):
    if mode not in {"eval", "check"}:
        raise ValueError(mode)
    app, run = app_root(), contained(run_dir)
    gym = app / "vendor/Gym"
    env = runtime_env()
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
    configs = ["responses_api_models/vllm_model/configs/vllm_model.yaml", "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml"]
    configs += ["benchmarks/swebench/verified/opencode.yaml" if mode == "eval" else "resources_servers/swebench/configs/swebench.yaml"]
    resource = "swebench_verified_opencode_resources_server" if mode == "eval" else "swebench_resources_server"
    config = {
        "config_paths": [str(gym / path) for path in configs],
        "head_server": {"host": host, "port": ports[0]}, "use_absolute_ip": True,
        "cache_dir": str(app / "cache/gym"), "results_dir": str(run / "results"),
        "uv_cache_dir": str(app / "cache/uv"), "uv_venv_dir": str(app / "runtime/servers"),
        "nemo_gym_log_dir": str(run / "gym-logs"), "skip_venv_if_present": True,
        "uv_pip_set_python": True,
        "policy_base_url": model_url, "policy_api_key": "unused-local", "policy_model_name": "lightning-3.5",
        "policy_model": {"responses_api_models": {"vllm_model": {
            "host": host, "port": ports[1], "uses_reasoning_parser": True,
            "uses_interleaved_reasoning": True,
            "chat_template_kwargs": {"enable_thinking": True, "force_nonempty_content": True},
            "sampling_overrides": {"temperature": 1.0, "top_p": 0.95},
        }}},
        resource: {"resources_servers": {"swebench": {
            "host": host, "port": ports[2], "is_verifying_golden_patch": mode == "check", "apply_anti_cheating": False,
        }}},
        "sandbox": {"default_metadata": {
            "nemo-gym.nvidia.com/run": run.name, "nemo-gym.nvidia.com/user": env["NEMO_GYM_USER"],
        }, "opensandbox": {"connection": {
            "domain": "${oc.env:ATP_OS_HOST}", "protocol": "${oc.env:ATP_OS_PROTOCOL}",
            "api_key": "${oc.env:OPENSANDBOX_API_KEY}",
            "tls_verify": env["ATP_OS_TLS_VERIFY"] == "true",
        }}},
    }
    if mode == "check":
        config["model_endpoint_readiness_timeout_seconds"] = 0
    else:
        config["swebench_verified_opencode_sandboxed_agent"] = {"responses_api_agents": {"opencode_sandboxed_agent": {
            "host": host, "port": ports[3], "opencode_version": "1.17.11", "opencode_max_context_window": 262144,
            "datasets": [{"name": "swebench_verified", "type": "benchmark",
                          "jsonl_fpath": "benchmarks/swebench/data/swebench_verified_benchmark.jsonl",
                          "prepare_script": "benchmarks/swebench/verified/prepare.py", "num_repeats": 1}],
        }}}
    path = run / "gym.yaml"
    path.write_text(json.dumps(config, indent=2) + "\n")  # JSON is valid YAML.
    return path
