"""Check the serving image and a pinned pure-Python OpenAI overlay on a CPU allocation."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import zipfile

from atp_runtime import app_root, contained, make_run_dir, runtime_env
from run_eval import MODEL_PYTHON, stop_child


OPENAI_VERSION = "2.44.0"
OVERLAY_MOUNT = "/opt/atp-serving-openai.zip"
PROBE = r'''
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

import openai
from openai.types.responses import NamespaceTool, FunctionTool, ToolChoiceFunction, Tool
import vllm.entrypoints.openai.api_server

assert openai.__version__ == "2.44.0", openai.__version__
assert openai.__file__.startswith("/opt/atp-serving-openai.zip/"), openai.__file__
assert metadata.version("openai") == "2.44.0"
result = {"openai": openai.__version__, "openai_file": openai.__file__,
          "namespace_tool": NamespaceTool.__name__, "api_server_import": True, "cli_help": False,
          "packages": {name: metadata.version(name) for name in
                       ("openai", "vllm", "httpx", "anyio", "jiter", "pydantic")}}
result_path = Path("/tmp/atp/serving-imports.json")
result_path.write_text(json.dumps(result, indent=2) + "\n")
with open("/tmp/atp/cli-help.txt", "w") as output:
    completed = subprocess.run([sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help"],
                               stdout=output, stderr=subprocess.STDOUT, check=False, timeout=180)
result["cli_help_returncode"] = completed.returncode
result["cli_help"] = completed.returncode == 0
if completed.returncode:
    import torch
    result["cuda_available"] = torch.cuda.is_available()
    help_output = Path("/tmp/atp/cli-help.txt").read_text()
    expected_error = (
        "RuntimeError: Failed to infer device type, please set the environment variable "
        "`VLLM_LOGGING_LEVEL=DEBUG` to turn on verbose logging to help debug the issue."
    )
    if (result["cuda_available"] is False
            and '/vllm/config/device.py", line ' in help_output
            and help_output.rstrip().endswith(expected_error)):
        result["cli_help_limitation"] = (
            "GPU-built vLLM requires a detected device; actual CLI must be validated by GPU smoke"
        )
    else:
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        raise RuntimeError("Unexpected serving CLI help failure; inspect cli-help.txt")
result_path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
'''


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def host_environment(environment):
    # Slurm/Pyxis and package staging need these settings; credentials never enter the step.
    names = {"PATH", "USER", "LOGNAME", "LANG", "LC_ALL", "SHELL", "LD_LIBRARY_PATH",
             "TMPDIR", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "XDG_CACHE_HOME"}
    prefixes = ("SLURM_", "SPANK_", "PMI_", "PMIX_", "ENROOT_", "UV_")
    return {key: value for key, value in environment.items() if key in names or key.startswith(prefixes)}


def container_environment():
    root = "/tmp/atp"
    paths = {
        "TMPDIR": "tmp", "XDG_CACHE_HOME": "cache", "XDG_CONFIG_HOME": "config",
        "XDG_DATA_HOME": "data", "XDG_STATE_HOME": "state", "HF_HOME": "hf",
        "HF_HUB_CACHE": "hf/hub", "HF_DATASETS_CACHE": "hf/datasets", "HF_MODULES_CACHE": "hf/modules",
        "VLLM_CACHE_ROOT": "vllm", "TORCH_HOME": "torch", "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TORCH_EXTENSIONS_DIR": "extensions", "TRITON_CACHE_DIR": "triton", "CUDA_CACHE_PATH": "cuda",
        "FLASHINFER_WORKSPACE_BASE": "flashinfer", "FLASHINFER_CACHE_DIR": "flashinfer-cache",
        "TVM_FFI_CACHE_DIR": "tvm", "NUMBA_CACHE_DIR": "numba", "MPLCONFIGDIR": "matplotlib",
        "RAY_TMPDIR": "ray", "UV_CACHE_DIR": "uv", "PIP_CACHE_DIR": "pip",
    }
    return {
        **{key: root + "/" + child for key, child in paths.items()},
        "PYTHONPATH": OVERLAY_MOUNT, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "OMP_NUM_THREADS": "8",
        "TOKENIZERS_PARALLELISM": "false",
    }


def stage_overlay(app, environment, log):
    requirements = app / "overlays/serving-requirements.txt"
    if requirements.read_text().strip() != "openai==" + OPENAI_VERSION:
        raise RuntimeError("Serving requirements do not match the validated SDK version")
    target = contained(app / "overlays/serving-python")
    target.mkdir(exist_ok=True)
    subprocess.run([
        str(app / "runtime/bin/uv"), "pip", "install", "--python", str(app / "runtime/controller/bin/python"),
        "--target", str(target), "--no-deps", "--requirement", str(requirements),
    ], env=environment, cwd=app, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    archive = contained(app / f"overlays/serving-openai-{OPENAI_VERSION}.zip")
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as output:
        for directory in (target / "openai", target / f"openai-{OPENAI_VERSION}.dist-info"):
            if not directory.is_dir():
                raise RuntimeError("Pinned OpenAI package or distribution metadata is missing")
            for path in sorted(directory.rglob("*")):
                if "__pycache__" in path.parts or path.suffix == ".pyc":
                    continue
                if path.is_symlink():
                    raise RuntimeError("Serving overlay must contain regular files, not symlinks")
                if path.is_file():
                    output.write(path, path.relative_to(target))
    temporary.replace(archive)
    return {"path": str(archive), "sha256": sha256(archive), "openai_version": OPENAI_VERSION}


def check(app, environment, job_tmp, run_dir, report):
    prepared = json.loads((app / "runtime/preparation.json").read_text())
    if prepared.get("status") != "ready":
        raise RuntimeError("Application preparation must be ready")
    image = prepared["image"]
    stat = Path(image["path"]).stat()
    if (stat.st_size, stat.st_mtime_ns) != (image["size"], image["mtime_ns"]):
        raise RuntimeError("Serving image differs from preparation")
    report.update(preparation_id=prepared["preparation_id"], image=image)
    launch_env = host_environment(environment)
    with (run_dir / "stage.log").open("w") as log:
        report["overlay"] = stage_overlay(app, launch_env, log)
    container_env = container_environment()
    for value in set(container_env.values()):
        if value.startswith("/tmp/atp/"):
            (job_tmp / value.removeprefix("/tmp/atp/")).mkdir(parents=True, exist_ok=True)
    command = [
        "srun", "--overlap", "--exact", "--nodes=1", "--ntasks=1", "--cpus-per-task=8",
        "--kill-on-bad-exit=1", "--job-name=atp-serving-imports", "--container-image=" + image["path"],
        "--container-writable", "--no-container-entrypoint", "--no-container-mount-home",
        "--container-workdir=/tmp/atp", "--container-mounts=" + report["overlay"]["path"]
        + ":" + OVERLAY_MOUNT + ":ro+rprivate," + str(job_tmp) + ":/tmp/atp:rw+rprivate",
        "env", *[key + "=" + value for key, value in container_env.items()], MODEL_PYTHON, "-c", PROBE,
    ]
    process = None
    try:
        with (run_dir / "container.log").open("w") as log:
            process = subprocess.Popen(command, env=launch_env, cwd=app, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            if process.wait(timeout=2700) != 0:
                raise RuntimeError("Serving imports or CLI help failed; inspect container.log and cli-help.txt")
    finally:
        if process is not None:
            stop_child(process)
        for filename in ("cli-help.txt", "serving-imports.json"):
            if (job_tmp / filename).is_file():
                shutil.copyfile(job_tmp / filename, run_dir / filename)
    result = json.loads((job_tmp / "serving-imports.json").read_text())
    cpu_limitation = (
        result.get("cli_help") is False and result.get("cuda_available") is False
        and result.get("cli_help_limitation") ==
        "GPU-built vLLM requires a detected device; actual CLI must be validated by GPU smoke"
    )
    if result.get("openai") != OPENAI_VERSION or result.get("api_server_import") is not True or not (result.get("cli_help") is True or cpu_limitation):
        raise RuntimeError("Serving validation returned incomplete evidence")
    if sha256(report["overlay"]["path"]) != report["overlay"]["sha256"]:
        raise RuntimeError("Serving overlay changed during validation")
    report.update(status="passed", versions=result)


def main():
    if len(sys.argv) != 1 or os.environ.get("SLURM_JOB_PARTITION") != "cpu_datamover" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use check_serving_cpu.sbatch in a CPU allocation")
    if os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS"):
        raise RuntimeError("Serving validation must not allocate GPUs")
    app = app_root()
    environment = runtime_env()
    job_tmp = Path(environment["ATP_JOB_TMP"])
    if job_tmp.is_symlink() or job_tmp.parent != Path("/tmp") or not job_tmp.name.startswith("atp-" + environment["SLURM_JOB_ID"] + "-"):
        raise RuntimeError("Serving validation requires its own node-local temporary directory")
    run_dir = make_run_dir("serving-check")
    report = {"status": "checking", "run_dir": str(run_dir), "slurm_job_id": environment["SLURM_JOB_ID"]}
    marker = app / "runtime/serving-check.json"
    with (app / "runtime/serving-check.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save_json(marker, report)
        def interrupted(signum, frame):
            raise KeyboardInterrupt("Serving CPU check interrupted")
        signal.signal(signal.SIGTERM, interrupted)
        try:
            check(app, environment, job_tmp, run_dir, report)
        except BaseException as error:
            report.update(status="failed", error_type=type(error).__name__, error=str(error))
        finally:
            try:
                shutil.rmtree(job_tmp)
                report["temporary_directory_removed"] = True
            except OSError as error:
                report.update(status="failed", cleanup_error=type(error).__name__)
            save_json(run_dir / "check.json", report)
            save_json(marker, report)
    print(f"Serving CPU check {report['status']}: {run_dir / 'check.json'}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
