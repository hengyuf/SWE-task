#!/usr/bin/env python3
"""Rebuild this app's local runtime: pristine vendor/Gym and the controller venv (hengyuf addition).

Mirrors ATP scripts/prepare.py bootstrap (TLS overlay only; uv sync --frozen --extra sandbox; safetensors
0.8.0; constraints export) and deepswe_stage.stage() (same filtered Gym.tar member list). Stdlib only:
    /usr/bin/python3 -I -B tools/build_runtime.py vendor       # vendor/Gym, read-only, pristine 2f11dec3
    /usr/bin/python3 -I -B tools/build_runtime.py controller   # vendor/Gym-controller-src + runtime/controller
    /usr/bin/python3 -I -B tools/build_runtime.py verify       # imports + freeze list
Needs runtime/bin/uv (>= 0.12.9) and network access (python-build-standalone, PyPI). No Slurm needed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

APP = Path(__file__).resolve().parents[1]
GYM_COMMIT = "2f11dec3a1c11c0c5f302ab199f97754f7b5c8bd"
PYTHON_VERSION = "3.13.14"
CONTROLLER = APP / "runtime/controller"
CONTROLLER_SOURCE = APP / "vendor/Gym-controller-src"
UV = APP / "runtime/bin/uv"
IMPORTS = ("httpx", "opensandbox", "pydantic", "aiohttp", "tomllib", "safetensors", "nemo_gym")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    """Clean uv environment rooted in this app (the caller's UV_*/PYTHON* settings never leak in)."""
    env = {key: os.environ[key] for key in ("HOME", "USER", "LOGNAME", "LANG", "TERM") if key in os.environ}
    paths = {"UV_CACHE_DIR": "cache/uv", "UV_PYTHON_INSTALL_DIR": "runtime/python", "UV_PYTHON_BIN_DIR": "runtime/bin",
             "UV_PROJECT_ENVIRONMENT": "runtime/controller", "XDG_CONFIG_HOME": "runtime/config",
             "XDG_CACHE_HOME": "cache/xdg", "XDG_DATA_HOME": "runtime/data"}
    for key, relative in paths.items():
        if key != "UV_PROJECT_ENVIRONMENT":  # uv creates the venv itself
            (APP / relative).mkdir(parents=True, exist_ok=True)
        env[key] = str(APP / relative)
    env.update(PATH=os.pathsep.join([str(APP / "runtime/bin"), "/usr/local/bin", "/usr/bin", "/bin"]),
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", UV_LINK_MODE="hardlink", TMPDIR=tempfile.gettempdir())
    return env


def run(command, cwd, env, **kwargs):
    print("Running:", " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True, **kwargs)


def extract(destination):
    """Same member filter as deepswe_stage.stage(); tar safety checks run on local disk, then copy to Lustre."""
    destination = Path(destination)
    if destination.exists():
        raise RuntimeError(f"{destination} exists; remove it first (chmod -R u+w; rm -rf) to rebuild")
    with tempfile.TemporaryDirectory(prefix="atp-gym-extract-") as temporary:
        staging = Path(temporary) / "Gym"
        staging.mkdir()
        with tarfile.open(APP / "vendor/Gym.tar") as archive:
            if archive.pax_headers.get("comment") != GYM_COMMIT:
                raise ValueError("Gym archive does not identify the pinned commit")
            members = [member for member in archive.getmembers()
                       if not member.name.startswith((".claude/", ".codex/")) and member.name != "CLAUDE.md"]
            archive.extractall(staging, members=members, filter="data")
        partial = destination.with_name(destination.name + f".partial-{os.getpid()}")
        shutil.copytree(staging, partial, symlinks=True)
    partial.rename(destination)
    return destination


def read_only(path):
    run(["chmod", "-R", "a-w", path], APP, environment())


def vendor():
    if sha256(APP / "vendor/Gym.tar") != (APP / "vendor/Gym.tar.sha256").read_text().split()[0]:
        raise ValueError("vendor/Gym.tar does not match vendor/Gym.tar.sha256")
    gym = extract(APP / "vendor/Gym")
    (APP / "vendor/Gym.commit").write_text(GYM_COMMIT + "\n")
    read_only(gym)
    print("Pristine Gym ready (read-only):", gym, flush=True)


def apply_tls_overlay(gym, env):
    """prepare.py stage(): only the OpenSandbox TLS overlay, with all recorded hashes checked."""
    overlay = json.loads((APP / "overlays/opensandbox-tls.metadata.json").read_text())
    patch = APP / "overlays/opensandbox-tls.patch"
    target = gym / overlay["target"]
    if overlay["base_commit"] != GYM_COMMIT or sha256(patch) != overlay["patch_sha256"]:
        raise RuntimeError("TLS overlay provenance does not match")
    if sha256(target) == overlay["base_sha256"]:
        run(["/usr/bin/patch", "--batch", "--forward", "-p1", "-i", patch], gym, env)
    if sha256(target) != overlay["patched_sha256"]:
        raise RuntimeError("TLS overlay target has unexpected contents")


def controller():
    env = environment()
    if not UV.is_file():
        raise RuntimeError("Copy uv >= 0.12.9 to runtime/bin/uv first")
    if CONTROLLER.exists():
        raise RuntimeError(f"{CONTROLLER} exists; remove it first to rebuild")
    gym = extract(CONTROLLER_SOURCE)
    apply_tls_overlay(gym, env)
    run([UV, "python", "install", PYTHON_VERSION], APP, env)
    python = subprocess.check_output([str(UV), "python", "find", "--managed-python", PYTHON_VERSION],
                                     cwd=APP, env=env, text=True).strip()
    if not Path(python).resolve().is_relative_to((APP / "runtime/python").resolve()):
        raise RuntimeError("uv selected a Python interpreter outside runtime/python")
    # uv installs the Gym project itself in editable mode: vendor/Gym-controller-src must stay in place.
    run([UV, "sync", "--frozen", "--python", python, "--extra", "sandbox"], gym, env)
    # Model header validation needs this small package; Gym's sandbox extra does not install it (prepare.py).
    run([UV, "pip", "install", "--python", CONTROLLER / "bin/python", "--no-deps", "safetensors==0.8.0"], gym, env)
    run([UV, "export", "--quiet", "--frozen", "--extra", "sandbox", "--extra", "dev", "--no-emit-project",
         "--no-hashes", "--output-file", APP / "runtime/constraints.txt"], gym, env)
    read_only(gym)
    verify()


def verify():
    env = environment()
    python = CONTROLLER / "bin/python"
    code = ("import importlib, importlib.metadata as m, json, sys; "
            f"[importlib.import_module(name) for name in {IMPORTS!r}]; "
            "print(json.dumps({'python': sys.version.split()[0], 'opensandbox': m.version('opensandbox'), "
            "'httpx': m.version('httpx'), 'pydantic': m.version('pydantic'), 'aiohttp': m.version('aiohttp'), "
            "'safetensors': m.version('safetensors'), 'nemo_gym': importlib.import_module('nemo_gym').__file__}))")
    result = json.loads(run([python, "-I", "-c", code], APP, env, capture_output=True, text=True).stdout)
    if result["python"] != PYTHON_VERSION or result["opensandbox"] != "0.1.15" or not result["pydantic"].startswith("2."):
        raise RuntimeError(f"Unexpected controller versions: {result}")
    if not Path(result["nemo_gym"]).resolve().is_relative_to(CONTROLLER_SOURCE.resolve()):
        raise RuntimeError("nemo_gym does not import from vendor/Gym-controller-src")
    freeze = run([UV, "pip", "freeze", "--python", python], APP, env, capture_output=True, text=True).stdout
    (APP / "runtime/controller.freeze.txt").write_text(freeze)
    print(json.dumps(result, indent=2), flush=True)
    print(f"{len(freeze.splitlines())} distributions -> {APP / 'runtime/controller.freeze.txt'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("vendor", "controller", "verify"))
    {"vendor": vendor, "controller": controller, "verify": verify}[parser.parse_args().step]()
