#!/usr/bin/env python3
"""One CPU allocation: archive pinned Gym, install selected environments, stage model/data."""
import hashlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

from atp_runtime import GYM_COMMIT, MODEL_REPO, MODEL_REVISION, app_root, build_config, make_run_dir, runtime_env


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, cwd, env):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)


def bootstrap(app, env):
    gym = app / "vendor/Gym"
    pin = app / "vendor/Gym.commit"
    if gym.exists() and not pin.exists():
        gym.rename(app / "vendor" / f"Gym.incomplete-{time.time_ns()}")
    if gym.exists():
        if not pin.exists() or pin.read_text().strip() != GYM_COMMIT:
            raise RuntimeError("Existing Gym snapshot has no matching pin; refusing to overwrite")
    else:
        source = app.parent / "pipeline/nemo_rl/3rdparty/Gym-workspace/Gym"
        extraction = app / "vendor" / f"Gym.extract-{os.getpid()}"
        (app / "vendor").mkdir(parents=True, exist_ok=True)
        staging = Path(env["TMPDIR"]) / f"Gym.extract-{os.getpid()}"
        staging.mkdir()
        archive = app / "vendor/Gym.tar"
        # git archive reads the immutable commit, never dirty working-tree files.
        if not archive.exists():
            run(["git", "-C", source, "archive", "--format=tar", "--output", archive, GYM_COMMIT], app, env)
        with tarfile.open(archive) as handle:
            if handle.pax_headers.get("comment") != GYM_COMMIT:
                raise RuntimeError("Gym archive does not identify the pinned commit")
            members = []
            for member in handle.getmembers():
                # Development-tool shortcuts can point outside the repository; runtime does not use them.
                if member.name.startswith((".claude/", ".codex/")) or member.name == "CLAUDE.md":
                    continue
                members.append(member)
            # Tar safety checks do many realpath calls; perform those on the job's local disk.
            handle.extractall(staging, members=members, filter="data")
        shutil.copytree(staging, extraction, symlinks=True)
        shutil.rmtree(staging)
        extraction.rename(gym)
        pin.write_text(GYM_COMMIT + "\n")
        (app / "vendor/Gym.archive.sha256").write_text(sha256(archive) + "\n")
    uv = app / "runtime/bin/uv"
    if not uv.exists():
        installed = Path("/home") / os.environ["USER"] / ".local/bin/uv"
        if not installed.is_file():
            raise RuntimeError("uv executable is required for preparation")
        shutil.copy2(installed, uv)
    run([uv, "python", "install", "3.13.14"], gym, env)
    run([uv, "sync", "--frozen", "--python", "3.13.14", "--extra", "sandbox"], gym, env)
    run([app / "runtime/controller/bin/python", __file__, "--stage"], gym, env)


def stage(app, env):
    from huggingface_hub import snapshot_download
    gym = app / "vendor/Gym"
    overlay = json.loads((app / "overlays/opensandbox-tls.metadata.json").read_text())
    patch = app / "overlays/opensandbox-tls.patch"
    target = gym / overlay["target"]
    if overlay["base_commit"] != GYM_COMMIT or sha256(patch) != overlay["patch_sha256"]:
        raise RuntimeError("TLS overlay provenance does not match")
    if sha256(target) == overlay["base_sha256"]:
        run(["/usr/bin/patch", "--batch", "--forward", "-p1", "-i", patch], gym, env)
    if sha256(target) != overlay["patched_sha256"]:
        raise RuntimeError("TLS overlay target has unexpected contents")
    preparation_run = make_run_dir("prepare")
    config = build_config(preparation_run, "http://127.0.0.1:9/v1")
    cli = app / "runtime/controller/bin/gym"
    constraints = app / "runtime/constraints.txt"
    run([app / "runtime/bin/uv", "export", "--frozen", "--extra", "sandbox", "--extra", "dev",
         "--no-emit-project", "--no-hashes", "--output-file", constraints], gym, env)
    env["UV_CONSTRAINT"] = str(constraints)
    # Model header validation needs this small package; Gym's sandbox extra does not install it.
    # Match the version already recorded in the pinned Gym lock without pulling an inference engine.
    run([app / "runtime/bin/uv", "pip", "install", "--python", app / "runtime/controller/bin/python",
         "--no-deps", "safetensors==0.8.0"], gym, env)
    # No Ray or servers are started during prefetch.
    run([cli, "env", "prefetch", f"+config_paths=[{config}]", "++skip_venv_if_present=false"], gym, env)
    run([cli, "eval", "prepare", "--config", config], gym, env)
    dataset = gym / "benchmarks/swebench/data/swebench_verified_benchmark.jsonl"
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    if len(rows) != 500 or len({row["instance_id"] for row in rows}) != 500:
        raise RuntimeError("Expected 500 distinct SWE-bench Verified tasks")
    model = app / "models/lightning-3.5"
    revision_path = app / "models/lightning-3.5.revision"
    revision_path.parent.mkdir(parents=True, exist_ok=True)
    token = env.get("HF_TOKEN") or False
    revision = revision_path.read_text().strip() if revision_path.exists() else MODEL_REVISION
    if revision != MODEL_REVISION:
        raise RuntimeError("Existing model revision does not match this application's release pin")
    revision_path.write_text(revision + "\n")
    snapshot_download(MODEL_REPO, revision=revision, local_dir=model, token=token,
                      max_workers=4, ignore_patterns=["*.gguf", "*.onnx"])
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    from safetensors import safe_open
    for shard in shards:
        with safe_open(model / shard, framework="numpy") as handle:
            if not list(handle.keys()):
                raise RuntimeError(f"Empty model shard: {shard}")
    for filename in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        json.loads((model / filename).read_text())
    image = app.parent / "images/nemo-rl.nightly-09032026.x86_64.sqsh"
    stat = image.stat()
    environments = {"controller": app / "runtime/controller/bin/python"}
    for component in ("responses_api_models/vllm_model", "resources_servers/swebench", "responses_api_agents/opencode_sandboxed_agent"):
        environments[component] = app / "runtime/servers" / component / ".venv/bin/python"
    versions = {name: json.loads(subprocess.check_output([
        str(python), "-c", "import importlib.metadata as m,json; print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()}))",
    ], cwd=gym, env=env, text=True)) for name, python in environments.items()}
    (app / "runtime/environment-versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    manifest = {
        "status": "ready", "gym_commit": GYM_COMMIT,
        "gym_archive_sha256": (app / "vendor/Gym.archive.sha256").read_text().strip(),
        "overlays": [overlay],
        "model": {"repo_id": MODEL_REPO, "revision": revision, "path": str(model),
                  "index_sha256": sha256(model / "model.safetensors.index.json"),
                  "shards": {s: (model / s).stat().st_size for s in shards}},
        "dataset": {"path": str(dataset), "sha256": sha256(dataset), "count": 500},
        "image": {"path": str(image), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "controller": str(app / "runtime/controller/bin/python"),
        "environment_versions_sha256": sha256(app / "runtime/environment-versions.json"),
        "constraints_sha256": sha256(constraints),
        "preparation_run": str(preparation_run),
    }
    manifest["preparation_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    target = app / "runtime/preparation.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(target)
    print("Preparation ready:", target, flush=True)


def main():
    if os.environ.get("SLURM_JOB_PARTITION") != "cpu_datamover":
        raise RuntimeError("Run prepare.sh to use the dedicated CPU partition")
    app = app_root()
    env = runtime_env()
    os.environ.clear()
    os.environ.update(env)
    if sys.argv[1:] == ["--stage"]:
        stage(app, env)
    elif not sys.argv[1:]:
        with (app / "runtime/preparation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            (app / "runtime/preparation.json").write_text(json.dumps({"status": "preparing"}) + "\n")
            bootstrap(app, env)
    else:
        raise RuntimeError("Unexpected preparation arguments")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        (app_root() / "runtime/preparation.json").write_text(json.dumps({
            "status": "failed", "error_type": type(error).__name__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "log": f"runs/slurm/prepare-{os.environ.get('SLURM_JOB_ID', 'unknown')}.log",
        }) + "\n")
        raise
