"""OpenSandbox utilities on top of swe-task/opensandbox: owned sandboxes, agent wrappers, snapshots, grading."""
from common import ATP_ENV, HELPER, DEEPSWE, log, read_json, scratch

import asyncio
from contextlib import asynccontextmanager
import importlib.util
import json
from pathlib import Path
import re
import shlex
import shutil
import sys
import uuid

from sandbox_helpers import Client, Config

RESOURCES = {"cpu": "4", "memory": "16Gi"}


def grader_module():
    """The DeepSWE grader (identical in all tasks), imported for its report parsers."""
    spec = importlib.util.spec_from_file_location("deepswe_grader", DEEPSWE / "abs-stepped-slices/tests/grader.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_report(path, grade_cfg) -> dict:
    """{node_id: status} for one CTRF/JUnit report, using the grader's own parser."""
    parsed = grader_module().PARSERS[grade_cfg["format"]](str(path), grade_cfg)
    return {node: status for node, (status, _message) in parsed.items()}


class Box:
    """One owned sandbox; commands run in /app unless `cwd` says otherwise."""

    def __init__(self, client, sid):
        self.client, self.sid = client, sid

    async def sh(self, command, *, cwd="/app", timeout=1800, check=True):
        result = await self.client.exec(self.sid, command, cwd=cwd, timeout=timeout)
        if check and result.exit_code != 0:
            raise RuntimeError(f"[{self.sid}] exit {result.exit_code}: {command[:200]}\n"
                               f"{(result.stdout + result.stderr)[-3000:]}")
        return result

    async def put(self, local, remote):
        await self.client.upload(self.sid, local, remote, timeout=600)

    async def put_text(self, text, remote):
        local = scratch() / f"upload-{uuid.uuid4().hex}"
        local.write_text(text)
        try:
            await self.put(local, remote)
        finally:
            local.unlink(missing_ok=True)

    async def get(self, remote, local):
        """Download one file; the helper only writes under /tmp or /raid/scratch, so stage there."""
        staged = scratch() / f"download-{uuid.uuid4().hex}"
        await self.client.download(self.sid, remote, staged, timeout=600)
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(staged, local)

    async def snapshot(self, name) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name)[:60]
        info = await self.client.snapshot(self.sid, name=safe, wait=True, timeout=2400)
        return "snapshot:" + info["id"]


async def _renew(client, sid, ttl):
    while True:
        await asyncio.sleep(1200)
        try:
            await client.renew(sid, ttl)
        except Exception as error:
            log(f"WARN renew {sid}: {type(error).__name__}")


# The OpenSandbox cluster rejects pods when its nemo.nvidia.com/sandbox-slot resource is used up (2026-10-17 09:06:
# "Node didn't have enough resource: nemo.nvidia.com/sandbox-slot, requested: 1, used: 140, capacity: 140" failed a
# whole evolve branch). ATP's probes back off by themselves; the pipeline's own sandboxes wait here instead (hengyuf).
CAPACITY_FULL = re.compile(r"sandbox-slot|didn't have enough resource|OutOf[A-Za-z.]*/?sandbox", re.I)
CAPACITY_PATIENCE_S = 1800


async def _create(client, start, *, ttl, network, metadata, sleep=asyncio.sleep):
    delay, waited = 30, 0
    while True:
        try:
            return await client.create(**start, ttl=ttl, timeout=1500, network=network, resource=RESOURCES,
                                       metadata=metadata)
        except Exception as error:
            if not CAPACITY_FULL.search(str(error)) or waited >= CAPACITY_PATIENCE_S:
                raise
            log(f"sandbox capacity full ({metadata.get('purpose')}): retrying in {delay} s")
            await sleep(delay)
            waited, delay = waited + delay, min(delay * 2, 300)


@asynccontextmanager
async def owned(source: str, *, run: str, purpose: str, network="deny", ttl=4 * 3600):
    """Create a sandbox from an image or `snapshot:<id>`, keep it alive, and always delete it."""
    async with Client(Config.load(ATP_ENV), timeout=600) as client:
        start = {"snapshot_id": source[len("snapshot:"):]} if source.startswith("snapshot:") else {"image": source}
        created = await _create(client, start, ttl=ttl, network=network,
                                metadata={"run": run[:60], "purpose": purpose[:60]})
        sid = created["id"]
        log(f"sandbox {sid} up ({purpose}, {network})")
        renew = asyncio.create_task(_renew(client, sid, ttl))
        try:
            yield Box(client, sid)
        finally:
            renew.cancel()
            try:
                await client.delete(sid, timeout=120)
            except Exception as error:
                log(f"WARN delete {sid}: {type(error).__name__}")


def write_wrappers(directory, sid):
    """bin/osb-exec, osb-put, osb-get: the only way agents touch their one owned sandbox."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base = f"{shlex.quote(sys.executable)} -B {shlex.quote(str(HELPER / 'sandbox_helpers.py'))} " \
           f"--env-file {shlex.quote(str(ATP_ENV))}"
    scripts = {
        "osb-exec": f'exec {base} exec {sid} "$1" --workdir /app --timeout "${{2:-1800}}"',
        "osb-put": f'exec {base} upload {sid} "$1" "$2" --timeout 600',
        "osb-get": f'tmp=$(mktemp -p /tmp osb-get.XXXXXX) && {base} download {sid} "$1" "$tmp" --timeout 600 '
                   f'>/dev/null && mv "$tmp" "$2"',
    }
    for name, body in scripts.items():
        path = directory / name
        path.write_text(f"#!/bin/sh\n# Bound to sandbox {sid} (repository at /app).\n{body}\n")
        path.chmod(0o755)


async def grade(source, task_dir, patch: str, *, run: str, out_dir) -> dict:
    """Replicates Gym's DeepSWE verifier (_stage_verifier + _run_verifier) in a fresh no-network sandbox."""
    task_dir, out_dir = Path(task_dir), Path(out_dir)
    config = json.loads((task_dir / "tests/config.json").read_text())
    async with owned(source, run=run, purpose="grade") as box:
        await box.sh("mkdir -p /tests /logs/artifacts /logs/verifier", cwd="/")
        for name in ("test.sh", "test.patch", "grader.py", "config.json"):
            await box.put(task_dir / "tests" / name, f"/tests/{name}")
        await box.put_text(patch, "/logs/artifacts/model.patch")
        await box.sh("chmod 0755 /tests/test.sh /tests/grader.py", cwd="/")
        result = await box.sh("bash /tests/test.sh", cwd="/app", timeout=1800, check=False)
        (out_dir).mkdir(parents=True, exist_ok=True)
        (out_dir / "output.log").write_text(result.stdout + result.stderr)
        reward = None
        if (await box.sh("test -f /logs/verifier/reward.json", cwd="/", check=False)).exit_code == 0:
            await box.get("/logs/verifier/reward.json", out_dir / "reward.json")
            reward = read_json(out_dir / "reward.json")
        suites = {}
        for remote in config["grade"]["reports"]:
            name = Path(remote).name
            for candidate in (f"/logs/verifier/reports/{name}", remote):  # the frame moves reports/ at the end
                if (await box.sh(f"test -f {candidate}", cwd="/", check=False)).exit_code == 0:
                    await box.get(candidate, out_dir / name)
                    suites[name] = parse_report(out_dir / name, config["grade"])
                    break
            else:
                suites[name] = {}
    return {"reward": (reward or {}).get("reward"), "reward_json": reward, "exit_code": result.exit_code,
            "timed_out": result.timed_out, "suites": suites}
