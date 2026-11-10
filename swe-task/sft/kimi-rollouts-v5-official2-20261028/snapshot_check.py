#!/usr/bin/env python3
"""kimi-rollouts-v5 (2026-10-28): low-cost check that every official2 task's OpenSandbox snapshot starts under hengyuf's
sandbox credential before the batch (kimi-rollouts-v4's baihe_snapshot_check.py, adapted: hengyuf's credential instead
of Baihe's, every distinct image of official2-tasks.json instead of one Baihe task). No hub call. Through
S/opensandbox/sandbox_helpers.py (the pipeline's own client), per distinct image (one task per image):
  1. snapshot_info(<snapshot id>)
  2. create a sandbox from it (network deny, the task's CPUs / memory, TTL 15 min), run
     `git -C /app rev-parse HEAD; git -C /app status --porcelain | wc -l` and compare HEAD with task.toml's
     base_commit_hash (informational), then delete the sandbox (always).
Credential: the OPENSANDBOX_* settings of S/atp/.env, i.e. exactly what ATP stages for the rollouts (its
OPENSANDBOX_API_KEY is CELL3_SANDBOX_API_KEY, written by S/tools/make_secrets.py). Before any call the script checks,
by NAME and without printing anything but booleans, that this value equals CELL3_SANDBOX_API_KEY in
/lustre/fsw/portfolios/nemotron/users/hengyuf/.env. OPENSANDBOX_* variables of the calling environment are dropped first
so that only the file's credential is used. Writes analysis/snapshot-check.json (ids, statuses, timings; never a
credential).

    <ATP controller python> -I -B snapshot_check.py [--parallel 2] [--task TASK_ID ...]
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import asyncio  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import time  # noqa: E402
import tomllib  # noqa: E402

HERE = Path(__file__).resolve().parent
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
ENV_FILE = S / "atp" / ".env"  # what ATP stages for the rollouts (deepswe_stage.read_dotenv)
USER_ENV = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/.env")
CREDENTIAL = "CELL3_SANDBOX_API_KEY"
TASKS_FILE = HERE / "official2-tasks.json"
OUT = HERE / "analysis" / "snapshot-check.json"
for _name in [name for name in os.environ if name.startswith("OPENSANDBOX_")]:
    del os.environ[_name]
sys.path.insert(0, str(S / "opensandbox"))
from sandbox_helpers import Client, Config, _literal_env, _safe_error  # noqa: E402


def credential_is_cell3() -> bool:
    """S/atp/.env's OPENSANDBOX_API_KEY == CELL3_SANDBOX_API_KEY of ~hengyuf/.env (values compared, never shown)."""
    spec = importlib.util.spec_from_file_location("kr5_make_secrets", S / "tools" / "make_secrets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = module.parse_source(USER_ENV.read_text())
    staged = _literal_env(ENV_FILE).get("OPENSANDBOX_API_KEY")
    return bool(source.get(CREDENTIAL)) and staged == source.get(CREDENTIAL)


async def check(task, client, semaphore):
    async with semaphore:
        snapshot = task["image"].removeprefix("snapshot:")
        out = {"task_id": task["task_id"], "run": task["run"], "image": task["image"], "credential": CREDENTIAL,
               "env_file": str(ENV_FILE), "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        config = tomllib.loads((Path(task["task_dir"]) / "task.toml").read_text())
        environment = config["environment"]
        out["task_toml_image_matches"] = environment["docker_image"] == task["image"]
        base = config["metadata"].get("base_commit_hash")
        resource = {"cpu": str(environment.get("cpus") or 2),
                    "memory": f"{max(1, int(environment.get('memory_mb') or 8192) // 1024)}Gi"}
        out["resource"] = resource
        started = time.time()
        try:
            info = await client.snapshot_info(snapshot)
            out["snapshot_info"] = {"ok": True, "state": ((info.get("status") or {}).get("state")
                                                          if isinstance(info, dict) else None),
                                    "seconds": round(time.time() - started, 2)}
        except Exception as error:  # noqa: BLE001
            out["snapshot_info"] = {"ok": False, "error": _safe_error(error), "seconds": round(time.time() - started, 2)}
        sid = None
        started = time.time()
        try:
            created = await client.create(snapshot_id=snapshot, ttl=900, timeout=1200, network="deny",
                                          resource=resource,
                                          metadata={"purpose": "kr5-snapshot-check", "user": "hengyuf"})
            sid = created["id"]
            out["create"] = {"ok": True, "sandbox_id": sid, "seconds": round(time.time() - started, 1)}
            result = await client.exec(sid, "git -C /app rev-parse HEAD; git -C /app status --porcelain | wc -l",
                                       timeout=120)
            lines = result.stdout.split()
            out["exec"] = {"exit_code": result.exit_code, "head": lines[0] if lines else None,
                           "dirty_paths": int(lines[1]) if len(lines) > 1 and lines[1].isdigit() else None,
                           "head_is_base_commit": bool(lines) and lines[0] == base, "stderr_tail": result.stderr[-300:]}
        except Exception as error:  # noqa: BLE001
            out["create"] = {"ok": False, "error": _safe_error(error), "seconds": round(time.time() - started, 1),
                             "sandbox_id": sid or getattr(error, "resource_id", None)}
            sid = sid or getattr(error, "resource_id", None)
        finally:
            if sid:
                try:
                    out["delete"] = await client.delete(sid, timeout=120)
                except Exception as error:  # noqa: BLE001
                    out["delete"] = {"id": sid, "deleted": False, "error": _safe_error(error)}
        out["ok"] = bool(out["snapshot_info"].get("ok") and out.get("create", {}).get("ok")
                         and out.get("exec", {}).get("exit_code") == 0 and (out.get("delete") or {}).get("deleted"))
        print(json.dumps({k: out.get(k) for k in ("task_id", "image", "ok")} |
                         {"create_s": (out.get("create") or {}).get("seconds"),
                          "head_is_base_commit": (out.get("exec") or {}).get("head_is_base_commit")}), flush=True)
        return out


async def run(tasks, parallel):
    semaphore = asyncio.Semaphore(parallel)
    async with Client(Config.load(ENV_FILE), timeout=120) as client:
        return await asyncio.gather(*(check(task, client, semaphore) for task in tasks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--task", nargs="*", help="only these task ids")
    args = parser.parse_args()
    if not credential_is_cell3():
        print(json.dumps({"ok": False, "error": f"{ENV_FILE} OPENSANDBOX_API_KEY is not {CREDENTIAL}; not checking"}))
        return 2
    tasks, seen = [], set()
    for task in json.loads(TASKS_FILE.read_text())["tasks"]:
        if (args.task and task["task_id"] not in args.task) or task["image"] in seen:
            continue
        seen.add(task["image"])
        tasks.append(task)
    started = time.time()
    results = asyncio.run(run(tasks, max(1, args.parallel)))
    summary = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "credential": CREDENTIAL, "credential_is_cell3": True,
               "env_file": str(ENV_FILE), "images": len(tasks), "ok": sum(r["ok"] for r in results),
               "failed": [r["task_id"] for r in results if not r["ok"]],
               "head_is_base_commit": sum(bool((r.get("exec") or {}).get("head_is_base_commit")) for r in results),
               "seconds": round(time.time() - started, 1)}
    OUT.parent.mkdir(exist_ok=True)
    previous = json.loads(OUT.read_text()) if OUT.exists() else []
    OUT.write_text(json.dumps(previous + [{"summary": summary, "results": results}], indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0 if summary["ok"] == len(tasks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
