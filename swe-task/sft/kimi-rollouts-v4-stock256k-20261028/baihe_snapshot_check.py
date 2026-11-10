#!/usr/bin/env python3
"""kimi-rollouts-v4 (2026-10-28) copy of kimi-rollouts-v3's baihe_snapshot_check.py, code unchanged except the sandbox metadata purpose kr4-baihe-snapshot-check; uses this directory's secrets/. v3 docstring follows:
kimi-rollouts-v3 (2026-10-26): low-cost check that a Baihe task's OpenSandbox snapshot starts under the
BAIHE_SANDBOX_API_KEY credential (secrets/baihe-opensandbox.env, make_baihe_secrets.py) before any Baihe campaign.
No hub call. Through S/opensandbox/sandbox_helpers.py (the pipeline's own client):
  1. snapshot_info(<task snapshot>)                      (the call that returned HTTP 404 for us on 2026-10-24)
  2. create a sandbox from it (network deny, the task's 2 CPU / 8 GiB, TTL 15 min), run
     `git -C /app rev-parse HEAD; git -C /app status --porcelain | wc -l` and compare HEAD with task.toml's
     base_commit_hash, then delete the sandbox (always).
Writes analysis/baihe-snapshot-check.json (ids, statuses, timings; never a credential). OPENSANDBOX_* variables of the
calling environment are dropped first so that only the file's credential is used.

    <synth controller python> -I -B baihe_snapshot_check.py [--task TASK_ID]
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import time  # noqa: E402
import tomllib  # noqa: E402

HERE = Path(__file__).resolve().parent
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
BAIHE_TSV = S / "sft/baihe-snapshot-access-20261026/baihe_v2_tasks.tsv"
ENV_FILE = HERE / "secrets" / "baihe-opensandbox.env"
for _name in [name for name in os.environ if name.startswith("OPENSANDBOX_")]:
    del os.environ[_name]
sys.path.insert(0, str(S / "opensandbox"))
from sandbox_helpers import Client, Config, _safe_error  # noqa: E402


async def check(task):
    out = {"task_id": task["task_id"], "snapshot": task["snapshot"], "credential": "BAIHE_SANDBOX_API_KEY",
           "env_file": str(ENV_FILE), "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    config = tomllib.loads((Path(task["task_dir"]) / "task.toml").read_text())
    image = config["environment"]["docker_image"]
    out["task_toml_image_matches"] = image == "snapshot:" + task["snapshot"]
    base = config["metadata"].get("base_commit_hash")
    async with Client(Config.load(ENV_FILE), timeout=120) as client:
        started = time.time()
        try:
            info = await client.snapshot_info(task["snapshot"])
            out["snapshot_info"] = {"ok": True, "state": ((info.get("status") or {}).get("state")
                                                          if isinstance(info, dict) else None),
                                    "seconds": round(time.time() - started, 2)}
        except Exception as error:  # noqa: BLE001
            out["snapshot_info"] = {"ok": False, "error": _safe_error(error), "seconds": round(time.time() - started, 2)}
        sid = None
        started = time.time()
        try:
            created = await client.create(snapshot_id=task["snapshot"], ttl=900, timeout=1200, network="deny",
                                          resource={"cpu": "2", "memory": "8Gi"},
                                          metadata={"purpose": "kr4-baihe-snapshot-check", "user": "hengyuf"})
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
    out["ok"] = bool(out.get("create", {}).get("ok") and out.get("exec", {}).get("exit_code") == 0
                     and (out.get("delete") or {}).get("deleted"))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="syn-citty-cand-c01-r1b0-6fa08bef")
    args = parser.parse_args()
    lines = BAIHE_TSV.read_text().splitlines()
    header = lines[0].split("\t")
    rows = {row["task_id"]: row for row in (dict(zip(header, line.split("\t"))) for line in lines[1:] if line.strip())}
    result = asyncio.run(check(rows[args.task]))
    path = HERE / "analysis" / "baihe-snapshot-check.json"
    previous = json.loads(path.read_text()) if path.exists() else []
    path.write_text(json.dumps(previous + [result], indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
