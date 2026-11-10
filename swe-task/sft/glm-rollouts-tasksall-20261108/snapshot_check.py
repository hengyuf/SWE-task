#!/usr/bin/env python3
"""Read-only check that the OpenSandbox snapshot of every built task still exists (task.toml docker_image
"snapshot:<id>"; the synth build's `cleanup` deletes non-accepted snapshots of a run). One GET /v1/snapshots/<id> per
task with the owner's credential (hengyuf: S/secrets/opensandbox.env; Baihe: glm-rollouts-sftv3's
secrets/baihe-opensandbox.env, BAIHE_SANDBOX_API_KEY), read by NAME and never printed.

    python3 -I -B snapshot_check.py [--limit N] [--workers 8]     # -> snapshots.json (task key -> status)
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import re
import ssl
import time
import tomllib
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
ALL_TASKS = S / "sft/tasks-all-20261106/all_tasks.jsonl"
SECRETS = {"hengyuf": S / "secrets/opensandbox.env",
           "baihe": S / "sft/glm-rollouts-sftv3-20261107/secrets/baihe-opensandbox.env"}
OUT = HERE / "snapshots.json"


def read_env(path: Path) -> dict:
    values = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.strip().removeprefix("export ").partition("=")
        if separator and key.startswith("OPENSANDBOX_"):
            value = value.strip()
            values[key] = value[1:-1] if value[:1] in "'\"" and value[-1:] == value[:1] else value
    return values


def key_of(row) -> str:
    """Unique task key (task ids collide across runs): source/run/cand/task_id."""
    return f"{row['source']}/{row['run']}/{row['cand']}/{row['task_id']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rows = [json.loads(line) for line in ALL_TASKS.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    clients = {}
    for owner, path in SECRETS.items():
        env = read_env(path)
        domain = env["OPENSANDBOX_DOMAIN"].rstrip("/")
        context = ssl.create_default_context()
        if env.get("OPENSANDBOX_TLS_VERIFY", "true").lower() == "false":
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        clients[owner] = {"base": domain if "://" in domain else "https://" + domain,
                          "key": env["OPENSANDBOX_API_KEY"],
                          "opener": urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                                                urllib.request.HTTPSHandler(context=context))}

    def check(row):
        out = {"key": key_of(row), "source": row["source"], "task_id": row["task_id"]}
        try:
            image = tomllib.loads((Path(row["task_dir"]) / "task.toml").read_text())["environment"]["docker_image"]
        except (OSError, KeyError, ValueError) as error:
            return dict(out, result="no_task_toml", error=f"{type(error).__name__}: {error}"[:200])
        out["image"] = image
        match = re.fullmatch(r"snapshot:([0-9a-f-]{36})", image)
        if not match:
            return dict(out, result="not_a_snapshot")
        client = clients[row["source"]]
        request = urllib.request.Request(f"{client['base']}/v1/snapshots/{match.group(1)}",
                                         headers={"OPEN-SANDBOX-API-KEY": client["key"]})
        for attempt in range(3):
            try:
                with client["opener"].open(request, timeout=60) as response:
                    body = json.loads(response.read(20000) or b"{}")
                    state = body.get("status") or body.get("state")
                    if isinstance(state, dict):
                        state = state.get("state") or state.get("phase") or json.dumps(state)[:80]
                    return dict(out, result="exists", http=response.status, state=state)
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return dict(out, result="missing", http=404)
                if error.code < 500 or attempt == 2:
                    return dict(out, result="error", http=error.code, error=error.read(300).decode("utf-8", "replace"))
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
                if attempt == 2:
                    return dict(out, result="error", error=f"{type(error).__name__}: {error}"[:200])
            time.sleep(2 * (attempt + 1))

    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(check, rows))
    summary = {}
    for result in results:
        name = f"{result['source']}:{result['result']}" + (f":{result.get('state')}" if result.get("state") else "")
        summary[name] = summary.get(name, 0) + 1
    OUT.with_suffix(".tmp").write_text(json.dumps({"checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                                   "seconds": round(time.time() - started), "summary": summary,
                                                   "tasks": {r["key"]: r for r in results}}, indent=1) + "\n")
    OUT.with_suffix(".tmp").replace(OUT)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
