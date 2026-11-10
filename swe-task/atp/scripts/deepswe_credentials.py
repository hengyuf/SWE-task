"""Fixed shard credentials: retries never rotate to another account."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path


CREDENTIAL_ENV = {"priority_1": "ATP_DEEPSWE_PRIORITY_1_API_KEY",
                  "priority_2": "ATP_DEEPSWE_PRIORITY_2_API_KEY",
                  "existing": "NVIDIA_API_KEY"}


def load_extra_credentials(app: Path, env: dict) -> None:
    for name in ("ATP_DEEPSWE_PRIORITY_1_API_KEY", "ATP_DEEPSWE_PRIORITY_2_API_KEY"):
        env.pop(name, None)
    path = Path(app) / ".api"
    if not path.exists():
        return
    keys = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(keys) != 2 or any(any(char.isspace() for char in key) for key in keys):
        raise ValueError("Application .api must contain exactly two nonempty raw API keys")
    if len(set(keys + [env.get("NVIDIA_API_KEY")])) != 3:
        raise ValueError("Application API credentials must be distinct")
    env.update(zip(("ATP_DEEPSWE_PRIORITY_1_API_KEY", "ATP_DEEPSWE_PRIORITY_2_API_KEY"), keys))


def credential_ids() -> tuple[str, ...]:
    extra = [bool(os.environ.get(CREDENTIAL_ENV[name])) for name in ("priority_1", "priority_2")]
    if any(extra) and not all(extra):
        raise ValueError("Both additional API credentials must be configured together")
    return tuple(CREDENTIAL_ENV) if all(extra) else ("existing",)


def credential_for_shard(shard: int) -> str:
    if type(shard) is not int or shard < 0:
        raise ValueError("Invalid credential shard")
    names = credential_ids()
    return names[shard % len(names)]


def credential_shards(name: str, shards: int) -> list[int]:
    if name not in credential_ids():
        raise ValueError("Unknown API credential identity")
    return [shard for shard in range(shards) if credential_for_shard(shard) == name]


def bind_credential(attempt_dir: str | None = None) -> tuple[str, str]:
    """Persist only the identity; every turn and transport retry uses the same key."""
    name = credential_for_shard(int(os.environ.get("ATP_DEEPSWE_SHARD_ID", "0")))
    secret = os.environ.get(CREDENTIAL_ENV[name])
    if not secret:
        raise ValueError("Selected API credential is missing")
    if attempt_dir:
        path = Path(attempt_dir) / "credential-binding.json"
        record = {"version": 1, "credential_id": name}
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                if json.loads(path.read_text()) != record:
                    raise ValueError("Attempt credential binding changed; refusing account rotation")
            else:
                from deepswe_campaign import atomic_json
                atomic_json(path, record)
    return name, secret
