"""Model-wide HTTP capacity and completion-proven DeepSWE rollout admission."""
from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
import fcntl
import json
import os
from pathlib import Path
import time
import uuid

from deepswe_campaign import atomic_json, load_manifest, read_json
from deepswe_credentials import credential_for_shard, credential_ids, credential_shards


MAX_CONCURRENCY_PER_MODEL = 2048
RAMP_INTERVAL_SECONDS = 30
PRESSURE_INTERVAL_SECONDS = 60
MIN_SUCCESSFUL_CALLS = 16
POLL_INTERVAL_SECONDS = 1.0
EVENTS = {"success_http", "rate_limit", "upstream_timeout", "upstream_error", "gateway_capacity",
          "sandbox_started", "sandbox_failed", "api_saturated"}
_active = {}
_api_active = {}
_api_waiters = {}
_rollout_waiters = {}


def split_budget(total: int, shards: int, shard_id: int) -> int:
    if (any(isinstance(value, bool) or not isinstance(value, int) for value in (total, shards, shard_id))
            or total < 0 or shards < 1 or not 0 <= shard_id < shards):
        raise ValueError("Invalid admission budget or shard")
    quotient, remainder = divmod(total, shards)
    return quotient + (shard_id < remainder)


def _credential_state(current, manifest, name, now):
    owned = credential_shards(name, manifest["shards"])
    maximum = manifest["settings"].get("max_concurrency_per_model", MAX_CONCURRENCY_PER_MODEL)
    maximum = sum(split_budget(maximum, manifest["shards"], shard) for shard in owned)
    initial = manifest["settings"].get("initial_concurrency_per_model", 16)
    initial = sum(split_budget(initial, manifest["shards"], shard) for shard in owned)
    states = dict(current.get("credentials", {}))
    state = {"limit": initial, "changed_at": now, "hold_until": now,
             "successful_calls": 0, "api_saturated": False, **states.get(name, {})}
    if type(state["limit"]) is not int or not len(owned) <= state["limit"] <= maximum:
        raise ValueError("Invalid credential admission limit")
    states[name] = state
    current["credentials"] = states
    return state, len(owned), maximum


def _credential_quota(current, manifest, shard, quota):
    if len(credential_ids()) == 1:
        return quota
    name = credential_for_shard(shard)
    state, _, _ = _credential_state(current, manifest, name, time.time())
    owned = credential_shards(name, manifest["shards"])
    return min(quota, split_budget(state["limit"], len(owned), owned.index(shard)))


def _note_credential(current, manifest, event, now):
    name = credential_for_shard(int(os.environ["ATP_DEEPSWE_SHARD_ID"]))
    state, minimum, maximum = _credential_state(current, manifest, name, now)
    if event in {"rate_limit", "upstream_timeout", "upstream_error"}:
        decreased = state.get("last_decrease_at")
        if decreased is None or now >= decreased + PRESSURE_INTERVAL_SECONDS:
            state["limit"] = max(minimum, state["limit"] // 2)
            state["last_decrease_at"] = now
        state.update(changed_at=now, hold_until=now + PRESSURE_INTERVAL_SECONDS,
                     successful_calls=0, api_saturated=False, last_pressure=event)
    else:
        if event == "success_http":
            state["successful_calls"] = min(MIN_SUCCESSFUL_CALLS, state["successful_calls"] + 1)
        elif event == "api_saturated":
            state["api_saturated"] = True
        if (state["limit"] < maximum and now >= state["changed_at"] + RAMP_INTERVAL_SECONDS
                and now >= state["hold_until"] and state["api_saturated"]
                and state["successful_calls"] >= MIN_SUCCESSFUL_CALLS):
            state.update(limit=min(maximum, state["limit"] * 2), changed_at=now,
                         successful_calls=0, api_saturated=False)


def _limits(campaign: Path):
    manifest = load_manifest(campaign)
    settings, shards = manifest["settings"], manifest["shards"]
    maximum = settings.get("max_concurrency_per_model", MAX_CONCURRENCY_PER_MODEL)
    initial = settings.get("initial_concurrency_per_model", 16)
    if (any(isinstance(value, bool) or not isinstance(value, int) for value in (initial, maximum, shards))
            or not 1 <= shards <= initial <= maximum <= MAX_CONCURRENCY_PER_MODEL):
        raise ValueError("Admission requires shards <= initial <= maximum <= 2048")
    saved = read_json(campaign / "concurrency.json", {})
    if not isinstance(saved, dict) or set(saved) - set(settings["models"]):
        raise ValueError("Concurrency file contains unknown models or is not an object")
    limits = {model: saved.get(model, initial) for model in settings["models"]}
    if any(isinstance(value, bool) or not isinstance(value, int) or not shards <= value <= maximum
           for value in limits.values()):
        raise ValueError("Live model concurrency is outside the campaign bounds")
    return manifest, limits, maximum


def _rollout_epoch(current, limit, now):
    current.update(rollout_limit=limit, rollout_epoch=uuid.uuid4().hex,
                   rollout_changed_at=now, healthy_completion=False)


def _state(manifest, limit, saved, now):
    current = dict(saved or {})
    current.setdefault("limit", limit)
    current.setdefault("changed_at", now)
    current.setdefault("hold_until", now)
    current.setdefault("successful_calls", 0)
    current.setdefault("sandbox_starts", 0)
    current.setdefault("api_saturated", False)
    current.setdefault("rollout_limit", min(limit, manifest["settings"].get("initial_concurrency_per_model", 16)))
    current.setdefault("rollout_epoch", uuid.uuid4().hex)
    current.setdefault("rollout_changed_at", now)
    current.setdefault("healthy_completion", False)
    current["api_waiting_shards"] = dict(current.get("api_waiting_shards", {}))
    current["rollout_waiting_shards"] = dict(current.get("rollout_waiting_shards", {}))
    rollout = current["rollout_limit"]
    if (type(rollout) is not int or not manifest["shards"] <= rollout <= MAX_CONCURRENCY_PER_MODEL
            or not isinstance(current["rollout_epoch"], str) or not current["rollout_epoch"]):
        raise ValueError("Invalid rollout admission state")
    if current["limit"] != limit or rollout > limit:
        current.update(limit=limit, changed_at=now, successful_calls=0, api_saturated=False)
        if rollout > limit:
            _rollout_epoch(current, limit, now)
    return current


def _grow_rollouts(current, now):
    if (current["rollout_limit"] < current["limit"] and current["healthy_completion"]
            and any(current["rollout_waiting_shards"].values())
            and not any(current["api_waiting_shards"].values())
            and now >= current["hold_until"]
            and now >= current["rollout_changed_at"] + RAMP_INTERVAL_SECONDS):
        _rollout_epoch(current, min(current["limit"], current["rollout_limit"] * 2), now)


def _write_change(directory, event, result):
    with (directory / "concurrency-changes.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": event, **result}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def set_concurrency(campaign: Path, value: int) -> dict:
    """Set API budgets; raising them never bypasses rollout reliability checks."""
    if type(value) is not int:
        raise ValueError("Concurrency must be an integer")
    campaign = Path(campaign).resolve()
    if not (campaign / "manifest.json").is_file():
        raise ValueError("Campaign manifest does not exist")
    directory = campaign / "rate-limits"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "admission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest, prior, maximum = _limits(campaign)
        if not manifest["shards"] <= value <= maximum:
            raise ValueError(f"Concurrency must be between {manifest['shards']} and {maximum}")
        now = time.time()
        limits = {model: value for model in prior}
        states = read_json(directory / "admission.json", {})
        for model in limits:
            current = _state(manifest, prior[model], states.get(model), now)
            current.update(limit=value, changed_at=now, successful_calls=0, api_saturated=False,
                           hold_until=max(now, current["hold_until"]), sandbox_starts=0)
            if current["rollout_limit"] > value:
                _rollout_epoch(current, value, now)
            states[model] = current
        atomic_json(directory / "admission.json", states)
        atomic_json(campaign / "concurrency.json", limits)
        result = {"prior": prior, "new": limits, "timestamp": now}
        _write_change(directory, "set_concurrency", result)
        return result


def reset_admission(campaign: Path, repair_revision: str) -> dict:
    """Explicit, idempotent repair reset. Caller must first stop all old writers."""
    if (not isinstance(repair_revision, str) or not repair_revision
            or not all(char.isalnum() or char in "._-" for char in repair_revision)):
        raise ValueError("Repair revision must be a simple nonempty name")
    campaign = Path(campaign).resolve()
    directory = campaign / "rate-limits"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "admission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest, prior, _ = _limits(campaign)
        record_path = directory / ("admission-reset-" + repair_revision + ".json")
        previous = read_json(record_path)
        if previous is not None:
            return previous
        now = time.time()
        initial = manifest["settings"].get("initial_concurrency_per_model", 16)
        limits = {model: initial for model in prior}
        states = {model: _state(manifest, initial, None, now) for model in prior}
        result = {"repair_revision": repair_revision, "prior": prior, "new": limits, "timestamp": now}
        atomic_json(directory / "admission.json", states)
        atomic_json(campaign / "concurrency.json", limits)
        atomic_json(record_path, result)
        _write_change(directory, "reset_admission", result)
        return result


def note_health(model: str, event: str) -> None:
    """Coalesce pressure by model; only shared service failures throttle all."""
    location = os.environ.get("ATP_DEEPSWE_CAMPAIGN")
    if not location:
        return
    if event not in EVENTS:
        raise ValueError("Unknown admission health event")
    campaign = Path(location).resolve()
    if not (campaign / "manifest.json").is_file():
        return
    directory = campaign / "rate-limits"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "admission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest, limits, maximum = _limits(campaign)
        if model not in limits:
            raise ValueError("Unknown model alias for admission health")
        now = time.time()
        states = read_json(directory / "admission.json", {})
        credential_event = len(credential_ids()) > 1 and event in {
            "rate_limit", "upstream_timeout", "upstream_error", "success_http", "api_saturated"}
        if credential_event:
            scoped = _state(manifest, limits[model], states.get(model), now)
            _note_credential(scoped, manifest, event, now)
            states[model] = scoped
            # An account-specific quota never throttles the other credentials.
            if event in {"rate_limit", "upstream_timeout", "upstream_error"}:
                atomic_json(directory / "admission.json", states)
                return
        if event in {"rate_limit", "upstream_timeout", "upstream_error", "gateway_capacity", "sandbox_failed"}:
            aliases = limits if event in {"gateway_capacity", "sandbox_failed"} else (model,)
            changed = False
            for alias in aliases:
                old_limit = limits[alias]
                current = _state(manifest, old_limit, states.get(alias), now)
                decrease_at = current.get("last_decrease_at")
                if decrease_at is None or now >= decrease_at + PRESSURE_INTERVAL_SECONDS:
                    limits[alias] = max(manifest["shards"], old_limit // 2)
                    decrease_at = now
                    changed |= old_limit != limits[alias]
                current.update(limit=limits[alias], changed_at=now,
                    hold_until=now + PRESSURE_INTERVAL_SECONDS, successful_calls=0,
                    api_saturated=False, sandbox_starts=0, last_pressure=event, last_decrease_at=decrease_at)
                _rollout_epoch(current, min(current["rollout_limit"], limits[alias]), now)
                states[alias] = current
            if changed or not (campaign / "concurrency.json").exists():
                atomic_json(campaign / "concurrency.json", limits)
            atomic_json(directory / "admission.json", states)
            return
        current = _state(manifest, limits[model], states.get(model), now)
        old_limit = limits[model]
        if event == "success_http":
            current["successful_calls"] = min(MIN_SUCCESSFUL_CALLS, current["successful_calls"] + 1)
        elif event == "sandbox_started":
            current["sandbox_starts"] = 1
        elif event == "api_saturated":
            current["api_saturated"] = True
        if (old_limit < maximum and now >= current["changed_at"] + RAMP_INTERVAL_SECONDS
                and now >= current["hold_until"] and current["api_saturated"]
                and current["successful_calls"] >= MIN_SUCCESSFUL_CALLS):
            limits[model] = min(maximum, old_limit * 2)
            current.update(limit=limits[model], changed_at=now, successful_calls=0,
                           api_saturated=False, sandbox_starts=0)
        _grow_rollouts(current, now)
        if old_limit != limits[model] or not (campaign / "concurrency.json").exists():
            atomic_json(campaign / "concurrency.json", limits)
        if credential_event or states.get(model) != current:
            states[model] = current
            atomic_json(directory / "admission.json", states)


def _update_shard(campaign, model, shard, *, queue=None, reset=False):
    directory = campaign / "rate-limits"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "admission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest, limits, _ = _limits(campaign)
        if model not in limits or type(shard) is not int or not 0 <= shard < manifest["shards"]:
            raise ValueError("Invalid model or shard for admission health")
        states = read_json(directory / "admission.json", {})
        current = _state(manifest, limits[model], states.get(model), time.time())
        if reset:
            for name in ("api_waiting_shards", "rollout_waiting_shards"):
                current[name].pop(str(shard), None)
        elif queue is not None:
            # Read the latest local queue under the shared lock, so concurrent
            # to_thread updates cannot publish stale queue snapshots.
            key = (str(campaign), model, shard)
            registry = _api_waiters if queue == "api" else _rollout_waiters
            current[queue + "_waiting_shards"][str(shard)] = bool(registry.get(key))
        _grow_rollouts(current, time.time())
        if states.get(model) != current:
            states[model] = current
            atomic_json(directory / "admission.json", states)
        return current["rollout_limit"], current["rollout_epoch"]


def reset_shard_health(campaign: Path, model: str, shard_id: int) -> None:
    """Clear stale queue flags only after that shard's old collector was cleaned."""
    _update_shard(Path(campaign).resolve(), model, shard_id, reset=True)


def note_rollout_completed(row: dict) -> None:
    """Caller has validated the complete generation and verifier result."""
    location = os.environ.get("ATP_DEEPSWE_CAMPAIGN")
    epoch = row.get("_atp_admission_epoch")
    if not location or not epoch or row.get("_atp_regrade"):
        return
    campaign = Path(location).resolve()
    model = os.environ["ATP_DEEPSWE_MODEL_ALIAS"]
    directory = campaign / "rate-limits"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "admission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest, limits, _ = _limits(campaign)
        states = read_json(directory / "admission.json", {})
        current = _state(manifest, limits[model], states.get(model), time.time())
        if current["rollout_epoch"] != epoch:
            return
        current["healthy_completion"] = True
        _grow_rollouts(current, time.time())
        states[model] = current
        atomic_json(directory / "admission.json", states)


@asynccontextmanager
async def api_slot():
    """FIFO capacity for upstream calls, independent of rollout lifetime."""
    location = os.environ.get("ATP_DEEPSWE_CAMPAIGN")
    if not location or not (Path(location) / "manifest.json").is_file():
        yield
        return
    campaign = Path(location).resolve()
    model = os.environ["ATP_DEEPSWE_MODEL_ALIAS"]
    shard_id = int(os.environ["ATP_DEEPSWE_SHARD_ID"])
    key = (str(campaign), model, shard_id)
    token = object()
    queue = _api_waiters.setdefault(key, deque())
    queue.append(token)
    admitted = False
    reported = False
    try:
        while True:
            manifest, limits, _ = _limits(campaign)
            quota = split_budget(limits[model], manifest["shards"], shard_id)
            saved = read_json(campaign / "rate-limits" / "admission.json", {}).get(model, {})
            quota = _credential_quota(saved, manifest, shard_id, quota)
            if queue[0] is token and _api_active.get(key, 0) < quota:
                queue.popleft()
                _api_active[key] = _api_active.get(key, 0) + 1
                admitted = True
                # Another waiter's threaded update can observe this token
                # while _limits reads disk, even if this request never waits.
                if reported or not queue:
                    await asyncio.to_thread(_update_shard, campaign, model, shard_id, queue="api")
                if _api_active[key] >= quota:
                    await asyncio.to_thread(note_health, model, "api_saturated")
                break
            if not reported:
                reported = True
                await asyncio.to_thread(_update_shard, campaign, model, shard_id, queue="api")
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
        yield
    finally:
        if admitted:
            _api_active[key] -= 1
            if not _api_active[key]:
                del _api_active[key]
        else:
            queue.remove(token)
        if not queue and _api_waiters.get(key) is queue:
            _api_waiters.pop(key, None)
        if reported:
            await asyncio.to_thread(_update_shard, campaign, model, shard_id, queue="api")


@asynccontextmanager
async def admission_slot(row: dict):
    """Gate new rollouts and stamp the epoch needed for completion evidence."""
    location = os.environ.get("ATP_DEEPSWE_CAMPAIGN")
    if not location:
        yield
        return
    campaign = Path(location).resolve()
    admitted = False
    key = None
    token = object()
    waiting = False
    try:
        while True:
            reason = None
            try:
                model = os.environ["ATP_DEEPSWE_MODEL_ALIAS"]
                manifest, limits, _ = _limits(campaign)
                shard_id = int(os.environ.get("ATP_DEEPSWE_SHARD_ID", row["_ng_task_index"] % manifest["shards"]))
                if row["_ng_task_index"] % manifest["shards"] != shard_id:
                    raise ValueError("Task belongs to a different shard")
                key = (str(campaign), model, shard_id)
                saved = read_json(campaign / "rate-limits" / "admission.json", {}).get(model, {})
                if "rollout_epoch" not in saved:
                    rollout_limit, epoch = await asyncio.to_thread(_update_shard, campaign, model, shard_id)
                else:
                    rollout_limit, epoch = saved["rollout_limit"], saved["rollout_epoch"]
                    if type(rollout_limit) is not int or not manifest["shards"] <= rollout_limit <= MAX_CONCURRENCY_PER_MODEL:
                        raise ValueError("Invalid rollout admission limit")
                    rollout_limit = min(rollout_limit, limits[model])
                quota = split_budget(rollout_limit, manifest["shards"], shard_id)
                quota = _credential_quota(saved, manifest, shard_id, quota)
                launch = os.environ.get("ATP_DEEPSWE_LAUNCH_DIR")
                closed = read_json(Path(launch) / "admission-closed.json") if launch else None
                if closed is not None:
                    reason = closed.get("reason") or "Launch admission is closed"
                elif time.time() >= float(os.environ.get("ATP_DEEPSWE_JOB_DEADLINE", "inf")):
                    reason = "Allocation deadline reached"
                elif _active.get(key, 0) < quota:
                    _active[key] = _active.get(key, 0) + 1
                    row["_atp_admission_epoch"] = epoch
                    admitted = True
                    break
                elif not waiting:
                    _rollout_waiters.setdefault(key, set()).add(token)
                    waiting = True
                    await asyncio.to_thread(_update_shard, campaign, model, shard_id, queue="rollout")
            except (KeyError, ValueError, TypeError, OSError) as error:
                reason = "Invalid admission configuration: " + type(error).__name__
            if reason:
                row["_atp_admission_closed"] = reason
                break
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
        if waiting:
            _rollout_waiters[key].remove(token)
            if not _rollout_waiters[key]:
                del _rollout_waiters[key]
            waiting = False
            await asyncio.to_thread(_update_shard, campaign, model, shard_id, queue="rollout")
        yield
    finally:
        if admitted:
            _active[key] -= 1
            if not _active[key]:
                del _active[key]
        if waiting:
            _rollout_waiters[key].remove(token)
            if not _rollout_waiters[key]:
                del _rollout_waiters[key]
            await asyncio.to_thread(_update_shard, campaign, model, shard_id, queue="rollout")
