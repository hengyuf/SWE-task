#!/usr/bin/env python3
# Usage (from atp/scripts): ../runtime/controller/bin/python3 -B ../../tools/raise_probe_rollout_limit.py <probe dir> <N>
# set-concurrency only LOWERS the live rollout_limit; after a cap was lifted, this raises it (hengyuf, 2026-10-19).
"""Raise ATP's rollout_limit (actual dispatch concurrency) of a running probe campaign to `value`, capped by its limit,
the same way ATP grows it (_rollout_epoch under rate-limits/admission.lock)."""
import fcntl, json, sys, time
from pathlib import Path
sys.path.insert(0, ".")
import deepswe_admission as da
campaign, value = Path(sys.argv[1]).resolve(), int(sys.argv[2])
directory = campaign / "rate-limits"
with (directory / "admission.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    manifest, limits, _ = da._limits(campaign)
    states = da.read_json(directory / "admission.json", {})
    now = time.time(); out = {}
    for model in limits:
        current = da._state(manifest, limits[model], states.get(model), now)
        before = current["rollout_limit"]
        target = min(value, limits[model])
        if before < target:
            da._rollout_epoch(current, target, now)
            current["hold_until"] = min(current["hold_until"], now)
        states[model] = current
        out[model] = (before, current["rollout_limit"])
    da.atomic_json(directory / "admission.json", states)
    da._write_change(directory, "main_raise_rollout_limit", {"value": value, "result": out, "timestamp": now})
print(json.dumps(out))
