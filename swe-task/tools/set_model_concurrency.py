#!/usr/bin/env python3
# Usage (from atp/scripts): ../runtime/controller/bin/python3 -B ../../tools/set_model_concurrency.py <probe dir> <model> <N> [--dry-run]
# ATP's set-concurrency sets EVERY model of a campaign to one value. This sets ONE model's API limit and caps its
# rollout_limit, the same way set_concurrency does (under rate-limits/admission.lock, concurrency.json + admission.json),
# so the student's admission can be lowered while serving is thin without slowing kimi (main, 2026-10-20). Raising it
# back: run this with the higher N, then tools/raise_probe_rollout_limit.py (rollout_limit only grows through it/ATP).
"""Set one model's admission limit of a running probe campaign."""
import fcntl, json, sys, time
from pathlib import Path
sys.path.insert(0, ".")
import deepswe_admission as da

args = [a for a in sys.argv[1:] if not a.startswith("--")]
dry = "--dry-run" in sys.argv
campaign, model, value = Path(args[0]).resolve(), args[1], int(args[2])
directory = campaign / "rate-limits"
with (directory / "admission.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    manifest, prior, maximum = da._limits(campaign)
    if model not in prior:
        raise SystemExit(f"{model} is not a model of {campaign} ({sorted(prior)})")
    if not manifest["shards"] <= value <= maximum:
        raise SystemExit(f"value must be between {manifest['shards']} and {maximum}")
    now = time.time()
    states = da.read_json(directory / "admission.json", {})
    current = da._state(manifest, prior[model], states.get(model), now)
    before = {"limit": current["limit"], "rollout_limit": current["rollout_limit"]}
    limits = dict(prior, **{model: value})
    current.update(limit=value, changed_at=now, successful_calls=0, api_saturated=False,
                   hold_until=max(now, current["hold_until"]), sandbox_starts=0)
    if current["rollout_limit"] > value:
        da._rollout_epoch(current, value, now)
    after = {"limit": current["limit"], "rollout_limit": current["rollout_limit"]}
    if not dry:
        states[model] = current
        da.atomic_json(directory / "admission.json", states)
        da.atomic_json(campaign / "concurrency.json", limits)
        da._write_change(directory, "main_set_model_concurrency", {"model": model, "value": value,
                                                                    "before": before, "after": after, "timestamp": now})
print(json.dumps({"model": model, "before": before, "after": after, "concurrency": limits, "dry_run": dry}))
