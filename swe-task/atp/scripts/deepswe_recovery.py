"""One explicit, finite recovery allowance for an immutable repair source."""
from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import time

from deepswe_campaign import (atomic_json, attempts_for, cleanup_confirmed, generation_failure,
                             infrastructure_error, load_manifest, read_json, slot_directory, slot_state)


ASSESSMENT_REVISION = 7
_ALLOWANCES = {}
TERMINAL_STATES = {"BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL",
                   "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT", "REVOKED"}


def require_terminal_jobs(campaign, run=subprocess.run):
    ledger = read_json(Path(campaign) / "submission.json")
    if ledger is None:
        return
    for record in ledger["jobs"]:
        if record.get("state") != "submitted" or not record.get("job_id"):
            raise ValueError("Previous submission has an unknown outcome; reconcile Slurm first")
        job = record["job_id"]
        response = run(["sacct", "-X", "-nP", "-j", job, "--format=JobIDRaw,State"],
                       capture_output=True, text=True, check=True, timeout=30)
        states = [line.split("|")[1].split()[0].rstrip("+") for line in response.stdout.splitlines()
                  if "|" in line and line.split("|", 1)[0] == job]
        if not states or any(state not in TERMINAL_STATES for state in states):
            raise ValueError(f"Previous Slurm job {job} is live or has no confirmed terminal state")


def load_allowance(campaign, path, source=None):
    from deepswe_source import select_source

    campaign, path = Path(campaign).resolve(), Path(path).resolve()
    if path.parent != campaign / "recovery-allowances" or path.suffix != ".json":
        raise ValueError("Recovery allowance must be a campaign recovery-allowances JSON file")
    signature = (path.stat().st_size, path.stat().st_mtime_ns)
    cached = _ALLOWANCES.get(path)
    if cached is None or cached[0] != signature:
        record = read_json(path)
        if (record.get("version") != 1 or record.get("assessment_revision") != ASSESSMENT_REVISION
                or record.get("campaign") != str(campaign) or record.get("id") != path.stem
                or record.get("generation_attempts") != 3 or record.get("regrade_attempts") != 3):
            raise ValueError("Invalid or incompatible recovery allowance")
        selected = select_source(campaign, Path(record["source"]))
        if selected.name != record["id"]:
            raise ValueError("Recovery allowance does not identify its frozen repair source")
        manifest = load_manifest(campaign)
        for key, entry in record["slots"].items():
            model, task, repeat = key.split("/")
            if (model not in manifest["settings"]["models"] or not 0 <= int(task) < len(manifest["tasks"])
                    or not 0 <= int(repeat) < manifest["settings"]["repeats"]
                    or type(entry.get("baseline_attempt_count")) is not int
                    or entry["baseline_attempt_count"] < 1
                    or entry.get("kind") not in {"generation", "regrade"}):
                raise ValueError("Invalid recovery slot or baseline")
        _ALLOWANCES[path] = (signature, record)
    record = _ALLOWANCES[path][1]
    # Bind this finite allowance to the already verified source manifest, so a
    # replacement at the same path cannot inherit its recovery authorization.
    digest = hashlib.sha256((Path(record["source"]) / "source-manifest.json").read_bytes()).hexdigest()
    if digest != record.get("source_manifest_sha256"):
        raise ValueError("Recovery source manifest changed after allowance preparation")
    expected_source = source or os.environ.get("ATP_DEEPSWE_SOURCE")
    if expected_source is not None and Path(expected_source).resolve() != Path(record["source"]):
        raise ValueError("Recovery allowance belongs to a different executable source")
    return record


def prepare_allowance(campaign, source, run=subprocess.run):
    from deepswe_source import select_source

    campaign = Path(campaign).resolve()
    source = select_source(campaign, source)
    if source.parent != campaign / "source-revisions":
        raise ValueError("Recovery requires a named frozen repair source")
    path = campaign / "recovery-allowances" / (source.name + ".json")
    with (campaign / "submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(item != path for item in path.parent.glob("*.json")):
            raise ValueError("Another recovery allowance already exists; refusing automatic budget refill or source migration")
        if path.exists():
            load_allowance(campaign, path, source)
            return path
        if os.environ.get("ATP_DEEPSWE_RECOVERY_ALLOWANCE"):
            raise ValueError("Prepare a recovery allowance outside an active recovery worker")
        require_terminal_jobs(campaign, run)
        manifest = load_manifest(campaign)
        slots = {}
        for model in manifest["settings"]["models"]:
            for task in range(len(manifest["tasks"])):
                for repeat in range(manifest["settings"]["repeats"]):
                    state = slot_state(campaign, model, task, repeat, source=source)
                    if state["status"] not in {"retryable", "exhausted"}:
                        continue
                    attempts = attempts_for(slot_directory(campaign, model, task, repeat))
                    if not attempts or not all(cleanup_confirmed(item) for item in attempts):
                        continue
                    latest = attempts[-1]
                    result = read_json(latest / "terminal.json")
                    invalid = generation_failure(latest, result)
                    failure = read_json(latest / "dispatch-failure.json") or read_json(latest / "failure.json") or {}
                    approved = state.get("historical_exception")
                    if invalid in {"fatal", "unknown"} and not approved:
                        continue
                    if approved:
                        kind, reason = "generation", approved["reason"]
                    elif invalid == "infrastructure":
                        kind, reason = "generation", "generation_infrastructure"
                    elif state["regrade"] and (infrastructure_error((result or {}).get("verifier_error"))
                            or failure.get("classification") == "infrastructure"
                            or failure.get("error_type") == "CancelledError"):
                        kind, reason = "regrade", "verifier_infrastructure"
                    elif failure.get("classification") == "infrastructure":
                        kind, reason = "generation", "dispatch_infrastructure"
                    else:
                        continue
                    slots[f"{model}/{task}/{repeat}"] = {
                        "baseline_attempt_count": len(attempts), "kind": kind, "reason": reason,
                        "evidence_attempt": str(latest)}
                    if approved:
                        slots[f"{model}/{task}/{repeat}"]["historical_exception"] = approved
        atomic_json(path, {"version": 1, "id": source.name, "campaign": str(campaign),
                          "source": str(source), "assessment_revision": ASSESSMENT_REVISION,
                          "source_manifest_sha256": hashlib.sha256((source / "source-manifest.json").read_bytes()).hexdigest(),
                          "created_at": time.time(), "generation_attempts": 3,
                          "regrade_attempts": 3, "slots": slots})
    load_allowance(campaign, path, source)
    return path


def recovery_budget(campaign, model, task, repeat, attempts, regrade):
    path = os.environ.get("ATP_DEEPSWE_RECOVERY_ALLOWANCE")
    if not path:
        return None
    record = load_allowance(campaign, path)
    entry = record["slots"].get(f"{model}/{task}/{repeat}")
    if entry is None:
        return None
    kind = "regrade" if regrade else "generation"
    if entry["kind"] == "regrade" and kind != "regrade":
        return {"id": record["id"], "kind": kind, "remaining": 0}
    baseline = entry["baseline_attempt_count"]
    if len(attempts) < baseline:
        raise ValueError("Recovery attempt history shrank below its immutable baseline")
    used = 0
    regrades_used = 0
    for attempt in attempts[baseline:]:
        started = read_json(attempt / "started.json") or {}
        if started.get("recovery_allowance") != record["id"]:
            raise ValueError("Attempt after the recovery baseline has different recovery provenance")
        used += bool(started.get("regrade")) == regrade
        regrades_used += bool(started.get("regrade"))
    return {"id": record["id"], "kind": kind,
            "remaining": 0 if regrades_used >= record["regrade_attempts"] else max(0, record[kind + "_attempts"] - used)}
