"""Evidence-bound authorization for the 61 reviewed historical generations."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re


APPROVAL_FILE = "configs/deepswe_historical_recovery.json"
APPROVAL_ID = "deepswe-historical-20261015-61-slots"
EVIDENCE_FILES = (
    "started.json", "http.jsonl", "terminal.json", "dispatch-failure.json", "failure.json",
    "agent-partial-snapshot.json", "agent-partial-inventory.json",
)
_APPROVALS = {}
_VERIFIED = {}


def _signature(path):
    try:
        value = path.lstat()
    except FileNotFoundError:
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("Historical recovery evidence must be a regular file")
    return value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_ino


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(root, source):
    from deepswe_campaign import read_json

    if source is None and os.environ.get("ATP_DEEPSWE_RECOVERY_ALLOWANCE"):
        from deepswe_recovery import load_allowance
        allowance_path = Path(os.environ["ATP_DEEPSWE_RECOVERY_ALLOWANCE"])
        source = (read_json(allowance_path) or {}).get("source")
        load_allowance(root, allowance_path, source)
    source = source or os.environ.get("ATP_DEEPSWE_SOURCE")
    if source is None:
        # Status uses the currently submitted source, never a future resume pin.
        source = (read_json(root / "submission.json") or {}).get("source")
    if source is None:
        return None
    supplied = Path(source)
    source = supplied.resolve()
    path, source_manifest = source / APPROVAL_FILE, source / "source-manifest.json"
    approval_signature = _signature(path)
    if approval_signature is None:
        return None  # Original/r9 sources have no historical exception.
    if supplied.is_symlink() or (source != root / "source" and source.parent != root / "source-revisions"):
        raise ValueError("Historical approval requires this campaign's frozen source")
    source_signature = _signature(source_manifest)
    key = source, source_signature, approval_signature
    if key not in _APPROVALS:
        record = read_json(source_manifest)
        digest = _digest(path)
        if (record or {}).get("files", {}).get(APPROVAL_FILE) != digest:
            raise ValueError("Historical approval is not bound to the frozen source manifest")
        approval = read_json(path)
        entries = approval.get("entries", [])
        if (approval.get("version") != 1 or approval.get("approval_id") != APPROVAL_ID
                or approval.get("campaign") != str(root) or approval.get("slot_count") != 61
                or len(entries) != 61 or approval.get("generation_attempts") != 3
                or approval.get("regrade_attempts") != 3):
            raise ValueError("Historical approval differs from the reviewed scope or budget")
        indexed, slots, counts = {}, set(), {"fatal": 0, "unknown": 0}
        for entry in entries:
            directory = Path(entry["attempt_dir"])
            try:
                relative = directory.relative_to(root / "models/super/attempts")
            except ValueError:
                raise ValueError("Historical approval contains an unrelated attempt") from None
            if (directory != directory.resolve() or len(relative.parts) != 2
                    or not re.fullmatch(r"[0-9]+-[0-9]+", relative.parts[0])
                    or not re.fullmatch(r"attempt-[0-9]+", relative.parts[1]) or directory.parent in slots):
                raise ValueError("Historical approval contains an invalid or duplicate attempt")
            failure = entry.get("expected_failure")
            reason = {"fatal": "reviewed_historical_super_route_404",
                      "unknown": "reviewed_incomplete_generation_lost_sandbox"}.get(failure)
            evidence = entry.get("evidence", {})
            if (reason is None or entry.get("reason") != reason or set(evidence) != set(EVIDENCE_FILES)
                    or any(value is not None and not re.fullmatch(r"[0-9a-f]{64}", value)
                           for value in evidence.values())
                    or not evidence.get("started.json")):
                raise ValueError("Historical approval has incomplete evidence or an unreviewed reason")
            indexed[directory] = entry
            slots.add(directory.parent)
            counts[failure] += 1
        if counts != {"fatal": 60, "unknown": 1}:
            raise ValueError("Historical approval does not contain the reviewed 60 route failures and one lost sandbox")
        if (_signature(source_manifest), _signature(path)) != (source_signature, approval_signature):
            raise ValueError("Historical approval changed during validation")
        _APPROVALS[key] = indexed, digest
    return _APPROVALS[key]


def historical_exception(campaign, directory, failure, attempts, *, source=None):
    """Return authorization only for the exact unchanged latest baseline attempt."""
    from deepswe_campaign import cleanup_confirmed, read_json

    if failure not in {"fatal", "unknown"}:
        return None
    root, directory = Path(campaign).resolve(), Path(directory).resolve()
    approval = _load(root, source)
    if approval is None:
        return None
    entries, approval_digest = approval
    entry = entries.get(directory)
    if entry is None or entry["expected_failure"] != failure:
        return None
    if not attempts or directory != attempts[-1] or not all(cleanup_confirmed(item) for item in attempts):
        return None
    started = read_json(directory / "started.json") or {}
    if started.get("regrade") or started.get("agent_artifacts_from"):
        return None  # Historical patches are never authorized for verifier-only recovery.
    signatures = tuple(_signature(directory / name) for name in EVIDENCE_FILES)
    key = directory, approval_digest, signatures
    if key not in _VERIFIED:
        for name, signature in zip(EVIDENCE_FILES, signatures):
            expected = entry["evidence"][name]
            if (signature is None) != (expected is None) or (signature is not None and _digest(directory / name) != expected):
                raise ValueError(f"Historical recovery evidence changed: {directory / name}")
        if tuple(_signature(directory / name) for name in EVIDENCE_FILES) != signatures:
            raise ValueError("Historical recovery evidence changed during hashing")
        _VERIFIED[key] = True
    return {"approval_id": APPROVAL_ID, "approval_sha256": approval_digest,
            "evidence_attempt": str(directory), "reason": entry["reason"]}
