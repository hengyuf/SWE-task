"""Verify tool-call delivery against durable OpenCode records, including child sessions."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tarfile
import tempfile


DEADLINE_DRAIN_SECONDS = 15


def _timestamp(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _snapshot_tool_ids(source):
    """Read only a consistent backup; never open a live database or one on Lustre."""
    marker = json.loads((source / "agent-final-snapshot.json").read_text())
    if not all(marker.get(key) is True for key in ("consistent_database", "all_sessions", "final")):
        raise ValueError("Final snapshot does not certify a consistent all-session database")
    captured_at = _timestamp(marker.get("captured_at"))
    if captured_at is None:
        raise ValueError("Final snapshot timestamp is unavailable")
    scratch = Path("/raid/scratch")
    if not scratch.is_dir() or not os.access(scratch, os.W_OK):
        scratch = Path("/tmp")
    with tempfile.TemporaryDirectory(prefix="deepswe-delivery-", dir=scratch) as temporary:
        database = Path(temporary) / "opencode.db"
        with tarfile.open(source / "opencode-data.tar.gz", "r|gz") as archive:
            for member in archive:
                if member.name not in {"snapshot/opencode.db", "./snapshot/opencode.db"}:
                    continue
                if not member.isfile() or member.size <= 0:
                    raise ValueError("Snapshot database is not a regular nonempty file")
                with archive.extractfile(member) as stream, database.open("xb") as output:
                    shutil.copyfileobj(stream, output)
                break
            else:
                raise ValueError("Snapshot archive has no consistent database")
        calls = set()
        connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
        try:
            # Check the complete session schema rather than a single exported conversation.
            connection.execute("select id, parent_id from session limit 0")
            rows = connection.execute(
                "select p.data, m.data from part p left join message m on m.id = p.message_id")
            for part_data, message_data in rows:
                part = json.loads(part_data)
                message = json.loads(message_data) if message_data is not None else {}
                if not isinstance(part, dict) or not isinstance(message, dict):
                    raise ValueError("Malformed OpenCode database record")
                identity = part.get("callID")
                if part.get("type") == "tool" and message.get("role") == "assistant" and isinstance(identity, str):
                    calls.add(identity)
        finally:
            connection.close()
    return calls, captured_at


def audit_tool_delivery(source, responses, later_tool_results, deadline):
    """Missing evidence is unknown; only a complete, sufficiently late snapshot proves loss."""
    pending = []
    required = 0
    for call_id, call in responses.items():
        for tool_id in call.get("tool_call_ids", []):
            required += 1
            if later_tool_results.get(tool_id, -1) <= call["response_sequence"]:
                pending.append((call_id, tool_id, call.get("response_timestamp")))
    report = {"required_tool_calls": required, "missing_tool_calls": [], "unverified_tool_calls": [],
              "status": "verified" if required else "not_applicable"}
    if not pending:
        return report
    try:
        observed, captured_at = _snapshot_tool_ids(Path(source))
        report["snapshot_captured_at"] = captured_at
    except (OSError, ValueError, TypeError, EOFError, sqlite3.Error, tarfile.TarError) as error:
        report.update(status="unknown", reason=type(error).__name__ + ": " + str(error),
                      unverified_tool_calls=[{"call_id": call, "tool_call_id": tool} for call, tool, _ in pending])
        return report
    for call_id, tool_id, response_at in pending:
        if tool_id in observed:
            continue
        record = {"call_id": call_id, "tool_call_id": tool_id, "response_timestamp": response_at}
        response_at = _timestamp(response_at)
        budget_end = _timestamp(deadline)
        if (response_at is None or budget_end is None or captured_at < response_at
                or response_at > budget_end - DEADLINE_DRAIN_SECONDS):
            report["unverified_tool_calls"].append(record)
        else:
            report["missing_tool_calls"].append(record)
    if report["missing_tool_calls"]:
        report["status"] = "missing"
    elif report["unverified_tool_calls"]:
        report["status"] = "unknown"
    return report
