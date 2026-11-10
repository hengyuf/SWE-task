"""Durable DeepSWE slots and attempts. Plain files are the source of truth."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
import uuid


TASK = "_ng_task_index"
REPEAT = "_ng_rollout_index"
_MANIFESTS = {}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def read_jsonl(path):
    """A trailing torn write is preserved verbatim; interior corruption is fatal."""
    path = Path(path)
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("JSONL rows must be objects")
            rows.append(row)
        except (ValueError, UnicodeError):
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError(f"Corrupt JSONL row {index + 1} in {path}") from None
            tail = path.with_name(path.name + ".interrupted-tail")
            if tail.exists() and tail.read_bytes() != line:
                raise ValueError(f"Conflicting interrupted tail in {path}")
            if not tail.exists():
                with tail.open("xb") as stream:
                    stream.write(line)
                    stream.flush()
                    os.fsync(stream.fileno())
    return rows


def load_manifest(campaign):
    path = Path(campaign).resolve() / "manifest.json"
    if path not in _MANIFESTS:
        value = read_json(path)
        if value is None:
            return None
        _MANIFESTS[path] = value
    return _MANIFESTS[path]


def create_campaign(path, dataset, settings, shards=10, task_limit=None, repeats=None):
    path = Path(path).resolve()
    settings = dict(settings)
    tasks = read_jsonl(dataset)
    if len(tasks) != settings["expected_tasks"]:
        raise ValueError(f"Expected {settings['expected_tasks']} native tasks, found {len(tasks)}")
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("DeepSWE task IDs must be unique")
    if task_limit is not None:
        if not 1 <= task_limit <= len(tasks):
            raise ValueError("Invalid task limit")
        tasks = tasks[:task_limit]
    if repeats is not None:
        settings["repeats"] = repeats
    if not 1 <= settings["repeats"] <= 16 or not 1 <= shards <= 10:
        raise ValueError("Campaign supports 1..16 repeats and 1..10 task shards")
    tasks = [dict(row, **{TASK: index}) for index, row in enumerate(tasks)]
    requested = {"version": 1, "settings": settings, "shards": shards, "tasks": tasks}
    existing = load_manifest(path)
    if existing:
        if {key: existing[key] for key in requested} != requested:
            raise ValueError("Campaign already exists with different immutable inputs/settings")
        return existing
    path.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects against two creators without relying on rename overwrite.
    with (path / "create.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (path / "manifest.json").exists():
            return create_campaign(path, dataset, settings, shards, task_limit, repeats)
        requested["created_at"] = time.time()
        atomic_json(path / "manifest.json", requested)
    return requested


@contextmanager
def shard_lock(campaign, model, shard_id):
    directory = Path(campaign) / "locks"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{model}-{shard_id}.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another worker owns {model} shard {shard_id}") from None
        yield


def write_shard(campaign, shard_id, destination):
    manifest = load_manifest(campaign)
    if not 0 <= shard_id < manifest["shards"]:
        raise ValueError("Invalid shard ID")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as stream:
        for row in manifest["tasks"]:
            if row[TASK] % manifest["shards"] == shard_id:
                stream.write(json.dumps(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return destination


def slot_directory(campaign, model, task, repeat):
    if model not in load_manifest(campaign)["settings"]["models"]:
        raise ValueError("Unknown model alias")
    return Path(campaign) / "models" / model / "attempts" / f"{int(task)}-{int(repeat)}"


def attempts_for(directory):
    return sorted((p for p in Path(directory).glob("attempt-*") if p.is_dir()),
                  key=lambda p: int(p.name.split("-")[-1]))


def outcome(result, directory=None):
    if result and result.get("evaluation_completed") is True and (
            directory is None or generation_failure(directory, result) is None):
        return "completed"
    return "incomplete"


def finalized_candidate(directory):
    # Only the normal verifier collect hook saves a finalized candidate. Recovery
    # may save a diagnostic patch, which must never silently become a new candidate.
    events = Path(directory) / "capture-events.jsonl"
    return (Path(directory) / "model.patch").is_file() and events.exists() and any(
        event.get("event") == "candidate_saved" for event in read_jsonl(events))


def cleanup_confirmed(directory):
    marker = read_json(Path(directory) / "cleanup-confirmed.json") or {}
    return marker.get("report", {}).get("remaining_sandboxes") == []


def _journal_signature(path):
    try:
        metadata = Path(path).stat()
    except FileNotFoundError:
        return None
    return {"size_bytes": metadata.st_size, "mtime_ns": metadata.st_mtime_ns}


def _generation_signatures(source):
    return {name: _journal_signature(source / name) for name in (
        "http.jsonl", "stream-events.jsonl", "capture-events.jsonl", "started.json", "verify-request.json",
        "terminal.json", "export.json", "agent-final-snapshot.json", "opencode-data.tar.gz",
        "generation-quiescence.json", "cleanup-confirmed.json", "interrupted.json", "failure.json",
        "stream-writer-error.json", "replay-prevented.log")}


def generation_failure(directory, result=None):
    """A verifier grade does not make an infrastructure-aborted generation valid."""
    directory = Path(directory)
    started = read_json(directory / "started.json") or {}
    source = Path(started.get("agent_artifacts_from", str(directory)))
    events = read_jsonl(source / "capture-events.jsonl") if (source / "capture-events.jsonl").exists() else []
    interrupted = read_json(source / "interrupted.json") or {}
    candidate_finalized = finalized_candidate(source)
    owner_interrupted = interrupted.get("reason") == "owner_job_finished" and not candidate_finalized
    quiescence = read_json(source / "generation-quiescence.json") or {}
    if quiescence.get("confirmed") is False or any(event.get("event") == "generation_not_quiesced" for event in events):
        # Recovery can make regeneration safe; it cannot finalize the interrupted patch.
        return "infrastructure" if cleanup_confirmed(source) else "unknown"
    interruptions = [event for event in events if event.get("event") == "agent_interrupted"]
    failure = read_json(source / "failure.json") or {}
    cancelled = any(event.get("error_type") == "cancelled" for event in interruptions) or (
        failure.get("error_type") == "CancelledError" and not candidate_finalized)
    if cancelled:
        return "infrastructure" if cleanup_confirmed(source) else "unknown"
    if any(event.get("error_type") == "SandboxPtyError" for event in interruptions):
        return "infrastructure" if quiescence.get("confirmed") or cleanup_confirmed(source) else "unknown"
    if (source / "replay-prevented.log").exists() or any(
            event.get("event") == "agent_command_replay_prevented" for event in events):
        return "infrastructure" if quiescence.get("confirmed") or cleanup_confirmed(source) else "unknown"
    writer_error = read_json(source / "stream-writer-error.json") or {}
    if any(writer_error.get(key) not in {None, 0} for key in ("stdout_exit_code", "stderr_exit_code")) or any(
            event.get("event") == "agent_stream_writer_error" for event in events):
        return "infrastructure" if quiescence.get("confirmed") or cleanup_confirmed(source) else "unknown"
    streams = read_jsonl(source / "stream-events.jsonl") if (source / "stream-events.jsonl").exists() else []
    stream_errors = any(event.get("event") == "stream_error" for event in streams)
    stream_cancelled = any(event.get("event") == "stream_cancelled" for event in streams)
    # Preserve older assessments. Every input that can change delivery or validity
    # participates in this cache, including a newly recovered final snapshot.
    journal = source / "http.jsonl"
    journal_signature = _journal_signature(journal)
    signatures = _generation_signatures(source)
    source_start = read_json(source / "started.json") or {}
    flags = (result or read_json(directory / "terminal.json") or {}) | (read_json(source / "verify-request.json") or {})
    flag_signature = {key: flags.get(key) for key in (
        "opencode_run_error_type", "opencode_exit_code", "opencode_finished", "opencode_export_found") if key in flags}
    cached = read_json(source / "generation-validity-v2-r7.json")
    if (cached is not None and cached.get("version") == 2 and cached.get("assessment_revision") == 7
            and cached.get("evidence_signatures") == signatures and cached.get("agent_flags") == flag_signature
            and cached.get("candidate_finalized", candidate_finalized) == candidate_finalized
            and (interrupted.get("reason") != "owner_job_finished" or "candidate_finalized" in cached)):
        return cached["failure"]
    deadline = source_start.get("deadline")
    budget_timeout = any(event.get("error_type") == "timeout" and deadline is not None and
                         event.get("time", 0) >= deadline - 1 for event in interruptions)
    calls = {}
    model_calls = {}
    later_tool_results = {}
    upstream_nonempty = False
    if journal.exists():
        from deepswe_http import malformed_success
        # Keep only call outcomes in memory; full prompts/responses stay in the journal.
        with journal.open("rb") as stream:
            for sequence, line in enumerate(stream):
                try:
                    event = json.loads(line)
                except ValueError:
                    calls["torn_event"] = {"event": "error", "error_type": "IncompleteHTTPJournal"}
                    break
                if event.get("scope") == "native_model_method":
                    model_calls[event.get("model_call_id")] = {
                        key: event[key] for key in ("event", "error_type", "error", "timestamp", "http_attempt", "outcome") if key in event}
                    continue
                if event.get("event") in {"api_gate_wait_started", "api_gate_acquired", "upstream_stream_chunk"}:
                    continue
                call = calls.setdefault(event.get("call_id"), {})
                call.update({key: event[key] for key in ("event", "status", "error_type", "error", "timestamp",
                             "retryable_gateway_capacity", "malformed_success", "model_call_id", "stream_id",
                             "http_attempt", "outcome", "buffered_upstream_failure") if key in event})
                call["sequence"] = sequence
                if event.get("event") == "request" and isinstance(event.get("payload"), dict):
                    canonical = json.dumps(event["payload"], sort_keys=True, separators=(",", ":"))
                    call["payload_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
                    for message in event["payload"].get("messages", []):
                        if isinstance(message, dict) and message.get("role") == "tool" and isinstance(message.get("tool_call_id"), str):
                            later_tool_results[message["tool_call_id"]] = sequence
                if event.get("event") == "response":
                    call.pop("error_type", None)
                    call.pop("error", None)
                    call.pop("outcome", None)
                    call.pop("buffered_upstream_failure", None)
                    call.pop("tool_call_ids", None)
                    # A later valid response clears an earlier malformed retry,
                    # including historical journals without the explicit marker.
                    call["malformed_success"] = event.get("malformed_success") is True
                    if event.get("status") == 200:
                        try:
                            response = json.loads(event.get("body", "{}"))
                            call["malformed_success"] |= malformed_success(response)
                            call["tool_call_ids"] = [tool["id"] for choice in response.get("choices", [])
                                if isinstance(choice, dict) and isinstance(choice.get("message"), dict)
                                for tool in choice["message"].get("tool_calls") or []
                                if isinstance(tool, dict) and isinstance(tool.get("id"), str) and tool["id"]]
                            call["response_sequence"] = sequence
                            call["response_timestamp"] = event.get("timestamp")
                            upstream_nonempty |= any(isinstance(choice, dict) and any(
                                (choice.get("message") or {}).get(key) for key in
                                ("content", "reasoning_content", "reasoning", "tool_calls"))
                                for choice in response.get("choices", []))
                        except (ValueError, TypeError, AttributeError):
                            pass
                elif event.get("event") == "rate_limit_wait":
                    call["status"] = 429
                if (event.get("event") == "error" and event.get("error_type") == "TimeoutError"
                        and event.get("outcome") != "not_sent" and deadline is not None):
                    budget_timeout |= event.get("timestamp", 0) >= deadline - 1
    # OpenCode can retry a failed native call itself. Only an identical request
    # succeeding later supersedes that failure; unrelated successful turns cannot.
    successful = {}
    for call in calls.values():
        native = model_calls.get(call.get("model_call_id"))
        if (call.get("payload_hash") and call.get("event") == "response" and call.get("status") == 200
                and not call.get("malformed_success") and (
                not call.get("model_call_id") or (native and native.get("event") == "model_call_completed"))):
            digest = call["payload_hash"]
            successful[digest] = max(successful.get(digest, -1), call["sequence"])
    recovered = {identity for identity, call in calls.items() if call.get("payload_hash") and
                 successful.get(call["payload_hash"], -1) > call.get("sequence", -1)}
    recovered_streams = {calls[identity].get("stream_id") for identity in recovered} - {None}
    unresolved = {identity: call for identity, call in calls.items() if identity not in recovered}
    recovered_models = {calls[identity].get("model_call_id") for identity in recovered} - {
        call.get("model_call_id") for call in unresolved.values()} - {None}
    native_errors = [call for identity, call in model_calls.items() if identity not in recovered_models and
                     call.get("event") == "model_call_error"]
    stream_errors = any(event.get("event") == "stream_error" and event.get("stream_id") not in recovered_streams
                        for event in streams)
    stream_cancelled = any(event.get("event") == "stream_cancelled" and event.get("stream_id") not in recovered_streams
                           for event in streams)
    deadline_stream_cancellation = budget_timeout and deadline is not None and all(
        event.get("timestamp", 0) >= deadline - 1 for event in streams
        if event.get("event") == "stream_cancelled" and event.get("stream_id") not in recovered_streams)
    export_progress = None
    if budget_timeout and (upstream_nonempty or stream_cancelled) and (source / "export.json").is_file():
        export = read_json(source / "export.json")
        if isinstance(export, dict) and isinstance(export.get("messages"), list) and export["messages"]:
            export_progress = any(
                part.get("type") == "tool" or
                (part.get("type") in {"text", "reasoning"} and bool(part.get("text")))
                for message in export["messages"] if message.get("info", {}).get("role") == "assistant"
                for part in message.get("parts", []))
    stream_errors = any(event.get("event") == "stream_error" and event.get("stream_id") not in recovered_streams
        and not (budget_timeout and export_progress and deadline is not None and
                 event.get("error_type") == "TimeoutError" and event.get("timestamp", 0) >= deadline - 1)
        for event in streams)
    failure = None
    if any(call.get("status") in {400, 401, 403, 404, 422} and not call.get("retryable_gateway_capacity")
           for call in unresolved.values()):
        failure = "fatal"
    elif any(call.get("malformed_success") or call.get("retryable_gateway_capacity") or call.get("buffered_upstream_failure") or
             call.get("error_type") == "LocalAdmissionDeadline" or
             (call.get("error_type") == "TimeoutError" and call.get("outcome") == "not_sent") or
             call.get("status") in {408, 429, 500, 502, 503, 504, 520} or
             infrastructure_error(str(call.get("error_type", "")) + ": " + str(call.get("error", "")))
             for call in unresolved.values()):
        failure = "infrastructure"
    elif any(event.get("error_type") == "cancelled" or infrastructure_error(event.get("error_type", ""))
             for event in interruptions):
        failure = "infrastructure"
    elif owner_interrupted:
        failure = "infrastructure" if cleanup_confirmed(source) else "unknown"
    elif any(call.get("error_type") in {"MalformedUpstreamResponse", "UpstreamStreamError", "LocalAdmissionDeadline"}
             or (call.get("error_type") == "TimeoutError" and call.get("outcome") == "not_sent")
             for call in native_errors) or any(
             event.get("event") == "stream_error" and event.get("stream_id") not in recovered_streams
             and event.get("error_type") in {"MalformedUpstreamResponse", "UpstreamStreamError", "LocalAdmissionDeadline"}
             for event in streams):
        failure = "infrastructure"
    elif any(not (budget_timeout and deadline is not None and call.get("timestamp", 0) >= deadline - 1 and (
            call.get("error_type") == "TimeoutError" or
            (call.get("error_type") == "CancelledError" and deadline_stream_cancellation and export_progress)))
            for call in native_errors):
        failure = "unknown"
    elif stream_errors:
        failure = "unknown"
    elif stream_cancelled and not (deadline_stream_cancellation and export_progress):
        failure = "infrastructure" if cleanup_confirmed(source) else "unknown"
    elif (upstream_nonempty and export_progress is False
            and not any(call.get("tool_call_ids") for call in unresolved.values())):
        # A successful upstream response is not proof that OpenCode received it.
        # Regenerate this empty delivery only after cleanup; never regrade its patch.
        failure = "infrastructure" if cleanup_confirmed(source) else "unknown"
    elif any(call.get("event") in {"request", "error", "cancelled"} for call in unresolved.values()) and not budget_timeout:
        failure = "unknown"
    elif not budget_timeout and (flags.get("opencode_run_error_type") or
            flags.get("opencode_exit_code") not in {None, 0} or flags.get("opencode_finished") is False):
        failure = "unknown"
    elif not calls and any(key in flags for key in ("opencode_finished", "opencode_exit_code", "opencode_run_error_type")):
        failure = "unknown"
    delivery = {"status": "not_checked"}
    if failure in {None, "unknown"}:
        from deepswe_delivery import audit_tool_delivery
        delivery = audit_tool_delivery(source, {
            identity: call for identity, call in unresolved.items()
            if call.get("event") == "response" and call.get("status") == 200 and not call.get("malformed_success")
        }, later_tool_results, deadline)
        if delivery["status"] == "missing":
            failure = "infrastructure"
        elif delivery["status"] == "unknown":
            failure = "unknown"
    # Candidate collection happens after agent shutdown. A recovered, stopped
    # owner can also leave a permanently open final call; its cleanup and every
    # journal remain signature-bound, so late evidence still invalidates the cache.
    calls_closed = all(call.get("event") in {"error", "cancelled", "retry_exhausted"} or (
                       call.get("event") == "response" and
                       not call.get("retryable_gateway_capacity") and
                       call.get("status") not in {408, 429, 500, 502, 503, 504, 520}) for call in unresolved.values())
    cacheable = (calls_closed and (candidate_finalized or (source / "terminal.json").exists())) or (
        owner_interrupted and cleanup_confirmed(source))
    if (cacheable and _generation_signatures(source) == signatures
            and finalized_candidate(source) == candidate_finalized):
        atomic_json(source / "generation-validity-v2-r7.json", {"version": 2, "assessment_revision": 7, "failure": failure,
                    "assessed_at": time.time(), "budget_timeout": budget_timeout, "deadline": deadline,
                    "http_journal_signature": journal_signature,
                    "evidence_signatures": signatures, "agent_flags": flag_signature,
                    "candidate_finalized": candidate_finalized,
                    "logical_calls": len(calls), "source": str(source),
                    "unresolved_http_outcomes": [dict(call_id=str(identity), **{
                        key: call[key] for key in ("event", "http_attempt", "outcome", "error_type", "status",
                                                  "buffered_upstream_failure") if key in call})
                        for identity, call in unresolved.items() if call.get("event") != "response"],
                    "upstream_nonempty": upstream_nonempty, "export_assistant_progress": export_progress,
                    "delivery_gap": (upstream_nonempty and export_progress is False
                                     and delivery["status"] in {"not_checked", "not_applicable"})
                                    or delivery["status"] == "missing",
                    "tool_delivery": delivery,
                    "stream_errors": stream_errors, "stream_cancelled": stream_cancelled,
                    "deadline_stream_cancellation": deadline_stream_cancellation,
                    "recovered_calls": sorted(str(identity) for identity in recovered),
                    "malformed_success_calls": sorted(str(identity) for identity, call in unresolved.items()
                                                      if call.get("malformed_success")),
                    "recovered_streams": sorted(recovered_streams),
                    "native_model_errors": [call.get("error_type") for call in native_errors]})
    return failure


def slot_state(campaign, model, task, repeat, *, source=None):
    directory = slot_directory(campaign, model, task, repeat)
    attempts = attempts_for(directory)
    state = {"task_index": task, "repeat_index": repeat, "attempts": len(attempts),
             "status": "pending", "retryable": True, "attempt_dir": None, "regrade": False}
    if not attempts:
        return state
    latest = attempts[-1]
    state["attempt_dir"] = str(latest)
    for attempt in attempts:
        terminal = read_json(attempt / "terminal.json")
        if outcome(terminal, attempt) == "completed":
            state.update(status="completed", retryable=False, attempt_dir=str(attempt))
            return state
    if not (latest / "started.json").exists():
        state.update(status="needs_review", retryable=False)
        return state
    terminal = read_json(latest / "terminal.json")
    failure = read_json(latest / "dispatch-failure.json") or read_json(latest / "failure.json") or {}
    interrupted = read_json(latest / "interrupted.json")
    candidate = finalized_candidate(latest) or bool(read_json(latest / "started.json").get("regrade"))
    invalid_generation = generation_failure(latest, terminal) if terminal or candidate or failure or interrupted else None
    state["regrade"] = candidate and invalid_generation is None
    from deepswe_historical import historical_exception
    approved = historical_exception(campaign, latest, invalid_generation, attempts, source=source)
    if approved:
        state.update(status="retryable", retryable=True, regrade=False, historical_exception=approved)
    elif not terminal and not failure and not interrupted:
        state.update(status="active", retryable=False)
    elif failure.get("classification") == "fatal" or invalid_generation == "fatal":
        state.update(status="fatal", retryable=False)
    elif invalid_generation == "infrastructure":
        state.update(status="retryable", retryable=True)
    elif invalid_generation == "unknown":
        state.update(status="needs_review", retryable=False)
    elif failure.get("error_type") == "CancelledError":
        state.update(status="retryable" if cleanup_confirmed(latest) else "needs_review",
                     retryable=cleanup_confirmed(latest))
    elif terminal and infrastructure_error(terminal.get("verifier_error", "")) and state["regrade"]:
        state.update(status="retryable", retryable=True)
    elif interrupted or failure.get("classification") == "infrastructure":
        state.update(status="retryable", retryable=True)
    else:
        state.update(status="needs_review", retryable=False)
    if state["status"] == "retryable":
        from deepswe_recovery import recovery_budget
        budget = recovery_budget(campaign, model, task, repeat, attempts, state["regrade"])
        if budget is not None:
            state["recovery"] = budget
            if not budget["remaining"]:
                state.update(status="exhausted", retryable=False)
        elif len(attempts) >= load_manifest(campaign)["settings"]["max_attempts"]:
            state.update(status="exhausted", retryable=False)
    return state


def infrastructure_error(message):
    # A vanished OpenSandbox command cannot yield a verifier grade. Keep this
    # specific to command-status polling; a generic HTTP 410 is not retry proof.
    if re.search(r"\bSandboxApiException: Get command status failed: HTTP 410(?:\s*\||$)", message or ""):
        return True
    return any(marker in (message or "") for marker in (
        "ConnectionError", "ConnectionResetError", "ConnectError", "ConnectTimeout", "ReadTimeout",
        "ClientConnectionError", "ServerDisconnectedError", "RateLimitDeadline", "HTTP 429",
        "ClientOSError", "RemoteProtocolError", "ConnectionClosedError", "WebSocketDisconnect",
        "MalformedUpstreamResponse", "UpstreamStreamError",
        "LocalAdmissionDeadline",
        "408 Request Timeout", "HTTP 408", "429 Too Many Requests", "502 Bad Gateway",
        "503 Service Unavailable", "504 Gateway Timeout",
    ))


def pending_slots(campaign, model, shard_id):
    manifest = load_manifest(campaign)
    return [state for task in manifest["tasks"] if task[TASK] % manifest["shards"] == shard_id
            for repeat in range(manifest["settings"]["repeats"])
            if (state := slot_state(campaign, model, task[TASK], repeat))["retryable"]]


def _environment():
    return Path(os.environ["ATP_DEEPSWE_CAMPAIGN"]), os.environ["ATP_DEEPSWE_MODEL_ALIAS"]


def filter_dispatch_rows(rows):
    if not os.environ.get("ATP_DEEPSWE_CAMPAIGN"):
        return rows
    campaign, model = _environment()
    return [row for row in rows if slot_state(campaign, model, row[TASK], row[REPEAT])["retryable"]]


class AdmissionClosed(RuntimeError):
    pass


def admission_closed(reason):
    atomic_json(Path(os.environ["ATP_DEEPSWE_LAUNCH_DIR"]) / "admission-closed.json",
                {"time": time.time(), "reason": reason})


def prepare_dispatch(row):
    if not os.environ.get("ATP_DEEPSWE_CAMPAIGN"):
        return
    if row.get("_atp_admission_closed"):
        raise AdmissionClosed(row["_atp_admission_closed"])
    campaign, model = _environment()
    settings = load_manifest(campaign)["settings"]
    state = slot_state(campaign, model, row[TASK], row[REPEAT])
    if not state["retryable"]:
        raise AdmissionClosed("Slot is no longer eligible for dispatch")
    if state.get("historical_exception") and not state.get("recovery"):
        raise AdmissionClosed("Historical recovery requires its finite source-bound allowance")
    if state.get("recovery") and state["attempt_dir"] and not cleanup_confirmed(Path(state["attempt_dir"])):
        raise AdmissionClosed("Recovery requires confirmed cleanup before another attempt")
    now = time.time()
    agent_timeout = settings["agent_timeout_seconds"]
    # Native verifier limit may vary by task; staging writes the actual timeout
    # into the model-visible row solely as dispatcher metadata.
    verifier_timeout = row.get("_atp_verifier_timeout", 3600)
    budget = (0 if state["regrade"] else agent_timeout) + verifier_timeout + settings["shutdown_grace_seconds"]
    if now + budget >= float(os.environ.get("ATP_DEEPSWE_JOB_DEADLINE", "inf")):
        raise AdmissionClosed("Insufficient allocation time for a complete attempt")
    attempt = slot_directory(campaign, model, row[TASK], row[REPEAT]) / f"attempt-{state['attempts']}"
    attempt.mkdir(parents=True, exist_ok=False)
    previous = Path(state["attempt_dir"]) if state["attempt_dir"] else None
    record = {"started_at": now, "job_id": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(),
              "pid": os.getpid(), "task_index": row[TASK], "repeat_index": row[REPEAT],
              "attempt_index": state["attempts"], "model": model, "regrade": state["regrade"],
              "source": os.environ.get("ATP_DEEPSWE_SOURCE"),
              "launch_dir": os.environ.get("ATP_DEEPSWE_LAUNCH_DIR"), "request": row.copy(),
              "deadline": now + (0 if state["regrade"] else agent_timeout)}
    if state.get("recovery"):
        record.update(recovery_allowance=state["recovery"]["id"], recovery_kind=state["recovery"]["kind"])
    if state.get("historical_exception"):
        record["historical_exception"] = state["historical_exception"]
    if state["regrade"]:
        previous_start = read_json(previous / "started.json")
        record["agent_artifacts_from"] = previous_start.get("agent_artifacts_from", str(previous))
        shutil.copyfile(previous / "model.patch", attempt / "model.patch")
        with (attempt / "model.patch").open("rb") as stream:
            os.fsync(stream.fileno())
        source = Path(record["agent_artifacts_from"])
        previous_result = ((read_json(previous / "terminal.json") or {}) |
                           (read_json(previous / "verify-request.json") or
                            read_json(source / "verify-request.json") or {}))
        row.update({key: value for key, value in previous_result.items()
                    if key in {"response", "opencode_exit_code", "opencode_run_error_type", "sandbox_handle",
                               "opencode_finished", "opencode_export_found"}})
        row["_atp_regrade"] = True
        row["_atp_grade_attempt"] = state["attempts"]
        # Verification itself has no model-call deadline, but schema requires a positive timestamp.
        record["deadline"] = now + verifier_timeout
    atomic_json(attempt / "started.json", record)
    row.update(_atp_attempt_dir=str(attempt), _atp_deadline=record["deadline"],
               _ng_attempt_index=state["attempts"])
    atomic_json(attempt / "request.json", row)


def dispatch_finished(row, result):
    if not row.get("_atp_attempt_dir"):
        return
    directory = Path(row["_atp_attempt_dir"])
    existing = read_json(directory / "terminal.json") or {}
    atomic_json(directory / "terminal.json", existing | result | {TASK: row[TASK], REPEAT: row[REPEAT],
                "_ng_attempt_index": row["_ng_attempt_index"], "_atp_attempt_dir": str(directory)})
    if outcome(result, directory) == "completed":
        from deepswe_admission import note_rollout_completed
        note_rollout_completed(row)
    elif infrastructure_error(result.get("verifier_error", "")):
        from deepswe_admission import note_health
        note_health(os.environ["ATP_DEEPSWE_MODEL_ALIAS"], "sandbox_failed")


def persist_native_result(row, result):
    if row.get("_atp_attempt_dir"):
        atomic_json(Path(row["_atp_attempt_dir"]) / "native-result.json", result)


def dispatch_failed(row, error, status=None):
    if not row.get("_atp_attempt_dir"):
        return
    from deepswe_http import redact
    response_content = getattr(error, "response_content", None)
    if isinstance(response_content, bytes):
        response_content = response_content.decode("utf-8", errors="replace")
    detail = f"{type(error).__name__}: {error}; {response_content or ''}"
    upstream_pressure = None
    for code, reason, health in ((408, "Request Timeout", "upstream_timeout"),
                                 (429, "Too Many Requests", "rate_limit")):
        if status == code or re.search(
                rf"\bHTTP\s+{code}\b|\b{code} {reason}\b|[\"'](?:code|status|status_code)[\"']\s*:\s*[\"']?{code}\b",
                detail, re.IGNORECASE) or (code == 429 and "RateLimitDeadline" in detail):
            upstream_pressure = health
            break
    classification = "unknown"
    if status in {400, 401, 403, 404, 422}:
        classification = "fatal"
    elif upstream_pressure or status in {502, 503, 504, 520} or isinstance(error, (TimeoutError, ConnectionError)) or (
            isinstance(error, OSError) and error.errno in {errno.ECONNRESET, errno.ECONNREFUSED, errno.ENETDOWN,
                errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ETIMEDOUT, errno.EPIPE, errno.ECONNABORTED}) or (
            infrastructure_error(detail)):
        classification = "infrastructure"
    atomic_json(Path(row["_atp_attempt_dir"]) / "dispatch-failure.json", redact({
        "time": time.time(), "error_type": type(error).__name__, "error": str(error),
        "http_status": status, "classification": classification, "response_content": response_content,
    }))
    # Drain work already admitted before recovering remote state. A transport
    # failure does not prove that the remote execution has stopped.
    admission_closed("Dispatch failed; drain and recover before more admission")
    if classification == "infrastructure":
        from deepswe_admission import note_health
        note_health(os.environ["ATP_DEEPSWE_MODEL_ALIAS"], upstream_pressure or "sandbox_failed")


def _owner_finished(started):
    job = started.get("job_id")
    if not job:
        return False
    # A lock by itself is insufficient: Gym descendants can outlive the collector.
    if str(job) == os.environ.get("SLURM_JOB_ID"):
        stopped = json.loads(os.environ.get("ATP_DEEPSWE_STOPPED_LAUNCH_DIRS", "[]"))
        stopped.append(os.environ.get("ATP_DEEPSWE_STOPPED_LAUNCH_DIR"))
        return bool(started.get("launch_dir") and started["launch_dir"] in stopped)
    result = subprocess.run(["sacct", "-X", "-nP", "-j", str(job), "--format=JobIDRaw,State"],
                            text=True, capture_output=True, check=True, timeout=30)
    states = [line.split("|")[1].split()[0].rstrip("+") for line in result.stdout.splitlines()
              if line.split("|")[0] == str(job)]
    return bool(states) and all(state in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
                                        "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE"} for state in states)


async def recover_incomplete(campaign, model, shard_id, full_resolved_config, *, concurrency=4, per_attempt_timeout=90):
    from deepswe_capture import recover_attempt
    manifest = load_manifest(campaign)
    progress = {"reports": [], "remaining": [], "errors": [], "attempted": 0, "recovered": 0, "already_clean": 0}
    if concurrency < 1 or per_attempt_timeout <= 0:
        raise ValueError("Recovery concurrency and per-attempt timeout must be positive")
    semaphore = asyncio.Semaphore(concurrency)
    owner_checks = {}
    owner_locks = {}

    async def recover_owned(directory, started):
        job = str(started.get("job_id") or "")
        if not job or job == os.environ.get("SLURM_JOB_ID"):
            # These checks use only this process's launch-stop markers, which may change.
            finished = _owner_finished(started)
        else:
            async with owner_locks.setdefault(job, asyncio.Lock()):
                if job not in owner_checks:
                    try:
                        owner_checks[job] = (await asyncio.to_thread(_owner_finished, started), None)
                    except Exception as error:
                        owner_checks[job] = (False, f"{type(error).__name__}: {error}")
                finished, error = owner_checks[job]
                if error:
                    raise RuntimeError(f"Owner job lookup failed: {error}")
        if not finished:
            raise RuntimeError("Cannot recover attempt while owning job may still be active")
        progress["attempted"] += 1
        return await recover_attempt(directory, full_resolved_config)

    async def recover_slot(task, repeat):
        async with semaphore:
            attempts = attempts_for(slot_directory(campaign, model, task[TASK], repeat))
            if not attempts:
                return
            directory = attempts[-1]
            try:
                # Cleanup does not need to rescan potentially huge HTTP journals for scoring.
                active = (directory / "started.json").exists() and not any(
                    (directory / name).exists() for name in
                    ("terminal.json", "dispatch-failure.json", "failure.json", "interrupted.json"))
                if cleanup_confirmed(directory):
                    if active:
                        atomic_json(directory / "interrupted.json", {"time": time.time(), "reason": "owner_job_finished"})
                    progress["already_clean"] += 1
                    return
                started = read_json(directory / "started.json")
                if not started:
                    raise RuntimeError("Cannot recover an attempt without ownership metadata")
                report = await asyncio.wait_for(recover_owned(directory, started), per_attempt_timeout)
                progress["reports"].append({"attempt_dir": str(directory), **report})
                if report["remaining_sandboxes"]:
                    raise RuntimeError("Sandbox recovery incomplete; refusing overlapping attempt")
                atomic_json(directory / "cleanup-confirmed.json", {"time": time.time(), "report": report})
                if active:
                    atomic_json(directory / "interrupted.json", {"time": time.time(), "reason": "owner_job_finished"})
                progress["recovered"] += 1
            except Exception as error:
                from deepswe_http import redact
                progress["remaining"].append(str(directory))
                progress["errors"].append(redact({"attempt_dir": str(directory),
                    "error_type": type(error).__name__, "error": str(error)}))

    await asyncio.gather(*(recover_slot(task, repeat) for task in manifest["tasks"]
                          if task[TASK] % manifest["shards"] == shard_id
                          for repeat in range(manifest["settings"]["repeats"])))
    return progress


def _regrade_response_evidence(directory, source, started):
    """Locate a preserved response without inventing the cancelled collector result."""
    attempt, original = directory.resolve(), source.resolve()
    attempt_index = re.fullmatch(r"attempt-(\d+)", attempt.name)
    original_index = re.fullmatch(r"attempt-(\d+)", original.name)
    if (started.get("regrade") is not True or attempt.parent != original.parent or
            not attempt_index or not original_index or
            int(original_index[1]) >= int(attempt_index[1])):
        return None
    source_started = read_json(source / "started.json") or {}
    if source_started.get("regrade") is not False or not finalized_candidate(source):
        return None
    terminal = read_json(directory / "terminal.json") or {}
    native = read_json(directory / "native-result.json") or {}
    response = (read_json(source / "verify-request.json") or {}).get("response")
    if (terminal.get("evaluation_completed") is not True or native.get("evaluation_completed") is not True or
            type(terminal.get("reward")) not in (int, float) or type(native.get("reward")) not in (int, float) or
            terminal["reward"] not in (0, 1) or terminal["reward"] != native["reward"] or
            not (directory / "model.patch").is_file() or
            (directory / "model.patch").read_bytes() != (source / "model.patch").read_bytes() or
            not isinstance(response, dict) or not isinstance(response.get("output"), list) or
            not response["output"] or response != native.get("response") or
            generation_failure(directory, terminal) is not None):
        return None
    return {"original_verify_request": str(source / "verify-request.json"),
            "completed_regrade_native_result": str(directory / "native-result.json"),
            "json_field": "response"}


def artifact_completeness(directory):
    directory = Path(directory)
    started = read_json(directory / "started.json") or {}
    source = Path(started.get("agent_artifacts_from", str(directory)))
    required = {"agent/" + name: source / name for name in (
        "agent-final-snapshot.json", "opencode-data.tar.gz", "streams-complete.json",
        "stdout.log", "stderr.log", "http.jsonl", "native-result.json")}
    required.update({"attempt/" + name: directory / name for name in (
        "started.json", "request.json", "terminal.json", "native-result.json", "model.patch")})
    response_evidence = None
    if not required["agent/native-result.json"].is_file():
        response_evidence = _regrade_response_evidence(directory, source, started)
        if response_evidence:
            del required["agent/native-result.json"]
    protocol = (read_json(source / "agent-command.json") or {}).get("capture_protocol_version", 1)
    stream_source = "pty" if (source / "streams-complete.json").is_file() else None
    stream_gaps = []
    if protocol >= 2:
        writers = read_json(source / "stream-writers-complete.json") or {}
        writers_complete = (all(writers.get(key) == 0 for key in ("stdout_exit_code", "stderr_exit_code")) and
                            not (source / "stream-writer-error.json").exists())
        if not writers_complete:
            stream_gaps.append("successful agent stream writers")
        if stream_source is None:
            remote = read_json(source / "remote-streams-complete.json") or {}
            control = read_json(source / "process-control.json") or {}
            complete = (writers_complete and remote.get("capture_protocol_version") == 2 and
                        remote.get("producer_and_wrapper_stopped") is True and remote.get("writers_flushed") is True and
                        control.get("confirmed") is True and control.get("wrapper", {}).get("confirmed") is True and
                        control.get("wrapper_forced_stop") is False)
            if complete:
                for name in ("stdout", "stderr"):
                    record = remote.get("files", {}).get(name) or {}
                    path = source / f"remote-{name}.log"
                    if record.get("path") != path.name or not path.is_file() or record.get("size_bytes") != path.stat().st_size:
                        complete = False
                        break
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        while chunk := stream.read(1024 * 1024):
                            digest.update(chunk)
                    if record.get("sha256") != digest.hexdigest():
                        complete = False
                        break
            if complete:
                del required["agent/streams-complete.json"]
                stream_source = "remote"
            else:
                stream_gaps.append("confirmed full remote streams or original PTY EOF")
    missing = [name for name, path in required.items() if not path.is_file()]
    missing.extend(stream_gaps)
    grading = sorted(directory.glob("grading/attempt-*/capture.json"))
    if not grading or not read_json(grading[-1]).get("artifact_complete"):
        missing.append("complete verifier artifacts")
    elif any(not (grading[-1].parent / name).is_file() for name in ("reward.json", "ctrf.json", "run.log")):
        missing.append("verifier reward.json, ctrf.json, and run.log")
    calls, responses = set(), set()
    if (source / "http.jsonl").is_file():
        with (source / "http.jsonl").open("rb") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    missing.append("parseable upstream HTTP events")
                    break
                identity = (event.get("call_id"), event.get("http_attempt"))
                if event.get("event") == "request":
                    calls.add(identity)
                elif event.get("event") in {"response", "error"}:
                    responses.add(identity)
                elif (event.get("event") == "buffered_upstream_failure" and identity in calls
                      and isinstance(identity[0], str) and identity[0]
                      and type(identity[1]) is int and identity[1] > 0
                      and event.get("status") == 200 and event.get("error_type") == "UpstreamStreamError"
                      and event.get("buffered_response") is True and event.get("buffered_upstream_failure") is True
                      and event.get("outcome") == "not_delivered"):
                    # The SSE adapter saves this explicit failed-attempt outcome before retrying.
                    responses.add(identity)
    if calls - responses:
        missing.append("response or explicit error for every upstream HTTP attempt")
    result = {"artifact_complete": not missing, "missing": missing,
            "agent_artifacts_from": str(source), "attempt_dir": str(directory),
            "capture_protocol_version": protocol, "stream_capture_source": stream_source}
    if response_evidence:
        result["agent_response_evidence"] = response_evidence
    return result


def summarize(campaign):
    import csv
    from contextlib import ExitStack

    campaign = Path(campaign).resolve()
    manifest = load_manifest(campaign)
    settings = manifest["settings"]
    repeats, task_count = settings["repeats"], len(manifest["tasks"])
    paths = {name: str(campaign / filename) for name, filename in (
        ("rollouts", "rollouts.jsonl"), ("attempts", "attempts.jsonl"),
        ("metrics", "metrics.csv"), ("status", "status.json"))}
    report = {"campaign": str(campaign), "models": {}, "outputs": paths,
              "expected_rollouts": task_count * repeats * len(settings["models"])}
    metrics = []
    temporary_paths = {name: campaign / (Path(path).name + "." + uuid.uuid4().hex + ".tmp")
                       for name, path in paths.items() if name != "status"}
    grade_fields = {"reward", "evaluation_completed", "apply_failed", "verifier_exit_code", "verifier_error",
        "f2p_total", "f2p_passed", "p2p_total", "p2p_passed", "f2p", "p2p", "partial", "model_patch",
        "model_patch_sha256", "model_patch_bytes", "log_dir", "patch_collection_time_s", "sandbox_start_time_s",
        "verification_time_s", "_atp_capture"}
    with ExitStack() as stack:
        rollout_stream = stack.enter_context(temporary_paths["rollouts"].open("w"))
        attempt_stream = stack.enter_context(temporary_paths["attempts"].open("w"))
        for model, model_settings in settings["models"].items():
            counts = {"expected": task_count * repeats, "completed": 0, "passed": 0,
                      "artifact_complete": 0, "attempts": 0, "artifact_complete_attempts": 0,
                      "attempt_statuses": {}, "missing_artifact_counts": {}}
            complete_task_passes = []
            for task in manifest["tasks"]:
                task_completed = task_passed = 0
                for repeat in range(repeats):
                    state = slot_state(campaign, model, task[TASK], repeat)
                    counts["attempts"] += state["attempts"]
                    counts[state["status"]] = counts.get(state["status"], 0) + 1
                    attempts = attempts_for(slot_directory(campaign, model, task[TASK], repeat))
                    selected, artifacts = {}, {"artifact_complete": False, "missing": ["attempt not started"],
                                                "agent_artifacts_from": None, "attempt_dir": None}
                    for directory in attempts:
                        started = read_json(directory / "started.json") or {}
                        terminal = read_json(directory / "terminal.json") or {}
                        native = read_json(directory / "native-result.json") or {}
                        failure = read_json(directory / "dispatch-failure.json") or read_json(directory / "failure.json")
                        if outcome(terminal, directory) == "completed":
                            attempt_status = "completed"
                        elif not started:
                            attempt_status = "orphaned"
                        elif (directory / "interrupted.json").is_file():
                            attempt_status = "interrupted"
                        elif failure:
                            attempt_status = {"fatal": "fatal", "infrastructure": "infrastructure_failure"}.get(
                                failure.get("classification"), "failed")
                        else:
                            attempt_status = "incomplete" if terminal else "active"
                        completeness = artifact_completeness(directory)
                        counts["artifact_complete_attempts"] += completeness["artifact_complete"]
                        counts["attempt_statuses"][attempt_status] = counts["attempt_statuses"].get(attempt_status, 0) + 1
                        for missing in completeness["missing"]:
                            counts["missing_artifact_counts"][missing] = counts["missing_artifact_counts"].get(missing, 0) + 1
                        chosen = str(directory) == state["attempt_dir"]
                        attempt_record = {"model_alias": model, TASK: task[TASK], REPEAT: repeat,
                            "attempt_index": int(directory.name.split("-")[-1]), "status": attempt_status,
                            "slot_status": state["status"], "selected_for_slot": chosen,
                            "attempt_dir": str(directory), "artifacts": completeness,
                            "terminal_result": str(directory / "terminal.json") if terminal else None,
                            "native_result": str(directory / "native-result.json") if native else None}
                        attempt_stream.write(json.dumps(attempt_record, allow_nan=False) + "\n")
                        if chosen:
                            artifacts = completeness
                            selected = terminal | native
                            # The completed verifier's grade is authoritative, including reward=0.
                            selected.update({key: value for key, value in terminal.items() if key in grade_fields})
                            source = Path(completeness["agent_artifacts_from"])
                            if started.get("regrade") and source != directory:
                                original = read_json(source / "native-result.json") or read_json(source / "terminal.json") or {}
                                # A cancelled collector may never have returned an envelope.
                                # Its exact saved response can still survive in the completed regrade.
                                response = selected.get("response") if not original and completeness.get("agent_response_evidence") else None
                                selected = original | {key: value for key, value in selected.items() if key in grade_fields}
                                if response is not None:
                                    selected["response"] = response
                                selected["_atp_regrade"] = True
                                selected["_atp_grade_result"] = str(directory / ("native-result.json" if native else "terminal.json"))
                    if state["status"] == "completed":
                        counts["artifact_complete"] += artifacts["artifact_complete"]
                        passed = selected.get("reward") == 1
                        counts["passed"] += passed
                        task_completed += 1
                        task_passed += passed
                    rollout_record = task | selected | {"model_alias": model, TASK: task[TASK], REPEAT: repeat,
                        "_atp_status": state["status"], "_atp_attempt_dir": state["attempt_dir"],
                        "_ng_attempt_index": int(Path(state["attempt_dir"]).name.split("-")[-1]) if state["attempt_dir"] else None,
                        "_atp_attempts": [str(path) for path in attempts], "_atp_artifacts": artifacts}
                    rollout_stream.write(json.dumps(rollout_record, allow_nan=False) + "\n")
                if task_completed == repeats:
                    complete_task_passes.append(task_passed)
            denominator = len(complete_task_passes)
            counts["pass_rate"] = counts["passed"] / counts["completed"] if counts["completed"] else None
            counts["fully_completed_tasks"] = denominator
            counts["pass_at_1"] = sum(complete_task_passes) / (repeats * denominator) if denominator else None
            counts["pass_at_1_task_denominator"] = denominator
            counts["pass_at_requested"] = sum(value > 0 for value in complete_task_passes) / denominator if denominator else None
            counts["pass_at_requested_k"] = repeats
            counts["pass_at_8"] = counts["pass_at_requested"] if repeats == 8 else None
            counts["pass_at_8_task_denominator"] = denominator if repeats == 8 else 0
            counts["artifact_incomplete_attempts"] = counts["attempts"] - counts["artifact_complete_attempts"]
            counts["artifact_incomplete_completed_rollouts"] = counts["completed"] - counts["artifact_complete"]
            report["models"][model] = counts
            metrics.append({"model_alias": model, "model_id": model_settings["id"], "expected_tasks": task_count,
                "repeats_per_task": repeats, "expected_rollouts": counts["expected"], "completed_rollouts": counts["completed"],
                "passed_rollouts": counts["passed"], "completed_rollout_pass_rate": counts["pass_rate"],
                "fully_completed_tasks": denominator, "pass_at_1": counts["pass_at_1"],
                "pass_at_1_task_denominator": denominator, "pass_at_8": counts["pass_at_8"],
                "pass_at_8_task_denominator": counts["pass_at_8_task_denominator"],
                "pass_at_requested_k": repeats, "pass_at_requested": counts["pass_at_requested"],
                "pass_at_requested_task_denominator": denominator, "attempts": counts["attempts"],
                "artifact_complete_rollouts": counts["artifact_complete"],
                "artifact_incomplete_completed_rollouts": counts["artifact_incomplete_completed_rollouts"],
                "artifact_incomplete_attempts": counts["artifact_incomplete_attempts"],
                "rollouts_path": paths["rollouts"], "attempts_path": paths["attempts"], "status_path": paths["status"]})
        metric_stream = stack.enter_context(temporary_paths["metrics"].open("w", newline=""))
        writer = csv.DictWriter(metric_stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
        for stream in (rollout_stream, attempt_stream, metric_stream):
            stream.flush()
            os.fsync(stream.fileno())
    for name, temporary in temporary_paths.items():
        os.replace(temporary, paths[name])
    atomic_json(campaign / "status.json", report)
    return report
