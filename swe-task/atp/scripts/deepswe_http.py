"""Durable hosted-model HTTP calls for the isolated DeepSWE evaluation."""

import asyncio
from contextvars import ContextVar
from datetime import timezone
from email.utils import parsedate_to_datetime
import fcntl
from functools import wraps
import json
import math
import os
from pathlib import Path
import random
import time
from urllib.parse import urlsplit
import uuid
from weakref import WeakKeyDictionary

from deepswe_upstream import BufferedChatResponse, UpstreamStreamError, collect_chat_sse


_CONTEXT = ContextVar("deepswe_http_context", default=None)
_SEMAPHORES = WeakKeyDictionary()
_RETRY_STATUSES = {408, 429, 500, 502, 503, 504, 520}
_RESPONSE_HEADERS = {"retry-after", "date", "request-id", "x-request-id"}


class RateLimitDeadline(TimeoutError):
    """The current attempt cannot wait for its upstream rate limit to clear."""


class LocalAdmissionDeadline(TimeoutError):
    """Local capacity waiting exhausted the deadline before a request was sent."""


class MalformedUpstreamResponse(RuntimeError):
    """A confirmed upstream reply cannot represent a valid model completion."""


def malformed_success(payload):
    """Reject empty zero-token stops, including replies containing only reasoning."""
    if not isinstance(payload, dict):
        return False
    choices, usage = payload.get("choices"), payload.get("usage")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(usage, dict):
        return False
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
        return False
    message = choice.get("message")
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    empty_content = content is None or content == [] or (isinstance(content, str) and not content.strip())
    return bool(empty_content and not message.get("tool_calls") and not message.get("function_call")
                and not message.get("refusal")
                and all(type(usage.get(key)) in (int, float) and usage[key] == 0
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens")))


def redact(value):
    secrets = sorted({item for key, item in os.environ.items() if item and
                      key.upper().endswith(("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD"))},
                     key=len, reverse=True)

    def clean(item):
        if isinstance(item, str):
            for secret in secrets:
                item = item.replace(secret, "[REDACTED]")
            return item
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        return item

    return clean(value)


def append_event(path, event):
    """Persist before delivering an upstream answer or starting a retry."""
    payload = (json.dumps(redact(event), ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    with Path(path).open("ab") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def retry_after(value, now=None):
    """Parse delta seconds or an HTTP date; None selects exponential backoff."""
    if value is None:
        return None
    now = time.time() if now is None else now
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            delay = date.timestamp() - now
        except (TypeError, ValueError, OverflowError):
            return None
        return max(0.0, delay)
    return delay if math.isfinite(delay) and delay >= 0 else None


def _gateway_capacity_error(status, body):
    """The gateway reports its observed database exhaustion as HTTP 401."""
    if status != 401:
        return False
    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        return False
    error = payload.get("error") if isinstance(payload, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    return (isinstance(message, str) and "error querying the database" in message.lower()
            and "too many clients already" in message.lower())


def _finite_time(value, name):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(name + " must be a positive finite time")
    return result


def correlated_model_call(function):
    """Keep sandbox correlation out of the provider's body and auth headers."""
    @wraps(function)
    async def wrapped(self, request, *args, **kwargs):
        if not os.environ.get("ATP_DEEPSWE_CAMPAIGN"):
            return await function(self, request, *args, **kwargs)
        headers = request.headers
        attempt = headers.get("x-atp-attempt-dir")
        deadline = headers.get("x-atp-deadline")
        from deepswe_streaming import STREAM_ID
        context = {"model_call_id": uuid.uuid4().hex, "stream_id": STREAM_ID.get()}
        if attempt or deadline:
            if not attempt or not deadline:
                raise ValueError("DeepSWE model requests need attempt directory and deadline")
            campaign = Path(os.environ["ATP_DEEPSWE_CAMPAIGN"]).resolve()
            directory = Path(attempt)
            if not directory.is_absolute() or not directory.resolve().is_relative_to(campaign) or not directory.is_dir():
                raise ValueError("DeepSWE attempt directory must exist inside the campaign")
            context.update({"attempt_dir": str(directory.resolve()),
                            "deadline": _finite_time(deadline, "attempt deadline"),
                            "rollout_id": headers.get("x-atp-rollout-id")})
        token = _CONTEXT.set(context)
        journal = Path(context.get("attempt_dir") or os.environ["ATP_DEEPSWE_LAUNCH_DIR"]) / "http.jsonl"

        async def record(event, **fields):
            await asyncio.to_thread(append_event, journal, {
                "event": event, "timestamp": time.time(), "model_call_id": context["model_call_id"],
                "stream_id": context.get("stream_id"),
                "attempt_dir": context.get("attempt_dir"), "rollout_id": context.get("rollout_id"),
                "scope": "native_model_method", **fields})

        try:
            result = await function(self, request, *args, **kwargs)
            # The provider response may still fail Gym's conversion. Returning
            # from this method is observable; downstream SSE receipt is not.
            await record("model_call_completed", result_type=type(result).__name__)
            return result
        except BaseException as error:
            await record("model_call_error", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            _CONTEXT.reset(token)
    return wrapped


def single_transport_attempt(method, url, internal):
    """Never replay an ambiguous generation or a side-effecting Gym POST."""
    return bool(os.environ.get("ATP_DEEPSWE_CAMPAIGN")) and (
        not internal or (method.upper() == "POST" and
                         urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1] in
                         {"run", "seed_session", "verify"}))


def _cooldown_path(model=None):
    # Model quotas are separate; database exhaustion affects the shared gateway.
    directory = Path(os.environ["ATP_DEEPSWE_CAMPAIGN"]) / "rate-limits"
    directory.mkdir(exist_ok=True)
    if model is not None and (not model or not all(char.isalnum() or char in "_-" for char in model)):
        raise ValueError("Invalid model alias for rate-limit cooldown")
    return directory / ("nvidia-api-" + model + ".json" if model is not None else "nvidia-api.json")


def cooldown(path, until=None, *, extend=True):
    """Explicit Retry-After extends waits; fallback bursts share one window."""
    path = Path(path)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = json.loads(path.read_text())["until"] if path.exists() else 0.0
        if not isinstance(current, (int, float)) or not math.isfinite(current) or current < 0:
            raise ValueError("Invalid persisted rate-limit cooldown")
        if until is not None and until > current and (extend or current <= time.time()):
            temporary = path.with_suffix(".tmp")
            with temporary.open("w") as stream:
                json.dump({"until": until}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            current = until
        return current


def _semaphore():
    limit = int(os.environ.get("ATP_DEEPSWE_API_CONCURRENCY", "4"))
    if limit < 1:
        raise ValueError("ATP_DEEPSWE_API_CONCURRENCY must be positive")
    loop = asyncio.get_running_loop()
    if loop not in _SEMAPHORES:
        _SEMAPHORES[loop] = asyncio.Semaphore(limit)
    return _SEMAPHORES[loop]


def canonicalize_reasoning(payload):
    """Hosted backends can interpret Gym's two reasoning keys as duplicate aliases."""
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        return payload
    messages = None
    for index, message in enumerate(payload["messages"]):
        if not isinstance(message, dict) or message.get("role") != "assistant" or not (
                "reasoning" in message and "reasoning_content" in message):
            continue
        if message["reasoning"] != message["reasoning_content"]:
            raise ValueError("Assistant reasoning and reasoning_content aliases disagree")
        if messages is None:
            messages = list(payload["messages"])
        messages[index] = {key: value for key, value in message.items() if key != "reasoning"}
    return payload if messages is None else dict(payload, messages=messages)


async def request_with_retry(send, request_kwargs):
    """Retry confirmed failures and buffered SSE failures before any delivery."""
    from deepswe_admission import api_slot, note_health

    context = _CONTEXT.get() or {}
    directory = Path(context.get("attempt_dir") or os.environ["ATP_DEEPSWE_LAUNCH_DIR"])
    journal = directory / "http.jsonl"
    deadline = context.get("deadline", time.time() +
                           _finite_time(os.environ.get("ATP_DEEPSWE_AGENT_TIMEOUT", "10800"), "agent timeout"))
    if os.environ.get("ATP_DEEPSWE_JOB_DEADLINE"):
        deadline = min(deadline, _finite_time(os.environ["ATP_DEEPSWE_JOB_DEADLINE"], "job deadline"))
    parsed = urlsplit(request_kwargs["url"])
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Hosted endpoint URLs cannot contain credentials, queries, or fragments")
    credential_id = None
    from deepswe_credentials import bind_credential, credential_ids
    hosted = (not request_kwargs.get("_internal", False)
              and parsed.path.rstrip("/").endswith("/chat/completions"))
    # hengyuf: only the configured hosted endpoint is hosted (bound credential, shared gateway cooldown
    # and capacity handling), whatever the credential count; e.g. a local student router is not.
    if hosted:
        base_url = os.environ.get("ATP_DEEPSWE_POLICY_BASE_URL")
        if not base_url:
            raise ValueError("Hosted model calls require their configured upstream endpoint")
        hosted = parsed.geturl().rstrip("/") == base_url.rstrip("/") + "/chat/completions"
    if hosted:
        credential_id, secret = await asyncio.to_thread(bind_credential, context.get("attempt_dir"))
        headers = {key: value for key, value in request_kwargs.get("headers", {}).items()
                   if key.lower() != "authorization"}
        request_kwargs = dict(request_kwargs, headers={**headers, "Authorization": "Bearer " + secret})
    identity = {"call_id": uuid.uuid4().hex, "model_call_id": context.get("model_call_id"),
                "stream_id": context.get("stream_id"),
                "rollout_id": context.get("rollout_id"),
                "attempt_dir": context.get("attempt_dir"), "url": parsed.geturl()}
    if credential_id is not None:
        identity["credential_id"] = credential_id
    model = os.environ.get("ATP_DEEPSWE_MODEL_ALIAS")
    shared_cooldown_path = _cooldown_path()
    model_scope = (model + "-" + credential_id if model and credential_id
                   and len(credential_ids()) > 1 else model)
    model_cooldown_path = _cooldown_path(model_scope)
    # Keep any pre-pool wait on the original account; new accounts have their own quotas.
    legacy_cooldown_path = _cooldown_path(model) if credential_id == "existing" else model_cooldown_path
    cooldown_paths = tuple(dict.fromkeys((shared_cooldown_path, model_cooldown_path, legacy_cooldown_path)))
    if not hosted:  # hengyuf: non-hosted calls keep only their model-scoped cooldown
        cooldown_paths = (model_cooldown_path,)
    retry_not_before = 0.0
    attempt = 0
    server_retries = 0
    rate_retries = 0

    async def record(event, **fields):
        await asyncio.to_thread(append_event, journal,
                                {**identity, "event": event, "http_attempt": attempt,
                                 "timestamp": time.time(), **fields})

    def current_cooldown():
        return max(retry_not_before, *(cooldown(path) for path in cooldown_paths))

    async def wait_for_cooldown():
        while True:
            until = await asyncio.to_thread(current_cooldown)
            delay = until - time.time()
            if until >= deadline:
                raise RateLimitDeadline("Upstream wait exceeds this attempt's deadline")
            if time.time() >= deadline:
                raise TimeoutError("Hosted model call exceeded this attempt's deadline")
            if delay <= 0:
                return
            await record("rate_limit_wait", wait_seconds=delay, retry_at=until, deadline=deadline)
            await asyncio.sleep(delay)

    try:
        original_payload = request_kwargs.get("json")
        try:
            payload = canonicalize_reasoning(original_payload)
        except ValueError as error:
            await record("request_rejected", payload=original_payload, error=str(error))
            raise
        upstream_stream = (os.environ.get("ATP_DEEPSWE_UPSTREAM_STREAM") == "1"
                           and not request_kwargs.get("_internal", False)
                           and request_kwargs.get("method", "POST").upper() == "POST"
                           and parsed.path.rstrip("/").endswith("/chat/completions")
                           and isinstance(payload, dict))
        if upstream_stream:
            payload = dict(payload, stream=True,
                           stream_options=dict(payload.get("stream_options") or {}, include_usage=True))
        request_fields = {"payload": payload}
        if payload is not original_payload:
            request_fields["source_payload"] = original_payload
            request_kwargs = dict(request_kwargs, json=payload)
        while True:
            await wait_for_cooldown()
            remaining = deadline - time.time()
            if remaining <= 0:
                raise LocalAdmissionDeadline("Deadline expired before hosted request admission")
            wait_started = time.time()
            send_started = False
            stream_error = None
            invalid_success = gateway_capacity = False
            await record("api_gate_wait_started", wait_started_at=wait_started, deadline=deadline,
                         next_http_attempt=attempt + 1, outcome="not_sent")
            try:
                async with asyncio.timeout(max(0.0, deadline - time.time())):
                    async with _semaphore(), api_slot():
                        await record("api_gate_acquired", wait_started_at=wait_started,
                                     wait_seconds=max(0.0, time.time() - wait_started), deadline=deadline,
                                     next_http_attempt=attempt + 1, outcome="not_sent")
                        # A sibling can extend the cooldown while we await capacity.
                        if await asyncio.to_thread(current_cooldown) > time.time():
                            continue
                        await record("request", http_attempt=attempt + 1,
                                     method=request_kwargs.get("method", "POST"),
                                     deadline=deadline, **request_fields)
                        if time.time() >= deadline:
                            raise LocalAdmissionDeadline("Deadline expired before sending the admitted request")
                        attempt += 1
                        send_started = True
                        response = await send(**request_kwargs)
                        headers = {key.lower(): value for key, value in response.headers.items()
                                   if key.lower() in _RESPONSE_HEADERS or key.lower().startswith("x-ratelimit-")}
                        try:
                            streamed = upstream_stream and response.status == 200
                            if streamed:
                                raw = await collect_chat_sse(response, record)
                                response = BufferedChatResponse(response, raw)
                            else:
                                raw = await response.read()
                            body = raw.decode("utf-8", errors="replace")
                            # hengyuf: only the hosted gateway reports shared capacity exhaustion.
                            gateway_capacity = hosted and _gateway_capacity_error(response.status, body)
                            try:
                                invalid_success = response.status == 200 and malformed_success(json.loads(body))
                            except ValueError:
                                invalid_success = False
                            await record("response", status=response.status, headers=headers,
                                         body=body, retryable_gateway_capacity=gateway_capacity,
                                         malformed_success=invalid_success,
                                         upstream_transport="sse" if streamed else "json")
                        except UpstreamStreamError as error:
                            response.release()
                            if not upstream_stream:
                                raise
                            # The entire answer is buffered: no model/tool delta has
                            # reached OpenCode, so retrying cannot execute a tool twice.
                            stream_error = error
                            await record("buffered_upstream_failure", status=response.status,
                                         headers=headers, error_type=type(error).__name__, error=str(error),
                                         buffered_response=True, buffered_upstream_failure=True,
                                         outcome="not_delivered")
                        except BaseException:
                            response.release()
                            raise
            except TimeoutError as error:
                if not send_started:
                    raise LocalAdmissionDeadline("Local API admission exceeded this attempt's deadline") from error
                raise
            status = response.status
            if stream_error:
                await asyncio.to_thread(note_health, model, "upstream_error")
            elif invalid_success:
                await record("malformed_success", status=status, reason="zero_token_empty_stop")
                await asyncio.to_thread(note_health, model, "upstream_error")
            elif status == 200:
                await asyncio.to_thread(note_health, model, "success_http")
            elif gateway_capacity:
                await asyncio.to_thread(note_health, model, "gateway_capacity")
            elif status in {408, 429}:
                await asyncio.to_thread(note_health, model, "rate_limit" if status == 429 else "upstream_timeout")
            elif status in _RETRY_STATUSES:
                await asyncio.to_thread(note_health, model, "upstream_error")
            if request_kwargs.get("_internal", False) and request_kwargs.get("method", "POST").upper() == "POST":
                return response
            if status not in _RETRY_STATUSES and not gateway_capacity and not invalid_success and not stream_error:
                return response
            if status != 429 and server_retries >= 6:
                await record("retry_exhausted", status=status, retryable_gateway_capacity=gateway_capacity,
                             malformed_success=invalid_success, buffered_upstream_failure=bool(stream_error))
                if stream_error:
                    raise stream_error
                if invalid_success:
                    response.release()
                    raise MalformedUpstreamResponse("Hosted model returned zero-token empty stop after bounded retries")
                return response
            response.release()
            if status == 429:
                rate_retries += 1
                header_delay = retry_after(headers.get("retry-after"))
                delay = header_delay
                if delay is None:
                    delay = min(60.0, 2.0 ** min(rate_retries - 1, 6))
                delay += random.uniform(0.0, min(1.0, max(0.1, delay * 0.1)))
                retry_not_before = time.time() + delay
                until = await asyncio.to_thread(cooldown, model_cooldown_path, retry_not_before,
                                                extend=header_delay is not None)
                until = max(until, retry_not_before)
                await record("retry", status=status, retry_at=until, wait_seconds=delay,
                             retryable_gateway_capacity=False)
            else:
                server_retries += 1
                delay = min(60.0, 2.0 ** (server_retries - 1)) + random.uniform(0.0, 1.0)
                header_delay = retry_after(headers.get("retry-after"))
                if header_delay is not None:
                    delay = max(delay, header_delay)
                if time.time() + delay >= deadline:
                    if stream_error:
                        raise stream_error
                    raise TimeoutError("Upstream retry exceeds this attempt's deadline")
                if gateway_capacity or status == 408:
                    retry_not_before = time.time() + delay
                    path = shared_cooldown_path if gateway_capacity else model_cooldown_path
                    until = await asyncio.to_thread(cooldown, path, retry_not_before,
                                                    extend=header_delay is not None)
                    until = max(until, retry_not_before)
                    await record("retry", status=status, retry_at=until, wait_seconds=delay,
                                 retryable_gateway_capacity=gateway_capacity)
                else:
                    await record("retry", status=status, wait_seconds=delay,
                                 retryable_gateway_capacity=False, buffered_upstream_failure=bool(stream_error))
                    await asyncio.sleep(delay)
    except asyncio.CancelledError:
        await record("cancelled", outcome="unknown" if attempt else "not_sent")
        raise
    except BaseException as error:
        await record("error", error_type=type(error).__name__, error=str(error),
                     outcome="not_sent" if isinstance(error, LocalAdmissionDeadline) or not attempt else "unknown",
                     next_http_attempt=attempt + 1)
        raise
