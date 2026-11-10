"""Keep campaign SSE connections active while Gym buffers one hosted response.

This mitigates idle connections; the historical OpenCode stall's network cause
has not been established. Model/tool chunks still come from Gym's synthesizer.
"""

import asyncio
from contextvars import ContextVar
import json
import os
from pathlib import Path
import time
import uuid

from deepswe_http import append_event, redact


HEARTBEAT = ": keep-alive\n\n"
STREAM_ID = ContextVar("deepswe_stream_id", default=None)


def stream_chat_completion(invoke, request, params, synthesize, *, include_usage=False, interval=15):
    """Return an SSE iterator after validating its durable destination eagerly."""
    campaign = Path(os.environ["ATP_DEEPSWE_CAMPAIGN"]).resolve()
    directory = Path(request.headers.get("x-atp-attempt-dir") or os.environ["ATP_DEEPSWE_LAUNCH_DIR"])
    if not directory.is_absolute() or not directory.resolve().is_relative_to(campaign) or not directory.is_dir():
        raise ValueError("DeepSWE stream directory must exist inside the campaign")
    if interval <= 0:
        raise ValueError("Heartbeat interval must be positive")
    journal = directory / "stream-events.jsonl"
    identity = {"stream_id": uuid.uuid4().hex, "rollout_id": request.headers.get("x-atp-rollout-id")}

    async def record(event, **fields):
        await asyncio.to_thread(append_event, journal, {
            "event": event, "timestamp": time.time(), **identity, **fields})

    async def stream():
        async def complete():
            token = STREAM_ID.set(identity["stream_id"])
            try:
                return await invoke(request, params)
            finally:
                STREAM_ID.reset(token)

        task = asyncio.create_task(complete())
        terminal_prepared = False
        try:
            await record("stream_started")
            yield HEARTBEAT
            while not task.done():
                done, _ = await asyncio.wait({task}, timeout=interval)
                if not done:
                    yield HEARTBEAT
            completion = await task
            payload = completion.model_dump(mode="json") if hasattr(completion, "model_dump") else dict(completion)
            for chunk in synthesize(payload, include_usage=include_usage):
                if chunk.strip() == "data: [DONE]":
                    # Readers can close as soon as DONE arrives. Persist before
                    # yielding it so that this normal closure is not an abort.
                    await record("stream_completed", evidence="model_chunks_emitted_terminal_prepared_no_client_ack")
                    terminal_prepared = True
                yield chunk
        except (asyncio.CancelledError, GeneratorExit):
            if not terminal_prepared:
                await record("stream_cancelled")
            raise
        except Exception as error:
            detail = redact({"message": str(error) or type(error).__name__,
                             "type": type(error).__name__, "code": getattr(error, "status", None) or 500})
            await record("stream_error", error_type=type(error).__name__, error=detail["message"])
            yield "data: " + json.dumps({"error": detail}) + "\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return stream()
