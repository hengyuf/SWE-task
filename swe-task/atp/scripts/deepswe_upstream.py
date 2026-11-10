"""Collect an opt-in hosted SSE response without changing Gym's JSON contract."""
import asyncio
import base64
import codecs
import json


class UpstreamStreamError(RuntimeError):
    """Upstream streaming failed before a complete, accounted answer was saved."""


class BufferedChatResponse:
    def __init__(self, response, body):
        self.response, self.body = response, body

    def __getattr__(self, name):
        return getattr(self.response, name)

    async def read(self):
        return self.body


def _append_text(target, key, value):
    if value is None:
        return
    if not isinstance(value, str):
        raise UpstreamStreamError(f"Non-text upstream delta for {key}")
    target[key] = target.get(key, "") + value


class _Completion:
    def __init__(self):
        self.value = {"object": "chat.completion"}
        self.choices = {}
        self.done = False

    def add(self, data):
        if self.done:
            raise UpstreamStreamError("Upstream sent data after DONE")
        if data == "[DONE]":
            self.done = True
            return
        chunk = json.loads(data)
        if not isinstance(chunk, dict) or chunk.get("error") is not None:
            raise UpstreamStreamError("Upstream SSE contained an error or non-object payload")
        if chunk.get("object") != "chat.completion.chunk" or not isinstance(chunk.get("choices"), list):
            raise UpstreamStreamError("Upstream SSE is not a chat completion chunk")
        for key in ("id", "created", "model", "system_fingerprint", "service_tier"):
            value = chunk.get(key)
            if value is not None:
                if key in self.value and self.value[key] != value:
                    raise UpstreamStreamError(f"Upstream {key} changed within one stream")
                self.value[key] = value
        if chunk.get("usage") is not None:
            self.value["usage"] = chunk["usage"]
        for incoming in chunk["choices"]:
            index = incoming.get("index")
            if type(index) is not int or index < 0:
                raise UpstreamStreamError("Upstream choice index is invalid")
            choice = self.choices.setdefault(index, {
                "index": index, "message": {"role": "assistant"}, "finish_reason": None, "tools": {},
            })
            delta = incoming.get("delta") or {}
            if not isinstance(delta, dict):
                raise UpstreamStreamError("Upstream choice delta is not an object")
            if choice["finish_reason"] is not None and any(delta.values()):
                raise UpstreamStreamError("Upstream emitted content after a terminal choice")
            if delta.get("role") not in (None, "assistant"):
                raise UpstreamStreamError("Upstream returned a non-assistant role")
            for key in ("content", "reasoning_content", "reasoning", "refusal"):
                _append_text(choice["message"], key, delta.get(key))
            for tool_delta in delta.get("tool_calls") or []:
                tool_index = tool_delta.get("index")
                if type(tool_index) is not int or tool_index < 0:
                    raise UpstreamStreamError("Upstream tool-call index is invalid")
                tool = choice["tools"].setdefault(tool_index, {"type": "function", "function": {}})
                for key in ("id", "type"):
                    if tool_delta.get(key) is not None:
                        if key in tool and tool[key] != tool_delta[key]:
                            raise UpstreamStreamError(f"Upstream tool {key} changed within one call")
                        tool[key] = tool_delta[key]
                function = tool_delta.get("function") or {}
                for key in ("name", "arguments"):
                    _append_text(tool["function"], key, function.get(key))
            known = {"role", "content", "reasoning_content", "reasoning", "refusal", "tool_calls"}
            if any(value is not None and key not in known for key, value in delta.items()):
                raise UpstreamStreamError("Upstream returned an unsupported message delta; raw chunks are preserved")
            if incoming.get("logprobs") is not None:
                logs = choice.setdefault("logprobs", {})
                for key, value in incoming["logprobs"].items():
                    if value is not None:
                        if not isinstance(value, list):
                            raise UpstreamStreamError("Upstream logprob field is not a list")
                        logs.setdefault(key, []).extend(value)
            reason = incoming.get("finish_reason")
            if reason is not None:
                if not isinstance(reason, str) or not reason or choice["finish_reason"] not in (None, reason):
                    raise UpstreamStreamError("Upstream terminal reason is invalid or changed")
                choice["finish_reason"] = reason

    def finish(self):
        if not self.done or not self.choices or any(choice["finish_reason"] is None for choice in self.choices.values()):
            raise UpstreamStreamError("Upstream stream ended without DONE and terminal choices")
        if any(key not in self.value for key in ("id", "created", "model")):
            raise UpstreamStreamError("Upstream stream omitted completion identity")
        usage = self.value.get("usage")
        if (not isinstance(usage, dict) or any(type(usage.get(key)) is not int or usage[key] < 0
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
                or usage["total_tokens"] <= 0):
            raise UpstreamStreamError("Upstream stream omitted valid nonzero token usage")
        choices = []
        for index in sorted(self.choices):
            choice = self.choices[index]
            tools = choice.pop("tools")
            if tools:
                for tool in tools.values():
                    if not tool.get("id") or tool.get("type") != "function" or not tool["function"].get("name"):
                        raise UpstreamStreamError("Upstream stream ended with an incomplete tool call")
                    tool["function"].setdefault("arguments", "")
                choice["message"]["tool_calls"] = [tools[key] for key in sorted(tools)]
            choice["message"].setdefault("content", None)
            choices.append(choice)
        return dict(self.value, choices=choices)


async def collect_chat_sse(response, record):
    """Persist every received byte chunk before assembling a complete answer."""
    completion = _Completion()
    decoder = codecs.getincrementaldecoder("utf-8")()
    pending, data_lines, count = "", [], 0
    recording = False
    await record("upstream_stream_started", status=response.status,
                 content_type=response.headers.get("Content-Type", ""))
    try:
        async for raw in response.content.iter_any():
            count += 1
            recording = True
            await record("upstream_stream_chunk", chunk_index=count,
                         data_base64=base64.b64encode(raw).decode("ascii"))
            recording = False
            pending += decoder.decode(raw)
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                line = line.removesuffix("\r")
                if not line:
                    if data_lines:
                        completion.add("\n".join(data_lines))
                        data_lines = []
                elif line.startswith("data:"):
                    value = line[5:]
                    data_lines.append(value[1:] if value.startswith(" ") else value)
            if completion.done:
                break
        pending += decoder.decode(b"", final=True)
        if pending.strip() or data_lines:
            raise UpstreamStreamError("Upstream stream ended with an incomplete SSE frame")
        value = completion.finish()
        recording = True
        await record("upstream_stream_completed", chunks=count, usage=value["usage"],
                     finish_reasons=[choice["finish_reason"] for choice in value["choices"]])
        recording = False
        response.release()
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    except asyncio.CancelledError:
        await record("upstream_stream_failed", chunks=count, error_type="CancelledError", classification="infrastructure")
        raise
    except Exception as error:
        if recording:
            # A local persistence failure cannot safely authorize another request.
            raise
        await record("upstream_stream_failed", chunks=count, error_type=type(error).__name__,
                     error=str(error), classification="infrastructure", terminal_received=completion.done,
                     buffered_response=True, outcome="not_delivered")
        if isinstance(error, UpstreamStreamError):
            raise
        raise UpstreamStreamError(f"Upstream stream failed: {type(error).__name__}: {error}") from error
