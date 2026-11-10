"""Offline tests of the student router and client against fake vLLM workers on 127.0.0.1 (no GPU, no Slurm).

    cd /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/student
    <python with aiohttp> -B -m unittest discover -s tests -p test_router.py -v
"""
import sys

sys.dont_write_bytecode = True
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))

import asyncio
import json
import logging
import os
import shutil
import socket
import tempfile
import time
import unittest

import aiohttp
from aiohttp import web

from student import client
from student import router as router_module
from student.router import CHAT_TEMPLATE_KWARGS, LOG, MAX_MODEL_LEN, Router, aborted, affinity_key, start, vllm_load

TEMPLATE = dict(CHAT_TEMPLATE_KWARGS)
# IsolatedAsyncioTestCase runs in asyncio debug mode; its slow-callback warnings are noise here.
logging.getLogger("asyncio").setLevel(logging.ERROR)


def overflow_message(prompt_tokens: int, requested: int, chars=False) -> str:
    """vLLM 0.29 renderers/params.py _token_len_check / _text_len_check (tokenizer stops at max_input + 1)."""
    max_input = MAX_MODEL_LEN - requested
    head = (f"This model's maximum context length is {MAX_MODEL_LEN} tokens. However, you requested "
            f"{requested} output tokens and your prompt ")
    tail = " Please reduce the length of the input prompt or the number of requested output tokens."
    if chars:
        return (head + f"contains {prompt_tokens * 5} characters (more than {max_input * 4} characters, which is "
                f"the upper bound for {max_input} input tokens)." + tail)
    count = min(prompt_tokens, max_input + 1)
    qualifier = "at least " if count == max_input + 1 else ""
    total = f"{qualifier}{count + requested}"
    return head + f"contains {qualifier}{count} input tokens, for a total of {total} tokens." + tail


def sse_events(name: str, finish="stop") -> list:
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "policy"}
    deltas = [{"role": "assistant", "content": ""}, {"content": "from "}, {"content": name}]
    events = [dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": None}]) for delta in deltas]
    events.append(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": finish}]))
    events.append(dict(base, choices=[], usage={"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}))
    return [f"data: {json.dumps(event)}\n\n".encode() for event in events] + [b"data: [DONE]\n\n"]


class FakeVLLM:
    """Fake vLLM OpenAI server. `plan` holds the next chat behaviours ("ok", "disconnect", "chars", "stream_break",
    "abort" or an HTTP status); requests are recorded. Over-long prompts get vLLM's real 400. GET /metrics reports
    `running` / `waiting` (vLLM 0.29's Prometheus text) or answers `metrics_status`."""

    def __init__(self, name: str, prompt_tokens=1000):
        self.name, self.prompt_tokens = name, prompt_tokens
        self.plan, self.chats, self.chat_headers, self.tokenizes = [], [], [], []
        self.health_status, self.health_calls, self.aborted = 200, 0, 0
        self.running = self.waiting = self.metrics_calls = 0
        self.metrics_status = 200

    async def start(self):
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.add_routes([web.post("/v1/chat/completions", self.chat), web.post("/tokenize", self.tokenize),
                        web.get("/health", self.health), web.get("/metrics", self.metrics)])
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", 0).start()
        self.port = self.runner.addresses[0][1]
        return self

    async def chat(self, request):
        body = await request.json()
        self.chats.append(body)
        self.chat_headers.append({key.lower() for key in request.headers})
        mode = self.plan.pop(0) if self.plan else "ok"
        if mode == "disconnect":  # the worker dies mid-generation
            await asyncio.sleep(0.05)
            request.transport.close()
            return web.Response()
        if mode == "slow":  # a long generation; vLLM aborts it when the router closes the connection
            for _ in range(200):
                await asyncio.sleep(0.05)
                if request.transport is None or request.transport.is_closing():
                    self.aborted += 1
                    return web.Response()
        if isinstance(mode, int):
            return web.json_response({"error": {"message": f"fake {mode}", "type": "InternalServerError",
                                                "param": None, "code": mode}}, status=mode)
        requested = body.get("max_completion_tokens")
        requested = body["max_tokens"] if requested is None else requested  # vLLM: max_completion_tokens wins
        if self.prompt_tokens + requested > MAX_MODEL_LEN:
            message = overflow_message(self.prompt_tokens, requested, chars=mode == "chars")
            return web.json_response({"error": {"message": message, "type": "BadRequestError",
                                                "param": "input_tokens", "code": 400}}, status=400)
        finish = "abort" if mode == "abort" else "stop"  # vLLM: the engine stopped while generating this call
        if body.get("stream"):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream; charset=utf-8"})
            await response.prepare(request)
            for event in sse_events(self.name, finish):
                await response.write(event)
                if mode == "stream_break":  # the worker dies after the client got the first event
                    await asyncio.sleep(1.0)
                    request.transport.close()
                    return response
                await asyncio.sleep(0.01)
            await response.write_eof()
            return response
        return web.json_response({
            "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": f"from {self.name}"},
                         "finish_reason": finish}],
            "usage": {"prompt_tokens": self.prompt_tokens, "completion_tokens": 3,
                      "total_tokens": self.prompt_tokens + 3}})

    async def tokenize(self, request):
        self.tokenizes.append(await request.json())
        return web.json_response({"count": self.prompt_tokens, "max_model_len": MAX_MODEL_LEN, "tokens": [1, 2, 3],
                                  "token_strs": None})

    async def health(self, request):
        self.health_calls += 1
        return web.Response(status=self.health_status)

    async def metrics(self, request):
        self.metrics_calls += 1
        if self.metrics_status != 200:
            return web.Response(status=self.metrics_status)
        labels = 'engine="0",model_name="policy"'
        return web.Response(text=(
            "# HELP vllm:num_requests_running Number of requests in model execution batches.\n"
            "# TYPE vllm:num_requests_running gauge\n"
            f"vllm:num_requests_running{{{labels}}} {float(self.running)}\n"
            "# HELP vllm:num_requests_waiting Number of requests waiting to be processed.\n"
            "# TYPE vllm:num_requests_waiting gauge\n"
            f"vllm:num_requests_waiting{{{labels}}} {float(self.waiting)}\n"
            f'vllm:num_requests_waiting_by_reason{{{labels},reason="capacity"}} 99.0\n'))


def register(registry: Path, job_id: str, port: int, *, status="ready", heartbeat_age=0.0, end_in=3600.0,
             drain_in=900.0, **extra):
    """Write a contract-schema registry entry (plus `extra` fields) atomically (as worker jobs do)."""
    now = time.time()
    entry = {"job_id": job_id, "slot": "0", "node": "fake-node", "ip": "127.0.0.1", "port": port,
             "base_url": f"http://127.0.0.1:{port}/v1", "served_model_name": "policy", "model_path": "/fake",
             "max_model_len": MAX_MODEL_LEN, "per_call_cap": 131072, "seed": 0, "started_at": now - 600,
             "ready_at": now - 300, "end_epoch": now + end_in, "drain_at": now + drain_in,
             "heartbeat_at": now - heartbeat_age, "status": status, "reason": None, **extra}
    path = registry / f"{job_id}.json"
    temporary = registry / f".{job_id}.json.tmp"
    temporary.write_text(json.dumps(entry))
    os.replace(temporary, path)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.02)


def user(text: str, system="You are OpenCode.") -> dict:
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": text}]}


class Harness(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="student-router-test-"))
        self.registry, self.logs = self.tmp / "registry", self.tmp / "logs"
        self.registry.mkdir()
        self.logs.mkdir()
        self.log_handler = logging.FileHandler(self.logs / "student-router.log")
        LOG.addHandler(self.log_handler)
        LOG.setLevel(logging.INFO)
        self.backends = {name: await FakeVLLM(name).start() for name in ("a", "b", "c")}
        self.http = aiohttp.ClientSession()
        self.runner = None

    async def asyncTearDown(self):
        await self.http.close()
        if self.runner is not None:
            await self.runner.cleanup()
        for backend in self.backends.values():
            await backend.runner.cleanup()
        LOG.removeHandler(self.log_handler)
        self.log_handler.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def register(self, name: str, **options):
        register(self.registry, name, self.backends[name].port, **options)

    async def start_router(self, **options):
        # metrics_s 0: no background /metrics loop; tests that need vLLM load call router.scrape_metrics()
        options = {"refresh_s": 3600.0, "health_s": 3600.0, "metrics_s": 0, "no_worker_wait_s": 1.0, **options}
        self.router = Router(self.registry, self.logs, **options)
        self.router.retry_backoff_s = 0.05
        self.runner, port = await start(self.router, "127.0.0.1", 0)
        self.url = f"http://127.0.0.1:{port}"

    async def get(self, path: str):
        async with self.http.get(self.url + path) as response:
            return response.status, await response.read(), response.headers

    async def chat(self, body: dict, headers=None):
        async with self.http.post(self.url + "/v1/chat/completions", json=body, headers=headers or {}) as response:
            return response.status, await response.read(), response.headers

    async def served_by(self, body: dict, headers=None) -> str:
        status, payload, _ = await self.chat(body, headers)
        self.assertEqual(status, 200, payload)
        return json.loads(payload)["choices"][0]["message"]["content"].removeprefix("from ")

    def journal(self) -> list:
        path = self.logs / "student-router-requests.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def order(self, body: dict) -> list:
        return [worker.job_id for worker in self.router.ranked(affinity_key({}, body))]


class Endpoints(Harness):
    async def test_models_health_and_status(self):
        await self.start_router()
        status, payload, _ = await self.get("/v1/models")
        self.assertEqual((status, json.loads(payload)),
                         (200, {"object": "list",
                                "data": [{"id": "policy", "object": "model", "owned_by": "swe-task"}]}))
        status, payload, _ = await self.get("/health")
        self.assertEqual((status, json.loads(payload)["routable"]), (503, 0))
        self.register("a")
        self.register("b", status="draining")
        self.register("c", status="starting")
        await self.router.refresh()
        status, payload, _ = await self.get("/health")
        health = json.loads(payload)
        self.assertEqual((status, health["routable"], health["ready"], health["draining"], health["known"]),
                         (200, 2, 1, 1, 3))
        status, payload, _ = await self.get("/router/status")
        workers = {worker["job_id"]: worker["routing"] for worker in json.loads(payload)["workers"]}
        self.assertEqual((status, workers), (200, {"a": "ready", "b": "draining", "c": None}))

    async def test_rejects_non_json(self):
        self.register("a")
        await self.start_router()
        async with self.http.post(self.url + "/v1/chat/completions", data=b"not json") as response:
            self.assertEqual(response.status, 400)
            self.assertEqual((await response.json())["error"]["type"], "invalid_request_error")
        status, _, _ = await self.chat([1, 2])
        self.assertEqual(status, 400)
        self.assertEqual(self.backends["a"].chats, [])


class Normalization(Harness):
    async def test_eval_semantics_and_no_credentials(self):
        self.register("a")
        await self.start_router()
        tool_call = [{"id": "t1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]
        body = {
            "model": "nvidia/moonshotai/kimi-k3", "temperature": 0.3, "top_p": 0.5, "reasoning_effort": "max",
            "chat_template_kwargs": {"enable_thinking": False, "custom": 1}, "max_tokens": 500000,
            "max_completion_tokens": 200000, "stream": False, "stream_options": {"include_usage": True},
            "tool_choice": "auto",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "secret task text"},
                {"role": "assistant", "content": "a1", "reasoning": ""},
                {"role": "assistant", "content": "a2", "reasoning_content": ""},
                {"role": "assistant", "content": None, "reasoning_content": "private thought",
                 "tool_calls": tool_call},
                {"role": "tool", "content": "out", "tool_call_id": "t1"},
                {"role": "assistant", "content": "a4", "reasoning": "kept", "reasoning_content": "dup"},
                {"role": "user", "content": "next"},
            ],
            "tools": [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"},
                                                        "strict": True}}],
        }
        headers = {"Authorization": "Bearer sk-secret-1", "api-key": "secret-2", "X-Api-Key": "secret-3",
                   "X-ATP-Attempt-Dir": "/campaign/attempt"}
        status, payload, _ = await self.chat(body, headers)
        self.assertEqual(status, 200, payload)
        expected = json.loads(json.dumps(body))
        expected.update(model="policy", temperature=1.0, top_p=1.0, max_tokens=131072, max_completion_tokens=131072,
                        chat_template_kwargs=TEMPLATE)
        del expected["reasoning_effort"], expected["stream_options"]
        expected["messages"][2] = {"role": "assistant", "content": "a1"}
        expected["messages"][3] = {"role": "assistant", "content": "a2"}
        expected["messages"][4] = {"role": "assistant", "content": None, "reasoning": "private thought",
                                   "tool_calls": tool_call}
        expected["messages"][6] = {"role": "assistant", "content": "a4", "reasoning": "kept"}
        del expected["tools"][0]["function"]["strict"]
        self.assertEqual(self.backends["a"].chats[-1], expected)
        self.assertFalse({"authorization", "api-key", "x-api-key", "x-atp-attempt-dir"}
                         & self.backends["a"].chat_headers[-1])

        await self.chat(user("defaults"))
        self.assertEqual(self.backends["a"].chats[-1]["max_tokens"], 131072)
        self.assertNotIn("max_completion_tokens", self.backends["a"].chats[-1])
        await self.chat(dict(user("small cap"), max_tokens=100))
        self.assertEqual(self.backends["a"].chats[-1]["max_tokens"], 100)

        record = self.journal()[0]
        self.assertEqual((record["status"], record["worker"], record["attempts"], record["prompt_tokens"],
                          record["completion_tokens"], record["finish_reason"]), (200, "a", 1, 1000, 3, "stop"))
        logs = "".join((self.logs / name).read_text()
                       for name in ("student-router-requests.jsonl", "student-router.log"))
        for text in ("sk-secret-1", "secret-2", "secret-3", "secret task text", "private thought"):
            self.assertNotIn(text, logs)

    async def test_tokenize_passthrough(self):
        self.register("a")
        await self.start_router()
        body = {"model": "x", "add_generation_prompt": True,
                "messages": [{"role": "assistant", "content": "c", "reasoning_content": "r"}]}
        async with self.http.post(self.url + "/tokenize", json=body) as response:
            self.assertEqual((response.status, (await response.json())["count"]), (200, 1000))
        self.assertEqual(self.backends["a"].tokenizes[-1], {
            "model": "policy", "add_generation_prompt": True, "chat_template_kwargs": TEMPLATE,
            "messages": [{"role": "assistant", "content": "c", "reasoning": "r"}]})


class ContextOverflow(Harness):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.backend = self.backends["a"]
        self.backend.prompt_tokens = 200000
        self.register("a")
        await self.start_router()
        self.body = {"messages": [{"role": "user", "content": "long task"},
                                  {"role": "assistant", "content": "x", "reasoning_content": "why"},
                                  {"role": "user", "content": "go on"}],
                     "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}, "strict": False}}]}

    async def test_clamps_to_exact_remaining_context(self):
        status, payload, _ = await self.chat(dict(self.body, max_tokens=131072))
        self.assertEqual(status, 200, payload)
        self.assertEqual([call["max_tokens"] for call in self.backend.chats], [131072, 62144])
        self.assertEqual(self.backend.tokenizes, [{
            "model": "policy", "add_generation_prompt": True, "chat_template_kwargs": TEMPLATE,
            "messages": [{"role": "user", "content": "long task"},
                         {"role": "assistant", "content": "x", "reasoning": "why"},
                         {"role": "user", "content": "go on"}],
            "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}]}])
        record = self.journal()[-1]
        self.assertEqual(record["clamps"], [{"prompt_tokens": 200000, "remaining": 62144, "field": "max_tokens",
                                             "applied": True}])
        self.assertEqual((record["status"], record["attempts"]), (200, 1))

    async def test_clamps_max_completion_tokens_and_character_variant(self):
        self.backend.plan = ["chars"]
        status, _, _ = await self.chat(dict(self.body, max_completion_tokens=131072))
        self.assertEqual(status, 200)
        self.assertEqual(self.backend.chats[-1]["max_completion_tokens"], 62144)
        self.assertNotIn("max_tokens", self.backend.chats[-1])
        self.assertEqual(len(self.backend.tokenizes), 1)

    async def test_no_room_returns_the_400_unchanged(self):
        self.backend.prompt_tokens = MAX_MODEL_LEN + 56
        status, payload, _ = await self.chat(self.body)
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(payload)["error"]["message"], overflow_message(MAX_MODEL_LEN + 56, 131072))
        self.assertEqual((len(self.backend.chats), len(self.backend.tokenizes)), (1, 1))
        self.assertEqual(self.journal()[-1]["clamps"], [{"prompt_tokens": MAX_MODEL_LEN + 56, "remaining": -56,
                                                         "field": "max_tokens", "applied": False}])


class Routing(Harness):
    async def test_affinity_is_sticky_and_spreads(self):
        for name in "abc":
            self.register(name)
        await self.start_router()
        conversation = user("task-7")["messages"]
        turn = {"role": "assistant", "content": "earlier turn"}
        later = {await self.served_by({"messages": conversation + [turn] * n}) for n in range(1, 5)}
        self.assertEqual(len(later), 1)  # every later turn of one conversation stays on its worker
        spread = {await self.served_by(user(f"task-{n}")) for n in range(40)}
        self.assertGreaterEqual(len(spread), 2)
        # OpenCode compaction replaces history with fixed messages; the per-conversation summary still spreads them
        compacted = [{"role": "system", "content": "sys"}, {"role": "user", "content": "What did we do so far?"}]
        after = {await self.served_by({"messages": compacted + [{"role": "assistant", "content": f"summary {n}"},
                                                                {"role": "user", "content": "Continue"}]})
                 for n in range(40)}
        self.assertGreaterEqual(len(after), 2)
        pinned = {await self.served_by(user(f"other-{n}"), {"X-Synth-Affinity": "task-7/r0"}) for n in range(5)}
        self.assertEqual(len(pinned), 1)
        parts = [{"type": "text", "text": "h"}, {"type": "text", "text": "i"}]
        self.assertEqual(affinity_key({}, {"messages": [{"role": "user", "content": parts}]}),
                         affinity_key({}, {"messages": [{"role": "user", "content": "hi"}]}))
        owner = await self.served_by(user("task-7"))  # rendezvous: removing another worker keeps the task on its worker
        (self.registry / f"{next(name for name in 'abc' if name != owner)}.json").unlink()
        await self.router.refresh()
        self.assertEqual(await self.served_by(user("task-7")), owner)

    async def test_soft_inflight_cap_moves_calls_to_peers(self):
        self.register("a")
        self.register("b")
        await self.start_router()
        key = affinity_key({}, user("task-1"))
        first = self.router.ranked(key)[0]
        first.inflight = self.router.soft_inflight_cap
        self.assertNotEqual(self.router.ranked(key)[0].job_id, first.job_id)
        first.inflight = 0
        self.assertEqual(self.router.ranked(key)[0].job_id, first.job_id)

    async def test_prefers_ready_over_draining(self):
        self.register("a", status="draining")
        self.register("b")
        self.register("c", drain_in=-1)  # "ready" but past drain_at: counts as draining
        await self.start_router()
        self.assertEqual({await self.served_by(user(f"task-{n}")) for n in range(12)}, {"b"})
        self.register("b", status="stopped")
        await self.router.refresh()
        served = {await self.served_by(user(f"task-{n}")) for n in range(12)}
        self.assertTrue(served and served <= {"a", "c"}, served)

    async def test_stale_or_ended_workers_are_unroutable_then_429(self):
        self.register("a", heartbeat_age=300)
        self.register("b", end_in=-5)
        self.register("c", status="starting")
        await self.start_router(no_worker_wait_s=0.5)
        self.assertEqual(client.routable_workers(self.registry), [])
        self.assertEqual((await self.get("/health"))[0], 503)
        started = time.monotonic()
        status, payload, headers = await self.chat(user("x"))
        self.assertGreaterEqual(time.monotonic() - started, 0.5)
        error = json.loads(payload)["error"]
        self.assertEqual((status, headers["Retry-After"], error["type"], error["code"]),
                         (429, "120", "no_routable_worker", 429))
        self.assertFalse(any(backend.chats for backend in self.backends.values()))
        record = self.journal()[-1]
        self.assertEqual((record["status"], record["attempts"]), (429, 0))
        self.assertGreaterEqual(record["waited_s"], 0.5)
        self.register("a")
        self.assertEqual([worker["job_id"] for worker in client.routable_workers(self.registry)], ["a"])
        self.assertEqual(await self.served_by(user("x")), "a")

    async def test_holds_until_a_worker_registers(self):
        await self.start_router(no_worker_wait_s=30, refresh_s=0.1)
        call = asyncio.create_task(self.chat(user("waiting task")))
        await asyncio.sleep(0.5)
        self.assertFalse(call.done())
        self.register("b")
        status, payload, _ = await asyncio.wait_for(call, 20)
        self.assertEqual(status, 200, payload)
        record = self.journal()[-1]  # held (no worker), then served on the first attempt
        self.assertEqual((record["status"], record["attempts"], record["worker"]), (200, 1, "b"))
        self.assertGreater(record["waited_s"], 0)

    async def test_health_failures_make_a_worker_unroutable_until_success(self):
        self.register("a")
        self.register("b")
        self.backends["a"].health_status = 500
        await self.start_router(health_s=0.05)
        await until(lambda: self.router.workers["a"].routing_status(time.time()) is None)
        self.assertEqual({await self.served_by(user(f"task-{n}")) for n in range(10)}, {"b"})
        self.backends["a"].health_status = 200
        await until(lambda: self.router.workers["a"].routing_status(time.time()) == "ready")


class LoadAware(Harness):
    """Load-aware ranking (2026-10-20): room before saturated before leaving, then ready before draining, then the
    rendezvous weight. Saturated: this router's in-flight >= soft cap, vLLM waiting > 0 or running >= max_num_seqs."""

    async def test_spills_to_draining_workers_when_the_ready_one_hits_the_soft_cap(self):
        self.register("a")
        self.register("b", status="draining")
        self.register("c", status="draining")
        await self.start_router()
        bodies = [user(f"task-{n}") for n in range(12)]
        self.assertEqual({await self.served_by(body) for body in bodies}, {"a"})  # ready before draining
        self.router.workers["a"].inflight = self.router.soft_inflight_cap  # this router's calls fill a
        spilled = [await self.served_by(body) for body in bodies]
        self.assertEqual(set(spilled), {"b", "c"})  # spread over the draining workers by rendezvous weight
        self.assertEqual({(record["moved_from"], record["moved_why"]) for record in self.journal()[-12:]},
                         {("a", "inflight")})
        self.assertEqual((self.router.counters["moved_inflight"], self.router.counters["spilled_to_draining"]),
                         (12, 12))
        self.assertEqual([await self.served_by(body) for body in bodies], spilled)  # sticky among spill targets
        self.router.workers["a"].inflight = 0
        self.assertEqual({await self.served_by(body) for body in bodies}, {"a"})
        self.assertIsNone(self.journal()[-1]["moved_from"])

    async def test_spills_when_vllm_queues_or_runs_a_full_batch(self):
        self.register("a")
        self.register("b", status="draining")
        await self.start_router()
        body, worker = user("task-1"), self.router.workers["a"]
        await self.router.scrape_metrics()
        self.assertEqual(await self.served_by(body), "a")
        self.backends["a"].waiting = 1  # calls of other routers or users queue on a; this router has none there
        await self.router.scrape_metrics()
        self.assertEqual(self.router.saturation(worker), "waiting")
        self.assertEqual(await self.served_by(body), "b")
        self.assertEqual((self.journal()[-1]["moved_from"], self.journal()[-1]["moved_why"]), ("a", "waiting"))
        self.backends["a"].waiting, self.backends["a"].running = 0, 64
        await self.router.scrape_metrics()
        self.assertEqual(await self.served_by(body), "b")
        self.assertEqual(self.journal()[-1]["moved_why"], "running")
        self.backends["a"].running = 63
        await self.router.scrape_metrics()
        self.assertEqual(await self.served_by(body), "a")
        self.assertIsNone(self.journal()[-1]["moved_from"])
        self.register("a", max_num_seqs=8)  # a registry record may set the worker's max_num_seqs
        await self.router.refresh()
        self.backends["a"].running = 8
        await self.router.scrape_metrics()
        self.assertEqual(await self.served_by(body), "b")
        log = (self.logs / "student-router.log").read_text()
        for line in ("a room (running 0/64, waiting 0); b room (running 0/64, waiting 0)",
                     "a saturated (running 0/64, waiting 1)", "a room (running 63/64, waiting 0)",
                     "a saturated (running 8/8, waiting 0)"):
            self.assertIn("vLLM load: " + line, log)

    async def test_stale_vllm_load_counts_as_unknown(self):
        self.register("a")
        self.register("b", status="draining")
        await self.start_router(metrics_s=0.05)  # the background loop scrapes every routable worker
        worker = self.router.workers["a"]
        self.backends["a"].waiting = 5
        await until(lambda: self.router.saturation(worker) == "waiting", timeout=20)
        self.assertEqual(await self.served_by(user("task-1")), "b")
        self.backends["a"].metrics_status = 500  # scrapes fail from now on; the last good sample ages
        await until(lambda: worker.metrics_error == "metrics HTTP 500", timeout=20)
        self.assertEqual(self.router.saturation(worker), "waiting")  # that sample is less than 45 s old
        worker.load_at -= 46  # ... 46 s later
        self.assertIsNone(self.router.saturation(worker))
        self.assertEqual(await self.served_by(user("task-1")), "a")  # unknown load: only the in-flight cap applies
        row = next(row for row in json.loads((await self.get("/router/status"))[1])["workers"] if row["job_id"] == "a")
        self.assertEqual((row["vllm_waiting"], row["metrics_error"], row["saturated"]), (5, "metrics HTTP 500", None))
        self.assertGreater(row["metrics_age_s"], 45)
        await until(lambda: "vLLM load: a unknown (no /metrics sample within 45 s: metrics HTTP 500)"
                    in (self.logs / "student-router.log").read_text(), timeout=20)

    async def test_affinity_unchanged_while_workers_have_room(self):
        self.register("a")
        self.register("b")
        self.register("c", status="draining")
        for backend in self.backends.values():
            backend.running = 10
        await self.start_router()
        await self.router.scrape_metrics()
        conversation, turn = user("task-7")["messages"], {"role": "assistant", "content": "earlier turn"}
        repeats = {await self.served_by(user("task-7")) for _ in range(5)}  # turn 1 of every repeat: one worker
        later = {await self.served_by({"messages": conversation + [turn] * n}) for n in range(1, 5)}
        self.assertEqual((len(repeats), len(later)), (1, 1))
        self.assertEqual({await self.served_by(user(f"task-{n}")) for n in range(40)}, {"a", "b"})
        self.assertFalse([record for record in self.journal() if record["moved_from"]])
        owner = later.pop()
        other = ({"a", "b"} - {owner}).pop()
        body = {"messages": conversation + [turn] * 5}
        self.backends[owner].waiting = 2  # the conversation's worker saturates: the next turn moves to a ready peer
        await self.router.scrape_metrics()
        self.assertEqual(await self.served_by(body), other)
        self.backends[other].waiting = 2  # every ready worker saturated: spill to the draining worker with room
        await self.router.scrape_metrics()
        self.assertEqual(await self.served_by(body), "c")
        self.backends["c"].running = 64  # every worker saturated: the old order (ready first, then weight)
        await self.router.scrape_metrics()
        self.assertEqual(await self.served_by(body), owner)
        self.assertIsNone(self.journal()[-1]["moved_from"])
        for backend in self.backends.values():
            backend.waiting, backend.running = 0, 10
        await self.router.scrape_metrics()
        self.assertEqual(await self.served_by(body), owner)  # back on its worker once that has room again

    async def test_leaving_workers_get_calls_last(self):
        self.register("a")
        self.register("b", status="draining", end_in=300)  # ends within 600 s: leaving
        self.register("c", status="draining")
        await self.start_router()
        self.backends["a"].waiting = 1
        await self.router.scrape_metrics()
        bodies = [user(f"task-{n}") for n in range(12)]
        self.assertEqual({await self.served_by(body) for body in bodies}, {"c"})  # not the ending worker
        self.backends["c"].waiting = 1
        await self.router.scrape_metrics()
        self.assertEqual({await self.served_by(body) for body in bodies}, {"a"})  # saturated before leaving
        self.register("a", status="stopped")
        self.register("c", status="stopped")
        await self.router.refresh()
        self.assertEqual({await self.served_by(body) for body in bodies}, {"b"})  # the last resort still serves
        self.assertEqual({record["moved_why"] for record in self.journal()[-12:]}, {None})
        row = next(row for row in json.loads((await self.get("/router/status"))[1])["workers"] if row["job_id"] == "b")
        self.assertEqual(row["leaving"], "ending")

    async def test_drain_flag_moves_calls_off_at_once(self):
        self.register("a")
        self.register("b")
        await self.start_router()
        bodies = [body for body in (user(f"task-{n}") for n in range(40)) if self.order(body)[0] == "a"][:5]
        self.assertEqual({await self.served_by(body) for body in bodies}, {"a"})
        (self.registry / "a.drain").touch()  # the operator's flag; the worker's record still says "ready"
        await self.router.refresh()
        self.assertEqual(client.drain_requested(self.registry), {"a"})
        self.assertEqual(self.router.workers["a"].routing_status(time.time()), "ready")
        self.assertEqual({await self.served_by(body) for body in bodies}, {"b"})
        self.assertEqual({(record["moved_from"], record["moved_why"]) for record in self.journal()[-5:]},
                         {("a", "drain_flag")})
        (self.registry / "a.drain").unlink()
        await self.router.refresh()
        self.assertEqual({await self.served_by(body) for body in bodies}, {"a"})
        log = (self.logs / "student-router.log").read_text()
        self.assertIn("worker a: drain flag set", log)
        self.assertIn("worker a: drain flag removed", log)

    async def test_never_spills_to_unroutable_workers(self):
        self.register("a")
        self.register("b", status="draining", heartbeat_age=300)  # stale heartbeat
        self.register("c", status="draining")
        self.backends["c"].health_status = 500
        await self.start_router(health_s=0.05)
        await until(lambda: self.router.workers["c"].routing_status(time.time()) is None, timeout=20)
        self.backends["a"].waiting = 30
        await self.router.scrape_metrics()
        self.assertEqual({await self.served_by(user(f"task-{n}")) for n in range(12)}, {"a"})
        self.assertEqual(self.backends["b"].metrics_calls + self.backends["c"].metrics_calls, 0)


class StaleHeartbeat(Harness):
    """2026-10-20 15:22-15:26: a Lustre stall hung the registry writes of all 4 bridges, so their heartbeats went stale
    while /health passed. A router keeps such a worker while its own /health checks of it keep passing."""

    async def start_checked(self, *names, **options):
        for name in names:
            self.register(name)
        await self.start_router(**options)  # health_s 3600: one startup round, then rounds by hand
        await until(lambda: all(self.router.workers[name].healthy_at for name in names), timeout=20)

    async def go_stale(self, *names, **options):
        for name in names:  # the last record that reached the registry before the writes hung
            self.register(name, heartbeat_age=300, **options)
        await self.router.refresh()

    def row(self, payload, name):
        return next(row for row in json.loads(payload)["workers"] if row["job_id"] == name)

    async def test_mass_stale_heartbeats_stay_routable_on_own_health(self):
        await self.start_checked("a", "b")
        await self.go_stale("a", "b")
        self.assertEqual(client.routable_workers(self.registry), [])  # the registry rules alone: nothing routable
        self.assertEqual({await self.served_by(user(f"task-{n}")) for n in range(12)}, {"a", "b"})
        await self.router.check_health()
        self.assertEqual(self.router.counters["stale_heartbeat_kept"], 2)
        payload = (await self.get("/router/status"))[1]
        for name in "ab":
            row = self.row(payload, name)
            self.assertEqual((row["routing"], row["on_own_health"]), ("ready", True))
            self.assertGreaterEqual(row["heartbeat_age_s"], 300)
            self.assertLess(row["health_age_s"], 60)
        log = self.logs / "student-router.log"
        self.assertIn("kept routable on this router's own /health checks", log.read_text())
        self.register("a")  # its writes go through again
        await self.router.refresh()
        await self.router.check_health()
        self.assertFalse(self.router.workers["a"].on_own_health)
        self.assertIn("worker a: no longer routable on its own /health checks", log.read_text())

    async def test_a_dead_worker_with_a_stale_heartbeat_still_drops_out(self):
        await self.start_checked("a", "b")
        await self.go_stale("a", "b")
        self.backends["a"].health_status = 500  # a's vLLM dies during the stall
        for _ in range(3):
            await self.router.check_health()
        self.assertIsNone(self.router.workers["a"].routing_status(time.time()))  # 3 failed checks, as before
        self.assertEqual({await self.served_by(user(f"task-{n}")) for n in range(12)}, {"b"})
        # once its last passing check is older than 60 s the router stops checking it: it stays out even when
        # something answers at that address again, until its registry heartbeat is fresh
        self.router.workers["a"].healthy_at -= client.HEALTH_GRACE_S + 1
        self.backends["a"].health_status = 200
        calls = self.backends["a"].health_calls
        await self.router.check_health()
        self.assertEqual(self.backends["a"].health_calls, calls)
        self.assertIsNone(self.router.workers["a"].routing_status(time.time()))
        self.register("a")
        await self.router.refresh()
        await self.router.check_health()
        self.assertEqual(self.router.workers["a"].routing_status(time.time()), "ready")

    async def test_registry_status_and_end_still_remove_stale_workers(self):
        await self.start_checked("a", "b", "c")
        await self.go_stale("a", "b", "c")
        self.register("a", status="stopped", heartbeat_age=300)
        self.register("b", status="unhealthy", heartbeat_age=300)
        await self.router.refresh()
        self.assertEqual({await self.served_by(user(f"task-{n}")) for n in range(12)}, {"c"})
        self.register("c", heartbeat_age=300, end_in=-1)  # past its end_epoch
        await self.router.refresh()
        self.assertEqual((await self.chat(user("x")))[0], 429)

    async def test_a_new_router_does_not_adopt_stale_records(self):
        self.register("a", heartbeat_age=300)
        await self.start_router(no_worker_wait_s=0.5)
        await self.router.check_health()
        self.assertEqual(self.backends["a"].health_calls, 0)  # never routable here, so never checked
        self.assertEqual((await self.chat(user("x")))[0], 429)


class Failover(Harness):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        for name in "abc":
            self.register(name)
        await self.start_router()
        self.body = user("failover task")
        self.first, self.second, self.third = self.order(self.body)

    async def test_disconnect_and_503_go_to_the_next_worker(self):
        for failure in ("disconnect", 503, 502, 504, 429):
            self.backends[self.first].plan = [failure]
            self.assertEqual(await self.served_by(self.body), self.second, failure)
            record = self.journal()[-1]
            self.assertEqual((record["attempts"], record["workers"], record["worker"]),
                             (2, [self.first, self.second], self.second))

    async def test_500_fails_over_once(self):
        self.backends[self.first].plan = [500]
        self.backends[self.second].plan = [500]
        status, _, _ = await self.chat(self.body)
        self.assertEqual(status, 500)
        self.assertEqual(self.journal()[-1]["workers"], [self.first, self.second])

    async def test_gives_up_after_four_attempts(self):
        self.backends[self.first].plan = [502, 502]
        self.backends[self.second].plan = [502]
        self.backends[self.third].plan = [502]
        status, _, _ = await self.chat(self.body)
        self.assertEqual(status, 502)
        record = self.journal()[-1]
        self.assertEqual((record["attempts"], record["workers"]),
                         (4, [self.first, self.second, self.third, self.first]))

    async def test_abort_is_retried_on_the_next_worker(self):
        self.backends[self.first].plan = ["abort"]  # the worker stopped while generating (drain timeout, scancel)
        self.assertEqual(await self.served_by(self.body), self.second)
        record = self.journal()[-1]
        self.assertEqual((record["status"], record["attempts"], record["aborts"], record["workers"], record["worker"],
                          record["finish_reason"], record["error"]),
                         (200, 2, 1, [self.first, self.second], self.second, "stop", "finish_reason abort"))
        self.assertEqual(self.router.counters["aborts"], 1)
        self.assertIn(f"on worker {self.first} failed: finish_reason abort",
                      (self.logs / "student-router.log").read_text())
        self.assertEqual(self.backends[self.first].chats[-1], self.backends[self.second].chats[-1])

    async def test_abort_retry_does_not_wait_for_a_stalled_registry(self):
        original = router_module.read_registry

        def stalled(registry):  # a registry read hung by a Lustre stall
            time.sleep(5)
            return original(registry)

        router_module.read_registry = stalled
        self.addCleanup(setattr, router_module, "read_registry", original)
        self.backends[self.first].plan = ["abort"]
        started = time.monotonic()
        self.assertEqual(await self.served_by(self.body), self.second)
        self.assertLess(time.monotonic() - started, 3.0)  # the re-read runs in the background

    async def test_every_attempt_aborted_gives_503(self):
        self.backends[self.first].plan = ["abort", "abort"]
        self.backends[self.second].plan = ["abort"]
        self.backends[self.third].plan = ["abort"]
        status, payload, _ = await self.chat(self.body)
        error = json.loads(payload)["error"]
        self.assertEqual((status, error["type"], error["code"]), (503, "upstream_aborted", 503))
        record = self.journal()[-1]
        self.assertEqual((record["status"], record["attempts"], record["aborts"], record["workers"]),
                         (503, 4, 4, [self.first, self.second, self.third, self.first]))

    async def test_client_hangup_closes_the_worker_connection(self):
        self.backends[self.first].plan = ["slow"]
        with self.assertRaises(TimeoutError):
            async with self.http.post(self.url + "/v1/chat/completions", json=self.body,
                                      timeout=aiohttp.ClientTimeout(total=1.5)) as response:
                await response.read()
        await until(lambda: self.backends[self.first].aborted == 1)
        await until(lambda: (self.logs / "student-router-requests.jsonl").exists() and self.journal())
        self.assertEqual((self.journal()[-1]["status"], len(self.backends[self.second].chats)),
                         ("client_disconnected", 0))

    async def test_refused_connections_count_as_health_failures(self):
        register(self.registry, "dead", free_port())
        await self.router.refresh()
        bodies = [user(f"dead-{n}") for n in range(200)]
        bodies = [body for body in bodies if self.order(body)[0] == "dead"][:3]
        self.assertEqual(len(bodies), 3)
        for body in bodies:
            self.assertIn(await self.served_by(body), "abc")
            self.assertEqual(self.journal()[-1]["workers"][0], "dead")
        self.assertIsNone(self.router.workers["dead"].routing_status(time.time()))


class Streaming(Harness):
    async def stream(self, body: dict):
        async with self.http.post(self.url + "/v1/chat/completions", json=body) as response:
            return response.status, response.headers.get("Content-Type", ""), await response.read()

    async def test_passes_sse_through(self):
        self.register("a")
        await self.start_router()
        body = dict(user("stream please"), stream=True, stream_options={"include_usage": True})
        status, content_type, data = await self.stream(body)
        self.assertEqual((status, content_type.split(";")[0], data),
                         (200, "text/event-stream", b"".join(sse_events("a"))))
        sent = self.backends["a"].chats[-1]
        self.assertEqual((sent["stream"], sent["stream_options"], sent["max_tokens"]),
                         (True, {"include_usage": True}, 131072))
        record = self.journal()[-1]
        self.assertEqual((record["stream"], record["prompt_tokens"], record["completion_tokens"],
                          record["finish_reason"]),
                         (True, 11, 3, "stop"))

    async def test_fails_over_before_the_first_byte_only(self):
        for name in "ab":
            self.register(name)
        await self.start_router()
        body = dict(user("stream failover"), stream=True)
        first, second = self.order(body)
        self.backends[first].plan = ["disconnect"]
        status, _, data = await self.stream(body)
        self.assertEqual((status, data), (200, b"".join(sse_events(second))))
        self.backends[first].plan = ["stream_break"]
        status, _, data = await self.stream(body)
        self.assertEqual(status, 200)
        self.assertTrue(data.startswith(sse_events(first)[0]))
        self.assertIn(b'"upstream_stream_error"', data)
        self.assertNotIn(b"[DONE]", data)
        self.assertTrue(self.journal()[-1]["error"].startswith("stream aborted"))

    async def test_streamed_abort_is_passed_through(self):
        for name in "ab":
            self.register(name)
        await self.start_router()
        body = dict(user("stream abort"), stream=True)
        first, second = self.order(body)
        self.backends[first].plan = ["abort"]
        status, _, data = await self.stream(body)
        self.assertEqual((status, data), (200, b"".join(sse_events(first, "abort"))))
        self.assertEqual(self.backends[second].chats, [])
        record = self.journal()[-1]
        self.assertEqual((record["attempts"], record["aborts"], record["finish_reason"]), (1, 0, "abort"))


class Client(Harness):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.saved = os.environ.pop(client.BASE_URL_ENV, None)

    async def asyncTearDown(self):
        await asyncio.to_thread(client.stop_router)
        os.environ.pop(client.BASE_URL_ENV, None)
        if self.saved is not None:
            os.environ[client.BASE_URL_ENV] = self.saved
        await super().asyncTearDown()

    async def test_ensure_router_starts_once_and_routes(self):
        url = await asyncio.to_thread(client.ensure_router, registry=self.registry, log_dir=self.logs)
        self.assertRegex(url, r"^http://127\.0\.0\.1:\d+/v1$")
        self.assertEqual(os.environ[client.BASE_URL_ENV], url)
        process = client._router[0]
        self.assertEqual(await asyncio.to_thread(client.ensure_router, registry=self.registry, log_dir=self.logs), url)
        self.assertIs(client._router[0], process)
        lines = []
        with self.assertRaises(TimeoutError):
            await asyncio.to_thread(client.wait_for_workers, 1, 0.3, self.registry, 0.1, lines.append)
        self.register("a")
        workers = await asyncio.to_thread(client.wait_for_workers, 1, 30, self.registry, 0.1, lines.append)
        self.assertEqual([worker["job_id"] for worker in workers], ["a"])
        self.assertTrue(lines)
        async with self.http.post(url + "/chat/completions", json=user("via subprocess"),
                                  headers={"Authorization": "Bearer sk-secret-9"}) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual((await response.json())["choices"][0]["message"]["content"], "from a")
        self.assertEqual(json.loads((self.logs / "student-router-requests.jsonl").read_text())["worker"], "a")
        self.assertIn("listening on", (self.logs / "student-router.log").read_text())
        await asyncio.to_thread(client.stop_router)
        self.assertEqual(process.poll(), 0)
        self.assertIn("shutting down", (self.logs / "student-router.log").read_text())
        self.assertNotIn(client.BASE_URL_ENV, os.environ)

    async def test_router_exits_when_its_parent_dies(self):
        script = (f"import os, sys; sys.path.insert(0, {str(Path(__file__).absolute().parents[2])!r}); "
                  f"from student import client; client.ensure_router(registry={str(self.registry)!r}, "
                  f"log_dir={str(self.logs)!r}); print(client._router[0].pid, flush=True); os._exit(0)")
        parent = await asyncio.create_subprocess_exec(sys.executable, "-B", "-c", script,
                                                      stdout=asyncio.subprocess.PIPE)
        pid = int((await parent.communicate())[0])

        def gone() -> bool:
            try:
                return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"
            except (OSError, IndexError):
                return True

        await until(gone, timeout=20)
        self.assertIn("parent process", (self.logs / "student-router.log").read_text())

    async def test_reuses_a_live_router_from_the_environment(self):
        await self.start_router()
        os.environ[client.BASE_URL_ENV] = self.url + "/v1"
        self.assertEqual(await asyncio.to_thread(client.ensure_router, registry=self.registry, log_dir=self.logs),
                         self.url + "/v1")
        self.assertIsNone(client._router)


class Helpers(unittest.TestCase):
    def test_vllm_load_parses_prometheus_text(self):
        text = ('# HELP vllm:num_requests_running Number of requests in model execution batches.\n'
                '# TYPE vllm:num_requests_running gauge\n'
                'vllm:num_requests_running{engine="0",model_name="policy"} 7.0\n'
                'vllm:num_requests_running{engine="1",model_name="policy"} 5.0\n'
                'vllm:num_requests_running{engine="0",model_name="other"} 40.0\n'
                'vllm:num_requests_waiting{engine="0",model_name="policy"} 2.0 1792531458000\n'
                'vllm:num_requests_waiting_by_reason{engine="0",model_name="policy",reason="capacity"} 9.0\n'
                'vllm:num_requests_swapped{engine="0",model_name="policy"} 3.0\n')
        self.assertEqual(vllm_load(text), (12, 2))
        self.assertIsNone(vllm_load(text, "other"))  # no waiting gauge for that model
        self.assertEqual(vllm_load("vllm:num_requests_running 3\nvllm:num_requests_waiting 0\n"), (3, 0))
        for broken in ("", "vllm:num_requests_running 3\n",
                       "vllm:num_requests_running NaN\nvllm:num_requests_waiting 0",
                       "vllm:num_requests_running{model_name=\"policy\"} x\nvllm:num_requests_waiting 0"):
            self.assertIsNone(vllm_load(broken), broken)

    def test_aborted(self):
        def body(finish, content="x"):
            return json.dumps({"choices": [{"index": 0, "message": {"content": content},
                                            "finish_reason": finish}]}).encode()

        self.assertTrue(aborted(body("abort")))
        for text in (body("stop"), body("stop", 'say "abort"'), body("length", "abort"), b'not json "abort"',
                     b'{"choices": "abort"}'):
            self.assertFalse(aborted(text), text)

    def test_worker_status_with_own_health(self):
        now = 10000.0
        entry = {"status": "ready", "heartbeat_at": now - 300, "end_epoch": now + 3600, "drain_at": now + 900,
                 "base_url": "http://10.0.0.1:8000/v1"}
        self.assertIsNone(client.worker_status(entry, now))
        self.assertEqual(client.worker_status(entry, now, healthy_at=now - 59), "ready")
        self.assertIsNone(client.worker_status(entry, now, healthy_at=now - 61))
        self.assertEqual(client.worker_status(dict(entry, drain_at=now - 1), now, healthy_at=now), "draining")
        for change in ({"status": "stopped"}, {"status": "unhealthy"}, {"end_epoch": now - 1}, {"heartbeat_at": None},
                       {"base_url": None}):
            self.assertIsNone(client.worker_status(dict(entry, **change), now, healthy_at=now), change)

    def test_client_load_rules(self):
        now = 1000.0
        self.assertEqual((client.max_num_seqs({}), client.max_num_seqs({"max_num_seqs": 8})), (64, 8))
        for bad in (0, -1, True, "8", 8.5, None):
            self.assertEqual(client.max_num_seqs({"max_num_seqs": bad}), 64)
        self.assertIsNone(client.leaving({"end_epoch": now + 601}, now))
        self.assertIsNone(client.leaving({}, now))
        self.assertEqual(client.leaving({"end_epoch": now + 599}, now), "ending")
        self.assertEqual(client.leaving({"end_epoch": now + 3600}, now, drain_flag=True), "drain_flag")
        with tempfile.TemporaryDirectory() as directory:
            for name in ("1.json", "1.drain", ".2.drain", "3.drain", "4.json.1.tmp"):
                (Path(directory) / name).touch()
            self.assertEqual(client.drain_requested(directory), {"1", "3"})
            self.assertEqual(client.drain_requested(Path(directory) / "missing"), set())


if __name__ == "__main__":
    unittest.main()
