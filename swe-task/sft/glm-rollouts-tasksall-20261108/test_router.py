#!/usr/bin/env python3
"""Offline test of router.py with fake vLLM backends (no GPUs, no Slurm). Run by explicit path:
    atp/runtime/controller/bin/python3 -B test_router.py
Checks: session affinity through tool-call ids, least-loaded placement of new sessions, failover when a backend dies
mid-call (the call is re-sent and its session moves), 4xx pass-through, 429 + Retry-After when no backend is up.
"""
import asyncio
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import uuid

import aiohttp
from aiohttp import web

HERE = Path(__file__).resolve().parent
PY = sys.executable


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Fake:
    def __init__(self, name):
        self.name, self.port = name, free_port()
        self.served = []          # (session tag, call index)
        self.delay = 0.05
        self.die_next = False     # close the connection mid-call
        self.healthy = True
        self.runner = None

    async def chat(self, request):
        body = await request.json()
        tag = body.get("tag")
        if body.get("bad"):
            return web.json_response({"error": "bad request"}, status=400)
        if self.die_next:
            self.die_next = False
            await asyncio.sleep(0.2)
            request.transport.close()
            await asyncio.sleep(1)
            return web.Response(status=200)
        await asyncio.sleep(self.delay)
        self.served.append(tag)
        call_id = "call-" + uuid.uuid4().hex[:10]
        return web.json_response({"choices": [{"message": {"role": "assistant", "content": f"{self.name}",
                                                           "tool_calls": [{"id": call_id, "type": "function",
                                                                           "function": {"name": "bash",
                                                                                        "arguments": "{}"}}]}}],
                                  "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    async def health(self, request):
        return web.Response(status=200 if self.healthy else 503)

    async def metrics(self, request):
        return web.Response(text='vllm:num_requests_running{engine="0"} 0.0\nvllm:num_requests_waiting{engine="0"} 0.0\n')

    async def models(self, request):
        return web.json_response({"data": [{"id": "glm-5.3", "max_model_len": 262144}]})

    async def start(self):
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.chat)
        app.router.add_get("/health", self.health)
        app.router.add_get("/metrics", self.metrics)
        app.router.add_get("/v1/models", self.models)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", self.port).start()

    async def stop(self):
        await self.runner.cleanup()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/v1"


async def call(http, router_url, tag, history, **extra):
    body = {"model": "glm-5.3", "tag": tag, "messages": [{"role": "user", "content": "task"}] + history, **extra}
    async with http.post(router_url + "/chat/completions", json=body) as response:
        status = response.status
        data = await response.json(content_type=None)
        return status, data, response.headers


async def session(http, router_url, tag, turns):
    history = []
    for _ in range(turns):
        status, data, _ = await call(http, router_url, tag, history)
        assert status == 200, (status, data)
        message = data["choices"][0]["message"]
        history = history + [message, {"role": "tool", "tool_call_id": message["tool_calls"][0]["id"], "content": "ok"}]
    return history


async def main():
    work = Path(tempfile.mkdtemp(prefix="router-test-", dir=os.environ.get("TEST_SCRATCH", str(HERE / "logs"))))
    a, b = Fake("A"), Fake("B")
    await a.start()
    await b.start()
    port = free_port()
    log = work / "router.jsonl"
    proc = await asyncio.create_subprocess_exec(
        PY, "-B", str(HERE / "router.py"), "--port", str(port), "--job-prefix", "", "--backend", a.url, "--backend", b.url,
        "--log", str(log), "--health-seconds", "1", "--health-failures", "2", "--discover-seconds", "1",
        "--wait-seconds", "3", "--retry-after", "7")
    router_url = f"http://127.0.0.1:{port}/v1"
    try:
        async with aiohttp.ClientSession() as http:
            for _ in range(50):
                try:
                    async with http.get(f"http://127.0.0.1:{port}/health") as response:
                        if (await response.json())["healthy_backends"] == 2:
                            break
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(0.2)
            # 1. affinity + spread: 8 sessions x 6 turns
            await asyncio.gather(*(session(http, router_url, f"s{i}", 6) for i in range(8)))
            per_tag = {}
            for fake in (a, b):
                for tag in fake.served:
                    per_tag.setdefault(tag, set()).add(fake.name)
            assert all(len(names) == 1 for names in per_tag.values()), per_tag
            spread = {fake.name: len({t for t in fake.served}) for fake in (a, b)}
            assert min(spread.values()) >= 2, spread
            print("affinity OK, sessions per backend", spread)
            # 2. failover mid-call: the backend of session "f" dies during its 3rd call
            a.served.clear(); b.served.clear()
            history = await session(http, router_url, "f", 2)
            owner = a if "f" in a.served else b
            other = b if owner is a else a
            owner.die_next = True
            status, data, _ = await call(http, router_url, "f", history)
            assert status == 200 and data["choices"][0]["message"]["content"] == other.name, (status, data)
            message = data["choices"][0]["message"]
            history += [message, {"role": "tool", "tool_call_id": message["tool_calls"][0]["id"], "content": "ok"}]
            status, data, _ = await call(http, router_url, "f", history)
            assert data["choices"][0]["message"]["content"] == other.name, "session did not stay on its new backend"
            print("failover OK:", owner.name, "->", other.name)
            # 3. 4xx passes through
            status, data, _ = await call(http, router_url, "bad", [], bad=True)
            assert status == 400, status
            print("4xx pass-through OK")
            # 4. no backend -> 429 + Retry-After after --wait-seconds
            a.healthy = b.healthy = False
            await asyncio.sleep(3)
            status, data, headers = await call(http, router_url, "late", [])
            assert status == 429 and headers.get("Retry-After") == "7", (status, data, dict(headers))
            print("no-backend 429 OK")
            # 5. a backend that comes back is used again
            b.healthy = True
            await asyncio.sleep(2.5)
            status, data, _ = await call(http, router_url, "back", [])
            assert status == 200 and data["choices"][0]["message"]["content"] == "B", (status, data)
            print("recovery OK")
            async with http.get(f"http://127.0.0.1:{port}/router/status") as response:
                print(json.dumps(await response.json())[:600])
    finally:
        proc.terminate()
        await proc.wait()
        await a.stop()
        await b.stop()
    events = [json.loads(line) for line in log.read_text().splitlines()]
    print("events:", {k: sum(e["event"] == k for e in events) for k in sorted({e["event"] for e in events})})
    print("ALL ROUTER TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
