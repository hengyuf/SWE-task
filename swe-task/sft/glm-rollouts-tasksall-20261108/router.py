#!/usr/bin/env python3
"""OpenAI-compatible router in front of several GLM-5.3 vLLM servers (glm53-serve 2-node jobs), one per campaign leg.

ATP's Gym model server (in the same CPU job) calls http://127.0.0.1:<port>/v1; the router forwards each call to one
backend and returns the backend's answer unchanged.

* Backends: RUNNING Slurm jobs of this user whose name starts with --job-prefix (squeue) and whose
  glm53-serve/logs/serve/<job>.status says "ready <url>" (and nothing after it), plus --backend URLs (tests) and the
  URLs listed in --extra-file (operator-managed, re-read every pass; removing a line removes the backend).
  /health every 10 s (3 consecutive failures = down; its in-flight calls are cancelled and re-sent elsewhere) and
  /metrics (vllm:num_requests_running + waiting) for the load of every backend, which all routers share.
* Session affinity (prefix cache): a call that carries tool-call ids (assistant tool_calls / tool tool_call_id) of an
  earlier answer belongs to that answer's session and goes to the session's backend. Calls without known ids start a
  new session on the least-loaded backend that has >= --drain-minutes of job time left. A session leaves its backend
  when the backend is down or has < --migrate-minutes left (the next call re-prefills elsewhere).
* Failures: a connection error, 5xx or cancelled call is re-sent to another backend (the answer was never delivered,
  so nothing is executed twice), up to --attempts backends. 4xx answers are returned as they are.
* Load cap: a NEW session waits in the router while every backend has >= --max-new-load running + waiting requests,
  for at most --cap-wait-seconds (then it goes to the least-loaded backend); existing sessions are never held.
* No backend: the call waits up to --wait-seconds for one, then gets 429 + Retry-After, which ATP retries until the
  rollout's deadline (a 5xx would exhaust ATP's 6 retries in about a minute; ATP's 429 cooldown pauses all calls of
  the campaign, which is what we want when no backend exists).
Logs one JSON line per call to --log; GET /router/status shows backends and sessions.
"""
import argparse
import asyncio
import collections
import json
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import time
import uuid

import aiohttp
from aiohttp import web

SERVE_LOGS = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/logs/serve")
METRIC = re.compile(r"^(vllm:num_requests_running|vllm:num_requests_waiting|vllm:kv_cache_usage_perc)\{[^}]*\}\s+([0-9.eE+-]+)$")
HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "content-length", "host"}
REQUEST_DROP = HOP_HEADERS | {"accept-encoding"}  # backends answer uncompressed; bodies pass through as bytes


class Backend:
    def __init__(self, url, job=None):
        self.url = url.rstrip("/")
        self.job = job
        self.time_left = None   # seconds of Slurm time left (None = unknown / static)
        self.fails = 0
        self.healthy = False
        self.seen_ok = False
        self.running = 0.0
        self.waiting = 0.0
        self.kv = 0.0
        self.recent_new = 0     # new sessions sent here since the last /metrics read
        self.inflight = set()   # forwarding tasks
        self.calls = 0
        self.errors = 0

    def load(self):
        return self.running + 2.0 * self.waiting + self.recent_new + 0.5 * len(self.inflight)

    def accepts_new(self, drain_s):
        return self.healthy and (self.time_left is None or self.time_left >= drain_s)

    def keeps_sessions(self, migrate_s):
        return self.healthy and (self.time_left is None or self.time_left >= migrate_s)

    def status(self):
        return {"url": self.url, "job": self.job, "healthy": self.healthy, "time_left_s": self.time_left,
                "running": self.running, "waiting": self.waiting, "kv": round(self.kv, 3),
                "inflight_here": len(self.inflight), "calls_here": self.calls, "errors_here": self.errors}


class Router:
    def __init__(self, args):
        self.args = args
        self.backends = {}                      # url -> Backend
        self.static = [u.rstrip("/") for u in args.backend]
        self.session_backend = {}               # session -> url
        self.session_seen = {}                  # session -> last activity
        self.id_session = {}                    # tool-call id -> session
        self.counts = collections.Counter()
        self.log = open(args.log, "a", buffering=1) if args.log else None
        self.http = None
        self.started = time.time()
        self.capped = False

    # ---------------------------------------------------------------- backends
    def discover(self):
        found = {}
        for url in self.static:
            found[url] = (None, None)
        if self.args.extra_file:  # operator-managed backends (e.g. a dev server), re-read every discovery pass
            try:
                for line in Path(self.args.extra_file).read_text().splitlines():
                    line = line.split("#", 1)[0].strip()
                    if line.startswith("http"):
                        found[line.rstrip("/")] = ("extra", None)
            except OSError:
                pass
        if self.args.job_prefix:
            try:
                out = subprocess.run(["squeue", "--me", "-h", "-o", "%i|%j|%T|%L"], capture_output=True, text=True,
                                     timeout=60).stdout
            except (OSError, subprocess.SubprocessError):
                return None  # keep the current set
            for line in out.splitlines():
                parts = line.strip().split("|")
                if len(parts) != 4 or parts[2] != "RUNNING" or not parts[1].startswith(self.args.job_prefix):
                    continue
                job, left = parts[0], parse_left(parts[3])
                status = SERVE_LOGS / f"{job}.status"
                try:
                    lines = status.read_text().splitlines()
                except OSError:
                    continue
                url = None
                for entry in lines:
                    if entry.startswith("ready "):
                        fields = entry.split()
                        url = fields[2] if len(fields) > 2 else None
                    elif entry.split(" ", 1)[0] in ("exited", "failed", "unhealthy"):
                        url = None
                if url:
                    found[url.rstrip("/")] = (job, left)
        return found

    async def discovery_loop(self):
        while True:
            found = await asyncio.to_thread(self.discover)
            if found is not None:
                for url, (job, left) in found.items():
                    backend = self.backends.get(url)
                    if backend is None:
                        backend = self.backends[url] = Backend(url, job)
                        self.event("backend_added", url=url, job=job, time_left_s=left)
                    backend.job, backend.time_left = job, left
                for url in list(self.backends):
                    if url not in found:
                        backend = self.backends.pop(url)
                        backend.healthy = False
                        self.event("backend_removed", url=url, job=backend.job)
                        for task in list(backend.inflight):
                            task.cancel()
            await asyncio.sleep(self.args.discover_seconds)

    async def health_loop(self):
        while True:
            await asyncio.gather(*(self.check(b) for b in list(self.backends.values())), return_exceptions=True)
            await asyncio.sleep(self.args.health_seconds)

    async def check(self, backend):
        base = backend.url[:-3] if backend.url.endswith("/v1") else backend.url
        ok = False
        try:
            async with self.http.get(base + "/health", timeout=aiohttp.ClientTimeout(total=8)) as response:
                ok = response.status == 200
            if ok:
                async with self.http.get(base + "/metrics", timeout=aiohttp.ClientTimeout(total=8)) as response:
                    text = await response.text()
                running = waiting = 0.0
                kv = 0.0
                for line in text.splitlines():
                    match = METRIC.match(line)
                    if not match:
                        continue
                    name, value = match.group(1), float(match.group(2))
                    if name.endswith("running"):
                        running += value
                    elif name.endswith("waiting"):
                        waiting += value
                    else:
                        kv = max(kv, value)
                backend.running, backend.waiting, backend.kv, backend.recent_new = running, waiting, kv, 0
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError):
            ok = False
        if ok:
            if not backend.healthy:
                self.event("backend_up", url=backend.url, job=backend.job, time_left_s=backend.time_left)
            backend.fails, backend.healthy, backend.seen_ok = 0, True, True
        else:
            backend.fails += 1
            if backend.healthy and backend.fails >= self.args.health_failures:
                backend.healthy = False
                self.event("backend_down", url=backend.url, job=backend.job, inflight=len(backend.inflight))
                for task in list(backend.inflight):
                    task.cancel()

    def choose_new(self, exclude=(), ignore_cap=False):
        drain = self.args.drain_minutes * 60
        candidates = [b for b in self.backends.values() if b.accepts_new(drain) and b.url not in exclude]
        if not candidates:  # every backend is draining: still better than waiting
            migrate = self.args.migrate_minutes * 60
            candidates = [b for b in self.backends.values() if b.keeps_sessions(migrate) and b.url not in exclude]
        self.capped = False
        if not ignore_cap:  # soft cap: no NEW session on a backend this busy (all routers share /metrics)
            under = [b for b in candidates if b.running + b.waiting + b.recent_new < self.args.max_new_load]
            self.capped = bool(candidates) and not under
            candidates = under
        if not candidates:
            return None
        best = min(b.load() for b in candidates)
        backend = random.choice([b for b in candidates if b.load() <= best + 0.5])
        backend.recent_new += 1
        return backend

    # ---------------------------------------------------------------- sessions
    @staticmethod
    def call_ids(payload):
        ids = []
        for message in payload.get("messages") or []:
            if not isinstance(message, dict):
                continue
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict) and call.get("id"):
                    ids.append(call["id"])
            if message.get("tool_call_id"):
                ids.append(message["tool_call_id"])
        return ids

    def session_of(self, payload):
        for call_id in reversed(self.call_ids(payload)):
            session = self.id_session.get(call_id)
            if session:
                return session, False
        return "s-" + uuid.uuid4().hex[:12], True

    def remember(self, session, body):
        try:
            data = json.loads(body)
        except ValueError:
            return
        for choice in data.get("choices") or []:
            message = choice.get("message") or {}
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict) and call.get("id"):
                    self.id_session[call["id"]] = session

    def backend_for(self, session, new, exclude, ignore_cap=False):
        migrate = self.args.migrate_minutes * 60
        url = self.session_backend.get(session)
        current = self.backends.get(url) if url else None
        if current is not None and current.keeps_sessions(migrate) and current.url not in exclude:
            return current, False
        backend = self.choose_new(exclude, ignore_cap)
        if backend is not None:
            self.session_backend[session] = backend.url
        return backend, current is not None or (url is not None)

    # ---------------------------------------------------------------- forwarding
    async def forward(self, backend, method, path, headers, body):
        async with self.http.request(method, backend.url + path, data=body, headers=headers) as response:
            payload = await response.read()
            return response.status, {k: v for k, v in response.headers.items() if k.lower() not in HOP_HEADERS}, payload

    async def handle(self, request):
        started = time.time()
        body = await request.read()
        path = request.path[3:] if request.path.startswith("/v1") else request.path
        payload = None
        if request.method == "POST" and body:
            try:
                payload = json.loads(body)
            except ValueError:
                payload = None
        session, new = self.session_of(payload) if isinstance(payload, dict) else ("anon-" + uuid.uuid4().hex[:8], True)
        headers = {k: v for k, v in request.headers.items() if k.lower() not in REQUEST_DROP}
        tried, attempt, waited, migrated, capped_wait = [], 0, 0.0, False, 0.0
        while True:
            backend, moved = self.backend_for(session, new, tried, ignore_cap=capped_wait >= self.args.cap_wait_seconds)
            migrated = migrated or moved
            if backend is None and self.capped:
                # Every backend is busy: hold this new session here (a 429 would make ATP pause ALL calls of the
                # campaign); after --cap-wait-seconds it goes to the least-loaded backend anyway.
                if capped_wait == 0:
                    self.counts["capped_waits"] += 1
                await asyncio.sleep(5)
                capped_wait += 5
                continue
            if backend is None:
                if waited >= self.args.wait_seconds:
                    self.counts["no_backend_429"] += 1
                    self.call_log(session, None, 429, started, attempt, tried, waited, payload, None, migrated)
                    return web.json_response({"error": {"message": "no GLM-5.3 backend available; retry later",
                                                        "type": "no_backend"}}, status=429,
                                             headers={"Retry-After": str(self.args.retry_after)})
                await asyncio.sleep(5)
                waited += 5
                if len(tried) >= len(self.backends):
                    tried = []  # every backend failed once: allow them again
                continue
            attempt += 1
            backend.calls += 1
            task = asyncio.ensure_future(self.forward(backend, request.method, path, headers, body))
            backend.inflight.add(task)
            try:
                status, out_headers, out = await asyncio.shield(task)
            except asyncio.CancelledError:
                if not task.cancelled():  # the client went away
                    task.cancel()
                    raise
                status, out_headers, out = None, None, b"backend cancelled"
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as error:
                status, out_headers, out = None, None, f"{type(error).__name__}: {error}".encode()
            finally:
                backend.inflight.discard(task)
            if status is not None and status < 500:
                if status == 200 and isinstance(payload, dict):
                    self.remember(session, out)
                self.session_backend[session] = backend.url
                self.session_seen[session] = time.time()
                self.counts[f"status_{status}"] += 1
                self.call_log(session, backend, status, started, attempt, tried, waited, payload, out, migrated)
                return web.Response(status=status, body=out, headers=out_headers)
            backend.errors += 1
            self.counts["retried_failures"] += 1
            self.event("call_failed", session=session, url=backend.url, job=backend.job, status=status,
                       error=out[:300].decode("utf-8", "replace") if isinstance(out, bytes) else None, attempt=attempt)
            tried.append(backend.url)
            if self.session_backend.get(session) == backend.url:
                self.session_backend.pop(session, None)
            if attempt >= self.args.attempts:
                self.counts["gave_up_503"] += 1
                self.call_log(session, backend, status or 503, started, attempt, tried, waited, payload, None, migrated)
                if status is not None:
                    return web.Response(status=status, body=out, headers=out_headers)
                return web.json_response({"error": {"message": "all backends failed", "type": "backend_error"}},
                                         status=503)

    async def models(self, request):
        for backend in self.backends.values():
            if backend.healthy:
                try:
                    status, headers, out = await self.forward(backend, "GET", "/models", {}, None)
                    return web.Response(status=status, body=out, headers=headers)
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                    continue
        return web.json_response({"error": {"message": "no backend", "type": "no_backend"}}, status=503)

    async def status(self, request):
        now = time.time()
        active = collections.Counter(url for session, url in self.session_backend.items()
                                     if now - self.session_seen.get(session, 0) < 900)
        return web.json_response({"uptime_s": round(now - self.started), "backends": [
            dict(b.status(), active_sessions_here=active.get(b.url, 0)) for b in self.backends.values()],
            "sessions": len(self.session_backend), "known_tool_call_ids": len(self.id_session),
            "counts": dict(self.counts)})

    async def health(self, request):
        return web.json_response({"ok": True, "healthy_backends": sum(b.healthy for b in self.backends.values())})

    # ---------------------------------------------------------------- logging
    def event(self, kind, **fields):
        if self.log:
            self.log.write(json.dumps({"t": round(time.time(), 3), "event": kind, **fields}) + "\n")

    def call_log(self, session, backend, status, started, attempt, tried, waited, payload, out, migrated):
        usage = {}
        if out:
            try:
                usage = (json.loads(out).get("usage") or {})
            except (ValueError, AttributeError):
                usage = {}
        self.event("call", session=session, url=backend.url if backend else None,
                   job=backend.job if backend else None, status=status, seconds=round(time.time() - started, 2),
                   attempts=attempt, failed_on=tried or None, waited_s=waited or None, migrated=migrated or None,
                   messages=len(payload.get("messages") or []) if isinstance(payload, dict) else None,
                   prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"))


def parse_left(text):
    """squeue %L (D-HH:MM:SS, HH:MM:SS, MM:SS, UNLIMITED) -> seconds."""
    text = text.strip()
    if not text or text in ("UNLIMITED", "NOT_SET", "INVALID"):
        return None
    days = 0
    if "-" in text:
        day, text = text.split("-", 1)
        days = int(day)
    parts = [int(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def keepalive_socket(addr_info):
    family, type_, proto, _, _ = addr_info
    sock = socket.socket(family=family, type=type_, proto=proto)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for option, value in (("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 20), ("TCP_KEEPCNT", 3)):
        if hasattr(socket, option):
            sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, option), value)
    return sock


async def main_async(args):
    router = Router(args)
    connector = aiohttp.TCPConnector(limit=0, socket_factory=keepalive_socket)
    router.http = aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(
        total=None, connect=20, sock_connect=20, sock_read=None), auto_decompress=False)
    app = web.Application(client_max_size=512 * 1024 * 1024)
    app.router.add_get("/health", router.health)
    app.router.add_get("/router/status", router.status)
    app.router.add_get("/v1/models", router.models)
    app.router.add_route("*", "/{tail:.*}", router.handle)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    router.event("router_start", host=args.host, port=args.port, pid=os.getpid(), job=os.environ.get("SLURM_JOB_ID"),
                 job_prefix=args.job_prefix, static=router.static)
    print(f"router listening on http://{args.host}:{args.port}/v1 (job prefix {args.job_prefix!r}, "
          f"static {router.static})", flush=True)
    await asyncio.gather(router.discovery_loop(), router.health_loop())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--job-prefix", default="glm53-serve-roll", help="Slurm job-name prefix of the backends")
    parser.add_argument("--backend", action="append", default=[], help="extra static backend URL (.../v1)")
    parser.add_argument("--extra-file", default=str(Path(__file__).resolve().parent / "backends-extra.txt"),
                        help="file of extra backend URLs (one per line), re-read every discovery pass")
    parser.add_argument("--log", help="JSONL call/event log")
    parser.add_argument("--discover-seconds", type=float, default=20)
    parser.add_argument("--health-seconds", type=float, default=10)
    parser.add_argument("--health-failures", type=int, default=3)
    parser.add_argument("--drain-minutes", type=float, default=45, help="no NEW sessions below this job time left")
    parser.add_argument("--migrate-minutes", type=float, default=12, help="sessions move away below this time left")
    parser.add_argument("--attempts", type=int, default=4, help="backends tried per call")
    parser.add_argument("--max-new-load", type=float, default=30,
                        help="new sessions only go to backends with running + waiting + just-assigned below this")
    parser.add_argument("--cap-wait-seconds", type=float, default=480,
                        help="a new session held by --max-new-load goes to the least-loaded backend after this")
    parser.add_argument("--wait-seconds", type=float, default=120, help="wait for a backend before answering 429")
    parser.add_argument("--retry-after", type=int, default=60)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
