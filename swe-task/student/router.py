"""Student router: this job's OpenAI endpoint for the student ("sc14", served as "policy") in front of the vLLM fleet.

ATP's Gym policy model and llm_engine.chat() call http://127.0.0.1:<port>/v1/chat/completions; each call goes to
one vLLM worker of the registry (student/registry/<job_id>.json, routing rules in client.py).

Every call gets the serving semantics of hengyuf's DeepSWE eval model server with the 131072 cap
(eval/deepswe-v1.1/gym/plugin/responses_api_models/deepswe_eval_vllm_model/app.py and its deepswe_eval.yaml):
  * model "policy", temperature 1.0, top_p 1.0, chat_template_kwargs {"enable_thinking": true,
    "truncate_history_thinking": false}; hosted-only parameters (reasoning_effort, ...) are dropped;
  * max_tokens / max_completion_tokens := min(requested, 131072); max_tokens := 131072 when both are absent;
  * history thinking: ATP's Gym model server already sends it inside the assistant content as <think>R</think>C
    (atp build_config gives a "local" spec the eval's preserve_reasoning_in_assistant_content), and the router never
    folds reasoning fields into content. For callers that do send reasoning fields, assistant reasoning /
    reasoning_content equal to "" are dropped and reasoning_content is renamed to reasoning (vLLM's chat request
    does this itself, its /tokenize request does not);
  * tools[].function.strict is dropped; client headers (Authorization, api keys, ...) are not forwarded;
  * a 400 "maximum context length" gets the exact prompt length from the same worker's /tokenize and is retried
    there with max_tokens = 262144 - prompt tokens when that is in (0, requested) (at most 2 such retries);
    otherwise the 400 is returned unchanged.
Routing (load-aware since 2026-10-20): rendezvous hashing of an affinity key over the routable workers (registry rules
in client.py and < 3 consecutive /health failures; nothing else is ever routed to), ranked by
  1. workers with room before SATURATED ones. Saturated: this router has >= $SYNTH_ROUTER_SOFT_INFLIGHT (48; 0
     disables) calls in flight there, or the worker's vLLM /metrics shows vllm:num_requests_waiting > 0 or
     vllm:num_requests_running >= its max_num_seqs (64 unless the registry record says otherwise). /metrics is scraped
     every --metrics-s (5 s; 5 s timeout); a scrape older than 45 s counts as unknown (then only the in-flight count
     applies). The vLLM counts include the calls of every router (one per synth job) and every other user, which the
     per-router in-flight count cannot see;
  2. LEAVING workers last (client.leaving): a job that ends within 600 s (the worker stops 30 s before its end and
     aborts what is still running) or one an operator drains (registry/<job_id>.drain, effective before the worker
     publishes "draining") gets calls only when every other routable worker is leaving too;
  3. then ready before draining, then the rendezvous weight.
So during a fleet rotation, when most workers drain at once, a draining worker with room takes the calls that the
saturated ready workers would queue; when every worker is saturated the order is the old one (ready, then weight).
The key is the X-Synth-Affinity header, else sha256 of the first system + first user message plus the first assistant
turn once there is one (affinity_key): turn 1 of every repeat of a task shares one worker's prefix cache, later turns
stay on their conversation's worker while it is not saturated, and compacted OpenCode conversations still spread.
Affinity is best effort: a conversation moves while its worker is saturated or leaving, when the worker starts
draining while a ready worker has room (continuing conversations too), or when it loses the rendezvous to a worker
that became routable. A first attempt that left the affinity's worker for load is logged as moved_from/moved_why.
Failover: connection failures, 429/502/503/504, one 500 and, for buffered (non-stream) chat calls, a 200 whose choices
carry finish_reason "abort" are retried on the next worker (calls are buffered, nothing has reached the client yet),
up to 4 worker attempts. vLLM answers "abort" for the generations still running when its worker stops (drain flag
timeout, scancel, walltime); Gym's chat completion model rejects that value and the rollout would be lost. A retried
generation is a fresh sample (temperature 1.0), so its reply differs from what the aborted attempt had produced; that
is fine, the client never saw the aborted one. If the last attempt aborted too, the call gets 503 (upstream_aborted)
instead of the aborted completion. Worker connections use TCP keepalive (60 s idle, 15 s interval, 4 probes), so a
silently dead node fails its calls over in ~2 min. With no routable worker a call waits up to
$SYNTH_ROUTER_NO_WORKER_WAIT_S (3600 s: absorbs a gap between 4h workers inside the router; ATP answers a 429 by
halving the model's concurrency), then gets 429 with Retry-After: 120. There is no read timeout: generations can run
30+ min. Streamed calls (stream=true) are passed through unchanged, aborts included; they fail over only before the
first byte. client.ensure_router starts this process without secret-like environment variables.
Stale heartbeats (2026-10-20): a registry heartbeat_at older than 120 s does not make a worker unroutable while this
router's own GET /health of it passed within the last 60 s (client.worker_status healthy_at), because a Lustre stall
can hang the workers' registry writes while vLLM keeps serving (15:22-15:26 all 4 bridges went stale at once and every
router dropped them). The router health-checks such a worker only while that holds, so a worker that stops answering
drops out until its registry heartbeat is fresh again. A "stopped"/"unhealthy" status, end_epoch, 3 consecutive
failed checks, refused connections and TCP keepalive still remove workers as before.
Logs: <log_dir>/student-router.log (text; worker routability and vLLM load changes) and student-router-requests.jsonl
(one line per call, no content/headers); GET /router/status lists the workers with their load and the counters.

    python -B router.py --registry DIR --host 127.0.0.1 --port 0 --port-file FILE --log-dir DIR
"""
import sys

sys.dont_write_bytecode = True

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import socket
import time

import aiohttp
from aiohttp import web

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))  # swe-task root, for `student.client`
from student.client import (HEARTBEAT_TIMEOUT_S, REGISTRY, drain_requested, leaving, max_num_seqs,  # noqa: E402
                            read_registry, root_url, worker_status)

LOG = logging.getLogger("student.router")
MODEL = "policy"
MAX_MODEL_LEN = 262144
PER_CALL_CAP = 131072
CHAT_TEMPLATE_KWARGS = {"enable_thinking": True, "truncate_history_thinking": False}
SAMPLING = {"temperature": 1.0, "top_p": 1.0}
# Parameters of the hosted (hub) model configs that the student must not receive.
HOSTED_ONLY = ("reasoning_effort", "verbosity", "service_tier", "store", "prediction", "web_search_options")
# Gym VLLMModel._TOKENIZE_CHAT_FIELDS plus the chat request's other prompt-rendering fields.
TOKENIZE_FIELDS = ("model", "messages", "tools", "chat_template_kwargs", "mm_processor_kwargs",
                   "required_prefix_token_ids", "chat_template", "add_generation_prompt", "continue_final_message",
                   "add_special_tokens")
CHAT, TOKENIZE = "/v1/chat/completions", "/tokenize"
FAILOVER_STATUS = {429, 502, 503, 504}  # and the first 500 of a call
ABORT = "finish_reason abort"  # a buffered chat 200 whose generation vLLM aborted: failed over like a 503
TRANSPORT_ERRORS = (aiohttp.ClientError, TimeoutError)  # connect/OS errors, disconnects, payload/protocol errors
MAX_ATTEMPTS, MAX_CLAMPS, HEALTH_FAILURES = 4, 2, 3
RECHECK_S, RETRY_AFTER_S = 15.0, 120
TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=None)
HEALTH_TIMEOUT = aiohttp.ClientTimeout(total=10)
METRICS_TIMEOUT = aiohttp.ClientTimeout(total=5)  # one /metrics page is ~60 kB and takes ~10 ms
METRICS_STALE_S = 45.0  # an older vLLM load sample counts as unknown
LOAD_GAUGES = {"vllm:num_requests_running": 0, "vllm:num_requests_waiting": 1}
LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')
MAX_BODY = 512 * 1024 * 1024
STREAM_TAIL = 65536


def normalize(body: dict) -> dict:
    """The eval model server's chat request semantics, in place (see module docstring)."""
    for key in HOSTED_ONLY:
        body.pop(key, None)
    body.update(SAMPLING)
    for key in ("max_tokens", "max_completion_tokens"):
        if body.get(key) is not None:
            body[key] = min(int(body[key]), PER_CALL_CAP)
    if body.get("max_tokens") is None and body.get("max_completion_tokens") is None:
        body["max_tokens"] = PER_CALL_CAP
    if body.get("stream") is not True:
        body.pop("stream_options", None)  # vLLM rejects stream_options without stream
    return normalize_prompt(body)


def normalize_prompt(body: dict) -> dict:
    """The prompt-rendering part, shared by chat and /tokenize: model, template kwargs, reasoning history, tools."""
    body.update(model=MODEL, chat_template_kwargs=dict(CHAT_TEMPLATE_KWARGS))
    for message in body.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for key in ("reasoning", "reasoning_content"):
            if message.get(key) == "":
                del message[key]
        if "reasoning_content" in message:  # as vLLM's ChatCompletionRequest does; TokenizeChatRequest does not
            value = message.pop("reasoning_content")
            if message.get("reasoning") is None and value is not None:
                message["reasoning"] = value
    for tool in body.get("tools") or []:
        if isinstance(tool, dict) and tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            tool["function"].pop("strict", None)
    return body


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part["text"] for part in content
                       if isinstance(part, dict) and isinstance(part.get("text"), str))
    return ""


def keepalive_socket(addr_info):
    family, type_, proto, _, _ = addr_info
    sock = socket.socket(family=family, type=type_, proto=proto)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for name, value in (("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 15), ("TCP_KEEPCNT", 4)):
        if hasattr(socket, name):
            sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, name), value)
    return sock


def affinity_key(headers, body: dict) -> str:
    """X-Synth-Affinity, else the conversation's opening: first system + first user message, plus the first assistant
    turn once there is one. Turn 1 of every repeat of a task shares a worker (and its prefix cache); later turns stay
    on their own conversation's worker. OpenCode's compaction replaces the history with fixed messages, so without
    the assistant turn every compacted conversation of every task would hash to one worker."""
    explicit = headers.get("X-Synth-Affinity")
    if explicit:
        seed = "header\0" + explicit
    else:
        first = {}
        for message in body.get("messages") or []:
            if isinstance(message, dict):
                role = "system" if message.get("role") == "developer" else message.get("role")
                if role in ("system", "user", "assistant") and role not in first:
                    first[role] = _text(message.get("content"))
                    if role == "assistant":
                        first[role] += "\0" + json.dumps(message.get("tool_calls") or [], sort_keys=True) + "\0" + \
                            str(message.get("reasoning") or message.get("reasoning_content") or "")
        seed = "messages\0" + first.get("system", "") + "\0" + first.get("user", "") + "\0" + first.get("assistant", "")
    return hashlib.sha256(seed.encode("utf-8", "surrogatepass")).hexdigest()


def error_response(status: int, message: str, kind: str, headers=None) -> web.Response:
    return web.json_response({"error": {"message": message, "type": kind, "param": None, "code": status}},
                             status=status, headers=headers)


def error_kind(body: bytes):
    """type/param of an OpenAI-style error body (never its message, which may quote the request)."""
    try:
        error = json.loads(body).get("error")
    except (ValueError, AttributeError):
        return None
    if not isinstance(error, dict):
        return None
    return "/".join(str(error[key]) for key in ("type", "param") if error.get(key)) or None


def vllm_load(text: str, model: str = MODEL):
    """(running, waiting) requests of `model` from a vLLM Prometheus /metrics page, summed over its engines; None when
    either gauge is missing or not a finite number."""
    totals, seen = [0.0, 0.0], [False, False]
    for line in text.splitlines():
        if not line.startswith("vllm:num_requests_"):
            continue
        name, brace, rest = line.partition("{")
        if brace:
            labels, _, value = rest.rpartition("}")
        else:
            name, _, value = line.partition(" ")
            labels = ""
        index = LOAD_GAUGES.get(name.strip())
        if index is None or dict(LABEL.findall(labels)).get("model_name", model) != model:
            continue
        try:
            number = float(value.split()[0])
        except (ValueError, IndexError):
            return None
        if not math.isfinite(number):
            return None
        totals[index] += number
        seen[index] = True
    return (int(totals[0]), int(totals[1])) if all(seen) else None


def aborted(body: bytes) -> bool:
    """Whether a chat completion body has a choice with finish_reason "abort" (vLLM: the worker stopped mid-call)."""
    if b'"abort"' not in body:  # cheap pre-check (quotes inside the generated text are escaped in the JSON)
        return False
    try:
        choices = json.loads(body).get("choices")
    except (ValueError, AttributeError):
        return False
    return isinstance(choices, list) and any(isinstance(choice, dict) and choice.get("finish_reason") == "abort"
                                             for choice in choices)


def sse_summary(tail: bytes):
    """(usage, finish_reason) from the last events of an SSE stream."""
    usage = finish = None
    for line in tail.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:") or line == b"data: [DONE]":
            continue
        try:
            event = json.loads(line[5:])
        except ValueError:
            continue
        if isinstance(event, dict):
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                if isinstance(choice, dict) and choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    return usage, finish


class Answer:
    """A worker's reply; for an open stream, body is its first chunk and stream the open response."""
    __slots__ = ("status", "body", "content_type", "stream")

    def __init__(self, status, body, content_type, stream=None):
        self.status, self.body, self.content_type, self.stream = status, body, content_type, stream


class Worker:
    """Router-side state of one registry entry."""

    def __init__(self, job_id: str):
        self.job_id, self.entry, self.present = job_id, {}, True
        self.health_failures = self.inflight = self.requests = self.errors = 0
        self.last_error = None
        self.healthy_at = None  # time.time() of this router's last passing GET /health
        self.on_own_health = False  # routable with a stale registry heartbeat (logged on change)
        self.drain_flag = False  # registry/<job_id>.drain exists
        # last good vLLM /metrics sample: (running, waiting), its time.monotonic(); the last scrape problem
        self.load, self.load_at, self.metrics_error = None, float("-inf"), None
        self.load_state = None  # last logged load state: "room", "saturated" or "unknown"

    @property
    def root(self) -> str:
        return root_url(self.entry["base_url"])

    def routing_status(self, now: float):
        """"ready" / "draining" when routable (registry rules, with this router's own recent /health pass standing in
        for a stale heartbeat, and < 3 consecutive health failures), else None."""
        if not self.present or self.health_failures >= HEALTH_FAILURES:
            return None
        return worker_status(self.entry, now, self.healthy_at)

    def heartbeat_stale(self, now: float) -> bool:
        heartbeat = self.entry.get("heartbeat_at")
        return not isinstance(heartbeat, (int, float)) or now - heartbeat > HEARTBEAT_TIMEOUT_S

    def vllm_load(self, monotonic: float, stale_s: float = METRICS_STALE_S):
        """(running, waiting) of the last /metrics sample if it is at most stale_s old, else None (unknown)."""
        return self.load if monotonic - self.load_at <= stale_s else None

    def leaving(self, now: float):
        return leaving(self.entry, now, self.drain_flag)

    def describe(self, now: float) -> dict:
        heartbeat, monotonic = self.entry.get("heartbeat_at"), time.monotonic()
        return {"job_id": self.job_id, "routing": self.routing_status(now), "status": self.entry.get("status"),
                "node": self.entry.get("node"), "base_url": self.entry.get("base_url"),
                "heartbeat_age_s": round(now - heartbeat, 1) if isinstance(heartbeat, (int, float)) else None,
                "drain_at": self.entry.get("drain_at"), "end_epoch": self.entry.get("end_epoch"),
                "in_registry": self.present, "health_failures": self.health_failures, "inflight": self.inflight,
                "requests": self.requests, "errors": self.errors, "last_error": self.last_error,
                "health_age_s": round(now - self.healthy_at, 1) if self.healthy_at else None,
                "on_own_health": self.on_own_health, "leaving": self.leaving(now), "drain_flag": self.drain_flag,
                "max_num_seqs": max_num_seqs(self.entry),
                "vllm_running": self.load[0] if self.load else None,
                "vllm_waiting": self.load[1] if self.load else None,
                "metrics_age_s": round(monotonic - self.load_at, 1) if self.load else None,
                "metrics_error": self.metrics_error}


class Router:
    retry_backoff_s = 2.0  # before re-trying a worker that already failed this call
    metrics_stale_s = METRICS_STALE_S

    def __init__(self, registry=REGISTRY, log_dir=".", *, refresh_s=10.0, health_s=15.0, metrics_s=5.0,
                 no_worker_wait_s=None):
        self.registry, self.log_dir = Path(registry), Path(log_dir)
        self.refresh_s, self.health_s, self.metrics_s = refresh_s, health_s, metrics_s  # metrics_s 0: no scraping
        if no_worker_wait_s is None:
            no_worker_wait_s = float(os.environ.get("SYNTH_ROUTER_NO_WORKER_WAIT_S", "3600"))
        self.no_worker_wait_s = no_worker_wait_s
        self.soft_inflight_cap = int(os.environ.get("SYNTH_ROUTER_SOFT_INFLIGHT", "48"))  # 0 disables
        self.workers, self.counters, self.inflight, self.closing = {}, Counter(), 0, False
        self.started_at = time.time()
        self.session = self._journal = None
        self._tasks, self._states, self._refreshed, self._refresh_task = [], {}, float("-inf"), None
        self._changed = asyncio.Event()

    def app(self) -> web.Application:
        app = web.Application(client_max_size=MAX_BODY)
        app.add_routes([web.post(CHAT, self.chat_completions), web.post(TOKENIZE, self.tokenize),
                        web.get("/v1/models", self.models), web.get("/health", self.health),
                        web.get("/router/status", self.status)])
        app.on_startup.append(self._startup)
        app.on_shutdown.append(self._shutdown)
        app.on_cleanup.append(self._cleanup)
        return app

    async def _startup(self, app) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._journal = (self.log_dir / "student-router-requests.jsonl").open("a", buffering=1)
        # No pooling: vLLM closes idle keep-alive connections after 5 s, and reusing one races that close.
        # TCP keepalive: a worker node that dies silently fails its in-flight calls over in ~2 min, not at ATP's 3 h deadline
        connector = aiohttp.TCPConnector(force_close=True, limit=0, socket_factory=keepalive_socket)
        self.session = aiohttp.ClientSession(connector=connector, timeout=TIMEOUT)
        await self.refresh()
        self._tasks = [asyncio.create_task(self._refresh_loop()), asyncio.create_task(self._health_loop())]
        if self.metrics_s > 0:
            self._tasks.append(asyncio.create_task(self._metrics_loop()))

    async def _shutdown(self, app) -> None:  # runs before in-flight calls are cancelled
        self.closing = True

    async def _cleanup(self, app) -> None:
        tasks = self._tasks + ([self._refresh_task] if self._refresh_task else [])
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.session.close()
        self._journal.close()

    # ------------------------------------------------------------------ worker state
    async def refresh(self) -> None:
        """Re-read the registry and its drain flags; a file that is momentarily unreadable keeps its previous entry."""
        self._refreshed = time.monotonic()
        entries = await asyncio.to_thread(read_registry, self.registry)
        flags = await asyncio.to_thread(drain_requested, self.registry)
        for job_id, entry in entries.items():
            worker = self.workers.get(job_id)
            if worker is None:
                if entry is None:
                    continue
                worker = self.workers[job_id] = Worker(job_id)
            worker.present = True
            if entry is not None:
                if entry.get("base_url") != worker.entry.get("base_url"):
                    worker.health_failures = 0  # a new endpoint starts with a clean health record and no load sample
                    worker.healthy_at, worker.load, worker.load_at = None, None, float("-inf")
                worker.entry = entry
        for job_id, worker in list(self.workers.items()):
            if job_id not in entries:
                worker.present = False
                if not worker.inflight:
                    del self.workers[job_id]
            elif (job_id in flags) != worker.drain_flag:
                worker.drain_flag = job_id in flags
                LOG.info("worker %s: drain flag %s", job_id, "set: calls go to it only when every other routable "
                         "worker is leaving too" if worker.drain_flag else "removed")
        self._update_states()

    def _update_states(self) -> None:
        """Log routability transitions and wake calls that wait for a worker."""
        now = time.time()
        states = {job_id: worker.routing_status(now) for job_id, worker in self.workers.items()}
        if states == self._states:
            return
        for job_id in sorted(set(states) | set(self._states)):
            old, new = self._states.get(job_id), states.get(job_id)
            if old != new:
                worker = self.workers.get(job_id)
                LOG.info("worker %s: %s -> %s %s", job_id, old or "unroutable", new or "unroutable",
                         json.dumps(worker.describe(now)) if worker else "(registry file removed)")
        self._states = states
        self._changed.set()
        self._changed = asyncio.Event()

    def _refresh_soon(self) -> None:
        """Re-read the registry in the background (a stopping worker has written "stopped" by now), at most once a
        second and one at a time: a stalled filesystem must neither delay the call nor pile up reads."""
        if (self._refresh_task is None or self._refresh_task.done()) and time.monotonic() - self._refreshed > 1.0:
            self._refresh_task = asyncio.create_task(self._refresh_logged())

    async def _refresh_logged(self) -> None:
        try:
            await self.refresh()
        except Exception:
            LOG.exception("registry refresh failed")

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(self.refresh_s)
            try:
                await self.refresh()
            except Exception:
                LOG.exception("registry refresh failed")

    async def _health_loop(self) -> None:
        while True:
            try:
                await self.check_health()
            except Exception:
                LOG.exception("health check round failed")
            await asyncio.sleep(self.health_s)

    async def check_health(self) -> None:
        """One round: GET /health of every registry-routable worker (a stale heartbeat counts while this router's own
        checks of the worker keep passing); 3 consecutive failures make a worker unroutable."""
        now = time.time()
        targets = [worker for worker in self.workers.values()
                   if worker.present and worker_status(worker.entry, now, worker.healthy_at)]
        await asyncio.gather(*(self._check(worker) for worker in targets))
        self._update_states()
        now = time.time()
        for job_id in sorted(self.workers):
            worker = self.workers[job_id]
            kept = worker.routing_status(now) is not None and worker.heartbeat_stale(now)
            if kept != worker.on_own_health:
                worker.on_own_health = kept
                heartbeat = worker.entry.get("heartbeat_at")
                age = f"{now - heartbeat:.0f} s old" if isinstance(heartbeat, (int, float)) else "missing"
                if kept:
                    self.counters["stale_heartbeat_kept"] += 1
                    LOG.warning("worker %s: registry heartbeat %s (filesystem or clock problem?); kept routable on "
                                "this router's own /health checks", job_id, age)
                else:
                    LOG.info("worker %s: no longer routable on its own /health checks (registry heartbeat %s, "
                             "routing %s)", job_id, age, worker.routing_status(now) or "unroutable")

    async def _check(self, worker: Worker) -> None:
        try:
            async with self.session.get(worker.root + "/health", timeout=HEALTH_TIMEOUT) as response:
                await response.read()
                problem = None if response.status == 200 else f"health HTTP {response.status}"
        except TRANSPORT_ERRORS as error:
            problem = f"health {type(error).__name__}"
        if problem is None:
            worker.health_failures, worker.healthy_at = 0, time.time()
        else:
            worker.health_failures += 1
            worker.last_error = problem

    async def _metrics_loop(self) -> None:
        while True:
            try:
                await self.scrape_metrics()
            except Exception:
                LOG.exception("metrics round failed")
            await asyncio.sleep(self.metrics_s)

    async def scrape_metrics(self) -> None:
        """One round: GET /metrics of every routable worker (vLLM's own request counts), then log load changes."""
        now = time.time()
        targets = sorted((worker for worker in self.workers.values() if worker.routing_status(now)),
                         key=lambda worker: worker.job_id)
        await asyncio.gather(*(self._scrape(worker) for worker in targets))
        monotonic, changes = time.monotonic(), []
        for worker in targets:
            load = worker.vllm_load(monotonic, self.metrics_stale_s)
            state = "unknown" if load is None else ("saturated" if self._load_saturation(worker, load) else "room")
            if state != worker.load_state:
                worker.load_state = state
                detail = (f"running {load[0]}/{max_num_seqs(worker.entry)}, waiting {load[1]}" if load else
                          f"no /metrics sample within {self.metrics_stale_s:.0f} s: {worker.metrics_error}")
                changes.append(f"{worker.job_id} {state} ({detail})")
        if changes:
            LOG.info("vLLM load: %s", "; ".join(changes))

    async def _scrape(self, worker: Worker) -> None:
        load = None
        try:
            async with self.session.get(worker.root + "/metrics", timeout=METRICS_TIMEOUT) as response:
                body = await response.read()
                if response.status != 200:
                    problem = f"metrics HTTP {response.status}"
                else:
                    load = vllm_load(body.decode("utf-8", "replace"), worker.entry.get("served_model_name") or MODEL)
                    problem = None if load else "metrics without vllm:num_requests_running/waiting"
        except TRANSPORT_ERRORS as error:
            problem = f"metrics {type(error).__name__}"
        if problem is None:
            worker.load, worker.load_at, worker.metrics_error = load, time.monotonic(), None
        else:
            worker.metrics_error = problem
            self.counters["metrics_errors"] += 1

    @staticmethod
    def _load_saturation(worker: Worker, load):
        if load[1] > 0:
            return "waiting"
        return "running" if load[0] >= max_num_seqs(worker.entry) else None

    def saturation(self, worker: Worker, monotonic=None):
        """Why a worker is saturated, else None: "inflight" (this router has >= the soft cap of calls there),
        "waiting" (its vLLM queues requests) or "running" (its vLLM runs max_num_seqs sequences)."""
        if self.soft_inflight_cap and worker.inflight >= self.soft_inflight_cap:
            return "inflight"
        load = worker.vllm_load(time.monotonic() if monotonic is None else monotonic, self.metrics_stale_s)
        return self._load_saturation(worker, load) if load else None

    def _ranking(self, key: str):
        """(routable workers in routing order for key, the worker that affinity alone would pick: the first by
        ready-before-draining and rendezvous weight). Order: room before saturated before leaving, then ready before
        draining, then rendezvous (highest random) weight."""
        now, monotonic = time.time(), time.monotonic()
        scored = []
        for worker in self.workers.values():
            status = worker.routing_status(now)
            if status:
                weight = int.from_bytes(hashlib.sha256(f"{key}\0{worker.job_id}".encode()).digest()[:8], "big")
                saturated = self.saturation(worker, monotonic) is not None
                tier = 2 if worker.leaving(now) else int(saturated)
                scored.append(((tier, saturated, status != "ready", -weight, worker.job_id),
                               (status != "ready", -weight, worker.job_id), worker))
        ranked = [item[-1] for item in sorted(scored, key=lambda item: item[0])]
        natural = min(scored, key=lambda item: item[1])[-1] if scored else None
        return ranked, natural

    def ranked(self, key: str) -> list:
        """Routable workers in routing order for key (see _ranking)."""
        return self._ranking(key)[0]

    async def _wait_for_change(self, timeout: float) -> None:
        try:
            await asyncio.wait_for(self._changed.wait(), timeout)
        except TimeoutError:
            pass

    def _moved(self, natural: Worker, worker: Worker, record: dict) -> None:
        """Journal and count a first attempt that leaves the affinity's worker because it is saturated or leaving."""
        now = time.time()
        why = natural.leaving(now) or self.saturation(natural) or "other"
        record["moved_from"], record["moved_why"] = natural.job_id, why
        self.counters[f"moved_{why}"] += 1
        if natural.routing_status(now) == "ready" and worker.routing_status(now) == "draining":
            self.counters["spilled_to_draining"] += 1

    async def _pick(self, key: str, tried: list, record: dict):
        """The best routable worker not yet tried by this call; waits (bounded) while none is routable."""
        while True:
            ranked, natural = self._ranking(key)
            for worker in ranked:
                if worker.job_id not in tried:
                    if not tried and worker is not natural:
                        self._moved(natural, worker, record)
                    return worker
            if ranked:  # every routable worker already failed this call: back off, then try the best again
                await asyncio.sleep(min(self.retry_backoff_s * len(tried), 10.0))
                ranked = self.ranked(key)
                if ranked:
                    return ranked[0]
                continue
            if time.monotonic() - self._refreshed > 1.0:  # a worker may have registered since the last re-read
                await self.refresh()
                continue
            budget = self.no_worker_wait_s - record["waited_s"]
            if budget <= 0:
                return None
            if not record["waited_s"]:
                self.counters["held"] += 1
                LOG.info("call %s: no routable worker, waiting up to %.0f s", record["affinity"], budget)
            started = time.monotonic()
            try:
                await self._wait_for_change(min(RECHECK_S, budget))
            finally:
                record["waited_s"] = round(record["waited_s"] + time.monotonic() - started, 3)

    # ------------------------------------------------------------------ forwarding
    async def _send(self, worker: Worker, path: str, body: dict, stream=False) -> Answer:
        response = await self.session.post(worker.root + path, data=json.dumps(body).encode(),
                                           headers={"Content-Type": "application/json"})
        opened = False
        try:
            content_type = response.headers.get("Content-Type", "application/json")
            if stream and response.status == 200:
                first = await response.content.readany()
                if not first:
                    raise aiohttp.ClientPayloadError("worker ended the stream before sending data")
                opened = True
                return Answer(200, first, content_type, response)
            return Answer(response.status, await response.read(), content_type)
        finally:
            if not opened:
                response.release()

    async def _clamp(self, worker: Worker, body: dict) -> dict:
        """eval app.py: retry budget = max_model_len - exact prompt tokens (worker /tokenize), if in (0, requested)."""
        field = "max_tokens" if body.get("max_completion_tokens") is None else "max_completion_tokens"
        requested = min(value for value in (body.get("max_tokens"), body.get("max_completion_tokens"), PER_CALL_CAP)
                        if value is not None)
        tokenize = {key: body[key] for key in TOKENIZE_FIELDS if key in body}
        tokenize.setdefault("add_generation_prompt", True)
        answer = await self._send(worker, TOKENIZE, tokenize)
        count = None
        if answer.status == 200:
            try:
                count = json.loads(answer.body).get("count")
            except (ValueError, AttributeError):
                pass
        limit = worker.entry.get("max_model_len")
        limit = limit if isinstance(limit, int) and limit > 0 else MAX_MODEL_LEN
        remaining = limit - count if isinstance(count, int) else None
        applied = remaining is not None and 0 < remaining < requested
        if applied:
            body[field] = remaining
        return {"prompt_tokens": count, "remaining": remaining, "field": field, "applied": applied}

    async def _call(self, worker: Worker, path: str, body: dict, record: dict) -> Answer:
        """One worker attempt, including the context-overflow clamp retries on that worker."""
        worker.inflight += 1
        worker.requests += 1
        try:
            while True:
                answer = await self._send(worker, path, body, record["stream"])
                if (path != CHAT or answer.status != 400 or b"maximum context length" not in answer.body
                        or len(record["clamps"]) >= MAX_CLAMPS):
                    return answer
                clamp = await self._clamp(worker, body)
                record["clamps"].append(clamp)
                LOG.info("call %s: context overflow on %s, prompt %s tokens -> %s %s", record["affinity"],
                         worker.job_id, clamp["prompt_tokens"], clamp["field"],
                         clamp["remaining"] if clamp["applied"] else "unchanged (400 returned)")
                if not clamp["applied"]:
                    return answer
                self.counters["clamps"] += 1
        finally:
            worker.inflight -= 1

    def _failed(self, worker: Worker, problem: str, record: dict) -> None:
        worker.errors += 1
        worker.last_error = record["error"] = problem
        self.counters["worker_failures"] += 1
        LOG.warning("call %s: attempt %d/%d on worker %s failed: %s", record["affinity"], record["attempts"],
                    MAX_ATTEMPTS, worker.job_id, problem)

    async def _proxy(self, request, body: dict, record: dict, path: str):
        """Route one call with failover; returns the worker's Answer or an error response."""
        key = affinity_key(request.headers, body)
        record["affinity"] = key[:12]
        tried, last, retried_500 = [], None, False
        while record["attempts"] < MAX_ATTEMPTS:
            worker = await self._pick(key, tried, record)
            if worker is None:
                self.counters["no_worker_429"] += 1
                LOG.warning("call %s: no routable worker after %.0f s, returning 429", record["affinity"],
                            record["waited_s"])
                record["error"] = "no_routable_worker"
                return error_response(429, f"No student vLLM worker became routable within "
                                      f"{self.no_worker_wait_s:.0f} s; retry later", "no_routable_worker",
                                      headers={"Retry-After": str(RETRY_AFTER_S)})
            record["attempts"] += 1
            record["workers"].append(worker.job_id)
            tried.append(worker.job_id)
            try:
                answer = await self._call(worker, path, body, record)
            except TRANSPORT_ERRORS as error:
                last = error
                self._failed(worker, type(error).__name__, record)
                if isinstance(error, aiohttp.ClientConnectorError):  # nothing listens there: counts as health
                    worker.health_failures += 1
                    self._update_states()
                continue
            if answer.status in FAILOVER_STATUS or (answer.status == 500 and not retried_500):
                retried_500 = retried_500 or answer.status == 500
                last = answer
                self._failed(worker, f"HTTP {answer.status}", record)
                continue
            if path == CHAT and not record["stream"] and answer.status == 200 and aborted(answer.body):
                last = ABORT  # the worker's vLLM stopped mid-call: a worker failure, retried as a fresh sample
                record["aborts"] += 1
                self.counters["aborts"] += 1
                self._failed(worker, ABORT, record)
                self._refresh_soon()
                continue
            record["worker"] = worker.job_id
            return answer
        record["worker"] = record["workers"][-1] if record["workers"] else None
        if last is ABORT:  # never hand an aborted completion to the client
            return error_response(503, f"Student worker generation aborted (worker stopping) on {record['aborts']} "
                                  f"of {record['attempts']} attempts", "upstream_aborted")
        if isinstance(last, Answer):
            return last
        return error_response(502, f"Student worker calls failed after {record['attempts']} attempts "
                              f"({type(last).__name__})", "upstream_unavailable")

    async def _respond(self, request, answer: Answer, record: dict, path: str) -> web.StreamResponse:
        if answer.stream is not None:
            return await self._pipe(request, answer, record)
        if answer.status != 200:
            record["error"] = f"HTTP {answer.status} {error_kind(answer.body) or ''}".strip()
        else:
            try:
                data = json.loads(answer.body)
            except ValueError:
                data = None
            if isinstance(data, dict) and path == TOKENIZE:
                record["prompt_tokens"] = data.get("count")
            elif isinstance(data, dict):
                usage, choices = data.get("usage") or {}, data.get("choices") or [{}]
                record["prompt_tokens"] = usage.get("prompt_tokens")
                record["completion_tokens"] = usage.get("completion_tokens")
                record["finish_reason"] = choices[0].get("finish_reason") if isinstance(choices[0], dict) else None
        return web.Response(status=answer.status, body=answer.body, headers={"Content-Type": answer.content_type})

    async def _pipe(self, request, answer: Answer, record: dict) -> web.StreamResponse:
        """Pass a worker's SSE through; once bytes reached the client a worker failure can no longer fail over."""
        upstream, tail = answer.stream, answer.body[-STREAM_TAIL:]
        response = web.StreamResponse(headers={"Content-Type": answer.content_type, "Cache-Control": "no-cache",
                                               "X-Accel-Buffering": "no"})
        try:
            record["streamed"] = True
            await response.prepare(request)
            await response.write(answer.body)
            while True:
                try:
                    chunk = await upstream.content.readany()
                except TRANSPORT_ERRORS as error:
                    record["error"] = f"stream aborted: {type(error).__name__}"
                    self.counters["stream_aborts"] += 1
                    LOG.warning("call %s: worker %s stream aborted: %s", record["affinity"], record["worker"],
                                type(error).__name__)
                    event = {"error": {"message": f"Student worker stream failed: {type(error).__name__}",
                                       "type": "upstream_stream_error", "param": None, "code": 502}}
                    await response.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                    break
                if not chunk:
                    break
                tail = (tail + chunk)[-STREAM_TAIL:]
                await response.write(chunk)
            await response.write_eof()
        finally:
            upstream.release()
        usage, record["finish_reason"] = sse_summary(tail)
        record["prompt_tokens"] = (usage or {}).get("prompt_tokens")
        record["completion_tokens"] = (usage or {}).get("completion_tokens")
        return response

    async def _handle(self, request, path: str) -> web.StreamResponse:
        started = time.time()
        record = {"ts": round(started, 3), "path": path, "stream": False, "affinity": None, "worker": None,
                  "workers": [], "status": None, "attempts": 0, "clamps": [], "waited_s": 0.0,
                  "prompt_tokens": None, "completion_tokens": None, "finish_reason": None, "error": None,
                  "aborts": 0, "moved_from": None, "moved_why": None}
        self.inflight += 1
        response = None
        try:
            try:
                body = json.loads(await request.read())
            except ValueError:
                body = None
            if not isinstance(body, dict):
                response = error_response(400, "Request body must be a JSON object", "invalid_request_error")
                return response
            try:
                if path == CHAT:
                    normalize(body)
                elif "messages" in body:
                    normalize_prompt(body)
                else:
                    body["model"] = MODEL
            except (TypeError, ValueError) as error:
                response = error_response(400, f"Invalid request ({type(error).__name__})", "invalid_request_error")
                return response
            record["stream"] = path == CHAT and body.get("stream") is True
            result = await self._proxy(request, body, record, path)
            response = await self._respond(request, result, record, path) if isinstance(result, Answer) else result
            return response
        except (asyncio.CancelledError, ConnectionResetError):  # the client hung up (worker errors never get here)
            record["status"] = "router_shutdown" if self.closing else "client_disconnected"
            raise
        except web.HTTPException as error:  # aiohttp's own answers, e.g. 413 for a body over MAX_BODY
            record["status"], record["error"] = error.status, error.reason
            raise
        except Exception as error:
            LOG.exception("call %s: router error", record["affinity"])
            record["error"] = f"router error: {type(error).__name__}"
            if record.get("streamed"):
                raise
            response = error_response(500, f"Student router error ({type(error).__name__})", "router_error")
            return response
        finally:
            self.inflight -= 1
            if record["status"] is None:
                record["status"] = response.status if response is not None else "failed"
            record["duration_s"] = round(time.time() - started, 3)
            self.counters["calls"] += 1
            self.counters[f"status_{record['status']}"] += 1
            self._journal.write(json.dumps(record, separators=(",", ":")) + "\n")

    # ------------------------------------------------------------------ routes
    async def chat_completions(self, request) -> web.StreamResponse:
        return await self._handle(request, CHAT)

    async def tokenize(self, request) -> web.StreamResponse:
        return await self._handle(request, TOKENIZE)

    async def models(self, request) -> web.Response:
        return web.json_response({"object": "list",
                                  "data": [{"id": MODEL, "object": "model", "owned_by": "swe-task"}]})

    async def health(self, request) -> web.Response:
        now = time.time()
        states = [worker.routing_status(now) for worker in self.workers.values()]
        body = {"routable": sum(map(bool, states)), "ready": states.count("ready"),
                "draining": states.count("draining"), "known": len(states), "inflight": self.inflight}
        return web.json_response(body, status=200 if body["routable"] else 503)

    async def status(self, request) -> web.Response:
        now = time.time()
        workers = []
        for job_id in sorted(self.workers):
            worker = self.workers[job_id]
            workers.append(dict(worker.describe(now), saturated=self.saturation(worker)))
        return web.json_response({
            "pid": os.getpid(), "started_at": self.started_at, "uptime_s": round(now - self.started_at, 1),
            "registry": str(self.registry), "no_worker_wait_s": self.no_worker_wait_s, "inflight": self.inflight,
            "soft_inflight_cap": self.soft_inflight_cap, "metrics_s": self.metrics_s,
            "metrics_stale_s": self.metrics_stale_s, "counters": dict(self.counters), "workers": workers})


async def start(router: Router, host: str, port: int):
    """Serve router on host:port (0 = any free port); returns (runner, bound port)."""
    # handler_cancellation: a client that hangs up cancels its call, which closes the worker connection and makes
    # vLLM abort the generation. On shutdown, in-flight calls get 2 s (twice: graceful, then after cancel).
    runner = web.AppRunner(router.app(), handler_cancellation=True, shutdown_timeout=2.0, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    return runner, runner.addresses[0][1]


async def serve(args) -> None:
    router = Router(args.registry, args.log_dir, refresh_s=args.refresh_s, health_s=args.health_s,
                    metrics_s=args.metrics_s)
    runner, port = await start(router, args.host, args.port)
    if args.port_file:
        temporary = args.port_file.with_name(f".{args.port_file.name}.{os.getpid()}.tmp")
        temporary.write_text(f"{port}\n")
        os.replace(temporary, args.port_file)
    LOG.info("listening on http://%s:%d/v1 (pid %d, registry %s, no-worker wait %.0f s; load-aware routing: soft "
             "in-flight cap %d, vLLM /metrics every %.0f s, abort retry)", args.host, port, os.getpid(), args.registry,
             router.no_worker_wait_s, router.soft_inflight_cap, router.metrics_s)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), 5)
        except TimeoutError:
            if args.parent_pid and os.getppid() != args.parent_pid:
                LOG.info("parent process %d exited", args.parent_pid)
                break
    LOG.info("shutting down (%d calls in flight)", router.inflight)
    await runner.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", type=Path, default=REGISTRY, help="worker registry directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 = any free port")
    parser.add_argument("--port-file", type=Path, help="written atomically with the bound port once listening")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, help="exit when this parent process is gone")
    parser.add_argument("--refresh-s", type=float, default=10.0, help="registry re-read interval")
    parser.add_argument("--health-s", type=float, default=15.0, help="worker GET /health interval")
    parser.add_argument("--metrics-s", type=float, default=5.0, help="worker vLLM GET /metrics interval (0 = off)")
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.log_dir / "student-router.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
