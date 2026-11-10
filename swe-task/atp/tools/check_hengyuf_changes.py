#!/usr/bin/env python3
"""Offline checks of hengyuf's ATP changes (no model APIs, no sandboxes, no Slurm submission).

    runtime/controller/bin/python -B tools/check_hengyuf_changes.py http
        deepswe_http hosted check with a stub `send`, plus infrastructure_error/generation_failure equal to
        Baihe's (deepswe_campaign.py is unchanged again).
    <scratch>/controller/bin/python -B tools/check_hengyuf_changes.py config --scratch <staged scratch>
        build_config, gym_command and run_model's launch environment for the local student "sc14" and the hosted
        models, compared with Baihe's pristine upstream/ code (kimi/glm gym.yaml and gym command, and kimi's launch
        environment, must be identical). Needs a staged scratch and a CPU-partition Slurm environment (or
        SLURM_JOB_ID/SLURM_JOB_PARTITION=cpu_* set by hand).
    runtime/controller/bin/python -B tools/check_hengyuf_changes.py capture [--scratch <parent of a private dir>]
        deepswe_capture.run_opencode's local stream copies (stdout.log/stderr.log) with fake PTY/sandbox objects and
        a slow os.fsync of those copies, compared with Baihe's upstream/ copy: identical bytes in order; the event loop
        keeps running during each fsync (upstream: blocked); a drain cancelled (twice) during a write still lands the
        handed-off chunk before its file closes (never an fsync on a closed descriptor); a write/fsync error ends
        run_opencode as upstream; no unretrieved-exception reports. Scratch: --scratch (default /tmp).
Secret values are never printed (only placeholders and booleans).
"""
import sys

sys.dont_write_bytecode = True

import argparse
import asyncio
import copy
import errno
import gc
import importlib.util
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import warnings

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP / "scripts"))
HOSTED = "https://inference-api.nvidia.com/v1"
LOCAL = "http://127.0.0.1:18123/v1"
STUDENT = {"id": "policy", "local": True, "base_url": "${oc.env:SYNTH_STUDENT_BASE_URL}", "api_key": "EMPTY",
           "context_window": 262144, "max_output_tokens": 131072,
           "chat_template_kwargs": {"enable_thinking": True, "truncate_history_thinking": False},
           "sampling_overrides": {"temperature": 1.0, "top_p": 1.0},
           "checkpoint": "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/models/swonlysc-step0014/hf"}


def upstream(name):
    """Baihe's pristine module (upstream/<name>.py) under a separate module name."""
    spec = importlib.util.spec_from_file_location("upstream_" + name, APP / "upstream" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contract_settings():
    """SETTINGS of docs/port-contract.md: the app's deepswe.json with models sc14 (local), kimi, glm."""
    settings = json.loads((APP / "configs/deepswe.json").read_text())
    return dict(settings, models={"sc14": STUDENT, "kimi": settings["models"]["kimi"], "glm": settings["models"]["glm"]})


def expect(condition, message):
    print(("PASS " if condition else "FAIL ") + message, flush=True)
    if not condition:
        raise SystemExit(1)


class Response:
    def __init__(self, status, body, headers=None):
        self.status, self.body, self.headers = status, json.dumps(body).encode(), headers or {}

    async def read(self):
        return self.body

    def release(self):
        pass


COMPLETION = {"id": "x", "object": "chat.completion", "choices": [{"index": 0, "finish_reason": "stop",
              "message": {"role": "assistant", "content": "ok"}}],
              "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}
GATEWAY_FULL = {"error": {"message": "error querying the database: FATAL: sorry, too many clients already"}}


def check_http():
    import deepswe_campaign
    import deepswe_http

    root = Path(tempfile.mkdtemp(prefix="atp-http-check-", dir="/tmp"))
    campaign, launch = root / "campaign", root / "campaign/launch"
    launch.mkdir(parents=True)
    for key in ("ATP_DEEPSWE_PRIORITY_1_API_KEY", "ATP_DEEPSWE_PRIORITY_2_API_KEY", "ATP_DEEPSWE_UPSTREAM_STREAM"):
        os.environ.pop(key, None)
    # Test-only credential in this throwaway process; it is the only *_API_KEY set here.
    os.environ.update(ATP_DEEPSWE_CAMPAIGN=str(campaign), ATP_DEEPSWE_LAUNCH_DIR=str(launch),
                      ATP_DEEPSWE_POLICY_BASE_URL=HOSTED, ATP_DEEPSWE_AGENT_TIMEOUT="60",
                      NVIDIA_API_KEY="test-only-hosted-credential")
    shared = campaign / "rate-limits/nvidia-api.json"

    def call(url, replies, model="sc14", internal=False, headers=None, payload=None):
        os.environ["ATP_DEEPSWE_MODEL_ALIAS"] = model
        sent = []

        async def send(**kwargs):
            sent.append(kwargs)
            return replies.pop(0)

        kwargs = {"method": "POST", "url": url, "headers": headers or {"Authorization": "Bearer EMPTY"},
                  "json": payload or {"model": "policy", "messages": [{"role": "user", "content": "hi"}]},
                  "_internal": internal}
        started = time.time()
        try:
            response = asyncio.run(deepswe_http.request_with_retry(send, kwargs))
        except Exception as error:  # noqa: BLE001 - reported to the caller
            return sent, error, time.time() - started
        return sent, response, time.time() - started

    def journal():
        return [json.loads(line) for line in (launch / "http.jsonl").read_text().splitlines()]

    # 1. Local student: caller's Authorization untouched, no credential binding, reasoning still canonicalized.
    payload = {"model": "policy", "messages": [{"role": "user", "content": "hi"},
               {"role": "assistant", "content": "a", "reasoning": "r", "reasoning_content": "r"}]}
    sent, response, _ = call(LOCAL + "/chat/completions", [Response(200, COMPLETION)], payload=payload)
    expect(response.status == 200 and len(sent) == 1, "local call returns the stub response after one send")
    expect(sent[0]["headers"].get("Authorization") == "Bearer EMPTY", "local call keeps Authorization 'Bearer EMPTY'")
    expect("reasoning" not in sent[0]["json"]["messages"][1] and sent[0]["json"]["messages"][1]["reasoning_content"] == "r",
           "canonicalize_reasoning still applied to the local payload")
    expect(all("credential_id" not in event for event in journal()), "local journal events carry no credential_id")

    # 2. A shared hosted-gateway cooldown does not delay the local student...
    deepswe_http.cooldown(shared, time.time() + 3600)
    sent, response, elapsed = call(LOCAL + "/chat/completions", [Response(200, COMPLETION)])
    expect(response.status == 200 and elapsed < 5, f"local call ignores an active shared cooldown ({elapsed:.2f}s)")
    # ...but a hosted call honours it (cooldown beyond the 60 s deadline -> RateLimitDeadline, nothing sent).
    sent, error, _ = call(HOSTED + "/chat/completions", [Response(200, COMPLETION)], model="kimi")
    expect(isinstance(error, deepswe_http.RateLimitDeadline) and not sent, "hosted call waits on the shared cooldown")
    shared.unlink()

    # 3. A gateway-capacity-shaped 401 from a local server is returned as-is; no shared cooldown is written.
    sent, response, _ = call(LOCAL + "/chat/completions", [Response(401, GATEWAY_FULL), Response(200, COMPLETION)])
    expect(response.status == 401 and len(sent) == 1 and not shared.exists(),
           "local 401 is not treated as hosted gateway capacity")
    expect(journal()[-1].get("retryable_gateway_capacity") is False, "journal marks it non-retryable")

    # 4. A local 429 uses the model-scoped cooldown only.
    sent, response, _ = call(LOCAL + "/chat/completions",
                             [Response(429, {"error": "busy"}, {"Retry-After": "0"}), Response(200, COMPLETION)])
    expect(response.status == 200 and len(sent) == 2, "local 429 is retried")
    expect((campaign / "rate-limits/nvidia-api-sc14.json").exists() and not shared.exists(),
           "local 429 writes only the model-scoped cooldown")

    # 5. Hosted kimi call: bound NVIDIA credential replaces the caller's Authorization.
    sent, response, _ = call(HOSTED + "/chat/completions", [Response(200, COMPLETION)], model="kimi",
                             headers={"Authorization": "Bearer something-else"})
    expect(response.status == 200 and sent[0]["headers"]["Authorization"] == "Bearer test-only-hosted-credential",
           "hosted call gets the bound credential")
    expect(journal()[-1].get("credential_id") == "existing", "hosted journal records credential_id 'existing'")
    expect("test-only-hosted-credential" not in (launch / "http.jsonl").read_text(), "journal never contains the key")

    # 6. Hosted gateway capacity: retried after a shared cooldown.
    sent, response, _ = call(HOSTED + "/chat/completions", [Response(401, GATEWAY_FULL), Response(200, COMPLETION)],
                             model="kimi")
    expect(response.status == 200 and len(sent) == 2 and shared.exists(), "hosted gateway capacity uses the shared cooldown")

    # 7. Internal Gym calls are untouched; chat calls without the configured hosted endpoint are refused.
    sent, response, _ = call("http://10.0.0.1:1234/run", [Response(200, {})], internal=True,
                             headers={"Authorization": "Bearer internal"})
    expect(sent[0]["headers"]["Authorization"] == "Bearer internal", "internal call keeps its headers")
    del os.environ["ATP_DEEPSWE_POLICY_BASE_URL"]
    sent, error, _ = call(LOCAL + "/chat/completions", [Response(200, COMPLETION)])
    expect(isinstance(error, ValueError) and not sent, "chat call without ATP_DEEPSWE_POLICY_BASE_URL is refused")
    os.environ["ATP_DEEPSWE_POLICY_BASE_URL"] = HOSTED

    # 8. infrastructure_error is Baihe's (the router-connection markers were reverted for kimi parity): an unresolved
    #    local transport error classifies exactly as upstream.
    for message in ("ClientConnectorError: Cannot connect to host 127.0.0.1:18123 ssl:default [Connect call failed]",
                    "ClientPayloadError: Response payload is not completed", "ValueError: bad request"):
        expect(deepswe_campaign.infrastructure_error(message) == upstream("deepswe_campaign").infrastructure_error(message),
               f"infrastructure_error({message.split(':')[0]}) == upstream")
    attempt = campaign / "models/sc14/attempts/0-0/attempt-0"
    attempt.mkdir(parents=True)
    (attempt / "started.json").write_text(json.dumps({"deadline": time.time() + 3600}))
    with (attempt / "http.jsonl").open("w") as stream:
        stream.write(json.dumps({"event": "request", "call_id": "c1", "http_attempt": 1, "payload": {"messages": []}}) + "\n")
        stream.write(json.dumps({"event": "error", "call_id": "c1", "http_attempt": 1, "outcome": "unknown",
                                 "error_type": "ClientConnectorError",
                                 "error": "Cannot connect to host 127.0.0.1:18123 ssl:default"}) + "\n")
    original = upstream("deepswe_campaign").generation_failure(attempt)
    expect(deepswe_campaign.generation_failure(attempt) == original,
           f"lost router connection: {original!r}, same as upstream (needs_review)")
    print("http checks passed; scratch:", root)


def check_config(scratch):
    import deepswe_campaign as campaign
    import deepswe_stage
    import run_deepswe

    original_stage, original_run = upstream("deepswe_stage"), upstream("run_deepswe")
    os.environ.pop("ATP_DEEPSWE_SOURCE", None)
    staged = deepswe_stage.stage(APP, Path(scratch))
    os.environ["SYNTH_STUDENT_BASE_URL"] = LOCAL  # set by student.client.ensure_router() in real runs
    settings = contract_settings()
    staged = dict(staged, settings=settings, env=dict(staged["env"], SYNTH_STUDENT_BASE_URL=LOCAL))
    os.environ.update(staged["env"])
    expect(staged["env"]["ATP_DEEPSWE_POLICY_BASE_URL"] == HOSTED, "staged ATP_DEEPSWE_POLICY_BASE_URL is the hosted URL")
    root = Path(tempfile.mkdtemp(prefix="atp-config-check-", dir="/tmp"))

    def ports_free(value):
        if isinstance(value, dict):
            return {key: None if key == "port" else ports_free(item) for key, item in value.items()}
        if isinstance(value, list):
            return [ports_free(item) for item in value]
        return value

    for model in ("kimi", "glm", "sc14"):
        launch = root / "configs" / model
        # Baihe's first, so launch/gym.yaml (resolved below) is the new one.
        old = json.loads(original_stage.build_config(staged, root, launch, model, staged["dataset"]).read_text())
        new = json.loads(deepswe_stage.build_config(staged, root, launch, model, staged["dataset"]).read_text())
        new_command = run_deepswe.gym_command(staged, launch / "gym.yaml", launch, settings, 7, model)
        old_command = original_run.gym_command(staged, launch / "gym.yaml", launch, settings, 7)
        if model != "sc14":
            expect(ports_free(new) == ports_free(old), f"{model}: gym.yaml identical to Baihe's build_config (ports aside)")
            expect(new_command == old_command, f"{model}: gym eval run command identical to Baihe's")
        else:
            changed = sorted(key for key in new if ports_free(new[key]) != ports_free(old.get(key)))
            print("sc14: top-level gym.yaml keys that differ from Baihe's build_config:", changed)
        model_server = new["policy_model"]["responses_api_models"]["vllm_model"]
        agent = new[deepswe_stage.AGENT_NAME]["responses_api_agents"]["opencode_sandboxed_agent"]
        resolved = run_deepswe.resolved_config(launch / "gym.yaml")
        resolved_server = resolved["policy_model"]["responses_api_models"]["vllm_model"]
        api_key = resolved_server["api_key"]
        print(json.dumps({"model": model, "policy_base_url": new["policy_base_url"],
                          "resolved_base_url": resolved_server["base_url"], "policy_model_name": new["policy_model_name"],
                          "policy_api_key_placeholder": new["policy_api_key"],
                          "resolved_api_key": "EMPTY" if api_key == "EMPTY" else
                          ("<NVIDIA_API_KEY from APP/.env>" if api_key == os.environ["NVIDIA_API_KEY"] else "<other>"),
                          "sampling_overrides": model_server["sampling_overrides"],
                          "chat_template_kwargs": resolved_server["chat_template_kwargs"],
                          "opencode_max_context_window": agent["opencode_max_context_window"],
                          "gym_eval_run_sampling": new_command[new_command.index("--temperature"):][:6]}, indent=1))

    # run_model's launch environment (collect stubbed: no Gym process, no model calls, no sandboxes).
    captured = {}

    async def fake_collect(staged_, config, launch, environment, settings_, concurrency, stopped, label):
        resolved = run_deepswe.resolved_config(config)
        server = resolved["policy_model"]["responses_api_models"]["vllm_model"]
        agent = resolved[deepswe_stage.AGENT_NAME]["responses_api_agents"]["opencode_sandboxed_agent"]
        captured[label] = {"environment": dict(environment), "command":
                           run_deepswe.gym_command(staged_, config, launch, settings_, concurrency,
                                                   environment["ATP_DEEPSWE_MODEL_ALIAS"]),
                           "resolved": {"base_url": server["base_url"], "model": server["model"],
                                        "api_key": "EMPTY" if server["api_key"] == "EMPTY" else
                                        ("<NVIDIA_API_KEY>" if server["api_key"] == os.environ["NVIDIA_API_KEY"] else "<other>"),
                                        "chat_template_kwargs": server["chat_template_kwargs"],
                                        "sampling_overrides": server["sampling_overrides"],
                                        "opencode_max_context_window": agent["opencode_max_context_window"]}}
        stopped.set()
        return 0

    async def launch(module, path):
        module.collect = fake_collect
        local = copy.deepcopy(staged)
        manifest = campaign.create_campaign(path, staged["dataset"], settings, shards=1, task_limit=1, repeats=1)
        local["settings"] = manifest["settings"]
        os.environ["ATP_DEEPSWE_CAMPAIGN"] = str(path)
        results = {}
        for model in ("sc14", "kimi"):
            stopped = asyncio.Event()
            await module.run_model(campaign, local, path, model, 0, time.time() + 6 * 3600, stopped)
            results[model] = captured.pop(f"{model}/0")
        return results

    new = asyncio.run(launch(run_deepswe, root / "campaign-new"))
    old = asyncio.run(launch(original_run, root / "campaign-old"))
    volatile = {"ATP_DEEPSWE_CAMPAIGN", "ATP_DEEPSWE_LAUNCH_DIR", "NEMO_GYM_RUN_ID", "RAY_TMPDIR", "ATP_DEEPSWE_JOB_DEADLINE"}
    for model in ("sc14", "kimi"):
        environment = new[model]["environment"]
        print(json.dumps({"model": model, "launch config (resolved)": new[model]["resolved"],
                          "ATP_DEEPSWE_MAX_OUTPUT_TOKENS (OpenCode limit.output)":
                          environment["ATP_DEEPSWE_MAX_OUTPUT_TOKENS"],
                          "ATP_DEEPSWE_POLICY_BASE_URL": environment["ATP_DEEPSWE_POLICY_BASE_URL"],
                          "ATP_DEEPSWE_UPSTREAM_STREAM": environment["ATP_DEEPSWE_UPSTREAM_STREAM"],
                          "SYNTH_STUDENT_BASE_URL in Gym env": "SYNTH_STUDENT_BASE_URL" in environment}, indent=1))
        expect(environment["ATP_DEEPSWE_POLICY_BASE_URL"] == HOSTED, f"{model}: launch keeps the hosted policy base URL")
        differing = sorted(key for key in set(environment) | set(old[model]["environment"])
                           if key not in volatile and environment.get(key) != old[model]["environment"].get(key))
        if model == "kimi":
            expect(not differing, "kimi: launch environment identical to Baihe's run_model (launch paths aside)")
        else:
            print("sc14: launch environment keys that differ from Baihe's run_model:", differing)
    expect(new["sc14"]["environment"]["ATP_DEEPSWE_MAX_OUTPUT_TOKENS"] == "131072", "sc14: OpenCode limit.output 131072")
    expect(new["sc14"]["resolved"] == {"base_url": LOCAL, "model": "policy", "api_key": "EMPTY",
                                       "chat_template_kwargs": STUDENT["chat_template_kwargs"],
                                       "sampling_overrides": {"temperature": 1.0, "top_p": 1.0, "max_tokens": 131072},
                                       "opencode_max_context_window": 262144},
           "sc14: resolved launch config = local router, 'EMPTY', chat_template_kwargs, top_p 1.0, context 262144")
    expect(new["kimi"]["resolved"] == old["kimi"]["resolved"] and new["kimi"]["resolved"]["base_url"] == HOSTED
           and new["kimi"]["resolved"]["api_key"] == "<NVIDIA_API_KEY>", "kimi: resolved launch config unchanged (hosted)")
    print("config checks passed; scratch:", root)


class SlowFsync:
    """`os` of one deepswe_capture module under test. fsync of a local stdout.log/stderr.log copy sleeps plan(path)
    seconds (or raises the OSError plan returns), fails if its descriptor no longer names the same file (closed under
    the write), then fsyncs. Other fsyncs are skipped: only the stream copies are under test, and Lustre latency stays
    out of the loop-lag measurement."""

    def __init__(self, plan):
        self.plan, self.lock, self.calls, self.in_flight, self.closed_under_write = plan, threading.Lock(), [], {}, []

    def __getattr__(self, name):
        return getattr(os, name)

    def fsync(self, fd):
        path = os.readlink(f"/proc/self/fd/{fd}")
        if not path.endswith(("/stdout.log", "/stderr.log")):
            return None
        before, delay = os.fstat(fd), self.plan(path)
        call = {"stream": Path(path).name, "thread": threading.current_thread().name, "start": time.monotonic()}
        with self.lock:
            self.calls.append(call)
            self.in_flight[path] = self.in_flight.get(path, 0) + 1
        try:
            if isinstance(delay, OSError):
                raise delay
            time.sleep(delay)
            try:
                after = os.fstat(fd)
            except OSError:
                after = None
            if after is None or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
                self.closed_under_write.append(path)
                raise OSError(errno.EBADF, "file closed under an in-flight write", path)
            return os.fsync(fd)
        finally:
            call["end"] = time.monotonic()
            with self.lock:
                self.in_flight[path] -= 1


class FakeSession:
    """PTY session of a command printing one binary chunk per stream every 20 ms, `chunks` times or until killed (then
    a tail chunk); records what read()/read_stderr() handed to their caller."""

    def __init__(self, chunks):
        self.session_id, self.killed = "check-session", False
        self.queues = {"stdout": asyncio.Queue(), "stderr": asyncio.Queue()}
        self.emitted = {"stdout": bytearray(), "stderr": bytearray()}
        self.handed = {"stdout": bytearray(), "stderr": bytearray()}
        self.exit, self.stopped = asyncio.get_running_loop().create_future(), asyncio.Event()
        self.producer = asyncio.create_task(self.produce(chunks))

    def emit(self, name, data):
        self.emitted[name] += data
        self.queues[name].put_nowait(data)

    async def produce(self, chunks):
        index = 0
        while index < chunks and not self.stopped.is_set():
            await asyncio.sleep(0.02)
            for name in ("stdout", "stderr"):
                self.emit(name, f"{name} {index:05d} ".encode() + bytes(range(256)) * (1 + index % 3) + b"\n")
            index += 1
        if self.killed:
            self.emit("stdout", b"terminated \x00\xff tail\n")
        for queue in self.queues.values():
            queue.put_nowait(None)
        self.exit.set_result(143 if self.killed else 0)

    async def read_stream(self, name):
        chunk = await self.queues[name].get()
        if chunk is None:
            self.queues[name].put_nowait(None)
            return b""
        self.handed[name] += chunk
        return chunk

    async def read(self):
        return await self.read_stream("stdout")

    async def read_stderr(self):
        return await self.read_stream("stderr")

    async def wait_exit(self):
        return await asyncio.shield(self.exit)

    async def close(self, *, preserve_session=False):
        self.stopped.set()
        self.producer.cancel()
        await asyncio.gather(self.producer, return_exceptions=True)


class FakeSandbox:
    """Sandbox for run_opencode: process control (_process_control_command) kills the FakeSession, or fails."""

    def __init__(self, chunks, control_ok=True):
        self.chunks, self.control_ok, self.session = chunks, control_ok, None
        self.pty = SimpleNamespace(create=self.create)

    async def create(self, command, pty):
        self.session = FakeSession(self.chunks)
        return self.session

    async def exec(self, command, timeout_s=None, cwd=None):
        if "control_channel" in command:
            if not self.control_ok:
                return SimpleNamespace(return_code=1, stdout="", stderr="simulated process-control failure")
            self.session.killed = True
            self.session.stopped.set()
            proof = {"confirmed": True, "control_channel": "sandbox_exec", "process": {}, "signals": ["SIGTERM"],
                     "wrapper_forced_stop": False, "wrapper": {"confirmed": True, "process": {}, "signals": []}}
            return SimpleNamespace(return_code=0, stdout=json.dumps(proof), stderr="")
        absent = command.startswith("test -f") and "stream-writer-error" in command
        return SimpleNamespace(return_code=1 if absent else 0, stdout="", stderr="")

    async def download(self, remote, local):
        name = remote.rsplit("/", 1)[-1]
        if name in ("stdout.log", "stderr.log"):
            data = bytes(self.session.emitted[name.split(".")[0]])
        elif name == "stream-writers-complete.json":
            data = json.dumps({"stdout_exit_code": 0, "stderr_exit_code": 0}).encode()
        else:
            data = b"archive"
        Path(local).write_bytes(data)


class DrainCancels(asyncio.Task):
    """Task logging each cancel() of a run_opencode drain task, with whether its stream's fsync was then in flight."""
    fsync, log = None, []

    def cancel(self, msg=None):
        frame = self.get_coro().cr_frame
        if frame is not None and frame.f_code.co_name == "drain" and DrainCancels.fsync is not None:
            name = frame.f_locals["name"]
            with DrainCancels.fsync.lock:
                busy = any(count for path, count in DrainCancels.fsync.in_flight.items() if path.endswith("/" + name))
            DrainCancels.log.append({"stream": name, "write_in_flight": busy})
        return super().cancel(msg)


async def loop_lag(stop):
    """(monotonic time, lateness of a 10 ms timer = how long the event loop was blocked), until `stop` is set."""
    loop, samples = asyncio.get_running_loop(), []
    while not stop.is_set():
        started = loop.time()
        await asyncio.sleep(0.01)
        samples.append((time.monotonic(), loop.time() - started - 0.01))
    return samples


def check_capture(scratch):
    import deepswe_capture as fixed  # this app's scripts/

    baihe = upstream("deepswe_capture")
    root = Path(tempfile.mkdtemp(prefix="atp-capture-check-", dir=scratch or "/tmp"))
    (root / "tmp").mkdir()
    tempfile.tempdir = str(root / "tmp")  # _download's temporary directories
    os.environ["ATP_DEEPSWE_CAMPAIGN"] = str(root)
    reports = []

    class Reports(logging.Handler):  # e.g. "Future exception was never retrieved" (asyncio logger)
        def emit(self, record):
            reports.append(record.getMessage())

    logging.getLogger("asyncio").addHandler(Reports(logging.WARNING))
    labels = iter(range(100))

    async def run(module, plan, *, chunks, timeout_s=60, control_ok=True):
        fsync = module.os = SlowFsync(plan)
        DrainCancels.fsync, DrainCancels.log = fsync, []
        attempt = root / "models/check/attempts" / f"{next(labels)}-0" / "attempt-0"
        attempt.mkdir(parents=True)
        sandbox = FakeSandbox(chunks, control_ok)
        body = {"_atp_attempt_dir": str(attempt), "_atp_deadline": time.time() + 600}

        async def request_json():
            return body

        stop = asyncio.Event()
        lag = asyncio.create_task(loop_lag(stop))
        try:
            result = await module.run_opencode(sandbox, "opencode run check", SimpleNamespace(json=request_json),
                                               timeout_s)
            outcome = f"returned {result.return_code}"
        except Exception as error:  # noqa: BLE001 - compared below
            outcome = f"{type(error).__name__} {getattr(error, 'errno', None) or ''}".strip()
        with fsync.lock:
            abandoned = sum(fsync.in_flight.values())  # stream writes still running after run_opencode ended
        stop.set()
        samples = await lag
        for _ in range(500):  # let abandoned writes (if any) finish before the files are inspected
            with fsync.lock:
                if not any(fsync.in_flight.values()):
                    break
            await asyncio.sleep(0.01)
        calls = [call for call in fsync.calls if "end" in call]
        window = (min([call["start"] for call in calls] or [0.0]), max([call["end"] for call in calls] or [0.0]) + 0.05)
        return SimpleNamespace(
            outcome=outcome, fsync=fsync, cancels=DrainCancels.log, session=sandbox.session, abandoned=abandoned,
            lag=max([late for at, late in samples if window[0] <= at <= window[1]] or [0.0]),  # while copying
            # 10 ms timer ticks completed inside each stalled (>= 0.2 s) fsync; 0 = the loop was blocked by it
            ticks=min([sum(call["start"] + 0.05 <= at <= call["end"] - 0.01 for at, _ in samples)
                       for call in calls if call["end"] - call["start"] >= 0.2] or [0]),
            threads={call["thread"].rstrip("0123456789").rstrip("_") for call in fsync.calls},
            events=[json.loads(line)["event"] for line in (attempt / "capture-events.jsonl").read_text().splitlines()],
            local={name: (attempt / f"{name}.log").read_bytes() for name in ("stdout", "stderr")},
            streams_complete=(attempt / "streams-complete.json").is_file())

    def enospc_on_third_stdout_fsync():
        count = [0]

        def plan(path):
            if not path.endswith("/stdout.log"):
                return 0
            count[0] += 1
            return OSError(errno.ENOSPC, "No space left on device (injected)", path) if count[0] == 3 else 0
        return plan

    async def cancelled_twice():
        finished = threading.Event()

        def write():
            time.sleep(0.4)
            finished.set()

        task = asyncio.ensure_future(fixed._in_stream_io(write))
        for _ in range(2):  # two separately delivered cancellations while the call runs
            await asyncio.sleep(0.1)
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return task.cancelled() and finished.is_set()

    async def cases():
        asyncio.get_running_loop().set_task_factory(lambda loop, coro, **kw: DrainCancels(coro, loop=loop, **kw))
        out = {"helper": await cancelled_twice()}
        for name, module in (("upstream", baihe), ("app", fixed)):
            out[name, "copy"] = await run(module, lambda path: 0.3, chunks=8)
            # Deadline, then process control fails: run_opencode's finally cancels each drain (task and gather).
            out[name, "cancel"] = await run(module, lambda path: 0.6, chunks=10**9, timeout_s=0.9, control_ok=False)
            out[name, "error"] = await run(module, enospc_on_third_stdout_fsync(), chunks=10**9)
        return out

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = asyncio.run(cases())
        gc.collect()
    streams = ("stdout", "stderr")
    for key, case in out.items():
        if key != "helper":
            print(json.dumps({"module": key[0], "case": key[1], "outcome": case.outcome,
                              "max_loop_lag_s": round(case.lag, 3), "min_ticks_per_stalled_fsync": case.ticks,
                              "stream_fsyncs": len(case.fsync.calls), "fsync_threads": sorted(case.threads),
                              "drain_cancels_during_write": sum(c["write_in_flight"] for c in case.cancels),
                              "writes_running_at_return": case.abandoned}))
    for name in ("upstream", "app"):
        case = out[name, "copy"]
        expect(case.outcome == "returned 0" and case.streams_complete and "agent_snapshot_saved" in case.events,
               f"{name}: returns 0 with streams-complete.json and a final snapshot")
        expect(all(case.local[s] == case.session.emitted[s] == case.session.handed[s] for s in streams),
               f"{name}: stdout.log/stderr.log hold the command's output bytes, in order")
    case = out["app", "copy"]
    expect(case.local == out["upstream", "copy"].local, "app copies byte-identical to upstream's")
    per_stream = [[call for call in case.fsync.calls if call["stream"] == s + ".log"] for s in streams]
    expect(all(len(calls) == 8 and all(a["end"] <= b["start"] for a, b in zip(calls, calls[1:]))
               for calls in per_stream), "app: one write + fsync per chunk, one at a time per stream")
    original = out["upstream", "copy"]
    expect(original.threads == {"MainThread"} and original.ticks == 0 and original.lag >= 0.25,
           f"upstream: each 0.3 s fsync blocks the event loop (no timer tick during any; max lag {original.lag:.2f} s)")
    expect(case.threads == {"atp-stream-io"} and case.ticks >= 1 and case.lag < 0.3,
           f"app: fsyncs run in atp-stream-io threads and the loop ticks during each (min {case.ticks} ticks; "
           f"max lag {case.lag:.3f} s)")
    expect(out["helper"], "app: _in_stream_io cancelled twice waits for its call, then ends cancelled")
    case = out["app", "cancel"]
    expect(case.outcome == "TimeoutError" and "generation_not_quiesced" in case.events and not case.streams_complete,
           "app: deadline + failed process control -> TimeoutError, streams incomplete")
    expect(all(sum(c["write_in_flight"] for c in case.cancels if c["stream"] == s + ".log") >= 2 for s in streams),
           "app: each drain was cancelled twice while its write was in flight")
    expect(all(case.local[s] == case.session.handed[s] for s in streams) and not case.fsync.closed_under_write
           and case.abandoned == 0, "app: every handed-off chunk is in its file; no file closed under a write; "
                                    "no write still running when run_opencode ended")
    expect(case.ticks >= 1 and case.lag < 0.3,
           f"app: the loop kept running during the 0.6 s fsyncs (min {case.ticks} ticks; max lag {case.lag:.3f} s)")
    case = out["upstream", "cancel"]
    expect(case.outcome == "TimeoutError" and all(case.local[s] == case.session.handed[s] for s in streams)
           and not any(c["write_in_flight"] for c in case.cancels) and case.ticks == 0 and case.lag >= 0.5,
           f"upstream: same outcome, never cancelled mid-write because each write blocked the loop "
           f"(max lag {case.lag:.2f} s)")
    app, original = out["app", "error"], out["upstream", "error"]
    expect(app.outcome == original.outcome == "OSError 28" and app.events == original.events
           and "agent_streams_incomplete" in app.events,
           f"app: an fsync error ends run_opencode as upstream ({app.outcome}; events {app.events})")
    expect(app.local["stdout"] == original.local["stdout"] == app.session.handed["stdout"],
           "app: stdout.log up to the failed chunk, as upstream")
    leaks = [str(w.message) for w in caught if issubclass(w.category, (RuntimeWarning, ResourceWarning))]
    expect(not reports and not leaks, f"no unretrieved-exception reports or warnings ({reports + leaks})")
    print("capture checks passed; scratch:", root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=("http", "config", "capture"))
    parser.add_argument("--scratch", type=Path)
    args = parser.parse_args()
    {"http": check_http, "config": lambda: check_config(args.scratch),
     "capture": lambda: check_capture(args.scratch)}[args.check]()
