"""Adversarial semantics review of the staged drain() fix (reviewer harness, independent of the implementer's fakes).

Usage (explicit path, private scratch; never discovery, never /tmp):
  <controller python> -I -B review_adversarial_drain.py <scratch dir> <mode> [seed] [runs] [loop]
modes: fuzz | compare | spin | saturate | micro
loop: asyncio | uvloop

Variants under test (all run the REAL run_opencode of the given module):
  upstream = live atp/scripts/deepswe_capture.py (== Baihe's original)
  fixed    = staged atp-fix/app/scripts/deepswe_capture.py
  sftv3    = upstream + glm-rollouts-sftv3 capture_patch.py NEW block, in memory (negative control)
"""
import sys

sys.dont_write_bytecode = True

import asyncio
import errno
import gc
import importlib.util
import json
import logging
import os
from pathlib import Path
import random
import shutil
import tempfile
import threading
import time
from types import SimpleNamespace
import warnings

LIVE = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp")
STAGED = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-tasksall-20261108/atp-fix/app/scripts/deepswe_capture.py")
SFTV3 = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-sftv3-20261107/capture_patch.py")
sys.path.insert(0, str(LIVE / "scripts"))  # deepswe_http (read only; -B: no pycache writes)

from nemo_gym.sandbox.providers.base import SandboxPtyError  # noqa: E402


def load(name, path, source=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if source is None:
        spec.loader.exec_module(module)
    else:
        exec(compile(source, f"<{name}>", "exec"), module.__dict__)
    return module


def variants():
    up = load("rv_upstream", LIVE / "scripts/deepswe_capture.py")
    fx = load("rv_fixed", STAGED)
    patch = load("rv_sftv3_patch", SFTV3)
    text = (LIVE / "scripts/deepswe_capture.py").read_text()
    assert text.count(patch.OLD) == 1
    sf = load("rv_sftv3", LIVE / "scripts/deepswe_capture.py", text.replace(patch.OLD, patch.NEW, 1))
    return {"upstream": up, "fixed": fx, "sftv3": sf}


FAILURES = []
LABELS = [0]
LAG_ONLY = []
REPORTS = []


def check(condition, message):
    if not condition:
        FAILURES.append(message)
        print("FAIL", message, flush=True)
    return condition


class OsProxy:
    """Module-level `os` replacement: stream-copy fsyncs sleep per plan, detect close-under-write (fd identity)."""

    def __init__(self, plan):
        self.plan, self.lock = plan, threading.Lock()
        self.calls, self.in_flight, self.closed_under_write, self.on_loop_thread = [], 0, 0, 0
        self.loop_thread = threading.get_ident()

    def __getattr__(self, name):
        return getattr(os, name)

    def fsync(self, fd):
        try:
            path = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            path = ""
        if not path.endswith(("/stdout.log", "/stderr.log")):
            return None  # JSON markers/events: skip the real fsync (speed); not under test
        before = os.fstat(fd)
        record = {"stream": Path(path).stem, "start": time.monotonic(), "thread": threading.current_thread().name}
        with self.lock:
            self.calls.append(record)
            self.in_flight += 1
            if threading.get_ident() == self.loop_thread:
                self.on_loop_thread += 1
        try:
            delay = self.plan(record["stream"])
            if isinstance(delay, BaseException):
                raise delay
            time.sleep(delay)
            try:
                after = os.fstat(fd)
            except OSError:
                after = None
            if after is None or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
                with self.lock:
                    self.closed_under_write += 1
                raise OSError(errno.EBADF, "closed under write")
            record["size"] = after.st_size
        finally:
            record["end"] = time.monotonic()
            with self.lock:
                self.in_flight -= 1


class FakeSession:
    """Mirrors OpenSandboxPtySession's reader contract: unbounded queues, sticky EOF (b'' on clean exit, else the
    exit error), wait_exit via shield, close() settles exit and EOFs the readers."""

    def __init__(self, rng, spec):
        self.session_id, self.rng, self.spec = "rv-session", rng, spec
        self.queues = {"stdout": asyncio.Queue(), "stderr": asyncio.Queue()}
        self.emitted = {"stdout": bytearray(), "stderr": bytearray()}
        self.handed = {"stdout": bytearray(), "stderr": bytearray()}
        self.read_times = {"stdout": [], "stderr": []}
        self.exit = asyncio.get_running_loop().create_future()
        self.killed, self.stopped, self.closed = False, asyncio.Event(), False
        self.producer = asyncio.create_task(self.produce())

    def emit(self, name, data):
        self.emitted[name] += data
        self.queues[name].put_nowait(data)

    def finish(self, code=None, error=None):
        if not self.exit.done():
            if error is not None:
                self.exit.set_exception(error)
                self.exit.exception()
            else:
                self.exit.set_result(code)
        for queue in self.queues.values():
            queue.put_nowait(None)

    async def produce(self):
        spec, rng = self.spec, self.rng
        try:
            for index in range(spec["chunks"]):
                if self.stopped.is_set():
                    break
                await asyncio.sleep(rng.uniform(0, spec["gap"]))
                for name in ("stdout", "stderr"):
                    if name == "stderr" and spec["silent_stderr"]:
                        continue
                    if rng.random() < spec["emit_p"]:
                        self.emit(name, f"{name}#{index}:".encode() + rng.randbytes(rng.randint(1, spec["size"])))
                if spec["pty_error_at"] is not None and index == spec["pty_error_at"]:
                    self.finish(error=SandboxPtyError("PTY connection closed unexpectedly (close code 1006)"))
                    return
            if self.killed:
                self.emit("stdout", b"terminated-tail\x00\xff\n")
            self.finish(code=143 if self.killed else 0)
        except asyncio.CancelledError:
            self.finish(error=SandboxPtyError("PTY session closed before process exit"))
            raise

    async def read_stream(self, name):
        chunk = await asyncio.wait_for(self.queues[name].get(), timeout=None)
        if chunk is None:
            self.queues[name].put_nowait(None)
            if self.exit.done() and not self.exit.cancelled() and self.exit.exception() is None:
                return b""
            raise self.exit.exception() if self.exit.done() else SandboxPtyError("closed")
        self.handed[name] += chunk
        self.read_times[name].append(time.monotonic())
        return chunk

    async def read(self, *, timeout_s=None):
        return await self.read_stream("stdout")

    async def read_stderr(self, *, timeout_s=None):
        return await self.read_stream("stderr")

    async def wait_exit(self, *, timeout_s=None):
        return await asyncio.wait_for(asyncio.shield(self.exit), timeout=timeout_s)

    async def close(self, *, preserve_session=False):
        self.closed = True
        self.stopped.set()
        if not self.producer.done():
            self.producer.cancel()
        await asyncio.gather(self.producer, return_exceptions=True)
        self.finish(error=SandboxPtyError("PTY session closed before process exit"))


class FakeSandbox:
    def __init__(self, rng, spec):
        self.rng, self.spec, self.session = rng, spec, None
        self.pty = SimpleNamespace(create=self.create)

    async def create(self, command, pty):
        self.session = FakeSession(self.rng, self.spec)
        return self.session

    async def exec(self, command, timeout_s=None, cwd=None):
        await asyncio.sleep(self.rng.uniform(0, self.spec["exec_delay"]))
        if "control_channel" in command:
            if not self.spec["control_ok"]:
                return SimpleNamespace(return_code=1, stdout="", stderr="simulated control failure")
            self.session.killed = True
            self.session.stopped.set()
            proof = {"confirmed": True, "control_channel": "sandbox_exec", "process": {}, "signals": ["SIGTERM"],
                     "wrapper_forced_stop": False, "wrapper": {"confirmed": True, "process": {}, "signals": []}}
            return SimpleNamespace(return_code=0, stdout=json.dumps(proof), stderr="")
        absent = command.startswith("test -f") and "stream-writer-error" in command
        return SimpleNamespace(return_code=1 if absent else 0, stdout="", stderr="")

    async def download(self, remote, local):
        await asyncio.sleep(0)
        name = remote.rsplit("/", 1)[-1]
        if name in ("stdout.log", "stderr.log"):
            data = bytes(self.session.emitted[name.split(".")[0]])
        elif name == "stream-writers-complete.json":
            data = json.dumps({"stdout_exit_code": 0, "stderr_exit_code": 0}).encode()
        else:
            data = b"archive"
        Path(local).write_bytes(data)


class CancelLog(asyncio.Task):
    """Counts cancel() calls per drain task (spin detection)."""
    counts = {}

    def cancel(self, msg=None):
        coro = self.get_coro()
        frame = getattr(coro, "cr_frame", None)
        if frame is not None and frame.f_code.co_name == "drain":
            CancelLog.counts[frame.f_locals.get("name")] = CancelLog.counts.get(frame.f_locals.get("name"), 0) + 1
        return super().cancel(msg)


def open_fds():
    return len(os.listdir("/proc/self/fd"))


async def lag_probe(stop, samples):
    loop = asyncio.get_running_loop()
    while not stop.is_set():
        start = loop.time()
        await asyncio.sleep(0.005)
        samples.append(loop.time() - start - 0.005)


async def run_one(module, root, label, rng, spec, *, disrupt=None, cancel_trigger=None, settle=True):
    """Run the real run_opencode once; returns observations. disrupt: None | ('cancel', t, n) | ('anyio', t)."""
    proxy = module.os = OsProxy(spec["plan"])
    LABELS[0] += 1
    attempt = root / "models/rv/attempts" / f"{LABELS[0]}-0" / "attempt-0"
    attempt.mkdir(parents=True)
    if spec.get("devfull"):
        os.symlink("/dev/full", attempt / "stdout.log")
    sandbox = FakeSandbox(rng, spec)
    body = {"_atp_attempt_dir": str(attempt), "_atp_deadline": time.time() + 600}

    async def request_json():
        return body

    request = SimpleNamespace(json=request_json)
    outcome = {}

    async def target():
        try:
            result = await module.run_opencode(sandbox, "opencode run rv", request, spec["timeout"])
            outcome["value"] = f"returned {result.return_code} {result.error_type}"
        except BaseException as error:  # noqa: BLE001
            outcome["value"] = f"{type(error).__name__}"
            raise

    stop, samples = asyncio.Event(), []
    probe = asyncio.create_task(lag_probe(stop, samples))
    started = time.monotonic()
    if disrupt is None:
        task = asyncio.create_task(target())
    elif disrupt[0] == "cancel":
        task = asyncio.create_task(target())

        async def canceller():
            if cancel_trigger is not None:
                await cancel_trigger()
            else:
                await asyncio.sleep(disrupt[1])
            for _ in range(disrupt[2]):
                task.cancel()
                await asyncio.sleep(rng.uniform(0, 0.02))
        asyncio.create_task(canceller())
    else:  # anyio level cancellation, as Gym's ClientDisconnectCancellationMiddleware
        import anyio

        async def anyio_host():
            async with anyio.create_task_group() as group:
                async def run_app():
                    try:
                        await target()
                    finally:
                        group.cancel_scope.cancel()

                async def disconnect():
                    if cancel_trigger is not None:
                        await cancel_trigger()
                    else:
                        await anyio.sleep(disrupt[1])
                    outcome["disconnect_at"] = time.monotonic()
                    group.cancel_scope.cancel()
                group.start_soon(run_app)
                group.start_soon(disconnect)
        task = asyncio.create_task(anyio_host())
    bound = spec["bound"]
    done, _ = await asyncio.wait({task}, timeout=bound)
    returned_at = time.monotonic()
    hung = task not in done
    with proxy.lock:
        in_flight_at_return = proxy.in_flight
    # Cleanup: end the remote command/PTY so orphaned drains reach EOF, then wait for every drain.
    session = sandbox.session
    if session is not None:
        session.stopped.set()
        if not session.producer.done():
            session.producer.cancel()
        await asyncio.gather(session.producer, return_exceptions=True)
    if hung:
        done, _ = await asyncio.wait({task}, timeout=30)
        check(task in done, f"{label}: run_opencode still not finished 30 s after the bound")
    drains = [t for t in asyncio.all_tasks() if getattr(t.get_coro(), "__qualname__", "").endswith("drain")]
    if drains:
        finished, pending = await asyncio.wait(drains, timeout=30)
        check(not pending, f"{label}: {len(pending)} drain task(s) never finished")
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    stop.set()
    await probe
    # Let every other background task of this run (shielded snapshots, cancellers, checkpoints) finish.
    others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if others and settle:
        _, still = await asyncio.wait(others, timeout=30)
        check(not still, f"{label}: {len(still)} background task(s) still running 30 s after the run: "
                         f"{[getattr(t.get_coro(), '__qualname__', '?') for t in still]}")
    events = []
    if (attempt / "capture-events.jsonl").is_file():
        events = [json.loads(line)["event"] for line in (attempt / "capture-events.jsonl").read_text().splitlines()]
    local = {}
    for name in ("stdout", "stderr"):
        path = attempt / f"{name}.log"
        local[name] = None if path.is_symlink() else (path.read_bytes() if path.is_file() else None)
    complete = None
    if (attempt / "streams-complete.json").is_file():
        complete = json.loads((attempt / "streams-complete.json").read_text())
    check(outcome.get("value") != "ValueError", f"{label}: run_opencode raised ValueError (harness setup?)")
    return SimpleNamespace(outcome=outcome.get("value"), hung=hung, elapsed=returned_at - started, proxy=proxy,
                           in_flight_at_return=in_flight_at_return, session=session, events=events, local=local,
                           complete=complete, max_lag=max(samples or [0.0]), attempt=attempt)


def invariants(name, label, obs, spec, disrupted):
    p, s = obs.proxy, obs.session
    ok = True
    ok &= check(p.closed_under_write == 0, f"{name}/{label}: {p.closed_under_write} stream file(s) closed under a write")
    if name != "upstream":
        ok &= check(p.on_loop_thread == 0, f"{name}/{label}: {p.on_loop_thread} stream fsync(s) on the event loop thread")
    if s is None:
        return ok
    for stream in ("stdout", "stderr"):
        local = obs.local[stream]
        if local is None:
            if not (spec.get("devfull") and stream == "stdout"):
                ok &= check(s.handed[stream] == b"" or obs.outcome is None,
                            f"{name}/{label}: {stream}.log missing although bytes were handed")
            continue
        handed = bytes(s.handed[stream])
        if spec.get("fsync_error"):
            ok &= check(handed.startswith(local) or local == handed, f"{name}/{label}: {stream} not a prefix")
        else:
            ok &= check(local == handed, f"{name}/{label}: {stream}.log ({len(local)} B) != handed ({len(handed)} B)")
        # durability order: chunk k fsynced before chunk k+1 was read
        calls = [c for c in p.calls if c["stream"] == stream and "end" in c]
        reads = s.read_times[stream]
        for k in range(min(len(calls), len(reads) - 1)):
            if calls[k]["end"] > reads[k + 1] + 1e-4:
                ok &= check(False, f"{name}/{label}: {stream} chunk {k + 1} read before chunk {k} fsync ended")
                break
    if obs.complete is not None:
        last = max([c["end"] for c in p.calls] or [0.0])
        ok &= check(all(obs.local[x] == bytes(s.emitted[x]) for x in ("stdout", "stderr") if obs.local[x] is not None),
                    f"{name}/{label}: streams-complete.json but local copy != emitted")
        ok &= check(p.in_flight == 0, f"{name}/{label}: streams-complete with writes in flight")
    if not disrupted and name == "fixed":
        ok &= check(obs.in_flight_at_return == 0, f"{name}/{label}: {obs.in_flight_at_return} write(s) running after "
                                                  f"run_opencode returned")
    return ok


def random_spec(rng, mode):
    stall_p = rng.choice([0.0, 0.05, 0.2, 0.5])
    stall = rng.uniform(0.05, 0.4)
    error_at = rng.choice([None, None, None, 2, 5])
    counter = {"stdout": 0, "stderr": 0}

    def plan(stream):
        counter[stream] += 1
        if error_at is not None and stream == "stdout" and counter[stream] == error_at:
            return OSError(errno.EIO, "injected fsync EIO")
        return stall if rng.random() < stall_p else rng.uniform(0, 0.002)
    chunks = rng.choice([0, 1, 3, 20, 60, 10**6])
    spec = {"plan": plan, "chunks": chunks, "gap": rng.choice([0.0, 0.002, 0.01]), "size": rng.choice([1, 64, 4096]),
            "silent_stderr": rng.random() < 0.3, "emit_p": rng.choice([1.0, 0.5]),
            "pty_error_at": rng.choice([None, None, None, 3, 15]), "exec_delay": rng.choice([0.0, 0.01, 0.05]),
            "control_ok": rng.random() < 0.8, "timeout": rng.choice([0.15, 0.4, 1.0, 30]),
            "fsync_error": error_at is not None, "devfull": rng.random() < 0.06}
    if chunks == 10**6 and spec["timeout"] == 30:
        spec["timeout"] = 0.5
    spec["drain_timeout"] = rng.choice([15, 0.05, 0.3])
    spec["bound"] = 30
    return spec


async def fuzz(root, seed, runs, names):
    rng = random.Random(seed)
    stats = {}
    for index in range(runs):
        spec_seed = rng.randrange(1 << 30)
        disrupt_kind = rng.choice([None, None, "cancel", "cancel2", "anyio"])
        when = rng.uniform(0.0, 0.6)
        for name, module in names.items():
            r = random.Random(spec_seed)
            spec = random_spec(r, "fuzz")
            module.STREAM_DRAIN_TIMEOUT_S = spec["drain_timeout"]
            disrupt = (None if disrupt_kind is None else ("cancel", when, 1) if disrupt_kind == "cancel"
                       else ("cancel", when, 2) if disrupt_kind == "cancel2" else ("anyio", when))
            label = f"s{seed}-{index}-{name}"
            fds = open_fds()
            reported = len(REPORTS)
            obs = await run_one(module, root, label, r, spec, disrupt=disrupt)
            gc.collect()
            await asyncio.sleep(0)
            gc.collect()
            leaked = open_fds() - fds
            check(leaked <= 0, f"{name}/{label}: {leaked} fd(s) leaked")
            invariants(name, label, obs, spec, disrupt is not None)
            key = (name, disrupt_kind or "none")
            entry = stats.setdefault(key, {"runs": 0, "hung": 0, "max_lag": 0.0, "cuw": 0, "abandoned": 0,
                                           "outcomes": {}, "reports": []})
            entry["reports"] += [m.split("\n")[0] + " | " + m.split("\n")[1][:90] for m in REPORTS[reported:]]
            entry["runs"] += 1
            entry["hung"] += obs.hung
            entry["max_lag"] = max(entry["max_lag"], obs.max_lag)
            entry["cuw"] += obs.proxy.closed_under_write
            entry["abandoned"] += obs.in_flight_at_return
            entry["outcomes"][obs.outcome] = entry["outcomes"].get(obs.outcome, 0) + 1
            shutil.rmtree(obs.attempt.parents[1], ignore_errors=True)
    for key in sorted(stats):
        e = stats[key]
        print(json.dumps({"variant": key[0], "disrupt": key[1], "runs": e["runs"], "hung": e["hung"],
                          "max_loop_lag_s": round(e["max_lag"], 3), "closed_under_write": e["cuw"],
                          "writes_running_at_return": e["abandoned"], "outcomes": e["outcomes"],
                          "asyncio_reports": e["reports"]}), flush=True)


async def compare(root, seed, runs, names):
    """No stalls/cancels: upstream and fixed must give identical outcome, events and bytes."""
    rng = random.Random(seed)
    for index in range(runs):
        spec_seed = rng.randrange(1 << 30)
        results = {}
        for name in ("upstream", "fixed"):
            r = random.Random(spec_seed)
            spec = random_spec(r, "compare")
            spec["plan"] = (lambda stream, _c={"n": 0}, _e=spec["fsync_error"], _r=r: 0.0)
            spec["fsync_error"] = False
            spec["chunks"] = min(spec["chunks"], 60) if spec["chunks"] != 10**6 else 40
            spec["timeout"], spec["drain_timeout"], spec["exec_delay"], spec["gap"] = 30, 15, 0.0, 0.0
            names[name].STREAM_DRAIN_TIMEOUT_S = 15
            obs = await run_one(names[name], root, f"c{seed}-{index}-{name}", r, spec)
            invariants(name, f"c{index}", obs, spec, False)
            results[name] = (obs.outcome, obs.events, obs.local, obs.complete is not None)
            detail = {s: {"local": None if obs.local[s] is None else len(obs.local[s]),
                          "handed": len(obs.session.handed[s]), "emitted": len(obs.session.emitted[s])}
                      for s in ("stdout", "stderr")}
            results[name + "-detail"] = (detail, {k: spec[k] for k in ("chunks", "pty_error_at", "control_ok",
                                                                       "silent_stderr", "devfull", "emit_p")})
            shutil.rmtree(obs.attempt.parents[1], ignore_errors=True)
        same = results["upstream"] == results["fixed"]
        up, fx = results["upstream"], results["fixed"]
        lag_only = (not same and up[:2] == fx[:2] and not up[3] and not fx[3] and
                    all((up[2][s] is None and fx[2][s] is None) or
                        (up[2][s] is not None and fx[2][s] is not None and up[2][s].startswith(fx[2][s]))
                        for s in ("stdout", "stderr")))
        if lag_only:
            LAG_ONLY.append(index)
            print(f"lag-only {index}: same outcome/events, both incomplete; fixed local copy is a prefix of upstream's",
                  results["upstream-detail"], "| fixed", results["fixed-detail"][0], flush=True)
            continue
        check(same, f"compare {index}: upstream {results['upstream'][:2]} vs fixed {results['fixed'][:2]}")
        if not same:
            print("  upstream", results["upstream-detail"], "complete", results["upstream"][3], flush=True)
            print("  fixed   ", results["fixed-detail"], "complete", results["fixed"][3], flush=True)
    print(f"compare: {runs} scenario(s) checked; lag-only differences: {len(LAG_ONLY)}", flush=True)


async def spin(root, names):
    """anyio level cancellation arriving while run_opencode's finally waits for a drain with a stalled write."""
    for name in ("upstream", "sftv3", "fixed"):
        module = names[name]
        module.STREAM_DRAIN_TIMEOUT_S = 15
        stall = 1.0
        state = {"n": 0}

        def plan(stream, state=state):
            state["n"] += 1
            return stall if state["n"] == 6 else 0.0
        rng = random.Random(1)
        spec = {"plan": plan, "chunks": 10**6, "gap": 0.002, "size": 64, "silent_stderr": True, "emit_p": 1.0,
                "pty_error_at": None, "exec_delay": 0.0, "control_ok": False, "timeout": 0.05, "fsync_error": False,
                "devfull": False, "drain_timeout": 15, "bound": 20}
        CancelLog.counts = {}
        loop = asyncio.get_running_loop()
        old_factory = loop.get_task_factory()
        loop.set_task_factory(lambda lp, coro, **kw: CancelLog(coro, loop=lp, **kw))

        async def trigger():  # disconnect once a drain has been cancelled by run_opencode itself (finally/drain wait)
            while not CancelLog.counts:
                await asyncio.sleep(0.001)
        cpu0 = time.process_time()
        obs = await run_one(module, root, f"spin-{name}", rng, spec, disrupt=("anyio", 0), cancel_trigger=trigger)
        cpu = time.process_time() - cpu0
        loop.set_task_factory(old_factory)
        print(json.dumps({"variant": name, "outcome": obs.outcome, "elapsed_s": round(obs.elapsed, 3),
                          "cpu_s": round(cpu, 3), "max_loop_lag_s": round(obs.max_lag, 4),
                          "drain_cancel_calls": CancelLog.counts,
                          "closed_under_write": obs.proxy.closed_under_write,
                          "writes_running_at_return": obs.in_flight_at_return,
                          "final_snapshot": "agent_snapshot_saved" in obs.events,
                          "events": obs.events}), flush=True)
        shutil.rmtree(obs.attempt.parents[1], ignore_errors=True)


class Watchdog:
    """Samples the event-loop thread's stack whenever the loop has not ticked for > threshold seconds."""

    def __init__(self, threshold=0.15):
        self.threshold, self.beat, self.loop_thread = threshold, time.monotonic(), threading.get_ident()
        self.stalls, self.running = [], True
        self.thread = threading.Thread(target=self.watch, name="rv-watchdog", daemon=True)
        self.thread.start()

    async def ticker(self):
        while self.running:
            self.beat = time.monotonic()
            await asyncio.sleep(0.005)

    def watch(self):
        import traceback
        reported = None
        while self.running:
            time.sleep(0.02)
            stale = time.monotonic() - self.beat
            if stale > self.threshold and reported != self.beat:
                frame = sys._current_frames().get(self.loop_thread)
                stack = traceback.extract_stack(frame)[-6:] if frame is not None else []
                self.stalls.append({"stale_s": round(stale, 3),
                                    "stack": [f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in stack]})
                reported = self.beat


async def saturate(root, names, sessions=20):
    """20 concurrent sessions (40 drains > 32 threads), every stream fsync stalls 0.3 s."""
    order = os.environ.get("RV_ORDER", "fixed,sftv3").split(",")
    for name in order:
        module = names[name]
        module.STREAM_DRAIN_TIMEOUT_S = 15
        proxy = OsProxy(lambda stream: 0.3)
        loop = asyncio.get_running_loop()
        default_latency = []
        stop = asyncio.Event()
        dog = Watchdog()
        ticker = asyncio.create_task(dog.ticker())

        async def default_probe():
            while not stop.is_set():
                t0 = time.monotonic()
                await loop.run_in_executor(None, lambda: None)
                default_latency.append(time.monotonic() - t0)
                await asyncio.sleep(0.05)
        probe = asyncio.create_task(default_probe())
        jobs = []
        for i in range(sessions):
            r = random.Random(100 + i)
            spec = {"plan": lambda stream: 0.3, "chunks": 12, "gap": 0.01, "size": 256, "silent_stderr": False,
                    "emit_p": 1.0, "pty_error_at": None, "exec_delay": 0.0, "control_ok": True, "timeout": 60,
                    "fsync_error": False, "devfull": False, "drain_timeout": 15, "bound": 60}
            jobs.append(run_one(module, root, f"sat-{name}-{i}", r, spec, settle=False))
        module.os = proxy
        # run_one replaces module.os per call; keep one shared proxy by patching OsProxy creation
        t0 = time.monotonic()
        results = await asyncio.gather(*jobs)
        elapsed = time.monotonic() - t0
        stop.set()
        await probe
        dog.running = False
        await ticker
        for stall in dog.stalls:
            print(json.dumps({"variant": name, "loop_stall": stall}), flush=True)
        bad = [i for i, o in enumerate(results) if o.local["stdout"] != bytes(o.session.handed["stdout"])
               or o.local["stderr"] != bytes(o.session.handed["stderr"]) or o.hung]
        print(json.dumps({"variant": name, "sessions": sessions, "elapsed_s": round(elapsed, 2),
                          "max_loop_lag_s": round(max(o.max_lag for o in results), 3),
                          "outcomes": sorted({o.outcome for o in results}), "bad_sessions": bad,
                          "streams_complete": sum(o.complete is not None for o in results),
                          "default_executor_max_latency_s": round(max(default_latency or [0]), 3)}), flush=True)
        check(not bad, f"saturate/{name}: sessions with wrong bytes or hung: {bad}")
        for o in results:
            shutil.rmtree(o.attempt.parents[1], ignore_errors=True)


async def micro(names):
    """_in_stream_io alone: cancellation landing at every phase, incl. the same-iteration race with completion."""
    fx = names["fixed"]
    loop = asyncio.get_running_loop()
    results = []
    for trial in range(400):
        rng = random.Random(trial)
        finished = threading.Event()
        holder = {}
        mode = trial % 4

        def work(holder=holder, finished=finished, mode=mode, rng=rng):
            time.sleep(rng.uniform(0, 0.004))
            if mode == 1:  # cancel queued from the worker just before the result is set
                loop.call_soon_threadsafe(holder["task"].cancel)
            finished.set()
            if mode == 2:
                raise OSError(errno.EIO, "boom")
            return "ok"

        async def caller():
            return await fx._in_stream_io(work)
        task = asyncio.ensure_future(caller())
        holder["task"] = task
        cancels = rng.randint(0, 3) if mode != 1 else 0
        for _ in range(cancels):
            await asyncio.sleep(rng.uniform(0, 0.003))
            task.cancel()
        res = await asyncio.gather(task, return_exceptions=True)
        ok = finished.is_set()
        if task.cancelled():
            kind = "cancelled"
        elif isinstance(res[0], OSError):
            kind = "oserror"
        else:
            kind = res[0]
        results.append(kind)
        check(ok, f"micro {trial}: task finished before its call")
    print("micro outcomes:", {k: results.count(k) for k in set(results)}, flush=True)


def main():
    scratch, mode = Path(sys.argv[1]), sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    runs = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    loop_kind = sys.argv[5] if len(sys.argv) > 5 else "asyncio"
    scratch.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"rv-{mode}-", dir=scratch))
    (root / "tmp").mkdir()
    tempfile.tempdir = str(root / "tmp")
    os.environ["ATP_DEEPSWE_CAMPAIGN"] = str(root)
    reports = REPORTS

    class Reports(logging.Handler):
        def emit(self, record):
            reports.append(record.getMessage())
    logging.getLogger("asyncio").addHandler(Reports(logging.WARNING))
    logging.getLogger("asyncio").propagate = False  # keep tracebacks out of the summary
    names = variants()
    if mode == "fuzz":
        names = {k: names[k] for k in ("upstream", "fixed", "sftv3")}
        coro = fuzz(root, seed, runs, names)
    elif mode == "compare":
        coro = compare(root, seed, runs, names)
    elif mode == "spin":
        coro = spin(root, names)
    elif mode == "saturate":
        coro = saturate(root, names)
    else:
        coro = micro(names)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if loop_kind == "uvloop":
            import uvloop
            with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
                runner.run(coro)
        else:
            asyncio.run(coro)
        gc.collect()
    leaks = [f"{w.category.__name__}: {w.message}" for w in caught
             if issubclass(w.category, (RuntimeWarning, ResourceWarning))]
    print("asyncio reports:", reports[:5], "count", len(reports))
    print("warnings:", leaks[:5], "count", len(leaks))
    shutil.rmtree(root, ignore_errors=True)
    print("RESULT", "FAIL" if FAILURES else "PASS", len(FAILURES), "failure(s)", flush=True)


if __name__ == "__main__":
    main()
