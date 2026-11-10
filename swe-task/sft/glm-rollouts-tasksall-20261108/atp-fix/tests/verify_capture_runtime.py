#!/usr/bin/env python3
"""Reviewer's independent runtime verification (round 1) of the staged ATP capture fix.

Drives the REAL modules through deepswe_capture.run_opencode() with a fake OpenSandbox-like PTY session/sandbox:
  baihe : S/atp/scripts/deepswe_capture.py (sha == Baihe's original; write + fsync inside the event loop)
  sftv3 : Baihe's source with glm-rollouts-sftv3 capture_patch.py's OLD->NEW applied IN MEMORY (asyncio.to_thread)
  fix   : atp-fix/app/scripts/deepswe_capture.py, imported by `import deepswe_capture` with
          sys.path[:2] = [staged scripts dir, private copy of deepswe_http.py + deepswe_upstream.py]
Test-process-only instrumentation:
  * os.fsync: an fsync of a stream copy (<attempt>/stdout.log|stderr.log) sleeps the planned delay, then checks the
    descriptor still names the same inode (else: closed/reused under the write), then really fsyncs. Other fsyncs are
    skipped (keeps Lustre metadata latency out of the measurement; the code under test is the stream copy).
  * pathlib.Path.open: the two stream files are wrapped to record writes/filenos after close and a close() while a
    write or fsync of that file is in progress (close-under-io).
  * a Task subclass records every cancel() of a run_opencode drain task and whether its stream's I/O was in flight.
  * a 5 ms timer measures event-loop lag; asyncio debug mode logs slow callbacks (>= 0.1 s).
Nothing outside the private scratch dir is written. No network, no Slurm.

Run (each group well under 2 min):
  <atp>/runtime/controller/bin/python -I -B -X dev -W error::RuntimeWarning verify_capture_runtime.py <scratch> <group>
groups: lag bytes cancel-run cancel-drain drainwait errors load load-baihe
"""
import sys

sys.dont_write_bytecode = True

import asyncio  # noqa: E402
import collections  # noqa: E402
import concurrent.futures  # noqa: E402
import errno  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import pathlib  # noqa: E402
from pathlib import Path  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from types import SimpleNamespace  # noqa: E402
import warnings  # noqa: E402

S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
ATP_SCRIPTS = S / "atp/scripts"
FIXDIR = S / "sft/glm-rollouts-tasksall-20261108/atp-fix"
STAGED_SCRIPTS = FIXDIR / "app/scripts"
BAIHE_CAPTURE = Path("/lustre/fsw/portfolios/nemotron/projects/nemotron_evals_research/users/baiheh/nano-rl/"
                     "agent-teacher-pipelines/scripts/deepswe_capture.py")
SFTV3_PATCHER = S / "sft/glm-rollouts-sftv3-20261107/capture_patch.py"
SHA = {"fix": "56f34ab96f42dca3a7561ca2ffb5d3d91d185ce32f8a45f89ea4d00ed019d89d",
       "baihe": "b1babe29bfe228fbc3808bdc6bffb6daac475fe798e9dd17720744b186949fe8"}
STREAMS = ("stdout", "stderr")
REAL_FSYNC = os.fsync
REAL_OPEN = pathlib.Path.open
RESULTS = []
LOG_RECORDS = []
UNRAISABLE = []
WARNED = []
THREAD_ERRORS = []
ATTEMPT_BY_SESSION = {}
COUNTER = itertools.count()
CAMPAIGN = None
PROBE = None


def check(ok, message):
    RESULTS.append((bool(ok), message))
    print(("PASS " if ok else "FAIL ") + message, flush=True)


def report(record):
    print("DATA " + json.dumps(record, sort_keys=True, default=str), flush=True)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha12(data):
    return hashlib.sha256(data).hexdigest()[:12]


# ----------------------------------------------------------------------------------------------- instrumentation
class Probe:
    def __init__(self, root):
        self.root = os.path.realpath(root) + "/"
        self.lock = threading.Lock()
        self.delay = lambda path: 0.0
        self.fsyncs = []
        self.io_busy = collections.Counter()
        self.problems = []
        self.closes = []
        self.other_fsync_threads = collections.Counter()
        self.concurrent = 0
        self.max_concurrent = 0

    def is_stream(self, path):
        return path.startswith(self.root) and path.endswith(("/stdout.log", "/stderr.log"))

    def fsync(self, fd):
        try:
            path = os.readlink(f"/proc/self/fd/{fd}")
        except OSError as error:
            with self.lock:
                self.problems.append(("fsync-on-closed-fd", None, repr(error)))
            return REAL_FSYNC(fd)
        if not self.is_stream(path):
            with self.lock:
                self.other_fsync_threads[threading.current_thread().name.rstrip("0123456789_")] += 1
            return None
        before = os.fstat(fd)
        record = {"path": path, "thread": threading.current_thread().name, "t0": time.monotonic(),
                  "size": before.st_size}
        with self.lock:
            self.fsyncs.append(record)
            self.io_busy[path] += 1
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            delay = self.delay(path)
            if isinstance(delay, OSError):
                record["error"] = "injected " + errno.errorcode.get(delay.errno, str(delay.errno))
                raise delay
            if delay:
                time.sleep(delay)
            try:
                after = os.fstat(fd)
            except OSError as error:
                record["error"] = "EBADF after delay"
                with self.lock:
                    self.problems.append(("closed-during-fsync", path, repr(error)))
                raise
            if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
                record["error"] = "descriptor reused"
                with self.lock:
                    self.problems.append(("fd-reused-during-fsync", path, ""))
                raise OSError(errno.EBADF, "descriptor reused during fsync", path)
            return REAL_FSYNC(fd)
        finally:
            record["t1"] = time.monotonic()
            with self.lock:
                self.io_busy[path] -= 1
                self.concurrent -= 1


class TrackedFile:
    """FileIO proxy for a stream copy: records write/fileno after close and close() during in-flight I/O."""

    def __init__(self, raw, path):
        self._raw, self._path = raw, path

    def write(self, data):
        with PROBE.lock:
            if self._raw.closed:
                PROBE.problems.append(("write-after-close", self._path, len(data)))
            PROBE.io_busy[self._path] += 1
        try:
            written = self._raw.write(data)
            if written != len(data):
                with PROBE.lock:
                    PROBE.problems.append(("short-write", self._path, (written, len(data))))
            return written
        finally:
            with PROBE.lock:
                PROBE.io_busy[self._path] -= 1

    def fileno(self):
        if self._raw.closed:
            with PROBE.lock:
                PROBE.problems.append(("fileno-after-close", self._path, ""))
        return self._raw.fileno()

    def close(self):
        with PROBE.lock:
            busy = PROBE.io_busy[self._path]
            PROBE.closes.append({"path": self._path, "t": time.monotonic(), "busy": busy})
            if busy:
                PROBE.problems.append(("close-under-io", self._path, busy))
        self._raw.close()

    @property
    def closed(self):
        return self._raw.closed

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def tracked_open(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
    raw = REAL_OPEN(self, mode, buffering, encoding, errors, newline)
    if mode == "ab" and str(self).endswith(("/stdout.log", "/stderr.log")):
        path = os.readlink(f"/proc/self/fd/{raw.fileno()}")
        if PROBE.is_stream(path):
            return TrackedFile(raw, path)
    return raw


class RecordingTask(asyncio.Task):
    def cancel(self, msg=None):
        coro = self.get_coro()
        if getattr(coro, "__qualname__", "").endswith("run_opencode.<locals>.drain") and coro.cr_frame is not None:
            local = coro.cr_frame.f_locals
            attempt = ATTEMPT_BY_SESSION.get(id(getattr(local.get("reader"), "__self__", None)))
            if attempt is not None:
                stream = str(local.get("name")).removesuffix(".log")
                attempt.drain_cancels.append({"stream": stream, "busy": attempt.busy(stream),
                                              "t": time.monotonic(), "done": self.done()})
        return super().cancel(msg)


# ----------------------------------------------------------------------------------------------- modules under test
def exec_source(name, text, filename):
    module = types.ModuleType(name)
    module.__file__ = filename
    sys.modules[name] = module
    exec(compile(text, filename, "exec"), module.__dict__)
    return module


def load_variants(run_dir):
    deps = run_dir / "deps"
    deps.mkdir()
    for name in ("deepswe_http.py", "deepswe_upstream.py"):
        shutil.copyfile(ATP_SCRIPTS / name, deps / name)

    class SandboxExecResult:  # stub of nemo_gym.sandbox.providers.base.SandboxExecResult
        def __init__(self, stdout=None, stderr=None, return_code=None, error_type=None):
            self.stdout, self.stderr, self.return_code, self.error_type = stdout, stderr, return_code, error_type

    for name in ("nemo_gym", "nemo_gym.sandbox", "nemo_gym.sandbox.providers", "nemo_gym.sandbox.providers.base"):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    sys.modules["nemo_gym.sandbox.providers.base"].SandboxExecResult = SandboxExecResult
    sys.path[:0] = [str(STAGED_SCRIPTS), str(deps)]
    import deepswe_capture as fix  # the staged module, by normal import

    fix_file = Path(fix.__file__).resolve()
    check(fix_file == (STAGED_SCRIPTS / "deepswe_capture.py").resolve() and sha256_file(fix_file) == SHA["fix"],
          f"fix imported from the staged scripts dir ({fix_file}), sha256 {SHA['fix'][:12]}")
    import deepswe_http
    check(Path(deepswe_http.__file__).resolve().parent == deps.resolve(), "deepswe_http from the private deps copy")
    live = ATP_SCRIPTS / "deepswe_capture.py"
    check(sha256_file(live) == sha256_file(BAIHE_CAPTURE) == SHA["baihe"],
          "S/atp/scripts/deepswe_capture.py == Baihe's original (pre-install state)")
    baihe = exec_source("capture_baihe", live.read_text(), str(live))
    namespace = {"__name__": "sftv3_capture_patch_constants"}
    exec(compile(SFTV3_PATCHER.read_text(), str(SFTV3_PATCHER), "exec"), namespace)  # defines OLD/NEW only
    text = live.read_text()
    check(text.count(namespace["OLD"]) == 1, "sftv3 OLD block found once in Baihe's source (in-memory patch)")
    sftv3 = exec_source("capture_sftv3", text.replace(namespace["OLD"], namespace["NEW"], 1),
                        "<sftv3-patched deepswe_capture.py>")
    return {"baihe": baihe, "sftv3": sftv3, "fix": fix}


# ----------------------------------------------------------------------------------------------- fakes
class FakeSession:
    """Like OpenSandboxPtySession: a pump task fills unbounded per-stream queues; read()/read_stderr() return one
    chunk, b'' at EOF (sticky). Records what was handed to the caller and when."""

    def __init__(self, plans, tails, label):
        self.session_id = "verify-" + label
        self.queues = {s: asyncio.Queue() for s in STREAMS}
        self.emitted = {s: bytearray() for s in STREAMS}
        self.handed = {s: [] for s in STREAMS}
        self.handed_at = {s: [] for s in STREAMS}
        self.kill = asyncio.Event()
        self.exited = asyncio.get_running_loop().create_future()
        self.close_calls = []
        self.pumps = [asyncio.create_task(self._pump(s, plans.get(s, ()), tails.get(s, ()))) for s in STREAMS]
        self.watch = asyncio.create_task(self._exit_when_done())

    async def _pump(self, stream, plan, tail):
        for delay, data in plan:
            await asyncio.sleep(delay)
            if self.kill.is_set():
                break
            self.emitted[stream] += data
            self.queues[stream].put_nowait(data)
        if self.kill.is_set():
            for delay, data in tail:  # output the command writes while it is being stopped
                await asyncio.sleep(delay)
                self.emitted[stream] += data
                self.queues[stream].put_nowait(data)
        self.queues[stream].put_nowait(None)

    async def _exit_when_done(self):
        await asyncio.gather(*self.pumps, return_exceptions=True)
        if not self.exited.done():
            self.exited.set_result(143 if self.kill.is_set() else 0)

    async def _read(self, stream):
        data = await self.queues[stream].get()
        if data is None:
            self.queues[stream].put_nowait(None)
            return b""
        self.handed[stream].append(data)
        self.handed_at[stream].append(time.monotonic())
        return data

    async def read(self, *, timeout_s=None):
        return await self._read("stdout")

    async def read_stderr(self, *, timeout_s=None):
        return await self._read("stderr")

    async def wait_exit(self, *, timeout_s=None):
        return await asyncio.shield(self.exited)

    async def close(self, *, preserve_session=False):
        self.close_calls.append(preserve_session)
        for task in self.pumps + [self.watch]:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.pumps, self.watch, return_exceptions=True)


class FakeSandbox:
    def __init__(self, attempt, plans, tails, label, control_ok):
        self.attempt, self.plans, self.tails, self.label, self.control_ok = attempt, plans, tails, label, control_ok
        self.session = None
        self.pty = SimpleNamespace(create=self._create)
        self.commands = collections.Counter()

    async def _create(self, command, pty):
        assert pty is False and command.startswith("bash -c ")
        self.session = FakeSession(self.plans, self.tails, self.label)
        ATTEMPT_BY_SESSION[id(self.session)] = self.attempt
        return self.session

    async def exec(self, command, timeout_s=None, **kwargs):
        await asyncio.sleep(0.002)
        if "control_channel" in command:
            self.commands["control"] += 1
            if not self.control_ok:
                return SimpleNamespace(return_code=1, stdout="", stderr="simulated: process control failed")
            self.session.kill.set()
            proof = {"confirmed": True, "control_channel": "sandbox_exec", "process": {"pid": 2, "pgid": 2},
                     "signals": ["SIGTERM"], "wrapper_forced_stop": False,
                     "wrapper": {"confirmed": True, "process": {"pid": 1, "pgid": 1}, "signals": []}}
            return SimpleNamespace(return_code=0, stdout=json.dumps(proof), stderr="")
        if command.startswith("test -f") and "stream-writer-error.json" in command:
            self.commands["absent"] += 1
            return SimpleNamespace(return_code=1, stdout="", stderr="")
        self.commands["other"] += 1
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def download(self, remote, local):
        await asyncio.sleep(0.002)
        name = remote.rsplit("/", 1)[-1]
        if name in ("stdout.log", "stderr.log"):
            data = bytes(self.session.emitted[name.removesuffix(".log")])
        elif name == "stream-writers-complete.json":
            data = json.dumps({"stdout_exit_code": 0, "stderr_exit_code": 0, "recorded_at": time.time()}).encode()
        else:
            data = b"verify-archive"
        Path(local).write_bytes(data)


class Attempt:
    def __init__(self, module, variant, plans, tails=None, timeout_s=300.0, control_ok=True):
        index = next(COUNTER)
        self.variant = variant
        self.directory = CAMPAIGN / "models" / "verify" / "attempts" / f"{index}-0" / "attempt-0"
        self.directory.mkdir(parents=True)
        real = os.path.realpath(self.directory)
        self.paths = {s: f"{real}/{s}.log" for s in STREAMS}
        self.drain_cancels = []
        self.cancels = []
        self.busy_at_return = None
        self.done_at = None
        self.sandbox = FakeSandbox(self, plans, tails or {}, f"{variant}-{index}", control_ok)
        body = {"_atp_attempt_dir": str(self.directory), "_atp_deadline": time.time() + 3600}

        async def request_json():
            return body

        self.started = time.monotonic()
        self.task = asyncio.create_task(module.run_opencode(self.sandbox, "opencode run verify",
                                                            SimpleNamespace(json=request_json), timeout_s))
        self.task.add_done_callback(self._on_done)

    def _on_done(self, task):
        with PROBE.lock:
            self.busy_at_return = {s: PROBE.io_busy[p] for s, p in self.paths.items()}
        self.done_at = time.monotonic()

    def busy(self, stream="stdout"):
        with PROBE.lock:
            return PROBE.io_busy[self.paths[stream]]

    def fsync_count(self, stream="stdout"):
        with PROBE.lock:
            return sum(1 for record in PROBE.fsyncs if record["path"] == self.paths[stream])

    def drain_tasks(self):
        found = {}
        for task in asyncio.all_tasks():
            coro = task.get_coro()
            if getattr(coro, "__qualname__", "").endswith("run_opencode.<locals>.drain") and coro.cr_frame:
                local = coro.cr_frame.f_locals
                if getattr(local.get("reader"), "__self__", None) is self.sandbox.session:
                    found[str(local.get("name")).removesuffix(".log")] = task
        return found

    def cancel(self, task, label):
        self.cancels.append({"target": label, "busy": {s: self.busy(s) for s in STREAMS}, "t": time.monotonic()})
        task.cancel()

    async def wait(self):
        await asyncio.wait([self.task])
        for _ in range(2000):  # abandoned (sftv3) writes: let them end before the files are inspected
            if not any(self.busy(s) for s in STREAMS):
                break
            await asyncio.sleep(0.01)
        if self.task.cancelled():
            self.outcome = "CancelledError"
        elif self.task.exception() is not None:
            error = self.task.exception()
            self.outcome = type(error).__name__ + (f" errno={error.errno}" if getattr(error, "errno", None) else "")
        else:
            self.outcome = f"returned {self.task.result().return_code}"
        self.files = {}
        for s in STREAMS:
            path = self.directory / f"{s}.log"
            self.files[s] = path.read_bytes() if path.exists() else None
        events = self.directory / "capture-events.jsonl"
        self.events = [json.loads(line)["event"] for line in events.read_text().splitlines()] if events.exists() else []
        marker = self.directory / "streams-complete.json"
        self.streams_complete = json.loads(marker.read_text()) if marker.exists() else None
        with PROBE.lock:
            self.fsyncs = [dict(record) for record in PROBE.fsyncs if record["path"] in self.paths.values()]
            self.problems = [problem for problem in PROBE.problems if problem[1] in self.paths.values()]
            self.closes = [close for close in PROBE.closes if close["path"] in self.paths.values()]
        session = self.sandbox.session
        self.handed = {s: b"".join(session.handed[s]) for s in STREAMS}
        self.emitted = {s: bytes(session.emitted[s]) for s in STREAMS}
        return self

    def order_problems(self, complete):
        """Per stream: fsyncs never overlap; file size at the k-th fsync == bytes of the first k handed chunks (one
        write per chunk, before its fsync, in order); chunk k+1 is handed out only after chunk k's fsync ended."""
        issues = []
        for s in STREAMS:
            records = sorted((r for r in self.fsyncs if r["path"] == self.paths[s]), key=lambda r: r["t0"])
            if any(a["t1"] > b["t0"] for a, b in zip(records, records[1:])):
                issues.append(f"{s}: overlapping fsyncs")
            chunks = self.sandbox.session.handed[s]
            cumulative = list(itertools.accumulate(len(c) for c in chunks))
            sizes = [r["size"] for r in records]
            if sizes != cumulative[:len(sizes)]:
                issues.append(f"{s}: fsync sizes {sizes[:5]}... != cumulative {cumulative[:5]}...")
            if complete and len(sizes) != len(cumulative):
                issues.append(f"{s}: {len(sizes)} fsyncs for {len(cumulative)} handed chunks")
            handed_at = self.sandbox.session.handed_at[s]
            for k in range(1, len(handed_at)):
                if k - 1 < len(records) and handed_at[k] < records[k - 1]["t1"]:
                    issues.append(f"{s}: chunk {k} handed before chunk {k - 1}'s fsync ended")
                    break
        return issues

    def summary(self):
        threads = sorted({r["thread"].rstrip("0123456789").rstrip("_") for r in self.fsyncs})
        return {"variant": self.variant, "outcome": self.outcome,
                "bytes": {s: len(self.files[s] or b"") for s in STREAMS},
                "handed_chunks": {s: len(self.sandbox.session.handed[s]) for s in STREAMS},
                "files_eq_handed": all(self.files[s] == self.handed[s] for s in STREAMS),
                "files_eq_emitted": all(self.files[s] == self.emitted[s] for s in STREAMS),
                "stream_fsyncs": len(self.fsyncs), "fsync_threads": threads,
                "problems": collections.Counter(p[0] for p in self.problems),
                "busy_at_return": self.busy_at_return, "events": self.events,
                "streams_complete": None if self.streams_complete is None else
                ("after_interruption" if self.streams_complete.get("after_interruption") else "normal")}


# ----------------------------------------------------------------------------------------------- measurement helpers
async def monitor(stop, samples, period=0.005):
    while not stop.is_set():
        start = time.monotonic()
        await asyncio.sleep(period)
        now = time.monotonic()
        samples.append((now, now - start - period))


async def executor_probe(stop, latencies, period=0.05):
    loop = asyncio.get_running_loop()
    while not stop.is_set():
        start = time.monotonic()
        await loop.run_in_executor(None, time.monotonic)
        latencies.append(time.monotonic() - start)
        await asyncio.sleep(period)


def lag_stats(samples, fsyncs, start, end):
    window = [late for at, late in samples if start <= at <= end + 0.05]
    # lag of timer wakeups that fall inside (or within 50 ms after) a stream fsync: did that fsync stop the loop?
    spans = sorted((r["t0"], r["t1"]) for r in fsyncs)
    during = [late for at, late in samples if any(t0 < at < t1 for t0, t1 in spans)]
    worst = max(((late, at) for at, late in samples if any(t0 < at < t1 for t0, t1 in spans)), default=(0.0, start))
    ticks = [sum(1 for at, _ in samples if r["t0"] + 0.02 < at < r["t1"] - 0.01)
             for r in fsyncs if r["t1"] - r["t0"] >= 0.2]
    return {"max_lag_s": round(max(window, default=0.0), 4),
            "max_lag_during_fsync_s": round(max(during, default=0.0), 4),
            "worst_during_fsync_at_s": round(worst[1] - start, 3), "_worst": worst,
            "p99_lag_s": round(sorted(window)[int(0.99 * (len(window) - 1))], 4) if window else None,
            "min_ticks_per_stalled_fsync": min(ticks, default=None),
            "median_ticks_per_stalled_fsync": statistics.median(ticks) if ticks else None,
            "stalled_fsyncs": len(ticks)}


SLOW = re.compile(r"took ([0-9.]+) seconds")


def slow_callbacks(start, end):
    found = [(float(SLOW.search(message).group(1)), "drain()" in message)
             for at, level, message in LOG_RECORDS
             if level == logging.WARNING and start <= at <= end + 0.5 and SLOW.search(message)]
    others = [(round(at - start, 3), message[:260]) for at, level, message in LOG_RECORDS
              if level == logging.WARNING and start <= at <= end + 0.5 and SLOW.search(message)
              and "drain()" not in message]
    return {"other_slow_detail": others[:4], "drain_slow_callbacks": sum(1 for _, drain in found if drain),
            "max_drain_slow_s": max((s for s, drain in found if drain), default=0.0),
            "other_slow_callbacks": sum(1 for _, drain in found if not drain),
            "max_other_slow_s": max((s for s, drain in found if not drain), default=0.0)}


CORO = re.compile(r"coro=<([A-Za-z_.<>]+)\(\)")


def loop_time_by_coroutine(start, end, worst_at=None, worst_lag=0.0):
    """Debug-mode 'Executing <handle> took X seconds' records (threshold set by the group) per coroutine name;
    plus the records that ended inside the worst lag sample's sleep interval."""
    totals, counts, maxima, around = collections.Counter(), collections.Counter(), {}, []
    for at, level, message in LOG_RECORDS:
        found = SLOW.search(message)
        if level != logging.WARNING or not found or not start <= at <= end + 0.5:
            continue
        name = (CORO.search(message).group(1) if CORO.search(message) else message[:40]).split(".")[-1]
        seconds = float(found.group(1))
        totals[name] += seconds
        counts[name] += 1
        maxima[name] = max(maxima.get(name, 0.0), seconds)
        if worst_at is not None and worst_at - worst_lag - 0.01 <= at <= worst_at + 0.001:
            around.append((name, seconds))
    return {"busy_s_by_coroutine": {k: round(v, 3) for k, v in totals.most_common()},
            "callbacks_by_coroutine": dict(counts), "max_s_by_coroutine": {k: round(v, 3) for k, v in maxima.items()},
            "in_worst_lag_window": collections.Counter(n for n, _ in around),
            "in_worst_lag_window_s": round(sum(s for _, s in around), 3)}


async def when(predicate, timeout, poll=0.003):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        await asyncio.sleep(poll)
    return predicate()


def rand_chunk(rng, low, high, tag=b""):
    return tag + rng.randbytes(rng.randint(low, high))


# ----------------------------------------------------------------------------------------------- groups
async def group_lag(V):
    for delay in (0.3, 1.0):
        rng = random.Random(int(delay * 1000))
        plans = {"stdout": [(0.05, rand_chunk(rng, 1500, 6000, b"o%d" % i)) for i in range(4)],
                 "stderr": [(0.08, rand_chunk(rng, 100, 900, b"e%d" % i)) for i in range(3)]}
        files = {}
        for variant in ("baihe", "sftv3", "fix"):
            PROBE.delay = lambda path, d=delay: d
            samples, stop = [], asyncio.Event()
            watcher = asyncio.create_task(monitor(stop, samples))
            start = time.monotonic()
            run = await Attempt(V[variant], variant, plans).wait()
            stop.set()
            await watcher
            stats = lag_stats(samples, run.fsyncs, start, run.done_at)
            stats.pop("_worst")
            slow = slow_callbacks(start, run.done_at)
            report({"group": "lag", "fsync_delay_s": delay, **run.summary(), **stats, **slow,
                    "wall_s": round(run.done_at - start, 2)})
            files[variant] = run.files
            ok = (run.outcome == "returned 0" and run.streams_complete is not None
                  and all(run.files[s] == run.emitted[s] == run.handed[s] for s in STREAMS))
            check(ok and not run.order_problems(True) and not run.problems,
                  f"lag {delay}s {variant}: rc 0, streams-complete, files == emitted == handed, per-stream order/"
                  f"durability ({run.order_problems(True) or 'ok'})")
            if variant == "baihe":
                check(stats["max_lag_s"] >= 0.9 * delay and stats["min_ticks_per_stalled_fsync"] == 0
                      and {r["thread"] for r in run.fsyncs} == {"MainThread"},
                      f"SENSITIVITY lag {delay}s baihe: fsync on the loop thread blocks it (max lag "
                      f"{stats['max_lag_s']} s, 0 ticks during each fsync)")
            else:
                prefix = "atp-stream-io_" if variant == "fix" else "asyncio_"
                check(stats["max_lag_during_fsync_s"] < 0.1 and (stats["min_ticks_per_stalled_fsync"] or 0) >= 10
                      and all(r["thread"].startswith(prefix) for r in run.fsyncs),
                      f"lag {delay}s {variant}: fsyncs in {prefix}* threads, loop responsive (max lag during fsyncs "
                      f"{stats['max_lag_during_fsync_s']} s, overall {stats['max_lag_s']} s, min "
                      f"{stats['min_ticks_per_stalled_fsync']} ticks per fsync)")
            if variant == "fix":
                check(slow["drain_slow_callbacks"] == 0, f"lag {delay}s fix: no drain step >= 0.1 s in asyncio debug "
                                                         f"(other slow callbacks: {slow['other_slow_callbacks']})")
        check(files["fix"] == files["baihe"] == files["sftv3"], f"lag {delay}s: identical stream files in all variants")


async def group_bytes(V):
    rng = random.Random(20261108)

    def size():
        r = rng.random()
        if r < 0.3:
            return rng.randint(1, 64)
        if r < 0.8:
            return rng.randint(65, 8192)
        if r < 0.99:
            return rng.randint(8193, 65536)
        return rng.randint(262144, 1048576)

    def delay():
        return 0.0 if rng.random() < 0.7 else rng.random() * 0.003

    plans = [{"stdout": [(delay(), rng.randbytes(size())) for _ in range(250)],
              "stderr": [(delay(), rng.randbytes(size())) for _ in range(150)]} for _ in range(3)]
    expected = [{s: b"".join(c for _, c in plan[s]) for s in STREAMS} for plan in plans]
    print(f"bytes: 3 concurrent sessions x (250 stdout + 150 stderr chunks), "
          f"{sum(len(b) for e in expected for b in e.values())} bytes per variant", flush=True)
    PROBE.delay = lambda path: random.random() * 0.002
    digests = {}
    for variant in ("baihe", "sftv3", "fix"):
        start = time.monotonic()
        runs = [Attempt(V[variant], variant, plans[i]) for i in range(3)]
        await asyncio.gather(*(run.wait() for run in runs))
        wall = time.monotonic() - start
        ok = all(run.outcome == "returned 0" and run.streams_complete is not None
                 and run.streams_complete.get("exit_code") == 0 and "after_interruption" not in run.streams_complete
                 and all(run.files[s] == expected[i][s] == run.handed[s] == run.emitted[s] for s in STREAMS)
                 for i, run in enumerate(runs))
        order = [issue for run in runs for issue in run.order_problems(True)]
        problems = [p for run in runs for p in run.problems]
        digests[variant] = [{s: sha12(run.files[s]) for s in STREAMS} for run in runs]
        report({"group": "bytes", "variant": variant, "wall_s": round(wall, 2),
                "fsyncs": sum(len(run.fsyncs) for run in runs), "digests": digests[variant],
                "fsync_threads": sorted({r["thread"].rstrip("0123456789").rstrip("_") for run in runs
                                         for r in run.fsyncs})})
        check(ok and not order and not problems,
              f"bytes {variant}: 3 sessions rc 0, stdout.log/stderr.log byte-exact (== planned == handed), one "
              f"write+fsync per chunk in order, next chunk read only after the previous fsync ({order[:2] or 'ok'})")
        for run in runs:
            shutil.rmtree(run.directory.parent)  # attempts/<i>-0: free the ~17 MB per variant (private scratch)
    check(digests["fix"] == digests["baihe"] == digests["sftv3"], "bytes: identical files across variants")


SCENARIOS = ("run-cancel-once", "run-cancel-twice", "deadline", "deadline-control-fails",
             "drain-cancel-once", "drain-cancel-twice")


async def scenario(V, variant, kind, delay=0.6):
    plans = {"stdout": [(0.3, b"[o%03d]" % i + random.Random(i).randbytes(700 + 37 * i)) for i in range(60)],
             "stderr": [(0.7, b"[e%03d]" % i + random.Random(1000 + i).randbytes(90 + 11 * i)) for i in range(30)]}
    tails = {"stdout": [(0.03, b"tail-o1\n"), (0.03, b"tail-o2 \x00\xff\n")], "stderr": [(0.03, b"tail-e1\n")]}
    deadline = kind.startswith("deadline")
    PROBE.delay = lambda path: delay
    run = Attempt(V[variant], variant, plans, tails, timeout_s=1.25 if deadline else 300.0,
                  control_ok=kind != "deadline-control-fails")
    run.kind = kind

    def writing(stream="stdout"):  # baihe: the loop only runs between writes, never during one
        return run.busy(stream) > 0 or variant == "baihe"

    if kind.startswith("run-cancel"):
        await when(lambda: run.fsync_count() >= 2 and writing(), 5)
        run.cancel(run.task, "run_opencode#1")
        if kind == "run-cancel-twice":
            await when(lambda: (run.directory / "generation-quiescence.json").exists() and writing(), 8)
            run.cancel(run.task, "run_opencode#2")
    elif kind.startswith("drain-cancel"):
        await when(lambda: run.fsync_count() >= 2 and writing(), 5)
        drain = run.drain_tasks()["stdout"]
        with PROBE.lock:
            inflight = [r for r in PROBE.fsyncs if r["path"] == run.paths["stdout"] and "t1" not in r]
        run.cancel(drain, "drain-stdout#1")
        if kind == "drain-cancel-twice":
            await asyncio.sleep(0.15)
            run.cancel(drain, "drain-stdout#2")
        await asyncio.wait([drain])
        run.drain_done_at = time.monotonic()
        run.inflight_at_cancel = inflight
    await run.wait()
    return run


async def group_cancel(V, kinds):
    for kind in kinds:
        runs = {}
        for variant in ("baihe", "sftv3", "fix"):
            run = runs[variant] = await scenario(V, variant, kind)
            summary = run.summary()
            summary.update(group="cancel", scenario=kind, cancels=[{"target": c["target"], "busy": c["busy"]}
                                                                   for c in run.cancels],
                           drain_cancels=len(run.drain_cancels),
                           drain_cancels_during_write=sum(1 for c in run.drain_cancels if c["busy"] and not c["done"]))
            report(summary)
        fix, baihe, sftv3 = runs["fix"], runs["baihe"], runs["sftv3"]
        expect_outcome = {"run-cancel-once": "CancelledError", "run-cancel-twice": "CancelledError",
                          "deadline": "TimeoutError", "deadline-control-fails": "TimeoutError",
                          "drain-cancel-once": "CancelledError", "drain-cancel-twice": "CancelledError"}[kind]
        if kind == "run-cancel-twice":
            # Baihe's drains clear their whole backlog in one blocking step (Queue.get does not suspend when items
            # are queued), so the 2nd cancel cannot land in Baihe's drain wait; it lands later (finally's shield).
            # Compare the outcome only, and check the fix's sequence for a cancel inside the drain wait: no
            # streams-complete, final snapshot still taken, session closed (quiesced).
            check(fix.outcome == baihe.outcome == expect_outcome
                  and fix.events == ["agent_interrupted", "agent_control_recovered", "agent_snapshot_saved"]
                  and fix.sandbox.session.close_calls == [False],
                  f"cancel {kind}: fix {fix.outcome} (Baihe {baihe.outcome}); fix events {fix.events}, session "
                  f"close calls {fix.sandbox.session.close_calls} (Baihe's 2nd cancel hit a later phase: events "
                  f"{baihe.events})")
        else:
            check(fix.outcome == baihe.outcome == expect_outcome and fix.events == baihe.events,
                  f"cancel {kind}: fix ends like Baihe's code ({fix.outcome}; events {fix.events})")
        fsync_errors = {name: sum(1 for r in run.fsyncs if "error" in r) for name, run in runs.items()}
        order = fix.order_problems(True)
        check(not order and fsync_errors["fix"] == 0,
              f"cancel {kind}: fix - each handed chunk has exactly one successful fsync, in order, before the next "
              f"read ({order[:2] or 'ok'}; fsync errors per variant {fsync_errors})")
        check(all(fix.files[s] == fix.handed[s] for s in STREAMS) and not fix.problems
              and fix.busy_at_return == {"stdout": 0, "stderr": 0},
              f"cancel {kind}: fix - every handed chunk is in its file, no write/fileno after close, no close under "
              f"a write/fsync, no fsync on a closed/reused fd, no write still running when run_opencode ended "
              f"(problems {dict(collections.Counter(p[0] for p in fix.problems))}, busy at return {fix.busy_at_return})")
        check(not baihe.problems and all(baihe.files[s] == baihe.handed[s] for s in STREAMS),
              f"cancel {kind}: baihe baseline clean")
        if kind in ("run-cancel-once", "deadline"):
            check(fix.streams_complete is not None and fix.streams_complete.get("after_interruption")
                  and all(fix.files[s] == fix.emitted[s] for s in STREAMS),
                  f"cancel {kind}: fix drains reach EOF after the stop; streams-complete.json after_interruption; "
                  f"files == all emitted output incl. the tail")
        if kind == "run-cancel-once":
            check(fix.cancels[0]["busy"]["stdout"] > 0, "run-cancel-once: fix cancel landed during a stdout write")
        if kind == "run-cancel-twice":
            check(fix.cancels[1]["busy"]["stdout"] > 0 and fix.streams_complete is None
                  and any(c["busy"] and not c["done"] for c in fix.drain_cancels),
                  f"run-cancel-twice: fix 2nd cancel landed in the post-stop drain wait during a write; drains "
                  f"cancelled mid-write ({len(fix.drain_cancels)} drain cancels)")
        if kind == "deadline-control-fails":
            check(fix.streams_complete is None and "generation_not_quiesced" in fix.events
                  and any(c["busy"] and not c["done"] for c in fix.drain_cancels),
                  f"deadline-control-fails: fix finally-path cancelled drains during a write "
                  f"({sum(1 for c in fix.drain_cancels if c['busy'] and not c['done'])} of {len(fix.drain_cancels)})")
        if kind.startswith("drain-cancel"):
            ended = [r["t1"] for r in fix.fsyncs if any(r["t0"] == i["t0"] for i in fix.inflight_at_cancel)]
            closes = [c["t"] for c in fix.closes if c["path"] == fix.paths["stdout"]]
            check(fix.cancels[0]["busy"]["stdout"] > 0 and fix.inflight_at_cancel and ended
                  and fix.drain_done_at >= max(ended) and closes and min(closes) >= max(ended),
                  f"{kind}: fix drain cancelled during a write ended only after that fsync finished "
                  f"(+{(fix.drain_done_at - max(ended or [0])):.3f} s), file closed after it")
            if kind == "drain-cancel-twice":
                check(fix.cancels[1]["busy"]["stdout"] > 0, "drain-cancel-twice: fix 2nd cancel also mid-write")
        check(all(not c["busy"]["stdout"] for c in baihe.cancels),
              f"cancel {kind}: baihe never cancelled during a write (loop blocked during each)")
        if kind in ("run-cancel-twice", "deadline-control-fails", "drain-cancel-once", "drain-cancel-twice"):
            report({"group": "cancel", "scenario": kind, "sftv3_problems":
                    dict(collections.Counter(p[0] for p in sftv3.problems)),
                    "sftv3_busy_at_return": sftv3.busy_at_return})
            check(any(p[0] == "close-under-io" for p in sftv3.problems),
                  f"SENSITIVITY cancel {kind}: sftv3 to_thread patch closes the file under an in-flight write "
                  f"({dict(collections.Counter(p[0] for p in sftv3.problems))})")


async def group_drainwait(V):
    """Deadline at 1.25 s, process control OK, then the 15 s post-stop drain wait (STREAM_DRAIN_TIMEOUT_S) expires
    with a stdout backlog of 2 s fsyncs. Fix: the expiry cancels the drains mid-write; TimeoutError of that wait only
    after the in-flight write ended; agent_streams_incomplete; every handed chunk fsynced; nothing closed under a
    write. Baihe's code cannot be interrupted while its drain clears the backlog (loop blocked throughout)."""
    plans = {"stdout": [(0.1, b"[o%03d]" % i + random.Random(i).randbytes(900)) for i in range(400)],
             "stderr": [(0.5, b"[e%03d]" % i + random.Random(500 + i).randbytes(90)) for i in range(80)]}
    tails = {"stdout": [(0.03, b"tail-o1\n")], "stderr": [(0.03, b"tail-e1\n")]}
    PROBE.delay = lambda path: 2.0 if path.endswith("/stdout.log") else 0.05
    runs = {}
    for variant in ("fix", "sftv3", "baihe"):
        samples, stop = [], asyncio.Event()
        watcher = asyncio.create_task(monitor(stop, samples))
        start = time.monotonic()
        run = runs[variant] = await Attempt(V[variant], variant, plans, tails, timeout_s=1.25).wait()
        stop.set()
        await watcher
        quiescence = run.directory / "generation-quiescence.json"
        report({"group": "drainwait", **run.summary(), "wall_s": round(run.done_at - start, 2),
                "max_lag_s": round(max((late for _, late in samples), default=0.0), 3),
                "drain_cancels": len(run.drain_cancels),
                "drain_cancels_during_write": sum(1 for c in run.drain_cancels if c["busy"] and not c["done"]),
                "fsync_errors": sum(1 for r in run.fsyncs if "error" in r),
                "quiescence": json.loads(quiescence.read_text()) if quiescence.exists() else None})
    fix = runs["fix"]
    check(fix.outcome == "TimeoutError" and "agent_streams_incomplete" in fix.events and fix.streams_complete is None
          and any(c["busy"] and not c["done"] for c in fix.drain_cancels),
          f"drainwait fix: 15 s drain wait expired mid-write -> agent_streams_incomplete, deadline TimeoutError "
          f"re-raised ({fix.outcome}; events {fix.events})")
    check(all(fix.files[s] == fix.handed[s] for s in STREAMS) and not fix.problems and not fix.order_problems(True)
          and not any("error" in r for r in fix.fsyncs) and fix.busy_at_return == {"stdout": 0, "stderr": 0},
          "drainwait fix: every handed chunk written + fsynced once, in order; no close under a write; nothing in "
          "flight at return")
    check(15.0 <= fix.done_at - fix.started <= 1.25 + 15 + 2.0 + 3.0,
          f"drainwait fix: run_opencode ended {fix.done_at - fix.started:.1f} s after start (deadline 1.25 s + 15 s "
          f"wait + <= one 2 s write + teardown)")


async def group_errors(V):
    """The 3rd stdout fsync fails (ENOSPC / EIO), after a 0.3 s stall; not cancelled. Fix must end exactly like
    Baihe's code: same exception type/errno, same events, same file contents, error not swallowed or logged."""
    for code in (errno.ENOSPC, errno.EIO):
        runs = {}
        for variant in ("baihe", "fix"):
            count = [0]

            def plan(path, count=count, code=code):
                if not path.endswith("/stdout.log"):
                    return 0.05
                count[0] += 1
                return OSError(code, os.strerror(code) + " (injected)", path) if count[0] == 3 else 0.3

            PROBE.delay = plan
            plans = {"stdout": [(0.05, b"[o%d]" % i + random.Random(i).randbytes(3000)) for i in range(40)],
                     "stderr": [(0.2, b"[e%d]" % i + random.Random(99 + i).randbytes(200)) for i in range(10)]}
            run = runs[variant] = await Attempt(V[variant], variant, plans).wait()
            report({"group": "errors", "errno": errno.errorcode[code], **run.summary()})
        fix, baihe = runs["fix"], runs["baihe"]
        name = errno.errorcode[code]
        check(fix.outcome == baihe.outcome == f"OSError errno={code}" and fix.events == baihe.events
              and "agent_streams_incomplete" in fix.events and fix.streams_complete is None,
              f"errors {name}: fix ends as Baihe's code ({fix.outcome}; events {fix.events})")
        check(fix.files["stdout"] == baihe.files["stdout"] == fix.handed["stdout"][:len(fix.files["stdout"])]
              and len(fix.sandbox.session.handed["stdout"]) == 3 and not fix.problems
              and fix.busy_at_return == {"stdout": 0, "stderr": 0},
              f"errors {name}: stdout.log identical to Baihe's (3 chunks written, failed one included), no I/O "
              f"problems, nothing in flight at return")


async def group_load(V, configs):
    loop = asyncio.get_running_loop()
    rng = random.Random(4848)
    n = 24
    plans = [{"stdout": [(0.02, rand_chunk(rng, 200, 16000, b"o%d." % k)) for k in range(6)],
              "stderr": [(0.02, rand_chunk(rng, 50, 2000, b"e%d." % k)) for k in range(6)]} for _ in range(n)]
    if configs == "baihe":
        plans = [{s: plan[s][:1] for s in STREAMS} for plan in plans]
        configs = [("baihe", 32)]
    PROBE.delay = lambda path: 0.3
    loop.slow_callback_duration = 0.005  # attribute loop busy time per coroutine (>= 5 ms callbacks)
    for variant, threads in configs:
        default = concurrent.futures.ThreadPoolExecutor(threads, thread_name_prefix=f"default{threads}x")
        loop.set_default_executor(default)
        with PROBE.lock:
            PROBE.max_concurrent = 0
        samples, latencies, stop = [], [], asyncio.Event()
        watchers = [asyncio.create_task(monitor(stop, samples)), asyncio.create_task(executor_probe(stop, latencies))]
        start = time.monotonic()
        runs = [Attempt(V[variant], variant, plans[i]) for i in range(n)]
        await asyncio.gather(*(run.wait() for run in runs))
        end = time.monotonic()
        stop.set()
        await asyncio.gather(*watchers)
        default.shutdown(wait=True)
        fsyncs = [r for run in runs for r in run.fsyncs]
        stats = lag_stats(samples, fsyncs, start, end)
        worst_lag, worst_at = stats.pop("_worst")
        slow = slow_callbacks(start, end)
        slow.pop("other_slow_detail", None)
        attribution = loop_time_by_coroutine(start, end, worst_at, worst_lag)
        exact = all(run.outcome == "returned 0" and run.streams_complete is not None
                    and all(run.files[s] == run.emitted[s] == run.handed[s] for s in STREAMS) for run in runs)
        order = [issue for run in runs for issue in run.order_problems(True)]
        problems = [p for run in runs for p in run.problems]
        report({"group": "load", "variant": variant, "default_executor_threads": threads, "sessions": n,
                "drains": 2 * n, "stream_fsyncs": len(fsyncs), "wall_s": round(end - start, 2),
                "max_concurrent_stream_fsyncs": PROBE.max_concurrent,
                "fsync_threads": sorted({r["thread"].rstrip("0123456789").rstrip("_x") for r in fsyncs}),
                "default_executor_latency_max_s": round(max(latencies, default=0.0), 3),
                "default_executor_latency_p50_s": round(statistics.median(latencies), 4) if latencies else None,
                "default_executor_probes": len(latencies), **stats, **slow, **attribution})
        check(exact and not order and not problems,
              f"load {variant} (default executor {threads}): {n} sessions / {2 * n} drains, all rc 0, every byte "
              f"landed in order ({order[:2] or 'ok'})")
        if variant == "fix":
            drain_max = attribution["max_s_by_coroutine"].get("drain", 0.0)
            check(stats["median_ticks_per_stalled_fsync"] >= 30 and (stats["min_ticks_per_stalled_fsync"] or 0) >= 1
                  and stats["max_lag_during_fsync_s"] < 0.3 and drain_max < 0.02
                  and PROBE.max_concurrent == 32,
                  f"load fix: loop responsive with {2 * n} drains (max lag during fsyncs "
                  f"{stats['max_lag_during_fsync_s']} s, overall {stats['max_lag_s']} s, min "
                  f"{stats['min_ticks_per_stalled_fsync']} / median {stats['median_ticks_per_stalled_fsync']} ticks per "
                  f"fsync; longest drain step {drain_max} s; loop busy by coroutine "
                  f"{attribution['busy_s_by_coroutine']}), _STREAM_IO caps at 32 concurrent fsyncs "
                  f"({PROBE.max_concurrent})")
            check(max(latencies, default=1.0) < 0.1,
                  f"load fix: default executor ({threads} threads) stays free (max latency "
                  f"{max(latencies, default=0.0):.3f} s)")
        if variant == "baihe":
            check(stats["max_lag_s"] >= 0.3 and stats["min_ticks_per_stalled_fsync"] == 0,
                  f"SENSITIVITY load baihe: loop blocked (max lag {stats['max_lag_s']} s)")


# ----------------------------------------------------------------------------------------------- main
def main():
    global CAMPAIGN, PROBE
    scratch = Path(sys.argv[1]).resolve()
    allowed = (FIXDIR / "scratch").resolve()
    if allowed not in scratch.parents:
        raise SystemExit(f"scratch must be under {allowed}")
    groups = sys.argv[2:]
    scratch.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-" + "-".join(groups) + "-", dir=scratch))
    (run_dir / "tmp").mkdir()
    tempfile.tempdir = str(run_dir / "tmp")  # _download's TemporaryDirectory
    CAMPAIGN = run_dir / "campaign"
    CAMPAIGN.mkdir()
    os.environ["ATP_DEEPSWE_CAMPAIGN"] = str(CAMPAIGN)
    print(f"python {sys.version.split()[0]}, flags dev_mode={sys.flags.dev_mode} isolated={sys.flags.isolated} "
          f"warnoptions={sys.warnoptions}; run dir {run_dir}", flush=True)

    class Capture(logging.Handler):
        def emit(self, record):
            LOG_RECORDS.append((time.monotonic(), record.levelno, record.getMessage()))

    logger = logging.getLogger("asyncio")
    logger.addHandler(Capture(logging.WARNING))
    logger.propagate = False
    sys.unraisablehook = lambda hook: UNRAISABLE.append(f"{hook.exc_type.__name__}: {hook.exc_value} ({hook.object!r})")
    threading.excepthook = lambda hook: THREAD_ERRORS.append(f"{hook.exc_type.__name__}: {hook.exc_value}")
    warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: WARNED.append(
        f"{category.__name__}: {message} ({filename}:{lineno})")

    V = load_variants(run_dir)
    PROBE = Probe(CAMPAIGN)
    os.fsync = PROBE.fsync
    pathlib.Path.open = tracked_open

    async def run_groups():
        loop = asyncio.get_running_loop()
        loop.set_task_factory(lambda loop, coro, **kwargs: RecordingTask(coro, loop=loop, **kwargs))
        check(loop.get_debug(), "asyncio debug mode on (slow callbacks >= 0.1 s are logged)")
        for group in groups:
            started = time.monotonic()
            if group == "lag":
                await group_lag(V)
            elif group == "bytes":
                await group_bytes(V)
            elif group == "cancel-run":
                await group_cancel(V, SCENARIOS[:4])
            elif group == "cancel-drain":
                await group_cancel(V, SCENARIOS[4:])
            elif group == "drainwait":
                await group_drainwait(V)
            elif group == "errors":
                await group_errors(V)
            elif group == "load":
                await group_load(V, [("fix", 8), ("sftv3", 8), ("sftv3", 32)])
            elif group == "load-baihe":
                await group_load(V, "baihe")
            else:
                raise SystemExit(f"unknown group {group}")
            print(f"group {group}: {time.monotonic() - started:.1f} s", flush=True)

    asyncio.run(run_groups(), debug=True)
    import gc
    gc.collect()
    errors = [message for _, level, message in LOG_RECORDS if level >= logging.ERROR]
    report({"other_fsync_threads": dict(PROBE.other_fsync_threads),
            "unattributed_problems": [p for p in PROBE.problems if p[1] is None]})
    check(not errors and not UNRAISABLE and not WARNED and not THREAD_ERRORS
          and not [p for p in PROBE.problems if p[1] is None],
          f"no asyncio error logs (e.g. exception never retrieved), unraisable exceptions, warnings or thread errors "
          f"({errors[:3]} {UNRAISABLE[:3]} {WARNED[:3]} {THREAD_ERRORS[:3]})")
    failed = [message for ok, message in RESULTS if not ok]
    print(f"SUMMARY {len(RESULTS) - len(failed)}/{len(RESULTS)} PASS" + (f"; FAILED: {failed}" if failed else ""),
          flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
