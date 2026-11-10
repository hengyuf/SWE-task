"""Behavioural tests of the ATP capture fix (atp-fix/app/scripts/deepswe_capture.py) against Baihe's original
(atp-fix/app/upstream/deepswe_capture.py) and glm-rollouts-sftv3's 2026-11-07 staged-copy patch (asyncio.to_thread,
built in memory from capture_patch.py's OLD/NEW). Real run_opencode() with the fake PTY/sandbox objects and the slow
os.fsync of tools/check_hengyuf_changes.py (imported from atp-fix/app/tools); no file outside the scratch dir is
written, no network, no sandbox, no Slurm.

Groups (each well under 2 minutes):
  unit  _in_stream_io: result/error/thread, cancellation once/twice/thrice while the call runs, a failing call under
        cancellation, asyncio.timeout/wait_for around it, saturation (40 calls > 32 threads), default executor unused.
  e2e   per variant: empty streams, normal copy (bytes, order, per-chunk durability before the next read and before
        streams-complete.json), deadline + 0.3 s drain wait expiring during writes, deadline + failed process control
        (finally cancels each drain twice during a write), injected ENOSPC from fsync.
  load  6 concurrent sessions (12 drains) with 0.3 s fsyncs; a default-executor job (stand-in for aiohttp's threaded
        DNS lookup) issued while 12 stream fsyncs stall, with the default executor sized as in a 4-CPU job (8).
Run: <atp>/runtime/controller/bin/python -I -B <this file> <private scratch dir> {unit,e2e,load} [<fix module file>]
(the optional last argument replaces atp-fix/app/scripts/deepswe_capture.py, e.g. for tests/mutation_check.py)
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import errno
import gc
import importlib.util
import json
import logging
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
import warnings

FIX = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-tasksall-20261108/atp-fix")
ATP_SCRIPTS = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp/scripts")  # deepswe_http (unchanged)
FAILED, REPORTS = [], []
FIX_MODULE = Path(sys.argv[3]) if len(sys.argv) > 3 else FIX / "app/scripts/deepswe_capture.py"


def expect(condition, message):
    print(("PASS " if condition else "FAIL ") + message, flush=True)
    if not condition:
        FAILED.append(message)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = load(FIX / "app/tools/check_hengyuf_changes.py", "check_tool")  # puts atp-fix/app/scripts first on sys.path
sys.path.insert(1, str(ATP_SCRIPTS))


def variants():
    patcher = load(FIX / "capture_patch.py", "capture_patch_constants")
    original = (FIX / "app/upstream/deepswe_capture.py").read_text()
    sftv3 = ModuleType("deepswe_capture_sftv3")
    sftv3.__file__ = str(FIX / "app/upstream/deepswe_capture.py")
    exec(compile(original.replace(patcher.OLD, patcher.NEW, 1), "<sftv3 patched>", "exec"), sftv3.__dict__)
    return {"upstream": load(FIX / "app/upstream/deepswe_capture.py", "deepswe_capture_upstream"),
            "sftv3": sftv3, "fix": load(FIX_MODULE, "deepswe_capture_fix")}


class Recorder(tool.SlowFsync):
    """SlowFsync that also records each stream file's durable size (st_size after a successful fsync)."""

    def __init__(self, plan):
        super().__init__(plan)
        self.durable = {}

    def fsync(self, fd):
        result = super().fsync(fd)
        path = os.readlink(f"/proc/self/fd/{fd}")
        if path.endswith(("/stdout.log", "/stderr.log")):
            with self.lock:
                self.durable[path] = os.fstat(fd).st_size
        return result


class Session(tool.FakeSession):
    def __init__(self, chunks, stdout_only=False):
        self.stdout_only, self.reads = stdout_only, {"stdout": [], "stderr": []}
        super().__init__(chunks)

    def emit(self, name, data):
        if not (self.stdout_only and name == "stderr"):
            super().emit(name, data)

    async def read_stream(self, name):
        chunk = await super().read_stream(name)
        if chunk:
            self.reads[name].append(time.monotonic())
        return chunk


class Sandbox(tool.FakeSandbox):
    def __init__(self, chunks, control_ok=True, stdout_only=False):
        super().__init__(chunks, control_ok)
        self.stdout_only = stdout_only

    async def create(self, command, pty):
        self.session = Session(self.chunks, self.stdout_only)
        return self.session


class Reports(logging.Handler):
    def emit(self, record):
        REPORTS.append(record.getMessage())


def setup(scratch):
    root = Path(tempfile.mkdtemp(prefix="capture-fix-", dir=scratch.resolve()))
    (root / "tmp").mkdir()
    tempfile.tempdir = str(root / "tmp")
    os.environ["ATP_DEEPSWE_CAMPAIGN"] = str(root)
    logging.getLogger("asyncio").addHandler(Reports(logging.WARNING))
    return root


LABELS = iter(range(10**6))


def instrument(module, plan):
    """module.os = a fresh Recorder; streams-complete.json writes record (size, durable size) per stream file."""
    recorder = module.os = Recorder(plan)
    recorder.complete_at = {}
    original = module.__dict__.setdefault("_unspied_write_json", module._write_json)

    def spy(path, value):
        path = Path(path)
        if path.name == "streams-complete.json":
            with recorder.lock:
                recorder.complete_at[path.parent] = {
                    name: (os.stat(path.parent / name).st_size,
                           recorder.durable.get(os.path.realpath(path.parent / name), 0))
                    for name in ("stdout.log", "stderr.log")}
        return original(path, value)

    module._write_json = spy
    tool.DrainCancels.fsync, tool.DrainCancels.log = recorder, []
    return recorder


async def run(root, module, recorder, *, chunks, timeout_s=60, control_ok=True, stdout_only=False):
    attempt = root / "models/fix/attempts" / f"{next(LABELS)}-0" / "attempt-0"
    attempt.mkdir(parents=True)
    sandbox = Sandbox(chunks, control_ok, stdout_only)
    body = {"_atp_attempt_dir": str(attempt), "_atp_deadline": time.time() + 600}

    async def request_json():
        return body

    stop = asyncio.Event()
    lag = asyncio.create_task(tool.loop_lag(stop))
    try:
        result = await module.run_opencode(sandbox, "opencode run test", SimpleNamespace(json=request_json), timeout_s)
        outcome = f"returned {result.return_code}"
    except Exception as error:  # noqa: BLE001 - asserted by the caller
        outcome = f"{type(error).__name__} {getattr(error, 'errno', None) or ''}".strip()
    returned = time.monotonic()
    with recorder.lock:  # stream writes still running after run_opencode ended (abandoned by their drain)
        abandoned = sum(recorder.in_flight.values())
    stop.set()
    samples = await lag
    for _ in range(500):  # let abandoned writes finish before files and records are inspected
        with recorder.lock:
            if not any(recorder.in_flight.values()):
                break
        await asyncio.sleep(0.01)
    calls = recorder.calls
    window = (min([c["start"] for c in calls] or [0.0]), max([c.get("end", returned) for c in calls] or [0.0]) + 0.05)
    inside = [(at, late) for at, late in samples if window[0] <= at <= window[1]]
    events = [json.loads(line)["event"] for line in (attempt / "capture-events.jsonl").read_text().splitlines()]
    return SimpleNamespace(
        outcome=outcome, recorder=recorder, session=sandbox.session, events=events, abandoned=abandoned,
        complete_at=recorder.complete_at.get(attempt, {}), cancels=list(tool.DrainCancels.log),
        lag=max([late for at, late in inside] or [0.0]),
        stalls=[(round(at - window[0], 3), round(late, 3)) for at, late in inside if late > 0.1],
        # 10 ms timer ticks completed inside each stalled (>= 0.2 s) fsync; 0 = the loop was blocked by it
        ticks=min([sum(c["start"] + 0.05 <= at <= c["end"] - 0.01 for at, _ in samples)
                   for c in calls if "end" in c and c["end"] - c["start"] >= 0.2
                   and samples and samples[0][0] <= c["start"] and c["end"] <= samples[-1][0]] or [0]),
        threads={c["thread"].rstrip("0123456789").rstrip("_") for c in calls},
        exists={name: (attempt / f"{name}.log").is_file() for name in ("stdout", "stderr")},
        local={name: (attempt / f"{name}.log").read_bytes() for name in ("stdout", "stderr")},
        streams_complete=(attempt / "streams-complete.json").is_file())


def describe(name, label, case):
    print(json.dumps({"variant": name, "case": label, "outcome": case.outcome, "max_loop_lag_s": round(case.lag, 3),
                      "stream_fsyncs": len(case.recorder.calls), "threads": sorted(case.threads),
                      "drain_cancels_during_write": sum(c["write_in_flight"] for c in case.cancels),
                      "writes_in_flight_at_return": case.abandoned, "min_ticks_per_stalled_fsync": case.ticks,
                      "closed_under_write": len(case.recorder.closed_under_write), "events": case.events,
                      "loop_stalls_over_0.1s(t_rel,lag)": case.stalls[:6]}), flush=True)


def handed_all(case):
    return all(case.local[s] == bytes(case.session.handed[s]) for s in ("stdout", "stderr"))


def ordered(case):
    """Per stream: one fsync per chunk read, one at a time, and chunk k+1 read only after chunk k's fsync ended."""
    for stream in ("stdout", "stderr"):
        calls = [c for c in case.recorder.calls if c["stream"] == stream + ".log"]
        reads = case.session.reads[stream]
        if len(calls) != len(reads) or any(a["end"] > b["start"] for a, b in zip(calls, calls[1:])):
            return False
        if any(read < call["end"] for read, call in zip(reads[1:], calls)):
            return False
    return True


async def e2e(root):
    modules = variants()
    asyncio.get_running_loop().set_task_factory(lambda loop, coro, **kw: tool.DrainCancels(coro, loop=loop, **kw))
    copies = {}
    for name, module in modules.items():
        expected_threads = {"upstream": {"MainThread"}, "sftv3": {"asyncio"}, "fix": {"atp-stream-io"}}[name]
        # Empty streams: both files exist (eager open), streams complete.
        case = await run(root, module, instrument(module, lambda path: 0.0), chunks=0)
        describe(name, "empty", case)
        expect(case.outcome == "returned 0" and case.streams_complete and all(case.exists.values())
               and case.local == {"stdout": b"", "stderr": b""}, f"{name}: no output -> both files exist, empty")
        # A stream with no output while the other has some.
        case = await run(root, module, instrument(module, lambda path: 0.05), chunks=5, stdout_only=True)
        describe(name, "stdout only", case)
        expect(case.outcome == "returned 0" and case.exists["stderr"] and case.local["stderr"] == b""
               and case.local["stdout"] == bytes(case.session.emitted["stdout"]), f"{name}: silent stderr -> empty file")
        # Normal copy, 0.3 s per stream fsync.
        case = copies[name] = await run(root, module, instrument(module, lambda path: 0.3), chunks=8)
        describe(name, "copy", case)
        expect(case.outcome == "returned 0" and case.streams_complete and "agent_snapshot_saved" in case.events,
               f"{name}: copy returns 0 with streams-complete.json and a final snapshot")
        expect(all(case.local[s] == bytes(case.session.emitted[s]) for s in ("stdout", "stderr")) and handed_all(case),
               f"{name}: local copies == the command's bytes, in order")
        expect(ordered(case), f"{name}: per stream one fsync per chunk, sequential, next read after the fsync")
        expect(all(size == durable for size, durable in case.complete_at.values()) and len(case.complete_at) == 2,
               f"{name}: streams-complete.json written only after every byte was fsynced {case.complete_at}")
        expect(case.threads == expected_threads, f"{name}: stream fsyncs ran in {sorted(case.threads)}")
        if name == "upstream":
            expect(case.ticks == 0 and case.lag >= 0.25,
                   f"upstream: the loop is blocked during each fsync (no tick; max lag {case.lag:.2f} s)")
        else:
            expect(case.ticks >= 1 and case.lag < 0.3,
                   f"{name}: the loop ticks during each fsync (min {case.ticks}; max lag {case.lag:.3f} s)")
        # Deadline at 0.6 s; process control works; the 0.3 s drain wait expires while 1.0 s fsyncs are in flight.
        module.STREAM_DRAIN_TIMEOUT_S = 0.3
        started = time.monotonic()
        case = await run(root, module, instrument(module, lambda path: 1.0 if time.monotonic() - started > 0.45 else 0.0),
                         chunks=10**9, timeout_s=0.6)
        module.STREAM_DRAIN_TIMEOUT_S = 15
        describe(name, "drain timeout", case)
        if name == "upstream":
            expect(handed_all(case) and not case.recorder.closed_under_write and case.lag >= 0.9
                   and not any(c["write_in_flight"] for c in case.cancels),
                   f"upstream: drain-timeout run keeps every read chunk; the loop was blocked ({case.lag:.2f} s)")
        elif name == "sftv3":
            expect(bool(case.recorder.closed_under_write),  # (writes still running at return: timing-dependent)
                   "sftv3 patch: the drain-timeout cancel closed a file under an in-flight fsync (the race the fix "
                   f"removes; reproduced {len(case.recorder.closed_under_write)}x, {case.abandoned} left running)")
        else:
            expect(case.outcome == "TimeoutError" and "agent_streams_incomplete" in case.events
                   and not case.streams_complete, "fix: drain wait expired -> TimeoutError, agent_streams_incomplete")
            expect(sum(c["write_in_flight"] for c in case.cancels) >= 2, "fix: both drains cancelled during a write")
            expect(handed_all(case) and not case.recorder.closed_under_write and case.abandoned == 0,
                   "fix: every handed-off chunk landed; no file closed under a write; no write left running")
            expect(case.ticks >= 1 and case.lag < 0.3,
                   f"fix: loop kept running (min {case.ticks} ticks per stalled fsync; max lag {case.lag:.3f} s)")
        # Deadline at 0.9 s, process control fails: finally cancels each drain twice during a 0.6 s fsync.
        case = await run(root, module, instrument(module, lambda path: 0.6), chunks=10**9, timeout_s=0.9, control_ok=False)
        describe(name, "finally double cancel", case)
        if name == "sftv3":
            expect(bool(case.recorder.closed_under_write),  # (writes still running at return: timing-dependent)
                   f"sftv3 patch: finally's cancel closed a file under an in-flight fsync (reproduced "
                   f"{len(case.recorder.closed_under_write)}x, {case.abandoned} left running)")
        elif name == "fix":
            expect(case.outcome == "TimeoutError" and "generation_not_quiesced" in case.events,
                   "fix: failed process control -> TimeoutError, generation_not_quiesced")
            expect(all(sum(c["write_in_flight"] for c in case.cancels if c["stream"] == s) >= 2
                       for s in ("stdout.log", "stderr.log")), "fix: each drain cancelled twice during its write")
            expect(handed_all(case) and not case.recorder.closed_under_write and case.abandoned == 0
                   and case.lag < 0.3, f"fix: handed-off chunks landed, no file closed under a write, no write left "
                                       f"running, lag {case.lag:.3f} s")
        else:
            expect(handed_all(case) and not case.recorder.closed_under_write and case.lag >= 0.5,
                   f"upstream: same outcome {case.outcome}, loop blocked ({case.lag:.2f} s)")
        # ENOSPC from the 3rd stdout fsync, not cancelled: the error ends the run as in Baihe's code.
        count = [0]

        def enospc(path):
            if not path.endswith("/stdout.log"):
                return 0
            count[0] += 1
            return OSError(errno.ENOSPC, "No space left on device (injected)", path) if count[0] == 3 else 0

        case = await run(root, module, instrument(module, enospc), chunks=10**9)
        describe(name, "enospc", case)
        copies[name, "error"] = case
        expect(case.outcome == "OSError 28" and case.local["stdout"] == bytes(case.session.handed["stdout"])
               and case.local["stderr"] == bytes(case.session.handed["stderr"]),
               f"{name}: ENOSPC from fsync propagates out of run_opencode ({case.outcome}); files hold what was read")
    expect(copies["fix"].local == copies["upstream"].local == copies["sftv3"].local,
           "copy: fix, sftv3 and upstream files byte-identical")
    expect(copies["fix", "error"].events == copies["upstream", "error"].events == copies["sftv3", "error"].events,
           f"enospc: same capture events as upstream {copies['fix', 'error'].events}")


async def load_group(root):
    modules = variants()
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=8, thread_name_prefix="asyncio"))  # 4-CPU job: 4 + 4
    for name, chunks in (("fix", 10), ("sftv3", 10), ("upstream", 3)):
        module = modules[name]
        started = time.monotonic()
        recorder = instrument(module, lambda path: 0.3)  # shared by the 6 concurrent sessions
        cases = await asyncio.gather(*(run(root, module, recorder, chunks=chunks) for _ in range(6)))
        elapsed = time.monotonic() - started
        lag, ticks = max(case.lag for case in cases), min(case.ticks for case in cases)
        durations = sorted(c["end"] - c["start"] for c in recorder.calls)
        print(json.dumps({"variant": name, "case": "6 sessions", "elapsed_s": round(elapsed, 2),
                          "max_loop_lag_s": round(lag, 3), "min_ticks_per_stalled_fsync": ticks,
                          "fsync_s_p50_p90_max": [round(durations[len(durations) // 2], 3),
                                                  round(durations[int(len(durations) * 0.9)], 3), round(durations[-1], 3)],
                          "outcomes": sorted({c.outcome for c in cases})}), flush=True)
        exact = all(c.outcome == "returned 0" and c.streams_complete and handed_all(c)
                    and all(c.local[s] == bytes(c.session.emitted[s]) for s in ("stdout", "stderr")) for c in cases)
        expect(exact, f"{name}: 6 concurrent sessions, every copy exact and complete")
        if name == "fix":
            expect(ticks >= 1 and lag < 0.3, f"fix: 12 drains x 0.3 s fsyncs, loop ticks during each (min {ticks}), "
                                             f"max lag {lag:.3f} s")
        elif name == "upstream":
            expect(ticks == 0 and lag >= 0.6, f"upstream: 12 drains x 0.3 s fsyncs on the loop, lag {lag:.2f} s")

    async def resolver_wait(module):
        """Latency of a default-executor job (aiohttp ThreadedResolver / anyio getaddrinfo stand-in, numeric host:
        no lookup) issued while 6 sessions' 12 stream fsyncs stall for 1.5 s."""
        start = [None]

        def plan(path):
            if start[0] is None:
                start[0] = time.monotonic()
            return 1.5 if time.monotonic() - start[0] < 1.0 else 0.0

        async def probe():
            while not (start[0] and time.monotonic() - start[0] > 0.3):
                await asyncio.sleep(0.01)
            began = time.monotonic()
            await loop.getaddrinfo("127.0.0.1", 80, flags=socket.AI_NUMERICHOST)
            return time.monotonic() - began

        recorder = instrument(module, plan)
        sessions = [run(root, module, recorder, chunks=4) for _ in range(6)]
        results = await asyncio.gather(probe(), *sessions)
        return results[0], results[1:]

    for name in ("fix", "sftv3"):
        wait, cases = await resolver_wait(modules[name])
        print(json.dumps({"variant": name, "case": "resolver during stall", "default_executor_wait_s": round(wait, 3)}))
        expect(all(c.outcome == "returned 0" and handed_all(c) for c in cases), f"{name}: stall sessions complete")
        if name == "fix":
            expect(wait < 0.2, f"fix: default executor free during the stall (job waited {wait:.3f} s)")
        else:
            expect(wait >= 0.5, f"sftv3 patch: default executor pinned by stalled fsyncs (job waited {wait:.2f} s)")


async def unit():
    fix = variants()["fix"]
    finished = threading.Event()

    def slow(seconds=0.3, error=None):
        time.sleep(seconds)
        finished.set()
        if error:
            raise error
        return "done"

    name = await fix._in_stream_io(lambda: threading.current_thread().name)
    expect(name.startswith("atp-stream-io_"), f"runs in the atp-stream-io pool ({name})")
    marker = OSError(errno.EIO, "injected")
    try:
        await fix._in_stream_io(slow, 0.01, marker)
        same = False
    except OSError as error:
        same = error is marker and error.__traceback__ is not None
    expect(same, "an error of the call propagates (same exception object)")
    for times in (1, 2, 3):
        finished.clear()
        task = asyncio.ensure_future(fix._in_stream_io(slow))
        at_done = []
        task.add_done_callback(lambda _: at_done.append(finished.is_set()))
        for _ in range(times):  # separately delivered: the task re-suspends between cancellations
            await asyncio.sleep(0.05)
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        expect(task.cancelled() and at_done == [True],
               f"cancelled {times}x while running: ends cancelled only after the call finished")
    finished.clear()
    task = asyncio.ensure_future(fix._in_stream_io(slow, 0.2, OSError(errno.EIO, "fails under cancellation")))
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    del task
    gc.collect()
    expect(finished.is_set() and not REPORTS, "call failing under cancellation: no unretrieved-exception report")
    for label in ("asyncio.timeout", "asyncio.wait_for"):
        finished.clear()
        began = time.monotonic()
        try:
            if label == "asyncio.timeout":
                async with asyncio.timeout(0.1):
                    await fix._in_stream_io(slow)
            else:
                await asyncio.wait_for(fix._in_stream_io(slow), 0.1)
            raised = None
        except TimeoutError:
            raised = "TimeoutError"
        elapsed = time.monotonic() - began
        expect(raised == "TimeoutError" and finished.is_set() and elapsed >= 0.28,
               f"{label}(0.1 s): TimeoutError after the call finished ({elapsed:.2f} s)")
    expect(await fix._in_stream_io(lambda: 42) == 42, "pool still healthy after errors and cancellations")
    names = set()

    def busy(index):
        names.add(threading.current_thread().name)
        time.sleep(0.1)
        return index

    results = await asyncio.gather(*(fix._in_stream_io(busy, index) for index in range(40)))
    expect(results == list(range(40)) and len(names) <= 32, f"40 concurrent calls complete on {len(names)} threads")
    used = []

    class Spy(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            used.append(args[0])
            return super().submit(*args, **kwargs)

    asyncio.get_running_loop().set_default_executor(Spy(max_workers=2))
    await asyncio.gather(*(fix._in_stream_io(time.sleep, 0.01) for _ in range(5)))
    expect(not used, "_in_stream_io never uses the loop's default executor")


def main():
    scratch, group = Path(sys.argv[1]), sys.argv[2]
    root = setup(scratch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run({"unit": unit, "e2e": lambda: e2e(root), "load": lambda: load_group(root)}[group]())
        gc.collect()
    leaks = [f"{w.category.__name__}: {w.message}" for w in caught if issubclass(w.category, (RuntimeWarning, ResourceWarning))]
    expect(not REPORTS and not leaks, f"no unretrieved-exception reports or Runtime/ResourceWarnings {REPORTS + leaks}")
    print("scratch:", root)
    print("RESULT", group, "FAIL" if FAILED else "PASS", f"({len(FAILED)} failed)")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
