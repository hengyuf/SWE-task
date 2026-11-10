"""blocking-io audit probe 2: ATP's real run_opencode() with fake PTY/sandbox objects and a stalled fsync.

Three variants of scripts/deepswe_capture.py are built IN MEMORY (no file anywhere is modified):
  original      : the shared app's file as is (write + os.fsync on the event loop per chunk)
  capture_patch : glm-rollouts-sftv3-20261107/capture_patch.py's OLD -> NEW replacement (to_thread per chunk)
  proposed      : open/append(+short-write loop)/fsync/close in a dedicated executor, per-stream lock so a
                  close can never run under an in-flight append; close is submitted, never awaited on the loop

Fake os.fsync stalls INSIDE the "syscall" (it dups the descriptor first, like the kernel holding the file),
only for the local stdout.log / stderr.log copies. A ticker task measures the event loop's worst lag.

Scenario A (success): 20 chunks per stream, one 2.5 s stall on the 5th stdout chunk.
Scenario B (deadline): chunks until killed; run_opencode timeout 1.0 s; drain wait patched to 0.5 s;
           every stream fsync after t0+0.8 s stalls 1.5 s, so the post-interruption drain wait times out and
           cancels drains while an append is in flight.

Run: <atp>/runtime/controller/bin/python -I -B <this file> <private scratch dir>
"""
import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

ATP = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp")
SCRIPTS = ATP / "scripts"
CAPTURE_PATCH = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-sftv3-20261107/"
                     "capture_patch.py")
sys.path.insert(0, str(SCRIPTS))  # read-only imports of deepswe_http / deepswe_upstream (-B: no __pycache__)

PROPOSED_DRAIN = """    async def drain(reader, name):
        # Lustre open/append/fsync/close of the live stream copy run off the event loop; same bytes, same order,
        # same per-chunk durability. The lock orders close() after an append still running in a worker.
        loop = asyncio.get_running_loop()
        stream = await loop.run_in_executor(_CAPTURE_IO, lambda: (directory / name).open("ab", buffering=0))
        guard = threading.Lock()

        def append(data):
            with guard:
                view = memoryview(data)
                while view:
                    view = view[stream.write(view):]
                os.fsync(stream.fileno())

        def close():
            with guard:
                stream.close()

        try:
            while chunk := await reader():
                await loop.run_in_executor(_CAPTURE_IO, append, chunk)
        finally:
            try:  # after any append still running (guard); never waits on Lustre on the loop
                _CAPTURE_IO.submit(close)
            except RuntimeError:  # executor shut down (interpreter exit)
                close()
"""
PROPOSED_HEADER = ("import threading\nfrom concurrent.futures import ThreadPoolExecutor\n"
                   "_CAPTURE_IO = ThreadPoolExecutor(max_workers=16, thread_name_prefix='atp-capture-io')\n")


def load_capture_patch_constants():
    spec = importlib.util.spec_from_file_location("glm_capture_patch_readonly", CAPTURE_PATCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OLD, module.NEW


def build_variants():
    original = (SCRIPTS / "deepswe_capture.py").read_text()
    old, new = load_capture_patch_constants()
    if original.count(old) != 1:
        raise SystemExit("capture_patch OLD block not found exactly once in the shared deepswe_capture.py")
    variants = {"original": original, "capture_patch": original.replace(old, new, 1)}
    marker = "_ATTEMPT = ContextVar("
    proposed = original.replace(old, PROPOSED_DRAIN, 1).replace(marker, PROPOSED_HEADER + marker, 1)
    variants["proposed"] = proposed
    return variants


class FakeOs(types.ModuleType):
    """os proxy for one variant module: fsync of stdout.log/stderr.log may stall inside the 'syscall'."""

    def __init__(self, real, stall):
        super().__init__("os")
        self._real, self._stall, self.errors, self.stalls, self.stalls_on_loop = real, stall, [], 0, 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def fsync(self, fd):
        try:
            target = self._real.readlink(f"/proc/self/fd/{fd}")
        except OSError as error:
            self.errors.append(f"readlink fd {fd}: {error!r}")
            raise
        delay = self._stall(target) if target.endswith(("/stdout.log", "/stderr.log")) else 0
        if not delay:
            return self._real.fsync(fd)
        self.stalls += 1
        if threading.current_thread() is threading.main_thread():
            self.stalls_on_loop += 1
        held = self._real.dup(fd)  # the kernel keeps the file during a stalled fsync, whatever happens to fd
        try:
            time.sleep(delay)
            return self._real.fsync(held)
        except OSError as error:
            self.errors.append(f"fsync {target}: {error!r}")
            raise
        finally:
            self._real.close(held)


class FakeSession:
    def __init__(self, plan, killed):
        self.session_id = "fake-pty-session"
        self._queues = {"stdout": asyncio.Queue(), "stderr": asyncio.Queue()}
        self._exit = asyncio.get_running_loop().create_future()
        self.emitted = {"stdout": bytearray(), "stderr": bytearray()}
        self.closed_with = None
        self._producer = asyncio.create_task(self._produce(plan, killed))

    async def _produce(self, plan, killed):
        index = 0
        while True:
            if plan["until_killed"] and killed.is_set():
                break
            if not plan["until_killed"] and index >= plan["chunks"]:
                break
            await asyncio.sleep(plan["interval"])
            for channel in ("stdout", "stderr"):
                data = f"{channel} chunk {index:04d} {'x' * 40}\n".encode()
                self._queues[channel].put_nowait(data)
                self.emitted[channel] += data
            index += 1
        if killed.is_set():
            data = b"terminated tail\n"
            self._queues["stdout"].put_nowait(data)
            self.emitted["stdout"] += data
        for queue in self._queues.values():
            queue.put_nowait(None)
        self._exit.set_result(143 if killed.is_set() else 0)

    async def _read(self, channel):
        chunk = await self._queues[channel].get()
        if chunk is None:
            self._queues[channel].put_nowait(None)
            return b""
        return chunk

    async def read(self):
        return await self._read("stdout")

    async def read_stderr(self):
        return await self._read("stderr")

    async def wait_exit(self):
        return await asyncio.shield(self._exit)

    async def close(self, *, preserve_session=False):
        self.closed_with = {"preserve_session": preserve_session}


class FakeSandbox:
    def __init__(self, plan):
        self.plan, self.killed, self.session = plan, asyncio.Event(), None
        self.pty = SimpleNamespace(create=self._create)

    async def _create(self, command, pty):
        self.session = FakeSession(self.plan, self.killed)
        return self.session

    async def exec(self, command, timeout_s=None, cwd=None):
        ok = SimpleNamespace(return_code=0, stdout="", stderr="")
        if "control_channel" in command:  # _process_control_command(): stop the producer, prove it
            self.killed.set()
            await asyncio.sleep(0.05)
            return SimpleNamespace(return_code=0, stderr="", stdout=json.dumps({
                "confirmed": True, "control_channel": "sandbox_exec", "process": {"pgid": 7}, "signals": ["SIGTERM"],
                "wrapper_forced_stop": False, "wrapper": {"confirmed": True, "process": {"pgid": 6}, "signals": []}}))
        if command.startswith("test -f") and "stream-writer-error.json" in command:
            return SimpleNamespace(return_code=1, stdout="", stderr="")
        return ok

    async def download(self, remote, local):
        name = remote.rsplit("/", 1)[-1]
        if name in {"stdout.log", "stderr.log"}:
            data = bytes(self.session.emitted[name.split(".")[0]])
        elif name == "stream-writers-complete.json":
            data = json.dumps({"stdout_exit_code": 0, "stderr_exit_code": 0}).encode()
        else:
            data = b"fake-archive"
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        Path(local).write_bytes(data)


async def run_case(module, attempt, plan, timeout_s):
    loop = asyncio.get_running_loop()
    stop, lag = asyncio.Event(), {"max": 0.0}

    async def ticker():
        while not stop.is_set():
            before = loop.time()
            await asyncio.sleep(0.01)
            lag["max"] = max(lag["max"], loop.time() - before - 0.01)

    tick = asyncio.create_task(ticker())
    sandbox = FakeSandbox(plan)
    body = {"_atp_attempt_dir": str(attempt), "_atp_deadline": time.time() + 600}

    async def request_json():
        return body

    started = time.monotonic()
    outcome = "returned"
    try:
        result = await module.run_opencode(sandbox, "opencode run fake", SimpleNamespace(json=request_json), timeout_s)
        outcome += f" rc={result.return_code} error_type={result.error_type}"
    except BaseException as error:  # noqa: BLE001 - recorded for the report
        outcome = f"raised {type(error).__name__}"
    elapsed = time.monotonic() - started
    stop.set()
    await tick
    return sandbox, {"outcome": outcome, "elapsed_s": round(elapsed, 2), "max_loop_lag_s": round(lag["max"], 3)}


def instrument(module):
    """Record wall time of the module's sync Lustre helpers when they run on the event loop thread."""
    timings = {}

    def wrap(name):
        function = getattr(module, name)

        def timed(*args, **kwargs):
            if threading.current_thread() is not threading.main_thread():
                return function(*args, **kwargs)
            started = time.monotonic()
            try:
                return function(*args, **kwargs)
            finally:
                timings.setdefault(name, []).append(time.monotonic() - started)
        setattr(module, name, timed)

    for name in ("_write_json", "_event"):
        wrap(name)
    return timings


def make_attempt(campaign, label):
    attempt = campaign / "models" / "m" / "attempts" / label / "attempt-0"
    attempt.mkdir(parents=True)
    return attempt


def events(attempt):
    path = attempt / "capture-events.jsonl"
    return [json.loads(line)["event"] for line in path.read_text().splitlines()] if path.exists() else []


def main():
    scratch = Path(sys.argv[1]).resolve()
    run_root = scratch / f"drain-probe-{os.getpid()}"
    (run_root / "tmp").mkdir(parents=True)
    tempfile.tempdir = str(run_root / "tmp")  # _download's TemporaryDirectory: private scratch, never shared /tmp
    campaign = run_root / "campaign"
    campaign.mkdir()
    os.environ["ATP_DEEPSWE_CAMPAIGN"] = str(campaign)
    variants = build_variants()
    report, ok = [], True
    for index, (name, source) in enumerate(variants.items()):
        for scenario in ("A", "B"):
            t0 = time.monotonic()
            stdout_stalled = {"n": 0}

            def stall(target, scenario=scenario, t0=t0, counter=stdout_stalled):
                if scenario == "A":
                    if target.endswith("/stdout.log"):
                        counter["n"] += 1
                        return 2.5 if counter["n"] == 5 else 0
                    return 0
                return 1.5 if time.monotonic() - t0 > 0.8 else 0

            module = types.ModuleType(f"deepswe_capture_{name}_{scenario}")
            module.__file__ = str(SCRIPTS / "deepswe_capture.py")
            exec(compile(source, f"<deepswe_capture {name}>", "exec"), module.__dict__)
            fake_os = FakeOs(os, stall)
            module.os = fake_os
            timings = instrument(module)
            attempt = make_attempt(campaign, f"{index}-{ord(scenario) - ord('A')}")
            if scenario == "A":
                plan, timeout_s = {"until_killed": False, "chunks": 20, "interval": 0.05}, 60
            else:
                module.STREAM_DRAIN_TIMEOUT_S = 0.5
                plan, timeout_s = {"until_killed": True, "chunks": 0, "interval": 0.05}, 1.0
            sandbox, row = asyncio.run(run_case(module, attempt, plan, timeout_s))
            time.sleep(4.0)  # let abandoned worker appends of patched variants finish before comparing bytes
            stdout_local = (attempt / "stdout.log").read_bytes()
            stderr_local = (attempt / "stderr.log").read_bytes()
            emitted = {key: bytes(value) for key, value in sandbox.session.emitted.items()}
            on_loop = {key: {"calls": len(value), "max_s": round(max(value), 3), "sum_s": round(sum(value), 3)}
                       for key, value in timings.items()}
            row.update(variant=name, scenario=scenario, fsync_stalls=fake_os.stalls,
                       stalled_fsyncs_on_loop=fake_os.stalls_on_loop, on_loop_helpers=on_loop,
                       fsync_errors=fake_os.errors,
                       streams_complete=(attempt / "streams-complete.json").is_file(),
                       final_snapshot=(attempt / "agent-final-snapshot.json").is_file(),
                       stdout_exact=stdout_local == emitted["stdout"], stderr_exact=stderr_local == emitted["stderr"],
                       stdout_is_prefix=emitted["stdout"].startswith(stdout_local),
                       stderr_is_prefix=emitted["stderr"].startswith(stderr_local),
                       local_vs_emitted_bytes=(len(stdout_local) + len(stderr_local),
                                               len(emitted["stdout"]) + len(emitted["stderr"])),
                       events=[e for e in events(attempt) if e != "agent_snapshot_saved"])
            if scenario == "A":
                good = (row["outcome"].startswith("returned rc=0") and row["streams_complete"] and row["final_snapshot"]
                        and row["stdout_exact"] and row["stderr_exact"] and not row["fsync_errors"])
                good = good and (row["stalled_fsyncs_on_loop"] >= 1 and row["max_loop_lag_s"] >= 2.0 if name == "original"
                                 else row["stalled_fsyncs_on_loop"] == 0 and row["max_loop_lag_s"] < 2.0)
            elif name == "original":
                # Sync drains cannot be cancelled mid-write: they fsync the whole backlog without yielding.
                good = (row["outcome"] == "raised TimeoutError" and row["stalled_fsyncs_on_loop"] >= 2
                        and row["max_loop_lag_s"] >= 3.0 and row["stdout_exact"] and row["stderr_exact"])
            else:
                good = (row["outcome"] == "raised TimeoutError" and not row["streams_complete"]
                        and row["stdout_is_prefix"] and row["stderr_is_prefix"] and not row["fsync_errors"]
                        and "agent_streams_incomplete" in row["events"] and row["stalled_fsyncs_on_loop"] == 0)
            row["expected"] = good
            ok = ok and good
            report.append(row)
            print(json.dumps(row), flush=True)
    print("PROBE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
