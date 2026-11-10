"""Reviewer probe: process exit while a staged-fix stream write is in flight.

  <python> -I -B review_exit_probe.py <scratch dir> <case>
case 'run'   : asyncio.run's main returns while a drain-like task is inside _in_stream_io (1 s write);
               asyncio.run cancels it -> the task must end only after the write; process exits ~1 s.
case 'close' : loop.close() without cancelling (abnormal teardown) then gc -> GeneratorExit path; reports whether
               the file was closed under the in-flight write.
case 'uvloop': as 'run' under uvloop (uvicorn's loop).
"""
import sys

sys.dont_write_bytecode = True
import asyncio
import gc
import importlib.util
import os
from pathlib import Path
import threading
import time

LIVE = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp")
STAGED = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-tasksall-20261108/atp-fix/app/scripts/deepswe_capture.py")
sys.path.insert(0, str(LIVE / "scripts"))
spec = importlib.util.spec_from_file_location("rv_fixed_exit", STAGED)
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)

scratch, case = Path(sys.argv[1]), sys.argv[2]
scratch.mkdir(parents=True, exist_ok=True)
path = scratch / f"exit-{case}-{os.getpid()}.log"
state = {"write_done": None, "drain_end": None, "closed_under_write": False}


def persist_factory(stream):
    def persist(data):
        fd = stream.fileno()
        before = os.fstat(fd)
        time.sleep(1.0)
        try:
            after = os.fstat(fd)
            state["closed_under_write"] = (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
        except OSError:
            state["closed_under_write"] = True
        if not state["closed_under_write"]:
            stream.write(data)
        state["write_done"] = time.monotonic()
    return persist


async def drain():
    try:
        with path.open("ab", buffering=0) as stream:
            await fixed._in_stream_io(persist_factory(stream), b"chunk\n")
    finally:
        state["drain_end"] = time.monotonic()


async def main_run():
    asyncio.get_running_loop().create_task(drain())
    await asyncio.sleep(0.1)  # return while the write is in flight


t0 = time.monotonic()
if case == "run":
    asyncio.run(main_run())
elif case == "uvloop":
    import uvloop
    with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
        runner.run(main_run())
else:
    loop = asyncio.new_event_loop()
    task = loop.create_task(drain())
    loop.run_until_complete(asyncio.sleep(0.1))
    loop.close()  # abnormal: pending task never cancelled
    del task
    gc.collect()
    time.sleep(1.2)
print(case, {"drain_ended_after_write": None if state["drain_end"] is None or state["write_done"] is None
             else state["drain_end"] >= state["write_done"],
             "closed_under_write": state["closed_under_write"],
             "bytes": path.read_bytes() if path.exists() else None,
             "elapsed_to_exit_s": round(time.monotonic() - t0, 2),
             "threads": sorted(t.name.rstrip("0123456789_") for t in threading.enumerate())}, flush=True)
path.unlink(missing_ok=True)
