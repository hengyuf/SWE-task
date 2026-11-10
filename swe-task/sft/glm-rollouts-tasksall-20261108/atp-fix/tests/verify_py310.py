"""Reviewer (round 1): the staged deepswe_capture.py imports and _in_stream_io behaves under the interpreter running
this script (meant for /usr/bin/python3 = 3.10, which may import the module via deepswe_campaign.recover_attempt).
Run: /usr/bin/python3 -I -B verify_py310.py <private scratch dir>
"""
import sys

sys.dont_write_bytecode = True

import asyncio  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
STAGED = S / "sft/glm-rollouts-tasksall-20261108/atp-fix/app/scripts"
ok = []


def check(condition, message):
    ok.append(bool(condition))
    print(("PASS " if condition else "FAIL ") + message, flush=True)


scratch = Path(sys.argv[1]).resolve()
if (S / "sft/glm-rollouts-tasksall-20261108/atp-fix/scratch").resolve() not in scratch.parents:
    raise SystemExit("scratch must be under atp-fix/scratch")
scratch.mkdir(parents=True, exist_ok=True)
deps = Path(tempfile.mkdtemp(prefix="py310-deps-", dir=scratch))
for name in ("deepswe_http.py", "deepswe_upstream.py"):
    shutil.copyfile(S / "atp/scripts" / name, deps / name)
sys.path[:0] = [str(STAGED), str(deps)]
import deepswe_capture as capture  # noqa: E402

check(Path(capture.__file__).resolve() == (STAGED / "deepswe_capture.py").resolve(),
      f"python {sys.version.split()[0]}: staged deepswe_capture imports ({capture._STREAM_IO._max_workers} workers)")


async def main():
    check(await capture._in_stream_io(lambda a, b: a + b, 2, 3) == 5, "result returned")
    error = OSError(28, "injected")

    def fail():
        raise error

    try:
        await capture._in_stream_io(fail)
    except OSError as raised:
        check(raised is error, "the same exception object is re-raised")
    finished = threading.Event()

    def slow():
        time.sleep(0.4)
        finished.set()

    task = asyncio.ensure_future(capture._in_stream_io(slow))
    for _ in range(2):
        await asyncio.sleep(0.1)
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    check(task.cancelled() and finished.is_set(), "cancelled twice: ends cancelled only after the call finished")
    finished.clear()
    started = time.monotonic()
    try:
        await asyncio.wait_for(capture._in_stream_io(slow), 0.1)
    except asyncio.TimeoutError:
        check(finished.is_set() and time.monotonic() - started >= 0.39, "wait_for timeout raised after the call ended")
    threads = {thread.name.rstrip("0123456789_") for thread in threading.enumerate()}
    check("atp-stream-io" in threads, f"calls ran in atp-stream-io threads ({sorted(threads)})")


asyncio.run(main())
print(f"SUMMARY {sum(ok)}/{len(ok)} PASS")
raise SystemExit(0 if all(ok) else 1)
