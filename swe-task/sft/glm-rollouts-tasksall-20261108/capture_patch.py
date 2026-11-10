"""glm-rollouts-tasksall (2026-11-08; successor of glm-rollouts-sftv3-20261107/capture_patch.py): checks, and if needed
patches, the JOB-LOCAL STAGED copy of ATP's deepswe_capture.py (never the shared S/atp tree, never a campaign's verified
source snapshot).

ATP's run_opencode() copied every OpenCode output chunk to the attempt dir on Lustre with write + os.fsync INSIDE the
Gym agent server's event loop. A stalled Lustre fsync (seconds) blocked that loop, aiohttp missed the PTY websocket
heartbeat pongs (30 s heartbeat, ~15 s pong window) and closed every PTY of the process at once: SandboxPtyError
("PTY connection closed unexpectedly", close code 1006/None), OpenCode killed, half-done workspace graded as a fail.
2026-11-07: 330 of 748 terminal slots in bursts of 3-32 sessions per campaign process.

Since 2026-11-08 the shared ATP app carries the fix (tracked hengyuf change in scripts/deepswe_capture.py: the write +
fsync run in its own thread pool, _STREAM_IO, via _in_stream_io), so patch_file() returns:
  "upstream"  the staged copy has the shared app's fix (no write);
  "already"   the staged copy carries this patch (NEW) already (no write);
  "patched"   the staged copy had Baihe's drain() (OLD) and now has NEW: glm-rollouts-sftv3's 2026-11-07 patch (same
              write + fsync via asyncio.to_thread), the fallback for a copy staged from an unfixed app;
and raises for anything else (fail closed). The app's fix mentions MARK in a comment, so glm-rollouts-sftv3's own
patcher reports "already" for it; UPSTREAM is therefore checked first here.
"""
from pathlib import Path

OLD = """    async def drain(reader, name):
        with (directory / name).open("ab", buffering=0) as stream:
            while chunk := await reader():
                stream.write(chunk)
                os.fsync(stream.fileno())
"""
NEW = """    async def drain(reader, name):
        with (directory / name).open("ab", buffering=0) as stream:
            def persist(data, stream=stream):  # glm-rollouts-sftv3 capture_patch.py: write + fsync off the event loop
                stream.write(data)
                os.fsync(stream.fileno())
            while chunk := await reader():
                await asyncio.to_thread(persist, chunk)
"""
MARK = "glm-rollouts-sftv3 capture_patch.py"
# Whole lines of the shared app's fix (hengyuf change 2026-11-08); each must occur exactly once for "upstream".
UPSTREAM = (
    '_STREAM_IO = ThreadPoolExecutor(max_workers=32, thread_name_prefix="atp-stream-io")\n',
    "async def _in_stream_io(function, *args):\n",
    "            while chunk := await reader():\n                await _in_stream_io(persist, chunk)\n",
)


def _lines_count(text: str, fragment: str) -> int:
    """Occurrences of fragment's whole lines in text, line-aligned (overlapping runs counted)."""
    lines, block = text.splitlines(keepends=True), fragment.splitlines(keepends=True)
    return sum(lines[index:index + len(block)] == block for index in range(len(lines) - len(block) + 1))


def patch_file(path: Path) -> str:
    """Check one staged deepswe_capture.py: 'upstream', 'already', or 'patched' (in place); raises otherwise."""
    text = path.read_text()
    fixed = [_lines_count(text, fragment) == 1 for fragment in UPSTREAM]
    if all(fixed) and OLD not in text and NEW not in text:
        return "upstream"
    if any(fragment in text for fragment in UPSTREAM):
        raise RuntimeError(f"{path}: partial or mixed ATP capture fix; review capture_patch.py")
    if text.count(NEW) == 1 and OLD not in text:
        return "already"
    if MARK in text or text.count(OLD) != 1:
        raise RuntimeError(f"{path}: drain() block not found exactly once; ATP changed -- review capture_patch.py")
    if "import asyncio" not in text:
        raise RuntimeError(f"{path}: no asyncio import")
    new = text.replace(OLD, NEW, 1)
    compile(new, str(path), "exec")
    tmp = path.with_name(path.name + ".glm-tmp")
    tmp.write_text(new)
    tmp.replace(path)
    for cached in path.parent.glob("__pycache__/deepswe_capture*.pyc"):
        cached.unlink()
    return "patched"
