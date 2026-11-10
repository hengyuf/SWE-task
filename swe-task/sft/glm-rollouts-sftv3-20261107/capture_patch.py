"""glm-rollouts-sftv3: patch for the JOB-LOCAL staged copy of ATP's deepswe_capture.py (never the shared S/atp tree,
never a campaign's verified source snapshot).

ATP's run_opencode() copies every OpenCode output chunk to the attempt dir on Lustre with write + os.fsync INSIDE the
Gym agent server's event loop. A stalled Lustre fsync (seconds) blocks that loop, aiohttp misses the PTY websocket
heartbeat pongs (30 s heartbeat, ~15 s pong window) and closes every PTY of the process at once: SandboxPtyError
("PTY connection closed unexpectedly", close code 1006/None), OpenCode killed, half-done workspace graded as a fail.
2026-11-07: 330 of 748 terminal slots in bursts of 3-32 sessions per campaign process (kimi-rollouts-v4, 8 sessions per
process: 1%). The patch runs the same write + fsync in a worker thread: same bytes, same order, same durability, but
the event loop (heartbeats) keeps running.
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


def patch_file(path: Path) -> str:
    """Patch one staged deepswe_capture.py in place; returns 'patched', 'already' or raises."""
    text = path.read_text()
    if MARK in text:
        return "already"
    if text.count(OLD) != 1:
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
