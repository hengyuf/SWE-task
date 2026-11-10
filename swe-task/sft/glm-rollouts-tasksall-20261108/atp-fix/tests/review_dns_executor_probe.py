"""Reviewer probe: does loop.getaddrinfo (aiohttp ThreadedResolver / anyio) wait on the default executor?

Saturates the loop's default executor (8 threads, as in a 4-CPU job) with 1 s sleeps, then times
loop.getaddrinfo('localhost', 80) under plain asyncio and under uvloop. No network (localhost only).
"""
import sys

sys.dont_write_bytecode = True
import asyncio
import concurrent.futures
import time


async def probe():
    loop = asyncio.get_running_loop()
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=8))
    blockers = [loop.run_in_executor(None, time.sleep, 1.0) for _ in range(8)]
    await asyncio.sleep(0.05)
    t0 = time.monotonic()
    await loop.getaddrinfo("localhost", 80)
    latency = time.monotonic() - t0
    await asyncio.gather(*blockers)
    return latency


def main():
    print("asyncio getaddrinfo latency with saturated default executor: %.3f s" % asyncio.run(probe()))
    import uvloop
    with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
        print("uvloop  getaddrinfo latency with saturated default executor: %.3f s" % runner.run(probe()))


if __name__ == "__main__":
    main()
