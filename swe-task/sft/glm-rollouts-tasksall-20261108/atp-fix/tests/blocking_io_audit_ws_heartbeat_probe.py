"""blocking-io audit probe 1: aiohttp websocket heartbeat vs. a blocked event loop.

Reproduces the 2026-11-07 PTY-drop mechanism in miniature, without network or sandboxes:
an aiohttp websocket server (own thread + own event loop, UNIX socket in a private scratch dir,
pong delayed by 50 ms to model the NLB/proxy RTT) and an idle client socket with heartbeat=2.0
(pong window 1.0 s; ATP's PTY sockets use 30 s / 15 s). The client loop then runs "drain" steps
that each block for 0.6 s, as ATP's run_opencode.drain does when a Lustre fsync stalls.
A socket dies when the blocking work of ONE loop iteration (all drains with a pending chunk run their
write+fsync step back to back) exceeds the pong window after a PING: then the PONG and the expired pong
timer are selected in the same iteration and the timer runs first (aiohttp defers the heartbeat reset
with call_soon). So it is the number of drains per loop times the fsync latency that matters.

  blocking x1: one drain, time.sleep(0.6) on the loop per chunk  -> 0.6 s/iteration < 1 s window: survives
  blocking x4: four drains, same                                 -> 2.4 s/iteration > 1 s window: close 1006
  offloop  x4: four drains, await asyncio.to_thread(...)         -> loop free: socket alive

Run: <atp>/runtime/controller/bin/python -I -B <this file> <private scratch dir>
"""
import asyncio
import os
import sys
import threading
import time

import aiohttp
from aiohttp import web

PONG_DELAY_S = 0.05
HEARTBEAT_S = 2.0
STEP_BLOCK_S = 0.6
STEPS = 6  # per drain; >= 3.6 s of drain work: covers at least one idle heartbeat period


def serve(socket_path, ready, stop):
    async def handler(request):
        ws = web.WebSocketResponse(autoping=False)
        await ws.prepare(request)
        async for message in ws:
            if message.type == aiohttp.WSMsgType.PING:
                await asyncio.sleep(PONG_DELAY_S)
                await ws.pong(message.data)
            elif message.type == aiohttp.WSMsgType.CLOSE:
                break
        return ws

    async def main():
        app = web.Application()
        app.router.add_get("/ws", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.UnixSite(runner, socket_path).start()
        ready.set()
        while not stop.is_set():
            await asyncio.sleep(0.05)
        await runner.cleanup()

    asyncio.run(main())


async def client_case(socket_path, mode, drains):
    connector = aiohttp.UnixConnector(path=socket_path)
    async with aiohttp.ClientSession(connector=connector) as session:
        ws = await session.ws_connect("http://localhost/ws", heartbeat=HEARTBEAT_S)
        pump_result = {}

        async def pump():  # stands in for OpenSandboxPtySession._pump_socket
            async for message in ws:
                pump_result.setdefault("messages", []).append(message.type.name)
            pump_result["ended_at"] = time.monotonic()

        pump_task = asyncio.create_task(pump())
        await asyncio.sleep(HEARTBEAT_S + 0.5)  # let one idle heartbeat round trip complete normally
        alive_before = not ws.closed
        started = time.monotonic()

        async def drain():
            for _ in range(STEPS):
                if mode == "blocking":
                    time.sleep(STEP_BLOCK_S)  # == stream.write + os.fsync stalled on Lustre, on the loop
                else:
                    await asyncio.to_thread(time.sleep, STEP_BLOCK_S)
                await asyncio.sleep(0)  # next chunk: the drain task yields between chunks

        await asyncio.gather(*(drain() for _ in range(drains)))
        elapsed = time.monotonic() - started
        await asyncio.sleep(0.2)
        result = {"mode": mode, "drains": drains, "alive_before": alive_before, "closed_after": ws.closed,
                  "close_code": ws.close_code, "exception": repr(ws.exception()),
                  "drain_work_seconds": round(elapsed, 2)}
        if not ws.closed:
            await ws.close()
        await asyncio.gather(pump_task, return_exceptions=True)
        return result


def main():
    scratch = os.path.abspath(sys.argv[1])
    os.makedirs(scratch, exist_ok=True)
    os.chdir(scratch)  # relative AF_UNIX path: short, private, no shared /tmp
    socket_path = "ws-heartbeat-probe.sock"
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    ready, stop = threading.Event(), threading.Event()
    server = threading.Thread(target=serve, args=(socket_path, ready, stop), daemon=True)
    server.start()
    if not ready.wait(10):
        raise SystemExit("server did not start")
    try:
        results = [asyncio.run(client_case(socket_path, mode, drains))
                   for mode, drains in (("blocking", 1), ("blocking", 4), ("offloop", 4))]
    finally:
        stop.set()
        server.join(5)
        if os.path.exists(socket_path):
            os.unlink(socket_path)
    for result in results:
        print(result)
    single, blocking, offloop = results
    ok = (single["alive_before"] and not single["closed_after"]
          and blocking["alive_before"] and blocking["closed_after"] and blocking["close_code"] == 1006
          and offloop["alive_before"] and not offloop["closed_after"])
    print("PROBE", "PASS" if ok else "FAIL",
          "(1 blocking drain survives; 4 blocking drains -> 1006 / no PONG; 4 off-loop drains -> alive)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
