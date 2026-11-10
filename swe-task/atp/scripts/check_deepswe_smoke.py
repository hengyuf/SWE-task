#!/usr/bin/env python3
"""Run one bounded live rollout per model, separate from production scores."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import time
import uuid

import deepswe_campaign as campaign
from deepswe_source import freeze_source
from deepswe_stage import stage
from run_deepswe import job_end_time, run_model, save_json


async def main(args):
    staged = stage(args.app, args.scratch)
    settings = dict(staged["settings"], repeats=1, max_attempts=1,
                    agent_timeout_seconds=args.agent_timeout, shards=1,
                    shutdown_grace_seconds=60,
                    initial_concurrency_per_model=16, max_concurrency_per_model=16)
    fake_server = None
    fake_calls = []
    if args.synthetic_delay is not None:
        from aiohttp import web
        async def reply(request):
            body = await request.json()
            number = len(fake_calls)
            call = {"number": number, "started_at": time.time()}
            fake_calls.append(call)
            if args.synthetic_error_status is not None:
                call["completed_at"] = time.time()
                return web.json_response({"error": {"message": "Synthetic upstream unavailable",
                                         "type": "server_error"}}, status=args.synthetic_error_status)
            if number == 0:
                await asyncio.sleep(args.synthetic_delay)
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_delivery_probe", "type": "function", "function": {
                        "name": "bash", "arguments": json.dumps({
                            "command": "printf 'repair-stream-delivered\\n'",
                            "description": "Verify delayed response delivery"})}}]}
                reason = "tool_calls"
            else:
                message = {"role": "assistant", "content": "Transport probe complete."}
                reason = "stop"
            call["completed_at"] = time.time()
            return web.json_response({"id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion", "created": int(time.time()), "model": body["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
        fake_app = web.Application()
        fake_app.router.add_post("/v1/chat/completions", reply)
        fake_server = web.AppRunner(fake_app, access_log=None)
        await fake_server.setup()
        site = web.TCPSite(fake_server, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        settings.update(base_url=f"http://127.0.0.1:{port}/v1", models={"super": {"id": "repair-stream-probe"}})
        staged["env"]["NVIDIA_API_KEY"] = "synthetic-probe-no-hosted-access"
    manifest = campaign.create_campaign(args.output, staged["dataset"], settings, shards=1, task_limit=1, repeats=1)
    source = freeze_source(staged["source"], args.output)
    # stage() has already prepared the same application code on local disk.
    for original in (source / "scripts").glob("*deepswe*.py"):
        if original.read_bytes() != (args.scratch / "scripts" / original.name).read_bytes():
            raise ValueError("Smoke source differs from staged code; refresh local scripts first")
    staged.update(settings=settings, source=source)
    staged["env"]["ATP_DEEPSWE_SOURCE"] = str(source)
    os.environ.update(staged["env"])
    os.environ["ATP_DEEPSWE_CAMPAIGN"] = str(args.output)
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stopped.set)
    end = job_end_time(os.environ)
    now = time.time()
    verifier_timeout = manifest["tasks"][0].get("_atp_verifier_timeout", 3600)
    grace = settings["shutdown_grace_seconds"]
    # run_model reserves one grace interval before allocation end; dispatch
    # reserves another after the unchanged native verifier timeout.
    required = args.agent_timeout + verifier_timeout + 2 * grace
    diagnostic_budget = {"checked_at": now, "agent_timeout_seconds": args.agent_timeout,
        "native_verifier_timeout_seconds": verifier_timeout, "shutdown_grace_seconds": grace,
        "required_allocation_seconds": required, "allocation_end": end,
        "remaining_allocation_seconds": end - now, "fits": now + required < end}
    purpose = "synthetic_delivery_probe" if fake_server else "live_smoke_excluded_from_evaluation"
    save_json(args.output / "check-result.json", {"passed": False, "phase": "preflight",
              "purpose": purpose, "diagnostic_budget": diagnostic_budget})
    try:
        if not diagnostic_budget["fits"]:
            raise ValueError("Insufficient allocation time for diagnostic agent, native verifier, and shutdown budget")
        results = await asyncio.gather(*(run_model(campaign, staged, args.output, model, 0, end, stopped)
                                       for model in settings["models"]), return_exceptions=True)
    finally:
        if fake_server is not None:
            await fake_server.cleanup()
            save_json(args.output / "synthetic-upstream.json", {"calls": fake_calls,
                      "delay_seconds": args.synthetic_delay, "hosted_model_calls": 0})
    details = {}
    for model, result in zip(settings["models"], results):
        state = campaign.slot_state(args.output, model, 0, 0)
        directory = Path(state["attempt_dir"]) if state["attempt_dir"] else None
        export = campaign.read_json(directory / "export.json", {}) if directory else {}
        tools = [part for message in export.get("messages", []) for part in message.get("parts", [])
                 if part.get("type") == "tool"]
        details[model] = {"state": state, "tool_parts": len(tools),
                          "error": str(result) if isinstance(result, BaseException) else None}
    passed = all(item["state"]["status"] == "completed" and item["tool_parts"] > 0
                 and item["error"] is None for item in details.values())
    if fake_calls and args.synthetic_error_status is not None:
        directory = Path(details["super"]["state"]["attempt_dir"])
        invalidity = campaign.generation_failure(directory)
        passed = details["super"]["state"]["status"] != "completed" and invalidity == "infrastructure"
        details["super"]["expected_infrastructure_failure"] = invalidity
    elif fake_calls:
        passed &= len(fake_calls) >= 2 and (
            fake_calls[0].get("completed_at", 0) - fake_calls[0]["started_at"] >= args.synthetic_delay)
    save_json(args.output / "check-result.json", {"passed": passed, "models": details,
              "purpose": purpose, "diagnostic_budget": diagnostic_budget,
              "finished_at": time.time()})
    return int(not passed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent-timeout", type=int, default=600)
    parser.add_argument("--synthetic-delay", type=float,
                        help="Use a local fake model with this first-response delay; no hosted model calls")
    parser.add_argument("--synthetic-error-status", type=int, choices=(408, 503),
                        help="With --synthetic-delay 0, verify exhausted server errors cannot become valid grades")
    args = parser.parse_args()
    if args.synthetic_error_status and args.synthetic_delay != 0:
        parser.error("--synthetic-error-status requires --synthetic-delay 0")
    raise SystemExit(asyncio.run(main(args)))
