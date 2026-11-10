#!/usr/bin/env python3
"""teacher-rollouts-v2 (2026-10-25): tiny hub checks of the two new teachers with the probe harness's sampling.

    /usr/bin/python3 -I -B hub_smoke.py [--models glm deepseek]

Per model, three small non-streaming /chat/completions calls (ATP sends glm/deepseek non-streaming:
run_deepswe ATP_DEEPSWE_UPSTREAM_STREAM=0 for every alias but kimi) with exactly the harness's request fields:
max_tokens 131072, temperature 1.0, top_p 0.95, reasoning_effort "max" (+ tools / tool_choice "auto"):
  plain     "What is 17*23?" (no tools)
  tool      one bash tool; the model should call it
  history   the tool turn replayed as the harness does (assistant content + tool_calls + reasoning_content, then
            the tool result): the hub must accept reasoning_content in the history
Records status, latency, usage (incl. LiteLLM usage.cost if present), the x-litellm-response-cost header,
reasoning presence/length, content and tool calls in hub_smoke.json. The key is read BY NAME (the pipeline's
active hub key, synth/hub_key_active.txt; must be NVAPI_KEY_OFFICIAL_2) and never printed.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

sys.path.insert(0, "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth")
from common import SETTINGS, USER_ENV, env_value, hub_key_name  # noqa: E402  (stdlib-only module)

TR = Path(__file__).resolve().parent
MODELS = dict(SETTINGS["models"], deepseek={"id": "nvidia/deepseek-ai/deepseek-v4-pro",
                                            "sampling_overrides": {"reasoning_effort": "max"}})
BASH = {"type": "function", "function": {
    "name": "bash", "description": "Executes a given bash command and returns its output.",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "string", "description": "The command to execute"},
        "description": {"type": "string", "description": "Clear, concise description of what this command does"}},
        "required": ["command", "description"]}}}
HEADERS = ("x-litellm-response-cost", "x-litellm-key-spend", "x-litellm-key-max-budget", "x-litellm-model-id",
           "x-litellm-model-group", "x-litellm-attempted-retries", "x-request-id", "request-id")


def call(key, body):
    request = urllib.request.Request(SETTINGS["base_url"].rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            status, headers, text = response.status, response.headers, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        status, headers, text = error.code, error.headers, error.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as error:
        return {"status": None, "error": f"{type(error).__name__}: {error}"[:500], "elapsed_s": time.time() - started}
    text = re.sub(r"sk-[A-Za-z0-9_.-]+", "<redacted>", text.replace(key, "<key>"))
    out = {"status": status, "elapsed_s": round(time.time() - started, 2),
           "headers": {name: headers.get(name) for name in HEADERS if headers.get(name) is not None}}
    try:
        payload = json.loads(text)
    except ValueError:
        out["error"] = text[:1000]
        return out
    if status != 200:
        out["error"] = payload
        return out
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    out.update(finish_reason=choice.get("finish_reason"), usage=payload.get("usage"), model=payload.get("model"),
               message_keys=sorted(message), reasoning_chars=len(reasoning) if isinstance(reasoning, str) else None,
               reasoning_head=reasoning[:400] if isinstance(reasoning, str) else repr(reasoning)[:400],
               content=(message.get("content") or "")[:400],
               content_has_think_tag="<think>" in (message.get("content") or ""),
               tool_calls=[{"id": call.get("id"), "name": (call.get("function") or {}).get("name"),
                            "arguments": (call.get("function") or {}).get("arguments")}
                           for call in message.get("tool_calls") or []],
               _message=message)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["glm", "deepseek"])
    args = parser.parse_args()
    name = hub_key_name()
    if name != "NVAPI_KEY_OFFICIAL_2":
        print(json.dumps({"error": f"active hub key is {name}, expected NVAPI_KEY_OFFICIAL_2; not calling"}))
        return 2
    key = env_value(USER_ENV, name)
    if not key:
        print(json.dumps({"error": f"no {name} in the credentials file"}))
        return 2
    results = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "key": name, "models": {}}
    for alias in args.models:
        spec = MODELS[alias]
        base = {"model": spec["id"], "max_tokens": SETTINGS["max_output_tokens"], "temperature": SETTINGS["temperature"],
                "top_p": SETTINGS["top_p"], **spec.get("sampling_overrides", {})}
        runs = {}
        runs["plain"] = call(key, dict(base, messages=[
            {"role": "user", "content": "What is 17*23? Answer with just the number."}]))
        user = {"role": "user", "content": "Use the bash tool to print the current working directory, then tell me "
                                           "what it is in one short sentence."}
        runs["tool"] = call(key, dict(base, messages=[user], tools=[BASH], tool_choice="auto"))
        message = runs["tool"].get("_message") or {}
        if message.get("tool_calls"):
            first = message["tool_calls"][0]
            assistant = {"role": "assistant", "content": message.get("content") or "",
                         "tool_calls": [{"id": first.get("id") or "call_0", "type": "function",
                                         "function": first.get("function")}]}
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            if reasoning:
                assistant["reasoning_content"] = reasoning
            runs["history"] = call(key, dict(base, tools=[BASH], tool_choice="auto", messages=[
                user, assistant, {"role": "tool", "tool_call_id": first.get("id") or "call_0", "content": "/app\n"}]))
            runs["history"]["sent_reasoning_content"] = bool(reasoning)
        for run in runs.values():
            run.pop("_message", None)
        results["models"][alias] = {"id": spec["id"], "request": {k: v for k, v in base.items() if k != "model"},
                                    "calls": runs}
    out = TR / "hub_smoke.json"
    out.write_text(json.dumps(results, indent=1) + "\n")
    for alias, entry in results["models"].items():
        for kind, run in entry["calls"].items():
            usage = run.get("usage") or {}
            print(alias, kind, run.get("status"), f"{run.get('elapsed_s')}s", run.get("finish_reason"),
                  "reasoning", run.get("reasoning_chars"), "tools", [c["name"] for c in run.get("tool_calls") or []],
                  "usage", {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "cost")},
                  "reasoning_tokens", (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                  "hdr_cost", (run.get("headers") or {}).get("x-litellm-response-cost"),
                  "err", json.dumps(run.get("error"))[:300] if run.get("error") else "")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
