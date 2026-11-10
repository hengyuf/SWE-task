#!/usr/bin/env python3
"""Replay real hub GLM-5.3 requests (from an ATP attempt's http.jsonl) against OUR OpenAI-compatible endpoint and
compare the reply with what the hub returned for the same payload. Stdlib only (login-node python 3.9 is fine).

    # dry run (default): list the calls that would be sent, nothing is sent anywhere
    /usr/bin/python3 -I -B replay_hub_requests.py --attempt <ATTEMPT_DIR> [--calls first,median,largest,compaction,other]
    # send to the local server (refuses any NVIDIA hub URL)
    /usr/bin/python3 -I -B replay_hub_requests.py --attempt <ATTEMPT_DIR> --base-url http://<ip>:8000/v1 --send \
        [--model glm-5.3] [--max-tokens N] [--repeat 2] [--timeout 3600] [--out results.jsonl]

What it checks per call (one JSON line each):
  * HTTP status; first 300 chars of a non-200 body (e.g. vLLM <= 0.31: 400 "maximum context length" when
    prompt + max_tokens > max_model_len; ATP makes any unresolved 400 a FATAL slot);
  * usage.prompt_tokens local vs hub for the identical payload: equal means same chat template / tokenizer /
    reasoning-history rendering as the hub (the hub kept history reasoning: a large negative delta means the server
    dropped it, e.g. reasoning_content not mapped to the template);
  * reasoning field name (vLLM: "reasoning"; hub: "reasoning_content"; Gym reads both) and length;
  * tool calls: names, JSON-parsable arguments, argument JSON types (hub: ints for bash.timeout / read.offset /
    read.limit, list for todowrite.todos), and chat-template markers leaked into content (<tool_call>, <arg_key>, ...);
  * finish_reason, completion tokens, wall latency; with --repeat 2 the second send's cached prompt tokens
    (prefix caching; vLLM reports them only with --enable-prompt-tokens-details).
The payload is sent exactly as the hub got it (messages incl. assistant reasoning_content, tools, tool_choice auto,
temperature 1.0, top_p 0.95, reasoning_effort max, max_tokens 131072), except "model" (and max_tokens with
--max-tokens). Never prints message contents or credentials.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

LEAK = ("<tool_call>", "<arg_key>", "<arg_value>", "</think>", "<think>", "<|observation|>", "<|user|>",
        "<|assistant|>", "[PAD]")
HUB_MARKERS = ("nvidia.com", "inference-api", "integrate.api", "build.nvidia")
COMPACTION_SYS = "You are an anchored context summarization assistant"


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def load_calls(attempt):
    """[(index, payload, hub_reply, kind)] of the attempt's status-200 calls in journal order."""
    requests, calls = {}, []
    with open(Path(attempt) / "http.jsonl", "rb") as stream:
        for raw in stream:
            head = raw[:2500]
            if b'"event":"request"' in head or b'"event": "request"' in head:
                event = json.loads(raw)
                requests[(event["call_id"], event.get("http_attempt"))] = event.get("payload") or {}
            elif b'"event":"response"' in head or b'"event": "response"' in head:
                event = json.loads(raw)
                payload = requests.pop((event["call_id"], event.get("http_attempt")), None)
                if event.get("status") != 200 or payload is None:
                    continue
                try:
                    reply = json.loads(event.get("body") or "{}")
                except ValueError:
                    continue
                messages = payload.get("messages") or []
                system = _text(messages[0].get("content")) if messages and messages[0].get("role") == "system" else ""
                kind = ("compaction" if system.startswith(COMPACTION_SYS) else
                        "main" if system.startswith("You are opencode") else "other")
                calls.append((len(calls), payload, reply, kind))
    return calls


def pick(calls, which):
    chosen = {}
    main = [c for c in calls if c[3] == "main"]
    usage = lambda c: (c[2].get("usage") or {}).get("prompt_tokens") or 0
    for name in which:
        if name == "first" and main:
            chosen[name] = main[0]
        elif name == "median" and main:
            chosen[name] = sorted(main, key=usage)[len(main) // 2]
        elif name == "largest" and main:
            chosen[name] = max(main, key=usage)
        elif name == "compaction":
            found = [c for c in calls if c[3] == "compaction"]
            if found:
                chosen[name] = found[0]
        elif name == "other":
            found = [c for c in calls if c[3] == "other"]
            if found:
                chosen[name] = found[0]
        elif name.isdigit() and int(name) < len(calls):
            chosen[name] = calls[int(name)]
    return chosen


def summarize_reply(reply):
    usage = reply.get("usage") or {}
    choice = (reply.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    content = content if isinstance(content, str) else json.dumps(content)
    reasoning_key = ("reasoning_content" if message.get("reasoning_content") else
                     "reasoning" if message.get("reasoning") else None)
    tools, arg_types, bad = [], {}, 0
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        tools.append(function.get("name"))
        try:
            args = json.loads(function.get("arguments") or "{}")
            for key, value in (args.items() if isinstance(args, dict) else []):
                arg_types[f"{function.get('name')}.{key}"] = type(value).__name__
        except ValueError:
            bad += 1
    return {"prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "finish_reason": choice.get("finish_reason"), "reasoning_key": reasoning_key,
            "reasoning_chars": len(message.get(reasoning_key) or "") if reasoning_key else 0,
            "content_chars": len(content), "tool_calls": tools, "tool_arg_types": arg_types,
            "unparsable_tool_args": bad, "content_leak_markers": [s for s in LEAK if s in content]}


def send(base_url, payload, timeout):
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
                                     method="POST")
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return response.status, body, time.time() - started
    except urllib.error.HTTPError as error:
        return error.code, error.read(), time.time() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--attempt", required=True, help="ATP attempt dir with http.jsonl (a hub GLM-5.3 rollout)")
    parser.add_argument("--calls", default="first,median,largest,compaction,other")
    parser.add_argument("--base-url", help="OUR endpoint, e.g. http://10.1.2.3:8000/v1")
    parser.add_argument("--model", default="glm-5.3", help="served model name")
    parser.add_argument("--max-tokens", type=int, help="override max_tokens (default: the hub payload's 131072)")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--send", action="store_true", help="actually send (default: dry run)")
    parser.add_argument("--out", help="also append the JSON lines to this file")
    args = parser.parse_args()
    if args.send:
        if not args.base_url:
            parser.error("--send needs --base-url")
        host = urllib.parse.urlsplit(args.base_url).hostname or ""
        if any(marker in args.base_url for marker in HUB_MARKERS) or not host:
            parser.error("refusing: --base-url must be our own endpoint, never the NVIDIA hub")
    calls = load_calls(args.attempt)
    chosen = pick(calls, [name.strip() for name in args.calls.split(",") if name.strip()])
    sink = open(args.out, "a") if args.out else None
    for name, (index, payload, hub_reply, kind) in chosen.items():
        body = dict(payload, model=args.model)
        if args.max_tokens:
            body["max_tokens"] = args.max_tokens
        hub = summarize_reply(hub_reply)
        record = {"pick": name, "call_index": index, "kind": kind, "messages": len(payload.get("messages") or []),
                  "tools": len(payload.get("tools") or []), "max_tokens": body.get("max_tokens"),
                  "payload_bytes": len(json.dumps(body)), "params": sorted(k for k in body if k not in ("messages", "tools")),
                  "hub": hub}
        if not args.send:
            print(json.dumps(dict(record, sent=False)))
            continue
        for attempt in range(args.repeat):
            status, raw, latency = send(args.base_url, body, args.timeout)
            result = dict(record, sent=True, repeat=attempt, status=status, latency_s=round(latency, 2))
            if status == 200:
                local = summarize_reply(json.loads(raw))
                result["local"] = local
                if local["prompt_tokens"] is not None and hub["prompt_tokens"] is not None:
                    result["prompt_tokens_delta_local_minus_hub"] = local["prompt_tokens"] - hub["prompt_tokens"]
            else:
                result["error_body"] = raw.decode("utf-8", "replace")[:300]
            line = json.dumps(result)
            print(line, flush=True)
            if sink:
                sink.write(line + "\n")
                sink.flush()
    if sink:
        sink.close()


if __name__ == "__main__":
    sys.exit(main())
