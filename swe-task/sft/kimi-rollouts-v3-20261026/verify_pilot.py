#!/usr/bin/env python3
"""kimi-rollouts-v3 (2026-10-26): verify one attempt of this batch against the previous Kimi rollout of the same task
(read-only; prints and writes analysis/verify-<campaign>-<slot>.json). Checks:
  agent-command.json  OpenCode config (OPENCODE_CONFIG_CONTENT): limit context = input = 262144, output = 131072,
                      agent.build.prompt = prompt/build-prompt.txt, and every other key equal to the previous attempt's
                      (headers aside: they carry per-attempt ATP ids)
  http.jsonl          first request: the system message contains the sentence exactly once, right before "You are
                      powered by the model named"; with the sentence removed it equals the previous Kimi system message
                      of the same task (the <env> date line normalised); tools and request parameters equal; base
                      (text before the <env> block) sha256 = prompt/PROMPT.json harness_base_new_expected; the
                      harness_system_sha256 (build_batch.py: sha256 of the JSON list of system messages)
  rate limits         x-ratelimit-* headers of every reply (first/last/min remaining), reply statuses, and the hub
                      tokens of this attempt per minute (prompt incl. cached + completion) over its call window
Never reads or prints request headers or credentials (http.jsonl holds none; reply headers are filtered to x-ratelimit-*).

    /usr/bin/python3 -I -B verify_pilot.py ATTEMPT_DIR [--previous PREVIOUS_ATTEMPT_DIR]
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import shlex  # noqa: E402

HERE = Path(__file__).resolve().parent
PROMPT = json.loads((HERE / "prompt" / "PROMPT.json").read_text())
SENTENCE = PROMPT["sentence"]
BUILD_PROMPT = (HERE / "prompt" / "build-prompt.txt").read_text()
MODEL_LINE = "You are powered by the model named"
ENV_MARK = "Here is some useful information about the environment you are running in:"
DATE = re.compile(r"Today's date: [A-Z][a-z]{2} [A-Z][a-z]{2} \d{2} \d{4}")


def sha(text):
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def sha_json(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def opencode_config(attempt: Path) -> dict:
    command = json.loads((attempt / "agent-command.json").read_text())["command"]
    at = command.index("OPENCODE_CONFIG_CONTENT=") + len("OPENCODE_CONFIG_CONTENT=")
    return json.loads(shlex.split(command[at:])[0])


def first_request(attempt: Path) -> dict:
    with open(attempt / "http.jsonl", "rb") as stream:
        for raw in stream:
            head = raw[:3000]
            if b'"event": "request"' in head or b'"event":"request"' in head:
                event = json.loads(raw)
                if event.get("event") == "request" and isinstance(event.get("payload"), dict):
                    return event["payload"]
    return {}


def replies(attempt: Path) -> dict:
    statuses = collections.Counter()
    limits = []
    calls = []
    errors = []
    with open(attempt / "http.jsonl", "rb") as stream:
        for raw in stream:
            head = raw[:3000]
            if b'"event": "response"' not in head and b'"event":"response"' not in head:
                continue
            event = json.loads(raw)
            statuses[str(event.get("status"))] += 1
            limits.append({"t": event.get("timestamp"), **{k.removeprefix("x-ratelimit-"): v for k, v in
                                                          (event.get("headers") or {}).items() if k.startswith("x-ratelimit-")}})
            if event.get("status") != 200:
                errors.append({"status": event.get("status"), "body": (event.get("body") or "")[:300]})
                continue
            try:
                body = json.loads(event.get("body") or "{}")
            except ValueError:
                continue
            usage = body.get("usage") or {}
            choice = (body.get("choices") or [{}])[0]
            calls.append({"t": event.get("timestamp"), "prompt": usage.get("prompt_tokens") or 0,
                          "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                          "completion": usage.get("completion_tokens") or 0, "cost": usage.get("cost"),
                          "finish": choice.get("finish_reason")})
    out = {"statuses": dict(statuses), "errors": errors[:10], "calls": len(calls)}
    if limits:
        numeric = collections.defaultdict(list)
        for row in limits:
            for k, v in row.items():
                if k != "t":
                    try:
                        numeric[k].append(float(v))
                    except (TypeError, ValueError):
                        pass
        out["ratelimit_first"] = {k: v for k, v in limits[0].items() if k != "t"}
        out["ratelimit_last"] = {k: v for k, v in limits[-1].items() if k != "t"}
        out["ratelimit_min"] = {k: min(v) for k, v in numeric.items()}
    if calls:
        window = max(c["t"] for c in calls) - min(c["t"] for c in calls)
        tokens = sum(c["prompt"] + c["completion"] for c in calls)
        out.update({"window_min": round(window / 60, 1), "prompt_tokens": sum(c["prompt"] for c in calls),
                    "cached_tokens": sum(c["cached"] for c in calls),
                    "completion_tokens": sum(c["completion"] for c in calls),
                    "max_prompt_tokens": max(c["prompt"] for c in calls),
                    "tokens_per_min": round(tokens / max(window / 60, 1e-9)) if window > 0 else None,
                    "calls_per_min": round(len(calls) / (window / 60), 2) if window > 0 else None,
                    "cost_reported": round(sum(c["cost"] for c in calls if isinstance(c["cost"], (int, float))), 4),
                    "finish_reasons": dict(collections.Counter(str(c["finish"]) for c in calls))})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--expect", choices=("new", "old"), default="new",
                        help="new = this batch (262144, sentence); old = the control ctrl-old (1048576, stock prompt)")
    args = parser.parse_args()
    attempt = args.attempt.resolve()
    report = {"attempt": str(attempt), "sentence": SENTENCE}
    config = opencode_config(attempt)
    limit = config["provider"]["nemo_gym"]["models"]["dummy_model"]["limit"]
    report["opencode_limit"] = limit
    report["expect"] = args.expect
    window = 262144 if args.expect == "new" else 1048576
    report["limit_ok"] = limit == {"context": window, "input": window, "output": 131072}
    if args.expect == "new":
        report["agent_build_prompt_ok"] = (config.get("agent") or {}).get("build", {}).get("prompt") == BUILD_PROMPT \
            and set(config["agent"]) == {"build"} and set(config["agent"]["build"]) == {"prompt"}
    else:
        report["no_agent_override_ok"] = "agent" not in config
    payload = first_request(attempt)
    system = [m.get("content") for m in payload.get("messages", []) if m.get("role") == "system"]
    text = system[0] if system and isinstance(system[0], str) else ""
    report["system_messages"] = len(system)
    report["sentence_count"] = text.count(SENTENCE)
    insert = SENTENCE + "\n\n" + MODEL_LINE
    report["sentence_right_before_model_line"] = text.count(insert) == 1
    base = text[:text.find(ENV_MARK)] if ENV_MARK in text else text
    report["harness_base_sha256"] = sha(base)
    report["harness_base_ok"] = sha(base) == PROMPT["harness_base_new_expected" if args.expect == "new" else "harness_base_old"]["sha256"]
    report["harness_system_sha256"] = sha_json(system)
    report["system_chars"] = len(text)
    report["tools_sha256"] = sha_json(payload.get("tools") or [])
    report["params"] = {k: v for k, v in payload.items() if k not in ("messages", "tools")}
    report["params_sha256"] = sha_json(report["params"])
    if args.previous:
        previous = args.previous.resolve()
        old_config = opencode_config(previous)
        for cfg in (config, old_config):
            cfg["provider"]["nemo_gym"]["options"].pop("headers", None)
            cfg["provider"]["nemo_gym"]["options"].pop("baseURL", None)
        changed = sorted(k for k in set(config) | set(old_config) if config.get(k) != old_config.get(k))
        old_limit = old_config["provider"]["nemo_gym"]["models"]["dummy_model"]["limit"]
        same_provider = (dict(config["provider"]["nemo_gym"], models=None) == dict(old_config["provider"]["nemo_gym"], models=None))
        report["vs_previous"] = {"previous_attempt": str(previous), "config_keys_changed": changed,
                                 "previous_limit": old_limit, "provider_options_equal": same_provider,
                                 "permission_equal": config.get("permission") == old_config.get("permission"),
                                 "tools_equal": config.get("tools") == old_config.get("tools")}
        old_payload = first_request(previous)
        old_system = [m.get("content") for m in old_payload.get("messages", []) if m.get("role") == "system"]
        old_text = old_system[0] if old_system else ""
        stripped = text.replace(SENTENCE + "\n\n", "", 1)
        report["vs_previous"].update({
            "system_equal_without_sentence": DATE.sub("DATE", stripped) == DATE.sub("DATE", old_text),
            "system_equal_without_sentence_and_date": True if DATE.sub("DATE", stripped) == DATE.sub("DATE", old_text) else
            sum(1 for a, b in zip(stripped, old_text) if a != b),
            "previous_harness_base_sha256": sha(old_text[:old_text.find(ENV_MARK)]),
            "tools_equal": (payload.get("tools") or []) == (old_payload.get("tools") or []),
            "params_previous": {k: v for k, v in old_payload.items() if k not in ("messages", "tools")},
            "params_equal": report["params"] == {k: v for k, v in old_payload.items() if k not in ("messages", "tools")},
            "first_user_equal": [m for m in payload.get("messages", []) if m.get("role") == "user"][:1] ==
            [m for m in old_payload.get("messages", []) if m.get("role") == "user"][:1]})
    report["replies"] = replies(attempt)
    out = HERE / "analysis" / f"verify-{attempt.parents[4].name}-{attempt.parent.name}-{attempt.name}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
