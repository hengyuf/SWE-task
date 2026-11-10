#!/usr/bin/env python3
"""kimi-rollouts-v4 (2026-10-28): verify one attempt of this batch (stock OpenCode prompt + 262,144 window; read-only on
the probes; prints and writes analysis/verify-<campaign>-<slot>-<attempt>.json). kimi-rollouts-v3's verify_pilot.py,
adapted: v4 must look exactly like the previous Kimi rollouts except for the window. Checks (each "*_ok" is a bool;
"all_ok" is their conjunction):
  agent-command.json  OpenCode config (OPENCODE_CONFIG_CONTENT): limit context = input = 262144, output = 131072, and
                      NO "agent" key (no agent.build.prompt override); with --previous every other key equal to the
                      previous attempt's (headers / baseURL aside: they carry per-attempt ATP ids)
  http.jsonl          first request: exactly one system message; it starts with prompt/opencode-1.17.11-default.txt
                      byte for byte and continues with "\nYou are powered by the model named" (nothing inserted); the
                      v3 sentence occurs 0 times; the base (text before the <env> block) sha256 = PROMPT.json
                      harness_base_old (the previous Kimi rollouts' base); with --previous the whole system message
                      equals the previous attempt's (the <env> date line normalised), tools / params / first user
                      message equal
  key / sandbox       the campaign's kr4-legs.jsonl (written by probe_worker.py at every ATP worker start) and
                      legs.jsonl: hub key NVAPI_KEY_NEW_2 (--expect-key), sandbox credential per taskset
                      (pilot / sel50 / rest47 -> CELL3_SANDBOX_API_KEY, baihe* -> BAIHE_SANDBOX_API_KEY), prompt stock,
                      context window 262144; worker.log's wrapper line names the same key; results metadata
                      (prompt "stock", context_window 262144, sandbox credential); for a Baihe task the agent sandbox
                      was created from the task's Baihe snapshot (task.toml docker_image "snapshot:<id>")
  grading             terminal.json evaluation_completed, a reward, grading/attempt-*/ctrf.json + reward.json, and the
                      results record graded pass / fail with its F2P / P2P counts
  spend               the leg's NVAPI_KEY_NEW_2 spend at start / end (results budget_start / budget_end) vs the
                      attempt's LiteLLM-reported cost (the key's spend must have grown at least by that much)
  rate limits         x-ratelimit-* headers of every reply, reply statuses, hub tokens per minute (as in v3)
Never reads or prints request headers or credentials (http.jsonl holds none; reply headers are filtered to
x-ratelimit-*).

    /usr/bin/python3 -I -B verify_pilot.py ATTEMPT_DIR [--previous PREVIOUS_ATTEMPT_DIR] [--expect-key NVAPI_KEY_NEW_2]
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
SENTENCE = PROMPT["sentence"]  # v3's added sentence: must NOT appear in v4
STOCK = (HERE / "prompt" / "opencode-1.17.11-default.txt").read_text()
STOCK_SHA256 = "962fbf3cb3ec659c9a5244425ee2e7bb141ad4428f489a630a7738566880dc6a"
MODEL_LINE = "You are powered by the model named"
ENV_MARK = "Here is some useful information about the environment you are running in:"
DATE = re.compile(r"Today's date: [A-Z][a-z]{2} [A-Z][a-z]{2} \d{2} \d{4}")
WINDOW = 262144
BAIHE_TSV = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/baihe-snapshot-access-20261026/baihe_v2_tasks.tsv")
SANDBOX_CREDENTIAL = {"pilot": "CELL3_SANDBOX_API_KEY", "sel50": "CELL3_SANDBOX_API_KEY", "rest47": "CELL3_SANDBOX_API_KEY",
                      "baihe-pilot": "BAIHE_SANDBOX_API_KEY", "baihe": "BAIHE_SANDBOX_API_KEY"}


def sha(text):
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def sha_json(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


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


def campaign_of(attempt: Path) -> Path:
    return attempt.parents[4]  # probes/kimi/<campaign>/models/kimi/attempts/<task>-<repeat>/attempt-<n>


def keys_and_sandbox(attempt: Path, expect_key: str) -> dict:
    campaign = campaign_of(attempt)
    name = campaign.name
    base = re.sub(r"-(rec|brk)\d+$", "", name)
    taskset = base if base in SANDBOX_CREDENTIAL else base.rsplit("-g", 1)[0]
    expected_credential = SANDBOX_CREDENTIAL.get(taskset)  # None: not a kimi-rollouts-v4 campaign
    path = campaign / "kr4-legs.jsonl"
    legs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    started = read_json(attempt / "started.json", {}) or {}
    job = str(started.get("job_id") or "")
    leg = next((entry for entry in legs if str(entry.get("job")) == job), legs[-1] if legs else {})
    driver_legs = [json.loads(line) for line in (HERE / "legs.jsonl").read_text().splitlines()
                   if line.strip()] if (HERE / "legs.jsonl").exists() else []
    starts = [entry for entry in driver_legs if entry.get("event") == "start" and entry.get("campaign") == base]
    worker_log = (campaign / "worker.log").read_text(errors="replace") if (campaign / "worker.log").exists() else ""
    wrapper_lines = [line for line in worker_log.splitlines() if "probe_worker wrapper: hub key" in line]
    results = read_json(HERE / "results" / f"kimi-{base}.json", {}) or {}
    out = {"campaign": name, "taskset": taskset, "attempt_job": job, "kr4_leg": leg, "driver_leg_starts": [
        {k: entry.get(k) for k in ("at", "job", "hub_key", "sandbox_credential", "prompt", "context_window")}
        for entry in starts], "worker_log_wrapper_lines": wrapper_lines[-3:],
        "expected_key": expect_key, "expected_sandbox_credential": expected_credential}
    out["hub_key_ok"] = bool(legs) and all(entry.get("hub_key") == expect_key for entry in legs) and \
        bool(starts) and all(entry.get("hub_key") == expect_key for entry in starts) and \
        bool(wrapper_lines) and all(f"hub key {expect_key}," in line for line in wrapper_lines)
    out["sandbox_credential_ok"] = bool(legs) and all(entry.get("sandbox_credential") == expected_credential
                                                      for entry in legs) and \
        all(entry.get("sandbox_credential") == expected_credential for entry in starts) and \
        (not results or results.get("sandbox_credential") == expected_credential)
    out["prompt_meta_ok"] = bool(legs) and all(entry.get("prompt") == "stock" and entry.get("context_window") == WINDOW
                                               and entry.get("system_prompt_sha256") == STOCK_SHA256 for entry in legs) \
        and all(entry.get("prompt") == "stock" and entry.get("context_window") == WINDOW for entry in starts)
    if results:
        out["results_meta"] = {k: results.get(k) for k in ("prompt", "context_window", "sandbox_credential",
                                                           "system_prompt", "hub_keys_by_leg")}
        out["results_meta_ok"] = results.get("prompt") == "stock" and results.get("context_window") == WINDOW and \
            (results.get("system_prompt") or {}).get("sha256") == STOCK_SHA256
        start, end = results.get("budget_start") or {}, results.get("budget_end") or {}
        if start.get("key") == end.get("key") and start.get("spend") is not None and end.get("spend") is not None:
            out["leg_key_spend"] = {"key": start["key"], "start": start["spend"], "end": end["spend"],
                                    "delta": round(end["spend"] - start["spend"], 4)}
    # Baihe: the task's image is a Baihe snapshot; an agent sandbox created from it means the Baihe credential worked
    request = started.get("request") or {}
    task_dir = campaign / "tasks" / str(request.get("task_id"))
    image = None
    if (task_dir / "task.toml").exists():  # /usr/bin/python3 has no tomllib: the one docker_image line
        found = re.search(r'(?m)^\s*docker_image\s*=\s*"([^"]*)"', (task_dir / "task.toml").read_text())
        image = found.group(1) if found else None
    sandbox = read_json(attempt / "agent-sandbox.json", {}) or {}
    out["task_image"] = image
    out["agent_sandbox_created"] = bool((sandbox.get("descriptor") or {}).get("sandbox_id"))
    if expected_credential == "BAIHE_SANDBOX_API_KEY":  # Baihe's snapshots 404 under hengyuf's credential (10-24)
        rows = [line.split("\t") for line in BAIHE_TSV.read_text().splitlines()[1:] if line.strip()]
        snapshot = {row[0]: row[2] for row in rows}.get(str(request.get("task_id")))
        out["baihe_snapshot"] = snapshot
        out["baihe_snapshot_sandbox_ok"] = bool(snapshot) and image == f"snapshot:{snapshot}" and \
            out["agent_sandbox_created"]
    return out


def grading(attempt: Path) -> dict:
    terminal = read_json(attempt / "terminal.json", {}) or {}
    gradings = sorted(attempt.glob("grading/attempt-*"))
    last = gradings[-1] if gradings else None
    name = campaign_of(attempt).name
    results = read_json(HERE / "results" / f"kimi-{re.sub(r'-(rec|brk)[0-9]+$', '', name)}.json", {}) or {}
    record = None
    for slot_records in (results.get("records") or {}).values():
        for candidate in slot_records:
            if candidate.get("attempt_dir") and Path(candidate["attempt_dir"]).resolve() == attempt.resolve():
                record = candidate
    out = {"evaluation_completed": terminal.get("evaluation_completed"), "reward": terminal.get("reward"),
           "f2p": f"{terminal.get('f2p_passed')}/{terminal.get('f2p_total')}",
           "p2p": f"{terminal.get('p2p_passed')}/{terminal.get('p2p_total')}",
           "verifier_error": terminal.get("verifier_error"),
           "grading_dirs": [p.name for p in gradings],
           "ctrf": bool(last and (last / "ctrf.json").exists()), "reward_json": bool(last and (last / "reward.json").exists()),
           "results_record": record}
    out["grading_ok"] = out["evaluation_completed"] is True and out["reward"] is not None and out["ctrf"] and \
        out["reward_json"] and bool(record) and record.get("status") in ("pass", "fail")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--previous", type=Path, help="a previous Kimi (1M, stock prompt) attempt of the same task")
    parser.add_argument("--expect-key", default="NVAPI_KEY_NEW_2")
    args = parser.parse_args()
    attempt = args.attempt.resolve()
    report = {"attempt": str(attempt)}
    config = opencode_config(attempt)
    limit = config["provider"]["nemo_gym"]["models"]["dummy_model"]["limit"]
    report["opencode_limit"] = limit
    report["limit_ok"] = limit == {"context": WINDOW, "input": WINDOW, "output": 131072}
    report["no_agent_override_ok"] = "agent" not in config
    report["config_keys"] = sorted(config)
    payload = first_request(attempt)
    system = [m.get("content") for m in payload.get("messages", []) if m.get("role") == "system"]
    text = system[0] if system and isinstance(system[0], str) else ""
    report["system_messages"] = len(system)
    report["stock_prompt_file_sha256_ok"] = sha(STOCK) == STOCK_SHA256
    report["system_starts_with_stock_prompt"] = text.startswith(STOCK)
    report["after_stock_prompt"] = text[len(STOCK):len(STOCK) + 60] if text.startswith(STOCK) else None
    report["nothing_inserted_after_stock_prompt"] = text.startswith(STOCK) and \
        text[len(STOCK):].startswith("\n" + MODEL_LINE)
    report["v3_sentence_count"] = text.count(SENTENCE)
    base = text[:text.find(ENV_MARK)] if ENV_MARK in text else text
    report["harness_base_sha256"] = sha(base)
    report["harness_base_ok"] = sha(base) == PROMPT["harness_base_old"]["sha256"]
    report["system_prompt_ok"] = len(system) == 1 and report["stock_prompt_file_sha256_ok"] and \
        report["nothing_inserted_after_stock_prompt"] and report["v3_sentence_count"] == 0 and report["harness_base_ok"]
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
        old_payload = first_request(previous)
        old_system = [m.get("content") for m in old_payload.get("messages", []) if m.get("role") == "system"]
        old_text = old_system[0] if old_system else ""
        report["vs_previous"] = {
            "previous_attempt": str(previous), "config_keys_changed": changed, "previous_limit": old_limit,
            "provider_options_equal": same_provider,
            "permission_equal": config.get("permission") == old_config.get("permission"),
            "tools_config_equal": config.get("tools") == old_config.get("tools"),
            "system_equal": DATE.sub("DATE", text) == DATE.sub("DATE", old_text),
            "previous_harness_base_sha256": sha(old_text[:old_text.find(ENV_MARK)]),
            "tools_equal": (payload.get("tools") or []) == (old_payload.get("tools") or []),
            "params_previous": {k: v for k, v in old_payload.items() if k not in ("messages", "tools")},
            "params_equal": report["params"] == {k: v for k, v in old_payload.items() if k not in ("messages", "tools")},
            "first_user_equal": [m for m in payload.get("messages", []) if m.get("role") == "user"][:1] ==
            [m for m in old_payload.get("messages", []) if m.get("role") == "user"][:1]}
        v = report["vs_previous"]
        report["vs_previous_ok"] = changed in ([], ["provider"]) and same_provider and v["permission_equal"] and \
            v["tools_config_equal"] and v["system_equal"] and v["tools_equal"] and v["params_equal"] and \
            v["first_user_equal"]
    report["keys_and_sandbox"] = keys_and_sandbox(attempt, args.expect_key)
    report["grading"] = grading(attempt)
    report["replies"] = replies(attempt)
    spend = report["keys_and_sandbox"].get("leg_key_spend")
    if spend and report["replies"].get("cost_reported") is not None:
        report["spend_ok"] = spend["key"] == args.expect_key and spend["delta"] + 0.01 >= report["replies"]["cost_reported"]
    checks = {k: v for k, v in report.items() if k.endswith("_ok") and isinstance(v, bool)}
    checks.update({k: v for k, v in report["keys_and_sandbox"].items() if k.endswith("_ok") and isinstance(v, bool)})
    checks["grading_ok"] = report["grading"]["grading_ok"]
    report["checks"] = checks
    report["all_ok"] = all(checks.values())
    out = HERE / "analysis" / f"verify-{campaign_of(attempt).name}-{attempt.parent.name}-{attempt.name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
