#!/usr/bin/env python3
"""teacher-deepswe (2026-10-30): verify one attempt of this campaign (any model: kimi / glm / deepseek; stock OpenCode
prompt + 262,144 window; read-only on the probes; prints and writes analysis/verify-<model>-<campaign>-<slot>-<attempt>.json).
kimi-rollouts-v4's verify_pilot.py, adapted: the model comes from the attempt path (probes/<model>/<campaign>/...), the
hub key must be the leg's key (NVAPI_KEY_NEW_2 for legs started before 2026-10-30 21:45 = the three pilot legs,
NVAPI_KEY_NEW_3 from then on: OPERATOR_NOTE 21:45, NVAPI_KEY_NEW_2 again from 2026-10-31 18:15: OPERATOR_NOTE 18:15)
and the sandbox credential CELL3_SANDBOX_API_KEY for every campaign (the Baihe branch is
gone), the ledger is tdw-legs.jsonl, and a request-parameter check is added:
  params              the first request's parameters: model = the alias's hub id, max_tokens 131072, temperature 1.0,
                      top_p 0.95, reasoning_effort max, streamed for kimi only (ATP_DEEPSWE_UPSTREAM_STREAM)
  spend               glm / deepseek replies carry no LiteLLM cost: their cost is priced with prices.json
kimi-rollouts-v4 checks follow (each "*_ok" is a bool; "all_ok" is their conjunction):
  agent-command.json  OpenCode config (OPENCODE_CONFIG_CONTENT): limit context = input = 262144, output = 131072, and
                      NO "agent" key (no agent.build.prompt override); with --previous every other key equal to the
                      previous attempt's (headers / baseURL aside: they carry per-attempt ATP ids)
  http.jsonl          first request: exactly one system message; it starts with prompt/opencode-1.17.11-default.txt
                      byte for byte and continues with "\nYou are powered by the model named" (nothing inserted); the
                      v3 sentence occurs 0 times; the base (text before the <env> block) sha256 = PROMPT.json
                      harness_base_old (the previous Kimi rollouts' base); with --previous the whole system message
                      equals the previous attempt's (the <env> date line normalised), tools / params / first user
                      message equal
  key / sandbox       the campaign's tdw-legs.jsonl (written by probe_worker.py at every ATP worker start) and
                      legs.jsonl: every leg's hub key = the key of its start time (NEW_2 before the 10-30 21:45
                      switch, NEW_3 until 10-31 18:15, NEW_2 after; --expect-key also pins the attempt's own leg),
                      sandbox credential CELL3_SANDBOX_API_KEY,
                      model alias / id, prompt stock, context window 262144; worker.log's wrapper line names the same
                      key; results metadata (model, prompt "stock", context_window 262144, sandbox credential); the
                      agent sandbox was created (agent-sandbox.json) with its ATP owner labels
  grading             terminal.json evaluation_completed, a reward, grading/attempt-*/ctrf.json + reward.json, and the
                      results record graded pass / fail with its F2P / P2P counts
  spend               the leg key's spend at start / end (results budget_start / budget_end) vs the
                      attempt's LiteLLM-reported cost (the key's spend must have grown at least by that much)
  rate limits         x-ratelimit-* headers of every reply, reply statuses, hub tokens per minute (as in v3)
Never reads or prints request headers or credentials (http.jsonl holds none; reply headers are filtered to
x-ratelimit-*). DeepSWE pilots: run without --previous (no earlier stock-prompt attempt of the same DeepSWE task exists
in hengyuf's campaigns; a different task's attempt differs in its <env> block, which fails vs_previous_ok).

Pilot-phase additions (2026-10-30, the workflow's pilot stage; all read-only on recorded artifacts):
  journal_all         EVERY request of http.jsonl (not only the first): model id of the alias, max_tokens 131072,
                      temperature 1.0, top_p 0.95, reasoning_effort max, streamed for kimi only; every system message
                      starts with the stock prompt + "\n" + the model line, without the v3 sentence; every request /
                      response / API-gate event carries ATP's credential id "existing" (NVIDIA_API_KEY, which
                      synth/probe_worker binds to api_keys()[0] = the wrapper's leg key), credential-binding.json
                      says "existing" and S/atp/.api (ATP's optional extra-key pool) does not exist (existence only)
  image               the campaign's task copy (tasks/<id>/: task.toml, instruction.md, tests/*) is byte-identical to the
                      pinned DeepSWE task dir and its task.toml sha256 = tasks/waves.json's; its docker_image is a
                      public.ecr.aws image = the snapshot's; request.json, started.json, terminal.json and Gym's own
                      rollout record (launch dir rollouts.jsonl) carry that image; terminal / Gym's sandbox_handle = the
                      registered agent sandbox (Gym's resources server creates every sandbox with image=task_image(task)
                      from that task store and refuses a request image that differs: resources_servers/deepswe/app.py)
  grading_detail      the graded patch is the committed diff: model.patch sha256 / bytes = terminal's
                      model_patch_sha256 / _bytes = the candidate_saved event size; the agent sandbox's collect command is
                      the task's [[verifier.collect]] command (git diff --binary <base> HEAD); a separate verifier-phase
                      sandbox was registered; apply_failed false; terminal / reward.json F2P and P2P totals = the task's
                      tests/config.json f2p_node_ids / p2p_node_ids counts, passed counts present, reward equal
  cost                the attempt's hub cost (kimi: LiteLLM usage.cost; glm / deepseek: prices.json) is reported
                      ("cost_usd"); spend_ok as before (the leg key's spend delta >= that cost)

    /usr/bin/python3 -I -B verify_pilot.py ATTEMPT_DIR [--previous PREVIOUS_ATTEMPT_DIR] [--expect-key KEY_NAME]
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
import datetime  # noqa: E402
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
SANDBOX_CREDENTIAL = "CELL3_SANDBOX_API_KEY"
MODEL_IDS = {"kimi": "nvidia/moonshotai/kimi-k3", "glm": "nvidia/zai-org/glm-5.3",
             "deepseek": "nvidia/deepseek-ai/deepseek-v4-pro"}
PRICES = {alias: fit["per_token"] for alias, fit in json.loads((HERE / "prices.json").read_text()).items()
          if alias in ("glm", "deepseek")}
WAVES = json.loads((HERE / "tasks" / "waves.json").read_text())
SNAPSHOT = {task["task_id"]: task for task in WAVES["tasks"]}
ATP_APP = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp")
ATP_CREDENTIAL = "existing"  # ATP's single hosted credential name (deepswe_credentials.CREDENTIAL_ENV: NVIDIA_API_KEY)
TASK_FILES = ("task.toml", "instruction.md", "tests/test.sh", "tests/grader.py", "tests/config.json", "tests/test.patch")
KEY_SWITCH_AT = "2026-10-30T21:45:00-0700"  # OPERATOR_NOTE 21:45: NVAPI_KEY_NEW_2 -> NVAPI_KEY_NEW_3 for every new leg
KEY_SWITCH2_AT = "2026-10-31T18:15:00-0700"  # OPERATOR_NOTE 2026-10-31 18:15: NVAPI_KEY_NEW_3 -> NVAPI_KEY_NEW_2


def key_at(at) -> str:
    """The campaign's hub key for a leg started at `at` (the ledgers' %Y-%m-%dT%H:%M:%S%z; driver.KEY_PERIODS)."""
    fmt = "%Y-%m-%dT%H:%M:%S%z"
    try:
        when = datetime.datetime.strptime(str(at), fmt)
    except ValueError:
        return None
    if when < datetime.datetime.strptime(KEY_SWITCH_AT, fmt):
        return "NVAPI_KEY_NEW_2"
    return "NVAPI_KEY_NEW_3" if when < datetime.datetime.strptime(KEY_SWITCH2_AT, fmt) else "NVAPI_KEY_NEW_2"


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


def replies(attempt: Path, model: str = "kimi") -> dict:
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
            prompt = usage.get("prompt_tokens") or 0
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            completion = usage.get("completion_tokens") or 0
            cost = usage.get("cost")
            if not isinstance(cost, (int, float)) and model in PRICES:  # glm / deepseek: no LiteLLM cost in the body
                a, b, c = PRICES[model]
                cost = a * (prompt - cached) + b * cached + c * completion
            calls.append({"t": event.get("timestamp"), "prompt": prompt, "cached": cached, "completion": completion,
                          "cost": cost, "finish": choice.get("finish_reason")})
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
    return attempt.parents[4]  # probes/<model>/<campaign>/models/<model>/attempts/<task>-<repeat>/attempt-<n>


def model_of(attempt: Path) -> str:
    return attempt.parents[2].name


def keys_and_sandbox(attempt: Path, expect_key) -> dict:
    campaign = campaign_of(attempt)
    model = model_of(attempt)
    name = campaign.name
    base = re.sub(r"-(rec|brk)\d+$", "", name)
    path = campaign / "tdw-legs.jsonl"
    legs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    started = read_json(attempt / "started.json", {}) or {}
    job = str(started.get("job_id") or "")
    leg = next((entry for entry in legs if str(entry.get("job")) == job), legs[-1] if legs else {})
    driver_legs = [json.loads(line) for line in (HERE / "legs.jsonl").read_text().splitlines()
                   if line.strip()] if (HERE / "legs.jsonl").exists() else []
    starts = [entry for entry in driver_legs if entry.get("event") == "start" and entry.get("campaign") == base
              and entry.get("model") == model]
    worker_log = (campaign / "worker.log").read_text(errors="replace") if (campaign / "worker.log").exists() else ""
    wrapper_lines = [line for line in worker_log.splitlines() if "probe_worker wrapper: model" in line]
    results = read_json(HERE / "results" / f"{model}-{base}.json", {}) or {}
    out = {"model": model, "campaign": name, "attempt_job": job, "tdw_leg": leg, "driver_leg_starts": [
        {k: entry.get(k) for k in ("at", "job", "hub_key", "sandbox_credential", "prompt", "context_window", "model_id")}
        for entry in starts], "worker_log_wrapper_lines": wrapper_lines[-3:],
        "expected_sandbox_credential": SANDBOX_CREDENTIAL}
    leg_key = key_at(leg.get("at")) if leg else None
    out["expected_key"] = expect_key or leg_key
    out["leg_keys"] = sorted({str(entry.get("hub_key")) for entry in legs})
    out["hub_key_ok"] = bool(legs) and all(entry.get("hub_key") == key_at(entry.get("at")) for entry in legs) and \
        bool(starts) and all(entry.get("hub_key") == key_at(entry.get("at")) for entry in starts) and \
        leg.get("hub_key") == out["expected_key"] and \
        bool(wrapper_lines) and all(any(f"hub key {key}," in line for key in out["leg_keys"]) for line in wrapper_lines)
    out["sandbox_credential_ok"] = bool(legs) and all(entry.get("sandbox_credential") == SANDBOX_CREDENTIAL
                                                      for entry in legs) and \
        all(entry.get("sandbox_credential") == SANDBOX_CREDENTIAL for entry in starts) and \
        (not results or results.get("sandbox_credential") == SANDBOX_CREDENTIAL)
    out["model_meta_ok"] = bool(legs) and all(entry.get("model") == model and entry.get("model_id") == MODEL_IDS[model]
                                              for entry in legs) and \
        all(entry.get("model_id") == MODEL_IDS[model] for entry in starts) and \
        (not results or (results.get("model") == model and results.get("model_id") == MODEL_IDS[model]))
    out["prompt_meta_ok"] = bool(legs) and all(entry.get("prompt") == "stock" and entry.get("context_window") == WINDOW
                                               and entry.get("system_prompt_sha256") == STOCK_SHA256 for entry in legs) \
        and all(entry.get("prompt") == "stock" and entry.get("context_window") == WINDOW for entry in starts)
    if results:
        out["results_meta"] = {k: results.get(k) for k in ("model", "model_id", "prompt", "context_window",
                                                           "sandbox_credential", "system_prompt", "hub_keys_by_leg")}
        out["results_meta_ok"] = results.get("prompt") == "stock" and results.get("context_window") == WINDOW and \
            (results.get("system_prompt") or {}).get("sha256") == STOCK_SHA256
        start, end = results.get("budget_start") or {}, results.get("budget_end") or {}
        if start.get("key") == end.get("key") and start.get("spend") is not None and end.get("spend") is not None:
            out["leg_key_spend"] = {"key": start["key"], "start": start["spend"], "end": end["spend"],
                                    "delta": round(end["spend"] - start["spend"], 4)}
    request = started.get("request") or {}
    task_dir = campaign / "tasks" / str(request.get("task_id"))
    image = None
    if (task_dir / "task.toml").exists():  # /usr/bin/python3 has no tomllib: the one docker_image line
        found = re.search(r'(?m)^\s*docker_image\s*=\s*"([^"]*)"', (task_dir / "task.toml").read_text())
        image = found.group(1) if found else None
    sandbox = read_json(attempt / "agent-sandbox.json", {}) or {}
    out["task_image"] = image
    out["agent_sandbox_created"] = bool((sandbox.get("descriptor") or {}).get("sandbox_id"))
    out["agent_sandbox_owner"] = sandbox.get("owner")
    return out


def grading(attempt: Path) -> dict:
    terminal = read_json(attempt / "terminal.json", {}) or {}
    gradings = sorted(attempt.glob("grading/attempt-*"))
    last = gradings[-1] if gradings else None
    name = campaign_of(attempt).name
    results = read_json(HERE / "results" / f"{model_of(attempt)}-{re.sub(r'-(rec|brk)[0-9]+$', '', name)}.json", {}) or {}
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



def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def journal_all(attempt: Path, model: str) -> dict:
    """Every request / response / API-gate event of http.jsonl: request parameters, system message, credential id."""
    combos = collections.Counter()
    systems = collections.Counter()
    credentials = collections.Counter()
    others = []
    requests = 0
    with open(attempt / "http.jsonl", "rb") as stream:
        for raw in stream:
            event = json.loads(raw)
            kind = event.get("event")
            if kind in ("request", "response", "api_gate_wait_started", "api_gate_acquired"):
                credentials[f"{kind}:{event.get('credential_id')}"] += 1
            if kind != "request" or not isinstance(event.get("payload"), dict):
                continue
            payload = event["payload"]
            requests += 1
            combos[json.dumps({k: payload.get(k) for k in ("model", "max_tokens", "temperature", "top_p",
                                                           "reasoning_effort", "stream")}, sort_keys=True)] += 1
            system = [m.get("content") for m in payload.get("messages", []) if m.get("role") == "system"]
            text = system[0] if system and isinstance(system[0], str) else ""
            stock = len(system) == 1 and text.startswith(STOCK + "\n" + MODEL_LINE) and SENTENCE not in text
            systems["stock" if stock else "other"] += 1
            if not stock and len(others) < 5:
                others.append({"call_id": event.get("call_id"), "system_messages": len(system),
                               "head": text[:160], "sha256": sha(text)})
    combo_list = [dict(json.loads(key), requests=n) for key, n in combos.items()]
    out = {"requests": requests, "param_combos": combo_list, "system_messages": dict(systems),
           "non_stock_examples": others, "credential_events": dict(credentials)}
    out["params_all_ok"] = requests > 0 and all(
        c["model"] == MODEL_IDS[model] and c["max_tokens"] == 131072 and c["temperature"] == 1.0
        and c["top_p"] == 0.95 and c["reasoning_effort"] == "max" and bool(c["stream"]) == (model == "kimi")
        for c in combo_list)
    out["system_prompt_all_ok"] = requests > 0 and systems.get("other", 0) == 0
    binding = read_json(attempt / "credential-binding.json", {}) or {}
    out["credential_binding"] = binding.get("credential_id")
    out["atp_extra_key_pool_absent"] = not (ATP_APP / ".api").exists()
    out["credential_ok"] = bool(credentials) and all(key.endswith(":" + ATP_CREDENTIAL) for key in credentials) \
        and binding.get("credential_id") == ATP_CREDENTIAL and out["atp_extra_key_pool_absent"]
    return out


def _toml_value(text: str, key: str):
    found = re.search(r'(?m)^\s*' + re.escape(key) + r'\s*=\s*"((?:[^"\\]|\\.)*)"', text)
    return found.group(1).encode().decode("unicode_escape") if found else None


def gym_rollout(launch_dir: Path, sandbox_id: str, attempt: Path):
    """Gym's own rollout record of this attempt (launch dir rollouts.jsonl): image and sandbox_handle."""
    path = launch_dir / "rollouts.jsonl"
    if not sandbox_id or not path.exists():
        return None
    with open(path, "rb") as stream:
        for raw in stream:
            if sandbox_id.encode() not in raw:
                continue
            row = json.loads(raw)
            if row.get("sandbox_handle") == sandbox_id:
                return {"image": row.get("image"), "sandbox_handle": row.get("sandbox_handle"),
                        "task_id": row.get("task_id"), "reward": row.get("reward")}
    return None


def image_and_grading(attempt: Path) -> dict:
    """The agent sandbox's image provenance and the grading details (see the module docstring)."""
    request = read_json(attempt / "request.json", {}) or {}
    started = read_json(attempt / "started.json", {}) or {}
    terminal = read_json(attempt / "terminal.json", {}) or {}
    task_id = request.get("task_id")
    snap = SNAPSHOT.get(task_id) or {}
    source_dir = Path(snap.get("task_dir", "/nonexistent"))
    task_copy = campaign_of(attempt) / "tasks" / str(task_id)
    files = {}
    for rel in TASK_FILES:
        files[rel] = {"campaign": sha_file(task_copy / rel) if (task_copy / rel).is_file() else None,
                      "source": sha_file(source_dir / rel) if (source_dir / rel).is_file() else None}
    toml_text = (task_copy / "task.toml").read_text() if (task_copy / "task.toml").is_file() else ""
    image = _toml_value(toml_text, "docker_image")
    out = {"task_id": task_id, "task_files": files,
           "task_files_ok": all(v["campaign"] and v["campaign"] == v["source"] for v in files.values())
           and files["task.toml"]["campaign"] == snap.get("task_toml_sha256")}
    agent_sandbox = read_json(attempt / "agent-sandbox.json", {}) or {}
    agent_id = (agent_sandbox.get("descriptor") or {}).get("sandbox_id")
    events = []
    if (attempt / "capture-events.jsonl").exists():
        events = [json.loads(line) for line in (attempt / "capture-events.jsonl").read_text().splitlines() if line.strip()]
    registered = [{"phase": e.get("phase"), "sandbox_id": (e.get("descriptor") or {}).get("sandbox_id")}
                  for e in events if e.get("event") == "sandbox_registered"]
    launch_dir = Path(started.get("launch_dir") or "/nonexistent")
    rollout = gym_rollout(launch_dir, agent_id, attempt)
    out["image"] = {"task_toml": image, "snapshot": snap.get("image"), "request": request.get("image"),
                    "started_request": (started.get("request") or {}).get("image"), "terminal": terminal.get("image"),
                    "gym_rollout": rollout, "agent_sandbox_id": agent_id,
                    "terminal_sandbox_handle": terminal.get("sandbox_handle"), "registered_sandboxes": registered}
    out["image_ok"] = bool(image) and image.startswith("public.ecr.aws/") and image == snap.get("image") \
        and request.get("image") == image and (started.get("request") or {}).get("image") == image \
        and terminal.get("image") == image and bool(agent_id) and terminal.get("sandbox_handle") == agent_id \
        and {"phase": "agent", "sandbox_id": agent_id} in registered \
        and rollout is not None and rollout["image"] == image and rollout["sandbox_handle"] == agent_id
    gradings = sorted(attempt.glob("grading/attempt-*"))
    last = gradings[-1] if gradings else None
    reward_json = read_json(last / "reward.json", {}) if last else {}
    reward_json = reward_json or {}
    verifier_sandbox = read_json(last / "verifier-sandbox.json", {}) if last else {}
    verifier_id = ((verifier_sandbox or {}).get("descriptor") or {}).get("sandbox_id")
    config = read_json(task_copy / "tests" / "config.json", {}) or {}
    f2p_n, p2p_n = len(config.get("f2p_node_ids") or []), len(config.get("p2p_node_ids") or [])
    patch = attempt / "model.patch"
    patch_bytes = patch.stat().st_size if patch.exists() else None
    candidates = [e.get("size") for e in events if e.get("event") == "candidate_saved"]
    collect = re.search(r'(?ms)^\[\[verifier\.collect\]\]\s*\ncommand\s*=\s*"((?:[^"\\]|\\.)*)"', toml_text)
    collect_command = collect.group(1).encode().decode("unicode_escape") if collect else None
    out["grading_detail"] = {
        "config_f2p_node_ids": f2p_n, "config_p2p_node_ids": p2p_n,
        "terminal": {k: terminal.get(k) for k in ("reward", "f2p_passed", "f2p_total", "p2p_passed", "p2p_total",
                                                  "apply_failed", "verifier_exit_code", "evaluation_completed",
                                                  "model_patch_bytes", "model_patch_sha256")},
        "reward_json": reward_json, "model_patch_bytes": patch_bytes,
        "model_patch_sha256": sha_file(patch) if patch.exists() else None, "candidate_saved_sizes": candidates,
        "collect_command_task": collect_command, "collect_command_agent_sandbox": agent_sandbox.get("collect_command"),
        "verifier_sandbox_id": verifier_id}
    g = out["grading_detail"]
    out["grading_counts_ok"] = f2p_n > 0 and p2p_n >= 0 and terminal.get("f2p_total") == f2p_n \
        and terminal.get("p2p_total") == p2p_n and reward_json.get("f2p_total") == f2p_n \
        and reward_json.get("p2p_total") == p2p_n and terminal.get("f2p_passed") is not None \
        and terminal.get("p2p_passed") is not None and reward_json.get("f2p_passed") == terminal.get("f2p_passed") \
        and reward_json.get("p2p_passed") == terminal.get("p2p_passed") \
        and reward_json.get("reward") is not None and float(reward_json["reward"]) == float(terminal.get("reward"))
    out["committed_diff_ok"] = patch.exists() and g["model_patch_sha256"] == terminal.get("model_patch_sha256") \
        and patch_bytes == terminal.get("model_patch_bytes") and bool(candidates) and candidates[-1] == patch_bytes \
        and bool(collect_command) and agent_sandbox.get("collect_command") == collect_command \
        and terminal.get("apply_failed") is False \
        and bool(verifier_id) and verifier_id != agent_id \
        and {"phase": "verifier", "sandbox_id": verifier_id} in registered
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--previous", type=Path, help="an earlier stock-prompt attempt (e.g. KR4 / TR2) to compare the "
                                                     "OpenCode config, system prompt, tools and params with")
    parser.add_argument("--expect-key", default=None, help="the attempt's leg key (default: the key of its start time: "
                                                            "NVAPI_KEY_NEW_2 before 2026-10-30 21:45, NEW_3 until "
                                                            "2026-10-31 18:15, NEW_2 after)")
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
    model = model_of(attempt)
    params = report["params"]
    report["params_ok"] = (params.get("model") == MODEL_IDS[model] and params.get("max_tokens") == 131072
                           and params.get("temperature") == 1.0 and params.get("top_p") == 0.95
                           and params.get("reasoning_effort") == "max"
                           and bool(params.get("stream")) == (model == "kimi"))
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
    report["replies"] = replies(attempt, model)
    report["cost_usd"] = report["replies"].get("cost_reported")
    report["cost_source"] = "LiteLLM usage.cost" if model not in PRICES else "prices.json fit x journal usage"
    spend = report["keys_and_sandbox"].get("leg_key_spend")
    if spend and report["replies"].get("cost_reported") is not None:
        report["spend_ok"] = spend["key"] == report["keys_and_sandbox"]["expected_key"] and \
            spend["delta"] + 0.01 >= report["replies"]["cost_reported"]
    report["journal_all"] = journal_all(attempt, model)
    report["image_and_grading"] = image_and_grading(attempt)
    checks = {k: v for k, v in report.items() if k.endswith("_ok") and isinstance(v, bool)}
    checks.update({k: v for k, v in report["keys_and_sandbox"].items() if k.endswith("_ok") and isinstance(v, bool)})
    checks.update({k: v for k, v in report["journal_all"].items() if k.endswith("_ok") and isinstance(v, bool)})
    checks.update({k: v for k, v in report["image_and_grading"].items() if k.endswith("_ok") and isinstance(v, bool)})
    checks["grading_ok"] = report["grading"]["grading_ok"]
    report["checks"] = checks
    report["all_ok"] = all(checks.values())
    out = HERE / "analysis" / f"verify-{model}-{campaign_of(attempt).name}-{attempt.parent.name}-{attempt.name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
