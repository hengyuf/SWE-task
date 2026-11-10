#!/usr/bin/env python3
"""teacher-deepswe (2026-10-30): the progress report PROGRESS.md (script-generated; kimi-rollouts-v4's progress.py
generalised to 3 models x 2 waves). Each run rewrites PROGRESS.md:
  1. header: the request, the configuration, the hub-key policy, the start time, the total (1,695 = 113 tasks x 3 x 5);
  2. a progress table that gains ONE row per run (rows kept in PROGRESS-rows.json; a run within --min-gap-min minutes of
     the previous row replaces it);
  3. per model x wave: slots, terminal / graded, pass@1 three ways, infra, broken-ended, output-cap runaways, re-runs,
     compaction, in flight, legs, estimated hub cost, ETA;
  4. per campaign; 5. cost (per model; the hub keys NVAPI_KEY_NEW_2 / NVAPI_KEY_NEW_3 by leg period) and the projection.
Hub key (OPERATOR_NOTE 2026-10-31 18:15, user decision relayed by main): NVAPI_KEY_NEW_2 for every leg started from
2026-10-31 18:15 (legs start at >= $1,700 left, stop below $1,500: NEW_2 also carries the live r6 RL run's rubric judge).
Before: NVAPI_KEY_NEW_3 for the legs of 2026-10-30 21:45 .. the 10-31 10:30 floor stop (floors $1,200 / $1,000), and
NVAPI_KEY_NEW_2 for the three pilot legs of 21:34.
Read-only on probes/ and results/ (no ATP code is called: ATP's slot_state writes validity caches into attempt dirs).
Writes only PROGRESS.md, PROGRESS-rows.json and progress-cache.json (facts of finished attempts, which never change),
plus key_budget.py's budget-log.jsonl line (one tiny NEW_2 call; --no-budget skips it). Slurm: one squeue (--no-squeue
skips it).

Definitions (per original slot of the 15 wave campaigns; the 3 pilots are reported in the header only):
  graded         a finished attempt with evaluation_completed (and no recorded generation failure); the slot's first
                 such attempt, as ATP's slot_state; for a completed campaign its results file is used instead
  terminal       graded + final infra (results files); "awaiting" = ended without a grade in a running campaign
  pass@1 raw     passes / terminal (infra counts as not passed), first attempts
  excl. infra    passes / graded (the usual pass@1), first attempts
  excl. broken   passes / graded slots that were not broken-ended
  after re-runs  passes / resolved slots, each broken-ended slot replaced by its graded re-run (pending ones left out)
  broken-ended   the graded attempt's session was ENDED by a broken reply (brokenreply.py, degenerate.py's definition)
  runaway        the session ended on a reply without a tool call that used the whole output budget (flag only)
  agent timeout  ended_by_agent_timeout: ATP's 10,800 s agent timeout ended the graded attempt (its
                 generation-validity-v2-r7.json "budget_timeout"; see agent_timeout()). OPERATOR_NOTE 2026-10-31 02:15:
                 such slots stay as graded (no re-run); pass@1 is reported with them and without them
  excl. timeout  passes / graded first attempts that did not end on the agent timeout
  compacted      graded rollouts whose OpenCode session compacted at least once
  in flight      started attempts without an ending, in campaigns whose tdw- job is RUNNING
  est. cost      OpenCode export token counts (main session) of every ended attempt, priced per model: glm / deepseek
                 prices.json (LiteLLM's per-token prices), kimi analysis/kimi-price-fit.json (fit to KR4's LiteLLM
                 costs); checked against journal costs: 98.4% (glm), 100.0% (deepseek), 98.8% (kimi) of the real cost
  key spend      the current key's (KEY) spend now - its spend at the first tdw leg start on it from KEY_FROM (rows
                 before 2026-10-31 18:15: NVAPI_KEY_NEW_3 since its first leg). NOT exact for this campaign: the r6 RL
                 run's rubric judge spends on the same key; est. cost is the campaign's own figure. Before the first
                 leg on the key the column shows its remaining budget only.
  ETA            remaining / rate: finished per hour over the last 3 h once >= 10 finished there; else, for a running
                 wave, the attempts in flight now (ATP's live concurrency, at least 1 per running campaign) / median
                 rollout hours; else the campaign cap / prior rollout hours (KR4 / TR2 agent wall medians + grading);
                 a model's wave 2 starts when its wave 1 ends, or now if it already runs (window-16 patch,
                 2026-11-02: before it, a running wave also used the manifest cap)

    /usr/bin/python3 -I -B progress.py [--no-budget] [--no-squeue] [--min-gap-min 10] [--root DIR]
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
import datetime  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

HERE = Path(__file__).resolve().parent
MODELS = ("kimi", "glm", "deepseek")
MODEL_IDS = {"kimi": "nvidia/moonshotai/kimi-k3", "glm": "nvidia/zai-org/glm-5.3",
             "deepseek": "nvidia/deepseek-ai/deepseek-v4-pro"}
WAVES = {"w1": 3, "w2": 2}
REPEATS = 5
CAP = {"kimi": 8, "glm": 8, "deepseek": 5}  # submit.sh defaults (a campaign's settings.json wins once it exists)
TOTAL = 113 * 3 * REPEATS
KEY = "NVAPI_KEY_NEW_2"   # every leg from 2026-10-31 18:15 (OPERATOR_NOTE 18:15)
KEY_FROM = "2026-10-31T18:15:00-0700"  # the switch NEW_3 -> NEW_2 (ledger times, %Y-%m-%dT%H:%M:%S%z)
START_FLOOR = 1700.0
GUARD_FLOOR = 1500.0
OLD_KEY = "NVAPI_KEY_NEW_3"   # the wave legs of 2026-10-30 21:45 .. the 10-31 10:30 floor stop (floors $1,200 / $1,000)
PILOT_KEY = "NVAPI_KEY_NEW_2"  # the three pilot legs started at 21:34 (they finished on it)
OUTPUT_CAP = 131072
PRIOR_HOURS = {"kimi": 1.1, "glm": 0.8, "deepseek": 0.6}  # KR4 57.8 / TR2 35.8 / 26.0 min agent wall + grading
PRIOR_BROKEN = {"kimi": 66 / 970, "glm": 0.0, "deepseek": 0.0}  # KR4 first-attempt broken-ended rate; TR2: none
REQUEST = ("我想在deepSWE 每一个task 上额外用kimi,Deepseek, GLM (Deepseek: nvidia/deepseek-ai/deepseek-v4-pro GLM: "
           "nvidia/zai-org/glm-5.3 ), 每个模型rollout 5遍. 用一个subagent 来收集他们的rollouts")
ROOT = HERE  # --root (tests only): where probes/, results/, reruns/, legs.jsonl and the outputs live


def _load(name):
    spec = importlib.util.spec_from_file_location(f"tdw_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BROKEN = _load("brokenreply")


def paused_by_main(model, campaign):
    """True while babysit/<model>-<campaign>/STOP is main's pause note (OPERATOR_NOTE 2026-11-01 05:15 PST)."""
    try:
        note = (ROOT / "babysit" / f"{model}-{campaign}" / "STOP").read_text()
    except OSError:
        return False
    return note.startswith("main ") and "paused" in note


PAUSE_LABEL = "paused by main (r6 judge 429s)"
CANCEL_LABEL = "cancelled by the user (OPERATOR_NOTE 2026-11-02 ~02:00 PST)"


def cancelled_by_user(model, campaign):
    """True while babysit/<model>-<campaign>/STOP is main's cancel note (OPERATOR_NOTE 2026-11-02 ~02:00 PST: the
    user cancelled the GLM teacher rollouts; the STOP files stay for good)."""
    try:
        note = (ROOT / "babysit" / f"{model}-{campaign}" / "STOP").read_text()
    except OSError:
        return False
    return "main" in note and "user cancelled" in note


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def prices() -> dict:
    fits = json.loads((HERE / "prices.json").read_text())
    out = {alias: fits[alias]["per_token"] for alias in ("glm", "deepseek")}
    out["kimi"] = json.loads((HERE / "analysis" / "kimi-price-fit.json").read_text())["per_token"]
    return out


def campaigns() -> dict:
    """{(model, campaign): {model, wave, tasks}} for the 15 wave campaigns (driver.py's task order and grouping)."""
    data = json.loads((HERE / "tasks" / "waves.json").read_text())
    out = {}
    for model in MODELS:
        for wave, groups in WAVES.items():
            ids = data["wave1"] if wave == "w1" else data["wave2"]
            for group in range(groups):
                out[(model, f"{wave}-g{group}")] = {"model": model, "wave": wave, "tasks": ids[group::groups]}
    return out


# ---------------------------------------------------------------- per-attempt facts (cached once finished)
def export_facts(attempt: Path, price) -> dict:
    export = read_json(attempt / "export.json", {}) or {}
    cost = 0.0
    calls = 0
    compactions = 0
    prompt_max = 0
    for message in export.get("messages") or []:
        info = message.get("info") or {}
        if info.get("role") != "assistant":
            continue
        if info.get("mode") == "compaction" or info.get("summary") is True:
            compactions += 1
        tokens = info.get("tokens") or {}
        if not tokens:
            continue
        cache = tokens.get("cache") or {}
        cached = cache.get("read") or 0
        uncached = (tokens.get("input") or 0) + (cache.get("write") or 0)
        completion = (tokens.get("output") or 0) + (tokens.get("reasoning") or 0)
        cost += price[0] * uncached + price[1] * cached + price[2] * completion
        calls += 1
        prompt_max = max(prompt_max, uncached + cached)
    return {"export": bool(export), "est_cost": round(cost, 4), "calls": calls, "compactions": compactions,
            "prompt_max": prompt_max}


def agent_timeout(attempt) -> bool:
    """True when ATP's 10,800 s agent timeout ended the attempt (OPERATOR_NOTE 2026-10-31 02:15: ended_by_agent_timeout).
    Source: the attempt's generation-validity-v2-r7.json "budget_timeout" (ATP deepswe_campaign.py: a timeout
    interruption or a TimeoutError journal error at / after started.json's deadline = start + the agent timeout); an
    attempt without that file: terminal.json opencode_run_error_type "TimeoutError" and terminal.json written at / after
    the deadline - 1 s."""
    attempt = Path(attempt)
    validity = read_json(attempt / "generation-validity-v2-r7.json")
    if isinstance(validity, dict) and "budget_timeout" in validity:
        return bool(validity["budget_timeout"])
    terminal = attempt / "terminal.json"
    try:
        end = terminal.stat().st_mtime
    except OSError:
        return False
    deadline = (read_json(attempt / "started.json", {}) or {}).get("deadline")
    return ((read_json(terminal, {}) or {}).get("opencode_run_error_type") == "TimeoutError" and deadline is not None
            and end >= deadline - 1)


def attempt_facts(attempt: Path, cache: dict, price) -> dict:
    """{ended, completed, status, reward, broken, kind, runaway, agent_timeout, compactions, duration_s, est_cost} of
    one attempt."""
    terminal = attempt / "terminal.json"
    validity = attempt / "generation-validity-v2-r7.json"
    markers = ("failure.json", "interrupted.json", "dispatch-failure.json")
    try:
        signature = [terminal.stat().st_mtime, validity.stat().st_mtime if validity.exists() else None]
    except OSError:
        ended = any((attempt / name).exists() for name in markers)
        if not ended:
            return {"ended": False, "completed": False, "started": (attempt / "started.json").exists()}
        key = str(attempt) + "#ended"
        entry = cache.get(key)
        if not entry:  # an attempt that ended without a terminal.json (infra): its export may still carry tokens
            entry = {"ended": True, "completed": False, "started": True, **export_facts(attempt, price)}
            cache[key] = entry
        return entry
    key = str(attempt)
    entry = cache.get(key)
    if entry and entry.get("signature") == signature and "agent_timeout" in entry:
        return entry
    result = read_json(terminal, {}) or {}
    failure = (read_json(validity, {}) or {}).get("failure") if validity.exists() else None
    completed = result.get("evaluation_completed") is True and failure is None
    entry = {"signature": signature, "ended": True, "started": True, "completed": completed,
             "reward": result.get("reward"),
             "status": ("pass" if result.get("reward") == 1 else "fail") if completed else None,
             **export_facts(attempt, price)}
    started = (read_json(attempt / "started.json", {}) or {}).get("started_at")
    entry["duration_s"] = round(signature[0] - started, 1) if started else None
    entry["finished_at"] = signature[0]
    entry["agent_timeout"] = agent_timeout(attempt)
    if completed:
        verdict = BROKEN.broken_end(attempt)
        detail = BROKEN.last_reply_detail(attempt)
        entry.update(broken=verdict["ended"], kind=verdict["kind"],
                     runaway=bool(detail) and not detail.get("tool") and (
                         (detail.get("completion_tokens") or 0) >= OUTPUT_CAP or detail.get("finish") == "length"))
    cache[key] = entry
    return entry


def slot_view(slot_dir: Path, cache: dict, price) -> dict:
    """ATP-like view of one slot from its attempt dirs: graded (first completed attempt) | active | ended | pending."""
    attempts = sorted((p for p in slot_dir.glob("attempt-*") if p.is_dir()), key=lambda p: int(p.name.split("-")[1]))
    if not attempts:
        return {"state": "pending"}
    for attempt in attempts:
        facts = attempt_facts(attempt, cache, price)
        if facts.get("completed"):
            return {"state": "graded", "attempt": str(attempt), **facts}
    facts = attempt_facts(attempts[-1], cache, price)
    if facts.get("ended"):
        return {"state": "ended", "attempt": str(attempts[-1])}
    return {"state": "active" if facts.get("started") else "pending", "attempt": str(attempts[-1])}


def facts_of(attempt_dir, cache: dict, price) -> dict:
    return attempt_facts(Path(attempt_dir), cache, price) if attempt_dir else {}


def scan(probe_dir: Path, model: str, cache: dict, price) -> dict:
    """{(task_id, slot): view} of a campaign dir (empty before its manifest exists)."""
    manifest = read_json(probe_dir / "manifest.json")
    if not manifest:
        return {}
    out = {}
    for task in manifest["tasks"]:
        for repeat in range(manifest["settings"]["repeats"]):
            slot_dir = probe_dir / "models" / model / "attempts" / f"{task['_ng_task_index']}-{repeat}"
            out[(task["task_id"], f"{model}-r{repeat}")] = slot_view(slot_dir, cache, price)
    return out


def model_cost(model: str, cache: dict, price) -> dict:
    """Estimated hub cost of every ended attempt of a model (all its campaigns: pilot, waves, rec, brk)."""
    total = 0.0
    attempts = 0
    timeouts = 0
    for attempt in (ROOT / "probes" / model).glob(f"*/models/{model}/attempts/*/attempt-*"):
        facts = attempt_facts(attempt, cache, price)
        if facts.get("ended"):
            total += facts.get("est_cost") or 0.0
            attempts += 1
            timeouts += bool(facts.get("agent_timeout"))
    return {"est_cost": round(total, 2), "ended_attempts": attempts, "agent_timeout_attempts": timeouts}


# ---------------------------------------------------------------- Slurm / budget
def squeue():
    try:
        out = subprocess.run(["squeue", "--noheader", "-u", os.environ.get("USER", "hengyuf"),
                              "--format=%i|%j|%T|%M|%P"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    jobs = collections.defaultdict(list)
    for line in out.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 5 and parts[1].startswith("tdw-"):
            jobs[parts[1][4:]].append({"id": parts[0], "state": parts[2], "time": parts[3], "part": parts[4]})
    return jobs


def key_budget():
    try:
        out = subprocess.run(["/usr/bin/python3", "-I", "-B", str(HERE / "key_budget.py"), "--key-name", KEY,
                              "--tag", "progress"], capture_output=True, text=True, timeout=300)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError, OSError, subprocess.SubprocessError):
        return {"key": KEY, "remaining": None}


def after(at, ref) -> bool:
    """True when the ledger time `at` (%Y-%m-%dT%H:%M:%S%z) is at or after `ref`."""
    fmt = "%Y-%m-%dT%H:%M:%S%z"
    try:
        return datetime.datetime.strptime(str(at), fmt) >= datetime.datetime.strptime(ref, fmt)
    except ValueError:
        return False


def old_key_spend(starts) -> tuple:
    """(spent, last reading) of OLD_KEY: its spend at the last budget-log.jsonl reading - its spend at the first
    OLD_KEY leg start (includes other users of the key, e.g. the r6 judge); (None, None) without readings."""
    first = next(((leg.get("budget") or {}).get("spend") for leg in starts if leg.get("hub_key") == OLD_KEY
                  and (leg.get("budget") or {}).get("spend") is not None), None)
    last = None
    path = HERE / "budget-log.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("key") == OLD_KEY and entry.get("spend") is not None:
                last = entry
    if first is None or last is None:
        return None, None
    return round(last["spend"] - first, 2), last


def legs():
    path = ROOT / "legs.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


# ---------------------------------------------------------------- one campaign
def campaign_status(model: str, name: str, info: dict, cache: dict, jobs, leg_rows: list, price) -> dict:
    probe = ROOT / "probes" / model / name
    result = read_json(ROOT / "results" / f"{model}-{name}.json")
    settings = read_json(probe / "settings.json", {}) or {}
    # GLM wave 2 is created at 4 (OPERATOR_NOTE 2026-10-31 02:15 / 2026-11-01 11:15 PST); settings.json wins once it exists
    cap = settings.get("max_concurrency_per_model") or (4 if (model, info["wave"]) == ("glm", "w2") else CAP[model])
    mine = (jobs or {}).get(f"{model}-{name}", [])
    running = any(j["state"] == "RUNNING" for j in mine)
    row = {"model": model, "campaign": name, "wave": info["wave"], "slots": len(info["tasks"]) * REPEATS, "cap": cap,
           "graded": 0, "pass": 0, "broken": 0, "broken_pass": 0, "clean_graded": 0, "clean_pass": 0, "runaway": 0,
           "infra_final": 0, "ended_ungraded": 0, "active": 0, "in_flight": 0, "compacted": 0, "graded_rollouts": 0,
           "timeout": 0, "timeout_pass": 0,
           "rerun_planned": 0, "rerun_done": 0, "rerun_pending": 0,
           "rerun_outcomes": collections.Counter(), "after_graded": 0, "after_pass": 0, "finished_times": [],
           "durations": [], "result": bool(result), "running": running, "queued": bool(mine) and not running,
           "jobs": [f"{j['id']}:{j['state']}:{j['time']}:{j['part']}" for j in mine],
           "legs": sum(1 for leg in leg_rows if leg.get("event") == "start" and leg.get("campaign") == name
                       and leg.get("model") == model),
           "started": probe.exists(),
           "recovery_campaigns": sorted(p.name for p in (ROOT / "probes" / model).glob(f"{name}-*"))
           if (ROOT / "probes" / model).exists() else []}

    def add_rollout(facts):
        if facts.get("completed"):
            row["graded_rollouts"] += 1
            row["compacted"] += bool(facts.get("compactions"))
            if facts.get("finished_at"):
                row["finished_times"].append(facts["finished_at"])
            if facts.get("duration_s"):
                row["durations"].append(facts["duration_s"])

    if result:  # first attempts (originals; infra-recovered slots with their recovery rollout)
        first = result.get("first_attempt_records") or result.get("records") or {}
        for task_id, records in first.items():
            for record in records:
                facts = facts_of(record.get("attempt_dir"), cache, price)
                if record["status"] in ("pass", "fail"):
                    row["graded"] += 1
                    row["pass"] += record["status"] == "pass"
                    broken = bool(record.get("ended_by_broken_reply"))
                    row["broken"] += broken
                    row["broken_pass"] += broken and record["status"] == "pass"
                    row["runaway"] += bool(record.get("ended_by_output_cap_runaway"))
                    timed_out = bool(record.get("ended_by_agent_timeout")) or bool(facts.get("agent_timeout"))
                    row["timeout"] += timed_out
                    row["timeout_pass"] += timed_out and record["status"] == "pass"
                    if not broken:
                        row["clean_graded"] += 1
                        row["clean_pass"] += record["status"] == "pass"
                        row["after_graded"] += 1
                        row["after_pass"] += record["status"] == "pass"
                    add_rollout(facts if facts.get("completed") else {"completed": True})
                else:
                    row["infra_final"] += 1
        for task_id, entries in (result.get("broken_reply_reruns") or {}).items():
            for entry in entries:
                row["rerun_planned"] += 1
                row["rerun_done"] += 1
                row["rerun_outcomes"][entry["outcome"]] += 1
                facts = facts_of(entry.get("rerun_attempt"), cache, price)
                row["after_graded"] += 1
                if entry["outcome"] == "infra":  # the original stays
                    row["after_pass"] += entry["original_status"] == "pass"
                else:
                    row["after_pass"] += entry["rerun_status"] == "pass"
                    add_rollout(facts if facts.get("completed") else {"completed": True})
        return row

    views = scan(probe, model, cache, price)
    for (task_id, slot), view in views.items():
        if view["state"] == "graded":
            row["graded"] += 1
            row["pass"] += view["status"] == "pass"
            row["broken"] += bool(view.get("broken"))
            row["broken_pass"] += bool(view.get("broken")) and view["status"] == "pass"
            row["runaway"] += bool(view.get("runaway"))
            row["timeout"] += bool(view.get("agent_timeout"))
            row["timeout_pass"] += bool(view.get("agent_timeout")) and view["status"] == "pass"
            if not view.get("broken"):
                row["clean_graded"] += 1
                row["clean_pass"] += view["status"] == "pass"
                row["after_graded"] += 1
                row["after_pass"] += view["status"] == "pass"
            add_rollout(view)
        elif view["state"] == "ended":
            row["ended_ungraded"] += 1
        elif view["state"] == "active":
            row["active"] += 1
            row["in_flight"] += running
    for rec in [p for p in row["recovery_campaigns"] if "-rec" in p]:
        for view in scan(ROOT / "probes" / model / rec, model, cache, price).values():
            if view["state"] == "active":
                row["active"] += 1
                row["in_flight"] += running
            add_rollout(view)
    plan = read_json(ROOT / "reruns" / f"{model}-{name}.json")
    if plan:
        brk_views = {}
        for entry in plan["slots"]:
            if entry["rerun_campaign"] not in brk_views:
                brk_views[entry["rerun_campaign"]] = scan(ROOT / "probes" / model / entry["rerun_campaign"], model,
                                                          cache, price)
            view = brk_views[entry["rerun_campaign"]].get((entry["task_id"], entry["rerun_slot"]), {"state": "pending"})
            row["rerun_planned"] += 1
            if view["state"] == "graded":
                row["rerun_done"] += 1
                row["rerun_outcomes"]["broken_again" if view.get("broken") else view["status"]] += 1
                row["after_graded"] += 1
                row["after_pass"] += view["status"] == "pass"
                add_rollout(view)
            else:
                row["rerun_pending"] += 1
                if view["state"] == "active":
                    row["active"] += 1
                    row["in_flight"] += running
    else:  # not planned yet: every broken-ended graded slot will get its re-run (if re-runs are on for the model)
        pending = sum(1 for view in views.values() if view["state"] == "graded" and view.get("broken"))
        row["rerun_planned"] += pending
        row["rerun_pending"] += pending
    return row


# ---------------------------------------------------------------- report
def pct(a, b):
    return f"{100.0 * a / b:.1f}" if b else "-"


def fmt_time(epoch):
    return time.strftime("%m-%d %H:%M", time.localtime(epoch)) if epoch else "-"


KEYS = ("slots", "graded", "pass", "broken", "broken_pass", "clean_graded", "clean_pass", "runaway", "infra_final",
        "ended_ungraded", "active", "in_flight", "compacted", "graded_rollouts", "rerun_planned", "rerun_done",
        "rerun_pending", "after_graded", "after_pass", "legs", "timeout", "timeout_pass")


def aggregate(rows):
    out = {key: sum(r[key] for r in rows) for key in KEYS}
    out["rerun_outcomes"] = sum((r["rerun_outcomes"] for r in rows), collections.Counter())
    out["finished_times"] = [t for r in rows for t in r["finished_times"]]
    out["durations"] = [d for r in rows for d in r["durations"]]
    out["running_cap"] = sum(r["cap"] for r in rows if r["running"])
    out["campaigns"] = len(rows)
    out["done"] = sum(r["result"] for r in rows)
    out["running"] = sum(r["running"] for r in rows)
    out["queued"] = sum(r["queued"] for r in rows)
    out["cap"] = sum(r["cap"] for r in rows)
    return out


def projection(by_mw: dict, now: float, start: float) -> dict:
    """ETA per (model, wave): remaining / rate; a model's wave 2 starts at its wave-1 ETA, or now if it runs."""
    out = {}
    for model in MODELS:
        begin = now
        for wave in WAVES:
            agg = by_mw[(model, wave)]
            final = agg["graded"] + agg["infra_final"]
            remaining_first = max(0, agg["slots"] - final)
            rate_broken = agg["broken"] / agg["graded"] if agg["graded"] >= 20 else PRIOR_BROKEN[model]
            remaining = remaining_first + agg["rerun_pending"] + remaining_first * rate_broken
            hours = statistics.median(agg["durations"]) / 3600 if len(agg["durations"]) >= 5 else PRIOR_HOURS[model]
            recent = [t for t in agg["finished_times"] if t >= now - 3 * 3600]
            if agg["running"]:  # a wave that already runs (GLM wave 2 beside wave 1) starts now
                begin = now
            if len(recent) >= 10 and now - start >= 3 * 3600:
                rate, basis = len(recent) / 3.0, "last 3 h"
            elif agg["running"]:  # ATP's live concurrency (attempts in flight now), not the manifest cap
                live = max(agg["in_flight"], agg["running"])
                rate, basis = live / hours, f"{live} in flight now / {hours:.2f} h"
            else:
                cap = agg["running_cap"] or agg["cap"]
                rate, basis = cap / hours, f"{cap} in flight / {hours:.2f} h"
            eta = begin + 3600 * remaining / rate if remaining and rate else begin
            out[(model, wave)] = {"remaining": remaining, "rate_per_h": rate, "basis": basis, "eta": eta}
            begin = eta
    out["eta"] = max(v["eta"] for v in out.values())
    return out


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-budget", action="store_true")
    parser.add_argument("--no-squeue", action="store_true")
    parser.add_argument("--min-gap-min", type=float, default=10.0)
    parser.add_argument("--root", type=Path, help="tests only: read probes/results/reruns/legs and write the outputs "
                                                  "there instead of this directory")
    args = parser.parse_args()
    if args.root:
        ROOT = args.root.resolve()
    out_md, rows_path, cache_path = ROOT / "PROGRESS.md", ROOT / "PROGRESS-rows.json", ROOT / "progress-cache.json"
    now = time.time()
    price = prices()
    cache = read_json(cache_path, {}) or {}
    jobs = None if args.no_squeue else squeue()
    leg_rows = legs()
    info = campaigns()
    rows = [campaign_status(model, name, data, cache, jobs, leg_rows, price[model]) for (model, name), data in info.items()]
    by_mw = {(model, wave): aggregate([r for r in rows if r["model"] == model and r["wave"] == wave])
             for model in MODELS for wave in WAVES}
    by_model = {model: aggregate([r for r in rows if r["model"] == model]) for model in MODELS}
    total = aggregate(rows)
    costs = {model: model_cost(model, cache, price[model]) for model in MODELS}

    starts = [leg for leg in leg_rows if leg.get("event") == "start"]
    wave_starts = [leg for leg in starts if leg.get("campaign") != "pilot"]
    start_at = min((time.mktime(time.strptime(leg["at"][:19], "%Y-%m-%dT%H:%M:%S")) for leg in starts), default=None)
    baseline = next(((leg.get("budget") or {}).get("spend") for leg in starts if leg.get("hub_key") == KEY
                     and after(leg.get("at"), KEY_FROM) and (leg.get("budget") or {}).get("spend") is not None), None)
    old_spent, old_last = old_key_spend(starts)
    record = {"remaining": None, "spend": None} if args.no_budget else key_budget()
    spent = round(record["spend"] - baseline, 2) if record.get("spend") is not None and baseline is not None else None
    proj = projection(by_mw, now, start_at or now)
    cancelled_models = [m for m in MODELS
                        if any(cancelled_by_user(m, f"{w}-g{g}") for w, n in WAVES.items() for g in range(n))]
    for model in cancelled_models:  # OPERATOR_NOTE 2026-11-02 ~02:00 PST: no more rollouts, re-runs or cost
        for wave, groups in WAVES.items():
            if all(cancelled_by_user(model, f"{wave}-g{g}") for g in range(groups)):
                proj[(model, wave)] = {"remaining": 0, "rate_per_h": 0.0, "basis": CANCEL_LABEL, "eta": now,
                                       "cancelled": True}
    proj["eta"] = max(v["eta"] for k, v in proj.items() if k != "eta")
    finished = {model: by_model[model]["graded"] + by_model[model]["infra_final"] + by_model[model]["rerun_done"]
                for model in MODELS}
    unit = {model: (costs[model]["est_cost"] / costs[model]["ended_attempts"]) if costs[model]["ended_attempts"] >= 10
            else None for model in MODELS}
    prior = read_json(HERE / "analysis" / "cost-projection.json", {}) or {}
    prior_unit = (prior.get("unit_cost_per_final_slot") or {})
    to_go = {}
    for model in MODELS:
        cost = 0.0
        for wave in WAVES:
            remaining = proj[(model, wave)]["remaining"]
            per_slot = unit[model] if unit[model] is not None else \
                ((prior_unit.get(wave[1:]) or {}).get(model) or {}).get("central", 4.0)
            cost += remaining * per_slot
        to_go[model] = round(cost, 2)
    projected_more = round(sum(to_go.values()), 2)

    def mw(model, wave):
        a = by_mw[(model, wave)]
        return [a["graded"], a["slots"]]

    history_row = {
        "at": now, "time": time.strftime("%m-%d %H:%M", time.localtime(now)),
        "graded": {model: [by_model[model]["graded"], by_model[model]["slots"]] for model in MODELS},
        "pass_at_1": {model: pct(by_model[model]["pass"], by_model[model]["graded"]) for model in MODELS},
        "agent_timeouts": {model: by_model[model]["timeout"] for model in MODELS},
        "pass_at_1_excl_timeout": {model: pct(by_model[model]["pass"] - by_model[model]["timeout_pass"],
                                              by_model[model]["graded"] - by_model[model]["timeout"])
                                   for model in MODELS},
        "graded_total": total["graded"], "pass": total["pass"], "broken": total["broken"], "runaway": total["runaway"],
        "infra_final": total["infra_final"], "ended_ungraded": total["ended_ungraded"],
        "rerun_pending": total["rerun_pending"], "rerun_done": total["rerun_done"],
        "in_flight": total["in_flight"], "active": total["active"], "running_campaigns": total["running"],
        "est_cost": {model: costs[model]["est_cost"] for model in MODELS}, "key": KEY, "key_spent": spent,
        "key_remaining": record.get("remaining"), "eta": proj["eta"], "projected_more": projected_more,
        "jobs_known": jobs is not None}
    history = read_json(rows_path, []) or []
    if history and now - history[-1]["at"] < 60 * args.min_gap_min:
        history[-1] = history_row
    else:
        history.append(history_row)
    write_atomic(rows_path, json.dumps(history, indent=1) + "\n")
    write_atomic(cache_path, json.dumps(cache) + "\n")

    # ---- markdown
    lines = [
        "# DeepSWE v1.1 teacher rollouts: Kimi-K3 / GLM-5.3 / DeepSeek-V4-Pro x 113 tasks x 5",
        "",
        f"_Generated by progress.py at {time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(now))}. The file is rewritten "
        "on every run; the first table keeps one row per run (PROGRESS-rows.json)._",
        "",
        f"- **Request (hengyuf, 2026-10-30, verbatim):** {REQUEST}",
        "- **Config:** " + "; ".join(f"{m} = `{MODEL_IDS[m]}`" for m in MODELS) + " (reasoning_effort max, "
        "temperature 1.0, top_p 0.95); OpenCode 1.17.11 stock system prompt; limit.context = limit.input = 262,144 "
        "(compaction at 242,144), limit.output = 131,072; ATP agent timeout 10,800 s; each task's own DeepSWE grader; "
        "hengyuf's OpenSandbox credential.",
        "- **Hub key:** **NVAPI_KEY_NEW_2** for every leg from 2026-10-31 18:15 (OPERATOR_NOTE 18:15, user decision "
        "relayed by main: \"kimi/GLM/DeepSeek 的rollout 用NEW_2 继续跑\"); legs start only at >= $1,700 left and stop "
        "below $1,500 (NEW_2 also carries the r6 RL rubric judge); at the floor the campaigns are stopped (STOP "
        "files), no other key. Before: NVAPI_KEY_NEW_3 for the wave-1 legs of 2026-10-30 23:07 .. 10-31 10:40 "
        "(OPERATOR_NOTE 21:45; stopped at its $1,000 floor, never used again); the three pilot legs of 10-30 21:34 "
        "ran on NVAPI_KEY_NEW_2.",
        "- **Agent timeouts (OPERATOR_NOTE 2026-10-31 02:15):** slots that ended on ATP's 10,800 s agent timeout are "
        "flagged (ended_by_agent_timeout) and stay graded as they are (no re-run); pass@1 is shown with and without "
        "them. Kimi auto-pause rule: if at least 3 and more than 20% of the Kimi slots that end in a monitor window hit "
        "the timeout, the Kimi campaigns are paused (resumable) and main is asked.",
    ]
    paused_models = [m for m in MODELS if any(paused_by_main(m, f"{w}-g{g}") for w, n in WAVES.items() for g in range(n))]
    for model in paused_models:
        lines.append(
            f"- **{model.upper() if model == 'glm' else model}: {PAUSE_LABEL}** (OPERATOR_NOTE 2026-11-01 05:15 PST): "
            "the r6 rubric judge (GLM-5.3 on NVAPI_KEY_NEW_2) got HTTP 429s; main wrote STOP files for every "
            f"{model}-* campaign and cancelled the running tdw-{model}-* legs (sacct: 2026-11-01 05:06:07 PST). No "
            f"{model} campaign (wave 1 or 2) is resubmitted until a later OPERATOR_NOTE entry; the cancelled in-flight "
            "slots stay retryable. The other models continue.")
    for model in cancelled_models:
        lines.append(
            f"- **{model.upper() if model == 'glm' else model}: CANCELLED by the user** (OPERATOR_NOTE entry "
            "headed 2026-11-02 ~02:00 PST; hengyuf, verbatim: \"cancel the DeepSWE teacher rollout for GLM\"): "
            f"main wrote STOP files for every {model}-* campaign and cancelled the tdw-{model}-* legs (sacct: "
            f"2026-11-02 01:46:45 PST); they are never resubmitted, and no {model} re-run (brk) or recovery (rec) "
            f"leg runs ({by_model[model]['rerun_pending']} planned re-run(s) cancelled). {model} is PARTIAL: its "
            "graded slots count as they stand; the slots in flight at the cancel are cancelled (not run, not "
            "graded); it is kept out of any 'complete' claim. Kimi and DeepSeek continue to completion under the "
            "existing rules.")
    lines += [
        f"- **Campaigns:** per model wave 1 (79 tasks without an accepted synthetic task) in 3 groups, then wave 2 "
        "(the other 34) in 2 groups; 5 rollouts per task; in flight per campaign kimi 8, glm 8, deepseek 5 (DeepSeek's "
        "hub-wide token limit); at most 9 campaigns at once. Broken-ended slots get one re-run (`<campaign>-brk<n>`); "
        "output-cap runaways are flagged only.",
        "- **GLM caps (OPERATOR_NOTE 2026-10-31 02:15, 2026-11-01 11:15, 14:15 and 15:35 PST):** GLM resumed at 2026-11-01 "
        "11:10 PST (main's pause of 05:15 lifted; r6 paused at a segment boundary), every GLM leg passing "
        "TDW_MAX_CONCURRENCY_OVERRIDE=4. GLM wave 2 (w2-g0, w2-g1) started at 14:06 PST with TDW_FORCE_W2=1 and cap 4 "
        "(OPERATOR_NOTE 14:15). At 14:23 PST GLM-5.3 429s passed the 20% rule over 30 min (rollout replies 26.4%, "
        "budget checks 30%), so GLM campaigns created from then on (rec / brk, any new leg) get 2 "
        "(TDW_MAX_CONCURRENCY_OVERRIDE=2). ATP takes the cap from each campaign's manifest (fixed when the campaign was "
        "created): GLM wave 1 keeps 8 and wave 2 keeps 4. Within that cap ATP's adaptive admission sets the live limit "
        "(halved on 429 / 408 / 503 replies, raised only after healthy completions). Main STOPPED GLM wave 2 at "
        "2026-11-01 15:21 PST (OPERATOR_NOTE entry headed 15:35 PST: STOP files in babysit/glm-w2-g0 and "
        "glm-w2-g1, legs 21158897 / 21158899 cancelled), because its 8 in-flight rollouts shared one API slot "
        "under ~58% GLM-5.3 429s; those attempts stay retryable. The entry's auto-restart rule (GLM-5.3 429s below "
        "10% of the rollout replies and of the budget checks for 60 consecutive minutes) was met at 19:21 PST, and "
        "GLM wave 2 resumed at 19:25-19:26 PST (legs 21167343 / 21167279, TDW_FORCE_W2=1, "
        "TDW_MAX_CONCURRENCY_OVERRIDE=2); ATP marked the 8 stopped attempts interrupted (retryable). ATP keeps "
        "a campaign's settings immutable (its source-snapshot check refused a cap-2 manifest edit at 19:23 PST; "
        "the edit was reverted), so both wave-2 campaigns keep their own cap 4, with ATP's live limit restarting "
        "at the saved 1. GLM wave 1 kept running until the user cancelled GLM (2026-11-02 01:46 PST; see "
        "the CANCELLED bullet).",
        f"- **Start:** {time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(start_at)) if start_at else 'not started'}; "
        f"**total:** {TOTAL} rollouts (565 per model) + re-runs.",
    ]
    for model in MODELS:
        data = read_json(ROOT / "results" / f"{model}-pilot.json")
        verify = sorted(ROOT.glob(f"analysis/verify-{model}-pilot-*.json"))
        if data:
            pilot = next(iter(next(iter(data["records"].values()))), {})
            checks = read_json(verify[-1], {}) if verify else {}
            state = ("all checks OK" if checks.get("all_ok") else
                     ("FAILED: " + ", ".join(k for k, v in (checks.get("checks") or {}).items() if not v))
                     if checks else "not run")
            cost = checks.get("cost_usd") if checks else None
            lines.append(f"- **{model} pilot:** {pilot.get('status')} (F2P {pilot.get('f2p')}, P2P {pilot.get('p2p')}), "
                         f"broken-ended {pilot.get('ended_by_broken_reply')}; hub key "
                         f"{', '.join(sorted({leg.get('hub_key') or '?' for leg in data.get('hub_keys_by_leg') or []})) or '?'}"
                         + (f", cost ${cost:,.2f}" if isinstance(cost, (int, float)) else "") + f"; verify_pilot {state}")
        else:
            lines.append(f"- **{model} pilot:** {'queued/running' if (jobs or {}).get(f'{model}-pilot') else 'no result yet'}")
    lines += [
        "",
        "## Progress (one row per report)",
        "",
        "| time | graded kimi · glm · deepseek | pass@1 excl. infra kimi · glm · deepseek | agent timeouts kimi · glm · "
        "deepseek | pass@1 excl. timeouts kimi · glm · deepseek | broken-ended | runaways "
        "| re-runs pending / done | infra / awaiting | in flight | campaigns running | est. cost kimi · glm · deepseek "
        "| hub key: spent / left | ETA | est. $ to go |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in history:
        g, p, c = r["graded"], r["pass_at_1"], r["est_cost"]
        t, x = r.get("agent_timeouts"), r.get("pass_at_1_excl_timeout")
        lines.append(
            f"| {time.strftime('%m-%d %H:%M %Z', time.localtime(r['at']))} | {' · '.join(f'{g[m][0]}/{g[m][1]}' for m in MODELS)} | {' · '.join(p[m] for m in MODELS)} "
            f"| {' · '.join(str(t[m]) for m in MODELS) if t else '-'} | {' · '.join(x[m] for m in MODELS) if x else '-'} "
            f"| {r['broken']} | {r['runaway']} | {r['rerun_pending']} / {r['rerun_done']} | {r['infra_final']} / "
            f"{r['ended_ungraded']} | {r['in_flight'] if r['jobs_known'] else str(r['active']) + '?'} | "
            f"{r['running_campaigns'] if r['jobs_known'] else '?'} | {' · '.join(f'${c[m]:,.0f}' for m in MODELS)} | "
            f"{str(r.get('key', 'NVAPI_KEY_NEW_2')).replace('NVAPI_KEY_', '')}: "
            f"{'$' + format(r.get('key_spent', r.get('new2_spent')), ',.0f') if r.get('key_spent', r.get('new2_spent')) is not None else '-'} / "
            f"{'$' + format(r.get('key_remaining', r.get('new2_remaining')), ',.0f') if r.get('key_remaining', r.get('new2_remaining')) is not None else '?'} | "
            f"{fmt_time(r['eta'])} | ${r['projected_more']:,.0f} |")
    lines += [
        "",
        "pass@1 values are percentages of first attempts. agent timeouts = graded first attempts that ended on ATP's "
        "10,800 s agent timeout (columns added 2026-10-31 02:30; '-' in earlier rows). in flight with '?' = squeue "
        "unavailable (count of started attempts). est. cost = OpenCode export tokens priced per model (main session; "
        "~98-100% of the journal cost). Times are local: PDT until daylight saving time ended at 2026-11-01 02:00 PDT "
        "(clocks went back to 01:00 PST), PST after; the first column shows the zone (added 2026-11-01).",
        "",
        "## Model x wave now",
        "",
        "| model | wave | slots | terminal / graded | passes | pass@1 raw / excl. infra / excl. broken | after re-runs "
        "| agent timeouts (passed) | pass@1 excl. agent timeouts "
        "| infra / awaiting | broken-ended | runaways | re-runs planned / done (pass/fail/broken again/infra) | compacted "
        "| in flight | campaigns done / running / queued | legs | ETA |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model in MODELS:
        for wave in WAVES:
            a = by_mw[(model, wave)]
            o = a["rerun_outcomes"]
            terminal = a["graded"] + a["infra_final"]
            lines.append(
                f"| {model} | {wave} | {a['slots']} | {terminal} / {a['graded']} | {a['pass']} | "
                f"{pct(a['pass'], terminal)} / {pct(a['pass'], a['graded'])} / {pct(a['clean_pass'], a['clean_graded'])} "
                f"| {pct(a['after_pass'], a['after_graded'])} | {a['timeout']} ({a['timeout_pass']}) | "
                f"{pct(a['pass'] - a['timeout_pass'], a['graded'] - a['timeout'])} "
                f"| {a['infra_final']} / {a['ended_ungraded']} | "
                f"{a['broken']} | {a['runaway']} | {a['rerun_planned']} / {a['rerun_done']} ({o.get('pass', 0)}/"
                f"{o.get('fail', 0)}/{o.get('broken_again', 0)}/{o.get('infra', 0)}) | {a['compacted']} of "
                f"{a['graded_rollouts']} | {a['in_flight']} | {a['done']} / {a['running']} / {a['queued']} of "
                f"{a['campaigns']} | {a['legs']} | "
                f"{'cancelled' if proj[(model, wave)].get('cancelled') else fmt_time(proj[(model, wave)]['eta'])} |")
    lines += [
        "",
        "## Campaigns now",
        "",
        "| model | campaign | slots | cap | graded | pass | pass@1 | agent timeouts | pass@1 excl. timeouts | broken-ended "
        "| runaways | re-runs planned / done | in flight | infra / awaiting | compacted | legs | jobs | results |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['campaign']} | {r['slots']} | {r['cap']} | {r['graded']} | {r['pass']} | "
            f"{pct(r['pass'], r['graded'])} | {r['timeout']} | {pct(r['pass'] - r['timeout_pass'], r['graded'] - r['timeout'])} "
            f"| {r['broken']} | {r['runaway']} | {r['rerun_planned']} / {r['rerun_done']} "
            f"| {r['in_flight']} | {r['infra_final']} / {r['ended_ungraded']} | {r['compacted']} | {r['legs']} | "
            f"{' '.join(r['jobs']) or (CANCEL_LABEL if not r['result'] and cancelled_by_user(r['model'], r['campaign']) else PAUSE_LABEL if not r['result'] and paused_by_main(r['model'], r['campaign']) else '-')} "
            f"| {'yes' if r['result'] else '-'} |")
    lines += ["", "## Cost and projection", ""]
    for model in MODELS:
        c = costs[model]
        lines.append(f"- {model}: est. ${c['est_cost']:,.2f} over {c['ended_attempts']} ended attempts "
                     f"({c['agent_timeout_attempts']} of them ended on the agent timeout, incl. ATP-retried ones)"
                     + (f" (${unit[model]:.2f} per attempt)" if unit[model] is not None else "")
                     + f"; ~${to_go[model]:,.0f} to go")
    lines.append(f"- {KEY}: " + (f"${spent:,.2f} spent since the first NEW_2 tdw leg from 2026-10-31 18:15 (includes "
                                  "the r6 judge's use); " if spent is not None else "no NEW_2 leg since 18:15 yet; ")
                 + (f"${record['remaining']:,.2f} left" if record.get("remaining") is not None else "budget not read")
                 + (f"; est. ${projected_more:,.0f} still needed vs ${record['remaining'] - GUARD_FLOOR:,.0f} usable "
                    f"above the ${GUARD_FLOOR:,.0f} guard floor (legs start only at >= ${START_FLOOR:,.0f})"
                    if record.get("remaining") is not None else ""))
    pilot_costs = []
    for model in MODELS:
        verify = sorted(ROOT.glob(f"analysis/verify-{model}-pilot-*.json"))
        cost = (read_json(verify[-1], {}) or {}).get("cost_usd") if verify else None
        if isinstance(cost, (int, float)):
            pilot_costs.append(f"{model} ${cost:,.2f}")
    lines.append(f"- {OLD_KEY} (wave-1 legs 2026-10-30 23:07 .. 10-31 10:40, stopped at its $1,000 floor; no longer "
                 "used): " + (f"${old_spent:,.2f} spent from its first leg start to its last reading "
                              f"(${old_last['remaining']:,.2f} left at {old_last.get('at')}; includes the r6 judge's use)"
                              if old_spent is not None else "no reading"))
    lines.append(f"- {PILOT_KEY} (pilots): the three pilot legs of 2026-10-30 21:34 (before the first key switch)"
                 + (f"; journal cost {', '.join(pilot_costs)}" if pilot_costs else ""))
    for model in MODELS:
        for wave in WAVES:
            p = proj[(model, wave)]
            if p.get('cancelled'):
                a = by_mw[(model, wave)]
                lines.append(f"- {model} {wave}: **{CANCEL_LABEL}**: {a['slots'] - a['graded'] - a['infra_final']} "
                             f"of {a['slots']} slots not graded ({a['active']} in flight at the cancel, "
                             f"{a['ended_ungraded']} ended ungraded (interrupted or infra) and not retried, the "
                             "rest never started); no more rollouts, re-runs or cost")
                continue
            paused = p['remaining'] > 0 and model in paused_models
            lines.append(f"- {model} {wave}: {p['remaining']:.0f} rollouts to go (incl. expected re-runs) at "
                         f"{p['rate_per_h']:.1f}/h ({p['basis']}) -> ~{fmt_time(p['eta'])}"
                         + (f" (**{PAUSE_LABEL}**: this ETA assumes an immediate resume)" if paused else ""))
    lines.append(f"- projected finish ~{fmt_time(proj['eta'])} (leg restarts and queue waits not modelled"
                 + (f"; {', '.join(paused_models)} {PAUSE_LABEL}, so its share assumes an immediate resume"
                    if paused_models else "")
                 + (f"; {', '.join(cancelled_models)} {CANCEL_LABEL}: not counted" if cancelled_models else "")
                 + ")")
    lines.append("")
    write_atomic(out_md, "\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in history_row.items() if k not in ("graded",)}, default=str))


if __name__ == "__main__":
    main()
