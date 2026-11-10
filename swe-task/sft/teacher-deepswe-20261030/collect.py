#!/usr/bin/env python3
"""teacher-deepswe (2026-10-30): the final collection -- the dataset, the per-task table and FINAL-REPORT.md.

    /usr/bin/python3 -I -B collect.py [--workers 16] [--no-journal-cost] [--allow-incomplete] [--root DIR]

Sources: results/<model>-<campaign>.json (driver.py writes it when a campaign completes) and, with --allow-incomplete,
the probe dir of every wave campaign that has no results file (read directly). Read-only on probes/ and results/;
writes only dataset/, per_task.csv, FINAL-REPORT.md and analysis/collect-*.

PARTIAL COLLECTION (2026-10-31). The run stopped at the NVAPI_KEY_NEW_3 budget floor ($1,000 guard, OPERATOR_NOTE
2026-10-30 21:45) on 2026-10-31 10:30-10:40 PDT, before any campaign finished: the guards stopped the 9 wave-1
campaigns and the 6 wave-2 campaigns never started. A campaign without a results file is read from its probe dir, slot
by slot:
  graded       ATP's own slot_state() says "completed" (analysis/atp-slot-states.json, written read-only by
               final/atp_states.py with ATP's unchanged code). It is cross-checked against this script's own read-only
               view: the slot's first attempt with evaluation_completed and no generation failure in its
               generation-validity-v2-r7.json cache. status pass (reward 1) or fail; the record is built as
               rollouts.probe_records() + driver.compact() would build it.
  cancelled    the latest attempt was cancelled by the budget stop (failure.json CancelledError). ATP retries the slot
               on resume.
  infra        the latest attempt ended without a grade before the stop. ATP's state says whether ATP would retry it
               ("retryable") or not ("needs_review", e.g. a verifier killed at its 1,800 s timeout).
  active       started, no end marker (none expected: no leg runs).
  not_started  no attempt.
Only graded slots are dataset rows. Every planned slot (113 tasks x 3 models x 5 = 1,695) is in dataset/slots.jsonl
with its state. Broken-reply re-runs (<campaign>-brk<n>) and budget-refusal recoveries (<campaign>-rec<n>) run only
after a campaign's slots are terminal, so none ran. Broken-ended slots are flagged and counted as they are.

STATUS OVERRIDE (2026-11-01, OPERATOR_NOTE 11:15 PST item 4 and 14:15 PST item 3, main): GRADED_FAIL_OVERRIDES names
slots that count as a GRADED FAIL although ATP left them ungraded (needs_review): kimi w2-g1 13-0 =
scriggo-method-declarations r0, whose OpenCode the agent's own `pkill -f opencode/bin` killed at 2026-11-01 08:20:26
PST; its evaluation completed (reward 0, f2p 0/48, empty patch). The override applies only while the slot's final
record in its results file is still that attempt's infra ending with evaluation_completed and reward 0 (otherwise the
check status_overrides_unmatched names it). The dataset row and ledger line get status "fail" and status_override
{from, to, atp_status, reason, source}; MANIFEST.json lists it (status_overrides); FINAL-REPORT.md lists it next to the
slots ATP left needs_review (the verifier killed at its 1,800 s timeout), which stay ungraded.

FINAL COLLECTION (2026-11-02; final/collect_final_20261102.py). After the OPERATOR_NOTE entry headed "2026-11-02 ~02:00
PST (main) -- USER DECISION, CANCEL GLM" (hengyuf, verbatim: "cancel the DeepSWE teacher rollout for GLM."): the 5 Kimi
and 5 DeepSeek campaigns are complete (results files); the 5 GLM campaigns were cancelled for good (main's STOP files in
babysit/glm-*, legs cancelled at 01:46:45 PST) and are read from their probe dirs (--allow-incomplete) with campaign
state "cancelled_by_user" (USER_CANCELLED; check user_cancel_stop_missing). GLM is PARTIAL (model_status
"partial_cancelled_by_user"): its graded slots count as they stand, and its started-ungraded slots are
  cancelled_in_flight  in flight at the user's cancel: started.json, no end marker, ATP "active"; cut off, not graded
  interrupted          ended only by ATP's interrupted.json (its leg was stopped: main's GLM pause of 2026-11-01 05:15
                       PST or wave-2 stop of 15:35 PST); ATP retryable, never retried before the cancel
  cancelled            failure.json CancelledError (the 2026-10-31 NEW_3 budget-floor stop), never retried
  infra                ended on an infra error (ATP retryable or needs_review), never retried
pass@1 raw counts infra as a fail and leaves the stopped ones (cancelled / interrupted / cancelled_in_flight) out. The
uncovered-task verdict is yes / no / undetermined (undetermined only through ungraded slots of an unfinished campaign).
No broken-reply re-run (brk) or recovery (rec) campaign ran: Kimi and DeepSeek had no broken-ended slot and no budget
refusal; GLM's one broken-ended slot lost its planned re-run with the cancel.

Outputs:
dataset/<model>.jsonl.gz  one line per dataset slot of the model's wave campaigns (a complete campaign: every final
  slot, as driver.py's results; a partial one: graded slots only), ordered by task_index, slot. Fields: every field of
  synth rollouts._record (slot, model, status, reward, f2p, p2p, failing_tests, agent_error, verifier_error, patch,
  output = the Responses-API transcript, attempt_dir), rebuilt from the slot's final attempt as
  rollouts.probe_records does (terminal.json; a regrade's generation from the slot's latest attempt with a response),
  plus: task_id, task_index (DeepSWE order), language, category, wave, covered_by_accepted_synth, base_k10, model_id,
  campaign (the wave campaign), campaign_complete, source_campaign (the probe dir of the attempt: <campaign>,
  -rec<n> or -brk<n>), repeat, ended_by_broken_reply, last_reply_kind, last_reply_stop_class, last_reply_finish,
  last_reply_completion_tokens, ended_by_output_cap_runaway, ended_by_agent_timeout /
  first_attempt_ended_by_agent_timeout (ATP's 10,800 s agent timeout ended the final / first attempt; OPERATOR_NOTE
  2026-10-31 02:15: such slots stay graded as they are), compactions, patch_chars, re-run provenance (rerun_reason,
  rerun_campaign, rerun_slot, original_attempt, rerun_attempt, rerun_replaced), budget-refusal recovery
  (recovered_from_attempt, recovery_campaign), first_attempt_status, first_attempt_ended_by_broken_reply,
  status_override (GRADED_FAIL_OVERRIDES, else null), and cost:
  cost_usd = the final attempt's hub cost, slot_cost_usd = every attempt the slot used, cost_basis = journal
  (costs.attempt_usage: LiteLLM usage.cost when reported, else prices.json per token) or export (OpenCode token
  counts priced as progress.py; --no-journal-cost), cost_by_key (hub key NAME -> USD, from the leg running at each
  reply), prompt_tokens / cached_tokens / completion_tokens / hub_calls_ok / hub_errors (non-200 replies by status).
dataset/pilots.jsonl.gz: the same for the 3 one-rollout pilots (not part of the 5 per task).
dataset/index.jsonl: every row of the dataset files without "output".
dataset/slots.jsonl: every planned slot (1,695) and the 3 pilots: model, campaign, campaign_state, task, repeat, state
  (graded / cancelled / infra / active / not_started, or "final" for a complete campaign's slot), status, ATP state,
  attempts, attempt_dir, end reason, in_dataset, cost of all its attempts (+ by key, tokens, hub errors); a dataset
  slot's line also has verifier_error and status_override.
dataset/MANIFEST.json: completeness, row counts, sha256 and bytes of every dataset file, the ATP-state file used,
  status_overrides (the GRADED_FAIL_OVERRIDES applied).
per_task.csv: per task (113, DeepSWE order): task_id, task_index, language, wave, covered_by_accepted_synth, base_k10
  (the base model's resolved out of 10), and per model: _planned (5), _graded, _pass, _complete (1 when all 5 slots
  are graded), _k5 (passes out of 5, only when complete), _cancelled, _infra, _not_started, _broken_first,
  _broken_final, _runaway, _agent_timeout, _compacted (2026-11-02: + _interrupted, _cancelled_in_flight after
  _cancelled, and _model_status = complete / partial_cancelled_by_user / partial).
FINAL-REPORT.md: script-generated: completeness, overview per model, wave 1 vs wave 2, matched comparisons, per
  language, uncovered tasks with base <= 2/10 and a teacher >= 3/5, Kimi vs the 1M-context Kimi export (context
  only), re-runs, cost and keys, checks, files.
analysis/collect-summary.json (the report's numbers), analysis/collect-kimi-vs-1m.csv (per task).
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import datetime  # noqa: E402
import csv  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

HERE = Path(__file__).resolve().parent
SYNTH = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth"
MODELS = ("kimi", "glm", "deepseek")
MODEL_IDS = {"kimi": "nvidia/moonshotai/kimi-k3", "glm": "nvidia/zai-org/glm-5.3",
             "deepseek": "nvidia/deepseek-ai/deepseek-v4-pro"}
CAMPAIGNS = ("w1-g0", "w1-g1", "w1-g2", "w2-g0", "w2-g1")
GROUPS = {"w1": 3, "w2": 2}
REPEATS = 5
OUTPUT_CAP = 131072
REQUEST = ("我想在deepSWE 每一个task 上额外用kimi,Deepseek, GLM (Deepseek: nvidia/deepseek-ai/deepseek-v4-pro GLM: "
           "nvidia/zai-org/glm-5.3 ), 每个模型rollout 5遍. 用一个subagent 来收集他们的rollouts")
ATP_STATES = Path("analysis") / "atp-slot-states.json"
KIMI_1M = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/deepswe-partial-analysis/per_task.csv")
KIMI_1M_NOTE = ("baiheh's export partial-selected-super-kimi-20261015T1603Z (hengyuf/deepswe-partial-analysis, "
                "read-only): Kimi-K3, OpenCode context 1,048,576 (never compacts), per-call cap 131,072, 5 of 16 "
                "repeats selected without looking at rewards, served 2026-10-15")
BUDGET_REFUSAL = re.compile(r"(?i)budget[_ ]exceeded|budget has been exceeded|exceededbudget|max budget")
ENDED = ("terminal.json", "failure.json", "dispatch-failure.json", "interrupted.json")
CHECK_KEYS = ("atp_state_missing", "atp_view_disagree", "graded_without_validity_cache", "budget_refusals",
              "dataset_rows_without_output", "slots_per_task_not_5", "duplicate_slots", "dataset_ledger_mismatch",
              "attempt_outside_campaign", "ungraded_dataset_rows", "pass_without_reward_1", "fail_with_reward_1",
              "status_overrides_unmatched", "user_cancel_stop_missing", "active_outside_cancel",
              "in_flight_started_after_cancel")
# OPERATOR_NOTE 2026-11-01 11:15 PST item 4 and 14:15 PST item 3 (main): (model, campaign, task_id, repeat) -> a slot
# that counts as a GRADED FAIL although ATP left it needs_review (see the docstring). Applied by graded_fail_override().
GRADED_FAIL_OVERRIDES = {
    ("kimi", "w2-g1", "scriggo-method-declarations", 0): {
        "attempt": "13-0/attempt-0", "from_status": "infra", "atp_status": "needs_review",
        "reason": "the agent's own `pkill -f opencode/bin` killed its OpenCode at 2026-11-01 08:20:26 PST (ATP "
                  "needs_review: generation validity failure unknown); the evaluation completed: reward 0, f2p 0/48",
        "source": "OPERATOR_NOTE 2026-11-01 11:15 PST item 4 and 14:15 PST item 3 (main): the model's own outcome, "
                  "no re-run"}}
OVERRIDES_APPLIED = {}   # key -> what was applied (MANIFEST status_overrides)
OVERRIDE_PROBLEMS = {}   # key -> why an override did not match (check status_overrides_unmatched)
# OPERATOR_NOTE entry headed "2026-11-02 ~02:00 PST (main) -- USER DECISION, CANCEL GLM" (items 1 and 3): the model's
# wave campaigns without a results file are "cancelled_by_user" (main's babysit/<model>-<campaign>/STOP must exist);
# the model is reported PARTIAL and kept out of any "complete" claim. The leg states are from sacct (2026-11-02).
USER_CANCELLED = {
    "glm": {"decision": "cancel the DeepSWE teacher rollout for GLM.",
            "source": "OPERATOR_NOTE entry headed 2026-11-02 ~02:00 PST (main), USER DECISION, CANCEL GLM",
            "cancelled_at": "2026-11-02T01:46:45-0800",
            "how": ("Main wrote babysit/glm-<campaign>/STOP for all 5 GLM campaigns at 2026-11-02 01:46:40 PST and "
                    "cancelled their legs (sacct: tdw-glm-w1-g0 21154933, w1-g1 21154935, w1-g2 21154938, w2-g0 "
                    "21167343 and w2-g1 21167279 CANCELLED at 01:46:45 PST). No GLM leg, broken-reply re-run (brk) or "
                    "recovery (rec) leg runs again.")}}
STOPPED_STATES = ("cancelled", "interrupted", "cancelled_in_flight")  # stopped before the slot ended on its own
INFRA_STATES = ("infra", "final_infra")
NEW2_RESET_AFTER = "2026-10-31T18:00:00-0700"  # NVAPI_KEY_NEW_2's spend read as reset at the 18:11 check
ROOT = HERE
_MODULES = {}


def _module(name: str):
    """This directory's costs / progress / brokenreply modules, and synth's rollouts (loaded once per process)."""
    if name not in _MODULES:
        if name == "rollouts":
            sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != HERE]
            sys.path.insert(0, SYNTH)
            import rollouts  # noqa: E402  (synth's: _record / _generation, read-only helpers)
            _MODULES[name] = rollouts
        else:
            spec = importlib.util.spec_from_file_location(f"tdw_{name}", HERE / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _MODULES[name] = module
    return _MODULES[name]


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def price_table() -> dict:
    """{alias: per-token (uncached, cached, completion)}: prices.json (glm, deepseek) + the Kimi fit (kimi)."""
    if "prices" not in _MODULES:
        fits = json.loads((HERE / "prices.json").read_text())
        table = {alias: fits[alias]["per_token"] for alias in ("glm", "deepseek")}
        table["kimi"] = json.loads((HERE / "analysis" / "kimi-price-fit.json").read_text())["per_token"]
        _MODULES["prices"] = table
    return _MODULES["prices"]


def campaign_of(attempt) -> str:
    """The probe dir name of an attempt: probes/<model>/<campaign>/models/<model>/attempts/<slot>/attempt-<n>."""
    return Path(attempt).parents[4].name


def attempt_cost(attempt: str, model: str, journal: bool) -> dict:
    """{usd, basis, by_key, prompt, cached, completion, calls, errors} of one attempt: the journal (exact, every reply
    attributed to the leg running at its timestamp, hence to that leg's hub key NAME) when asked and present, else the
    OpenCode export estimate (main session)."""
    if not attempt:
        return {"usd": 0.0, "basis": None, "by_key": {}}
    path = Path(attempt)
    prices = price_table()
    if journal and (path / "http.jsonl").exists():
        costs = _module("costs")
        usage = costs.attempt_usage(path, prices, model, costs.legs_of(model, campaign_of(path)))
        return {"usd": round(usage["cost"], 6), "basis": "journal", "prompt": usage["prompt"], "cached": usage["cached"],
                "completion": usage["completion"], "calls": usage["ok"], "errors": usage["errors"],
                "by_key": {key: round(value["cost"], 6) for key, value in usage["by_key"].items()}}
    facts = _module("progress").export_facts(path, prices[model])
    return {"usd": facts["est_cost"], "basis": "export" if facts["export"] else None, "calls": facts["calls"],
            "by_key": {}}


def sum_costs(costs: list) -> dict:
    """The sum of attempt_cost() results (one slot's attempts)."""
    by_key, errors = collections.Counter(), collections.Counter()
    for cost in costs:
        by_key.update(cost.get("by_key") or {})
        errors.update(cost.get("errors") or {})
    return {"usd": round(sum(c["usd"] for c in costs), 6), "attempts": len(costs),
            "basis": sorted({c["basis"] for c in costs if c["basis"]}),
            "by_key": {key: round(value, 6) for key, value in by_key.items()},
            "prompt": sum(c.get("prompt") or 0 for c in costs), "cached": sum(c.get("cached") or 0 for c in costs),
            "completion": sum(c.get("completion") or 0 for c in costs),
            "calls_ok": sum(c.get("calls") or 0 for c in costs), "errors": dict(errors)}


def slot_attempts(attempt: str) -> list:
    """Every attempt dir of the slot that holds `attempt` (ATP retries included)."""
    if not attempt:
        return []
    return sorted(str(p) for p in Path(attempt).parent.glob("attempt-*") if p.is_dir())


def compact_fields(attempt: str) -> dict:
    """driver.compact()'s broken-ending verdict, last-reply detail and runaway flag of one attempt (same readers)."""
    broken = _module("brokenreply")
    verdict = broken.broken_end(attempt)
    detail = broken.last_reply_detail(attempt) if attempt else {}
    runaway = bool(detail) and not detail.get("tool") and (
        (detail.get("completion_tokens") or 0) >= OUTPUT_CAP or detail.get("finish") == "length")
    return {"ended_by_broken_reply": verdict["ended"], "last_reply_kind": verdict["kind"],
            "last_reply_stop_class": detail.get("stop_class"), "last_reply_finish": detail.get("finish"),
            "last_reply_completion_tokens": detail.get("completion_tokens"), "ended_by_output_cap_runaway": runaway}


def build_row(job: dict) -> dict:
    """One dataset row (with output) for a final record; runs in a worker process."""
    rollouts = _module("rollouts")
    model, record = job["model"], dict(job["record"])
    attempt = Path(record["attempt_dir"]) if record.get("attempt_dir") else None
    if attempt and "ended_by_broken_reply" not in record:  # a partial campaign's slot: driver.compact()'s fields
        record.update(compact_fields(str(attempt)))
    first = job["first"] if job.get("first") is not job["record"] else record
    if "ended_by_broken_reply" not in first:
        first = dict(first, **{k: record[k] for k in ("ended_by_broken_reply",)})
    grade = rollouts.read_json(attempt / "terminal.json", {}) if attempt else {}
    generation = grade if (grade.get("response") or {}).get("output") else (rollouts._generation(attempt) if attempt else {})
    repeat = int(record["slot"].rsplit("-r", 1)[1])
    row = rollouts._record(model, repeat, record["status"], generation, grade, attempt)
    assert row["slot"] == record["slot"], (row["slot"], record["slot"])
    export = _module("progress").export_facts(attempt, price_table()[model]) if attempt else {}
    first_dir = first.get("attempt_dir")
    final_cost = attempt_cost(record.get("attempt_dir"), model, job["journal"])
    used = set(slot_attempts(record.get("attempt_dir")))
    original = record.get("original_attempt") or {}
    used |= set(slot_attempts(original.get("attempt_dir")))
    used |= set(slot_attempts((record.get("rerun_attempt") or {}).get("attempt_dir")))
    if job.get("recovered_from"):
        used |= set(slot_attempts(job["recovered_from"]))
    costs = [final_cost if path == record.get("attempt_dir") else attempt_cost(path, model, job["journal"])
             for path in sorted(used)]
    total = sum_costs(costs)
    source = campaign_of(record["attempt_dir"]) if record.get("attempt_dir") else None
    row.update(job["task"])
    row.update({
        "model_id": MODEL_IDS.get(model, job.get("model_id")), "campaign": job["campaign"],
        "campaign_complete": job["complete"], "source_campaign": source, "repeat": repeat,
        "ended_by_broken_reply": record.get("ended_by_broken_reply"), "last_reply_kind": record.get("last_reply_kind"),
        "last_reply_stop_class": record.get("last_reply_stop_class"),
        "last_reply_finish": record.get("last_reply_finish"),
        "last_reply_completion_tokens": record.get("last_reply_completion_tokens"),
        "ended_by_output_cap_runaway": record.get("ended_by_output_cap_runaway"),
        "ended_by_agent_timeout": _module("progress").agent_timeout(attempt) if attempt else None,
        "first_attempt_ended_by_agent_timeout": _module("progress").agent_timeout(first_dir) if first_dir else None,
        "compactions": export.get("compactions"), "patch_chars": len(row["patch"] or ""),
        "first_attempt_status": first.get("status"),
        "first_attempt_ended_by_broken_reply": first.get("ended_by_broken_reply"),
        "status_override": record.get("status_override"),
        "rerun_reason": record.get("rerun_reason"), "rerun_campaign": record.get("rerun_campaign"),
        "rerun_slot": record.get("rerun_slot"), "original_attempt": record.get("original_attempt"),
        "rerun_attempt": record.get("rerun_attempt"), "rerun_replaced": record.get("rerun_replaced"),
        "recovered_from_attempt": job.get("recovered_from"), "recovery_campaign": job.get("recovery_campaign"),
        "cost_usd": final_cost["usd"], "cost_basis": final_cost["basis"],
        "slot_cost_usd": total["usd"], "slot_attempts": total["attempts"], "slot_cost_basis": total["basis"],
        "cost_by_key": total["by_key"], "prompt_tokens": total["prompt"], "cached_tokens": total["cached"],
        "completion_tokens": total["completion"], "hub_calls_ok": total["calls_ok"], "hub_errors": total["errors"]})
    return row


def graded_fail_override(model: str, name: str, task_id: str, record: dict) -> dict:
    """A GRADED_FAIL_OVERRIDES slot's final record as a graded fail (a copy with status "fail" and status_override),
    while it is still the named attempt's ATP infra ending with a completed evaluation and reward 0; else `record`
    itself (an override that does not match is reported by the status_overrides_unmatched check)."""
    repeat = int(record["slot"].rsplit("-r", 1)[1])
    key = (model, name, task_id, repeat)
    override = GRADED_FAIL_OVERRIDES.get(key)
    if not override:
        return record
    attempt = record.get("attempt_dir") or ""
    terminal = (read_json(Path(attempt) / "terminal.json", {}) or {}) if attempt else {}
    if record.get("status") != override["from_status"]:
        problem = f"final record status {record.get('status')}, not {override['from_status']}"
    elif not attempt.endswith("/" + override["attempt"]):
        problem = f"final attempt {attempt or None}, not .../{override['attempt']}"
    elif terminal.get("evaluation_completed") is not True or terminal.get("reward") != 0:
        problem = (f"terminal.json evaluation_completed {terminal.get('evaluation_completed')}, "
                   f"reward {terminal.get('reward')}")
    else:
        problem = None
    if problem:
        OVERRIDE_PROBLEMS[key] = problem
        return record
    applied = {"from": override["from_status"], "to": "fail", "atp_status": override["atp_status"],
               "reason": override["reason"], "source": override["source"]}
    OVERRIDES_APPLIED[key] = dict(applied, model=model, campaign=name, task_id=task_id, repeat=repeat,
                                  attempt_dir=attempt, f2p=record.get("f2p"), p2p=record.get("p2p"))
    return dict(record, status="fail", status_override=applied)


def jobs_of(model: str, name: str, result: dict, tasks: dict, journal: bool) -> list:
    """Dataset jobs of a campaign with a results file (driver.py's records)."""
    recovered = {(task_id, entry["slot"]): entry for task_id, entries in (result.get("recovered_budget_refusals") or {}).items()
                 for entry in entries}
    out = []
    for task_id, records in result["records"].items():
        firsts = {r["slot"]: r for r in (result.get("first_attempt_records") or {}).get(task_id, [])}
        for record in records:
            rec = recovered.get((task_id, record["slot"])) or {}
            first = firsts.get(record["slot"], record)
            overridden = graded_fail_override(model, name, task_id, record)
            if overridden is not record:  # the first attempt too when it is the same attempt (no re-run)
                first = overridden if first.get("attempt_dir") == record.get("attempt_dir") else first
                record = overridden
            out.append({"model": model, "campaign": name, "complete": True, "record": record,
                        "first": first, "task": task_meta(tasks, task_id),
                        "recovered_from": rec.get("refused_attempt"), "recovery_campaign": rec.get("recovery_campaign"),
                        "journal": journal, "model_id": result.get("model_id")})
    out.sort(key=lambda j: (j["task"]["task_index"] if j["task"]["task_index"] is not None else 10 ** 6,
                            j["task"]["task_id"], j["record"]["slot"]))
    return out


# ---------------------------------------------------------------- partial campaigns (no results file)
def task_meta(tasks: dict, task_id: str) -> dict:
    return tasks.get(task_id) or {"task_id": task_id, "task_index": None, "language": None, "category": None,
                                  "wave": None, "covered_by_accepted_synth": None, "base_k10": None}


def campaign_tasks(waves: dict, name: str) -> list:
    """driver.py's task list of a wave campaign: wave_tasks[K::G]."""
    wave, group = name.split("-g")
    ids = waves["wave1"] if wave == "w1" else waves["wave2"]
    return ids[int(group)::GROUPS[wave]]


def parse_stamp(text: str) -> float:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S%z").timestamp()


def stamp_of(path: Path):
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(path.stat().st_mtime))
    except OSError:
        return None


def last_change(directory: Path):
    """The latest mtime of a file directly in an attempt dir (where ATP writes its logs and markers)."""
    stamps = []
    for path in directory.iterdir():
        try:
            if path.is_file():
                stamps.append(path.stat().st_mtime)
        except OSError:
            continue
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(max(stamps))) if stamps else None


def own_view(slot: dict) -> dict:
    """This script's read-only view of one slot (runs in a worker): the first attempt with evaluation_completed and no
    cached generation failure is the graded one (ATP's slot_state rule; ATP decides from its validity cache), else the
    latest attempt's ending. For an ungraded slot also: the end reason, a budget-refusal check of its journal (driver
    refused_slots: a 4xx with a budget message) and the cost of all its attempts."""
    if not slot.get("slot_dir"):
        return {"state": "not_started", "attempts": 0}
    slot_dir = Path(slot["slot_dir"])
    attempts = sorted((p for p in slot_dir.glob("attempt-*") if p.is_dir()), key=lambda p: int(p.name.split("-")[1]))
    if not attempts:
        return {"state": "not_started", "attempts": 0}
    for attempt in attempts:
        terminal = read_json(attempt / "terminal.json", {}) or {}
        if terminal.get("evaluation_completed") is True:
            validity = read_json(attempt / "generation-validity-v2-r7.json")
            if validity is None or validity.get("failure") is None:
                return {"state": "graded", "attempts": len(attempts), "attempt_dir": str(attempt),
                        "status": "pass" if terminal.get("reward") == 1 else "fail",
                        "validity_cached": validity is not None}
    latest = attempts[-1]
    failure = read_json(latest / "failure.json", {}) or read_json(latest / "dispatch-failure.json", {}) or {}
    terminal = read_json(latest / "terminal.json", {}) or {}
    validity = read_json(latest / "generation-validity-v2-r7.json", {}) or {}
    ended = [name for name in ENDED if (latest / name).exists()]
    interrupted = read_json(latest / "interrupted.json", {}) or {}
    if failure.get("error_type") == "CancelledError":
        state = "cancelled"
    elif ended == ["interrupted.json"]:  # ATP's recovery after the attempt's leg was stopped: no grade, no failure
        state = "interrupted"
    elif ended:
        state = "infra"
    else:
        state = "active" if (latest / "started.json").exists() else "infra"
    stamps = [(latest / name).stat().st_mtime for name in ended]
    reason = {"failure_error_type": failure.get("error_type"), "failure_classification": failure.get("classification"),
              "verifier_error": (terminal.get("verifier_error") or "")[:300] or None,
              "evaluation_completed": terminal.get("evaluation_completed") if terminal else None,
              "generation_failure": validity.get("failure"), "budget_timeout": validity.get("budget_timeout"),
              "agent_error": terminal.get("opencode_run_error_type") or terminal.get("failure_reason") if terminal else None,
              "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(max(stamps))) if stamps else None,
              "interrupted_reason": interrupted.get("reason"), "started_at": stamp_of(latest / "started.json"),
              "last_file_change": last_change(latest)}
    refused = False
    journal = latest / "http.jsonl"
    if journal.exists():
        with open(journal, "rb") as stream:
            for raw in stream:
                head = raw[:3000]
                if b'"event": "response"' not in head and b'"event":"response"' not in head:
                    continue
                try:
                    event = json.loads(raw)
                except ValueError:
                    continue
                if event.get("status") in (400, 401, 403, 404, 422) and BUDGET_REFUSAL.search(event.get("body") or ""):
                    refused = True
                    break
    cost = sum_costs([attempt_cost(str(attempt), slot["model"], slot["journal"]) for attempt in attempts])
    sandbox = None
    if state == "active":  # no end marker: the sandbox it ran in (descriptor only; the file holds no credential)
        sandbox = ((read_json(latest / "agent-sandbox.json", {}) or {}).get("descriptor") or {}).get("sandbox_id")
    return {"state": state, "attempts": len(attempts), "attempt_dir": str(latest), "reason": reason,
            "budget_refusal": refused, "cost": cost, "sandbox_id": sandbox}


def partial_slots(model: str, name: str, waves: dict, journal: bool) -> tuple:
    """([slot], probe dir exists) of a wave campaign without a results file."""
    probe = ROOT / "probes" / model / name
    manifest = read_json(probe / "manifest.json")
    task_ids = campaign_tasks(waves, name)
    index_of = {}
    if manifest:
        index_of = {task["task_id"]: task["_ng_task_index"] for task in manifest["tasks"]}
        assert set(index_of) == set(task_ids), (model, name, "manifest tasks differ from driver.py's task list")
        assert manifest["settings"]["repeats"] == REPEATS, (model, name, manifest["settings"]["repeats"])
        assert list(manifest["settings"]["models"]) == [model], (model, name, list(manifest["settings"]["models"]))
    slots = []
    for task_id in task_ids:
        for repeat in range(REPEATS):
            slot_dir = probe / "models" / model / "attempts" / f"{index_of[task_id]}-{repeat}" if manifest else None
            slots.append({"model": model, "campaign": name, "task_id": task_id, "repeat": repeat,
                          "slot": f"{model}-r{repeat}", "slot_dir": str(slot_dir) if slot_dir else None,
                          "journal": journal})
    return slots, bool(manifest)


def load_atp_states() -> tuple:
    """{(model, campaign, task_id, repeat): ATP slot state} from final/atp_states.py's output, and its metadata."""
    path = ROOT / ATP_STATES
    data = read_json(path)
    if not data:
        return {}, None
    out = {}
    for key, campaign in data["campaigns"].items():
        model, name = key.split("/", 1)
        for slot in campaign["slots"]:
            out[(model, name, slot["task_id"], slot["repeat"])] = slot
    meta = {"file": str(ATP_STATES), "sha256": sha256_file(path), "created_at": data.get("created_at"),
            "probes_unchanged": data.get("probes_unchanged"), "files_checked": data.get("files_checked"),
            "would_write": len(data.get("would_write") or []), "campaigns": sorted(data["campaigns"])}
    return out, meta


# ---------------------------------------------------------------- helpers for the report
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(a, b):
    return f"{100.0 * a / b:.1f}" if b else "-"


def money(value):
    return f"${value:,.2f}" if value is not None else "-"


def summarize(entries: list, base: dict) -> dict:
    """Numbers of a set of wave slots (slot ledger entries)."""
    graded = [e for e in entries if e["status"] in ("pass", "fail")]
    passes = sum(e["status"] == "pass" for e in graded)
    infra = [e for e in entries if e["state"] in INFRA_STATES]
    cancelled = [e for e in entries if e["state"] == "cancelled"]
    interrupted = [e for e in entries if e["state"] == "interrupted"]
    in_flight = [e for e in entries if e["state"] == "cancelled_in_flight"]
    not_started = [e for e in entries if e["state"] == "not_started"]
    active = [e for e in entries if e["state"] == "active"]
    clean = [e for e in graded if not e.get("ended_by_broken_reply")]
    no_timeout = [e for e in graded if not e.get("ended_by_agent_timeout")]
    by_task = collections.defaultdict(list)
    for e in entries:
        by_task[e["task_id"]].append(e)
    task_pass = {t: sum(x["status"] == "pass" for x in v) for t, v in by_task.items()}
    task_graded = {t: sum(x["status"] in ("pass", "fail") for x in v) for t, v in by_task.items()}
    started_tasks = [t for t in by_task if task_graded[t] > 0]
    complete_tasks = [t for t in by_task if task_graded[t] == REPEATS]
    graded_cost = [e["cost_usd"] for e in graded if e.get("cost_usd") is not None]
    graded_slot_cost = [e["slot_cost_usd"] for e in graded if e.get("slot_cost_usd") is not None]
    all_cost = sum(e.get("slot_cost_usd") or 0.0 for e in entries)
    base_rate = (sum((base.get(e["task_id"]) or {}).get("base_k10") or 0 for e in graded) / (10.0 * len(graded))
                 if graded else None)
    task_means = [task_pass[t] / task_graded[t] for t in started_tasks]
    tasks_any_pass = sum(task_pass[t] > 0 for t in started_tasks)
    return {
        "planned": len(entries), "graded": len(graded), "pass": passes, "fail": len(graded) - passes,
        "infra": len(infra), "infra_needs_review": sum(e.get("atp_status") == "needs_review" for e in infra),
        "cancelled": len(cancelled), "interrupted": len(interrupted), "cancelled_in_flight": len(in_flight),
        "stopped": len(cancelled) + len(interrupted) + len(in_flight),
        "not_started": len(not_started), "active": len(active),
        "retried_slots": sum((e.get("attempts") or 0) > 1 for e in entries),
        "pass_at_1_raw": pct(passes, len(graded) + len(infra)),
        "pass_at_1_excl_infra": pct(passes, len(graded)),
        "pass_at_1_excl_broken": pct(sum(e["status"] == "pass" for e in clean), len(clean)),
        "broken_ended": len(graded) - len(clean),
        "agent_timeouts": len(graded) - len(no_timeout),
        "agent_timeouts_passed": sum(e["status"] == "pass" for e in graded if e.get("ended_by_agent_timeout")),
        "pass_at_1_excl_timeouts": pct(sum(e["status"] == "pass" for e in no_timeout), len(no_timeout)),
        "runaways": sum(bool(e.get("ended_by_output_cap_runaway")) for e in graded),
        "compacted": sum(bool(e.get("compactions")) for e in graded),
        "pass_no_compaction": sum(e["status"] == "pass" and not e.get("compactions") for e in graded),
        "pass_clean": sum(e["status"] == "pass" and not e.get("ended_by_broken_reply")
                          and not e.get("ended_by_agent_timeout") for e in graded),
        "compaction_rate": pct(sum(bool(e.get("compactions")) for e in graded), len(graded)),
        "tasks": len(by_task), "tasks_started": len(started_tasks), "tasks_complete": len(complete_tasks),
        "tasks_any_pass": tasks_any_pass, "pass_at_5_rate": pct(tasks_any_pass, len(started_tasks)),
        "tasks_complete_any_pass": sum(task_pass[t] > 0 for t in complete_tasks),
        "tasks_complete_all_pass": sum(task_pass[t] == REPEATS for t in complete_tasks),
        "task_mean_pass_rate": pct(sum(task_means), len(task_means)),
        "base_pass_at_1_same_slots": pct(base_rate, 1) if base_rate is not None else "-",
        "cost_graded": round(sum(graded_cost), 2),
        "cost_per_graded_rollout": round(statistics.mean(graded_cost), 3) if graded_cost else None,
        "cost_per_graded_slot_all_attempts": round(statistics.mean(graded_slot_cost), 3) if graded_slot_cost else None,
        "cost_all_attempts": round(all_cost, 2),
        "cost_all_per_graded": round(all_cost / len(graded), 3) if graded else None}


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--no-journal-cost", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="read the probe dir of every wave campaign without a results file (partial collection)")
    parser.add_argument("--root", type=Path, help="tests only: read probes/ and results/ and write the outputs there")
    args = parser.parse_args()
    if args.root:
        ROOT = args.root.resolve()
    journal = not args.no_journal_cost
    started_at = time.time()
    waves = json.loads((HERE / "tasks" / "waves.json").read_text())
    base = json.loads((HERE / "tasks" / "base_k10.json").read_text())["tasks"]
    tasks = {t["task_id"]: {"task_id": t["task_id"], "task_index": t["task_index"], "language": t["language"],
                            "category": t["category"], "wave": t["wave"],
                            "covered_by_accepted_synth": t["covered_by_accepted_synth"],
                            "base_k10": (base.get(t["task_id"]) or {}).get("base_k10")} for t in waves["tasks"]}
    results, missing = {}, []
    for model in MODELS:
        for name in CAMPAIGNS + ("pilot",):
            data = read_json(ROOT / "results" / f"{model}-{name}.json")
            if data is None:
                missing.append(f"{model}-{name}")
                continue
            assert data["model"] == model and data.get("campaign", name) in (name, None) or args.root, (model, name)
            results[(model, name)] = data
    wave_missing = [m for m in missing if not m.endswith("-pilot")]
    if wave_missing and not args.allow_incomplete:
        raise SystemExit(f"campaigns without results: {wave_missing} (use --allow-incomplete for a partial collection)")
    states, states_meta = load_atp_states()
    out_dir = ROOT / "dataset"
    out_dir.mkdir(exist_ok=True)
    index_rows, ledger, counts, checks = [], [], {}, collections.defaultdict(list)
    for key in CHECK_KEYS:
        checks[key] = []
    campaign_state = {}
    with ProcessPoolExecutor(args.workers) as pool:
        # 1. the slots of every wave campaign without a results file, viewed in parallel
        partial = {}
        for model in MODELS:
            for name in CAMPAIGNS:
                if (model, name) in results:
                    campaign_state[(model, name)] = "complete"
                    continue
                slots, exists = partial_slots(model, name, waves, journal)
                views = list(pool.map(own_view, slots, chunksize=8))
                cancelled_by_user = model in USER_CANCELLED
                campaign_state[(model, name)] = ("not_started" if not exists else
                                                 "cancelled_by_user" if cancelled_by_user else "stopped")
                if exists and cancelled_by_user and not (HERE / "babysit" / f"{model}-{name}" / "STOP").exists():
                    checks["user_cancel_stop_missing"].append(f"{model}-{name}")
                for slot, view in zip(slots, views):
                    atp = states.get((model, name, slot["task_id"], slot["repeat"]))
                    if view["state"] == "active" and cancelled_by_user:  # cut off by the user's cancel, never graded
                        view = dict(view, state="cancelled_in_flight")
                        started = (view.get("reason") or {}).get("started_at")
                        if not started or parse_stamp(started) >= parse_stamp(USER_CANCELLED[model]["cancelled_at"]):
                            checks["in_flight_started_after_cancel"].append(view["attempt_dir"])
                    elif view["state"] == "active":
                        checks["active_outside_cancel"].append(view["attempt_dir"])
                    if exists and states_meta and name not in ("pilot",):
                        if atp is None:
                            checks["atp_state_missing"].append(f"{model}/{name} {slot['task_id']} r{slot['repeat']}")
                        elif (atp["status"] == "completed") != (view["state"] == "graded") or (
                                atp["status"] == "completed" and atp["attempt_dir"] != view.get("attempt_dir")):
                            checks["atp_view_disagree"].append(
                                f"{model}/{name} {slot['task_id']} r{slot['repeat']}: ATP {atp['status']} "
                                f"{atp['attempt_dir']} vs view {view['state']} {view.get('attempt_dir')}")
                        elif atp["status"] == "pending" and view["state"] != "not_started":
                            checks["atp_view_disagree"].append(
                                f"{model}/{name} {slot['task_id']} r{slot['repeat']}: ATP pending vs view {view['state']}")
                        elif (view["state"] == "cancelled_in_flight") != (atp["status"] == "active"):
                            checks["atp_view_disagree"].append(
                                f"{model}/{name} {slot['task_id']} r{slot['repeat']}: ATP {atp['status']} vs view "
                                f"{view['state']}")
                    if view["state"] == "graded" and not view.get("validity_cached"):
                        checks["graded_without_validity_cache"].append(view["attempt_dir"])
                    if view.get("budget_refusal"):
                        checks["budget_refusals"].append(view["attempt_dir"])
                    slot.update(view=view, atp=atp)
                partial[(model, name)] = slots

        # 2. dataset rows (graded / final slots) with transcripts
        for model in MODELS + ("pilots",):
            if model == "pilots":
                jobs = [job for m in MODELS if (m, "pilot") in results
                        for job in jobs_of(m, "pilot", results[(m, "pilot")], tasks, journal)]
            else:
                jobs = []
                for name in CAMPAIGNS:
                    if (model, name) in results:
                        jobs += jobs_of(model, name, results[(model, name)], tasks, journal)
                        continue
                    for slot in partial[(model, name)]:
                        if slot["view"]["state"] != "graded":
                            continue
                        record = {"slot": slot["slot"], "status": slot["view"]["status"],
                                  "attempt_dir": slot["view"]["attempt_dir"]}
                        jobs.append({"model": model, "campaign": name, "complete": False, "record": record,
                                     "first": record, "task": task_meta(tasks, slot["task_id"]), "journal": journal,
                                     "model_id": MODEL_IDS[model]})
                jobs.sort(key=lambda j: (j["task"]["task_index"] if j["task"]["task_index"] is not None else 10 ** 6,
                                         j["task"]["task_id"], j["record"]["slot"]))
            path = out_dir / f"{model}.jsonl.gz"
            tmp = path.with_name("." + path.name + ".tmp")
            rows = 0
            with gzip.open(tmp, "wt", encoding="utf-8") as stream:
                for start in range(0, len(jobs), 4 * args.workers):  # bounded: rows carry whole transcripts
                    for job, row in zip(jobs[start:start + 4 * args.workers],
                                        pool.map(build_row, jobs[start:start + 4 * args.workers])):
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        compact = {key: value for key, value in row.items() if key != "output"} | {
                            "file": path.name, "output_items": len(row["output"] or [])}
                        index_rows.append(compact)
                        if not row["output"]:
                            checks["dataset_rows_without_output"].append(row["attempt_dir"])
                        rows += 1
            os.replace(tmp, path)
            counts[model] = rows
            print(json.dumps({"file": path.name, "rows": rows, "elapsed_s": round(time.time() - started_at)}),
                  flush=True)

    for key in GRADED_FAIL_OVERRIDES:  # every override must have been applied (complete campaign, matching record)
        if key not in OVERRIDES_APPLIED:
            checks["status_overrides_unmatched"].append(
                f"{key[0]}/{key[1]} {key[2]} r{key[3]}: " + OVERRIDE_PROBLEMS.get(
                    key, "not applied (the campaign has no results file, or the slot is not in it)"))

    # 3. the slot ledger: every planned slot (+ the pilots)
    by_attempt = {row["attempt_dir"]: row for row in index_rows}
    for model in MODELS:
        for name in CAMPAIGNS:
            state = campaign_state[(model, name)]
            if state == "complete":
                for row in [r for r in index_rows if r["model"] == model and r["campaign"] == name
                            and r["file"] != "pilots.jsonl.gz"]:
                    ledger.append(ledger_entry(row, "final" if row["status"] in ("pass", "fail") else "final_infra",
                                               state, None))
                continue
            for slot in partial[(model, name)]:
                view, atp = slot["view"], slot["atp"]
                if view["state"] == "graded":
                    ledger.append(ledger_entry(by_attempt[view["attempt_dir"]], "graded", state, atp))
                    continue
                meta = task_meta(tasks, slot["task_id"])
                cost = view.get("cost") or sum_costs([])
                ledger.append({
                    "model": model, "model_id": MODEL_IDS[model], "campaign": name, "campaign_state": state,
                    "wave": meta["wave"], "task_id": slot["task_id"], "task_index": meta["task_index"],
                    "language": meta["language"], "base_k10": meta["base_k10"], "repeat": slot["repeat"],
                    "slot": slot["slot"], "state": view["state"], "status": None,
                    "atp_status": (atp or {}).get("status"), "atp_retryable": (atp or {}).get("retryable"),
                    "attempts": view["attempts"], "attempt_dir": view.get("attempt_dir"),
                    "reason": view.get("reason"), "budget_refusal": view.get("budget_refusal"), "in_dataset": False,
                    "sandbox_id": view.get("sandbox_id"),
                    "cost_usd": None, "slot_cost_usd": cost["usd"], "cost_by_key": cost["by_key"],
                    "prompt_tokens": cost["prompt"], "cached_tokens": cost["cached"],
                    "completion_tokens": cost["completion"], "hub_calls_ok": cost["calls_ok"],
                    "hub_errors": cost["errors"]})
    for row in [r for r in index_rows if r["file"] == "pilots.jsonl.gz"]:
        ledger.append(ledger_entry(row, "final", "complete", None))
    ledger_path = out_dir / "slots.jsonl"
    tmp = ledger_path.with_name(".slots.jsonl.tmp")
    tmp.write_text("".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in ledger))
    os.replace(tmp, ledger_path)
    index = out_dir / "index.jsonl"
    tmp = index.with_name(".index.jsonl.tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in index_rows))
    os.replace(tmp, index)

    model_status = {}
    for model in MODELS:
        mine = [campaign_state[(model, name)] for name in CAMPAIGNS]
        model_status[model] = ("complete" if all(s == "complete" for s in mine) else
                               "partial_cancelled_by_user" if "cancelled_by_user" in mine else
                               "not_started" if all(s == "not_started" for s in mine) else "partial")

    # 4. checks (slot counts per model per task, dataset vs ledger)
    wave_ledger = [e for e in ledger if e["campaign"] != "pilot"]
    per_task_model = collections.Counter((e["model"], e["task_id"]) for e in wave_ledger)
    for model in MODELS:
        for task_id in tasks:
            if per_task_model[(model, task_id)] != REPEATS:
                checks["slots_per_task_not_5"].append(f"{model} {task_id}: {per_task_model[(model, task_id)]}")
    keys = collections.Counter((e["model"], e["campaign"], e["task_id"], e["repeat"]) for e in wave_ledger)
    checks["duplicate_slots"] = [str(k) for k, v in keys.items() if v > 1]
    for entry in wave_ledger:
        if entry["in_dataset"] != (entry["status"] in ("pass", "fail") or entry["state"] == "final_infra"):
            checks["dataset_ledger_mismatch"].append(f"{entry['model']}/{entry['campaign']} {entry['task_id']} "
                                                     f"r{entry['repeat']}")
        if entry["state"] == "graded" and entry.get("attempt_dir") and campaign_of(entry["attempt_dir"]) != entry["campaign"]:
            checks["attempt_outside_campaign"].append(entry["attempt_dir"])
    for row in index_rows:
        if row["status"] not in ("pass", "fail") and not row.get("campaign_complete"):
            checks["ungraded_dataset_rows"].append(row["attempt_dir"])
        if row["status"] == "pass" and row.get("reward") != 1:
            checks["pass_without_reward_1"].append(row["attempt_dir"])
        if row["status"] == "fail" and row.get("reward") == 1:
            checks["fail_with_reward_1"].append(row["attempt_dir"])
    checks = {key: value for key, value in checks.items()}
    check_summary = {key: len(value) for key, value in checks.items()}
    complete = all(state == "complete" for state in campaign_state.values())
    manifest = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "complete": complete,
                "collection": "complete" if complete else "PARTIAL", "model_status": model_status,
                "user_cancelled": {m: USER_CANCELLED[m] for m in MODELS if model_status[m] == "partial_cancelled_by_user"},
                "campaign_states": {f"{m}-{n}": s for (m, n), s in campaign_state.items()},
                "rows": counts, "slots_ledger_rows": len(ledger), "missing_results": missing,
                "status_overrides": list(OVERRIDES_APPLIED.values()),
                "cost_basis": "journal" if journal else "export", "atp_states": states_meta,
                "checks": check_summary,
                "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                          for p in sorted(out_dir.iterdir()) if p.is_file() and not p.name.startswith(".")
                          and p.name != "MANIFEST.json"}}
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")

    # ---------------------------------------------------------------- per_task.csv
    by_model_task = collections.defaultdict(list)
    for entry in wave_ledger:
        by_model_task[(entry["model"], entry["task_id"])].append(entry)
    header = ["task_id", "task_index", "language", "wave", "covered_by_accepted_synth", "base_k10"]
    for model in MODELS:
        header += [f"{model}_{c}" for c in ("planned", "graded", "pass", "complete", "k5", "cancelled", "interrupted",
                                             "cancelled_in_flight", "infra", "not_started", "broken_first",
                                             "broken_final", "runaway", "agent_timeout", "compacted", "model_status")]
    table = []
    for task_id in sorted(tasks, key=lambda t: tasks[t]["task_index"]):
        meta = tasks[task_id]
        line = {"task_id": task_id, "task_index": meta["task_index"], "language": meta["language"], "wave": meta["wave"],
                "covered_by_accepted_synth": meta["covered_by_accepted_synth"], "base_k10": meta["base_k10"]}
        for model in MODELS:
            slots = by_model_task.get((model, task_id), [])
            graded = [e for e in slots if e["status"] in ("pass", "fail")]
            passes = sum(e["status"] == "pass" for e in graded)
            done = len(graded) == REPEATS
            line.update({f"{model}_planned": len(slots), f"{model}_graded": len(graded), f"{model}_pass": passes,
                         f"{model}_complete": int(done), f"{model}_k5": passes if done else "",
                         f"{model}_cancelled": sum(e["state"] == "cancelled" for e in slots),
                         f"{model}_interrupted": sum(e["state"] == "interrupted" for e in slots),
                         f"{model}_cancelled_in_flight": sum(e["state"] == "cancelled_in_flight" for e in slots),
                         f"{model}_model_status": model_status[model],
                         f"{model}_infra": sum(e["state"] in ("infra", "final_infra") for e in slots),
                         f"{model}_not_started": sum(e["state"] == "not_started" for e in slots),
                         f"{model}_broken_first": sum(bool(e.get("first_attempt_ended_by_broken_reply")) for e in graded),
                         f"{model}_broken_final": sum(bool(e.get("ended_by_broken_reply")) for e in graded),
                         f"{model}_runaway": sum(bool(e.get("ended_by_output_cap_runaway")) for e in graded),
                         f"{model}_agent_timeout": sum(bool(e.get("ended_by_agent_timeout")) for e in graded),
                         f"{model}_compacted": sum(bool(e.get("compactions")) for e in graded)})
        table.append(line)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(table)
    (ROOT / "per_task.csv").write_text(buffer.getvalue())

    # ---------------------------------------------------------------- numbers
    summary = {"overview": {}, "by_wave": {}, "by_campaign": {}, "by_language": {}, "matched": {}, "uncovered": {},
               "kimi_vs_1m": {}, "cost": {}, "checks": check_summary, "complete": complete,
               "model_status": model_status, "user_cancelled": {}}
    for model in MODELS:
        if model_status[model] == "partial_cancelled_by_user":
            summary["user_cancelled"][model] = dict(USER_CANCELLED[model], stop_files={
                name: stamp_of(HERE / "babysit" / f"{model}-{name}" / "STOP") for name in CAMPAIGNS})
    summary["rerun_dirs"] = sorted(f"{p.parent.name}/{p.name}" for model in MODELS
                                   if (ROOT / "probes" / model).is_dir()
                                   for p in (ROOT / "probes" / model).iterdir() if p.is_dir()
                                   and re.search(r"-(brk|rec)\d*$", p.name))
    summary["results_provenance"] = {}
    for (model, name), data in results.items():
        if name == "pilot":
            continue
        firsts = {(t, r["slot"]): r.get("attempt_dir") for t, recs in (data.get("first_attempt_records") or {}).items()
                  for r in recs}
        finals = {(t, r["slot"]): r.get("attempt_dir") for t, recs in data["records"].items() for r in recs}
        summary["results_provenance"][(model, name)] = {
            "slots": len(finals), "same_as_first": sum(firsts.get(k) == v for k, v in finals.items()),
            "broken_reply_reruns": data.get("broken_reply_reruns") or {},
            "recovered": data.get("recovered_budget_refusals") or {},
            "stats_final": data.get("stats_final"), "hub_keys_by_leg": data.get("hub_keys_by_leg")}
    for model in MODELS:
        mine = [e for e in wave_ledger if e["model"] == model]
        summary["overview"][model] = summarize(mine, base)
        for wave in (1, 2):
            summary["by_wave"][f"{model} w{wave}"] = summarize([e for e in mine if e["wave"] == wave], base)
        for name in CAMPAIGNS:
            summary["by_campaign"][f"{model} {name}"] = summarize([e for e in mine if e["campaign"] == name], base) | {
                "state": campaign_state[(model, name)]}
    languages = sorted({t["language"] for t in tasks.values() if t["language"]})
    for language in languages:
        summary["by_language"][language] = {
            model: summarize([e for e in wave_ledger if e["model"] == model and e["language"] == language], base)
            for model in MODELS}
    graded_tasks = {model: {t for (m, t), v in by_model_task.items() if m == model
                            and any(e["status"] in ("pass", "fail") for e in v)} for model in MODELS}
    complete_tasks = {model: {t for (m, t), v in by_model_task.items() if m == model
                              and sum(e["status"] in ("pass", "fail") for e in v) == REPEATS} for model in MODELS}
    for label, chosen in (("all three models have >= 1 graded slot", set.intersection(*graded_tasks.values())),
                          ("all three models have all 5 slots graded", set.intersection(*complete_tasks.values()))):
        summary["matched"][label] = {"tasks": len(chosen), "per_model": {
            model: summarize([e for e in wave_ledger if e["model"] == model and e["task_id"] in chosen], base)
            for model in MODELS}, "task_ids": sorted(chosen, key=lambda t: tasks[t]["task_index"])}

    # uncovered (wave 1) tasks with base <= 2/10: which teacher reaches >= 3 of 5
    uncovered = [t for t in waves["wave1"] if (tasks[t]["base_k10"] is not None and tasks[t]["base_k10"] <= 2)]
    verdicts = {}
    for task_id in uncovered:
        verdicts[task_id] = {}
        for model in MODELS:
            slots = by_model_task.get((model, task_id), [])
            graded = sum(e["status"] in ("pass", "fail") for e in slots)
            passes = sum(e["status"] == "pass" for e in slots)
            # only an unfinished campaign's ungraded slots leave the verdict open (a complete campaign's final infra
            # slot is final); GLM's are cut off by the user's cancel, so "undetermined"
            left = sum(e["status"] not in ("pass", "fail") and e["campaign_state"] != "complete" for e in slots)
            verdicts[task_id][model] = {"pass": passes, "graded": graded, "ungraded_unfinished": left,
                                        "verdict": "yes" if passes >= 3 else "undetermined" if passes + left >= 3
                                        else "no"}
    per_teacher = {model: {v: [t for t in uncovered if verdicts[t][model]["verdict"] == v]
                           for v in ("yes", "no", "undetermined")} for model in MODELS}
    any_yes = [t for t in uncovered if any(verdicts[t][m]["verdict"] == "yes" for m in MODELS)]
    any_open = [t for t in uncovered if t not in any_yes
                and any(verdicts[t][m]["verdict"] == "undetermined" for m in MODELS)]
    complete_teachers = [m for m in MODELS if model_status[m] == "complete"]
    summary["uncovered"] = {"definition": "wave-1 tasks (no accepted synthetic task) with base_k10 <= 2; a teacher "
                                          "reaches >= 3/5 = at least 3 passing slots among its 5; yes = >= 3 passes; "
                                          "no = < 3 even if every ungraded slot of an unfinished campaign had passed; "
                                          "undetermined = neither (GLM: cancelled by the user)",
                            "tasks": len(uncovered), "task_ids": uncovered,
                            "per_teacher": {m: {v: len(ids) for v, ids in d.items()} for m, d in per_teacher.items()},
                            "per_teacher_yes": {m: d["yes"] for m, d in per_teacher.items()},
                            "per_teacher_undetermined": {m: d["undetermined"] for m, d in per_teacher.items()},
                            "complete_teachers": complete_teachers,
                            "complete_teachers_yes": [t for t in uncovered if any(
                                verdicts[t][m]["verdict"] == "yes" for m in complete_teachers)],
                            "any_teacher_yes": any_yes, "any_teacher_undetermined_only": any_open,
                            "no_teacher_yes": [t for t in uncovered if t not in any_yes], "verdicts": verdicts}

    # Kimi vs the 1M-context Kimi export (context only)
    k1m = {}
    if KIMI_1M.exists():
        for line in csv.DictReader(open(KIMI_1M)):
            k1m[line["task_id"]] = {"correct": int(line["kimi_correct"]), "scored": int(line["kimi_scored"])}
    rows_1m = []
    for task_id in sorted(graded_tasks["kimi"], key=lambda t: tasks[t]["task_index"]):
        slots = by_model_task[("kimi", task_id)]
        graded = sum(e["status"] in ("pass", "fail") for e in slots)
        passes = sum(e["status"] == "pass" for e in slots)
        ref = k1m.get(task_id)
        rows_1m.append({"task_id": task_id, "task_index": tasks[task_id]["task_index"], "wave": tasks[task_id]["wave"],
                        "language": tasks[task_id]["language"], "base_k10": tasks[task_id]["base_k10"],
                        "kimi256k_graded": graded, "kimi256k_pass": passes,
                        "kimi256k_timeouts": sum(bool(e.get("ended_by_agent_timeout")) for e in slots
                                                 if e["status"] in ("pass", "fail")),
                        "kimi256k_rate": round(passes / graded, 4),
                        "kimi1m_correct": ref["correct"] if ref else "", "kimi1m_scored": ref["scored"] if ref else "",
                        "kimi1m_rate": round(ref["correct"] / ref["scored"], 4) if ref and ref["scored"] else ""})
    if rows_1m:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows_1m[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_1m)
        (ROOT / "analysis").mkdir(exist_ok=True)
        (ROOT / "analysis" / "collect-kimi-vs-1m.csv").write_text(buffer.getvalue())
    paired = [r for r in rows_1m if r["kimi1m_scored"] != ""]
    weight = sum(r["kimi256k_graded"] for r in paired)
    full = [r for r in paired if r["kimi256k_graded"] == REPEATS]
    summary["kimi_vs_1m"] = {
        "source": str(KIMI_1M), "note": KIMI_1M_NOTE, "tasks_with_kimi_graded": len(paired),
        "slots": weight, "kimi256k_pass": sum(r["kimi256k_pass"] for r in paired),
        "kimi256k_pass_at_1": pct(sum(r["kimi256k_pass"] for r in paired), weight),
        "kimi1m_same_slots": pct(sum(r["kimi256k_graded"] * r["kimi1m_rate"] for r in paired), weight),
        "kimi256k_task_mean": pct(sum(r["kimi256k_rate"] for r in paired), len(paired)),
        "slots_excl_timeouts": sum(r["kimi256k_graded"] - r["kimi256k_timeouts"] for r in paired),
        "kimi256k_pass_at_1_excl_timeouts": pct(sum(r["kimi256k_pass"] for r in paired),
                                                sum(r["kimi256k_graded"] - r["kimi256k_timeouts"] for r in paired)),
        "kimi1m_same_slots_excl_timeouts": pct(sum((r["kimi256k_graded"] - r["kimi256k_timeouts"]) * r["kimi1m_rate"]
                                                   for r in paired),
                                               sum(r["kimi256k_graded"] - r["kimi256k_timeouts"] for r in paired)),
        "kimi1m_task_mean": pct(sum(r["kimi1m_rate"] for r in paired), len(paired)),
        "tasks_256k_better": sum(r["kimi256k_rate"] > r["kimi1m_rate"] for r in paired),
        "tasks_256k_worse": sum(r["kimi256k_rate"] < r["kimi1m_rate"] for r in paired),
        "tasks_tied": sum(r["kimi256k_rate"] == r["kimi1m_rate"] for r in paired),
        "complete_tasks": len(full),
        "complete_256k_avg": pct(sum(r["kimi256k_rate"] for r in full), len(full)),
        "complete_1m_avg": pct(sum(r["kimi1m_rate"] for r in full), len(full)),
        "complete_256k_any_pass": sum(r["kimi256k_pass"] > 0 for r in full),
        "complete_1m_any_pass": sum(r["kimi1m_correct"] > 0 for r in full),
        "kimi1m_all_tasks_avg": pct(sum(v["correct"] for v in k1m.values()), sum(v["scored"] for v in k1m.values())),
        "kimi1m_all_tasks_any_pass": sum(v["correct"] > 0 for v in k1m.values()), "kimi1m_tasks": len(k1m),
        "kimi1m_wave1_avg": pct(sum(k1m[t]["correct"] for t in waves["wave1"] if t in k1m),
                                sum(k1m[t]["scored"] for t in waves["wave1"] if t in k1m))}
    for wave in (1, 2):
        mine = [r for r in paired if r["wave"] == wave]
        summary["kimi_vs_1m"][f"wave{wave}_256k"] = pct(sum(r["kimi256k_pass"] for r in mine),
                                                         sum(r["kimi256k_graded"] for r in mine))
        summary["kimi_vs_1m"][f"wave{wave}_1m_same_tasks"] = pct(sum(r["kimi1m_correct"] for r in mine),
                                                                  sum(r["kimi1m_scored"] for r in mine))
    summary["kimi_vs_1m"].update({
        "kimi1m_wave2_avg": pct(sum(k1m[t]["correct"] for t in waves["wave2"] if t in k1m),
                                sum(k1m[t]["scored"] for t in waves["wave2"] if t in k1m)),
        "kimi1m_correct_total": sum(v["correct"] for v in k1m.values()),
        "kimi1m_scored_total": sum(v["scored"] for v in k1m.values()),
        "all5_256k": sum(r["kimi256k_pass"] == REPEATS for r in full),
        "all5_1m": sum(r["kimi1m_correct"] == REPEATS for r in full),
        "uncovered_yes_256k": [t for t in uncovered if verdicts[t]["kimi"]["verdict"] == "yes"],
        "uncovered_yes_1m": [t for t in uncovered if t in k1m and k1m[t]["correct"] >= 3]})

    # cost by model and key (every attempt: graded, cancelled, infra; pilots apart)
    for model in MODELS + ("pilots",):
        entries = ([e for e in ledger if e["campaign"] == "pilot"] if model == "pilots"
                   else [e for e in wave_ledger if e["model"] == model])
        by_key = collections.Counter()
        errors = collections.Counter()
        for e in entries:
            by_key.update(e.get("cost_by_key") or {})
            errors.update(e.get("hub_errors") or {})
        graded = [e for e in entries if e["status"] in ("pass", "fail")]
        summary["cost"][model] = {
            "all_attempts": round(sum(e.get("slot_cost_usd") or 0.0 for e in entries), 2),
            "graded_final_attempts": round(sum(e.get("cost_usd") or 0.0 for e in graded), 2),
            "ungraded_attempts": round(sum(e.get("slot_cost_usd") or 0.0 for e in entries
                                           if e["status"] not in ("pass", "fail")), 2),
            "by_key": {key: round(value, 2) for key, value in by_key.items()},
            "prompt_tokens": sum(e.get("prompt_tokens") or 0 for e in entries),
            "cached_tokens": sum(e.get("cached_tokens") or 0 for e in entries),
            "completion_tokens": sum(e.get("completion_tokens") or 0 for e in entries),
            "hub_calls_ok": sum(e.get("hub_calls_ok") or 0 for e in entries), "hub_errors": dict(errors),
            "per_pilot": {e["model"]: round(e.get("slot_cost_usd") or 0.0, 2) for e in entries}
            if model == "pilots" else None}
    budget_log = [json.loads(line) for line in (HERE / "budget-log.jsonl").read_text().splitlines() if line.strip()]
    new3 = [b for b in budget_log if b.get("key") == "NVAPI_KEY_NEW_3" and b.get("remaining") is not None]
    summary["new3_last_reading"] = ({k: new3[-1].get(k) for k in ("at", "remaining", "spend", "max_budget", "tag")}
                                    if new3 else None)
    first_leg = [b for b in new3 if str(b.get("tag", "")).startswith("start ")]
    summary["new3_at_first_leg"] = ({k: min(first_leg, key=lambda b: b["remaining"]).get(k)
                                     for k in ("at", "remaining", "spend")} if first_leg else None)
    fields = ("at", "remaining", "spend", "max_budget", "tag")
    new2 = [b for b in budget_log if b.get("key") == "NVAPI_KEY_NEW_2" and b.get("remaining") is not None
            and b.get("spend") is not None and b.get("at")]
    before = [b for b in new2 if parse_stamp(b["at"]) < parse_stamp(NEW2_RESET_AFTER)]
    after = [b for b in new2 if parse_stamp(b["at"]) >= parse_stamp(NEW2_RESET_AFTER)]
    summary["new2_readings"] = {
        "first": {k: new2[0].get(k) for k in fields} if new2 else None,
        "last_before_reset": {k: before[-1].get(k) for k in fields} if before else None,
        "first_after_reset": {k: after[0].get(k) for k in fields} if after else None,
        "last": {k: new2[-1].get(k) for k in fields} if new2 else None,
        "max_spend_after_reset": max((b["spend"] for b in after), default=None)}
    legs = []
    for model in MODELS:
        for name in CAMPAIGNS + ("pilot",):
            for start, job, key in _module("costs").legs_of(model, name):
                legs.append({"model": model, "campaign": name, "job": job, "key": key, "start": start,
                             "at": time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(start))})
    summary["legs"] = [{k: v for k, v in leg.items() if k != "start"} for leg in sorted(legs, key=lambda x: x["start"])]
    summary["legs_by_key"] = {}
    for key in sorted({leg["key"] for leg in legs}):
        mine = sorted((leg for leg in legs if leg["key"] == key), key=lambda x: x["start"])
        summary["legs_by_key"][key] = {"legs": len(mine), "pilot_legs": sum(x["campaign"] == "pilot" for x in mine),
                                       "wave_legs": sum(x["campaign"] != "pilot" for x in mine),
                                       "first": mine[0]["at"], "last": mine[-1]["at"]}

    # ---------------------------------------------------------------- FINAL-REPORT.md
    lines = render_report(summary, campaign_state, tasks, waves, manifest, counts, checks, ledger, by_model_task,
                          per_teacher, any_yes, any_open, verdicts, uncovered, rows_1m, len(table), states_meta)
    (ROOT / "FINAL-REPORT.md").write_text("\n".join(lines) + "\n")
    (ROOT / "analysis").mkdir(exist_ok=True)
    (ROOT / "analysis" / "collect-summary.json").write_text(json.dumps(
        summary | {"manifest": manifest, "checks_detail": checks,
                   "results_provenance": {f"{m}-{n}": v for (m, n), v in summary["results_provenance"].items()}},
        indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": counts, "ledger": len(ledger), "complete": complete, "model_status": model_status,
                      "checks": check_summary,
                      "overview": {m: {k: summary["overview"][m][k] for k in ("graded", "pass", "pass_at_1_excl_infra",
                                                                             "cost_all_attempts")}
                                   for m in MODELS}, "elapsed_s": round(time.time() - started_at)}, indent=1))


def ledger_entry(row: dict, state: str, campaign_state: str, atp) -> dict:
    """A slot-ledger line from a dataset row (no transcript)."""
    keep = ("model", "model_id", "campaign", "wave", "task_id", "task_index", "language", "base_k10", "repeat", "slot",
            "status", "reward", "f2p", "p2p", "attempt_dir", "source_campaign", "ended_by_broken_reply",
            "first_attempt_ended_by_broken_reply", "first_attempt_status", "ended_by_output_cap_runaway",
            "ended_by_agent_timeout", "first_attempt_ended_by_agent_timeout", "compactions", "rerun_reason",
            "verifier_error", "status_override",
            "cost_usd", "slot_cost_usd", "cost_by_key", "prompt_tokens", "cached_tokens", "completion_tokens",
            "hub_calls_ok", "hub_errors")
    entry = {key: row.get(key) for key in keep}
    entry.update(campaign_state=campaign_state, state=state, atp_status=(atp or {}).get("status"),
                 atp_retryable=(atp or {}).get("retryable"), attempts=row.get("slot_attempts"), in_dataset=True,
                 reason=None)
    return entry


def reason_text(reason) -> str:
    """A short end reason of an ungraded slot (own_view's reason)."""
    reason = reason or {}
    parts = []
    if reason.get("verifier_error"):
        parts.append("verifier: " + reason["verifier_error"][:90])
    if reason.get("generation_failure"):
        parts.append(f"generation {reason['generation_failure']}")
    if reason.get("budget_timeout"):
        parts.append("ended by the agent timeout")
    if reason.get("failure_error_type"):
        parts.append(f"failure {reason['failure_error_type']}")
    if reason.get("ended_at"):
        parts.append(f"ended {reason['ended_at']}")
    return ", ".join(parts) or "no end detail"


def render_report(summary, campaign_state, tasks, waves, manifest, counts, checks, ledger, by_model_task, per_teacher,
                  any_yes, any_open, verdicts, uncovered, rows_1m, n_table, states_meta) -> list:
    complete = manifest["complete"]
    ov = summary["overview"]
    status = summary["model_status"]
    done = [m for m in MODELS if status[m] == "complete"]
    partial = [m for m in MODELS if status[m] != "complete"]
    cancelled_models = [m for m in MODELS if status[m] == "partial_cancelled_by_user"]
    tag = {m: ("complete" if status[m] == "complete" else "PARTIAL, cancelled by the user"
               if status[m] == "partial_cancelled_by_user" else "PARTIAL") for m in MODELS}
    label = {m: (m if status[m] == "complete" else m + " (PARTIAL)") for m in MODELS}
    wave_ledger = [e for e in ledger if e["campaign"] != "pilot"]
    title = "# DeepSWE v1.1 teacher rollouts: final report"
    if not complete:
        title += f" ({', '.join(done) or 'no model'} complete; {', '.join(partial)} PARTIAL)"
    lines = [title, "", f"_Generated by collect.py at {time.strftime('%Y-%m-%d %H:%M %Z')}. Sources: results/*.json for "
             "complete campaigns and the probe dirs of the others (read-only). Every planned slot with its state: "
             "dataset/slots.jsonl._", ""]
    for model in cancelled_models:
        info = summary["user_cancelled"][model]
        b, w1, w2 = ov[model], summary["by_wave"][f"{model} w1"], summary["by_wave"][f"{model} w2"]
        ended = b["interrupted"] + b["cancelled"] + b["infra"]
        lines += [
            f"> **{model.upper()} IS PARTIAL: CANCELLED BY THE USER.** {info['source']}: hengyuf, verbatim: "
            f"\"{info['decision']}\" {info['how']}",
            f"> - {model}: **{b['graded']} of {b['planned']} slots graded** (wave 1: {w1['graded']} of {w1['planned']}; "
            f"wave 2: {w2['graded']} of {w2['planned']}). Not graded: {b['cancelled_in_flight']} in flight at the cancel "
            f"(cut off, not graded), {ended} that had ended ungraded earlier and were never retried "
            f"({b['interrupted']} interrupted by an earlier stop, {b['cancelled']} cancelled by the 2026-10-31 "
            f"budget-floor stop, {b['infra']} infra endings), and {b['not_started']} never started.",
            f"> - Every {model} number below counts only its graded slots and is labelled PARTIAL. Within each campaign, "
            f"ATP admits slots in a fixed (task, repeat) order, so {model}'s graded slots are the early tasks of each "
            f"campaign: {b['tasks_started']} of 113 tasks have at least one graded slot and {b['tasks_complete']} have "
            f"all 5. Like-for-like numbers: \"Matched task sets\"."]
    if not complete:
        for model in done:
            b = ov[model]
            lines.append(f"> - {model}: complete (all 5 campaigns finished; {b['planned']} of {b['planned']} slots "
                         f"final: {b['graded']} graded, {b['infra']} final infra).")
        stopped = [f"{m} {n}" for (m, n), s in campaign_state.items() if s == "stopped"]
        never = [f"{m} {n}" for (m, n), s in campaign_state.items() if s == "not_started"]
        if stopped:
            lines.append(f"> - Campaigns stopped before finishing (not cancelled by the user): {', '.join(stopped)}.")
        if never:
            lines.append(f"> - Campaigns never started: {', '.join(never)}.")
        lines += ["> - A row that adds models up (\"all\") mixes complete and PARTIAL counts and says so; no pass rate "
                  "is pooled across models.", ""]
    legs = summary.get("legs_by_key") or {}
    leg_text = "; ".join(f"{key}: {v['legs']} legs ({v['pilot_legs']} pilot, {v['wave_legs']} wave), started "
                         f"{v['first']} .. {v['last']}" for key, v in sorted(legs.items()))
    lines += [f"- **Request (hengyuf, 2026-10-30, verbatim):** {REQUEST}",
              "- **Settings:** kimi = `nvidia/moonshotai/kimi-k3`, glm = `nvidia/zai-org/glm-5.3`, deepseek = "
              "`nvidia/deepseek-ai/deepseek-v4-pro` (reasoning_effort max, temperature 1.0, top_p 0.95); OpenCode 1.17.11 "
              "stock system prompt; limit.context = limit.input = 262,144 (compaction at 242,144), limit.output = "
              "131,072; ATP agent timeout 10,800 s; each task's own DeepSWE grader; hengyuf's OpenSandbox credential "
              "(CELL3_SANDBOX_API_KEY). Campaigns: per model wave 1 (79 tasks) in 3 groups and wave 2 (34 tasks) in 2 "
              "groups, 5 rollouts per task; in flight per campaign at most kimi 8, glm 8 (wave 2: 4), deepseek 5, "
              "lowered further by ATP's adaptive admission.",
              "- **Hub keys (key names only):** the 3 pilot legs (2026-10-30 21:35 PDT) ran on NVAPI_KEY_NEW_2; the "
              "9 wave-1 legs of 10-30 23:07 PDT ran on NVAPI_KEY_NEW_3 (OPERATOR_NOTE 10-30 21:45, user decision) until "
              "its $1,000 floor stop on 10-31 10:30-10:40 PDT; every leg from 10-31 18:29 PDT ran on NVAPI_KEY_NEW_2 "
              "(OPERATOR_NOTE 10-31 18:15, user decision; floors $1,700 start / $1,500 guard). Legs per key "
              f"(tdw-legs.jsonl): {leg_text or 'none recorded'}. Each hub reply's cost is attributed to the leg that "
              "was running when the reply arrived, so to that leg's key.",
              "- **Definitions:** graded = the slot's final record is pass or fail: for a complete campaign, driver.py's "
              "results/<model>-<campaign>.json, which holds ATP's grade of the slot's graded attempt; for a campaign "
              "without a results file, ATP's slot_state \"completed\", cross-checked with this script's own view. Main's "
              "decision adds one graded fail (GRADED_FAIL_OVERRIDES: kimi w2-g1 13-0). **pass@1 raw** = passes / "
              "(graded + slots that ended on an infra error), so infra counts as a fail. Slots stopped before they "
              "ended on their own are left out: cancelled or interrupted by a stop, or in flight at the user's cancel. "
              "Never-started slots are also left out. **excl. broken-ended** = passes / graded slots not ended by a "
              "broken reply (brokenreply.py). **excl. infra** = passes / graded. **excl. agent timeouts** = passes / "
              "graded slots not ended by ATP's 10,800 s agent timeout; those slots stay graded (OPERATOR_NOTE "
              "2026-10-31 02:15). **pass@5** = tasks with at least one pass, out of the tasks with at least one graded "
              "slot; it is exact for a task with all 5 graded and a lower bound otherwise. **Compaction rate** = "
              "graded rollouts whose OpenCode session compacted at least once (export.json). **$ per graded "
              "rollout** = mean hub cost of a graded slot's final attempt. The cost comes from the journal: LiteLLM "
              "usage.cost when reported, else prices.json per token. **all spend / graded** = the cost of every "
              "attempt (graded, infra, stopped) / graded slots. **Base** = the base student swonlysc-step0014, from "
              "eval runs base-step0-rep5-out131k and its rerun1, resolved out of 5 each = k/10 (tasks/base_k10.json).",
              ""]

    lines += ["## Completeness", "",
              "| model | status | planned slots | graded | pass | fail | infra (ATP: retry / needs_review) | stopped "
              "before the end: cancelled / interrupted / in flight at the user's cancel | not started | campaigns "
              "complete / cancelled by the user / stopped / never started | tasks with >= 1 graded / all 5 graded |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for model in MODELS:
        b = ov[model]
        states = collections.Counter(s for (m, _), s in campaign_state.items() if m == model)
        lines.append(f"| {model} | {tag[model]} | {b['planned']} | {b['graded']} | {b['pass']} | {b['fail']} | "
                     f"{b['infra']} ({b['infra'] - b['infra_needs_review']} / {b['infra_needs_review']}) | "
                     f"{b['cancelled']} / {b['interrupted']} / {b['cancelled_in_flight']} | {b['not_started']} | "
                     f"{states['complete']} / {states['cancelled_by_user']} / {states['stopped']} / "
                     f"{states['not_started']} | {b['tasks_started']} / {b['tasks_complete']} of 113 |")
    tot = {k: sum(ov[m][k] for m in MODELS) for k in ("planned", "graded", "pass", "infra", "cancelled", "interrupted",
                                                       "cancelled_in_flight", "not_started")}
    lines += [f"| all three ({'mixes complete and PARTIAL counts' if not complete else 'complete'}) | | "
              f"{tot['planned']} | {tot['graded']} | {tot['pass']} | {tot['graded'] - tot['pass']} | {tot['infra']} | "
              f"{tot['cancelled']} / {tot['interrupted']} / {tot['cancelled_in_flight']} | {tot['not_started']} | | |",
              ""]

    lines += ["## Overview per model", "",
              "| model | status | graded | pass | pass@1 raw | pass@1 excl. broken-ended | pass@1 excl. infra | agent "
              "timeouts (passed) | pass@1 excl. agent timeouts | base pass@1, same slots | pass@5: tasks >= 1 pass / tasks "
              "graded | tasks all 5 graded: >= 1 pass / 5 of 5 | broken-ended | runaways | compaction rate | passes "
              "without compaction | $ per graded rollout | all spend / graded |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for model in MODELS:
        b = ov[model]
        lines.append(f"| {model} | {tag[model]} | {b['graded']} | {b['pass']} | {b['pass_at_1_raw']} | "
                     f"{b['pass_at_1_excl_broken']} | {b['pass_at_1_excl_infra']} | {b['agent_timeouts']} "
                     f"({b['agent_timeouts_passed']}) | {b['pass_at_1_excl_timeouts']} | {b['base_pass_at_1_same_slots']} "
                     f"| {b['tasks_any_pass']} / {b['tasks_started']} ({b['pass_at_5_rate']}%) | "
                     f"{b['tasks_complete_any_pass']} / {b['tasks_complete_all_pass']} of {b['tasks_complete']} | "
                     f"{b['broken_ended']} | {b['runaways']} | {b['compaction_rate']}% ({b['compacted']}) | "
                     f"{b['pass_no_compaction']} of {b['pass']} | {money(b['cost_per_graded_rollout'])} | "
                     f"{money(b['cost_all_per_graded'])} |")
    notes = []
    for over in manifest.get("status_overrides") or []:
        b = ov[over["model"]]
        notes.append(f"{over['model']} counts {over['campaign']} {over['task_id']} r{over['repeat']} as a graded fail "
                     f"(GRADED_FAIL_OVERRIDES; ATP leaves it needs_review). In ATP's own view {over['model']} has "
                     f"{b['graded'] - 1} graded and pass@1 {pct(b['pass'], b['graded'] - 1)} "
                     f"({pct(b['pass'] - b['agent_timeouts_passed'], b['graded'] - 1 - b['agent_timeouts'])} without "
                     "the agent timeouts), as PROGRESS.md shows.")
    for model in partial:
        notes.append(f"{model} is PARTIAL: its pass@5 is a lower bound on the {ov[model]['tasks_started'] - ov[model]['tasks_complete']} "
                     "tasks with 1-4 graded slots, and its numbers cover a different task mix than the complete "
                     "models.")
    lines += [""] + [f"- {note}" for note in notes] + [
        "- Base = the base student (swonlysc-step0014, 10 rollouts per task) on the same tasks, weighted by the "
        "graded slots.", ""]

    lines += ["## Wave 1 (79 uncovered tasks) vs wave 2 (34 covered tasks)", "",
              "| model x wave | status | tasks | planned | graded | pass@1 raw | excl. broken | excl. infra | excl. agent "
              "timeouts | base, same slots | pass@5: tasks >= 1 pass / graded | tasks all 5 graded | compaction | $ per "
              "graded rollout | hub cost (all attempts) | campaigns |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for model in MODELS:
        for wave in (1, 2):
            b = summary["by_wave"][f"{model} w{wave}"]
            states = collections.Counter(s for (m, n), s in campaign_state.items()
                                         if m == model and n.startswith(f"w{wave}"))
            state_text = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(states.items()))
            lines.append(f"| {model} w{wave} | {tag[model]} | {b['tasks']} | {b['planned']} | {b['graded']} | "
                         f"{b['pass_at_1_raw']} | {b['pass_at_1_excl_broken']} | {b['pass_at_1_excl_infra']} | "
                         f"{b['pass_at_1_excl_timeouts']} | {b['base_pass_at_1_same_slots']} | {b['tasks_any_pass']} / "
                         f"{b['tasks_started']} | {b['tasks_complete']} | {b['compaction_rate']}% | "
                         f"{money(b['cost_per_graded_rollout'])} | {money(b['cost_all_attempts'])} | {state_text} |")
    lines.append("")

    lines += ["## Per campaign", "",
              "| model | campaign | state | planned | graded | pass | pass@1 excl. infra | infra | stopped before the end "
              "| not started | tasks all 5 graded | ATP retries | hub cost |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for model in MODELS:
        for name in CAMPAIGNS:
            b = summary["by_campaign"][f"{model} {name}"]
            lines.append(f"| {model} | {name} | {b['state'].replace('_', ' ')} | {b['planned']} | {b['graded']} | "
                         f"{b['pass']} | {b['pass_at_1_excl_infra']} | {b['infra']} | {b['stopped']} | "
                         f"{b['not_started']} | {b['tasks_complete']} | {b['retried_slots']} | "
                         f"{money(b['cost_all_attempts'])} |")
    lines.append("")

    lines += ["## Matched task sets (like-for-like across models)", ""]
    for name, data in summary["matched"].items():
        lines += [f"**Tasks where {name}: {data['tasks']}.**", ""]
        if not data["tasks"]:
            lines += ["(none)", ""]
            continue
        lines += ["| model | graded slots | pass@1 excl. infra | task-mean pass rate | tasks >= 1 pass | base, same "
                  "slots |", "|---|---|---|---|---|---|"]
        for model in MODELS:
            b = data["per_model"][model]
            lines.append(f"| {label[model]} | {b['graded']} | {b['pass_at_1_excl_infra']} | {b['task_mean_pass_rate']} | "
                         f"{b['tasks_any_pass']} / {b['tasks_started']} | {b['base_pass_at_1_same_slots']} |")
        lines.append("")

    lines += ["## pass@1 (excl. infra) per language", "",
              "Cell = pass@1 over the model's graded slots (graded slots; base pass@1 on the same slots).", "",
              "| language | " + " | ".join(label[m] for m in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    for language, per_model in summary["by_language"].items():
        ids = [t for t in tasks if tasks[t]["language"] == language]
        cells = [f"{per_model[m]['pass_at_1_excl_infra']} ({per_model[m]['graded']}; base "
                 f"{per_model[m]['base_pass_at_1_same_slots']})" for m in MODELS]
        lines.append(f"| {language} ({len(ids)} tasks) | " + " | ".join(cells) + " |")
    lines.append("")

    u = summary["uncovered"]
    lines += ["## Uncovered tasks with base <= 2/10 and a teacher >= 3/5", "",
              f"Wave 1 = the 79 tasks with no accepted synthetic task. {u['tasks']} of them have base_k10 <= 2, meaning "
              "the base student resolved at most 2 of its 10 rollouts. A teacher reaches >= 3/5 when at least 3 of its "
              "5 slots pass. For a complete model every task is **yes** or **no**. A complete campaign's final infra "
              "slot counts as not passed. For PARTIAL GLM: **yes** = at least 3 passes among its graded slots, which "
              "is final. **no** = fewer than 3 passes even if every ungraded slot had passed. **undetermined** = "
              "neither, because the user's cancel came before enough slots ran.", "",
              "| teacher | status | yes | no | undetermined |", "|---|---|---|---|---|"]
    for model in MODELS:
        d = u["per_teacher"][model]
        lines.append(f"| {model} | {tag[model]} | {d['yes']} | {d['no']} | {d['undetermined']} |")
    lines.append(f"| any complete teacher ({' or '.join(u['complete_teachers']) or 'none'}) | complete | "
                 f"{len(u['complete_teachers_yes'])} | {u['tasks'] - len(u['complete_teachers_yes'])} | 0 |")
    lines.append(f"| any teacher (with {', '.join(partial) or 'no'} PARTIAL) | | {len(any_yes)} | "
                 f"{u['tasks'] - len(any_yes) - len(any_open)} | {len(any_open)} (no yes; at least one undetermined) |")
    lines += ["", f"Tasks with at least one teacher at >= 3/5 ({len(any_yes)}):", "",
              "| task | language | base_k10 | " + " | ".join(f"{label[m]} pass / graded" for m in MODELS) + " |",
              "|---|---|---|" + "---|" * len(MODELS)]
    for task_id in any_yes:
        cells = []
        for model in MODELS:
            v = verdicts[task_id][model]
            cells.append(f"{v['pass']} / {v['graded']}" + (" **yes**" if v["verdict"] == "yes" else
                                                            " (undetermined)" if v["verdict"] == "undetermined" else ""))
        lines.append(f"| {task_id} | {tasks[task_id]['language']} | {tasks[task_id]['base_k10']} | "
                     + " | ".join(cells) + " |")
    lines.append("")
    for model in MODELS:
        ids = u["per_teacher_yes"][model]
        lines.append(f"- {label[model]} yes ({len(ids)}): {', '.join(ids) or 'none'}.")
        if u["per_teacher_undetermined"][model]:
            lines.append(f"- {label[model]} undetermined ({len(u['per_teacher_undetermined'][model])}): "
                         + ", ".join(f"{t} ({verdicts[t][model]['pass']} / {verdicts[t][model]['graded']} graded)"
                                     for t in u["per_teacher_undetermined"][model]) + ".")
    no_yes = u["no_teacher_yes"]
    lines += [f"- No teacher at >= 3/5 ({len(no_yes)}): " + ", ".join(
        f"{t} (base {tasks[t]['base_k10']}; " + ", ".join(f"{m} {verdicts[t][m]['pass']}/{verdicts[t][m]['graded']}"
                                                         for m in MODELS) + ")" for t in no_yes) + ".",
              "- Per task and teacher: analysis/collect-summary.json (uncovered.verdicts).", ""]

    k = summary["kimi_vs_1m"]
    lines += ["## Kimi here (256k context) vs the 1M-context Kimi export (context only)", "",
              f"Reference: {KIMI_1M_NOTE}. It uses a different pipeline, context limit and serving period, so this is "
              "context, not a controlled comparison. Per task: analysis/collect-kimi-vs-1m.csv.", ""]
    if k.get("tasks_with_kimi_graded"):
        lines += ["| | Kimi here (256k) | Kimi 1M export |", "|---|---|---|",
                  f"| tasks / slots | {k['tasks_with_kimi_graded']} / {k['slots']} | {k['kimi1m_tasks']} / "
                  f"{k['kimi1m_scored_total']} |",
                  f"| avg@5 (pass@1), all tasks | {k['kimi256k_pass_at_1']} ({k['kimi256k_pass']} / {k['slots']}) | "
                  f"{k['kimi1m_all_tasks_avg']} ({k['kimi1m_correct_total']} / {k['kimi1m_scored_total']}) |",
                  f"| wave 1 (79 uncovered tasks) | {k['wave1_256k']} | {k['kimi1m_wave1_avg']} |",
                  f"| wave 2 (34 covered tasks) | {k['wave2_256k']} | {k['kimi1m_wave2_avg']} |",
                  f"| pass@5: tasks >= 1 pass (tasks with all 5 graded here) | {k['complete_256k_any_pass']} of "
                  f"{k['complete_tasks']} | {k['complete_1m_any_pass']} of {k['complete_tasks']} |",
                  f"| tasks with 5 of 5 passing | {k['all5_256k']} | {k['all5_1m']} |",
                  f"| uncovered base <= 2/10 tasks at >= 3/5 | {len(k['uncovered_yes_256k'])} of {u['tasks']} | "
                  f"{len(k['uncovered_yes_1m'])} of {u['tasks']} |", "",
                  f"- Without the {k['slots'] - k['slots_excl_timeouts']} graded agent timeouts here "
                  f"({ov['kimi']['agent_timeouts_passed']} of them passed; the numerator keeps every pass): "
                  f"{k['kimi256k_pass_at_1_excl_timeouts']} over {k['slots_excl_timeouts']} slots vs "
                  f"{k['kimi1m_same_slots_excl_timeouts']} for the 1M export on the same tasks, weighted the same way.",
                  f"- Per task (rate here vs the 1M rate): higher here {k['tasks_256k_better']}, lower "
                  f"{k['tasks_256k_worse']}, tied {k['tasks_tied']}. Task-mean: {k['kimi256k_task_mean']} vs "
                  f"{k['kimi1m_task_mean']}.",
                  "- Uncovered base <= 2/10 tasks at >= 3/5 for one of the two only: here only: "
                  + (", ".join(t for t in k["uncovered_yes_256k"] if t not in k["uncovered_yes_1m"]) or "none")
                  + "; 1M only: " + (", ".join(t for t in k["uncovered_yes_1m"] if t not in k["uncovered_yes_256k"])
                                     or "none") + ".", ""]
    else:
        lines += ["(no Kimi slot graded)", ""]

    for model in cancelled_models:
        mine = [e for e in wave_ledger if e["model"] == model]
        b = ov[model]
        lines += [f"## {model.upper()}: what the user's cancel left (PARTIAL)", ""]
        flight = [e for e in mine if e["state"] == "cancelled_in_flight"]
        lines.append(f"- In flight at the cancel ({len(flight)}; cut off, not graded; their dirs have no end marker and "
                     "ATP's slot_state says \"active\"): " + ("; ".join(
                         f"{e['campaign']} {e['task_id']} r{e['repeat']} (attempt {Path(e['attempt_dir']).parent.name}/"
                         f"{Path(e['attempt_dir']).name}, started {(e.get('reason') or {}).get('started_at')}, last file "
                         f"change {(e.get('reason') or {}).get('last_file_change')}, sandbox {e.get('sandbox_id')})"
                         for e in flight) or "none") + ". Main deletes their sandboxes (OPERATOR_NOTE 2026-11-02 "
                     "item 4).")
        earlier = [e for e in mine if e["state"] in ("interrupted", "cancelled", "infra")]
        lines.append(f"- Ended ungraded before the cancel and never retried ({len(earlier)}): " + ("; ".join(
            f"{e['campaign']} {e['task_id']} r{e['repeat']} ({e['state']}; ATP {e.get('atp_status')}; "
            + (f"interrupted ({(e.get('reason') or {}).get('interrupted_reason')}), "
               if e["state"] == "interrupted" else "")
            + reason_text(e.get("reason")) + ")" for e in earlier) or "none") + ".")
        per_campaign = collections.Counter(e["campaign"] for e in mine if e["state"] == "not_started")
        lines.append(f"- Never started: {b['not_started']} ("
                     + ", ".join(f"{n} {per_campaign[n]}" for n in CAMPAIGNS) + ").")
        task_graded = collections.Counter(e["task_id"] for e in mine if e["status"] in ("pass", "fail"))
        lines.append(f"- Tasks: {sum(v == REPEATS for v in task_graded.values())} with all 5 graded, "
                     f"{sum(0 < v < REPEATS for v in task_graded.values())} with 1-4 graded, "
                     f"{113 - len(task_graded)} with none.")
        broken = [e for e in mine if e.get("ended_by_broken_reply")]
        lines.append(f"- Broken-ended graded slots: {len(broken)}" + ((": " + "; ".join(
            f"{e['campaign']} {e['task_id']} r{e['repeat']} ({e['status']})" for e in broken)
            + ". Its one-time re-run (brk) was planned for the end of the campaign; the cancel removed it, so the slot "
              "counts as it stands (flagged ended_by_broken_reply).") if broken else "."))
        ungraded_cost = sum(e.get("slot_cost_usd") or 0.0 for e in mine if e["status"] not in ("pass", "fail"))
        lines.append(f"- Hub spend of the {model} attempts that ended ungraded or were cut off: {money(ungraded_cost)} "
                     f"of {money(summary['cost'][model]['all_attempts'])} (all {model} wave attempts).")
        lines.append("")

    lines += ["## Re-runs, ATP retries, agent timeouts, broken replies", ""]
    reruns = summary.get("rerun_dirs") or []
    lines.append("- Broken-reply re-runs (`<campaign>-brk<n>`) and budget-refusal recoveries (`<campaign>-rec<n>`): "
                 + (f"probe dirs {', '.join(reruns)}" if reruns else "**none ran** (no -brk / -rec probe dir exists)")
                 + ". Complete campaigns: " + "; ".join(
                     f"{m} {n}: re-runs {len(v['broken_reply_reruns'])}, recoveries {len(v['recovered'])}, final "
                     f"record = first-attempt record for {v['same_as_first']} of {v['slots']} slots"
                     for (m, n), v in sorted(summary["results_provenance"].items())) + ".")
    retried = [e for e in wave_ledger if (e.get("attempts") or 0) > 1]
    lines.append("- ATP retries inside a campaign (slots with more than one attempt dir; the dataset row is ATP's "
                 f"graded attempt, the cost counts every attempt): {len(retried)} (" + ", ".join(
                     f"{m} {sum(e['model'] == m for e in retried)}" for m in MODELS) + ").")
    broken = [e for e in wave_ledger if e.get("ended_by_broken_reply")]
    lines.append(f"- Broken-ended graded slots: {len(broken)}" + (": " + "; ".join(
        f"{e['model']} {e['campaign']} {e['task_id']} r{e['repeat']} ({e['status']})" for e in broken) + "."
        if broken else "."))
    timeouts = [e for e in wave_ledger if e.get("ended_by_agent_timeout") and e["status"] in ("pass", "fail")]
    lines.append(f"- Graded slots ended by the agent timeout: {len(timeouts)}" + (": " + "; ".join(
        f"{e['model']} {e['campaign']} {e['task_id']} r{e['repeat']} ({e['status']})" for e in timeouts) + "."
        if timeouts else "."))
    review = [e for e in wave_ledger if not e.get("status_override") and (
        e["state"] == "final_infra" or (e["state"] == "infra" and e.get("atp_status") == "needs_review"))]
    lines.append("- Slots ATP left ungraded and does not retry (needs_review: e.g. the verifier killed at its 1,800 s "
                 f"timeout; final infra slots of complete campaigns): {len(review)}" + (": " + "; ".join(
                     f"{e['model']} {e['campaign']} {e['task_id']} r{e['repeat']} ("
                     + (f"verifier: {e['verifier_error'][:90]}" if e.get("verifier_error") else reason_text(e.get("reason")))
                     + ")" for e in review) + ". They stay ungraded (status infra)." if review else "."))
    overridden = [e for e in wave_ledger if e.get("status_override")]
    lines.append("- Counted as a GRADED FAIL although ATP left them needs_review (GRADED_FAIL_OVERRIDES): "
                 f"{len(overridden)}" + (": " + "; ".join(
                     f"{e['model']} {e['campaign']} {e['task_id']} r{e['repeat']} (f2p {e.get('f2p')}; "
                     f"{e['status_override']['reason']}; {e['status_override']['source']})" for e in overridden) + "."
                     if overridden else "."))
    refusals = checks.get("budget_refusals") or []
    lines.append(f"- Hub budget refusals (4xx with a budget message) in ungraded attempts: {len(refusals)}.")
    lines.append("")

    c = summary["cost"]
    keys = sorted({key for m in MODELS + ("pilots",) for key in c[m]["by_key"]})
    lines += ["## Cost and hub keys (journal)", "",
              "| model | status | all attempts | graded rollouts (final attempt) | ungraded attempts | "
              + " | ".join(keys) + " | prompt / cached / completion tokens | hub replies 200 | non-200 replies |",
              "|---|---|---|---|---|" + "---|" * len(keys) + "---|---|---|"]
    for model in MODELS + ("pilots",):
        x = c[model]
        errors = ", ".join(f"{s}: {n}" for s, n in sorted(x["hub_errors"].items())) or "0"
        lines.append(f"| {model} | {tag.get(model, 'complete (3 one-rollout pilots)')} | {money(x['all_attempts'])} | "
                     f"{money(x['graded_final_attempts'])} | {money(x['ungraded_attempts'])} | "
                     + " | ".join(money(x["by_key"].get(key, 0.0)) for key in keys)
                     + f" | {x['prompt_tokens']:,} / {x['cached_tokens']:,} / {x['completion_tokens']:,} | "
                     f"{x['hub_calls_ok']:,} | {errors} |")
    wave_total = sum(c[m]["all_attempts"] for m in MODELS)
    key_totals = {key: sum(c[m]["by_key"].get(key, 0.0) for m in MODELS) for key in keys}
    lines += [f"| all wave legs ({'mixes complete and PARTIAL' if not complete else 'complete'}) | | {money(wave_total)} | "
              f"{money(sum(c[m]['graded_final_attempts'] for m in MODELS))} | "
              f"{money(sum(c[m]['ungraded_attempts'] for m in MODELS))} | "
              + " | ".join(money(key_totals[key]) for key in keys) + " | | | |", ""]
    lines.append("- all attempts = the graded slots' final attempts + the ungraded slots' attempts + the earlier attempts "
                 "of slots that ATP retried before grading them: " + ", ".join(
                     f"{m} {money(c[m]['all_attempts'] - c[m]['graded_final_attempts'] - c[m]['ungraded_attempts'])}"
                     for m in MODELS) + " in earlier attempts.")
    lines.append("- Pilots (NVAPI_KEY_NEW_2): " + ", ".join(
        f"{m} {money(v)}" for m, v in (c["pilots"]["per_pilot"] or {}).items()) + f" = {money(c['pilots']['all_attempts'])}.")
    lines.append("- Wave legs per key: " + "; ".join(
        f"{key} {money(key_totals[key])} (" + ", ".join(f"{m} {money(c[m]['by_key'].get(key, 0.0))}" for m in MODELS)
        + ")" for key in keys) + ".")
    last, first = summary.get("new3_last_reading"), summary.get("new3_at_first_leg")
    if last and first:
        mine3 = key_totals.get("NVAPI_KEY_NEW_3", 0.0)
        lines.append(f"- NVAPI_KEY_NEW_3 (budget-log.jsonl, key_budget.py): {money(first['remaining'])} left at the "
                     f"wave-leg starts (lowest start reading, {first['at']}), {money(last['remaining'])} at its last "
                     f"reading ({last['at']}): {money(first['remaining'] - last['remaining'])} spent in between. This "
                     f"campaign's journals account for {money(mine3)}. The rest is other use of the key (the r6 RL "
                     "run's rubric judge, key_budget.py's own checks) plus any gap between the journal's cost and "
                     "LiteLLM's spend figure. That figure lags, and replicas differ by about $20.")
    n2 = summary.get("new2_readings") or {}
    if n2.get("last"):
        mine2 = key_totals.get("NVAPI_KEY_NEW_2", 0.0) + c["pilots"]["by_key"].get("NVAPI_KEY_NEW_2", 0.0)
        text = (f"- NVAPI_KEY_NEW_2 (budget-log.jsonl, key_budget.py): {money(n2['first']['spend'])} spent / "
                f"{money(n2['first']['remaining'])} left at the setup check ({n2['first']['at']})")
        if n2.get("last_before_reset") and n2.get("first_after_reset"):
            text += (f"; {money(n2['last_before_reset']['spend'])} spent at {n2['last_before_reset']['at']}. The hub "
                     f"then reported the spend as reset: {money(n2['first_after_reset']['spend'])} spent / "
                     f"{money(n2['first_after_reset']['remaining'])} left at {n2['first_after_reset']['at']}. This "
                     "agent does not know why")
        text += (f". **Last reading: {money(n2['last']['spend'])} spent / {money(n2['last']['remaining'])} left "
                 f"({n2['last']['at']}, tag {n2['last'].get('tag')}).**")
        if n2.get("first_after_reset"):
            since = n2["last"]["spend"] - n2["first_after_reset"]["spend"]
            text += (f" From the reset on, NEW_2's spend grew by {money(since)}. This campaign's NEW_2 wave legs "
                     f"(from 10-31 18:29) account for {money(key_totals.get('NVAPI_KEY_NEW_2', 0.0))}. The rest "
                     f"({money(since - key_totals.get('NVAPI_KEY_NEW_2', 0.0))}) is other use of the key: the r6 RL "
                     "rubric judge (on NEW_2 since 10-31 18:15) and key_budget.py's own checks. It also holds any gap "
                     "between the journal's cost and LiteLLM's spend figure.")
        text += f" All of this campaign's NEW_2 cost, pilots included: {money(mine2)}."
        lines.append(text)
    lines.append("")

    lines += ["## Checks", "",
              f"- ATP state file: {states_meta['file'] if states_meta else 'none'}"
              + (f" (created {states_meta['created_at']}, sha256 {states_meta['sha256'][:16]}..., probes unchanged by "
                 f"the check: {states_meta['probes_unchanged']}, {states_meta['files_checked']:,} files compared, "
                 f"validity caches it would have written: {states_meta['would_write']})" if states_meta else "")]
    for key, value in sorted(manifest["checks"].items()):
        lines.append(f"- {key}: {value}")
    lines.append("- Independent checks: final/verify_collection.py -> analysis/collect-verify.json.")
    lines += ["", "## Files", "", "| file | rows | bytes | sha256 |", "|---|---|---|---|"]
    for name, meta in manifest["files"].items():
        rows = (counts.get(name.split(".")[0]) if name.endswith(".jsonl.gz") else
                manifest["slots_ledger_rows"] if name == "slots.jsonl" else
                sum(counts.values()) if name == "index.jsonl" else "")
        lines.append(f"| dataset/{name} | {rows} | {meta['bytes']:,} | `{meta['sha256'][:16]}...` |")
    lines.append(f"| per_task.csv | {n_table} | | |")
    lines.append("| analysis/collect-summary.json | | | |")
    lines.append(f"| analysis/collect-kimi-vs-1m.csv | {len(rows_1m)} | | |")
    return lines

if __name__ == "__main__":
    main()
