#!/usr/bin/env python3
"""Validate the collector's output (D/data) against D/CONTRACT.md, and optionally cross-check it against the pipeline's
own files (READ-ONLY).

    /usr/bin/python3 -I -B tests/validate_contract.py [--data DIR] [--state DIR] [--sources] [--show N]

Contract checks (always): every file is strict JSON (no NaN); schema 1; the same generated_at in index / task / fleet /
api (one cycle); every key, type, enum and length limit of CONTRACT.md; derived fields agree with their inputs
(passed/total/unscored vs the rollouts, eligibility and queue rank, parent/lineage/round/branch/kind vs the node id,
children vs parents, the decision rules, a node's focus pattern vs the failure patterns, a probing node's partial
counts vs its progress, the teacher audit vs the teacher's passing rollouts, a node's failure-match audit scope
(audit_eligible) vs its score, rollouts, seeds.json, withdrawal and build root, a node's seed note (seed_invalid) vs
seeds.json and its decision, a run's failure-match audit counts vs its nodes, busy workers vs the workers, the burn
windows and the exhaustion ETA vs the hub sample); task files exist exactly for the tasks with a run; histories are
trimmed; no key-like string anywhere (independent patterns, not collect.redact()); state/ is private. Fields added after
v1.1 (CONTRACT "v1.2" .. "v1.5") are optional: checked when present, so output of an older collector still passes. The
decision rules are the current ones: since v1.5 an accepted task is accepted even when its round's seeds.json lists it
invalid (seed validity only governs evolution), so an older collector's discard of such a task is a problem.

--sources adds cross-checks against the real tree under S (read-only, a few hundred KB of reads): the node set equals
the cand-* directories, nodes with round-<k>/<cand>/metrics.json are scored with the same accept, seeds.json
valid/invalid agree with the decisions (invalid: discarded with its reason, unless metrics.json accepted the task and
main did not withdraw it, then accepted with the reason as its seed_invalid note), runs/<run>/accepted/ holds exactly
the accepted tasks (a withdrawn task never counts as accepted), the probe directories
are all accounted for, each evolve node's focus pattern is the one its evolve prompt (evolve/.agent/prompt.md) used,
each attached failure-match audit has the verdict of its file (runs/<run>/audit/failure_match/<task_id>.json), a node
is withdrawn exactly when its metrics.json says so or its task dir sits in runs/<run>/withdrawn/, an eligible node has
its round-<k>/<cand>/metrics.json and rollouts/, and every task's accuracy cards equal three_way_per_task.csv (kimi,
super_vl) and the sc14 targets (per_task.csv counts, index.json records).

Batches (CONTRACT "v1.6", e.g. Baihe's): runs[] entries with a "source" are a batch's runs (status finished / stopped,
no Slurm state, audit {"status": "not_audited"}, "batch" tasks of outputs/tasks.jsonl, their document at "detail");
tasks[].batch_runs names them; index "batches" carries each batch's own totals (tasks, runs, targets, in_batch,
accepted, never added to the official runs'). data/run/<source>/<run>.json exist exactly for those runs and are checked
like task files (same NODE and decision rules), plus: generated_at is runs[].detail_generated_at (the cycle that last
changed the file, never later than index.json's), no job state or log, rounds done / unfinished / pending, every node
never audited (failure_match null, audit_eligible false, audit not_audited) with in_batch, every in_batch node accepted
and their count runs[].batch; the official audit counts never include them. --sources also reads each batch's
outputs/tasks.jsonl and outputs/<task_id>/provenance.json (the run that built each task) and the runs' own trees as
above: every batch task is an in_batch accepted node of its run, runs[].batch counts them, and each run's
accepted/tasks.jsonl lists exactly them.

The Explorer (CONTRACT "v1.7"): data/explorer.json, when index.json names it: its shape (column headers, each cell's
type and enum), its stamps (no later than index.json's, = index.json explorer.generated_at, bytes and row counts), its
sources (official first and the only audited one), and every run, target, task row, trajectory row and total recomputed
from the task and run files by the contract's rules (explorer_expected: an implementation independent of collect.py's).
--sources also checks the candidate dirs and the transcripts and patches it points at.

Every official run and the synthesis queue (CONTRACT "v1.8"): official runs[] entries carry latest (exactly one per
task, the most recently submitted, = tasks[].run) and detail (task/<task>.json for the latest, run/hengyuf/<run>.json
for an older one, with detail_generated_at); tasks[].official_runs lists them all (latest first, then newest launch
first); data/run/hengyuf/ holds exactly the older runs' documents, checked like task files (generated_at =
detail_generated_at <= index.json's); a document's job.jobs lists each job once; the Explorer covers every official run
(older ones linked with run=). index.json queues[]: read_at within the cycle, each queue's runner (age_s and alive as of
read_at), its budget (a key NAME and a number, nothing else), its flags, its runs (status of the runner's set, leg,
winner among the candidates) and its log tail (the runner's pid masked). --sources also lists
synth/runs/official*: every dir with launch.jsonl or a build is exactly one official run and the latest of each task is
its last launch; and compares the queues with their runner files (queue.json, state.json, heartbeat.json, runner.log)
when those did not change since the cycle.

Exit status 0 when nothing is wrong; otherwise one line per problem and exit status 1. validate(...) returns the
problem list for tests (tests/test_contract.py). Safe next to a running collector: files are replaced atomically, a
read that straddles two cycles is retried, the collector's temp files (".<name>.<pid>[.<thread>].tmp") are skipped and
only those older than 10 min count as leftovers, and a file renamed away between the listing and its stat is absent.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import stat
import time
import urllib.parse

D = Path(__file__).resolve().parent.parent
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
THREE_WAY = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/deepswe-partial-analysis/three-way/"
                 "three_way_per_task.csv")
TARGETS = S / "data/targets/sc14-out131k-kimi"

MODELS = ["kimi", "super_vl", "sc14"]  # CONTRACT: display order
STUDENT, TEACHER = "sc14", "kimi"  # the probes' model aliases (synth/common.py)
RUN_STATUS = {"running", "pending", "finished", "stopped", "failed"}
ROUND_STATUS = {"done", "probing", "building", "pending"}
BATCH_RUN_STATUS = {"finished", "stopped"}  # v1.6: a batch's runs have ended (not polled)
BATCH_ROUND_STATUS = {"done", "unfinished", "pending"}  # v1.6: an ended run's rounds
NOT_AUDITED = {"status": "not_audited"}  # v1.6: NODE.audit / runs[].audit / batches[].audit of a batch
SAFE_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")  # collect.py SAFE_NAME: run and source names
NODE_STATUS = {"planning", "authoring", "building", "build_failed", "built", "probing", "scored", "abandoned"}
DECISIONS = {"evolve", "discard", "accepted", "accepted_evolve", "not_selected", "end", "pending", "abandoned"}
ROLLOUT_STATUS = {"pass", "fail", "infra"}
AUDIT_VERDICTS = ["strong", "partial", "weak", "none"]  # v1.3 failure-match audits
AUDIT_REPRODUCED = {"yes", "partial", "no"}
FAILURE_MATCH_KEYS = ["verdict", "summary", "patterns", "student_rollouts", "other_causes", "similarity_judge",
                      "audited_at"]
WITHDRAWN_LIMITS = {"at": 64, "by": 100, "why": 600}  # v1.4 NODE.withdrawn
ACCEPTED = ("accepted", "accepted_evolve")
FLEET_BUCKETS = ["ready", "draining", "starting", "pending", "held", "stale", "unhealthy", "failed", "stopped",
                 "vanished", "other"]
DETAIL_LIMITS = {"instruction_md": 20000, "evolve_plan_md": 20000, "synthetic_failure_analysis_md": 20000,
                 "rubric_md": 8000}  # + teacher_audit (check_audit)
ROUTER_SCOPE = "official-run probes"
# v1.7, data/explorer.json (the Explorer's tables)
EXPLORER_KEYS = ["schema", "generated_at", "sources", "runs", "targets", "task_columns", "tasks", "trajectory_columns",
                 "trajectories", "totals"]
EXPLORER_TASK_COLUMNS = ["run", "node", "task_id", "round", "branch", "kind", "title", "decision", "accepted",
                         "withdrawn", "in_batch", "similarity", "student_rate", "teacher_clean_rate", "teacher_rate",
                         "valid", "audit", "reproduced_yes", "reproduced_partial", "f2p", "p2p", "scored",
                         "rates_differ"]
EXPLORER_TRAJECTORY_COLUMNS = ["task", "slot", "model", "status", "reward", "f2p", "p2p", "tool_calls", "patch_chars",
                               "agent_error", "verifier_error", "clean"]
EXPLORER_AUDITS = AUDIT_VERDICTS + ["not_audited", "not_eligible"]
EXPLORER_TOTALS = ["runs", "tasks", "trajectories", "accepted", "audited", "unmatched_patterns", "rates_differ"]
OFFICIAL = {"source": "hengyuf", "name": "hengyuf", "label": "hengyuf · official runs", "audited": True}
OFFICIAL_RUNS_DIR = "run/hengyuf"  # v1.8: an older official run's document
QUEUE_KEYS = ["name", "root", "read_at", "max_active", "runner", "budget", "serving", "stop_launching", "updated_at",
              "finished_at", "runs", "log_tail"]
QUEUE_RUNNER_KEYS = ["heartbeat_at", "age_s", "host", "slurm_job_id", "note", "alive"]
QUEUE_BUDGET_KEYS = ["key", "remaining", "at"]  # the key's NAME and a number: nothing else of the runner's budget
QUEUE_RUN_KEYS = ["run", "task", "repo", "batch", "status", "leg", "winner", "candidates"]
QUEUE_STATUSES = {"queued", "pending", "running", "waiting_to_resume", "done", "failed", "stuck"}  # run_queue.py
QUEUE_STALE_S = 900  # run_queue.py HEARTBEAT_STALE_S
JOB_ID = re.compile(r"\d{1,12}")
NODE_ID = re.compile(r"(cand-[A-Za-z0-9_]+?)((?:-r\d+b\d+)*)")
FOCUS_LINE = re.compile(r'Focus pattern for this branch: "(.*)"\.')  # synth/prompts/evolve_plan.md, rendered
FRACTION = re.compile(r"(\d+)/(\d+)")
ERROR_LINE = re.compile(r"[^:\n]+: .+", re.S)  # "<source>: <message>"
KEY_NAME = re.compile(r"[A-Z][A-Z0-9_]*")  # api.json hub.key is a variable NAME, never a value
DAY = 86400.0
# Credentials that must never appear in an output (independent of collect.redact(), so a hole there shows up here).
SECRET_PATTERNS = [
    re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"nvapi-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    re.compile(r"glpat-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![0-9A-Z])"),
    re.compile(r"(?i)\bbearer\s+(?!<redacted>)[A-Za-z0-9._~+/=-]{20,}"),
    re.compile(r"(?i)(?:key|token|secret)[A-Za-z0-9_.-]{0,40}[\"']?\s{0,3}[=:]\s{0,3}[\"']?[A-Za-z0-9+/_-]{32,}"),
]


# ------------------------------------------------------------------------------------------------------- helpers

def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def is_num(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)) and math.isfinite(value)


def is_str(value):
    return isinstance(value, str)


def opt(check):
    """check, or None."""
    return lambda value: value is None or check(value)


def str_list(value, limit=None):
    return isinstance(value, list) and all(is_str(item) for item in value) and (limit is None or len(value) <= limit)


def fraction_ok(value):
    match = FRACTION.fullmatch(value) if is_str(value) else None
    return bool(match) and int(match.group(1)) <= int(match.group(2))


def pair_ok(value):
    """[done, total] with 0 <= done <= total."""
    return (isinstance(value, list) and len(value) == 2 and all(is_num(v) for v in value)
            and 0 <= value[0] <= value[1])


def refuse_constant(name):
    """json parse_constant hook: NaN / Infinity are no JSON (the browser's JSON.parse refuses them)."""
    raise ValueError(f"non-JSON constant {name}")


def strict_json(path):
    """json.load that refuses NaN/Infinity (the browser's JSON.parse would)."""
    with open(path, encoding="utf-8") as stream:
        return json.load(stream, parse_constant=refuse_constant)


def strings(value, path="$"):
    """Every (json path, string) inside a document, keys included."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield f"{path}.<key>", str(key)
            yield from strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for position, item in enumerate(value):
            yield from strings(item, f"{path}[{position}]")


def iso_epoch(text):
    """Epoch seconds of an ISO time with an offset, "-07:00" or (v1.8: run_queue.py's time.strftime("%z")) "-0700"."""
    try:
        return datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError):
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except (TypeError, ValueError):
            pass
    return None


def read_csv(path):
    with open(path, newline="") as stream:
        return {row["task_id"]: row for row in csv.DictReader(stream) if row.get("task_id")}


class Report:
    """Collects problems as '<file> <json path>: <message>'."""

    def __init__(self):
        self.problems = []
        self.checked = {}

    def add(self, where, message):
        self.problems.append(f"{where}: {message}")

    def need(self, condition, where, message):
        if not condition:
            self.add(where, message)
        return bool(condition)

    def keys(self, doc, required, where):
        """doc is a dict with every required key; returns whether it is a dict at all."""
        if not isinstance(doc, dict):
            self.add(where, f"expected an object, got {type(doc).__name__}")
            return False
        missing = [key for key in required if key not in doc]
        if missing:
            self.add(where, f"missing keys {missing}")
        return True

    def field(self, doc, key, check, where, what):
        """doc[key] passes check (doc is a dict; a missing key was already reported by keys())."""
        if isinstance(doc, dict) and key in doc and not check(doc[key]):
            self.add(f"{where}.{key}", f"expected {what}, got {json.dumps(doc[key])[:120]}")


# ------------------------------------------------------------------------------------------------- index.json

def check_index(r, index):
    w = "index.json"
    if not r.keys(index, ["schema", "generated_at", "generated_at_iso", "collect_seconds", "errors", "models", "runs",
                          "tasks"], w):
        return
    r.need(index.get("schema") == 1, w, "schema must be 1")
    r.field(index, "generated_at", is_num, w, "epoch seconds")
    r.field(index, "collect_seconds", lambda v: is_num(v) and v >= 0, w, "seconds >= 0")
    stamp = iso_epoch(index.get("generated_at_iso"))
    r.need(stamp is not None and is_num(index.get("generated_at")) and abs(stamp - index["generated_at"]) < 1.5,
           w, "generated_at_iso must be the ISO form of generated_at")
    for position, error in enumerate(index.get("errors") or []):
        r.need(is_str(error) and ERROR_LINE.fullmatch(error), f"{w} errors[{position}]",
               "expected '<source>: <message>'")
    models = index.get("models")
    if r.keys(models, MODELS, f"{w} models"):
        r.need(list(models)[:3] == MODELS, f"{w} models", f"display order must be {MODELS}")
        for key in MODELS:
            if r.keys(models.get(key), ["label", "role", "source"], f"{w} models.{key}"):
                for field in ("label", "role", "source"):
                    r.field(models[key], field, lambda v: is_str(v) and v.strip(), f"{w} models.{key}",
                            "a non-empty string")

    runs = index.get("runs") if isinstance(index.get("runs"), list) else []
    tasks = index.get("tasks") if isinstance(index.get("tasks"), list) else []
    task_ids = [task.get("task_id") for task in tasks if isinstance(task, dict)]
    r.need(len(task_ids) == len(set(task_ids)), w, "duplicate task ids")
    for position, run in enumerate(runs):
        where = f"{w} runs[{position}]"
        if not r.keys(run, ["run", "task_id", "repo", "job_id", "slurm_state", "partition", "submitted_at",
                            "started_at", "ended_at", "status", "stage", "rounds_started", "accepted", "tasks_built",
                            "tasks_scored"], where):
            continue
        if "source" in run:  # v1.6: a batch's run (a missing source: hengyuf's official run)
            check_batch_run(r, run, where)
        else:
            r.field(run, "run", lambda v: is_str(v) and v.startswith("official"), where, "an official* run name")
            if "latest" in run or "detail" in run:  # v1.8: its task's latest run, and where its document is
                check_official_run(r, run, where)
        r.field(run, "task_id", lambda v: v in task_ids, where, "a task of tasks[]")
        r.field(run, "status", lambda v: v in RUN_STATUS, where, f"one of {sorted(RUN_STATUS)}")
        r.field(run, "stage", lambda v: is_str(v) and v.strip() and "\n" not in v, where, "a one-line stage")
        for key in ("repo", "job_id", "slurm_state", "partition"):
            r.field(run, key, opt(is_str), where, "a string or null")
        for key in ("submitted_at", "started_at", "ended_at"):
            r.field(run, key, opt(is_num), where, "epoch seconds or null")
        for key in ("rounds_started", "accepted", "tasks_built", "tasks_scored"):
            r.field(run, key, lambda v: is_int(v) and v >= 0, where, "an int >= 0")
        if all(is_int(run.get(key)) for key in ("accepted", "tasks_scored", "tasks_built")):
            r.need(run["accepted"] <= run["tasks_scored"] <= run["tasks_built"], where,
                   "expected accepted <= tasks_scored <= tasks_built")
        if "audit" in run and "source" not in run:  # v1.3: failure-match audit coverage of the run's accepted tasks
            check_audit_counts(r, run, f"{where}.audit")
    run_names = {run.get("run"): run for run in runs if isinstance(run, dict) and "source" not in run}
    check_official_runs(r, runs, tasks, w)
    batch_runs = {(run.get("source"), run.get("run")): run for run in runs if isinstance(run, dict) and "source" in run}
    r.need(len(batch_runs) == sum(isinstance(run, dict) and "source" in run for run in runs), w,
           "duplicate batch runs (source, run)")

    eligible = []
    for position, task in enumerate(tasks):
        where = f"{w} tasks[{position}]"
        if not r.keys(task, ["task_id", "task_index", "f2p_total", "accuracy", "eligible", "queue_rank", "run"],
                      where):
            continue
        where = f"{w} {task.get('task_id')}"
        r.field(task, "task_id", lambda v: is_str(v) and v, where, "a task id")
        r.field(task, "task_index", is_int, where, "an int")
        r.field(task, "f2p_total", opt(lambda v: is_int(v) and v >= 0), where, "an int or null")
        accuracy = task.get("accuracy")
        cards = {}
        if r.keys(accuracy, MODELS, f"{where}.accuracy"):
            for model in MODELS:
                cards[model] = check_card(r, accuracy.get(model), model, f"{where}.accuracy.{model}")
        sc14, kimi = cards.get("sc14"), cards.get("kimi")
        rule = bool(sc14 and kimi and sc14.get("passed") <= 1 and kimi.get("passed") == 5 and kimi.get("total") == 5)
        r.need(task.get("eligible") is rule, where,
               f"eligible must be {rule} (sc14 <= 1 valid pass and kimi 5/5)")
        rank = task.get("queue_rank")
        if task.get("eligible") is True:
            if r.need(is_int(rank) and rank >= 1, where, "an eligible task needs queue_rank >= 1"):
                eligible.append((rank, (sc14 or {}).get("mean_f2p_ratio"), task.get("task_id")))
        else:
            r.need(rank is None, where, "queue_rank must be null when not eligible")
        run = task.get("run")
        if run is not None:
            r.need(run in run_names and run_names[run].get("task_id") == task.get("task_id"), where,
                   f"run {run!r} is not a runs[] entry for this task")
        if "batch_runs" in task:  # v1.6: the batch runs that converted this task
            listed = task["batch_runs"]
            if r.need(isinstance(listed, list) and listed and all(
                    isinstance(item, dict) and set(item) == {"source", "run"} for item in listed), where,
                    "batch_runs must be a non-empty list of {source, run}"):
                for item in listed:
                    entry = batch_runs.get((item["source"], item["run"]))
                    r.need(entry is not None and entry.get("task_id") == task.get("task_id"), where,
                           f"batch_runs {item} is not a batch runs[] entry for this task")
    eligible.sort()
    r.need([rank for rank, _, _ in eligible] == list(range(1, len(eligible) + 1)), w,
           "queue ranks must be 1..N over the eligible tasks")
    ratios = [(-1.0 if ratio is None else ratio) for _, ratio, _ in eligible]
    r.need(all(a >= b for a, b in zip(ratios, ratios[1:])), w,
           "queue_rank must follow sc14 mean_f2p_ratio, descending")
    named = [(item.get("source"), item.get("run")) for task in tasks if isinstance(task, dict)
             for item in (task.get("batch_runs") if isinstance(task.get("batch_runs"), list) else [])
             if isinstance(item, dict)]
    r.need(sorted(named, key=str) == sorted(batch_runs, key=str), w,
           "every batch run must be named once, in its task's batch_runs")
    check_batches(r, index.get("batches"), batch_runs, "batches" in index, w)
    if "queues" in index:  # v1.8: the synthesis queues' runner state
        check_queues(r, index.get("queues"), index.get("generated_at"), set(task_ids), index.get("errors") or [], w,
                     index.get("collect_seconds"))
    r.checked["index tasks"] = len(tasks)
    r.checked["eligible"] = len(eligible)
    if batch_runs:
        r.checked["batch runs"] = len(batch_runs)


def check_batch_run(r, run, where):
    """A batch's runs[] entry (v1.6): the official keys (no Slurm state, the run has ended) plus source, source_label,
    root, batch (its tasks in the batch's outputs/tasks.jsonl), detail (its document) and detail_generated_at, never
    audited."""
    if not r.keys(run, ["source", "source_name", "source_label", "root", "batch", "detail", "detail_generated_at",
                        "audit"], where):
        return
    r.field(run, "source", lambda v: is_str(v) and SAFE_RUN.fullmatch(v) and v != "hengyuf", where,
            "a batch source name (hengyuf's official runs carry no source)")
    r.field(run, "run", lambda v: is_str(v) and SAFE_RUN.fullmatch(v), where, "a run name")
    for key in ("source_name", "source_label"):
        r.field(run, key, lambda v: is_str(v) and v.strip(), where, "a non-empty name / label")
    r.field(run, "root", lambda v: is_str(v) and os.path.isabs(v) and v.endswith(f"/synth/runs/{run.get('run')}"),
            where, "the run's absolute directory <root>/synth/runs/<run>")
    r.field(run, "batch", lambda v: is_int(v) and v >= 0, where, "an int >= 0")
    r.field(run, "detail", lambda v: is_str(v) and re.fullmatch(
        rf"run/{re.escape(str(run.get('source')))}/{re.escape(str(run.get('run')))}(--[A-Za-z0-9._-]+)?\.json", v),
        where, "run/<source>/<run>.json")
    r.field(run, "detail_generated_at", is_num, where, "epoch seconds")
    r.field(run, "status", lambda v: v in BATCH_RUN_STATUS, where, f"one of {sorted(BATCH_RUN_STATUS)} (an ended run)")
    for key in ("slurm_state", "started_at", "ended_at"):
        r.field(run, key, lambda v: v is None, where, "null (a batch's jobs are not polled)")
    r.field(run, "audit", lambda v: v == NOT_AUDITED, where, f"{NOT_AUDITED} (a batch is never audited)")
    if is_int(run.get("batch")) and is_int(run.get("accepted")):
        r.need(run["batch"] <= run["accepted"], where, "batch tasks must be accepted tasks (batch <= accepted)")


def check_official_run(r, run, where):
    """v1.8, an official runs[] entry: latest (bool) and detail (task/<task>.json for its task's latest run, else
    run/hengyuf/<run>.json, or <run>--<task>.json for a run that builds several targets), and detail_generated_at
    exactly for an older run (its file keeps the stamp of the cycle that last changed it)."""
    r.field(run, "latest", lambda v: isinstance(v, bool), where, "a bool")
    name, task = str(run.get("run")), str(run.get("task_id"))
    if run.get("latest") is True:
        r.field(run, "detail", lambda v: v == f"task/{task}.json", where, "task/<task_id>.json (the latest run)")
        r.need("detail_generated_at" not in run, where, "no detail_generated_at: the latest run's task file is one "
                                                        "cycle's")
    elif run.get("latest") is False:
        r.field(run, "detail", lambda v: is_str(v) and re.fullmatch(
            rf"{OFFICIAL_RUNS_DIR}/{re.escape(name)}(--{re.escape(task)})?\.json", v), where,
            f"{OFFICIAL_RUNS_DIR}/<run>.json (an older official run)")
        r.field(run, "detail_generated_at", is_num, where, "epoch seconds")


def check_official_runs(r, runs, tasks, w):
    """v1.8: per task, its official runs[] entries: exactly one latest, the most recently submitted one, which is
    tasks[].run; tasks[].official_runs lists every one ({run, latest, detail}: the latest first, then newest launch
    first). Skipped for an older collector's output (no official entry carries latest)."""
    official = [run for run in runs if isinstance(run, dict) and "source" not in run]
    if not any("latest" in run for run in official):
        r.need(not any(isinstance(task, dict) and "official_runs" in task for task in tasks), w,
               "tasks[].official_runs without runs[].latest")
        return
    by_task = {}
    for run in official:
        by_task.setdefault(run.get("task_id"), []).append(run)
    older = 0
    for task in tasks:
        if not isinstance(task, dict):
            continue
        where = f"{w} {task.get('task_id')}"
        mine = by_task.pop(task.get("task_id"), [])
        if not mine:
            r.need("official_runs" not in task and task.get("run") is None, where,
                   "official_runs / run without an official runs[] entry")
            continue
        latest = [run for run in mine if run.get("latest") is True]
        if not r.need(len(latest) == 1, where, f"{len(latest)} latest official runs: expected exactly one"):
            continue
        when = lambda run: run.get("submitted_at") if is_num(run.get("submitted_at")) else 0  # noqa: E731
        r.need(all(when(run) <= when(latest[0]) for run in mine), where,
               f"latest {latest[0].get('run')} is not the most recently submitted official run")
        r.need(task.get("run") == latest[0].get("run"), where, f"run must be the latest official run "
                                                               f"{latest[0].get('run')}")
        rest = sorted((run for run in mine if run is not latest[0]), key=lambda run: (-when(run), str(run.get("run"))))
        want = [{"run": run.get("run"), "latest": run.get("latest"), "detail": run.get("detail")}
                for run in latest + rest]
        r.need(task.get("official_runs") == want, where,
               f"official_runs {json.dumps(task.get('official_runs'))[:200]} != {json.dumps(want)[:200]}")
        older += len(rest)
    r.need(not by_task, w, f"official runs of tasks not in tasks[]: {sorted(map(str, by_task))}")
    r.checked["official runs"] = len(official)
    r.checked["older official runs"] = older


def check_queues(r, queues, generated_at, task_ids, errors, w, collect_seconds=None):
    """v1.8, index.json queues[]: read_at (when the runner files were read: within the cycle), each queue's runner (age_s
    = read_at - heartbeat_at, alive = a heartbeat at most QUEUE_STALE_S old then whose note is not "stopped"), budget
    (null or exactly {key: a variable NAME or null, remaining, at}), flags, runs in queue order (a known status, or null
    with an error naming it; leg >= 0; winner one of the candidates; job ids are digits) and a log tail of <= 40 lines
    of <= 300 chars (a clipped line ends with the clip marker) without the runner's pid ("pid ⋯")."""
    if not r.need(isinstance(queues, list), f"{w} queues", "expected a list"):
        return
    names = set()
    for position, queue in enumerate(queues):
        where = f"{w} queues[{position}]"
        if not r.need(isinstance(queue, dict) and list(queue) == QUEUE_KEYS, where, f"keys must be {QUEUE_KEYS}"):
            continue
        name = queue.get("name")
        r.need(is_str(name) and SAFE_RUN.fullmatch(name) and name not in names, where, "a queue name, once")
        names.add(name)
        r.field(queue, "root", lambda v: is_str(v) and os.path.isabs(v), where, "an absolute path")
        read_at = queue.get("read_at")
        if r.need(is_num(read_at), f"{where}.read_at", "expected epoch seconds") and is_num(generated_at):
            r.need(generated_at - 1 <= read_at <= generated_at + (collect_seconds if is_num(collect_seconds) else 0) + 5,
                   where, f"read_at {read_at} is not within the cycle ({generated_at} + {collect_seconds} s)")
        r.field(queue, "max_active", opt(lambda v: is_int(v) and v > 0), where, "an int > 0 or null")
        for key in ("serving", "stop_launching"):
            r.field(queue, key, lambda v: isinstance(v, bool), where, "a bool")
        for key in ("updated_at", "finished_at"):
            r.field(queue, key, opt(is_num), where, "epoch seconds or null")
        runner = queue.get("runner")
        if r.need(isinstance(runner, dict) and list(runner) == QUEUE_RUNNER_KEYS, f"{where}.runner",
                  f"keys must be {QUEUE_RUNNER_KEYS}"):
            at, age = runner.get("heartbeat_at"), runner.get("age_s")
            r.need((at is None) == (age is None), f"{where}.runner", "heartbeat_at and age_s: both or neither")
            if is_num(at) and is_num(read_at):
                r.need(is_num(age) and age >= 0 and abs(age - max(0.0, read_at - at)) < 0.15, f"{where}.runner",
                       f"age_s {age} != read_at - heartbeat_at ({read_at - at:.1f})")
            r.field(runner, "host", opt(lambda v: is_str(v) and len(v) <= 200), f"{where}.runner", "<= 200 chars")
            r.field(runner, "note", opt(lambda v: is_str(v) and len(v) <= 40), f"{where}.runner", "<= 40 chars")
            r.field(runner, "slurm_job_id", opt(lambda v: is_str(v) and JOB_ID.fullmatch(v)), f"{where}.runner",
                    "a job id or null")
            want = bool(is_num(at) and is_num(age) and age <= QUEUE_STALE_S and runner.get("note") != "stopped")
            r.need(runner.get("alive") is want, f"{where}.runner",
                   f"alive must be {want} (a heartbeat <= {QUEUE_STALE_S} s old, not stopped)")
        budget = queue.get("budget")
        if budget is not None and r.need(isinstance(budget, dict) and list(budget) == QUEUE_BUDGET_KEYS,
                                         f"{where}.budget", f"null or exactly {QUEUE_BUDGET_KEYS}"):
            r.field(budget, "key", opt(lambda v: is_str(v) and KEY_NAME.fullmatch(v)), f"{where}.budget",
                    "a key NAME or null")
            r.field(budget, "remaining", opt(is_num), f"{where}.budget", "dollars or null")
            r.field(budget, "at", opt(is_num), f"{where}.budget", "epoch seconds or null")
        tail = queue.get("log_tail")
        r.need(str_list(tail, 40) and all(len(line) <= 300 for line in tail), f"{where}.log_tail",
               "<= 40 lines of <= 300 chars")
        r.need(not any(re.search(r"\bpid \d+", line) for line in tail if is_str(line)), f"{where}.log_tail",
               "the runner's pid must be masked (pid ⋯)")
        seen = set()
        for index, run in enumerate(queue.get("runs") if isinstance(queue.get("runs"), list) else []):
            here = f"{where}.runs[{index}]"
            if not r.need(isinstance(run, dict) and list(run) == QUEUE_RUN_KEYS, here, f"keys must be {QUEUE_RUN_KEYS}"):
                continue
            r.need(is_str(run.get("run")) and SAFE_RUN.fullmatch(run["run"]) and run["run"] not in seen, here,
                   "a run name, once")
            seen.add(run.get("run"))
            r.field(run, "task", lambda v: is_str(v) and v in task_ids, here, "a DeepSWE task of tasks[]")
            for key in ("repo", "batch"):
                r.field(run, key, opt(is_str), here, "a string or null")
            status = run.get("status")
            if status is None:
                r.need(any(str(error).startswith(f"queue {name}: {run.get('run')}: unknown status") for error in errors),
                       here, "status null without its 'unknown status' error")
            else:
                r.field(run, "status", lambda v: v in QUEUE_STATUSES, here, f"one of {sorted(QUEUE_STATUSES)}")
            r.field(run, "leg", lambda v: is_int(v) and v >= 0, here, "an int >= 0")
            r.field(run, "candidates", lambda v: str_list(v) and len(v) <= 8 and all(JOB_ID.fullmatch(x) for x in v),
                    here, "<= 8 job ids")
            r.field(run, "winner", opt(lambda v: is_str(v) and JOB_ID.fullmatch(v)), here, "a job id or null")
            if is_str(run.get("winner")) and isinstance(run.get("candidates"), list) and run["candidates"]:
                r.need(run["winner"] in run["candidates"], here, "the winner must be one of the candidates")
        r.need(isinstance(queue.get("runs"), list), where, "runs must be a list")
        r.checked[f"queue {name} runs"] = len(seen)


def check_batches(r, batches, batch_runs, present, w):
    """index.json "batches" (v1.6): one entry per batch with its own totals, which must agree with its runs[]."""
    if not present:
        r.need(not batch_runs, w, "batch runs without index.json batches")
        return
    if not r.need(isinstance(batches, list), f"{w} batches", "expected a list"):
        return
    sources = set()
    for position, batch in enumerate(batches):
        where = f"{w} batches[{position}]"
        if not r.keys(batch, ["source", "name", "label", "root", "outputs", "tasks", "runs", "targets", "in_batch",
                              "accepted", "unmatched", "not_in_batch", "audit"], where):
            continue
        for key in ("name", "label"):
            r.field(batch, key, lambda v: is_str(v) and v.strip(), where, "a non-empty string")
        source = batch.get("source")
        r.need(source not in sources, where, f"batch {source!r} twice")
        sources.add(source)
        for key in ("tasks", "runs", "targets", "in_batch", "accepted"):
            r.field(batch, key, lambda v: is_int(v) and v >= 0, where, "an int >= 0")
        for key in ("unmatched", "not_in_batch"):
            r.field(batch, key, lambda v: str_list(v, 50), where, "<= 50 strings")
        r.field(batch, "audit", lambda v: v == NOT_AUDITED, where, f"{NOT_AUDITED}")
        r.field(batch, "outputs", lambda v: is_str(v) and v == f"{batch.get('root')}/outputs", where, "<root>/outputs")
        mine = [run for (src, _), run in batch_runs.items() if src == source]
        r.need(batch.get("runs") == len(mine), where, f"runs must count its runs[] entries ({len(mine)})")
        r.need(batch.get("targets") == len({run.get("task_id") for run in mine}), where,
               "targets must count its runs' distinct tasks")
        r.need(batch.get("accepted") == sum(run.get("accepted") or 0 for run in mine), where,
               "accepted must sum its runs' accepted")
        r.need(all(run.get("source_name") == batch.get("name") for run in mine), where,
               "its runs' source_name must be its name")
        counted = sum(run.get("batch") or 0 for run in mine)
        r.need(counted <= (batch.get("tasks") or 0), where, f"its runs count {counted} tasks of {batch.get('tasks')}")
        # every task of the batch must be an accepted node of the run that built it (CONTRACT "v1.6")
        r.need(not batch.get("unmatched"), where, f"outputs tasks that are no accepted node: {batch.get('unmatched')}")
        if not batch.get("unmatched"):
            r.need(counted == batch.get("tasks") == batch.get("in_batch"), where,
                   f"tasks {batch.get('tasks')}, runs[].batch {counted} and in_batch {batch.get('in_batch')} differ")
        for run in mine:
            r.need(str(run.get("root", "")).startswith(f"{batch.get('root')}/synth/runs/"), where,
                   f"run {run.get('run')} is not under {batch.get('root')}/synth/runs")
        r.checked[f"batch {source} tasks"] = batch.get("tasks")
        r.checked[f"batch {source} in_batch"] = batch.get("in_batch")
    r.need({src for src, _ in batch_runs} <= sources, w, "batch runs of a source without a batches[] entry")


def check_card(r, card, model, where):
    """One model's accuracy card (or null); returns it when well-formed."""
    if card is None:
        return None
    count_key = "infra" if model == "sc14" else "unscored"
    required = ["passed", "total", count_key, "rollouts"] + (["mean_f2p_ratio"] if model == "sc14" else [])
    if not r.keys(card, required, where):
        return None
    good = True
    for key in ("passed", "total", count_key):
        good &= r.need(is_int(card.get(key)) and card[key] >= 0, f"{where}.{key}", "expected an int >= 0")
    if model == "sc14":
        r.field(card, "mean_f2p_ratio", opt(lambda v: is_num(v) and 0 <= v <= 1), where, "a ratio or null")
    rollouts = card.get("rollouts")
    if not r.need(isinstance(rollouts, list), f"{where}.rollouts", "expected a list"):
        return None
    for position, rollout in enumerate(rollouts):
        here = f"{where}.rollouts[{position}]"
        if r.keys(rollout, ["f2p", "p2p", "score"], here):
            r.field(rollout, "f2p", opt(fraction_ok), here, "'a/b' or null")
            r.field(rollout, "p2p", opt(fraction_ok), here, "'a/b' or null")
            r.field(rollout, "score", lambda v: v in (0, 1, None) and not isinstance(v, bool), here, "1, 0 or null")
    if good and rollouts:
        scores = [item.get("score") for item in rollouts if isinstance(item, dict)]
        r.need(card["passed"] == scores.count(1), where, "passed must count the score-1 rollouts")
        r.need(card["total"] == len(rollouts), where, "total must be the number of rollouts")
        r.need(card[count_key] == scores.count(None), where, f"{count_key} must count the null-score rollouts")
    elif good:
        r.need(card["passed"] + card[count_key] <= card["total"], where, "counts exceed total")
    return card if good else None


# ------------------------------------------------------------------------------------------- data/task/<id>.json

def check_task(r, doc, task_id, index_task, index_run, generated_at, batch=False, older=False):
    """data/task/<task_id>.json; batch (v1.6): data/run/<source>/<run>.json of the batch run index_run (the same
    document, with the batch rules); older (v1.8): data/run/hengyuf/<run>.json of an official run that is not its
    task's latest (the official rules; its stamp is the cycle that last changed it, as a batch run's)."""
    w = index_run.get("detail") if batch or older else f"task/{task_id}.json"
    if not r.keys(doc, ["schema", "generated_at", "task_id", "run", "repo", "build_root", "max_rounds", "branches",
                        "job", "failure_analysis", "plan", "rounds", "nodes"], w):
        return
    r.need(doc.get("schema") == 1, w, "schema must be 1")
    if batch:  # rewritten only when its content changes: the stamp of that cycle, which index.json names
        r.need(doc.get("generated_at") == index_run.get("detail_generated_at"), w,
               "generated_at differs from index runs[].detail_generated_at (not one cycle)")
        r.need(is_num(doc.get("generated_at")) and is_num(generated_at) and doc["generated_at"] <= generated_at, w,
               "generated_at is later than index.json's")
        r.need(doc.get("task_id") == task_id, w, "task_id differs from index runs[].task_id")
        r.need(doc.get("run") == index_run.get("run"), w, "run differs from index runs[].run")
        check_batch_doc_fields(r, doc, index_run, w)
    elif older:  # v1.8: an older official run's document, rewritten only when its content changes
        r.need(doc.get("generated_at") == index_run.get("detail_generated_at"), w,
               "generated_at differs from index runs[].detail_generated_at (not one cycle)")
        r.need(is_num(doc.get("generated_at")) and is_num(generated_at) and doc["generated_at"] <= generated_at, w,
               "generated_at is later than index.json's")
        r.need(doc.get("task_id") == task_id, w, "task_id differs from index runs[].task_id")
        r.need(doc.get("run") == index_run.get("run"), w, "run differs from index runs[].run")
        r.need("source" not in doc, w, "an official run's document carries no batch source")
    else:
        r.need(doc.get("generated_at") == generated_at, w, "generated_at differs from index.json (not one cycle)")
        r.need(doc.get("task_id") == task_id, w, "task_id differs from the file name")
        r.need(doc.get("run") == (index_task or {}).get("run"), w, "run differs from index.json tasks[].run")
    # A launched run whose build directory does not exist yet has no repo, no build root and no tree (rounds []).
    r.field(doc, "repo", opt(is_str), w, "a string or null")
    r.field(doc, "build_root",
            opt(lambda v: is_str(v) and os.path.isabs(v) and f"/synth/runs/{doc.get('run')}/build/" in v),
            w, "the absolute build root of the run, or null")
    r.field(doc, "max_rounds", lambda v: is_int(v) and v >= 0, w, "an int >= 0")
    r.field(doc, "branches", lambda v: is_int(v) and v >= 1, w, "an int >= 1")

    job = doc.get("job")
    if r.keys(job, ["job_id", "slurm_state", "partition", "node", "elapsed", "log_path", "log_tail", "stage"],
              f"{w} job"):
        r.field(job, "log_tail", lambda v: str_list(v, 40) and all(len(line) <= 300 for line in v), f"{w} job",
                "<= 40 lines of <= 300 chars")
        if isinstance(job.get("jobs"), list):  # v1.8: each launched job once (a resolved babysit leg repeats its winner)
            r.need(len(job["jobs"]) == len(set(map(str, job["jobs"]))), f"{w} job", f"jobs repeat a job: {job['jobs']}")
        r.need(job.get("stage") == (index_run or {}).get("stage"), f"{w} job.stage", "differs from index runs[].stage")
        for key in ("job_id", "slurm_state", "partition", "node", "elapsed", "log_path"):
            r.field(job, key, opt(is_str), f"{w} job", "a string or null")

    analysis = doc.get("failure_analysis")
    if analysis is not None and r.keys(analysis, ["patterns", "transferable_signature", "excluded", "pass_stats",
                                                  "markdown"], f"{w} failure_analysis"):
        where = f"{w} failure_analysis"
        patterns = analysis.get("patterns")
        if r.need(isinstance(patterns, list), f"{where}.patterns", "expected a list"):
            for position, pattern in enumerate(patterns):
                here = f"{where}.patterns[{position}]"
                if r.keys(pattern, ["name", "category", "description", "student_slots", "teacher_contrast"], here):
                    for key in ("name", "category", "description", "teacher_contrast", "evidence"):
                        r.field(pattern, key, opt(is_str), here, "a string or null")
                    r.field(pattern, "student_slots", str_list, here, "a list of slot names")
        r.field(analysis, "transferable_signature", opt(is_str), where, "a string or null")
        r.field(analysis, "excluded", lambda v: isinstance(v, list), where, "a list")
        r.field(analysis, "pass_stats", opt(lambda v: isinstance(v, dict)), where, "an object or null")
        r.field(analysis, "markdown", opt(lambda v: is_str(v) and len(v) <= 60000), where, "<= 60000 chars")

    plan = doc.get("plan")
    if plan is not None and r.keys(plan, ["candidates", "mask_plan_excerpt"], f"{w} plan"):
        for position, candidate in enumerate(plan.get("candidates") or []):
            here = f"{w} plan.candidates[{position}]"
            if r.keys(candidate, ["id", "title", "summary", "files", "tests", "built"], here):
                r.field(candidate, "files", lambda v: isinstance(v, list), here, "a list")
                r.field(candidate, "tests", opt(lambda v: is_int(v) or isinstance(v, list)), here, "a count or list")
                r.field(candidate, "built", lambda v: isinstance(v, bool), here, "a bool")
        r.field(plan, "mask_plan_excerpt", opt(lambda v: is_str(v) and len(v) <= 4000), f"{w} plan", "<= 4000 chars")

    nodes = [node for node in doc.get("nodes") or [] if isinstance(node, dict)]
    by_id = {}
    for node in nodes:
        r.need(node.get("id") not in by_id, w, f"duplicate node {node.get('id')}")
        by_id[node.get("id")] = node
    rounds = doc.get("rounds") if isinstance(doc.get("rounds"), list) else []
    r.need([item.get("index") for item in rounds if isinstance(item, dict)] == list(range(len(rounds))), w,
           "rounds[] must be indexed 0..n-1 in order")
    if is_int(doc.get("max_rounds")) and doc.get("build_root") is not None:
        r.need(len(rounds) >= doc["max_rounds"] + 1, w, "rounds[] must cover 0..max_rounds")
    elif doc.get("build_root") is None:
        r.need(not rounds and not doc.get("nodes"), w, "no build root but a tree")
    seeds_by_round = {}
    for item in rounds:
        check_round(r, item, by_id, w, batch)
        if isinstance(item, dict) and isinstance(item.get("seeds"), dict):
            seeds_by_round[item.get("index")] = item["seeds"]
    # The pattern names an evolve branch may aim at (None: no failure analysis shown, nothing to compare with).
    names = ([pattern.get("name") for pattern in analysis["patterns"] if isinstance(pattern, dict)]
             if isinstance(analysis, dict) and isinstance(analysis.get("patterns"), list) else None)
    root = doc.get("build_root")  # runs/<run>/build/<task>/<repo>: an abandoned-* root holds no eligible task (v1.4)
    abandoned_root = is_str(root) and any(part.startswith("abandoned-") for part in Path(root).parts[-2:])
    for node in nodes:
        check_node(r, node, by_id, seeds_by_round, doc.get("max_rounds"), f"{w} {node.get('id')}", names,
                   abandoned_root, batch)
    if batch:  # v1.6: the batch's tasks of this run are its in_batch nodes, every one accepted (check_node)
        flagged = sum(node.get("in_batch") is True for node in nodes)
        r.need(flagged == index_run.get("batch"), w,
               f"{flagged} in_batch nodes, but index runs[].batch is {index_run.get('batch')}")
        r.checked[f"{index_run.get('run')} nodes"] = len(nodes)
        r.checked[f"{index_run.get('run')} in_batch"] = flagged
        return
    label = index_run.get("run") if older else task_id  # v1.8: an older run's counts under its own name
    r.checked[f"{label} nodes"] = len(nodes)
    audit = (index_run or {}).get("audit")
    if isinstance(audit, dict) and any("failure_match" in node for node in nodes):  # v1.3: the counts are the nodes'
        accepted = [node for node in nodes if node.get("decision") in ACCEPTED]
        verdicts = [node["failure_match"].get("verdict") for node in accepted
                    if isinstance(node.get("failure_match"), dict)]
        want = dict({"accepted": len(accepted), "audited": len(verdicts)},
                    **{verdict: verdicts.count(verdict) for verdict in AUDIT_VERDICTS})
        r.need({key: audit.get(key) for key in want} == want, f"{w} failure_match",
               f"index runs[].audit {json.dumps(audit)} != the counts of the accepted nodes {json.dumps(want)}")
        r.checked[f"{label} audited"] = len(verdicts)
    if isinstance(audit, dict) and nodes:  # v1.4: eligibility on every node exactly when the index counts it
        flagged = sum("audit_eligible" in node for node in nodes)
        r.need(flagged in (0, len(nodes)), w, f"audit_eligible on {flagged} of {len(nodes)} nodes")
        r.need(("eligible" in audit) == bool(flagged), w, "index runs[].audit has the eligible counts exactly when "
                                                           "the nodes carry audit_eligible")
    if isinstance(audit, dict) and "eligible" in audit and (not nodes or any("audit_eligible" in n for n in nodes)):
        eligible = [node for node in nodes if node.get("audit_eligible") is True]
        found = [node["failure_match"].get("verdict") for node in eligible
                 if isinstance(node.get("failure_match"), dict)]
        want = {"eligible": len(eligible), "audited_eligible": len(found),
                "eligible_verdicts": {verdict: found.count(verdict) for verdict in AUDIT_VERDICTS}}
        shown = {key: audit.get(key) for key in want}
        r.need(shown == want, f"{w} audit_eligible",
               f"index runs[].audit {json.dumps(shown)} != the counts of the eligible nodes {json.dumps(want)}")
        r.checked[f"{label} eligible"] = len(eligible)
        r.checked[f"{label} audited eligible"] = len(found)


def check_batch_doc_fields(r, doc, run, w):
    """A batch run's document (v1.6): its source fields agree with its index runs[] entry, it is never audited, and its
    job carries what launch.jsonl says, without Slurm state or log (a batch's jobs are not polled, its logs not read)."""
    if not r.keys(doc, ["source", "source_name", "source_label", "root", "outputs", "batch", "audit"], w):
        return
    for key in ("source", "source_name", "source_label", "root", "batch"):
        r.need(doc.get(key) == run.get(key), w, f"{key} differs from index runs[].{key}")
    r.field(doc, "audit", lambda v: v == NOT_AUDITED, w, f"{NOT_AUDITED} (a batch is never audited)")
    r.field(doc, "outputs", lambda v: is_str(v) and os.path.isabs(v) and v.endswith("/outputs"), w,
            "the batch's outputs dir")
    job = doc.get("job")
    if isinstance(job, dict):
        for key in ("slurm_state", "node", "elapsed", "log_path", "started_at", "ended_at"):
            r.need(job.get(key) is None, f"{w} job", f"{key} must be null (a batch's jobs are not polled, its logs "
                                                     "not read)")
        r.need(job.get("log_tail") == [], f"{w} job", "log_tail must be [] (a batch's logs are not read)")
        r.need(job.get("status") == run.get("status"), f"{w} job", "status differs from index runs[].status")


def check_batch_node(r, node, where):
    """A batch run's NODE (v1.6): never audited (failure_match null, audit_eligible false, audit not_audited); in_batch
    true or false; an in_batch node is one of the batch's tasks, so it is built and accepted (every task of the batch
    must be an accepted node)."""
    if not r.keys(node, ["in_batch", "audit", "failure_match", "audit_eligible"], where):
        return
    r.need(node.get("failure_match") is None, where, "failure_match must be null (a batch is never audited)")
    r.need(node.get("audit_eligible") is False, where, "audit_eligible must be False (a batch is never audited)")
    r.need(node.get("audit") == NOT_AUDITED, where, f"audit must be {NOT_AUDITED}")
    if r.need(isinstance(node.get("in_batch"), bool), where, "in_batch must be true or false") and node["in_batch"]:
        r.need(is_str((node.get("build") or {}).get("task_id")), where, "an in_batch node needs its build.task_id")
        r.need(node.get("decision") in ACCEPTED, where,
               f"an in_batch node must be accepted (every task of the batch), got {node.get('decision')}")


def check_round(r, item, by_id, w, batch=False):
    where = f"{w} rounds[{item.get('index') if isinstance(item, dict) else '?'}]"
    if not r.keys(item, ["index", "status", "probe", "abandoned_probes", "seeds"], where):
        return
    statuses = BATCH_ROUND_STATUS if batch else ROUND_STATUS  # v1.6: an ended batch run's rounds
    r.field(item, "status", lambda v: v in statuses, where, f"one of {sorted(statuses)}")
    r.field(item, "abandoned_probes", str_list, where, "a list of probe dirs")
    if batch and r.need("unfinished_probes" in item and str_list(item["unfinished_probes"]), where,
                        "unfinished_probes must list probe dirs"):
        dirs = (item.get("probe") or {}).get("dirs") or []
        r.need(set(item["unfinished_probes"]) <= set(dirs), where, "unfinished_probes must be probe.dirs")
        r.need(not item["unfinished_probes"] or item.get("status") == "unfinished", where,
               "a round with unfinished probes is unfinished")
    probe = item.get("probe")
    if probe is not None and r.keys(probe, ["dir", "tasks", "slots_total", "slots_done", "by_model"], f"{where}.probe"):
        r.field(probe, "dir", lambda v: is_str(v) and re.fullmatch(r"round-\d+/probe-[A-Za-z0-9_-]+", v),
                f"{where}.probe", "round-<k>/probe-<key>")
        for key in ("tasks", "slots_total", "slots_done"):
            r.field(probe, key, lambda v: is_int(v) and v >= 0, f"{where}.probe", "an int >= 0")
        if is_int(probe.get("slots_done")) and is_int(probe.get("slots_total")):
            r.need(probe["slots_done"] <= probe["slots_total"], f"{where}.probe", "slots_done > slots_total")
        by_model = probe.get("by_model")
        r.need(isinstance(by_model, dict) and all(pair_ok(v) for v in by_model.values()), f"{where}.probe.by_model",
               "expected {model: [done, total]}")
        if "live" in probe:
            r.need((item.get("status") == "probing") == bool(probe["live"]), where,
                   "status 'probing' iff the round has a live probe")
    seeds = item.get("seeds")
    if seeds is not None and r.keys(seeds, ["valid", "invalid"], f"{where}.seeds"):
        r.field(seeds, "valid", str_list, f"{where}.seeds", "a list of node ids")
        r.field(seeds, "invalid", lambda v: isinstance(v, dict) and all(is_str(x) for x in v.values()),
                f"{where}.seeds", "{node id: reason}")
        for name in list(seeds.get("valid") or []) + list(seeds.get("invalid") or {}):
            node = by_id.get(name)
            r.need(node is not None and node.get("round") == item.get("index"), f"{where}.seeds",
                   f"{name} is not a node of this round")


def check_node(r, node, by_id, seeds_by_round, max_rounds, where, pattern_names=None, abandoned_root=False,
               batch=False):
    required = ["id", "parent", "lineage", "round", "branch", "kind", "title", "summary", "status", "build", "probe",
                "score", "decision", "decision_reason", "children", "details"]
    if not r.keys(node, required, where):
        return
    if batch:  # v1.6: never audited, in_batch
        check_batch_node(r, node, where)
    node_id = node.get("id")
    match = NODE_ID.fullmatch(node_id) if is_str(node_id) else None
    if not r.need(match, where, "id must be cand-<id>[-r<k>b<j>...]"):
        return
    steps = [(int(k), int(j)) for k, j in re.findall(r"-r(\d+)b(\d+)", match.group(2))]
    r.need([k for k, _ in steps] == list(range(1, len(steps) + 1)), where, "evolve rounds must count up from 1")
    expect = {"parent": node_id[:node_id.rfind("-r")] if steps else None, "lineage": match.group(1),
              "round": steps[-1][0] if steps else 0, "branch": steps[-1][1] if steps else None,
              "kind": "evolve" if steps else "mask"}
    for key, value in expect.items():
        r.need(node.get(key) == value, where, f"{key} must be {value!r} (from the id), got {node.get(key)!r}")
    if "focus_pattern" in node:  # v1.2: task_builder evolve() aims branch j at patterns[j % n] ("overall" if n == 0)
        focus = node["focus_pattern"]
        if not steps:
            r.need(focus is None, where, "a mask node has no focus_pattern")
        elif r.need(focus is None or is_str(focus), where, f"focus_pattern must be a string or null, got {focus!r}") \
                and is_str(focus) and pattern_names is not None:
            want = pattern_names[steps[-1][1] % len(pattern_names)] if pattern_names else "overall"
            r.need(focus == want, where, f"focus_pattern must be {want!r} (branch {steps[-1][1]} of "
                                         f"{len(pattern_names)} patterns), got {focus!r}")
    if "failure_match" in node:  # v1.3: the audit of the node's task, or null
        check_failure_match(r, node["failure_match"], node, f"{where}.failure_match")
    if "withdrawn" in node:  # v1.4: main withdrew the task after accepting it, or null
        check_withdrawn(r, node["withdrawn"], node, f"{where}.withdrawn")
    if "audit_eligible" in node and not batch:  # v1.4: the failure-match audit's scope (a batch has none: v1.6)
        check_eligibility(r, node, seeds_by_round, abandoned_root, where)
    if "seed_invalid" in node:  # v1.5: why the scored task may not seed evolution, or null
        check_seed_invalid(r, node, seeds_by_round, where)
    r.field(node, "title", opt(is_str), where, "a string or null")
    r.field(node, "summary", opt(lambda v: is_str(v) and len(v) <= 600), where, "<= 600 chars")
    r.field(node, "status", lambda v: v in NODE_STATUS, where, f"one of {sorted(NODE_STATUS)}")
    r.field(node, "decision", lambda v: v in DECISIONS, where, f"one of {sorted(DECISIONS)}")
    r.field(node, "decision_reason", lambda v: is_str(v) and v.strip() and "\n" not in v, where, "one line of text")

    build = node.get("build")
    if r.keys(build, ["ok", "attempts", "f2p", "p2p", "f2p_tests", "p2p_tests_sample", "problems", "task_id"],
              f"{where}.build"):
        here = f"{where}.build"
        r.field(build, "ok", lambda v: v in (True, False, None), here, "true, false or null")
        r.field(build, "attempts", opt(lambda v: is_int(v) and v >= 0), here, "an int >= 0")
        r.field(build, "f2p", opt(lambda v: is_int(v) and v >= 0), here, "an int or null")
        r.field(build, "p2p", opt(lambda v: is_int(v) and v >= 0), here, "an int or null")
        r.field(build, "f2p_tests", lambda v: str_list(v, 60), here, "<= 60 test names")
        r.field(build, "p2p_tests_sample", lambda v: str_list(v, 20), here, "<= 20 test names")
        r.field(build, "problems", opt(lambda v: is_str(v) and len(v) <= 3000), here, "<= 3000 chars or null")
        r.field(build, "task_id", opt(is_str), here, "a string or null")
        if build.get("ok") is True:
            r.need(is_str(build.get("task_id")), here, "a built task needs its task_id")

    probe = node.get("probe")
    if probe is not None and r.keys(probe, ["progress", "student", "teacher", "rollouts"], f"{where}.probe"):
        here = f"{where}.probe"
        r.need(isinstance(probe.get("progress"), dict) and all(pair_ok(v) for v in probe["progress"].values()),
               f"{here}.progress", "expected {model: [done, total]}")
        if r.keys(probe.get("student"), ["pass", "fail", "infra", "rate"], f"{here}.student"):
            r.field(probe["student"], "rate", opt(lambda v: is_num(v) and 0 <= v <= 1), f"{here}.student", "a rate")
        if r.keys(probe.get("teacher"), ["pass", "fail", "infra", "clean_rate", "cheated"], f"{here}.teacher"):
            r.field(probe["teacher"], "clean_rate", opt(lambda v: is_num(v) and 0 <= v <= 1), f"{here}.teacher",
                    "a rate")
        for position, row in enumerate(probe.get("rollouts") or []):
            spot = f"{here}.rollouts[{position}]"
            if r.keys(row, ["slot", "model", "status", "reward", "f2p", "p2p", "tool_calls", "patch_chars",
                            "agent_error", "verifier_error"], spot):
                r.field(row, "status", lambda v: v in ROLLOUT_STATUS, spot, f"one of {sorted(ROLLOUT_STATUS)}")
                r.field(row, "f2p", opt(fraction_ok), spot, "'a/b' or null")
                r.field(row, "p2p", opt(fraction_ok), spot, "'a/b' or null")
                r.field(row, "reward", opt(is_num), spot, "a number or null")
                r.field(row, "slot", lambda v: is_str(v) and v.startswith(f"{row.get('model')}-r"), spot,
                        "<model>-r<n>")
        check_partial(r, node, probe, here)

    score = node.get("score")
    if score is not None and r.keys(score, ["accept", "reason", "similarity", "similarity_detail", "valid",
                                            "grading_failures", "answer_retrieval", "target_student_rate",
                                            "target_teacher_rate"], f"{where}.score"):
        here = f"{where}.score"
        r.field(score, "accept", lambda v: isinstance(v, bool), here, "a bool")
        r.field(score, "similarity", opt(is_num), here, "a number or null")
        r.field(score, "similarity_detail", opt(lambda v: is_str(v) and len(v) <= 4000), here, "<= 4000 chars")
        r.field(score, "grading_failures", str_list, here, "a list of slots")
        r.field(score, "valid", opt(lambda v: isinstance(v, list) and all(is_num(x) for x in v)), here,
                "[student, teacher] valid rollouts")

    details = node.get("details")
    if r.keys(details, list(DETAIL_LIMITS) + ["teacher_audit"], f"{where}.details"):
        for key, limit in DETAIL_LIMITS.items():
            r.field(details, key, opt(lambda v, limit=limit: is_str(v) and len(v) <= limit), f"{where}.details",
                    f"<= {limit} chars or null")
        check_audit(r, details.get("teacher_audit"), probe, f"{where}.details.teacher_audit")

    # Tree structure: children <-> parents.
    children = node.get("children")
    if r.need(str_list(children), f"{where}.children", "expected a list of node ids"):
        for child in children:
            r.need(child in by_id and by_id[child].get("parent") == node_id, f"{where}.children",
                   f"{child} is not a node whose parent is {node_id}")
        missing = [other for other, item in by_id.items() if item.get("parent") == node_id and other not in children]
        r.need(not missing, f"{where}.children", f"misses {missing}")
    if node.get("parent") is not None:
        r.need(node["parent"] in by_id, where, f"parent {node['parent']} is not a node")
    check_decision(r, node, by_id, seeds_by_round, max_rounds, where)


def check_partial(r, node, probe, where):
    """v1.2, a probing node's live counts: partial {model: {done, total, pass, fail, infra}} over its graded / final
    slots agrees with progress [done, total] and with the student / teacher counts; own_done iff every own slot is
    done (metrics.json, i.e. status scored, comes only once the whole campaign is)."""
    partial = probe.get("partial")
    if partial is None and "own_done" not in probe:
        return
    r.need(node.get("status") == "probing" and probe.get("live") is True, where,
           "partial / own_done belong to a probing node's live probe")
    if partial is not None and r.need(isinstance(partial, dict), f"{where}.partial", "expected {model: counts}"):
        progress = probe.get("progress") if isinstance(probe.get("progress"), dict) else {}
        r.need(set(partial) == set(progress), f"{where}.partial", "must cover the models of progress")
        for model, counts in partial.items():
            spot = f"{where}.partial.{model}"
            if not r.keys(counts, ["done", "total", "pass", "fail", "infra"], spot) or not r.need(
                    all(is_int(counts.get(k)) and counts[k] >= 0 for k in ("done", "total", "pass", "fail", "infra")),
                    spot, "expected ints >= 0"):
                continue
            r.need(counts["pass"] + counts["fail"] + counts["infra"] == counts["done"] <= counts["total"], spot,
                   "expected pass + fail + infra == done <= total")
            r.need(progress.get(model) == [counts["done"], counts["total"]], spot,
                   "differs from progress [done, total]")
        for model, key in ((STUDENT, "student"), (TEACHER, "teacher")):
            counts, shown = partial.get(model), probe.get(key)
            if isinstance(counts, dict) and isinstance(shown, dict):
                r.need(all(counts.get(k) == shown.get(k) for k in ("pass", "fail", "infra")),
                       f"{where}.partial.{model}", f"pass/fail/infra differ from probe.{key}")
    if "own_done" in probe and r.need(isinstance(probe["own_done"], bool), f"{where}.own_done", "expected a bool"):
        if isinstance(partial, dict) and partial:
            done = all(isinstance(c, dict) and c.get("done") == c.get("total") for c in partial.values())
            r.need(probe["own_done"] == done, f"{where}.own_done", f"must be {done} (every own slot done)")


def check_audit(r, audit, probe, where):
    """NODE.details.teacher_audit: null (not scored); since v1.2 the list [{slot, cheated, categories, summary <= 400
    chars, rubric_score?}], one entry per passing teacher rollout (task_builder score() audits exactly those); from an
    older collector, the JSON-dumped teacher_audit.json (<= 6000 chars). Both shapes are accepted."""
    if audit is None:
        return
    if is_str(audit):
        r.need(len(audit) <= 6000, where, "the JSON-dumped audit must be <= 6000 chars")
        return
    if not r.need(isinstance(audit, list), where, "expected a list of verdicts, a JSON string or null"):
        return
    for position, entry in enumerate(audit):
        here = f"{where}[{position}]"
        if r.keys(entry, ["slot", "cheated", "categories", "summary"], here):
            r.field(entry, "slot", lambda v: is_str(v) and v.startswith(f"{TEACHER}-r"), here, f"a {TEACHER}-r<n> slot")
            r.field(entry, "cheated", lambda v: v is None or isinstance(v, bool), here, "true, false or null")
            r.field(entry, "categories", lambda v: str_list(v, 20), here, "<= 20 category names")
            r.field(entry, "summary", opt(lambda v: is_str(v) and len(v) <= 400), here, "<= 400 chars or null")
            r.field(entry, "rubric_score", opt(is_num), here, "a number or null")
    slots = sorted(str(entry.get("slot")) for entry in audit if isinstance(entry, dict))
    rows = probe.get("rollouts") if isinstance(probe, dict) and not probe.get("live") else None
    if isinstance(rows, list) and rows:
        passes = sorted(str(row.get("slot")) for row in rows if isinstance(row, dict) and row.get("model") == TEACHER
                        and row.get("status") == "pass")
        r.need(slots == passes, where, f"audited slots {slots} != the teacher's passing rollouts {passes}")


def check_audit_counts(r, run, where):
    """index.json runs[].audit (v1.3): {accepted (= runs[].accepted), audited (<= accepted), strong, partial, weak,
    none (summing to audited)}; v1.4 adds the same over the run's eligible tasks: eligible, audited_eligible
    (<= eligible) and eligible_verdicts {strong, partial, weak, none} (summing to audited_eligible)."""
    audit = run["audit"]
    keys = ["accepted", "audited"] + AUDIT_VERDICTS
    if not r.keys(audit, keys, where):
        return
    for key in keys:
        r.field(audit, key, lambda v: is_int(v) and v >= 0, where, "an int >= 0")
    if all(is_int(audit.get(key)) for key in keys):
        r.need(audit["accepted"] == run.get("accepted"), where, "accepted must equal runs[].accepted")
        r.need(audit["audited"] <= audit["accepted"], where, "audited must be <= accepted")
        r.need(sum(audit[verdict] for verdict in AUDIT_VERDICTS) == audit["audited"], where,
               "strong + partial + weak + none must equal audited")
    v14 = ["eligible", "audited_eligible", "eligible_verdicts"]
    if not any(key in audit for key in v14):  # a collector that predates v1.4
        return
    r.need(all(key in audit for key in v14), where, f"missing keys {[key for key in v14 if key not in audit]}")
    for key in ("eligible", "audited_eligible"):
        r.field(audit, key, lambda v: is_int(v) and v >= 0, where, "an int >= 0")
    if is_int(audit.get("eligible")) and is_int(audit.get("audited_eligible")):
        r.need(audit["audited_eligible"] <= audit["eligible"], where, "audited_eligible must be <= eligible")
    verdicts = audit.get("eligible_verdicts")
    if "eligible_verdicts" in audit and r.keys(verdicts, AUDIT_VERDICTS, f"{where}.eligible_verdicts"):
        for key in AUDIT_VERDICTS:
            r.field(verdicts, key, lambda v: is_int(v) and v >= 0, f"{where}.eligible_verdicts", "an int >= 0")
        if is_int(audit.get("audited_eligible")) and all(is_int(verdicts.get(key)) for key in AUDIT_VERDICTS):
            r.need(sum(verdicts[key] for key in AUDIT_VERDICTS) == audit["audited_eligible"], where,
                   "eligible_verdicts must sum to audited_eligible")


def check_withdrawn(r, withdrawn, node, where):
    """NODE.withdrawn (v1.4): null, or {at, by, why} (each a string within its limit, or null) for a task main
    withdrew after accepting it; such a task is never accepted."""
    if withdrawn is None:
        return
    if not r.keys(withdrawn, list(WITHDRAWN_LIMITS), where):
        return
    for key, limit in WITHDRAWN_LIMITS.items():
        r.field(withdrawn, key, opt(lambda v, limit=limit: is_str(v) and len(v) <= limit), where,
                f"<= {limit} chars or null")
    r.need(node.get("decision") not in ACCEPTED, where, f"a withdrawn task is never accepted, got {node.get('decision')}")


def accepted_task(node):
    """Whether the node's task stands accepted: its metrics.json accept (score.accept) and main did not withdraw it
    (v1.4). Since v1.5 that decides accepted / accepted_evolve whatever its round's seeds.json says."""
    score = node.get("score")
    return isinstance(score, dict) and score.get("accept") is True and not node.get("withdrawn")


def expected_eligibility(node, seeds_by_round, abandoned_root):
    """(NODE.audit_eligible as CONTRACT "v1.4" defines it, why), from the node's own fields: built and probed (a score),
    a graded sc14 rollout, and not discarded (invalid in its round's seeds.json and not accepted, v1.5; without a
    seeds.json verdict, the decision discard of the recomputed seed_check()), not withdrawn, not under an abandoned-*
    build root. (None, ...) when the file cannot tell: an abandoned node its round's seeds.json does not list."""
    probe = node.get("probe") if isinstance(node.get("probe"), dict) else {}
    rows = [row for row in probe.get("rollouts") or [] if isinstance(row, dict)]
    seeds = seeds_by_round.get(node.get("round")) or {}
    if abandoned_root:
        return False, "under an abandoned-* build root"
    if node.get("withdrawn"):
        return False, "withdrawn"
    if not isinstance(node.get("score"), dict) or (node.get("build") or {}).get("ok") is False:
        return False, "not scored"
    if not any(row.get("model") == STUDENT and row.get("status") in ("pass", "fail") for row in rows):
        return False, "no graded sc14 rollout"
    if node.get("id") in (seeds.get("invalid") or {}):
        if accepted_task(node):  # v1.5: accepted, so not discarded; seeds.json bars it from seeding evolution only
            return True, "scored, a graded sc14 rollout, accepted (seeds.json lists it invalid as a seed only)"
        return False, "invalid in its round's seeds.json"
    if node.get("id") in (seeds.get("valid") or []):
        return True, "scored, a graded sc14 rollout, valid in seeds.json"
    if node.get("decision") == "discard":
        return False, "discarded by the recomputed seed_check()"
    if node.get("decision") == "abandoned":
        return None, "abandoned, validity not shown"
    return True, "scored, a graded sc14 rollout, not discarded"


def check_eligibility(r, node, seeds_by_round, abandoned_root, where):
    """NODE.audit_eligible (v1.4): true or false, as expected_eligibility() recomputes it."""
    flag = node["audit_eligible"]
    if not r.need(isinstance(flag, bool), where, f"audit_eligible must be true or false, got {json.dumps(flag)}"):
        return
    want, why = expected_eligibility(node, seeds_by_round, abandoned_root)
    if want is not None:
        r.need(flag is want, where, f"audit_eligible must be {want} ({why})")


def check_seed_invalid(r, node, seeds_by_round, where):
    """NODE.seed_invalid (v1.5): null, or one line saying why the scored task may not seed evolution: the reason of its
    round's seeds.json when that file lists it invalid (null when it lists it valid), else seed_check() recomputed. Null
    for a task never scored and for a failed build. It discards only a task that was not accepted (the discard's
    decision_reason); an accepted task stays accepted (never accepted_evolve: an invalid seed seeds no branch), with the
    reason as a note; no other decision goes with it (abandoned aside)."""
    note = node["seed_invalid"]
    if not r.need(note is None or (is_str(note) and note.strip() and "\n" not in note), where,
                  f"seed_invalid must be null or one line of text, got {json.dumps(note)[:80]}"):
        return
    if not isinstance(node.get("score"), dict) or (node.get("build") or {}).get("ok") is False:
        r.need(note is None, where, f"seed_invalid must be null for a task never scored or not built, got {note!r}")
        return
    seeds = seeds_by_round.get(node.get("round")) or {}
    invalid = seeds.get("invalid") if isinstance(seeds.get("invalid"), dict) else {}
    if node.get("id") in invalid:
        want = " ".join(str(invalid[node["id"]]).split())
        r.need(note == want, where, f"seed_invalid must be its round's seeds.json reason {want!r}, got {note!r}")
    elif node.get("id") in (seeds.get("valid") or []):
        r.need(note is None, where, f"seeds.json lists it as a valid seed, but seed_invalid is {note!r}")
    decision = node.get("decision")
    if decision == "discard":
        r.need(note is not None and note == node.get("decision_reason"), where,
               "a scored task is discarded only as an invalid seed: its decision_reason is its seed_invalid")
    elif note is not None and decision != "abandoned":
        want = "accepted" if accepted_task(node) else "discard"
        r.need(decision == want, where, f"a task with seed_invalid is {want} (acceptance beats an invalid seed, which "
                                        f"never seeds branches), got {decision}")


def check_failure_match(r, match, node, where):
    """NODE.failure_match (v1.3): null, or an auditor's verdict on the node's task (does sc14 repeat the target's failure
    patterns there?) as the collector validated and clipped it from runs/<run>/audit/failure_match/<task_id>.json."""
    if match is None:
        return
    if not r.keys(match, FAILURE_MATCH_KEYS, where):
        return
    r.need(is_str((node.get("build") or {}).get("task_id")), where, "only a built task (build.task_id) is audited")
    r.field(match, "verdict", lambda v: v in AUDIT_VERDICTS, where, f"one of {AUDIT_VERDICTS}")
    for key, limit in (("summary", 1000), ("other_causes", 400), ("audited_at", 64)):
        r.field(match, key, opt(lambda v, limit=limit: is_str(v) and len(v) <= limit), where, f"<= {limit} chars or null")
    patterns = match.get("patterns")
    if r.need(isinstance(patterns, list) and len(patterns) <= 20, f"{where}.patterns", "expected <= 20 patterns"):
        for position, item in enumerate(patterns):
            here = f"{where}.patterns[{position}]"
            if r.keys(item, ["name", "reproduced", "slots", "evidence"], here):
                r.field(item, "name", opt(lambda v: is_str(v) and len(v) <= 300), here, "<= 300 chars or null")
                r.field(item, "reproduced", lambda v: v is None or v in AUDIT_REPRODUCED, here, "yes, partial, no or null")
                r.field(item, "slots", lambda v: str_list(v, 20) and all(len(slot) <= 100 for slot in v), here,
                        "<= 20 slots")
                r.field(item, "evidence", opt(lambda v: is_str(v) and len(v) <= 600), here, "<= 600 chars or null")
    rollouts = match.get("student_rollouts")
    if r.need(isinstance(rollouts, list) and len(rollouts) <= 10, f"{where}.student_rollouts",
              "expected <= 10 rollouts"):
        for position, item in enumerate(rollouts):
            here = f"{where}.student_rollouts[{position}]"
            if r.keys(item, ["slot", "status", "failing_tests", "cause", "pattern"], here):
                r.field(item, "slot", opt(lambda v: is_str(v) and len(v) <= 100), here, "<= 100 chars or null")
                r.field(item, "status", lambda v: v is None or v in ROLLOUT_STATUS, here,
                        f"one of {sorted(ROLLOUT_STATUS)} or null")
                r.field(item, "failing_tests", lambda v: str_list(v, 8) and all(len(name) <= 300 for name in v), here,
                        "<= 8 test names")
                for key in ("cause", "pattern"):
                    r.field(item, key, opt(lambda v: is_str(v) and len(v) <= 300), here, "<= 300 chars or null")
    judge = match.get("similarity_judge")
    if judge is not None and r.keys(judge, ["score", "agrees", "comment"], f"{where}.similarity_judge"):
        here = f"{where}.similarity_judge"
        r.field(judge, "score", opt(is_num), here, "a number or null")
        r.field(judge, "agrees", lambda v: v is None or isinstance(v, bool), here, "true, false or null")
        r.field(judge, "comment", opt(lambda v: is_str(v) and len(v) <= 300), here, "<= 300 chars or null")


def check_decision(r, node, by_id, seeds_by_round, max_rounds, where):
    """CONTRACT "Decision rules" as invariants between a node's fields, its round's seeds.json and its relatives."""
    status, decision, k = node.get("status"), node.get("decision"), node.get("round")
    build, score = node.get("build") or {}, node.get("score")
    # v1.4: a task main withdrew is decided as not accepted, whatever its metrics.json still says
    accept = accepted_task(node)
    seeds = seeds_by_round.get(k) or {}
    parent = by_id.get(node.get("parent"))
    live_children = [by_id[c] for c in node.get("children") or [] if c in by_id
                     and by_id[c].get("status") != "abandoned"]
    r.need((status == "abandoned") == (decision == "abandoned"), where, "status and decision 'abandoned' go together")
    parent_seeds = seeds_by_round.get((parent or {}).get("round")) or {}
    if parent is not None and parent.get("id") in (parent_seeds.get("invalid") or {}):
        r.need(decision == "abandoned", where, "a child of an invalid seed must be abandoned")
    if decision == "abandoned":
        return
    if build.get("ok") is False:
        r.need(status == "build_failed" and decision == "discard"
               and str(node.get("decision_reason", "")).startswith("build failed"), where,
               "a failed build is status build_failed, decision discard ('build failed: ...')")
        return
    listed_invalid = node.get("id") in (seeds.get("invalid") or {})
    if listed_invalid:  # v1.5: acceptance beats an invalid seed (seeds.json only picks the parents of the next round)
        r.need(decision == ("accepted" if accept else "discard"), where,
               f"listed invalid in seeds.json and {'accepted' if accept else 'not accepted'}: must be "
               f"{'accepted' if accept else 'discard'}, got {decision}")
    if score is None:
        r.need(decision == "pending", where, f"an unscored node must be pending, got {decision}")
        r.need(status in ("planning", "authoring", "building", "built", "probing"), where,
               f"an unscored node cannot be {status}")
        if status == "probing":
            r.need(isinstance(node.get("probe"), dict), where, "a probing node needs its probe progress")
        return
    r.need(status == "scored", where, f"a scored node must have status scored, got {status}")
    valid_seed = node.get("id") in (seeds.get("valid") or [])
    # an invalid seed: listed so in seeds.json, else (before that file is written) the collector's seed_check()
    invalid_seed = listed_invalid or (not valid_seed and is_str(node.get("seed_invalid")))
    if decision == "discard":
        r.need(listed_invalid or not valid_seed, where, "discard of a node seeds.json lists as valid")
        r.need(not accept, where, "an accepted task is never discarded (v1.5: acceptance beats an invalid seed)")
    elif k == 0 and (not is_int(max_rounds) or max_rounds >= 1):
        want = (("accepted" if invalid_seed else "accepted_evolve") if accept
                else "discard" if invalid_seed else "evolve")
        r.need(decision == want, where, f"a round-0 node, {'accepted' if accept else 'not accepted'} and "
                                        f"{'an invalid' if invalid_seed else 'a valid'} seed, is {want}, "
                                        f"got {decision}")
    elif accept:
        r.need(decision == "accepted", where, f"an accepted round>=1 node is accepted, got {decision}")
    elif decision == "evolve":
        r.need(bool(live_children), where, "evolve (round >= 1) needs children")
    elif decision == "not_selected":
        r.need(not live_children, where, "not_selected with children")
        r.need(any(other.get("round") == k + 1 and other.get("lineage") == node.get("lineage")
                   and other.get("status") != "abandoned" for other in by_id.values()), where,
               "not_selected needs a next-round node of the same lineage")
    elif decision == "end":
        r.need(is_int(max_rounds) and k >= max_rounds, where, "end before the last round")
    elif decision != "pending":
        r.add(where, f"decision {decision} does not fit a scored round-{k} node")


# ------------------------------------------------------------------------------------------- fleet.json, api.json

def section(r, doc, key, required, where, degraded):
    """doc[key] when it is an object with the required keys; a null/missing section is a problem unless the collector
    reported its source as broken (degraded), in which case its fallback document leaves it out."""
    value = doc.get(key)
    if value is None and degraded:
        return None
    return value if r.keys(value, required, f"{where} {key}") else None


def check_fleet(r, fleet, generated_at, degraded=False):
    """fleet.json; degraded: index.json reports a broken fleet/usage source (sections may then be null)."""
    w = "fleet.json"
    sections = ["daemon", "config", "coverage", "counts", "workers", "totals", "router", "history"]
    if not r.keys(fleet, ["schema", "generated_at"] + ([] if degraded else sections), w):
        return
    r.need(fleet.get("schema") == 1, w, "schema must be 1")
    r.need(fleet.get("generated_at") == generated_at, w, "generated_at differs from index.json (not one cycle)")
    section(r, fleet, "daemon", ["job_id", "host", "heartbeat_age_s", "status_age_s", "breaker"], w, degraded)
    section(r, fleet, "config", ["target", "horizon_effective_min", "horizon_min", "stagger_min", "max_jobs", "time",
                                 "coverage_partitions"], w, degraded)
    section(r, fleet, "coverage", ["value", "target"], w, degraded)
    counts = section(r, fleet, "counts", FLEET_BUCKETS, w, degraded)
    workers = fleet.get("workers") if isinstance(fleet.get("workers"), list) else []
    if counts:  # the fleet daemon's own buckets (not recounted from workers)
        r.need(all(is_int(counts.get(key)) and counts[key] >= 0 for key in FLEET_BUCKETS), f"{w} counts",
               "expected ints >= 0")
    for position, worker in enumerate(workers):
        here = f"{w} workers[{position}]"
        if not r.keys(worker, ["job_id", "slot", "partition", "node", "state", "registry_status", "ready_at", "start",
                               "end", "left_s", "heartbeat_age_s", "base_url", "pending_reason", "est_start",
                               "metrics"], here):
            continue
        r.field(worker, "base_url", opt(lambda v: is_str(v) and re.fullmatch(r"http://[A-Za-z0-9.-]+:\d+(/v1)?/?", v)),
                here, "http://host:port/v1 (no credentials)")
        metrics = worker.get("metrics")
        if metrics is not None and r.keys(metrics, ["running", "waiting", "kv_usage", "gen_tps", "scraped_at",
                                                    "error"], f"{here}.metrics"):
            r.field(metrics, "kv_usage", opt(lambda v: is_num(v) and 0 <= v <= 1), f"{here}.metrics", "a fraction")
            check_prompt_tps(r, metrics, f"{here}.metrics")
            for key in ("prompt_tps_incl_cache", "prompt_tps", "gen_tps"):
                r.field(metrics, key, opt(lambda v: is_num(v) and v >= 0), f"{here}.metrics", "tok/s >= 0 or null")
    totals = section(r, fleet, "totals", ["routable", "running", "waiting", "kv_usage_mean", "gen_tps"], w, degraded)
    if totals:
        check_prompt_tps(r, totals, f"{w} totals")
        if "busy_workers" in totals:  # v1.2: routable workers with at least 1 running request
            busy = sum(1 for worker in workers if isinstance(worker, dict) and worker.get("routing")
                       and isinstance(worker.get("metrics"), dict) and worker["metrics"].get("error") is None
                       and is_num(worker["metrics"].get("running")) and worker["metrics"]["running"] >= 1)
            r.need(totals["busy_workers"] == busy and is_int(totals["busy_workers"]), f"{w} totals.busy_workers",
                   f"must count the routable workers with >= 1 running request ({busy}), got {totals['busy_workers']}")
    router = section(r, fleet, "router", ["window_s", "requests", "errors", "error_rate", "p50_s", "p90_s",
                                          "prompt_tokens", "completion_tokens", "clamps", "by_worker", "last_hour"],
                     w, degraded)
    if router:
        if "scope" in router:  # v1.2: which journals the stats cover
            r.field(router, "scope", lambda v: v == ROUTER_SCOPE, f"{w} router", repr(ROUTER_SCOPE))
        r.field(router, "error_rate", opt(lambda v: is_num(v) and 0 <= v <= 1), f"{w} router", "a fraction")
        r.field(router, "by_worker", lambda v: isinstance(v, dict) and all(
            isinstance(x, dict) and {"requests", "errors", "p90_s"} <= set(x) for x in v.values()), f"{w} router",
            "{job_id: {requests, errors, p90_s}}")
        r.keys(router.get("last_hour"), ["requests", "errors"], f"{w} router.last_hour")
    history = fleet.get("history") if isinstance(fleet.get("history"), list) else []
    check_series(r, history, ["t", "ready", "routable", "running", "waiting", "kv", "gen_tps", "req", "err", "p90_s"],
                 generated_at, DAY + 600, f"{w} history")
    for position, point in enumerate(history):  # v1.2 points add prompt_tps_incl_cache (older points lack it)
        if isinstance(point, dict):
            r.field(point, "prompt_tps_incl_cache", opt(lambda v: is_num(v) and v >= 0), f"{w} history[{position}]",
                    "tok/s >= 0 or null")
    r.checked["workers"] = len(workers)
    r.checked["fleet history points"] = len(history)


def check_prompt_tps(r, doc, where):
    """Prompt tok/s, named prompt_tps_incl_cache since v1.2 (vLLM's prompt counter includes prefix-cache hits); the
    old name prompt_tps stays one release as an alias: at least one of them, equal when both are present."""
    if r.need("prompt_tps_incl_cache" in doc or "prompt_tps" in doc, where,
              "missing keys ['prompt_tps_incl_cache'] (or the old 'prompt_tps')"):
        if "prompt_tps_incl_cache" in doc and "prompt_tps" in doc:
            r.need(doc["prompt_tps"] == doc["prompt_tps_incl_cache"], where,
                   "prompt_tps (the old name) must equal prompt_tps_incl_cache")


def check_api(r, api, generated_at, run_names, degraded=False):
    """api.json; degraded: index.json reports a broken api/usage source (the fallback document has no sections)."""
    w = "api.json"
    if not r.keys(api, ["schema", "generated_at"] + ([] if degraded else ["hub", "burn", "history", "by_run",
                                                                          "notes"]), w):
        return
    r.need(api.get("schema") == 1, w, "schema must be 1")
    r.need(api.get("generated_at") == generated_at, w, "generated_at differs from index.json (not one cycle)")
    hub = section(r, api, "hub", ["spend", "max_budget", "remaining", "status", "sampled_at", "key", "error"], w,
                  degraded)
    if hub:
        for key in ("spend", "max_budget", "remaining", "sampled_at"):
            r.field(hub, key, opt(is_num), f"{w} hub", "a number or null")
        r.field(hub, "key", opt(lambda v: is_str(v) and KEY_NAME.fullmatch(v)), f"{w} hub",
                "the NAME of the key variable")
        if all(is_num(hub.get(key)) for key in ("spend", "max_budget", "remaining")):
            r.need(abs(hub["max_budget"] - hub["spend"] - hub["remaining"]) < 0.05, f"{w} hub",
                   "remaining must be max_budget - spend")
    burn = section(r, api, "burn", ["per_hour_1h", "per_hour_6h", "eta_exhaust_iso"], w, degraded)
    if burn:
        for key in ("per_hour_1h", "per_hour_6h"):
            r.field(burn, key, opt(is_num), f"{w} burn", "$/h or null")
        r.field(burn, "eta_exhaust_iso", opt(lambda v: iso_epoch(v) is not None), f"{w} burn", "an ISO time or null")
        check_burn(r, burn, hub, f"{w} burn")
    scan = api.get("scan")
    if scan is not None and r.keys(scan, ["complete", "pending_bytes"], f"{w} scan"):
        r.field(scan, "complete", lambda v: isinstance(v, bool), f"{w} scan", "a bool")
        r.field(scan, "pending_bytes", lambda v: is_int(v) and v >= 0, f"{w} scan", "bytes >= 0")
        if scan.get("complete") is True:
            r.need(scan.get("pending_bytes") == 0, f"{w} scan", "a complete scan has no pending bytes")
    history = api.get("history") if isinstance(api.get("history"), list) else []
    check_series(r, history, ["t", "spend", "max_budget"], generated_at, 30 * DAY + DAY, f"{w} history")
    for position, row in enumerate(api.get("by_run") or []):
        here = f"{w} by_run[{position}]"
        if not r.keys(row, ["run", "kimi", "codex", "glm"], here):
            continue
        r.need(row.get("run") in run_names, here, f"{row.get('run')!r} is not an index.json run")
        # kimi / codex are null for a run without any such usage yet (e.g. before its first probe).
        if row.get("kimi") is not None:
            r.keys(row["kimi"], ["calls", "cost", "input_tokens", "cached_tokens", "output_tokens"], f"{here}.kimi")
        if row.get("codex") is not None:
            r.keys(row["codex"], ["sessions", "input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens"],
                   f"{here}.codex")
        r.keys(row.get("glm"), ["calls"], f"{here}.glm")
    if "notes" in api:
        r.field(api, "notes", str_list, w, "a list of strings")


def check_burn(r, burn, hub, where):
    """window_min_1h / window_min_6h (v1.2) are the spans the rates were measured over: a rate needs >= 2 hub samples
    inside its window, so a rate and its window are null together. eta_exhaust (with its ISO form) projects the hub
    sample's remaining budget at the HIGHER of the 1 h and 6 h rates (CONTRACT "Integration notes (v1.1)": conservative
    while probes burst; the page names the one used; collect.py _burn()) from the sample's own time, not from the
    cycle's."""
    for label, minutes in (("1h", 60), ("6h", 360)):
        key = f"window_min_{label}"
        if key in burn:
            r.field(burn, key, opt(lambda v, minutes=minutes: is_num(v) and 0 < v <= minutes + 0.05), where,
                    f"minutes in (0, {minutes}] or null")
            r.need((burn[key] is None) == (burn.get(f"per_hour_{label}") is None), where,
                   f"per_hour_{label} and {key} must be null together")
    eta = burn.get("eta_exhaust")
    if eta is not None and r.need(is_num(eta), where, "eta_exhaust must be epoch seconds or null"):
        stamp = iso_epoch(burn.get("eta_exhaust_iso"))
        r.need(stamp is not None and abs(stamp - eta) < 1.5, where,
               "eta_exhaust_iso must be the ISO form of eta_exhaust")
    rates = [value for value in (burn.get("per_hour_1h"), burn.get("per_hour_6h")) if is_num(value)]
    rate = max(rates) if rates else None
    hub = hub if isinstance(hub, dict) else {}
    if "window_min_1h" in burn and all(is_num(v) for v in (rate, hub.get("sampled_at"), hub.get("remaining"))) \
            and rate > 0:
        want = hub["sampled_at"] + max(0.0, hub["remaining"]) / rate * 3600
        r.need(is_num(eta) and abs(eta - want) <= 2, where,
               f"eta_exhaust must be hub.sampled_at + remaining / rate ({round(want)}), got {eta}")


def check_series(r, points, keys, generated_at, max_age, where):
    """History points: objects with `keys`, time-ordered, none older than max_age or from the future."""
    times = []
    for position, point in enumerate(points):
        if r.keys(point, keys, f"{where}[{position}]") and is_num(point.get("t")):
            times.append(point["t"])
    r.need(times == sorted(times), where, "points must be in time order")
    if times and is_num(generated_at):
        r.need(times[0] >= generated_at - max_age, where, f"oldest point is older than {max_age / DAY:g} days")
        r.need(times[-1] <= generated_at + 120, where, "a point lies in the future")


def check_jsonl(r, path, max_age, now):
    """An append-only history file: one JSON object with "t" per line, in time order, trimmed."""
    rows, bad = [], 0
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
                rows.append(float(row["t"]))
            except (ValueError, KeyError, TypeError):
                bad += 1
    where = str(path.relative_to(path.parent.parent))
    r.need(bad == 0, where, f"{bad} unparsable line(s)")
    r.need(rows == sorted(rows), where, "lines must be in time order")
    if rows and now:
        r.need(rows[0] >= now - max_age, where, f"not trimmed: oldest line older than {max_age / DAY:g} days")


# ------------------------------------------------------------------------------------------- explorer (v1.7)

def explorer_documents(index, task_docs, batch_docs, official_docs=None):
    """(source, detail path, document, latest) of every document the Explorer covers, in its runs[] order: each official
    run (index.json runs[] order; v1.8: every one, the latest of its task with data/task/<task_id>.json, an older one
    with data/run/hengyuf/<run>.json), then each batch run (runs[] entries with a source) and its
    data/run/<source>/<run>.json (latest None)."""
    out = []
    for run in index.get("runs") or []:
        if isinstance(run, dict) and "source" not in run:
            if run.get("latest") is False:
                doc = (official_docs or {}).get(run.get("detail"))
                if isinstance(doc, dict) and doc.get("run") == run.get("run"):
                    out.append((OFFICIAL["source"], run.get("detail"), doc, False))
                continue
            doc = task_docs.get(run.get("task_id"))
            if isinstance(doc, dict) and doc.get("run") == run.get("run"):
                out.append((OFFICIAL["source"], f"task/{run.get('task_id')}.json", doc, True))
    for run in index.get("runs") or []:
        if isinstance(run, dict) and "source" in run and isinstance(batch_docs.get(run.get("detail")), dict):
            out.append((run.get("source"), run.get("detail"), batch_docs[run["detail"]], None))
    return out


def _finite(value):
    return value if is_num(value) else None


def _loose(name):
    return " ".join(name.split()).casefold() if is_str(name) else None


def explorer_expected(documents, sources):
    """explorer.json's runs, targets, task rows, trajectory rows and totals as CONTRACT "v1.7" derives them from the task
    and run files (an implementation of the rules independent of collect.py's)."""
    audited = {source.get("source"): source.get("audited") is True for source in sources if isinstance(source, dict)}
    totals = {source.get("source"): dict.fromkeys(EXPLORER_TOTALS, 0) for source in sources if isinstance(source, dict)}
    runs, targets, tasks, trajectories = [], {}, [], []
    for source, detail, doc, latest in documents:
        total = totals.setdefault(source, dict.fromkeys(EXPLORER_TOTALS, 0))
        batch = source != OFFICIAL["source"]
        target, name = doc.get("task_id"), doc.get("run")
        quote = lambda text: urllib.parse.quote(str(text), safe="-_.!~*'()")  # noqa: E731 (encodeURIComponent)
        runs.append({"run": name, "source": source, "deepswe_task": target, "build_root": doc.get("build_root"),
                     "max_rounds": doc.get("max_rounds"), "detail": detail,
                     "link": f"#task={quote(target)}" + (f"&run={quote(name)}" if batch or latest is False else ""),
                     "latest": latest})  # v1.8: an official run's latest flag (an older one is linked with run=)
        total["runs"] += 1
        info = targets.setdefault(target, {"deepswe_task": target, "runs": [], "patterns": [], "exact": {}, "loose": {}})
        info["runs"].append(len(runs) - 1)
        analysis = doc.get("failure_analysis") if isinstance(doc.get("failure_analysis"), dict) else {}
        for pattern in analysis.get("patterns") or []:
            pattern = pattern if isinstance(pattern, dict) else {}
            pname = pattern.get("name") if is_str(pattern.get("name")) else None
            if len(info["runs"]) > 1 and (pname is None or pname in info["exact"]):
                continue  # a later run adds only names the target's list lacks
            if pname is not None:
                info["exact"].setdefault(pname, len(info["patterns"]))
                info["loose"].setdefault(_loose(pname), len(info["patterns"]))
            info["patterns"].append({"name": pname, "category": pattern.get("category")
                                     if is_str(pattern.get("category")) else None})
        for node in doc.get("nodes") or []:
            build = node.get("build") if isinstance(node, dict) and isinstance(node.get("build"), dict) else {}
            if not build.get("task_id"):
                continue
            score = node.get("score") if isinstance(node.get("score"), dict) else None
            probe = node.get("probe") if isinstance(node.get("probe"), dict) else {}
            match = node.get("failure_match")
            if not audited.get(source, not batch) or (node.get("audit") or {}).get("status") == "not_audited":
                audit = "not_audited"
            elif node.get("audit_eligible") is not True:
                audit = "not_eligible"
            else:
                verdict = match.get("verdict") if isinstance(match, dict) else None
                audit = verdict if verdict in AUDIT_VERDICTS else "not_audited"
            yes = partial = None
            if audit in AUDIT_VERDICTS:
                yes, partial = set(), set()
                for pattern in match.get("patterns") or []:
                    pattern = pattern if isinstance(pattern, dict) else {}
                    pname = pattern.get("name")
                    index = info["exact"].get(pname, info["loose"].get(_loose(pname))) if is_str(pname) else None
                    if index is None:
                        total["unmatched_patterns"] += 1
                    elif pattern.get("reproduced") == "yes":
                        yes.add(index)
                    elif pattern.get("reproduced") == "partial":
                        partial.add(index)
                yes, partial = sorted(yes), sorted(partial - yes)
            valid = score.get("valid") if score else None
            valid = list(valid) if isinstance(valid, list) and len(valid) == 2 and all(is_num(v) for v in valid) else None
            accepted = node.get("decision") in ACCEPTED
            details = node.get("details") if isinstance(node.get("details"), dict) else {}
            cheats = {entry.get("slot"): entry.get("cheated") for entry in details.get("teacher_audit") or []
                      if isinstance(entry, dict)} if isinstance(details.get("teacher_audit"), list) else {}
            # task_builder score()'s rates from the rollout rows: do metrics.json's still agree (else re-scored)?
            graded = {model: [r for r in probe.get("rollouts") or [] if isinstance(r, dict) and r.get("model") == model
                              and r.get("status") in ("pass", "fail")] for model in (STUDENT, TEACHER)}
            rows_rates = (sum(r["status"] == "pass" for r in graded[STUDENT]) / len(graded[STUDENT])
                          if graded[STUDENT] else None,
                          sum(r["status"] == "pass" and cheats.get(r.get("slot")) is not True for r in graded[TEACHER])
                          / len(graded[TEACHER]) if graded[TEACHER] else None)
            stored = ((_finite(score.get("student_rate")), _finite(score.get("teacher_clean_rate")))
                      if score is not None else ())
            differ = score is not None and (any((a is None) != (b is None) or (a is not None and abs(a - b) >= 1e-9)
                                          for a, b in zip(stored, rows_rates))
                                      or (valid is not None and valid != [len(graded[STUDENT]), len(graded[TEACHER])]))
            tasks.append([len(runs) - 1, node.get("id"), build["task_id"], node.get("round"), node.get("branch"),
                          node.get("kind"), node.get("title"), node.get("decision"), accepted, bool(node.get("withdrawn")),
                          node.get("in_batch") is True, _finite(score.get("similarity")) if score else None,
                          _finite(score.get("student_rate")) if score else None,
                          _finite(score.get("teacher_clean_rate")) if score else None,
                          _finite((probe.get("teacher") or {}).get("rate")) if score else None, valid, audit, yes,
                          partial, _finite(build.get("f2p")), _finite(build.get("p2p")), score is not None, differ])
            total["tasks"] += 1
            total["accepted"] += accepted
            total["audited"] += audit in AUDIT_VERDICTS
            total["rates_differ"] += differ
            for rollout in probe.get("rollouts") or []:
                if not isinstance(rollout, dict):
                    continue
                cheated = cheats.get(rollout.get("slot"))
                clean = ((not cheated) if isinstance(cheated, bool) else None) \
                    if rollout.get("model") == TEACHER and rollout.get("status") == "pass" else None
                trajectories.append([len(tasks) - 1, rollout.get("slot"), rollout.get("model"), rollout.get("status"),
                                     _finite(rollout.get("reward")), rollout.get("f2p"), rollout.get("p2p"),
                                     _finite(rollout.get("tool_calls")), _finite(rollout.get("patch_chars")),
                                     bool(rollout.get("agent_error")), bool(rollout.get("verifier_error")), clean])
                total["trajectories"] += 1
    return {"runs": runs, "targets": [{key: info[key] for key in ("deepswe_task", "runs", "patterns")}
                                      for _, info in sorted(targets.items())],
            "tasks": tasks, "trajectories": trajectories, "totals": totals}


def check_explorer_rows(r, rows, columns, where):
    """The shape of explorer.json task or trajectory rows (one list per row, a value per column, each of its column's
    type), independent of the documents they come from."""
    tasks = columns is EXPLORER_TASK_COLUMNS
    checks = {"run": is_int, "node": lambda v: is_str(v) and NODE_ID.fullmatch(v), "task_id": lambda v: is_str(v) and v,
              "round": is_int, "branch": opt(is_int), "kind": lambda v: v in ("mask", "evolve"), "title": opt(is_str),
              "decision": lambda v: v in DECISIONS, "audit": lambda v: v in EXPLORER_AUDITS,
              "reproduced_yes": opt(lambda v: isinstance(v, list) and all(is_int(i) and i >= 0 for i in v)),
              "valid": opt(lambda v: isinstance(v, list) and len(v) == 2 and all(is_num(i) for i in v)),
              "scored": lambda v: isinstance(v, bool), "rates_differ": lambda v: isinstance(v, bool)} if tasks else {
        "task": lambda v: is_int(v) and v >= 0, "slot": is_str, "model": is_str,
        "status": lambda v: v in ROLLOUT_STATUS, "f2p": opt(fraction_ok), "p2p": opt(fraction_ok),
        "clean": opt(lambda v: isinstance(v, bool))}
    for key in (("accepted", "withdrawn", "in_batch") if tasks else ("agent_error", "verifier_error")):
        checks[key] = lambda v: isinstance(v, bool)
    for key in (("similarity", "student_rate", "teacher_clean_rate", "teacher_rate", "f2p", "p2p") if tasks
                else ("reward", "tool_calls", "patch_chars")):
        checks[key] = opt(is_num)
    if tasks:
        checks["reproduced_partial"] = checks["reproduced_yes"]
    bad = 0
    for position, row in enumerate(rows if isinstance(rows, list) else []):
        if not (isinstance(row, list) and len(row) == len(columns)):
            r.add(f"{where}[{position}]", f"expected a list of {len(columns)} values")
            bad += 1
            continue
        for key, value in zip(columns, row):
            if key in checks and not checks[key](value) and bad < 20:
                bad += 1
                r.add(f"{where}[{position}].{key}", f"unexpected value {json.dumps(value)[:80]}")
    for position, row in enumerate(rows if tasks and isinstance(rows, list) else []):
        if isinstance(row, list) and len(row) == len(columns):
            value = dict(zip(columns, row))
            if value["audit"] in AUDIT_VERDICTS:
                r.need(isinstance(value["reproduced_yes"], list) and isinstance(value["reproduced_partial"], list)
                       and not set(value["reproduced_yes"]) & set(value["reproduced_partial"]), f"{where}[{position}]",
                       "an audited task lists its reproduced patterns, none both yes and partial")
            else:
                r.need(value["reproduced_yes"] is None and value["reproduced_partial"] is None, f"{where}[{position}]",
                       "reproduced patterns without a counted audit")
            r.need(value["accepted"] is (value["decision"] in ACCEPTED), f"{where}[{position}]",
                   "accepted must be decision accepted / accepted_evolve")


def check_explorer(r, ex, index, task_docs, batch_docs, size=None, official_docs=None):
    """data/explorer.json (CONTRACT "v1.7") against index.json and the task and run files: the shape, then every run,
    target, task row, trajectory row and total recomputed from those documents (explorer_expected)."""
    w = "explorer.json"
    if not r.keys(ex, EXPLORER_KEYS, w):
        return
    r.need(ex.get("schema") == 1, w, "schema must be 1")
    info, stamp, generated = index.get("explorer"), ex.get("generated_at"), index.get("generated_at")
    if r.need(is_num(stamp), w, "generated_at must be epoch seconds") and is_num(generated):
        r.need(stamp <= generated, w, f"generated_at {stamp} is later than index.json's {generated}: not one cycle")
    if r.keys(info, ["path", "generated_at", "bytes", "tasks", "trajectories"], "index.json explorer"):
        r.need(info.get("path") == "explorer.json", "index.json explorer", "path must be explorer.json")
        r.need(info.get("generated_at") == stamp, "index.json explorer",
               f"generated_at {info.get('generated_at')} != the file's {stamp}: not one cycle")
        r.need(info.get("tasks") == len(ex.get("tasks") or []) and info.get("trajectories") == len(
            ex.get("trajectories") or []), "index.json explorer", "tasks / trajectories != the file's row counts")
        if size is not None:
            r.need(info.get("bytes") == size, "index.json explorer", f"bytes {info.get('bytes')} != the file's {size}")
    r.need(ex.get("task_columns") == EXPLORER_TASK_COLUMNS, w, f"task_columns must be {EXPLORER_TASK_COLUMNS}")
    r.need(ex.get("trajectory_columns") == EXPLORER_TRAJECTORY_COLUMNS, w,
           f"trajectory_columns must be {EXPLORER_TRAJECTORY_COLUMNS}")
    sources = ex.get("sources") if isinstance(ex.get("sources"), list) else []
    r.need(sources[:1] == [OFFICIAL], w, f"sources must start with the official runs' {OFFICIAL}")
    for batch in index.get("batches") or []:  # every batch index.json lists, never audited
        if isinstance(batch, dict):
            entry = {"source": batch.get("source"), "name": batch.get("name"), "label": batch.get("label"),
                     "audited": False}
            r.need(entry in sources, w, f"sources must list batch {batch.get('source')} as {entry}")
    r.need(len({s.get("source") for s in sources if isinstance(s, dict)}) == len(sources)
           and all(isinstance(s, dict) and (s.get("audited") is True) == (s.get("source") == OFFICIAL["source"])
                   for s in sources), w, "sources: one entry each; only the official runs are audited")
    check_explorer_rows(r, ex.get("tasks"), EXPLORER_TASK_COLUMNS, f"{w} tasks")
    check_explorer_rows(r, ex.get("trajectories"), EXPLORER_TRAJECTORY_COLUMNS, f"{w} trajectories")
    expected = explorer_expected(explorer_documents(index, task_docs, batch_docs, official_docs), sources)
    if not any(isinstance(run, dict) and "latest" in run for run in ex.get("runs") or []):  # a v1.7 collector's file
        for run in expected["runs"]:
            run.pop("latest", None)
    for key in ("runs", "targets"):
        got, want = ex.get(key), expected[key]
        if not r.need(got == want, f"{w} {key}", f"{len(got) if isinstance(got, list) else got} entries differ from "
                                                  f"the task and run files' {len(want)}"):
            for position, (a, b) in enumerate(zip(got if isinstance(got, list) else [], want)):
                if a != b:
                    r.add(f"{w} {key}[{position}]", f"{json.dumps(a)[:160]} != expected {json.dumps(b)[:160]}")
                    break
    for key, columns in (("tasks", EXPLORER_TASK_COLUMNS), ("trajectories", EXPLORER_TRAJECTORY_COLUMNS)):
        got, want = ex.get(key) if isinstance(ex.get(key), list) else [], expected[key]
        r.need(len(got) == len(want), f"{w} {key}", f"{len(got)} rows, the task and run files hold {len(want)}")
        shown = 0
        for position, (a, b) in enumerate(zip(got, want)):
            if a != b and shown < 10:
                shown += 1
                if isinstance(a, list) and len(a) == len(b):
                    cells = [f"{c}: {json.dumps(x)[:60]} != {json.dumps(y)[:60]}" for c, x, y in zip(columns, a, b)
                             if x != y]
                else:
                    cells = [f"{json.dumps(a)[:120]} != {json.dumps(b)[:120]}"]
                r.add(f"{w} {key}[{position}]", "differs from its node: " + "; ".join(cells))
    r.need(ex.get("totals") == expected["totals"], f"{w} totals",
           f"{json.dumps(ex.get('totals'))[:300]} != recounted {json.dumps(expected['totals'])[:300]}")
    r.checked["explorer tasks"] = len(ex.get("tasks") or [])
    r.checked["explorer trajectories"] = len(ex.get("trajectories") or [])


def check_explorer_sources(r, ex):
    """--sources, v1.7: every task row's candidate dir (<build_root>/<node>) is a directory, and every trajectory of a
    scored task has its transcript and patch (<build_root>/round-<k>/<node>/rollouts/<slot>.md and .patch)."""
    runs = ex.get("runs") if isinstance(ex.get("runs"), list) else []
    slots = {}
    for row in ex.get("trajectories") or []:
        if isinstance(row, list) and len(row) == len(EXPLORER_TRAJECTORY_COLUMNS):
            slots.setdefault(row[0], []).append(row[1])
    dirs = files = 0
    for position, row in enumerate(ex.get("tasks") or []):
        if not (isinstance(row, list) and len(row) == len(EXPLORER_TASK_COLUMNS)):
            continue  # reported by check_explorer_rows()
        value = dict(zip(EXPLORER_TASK_COLUMNS, row))
        run = runs[value["run"]] if is_int(value["run"]) and 0 <= value["run"] < len(runs) else {}
        root = run.get("build_root") if isinstance(run, dict) else None
        if not is_str(root):
            r.add(f"sources explorer tasks[{position}]", "its run has no build_root")
            continue
        dirs += 1
        r.need((Path(root) / str(value["node"])).is_dir(), f"sources explorer tasks[{position}]",
               f"no candidate dir {root}/{value['node']}")
        if value["scored"] is True and slots.get(position):
            rollouts = Path(root) / f"round-{value['round']}" / str(value["node"]) / "rollouts"
            try:
                listing = set(os.listdir(rollouts))
            except OSError:
                listing = set()
            for slot in slots[position]:
                files += 1
                r.need(f"{slot}.md" in listing and f"{slot}.patch" in listing, f"sources explorer tasks[{position}]",
                       f"no transcript / patch {rollouts}/{slot}.md|.patch")
    r.checked["explorer cand dirs vs disk"] = dirs
    r.checked["explorer transcripts vs disk"] = files


# --------------------------------------------------------------------------------------------- cross-checks

def cross_check(r, index, task_docs, batch_docs=None, official_docs=None):
    """Compare the outputs with the pipeline's own files (read-only); a batch run's document (v1.6) and an older
    official run's (v1.8) against its own tree the same way, then the batches against their outputs/."""
    for task_id, doc in list(task_docs.items()) + list((batch_docs or {}).items()) + list((official_docs or {}).items()):
        if not isinstance(doc, dict) or not is_str(doc.get("build_root")):
            continue
        root, w = Path(doc["build_root"]), f"sources {doc.get('run')}"
        nodes = {node["id"]: node for node in doc.get("nodes") or [] if isinstance(node, dict) and "id" in node}
        dirs = {name for name in os.listdir(root) if name.startswith("cand-") and (root / name).is_dir()
                and NODE_ID.fullmatch(name)}
        r.need(set(nodes) == dirs, w, f"node set != cand-* dirs (only nodes: {sorted(set(nodes) - dirs)}, "
                                      f"only dirs: {sorted(dirs - set(nodes))})")
        # task dirs main moved to runs/<run>/withdrawn/ after accepting them (v1.4; a withdrawn task is never accepted)
        withdrawn_dir = root.parent.parent.parent / "withdrawn"
        moved = {entry for entry in (os.listdir(withdrawn_dir) if withdrawn_dir.is_dir() else [])
                 if (withdrawn_dir / entry).is_dir()}
        accepted_invalid = 0
        for name in sorted(os.listdir(root)):
            match = re.fullmatch(r"round-(\d+)", name)
            if not match or not (root / name).is_dir():
                continue
            k, round_dir = int(match.group(1)), root / name
            seeds = read_json_quiet(round_dir / "seeds.json")
            for node_id, reason in ((seeds or {}).get("invalid") or {}).items():
                node = nodes.get(node_id) or {}
                one_line = " ".join(str(reason).split())  # decision_reason is one line (CONTRACT "Decision rules")
                metrics = read_json_quiet(round_dir / node_id / "metrics.json")
                metrics = metrics if isinstance(metrics, dict) else {}
                task_id = (node.get("build") or {}).get("task_id")
                withdrawn = metrics.get("withdrawn") not in (None, False) or (is_str(task_id) and task_id in moved)
                if metrics.get("accept") is True and not withdrawn:
                    # v1.5: task_builder accepted it on its score; seeds.json only kept it from seeding evolution
                    accepted_invalid += 1
                    r.need(node.get("decision") == "accepted" and node.get("seed_invalid", one_line) == one_line,
                           f"{w} {name}", f"seeds.json invalid {node_id}, accepted by its metrics.json: expected "
                                          f"accepted with its reason as seed_invalid, got {node.get('decision')} "
                                          f"(seed_invalid {node.get('seed_invalid')!r})")
                    continue
                r.need(node.get("decision") == "discard" and node.get("decision_reason") == one_line, f"{w} {name}",
                       f"seeds.json invalid {node_id}: expected discard with its reason, got {node.get('decision')}")
            for node_id in (seeds or {}).get("valid") or []:
                node = nodes.get(node_id) or {}
                r.need(node.get("decision") not in ("discard", "abandoned", None), f"{w} {name}",
                       f"seeds.json valid {node_id} shows {node.get('decision')}")
            for entry in os.listdir(round_dir):
                metrics = read_json_quiet(round_dir / entry / "metrics.json") if entry.startswith("cand-") else None
                if isinstance(metrics, dict) and entry in nodes:
                    score = nodes[entry].get("score") or {}
                    r.need(score.get("accept") == metrics.get("accept"), f"{w} {name}/{entry}",
                           "score.accept differs from metrics.json")
            probes = {f"{name}/{entry}" for entry in os.listdir(round_dir)
                      if re.fullmatch(r"probe-[A-Za-z0-9_-]+", entry) and (round_dir / entry).is_dir()}
            round_doc = next((item for item in doc.get("rounds") or [] if item.get("index") == k), {})
            listed = set((round_doc.get("probe") or {}).get("dirs") or [(round_doc.get("probe") or {}).get("dir")])
            listed |= set(round_doc.get("abandoned_probes") or [])
            r.need(probes <= listed, f"{w} {name}", f"probe dirs not accounted for: {sorted(probes - listed)}")
        accepted_dir = root.parent.parent.parent / "accepted"
        on_disk = {entry for entry in (os.listdir(accepted_dir) if accepted_dir.is_dir() else [])
                   if (accepted_dir / entry).is_dir() and not entry.endswith((".workspace", ".partial"))}
        shown = {node["build"]["task_id"] for node in nodes.values()
                 if node.get("decision") in ("accepted", "accepted_evolve")
                 and (node.get("build") or {}).get("task_id")}
        r.need(on_disk == shown, f"{w} accepted/", f"accepted dir {sorted(on_disk)} != accepted nodes {sorted(shown)}")
        prompts = 0
        for node_id, node in sorted(nodes.items()):  # the focus pattern evolve() actually rendered into the prompt
            if node.get("kind") != "evolve" or "focus_pattern" not in node:
                continue
            match = FOCUS_LINE.search(read_head_quiet(root / node_id / "evolve/.agent/prompt.md") or "")
            if match:
                prompts += 1
                r.need(node["focus_pattern"] == match.group(1), f"{w} {node_id}",
                       f"focus_pattern {node['focus_pattern']!r} != its evolve prompt's {match.group(1)!r}")
        audit_dir, audited = root.parent.parent.parent / "audit" / "failure_match", 0
        for node_id, node in sorted(nodes.items()):  # an attached audit carries the verdict of its task's file
            if not isinstance(node.get("failure_match"), dict):
                continue
            audited += 1
            data = read_json_quiet(audit_dir / f"{(node.get('build') or {}).get('task_id')}.json")
            verdict = data.get("verdict") if isinstance(data, dict) else None
            r.need(is_str(verdict) and verdict.strip().lower() == node["failure_match"].get("verdict"), f"{w} {node_id}",
                   f"failure_match.verdict {node['failure_match'].get('verdict')!r} != its audit file's {verdict!r}")
        # v1.4: withdrawn exactly when metrics.json says so or the task dir sits in runs/<run>/withdrawn/ (`moved`,
        # above); an eligible task has its scored dir (metrics.json and rollouts/)
        withdrawals = 0
        for node_id, node in sorted(nodes.items()):
            scored = root / f"round-{node.get('round')}" / node_id
            if "withdrawn" in node:
                metrics = read_json_quiet(scored / "metrics.json")
                mark = metrics.get("withdrawn") if isinstance(metrics, dict) else None
                task_id = (node.get("build") or {}).get("task_id")
                want = mark not in (None, False) or (is_str(task_id) and task_id in moved)
                withdrawals += want
                r.need(bool(node["withdrawn"]) == want, f"{w} {node_id}",
                       f"withdrawn {json.dumps(node['withdrawn'])[:80]}, but its sources say "
                       f"{'withdrawn' if want else 'not withdrawn'}")
            if node.get("audit_eligible") is True:
                r.need((scored / "metrics.json").is_file() and (scored / "rollouts").is_dir(), f"{w} {node_id}",
                       f"eligible without {scored.relative_to(root)}/metrics.json and rollouts/")
        r.checked[f"{doc.get('run')} cross-checked nodes"] = len(nodes)
        r.checked[f"{doc.get('run')} focus patterns vs prompts"] = prompts
        r.checked[f"{doc.get('run')} audits vs files"] = audited
        r.checked[f"{doc.get('run')} withdrawn"] = withdrawals
        if accepted_invalid:  # v1.5: accepted although seeds.json lists them invalid (as seeds)
            r.checked[f"{doc.get('run')} accepted invalid seeds"] = accepted_invalid
    check_batch_sources(r, index, batch_docs or {})
    check_accuracy_sources(r, index)


def check_official_sources(r, index, task_docs, official_docs):
    """v1.8, --sources: synth/runs/official* (found from an official document's build root): every directory with a
    launch.jsonl or a build is exactly one official runs[] entry (a queued run with only analysis/ is none), and the
    latest run of each task is its last submission (launch.jsonl, the last line naming the task, else the last line)."""
    roots = [doc.get("build_root") for doc in list(task_docs.values()) + list(official_docs.values())
             if isinstance(doc, dict) and is_str(doc.get("build_root"))]
    if not roots:
        return
    runs_dir = Path(roots[0]).parents[3]  # <runs>/<run>/build/<task>/<repo>
    official = [run for run in index.get("runs") or [] if isinstance(run, dict) and "source" not in run]
    expected, bare = set(), 0
    for entry in sorted(os.listdir(runs_dir)):
        path = runs_dir / entry
        if not entry.startswith("official") or not SAFE_RUN.fullmatch(entry) or not path.is_dir():
            continue
        built = path / "build"
        has_build = built.is_dir() and any((built / task).is_dir() and any(
            (built / task / repo).is_dir() for repo in os.listdir(built / task)) for task in os.listdir(built))
        if (path / "launch.jsonl").is_file() or has_build:
            expected.add(entry)
        else:
            bare += 1
    shown = {run.get("run") for run in official}
    r.need(shown == expected, "sources official runs", f"runs[] official {sorted(shown - expected)} without a launch or "
                                                       f"build; run dirs not shown {sorted(expected - shown)}")
    if any("latest" in run for run in official):
        submitted = {}
        for run in official:
            launches = read_jsonl_quiet(runs_dir / str(run.get("run")) / "launch.jsonl")
            mine = [row for row in launches if row.get("task") == run.get("task_id")] or launches
            submitted[(run.get("run"), run.get("task_id"))] = (iso_epoch(mine[-1].get("submitted_at")) or 0) if mine else 0
        for run in official:
            if run.get("latest") is True:
                others = [submitted[(other.get("run"), other.get("task_id"))] for other in official
                          if other.get("task_id") == run.get("task_id")]
                r.need(submitted[(run.get("run"), run.get("task_id"))] >= max(others), "sources official runs",
                       f"{run.get('run')} is latest for {run.get('task_id')}, but another official run of it was "
                       "submitted later (launch.jsonl)")
    r.checked["sources official run dirs"] = len(expected)
    r.checked["sources official run dirs without launch or build"] = bare


def check_queue_sources(r, index, now=None):
    """v1.8, --sources: each queue against its runner's files under its root: the runs of queue.json (run, task, repo,
    batch, in order); and, for each file not modified since the cycle began (index.json generated_at; the runner
    rewrites state.json and heartbeat.json every pass), the statuses, legs, winners and candidates, budget and flags
    of state.json, the heartbeat, and the number of log lines."""
    generated = index.get("generated_at")
    for queue in index.get("queues") or []:
        if not isinstance(queue, dict) or not is_str(queue.get("root")):
            continue
        root, w = Path(queue["root"]), f"sources queue {queue.get('name')}"
        unchanged = lambda name: (stat_quiet(root / name) is not None and is_num(generated)  # noqa: E731
                                  and stat_quiet(root / name).st_mtime < generated)
        spec = read_json_quiet(root / "queue.json")
        listed = [(run.get("run"), run.get("task"), run.get("repo"), run.get("batch"))
                  for run in (spec.get("runs") if isinstance(spec, dict) and isinstance(spec.get("runs"), list) else [])
                  if isinstance(run, dict) and is_str(run.get("run")) and SAFE_RUN.fullmatch(run["run"])]
        runs = [run for run in queue.get("runs") or [] if isinstance(run, dict)]
        r.need([(run.get("run"), run.get("task"), run.get("repo"), run.get("batch")) for run in runs] == listed, w,
               "runs (run, task, repo, batch) differ from queue.json")
        compared = []
        if unchanged("state.json"):
            state = read_json_quiet(root / "state.json")
            state = state if isinstance(state, dict) else {}
            recorded = state.get("runs") if isinstance(state.get("runs"), dict) else {}
            for run in runs:
                info = recorded.get(run.get("run")) if isinstance(recorded.get(run.get("run")), dict) else {}
                want = {"status": info.get("status", "queued"), "leg": info.get("leg") or 0,
                        "winner": None if info.get("winner") is None else str(info.get("winner")),
                        "candidates": [str(job) for job in info.get("candidates") or []]}
                got = {key: run.get(key) for key in want}
                r.need(got == want, f"{w} {run.get('run')}", f"{json.dumps(got)} != state.json's {json.dumps(want)}")
            budget = state.get("budget") if isinstance(state.get("budget"), dict) else None
            want = None if budget is None else {"key": budget.get("key"), "remaining": budget.get("remaining"),
                                                "at": iso_epoch(budget.get("at"))}
            got = queue.get("budget")
            r.need(got == want or (isinstance(got, dict) and isinstance(want, dict) and got.get("key") == want["key"]
                                   and got.get("at") == want["at"] and is_num(got.get("remaining"))
                                   and is_num(want["remaining"]) and abs(got["remaining"] - want["remaining"]) < 0.006),
                   f"{w} budget", f"{json.dumps(got)} != state.json's key, remaining and at")
            for key in ("serving", "stop_launching"):
                r.need(queue.get(key) is (state.get(key) is True), w, f"{key} differs from state.json")
            compared.append("state.json")
        if unchanged("heartbeat.json"):
            beat = read_json_quiet(root / "heartbeat.json")
            beat = beat if isinstance(beat, dict) else {}
            runner = queue.get("runner") if isinstance(queue.get("runner"), dict) else {}
            r.need(runner.get("heartbeat_at") == (beat.get("at") if is_num(beat.get("at")) else None)
                   and runner.get("note") == beat.get("note") and runner.get("host") == beat.get("host"), f"{w} runner",
                   "heartbeat_at / note / host differ from heartbeat.json")
            compared.append("heartbeat.json")
        if unchanged("runner.log"):
            try:
                lines = (root / "runner.log").read_text(errors="replace").splitlines()
            except OSError:
                lines = []
            r.need(len(queue.get("log_tail") or []) == min(40, len(lines)), f"{w} log_tail",
                   f"{len(queue.get('log_tail') or [])} lines, runner.log ends with {min(40, len(lines))}")
            compared.append("runner.log")
        r.checked[f"sources queue {queue.get('name')} runs vs queue.json"] = len(listed)
        r.checked[f"sources queue {queue.get('name')} files compared"] = len(compared) + 1


def check_batch_sources(r, index, batch_docs):
    """v1.6, --sources: every task of a batch's outputs/tasks.jsonl is the in_batch accepted node of the run its
    outputs/<task_id>/provenance.json task_dir names (synth/runs/<run>/build/<target>/<repo>/<cand>/task/<task_id>);
    runs[].batch counts them per run; each run's accepted/tasks.jsonl lists exactly them; no run is listed without one."""
    runs = [run for run in index.get("runs") or [] if isinstance(run, dict) and "source" in run]
    task_dir_shape = re.compile(r".*/synth/runs/([^/]+)/build/([^/]+)/[^/]+/([^/]+)/task/([^/]+)")
    for batch in index.get("batches") or []:
        if not isinstance(batch, dict) or not is_str(batch.get("outputs")):
            continue
        source, outputs = batch.get("source"), Path(batch["outputs"])
        w = f"sources batch {source}"
        rows = read_jsonl_quiet(outputs / "tasks.jsonl")
        task_ids = list(dict.fromkeys(row.get("task_id") for row in rows if is_str(row.get("task_id"))))
        r.need(len(task_ids) == batch.get("tasks"), w,
               f"outputs/tasks.jsonl lists {len(task_ids)} tasks, batches[].tasks is {batch.get('tasks')}")
        mine = {(run.get("run"), run.get("task_id")): run for run in runs if run.get("source") == source}
        by_run, by_target = {}, {}
        for task_id in task_ids:
            provenance = read_json_quiet(outputs / task_id / "provenance.json")
            task_dir = provenance.get("task_dir") if isinstance(provenance, dict) else None
            match = task_dir_shape.fullmatch(task_dir) if is_str(task_dir) else None
            if not r.need(match and match.group(4) == task_id, w,
                          f"{task_id}: provenance.json task_dir {task_dir!r} names no task of a run"):
                continue
            name, target, cand = match.group(1), match.group(2), match.group(3)
            by_run.setdefault(name, []).append(task_id)
            by_target.setdefault((name, target), []).append(task_id)
            run = mine.get((name, target))
            if not r.need(run is not None, w, f"{task_id}: built by run {name} (target {target}), which index.json "
                                              "does not list"):
                continue
            nodes = (batch_docs.get(run.get("detail")) or {}).get("nodes") or []
            node = next((n for n in nodes if isinstance(n, dict) and (n.get("build") or {}).get("task_id") == task_id),
                        None)
            r.need(node is not None and node.get("id") == cand and node.get("in_batch") is True
                   and node.get("decision") in ACCEPTED, w,
                   f"{task_id}: expected {name} {cand} accepted and in_batch, got "
                   f"{None if node is None else (node.get('id'), node.get('decision'), node.get('in_batch'))}")
        for (name, target), run in sorted(mine.items()):
            want = sorted(by_target.get((name, target), []))
            r.need(run.get("batch") == len(want), w,
                   f"run {name}: runs[].batch {run.get('batch')} != its {len(want)} tasks in outputs/tasks.jsonl")
            accepted = sorted(row.get("task_id") for row in read_jsonl_quiet(Path(str(run.get("root"))) /
                                                                          "accepted/tasks.jsonl")
                              if is_str(row.get("task_id")))
            r.need(accepted == want, w, f"run {name}: accepted/tasks.jsonl {accepted} != its outputs tasks {want}")
            r.need(bool(by_run.get(name)), w, f"run {name} is listed but built no task of the batch")
        r.checked[f"{w} tasks vs provenance"] = len(task_ids)
        r.checked[f"{w} runs vs outputs"] = len(mine)


def read_jsonl_quiet(path):
    """The JSON object rows of a .jsonl file ([] when it cannot be read)."""
    rows = []
    try:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        pass
    return rows


def check_accuracy_sources(r, index):
    """Every task's kimi / super_vl card against three_way_per_task.csv, sc14 against the targets."""
    three = read_csv(THREE_WAY)
    per_task = read_csv(TARGETS / "per_task.csv")
    records = read_json_quiet(TARGETS / "index.json") or {}
    compared = 0
    for task in index.get("tasks") or []:
        task_id, accuracy = task.get("task_id"), task.get("accuracy") or {}
        where = f"sources accuracy {task_id}"
        row = three.get(task_id)
        if row:
            for model, column in (("kimi", "kimi_rollouts(f2p|p2p|score)"),
                                  ("super_vl", "super_rollouts(f2p|p2p|score)")):
                items = [item.split("|") for item in (row.get(column) or "").split()]
                expect = [{"f2p": None if "None" in parts[0] else parts[0],
                           "p2p": None if "None" in parts[1] else parts[1],
                           "score": {"1": 1, "0": 0}.get(parts[2])} for parts in items if len(parts) == 3]
                card = accuracy.get(model) or {}
                r.need(card.get("rollouts") == expect, where, f"{model} rollouts differ from three_way_per_task.csv")
                r.need(card.get("passed") == sum(item["score"] == 1 for item in expect), where,
                       f"{model} passed differs from the CSV")
            correct = row.get("kimi_correct")
            if correct and correct.isdigit():
                r.need((accuracy.get("kimi") or {}).get("passed") == int(correct), where, "kimi_correct differs")
            if row.get("f2p_total", "").isdigit():
                r.need(task.get("f2p_total") == int(row["f2p_total"]), where, "f2p_total differs from the CSV")
        counts = per_task.get(task_id)
        sc14 = accuracy.get("sc14") or {}
        if counts:
            passed, failed, infra = (int(counts.get(f"sc14_{key}") or 0) for key in ("pass", "fail", "infra"))
            shown = (sc14.get("passed"), sc14.get("total"), sc14.get("infra"))
            r.need(shown == (passed, passed + failed + infra, infra),
                   where, f"sc14 {sc14.get('passed')}/{sc14.get('total')} (infra {sc14.get('infra')}) differs from "
                          f"per_task.csv {passed}/{passed + failed + infra} (infra {infra})")
        mine = [rec for rec in ((records.get(task_id) or {}).get("records") or []) if rec.get("model") == "sc14"]
        ratios = []
        for rec in mine:
            match = FRACTION.fullmatch(str(rec.get("f2p")))
            if rec.get("status") in ("pass", "fail") and match and int(match.group(2)) > 0:
                ratios.append(int(match.group(1)) / int(match.group(2)))
        if ratios:
            expect = round(sum(ratios) / len(ratios), 4)
            r.need(sc14.get("mean_f2p_ratio") == expect, where,
                   f"sc14 mean_f2p_ratio {sc14.get('mean_f2p_ratio')} != {expect} from the records")
        compared += 1
    r.checked["accuracy cross-checked tasks"] = compared


def read_json_quiet(path):
    try:
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


def read_head_quiet(path, size=16384):
    try:
        with open(path, "rb") as stream:
            return stream.read(size).decode("utf-8", errors="replace")
    except OSError:
        return None


def stat_quiet(path):
    """path.stat(), or None when the listed entry is gone by now: the collector replaces files by renaming its temp
    files over them, so an entry of a listing may vanish before it is looked at (absent, not a crash)."""
    try:
        return path.stat()
    except FileNotFoundError:
        return None


def collector_temp(name):
    """A temp file of an atomic write in progress: ".<name>.<pid>.<thread>.tmp" (collect.py / server.py
    write_json_atomic) or ".<name>.<pid>.tmp" (collect.py trim_jsonl). Not a data file; never served."""
    return name.startswith(".") and name.endswith(".tmp")


# ------------------------------------------------------------------------------------------------- the whole

def validate(data_dir=D / "data", state_dir=D / "state", sources=False):
    """Problems found in data_dir (and state_dir's permissions) -> (problems, counts of what was checked)."""
    r, data_dir = Report(), Path(data_dir)
    docs = {}
    for name in ("index.json", "fleet.json", "api.json"):
        try:
            docs[name] = strict_json(data_dir / name)
        except (OSError, ValueError) as error:
            r.add(name, f"unreadable: {error}")
    task_docs = {}
    for path in sorted((data_dir / "task").glob("*.json")) if (data_dir / "task").is_dir() else []:
        try:
            task_docs[path.stem] = strict_json(path)
        except (OSError, ValueError) as error:
            r.add(f"task/{path.name}", f"unreadable: {error}")
    index = docs.get("index.json") if isinstance(docs.get("index.json"), dict) else {}
    generated_at = index.get("generated_at")
    if index:
        check_index(r, index)
    tasks = {task.get("task_id"): task for task in index.get("tasks") or [] if isinstance(task, dict)}
    runs = {run.get("run"): run for run in index.get("runs") or [] if isinstance(run, dict)}
    with_run = {task_id for task_id, task in tasks.items() if task.get("run")}
    r.need(set(task_docs) == with_run, "task/",
           f"task files {sorted(task_docs)} != tasks with a run {sorted(with_run)}")
    for task_id, doc in task_docs.items():
        task = tasks.get(task_id)
        check_task(r, doc, task_id, task, runs.get((task or {}).get("run")), generated_at)
    # v1.6: a batch run's document at runs[].detail, for exactly the batch runs
    batch_docs = {}
    run_dir = data_dir / "run"
    on_disk = sorted(str(path.relative_to(data_dir)) for path in run_dir.rglob("*.json")
                     if not collector_temp(path.name)) if run_dir.is_dir() else []
    batch_entries = [run for run in index.get("runs") or [] if isinstance(run, dict) and "source" in run]
    older_entries = [run for run in index.get("runs") or [] if isinstance(run, dict) and "source" not in run
                     and run.get("latest") is False]  # v1.8: official runs that are not their task's latest
    details = sorted(str(run.get("detail")) for run in batch_entries + older_entries)
    r.need(on_disk == details, "run/", f"run files {on_disk} != index runs[].detail {details} (batch and older "
                                       "official runs)")
    official_docs = {}
    for run in older_entries:
        if not is_str(run.get("detail")) or not (data_dir / run["detail"]).is_file():
            continue
        try:
            doc = strict_json(data_dir / run["detail"])
        except (OSError, ValueError) as error:
            r.add(run["detail"], f"unreadable: {error}")
            continue
        official_docs[run["detail"]] = doc
        check_task(r, doc, run.get("task_id"), tasks.get(run.get("task_id")), run, generated_at, older=True)
    for run in batch_entries:
        if not is_str(run.get("detail")) or not (data_dir / run["detail"]).is_file():
            continue
        try:
            doc = strict_json(data_dir / run["detail"])
        except (OSError, ValueError) as error:
            r.add(run["detail"], f"unreadable: {error}")
            continue
        batch_docs[run["detail"]] = doc
        check_task(r, doc, run.get("task_id"), tasks.get(run.get("task_id")), run, generated_at, batch=True)
    # v1.7: the Explorer's tables, when index.json names them (null: not built this cycle, its errors say why)
    explorer = None
    if isinstance(index.get("explorer"), dict):
        try:
            raw = (data_dir / "explorer.json").read_bytes()
            explorer = json.loads(raw, parse_constant=refuse_constant)
        except (OSError, ValueError) as error:
            r.add("explorer.json", f"unreadable: {error}")
        else:
            check_explorer(r, explorer, index, task_docs, batch_docs, size=len(raw), official_docs=official_docs)
    elif index and "explorer" in index:
        said = any(str(error).startswith("explorer:") for error in index.get("errors") or [])
        r.need(index["explorer"] is None and said, "index.json explorer",
               "must be the explorer.json entry, or null with an 'explorer: ...' error")
    broken = {str(error).split(":", 1)[0].split()[0] for error in index.get("errors") or [] if str(error).strip()}
    if "fleet.json" in docs:
        check_fleet(r, docs["fleet.json"], generated_at, degraded=bool(broken & {"fleet", "usage"}))
    if "api.json" in docs:
        check_api(r, docs["api.json"], generated_at, set(runs), degraded=bool(broken & {"api", "usage"}))
    for name, max_age in (("fleet.jsonl", 7 * DAY + DAY), ("hub.jsonl", 30 * DAY + DAY)):
        path = data_dir / "history" / name
        if path.exists():  # absent until the first fleet cycle / hub sample
            check_jsonl(r, path, max_age, generated_at)
    status = read_json_quiet(data_dir / "server_status.json")
    if status is not None:  # written by server.py: must never carry the token, the port or a URL
        leaks = [key for path, key in strings(status) if path.endswith(".<key>")
                 and re.fullmatch(r"(?i)token|url|local_url|port|cookie", key)]
        r.need(not leaks, "server_status.json", f"connection info keys {leaks}")
    # No credential anywhere.
    for name, doc in list(docs.items()) + [(f"task/{k}.json", v) for k, v in task_docs.items()] + \
            list(batch_docs.items()) + list(official_docs.items()) + [("server_status.json", status),
                                                                      ("explorer.json", explorer)]:
        for path, value in strings(doc):
            for pattern in SECRET_PATTERNS:
                if pattern.search(value):
                    r.add(f"{name} {path}", f"key-like text matches {pattern.pattern[:40]}")
    # Files: no stray temp files; nothing group/world-writable; state/ private (it holds the token). One stat per
    # entry: an entry renamed away since the listing is absent.
    for path in data_dir.rglob("*"):
        info = stat_quiet(path)
        if info is None:
            continue
        if path.name.endswith(".tmp") and time.time() - info.st_mtime > 600:  # younger: a write in progress
            r.add(str(path.relative_to(data_dir)), "leftover temp file")
        if collector_temp(path.name):
            continue
        if stat.S_ISREG(info.st_mode) and info.st_mode & 0o022:
            r.add(str(path.relative_to(data_dir)), "group/world-writable")
    state_dir = Path(state_dir)
    if state_dir.is_dir():
        r.need(stat.S_IMODE(state_dir.stat().st_mode) & 0o077 == 0, "state/", "must be mode 700")
        for path in state_dir.iterdir():
            info = stat_quiet(path)
            if info is not None and stat.S_ISREG(info.st_mode):
                r.need(stat.S_IMODE(info.st_mode) & 0o077 == 0, f"state/{path.name}", "must be mode 600")
    if sources and index:
        cross_check(r, index, task_docs, batch_docs, official_docs)
        check_official_sources(r, index, task_docs, official_docs)  # v1.8: every official run dir, once
        check_queue_sources(r, index)  # v1.8: the queues against their runner files
        if isinstance(explorer, dict):  # v1.7: the paths the Explorer shows exist
            check_explorer_sources(r, explorer)
    return r.problems, r.checked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(D / "data"))
    parser.add_argument("--state", default=str(D / "state"))
    parser.add_argument("--sources", action="store_true", help="also cross-check against the pipeline's files")
    parser.add_argument("--show", type=int, default=200, help="print at most this many problems")
    args = parser.parse_args(argv)
    for attempt in range(3):  # a collector cycle may land between our reads: then read everything again
        problems, checked = validate(args.data, args.state, sources=args.sources)
        if not any("not one cycle" in problem for problem in problems):
            break
        time.sleep(3)
    print("checked: " + ", ".join(f"{key} {value}" for key, value in checked.items()))
    for problem in problems[:args.show]:
        print(f"PROBLEM {problem}")
    if len(problems) > args.show:
        print(f"... {len(problems) - args.show} more")
    print(f"{len(problems)} problem(s)" if problems else "OK: the data conforms to CONTRACT.md")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
