#!/usr/bin/env python3
"""Recover probe slots that platform faults lost, and say whether they change a task's accept decision.

Acceptance needs sc14 <= 0.2 and clean kimi >= 0.6 over VALID slots, >= 3 valid slots per model and similarity >= 6.
ATP never retries a slot that a hub 4xx made fatal (422 budget_exceeded / 401 "remaining connection slots", README
"Hub incident"), nor one that a student worker left needs_review by stopping mid-call (finish_reason "abort", which
Gym's validation rejects); score() then excludes such slots as infra, and the loss can decide a task (sc14 1/4 = 0.25
instead of 1/5 or 2/5; clean kimi 1/2 with < 3 valid). This tool runs fresh rollouts of the lost model through the
pipeline's own probe path and records the outcome next to the build, without touching the build:

  plan    --run R --task T [--model M -n N]      read-only: the recorded slots (status, the lost ones and why), the lost
                                                 slots N rollouts would replace, and what each outcome does to the rates
  run     --run R --task T --model M -n N        (CPU job) scratch copy of the probed task dir -> rollouts.probe() (an ATP
          [--regrade-slots S ...] [--label L]    campaign with settings.json's semantics, the job's student router for
          [--no-retry] [--budget kimi=10,sc14=4] sc14) -> every recovered rollout graded with the task's recorded grader
                                                 (the probe's own verifier, plus a sandbox control) AND the sanitizing
                                                 grader (verifier/grader.py on a scratch copy staged like grading_repair) ->
                                                 the teacher audit of score() (judge_pass) on kimi passes -> one retry of
                                                 slots lost again (budget permitting) -> record. --regrade-slots also
                                                 regrades RECORDED slots with both graders (judged when a kimi slot passes)
  batch   --spec FILE                            (CPU job) the spec's recoveries concurrently: <= 8 rollouts per model at
                                                 a time, one rollout budget per model for the whole job
  submit  --spec FILE [--time 10:00:00] [--dry-run]   one CPU job (tools/pick_partition.py over cpu_interactive and
                                                 cpu_dataprocessing) running `batch` through run.sbatch, scrubbed env
  record  --run R --task T                       rebuild the record from saved outputs (no sandbox, no LLM), e.g. once
                                                 runs/<R>/audit/grading_repair/<T>.json exists
  export  --run R --task T [--grader sanitizing|recorded]   the 10 slot records score() would take after the
                                                 recovery (recovered rollouts renamed to the slots they replace) ->
                                                 audit/recovery/<T>/after-records-<grader>.jsonl (+ .json: replacements,
                                                 the teacher verdicts of every kimi pass) for main's re-score

Spec (batch/submit): {"label": "...", "budget": {"kimi": 10, "sc14": 4}, "items": [{"run": R, "task": T, "model": M,
"n": N, "regrade_slots": [...]}, ...]}. A rerun with the same label resumes (same probe ids, cached grades/verdicts).
Budget: rollout slots reserved per model in one job; a retry also needs (attempts ATP started in this job's probes +
retry) <= budget. Retries: a recovery slot lost again (hub 4xx, student abort, PTY, or "unfinished": the campaign
stopped before the slot ended, e.g. NODE_FAIL; Slurm then requeues the job under the SAME id and ATP refuses to recover
attempts whose owner job is still running) is re-run once in a fresh probe (seq 1); an existing retry probe resumes
without new budget. sc14 probes first wait (never start workers) for a 'ready' worker with >= RUNWAY_S left.
Lesson of 2026-10-20 (job 20573337): a node failure killed all 8 first rollouts after ~13 min; the orphaned agent
sandboxes (no TTL set, 5 h default) had to be deleted by hand.

Outputs (nothing else is written):
  runs/<R>/audit/recovery/<T>.json        the record (atomic): recorded slots (status, lost + why), recovered slots
                                          (outcome per grader, attempt dirs, judge result), before/after valid counts,
                                          sc14 rate and clean kimi under both graders, recommendation
                                          newly_in_range | still_out_of_range | unclear
  runs/<R>/audit/recovery/<T>/probe-<id>/ the recovery probe campaigns (ATP layout; probe id = sha1(task|model|n|label|seq))
  runs/<R>/audit/recovery/<T>/state.json  probes, grades, judge verdicts (what a rerun resumes from)
  repair/recovery/<R>/<T>/                scratch: task/<T> (copy of the probed task dir, the rollouts' input),
                                          new/<T> (sanitizing grader + "test_paths"), old/<T>/tests (the recorded tests),
                                          stage.json, validate.json, validate/, grade/
  repair/recovery/batches/<label>/        spec.json, launch.json, summary.json, the job log
Decision rule: the recommendation follows the SANITIZING-grader view (hengyuf: these runs' tasks are re-graded with
verifier/grader.py): recorded slots enter it with this tool's own regrade, else the grading_repair record
(runs/<R>/audit/grading_repair/<T>.json), else their recorded outcome (listed as pending_repair when their patch edits
test paths that test.patch does not reset). The recorded-grader view is reported beside it. The builds' probe data,
metrics.json, stage.json, seeds.json, cand-*/, accepted/ and summary.json are only read (probe reads through
grading_repair.probe_reads_only); nothing is accepted or withdrawn. No credential is read here: probes, judge calls and
tools/hub_budget.py read the pipeline's own credential files themselves.
Run from S/synth with the pipeline venv:  ../atp/runtime/controller/bin/python3 -B tools/recover_slots.py plan ...
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import asyncio  # noqa: E402
import contextlib  # noqa: E402
import contextvars  # noqa: E402
import datetime  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import pwd  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402

SYNTH = Path(__file__).resolve().parents[1]
for _path in (SYNTH, SYNTH / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import common  # noqa: E402
from common import JUDGE, PROMPTS, RUNS, S, SETTINGS, STUDENT, TEACHER, hub_key_name, log, read_json, write_json  # noqa: E402
import grading_repair as gr  # noqa: E402  (staging helpers, both graders, read-only probe reads, grade summaries)
import regrade  # noqa: E402  (regraded_record: a slot record as score() takes it)
import rollouts  # noqa: E402
import sandbox  # noqa: E402
import task_builder  # noqa: E402  (judge_pass is looked up here at call time, as score() does)

TOOL = "tools/recover_slots.py"
AUDIT = Path("audit/recovery")          # under runs/<run>/
SCRATCH = SYNTH / "repair/recovery"     # scratch copies, gradings, batch specs and job logs
BATCHES = SCRATCH / "batches"
VALID = ("pass", "fail")
MAX_PER_MODEL = 8                       # concurrent recovery rollouts per model (sandbox pool and student fleet are shared)
DEFAULT_BUDGET = {STUDENT: 4, TEACHER: 10}
HUB_BUDGET = S / "tools/hub_budget.py"
HUB_FLOOR = 500.0                       # the builds' watchers stop below $500 left on the active key (README)
HUB_WAIT_S = 7200                       # how long a kimi probe or judge call waits for a key with budget (manager switch)
RUNWAY_S = 5400                         # an sc14 probe starts once a "ready" worker has >= this much job time left
RUNWAY_WAIT_S = 4 * 3600
RETRIEVAL = "Answer retrieval by action"
GRADERS = ("new", "old")                # verifier/grader.py (sanitizing) and the task's recorded grader (control)
SLURM_CONF = "/cm/shared/apps/slurm/var/etc/cw-dfw-cs-001/slurm.conf"
SLURM_PATH = "/cm/shared/apps/slurm/current/bin:/usr/bin:/bin"
SLOT = re.compile(r"([A-Za-z0-9_.]+)-r(\d+)")


class RecoveryError(RuntimeError):
    pass


# ============================================================================================ pure helpers (tested)

def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def iso(epoch):
    try:
        return datetime.datetime.fromtimestamp(float(epoch)).astimezone().isoformat(timespec="seconds")
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def parse_slot(name: str) -> tuple:
    match = SLOT.fullmatch(str(name))
    if not match:
        raise RecoveryError(f"slot {name!r}: use <model>-r<repeat>, e.g. kimi-r1")
    return match.group(1), int(match.group(2))


def slot_order(name: str) -> tuple:
    model, repeat = parse_slot(name)
    return (model != STUDENT, model, repeat)


KEY_PATTERNS = (re.compile(r"Key=\S+"), re.compile(r"\(sk-[^)]*\)"), re.compile(r"\bsk-[A-Za-z0-9_.*\-]+"),
                re.compile(r"(?i)bearer\s+\S+"), re.compile(r"(?i)(api[_-]?key[\"'=:\s]+)\S+"))


def redact(text, limit: int = 240) -> str:
    """Hub error text without key identifiers (the 422 body names the key alias and its masked value)."""
    text = str(text if text is not None else "")
    for pattern in KEY_PATTERNS:
        text = pattern.sub("<redacted>", text)
    return text[:limit]


def hub_error(body) -> dict:
    """{"type", "message"} of a hub error response body (redacted, short)."""
    data = body
    if isinstance(body, (str, bytes)):
        try:
            data = json.loads(body)
        except ValueError:
            data = None
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        message = str(error.get("message") or "")
        return {"type": error.get("type") or error.get("code"), "message": redact(message.split(" Key=")[0], 160)}
    return {"type": None, "message": redact(body, 160)}


def classify_loss(atp_status: str, evidence: dict, verifier_error=None) -> dict:
    """Why a slot was lost, from its latest attempt's evidence (pure). kind: hub_422 | hub_401 | hub_<4xx> |
    student_abort | pty_closed | grading_failure | atp_<status>. retry: whether a fresh rollout can recover it (a
    grading failure is the task's, not the platform's)."""
    ev = evidence or {}

    def first(items):
        timed = [item for item in items if number(item.get("at")) is not None]
        return min(timed, key=lambda item: item["at"]) if timed else (items[0] if items else {})

    fatal = [e for e in ev.get("http_errors") or [] if e.get("status") in (400, 401, 403, 404, 422)]
    if fatal:
        hit = first(fatal)
        status = hit.get("status")
        kind = {422: "hub_422", 401: "hub_401"}.get(status, f"hub_{status}")
        detail = f"HTTP {status} {hit.get('type') or ''}: {hit.get('message') or ''}".replace("  ", " ").strip()
        return {"kind": kind, "at": iso(hit.get("at")), "detail": detail, "platform": True, "retry": True,
                "atp_status": atp_status, "fatal_responses": len(fatal)}
    aborts = list(ev.get("aborts") or []) + [e for e in ev.get("model_errors") or []
                                              if "abort" in str(e.get("error"))]
    if aborts:
        hit = first(aborts)
        return {"kind": "student_abort", "at": iso(hit.get("at")), "platform": True, "retry": True,
                "detail": 'a student call ended with finish_reason "abort" (worker stopped mid-call); Gym rejects '
                          "it (ValidationError), so the rollout ended unfinished", "atp_status": atp_status}
    pty = [i for i in ev.get("interruptions") or [] if i.get("error_type") == "SandboxPtyError"]
    if pty:
        hit = first(pty)
        return {"kind": "pty_closed", "at": iso(hit.get("at")), "platform": True, "retry": True,
                "detail": "PTY connection closed (SandboxPtyError)", "atp_status": atp_status}
    terminal = ev.get("terminal") or {}
    if verifier_error or terminal.get("verifier_error"):
        return {"kind": "grading_failure", "at": None, "platform": False, "retry": False, "atp_status": atp_status,
                "detail": "verifier produced no grade: " + redact(verifier_error or terminal.get("verifier_error"), 200)}
    errors = ev.get("model_errors") or []
    detail = "; ".join(f"{e.get('error_type')}: {redact(e.get('error'), 120)}" for e in errors[:2]) or \
        "; ".join(f"HTTP {e.get('status')} {e.get('type') or ''}" for e in (ev.get("http_errors") or [])[:2])
    validity = (ev.get("validity") or {}).get("failure")
    return {"kind": f"atp_{atp_status}", "at": iso(first(errors).get("at")) if errors else None, "platform": True,
            "retry": True, "atp_status": atp_status,
            "detail": detail or f"ATP slot {atp_status} (generation validity {validity})"}


def select_replacements(recorded: list, model: str, n: int) -> list:
    """The recorded slots of `model` that n fresh rollouts replace: its lost slots (not pass/fail) that a rollout can
    recover (no grading failure), in repeat order (pure)."""
    lost = sorted((r["slot"] for r in recorded if r["model"] == model and r["status"] not in VALID
                   and (r.get("loss") or {}).get("retry", True)), key=slot_order)
    if not isinstance(n, int) or n < 1:
        raise RecoveryError(f"n must be a positive integer (got {n!r})")
    if n > len(lost):
        raise RecoveryError(f"{model}: {n} rollout(s) requested but only {len(lost)} recoverable lost slot(s) "
                            f"{lost}: more would add slots instead of replacing lost ones")
    return lost[:n]


def evaluate(slots: list) -> dict:
    """score()'s acceptance arithmetic over [{slot, model, status pass|fail|infra|unknown, cheated, retrieval}]
    (pure). "unknown" = a completed rollout whose outcome is not known under this grader; a kimi pass with cheated None
    has no teacher audit. Unknowns give bounds (best: unknown sc14 FAIL, unknown/unaudited kimi a clean PASS; worst:
    the opposite); in_range is True/False only when both bounds agree, None otherwise or with < 3 valid per model."""
    counted = VALID + ("unknown",)
    student = [x for x in slots if x["model"] == STUDENT and x["status"] in counted]
    teacher = [x for x in slots if x["model"] == TEACHER and x["status"] in counted]
    s_pass = sum(x["status"] == "pass" for x in student)
    s_unknown = sum(x["status"] == "unknown" for x in student)
    t_passes = [x for x in teacher if x["status"] == "pass"]
    clean = sum(x.get("cheated") is False for x in t_passes)
    cheated = sum(x.get("cheated") is True for x in t_passes)
    unaudited = sorted((x["slot"] for x in t_passes if x.get("cheated") not in (True, False)), key=slot_order)
    t_unknown = sum(x["status"] == "unknown" for x in teacher)
    retrieval = sorted((x["slot"] for x in t_passes if x.get("retrieval")), key=slot_order)
    ns, nt = len(student), len(teacher)

    def rate(a, b):
        return a / b if b else None

    best = {"sc14": rate(s_pass, ns), "kimi_clean": rate(clean + len(unaudited) + t_unknown, nt)}
    worst = {"sc14": rate(s_pass + s_unknown, ns), "kimi_clean": rate(clean, nt)}
    exact = best == worst
    out = {"valid": [ns, nt], "sc14": best["sc14"] if exact else None,
           "kimi_clean": best["kimi_clean"] if exact else None, "sc14_pass": s_pass, "sc14_unknown": s_unknown,
           "kimi_clean_pass": clean, "kimi_cheated_pass": cheated, "kimi_unaudited_pass": unaudited,
           "kimi_unknown": t_unknown, "answer_retrieval": retrieval}
    if not exact:
        out["bounds"] = {"best": best, "worst": worst}
    if min(ns, nt) < 3:
        out.update(in_range=None, reason=f"inconclusive: fewer than 3 valid rollouts per model (valid [{ns}, {nt}])")
    elif retrieval:
        out.update(in_range=False, reason=f"teacher retrieved the answer by action (leaky task): {', '.join(retrieval)}")
    elif gr.in_range(worst["sc14"], worst["kimi_clean"]):
        out.update(in_range=True, reason=f"pass rates in range (student {worst['sc14']:.2f} <= 0.2, clean teacher "
                                         f"{worst['kimi_clean']:.2f} >= 0.6)")
    elif not gr.in_range(best["sc14"], best["kimi_clean"]):
        out.update(in_range=False, reason=f"pass rates out of range (student {best['sc14']:.2f} > 0.2 or clean "
                                          f"teacher {best['kimi_clean']:.2f} < 0.6)")
    else:
        out.update(in_range=None, reason=f"depends on {s_unknown + t_unknown} unknown outcome(s) and "
                                         f"{len(unaudited)} unaudited kimi pass(es)")
    return out


def outcome_table(view: list, replaces: list, model: str) -> list:
    """What k passes (kimi: clean passes) among the rollouts replacing `replaces` would give, k = 0..n (pure)."""
    rows = []
    for k in range(len(replaces) + 1):
        winners = set(replaces[:k])
        trial = [dict(x, status="pass" if x["slot"] in winners else "fail",
                      cheated=False if x["slot"] in winners else None, retrieval=False)
                 if x["slot"] in replaces else x for x in view]
        result = evaluate(trial)
        rows.append({"passes": k, "of": len(replaces), "model": model, "valid": result["valid"],
                     "sc14": result["sc14"], "kimi_clean": result["kimi_clean"], "in_range": result["in_range"]})
    return rows


def final_replacements(recovered: list) -> dict:
    """{recorded slot: recovered entry}: per replaced slot the valid recovery from the latest probe that has one, else
    the latest (lost) recovery (pure)."""
    chosen = {}
    for entry in sorted(recovered, key=lambda e: (e.get("seq", 0), slot_order(e["slot"]))):
        slot = entry.get("replaces")
        if not slot:
            continue
        if slot not in chosen or entry["status"] in VALID or chosen[slot]["status"] not in VALID:
            chosen[slot] = entry
    return chosen


RECORD_FIELDS = ("status", "reward", "f2p", "p2p", "failing_tests", "verifier_error")


def after_records(recorded: list, recovered: dict, overrides: dict) -> list:
    """The slot records score() would take after the recovery (pure): the recorded records, each replaced slot by its
    recovered rollout's record renamed to that slot (with a "recovery" field), and per-slot grade overrides
    ({slot: {status, reward, f2p, p2p, failing_tests, verifier_error, info}}, e.g. the sanitizing grader's)."""
    out = []
    for record in recorded:
        slot = record["slot"]
        if slot in recovered:
            source = recovered[slot]
            record = dict(source["record"], slot=slot,
                          recovery={"probe": source["probe"], "probe_slot": source["record"]["slot"],
                                    "attempt_dir": source["record"].get("attempt_dir"), "replaces": slot})
        if slot in overrides:
            override = overrides[slot]
            record = dict(record, **{k: override[k] for k in RECORD_FIELDS if k in override},
                          regrade=override.get("info"))
        out.append(record)
    return out


def is_retrieval(categories) -> bool:
    return RETRIEVAL in " ".join(str(c) for c in categories or [])


def view_slots(recorded: list, replacements: dict, *, grader: str, regraded=None, repair=None) -> list:
    """The task's slots under one grader after the recovery (pure).

    recorded: [{slot, model, status (score()'s), cheated, categories, surviving_test_edits}]
    replacements: {recorded slot: {status (probe), grades {"new": s, "old": s}, cheated, categories}}
    grader: "recorded" (the task's grader: recorded statuses, the probe's own grading of recovered rollouts) or "new"
    (sanitizing: recovered rollouts by their new-grader outcome; recorded slots by this tool's regrade, else the
    grading_repair record, else as recorded)
    regraded: {recorded slot: {"new", "old", "cheated", "categories"}} (this tool's regrades)
    repair: {recorded slot: status} from runs/<run>/audit/grading_repair/<task>.json
    Returns [{slot, model, status, cheated, retrieval, source}]."""
    regraded, repair = regraded or {}, repair or {}
    out = []
    for r in recorded:
        slot = r["slot"]
        cheated, categories = r.get("cheated"), r.get("categories")
        if slot in replacements:
            rep = replacements[slot]
            cheated, categories = rep.get("cheated"), rep.get("categories")
            if rep["status"] not in VALID:
                status, source = "infra", "recovered, lost again"
            elif grader == "recorded":
                status, source = rep["status"], "recovered (probe verifier, recorded grader)"
            else:
                new = (rep.get("grades") or {}).get("new")
                status = new if new in VALID else "unknown"
                source = "recovered (sanitizing grader)" if new in VALID else f"recovered, sanitizing grade {new}"
        else:
            status, source = r["status"], "recorded"
            own = regraded.get(slot) or {}
            if grader != "recorded" and r["status"] in VALID:
                if own.get("new") in VALID:
                    status, source = own["new"], "regraded here (sanitizing grader)"
                elif slot in repair and repair[slot] in VALID + ("unknown",):
                    status, source = repair[slot], "grading_repair record"
                elif own:
                    status, source = "unknown", f"regraded here, sanitizing grade {own.get('new')}"
            if status == "pass" and r["status"] != "pass":  # a pass the first scoring did not audit
                cheated, categories = own.get("cheated"), own.get("categories")
        if status != "pass":
            cheated, categories = None, None
        out.append({"slot": slot, "model": r["model"], "status": status, "cheated": cheated,
                    "retrieval": is_retrieval(categories), "source": source})
    return out


def pending_repair(view: list, recorded: list) -> list:
    """Valid recorded slots that enter the sanitizing view as recorded although their patch edits test paths that
    test.patch does not reset: the pending grading repair could still flip them (pure)."""
    edits = {r["slot"]: r.get("surviving_test_edits") or [] for r in recorded}
    return [x["slot"] for x in view if x["source"] == "recorded" and x["status"] in VALID and edits.get(x["slot"])]


def robustness(view: list, pending: list) -> dict:
    """Would the verdict survive any outcome of the pending slots? (bounds with them unknown; pure)"""
    if not pending:
        return {"pending": [], "verdict_can_change": False}
    trial = [dict(x, status="unknown", cheated=None) if x["slot"] in pending else x for x in view]
    result = evaluate(trial)
    return {"pending": pending, "verdict_can_change": result["in_range"] is None,
            "bounds": result.get("bounds"), "in_range_if_unknown": result["in_range"]}


def recommend(before: dict, after: dict) -> str:
    """newly_in_range | still_out_of_range | unclear (pure)."""
    if after.get("in_range") is True and before.get("in_range") is not True:
        return "newly_in_range"
    if after.get("in_range") is False:
        return "still_out_of_range"
    return "unclear"


def repair_statuses(record) -> tuple:
    """({slot: pass|fail|infra|unknown}, field used) from a grading_repair audit record (both known formats; pure)."""
    if not isinstance(record, dict):
        return {}, "none"
    after = record.get("statuses_after")
    if isinstance(after, dict) and after:
        statuses = {str(k): (v if v in VALID + ("infra", "unknown") else "unknown") for k, v in after.items()}
        for item in record.get("unknown_slots") or []:
            name = item if isinstance(item, str) else (item or {}).get("slot")
            if name:
                statuses[name] = "unknown"
        return statuses, "statuses_after"
    slots = record.get("slots")
    if isinstance(slots, list) and slots:
        statuses = {}
        for item in slots:
            if not isinstance(item, dict) or not item.get("slot"):
                continue
            effective, verdict = item.get("new_effective"), item.get("verdict")
            if effective in VALID + ("infra",):
                statuses[item["slot"]] = effective
            elif verdict in ("unchanged", "flip", "changed") and item.get("new") in VALID:
                statuses[item["slot"]] = item["new"]
            elif verdict in ("not_regraded", "inferred", "unaffected") and not item.get("surviving_test_files"):
                statuses[item["slot"]] = item.get("recorded") if item.get("recorded") in VALID else "unknown"
            else:
                statuses[item["slot"]] = "unknown"
        return statuses, "slots"
    return {}, "unrecognized"


class Budget:
    """Rollout budget per model for one job: slots reserved by initial probes and retries (pure bookkeeping)."""

    def __init__(self, limits: dict):
        self.limits = {str(k): int(v) for k, v in (limits or {}).items()}
        self.reserved, self.entries = {}, []

    def allow(self, model: str, n: int, *, attempts_started: int = 0) -> bool:
        limit = self.limits.get(model, 0)
        return self.reserved.get(model, 0) + n <= limit and attempts_started + n <= limit

    def reserve(self, model: str, n: int, why: str, *, attempts_started: int = 0) -> bool:
        ok = self.allow(model, n, attempts_started=attempts_started)
        self.entries.append({"model": model, "n": n, "why": why, "granted": ok, "at": now(),
                             "attempts_started": attempts_started, "reserved_before": self.reserved.get(model, 0)})
        if ok:
            self.reserved[model] = self.reserved.get(model, 0) + n
        return ok

    def summary(self) -> dict:
        return {"limits": self.limits, "reserved": self.reserved, "entries": self.entries}


class Limiter:
    """At most `cap` concurrent recovery rollouts per model (across the job's probes)."""

    def __init__(self, cap: int = MAX_PER_MODEL):
        self.cap, self.used, self._cond = cap, {}, None

    @contextlib.asynccontextmanager
    async def hold(self, model: str, n: int):
        if n > self.cap:
            raise RecoveryError(f"{n} concurrent {model} rollouts requested; the cap is {self.cap}")
        self._cond = self._cond or asyncio.Condition()
        async with self._cond:
            await self._cond.wait_for(lambda: self.used.get(model, 0) + n <= self.cap)
            self.used[model] = self.used.get(model, 0) + n
        try:
            yield
        finally:
            async with self._cond:
                self.used[model] -= n
                self._cond.notify_all()


def probe_id(task_id: str, model: str, n: int, label: str, seq: int) -> str:
    return hashlib.sha1(f"{task_id}|{model}|{n}|{label}|{seq}".encode()).hexdigest()[:8]


def parse_budget(text: str) -> dict:
    budget = dict(DEFAULT_BUDGET)
    for part in [p.strip() for p in (text or "").split(",") if p.strip()]:
        model, _, value = part.partition("=")
        if model not in SETTINGS["models"] or not value.isdigit():
            raise RecoveryError(f"--budget {part!r}: use <model>=<rollouts>, e.g. kimi=10")
        budget[model] = int(value)
    return budget


# ============================================================================================ the build (read-only)

def build_root(run: str) -> Path:
    roots = sorted(p for p in (RUNS / run / "build").glob("*/*") if p.is_dir())
    if len(roots) != 1:
        raise RecoveryError(f"runs/{run}/build holds {len(roots)} builds")
    return roots[0]


def tree_shas(directory: Path) -> dict:
    directory = Path(directory)
    return {str(p.relative_to(directory)): gr.file_sha(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def locate(run: str, task_id: str) -> dict:
    """Where a scored task lives: build root, round, scored dir, metrics, the probe whose records score() rendered,
    those records, the probed task copy and the packaged task dir (read-only)."""
    root = build_root(run)
    found = [p.parent for p in sorted(root.glob("round-*/cand-*/metrics.json"))
             if (read_json(p) or {}).get("task_id") == task_id]
    if len(found) != 1:
        raise RecoveryError(f"{task_id}: {len(found)} scored candidates in {root}")
    scored = found[0]
    metrics = read_json(scored / "metrics.json")
    with _READ_GUARD:  # scored_probe reads through grading_repair.probe_reads_only()
        probe, records = gr.scored_probe(scored.parent, task_id, scored)
    probed = probe / "tasks" / task_id
    if not (probed / "task.toml").exists():
        raise RecoveryError(f"{probed}: the probed task copy is missing")
    target = root.parent.name
    repo = (read_json(root / "summary.json") or {}).get("repo") or root.name.replace("__", "/", 1)
    language = gr.repo_language(target, repo)
    manifest = read_json(probe / "manifest.json")
    index = next(t["_ng_task_index"] for t in manifest["tasks"] if t["task_id"] == task_id)
    return {"run": run, "task_id": task_id, "root": root, "target": target, "repo": repo, "language": language,
            "round": scored.parent.name, "scored": scored, "cand": scored.name, "metrics": metrics, "probe": probe,
            "task_index": index, "records": records, "probed": probed,
            "packaged": Path(metrics["task_dir"]), "stage": read_json(root / scored.name / "stage.json") or {}}


ABORT = re.compile(rb'finish_reason\\*"\s*:\s*\\*"abort')


def loss_evidence(attempt) -> dict:
    """Read-only facts about why an attempt did not complete (hub error responses, student aborts, interruptions,
    ATP's failure files and generation-validity cache)."""
    attempt = Path(attempt)
    ev = {"attempt_dir": str(attempt), "http_errors": [], "aborts": [], "model_errors": [], "interruptions": [],
          "validity": None, "failure": None, "interrupted": None, "terminal": None}
    validity = read_json(attempt / "generation-validity-v2-r7.json")
    if isinstance(validity, dict):
        ev["validity"] = {k: validity.get(k) for k in ("failure", "native_model_errors", "budget_timeout",
                                                        "stream_errors", "delivery_gap")}
    journal = attempt / "http.jsonl"
    if journal.exists():
        with journal.open("rb") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                at = number(event.get("timestamp"))
                if event.get("scope") == "native_model_method":
                    if event.get("event") == "model_call_error":
                        ev["model_errors"].append({"at": at, "error_type": event.get("error_type"),
                                                   "error": redact(event.get("error"), 300)})
                elif event.get("event") == "response":
                    status = event.get("status")
                    if isinstance(status, int) and status >= 400:
                        ev["http_errors"].append({"at": at, "status": status, **hub_error(event.get("body"))})
                    elif ABORT.search(line):
                        ev["aborts"].append({"at": at})
    events = attempt / "capture-events.jsonl"
    if events.exists():
        for line in events.read_bytes().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("event") == "agent_interrupted":
                ev["interruptions"].append({"at": number(event.get("time") or event.get("timestamp")),
                                            "error_type": event.get("error_type")})
    for name in ("failure.json", "dispatch-failure.json"):
        value = read_json(attempt / name)
        if isinstance(value, dict):
            ev["failure"] = {"file": name, **{k: redact(value[k], 300) if isinstance(value[k], str) else value[k]
                                              for k in ("classification", "error_type", "error") if k in value}}
            break
    interrupted = read_json(attempt / "interrupted.json")
    if isinstance(interrupted, dict):
        ev["interrupted"] = {"reason": interrupted.get("reason")}
    terminal = read_json(attempt / "terminal.json")
    if isinstance(terminal, dict):
        ev["terminal"] = {k: terminal.get(k) for k in ("evaluation_completed", "verifier_error", "failure_reason",
                                                        "opencode_run_error_type", "opencode_exit_code")}
    ev["http_errors"] = ev["http_errors"][:20]
    ev["model_errors"] = ev["model_errors"][:10]
    ev["aborts"] = ev["aborts"][:10]
    return ev


_READ_GUARD = threading.RLock()


@contextlib.contextmanager
def read_only_probe():
    """grading_repair.probe_reads_only() swaps functions of the ATP campaign module for the duration; two threads
    inside it at once could leave the swap in place, so every guarded read here goes through one lock."""
    with _READ_GUARD, gr.probe_reads_only():
        yield


def slot_states(probe: Path, task_index: int, slots: list) -> dict:
    """{slot: {status, attempts, attempt_dirs}} from ATP's slot_state, through grading_repair's read-only guard."""
    import deepswe_campaign as campaign
    out = {}
    with read_only_probe():
        for slot in slots:
            model, repeat = parse_slot(slot)
            state = campaign.slot_state(probe, model, task_index, repeat)
            attempts = campaign.attempts_for(campaign.slot_directory(probe, model, task_index, repeat))
            out[slot] = {"status": state["status"], "attempts": len(attempts),
                         "attempt_dirs": [str(a) for a in attempts], "attempt_dir": state.get("attempt_dir")}
    return out


def owner_job(attempt) -> dict:
    """The Slurm job that owned an attempt (started.json job_id) and every state sacct has for it, requeues included
    (e.g. NODE_FAIL then RUNNING: the node died and Slurm restarted the job under the same id)."""
    started = read_json(Path(attempt) / "started.json") or {}
    job = str(started.get("job_id") or "")
    out = {"job_id": job or None, "states": None}
    sacct = shutil.which("sacct") or "/cm/shared/apps/slurm/current/bin/sacct"
    if job and os.path.exists(sacct):
        try:
            text = subprocess.run([sacct, "-X", "-nP", "--duplicates", "-j", job, "--format=JobIDRaw,State,NodeList"],
                                  capture_output=True, text=True, timeout=30).stdout
            out["states"] = [f"{parts[1].split()[0]} {parts[2]}" for parts in
                             (line.split("|") for line in text.splitlines()) if len(parts) >= 3 and parts[0] == job]
        except (OSError, subprocess.SubprocessError):
            pass
    return out


def recovery_errors(probe_dir: Path, model: str) -> list:
    """The errors of ATP's last recovery pass in the probe (e.g. "Cannot recover attempt while owning job may still
    be active": ATP refuses to clean up attempts whose owner job id is still running, which a requeue keeps)."""
    launches = sorted(Path(probe_dir).glob(f"models/{model}/shards/*/launch-*"), key=lambda p: p.stat().st_mtime)
    for launch in reversed(launches):
        passes = sorted(launch.glob("recovery-*.json"))
        if passes:
            errors = (read_json(passes[-1]) or {}).get("errors") or []
            return sorted({redact(e.get("error"), 160) for e in errors if isinstance(e, dict)})
    return []


def surviving_edits(patch: str, globs: list, reset: set) -> list:
    """Test paths a patch touches that the recorded grader leaves in the graded run (outside test.patch's files)."""
    return [p for p in gr.matching(gr.NEW.touched_paths(patch or ""), globs) if p not in reset]


def recorded_slots(loc: dict) -> list:
    """The task's probe slots as score() used them: status from its rollouts/<slot>.md header, teacher verdicts from
    teacher_audit.json, ATP slot state, and for lost slots the evidence of why."""
    audit = read_json(loc["scored"] / "teacher_audit.json") or {}
    globs = gr.task_globs(loc["task_id"], loc["language"], loc["run"])
    reset = set(gr.OLD.patch_paths((loc["probed"] / "tests/test.patch").read_text(errors="surrogateescape")))
    states = slot_states(loc["probe"], loc["task_index"], [r["slot"] for r in loc["records"]])
    out = []
    for record in sorted(loc["records"], key=lambda r: slot_order(r["slot"])):
        slot = record["slot"]
        md_slot, status = gr.md_header(loc["scored"] / "rollouts" / f"{slot}.md")
        status = status if md_slot == slot else record["status"]
        verdict = audit.get(slot) if isinstance(audit.get(slot), dict) else None
        state = states[slot]
        entry = {"slot": slot, "model": record["model"], "status": status, "probe_status": record["status"],
                 "atp_state": state["status"], "attempts": state["attempts"], "attempt_dirs": state["attempt_dirs"],
                 "f2p": record["f2p"], "p2p": record["p2p"], "agent_error": record["agent_error"],
                 "verifier_error": record["verifier_error"], "patch_sha256": gr.sha256(record["patch"]),
                 "patch_chars": len(record["patch"]),
                 "surviving_test_edits": surviving_edits(record["patch"], globs, reset),
                 "cheated": verdict.get("cheated") if verdict and status == "pass" else None,
                 "categories": verdict.get("categories") if verdict and status == "pass" else None,
                 "lost": status not in VALID, "loss": None}
        if entry["lost"]:
            latest = state["attempt_dirs"][-1] if state["attempt_dirs"] else None
            entry["loss"] = classify_loss(state["status"], loss_evidence(latest) if latest else {},
                                          record["verifier_error"])
        out.append(entry)
    return out


def read_repair(run: str, task_id: str, probe: Path) -> dict:
    path = RUNS / run / "audit/grading_repair" / f"{task_id}.json"
    record = read_json(path)
    if not isinstance(record, dict):
        return {"path": str(path), "present": path.exists(), "statuses": {}, "format": "none"}
    statuses, field = repair_statuses(record)
    problems = []
    if record.get("probe") and Path(record["probe"]).resolve() != Path(probe).resolve():
        problems.append(f"repair record probe {record['probe']} is not the scored probe {probe}")
    return {"path": str(path), "present": True, "statuses": statuses, "format": field, "problems": problems,
            "recommendation": record.get("recommendation"), "rates_after": record.get("rates_after"),
            "test_paths_mode": record.get("test_paths_mode"), "at": record.get("at") or record.get("generated_at"),
            "sha256": gr.file_sha(path)}


# ============================================================================================ one task

_ATP_SCRATCH = contextvars.ContextVar("recover_slots_atp_scratch", default=None)


def _scratch() -> Path:
    base = _ATP_SCRATCH.get()
    return base if base is not None else common.scratch()


@contextlib.contextmanager
def per_probe_scratch():
    """rollouts.probe() stages ATP into scratch()/atp and copies its stage scripts there on every call; concurrent
    probes in one job would race on that directory. Inside this context each probe task stages into its own
    subdirectory of the job's scratch (a ContextVar per asyncio task); nothing else in probe() changes."""
    real = rollouts.scratch
    rollouts.scratch = _scratch
    try:
        yield
    finally:
        rollouts.scratch = real


def job_end() -> float:
    end = number(os.environ.get("SLURM_JOB_END_TIME"))
    if end:
        return end
    job = os.environ.get("SLURM_JOB_ID")
    scontrol = shutil.which("scontrol") or "/cm/shared/apps/slurm/current/bin/scontrol"
    if job and os.path.exists(scontrol):
        try:
            text = subprocess.run([scontrol, "show", "job", "--oneliner", job], capture_output=True, text=True,
                                  timeout=30).stdout
            match = re.search(r"(?:^|\s)EndTime=(\S+)", text)
            if match:
                return datetime.datetime.fromisoformat(match.group(1)).timestamp()
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return 0.0


def hub_status(min_remaining: float = HUB_FLOOR) -> dict:
    """tools/hub_budget.py on the active key (it prints the key NAME and numbers only)."""
    python = "/usr/bin/python3" if os.path.exists("/usr/bin/python3") else sys.executable
    try:
        proc = subprocess.run([python, "-I", "-B", str(HUB_BUDGET), "--min-remaining", str(min_remaining)],
                              capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as error:
        return {"exit": 2, "error": f"{type(error).__name__}"}
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "{}"
    try:
        info = json.loads(line)
    except ValueError:
        info = {}
    return {"exit": proc.returncode, "key": info.get("key"), "remaining": info.get("remaining"),
            "status": info.get("status"), "error": redact(info.get("error"), 200) if info.get("error") else None,
            "at": now()}


async def hub_guard(why: str, *, floor: float = HUB_FLOOR, max_wait_s: float = HUB_WAIT_S, poll_s: float = 300) -> dict:
    """Wait until the active hub key has >= floor dollars left (the manager switches keys); unknown status 3x -> go."""
    deadline, unknown = time.time() + max_wait_s, 0
    while True:
        info = await asyncio.to_thread(hub_status, floor)
        if info["exit"] == 0:
            return info
        if info["exit"] == 2:
            unknown += 1
            if unknown >= 3:
                log(f"hub budget unknown 3x before {why}; proceeding ({info.get('error')})")
                return dict(info, proceeded_unknown=True)
        else:
            log(f"hub key {info.get('key')} below ${floor:.0f} ({info.get('remaining')}) before {why}; waiting")
        if time.time() >= deadline:
            raise RecoveryError(f"hub key {info.get('key')} has no budget (>= ${floor:.0f}) for {why} after "
                                f"{max_wait_s:.0f} s")
        await asyncio.sleep(poll_s if info["exit"] == 3 else 60)


async def wait_runway(runway_s: float = RUNWAY_S, max_wait_s: float = RUNWAY_WAIT_S, poll_s: float = 60) -> dict:
    """Before an sc14 probe: wait (never start workers) until a 'ready' student worker has >= runway_s of job time
    left, so that the rollout does not start on a worker about to stop (its in-flight calls end in finish_reason
    "abort", the loss being recovered). Gives up at max_wait_s or when the probe would no longer fit in this job;
    probe() then waits for any routable worker as usual."""
    from student.client import routable_workers
    started = time.time()
    end = job_end()
    need = SETTINGS["agent_timeout_seconds"] + 1800 + 600 + 300
    deadline = min(started + max_wait_s, end - need) if end else started + max_wait_s
    reported = 0.0
    while True:
        moment = time.time()
        workers = await asyncio.to_thread(routable_workers)
        fresh = [w for w in workers if w.get("routing_status") == "ready" and number(w.get("end_epoch"))
                 and w["end_epoch"] - moment >= runway_s]
        if fresh:
            return {"waited_s": round(moment - started), "fresh_workers": [str(w.get("job_id")) for w in fresh],
                    "runway_s": runway_s}
        if moment >= deadline:
            return {"waited_s": round(moment - started), "fresh_workers": [], "gave_up": True, "runway_s": runway_s,
                    "routable": [f"{w.get('job_id')}:{w.get('routing_status')}" for w in workers]}
        if moment - reported >= 600:
            log(f"sc14 probe: waiting for a ready worker with >= {runway_s:.0f} s left (routable now: "
                f"{', '.join(str(w.get('job_id')) + ':' + str(w.get('routing_status')) for w in workers) or 'none'})")
            reported = moment
        await asyncio.sleep(poll_s)


def grade_short(summary) -> dict:
    if not isinstance(summary, dict):
        return None
    keys = ("status", "f2p", "p2p", "reward", "error", "out_dir", "grader_sha256", "config_sha256", "at", "timed_out",
            "apply_failed")
    out = {k: summary.get(k) for k in keys if k in summary}
    out["failing_tests"] = (summary.get("failing_tests") or [])[:20]
    actions = ((summary.get("sanitize") or {}).get("actions")) or {}
    if actions:
        out["sanitize"] = {k: v[:20] for k, v in actions.items()}
    return out


class Task:
    """One task's recovery: the build facts (read-only), scratch staging, probes, grades, verdicts, the record."""

    def __init__(self, run: str, task_id: str, label: str):
        self.run, self.task_id, self.label = run, task_id, label
        self.home = RUNS / run / AUDIT / task_id
        self.record_path = RUNS / run / AUDIT / f"{task_id}.json"
        self.scratch = SCRATCH / run / task_id
        self.notes, self.problems = [], []
        self._lock = None
        self.loc = self.slots = self.stage_info = None

    # ---------------------------------------------------------------------------------------------- facts
    def load(self) -> "Task":
        self.loc = locate(self.run, self.task_id)
        self.slots = recorded_slots(self.loc)
        self.by_slot = {r["slot"]: r for r in self.loc["records"]}
        self.snapshot = rollouts.task_toml(self.loc["probed"])["environment"]["docker_image"]
        if self.snapshot != self.loc["metrics"].get("snapshot"):
            self.problems.append(f"probed task.toml image {self.snapshot} != metrics snapshot "
                                 f"{self.loc['metrics'].get('snapshot')}")
        return self

    @property
    def before(self) -> dict:
        view = [{"slot": r["slot"], "model": r["model"], "status": r["status"], "cheated": r["cheated"],
                 "retrieval": is_retrieval(r["categories"])} for r in self.slots]
        return evaluate(view)

    def plan(self, recover: dict) -> dict:
        view = view_slots(self.slots, {}, grader="recorded")
        out = {}
        for model, n in recover.items():
            replaces = select_replacements(self.slots, model, n)
            table = outcome_table(view, replaces, model)
            out[model] = {"n": n, "replaces": replaces, "outcomes": table,
                          "decisive": len({row["in_range"] for row in table}) > 1}
        return out

    # ---------------------------------------------------------------------------------------------- state
    def state(self) -> dict:
        return read_json(self.home / "state.json") or {"run": self.run, "task_id": self.task_id, "probes": [],
                                                       "grades": {}, "verdicts": {}}

    @contextlib.asynccontextmanager
    async def edit(self):
        """Read-modify-write of state.json: one coroutine at a time, and one process at a time (flock), since two
        jobs can work on one task (e.g. a requeued job and a follow-up job)."""
        self._lock = self._lock or asyncio.Lock()
        async with self._lock:
            self.home.mkdir(parents=True, exist_ok=True)
            with (self.home / ".state.lock").open("a") as handle:
                await asyncio.to_thread(fcntl.flock, handle, fcntl.LOCK_EX)
                try:
                    state = self.state()
                    yield state
                    state["updated_at"] = now()
                    write_json(self.home / "state.json", state)
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    # ---------------------------------------------------------------------------------------------- scratch
    def stage(self) -> dict:
        """repair/recovery/<run>/<task>/: task/<T> (the probed copy, the rollouts' input), new/<T> (+ the sanitizing
        grader and "test_paths", staged like grading_repair.stage_task), old/<T>/tests (the recorded tests)."""
        loc, home = self.loc, self.scratch
        probed = loc["probed"]
        shas = tree_shas(probed)
        globs = gr.task_globs(self.task_id, loc["language"], self.run)
        mode = gr.task_mode(loc["language"])
        solution = (probed / "solution/solution.patch").read_text(errors="surrogateescape")
        hits = gr.matching(gr.NEW.touched_paths(solution), globs)
        config_text = None if hits else gr.sanitized_config((probed / "tests/config.json").read_text(), globs, mode)
        packaged = tree_shas(loc["packaged"]) if loc["packaged"].exists() else {}
        info = {"source": str(probed), "source_sha256": shas, "globs": globs, "test_paths_mode": mode,
                "language": loc["language"], "grader_sha256": gr.file_sha(gr.NEW_GRADER),
                "config_sha256": gr.sha256(config_text) if config_text else None,
                "recorded_grader_sha256": shas.get("tests/grader.py"),
                "recorded_grader": gr.grader_kind(shas.get("tests/grader.py")) or "unknown",
                "solution_test_path_hits": hits, "sanitizing": not hits,
                "packaged": str(loc["packaged"]),
                "packaged_differs": sorted(k for k in set(shas) | set(packaged) if shas.get(k) != packaged.get(k))}
        keys = ("source_sha256", "grader_sha256", "config_sha256", "globs", "test_paths_mode")
        previous = read_json(home / "stage.json")
        if previous and {k: previous.get(k) for k in keys} == {k: info[k] for k in keys} and self._staged_ok(previous):
            self.stage_info = previous
            return previous
        home.mkdir(parents=True, exist_ok=True)
        tmp = home / f".stage.{uuid.uuid4().hex}.tmp"
        try:
            shutil.copytree(probed, tmp / "task" / self.task_id, symlinks=True,
                            ignore=shutil.ignore_patterns("__pycache__"))
            (tmp / "old" / self.task_id / "tests").mkdir(parents=True)
            for name in gr.TESTS_FILES + ("grader.py",):
                shutil.copy2(probed / "tests" / name, tmp / "old" / self.task_id / "tests" / name)
            if config_text:
                shutil.copytree(probed, tmp / "new" / self.task_id, symlinks=True,
                                ignore=shutil.ignore_patterns("__pycache__"))
                shutil.copyfile(gr.NEW_GRADER, tmp / "new" / self.task_id / "tests/grader.py")
                (tmp / "new" / self.task_id / "tests/config.json").write_text(config_text)
                task_builder.validate_task(tmp / "new" / self.task_id)  # Gym's DeepSWE loader accepts the copy
            task_builder.validate_task(tmp / "task" / self.task_id)
            for name in ("task", "new", "old"):
                if (home / name).exists():
                    shutil.rmtree(home / name)
                if (tmp / name).exists():
                    (tmp / name).rename(home / name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        info["at"] = now()
        write_json(home / "stage.json", info)
        self.stage_info = info
        return info

    def _staged_ok(self, stage: dict) -> bool:
        task, new, old = (self.scratch / name / self.task_id for name in ("task", "new", "old"))
        if tree_shas(task) != stage["source_sha256"]:
            return False
        if any(gr.file_sha(old / "tests" / name) != stage["source_sha256"].get(f"tests/{name}")
               for name in gr.TESTS_FILES + ("grader.py",)):
            return False
        if stage.get("sanitizing"):
            staged = tree_shas(new)
            return (staged.get("tests/grader.py") == stage["grader_sha256"]
                    and staged.get("tests/config.json") == stage["config_sha256"]
                    and all(staged.get(k) == v for k, v in stage["source_sha256"].items()
                            if k not in ("tests/grader.py", "tests/config.json")))
        return True

    @property
    def task_copy(self) -> Path:
        return self.scratch / "task" / self.task_id

    def grader_dir(self, grader: str) -> Path:
        return self.scratch / ("new" if grader == "new" else "old") / self.task_id

    # ---------------------------------------------------------------------------------------------- sandboxes
    async def validate(self, gate) -> dict:
        """The sanitizing copy grades the reference solution 1 and an empty patch 0 (grading_repair's checks)."""
        stage = self.stage_info
        if not stage.get("sanitizing"):
            return {"skipped": f"reference solution touches test paths {stage['solution_test_path_hits']}"}
        cached = read_json(self.scratch / "validate.json")
        if isinstance(cached, dict) and cached.get("all_ok") and cached.get("grader_sha256") == stage["grader_sha256"] \
                and cached.get("config_sha256") == stage["config_sha256"] \
                and cached.get("source_sha256") == stage["source_sha256"]:
            return cached
        new_dir = self.grader_dir("new")
        config = json.loads((new_dir / "tests/config.json").read_text())
        patches = {"solution": (new_dir / "solution/solution.patch").read_text(), "empty": ""}

        async def one(name):
            out_dir = self.scratch / "validate" / name
            async with gate:
                shutil.rmtree(out_dir, ignore_errors=True)
                return await sandbox.grade(self.snapshot, new_dir, patches[name], run=self.run, out_dir=out_dir)

        results = await asyncio.gather(*(one(name) for name in patches), return_exceptions=True)
        graded = {}
        for name, result in zip(patches, results):
            graded[name] = {"status": "error", "error": redact(f"{type(result).__name__}: {result}", 500)} \
                if isinstance(result, BaseException) else gr.grade_summary(result, self.scratch / "validate" / name,
                                                                           config)
        sol, emp = graded["solution"], graded["empty"]
        checks = {"solution_scores_1": sol.get("reward_json_present") is True and sol.get("reward") == 1
                  and (sol.get("sanitize") or {}).get("applied") is True
                  and any("0 path(s) of model.patch match" in line for line in (sol.get("sanitize") or {}).get("lines", [])),
                  "empty_scores_0": emp.get("reward_json_present") is True and emp.get("reward") == 0}
        if stage["test_paths_mode"] != "all":
            checks["mode_logged"] = any(f"test_paths_mode {stage['test_paths_mode']}:" in line
                                        for line in (sol.get("sanitize") or {}).get("lines", []))
        record = {"task_id": self.task_id, "snapshot": self.snapshot, "grader_sha256": stage["grader_sha256"],
                  "config_sha256": stage["config_sha256"], "source_sha256": stage["source_sha256"],
                  "results": {k: grade_short(v) for k, v in graded.items()}, "checks": checks,
                  "all_ok": all(checks.values()),
                  "sandbox_ok": any(v.get("reward_json_present") for v in graded.values()), "at": now()}
        write_json(self.scratch / "validate.json", record)
        return record

    async def grade(self, key: str, record: dict, gate, graders=GRADERS) -> dict:
        """Grade one patch with the sanitizing grader ("new") and the recorded grader ("old") on the scratch copies
        (grading_repair's regrade: sandbox.grade + grade_summary + regrade.regraded_record); cached per patch."""
        patch_sha = gr.sha256(record["patch"])
        cached = (self.state()["grades"].get(key)) or {}
        graders = [g for g in graders if g == "old" or self.stage_info.get("sanitizing")]
        todo = [g for g in graders if not (isinstance(cached.get(g), dict) and cached[g].get("patch_sha256") == patch_sha
                                           and cached[g].get("status") in VALID
                                           and cached[g].get("grader_sha256") == gr.file_sha(self.grader_dir(g) / "tests/grader.py")
                                           and cached[g].get("config_sha256") == gr.file_sha(self.grader_dir(g) / "tests/config.json"))]

        async def one(grader):
            out_dir = self.scratch / "grade" / key.replace(":", "__") / grader
            async with gate:
                shutil.rmtree(out_dir, ignore_errors=True)
                return await sandbox.grade(self.snapshot, self.grader_dir(grader), record["patch"], run=self.run,
                                           out_dir=out_dir)

        results = await asyncio.gather(*(one(g) for g in todo), return_exceptions=True)
        fresh = {}
        for grader, result in zip(todo, results):
            directory = self.grader_dir(grader)
            out_dir = self.scratch / "grade" / key.replace(":", "__") / grader
            config = json.loads((directory / "tests/config.json").read_text())
            if isinstance(result, BaseException):
                summary = {"status": "error", "error": redact(f"{type(result).__name__}: {result}", 500)}
            else:
                summary = gr.grade_summary(result, out_dir, config)
                info = {"tool": TOOL, "grader": grader, "out_dir": str(out_dir)}
                summary["record"] = {k: regrade.regraded_record(record, result, config, info).get(k)
                                     for k in ("status", "reward", "f2p", "p2p", "failing_tests", "verifier_error")}
            summary.update(at=now(), patch_sha256=patch_sha, grader=grader,
                           grader_sha256=gr.file_sha(directory / "tests/grader.py"),
                           config_sha256=gr.file_sha(directory / "tests/config.json"))
            fresh[grader] = summary
        if fresh:
            async with self.edit() as state:
                state["grades"].setdefault(key, {}).update(fresh)
        return dict(cached, **fresh)

    async def judge(self, key: str, record: dict) -> dict:
        """score()'s teacher audit of one kimi pass (task_builder.judge_pass with the task's instruction and rubric);
        cached per rendered transcript."""
        transcript = gr.sha256(rollouts.render(record))
        cached = self.state()["verdicts"].get(key)
        if isinstance(cached, dict) and cached.get("transcript_sha256") == transcript \
                and isinstance(cached.get("cheated"), bool):
            return cached
        texts = self.loc["packaged"] if (self.loc["packaged"] / "instruction.md").exists() else self.loc["probed"]
        instruction, rubric = (texts / "instruction.md").read_text(), (texts / "rubric.md").read_text()
        guard = await hub_guard(f"judging {key}")
        error = None
        for attempt in range(3):
            try:
                verdict = await task_builder.judge_pass(record, instruction, rubric)
                break
            except Exception as exc:  # noqa: BLE001 - recorded, retried
                error = redact(f"{type(exc).__name__}: {exc}", 400)
                log(f"judge {key} attempt {attempt} failed: {error}")
                await asyncio.sleep(30)
        else:
            verdict = {"cheated": None, "error": error}
        entry = dict(verdict, transcript_sha256=transcript, judge=JUDGE, at=now(), hub_key=guard.get("key"),
                     prompt_sha256=gr.file_sha(PROMPTS / "judge_pass.md"), instruction_from=str(texts))
        async with self.edit() as state:
            state["verdicts"][key] = entry
        return entry

    # ---------------------------------------------------------------------------------------------- probes
    def probe_records(self, probe_dir: Path) -> dict:
        if not (probe_dir / "manifest.json").exists():
            return {}
        with read_only_probe():
            return {r["slot"]: r for r in rollouts.probe_records(probe_dir).get(self.task_id, [])}

    def probe_slots(self, probe_dir: Path, replaces: list, probe_error: str = None) -> list:
        records = self.probe_records(probe_dir)
        if not records:
            return []
        manifest = read_json(probe_dir / "manifest.json")
        index = next(t["_ng_task_index"] for t in manifest["tasks"] if t["task_id"] == self.task_id)
        states = slot_states(probe_dir, index, list(records))
        out = []
        for slot, record in sorted(records.items(), key=lambda item: slot_order(item[0])):
            model, repeat = parse_slot(slot)
            state = states[slot]
            entry = {"slot": slot, "replaces": replaces[repeat] if repeat < len(replaces) else None,
                     "status": record["status"], "atp_state": state["status"], "attempts": state["attempts"],
                     "attempt_dirs": state["attempt_dirs"], "f2p": record["f2p"], "p2p": record["p2p"],
                     "agent_error": record["agent_error"], "verifier_error": record["verifier_error"],
                     "patch_sha256": gr.sha256(record["patch"]), "patch_chars": len(record["patch"]), "loss": None}
            latest = state["attempt_dirs"][-1] if state["attempt_dirs"] else None
            if record["status"] not in VALID and state["status"] not in rollouts.TERMINAL_STATES:
                # The campaign stopped before this slot ended (node failure, requeue, job time): not a recorded loss
                # of the model's rollout but an unfinished one; ATP itself retries it once its owner job has ended.
                entry["loss"] = {"kind": "unfinished", "retry": True, "platform": True, "atp_status": state["status"],
                                 "owner_job": owner_job(latest) if latest else None,
                                 "atp_recovery_errors": recovery_errors(probe_dir, model),
                                 "detail": "the probe stopped before this slot finished"
                                           + (f": {probe_error}" if probe_error else "")}
            elif record["status"] not in VALID:
                entry["loss"] = classify_loss(state["status"], loss_evidence(latest) if latest else {},
                                              record["verifier_error"])
            out.append(entry)
        return out

    async def run_probe(self, model: str, replaces: list, seq: int, budget=None) -> dict:
        """One recovery campaign through rollouts.probe() (models=(model,), repeats=len(replaces)), in its own ATP
        scratch; resumable (deterministic probe dir)."""
        n = len(replaces)
        pid = probe_id(self.task_id, model, n, self.label, seq)
        probe_dir = self.home / f"probe-{pid}"
        entry = {"id": pid, "dir": str(probe_dir), "model": model, "repeats": n, "seq": seq, "replaces": replaces,
                 "label": self.label}
        complete = (probe_dir / "manifest.json").exists() and rollouts.probe_complete(probe_dir)
        if not complete:
            if SETTINGS["models"][model].get("local"):
                entry["runway"] = await wait_runway()
            else:
                entry["hub_guard"] = await hub_guard(f"{self.task_id} {model} probe")
        if budget is not None:
            budget.probes.append((model, probe_dir))
        base = common.scratch() / f"rcv-{pid}"
        base.mkdir(mode=0o700, parents=True, exist_ok=True)
        _ATP_SCRATCH.set(base)
        earlier = next((p for p in self.state()["probes"] if p["id"] == pid), {})
        entry.update(started_at=earlier.get("started_at") or now(), hub_key=hub_key_name(),  # what this worker bills
                     hub_key_first=earlier.get("hub_key_first") or hub_key_name(),
                     job=os.environ.get("SLURM_JOB_ID"), attached_at=now())
        async with self.edit() as state:
            state["probes"] = [p for p in state["probes"] if p["id"] != pid] + [entry]
        log(f"{self.task_id}: {model} x{n} probe {probe_dir.name} (replaces {', '.join(replaces)})")
        try:
            await rollouts.probe(probe_dir, [self.task_copy], models=(model,), repeats=n)
            entry["complete"] = True
        except Exception as error:  # noqa: BLE001 - recorded; the slots it left are read below
            entry.update(complete=False, error=redact(f"{type(error).__name__}: {error}", 600))
            log(f"{self.task_id}: probe {probe_dir.name} failed: {entry['error']}")
        entry["finished_at"] = now()
        entry["slots"] = await asyncio.to_thread(self.probe_slots, probe_dir, replaces, entry.get("error"))
        if not entry["slots"] and entry.get("complete") is False:
            entry["slots"] = [{"slot": f"{model}-r{i}", "replaces": slot, "status": "infra", "atp_state": "not_started",
                               "attempts": 0, "attempt_dirs": [], "loss": {"kind": "not_started", "retry": False,
                                                                           "platform": True, "detail": entry["error"]}}
                              for i, slot in enumerate(replaces)]
        entry["settings"] = {k: (read_json(probe_dir / "settings.json") or {}).get(k)
                             for k in ("repeats", "max_concurrency_per_model", "initial_concurrency_per_model",
                                       "max_attempts", "agent_timeout_seconds")}
        async with self.edit() as state:
            state["probes"] = [p for p in state["probes"] if p["id"] != pid] + [entry]
        return entry

    async def grade_and_judge(self, probe: dict, gate) -> None:
        records = await asyncio.to_thread(self.probe_records, Path(probe["dir"]))
        jobs = []
        for slot in probe["slots"]:
            if slot["status"] not in VALID:
                continue
            key, record = f"{probe['id']}:{slot['slot']}", records[slot["slot"]]
            jobs.append(self._grade_then_judge(key, record, gate, judge_if_pass=probe["model"] == TEACHER))
        for result in await asyncio.gather(*jobs, return_exceptions=True):
            if isinstance(result, BaseException):
                self.problems.append(f"probe {probe['id']}: {redact(f'{type(result).__name__}: {result}', 300)}")

    async def _grade_then_judge(self, key: str, record: dict, gate, *, judge_if_pass: bool,
                                recorded_pass: bool = False) -> None:
        grades = await self.grade(key, record, gate)
        passed = record["status"] == "pass" or any((grades.get(g) or {}).get("status") == "pass" for g in GRADERS)
        if judge_if_pass and passed and not recorded_pass:
            await self.judge(key, record)

    async def regrade_recorded(self, slot: str, gate) -> None:
        """Grade a RECORDED slot's patch with both graders (its old-grader control should reproduce the recorded
        outcome); a kimi slot that passes and had no audit is judged."""
        entry = next((r for r in self.slots if r["slot"] == slot), None)
        if entry is None:
            raise RecoveryError(f"{slot}: not a slot of {self.task_id}")
        if entry["status"] not in VALID:
            self.notes.append(f"{slot}: recorded {entry['status']} (lost), not regraded")
            return
        await self._grade_then_judge(f"recorded:{slot}", self.by_slot[slot], gate,
                                     judge_if_pass=entry["model"] == TEACHER,
                                     recorded_pass=entry["status"] == "pass")

    async def note(self, text: str) -> None:
        """A note that survives a job restart (state.json) and goes into the record."""
        log(f"{self.task_id}: {text}")
        async with self.edit() as state:
            state.setdefault("notes", []).append({"at": now(), "label": self.label, "note": text})

    async def recover_model(self, model: str, replaces: list, *, budget, limiter, gate, retry: bool) -> None:
        async with limiter.hold(model, len(replaces)):
            first = await asyncio.create_task(self.run_probe(model, replaces, 0, budget))
        await self.grade_and_judge(first, gate)
        self.write_record()
        lost = [s for s in first["slots"] if s["status"] not in VALID and (s.get("loss") or {}).get("retry", True)
                and s.get("replaces")]
        if not lost:
            return
        again = [s["replaces"] for s in lost]
        kinds = ", ".join(sorted({(s.get("loss") or {}).get("kind", "?") for s in lost}))
        if not retry:
            await self.note(f"{model}: {len(lost)} recovery slot(s) lost again ({', '.join(again)}: {kinds}); --no-retry")
            return
        retry_dir = self.home / f"probe-{probe_id(self.task_id, model, len(again), self.label, 1)}"
        started = budget.attempts(model) if budget is not None else 0
        if (retry_dir / "manifest.json").exists():  # granted by an earlier run of this label (e.g. before a requeue)
            if budget is not None:
                budget.entries.append({"model": model, "n": len(again), "why": f"resume {retry_dir.name}",
                                       "granted": True, "at": now(), "attempts_started": started, "resume": True})
        elif budget is not None and not budget.reserve(model, len(again), f"retry {self.task_id} {again}",
                                                       attempts_started=started):
            await self.note(f"{model}: retry of {again} ({kinds}) skipped: rollout budget {budget.limits.get(model)} "
                            f"(reserved {budget.reserved.get(model, 0)}, attempts started {started})")
            return
        await self.note(f"{model}: {len(lost)} recovery slot(s) lost ({', '.join(again)}: {kinds}); one retry "
                        f"({retry_dir.name})")
        async with limiter.hold(model, len(again)):
            second = await asyncio.create_task(self.run_probe(model, again, 1, budget))
        await self.grade_and_judge(second, gate)
        still = [s["replaces"] for s in second["slots"] if s["status"] not in VALID]
        if still:
            await self.note(f"{model}: {', '.join(still)} lost again after the retry; not retried further")

    # ---------------------------------------------------------------------------------------------- record
    def recovered(self, state: dict) -> list:
        verdicts, grades = state.get("verdicts") or {}, state.get("grades") or {}
        out = []
        for probe in sorted(state.get("probes") or [], key=lambda p: p.get("seq", 0)):
            for slot in probe.get("slots") or []:
                key = f"{probe['id']}:{slot['slot']}"
                g, v = grades.get(key) or {}, verdicts.get(key)
                out.append(dict(slot, key=key, model=probe["model"], probe=probe["dir"], probe_id=probe["id"],
                                seq=probe.get("seq", 0), hub_key=probe.get("hub_key"),
                                grading={"recorded": {"status": slot["status"], "f2p": slot.get("f2p"),
                                                      "p2p": slot.get("p2p"), "by": "probe verifier (ATP, the task's "
                                                                                     "recorded grader)"},
                                         "old_control": grade_short(g.get("old")), "new": grade_short(g.get("new"))},
                                judge={k: v.get(k) for k in ("cheated", "categories", "evidence", "rubric_score",
                                                             "notes", "error", "at", "hub_key")}
                                if isinstance(v, dict) else None))
        return out

    def build_record(self, *, refresh: bool = False) -> dict:
        """The record from state.json (+ the recorded facts, the scratch staging and any grading_repair record).
        refresh: re-read each recovery probe's slots from its directory (in memory only; state.json is not written)."""
        state = self.state()
        if refresh:
            for probe in state.get("probes") or []:
                if (Path(probe["dir"]) / "manifest.json").exists():
                    probe["slots"] = self.probe_slots(Path(probe["dir"]), probe["replaces"], probe.get("error"))
        recovered = self.recovered(state)
        chosen = final_replacements(recovered)
        replacements = {slot: {"status": e["status"], "grades": {g: (e["grading"].get("new" if g == "new" else
                                                                                       "old_control") or {}).get("status")
                                                                 for g in GRADERS},
                               "cheated": (e.get("judge") or {}).get("cheated"),
                               "categories": (e.get("judge") or {}).get("categories")} for slot, e in chosen.items()}
        regraded = {}
        for slot in [k.split(":", 1)[1] for k in state.get("grades") or {} if k.startswith("recorded:")]:
            g = state["grades"][f"recorded:{slot}"]
            verdict = (state.get("verdicts") or {}).get(f"recorded:{slot}") or {}
            regraded[slot] = {"new": (g.get("new") or {}).get("status"), "old": (g.get("old") or {}).get("status"),
                              "cheated": verdict.get("cheated"), "categories": verdict.get("categories"),
                              "new_detail": grade_short(g.get("new")), "old_detail": grade_short(g.get("old")),
                              "judge": {k: verdict.get(k) for k in ("cheated", "categories", "evidence",
                                                                    "rubric_score", "notes", "error", "at")}
                              if verdict else None}
        repair = read_repair(self.run, self.task_id, self.loc["probe"])
        validation = read_json(self.scratch / "validate.json") or {}
        sanitizing_ok = bool(self.stage_info and self.stage_info.get("sanitizing") and validation.get("all_ok"))
        problems = list(self.problems) + list(repair.get("problems") or [])
        if self.stage_info and not self.stage_info.get("sanitizing"):
            problems.append("no sanitizing grading: " + str(self.stage_info.get("solution_test_path_hits")))
        elif self.stage_info and validation and not validation.get("all_ok"):
            problems.append(f"sanitizing copy failed validation: {validation.get('checks')}")
        flaky = []
        for e in recovered:
            old = (e["grading"].get("old_control") or {}).get("status")
            if e["status"] in VALID and old in VALID and old != e["status"]:
                flaky.append(f"{e['key']}: probe {e['status']}, recorded-grader control {old}")
        for slot, info in regraded.items():
            recorded = next(r["status"] for r in self.slots if r["slot"] == slot)
            if info["old"] in VALID and info["old"] != recorded:
                flaky.append(f"recorded:{slot}: recorded {recorded}, recorded-grader control {info['old']}")
        views, evals = {}, {}
        for name, grader in (("recorded_grader", "recorded"), ("sanitizing_grader", "new")):
            slots = view_slots(self.slots, replacements if (grader == "recorded" or sanitizing_ok) else
                               {k: dict(v, grades={}) for k, v in replacements.items()},
                               grader=grader, regraded=regraded if sanitizing_ok else {},
                               repair=repair["statuses"] if grader == "new" else None)
            views[name], evals[name] = slots, evaluate(slots)
        pending = pending_repair(views["sanitizing_grader"], self.slots)
        before = self.before
        decision = evals["sanitizing_grader"]
        recommendation = recommend(before, decision)
        metrics = self.loc["metrics"]
        if (before["sc14"], before["kimi_clean"], before["valid"]) != \
                (metrics.get("student_rate"), metrics.get("teacher_clean_rate"), metrics.get("valid")):
            problems.append(f"recomputed before-rates {before['sc14']}, {before['kimi_clean']}, {before['valid']} != "
                            f"metrics {metrics.get('student_rate')}, {metrics.get('teacher_clean_rate')}, "
                            f"{metrics.get('valid')}")
        notes = [n.get("note") if isinstance(n, dict) else str(n) for n in state.get("notes") or []] + \
            list(self.notes)
        if evals["recorded_grader"]["in_range"] != decision["in_range"]:
            notes.append(f"the graders disagree on the verdict: recorded grader {evals['recorded_grader']['reason']}; "
                         f"sanitizing grader {decision['reason']}")
        if recommendation == "newly_in_range":
            notes.append("main: run the similarity judge (score()'s failure analysis + judge_similarity); accepting "
                         "also needs the sanitizing grader installed (grading_repair install) for consistency")
        summary_slots = []
        for r in self.slots:
            item = {k: r[k] for k in ("slot", "model", "status", "atp_state", "attempts", "f2p", "p2p", "agent_error",
                                      "verifier_error", "patch_sha256", "patch_chars", "cheated", "categories", "lost",
                                      "loss", "surviving_test_edits", "attempt_dirs")}
            if r["slot"] in regraded:
                item["regraded_here"] = {k: regraded[r["slot"]][k] for k in ("new_detail", "old_detail", "judge")}
            summary_slots.append(item)
        probes = [{k: p.get(k) for k in ("id", "dir", "model", "repeats", "seq", "replaces", "label", "started_at",
                                         "finished_at", "complete", "error", "hub_key", "hub_key_first", "hub_key_note", "node_failure", "job",
                                         "runway", "hub_guard",
                                         "settings")} for p in state.get("probes") or []]
        return {
            "tool": TOOL, "at": now(), "label": self.label, "run": self.run, "task_id": self.task_id,
            "cand": self.loc["cand"], "round": self.loc["round"], "build_root": str(self.loc["root"]),
            "scored_dir": str(self.loc["scored"]), "probe": str(self.loc["probe"]),
            "task_dir_probed": str(self.loc["probed"]), "task_dir_packaged": str(self.loc["packaged"]),
            "probed_vs_packaged_differs": (self.stage_info or {}).get("packaged_differs"),
            "metrics_before": {k: metrics.get(k) for k in ("student_rate", "teacher_clean_rate", "valid", "accept",
                                                           "reason", "similarity", "answer_retrieval", "cheated")},
            "recorded_slots": summary_slots,
            "lost_slots": {model: [r["slot"] for r in self.slots if r["model"] == model and r["lost"]]
                           for model in (STUDENT, TEACHER)},
            "recovery_probes": probes,
            "recovered_slots": recovered,
            "replacements": {slot: e["key"] for slot, e in chosen.items()},
            "regraded_recorded_slots": {slot: {k: v for k, v in info.items() if k not in ("cheated", "categories")}
                                        for slot, info in regraded.items()},
            "grading": {"stage": {k: (self.stage_info or {}).get(k) for k in (
                "globs", "test_paths_mode", "grader_sha256", "config_sha256", "recorded_grader",
                "recorded_grader_sha256", "sanitizing", "solution_test_path_hits")},
                "scratch": str(self.scratch),
                "validation": {k: validation.get(k) for k in ("checks", "all_ok", "sandbox_ok", "at")} or None,
                "flaky": flaky},
            "repair_record": {k: repair.get(k) for k in ("path", "present", "format", "recommendation", "rates_after",
                                                          "at", "sha256", "statuses")},
            "before": before,
            "after": {name: dict(evals[name], statuses={x["slot"]: x["status"] for x in views[name]},
                                 sources={x["slot"]: x["source"] for x in views[name]}) for name in views},
            "decision_view": "sanitizing_grader",
            "decision_basis": ("sanitizing grader (hengyuf: these runs' tasks are re-graded with verifier/grader.py); "
                               "recovered rollouts by their sanitizing grade; recorded slots by "
                               + ("the grading_repair record" if repair["present"] and repair["statuses"] else
                                  "their recorded outcome (no grading_repair record yet)")
                               + (" and this tool's regrade of " + ", ".join(sorted(regraded)) if regraded else "")),
            "pending_repair": robustness(views["sanitizing_grader"], pending),
            "recommendation": recommendation,
            "notes": notes, "problems": problems}

    def export(self, grader: str) -> dict:
        """audit/recovery/<T>/after-records-<grader>.jsonl: the slot records score() would take after the recovery
        (grader "recorded": as graded by the probes; "sanitizing": recovered and regraded-here slots with their
        sanitizing grade, other recorded slots as recorded, listed under "unrepaired" when they edit test paths), plus a
        sidecar .json with the replacements and the teacher verdicts of every kimi pass (for reuse by a re-score)."""
        state = self.state()
        chosen = final_replacements(self.recovered(state))
        recovered, overrides, cache = {}, {}, {}
        for slot, entry in chosen.items():
            if entry["status"] not in VALID:
                continue
            probe_dir = Path(entry["probe"])
            if probe_dir not in cache:
                cache[probe_dir] = self.probe_records(probe_dir)
            recovered[slot] = {"probe": str(probe_dir), "record": cache[probe_dir][entry["slot"]], "key": entry["key"]}
        sources = {slot: (item["key"], f"recovered {item['key']}") for slot, item in recovered.items()}
        sources.update({k.split(":", 1)[1]: (k, f"regraded here {k}") for k in state["grades"] if k.startswith("recorded:")})
        if grader == "sanitizing":
            for slot, (key, _) in sources.items():
                new = (state["grades"].get(key) or {}).get("new") or {}
                if new.get("status") not in VALID or not new.get("record"):
                    raise RecoveryError(f"{key}: no sanitizing grade to export")
                overrides[slot] = dict(new["record"], info={"tool": TOOL, "grader": "new", "out_dir": new.get("out_dir"),
                                                            "grader_sha256": new.get("grader_sha256"),
                                                            "config_sha256": new.get("config_sha256")})
        records = after_records(sorted(self.loc["records"], key=lambda r: slot_order(r["slot"])), recovered, overrides)
        audit = read_json(self.loc["scored"] / "teacher_audit.json") or {}
        verdicts = {}
        for record in records:
            if record["model"] != TEACHER or record["status"] != "pass":
                continue
            slot = record["slot"]
            if slot in recovered:  # a recovered rollout: this tool's audit only
                key = recovered[slot]["key"]
                verdicts[slot] = {"verdict": state["verdicts"].get(key), "from": key}
            elif f"recorded:{slot}" in state["verdicts"]:  # a recorded slot that passes only after a regrade here
                verdicts[slot] = {"verdict": state["verdicts"][f"recorded:{slot}"], "from": f"recorded:{slot}"}
            else:  # a recorded pass: the first scoring's audit
                verdicts[slot] = {"verdict": audit.get(slot), "from": str(self.loc["scored"] / "teacher_audit.json")}
        path = self.home / f"after-records-{grader}.jsonl"
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
        temporary.replace(path)
        edits = {r["slot"]: r.get("surviving_test_edits") for r in self.slots}
        repair = read_repair(self.run, self.task_id, self.loc["probe"])
        if grader == "sanitizing":  # a repair record that disagrees with a recorded outcome must not be exported over
            disagree = [r["slot"] for r in records if r["slot"] not in overrides and r["status"] in VALID
                        and repair["statuses"].get(r["slot"], r["status"]) != r["status"]]
            if disagree:
                raise RecoveryError(f"the grading_repair record changes {disagree}: export needs their regraded records")
        side = {"at": now(), "tool": TOOL, "grader": grader, "records": str(path), "sha256": gr.file_sha(path),
                "grading_repair_record": repair["path"] if repair["present"] else None,
                "task_dir": str(self.loc["packaged"]), "snapshot": self.loc["metrics"].get("snapshot"),
                "replaced": {slot: f"{item['probe']} {item['record']['slot']}" for slot, item in recovered.items()},
                "grade_overrides": sorted(overrides), "kimi_pass_verdicts": verdicts,
                "unrepaired_with_test_edits": sorted((r["slot"] for r in records if r["slot"] not in overrides
                                                      and r["slot"] not in repair["statuses"]
                                                      and r["status"] in VALID and edits.get(r["slot"])),
                                                     key=slot_order) if grader == "sanitizing" else None}
        write_json(self.home / f"after-records-{grader}.json", side)
        return side

    def write_record(self, *, refresh: bool = False) -> dict:
        record = self.build_record(refresh=refresh)
        write_json(self.record_path, record)
        return record


# ============================================================================================ orchestration

async def run_task(task: Task, recover: dict, regrade_slots: list, *, budget, limiter, gate, retry=True) -> dict:
    """stage -> validate -> (probe per model || regrade named recorded slots) -> grade, judge -> one retry -> record."""
    await asyncio.to_thread(task.load)
    plan = task.plan(recover)
    for model, item in plan.items():
        if not item["decisive"]:
            task.notes.append(f"{model}: no outcome of {item['n']} rollout(s) changes the verdict ({item['outcomes']})")
    await asyncio.to_thread(task.stage)
    task.write_record()
    validation = await task.validate(gate)
    if not validation.get("skipped") and not validation.get("sandbox_ok"):
        log(f"{task.task_id}: validation produced no grade; one more try in 300 s")
        await asyncio.sleep(300)
        validation = await task.validate(gate)
    if validation.get("skipped"):
        task.notes.append(f"sanitizing grader unavailable: {validation['skipped']}")
    elif not validation.get("sandbox_ok"):
        errors = [str((r or {}).get("error"))[:200] for r in (validation.get("results") or {}).values()]
        task.problems.append(f"no sandbox from the task snapshot {task.snapshot} (validation produced no grade: "
                             f"{errors}); no rollouts run")
        return task.write_record()
    task.write_record()
    jobs = [task.regrade_recorded(slot, gate) for slot in regrade_slots]
    jobs += [task.recover_model(model, item["replaces"], budget=budget, limiter=limiter, gate=gate, retry=retry)
             for model, item in plan.items()]
    for result in await asyncio.gather(*jobs, return_exceptions=True):
        if isinstance(result, BaseException):
            task.problems.append(redact(f"{type(result).__name__}: {result}", 600))
    return task.write_record()


def load_spec(path) -> dict:
    spec = read_json(path)
    if not isinstance(spec, dict) or not spec.get("label") or not isinstance(spec.get("items"), list):
        raise RecoveryError(f"{path}: a spec needs a label and items")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", spec["label"]):
        raise RecoveryError(f"label {spec['label']!r}: letters, digits, . _ - only")
    budget = dict(DEFAULT_BUDGET, **(spec.get("budget") or {}))
    tasks = {}
    for item in spec["items"]:
        run, task_id, model, n = item.get("run"), item.get("task"), item.get("model"), item.get("n")
        if not run or not task_id:
            raise RecoveryError(f"item {item}: run and task are required")
        entry = tasks.setdefault((run, task_id), {"run": run, "task": task_id, "recover": {}, "regrade_slots": []})
        if model is not None:
            if model not in SETTINGS["models"] or model == JUDGE:
                raise RecoveryError(f"item {item}: model must be {STUDENT} or {TEACHER}")
            if model in entry["recover"]:
                raise RecoveryError(f"item {item}: {model} given twice for {task_id}")
            if not isinstance(n, int) or isinstance(n, bool) or n < 1:
                raise RecoveryError(f"item {item}: n must be a positive integer")
            entry["recover"][model] = n
        entry["regrade_slots"] += [s for s in item.get("regrade_slots") or [] if s not in entry["regrade_slots"]]
    totals = {}
    for entry in tasks.values():
        for model, n in entry["recover"].items():
            totals[model] = totals.get(model, 0) + n
    over = {m: (t, budget.get(m, 0)) for m, t in totals.items() if t > budget.get(m, 0)}
    if over:
        raise RecoveryError(f"initial rollouts exceed the budget: {over}")
    return {"label": spec["label"], "budget": budget, "tasks": list(tasks.values()), "raw": spec}


class JobBudget(Budget):
    """Budget plus the job's probe dirs, so a retry also counts attempts ATP actually started."""

    def __init__(self, limits):
        super().__init__(limits)
        self.probes = []

    def attempts(self, model: str) -> int:
        total = 0
        for probe_model, probe_dir in self.probes:
            if probe_model == model:
                total += sum(1 for _ in Path(probe_dir).glob(f"models/{model}/attempts/*/attempt-*/started.json"))
        return total


async def run_batch(spec: dict, *, retry=True) -> dict:
    budget, limiter, gate = JobBudget(spec["budget"]), Limiter(MAX_PER_MODEL), asyncio.Semaphore(gr.MAX_SANDBOXES)
    tasks = []
    for entry in spec["tasks"]:
        for model, n in entry["recover"].items():
            if not budget.reserve(model, n, f"initial {entry['task']}"):
                raise RecoveryError(f"budget refused {model} x{n} for {entry['task']}")
        tasks.append((entry, Task(entry["run"], entry["task"], spec["label"])))
    out = BATCHES / spec["label"]
    started = now()
    with per_probe_scratch():
        results = await asyncio.gather(*(run_task(task, entry["recover"], entry["regrade_slots"], budget=budget,
                                                  limiter=limiter, gate=gate, retry=retry)
                                         for entry, task in tasks), return_exceptions=True)
    summary = {"label": spec["label"], "started_at": started, "finished_at": now(),
               "job": os.environ.get("SLURM_JOB_ID"), "budget": budget.summary(),
               "attempts_started": {m: budget.attempts(m) for m in budget.limits}, "tasks": []}
    for (entry, task), result in zip(tasks, results):
        if isinstance(result, BaseException):
            summary["tasks"].append({"run": entry["run"], "task": entry["task"],
                                     "error": redact(f"{type(result).__name__}: {result}", 600)})
            try:
                task.problems.append(redact(f"{type(result).__name__}: {result}", 600))
                if task.loc is not None:
                    task.write_record()
            except Exception as error:  # noqa: BLE001
                log(f"{entry['task']}: record not written: {type(error).__name__}: {error}")
            continue
        summary["tasks"].append(table_row(result))
    write_json(out / "summary.json", summary)
    print_table(summary["tasks"])
    return summary


def fmt(value) -> str:
    return "-" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)


def table_row(record: dict) -> dict:
    before, after = record["before"], record["after"]
    return {"run": record["run"], "task": record["task_id"], "record": str(RUNS / record["run"] / AUDIT /
                                                                           f"{record['task_id']}.json"),
            "lost": record["lost_slots"], "replacements": record["replacements"],
            "before": {k: before.get(k) for k in ("valid", "sc14", "kimi_clean", "in_range")},
            "after_recorded": {k: after["recorded_grader"].get(k) for k in ("valid", "sc14", "kimi_clean", "in_range")},
            "after_sanitizing": {k: after["sanitizing_grader"].get(k) for k in ("valid", "sc14", "kimi_clean",
                                                                                "in_range")},
            "recommendation": record["recommendation"], "problems": record["problems"]}


def print_table(rows: list) -> None:
    print("| task | before valid / sc14 / kimi | after (recorded grader) | after (sanitizing) | recommendation |")
    print("|---|---|---|---|---|")
    for row in rows:
        if "error" in row:
            print(f"| {row['task']} | ERROR {row['error'][:120]} | | | |")
            continue

        def cell(item):
            return f"{item['valid']} / {fmt(item['sc14'])} / {fmt(item['kimi_clean'])}"
        print(f"| {row['task']} | {cell(row['before'])} | {cell(row['after_recorded'])} | "
              f"{cell(row['after_sanitizing'])} | {row['recommendation']} |")


# ============================================================================================ CLI

def scrubbed_env() -> dict:
    user = pwd.getpwuid(os.getuid())
    return {"HOME": user.pw_dir, "USER": user.pw_name, "LANG": "C.UTF-8", "SLURM_CONF": SLURM_CONF,
            "PATH": SLURM_PATH}


def cmd_plan(args) -> int:
    task = Task(args.run, args.task, args.label).load()
    recover = {args.model: args.n} if args.model else {}
    view = view_slots(task.slots, {}, grader="recorded")
    out = {"run": args.run, "task_id": args.task, "probe": str(task.loc["probe"]), "scored": str(task.loc["scored"]),
           "metrics_before": {k: task.loc["metrics"].get(k) for k in ("student_rate", "teacher_clean_rate", "valid",
                                                                      "reason")},
           "before": task.before,
           "slots": [{k: r[k] for k in ("slot", "status", "atp_state", "attempts", "f2p", "p2p", "cheated", "loss")}
                     | {"surviving_test_edits": len(r["surviving_test_edits"])} for r in task.slots],
           "pending_repair_if_unregraded": pending_repair(view, task.slots), "problems": task.problems}
    if recover:
        out["plan"] = task.plan(recover)
    print(json.dumps(out, indent=2, default=str))
    return 0


async def _run(args) -> int:
    recover = {args.model: args.n}
    budget = JobBudget(parse_budget(args.budget))
    if not budget.reserve(args.model, args.n, f"initial {args.task}"):
        raise RecoveryError(f"{args.model} x{args.n} exceeds the budget {budget.limits}")
    task = Task(args.run, args.task, args.label)
    with per_probe_scratch():
        record = await run_task(task, recover, args.regrade_slots or [], budget=budget, limiter=Limiter(),
                                gate=asyncio.Semaphore(gr.MAX_SANDBOXES), retry=not args.no_retry)
    print_table([table_row(record)])
    print(f"record: {task.record_path}")
    return 0


def cmd_record(args) -> int:
    task = Task(args.run, args.task, args.label).load()
    task.stage_info = read_json(task.scratch / "stage.json")
    if not task.stage_info:
        raise RecoveryError(f"{task.task_id}: nothing staged yet ({task.scratch})")
    state = task.state()
    if state.get("probes") and args.label == "manual":
        task.label = state["probes"][-1].get("label") or task.label
    record = task.write_record(refresh=True)
    print_table([table_row(record)])
    print(f"record: {task.record_path}")
    return 0


def cmd_export(args) -> int:
    task = Task(args.run, args.task, args.label).load()
    side = task.export(args.grader)
    print(json.dumps(side, indent=2, default=str))
    return 0


def cmd_batch(args) -> int:
    spec = load_spec(args.spec)
    summary = asyncio.run(run_batch(spec, retry=not args.no_retry))
    return 1 if any("error" in t for t in summary["tasks"]) else 0


def cmd_submit(args) -> int:
    spec = load_spec(args.spec)
    label = spec["label"]
    out = BATCHES / label
    out.mkdir(parents=True, exist_ok=True)
    spec_path = out / "spec.json"
    if Path(args.spec).resolve() != spec_path.resolve():
        write_json(spec_path, spec["raw"])
    name = f"synth-recover-{label}"
    # --open-mode=append: a job that Slurm requeues (e.g. after NODE_FAIL; same job id) appends to its log
    job_args = [f"--job-name={name}", f"--time={args.time}", f"--cpus-per-task={args.cpus}", f"--mem={args.mem}",
                f"--output={out}/%x-%j.out", "--open-mode=append", str(SYNTH / "run.sbatch"),
                "tools/recover_slots.py", "batch", "--spec", str(spec_path)]
    env = scrubbed_env()
    python = "/usr/bin/python3" if os.path.exists("/usr/bin/python3") else sys.executable
    pick = subprocess.run([python, "-I", "-B", str(S / "tools/pick_partition.py"), "--partitions", args.partitions,
                           "--", *job_args], cwd=SYNTH, env=env, capture_output=True, text=True, timeout=900)
    try:
        choice = json.loads(pick.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise RecoveryError(f"pick_partition.py failed: {pick.stderr[-500:]}") from None
    argv = ["sbatch", "--parsable", f"--partition={choice['partition']}", *job_args]
    if args.dry_run:
        print(json.dumps({"partition_probe": choice, "argv": argv}, indent=2))
        return 0
    result = subprocess.run(argv, cwd=SYNTH, env=env, capture_output=True, text=True, timeout=300)
    if result.returncode:
        raise RecoveryError(f"sbatch failed: {result.stderr.strip()[-500:]}")
    job = result.stdout.strip().split(";")[0]
    launch = {"submitted_at": now(), "job_id": job, "label": label, "partition": choice["partition"],
              "partition_probe": choice, "argv": argv, "log": f"{out}/{name}-{job}.out", "spec": str(spec_path)}
    write_json(out / f"launch-{job}.json", launch)
    with (BATCHES / "launch.jsonl").open("a") as stream:
        stream.write(json.dumps(launch) + "\n")
    print(json.dumps(launch, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def task_args(sub, model_required=False):
        sub.add_argument("--run", required=True)
        sub.add_argument("--task", required=True, help="task id (syn-...)")
        sub.add_argument("--model", choices=(STUDENT, TEACHER), required=model_required)
        sub.add_argument("-n", "--n", type=int, required=model_required, help="fresh rollouts (<= lost slots)")
        sub.add_argument("--label", default="manual", help="recovery label (part of the probe id; reuse to resume)")

    sub = commands.add_parser("plan", help="read-only: recorded slots, lost slots and why, what N rollouts can do")
    task_args(sub)
    sub = commands.add_parser("run", help="(CPU job) one task's recovery")
    task_args(sub, model_required=True)
    sub.add_argument("--regrade-slots", nargs="+", help="recorded slots to regrade with both graders, e.g. kimi-r0")
    sub.add_argument("--no-retry", action="store_true", help="do not retry slots lost again")
    sub.add_argument("--budget", default="", help="rollout budget per model, e.g. kimi=10,sc14=4")
    sub = commands.add_parser("batch", help="(CPU job) every recovery of a spec")
    sub.add_argument("--spec", required=True)
    sub.add_argument("--no-retry", action="store_true")
    sub = commands.add_parser("submit", help="submit `batch` as one CPU job")
    sub.add_argument("--spec", required=True)
    sub.add_argument("--time", default="10:00:00")
    sub.add_argument("--cpus", type=int, default=16)
    sub.add_argument("--mem", default="64G")
    sub.add_argument("--partitions", default="cpu_interactive,cpu_dataprocessing,cpu_datamover")
    sub.add_argument("--dry-run", action="store_true")
    sub = commands.add_parser("record", help="rebuild the record from saved outputs (no sandbox, no LLM)")
    task_args(sub)
    sub = commands.add_parser("export", help="the slot records score() would take after the recovery (JSONL)")
    task_args(sub)
    sub.add_argument("--grader", choices=("recorded", "sanitizing"), default="sanitizing")
    args = parser.parse_args(argv)
    if args.command == "plan" and bool(args.model) != (args.n is not None):
        parser.error("plan takes --model and -n together")
    try:
        if args.command == "run":
            return asyncio.run(_run(args))
        return {"plan": cmd_plan, "record": cmd_record, "batch": cmd_batch, "submit": cmd_submit,
                "export": cmd_export}[args.command](args)
    except RecoveryError as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
