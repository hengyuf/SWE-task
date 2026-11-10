"""Audit historical grading with revision 7 without changing any campaign artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time

import deepswe_campaign as state


@contextmanager
def readonly_assessments(assessment_timestamp=None):
    assessments, torn = {}, set()
    original_read, original_write, original_rows = state.read_json, state.atomic_json, state.read_jsonl

    def read(path, default=None):
        path = Path(path)
        return assessments[path] if path in assessments else original_read(path, default)

    def write(path, value):
        path = Path(path)
        if path.name != "generation-validity-v2-r7.json":
            raise RuntimeError("Read-only audit rejected a write to " + str(path))
        assessments[path] = dict(value, assessed_at=assessment_timestamp) if assessment_timestamp is not None else value

    def rows(path):
        path = Path(path)
        values = []
        with path.open("rb") as stream:
            line = stream.readline()
            while line:
                following = stream.readline()
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("JSONL record is not an object")
                except (ValueError, UnicodeError):
                    if following or line.endswith(b"\n"):
                        raise ValueError("Corrupt JSONL record in " + str(path)) from None
                    torn.add(str(path))
                    break
                values.append(value)
                line = following
        return values

    state.read_json, state.atomic_json, state.read_jsonl = read, write, rows
    try:
        yield assessments, torn
    finally:
        state.read_json, state.atomic_json, state.read_jsonl = original_read, original_write, original_rows


def evidence(directory, assessments):
    started = state.read_json(directory / "started.json", {})
    source = Path(started.get("agent_artifacts_from", str(directory)))
    result = assessments.get(source / "generation-validity-v2-r7.json")
    if result is None:
        result = state.read_json(source / "generation-validity-v2-r7.json", {})
    if result.get("evidence_signatures") != state._generation_signatures(source):
        return {"assessment_detail": "No current cached assessment; an early validity guard or changing evidence prevented caching."}
    return {key: result[key] for key in (
        "failure", "assessment_revision", "budget_timeout", "malformed_success_calls",
        "unresolved_http_outcomes", "tool_delivery", "native_model_errors", "delivery_gap",
        "stream_errors", "stream_cancelled", "recovered_calls") if key in result}


def audit(campaign, models=None, task_indices=None, slots=None):
    campaign = Path(campaign).resolve()
    manifest = state.load_manifest(campaign)
    if manifest is None:
        raise ValueError("Campaign manifest is missing")
    models = models or list(manifest["settings"]["models"])
    if set(models) - set(manifest["settings"]["models"]):
        raise ValueError("Unknown model alias")
    known_tasks = {task[state.TASK] for task in manifest["tasks"]}
    if task_indices and set(task_indices) - known_tasks:
        raise ValueError("Unknown task index")
    if slots and any(task not in known_tasks or not 0 <= repeat < manifest["settings"]["repeats"] for task, repeat in slots):
        raise ValueError("Unknown task-repeat slot")
    started_at = time.time()
    report = {"assessment_timestamp": started_at, "campaign": str(campaign), "models": {},
              "baseline": "Cached revision-6 accepted grades with unchanged HTTP journals; absent or stale caches are unknown.",
              "snapshot_note": "Assessment timestamp is fixed at scan start; this is a scan interval, not an atomic snapshot of live files.",
              "changed_slots": [], "spot_checks": {}, "errors": [], "production_writes": 0}
    with readonly_assessments(started_at) as (assessments, torn):
        for model in models:
            counts = Counter({key: 0 for key in ("slots_inspected", "cached_r6_accepted_slots", "cached_r6_passed_slots",
                "baseline_unknown_attempts", "corrected_completed_slots", "corrected_passed_slots",
                "discarded_cached_r6_scores", "audit_errors")})
            statuses, failures = Counter(), Counter()
            known_checks = {(11, 12), (65, 12), (72, 4)} if model == "glm" else set()
            for task in manifest["tasks"]:
                index = task[state.TASK]
                if task_indices and index not in task_indices:
                    continue
                for repeat in range(manifest["settings"]["repeats"]):
                    if slots and (index, repeat) not in slots:
                        continue
                    counts["slots_inspected"] += 1
                    directory = state.slot_directory(campaign, model, index, repeat)
                    try:
                        attempts = state.attempts_for(directory)
                        baseline = None
                        for attempt in attempts:
                            terminal = state.read_json(attempt / "terminal.json", {})
                            if not terminal.get("evaluation_completed") is True:
                                continue
                            attempt_start = state.read_json(attempt / "started.json", {})
                            source = Path(attempt_start.get("agent_artifacts_from", str(attempt)))
                            cached = state.read_json(source / "generation-validity-v2-r6.json")
                            if cached is None or cached.get("http_journal_signature") != state._journal_signature(source / "http.jsonl"):
                                counts["baseline_unknown_attempts"] += 1
                                continue
                            if cached.get("failure") is None and baseline is None:
                                baseline = attempt
                        current = state.slot_state(campaign, model, index, repeat)
                        statuses[current["status"]] += 1
                        if baseline is not None:
                            counts["cached_r6_accepted_slots"] += 1
                            old_grade = state.read_json(baseline / "terminal.json", {})
                            counts["cached_r6_passed_slots"] += old_grade.get("reward") == 1
                        if current["status"] == "completed":
                            selected = Path(current["attempt_dir"])
                            counts["corrected_completed_slots"] += 1
                            counts["corrected_passed_slots"] += state.read_json(selected / "terminal.json", {}).get("reward") == 1
                        if baseline is not None:
                            failure = state.generation_failure(baseline, state.read_json(baseline / "terminal.json"))
                            if failure is not None:
                                counts["discarded_cached_r6_scores"] += 1
                                failures[failure] += 1
                                report["changed_slots"].append({"model": model, "task_index": index,
                                    "repeat_index": repeat, "previous_attempt": str(baseline),
                                    "previous_reward": state.read_json(baseline / "terminal.json", {}).get("reward"),
                                    "corrected_slot_status": current["status"], "failure": failure,
                                    "evidence": evidence(baseline, assessments)})
                        if (index, repeat) in known_checks or (model == "kimi" and "kimi_malformed" not in report["spot_checks"]):
                            chosen = baseline or (Path(current["attempt_dir"]) if current["attempt_dir"] else None)
                            detail = evidence(chosen, assessments) if chosen is not None else {}
                            if (index, repeat) in known_checks or detail.get("malformed_success_calls"):
                                key = f"{model}/{index}-{repeat}" if model == "glm" else "kimi_malformed"
                                report["spot_checks"][key] = {"attempt": str(chosen), "slot_status": current["status"], "evidence": detail}
                    except (OSError, ValueError, RuntimeError) as error:
                        counts["audit_errors"] += 1
                        report["errors"].append({"model": model, "task_index": index, "repeat_index": repeat,
                                                 "error_type": type(error).__name__, "error": str(error)})
                    if counts["slots_inspected"] % 100 == 0:
                        print(json.dumps({"model": model, "slots_inspected": counts["slots_inspected"],
                                          "corrected_completed": counts["corrected_completed_slots"]}), file=sys.stderr, flush=True)
            report["models"][model] = dict(counts, corrected_slot_statuses=dict(statuses), discarded_score_failures=dict(failures))
        report["in_memory_assessments"] = len(assessments)
        report["torn_jsonl_files"] = sorted(torn)
    report["finished_at"] = time.time()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--tasks", nargs="+", type=int)
    parser.add_argument("--slots", nargs="+", help="Optional task-repeat pairs, for example 11-12 65-12 72-4")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_JOB_PARTITION", "").startswith("cpu"):
        parser.error("Run inside the existing CPU allocation from node-local scratch")
    if not any(Path.cwd().is_relative_to(path) for path in (Path("/raid/scratch"), Path("/tmp"))):
        parser.error("Run from node-local scratch")
    output, campaign = arguments.output.resolve(), arguments.campaign.resolve()
    if output.is_relative_to(campaign):
        parser.error("Audit output must be outside the production campaign")
    if output.exists():
        parser.error("Audit output already exists; choose a new evidence file")
    slots = None
    if arguments.slots:
        try:
            slots = {tuple(map(int, item.split("-"))) for item in arguments.slots}
        except ValueError:
            parser.error("Slots must be task-repeat integer pairs")
        if any(len(slot) != 2 or min(slot) < 0 for slot in slots):
            parser.error("Slots must be nonnegative task-repeat pairs")
    report = audit(campaign, arguments.models, arguments.tasks, slots)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "models": report["models"], "errors": len(report["errors"])}))
    return int(bool(report["errors"]))


if __name__ == "__main__":
    sys.exit(main())
