#!/usr/bin/env python3
"""Report a conservative SWE score without changing frozen evaluation artifacts."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import shard_plan as sp


def digest(data):
    return hashlib.sha256(data).hexdigest()


def artifact_path(value):
    path = Path(value)
    return sp.contained(path if path.is_absolute() else sp.APP / path)


def records(path):
    with sp.contained(path).open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            if raw.strip():
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError("Result must be a JSON object")
                yield row, {"path": str(path), "line": number, "row_sha256": digest(raw.strip())}


def outcome(row, source, classification, evidence=None):
    return {"instance_id": row["instance_id"], "task_index": row["_ng_task_index"], "rollout_index": 0,
            "resolved": row["resolved"], "classification": classification, "source": source,
            "evidence": evidence or []}


def budget_evidence(path, audit, row, source, slot, unresolved_interruption=False):
    classification = audit["classification"]
    if not unresolved_interruption and (classification.get("confidence") != "strongly_supported"
            or classification.get("most_likely_cause") != "agent_wall_clock_budget_exhaustion"
            or classification.get("infra_failure_observed") is not False
            or classification.get("budget_allowlist_eligible") is False
            # Admission guard in addition to explicit audit, never automatic cause inference.
            or abs(audit.get("timing", {}).get("observed_session_seconds", 0) - 10800) > 120
            or audit["export"].get("assistant_error_count") != 0
            or audit["result"].get("command_status_error") != "signal: killed"
            or "signal: killed" not in ((row.get("opencode_run_stdout") or "") + (row.get("opencode_run_stderr") or ""))):
        raise ValueError("Budget audit is not strongly supported: " + str(path))
    if unresolved_interruption and row.get("resolved") is not False:
        raise ValueError("Conservative interruption admission requires an unresolved original")
    expected = {"instance_id": row["instance_id"], "global_task_index": row["_ng_task_index"],
                "rollout_index": 0, "slot": slot}
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError("Budget audit task identity changed")
    recorded = audit["source_rollout"]
    if artifact_path(recorded["path"]) != Path(source["path"]) or recorded["row_sha256"] != source["row_sha256"]:
        raise ValueError("Budget audit row hash or source changed")
    fields = ("evaluation_completed", "opencode_finished", "opencode_export_found", "resolved")
    if (sp.invalid_reasons(row) != ["opencode_finished_not_true"] or row.get("opencode_finished") is not False
            or row.get("failure_reason") is not None
            or any(audit["result"].get(key) != row.get(key) for key in fields)
            or audit["result"].get("failure_class") is not None):
        raise ValueError("Budget audit does not describe a completed graded candidate")
    export = artifact_path(audit["export"]["path"])
    candidate = (row.get("model_patch") or "").encode()
    budget = audit["budget"]
    if (artifact_path(row["opencode_results_fpath"]) != export or sp.sha256(export) != audit["export"]["sha256"]
            or digest(candidate) != audit["candidate_patch"]["sha256"]
            or len(candidate) != audit["candidate_patch"]["bytes"]
            or budget["seconds"] != 10800 or budget.get("launcher_timeout_overrides") != []
            or sp.sha256(artifact_path(budget["config_path"])) != budget["config_sha256"]):
        raise ValueError("Budget audit candidate, export or budget evidence changed")
    return [{"path": str(path), "sha256": sp.sha256(path),
             "confidence": classification["confidence"],
             "exact_kill_trigger_proven": classification.get("exact_kill_trigger_proven", False)}]


def score(plan_path, policy_path, budget_audits=(), out_dir=None, unresolved_interruption_audits=()):
    plan_path, policy_path = artifact_path(plan_path), artifact_path(policy_path)
    prepared = sp.read_json(sp.APP / "runtime/preparation.json")
    plan, master, retained, manifests = sp.validate_plan(plan_path, prepared)
    indices = {row["instance_id"]: i for i, row in enumerate(master)}
    policy = sp.read_json(policy_path)
    source_run = sp.contained(plan["source"]["run_dir"])
    source_file = source_run / "rollouts.jsonl"
    if (policy.get("reported_score_policy") != "conservative_original_oom_is_unresolved"
            or policy.get("instance_id") != "sympy__sympy-18698"
            or policy.get("global_task_index") != indices.get(policy["instance_id"])
            or artifact_path(policy["original_source"]) != source_file
            or policy["original_source_sha256"] != plan["source"]["rollouts_sha256"]):
        raise ValueError("Conservative OOM policy does not match the frozen source")
    oom_index = policy["global_task_index"]
    audits = {}
    for filename, interruption in [(p, False) for p in budget_audits] + [(p, True) for p in unresolved_interruption_audits]:
        path = artifact_path(filename)
        audit = sp.read_json(path)
        key = audit["global_task_index"]
        if type(key) is not int or not 0 <= key < 500 or key in audits or key == oom_index:
            raise ValueError("Duplicate or invalid budget audit identity")
        audits[key] = (path, audit, interruption)
    accepted, unadjudicated, diagnostics, file_hashes, sidecars = {}, [], [], {}, []
    conservative_interruptions = {}
    used_audits, runner_hashes, cleanup = set(), set(), []
    with ExitStack() as locks:
        locks.enter_context(sp.stopped_run(source_run))
        for manifest in manifests:
            locks.enter_context(sp.stopped_run(manifest["run_dir"]))
        if sp.sha256(source_file) != policy["original_source_sha256"]:
            raise ValueError("Original OOM source hash changed")
        file_hashes[str(source_file)] = policy["original_source_sha256"]
        for row, reference in records(plan["retained"]["path"]):
            accepted[sp.row_index(row, indices)] = outcome(row, reference, "completed_attempt")
        original = [(row, ref) for row, ref in records(source_file) if row.get("instance_id") == policy["instance_id"]]
        if len(original) != 1 or oom_index in accepted:
            raise ValueError("Original OOM outcome missing or duplicated")
        row, reference = original[0]
        if (sp.row_index(row, indices) != oom_index or row.get("resolved") is not False
                or row.get("evaluation_completed") is not True or row.get("opencode_finished") is not False
                or row.get("opencode_export_found") is not False
                or any(row.get(key) != value for key, value in policy["original_outcome"].items())):
            raise ValueError("Original OOM outcome differs from conservative policy")
        accepted[oom_index] = outcome(row, reference, "conservative_original_oom_unresolved",
                                      [{"path": str(policy_path), "sha256": sp.sha256(policy_path)}])
        for manifest in manifests:
            run = sp.contained(manifest["run_dir"])
            settings = sp.read_json(run / "settings.json")
            if (sp.protocol_settings(settings) != plan["protocol_settings"]
                    or settings["dataset"] != manifest["dataset"] or settings["limit"] != manifest["dataset"]["count"]
                    or settings.get("shard_manifest") != manifest["_identity"]):
                raise ValueError("Shard protocol or assignment changed")
            for field, filename in (("runner_sha256", "run_eval.py"), ("runtime_helper_sha256", "atp_runtime.py"),
                                    ("gpu_monitor_sha256", "gpu_monitor.py")):
                if settings[field] != sp.sha256(sp.APP / "scripts" / filename):
                    raise ValueError("Pinned runtime file changed: " + filename)
            runner_hashes.add(settings["runner_sha256"])
            file_hashes[str(run / "settings.json")] = sp.sha256(run / "settings.json")
            status = sp.read_json(run / "status.json")
            cleanup.append({"slot": manifest["slot"], "status": status["status"],
                            "sandbox_cleanup_passed": status.get("sandbox_cleanup_passed")})
            allowed, seen = set(manifest["expected_task_indices"]), set()
            output = run / "rollouts.jsonl"
            if output.exists():
                file_hashes[str(output)] = sp.sha256(output)
            for row, reference in records(output) if output.exists() else []:
                index = sp.row_index(row, indices)
                if index not in allowed or index in seen or (index in accepted and index != oom_index):
                    raise ValueError("Duplicate or out-of-shard outcome")
                seen.add(index)
                if index == oom_index:
                    diagnostics.append({"instance_id": row["instance_id"], "resolved": row.get("resolved"),
                                        "evaluation_completed": row.get("evaluation_completed"),
                                        "opencode_finished": row.get("opencode_finished"),
                                        "opencode_export_found": row.get("opencode_export_found"),
                                        "invalid_reasons": sp.invalid_reasons(row),
                                        "source": reference, "excluded_from_score": True})
                    continue
                if index in audits:
                    path, audit, interruption = audits[index]
                    evidence = budget_evidence(path, audit, row, reference, manifest["slot"], interruption)
                    if interruption:
                        conservative_interruptions[index] = outcome(row, reference, "audited_unresolved_interruption_cause_unknown", evidence)
                        unadjudicated.append({"instance_id": row["instance_id"], "task_index": index,
                                              "reasons": ["interruption_cause_unknown"], "source": reference})
                    else:
                        accepted[index] = outcome(row, reference, "adjudicated_agent_budget_exhaustion", evidence)
                    used_audits.add(index)
                elif sp.invalid_reasons(row):
                    unadjudicated.append({"instance_id": row["instance_id"], "task_index": index,
                                          "reasons": sp.invalid_reasons(row), "source": reference})
                else:
                    accepted[index] = outcome(row, reference, "completed_attempt")
            failures = run / "rollouts_failures.jsonl"
            if failures.exists():
                file_hashes[str(failures)] = sp.sha256(failures)
                for row, reference in records(failures):
                    if sp.row_index(row, indices) not in allowed:
                        raise ValueError("Sidecar failure is outside its shard")
                    sidecars.append({"instance_id": row["instance_id"], "failure_class": row.get("_ng_failure_class"),
                                     "terminal": row.get("_ng_failure_terminal", False), "source": reference})
        if used_audits != set(audits) or len(runner_hashes) != 1:
            raise ValueError("Unused budget audit or incompatible shard runners")
        missing = [master[i]["instance_id"] for i in range(500) if i not in accepted]
        conservative_missing = [master[i]["instance_id"] for i in range(500)
                                if i not in accepted and i not in conservative_interruptions]
        complete = not missing
        resolved = sum(row["resolved"] for row in accepted.values())
        summary = {"schema": 1, "score_policy": "conservative_original_oom_and_adjudicated_budgets",
                   "plan": {"path": str(plan_path), "sha256": sp.sha256(plan_path)},
                   "reporter_sha256": sp.sha256(__file__), "expected_tasks": 500, "scored_tasks": len(accepted),
                   "complete": complete, "resolved_count": resolved, "resolved_mean": resolved / 500 if complete else None,
                   "conservative_known_outcomes_count": len(accepted) + len(conservative_interruptions),
                   "conservative_resolved_count": resolved, "conservative_complete": not conservative_missing,
                   "conservative_resolved_mean": resolved / 500 if not conservative_missing else None,
                   "conservative_missing_instance_ids": conservative_missing,
                   "conservative_unresolved_interruptions": list(conservative_interruptions.values()),
                   "classification_counts": dict(Counter(row["classification"] for row in accepted.values())),
                   "missing_instance_ids": missing, "unadjudicated_results": unadjudicated,
                   "diagnostic_retries": diagnostics, "missing_diagnostic_instance_ids": [] if diagnostics else [policy["instance_id"]],
                   "sidecar_failures": sidecars, "worker_cleanup": cleanup,
                   "all_worker_cleanup_passed": all(item["sandbox_cleanup_passed"] is True for item in cleanup),
                   "source_files_sha256": file_hashes}
        if out_dir is not None:
            destination = artifact_path(out_dir)
            if not destination.is_relative_to(sp.APP / "runs") or destination == sp.APP / "runs":
                raise ValueError("Reports must be a new directory under APP/runs")
            destination.mkdir(exist_ok=False)
            sp.write_rows(destination / "scored-outcomes.jsonl", [accepted[i] for i in sorted(accepted)])
            summary["outcomes"] = {"path": str(destination / "scored-outcomes.jsonl"),
                                   "sha256": sp.sha256(destination / "scored-outcomes.jsonl")}
            sp.write_json(destination / "summary.json", summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--budget-audit", action="append", default=[])
    parser.add_argument("--unresolved-interruption-audit", action="append", default=[])
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    result = score(args.plan, args.policy, args.budget_audit, args.out_dir, args.unresolved_interruption_audit)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
