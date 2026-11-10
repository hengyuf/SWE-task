#!/usr/bin/env python3
"""Score selected DeepSWE models with one explicitly approved, evidence-pinned timeout failure."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path


MODELS = {"super": "nvidia/nvidia/nemotron-3.5-super-vl", "kimi": "nvidia/moonshotai/kimi-k3",
          "glm": "nvidia/zai-org/glm-5.3"}
POLICY = "explicit_kimi_57_2_timeout_failure"
TASK, REPEAT, ATTEMPT = "_ng_task_index", "_ng_rollout_index", "_ng_attempt_index"
NATIVE_FIELDS = ("evaluation_completed", "reward", "verifier_exit_code", "verifier_error",
                 "opencode_finished", "opencode_export_found", "model_patch_sha256", "model_patch_bytes")
GRADE_FIELDS = {"evaluation_completed", "reward", "verifier_exit_code", "verifier_error",
                "model_patch_sha256", "model_patch_bytes"}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    with Path(path).open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return value


def exact_path(value, expected):
    return isinstance(value, str) and Path(value).is_absolute() and Path(value).resolve() == expected.resolve()


def same_fields(actual, expected):
    return all(key in actual and type(actual[key]) is type(value) and actual[key] == value for key, value in expected.items())


def validate_approval(campaign, manifest, path):
    approval = read_json(path)
    attempt = campaign / "models/kimi/attempts/57-2/attempt-2"
    expected_identity = {"model_alias": "kimi", "model_id": MODELS["kimi"], "task_id": "koota-query-predicates",
                         "task_index": 57, "repeat_index": 2, "attempt_index": 2}
    identity = approval.get("identity", {})
    if (type(approval.get("schema")) is not int or approval["schema"] != 1 or approval.get("policy") != POLICY or approval.get("approved") is not True
            or type(approval.get("score")) is not int or approval["score"] != 0
            or not exact_path(approval.get("campaign"), campaign)
            or approval.get("manifest_sha256") != sha256(campaign / "manifest.json")
            or not isinstance(identity, dict) or any(identity.get(k) != v for k, v in expected_identity.items())
            or not exact_path(identity.get("attempt_dir"), attempt)
            or manifest["tasks"][57]["task_id"] != identity["task_id"]):
        raise ValueError("Approval policy, campaign, manifest or exact target identity mismatch")
    artifacts = approval.get("raw_artifacts", {})
    for key, filename in (("terminal", "terminal.json"), ("native_result", "native-result.json"), ("model_patch", "model.patch")):
        reference = artifacts.get(key, {})
        if (not exact_path(reference.get("path"), attempt / filename)
                or reference.get("sha256") != sha256(attempt / filename)):
            raise ValueError("Approved raw artifact path or hash changed: " + key)
    evidence = approval.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Explicit hashed diagnostic evidence is required")
    for reference in evidence:
        if (not isinstance(reference, dict) or not isinstance(reference.get("path"), str)
                or not Path(reference["path"]).is_absolute() or reference.get("sha256") != sha256(reference["path"])):
            raise ValueError("Approved diagnostic evidence path or hash changed")
    absent = approval.get("expected_absent_artifacts")
    reward = attempt / "grading/attempt-0/reward.json"
    if (not isinstance(absent, list) or not absent or not any(exact_path(p, reward) for p in absent)
            or any(not isinstance(p, str) or not Path(p).is_absolute()
                   or not Path(p).resolve().is_relative_to(attempt.resolve()) or Path(p).exists() for p in absent)):
        raise ValueError("Approved missing native reward evidence changed")
    expected = approval.get("expected_native_fields", {})
    if (not isinstance(expected, dict) or set(expected) - set(NATIVE_FIELDS)
            or expected.get("evaluation_completed") is not False or type(expected.get("reward")) not in (int, float)
            or expected["reward"] != 0 or not isinstance(expected.get("verifier_error"), str) or not expected["verifier_error"]):
        raise ValueError("Approval must preserve an explicitly incomplete native evaluation and its error")
    terminal, native = read_json(attempt / "terminal.json"), read_json(attempt / "native-result.json")
    original = terminal | native | {key: value for key, value in terminal.items() if key in GRADE_FIELDS}
    if not same_fields(original, expected):
        raise ValueError("Approved native fields differ from original artifacts")
    return approval


def row_identity(row, campaign, tasks, manifest_models):
    model, task, repeat, attempt = row.get("model_alias"), row.get(TASK), row.get(REPEAT), row.get(ATTEMPT)
    if (model not in manifest_models or type(task) is not int or not 0 <= task < 113
            or type(repeat) is not int or not 0 <= repeat < 16 or row.get("task_id") != tasks[task]["task_id"]
            or ("model_id" in row and row["model_id"] != MODELS[model])):
        raise ValueError("Rollout model, task or repeat identity mismatch/out of range")
    directory = row.get("_atp_attempt_dir")
    if attempt is None:
        if directory is not None:
            raise ValueError("Attempt index and directory disagree")
    elif (type(attempt) is not int or attempt < 0 or not exact_path(directory,
          campaign / "models" / model / "attempts" / f"{task}-{repeat}" / f"attempt-{attempt}")):
        raise ValueError("Rollout campaign or attempt identity mismatch")
    artifacts = row.get("_atp_artifacts")
    if (not isinstance(artifacts, dict) or type(artifacts.get("artifact_complete")) is not bool
            or not isinstance(artifacts.get("missing"), list) or any(not isinstance(item, str) for item in artifacts["missing"])
            or artifacts["artifact_complete"] != (not artifacts["missing"])
            or (attempt is not None and not exact_path(artifacts.get("attempt_dir"), Path(directory)))):
        raise ValueError("Canonical exporter artifact proof is missing or inconsistent")
    return model, task, repeat


def adjudicate(row, approval):
    identity = approval["identity"]
    if (row.get(ATTEMPT) != identity["attempt_index"]
            or not exact_path(row.get("_atp_attempt_dir"), Path(identity["attempt_dir"]))
            or row.get("_atp_status") == "completed"
            or not same_fields(row, approval["expected_native_fields"])):
        raise ValueError("Selected rollout differs from approved attempt/native timeout")
    patch = row.get("model_patch")
    if not isinstance(patch, str) or hashlib.sha256(patch.encode()).hexdigest() != approval["raw_artifacts"]["model_patch"]["sha256"]:
        raise ValueError("Selected rollout differs from the approved exact candidate patch")
    return 0


def validate_selection(campaign, models, path):
    raw = path.read_bytes()
    record = json.loads(raw)
    if (not isinstance(record, dict) or type(record.get("version")) is not int or record["version"] != 1
            or not exact_path(record.get("campaign"), campaign) or set(models) != {"super", "kimi"}
            or record.get("models") != ["super", "kimi"] or type(record.get("requested_repeats")) is not int
            or record["requested_repeats"] != 5 or type(record.get("expected_tasks")) is not int
            or record["expected_tasks"] != 113 or record.get("selection_uses_no_rewards") is not True
            or record.get("selection_policy") not in {"any_completed", "fixed_0_4"}
            or not isinstance(record.get("source"), str) or not Path(record["source"]).is_absolute()):
        raise ValueError("Five-slot selection contract or score-blind policy mismatch")
    source = Path(record["source"]).resolve()
    allowance_path = campaign / "recovery-allowances" / (source.name + ".json")
    if (source.parent != campaign / "source-revisions"
            or not exact_path(record.get("recovery_allowance"), allowance_path)):
        raise ValueError("Selection source or recovery allowance path mismatch")
    allowance = read_json(allowance_path)
    source_digest = sha256(source / "source-manifest.json")
    if (allowance.get("id") != source.name or not exact_path(allowance.get("campaign"), campaign)
            or not exact_path(allowance.get("source"), source)
            or allowance.get("source_manifest_sha256") != source_digest):
        raise ValueError("Selection recovery allowance/source binding mismatch")
    rows = record.get("slots")
    if not isinstance(rows, list) or len(rows) != 1130:
        raise ValueError("Selection must contain exactly 1130 original slots")
    selected, counts = set(), Counter()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"model", "task_index", "repeat_index"}:
            raise ValueError("Invalid selected slot identity")
        model, task, repeat = row["model"], row["task_index"], row["repeat_index"]
        if (model not in {"super", "kimi"} or type(task) is not int or not 0 <= task < 113
                or type(repeat) is not int or not 0 <= repeat < 16
                or (record["selection_policy"] == "fixed_0_4" and repeat >= 5)):
            raise ValueError("Selected slot is out of scope/range or violates its fixed-repeat policy")
        key = model, task, repeat
        if key in selected:
            raise ValueError("Duplicate selected original slot")
        selected.add(key)
        counts[model, task] += 1
    if any(counts[model, task] != 5 for model in ("super", "kimi") for task in range(113)):
        raise ValueError("Selection requires exactly five unique original repeats per task/model")
    reference = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                 "record": {key: value for key, value in record.items() if key != "slots"},
                 "source_manifest_sha256": source_digest, "recovery_allowance_sha256": sha256(allowance_path)}
    return selected, reference


def metrics(groups, repeats=16):
    completed = [values for values in groups if len(values) == repeats]
    return {"rollouts": sum(len(values) for values in groups),
            "passed_rollouts": sum(sum(values) for values in groups),
            "fully_completed_tasks": len(completed), "task_denominator": len(completed),
            "pass_at_1": sum(sum(values) for values in completed) / (repeats * len(completed)) if completed else None,
            f"pass_at_{repeats}": sum(any(values) for values in completed) / len(completed) if completed else None,
            "complete": len(completed) == 113}


def score(campaign, rollouts, approval_path, models, out_dir=None, selection_path=None):
    campaign, rollouts = (Path(p).resolve() for p in (campaign, rollouts))
    approval_path = Path(approval_path).resolve() if approval_path is not None else None
    manifest = read_json(campaign / "manifest.json")
    settings, tasks = manifest["settings"], manifest["tasks"]
    if (not models or len(set(models)) != len(models) or "kimi" not in models
            or set(models) - set(settings["models"]) or set(settings["models"]) - set(MODELS)
            or any(value.get("id") != MODELS[key] for key, value in settings["models"].items())
            or settings.get("benchmark") != "deepswe_v1_1" or settings.get("expected_tasks") != 113
            or settings.get("repeats") != 16 or len(tasks) != 113
            or any(type(task.get(TASK)) is not int or task[TASK] != index or not isinstance(task.get("task_id"), str) for index, task in enumerate(tasks))
            or len({task["task_id"] for task in tasks}) != 113):
        raise ValueError("Explicit selected models or pinned DeepSWE 113-task/16-repeat contract mismatch")
    selected, selection_ref = None, None
    if selection_path is not None:
        selected, selection_ref = validate_selection(campaign, models, Path(selection_path).resolve())
    repeats = 5 if selected is not None else 16
    approval_needed = selected is None or ("kimi", 57, 2) in selected
    approval, approval_ref = None, None
    if approval_needed:
        if approval_path is None:
            raise ValueError("Selected adjudication requires its explicit approval")
        if selection_ref is not None:
            reference = selection_ref["record"].get("adjudication", {})
            if (not isinstance(reference, dict) or not exact_path(reference.get("path"), approval_path)
                    or reference.get("sha256") != sha256(approval_path)
                    or reference.get("validator_sha256") != sha256(__file__)):
                raise ValueError("Selection adjudication approval or validator hash mismatch")
        approval = validate_approval(campaign, manifest, approval_path)
        approval_ref = {"path": str(approval_path), "sha256": sha256(approval_path)}
    native = {model: [[] for _ in tasks] for model in models}
    scored = {model: [[] for _ in tasks] for model in models}
    classifications = {model: Counter() for model in models}
    seen, used, excluded, excluded_selection, file_hash = set(), False, 0, 0, hashlib.sha256()
    output = None
    if out_dir is not None:
        out_dir = Path(out_dir).resolve()
        if out_dir.is_relative_to(campaign):
            raise ValueError("Scored reports must be separate from the raw campaign")
        out_dir.mkdir(parents=True, exist_ok=False)
        output = (out_dir / "scored-outcomes.jsonl.tmp").open("x")
    try:
        with rollouts.open("rb") as stream:
            before = os.fstat(stream.fileno())
            for line, raw in enumerate(stream, 1):
                file_hash.update(raw)
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError("Rollout must be a JSON object")
                key = row_identity(row, campaign, tasks, settings["models"])
                if key in seen:
                    raise ValueError("Duplicate model/task/repeat rollout")
                seen.add(key)
                model, task, repeat = key
                if model not in models:
                    excluded += 1
                    continue
                if selected is not None and key not in selected:
                    excluded_selection += 1
                    continue
                value, classification = None, "unscored"
                if key == ("kimi", 57, 2):
                    value, classification, used = adjudicate(row, approval), "adjudicated_timeout_failure", True
                elif row.get("_atp_status") == "completed":
                    if (row.get("evaluation_completed") is not True or type(row.get("reward")) not in (int, float)
                            or row["reward"] not in (0, 1) or row.get(ATTEMPT) is None):
                        raise ValueError("Completed native rollout lacks an explicit binary native grade")
                    value, classification = int(row["reward"]), "native_completed"
                    native[model][task].append(value)
                    if not row["_atp_artifacts"]["artifact_complete"]:
                        value, classification = None, "unscored_artifact_gap"
                if value is not None:
                    scored[model][task].append(value)
                classifications[model][classification] += 1
                record = {"model_alias": model, "model_id": MODELS[model], "task_id": row["task_id"],
                          TASK: task, REPEAT: repeat, ATTEMPT: row.get(ATTEMPT), "attempt_dir": row.get("_atp_attempt_dir"),
                          "classification": classification, "score": value,
                          "native": {key: row[key] for key in NATIVE_FIELDS if key in row},
                          "native_fields_missing": [key for key in NATIVE_FIELDS if key not in row],
                          "native_status": row.get("_atp_status"),
                          "native_artifacts": row["_atp_artifacts"],
                          "source": {"path": str(rollouts), "line": line, "row_sha256": hashlib.sha256(raw.strip()).hexdigest()}}
                if classification == "adjudicated_timeout_failure":
                    record.update(approval=approval_ref, expected_absent_artifacts=approval["expected_absent_artifacts"],
                                  raw_artifacts=approval["raw_artifacts"], diagnostic_evidence=approval["evidence"])
                if output is not None:
                    output.write(json.dumps(record, allow_nan=False) + "\n")
            signature = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if signature(before) != signature(os.fstat(stream.fileno())) or signature(before) != signature(rollouts.stat()):
                raise ValueError("Canonical rollout export changed during scoring")
        if approval_needed and not used:
            raise ValueError("Explicit approval target is absent/unused")
        summary = {"schema": 1, "policy": POLICY, "campaign": str(campaign), "selected_models": list(models),
                   "expected_tasks": 113, "repeats": repeats, "expected_rollouts": 113 * repeats * len(models),
                   "approval": approval_ref, "manifest_sha256": sha256(campaign / "manifest.json"),
                   "source": {"path": str(rollouts), "sha256": file_hash.hexdigest()},
                   "reporter_sha256": sha256(__file__), "excluded_model_rows": excluded, "models": {}}
        if selection_ref is not None:
            if sha256(selection_ref["path"]) != selection_ref["sha256"]:
                raise ValueError("Five-slot selection changed during scoring")
            summary.update(selection=selection_ref, selection_uses_no_rewards=True,
                           excluded_unselected_slot_rows=excluded_selection, original_manifest_repeats=16)
        summary["native_validity_basis"] = "Previously validated canonical DeepSWE exporter classification, explicit native binary grade, and exported artifact proof; no raw trace reclassification."
        summary["artifact_audit"] = "Final artifact audit is separate. Native grades with exported artifact gaps remain visible in native metrics but are unscored pending complete artifact proof."
        for model in models:
            summary["models"][model] = {"native": metrics(native[model], repeats), "scored": metrics(scored[model], repeats),
                "classification_counts": dict(classifications[model]),
                "missing_rollouts": 113 * repeats - sum(classifications[model].values())}
        summary["complete"] = all(item["scored"]["complete"] for item in summary["models"].values())
        if output is not None:
            output.close()
            output = None
            temporary = out_dir / "scored-outcomes.jsonl.tmp"
            temporary.rename(out_dir / "scored-outcomes.jsonl")
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        return summary
    finally:
        if output is not None:
            output.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--approval")
    parser.add_argument("--models", nargs="+", required=True, choices=sorted(MODELS))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--selection", help="Optional immutable five-slot selection; original repeat indices are preserved")
    args = parser.parse_args()
    summary = score(args.campaign, args.rollouts, args.approval, args.models, args.out_dir, args.selection)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
