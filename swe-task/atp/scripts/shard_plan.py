#!/usr/bin/env python3
"""Plan disjoint SWE evaluation shards and merge only complete, compatible results."""

import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import json
from pathlib import Path


APP = Path(__file__).resolve().parents[1]
MASTER_COUNT = 500
STOPPED = {"completed", "failed", "interrupted", "incomplete", "smoke_failed", "completed_cleanup_failed"}
ROUTING_SETTINGS = {"dataset", "limit", "runner_sha256", "shard_manifest"}
PROTOCOL_FIELDS = {
    "preparation_id", "benchmark", "model", "image", "serving_overlay", "gym_commit",
    "runtime_helper_sha256", "gpu_monitor_sha256", "smoke", "concurrency", "num_repeats",
    "max_output_tokens", "temperature", "top_p", "context", "tensor_parallel_size", "opencode_version",
}


def contained(path):
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(APP):
        raise ValueError("Artifact path must remain inside this application")
    return path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(contained(path).read_text())


def read_rows(path):
    with contained(path).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("JSONL records must be objects")
    return rows


def write_json(path, value):
    path = contained(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_rows(path, rows):
    path = contained(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(path)


def protocol_settings(settings):
    if not PROTOCOL_FIELDS.issubset(settings):
        raise ValueError("Evaluation settings lack required protocol fields")
    return {key: value for key, value in settings.items() if key not in ROUTING_SETTINGS}


def master_rows(prepared):
    dataset = prepared["dataset"]
    if prepared.get("status") != "ready" or dataset["count"] != MASTER_COUNT:
        raise ValueError("A ready 500-task preparation is required")
    if sha256(contained(dataset["path"])) != dataset["sha256"]:
        raise ValueError("Master dataset hash changed")
    rows = read_rows(dataset["path"])
    ids = [row.get("instance_id") for row in rows]
    if len(rows) != MASTER_COUNT or any(not isinstance(value, str) for value in ids) or len(set(ids)) != MASTER_COUNT:
        raise ValueError("Master dataset must contain 500 distinct instance IDs")
    return rows


def row_index(row, indices):
    instance = row.get("instance_id")
    index = row.get("_ng_task_index")
    repeat = row.get("_ng_rollout_index")
    if not isinstance(instance, str) or instance not in indices:
        raise ValueError("Result contains an out-of-plan instance ID")
    if type(index) is not int or index != indices[instance] or type(repeat) is not int or repeat != 0:
        raise ValueError("Result has an incorrect global task index or repeat")
    return index


def invalid_reasons(row):
    reasons = []
    if row.get("_ng_failure_class") is not None:
        reasons.append("infrastructure_failure")
    for field in ("evaluation_completed", "opencode_finished", "opencode_export_found"):
        if row.get(field) is not True:
            reasons.append(field + "_not_true")
    if type(row.get("resolved")) is not bool:
        reasons.append("resolved_not_boolean")
    return reasons


@contextmanager
def stopped_run(run_dir):
    run_dir = contained(run_dir)
    with contained(run_dir / ".lock").open("r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Run still has an active writer: " + str(run_dir)) from error
        if read_json(run_dir / "status.json").get("status") not in STOPPED:
            raise ValueError("Run must have a terminal status before snapshotting: " + str(run_dir))
        yield run_dir


def create_plan(source_run, out_dir, n_shards=4):
    source_run, out_dir = contained(source_run), contained(out_dir)
    if type(n_shards) is not int or n_shards != 4:
        raise ValueError("This approved plan requires exactly four shards")
    if not out_dir.is_relative_to(APP / "runs") or out_dir == APP / "runs" or out_dir.is_relative_to(source_run):
        raise ValueError("Plan directory must be a new APP/runs directory outside the source run")
    prepared = read_json(APP / "runtime/preparation.json")
    master = master_rows(prepared)
    indices = {row["instance_id"]: index for index, row in enumerate(master)}
    with stopped_run(source_run):
        settings = read_json(source_run / "settings.json")
        protocol = protocol_settings(settings)
        if settings["dataset"] != prepared["dataset"] or settings["limit"] != MASTER_COUNT:
            raise ValueError("Source run must use the exact complete prepared benchmark")
        if any(settings[key] != prepared[key] for key in ("preparation_id", "model", "image", "gym_commit")):
            raise ValueError("Source run does not match the current preparation")
        if settings["smoke"] is not False or settings["concurrency"] != 16 or settings["num_repeats"] != 1:
            raise ValueError("Source protocol must be the full run with concurrency 16 and one repeat")
        source_path = source_run / "rollouts.jsonl"
        retained, excluded, seen = [], [], set()
        for row in read_rows(source_path):
            index = row_index(row, indices)
            if index in seen:
                raise ValueError("Source results contain duplicate task IDs")
            seen.add(index)
            reasons = invalid_reasons(row)
            if reasons:
                excluded.append({"instance_id": row["instance_id"], "task_index": index,
                                 "reasons": reasons, "candidate_patch_available": bool(row.get("model_patch")),
                                 "action": "retry_original_task"})
            else:
                retained.append(row)
        retained.sort(key=lambda row: row["_ng_task_index"])
        kept = {row["_ng_task_index"] for row in retained}
        remaining = [index for index in range(MASTER_COUNT) if index not in kept]
        out_dir.mkdir(parents=True, exist_ok=False)
        (out_dir / "datasets").mkdir()
        (out_dir / "manifests").mkdir()
        retained_path, excluded_path = out_dir / "retained.jsonl", out_dir / "excluded.json"
        write_rows(retained_path, retained)
        write_json(excluded_path, excluded)
        plan_path = out_dir / "plan.json"
        plan = {"schema": 1, "plan_path": str(plan_path), "preparation_id": prepared["preparation_id"],
                "master_dataset": prepared["dataset"], "protocol_settings": protocol,
                "source": {"run_dir": str(source_run), "rollouts_sha256": sha256(source_path),
                           "row_count": len(seen), "settings_sha256": sha256(source_run / "settings.json"),
                           "settings": settings},
                "retained": {"path": str(retained_path), "sha256": sha256(retained_path), "count": len(retained)},
                "excluded": {"path": str(excluded_path), "sha256": sha256(excluded_path), "count": len(excluded)},
                "shards": []}
        for slot in range(n_shards):
            selected = remaining[slot::n_shards]
            dataset_path = out_dir / "datasets" / f"shard-{slot}.jsonl"
            write_rows(dataset_path, [dict(master[index], _ng_task_index=index) for index in selected])
            manifest_path = out_dir / "manifests" / f"shard-{slot}.json"
            run_dir = out_dir / f"eval-shard-{slot}"
            manifest = {"schema": 1, "slot": slot, "plan_path": str(plan_path),
                        "dataset": {"path": str(dataset_path), "sha256": sha256(dataset_path), "count": len(selected)},
                        "expected_task_indices": selected, "preparation_id": prepared["preparation_id"],
                        "master_dataset_sha256": prepared["dataset"]["sha256"], "run_dir": str(run_dir)}
            write_json(manifest_path, manifest)
            plan["shards"].append({"slot": slot, "path": str(manifest_path), "sha256": sha256(manifest_path),
                                   "count": len(selected), "run_dir": str(run_dir)})
        write_json(plan_path, plan)
    return plan


def validate_plan(plan_path, prepared):
    plan_path = contained(plan_path)
    plan = read_json(plan_path)
    master = master_rows(prepared)
    if (plan.get("schema") != 1 or plan.get("plan_path") != str(plan_path)
            or plan.get("preparation_id") != prepared["preparation_id"] or plan.get("master_dataset") != prepared["dataset"]):
        raise ValueError("Plan does not match this preparation")
    protocol = protocol_settings(plan["source"]["settings"])
    if protocol != plan["protocol_settings"]:
        raise ValueError("Plan protocol does not match its frozen source settings")
    for key in ("preparation_id", "model", "image", "gym_commit", "serving_overlay"):
        if key in prepared and protocol[key] != prepared[key]:
            raise ValueError("Shard preparation differs from source protocol: " + key)
    indices = {row["instance_id"]: index for index, row in enumerate(master)}
    for artifact in (plan["retained"], plan["excluded"]):
        if sha256(contained(artifact["path"])) != artifact["sha256"]:
            raise ValueError("Frozen plan artifact changed")
    retained = read_rows(plan["retained"]["path"])
    assigned = set()
    for row in retained:
        index = row_index(row, indices)
        if index in assigned or invalid_reasons(row):
            raise ValueError("Retained results are duplicate or invalid")
        assigned.add(index)
    if len(retained) != plan["retained"]["count"]:
        raise ValueError("Retained result count changed")
    manifests, run_dirs = [], set()
    if len(plan["shards"]) != 4:
        raise ValueError("Plan must contain exactly four shard manifests")
    for slot, entry in enumerate(plan["shards"]):
        path = contained(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError("Shard manifest hash changed")
        manifest = read_json(path)
        run_dir = contained(manifest["run_dir"])
        dataset = manifest["dataset"]
        if (manifest.get("schema") != 1 or manifest["slot"] != slot or entry["slot"] != slot
                or contained(manifest["plan_path"]) != plan_path or manifest["preparation_id"] != prepared["preparation_id"]
                or manifest["master_dataset_sha256"] != prepared["dataset"]["sha256"]
                or entry["run_dir"] != str(run_dir) or not run_dir.is_relative_to(APP / "runs")
                or run_dir == APP / "runs" or run_dir in run_dirs):
            raise ValueError("Invalid shard identity or run directory")
        run_dirs.add(run_dir)
        if sha256(contained(dataset["path"])) != dataset["sha256"]:
            raise ValueError("Shard dataset hash changed")
        selected = manifest["expected_task_indices"]
        if (len(selected) != dataset["count"] or dataset["count"] != entry["count"]
                or any(type(index) is not int or not 0 <= index < MASTER_COUNT for index in selected)
                or len(set(selected)) != len(selected) or assigned.intersection(selected)):
            raise ValueError("Shard assignments overlap or have invalid counts/indices")
        if read_rows(dataset["path"]) != [dict(master[index], _ng_task_index=index) for index in selected]:
            raise ValueError("Shard rows differ from canonical tasks")
        assigned.update(selected)
        manifest["_identity"] = {"path": str(path), "sha256": entry["sha256"],
                                 "plan_sha256": sha256(plan_path), "slot": slot,
                                 "validator_sha256": sha256(__file__)}
        manifests.append(manifest)
    if assigned != set(range(MASTER_COUNT)):
        raise ValueError("Retained results and shard assignments do not cover exactly 500 tasks")
    return plan, master, retained, manifests


def load_shard(manifest_path, prepared):
    manifest_path = contained(manifest_path)
    manifest = read_json(manifest_path)
    _, _, _, manifests = validate_plan(manifest["plan_path"], prepared)
    for candidate in manifests:
        if candidate["_identity"]["path"] == str(manifest_path):
            return candidate
    raise ValueError("Manifest is not a member of its plan")


def merge_plan(plan_path):
    plan_path = contained(plan_path)
    prepared = read_json(APP / "runtime/preparation.json")
    plan, master, retained, manifests = validate_plan(plan_path, prepared)
    indices = {row["instance_id"]: index for index, row in enumerate(master)}
    results = {row["_ng_task_index"]: row for row in retained}
    invalid, pending, runner_hashes = [], [], set()
    with ExitStack() as stack:
        for manifest in manifests:
            if not manifest["dataset"]["count"]:
                continue  # An empty assignment needs no GPU job.
            run_dir = contained(manifest["run_dir"])
            if not (run_dir / "status.json").exists() or read_json(run_dir / "status.json").get("status") not in STOPPED:
                pending.append(manifest["slot"])
                continue
            stack.enter_context(stopped_run(run_dir))
            settings = read_json(run_dir / "settings.json")
            if (protocol_settings(settings) != plan["protocol_settings"] or settings["dataset"] != manifest["dataset"]
                    or settings["limit"] != manifest["dataset"]["count"] or settings.get("shard_manifest") != manifest["_identity"]):
                raise ValueError("Shard settings changed evaluation protocol or assignment")
            if not isinstance(settings.get("runner_sha256"), str) or not settings["runner_sha256"]:
                raise ValueError("Shard runner hash is missing")
            runner_hashes.add(settings["runner_sha256"])
            allowed, seen = set(manifest["expected_task_indices"]), set()
            output = run_dir / "rollouts.jsonl"
            for row in read_rows(output) if output.exists() else []:
                index = row_index(row, indices)
                if index not in allowed or index in seen or index in results:
                    raise ValueError("Duplicate or out-of-shard result")
                seen.add(index)
                if invalid_reasons(row):
                    invalid.append({"instance_id": row["instance_id"], "reasons": invalid_reasons(row)})
                else:
                    results[index] = row
        if len(runner_hashes) > 1:
            raise ValueError("Shards used different runner versions")
        missing = [master[index]["instance_id"] for index in range(MASTER_COUNT) if index not in results]
        complete = not (missing or invalid or pending)
        resolved = sum(row["resolved"] for row in results.values())
        summary = {"schema": 1, "plan_path": str(plan_path), "plan_sha256": sha256(plan_path),
                   "complete": complete, "expected_tasks": MASTER_COUNT, "valid_tasks": len(results),
                   "retained_tasks": len(retained), "pending_slots": pending, "missing_instance_ids": missing,
                   "invalid_results": invalid, "resolved_count": resolved,
                   "resolved_mean": resolved / MASTER_COUNT if complete else None}
        if complete:
            merged = plan_path.parent / "merged.jsonl"
            write_rows(merged, [results[index] for index in range(MASTER_COUNT)])
            summary["merged"] = {"path": str(merged), "sha256": sha256(merged), "count": MASTER_COUNT}
        write_json(plan_path.parent / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--source-run", required=True)
    create.add_argument("--out-dir", required=True)
    create.add_argument("--shards", type=int, default=4)
    merge = commands.add_parser("merge")
    merge.add_argument("--plan", required=True)
    args = parser.parse_args()
    if args.command == "create":
        result = create_plan(args.source_run, args.out_dir, args.shards)
    else:
        result = merge_plan(args.plan)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if args.command == "create" or result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
