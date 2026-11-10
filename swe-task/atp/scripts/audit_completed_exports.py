#!/usr/bin/env python3
"""Incremental, evidence-only audit of completed OpenCode root-session exports."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

APP = Path(__file__).resolve().parents[1]


def owned(path):
    path = Path(path).resolve()
    path.relative_to(APP)
    return path


def signature(path):
    st = path.stat()
    return dict(size=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino)


def inspect_export(path, previous):
    stat = signature(path)
    if previous and previous.get("export_stat") == stat:
        return dict(previous), 0
    raw = path.read_bytes()
    if signature(path) != stat:
        raise RuntimeError("export changed during audit")
    messages = json.loads(raw)["messages"]
    assistants = [m["info"] for m in messages if m["info"].get("role") == "assistant"]
    errors = [m["error"] for m in assistants if m.get("error")]
    states, tool_errors = Counter(), Counter()
    for message in messages:
        for part in message.get("parts", []):
            if part.get("type") == "tool":
                state = part.get("state", {}).get("status", "missing")
                states[state] += 1
                if state == "error":
                    tool_errors[part.get("tool", "unknown")] += 1
    last = assistants[-1] if assistants else {}
    return dict(export_sha256=hashlib.sha256(raw).hexdigest(), export_bytes=len(raw),
                export_stat=stat, assistant_count=len(assistants), assistant_error_count=len(errors),
                assistant_error_types=dict(Counter(e.get("name", "unknown") if isinstance(e, dict)
                                                    else type(e).__name__ for e in errors)),
                final_finish=last.get("finish"), final_completed=bool(last.get("time", {}).get("completed")),
                final_input_tokens=last.get("tokens", {}).get("input"),
                final_output_tokens=last.get("tokens", {}).get("output"),
                finish_counts=dict(Counter(m.get("finish", "missing") for m in assistants)),
                tool_status_counts=dict(states), tool_error_names=dict(tool_errors),
                unfinished_tool_count=sum(n for state, n in states.items() if state not in ("completed", "error"))), len(raw)


def audit(plan_dir):
    started = time.monotonic()
    plan_dir = owned(plan_dir)
    plan_dir.relative_to(APP / "runs")
    output = owned(plan_dir / "completed-export-audit.json")
    plan = json.loads((plan_dir / "plan.json").read_text())
    old = json.loads(output.read_text()) if output.exists() else {}
    cache = {r["export_path"]: r for r in old.get("records", []) if "export_path" in r}
    records, skipped, partial, malformed, workers = [], [], [], [], []
    bytes_read = reused = 0
    sources = [(None, owned(plan["retained"]["path"]))]
    for shard in plan["shards"]:
        run = owned(shard["run_dir"])
        status = json.loads((run / "status.json").read_text()) if (run / "status.json").exists() else {}
        workers.append(dict(slot=shard["slot"], status=status.get("status", "unknown")))
        sources.append((shard["slot"], run / "rollouts.jsonl"))
    for slot, source in sources:
        if not source.exists():
            continue
        with source.open("rb") as stream:
            for number, raw in enumerate(stream, 1):
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    (malformed if raw.endswith(b"\n") else partial).append(dict(slot=slot, line=number))
                    continue
                identity = dict(instance_id=row.get("instance_id"), global_task_index=row.get("_ng_task_index"),
                                rollout_index=row.get("_ng_rollout_index"), slot=slot)
                if row.get("opencode_finished") is not True:
                    skipped.append(identity)
                    continue
                record = dict(identity, source=str(source.relative_to(APP)), source_line=number,
                              row_sha256=hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest(),
                              evaluation_completed=row.get("evaluation_completed"), resolved=row.get("resolved"))
                try:
                    path = owned(row["opencode_results_fpath"])
                    relative = str(path.relative_to(APP))
                    details, size = inspect_export(path, cache.get(relative))
                    record = dict(details, **record, export_path=relative)
                    bytes_read += size
                    reused += size == 0
                except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                    record["issue"] = type(exc).__name__
                record["terminal_checks_passed"] = bool(not record.get("issue") and record.get("assistant_count", 0)
                    and not record.get("assistant_error_count") and record.get("final_finish") == "stop"
                    and record.get("final_completed") and not record.get("unfinished_tool_count"))
                if record.get("final_finish") == "length":
                    record["termination_classification"] = ("per_response_token_limit" if record.get("final_output_tokens")
                        == plan["protocol_settings"]["max_output_tokens"] else "length_finish_unclassified")
                records.append(record)
    flagged = [{k: r[k] for k in ("instance_id", "global_task_index", "slot", "issue", "assistant_error_types",
        "final_finish", "final_completed", "unfinished_tool_count", "termination_classification") if k in r}
        for r in records if not r["terminal_checks_passed"]]
    result = dict(schema_version=2, recorded_at_utc=datetime.now(timezone.utc).isoformat(), plan_id=plan_dir.name,
        status="snapshot_not_final", final_run_validation=False, workers=workers,
        cache_key=["export_path", "export_stat.size", "export_stat.mtime_ns", "export_stat.inode"],
        summary=dict(checked_exports=len(records), bytes_read=bytes_read, cached_exports_reused=reused,
            exports_read=len(records)-reused, elapsed_seconds=round(time.monotonic()-started, 3),
            terminal_checks_passed=sum(r["terminal_checks_passed"] for r in records),
            assistant_error_records=sum(bool(r.get("assistant_error_count")) for r in records),
            non_stop_final_turns=sum(r.get("final_finish") != "stop" for r in records),
            unfinished_tool_count=sum(r.get("unfinished_tool_count", 0) for r in records),
            ordinary_or_unclassified_tool_error_count=sum(r.get("tool_status_counts", {}).get("error", 0) for r in records),
            flagged_count=len(flagged), skipped_unfinished_count=len(skipped),
            partial_jsonl_line_count=len(partial), malformed_complete_jsonl_line_count=len(malformed)),
        flagged=flagged, skipped_unfinished=skipped, partial_jsonl_lines=partial,
        malformed_complete_jsonl_lines=malformed, records=records,
        limitations=["Evidence snapshot only; the separate strict scorer controls completeness and score.",
            "Root-session exports only; child-session trajectories may be absent.",
            "Tool errors are counted separately and do not by themselves establish infrastructure failure.",
            "Unchanged file stat signatures reuse prior content hashes; this is a local cache, not tamper proofing."])
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return result["summary"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().plan_dir), sort_keys=True))
