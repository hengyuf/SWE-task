#!/usr/bin/env python3
"""official3 audit loop helper (hengyuf, 2026-10-27): list the scored official3 candidates that still need a
failure-match audit (synth/workflows/official3_audit_loop.js runs it from its poll agents). Read-only; stdlib only.

A round counts once its seeds.json exists (every candidate of the round scored, valid/invalid decided); its candidates
with a metrics.json are "scored". A scored candidate is new when it has no audit file
(runs/<run>/audit/failure_match/<task_id>.json) and its key "<run>/<cand>" is not in --exclude (the keys the loop already
handed to the annotate workflow, which may have skipped them as not eligible). Eligibility itself is the annotate scan's
call, not this script's.

    /usr/bin/python3 -I -B audit_poll.py [--exclude KEY,KEY,...]

Prints one JSON line: {"at", "new": [{"key", "run", "target", "repo", "round", "cand", "task_id", "accept"}],
"scored", "audited", "rounds_open", "queue": {"runs", "by_status", "finished", "runner_note"}}.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
Q = S / "synth/queue3"
RUNS = S / "synth/runs"
ROUND = re.compile(r"round-(\d+)")


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude", default="", help="comma-separated <run>/<cand> keys already handed out")
    exclude = {key for key in parser.parse_args().exclude.split(",") if key}
    # 2026-10-28: keys handed out by earlier audit-loop runs (main writes them when relaunching the loop)
    seen_file = Q / "audit_seen.json"
    if seen_file.is_file():
        exclude |= set(read_json(seen_file) or [])
    queue = read_json(Q / "queue.json") or {}
    names = [entry["run"] for entry in queue.get("runs", [])]
    new, scored, audited, rounds_open = [], 0, 0, 0
    for run in names:
        audit_dir = RUNS / run / "audit/failure_match"
        for root in sorted((RUNS / run / "build").glob("*/*")):
            if not root.is_dir() or root.name.startswith("abandoned-") or root.parent.name.startswith("abandoned-"):
                continue
            for round_dir in sorted(root.glob("round-*")):
                match = ROUND.fullmatch(round_dir.name)
                if not match or not round_dir.is_dir():
                    continue
                if not (round_dir / "seeds.json").is_file():
                    rounds_open += 1
                    continue
                for cand in sorted(round_dir.glob("cand-*")):
                    metrics = read_json(cand / "metrics.json")
                    if not isinstance(metrics, dict) or not metrics.get("task_id"):
                        continue
                    scored += 1
                    task_id = str(metrics["task_id"])
                    if (audit_dir / f"{task_id}.json").is_file():
                        audited += 1
                        continue
                    key = f"{run}/{cand.name}"
                    if key in exclude:
                        continue
                    new.append({"key": key, "run": run, "target": root.parent.name, "repo": root.name,
                                "round": int(match.group(1)), "cand": cand.name, "task_id": task_id,
                                "accept": bool(metrics.get("accept"))})
    state = read_json(Q / "state.json") or {}
    statuses = {run: (state.get("runs", {}).get(run) or {}).get("status", "queued") for run in names}
    by_status = {}
    for status in statuses.values():
        by_status[status] = by_status.get(status, 0) + 1
    heartbeat = read_json(Q / "heartbeat.json") or {}
    finished = bool(names) and all(status in ("done", "failed", "stuck") for status in statuses.values())
    print(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "new": new, "scored": scored, "audited": audited,
                      "rounds_open": rounds_open,
                      "queue": {"runs": len(names), "by_status": by_status, "finished": finished,
                                "runner_note": heartbeat.get("note")}}))


if __name__ == "__main__":
    main()
