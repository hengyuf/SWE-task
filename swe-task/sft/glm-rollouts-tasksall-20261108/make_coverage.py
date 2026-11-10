#!/usr/bin/env python3
"""glm-rollouts-tasksall: coverage.json = the tasks-all tasks that already have HOSTED GLM-5.3 rollouts (user 2026-11-08:
"you don't need to redo the rollout for the tasks that already has the hosted GLM 5.3 rollouts").

    python3 -I -B make_coverage.py [--write]

Sources (from the verified coverage sweep, workflow glm-coverage-sweep, 2026-11-08): the only hosted GLM-5.3 agent
rollouts on synthetic tasks are sft/teacher-rollouts-v2-20261025 (nvidia/zai-org/glm-5.3, stock OpenCode prompt, 5 per
task on the 50 train-acc-v2 tasks; results/glm-*.json). Not counted: teacher-deepswe-20261030 (DeepSWE benchmark tasks,
not synthetic), glm53-serve/e2e (self-hosted 1-2 test rollouts on 16 of those same 50 tasks), the synth build pipeline
(GLM is only the judge there). The 220 self-hosted SFT-v3 tasks are excluded by make_tasks.py itself.
Tasks are matched to inventory rows by (run, task_id) (task ids collide across runs).
"""
import argparse
import collections
import glob
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
INVENTORY = S / "sft/tasks-all-20261106/all_tasks.jsonl"
HOSTED = [{"name": "teacher-rollouts-v2-20261025", "results": S / "sft/teacher-rollouts-v2-20261025/results",
           "pattern": "glm-*.json", "tasks": S / "sft/train-acc-v2-20261024/tasks.json",
           "model": "nvidia/zai-org/glm-5.3"}]
OUT = HERE / "coverage.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in INVENTORY.read_text().splitlines() if line.strip()]
    by_run_id = collections.defaultdict(list)
    for row in rows:
        by_run_id[(row["run"], row["task_id"])].append(row)
    covered = {}
    for source in HOSTED:
        runs = {t["task_id"]: t["run"] for t in json.loads(source["tasks"].read_text())["tasks"]}
        statuses = collections.defaultdict(list)
        for path in sorted(glob.glob(str(source["results"] / source["pattern"]))):
            data = json.loads(Path(path).read_text())
            assert data.get("model_id") == source["model"], (path, data.get("model_id"))
            for task_id, records in (data.get("records") or {}).items():
                statuses[task_id] += [r.get("status") for r in records]
        for task_id, status in statuses.items():
            matches = by_run_id[(runs[task_id], task_id)]
            assert len(matches) == 1, (task_id, runs[task_id], len(matches))
            row = matches[0]
            key = f"{row['source']}/{row['run']}/{row['cand']}/{row['task_id']}"
            entry = covered.setdefault(key, {"key": key, "task_id": task_id, "run": row["run"], "sources": []})
            entry["sources"].append({"source": source["name"], "model": source["model"], "slots": len(status),
                                     "graded": sum(s in ("pass", "fail") for s in status),
                                     "pass": sum(s == "pass" for s in status)})
    out = {"made_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rule": "tasks with hosted GLM-5.3 rollouts (standard "
           "OpenCode prompt)", "sources": [{k: str(v) for k, v in s.items()} for s in HOSTED],
           "covered": sorted(covered.values(), key=lambda e: e["key"])}
    print(json.dumps({"covered": len(out["covered"]), "slots": sum(s["slots"] for e in out["covered"]
                                                                    for s in e["sources"])}, indent=1))
    if args.write:
        tmp = OUT.with_name(".coverage.json.tmp")
        tmp.write_text(json.dumps(out, indent=1) + "\n")
        tmp.replace(OUT)
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
