#!/usr/bin/env python3
"""Freeze the task list of glm-rollouts-sftv3: every task that has at least one conversation in SFT-v3
(sft/corpus/sft-v3-20261029/selection/sft_v3.jsonl), with its task dir, sandbox owner and file digests.

    /usr/bin/python3 -I -B make_tasks.py [--write]      # prints counts; --write writes tasks.json (refuses to overwrite)

Task dirs come from the sources the Kimi rollouts used: hengyuf's v2 tasks (train-acc-v2 tasks.json + pool.json),
Baihe's v2 tasks (baihe_v2_tasks.tsv, Baihe's sandbox credential) and official2's v5 tasks (official2-tasks.json).
Digests = sha256 of instruction.md, task.toml and every file under tests/ (as kimi-rollouts-v5's official2_tasks.py);
driver.py refuses to run a task whose files changed since this list was frozen.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
SELECTION = S / "sft/corpus/sft-v3-20261029/selection/sft_v3.jsonl"
TA = S / "sft/train-acc-v2-20261024"
BAIHE_TSV = S / "sft/baihe-snapshot-access-20261026/baihe_v2_tasks.tsv"
V5_TASKS = S / "sft/kimi-rollouts-v5-official2-20261028/official2-tasks.json"
OUT = HERE / "tasks.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digests(task_dir: Path) -> dict:
    out = {"instruction.md": sha256(task_dir / "instruction.md"), "task.toml": sha256(task_dir / "task.toml")}
    for path in sorted(p for p in (task_dir / "tests").rglob("*") if p.is_file()):
        out[str(path.relative_to(task_dir))] = sha256(path)
    return out


def sources() -> dict:
    """task_id -> {task_dir, owner, source, run, snapshot}"""
    out = {}
    for name in ("tasks.json", "pool.json"):
        for task in json.loads((TA / name).read_text())["tasks"]:
            out[task["task_id"]] = {"task_dir": task["task_dir"], "owner": "hengyuf", "source": "v2-hengyuf",
                                    "run": task.get("run"), "snapshot": None}
    lines = BAIHE_TSV.read_text().splitlines()
    header = lines[0].split("\t")
    for line in lines[1:]:
        if line.strip():
            row = dict(zip(header, line.split("\t")))
            assert row["task_id"] not in out, row["task_id"]
            out[row["task_id"]] = {"task_dir": row["task_dir"], "owner": "baihe", "source": "v2-baihe",
                                   "run": row["run"], "snapshot": row["snapshot"]}
    for task in json.loads(V5_TASKS.read_text())["tasks"]:
        assert task["task_id"] not in out, task["task_id"]
        out[task["task_id"]] = {"task_dir": task["task_dir"], "owner": "hengyuf", "source": "v5-official2",
                                "run": task["run"], "snapshot": None}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in SELECTION.read_text().splitlines() if line.strip()]
    conversations = collections.Counter(row["task_id"] for row in rows)
    by_source = collections.defaultdict(collections.Counter)
    for row in rows:
        by_source[row["task_id"]][row["source"]] += 1
    known = sources()
    missing = sorted(set(conversations) - set(known))
    assert not missing, missing
    tasks = []
    for task_id in sorted(conversations, key=lambda t: (known[t]["owner"], known[t]["source"], t)):
        info = known[task_id]
        task_dir = Path(info["task_dir"])
        assert (task_dir / "task.toml").is_file(), task_dir
        tasks.append({"task_id": task_id, **info, "sft_v3_conversations": conversations[task_id],
                      "sft_v3_sources": dict(by_source[task_id]), "digests": digests(task_dir)})
    counts = collections.Counter((t["owner"], t["source"]) for t in tasks)
    left_out = sorted(set(known) - set(conversations))
    summary = {"tasks": len(tasks), "by_owner_source": {f"{o}/{s}": n for (o, s), n in sorted(counts.items())},
               "not_in_sft_v3": left_out}
    print(json.dumps(summary, indent=1))
    if args.write:
        if OUT.exists():
            raise SystemExit(f"{OUT} exists; refusing to overwrite a frozen list")
        OUT.write_text(json.dumps({
            "request": "user 2026-11-07: rollout GLM-5.3 (self-hosted) on the SFT-v3 tasks, 5 per task, 256k output "
                       "limit, compaction allowed",
            "selection": str(SELECTION), "selection_sha256": sha256(SELECTION), "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "summary": summary, "tasks": tasks}, indent=1) + "\n")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
