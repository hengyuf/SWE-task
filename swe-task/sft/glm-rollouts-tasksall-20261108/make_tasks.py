#!/usr/bin/env python3
"""glm-rollouts-tasksall (2026-11-08): freeze the target list and the campaign plan.

User (hengyuf, 2026-11-08, verbatim): "... Then do the rollout for GLM 5.3 on all the tasks in
swe-task/sft/tasks-all-20261106/, you don't need to redo the rollout for the tasks that already has the hosted GLM 5.3
rollouts. Prioritize the task that are accepted or marked as strong (first do the accepted/strong tasks, then do
partial, then weak, then none or without any labels)."

    <atp controller python> -I -B make_tasks.py [--write]       # Python >= 3.11; --write refuses to overwrite

* Inventory: tasks-all-20261106/all_tasks.jsonl (2,203 built tasks). Task ids collide across runs (17 ids), so a task
  is keyed source/run/cand/task_id and no campaign holds two tasks with the same id.
* Task dir: the accepted copy (accepted_dir) for accepted tasks (5 of them carry later grading repairs; the rest differ
  only in provenance.json), the withdrawn copy (<run>/withdrawn/<task_id>) for withdrawn tasks, the repaired copy for
  the 2 rejected official1 tasks with a grading repair (flag grading_repair: repair.json repaired_task_dir, or the
  rescore's analyze_task_dir), else the build's task dir. Digests (instruction.md, task.toml, tests/**) are frozen here
  and re-checked by every driver leg.
* Excluded, already rolled out with GLM-5.3: coverage.json (hosted GLM-5.3, verified coverage sweep) and the 220 SFT-v3
  tasks of glm-rollouts-sftv3-20261107 (self-hosted, 5 each; inventory sft_v3 > 0).
* Tiers (priority): 1 accepted_or_strong (accepted, or failure-match audit strong); 2 audit partial; 3 weak; 4 none or
  not_audited (incl. every non-accepted Baihe task: Baihe never ran the audit).
* Snapshots: snapshots.json (snapshot_check.py) must report the task's snapshot as existing.
* Campaigns: per tier and sandbox owner, chunks of <= CHUNK[owner] tasks (no repeated task id); run order tier-major,
  hengyuf and Baihe interleaved within a tier. 5 rollouts per task.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import time
import tomllib

HERE = Path(__file__).resolve().parent
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
INVENTORY = S / "sft/tasks-all-20261106/all_tasks.jsonl"
SFTV3 = S / "sft/glm-rollouts-sftv3-20261107/tasks.json"
COVERAGE = HERE / "coverage.json"
SNAPSHOTS = HERE / "snapshots.json"
OUT = HERE / "tasks.json"
CHUNK = {"hengyuf": 25, "baihe": 25}
CONCURRENCY = {"hengyuf": 24, "baihe": 12}
REPEATS = 5
TIER_NAMES = {1: "accepted_or_strong", 2: "partial", 3: "weak", 4: "none_or_unlabeled"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digests(task_dir: Path) -> dict:
    out = {"instruction.md": sha256(task_dir / "instruction.md"), "task.toml": sha256(task_dir / "task.toml")}
    for path in sorted(p for p in (task_dir / "tests").rglob("*") if p.is_file()):
        out[str(path.relative_to(task_dir))] = sha256(path)
    return out


def key_of(row) -> str:
    return f"{row['source']}/{row['run']}/{row['cand']}/{row['task_id']}"


def tier_of(row) -> int:
    assert row["audit"] in ("strong", "partial", "weak", "none", "not_audited"), row["audit"]
    if row["accepted_or_strong"]:
        return 1
    return {"partial": 2, "weak": 3}.get(row["audit"], 4)


def repaired_dir(row):
    """The repaired task copy of a grading-repaired rejected task (status_dir has repair.json or rescore.json)."""
    status_dir = Path(row["status_dir"])
    if (status_dir / "repair.json").is_file():
        return Path(json.loads((status_dir / "repair.json").read_text())["repaired_task_dir"])
    if (status_dir / "rescore.json").is_file():
        text = json.loads((status_dir / "rescore.json").read_text())["substitutions"]["analyze_task_dir"]
        return Path(text.split(" (", 1)[0])
    raise AssertionError(f"grading_repair without repair.json / rescore.json: {row['task_id']}")


def chosen_dir(row) -> tuple:
    if row["status"] == "accepted":
        return Path(row["accepted_dir"]), "accepted_dir"
    task_dir = Path(row["task_dir"])
    if row["status"] == "withdrawn":
        # <runs>/<run>/build/<target>/<owner__repo>/<cand>/task/<task_id> -> <runs>/<run>/withdrawn/<task_id>
        run_root = next(p for p in task_dir.parents if p.name == row["run"])
        withdrawn = run_root / "withdrawn" / row["task_id"]
        assert (withdrawn / "task.toml").is_file(), f"withdrawn copy missing: {withdrawn}"
        return withdrawn, "withdrawn_dir"
    if "grading_repair" in row["flags"]:
        repaired = repaired_dir(row)
        assert (repaired / "task.toml").is_file() and repaired.name == row["task_id"], repaired
        return repaired, "repaired_dir"
    return task_dir, "task_dir"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in INVENTORY.read_text().splitlines() if line.strip()]
    assert len(rows) == 2203 and len({key_of(r) for r in rows}) == 2203
    sftv3_pairs = {(t["run"], t["task_id"]) for t in json.loads(SFTV3.read_text())["tasks"]}
    done_v3 = {key_of(r) for r in rows if r["sft_v3"] > 0}
    assert len(done_v3) == 220 and {(r["run"], r["task_id"]) for r in rows if r["sft_v3"] > 0} == sftv3_pairs, \
        "SFT-v3 rows != the 220 (run, task_id) of glm-rollouts-sftv3"
    coverage = json.loads(COVERAGE.read_text())
    hosted = {}
    for entry in coverage["covered"]:
        assert entry["key"] in {key_of(r) for r in rows}, entry
        hosted[entry["key"]] = entry
    snapshots = json.loads(SNAPSHOTS.read_text())["tasks"]
    tasks, excluded = [], collections.Counter()
    for row in rows:
        key = key_of(row)
        if key in done_v3:
            excluded["self_hosted_sftv3"] += 1
            continue
        if key in hosted:
            excluded["hosted_glm"] += 1
            continue
        task_dir, dir_kind = chosen_dir(row)
        image = tomllib.loads((task_dir / "task.toml").read_text())["environment"]["docker_image"]
        snap = snapshots[key]
        assert snap["result"] == "exists" and snap["image"] == image, (key, snap, image)
        tasks.append({"key": key, "task_id": row["task_id"], "source": row["source"], "owner": row["source"],
                      "run": row["run"], "cand": row["cand"], "repo": row["repo"], "target": row["target"],
                      "status": row["status"], "audit": row["audit"], "accepted_or_strong": row["accepted_or_strong"],
                      "tier": tier_of(row), "task_dir": str(task_dir), "dir_kind": dir_kind, "image": image,
                      "flags": row["flags"], "digests": digests(task_dir)})
    tasks.sort(key=lambda t: (t["tier"], t["owner"], t["run"], t["cand"], t["task_id"]))
    campaigns = []
    for tier in sorted({t["tier"] for t in tasks}):
        per_owner = {}
        for owner in ("hengyuf", "baihe"):
            pool = [t for t in tasks if t["tier"] == tier and t["owner"] == owner]
            chunks = []
            for task in pool:  # first chunk with room and without this task id
                for chunk in chunks:
                    if len(chunk) < CHUNK[owner] and all(t["task_id"] != task["task_id"] for t in chunk):
                        chunk.append(task)
                        break
                else:
                    chunks.append([task])
            per_owner[owner] = chunks
        order = []
        for i in range(max(len(c) for c in per_owner.values())):
            for owner in ("hengyuf", "baihe"):
                if i < len(per_owner[owner]):
                    order.append((owner, i, per_owner[owner][i]))
        for owner, i, chunk in order:
            name = f"t{tier}-{owner[0]}{i:02d}"
            for task in chunk:
                task["campaign"] = name
            campaigns.append({"name": name, "tier": tier, "tier_name": TIER_NAMES[tier], "sandbox": owner,
                              "tasks": [t["key"] for t in chunk], "repeats": REPEATS,
                              "max_concurrency": CONCURRENCY[owner], "slots": len(chunk) * REPEATS})
    for campaign in campaigns:
        ids = [k.rsplit("/", 1)[1] for k in campaign["tasks"]]
        assert len(ids) == len(set(ids)), campaign["name"]
    summary = {
        "inventory": len(rows), "excluded": dict(excluded), "tasks": len(tasks), "slots": len(tasks) * REPEATS,
        "by_tier": {f"{tier} {TIER_NAMES[tier]}": dict(collections.Counter(t["owner"] for t in tasks if t["tier"] == tier))
                    for tier in sorted(TIER_NAMES)},
        "by_tier_status": {f"{t} {s}": n for (t, s), n in sorted(collections.Counter((t["tier"], t["status"])
                                                                                   for t in tasks).items())},
        "dir_kind": dict(collections.Counter(t["dir_kind"] for t in tasks)),
        "campaigns": len(campaigns), "campaign_order": [c["name"] for c in campaigns],
    }
    print(json.dumps(summary, indent=1))
    if args.write:
        if OUT.exists():
            raise SystemExit(f"{OUT} exists; refusing to overwrite a frozen list")
        OUT.write_text(json.dumps({
            "request": "user 2026-11-08: GLM-5.3 rollouts (5 each) on all tasks of sft/tasks-all-20261106 without "
                       "GLM-5.3 rollouts, priority accepted/strong > partial > weak > none/unlabeled",
            "inventory": str(INVENTORY), "inventory_sha256": sha256(INVENTORY), "coverage": str(COVERAGE),
            "coverage_sha256": sha256(COVERAGE), "snapshots_checked_at": json.loads(SNAPSHOTS.read_text())["checked_at"],
            "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "summary": summary, "campaigns": campaigns,
            "tasks": tasks}, indent=1) + "\n")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
