#!/usr/bin/env python3
"""kimi-rollouts-v5 (2026-10-28): the task list of this batch. Read-only on synth/runs.

User (hengyuf, 2026-10-28, verbatim): "这组rollout 跑完之后,再给official的第二批(找rank2-5 repo 的那一批) 里的accepted tasks
和标记为strong的task 每个task 额外做5个kimi 的256 k context rollout".

The second official batch = the 14 runs synth/runs/official2-* (the Codex-reselected rank 2-5 host repos; queue2).
Per run (runs and tasks sorted by name):
  accepted  every row of <run>/accepted/tasks.jsonl. Rolled out from <run>/accepted/<task_id> (the installed grader).
            Its candidate metrics.json must say accept=true with no "withdrawn" field, and the task must not be under
            <run>/withdrawn/ or in <run>/withdrawn.jsonl (S/README "Withdrawals"); a withdrawn task is excluded.
  strong    every <run>/audit/failure_match/<task_id>.json whose "verdict" is "strong" (the reviewed label; the
            *.json.pre-review backups are ignored). When audit/failure_match_review/<task_id>.json exists, its
            reviewed_verdict must agree.
            A strong task that is not accepted is rolled out from its scored candidate dir: the "task_dir" of the
            <run>/build/<target>/<repo>/round-<n>/<cand>/metrics.json with that task_id (the same tree as the probe
            copy round-<n>/probe-*/tasks/<task_id>; the driver reads the scored dir).
  why       accepted | strong | both.
Checks per task dir (all must hold):
  - the basename equals the task_id (rollouts.probe() copies the dir under that name);
  - instruction.md is non-empty;
  - task.toml has metadata.task_id = task_id, environment.docker_image = "snapshot:<uuid>" (for an accepted task it
    must equal the tasks.jsonl row's image), verifier.timeout_sec and agent.timeout_sec;
  - tests/ holds grader.py, test.sh, config.json and test.patch; grader.py's sha256 is recorded with the pipeline's
    current synth/verifier/grader.py for reference;
  - synth rollouts.dataset_row() builds the row from it (only under the ATP controller python, which has tomllib).
Each entry also records file digests (instruction.md, task.toml, every file under tests/). driver.py re-checks them at
every leg, so a task changed after this list was frozen stops the batch rather than mixing graders.

    <ATP controller python> -I -B official2_tasks.py --write   # derive and write official2-tasks.json (refuses to overwrite)
    <ATP controller python> -I -B official2_tasks.py           # derive again and compare with official2-tasks.json
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import tomllib  # noqa: E402

HERE = Path(__file__).resolve().parent
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
RUNS = S / "synth" / "runs"
PIPELINE_GRADER = S / "synth" / "verifier" / "grader.py"
OUT = HERE / "official2-tasks.json"
EXPECTED_RUNS = 14
TEST_FILES = ("grader.py", "test.sh", "config.json", "test.patch")
SNAPSHOT = re.compile(r"snapshot:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
REQUEST = ("这组rollout 跑完之后,再给official的第二批(找rank2-5 repo 的那一批) 里的accepted tasks 和标记为strong的task "
           "每个task 额外做5个kimi 的256 k context rollout")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digests(task_dir: Path) -> dict:
    """sha256 of instruction.md, task.toml and every file under tests/ (relative path -> digest)."""
    task_dir = Path(task_dir)
    out = {"instruction.md": sha256(task_dir / "instruction.md"), "task.toml": sha256(task_dir / "task.toml")}
    for path in sorted(p for p in (task_dir / "tests").rglob("*") if p.is_file()):
        out[str(path.relative_to(task_dir))] = sha256(path)
    return out


def metrics_by_task(run: Path) -> dict:
    """{task_id: (metrics.json path, metrics)} of every scored candidate of a run."""
    out = {}
    for path in sorted(run.glob("build/*/*/round-*/cand-*/metrics.json")):
        metrics = json.loads(path.read_text())
        if metrics.get("task_id"):
            out.setdefault(metrics["task_id"], []).append((path, metrics))
    return out


def withdrawn_ids(run: Path) -> set:
    ids = {p.name for p in (run / "withdrawn").glob("syn-*") if p.is_dir() and not p.name.endswith(".workspace")}
    log = run / "withdrawn.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            if line.strip():
                ids.add(json.loads(line).get("task_id"))
    return ids


def check_dir(task_id: str, task_dir: Path, image_expected=None) -> dict:
    """The checks above; returns the entry's facts plus "problems" (empty when every check holds)."""
    problems = []
    facts = {"task_dir": str(task_dir)}
    if task_dir.name != task_id:
        problems.append(f"basename {task_dir.name} != task_id")
    instruction = task_dir / "instruction.md"
    if not instruction.is_file() or not instruction.read_text().strip():
        problems.append("instruction.md missing or empty")
    try:
        config = tomllib.loads((task_dir / "task.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        problems.append(f"task.toml unreadable: {type(error).__name__}")
        config = {}
    image = (config.get("environment") or {}).get("docker_image")
    facts.update(image=image, base_commit=(config.get("metadata") or {}).get("base_commit_hash"),
                 language=(config.get("metadata") or {}).get("language"),
                 repository_url=(config.get("metadata") or {}).get("repository_url"),
                 title=(config.get("metadata") or {}).get("display_title"),
                 verifier_timeout_sec=(config.get("verifier") or {}).get("timeout_sec"),
                 agent_timeout_sec=(config.get("agent") or {}).get("timeout_sec"),
                 cpus=(config.get("environment") or {}).get("cpus"),
                 memory_mb=(config.get("environment") or {}).get("memory_mb"))
    if (config.get("metadata") or {}).get("task_id") != task_id:
        problems.append("task.toml metadata.task_id differs")
    if not (isinstance(image, str) and SNAPSHOT.fullmatch(image)):
        problems.append(f"task.toml environment.docker_image is not snapshot:<uuid>: {image!r}")
    if image_expected is not None and image != image_expected:
        problems.append(f"task.toml image {image} != tasks.jsonl image {image_expected}")
    if not facts["verifier_timeout_sec"] or not facts["agent_timeout_sec"]:
        problems.append("task.toml lacks verifier.timeout_sec or agent.timeout_sec")
    missing = [name for name in TEST_FILES if not (task_dir / "tests" / name).is_file()]
    if missing:
        problems.append(f"tests/ lacks {missing}")
    if not missing:
        facts["grader_sha256"] = sha256(task_dir / "tests" / "grader.py")
        facts["tests"] = sorted(str(p.relative_to(task_dir / "tests")) for p in (task_dir / "tests").rglob("*")
                                if p.is_file())
        facts["digests"] = digests(task_dir)
    try:  # the row rollouts.probe() writes for this task (needs the synth modules: ATP controller python)
        sys.path.insert(0, str(S / "synth"))
        import rollouts  # noqa: PLC0415
        row = rollouts.dataset_row(task_dir)
        facts["dataset_row_ok"] = row["task_id"] == task_id and row["image"] == image
        if not facts["dataset_row_ok"]:
            problems.append("rollouts.dataset_row() gives another task_id / image")
    except Exception as error:  # noqa: BLE001
        problems.append(f"rollouts.dataset_row() failed: {type(error).__name__}: {str(error)[:200]}")
    facts["problems"] = problems
    return facts


def derive() -> dict:
    runs = sorted(p for p in RUNS.glob("official2-*") if p.is_dir())
    tasks, excluded, notes = [], [], []
    for run in runs:
        accepted_rows = []
        if (run / "accepted" / "tasks.jsonl").exists():
            accepted_rows = [json.loads(line) for line in (run / "accepted" / "tasks.jsonl").read_text().splitlines()
                             if line.strip()]
        accepted = {row["task_id"]: row for row in accepted_rows}
        accepted_dirs = {p.name for p in (run / "accepted").glob("syn-*")
                         if p.is_dir() and not p.name.endswith(".workspace")}
        if set(accepted) != accepted_dirs:
            notes.append(f"{run.name}: accepted/tasks.jsonl rows {sorted(set(accepted) - accepted_dirs)} have no dir, "
                         f"dirs {sorted(accepted_dirs - set(accepted))} have no row")
        strong, review = {}, {}
        for path in sorted((run / "audit" / "failure_match").glob("*.json")):
            label = json.loads(path.read_text())
            if label.get("verdict") == "strong":
                strong[label.get("task_id") or path.stem] = path
        for task_id in strong:
            reviewed = run / "audit" / "failure_match_review" / f"{task_id}.json"
            if reviewed.exists():
                review[task_id] = json.loads(reviewed.read_text()).get("reviewed_verdict")
        metrics = metrics_by_task(run)
        withdrawn = withdrawn_ids(run)
        for task_id in sorted(set(accepted) | set(strong)):
            found = metrics.get(task_id, [])
            why = "both" if task_id in accepted and task_id in strong else "accepted" if task_id in accepted else "strong"
            marked = [str(path) for path, m in found if "withdrawn" in m or str(m.get("reason", "")).startswith("withdrawn")]
            if task_id in withdrawn or marked:
                excluded.append({"task_id": task_id, "run": run.name, "why": why, "reason": "withdrawn",
                                 "evidence": marked or [str(run / "withdrawn")]})
                continue
            entry = {"task_id": task_id, "run": run.name, "why": why,
                     "failure_match_verdict": json.loads(strong[task_id].read_text()).get("verdict")
                     if task_id in strong else json.loads((run / "audit" / "failure_match" / f"{task_id}.json")
                                                          .read_text()).get("verdict")
                     if (run / "audit" / "failure_match" / f"{task_id}.json").exists() else None,
                     "reviewed_verdict": review.get(task_id) if task_id in strong else None,
                     "candidate_metrics": [str(path) for path, _ in found]}
            if len(found) != 1:
                entry.setdefault("problems_pre", []).append(f"{len(found)} metrics.json files with this task_id")
            scored = found[0][1] if found else {}
            entry["candidate_accept"] = scored.get("accept")
            entry["candidate_task_dir"] = scored.get("task_dir")
            entry["rates"] = {k: scored.get(k) for k in ("student_rate", "teacher_clean_rate", "similarity", "reason")}
            if task_id in accepted:
                task_dir = run / "accepted" / task_id
                entry["task_dir_kind"] = "accepted"
                facts = check_dir(task_id, task_dir, accepted[task_id].get("image"))
                if scored.get("accept") is not True:
                    facts["problems"].append("accepted row but its metrics.json does not say accept=true")
            else:
                task_dir = Path(scored.get("task_dir") or "")
                entry["task_dir_kind"] = "scored candidate (not accepted; failure-match strong)"
                facts = check_dir(task_id, task_dir)
                probes = sorted(task_dir.parents[2].glob(f"round-*/probe-*/tasks/{task_id}")) if scored else []
                entry["probe_copies"] = [str(p) for p in probes]
                entry["probe_copy_identical"] = bool(probes) and all(digests(p) == digests(task_dir) for p in probes)
            if task_id in strong and review.get(task_id) not in (None, "strong"):
                facts["problems"].append(f"failure_match says strong but the review says {review[task_id]}")
            facts["problems"] = entry.pop("problems_pre", []) + facts["problems"]
            entry.update(facts)
            tasks.append(entry)
    pipeline = sha256(PIPELINE_GRADER)
    for entry in tasks:
        entry["grader_is_pipeline_grader"] = entry.get("grader_sha256") == pipeline
    counts = {"runs": len(runs), "tasks": len(tasks),
              "accepted": sum(e["why"] in ("accepted", "both") for e in tasks),
              "strong": sum(e["why"] in ("strong", "both") for e in tasks),
              "both": sum(e["why"] == "both" for e in tasks),
              "strong_not_accepted": sum(e["why"] == "strong" for e in tasks),
              "withdrawn_excluded": len(excluded),
              "distinct_images": len({e.get("image") for e in tasks}),
              "per_run": {run.name: sum(e["run"] == run.name for e in tasks) for run in runs}}
    return {"request": REQUEST, "runs_glob": str(RUNS / "official2-*"), "runs": [run.name for run in runs],
            "pipeline_grader": {"path": str(PIPELINE_GRADER), "sha256": pipeline},
            "counts": counts, "excluded": excluded, "notes": notes, "tasks": tasks}


def comparable(data: dict) -> dict:
    """The parts of a derivation that must not change between --write and a later check."""
    return {"runs": data["runs"], "counts": data["counts"], "excluded": data["excluded"],
            "tasks": [{k: e.get(k) for k in ("task_id", "run", "why", "task_dir", "image", "grader_sha256", "digests")}
                      for e in data["tasks"]]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write official2-tasks.json (refuses to overwrite)")
    args = parser.parse_args()
    data = derive()
    problems = {e["task_id"]: e["problems"] for e in data["tasks"] if e["problems"]}
    summary = {"counts": data["counts"], "excluded": data["excluded"], "notes": data["notes"], "problems": problems}
    if len(data["runs"]) != EXPECTED_RUNS:
        problems["_runs"] = [f"{len(data['runs'])} official2 runs, expected {EXPECTED_RUNS}"]
    if args.write:
        if OUT.exists():
            print(json.dumps(dict(summary, error=f"{OUT} exists; not overwriting"), indent=1))
            return 2
        if problems:
            print(json.dumps(dict(summary, error="problems found; not writing"), indent=1))
            return 1
        data["derived_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        tmp = OUT.with_name("." + OUT.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
        os.replace(tmp, OUT)
        print(json.dumps(dict(summary, wrote=str(OUT)), indent=1, ensure_ascii=False))
        return 0
    frozen = json.loads(OUT.read_text()) if OUT.exists() else None
    same = frozen is not None and comparable(frozen) == comparable(data)
    print(json.dumps(dict(summary, frozen=str(OUT) if frozen else None, frozen_matches_runs=same), indent=1,
                     ensure_ascii=False))
    return 0 if same and not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
