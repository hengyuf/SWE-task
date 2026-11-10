#!/usr/bin/env python3
"""teacher-deepswe setup (2026-10-30): snapshot the task list, the wave split and the base rates into tasks/ (read-only on
every source). driver.py, progress.py and collect.py read only these snapshots, so later edits of the sources cannot
change a running campaign's task set.

    /usr/bin/python3 -I -B setup/make_tasks.py [--check]     # --check: verify tasks/*.json against the sources

Sources (all read-only):
  TASKS   the 113 Harbor task dirs of DeepSWE v1.1 (deep-swe@435ee89, ATP's dataset_revision; Gym-prepared layout)
  WAVE    uncovered_base_k10.csv: task_id column = the 79 tasks without an accepted synthetic task = wave 1;
          the other 34 = wave 2
  SYNTH   synth_targets.json next to it: per target task, synthetic build runs / scored / accepted (cross-check: the
          wave-2 tasks are exactly the targets with accepted > 0)
  BASE    the base model's two 5-rollout DeepSWE evals (resolved out of 5 each): base_k10 = resolved sum (out of 10)
Writes tasks/waves.json (task list in a stable order = sorted task_id = the DeepSWE task_index order, with language,
image, task.toml sha256, wave) and tasks/base_k10.json (base rates per task), each with the sources' sha256.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402

TD = Path(__file__).resolve().parents[1]
E = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/eval/deepswe-v1.1")
TASKS = E / "cache/tasks"
WAVE = E / "analysis/base-uncovered-synth-20261030/uncovered_base_k10.csv"
SYNTH = E / "analysis/base-uncovered-synth-20261030/synth_targets.json"
BASE = (E / "runs/base-step0-rep5-out131k/per_task.csv", E / "runs/base-step0-rep5-out131k-rerun1/per_task.csv")
TARGETS_TASK_IDS = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/data/targets/sc14-out131k-kimi/task_ids.json")
REVISION = "435ee89ec2f2e2289f33b0da4f992f0b7b7266b9"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def toml_field(text: str, section: str, key: str):
    """One string / number field of a task.toml (/usr/bin/python3 3.10 has no tomllib; the files are flat)."""
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        header = re.fullmatch(r"\[\[?([^\]]+)\]\]?", stripped)
        if header:
            current = header.group(1)
            continue
        if current == section:
            found = re.fullmatch(rf'{re.escape(key)}\s*=\s*(.+)', stripped)
            if found:
                value = found.group(1).strip()
                return json.loads(value) if value.startswith('"') else float(value)
    return None


def build() -> tuple:
    marker = json.loads((TASKS / ".nemo_gym_deepswe.json").read_text())
    assert marker["source_revision"] == REVISION, marker
    dirs = sorted(p for p in TASKS.iterdir() if p.is_dir())
    tasks = []
    for d in dirs:
        text = (d / "task.toml").read_text()
        task_id = toml_field(text, "metadata", "task_id")
        assert task_id == d.name, (d, task_id)
        for required in ("task.toml", "instruction.md", "tests/test.sh", "tests/test.patch", "tests/grader.py",
                         "tests/config.json", "solution/solution.patch"):
            assert (d / required).is_file(), (d, required)
        tasks.append({"task_id": task_id, "task_dir": str(d), "language": toml_field(text, "metadata", "language"),
                      "category": toml_field(text, "metadata", "category"),
                      "image": toml_field(text, "environment", "docker_image"),
                      "verifier_timeout_sec": toml_field(text, "verifier", "timeout_sec"),
                      "task_toml_sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "instruction_sha256": sha256(d / "instruction.md")})
    assert len(tasks) == 113 and len({t["task_id"] for t in tasks}) == 113, len(tasks)
    order = json.loads(TARGETS_TASK_IDS.read_text())
    assert order == [t["task_id"] for t in tasks], "sorted task ids differ from the DeepSWE task_index order"
    with open(WAVE, newline="") as stream:
        uncovered = [row["task_id"] for row in csv.DictReader(stream)]
    assert len(uncovered) == 79 == len(set(uncovered)), len(uncovered)
    known = {t["task_id"] for t in tasks}
    assert set(uncovered) <= known, set(uncovered) - known
    synth = json.loads(SYNTH.read_text())
    accepted = {task for task, entry in synth.items() if entry.get("accepted", 0) > 0}
    for index, task in enumerate(tasks):
        task["task_index"] = index
        task["wave"] = 1 if task["task_id"] in uncovered else 2
        task["covered_by_accepted_synth"] = task["task_id"] in accepted
        task["synth_runs"] = (synth.get(task["task_id"]) or {}).get("runs", [])
        task["synth_accepted"] = (synth.get(task["task_id"]) or {}).get("accepted", 0)
    wave2 = [t["task_id"] for t in tasks if t["wave"] == 2]
    assert len(wave2) == 34, len(wave2)
    assert set(wave2) == accepted, ("wave 2 != accepted synth targets", sorted(set(wave2) ^ accepted))
    waves = {
        "created_by": "setup/make_tasks.py", "dataset_revision": REVISION,
        "sources": {"tasks_dir": str(TASKS), "marker": marker,
                    "wave_csv": {"file": str(WAVE), "sha256": sha256(WAVE)},
                    "synth_targets": {"file": str(SYNTH), "sha256": sha256(SYNTH)},
                    "task_index_order": {"file": str(TARGETS_TASK_IDS), "sha256": sha256(TARGETS_TASK_IDS)}},
        "order": "sorted task_id (= DeepSWE task_index order); group K of G = wave_tasks[K::G]",
        "counts": {"tasks": len(tasks), "wave1": len(uncovered), "wave2": len(wave2)},
        "wave1": [t["task_id"] for t in tasks if t["wave"] == 1], "wave2": wave2, "tasks": tasks}
    base = {}
    sources = []
    for path in BASE:
        with open(path, newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 113, (path, len(rows))
        sources.append({"file": str(path), "sha256": sha256(path)})
        for row in rows:
            entry = base.setdefault(row["task_id"], {"resolved": [], "completed": []})
            entry["resolved"].append(int(row["resolved"]))
            entry["completed"].append(int(row["completed"]))
    for task_id, entry in base.items():
        entry["base_k10"] = sum(entry["resolved"])
        entry["base_completed"] = sum(entry["completed"])
    assert set(base) == known
    base_out = {"created_by": "setup/make_tasks.py", "sources": sources,
                "definition": "base_k10 = resolved out of 5 in base-step0-rep5-out131k + resolved out of 5 in its rerun1",
                "tasks": base}
    return waves, base_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    waves, base = build()
    out = TD / "tasks"
    if args.check:
        same = (json.loads((out / "waves.json").read_text()) == waves and
                json.loads((out / "base_k10.json").read_text()) == base)
        print(json.dumps({"snapshots_match_sources": same}))
        raise SystemExit(0 if same else 1)
    out.mkdir(exist_ok=True)
    for name, value in (("waves.json", waves), ("base_k10.json", base)):
        tmp = out / ("." + name + ".tmp")
        tmp.write_text(json.dumps(value, indent=1) + "\n")
        tmp.replace(out / name)
    print(json.dumps({"tasks": waves["counts"], "languages": {
        wave: {lang: sum(1 for t in waves["tasks"] if t["wave"] == wave and t["language"] == lang)
               for lang in sorted({t["language"] for t in waves["tasks"]})} for wave in (1, 2)}}))


if __name__ == "__main__":
    main()
