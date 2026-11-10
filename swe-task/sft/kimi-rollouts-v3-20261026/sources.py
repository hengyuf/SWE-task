"""kimi-rollouts-v3 (2026-10-26): the rollout slots of this batch for features.py (a byte-identical copy of
teacher-rollouts-v2's extractor, which imports this module for read_json and slots). Read-only.

slots(source) -> list of {source, task_id, run, repeat, slot, status (pass|fail|infra), reward, f2p, p2p,
                          attempt_dir (the slot's final attempt, as rollouts.probe_records() chose it), origin}
Sources:
  kimi3          this batch: results/kimi-<taskset>-g<K>.json (driver.py's compact records; sel50, rest47, baihe)
  kimi3-pilot    the 1-rollout pilots: results/kimi-pilot.json, results/kimi-baihe-pilot.json
  ctrl-old       the control arm (previous Kimi settings, same harness/key/sandbox): results/kimi-ctrl-old.json
The baselines on the 50 tasks are teacher-rollouts-v2's own features files, copied unchanged to features/kimi.jsonl
(previous Kimi-K3, 1M window, no instruction) and features/sc14.jsonl (base Nemotron sc14).
"""
import json
from pathlib import Path

S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
KR = Path(__file__).resolve().parent
SELECTION = S / "sft/corpus/kimi-pass-v2-20261023/selection/kimi_pass_v2.jsonl"


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def runs():
    rows = [json.loads(line) for line in SELECTION.read_text().splitlines() if line.strip()]
    return {row["task_id"]: row.get("run") for row in rows}


def _from_results(paths, source):
    out = []
    run_of = runs()
    for path in paths:
        data = read_json(path)
        for task_id, records in (data or {}).get("records", {}).items():
            for record in records:
                out.append({"source": source, "task_id": task_id, "run": run_of.get(task_id),
                            "repeat": int(record["slot"].rsplit("-r", 1)[1]), "slot": record["slot"],
                            "status": record["status"], "reward": record.get("reward"), "f2p": record.get("f2p"),
                            "p2p": record.get("p2p"), "attempt_dir": record.get("attempt_dir"), "origin": str(path)})
    return out


def slots(source):
    if source == "kimi3":
        return _from_results(sorted(KR.glob("results/kimi-*-g*.json")), source)
    if source == "ctrl-old":  # control arm: previous Kimi settings (no sentence, 1M window), driver CTRL_TASKS x 5
        return _from_results([KR / "results/kimi-ctrl-old.json"] if (KR / "results/kimi-ctrl-old.json").exists() else [],
                             source)
    if source == "kimi3-pilot":
        return _from_results(sorted(p for p in (KR / "results/kimi-pilot.json", KR / "results/kimi-baihe-pilot.json")
                                    if p.exists()), source)
    raise ValueError(f"unknown source {source}")
