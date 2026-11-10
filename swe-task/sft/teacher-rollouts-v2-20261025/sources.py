"""teacher-rollouts-v2 (2026-10-25): the rollout slots of every source on the 50 train-acc-v2 tasks (read-only).

slots(source) -> list of {source, task_id, run, repeat, slot, status (pass|fail|infra), reward, f2p, p2p,
                          attempt_dir (the slot's final attempt), origin}
Sources:
  sc14, sftv2      train-acc-v2 probes: TA/results/<model>-group-<K>.json (driver.py's compact records; attempt_dir =
                   the slot's final attempt as rollouts.probe_records() chose it)
  kimi             the original build probes of each task (all 5 kimi slots, pass or fail): probe dir =
                   attempt_dir.parents[4] of the task's rows in the v2 SFT selection (kimi_pass_v2.jsonl); a task whose
                   lost slots were re-run post hoc (official1-scriggo gomacro c01) takes its final records from
                   audit/recovery/<task>/after-records-sanitizing.jsonl (the recovery probe numbers its re-runs from 0,
                   so slot names map only through that file). Per slot: the last completed attempt (terminal.json
                   evaluation_completed and no cached generation failure, generation-validity-v2-r7.json), else the
                   last attempt (infra); read-only (deepswe_campaign.slot_state may write validity caches).
  glm, deepseek    this directory's probes: TR/results/<model>-<name>.json (driver.py), same record shape as sc14.
Pass/fail is the grade recorded in the attempt's terminal.json (ATP's view). kimi's grades come from the build-time
grader of each task (later grading repairs regraded a few near-bar slots; not applied here), the others from the
task_dir's grader at probe time (tasks.json).
"""
import collections
import json
from pathlib import Path

S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
TA = S / "sft/train-acc-v2-20261024"
TR = Path(__file__).resolve().parent
SELECTION = S / "sft/corpus/kimi-pass-v2-20261023/selection/kimi_pass_v2.jsonl"


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def tasks():
    return read_json(TA / "tasks.json")["tasks"]


def _status(terminal, completed):
    if not completed:
        return "infra"
    return "pass" if terminal.get("reward") == 1 else "fail"


def _from_results(paths, source):
    out = []
    by_task = {task["task_id"]: task for task in tasks()}
    for path in paths:
        data = read_json(path)
        for task_id, records in (data or {}).get("records", {}).items():
            for record in records:
                attempt = record.get("attempt_dir")
                out.append({"source": source, "task_id": task_id, "run": by_task.get(task_id, {}).get("run"),
                            "repeat": int(record["slot"].rsplit("-r", 1)[1]), "slot": record["slot"],
                            "status": record["status"], "reward": record.get("reward"), "f2p": record.get("f2p"),
                            "p2p": record.get("p2p"), "attempt_dir": attempt, "origin": str(path)})
    return out


def _attempts(slot_dir):
    return sorted((p for p in Path(slot_dir).glob("attempt-*") if p.is_dir()), key=lambda p: int(p.name.split("-")[-1]))


def _kimi_slot(probe_dirs, task_id, repeat):
    """(attempt_dir, completed, terminal) for one kimi repeat: the slot's LAST completed attempt (terminal.json
    evaluation_completed and no cached generation failure) over the task's probe dirs in order (build probe, then a
    post-hoc recovery probe), else its last attempt (infra). This is the attempt the build scored (checked against the
    candidates' rollouts/index.md: 247/250 equal; the 2 others are the recovered gomacro c01 slots)."""
    attempts = []
    for probe in probe_dirs:
        manifest = read_json(probe / "manifest.json") or {}
        index = next((t["_ng_task_index"] for t in manifest.get("tasks", []) if t["task_id"] == task_id), None)
        if index is not None:
            attempts += _attempts(probe / "models/kimi/attempts" / f"{index}-{repeat}")
    for attempt in reversed(attempts):
        terminal = read_json(attempt / "terminal.json") or {}
        validity = read_json(attempt / "generation-validity-v2-r7.json") or {}
        if terminal.get("evaluation_completed") is True and validity.get("failure") is None:
            return attempt, True, terminal
    if attempts:
        return attempts[-1], False, read_json(attempts[-1] / "terminal.json") or {}
    return None, False, {}


def kimi_probe_dirs():
    rows = [json.loads(line) for line in SELECTION.read_text().splitlines() if line.strip()]
    dirs = collections.defaultdict(set)
    for row in rows:
        if row.get("source") == "hengyuf":
            dirs[row["task_id"]].add(Path(row["attempt_dir"]).parents[4])
    # build probe first, post-hoc recovery probes after (a repeat completed in the build probe keeps that attempt)
    return {task_id: sorted(found, key=lambda p: ("/audit/recovery/" in str(p), str(p))) for task_id, found in dirs.items()}


def recovered_records(task):
    """The final records of a task whose lost slots were recovered post hoc (recover_slots.py): its recovery probe
    numbers the re-run slots from 0, so the build's slot names map through after-records-sanitizing.jsonl."""
    path = S / "synth/runs" / task["run"] / "audit/recovery" / task["task_id"] / "after-records-sanitizing.jsonl"
    if not path.exists():
        return None
    return {r["slot"]: r for r in map(json.loads, path.read_text().splitlines()) if r.get("model") == "kimi"
            or str(r.get("slot", "")).startswith("kimi-")}


def kimi_slots():
    probes = kimi_probe_dirs()
    out = []
    for task in tasks():
        recovered = recovered_records(task)
        if recovered:
            for repeat in range(5):
                record = recovered[f"kimi-r{repeat}"]
                out.append({"source": "kimi", "task_id": task["task_id"], "run": task["run"], "repeat": repeat,
                            "slot": f"kimi-r{repeat}", "status": record["status"], "reward": record.get("reward"),
                            "f2p": record.get("f2p"), "p2p": record.get("p2p"), "attempt_dir": record.get("attempt_dir"),
                            "origin": "after-records-sanitizing.jsonl (post-hoc recovery)"})
            continue
        for repeat in range(5):
            attempt, completed, terminal = _kimi_slot(probes[task["task_id"]], task["task_id"], repeat)
            out.append({"source": "kimi", "task_id": task["task_id"], "run": task["run"], "repeat": repeat,
                        "slot": f"kimi-r{repeat}", "status": _status(terminal, completed) if attempt else "infra",
                        "reward": terminal.get("reward"),
                        "f2p": f"{terminal.get('f2p_passed')}/{terminal.get('f2p_total')}",
                        "p2p": f"{terminal.get('p2p_passed')}/{terminal.get('p2p_total')}",
                        "attempt_dir": str(attempt) if attempt else None,
                        "origin": str(attempt.parents[4]) if attempt else None})
    return out


def slots(source):
    if source in ("sc14", "sftv2"):
        return _from_results(sorted(TA.glob(f"results/{source}-group-*.json")), source)
    if source == "kimi":
        return kimi_slots()
    return _from_results(sorted(TR.glob(f"results/{source}-*.json")), source)
