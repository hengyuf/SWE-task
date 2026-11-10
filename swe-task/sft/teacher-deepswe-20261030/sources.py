"""kimi-rollouts-v4 (2026-10-28; final-summary sources kimi4-all / kimi-1m added 2026-10-29): the rollout slots of this
batch (and of kimi-rollouts-v3 and of the earlier 1M Kimi-K3 build probes, read-only) for features.py / window.py /
degenerate.py / phase_report.py (kimi-rollouts-v3's sources.py with the v4 sources added; features.py imports this module
for read_json and slots). Read-only on every input (results files, probe campaigns, the synth runs). The pre-final file
is backup-pre-final-20261029/sources.py; the sources it had are unchanged.

slots(source) -> list of {source, task_id, run, repeat, slot, status (pass|fail|infra), reward, f2p, p2p,
                          attempt_dir (the slot's final attempt, as rollouts.probe_records() chose it), origin, ...}
Sources:
  kimi4          this batch (stock prompt + 262k): results/kimi-<taskset>-g<K>.json records = the FINAL slots, i.e.
                 a slot ended by a broken reply is its one re-run when that re-run was graded (driver.py)
  kimi4-first    this batch's first attempts: the same files' first_attempt_records (before the broken-reply re-runs)
  kimi4-all      (2026-10-29) EVERY graded rollout of this batch once: the 970 first attempts (role "first") plus the
                 broken-reply re-runs (role "rerun"). Extra fields: taskset (sel50 | rest47 | baihe); group; role; final
                 (this rollout is its slot's final record); campaign (probe campaign dir, e.g. sel50-g0 / sel50-g0-brk1);
                 ended_by_broken_reply and last_reply_kind (driver.py's brokenreply.py verdict, from the results file);
                 agent_error; verifier_error; patch_chars_results; re-runs also rerun_slot (the slot label inside the
                 -brk campaign), rerun_of (the original attempt) and rerun_outcome
  kimi4-pilot    the 1-rollout pilots: results/kimi-pilot.json, results/kimi-baihe-pilot.json
  kimi3          kimi-rollouts-v3's sentence arm (256k + one reasoning sentence): V3/results/kimi-*-g*.json
  ctrl-old       kimi-rollouts-v3's control arm (previous Kimi settings: 1M, stock prompt): V3/results/kimi-ctrl-old.json
  kimi-1m        (2026-10-29) the earlier Kimi-K3 rollouts of the 50 sel50 tasks (1M window, stock prompt, 2026-10-17/21;
                 the synth pipeline's build-time probe slots kimi-r0..r4 of each task), i.e. the "Kimi-K3 previous (1M,
                 no sentence)" column of V3/analysis/phase1-report.md: teacher-rollouts-v2's slot list as stored in
                 V3/features/kimi.jsonl (the slot's attempt_dir as rollouts.probe_records() chose it, status, F2P, P2P),
                 read-only; extra fields campaign (the probe campaign dir name) and all_attempts (every attempt dir of
                 the slot, for degenerate.py)
The baselines on the 50 tasks in V3/features/ (kimi.jsonl = previous Kimi-K3, 1M window, no instruction; sc14.jsonl = base
Nemotron sc14) are teacher-rollouts-v2's features files.
"""
import json
from pathlib import Path

S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
KR = Path(__file__).resolve().parent
V3 = S / "sft/kimi-rollouts-v3-20261026"
SELECTION = S / "sft/corpus/kimi-pass-v2-20261023/selection/kimi_pass_v2.jsonl"
KIMI_1M_FEATURES = V3 / "features" / "kimi.jsonl"
SLOT_KEYS = ("source", "task_id", "run", "repeat", "slot", "status", "reward", "f2p", "p2p", "attempt_dir", "origin")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def runs():
    rows = [json.loads(line) for line in SELECTION.read_text().splitlines() if line.strip()]
    return {row["task_id"]: row.get("run") for row in rows}


def _from_results(paths, source, field="records"):
    out = []
    run_of = runs()
    for path in paths:
        data = read_json(path)
        for task_id, records in ((data or {}).get(field) or (data or {}).get("records") or {}).items():
            for record in records:
                out.append({"source": source, "task_id": task_id, "run": run_of.get(task_id),
                            "repeat": int(record["slot"].rsplit("-r", 1)[1]), "slot": record["slot"],
                            "status": record["status"], "reward": record.get("reward"), "f2p": record.get("f2p"),
                            "p2p": record.get("p2p"), "attempt_dir": record.get("attempt_dir"), "origin": str(path)})
    return out


# ---------------------------------------------------------------- kimi4-all: every graded rollout of this batch once
def campaign_of(attempt_dir):
    """.../probes/kimi/<campaign>/models/kimi/attempts/<i>-<r>/attempt-<n> -> <campaign>."""
    return Path(attempt_dir).parents[4].name if attempt_dir else None


def v4_rollouts(source="kimi4-all") -> list:
    out = []
    run_of = runs()
    for path in sorted(KR.glob("results/kimi-*-g*.json")):
        data = read_json(path) or {}
        taskset, group = data.get("taskset"), data.get("group")
        final_dirs = {r.get("attempt_dir") for records in (data.get("records") or {}).values() for r in records}
        for task_id, records in (data.get("first_attempt_records") or data.get("records") or {}).items():
            for record in records:
                out.append({"source": source, "task_id": task_id, "run": run_of.get(task_id),
                            "repeat": int(record["slot"].rsplit("-r", 1)[1]), "slot": record["slot"],
                            "status": record["status"], "reward": record.get("reward"), "f2p": record.get("f2p"),
                            "p2p": record.get("p2p"), "attempt_dir": record.get("attempt_dir"), "origin": str(path),
                            "taskset": taskset, "group": group, "role": "first",
                            "final": record.get("attempt_dir") in final_dirs,
                            "campaign": campaign_of(record.get("attempt_dir")),
                            "ended_by_broken_reply": bool(record.get("ended_by_broken_reply")),
                            "last_reply_kind": record.get("last_reply_kind"), "agent_error": record.get("agent_error"),
                            "verifier_error": record.get("verifier_error"),
                            "patch_chars_results": record.get("patch_chars")})
        for task_id, entries in (data.get("broken_reply_reruns") or {}).items():
            for entry in entries:
                attempt = entry.get("rerun_attempt")
                final = next((r for r in (data.get("records") or {}).get(task_id, [])
                              if r.get("attempt_dir") == attempt), None)
                terminal = read_json(Path(attempt) / "terminal.json", {}) if attempt else {}
                f2p = final.get("f2p") if final else f"{terminal.get('f2p_passed')}/{terminal.get('f2p_total')}"
                p2p = final.get("p2p") if final else f"{terminal.get('p2p_passed')}/{terminal.get('p2p_total')}"
                out.append({"source": source, "task_id": task_id, "run": run_of.get(task_id),
                            "repeat": int(entry["slot"].rsplit("-r", 1)[1]), "slot": entry["slot"],
                            "status": entry.get("rerun_status") or "infra",
                            "reward": final.get("reward") if final else terminal.get("reward"),
                            "f2p": f2p, "p2p": p2p, "attempt_dir": attempt, "origin": str(path),
                            "taskset": taskset, "group": group, "role": "rerun",
                            "final": bool(entry.get("replaced")) and attempt in final_dirs,
                            "campaign": entry.get("rerun_campaign") or campaign_of(attempt),
                            "ended_by_broken_reply": bool(entry.get("rerun_ended_by_broken_reply")),
                            "last_reply_kind": (final or {}).get("last_reply_kind"),
                            "agent_error": (final or {}).get("agent_error", terminal.get("opencode_run_error_type")),
                            "verifier_error": (final or {}).get("verifier_error", terminal.get("verifier_error")),
                            "patch_chars_results": (final or {}).get("patch_chars"),
                            "rerun_slot": entry.get("rerun_slot"), "rerun_of": entry.get("original_attempt"),
                            "rerun_outcome": entry.get("outcome")})
    return out


# ---------------------------------------------------------------- kimi-1m: the earlier 1M Kimi-K3 rollouts (50 tasks)
def kimi_1m(source="kimi-1m") -> list:
    out = []
    for line in KIMI_1M_FEATURES.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        slot = {key: row.get(key) for key in SLOT_KEYS}
        slot["source"] = source
        attempt = Path(row["attempt_dir"]) if row.get("attempt_dir") else None
        slot["campaign"] = Path(row["origin"]).name if row.get("origin") else None
        slot["all_attempts"] = [str(p) for p in sorted(
            (p for p in attempt.parent.glob("attempt-*") if p.is_dir() and p.name.split("-", 1)[1].isdigit()),
            key=lambda p: int(p.name.split("-", 1)[1]))] if attempt else []
        out.append(slot)
    return out


def slots(source):
    if source == "kimi4":
        return _from_results(sorted(KR.glob("results/kimi-*-g*.json")), source)
    if source == "kimi4-first":
        return _from_results(sorted(KR.glob("results/kimi-*-g*.json")), source, field="first_attempt_records")
    if source == "kimi4-all":
        return v4_rollouts(source)
    if source == "kimi4-pilot":
        return _from_results(sorted(p for p in (KR / "results/kimi-pilot.json", KR / "results/kimi-baihe-pilot.json")
                                    if p.exists()), source)
    if source == "kimi3":
        return _from_results(sorted(V3.glob("results/kimi-*-g*.json")), source)
    if source == "ctrl-old":  # v3 control arm: previous Kimi settings (no sentence, 1M window)
        return _from_results([V3 / "results/kimi-ctrl-old.json"] if (V3 / "results/kimi-ctrl-old.json").exists() else [],
                             source)
    if source == "kimi-1m":
        return kimi_1m(source)
    raise ValueError(f"unknown source {source}")
