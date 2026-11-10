"""kimi-rollouts-v5 (2026-10-28; final-summary sources kimi5-all / kimi5-build added 2026-10-29): the rollout slots of
this batch (and of kimi-rollouts-v4 / v3, read-only) for features.py / window.py / degenerate.py / phase_report.py
(kimi-rollouts-v4's sources.py with the v5 sources added and v4's read from its own directory). Read-only on every
input (results files, probe campaigns, the synth runs).

slots(source) -> list of {source, task_id, run, repeat, slot, status (pass|fail|infra), reward, f2p, p2p,
                          attempt_dir (the slot's final attempt, as rollouts.probe_records() chose it), origin, ...}
Sources:
  kimi5          this batch (official2 tasks, stock prompt + 262k): results/kimi-official2-g<K>.json records = the FINAL
                 slots, i.e. a slot ended by a broken reply is its one re-run when that re-run was graded (driver.py)
  kimi5-first    this batch's first attempts: the same files' first_attempt_records (before the broken-reply re-runs)
  kimi5-all      (2026-10-29) EVERY graded rollout of this batch once: the 165 first attempts (role "first") plus the
                 broken-reply re-runs (role "rerun"). Extra fields: role; final (this rollout is its slot's final
                 record); campaign (probe campaign dir, e.g. official2-g0 / official2-g0-brk1); ended_by_broken_reply and
                 last_reply_kind (driver.py's brokenreply.py verdict, from the results file); agent_error; verifier_error;
                 re-runs also rerun_slot (the slot label inside the -brk campaign) and rerun_of (the original attempt)
  kimi5-build    (2026-10-29) the BUILD-TIME Kimi-K3 probe rollouts of the same 33 tasks: the synth pipeline's own probe
                 campaign of the round that scored the task (<round>/probe-*/, which holds tasks/<task_id>), Kimi slots
                 kimi-r0..r4 = the "5 earlier Kimi slots" of each task. Those campaigns ran with a 1M window
                 (settings.context_window 1,048,576 and no window of kimi's own; v5 sets 262,144), the same model id,
                 reasoning_effort max, OpenCode 1.17.11 and stock prompt (verify_pilot.py --previous checked a build
                 attempt against v5's pilot: only the provider limits differ). Per slot, read-only (no ATP code is
                 called: ATP's slot_state writes validity caches into attempt dirs):
                   attempt_dir   the first attempt that ATP counts as completed (terminal.json evaluation_completed and
                                 no failure in the generation-validity-v2-r7.json cache that ATP wrote when the build
                                 scored the probe; a missing cache counts as valid and is flagged validity_cached False)
                                 = deepswe_campaign.slot_state(); else the latest attempt (status infra)
                   export_dir    started.json agent_artifacts_from (a regrade attempt's generation) or the attempt
                   status        score()'s graded status from <round>/<cand>/rollouts/index.md (written by
                                 task_builder.score() from rollouts.probe_records(), verifier faults turned to infra);
                                 atp_status = the replica's own pass / fail / infra
                   index_match   the replica's status, F2P, P2P and patch chars equal that index.md row
                   all_attempts  every attempt dir of the slot (degenerate.py scans all of them)
                 When a round has several probe campaigns holding the task (kapacitor c05-r1b1: probe-25dbb06e, a
                 partial earlier one, and probe-3380a4da), the campaign whose 5 slots match index.md is used.
  kimi4          kimi-rollouts-v4 (v2 tasks, same settings): V4/results/kimi-*-g*.json (final slots)
  kimi4-first    kimi-rollouts-v4's first attempts
  kimi4-pilot    kimi-rollouts-v4's 1-rollout pilots: V4/results/kimi-pilot.json, V4/results/kimi-baihe-pilot.json
  kimi3          kimi-rollouts-v3's sentence arm (256k + one reasoning sentence): V3/results/kimi-*-g*.json
  ctrl-old       kimi-rollouts-v3's control arm (previous Kimi settings: 1M, stock prompt): V3/results/kimi-ctrl-old.json
The baselines on the 50 tasks are teacher-rollouts-v2's features files (V3/features/kimi.jsonl = previous Kimi-K3, 1M
window, no instruction; V3/features/sc14.jsonl = base Nemotron sc14).
"""
import json
from pathlib import Path

S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
KR = Path(__file__).resolve().parent
V3 = S / "sft/kimi-rollouts-v3-20261026"
V4 = S / "sft/kimi-rollouts-v4-stock256k-20261028"
SELECTION = S / "sft/corpus/kimi-pass-v2-20261023/selection/kimi_pass_v2.jsonl"
TASKS_FILE = KR / "official2-tasks.json"
BUILD_MODEL = "kimi"  # the teacher alias in the synth probe campaigns (synth common.TEACHER)
REPEATS = 5


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def official2_tasks() -> list:
    return (read_json(TASKS_FILE, {}) or {}).get("tasks", [])


def runs():
    """{task_id: run}: the v2 SFT selection (v3 / v4 tasks) plus this batch's official2 task list."""
    rows = [json.loads(line) for line in SELECTION.read_text().splitlines() if line.strip()]
    out = {row["task_id"]: row.get("run") for row in rows}
    out.update({task["task_id"]: task["run"] for task in official2_tasks()})
    return out


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


# ---------------------------------------------------------------- kimi5-all: every graded rollout of this batch once
def _campaign_of(attempt_dir) -> str:
    """probes/kimi/<campaign>/models/kimi/attempts/<i>-<r>/attempt-<n> -> <campaign>."""
    return Path(attempt_dir).parents[4].name if attempt_dir else None


def v5_rollouts(source="kimi5-all") -> list:
    out = []
    run_of = runs()
    for path in sorted(KR.glob("results/kimi-official2-g*.json")):
        data = read_json(path) or {}
        final_dirs = {r.get("attempt_dir") for records in (data.get("records") or {}).values() for r in records}
        for task_id, records in (data.get("first_attempt_records") or data.get("records") or {}).items():
            for record in records:
                out.append({"source": source, "task_id": task_id, "run": run_of.get(task_id),
                            "repeat": int(record["slot"].rsplit("-r", 1)[1]), "slot": record["slot"],
                            "status": record["status"], "reward": record.get("reward"), "f2p": record.get("f2p"),
                            "p2p": record.get("p2p"), "attempt_dir": record.get("attempt_dir"), "origin": str(path),
                            "role": "first", "final": record.get("attempt_dir") in final_dirs,
                            "campaign": _campaign_of(record.get("attempt_dir")),
                            "ended_by_broken_reply": bool(record.get("ended_by_broken_reply")),
                            "last_reply_kind": record.get("last_reply_kind"), "agent_error": record.get("agent_error"),
                            "verifier_error": record.get("verifier_error")})
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
                            "role": "rerun", "final": bool(entry.get("replaced")) and attempt in final_dirs,
                            "campaign": entry.get("rerun_campaign") or _campaign_of(attempt),
                            "ended_by_broken_reply": bool(entry.get("rerun_ended_by_broken_reply")),
                            "last_reply_kind": (final or {}).get("last_reply_kind"),
                            "agent_error": (final or {}).get("agent_error", terminal.get("opencode_run_error_type")),
                            "verifier_error": (final or {}).get("verifier_error", terminal.get("verifier_error")),
                            "rerun_slot": entry.get("rerun_slot"), "rerun_of": entry.get("original_attempt"),
                            "rerun_outcome": entry.get("outcome")})
    return out


# ---------------------------------------------------------------- kimi5-build: the build-time 1M Kimi probes
def _index_md(path: Path) -> dict:
    """{slot: {status, reward, f2p, p2p, tool_calls, patch_chars, agent_error}} of a rollouts/index.md (score())."""
    rows = {}
    try:
        text = path.read_text()
    except OSError:
        return rows
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 8 and cells[0] and cells[0] != "slot" and not cells[0].startswith("-"):
            rows[cells[0]] = dict(zip(("status", "reward", "f2p", "p2p", "tool_calls", "patch_chars", "agent_error"),
                                      cells[1:]))
    return rows


def _attempts(slot_dir: Path) -> list:
    return sorted((p for p in slot_dir.glob("attempt-*") if p.is_dir() and p.name.split("-", 1)[1].isdigit()),
                  key=lambda p: int(p.name.split("-", 1)[1]))


def generation_dir(attempt) -> Path:
    """Where an attempt's generation artifacts (export.json, http.jsonl) live: started.json agent_artifacts_from (a
    regrade attempt reuses an earlier attempt's generation), else the attempt itself."""
    started = read_json(Path(attempt) / "started.json", {}) or {}
    return Path(started.get("agent_artifacts_from") or attempt)


def _completed(attempt: Path):
    """(completed, terminal, validity_cached): ATP's outcome() == "completed", read-only (see the module docstring)."""
    terminal = read_json(attempt / "terminal.json", {}) or {}
    if terminal.get("evaluation_completed") is not True:
        return False, terminal, None
    cache = read_json(generation_dir(attempt) / "generation-validity-v2-r7.json")
    return (cache is None or cache.get("failure") is None), terminal, cache is not None


def _build_slot(probe: Path, index: int, repeat: int) -> dict:
    attempts = _attempts(probe / "models" / BUILD_MODEL / "attempts" / f"{index}-{repeat}")
    chosen, terminal, cached = None, {}, None
    for attempt in attempts:
        done, data, has_cache = _completed(attempt)
        if done:
            chosen, terminal, cached = attempt, data, has_cache
            break
    if chosen is None and attempts:
        chosen = attempts[-1]
        terminal = read_json(chosen / "terminal.json", {}) or {}
    completed = cached is not None
    status = ("pass" if terminal.get("reward") == 1 else "fail") if completed else "infra"
    patch = terminal.get("model_patch") or ""
    return {"attempt_dir": str(chosen) if chosen else None, "atp_status": status, "reward": terminal.get("reward"),
            "f2p": f"{terminal.get('f2p_passed')}/{terminal.get('f2p_total')}",
            "p2p": f"{terminal.get('p2p_passed')}/{terminal.get('p2p_total')}", "patch_chars_terminal": len(patch),
            "validity_cached": cached, "all_attempts": [str(a) for a in attempts],
            "export_dir": str(generation_dir(chosen)) if chosen else None,
            "agent_error": terminal.get("opencode_run_error_type") or terminal.get("failure_reason"),
            "verifier_error": terminal.get("verifier_error")}


def _matches(slot: dict, row: dict) -> bool:
    return bool(row) and row.get("status") == slot["atp_status"] and row.get("f2p") == slot["f2p"] \
        and row.get("p2p") == slot["p2p"] and str(row.get("patch_chars")) == str(slot["patch_chars_terminal"])


def build_probe_slots(source="kimi5-build") -> list:
    out = []
    for task in official2_tasks():
        task_id = task["task_id"]
        metrics_path = Path(task["candidate_metrics"][0])  # <build>/<round>/<cand>/metrics.json
        scored = metrics_path.parent
        index_rows = _index_md(scored / "rollouts" / "index.md")
        candidates = []
        for probe in sorted(p for p in scored.parent.glob("probe-*") if (p / "tasks" / task_id).is_dir()):
            manifest = read_json(probe / "manifest.json", {}) or {}
            indices = [t["_ng_task_index"] for t in manifest.get("tasks", []) if t.get("task_id") == task_id]
            if not indices:
                continue
            settings = manifest.get("settings") or {}
            model = (settings.get("models") or {}).get(BUILD_MODEL) or {}
            slots = [_build_slot(probe, indices[0], repeat) for repeat in range(settings.get("repeats", REPEATS))]
            matched = sum(_matches(s, index_rows.get(f"kimi-r{r}")) for r, s in enumerate(slots))
            candidates.append((matched, manifest.get("created_at") or 0, probe, indices[0], settings, model, slots))
        if not candidates:
            continue
        matched, _, probe, index, settings, model, slots = max(candidates, key=lambda c: (c[0], c[1]))
        for repeat, slot in enumerate(slots):
            row = index_rows.get(f"kimi-r{repeat}") or {}
            status = row.get("status") or slot["atp_status"]
            out.append({"source": source, "task_id": task_id, "run": task.get("run"), "repeat": repeat,
                        "slot": f"kimi-r{repeat}", "status": status, **slot,
                        "origin": str(metrics_path), "campaign": f"{task.get('run')}/{probe.name}",
                        "probe_dir": str(probe), "task_index": index, "index_md": str(scored / "rollouts" / "index.md"),
                        "index_row": row, "index_match": _matches(slot, row),
                        "verifier_fault": status == "infra" and slot["atp_status"] in ("pass", "fail"),
                        "probe_candidates": len(candidates), "probe_slots_matched": matched,
                        "context_window": model.get("context_window") or settings.get("context_window"),
                        "model_id": model.get("id"),
                        "reasoning_effort": (model.get("sampling_overrides") or {}).get("reasoning_effort"),
                        "opencode_version": settings.get("opencode_version")})
    return out


def slots(source):
    if source == "kimi5":
        return _from_results(sorted(KR.glob("results/kimi-official2-g*.json")), source)
    if source == "kimi5-first":
        return _from_results(sorted(KR.glob("results/kimi-official2-g*.json")), source, field="first_attempt_records")
    if source == "kimi5-all":
        return v5_rollouts(source)
    if source == "kimi5-build":
        return build_probe_slots(source)
    if source == "kimi4":
        return _from_results(sorted(V4.glob("results/kimi-*-g*.json")), source)
    if source == "kimi4-first":
        return _from_results(sorted(V4.glob("results/kimi-*-g*.json")), source, field="first_attempt_records")
    if source == "kimi4-pilot":
        return _from_results(sorted(p for p in (V4 / "results/kimi-pilot.json", V4 / "results/kimi-baihe-pilot.json")
                                    if p.exists()), source)
    if source == "kimi3":
        return _from_results(sorted(V3.glob("results/kimi-*-g*.json")), source)
    if source == "ctrl-old":  # v3 control arm: previous Kimi settings (no sentence, 1M window)
        return _from_results([V3 / "results/kimi-ctrl-old.json"] if (V3 / "results/kimi-ctrl-old.json").exists() else [],
                             source)
    raise ValueError(f"unknown source {source}")
