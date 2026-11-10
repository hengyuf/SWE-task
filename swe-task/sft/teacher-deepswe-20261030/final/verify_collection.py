#!/usr/bin/env python3
"""teacher-deepswe collect phase (2026-10-31): independent checks of collect.py's outputs (read-only; writes only
--out, default analysis/collect-verify.json).

1. Slot accounting: dataset/slots.jsonl has every planned slot once (113 tasks x 3 models x 5 = 1,695) + the 3 pilots;
   5 per model per task; the graded / ungraded split equals ATP's own slot states (analysis/atp-slot-states.json) and
   the last PROGRESS-rows.json row (progress.py's independent scan).
2. Rows: dataset/index.jsonl rows = dataset files' line counts (gz decompressed and counted) = MANIFEST rows; every
   row's status agrees with its attempt's terminal.json reward (re-read here); no duplicate (model, task, slot); every
   row has a transcript (output_items > 0) and an attempt under its own campaign.
3. Files: every dataset file's sha256 / bytes = MANIFEST.json.
4. Cost: the journal cost per model (collect.py) vs progress.py's export-token estimate over the same ended attempts
   (progress-cache.json est_cost), and per hub key.
Status overrides (2026-11-01, OPERATOR_NOTE 11:15 / 14:15 PST): a slot in MANIFEST.json status_overrides (collect.py's
GRADED_FAIL_OVERRIDES) is a graded fail in the ledger, but ATP calls it needs_review and progress.py counts it as final
infra, so check 1 compares ATP and progress.py with the ledger's graded count minus the overrides. Ungraded slots on the
ledger side are cancelled / infra / active / final_infra (a complete campaign's final infra slot is ATP needs_review or
retryable too) plus the overrides.
Final collection (2026-11-02, final/collect_final_20261102.py; GLM cancelled by the user): the ledger states
"interrupted" (ATP retryable) and "cancelled_in_flight" (ATP "active"; progress.py's "active") count as started-ungraded
too; progress.py's ended_ungraded = cancelled + interrupted + infra. 5. complete campaigns vs their results files
(pass / fail / infra per campaign, every final record = its first-attempt record, no re-run or recovery). 6. a model
cancelled by the user (MANIFEST model_status partial_cancelled_by_user) has main's STOP file in every
babysit/<model>-<campaign>/ and no results file.

    /usr/bin/python3 -I -B final/verify_collection.py [--out PATH]
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

TD = Path(__file__).resolve().parent.parent
MODELS = ("kimi", "glm", "deepseek")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=TD / "analysis" / "collect-verify.json")
    args = parser.parse_args()
    problems = []
    out = {}
    dataset = TD / "dataset"
    manifest = json.loads((dataset / "MANIFEST.json").read_text())
    ledger = [json.loads(line) for line in (dataset / "slots.jsonl").read_text().splitlines() if line.strip()]
    index = [json.loads(line) for line in (dataset / "index.jsonl").read_text().splitlines() if line.strip()]
    waves = json.loads((TD / "tasks" / "waves.json").read_text())

    # 1. slot accounting
    wave = [e for e in ledger if e["campaign"] != "pilot"]
    pilots = [e for e in ledger if e["campaign"] == "pilot"]
    out["ledger"] = {"rows": len(ledger), "wave_slots": len(wave), "pilots": len(pilots)}
    if len(wave) != 1695 or len(pilots) != 3:
        problems.append(f"ledger has {len(wave)} wave slots and {len(pilots)} pilots (want 1695 and 3)")
    per = collections.Counter((e["model"], e["task_id"]) for e in wave)
    bad = {f"{m} {t}": per[(m, t)] for m in MODELS for t in waves["wave1"] + waves["wave2"] if per[(m, t)] != 5}
    if bad:
        problems.append(f"slots per model per task != 5: {bad}")
    keys = collections.Counter((e["model"], e["campaign"], e["task_id"], e["repeat"]) for e in wave)
    if any(v > 1 for v in keys.values()):
        problems.append("duplicate slots in the ledger")
    states = collections.Counter((e["model"], e["state"]) for e in wave)
    out["states"] = {m: {s: n for (mm, s), n in states.items() if mm == m} for m in MODELS}
    atp = read_json(TD / "analysis" / "atp-slot-states.json")
    atp_counts = collections.Counter()
    for key, campaign in atp["campaigns"].items():
        model, name = key.split("/", 1)
        if name == "pilot":
            continue
        for slot in campaign["slots"]:
            atp_counts[(model, slot["status"])] += 1
    out["atp_states"] = {m: {s: n for (mm, s), n in atp_counts.items() if mm == m} for m in MODELS}
    overrides = collections.Counter(o["model"] for o in manifest.get("status_overrides") or [])
    out["status_overrides"] = manifest.get("status_overrides") or []
    for model in MODELS:
        graded = sum(e["status"] in ("pass", "fail") for e in wave if e["model"] == model) - overrides[model]
        if graded != atp_counts[(model, "completed")]:
            problems.append(f"{model}: {graded} graded in the ledger (status overrides left out) vs ATP completed "
                            f"{atp_counts[(model, 'completed')]}")
        ungraded_started = sum(e["state"] in ("cancelled", "infra", "active", "final_infra", "interrupted",
                                              "cancelled_in_flight")
                               for e in wave if e["model"] == model) + overrides[model]
        atp_other = sum(n for (m, s), n in atp_counts.items() if m == model and s not in ("completed", "pending"))
        if ungraded_started != atp_other:
            problems.append(f"{model}: {ungraded_started} started-ungraded slots vs ATP {atp_other}")
    rows = read_json(TD / "PROGRESS-rows.json")
    last = (rows or [])[-1] if isinstance(rows, list) and rows else None
    if isinstance(rows, dict):
        last = (rows.get("rows") or [None])[-1]
    out["progress_last_row"] = {k: last.get(k) for k in ("time", "graded", "graded_total", "pass", "ended_ungraded",
                                                         "active", "infra_final") if k in last} if last else None
    if last:
        ledger_graded = sum(e["status"] in ("pass", "fail") for e in wave) - sum(overrides.values())
        if last.get("graded_total") is not None and last["graded_total"] != ledger_graded:
            problems.append(f"progress.py graded_total {last['graded_total']} vs ledger {ledger_graded} (status "
                            "overrides left out)")
        if last.get("pass") is not None and last["pass"] != sum(e["status"] == "pass" for e in wave):
            problems.append(f"progress.py pass {last['pass']} vs ledger {sum(e['status'] == 'pass' for e in wave)}")
        ended_states = ("cancelled", "infra", "interrupted")
        if last.get("ended_ungraded") is not None and last["ended_ungraded"] != sum(
                e["state"] in ended_states for e in wave):
            problems.append(f"progress.py ended_ungraded {last['ended_ungraded']} vs ledger "
                            f"{sum(e['state'] in ended_states for e in wave)}")
        in_flight = sum(e["state"] in ("cancelled_in_flight", "active") for e in wave)
        if last.get("active") is not None and last["active"] != in_flight:
            problems.append(f"progress.py active {last['active']} vs ledger in flight (cancelled_in_flight / active) "
                            f"{in_flight}")

    # 2. rows
    counts = collections.Counter(row["file"] for row in index)
    seen = collections.Counter((row["model"], row["task_id"], row["slot"], row["campaign"]) for row in index)
    if any(v > 1 for v in seen.values()):
        problems.append("duplicate dataset rows")
    status_checked = 0
    final_infra_rows = []
    for row in index:
        terminal = read_json(Path(row["attempt_dir"]) / "terminal.json", {}) or {}
        if row["status"] == "infra":  # a complete campaign's final infra slot (driver.py's record, ATP not graded)
            if not row.get("campaign_complete") or terminal.get("evaluation_completed") is True:
                problems.append(f"infra row outside a complete campaign or with a completed evaluation "
                                f"(evaluation_completed {terminal.get('evaluation_completed')}): {row['attempt_dir']}")
            final_infra_rows.append(row["attempt_dir"])
            status_checked += 1
        else:
            want = "pass" if terminal.get("reward") == 1 else "fail"
            if terminal.get("evaluation_completed") is not True or row["status"] != want:
                problems.append(f"status {row['status']} vs terminal reward {terminal.get('reward')} "
                                f"(evaluation_completed {terminal.get('evaluation_completed')}): {row['attempt_dir']}")
            status_checked += 1
        if not row.get("output_items"):
            problems.append(f"row without transcript: {row['attempt_dir']}")
        parts = Path(row["attempt_dir"]).parts
        if parts[-4] != row["model"] or parts[-5] != "models" or parts[-6] != row["campaign"]:
            problems.append(f"attempt dir outside its campaign: {row['attempt_dir']}")
    gz_lines = {}
    for name in ("kimi.jsonl.gz", "glm.jsonl.gz", "deepseek.jsonl.gz", "pilots.jsonl.gz"):
        with gzip.open(dataset / name, "rt", encoding="utf-8") as stream:
            gz_lines[name] = sum(1 for _ in stream)
        if gz_lines[name] != counts[name] or gz_lines[name] != manifest["rows"][name.split(".")[0]]:
            problems.append(f"{name}: {gz_lines[name]} lines vs index {counts[name]} vs manifest "
                            f"{manifest['rows'][name.split('.')[0]]}")
    out["rows"] = {"gz_lines": gz_lines, "index_rows": len(index), "status_checked": status_checked,
                   "final_infra_rows": final_infra_rows}

    # 3. files
    for name, meta in manifest["files"].items():
        path = dataset / name
        if path.stat().st_size != meta["bytes"] or sha256_file(path) != meta["sha256"]:
            problems.append(f"{name} differs from MANIFEST.json")
    out["files_checked"] = sorted(manifest["files"])

    # 4. cost: journal (collect.py) vs export-token estimate (progress.py cache) over the same attempts
    cache = read_json(TD / "progress-cache.json", {}) or {}
    cost = {}
    for model in MODELS + ("pilot",):
        entries = pilots if model == "pilot" else [e for e in wave if e["model"] == model]
        journal = sum(e.get("slot_cost_usd") or 0.0 for e in entries)
        estimate, missing = 0.0, 0
        for e in entries:
            if not e.get("attempt_dir"):
                continue
            for attempt in sorted(Path(e["attempt_dir"]).parent.glob("attempt-*")):
                facts = cache.get(str(attempt)) or cache.get(str(attempt) + "#ended")
                if facts and facts.get("est_cost") is not None:
                    estimate += facts["est_cost"]
                else:
                    missing += 1
        by_key = collections.Counter()
        for e in entries:
            by_key.update(e.get("cost_by_key") or {})
        cost[model] = {"journal": round(journal, 2), "export_estimate": round(estimate, 2),
                       "estimate_over_journal": round(estimate / journal, 4) if journal else None,
                       "attempts_without_cached_estimate": missing,
                       "by_key": {k: round(v, 2) for k, v in by_key.items()}}
        # pilots: NEW_2 (10-30 21:34); wave legs: NEW_3 (10-30 21:45 .. 10-31 10:40), NEW_2 again from 10-31 18:15
        expected_keys = {"NVAPI_KEY_NEW_2"} if model == "pilot" else {"NVAPI_KEY_NEW_3", "NVAPI_KEY_NEW_2"}
        other = {k: v for k, v in by_key.items() if k not in expected_keys and v > 0}
        if other:
            problems.append(f"{model}: cost attributed to other keys than {sorted(expected_keys)}: {other}")
    out["cost"] = cost

    # 5. complete campaigns vs their results files; 6. models cancelled by the user
    status = manifest.get("model_status") or {}
    out["model_status"] = status
    provenance = {}
    for model in MODELS:
        for name in ("w1-g0", "w1-g1", "w1-g2", "w2-g0", "w2-g1"):
            result = read_json(TD / "results" / f"{model}-{name}.json")
            mine = [e for e in wave if e["model"] == model and e["campaign"] == name]
            stop = TD / "babysit" / f"{model}-{name}" / "STOP"
            if result is None:
                if status.get(model) == "complete":
                    problems.append(f"{model}-{name}: no results file but model_status complete")
                if status.get(model) == "partial_cancelled_by_user" and not stop.exists():
                    problems.append(f"{model}-{name}: cancelled by the user but no STOP file")
                provenance[f"{model}-{name}"] = {"results": False, "stop_file": stop.exists(),
                                                 "states": dict(collections.Counter(e["state"] for e in mine))}
                continue
            final = {"pass": 0, "fail": 0, "infra": 0}
            same = slots = 0
            firsts = {(t, r["slot"]): r.get("attempt_dir") for t, recs in (result.get("first_attempt_records") or {}).items()
                      for r in recs}
            for task_id, records in result["records"].items():
                for record in records:
                    final[record["status"] if record["status"] in final else "infra"] += 1
                    slots += 1
                    same += firsts.get((task_id, record["slot"])) == record.get("attempt_dir")
            mine_overrides = sum(1 for o in manifest.get("status_overrides") or [] if o["model"] == model
                                 and o["campaign"] == name)
            ledger_counts = {"pass": sum(e["status"] == "pass" for e in mine),
                             "fail": sum(e["status"] == "fail" for e in mine) - mine_overrides,
                             "infra": sum(e["state"] == "final_infra" for e in mine) + mine_overrides}
            if ledger_counts != final:
                problems.append(f"{model}-{name}: ledger {ledger_counts} (overrides as infra) vs results {final}")
            stats = result.get("stats_final") or {}
            if {k: stats.get(k) for k in final} != final:
                problems.append(f"{model}-{name}: results stats_final {stats} vs records {final}")
            if same != slots:
                problems.append(f"{model}-{name}: {slots - same} final records differ from their first-attempt record")
            if result.get("broken_reply_reruns") or result.get("recovered_budget_refusals"):
                problems.append(f"{model}-{name}: re-runs / recoveries recorded; check their provenance")
            provenance[f"{model}-{name}"] = {"results": True, "final": final, "slots": slots,
                                             "final_equals_first": same, "stop_file": stop.exists(),
                                             "broken_reply_reruns": len(result.get("broken_reply_reruns") or {}),
                                             "recovered_budget_refusals": len(result.get("recovered_budget_refusals") or {})}
    rerun_dirs = sorted(f"{p.parent.name}/{p.name}" for model in MODELS for p in (TD / "probes" / model).iterdir()
                        if p.is_dir() and ("-brk" in p.name or "-rec" in p.name))
    if rerun_dirs:
        problems.append(f"re-run / recovery probe dirs exist: {rerun_dirs}")
    out["campaign_provenance"] = provenance
    out["rerun_dirs"] = rerun_dirs
    out["problems"] = problems
    out["ok"] = not problems
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
