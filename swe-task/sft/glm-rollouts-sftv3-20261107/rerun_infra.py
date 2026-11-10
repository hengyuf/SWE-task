#!/usr/bin/env python3
"""glm-rollouts-sftv3: re-run the slots that ended without a grade (user 2026-11-08: "rerun the 3 ungraded slots").

    (cd swe-task/synth && atp/runtime/controller/bin/python3 -B <GR>/rerun_infra.py --base-url http://127.0.0.1:<port>/v1
        [--dry-run])

Every final record whose status is neither pass nor fail in the 8 campaign results (results/glmlocal-<campaign>.json) is
re-run ONCE in a new ATP campaign probes/glmlocal/infra-rerun1-<sandbox>-n<k> (k = such slots per task, one rollout
each), with driver.py's model spec, stock prompt, sandbox credential, task digests and sandbox-failure re-run rule.
Writes results/glmlocal-infra-rerun1.json = the new records + the mapping to the original (campaign, task, slot,
attempt). The campaign results files are not modified; final_report.py applies the re-runs (provenance kept).
Started by rerun_infra.sbatch (router on the glm53-serve-ta-* servers); resumable like a driver leg.
"""
import argparse
import asyncio
import collections
import importlib.util
import json
import os
from pathlib import Path
import time

GR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("gr_driver", GR / "driver.py")
driver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(driver)  # sets up the synth import path, as running driver.py does

RERUN = "infra-rerun1"
OUT = GR / "results" / f"{driver.ALIAS}-{RERUN}.json"
CAMPAIGNS = [f"hengyuf-g{g}" for g in range(4)] + [f"baihe-g{g}" for g in range(4)]


def ungraded_slots():
    """[(campaign, task_id, record)] for every final record without a pass/fail grade."""
    out = []
    for name in CAMPAIGNS:
        data = json.loads((GR / "results" / f"{driver.ALIAS}-{name}.json").read_text())
        for task_id, records in data["records"].items():
            out += [(name, task_id, record) for record in records if record["status"] not in ("pass", "fail")]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="this leg's router, http://127.0.0.1:<port>/v1")
    parser.add_argument("--max-wait-backend", type=float, default=12 * 3600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(marker in args.base_url for marker in driver.HUB_MARKERS):
        raise SystemExit("--base-url must be our own router, never the NVIDIA hub")
    tasks = {t["task_id"]: t for t in driver.load_tasks()}
    slots = ungraded_slots()
    groups = collections.defaultdict(lambda: collections.defaultdict(list))  # (sandbox, k) -> task -> [(campaign, record)]
    per_task = collections.Counter(task_id for _, task_id, _ in slots)
    for campaign, task_id, record in slots:
        sandbox = driver.SANDBOX_OF[campaign.rsplit("-g", 1)[0]]
        groups[(sandbox, per_task[task_id])][task_id].append((campaign, record))
    plan = {f"{RERUN}-{sandbox}-n{k}": {task_id: [{"campaign": c, "slot": r["slot"], "status": r["status"],
                                                   "attempt_dir": r.get("attempt_dir")} for c, r in entries]
                                        for task_id, entries in sorted(by_task.items())}
            for (sandbox, k), by_task in sorted(groups.items())}
    print(json.dumps({"slots": len(slots), "campaigns": plan}, indent=1))
    if args.dry_run or not slots:
        return
    changed = [t for t in per_task if driver.digests(Path(tasks[t]["task_dir"])) != tasks[t]["digests"]]
    if changed:
        raise SystemExit(f"task files changed since tasks.json was frozen: {changed}")
    assert driver.SETTINGS["temperature"] == 1.0 and driver.SETTINGS["top_p"] == 0.95
    os.environ["GLM53_BASE_URL"] = args.base_url
    os.environ["GR_PROMPT"] = driver.PROMPT
    health = driver.wait_for_backend(args.base_url, args.max_wait_backend)
    driver.SETTINGS["models"][driver.ALIAS] = dict(driver.SPEC)
    driver.rollouts.SYNTH = GR
    previous = json.loads(OUT.read_text()) if OUT.exists() else {}
    results = dict(previous.get("campaigns") or {})
    for (sandbox, k), by_task in sorted(groups.items()):
        name = f"{RERUN}-{sandbox}-n{k}"
        if name in results:
            continue
        probe_dir = GR / "probes" / driver.ALIAS / name
        legs = probe_dir / "gr-legs.jsonl"
        if legs.exists():
            earlier = [json.loads(line) for line in legs.read_text().splitlines() if line.strip()]
            assert all(leg["sandbox"] == sandbox for leg in earlier), f"{probe_dir}: earlier legs used another sandbox"
        os.environ["GR_SANDBOX"] = sandbox
        driver.rollouts.PROBE_MAX_CONCURRENCY = max(1, len(by_task) * k)
        task_ids = sorted(by_task)
        driver.log(f"glm-rollouts: {name}: re-running {len(task_ids) * k} ungraded slot(s) via {args.base_url} "
                   f"(router {health}), sandbox {driver.SANDBOX_CREDENTIAL[sandbox]}")
        driver.leg_record(campaign=name, event="start", reason="ungraded slots (user 2026-11-08)",
                          slots=plan[name], sandbox_credential=driver.SANDBOX_CREDENTIAL[sandbox])
        records = asyncio.run(driver.rollouts.probe(probe_dir, [Path(tasks[t]["task_dir"]) for t in task_ids],
                                                    models=(driver.ALIAS,), repeats=k))
        first = {t: [driver.compact(r) for r in rs] for t, rs in records.items()}
        recovered = driver.recover(name, records, {t: Path(tasks[t]["task_dir"]) for t in task_ids}, "")
        mapping = []
        for task_id in task_ids:
            for (campaign, original), new in zip(by_task[task_id], records[task_id]):
                mapping.append({"campaign": campaign, "task_id": task_id, "slot": original["slot"],
                                "original_status": original["status"], "original_attempt": original.get("attempt_dir"),
                                "rerun_campaign": name, "rerun_slot": new["slot"], "rerun_status": new["status"],
                                "rerun_attempt": new.get("attempt_dir"), "rerun": driver.compact(new)})
        results[name] = {"sandbox": sandbox, "stats": driver.stats(records), "first_attempt_records": first,
                         "records": {t: [driver.compact(r) for r in rs] for t, rs in records.items()},
                         "sandbox_failure_reruns": recovered, "mapping": mapping}
        driver.write_json_atomic(OUT, {"request": "user 2026-11-08: rerun the 3 ungraded slots", "alias": driver.ALIAS,
                                       "spec": driver.SPEC, "prompt": driver.PROMPT,
                                       "sandbox_failure_rule": driver.INFRA_RERUN.pattern,
                                       "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                       "job": os.environ.get("SLURM_JOB_ID"), "campaigns": results})
        driver.leg_record(campaign=name, event="complete", stats=driver.stats(records),
                          mapping=[{k_: m[k_] for k_ in ("campaign", "task_id", "slot", "rerun_status")} for m in mapping])
        driver.log(f"glm-rollouts: {name} done: {json.dumps(driver.stats(records))}")


if __name__ == "__main__":
    main()
