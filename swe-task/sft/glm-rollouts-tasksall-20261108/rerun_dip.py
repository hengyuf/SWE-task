#!/usr/bin/env python3
"""glm-rollouts-tasksall: re-run once the slots whose rollout hit ATP's 3 h agent timeout while the GPU fleet was down to
2 servers (2026-11-09 03:10-04:50: the first 8 batch servers expired before their replacements started, ~230 sessions
on 2 servers). 62 of the 66 agent timeouts up to 09:00 overlapped that window (the other hours: ~1 per 2-3 h), so those
grades say more about the fleet than about the model; the same policy as the sandbox-failure re-runs.

    (cd swe-task/synth && atp/runtime/controller/bin/python3 -B <GR2>/rerun_dip.py --base-url http://127.0.0.1:<port>/v1
        [--dry-run] [--only-done])

Candidates: a final record (results/glmlocal-<campaign>.json) whose attempt has an agent_interrupted event with
error_type "timeout" and whose attempt interval [started.json started_at, timeout event] overlaps the window. Each is
re-run ONCE in campaigns dip1-<h|b>-n<k>[-<i>] (k = such slots of that task, one campaign per sandbox owner and k; a
repeated task id goes to another campaign), with driver.py's model spec, stock prompt, sandbox credential, task
digests, staging wrapper and sandbox-failure re-run rule. Writes results/glmlocal-dip1.json (new records + mapping to
the original campaign/task/slot/attempt); the campaign results files are not modified; final_report.py applies it.
Campaigns without a results file yet are skipped: run it again later for them. Each invocation that finds new
candidates plans a ROUND (campaign names dip1r<round>-...), stored in the results file before anything runs, so a leg
that ends early resumes its round; slots planned in any earlier round are never re-run twice.
Started by rerun_dip.sbatch (router on the glm53-serve-ta-* servers).

Parallel mode (2026-11-09, for --name dip2 and later): `--plan-only` plans the round and exits; `--campaign NAME` runs
one planned campaign of the latest round and writes its result to results/<alias>-<name>-parts/<campaign>.json (one
CPU job per campaign, so the campaigns run side by side; no shared-file race); final_report.py reads the parts too.
`--tiers 1` (with --plan-only) plans only candidates of tier-1 campaigns.
"""
import argparse
import collections
import importlib.util
import json
import os
from pathlib import Path
import time

GR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("gr2_driver", GR / "driver.py")
driver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(driver)  # synth import path, staging wrapper hook, probe() with staging retries

NAME = "dip1"  # --name: each name has its own state file results/glmlocal-<name>.json, so two names can run at once
OUT = GR / "results" / f"{driver.ALIAS}-{NAME}.json"
INCLUDE_UNGRADED = False  # --include-ungraded: also re-run once every final record without a grade (status not pass/fail)
WINDOW = ("2026-11-09 03:10", "2026-11-09 04:50")  # local time (PST): fewer than 3 ready servers


def window_epoch():
    return tuple(time.mktime(time.strptime(text, "%Y-%m-%d %H:%M")) for text in WINDOW)


def timeout_interval(attempt: Path):
    """(started_at, timeout time) when the attempt ended in ATP's agent timeout, else None."""
    try:
        started = json.loads((attempt / "started.json").read_text())["started_at"]
        with open(attempt / "capture-events.jsonl") as stream:
            for line in stream:
                if '"agent_interrupted"' in line:
                    event = json.loads(line)
                    if event.get("error_type") == "timeout":
                        return started, event["time"]
    except (OSError, ValueError, KeyError):
        return None
    return None


def candidates(tiers=None):
    """[(campaign, task_key, task_id, record, (start, end))] of done campaigns, and the campaigns not done yet."""
    w0, w1 = window_epoch()
    _, campaigns = driver.load_plan()
    out, pending = [], []
    for name, campaign in campaigns.items():
        if tiers and campaign["tier"] not in tiers:
            continue
        path = GR / "results" / f"{driver.ALIAS}-{name}.json"
        if not path.exists():
            pending.append(name)
            continue
        data = json.loads(path.read_text())
        for task_id, records in data["records"].items():
            for record in records:
                if INCLUDE_UNGRADED and record["status"] not in ("pass", "fail"):
                    out.append((name, data["task_keys"][task_id], task_id, record, [None, None]))
                    continue
                if record["status"] != "fail" or not record.get("attempt_dir"):
                    continue
                interval = timeout_interval(Path(record["attempt_dir"]))
                if interval and interval[0] < w1 and interval[1] > w0:
                    out.append((name, data["task_keys"][task_id], task_id, record, interval))
    return out, pending


def plan_campaigns(slots, already, prefix):
    """{campaign name: {task_key: [(campaign, task_id, record, interval)]}} grouped by owner and slots per task;
    a task id that is already in a campaign goes to the next one of the same group."""
    tasks, _ = driver.load_plan()
    per_task = collections.defaultdict(list)
    for campaign, key, task_id, record, interval in slots:
        if (campaign, task_id, record["slot"]) in already:
            continue
        per_task[key].append((campaign, task_id, record, interval))
    groups = collections.defaultdict(list)  # (owner, k) -> [[key, ...], ...]
    for key in sorted(per_task):
        owner, k = tasks[key]["owner"], len(per_task[key])
        for chunk in groups[(owner, k)]:
            if all(tasks[other]["task_id"] != tasks[key]["task_id"] for other in chunk):
                chunk.append(key)
                break
        else:
            groups[(owner, k)].append([key])
    plan = {}
    for (owner, k), chunks in sorted(groups.items()):
        for i, chunk in enumerate(chunks):
            name = f"{prefix}-{owner[0]}-n{k}" + (f"-{i}" if i else "")
            plan[name] = {"owner": owner, "repeats": k,
                          "tasks": {key: [[c, tid, driver.compact(r) if "patch" in r else r, list(iv)]
                                          for c, tid, r, iv in per_task[key]] for key in chunk}}
    return plan


def write_state(state):
    state.update({"rule": f"final fail records with an ATP agent timeout whose attempt overlapped {WINDOW} (fleet down "
                          f"to 2 servers)" + (", and final records without a grade" if INCLUDE_UNGRADED else "")
                          + ", re-run once", "alias": driver.ALIAS, "spec": driver.SPEC,
                  "prompt": driver.PROMPT, "sandbox_failure_rule": driver.INFRA_RERUN.pattern,
                  "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "job": os.environ.get("SLURM_JOB_ID")})
    driver.write_json_atomic(OUT, state)


def main():
    global NAME, OUT, INCLUDE_UNGRADED
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="this job's router, http://127.0.0.1:<port>/v1")
    parser.add_argument("--max-wait-backend", type=float, default=12 * 3600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--name", default=NAME, help="state name (results/glmlocal-<name>.json, campaigns <name>r<n>-...)")
    parser.add_argument("--include-ungraded", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="plan the next round (parallel mode) and exit")
    parser.add_argument("--campaign", help="run this one planned campaign of the latest round (parallel mode)")
    parser.add_argument("--tiers", help="comma-separated tiers whose campaigns give candidates (with --plan-only)")
    args = parser.parse_args()
    NAME, INCLUDE_UNGRADED = args.name, args.include_ungraded
    OUT = GR / "results" / f"{driver.ALIAS}-{NAME}.json"
    if any(marker in args.base_url for marker in driver.HUB_MARKERS):
        raise SystemExit("--base-url must be our own router, never the NVIDIA hub")
    state = json.loads(OUT.read_text()) if OUT.exists() else {"rounds": []}
    rounds = state["rounds"]
    planned = set()  # every slot planned under ANY name (results/glmlocal-dip*.json): a slot is re-run at most once
    for path in sorted((GR / "results").glob(f"{driver.ALIAS}-dip*.json")):
        other = state if path == OUT else json.loads(path.read_text())
        planned |= {(entry[0], entry[1], entry[2]["slot"]) for r in other.get("rounds", []) for p in r["plan"].values()
                    for entries in p["tasks"].values() for entry in entries}
    parts = GR / "results" / f"{driver.ALIAS}-{NAME}-parts"
    if args.campaign:  # parallel mode: one planned campaign of the latest round
        if not rounds or args.campaign not in rounds[-1]["plan"]:
            raise SystemExit(f"{args.campaign} is not in the latest {NAME} round")
        current = rounds[-1]
        plan = {args.campaign: current["plan"][args.campaign]}
        done_part = parts / f"{args.campaign}.json"
        if done_part.exists():
            print(f"{args.campaign} already done ({done_part})")
            return
        results = {}
        return run_campaigns(args, plan, results, state, part_dir=parts)
    tiers = {int(x) for x in args.tiers.split(",")} if args.tiers else None
    slots, pending = candidates(tiers)
    unfinished = set(rounds[-1]["plan"]) - set(rounds[-1]["campaigns"]) - {p.stem for p in parts.glob("*.json")} \
        if rounds else set()
    current = rounds[-1] if rounds and unfinished and not args.plan_only else None
    if args.plan_only and rounds and unfinished:
        raise SystemExit(f"latest {NAME} round still has unfinished campaigns {sorted(unfinished)}")
    if current is None:
        prefix = f"{NAME}r{len(rounds) + 1}"
        new_plan = plan_campaigns(slots, planned, prefix)
        if new_plan and not args.dry_run:
            current = {"round": len(rounds) + 1, "planned_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       "campaigns_not_done_then": pending, "plan": new_plan, "campaigns": {}}
            rounds.append(current)
            write_state(state)
        plan = new_plan
    else:
        plan = current["plan"]
    summary = {"window": WINDOW, "candidates": len(slots), "planned_in_earlier_rounds": len(planned),
               "campaigns_not_done": pending, "round": current["round"] if current else None,
               "plan": {name: {"owner": p["owner"], "repeats": p["repeats"], "tasks": len(p["tasks"]),
                               "slots": sum(len(v) for v in p["tasks"].values())} for name, p in plan.items()}}
    print(json.dumps(summary, indent=1))
    if args.dry_run or args.plan_only or not plan:
        return
    return run_campaigns(args, plan, current["campaigns"], state)


def run_campaigns(args, plan, results, state, part_dir=None):
    """Run the planned campaigns (sequentially) and record each result: in the state file (results[name]) or, in
    parallel mode, in part_dir/<name>.json."""
    tasks, _ = driver.load_plan()
    changed = [key for p in plan.values() for key in p["tasks"]
               if driver.digests(Path(tasks[key]["task_dir"])) != tasks[key]["digests"]]
    if changed:
        raise SystemExit(f"task files changed since tasks.json was frozen: {changed}")
    assert driver.SETTINGS["temperature"] == 1.0 and driver.SETTINGS["top_p"] == 0.95
    os.environ["GLM53_BASE_URL"] = args.base_url
    os.environ["GR_PROMPT"] = driver.PROMPT
    health = driver.wait_for_backend(args.base_url, args.max_wait_backend)
    driver.SETTINGS["models"][driver.ALIAS] = dict(driver.SPEC)
    driver.rollouts.SYNTH = GR
    for name, p in plan.items():
        if name in results:
            continue
        sandbox, k = p["owner"], p["repeats"]
        probe_dir = GR / "probes" / driver.ALIAS / name
        legs = probe_dir / "gr-legs.jsonl"
        if legs.exists():
            earlier = [json.loads(line) for line in legs.read_text().splitlines() if line.strip()]
            assert all(leg["sandbox"] == sandbox for leg in earlier), f"{probe_dir}: earlier legs used another sandbox"
        os.environ["GR_SANDBOX"] = sandbox
        keys = sorted(p["tasks"])
        driver.rollouts.PROBE_MAX_CONCURRENCY = max(1, min(24, len(keys) * k))
        task_dirs = {tasks[key]["task_id"]: Path(tasks[key]["task_dir"]) for key in keys}
        driver.log(f"glm-rollouts-tasksall: {name}: re-running {len(keys) * k} dip-timeout slot(s) via {args.base_url} "
                   f"(router {health}), sandbox {driver.SANDBOX_CREDENTIAL[sandbox]}")
        driver.leg_record(campaign=name, event="start", reason=f"agent timeouts overlapping {WINDOW}",
                          slots={key: [[c, s["slot"]] for c, _, s, _ in p["tasks"][key]] for key in keys})
        records = driver.probe(probe_dir, [task_dirs[tasks[key]["task_id"]] for key in keys], repeats=k)
        first = {t: [driver.compact(r) for r in rs] for t, rs in records.items()}
        recovered = driver.recover(name, records, task_dirs)
        mapping = []
        for key in keys:
            task_id = tasks[key]["task_id"]
            for (campaign, _, original, interval), new in zip(p["tasks"][key], records[task_id]):  # noqa: B007
                mapping.append({"campaign": campaign, "task_key": key, "task_id": task_id, "slot": original["slot"],
                                "original_status": original["status"], "original_attempt": original.get("attempt_dir"),
                                "original_interval": interval, "rerun_campaign": name, "rerun_slot": new["slot"],
                                "rerun_status": new["status"], "rerun_attempt": new.get("attempt_dir"),
                                "rerun": driver.compact(new)})
        results[name] = {"sandbox": sandbox, "stats": driver.stats(records), "first_attempt_records": first,
                         "records": {t: [driver.compact(r) for r in rs] for t, rs in records.items()},
                         "sandbox_failure_reruns": recovered, "mapping": mapping}
        if part_dir is not None:
            driver.write_json_atomic(part_dir / f"{name}.json", dict(results[name], campaign=name,
                                                                     finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                                                     job=os.environ.get("SLURM_JOB_ID")))
        else:
            write_state(state)
        driver.leg_record(campaign=name, event="complete", stats=driver.stats(records))
        driver.log(f"glm-rollouts-tasksall: {name} done: {json.dumps(driver.stats(records))}")


if __name__ == "__main__":
    main()
