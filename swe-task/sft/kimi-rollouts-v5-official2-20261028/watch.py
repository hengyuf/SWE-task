#!/usr/bin/env python3
"""kimi-rollouts-v5 (2026-10-28): background watcher of this batch's campaigns (kimi-rollouts-v4's watch.py with the
kr5- job prefix and no single-campaign tasksets; one model spec for every campaign, the campaigns' rec / brk recovery
campaigns included in the attempt scan). Checks every --interval seconds and EXITS (printing one JSON snapshot with "events") only when something
needs attention (--once: one check, always prints):
  bad_settings      a new probes/kimi/<name>/settings.json whose model spec / context window / concurrency cap differ
                    from the launch (kimi-k3, context_window 262144; cap = the leg's --max-concurrency in legs.jsonl)
  all_done          every watched campaign has results/kimi-<name>.json
  campaign_ended    no queued/running job kr5-<name> and no results (needs another leg); with --auto-leg-cap N the
                    watcher itself submits up to N legs per campaign (bash submit.sh <taskset> <group>) and logs them
                    (the count persists in watch-state.json across runs)
  auto_leg_failed   such a submission failed
  guard_stopped     a leg stopped itself because its hub key ran low (legs.jsonl); the next leg picks NVAPI_KEY_NEW_3
  key_switched      a leg started on NVAPI_KEY_NEW_3 (reported once)
  budget_low        the key in use (NEW_2, or NEW_3 once NEW_2 < $100) has < --budget-floor left (reported once per key)
  budget_unknown    3 budget checks in a row failed
  hub_4xx           replies with status 400/401/403/404/422 (fatal for ATP: budget refusal, context overflow, ...),
                    reported once per new attempt with the reply body's first 300 characters (main campaign and its
                    <name>-rec<n> / <name>-brk<n> recovery campaigns)
  rerun_planned     a campaign wrote its broken-reply re-run plan (reruns/<name>.json; reported once)
  stalled           running jobs but no new terminal.json and no http.jsonl growth for --stall-hours
  rate_limited      >= 10% of the hub replies seen in the last interval were 429 (and >= 20 of them)
  campaign_done     a campaign named in --notify-done completed (results/kimi-<name>.json written)
  quota_low         hengyuf's Lustre user quota has < 2 TiB free (2026-10-26: a full quota killed legs with EDQUOT)
  deadline          --max-hours elapsed (re-arm)
State in watch-state.json; every check appends a line to watch-log.jsonl. Read-only on the campaigns; Slurm only via
squeue (and submit.sh for auto legs).

    /usr/bin/python3 -I -B watch.py --campaigns official2-g0 ... [--interval 600] [--max-hours 10] [--budget-floor 500]
                                    [--stall-hours 2] [--auto-leg-cap N] [--partition P]
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

HERE = Path(__file__).resolve().parent
SPEC = {"id": "nvidia/moonshotai/kimi-k3", "sampling_overrides": {"reasoning_effort": "max"}, "context_window": 262144}
SINGLE = ()  # campaigns without groups (none in v5)
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "RESIZING", "SUSPENDED"}
FATAL = {"400", "401", "403", "404", "422"}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def squeue():
    out = subprocess.run(["squeue", "--noheader", "-u", os.environ.get("USER", "hengyuf"), "--format=%i|%j|%T|%M|%P"],
                         capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        return None
    jobs = {}
    for line in out.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 5 and parts[1].startswith("kr5-"):
            jobs.setdefault(parts[1], []).append({"id": parts[0], "state": parts[2], "time": parts[3], "part": parts[4]})
    return jobs


def legs():
    path = HERE / "legs.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def campaign_dirs(name):
    """probes/kimi/<name> and its recovery campaigns <name>-rec<n> / <name>-brk<n> (driver.py)."""
    root = HERE / "probes" / "kimi"
    extra = sorted(p for p in root.glob(f"{name}-*") if re.fullmatch(re.escape(name) + r"-(rec|brk)\d+", p.name))
    return [root / name] + extra


def auto_leg(name, partition):
    taskset, _, group = name.rpartition("-g")
    args = ["bash", str(HERE / "submit.sh")] + ([name] if name in SINGLE else [taskset, group])
    env = dict(os.environ, KR_PARTITION=partition)
    cap = (read_json(HERE / "probes" / "kimi" / name / "settings.json", {}) or {}).get("max_concurrency_per_model")
    if cap:  # the campaign's own cap (fixed at creation), so legs.jsonl records what the campaign runs with
        env["KR_MAX_CONCURRENCY_OVERRIDE"] = str(cap)
    out = subprocess.run(args, capture_output=True, text=True, timeout=300, env=env, cwd=str(HERE))
    return {"ok": out.returncode == 0, "stdout": out.stdout.strip()[-300:], "stderr": out.stderr.strip()[-300:]}


def budget(key):
    out = subprocess.run(["/usr/bin/python3", "-I", "-B", str(HERE / "key_budget.py"), "--key-name", key, "--tag", "watch"],
                         capture_output=True, text=True, timeout=300)
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"key": key, "error": (out.stdout + out.stderr)[-300:]}


def quota_free_tib():
    """hengyuf's free Lustre user quota (TiB) from `lfs quota -u` (KiB columns), or None."""
    try:
        out = subprocess.run(["lfs", "quota", "-u", os.environ.get("USER", "hengyuf"), "/lustre/fsw"],
                             capture_output=True, text=True, timeout=120).stdout.splitlines()
        fields = [line.split() for line in out if line.strip().startswith("/lustre/fsw")][0]
        used, limit = int(fields[1].rstrip("*")), int(fields[3])
        return round((limit - used) / 2 ** 30, 2)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def tail_replies(path, since_size):
    """Reply statuses (and 4xx bodies) in the part of an http.jsonl written since the last check (<= last 64 MB)."""
    counts, bodies = {}, []
    try:
        size = path.stat().st_size
        start = max(since_size, size - (64 << 20))
        if size <= start:
            return counts, bodies, size
        with open(path, "rb") as stream:
            stream.seek(start)
            if start:
                stream.readline()
            for raw in stream:
                head = raw[:3000]
                if b'"event": "response"' not in head and b'"event":"response"' not in head:
                    continue
                match = re.search(rb'"status": ?(\d{3})', raw[:6000])
                if not match:
                    continue
                code = match.group(1).decode()
                counts[code] = counts.get(code, 0) + 1
                if code in FATAL:
                    try:
                        bodies.append({"status": code, "body": (json.loads(raw).get("body") or "")[:300]})
                    except ValueError:
                        bodies.append({"status": code, "body": "?"})
        return counts, bodies, size
    except OSError:
        return counts, bodies, since_size


def snapshot(state, campaigns):
    now = time.time()
    events = []
    jobs = squeue()
    entries = {}
    statuses = {}
    sizes = state.setdefault("http_sizes", {})
    caps = {}
    for leg in legs():
        if leg.get("event") == "start":
            caps[leg["campaign"]] = leg.get("max_concurrency")
            if leg.get("hub_key") == "NVAPI_KEY_NEW_3" and not state.get("reported_key_switch"):
                state["reported_key_switch"] = True
                events.append({"event": "key_switched", "campaign": leg["campaign"], "job": leg.get("job"),
                               "at": leg.get("at")})
        if leg.get("event") == "incomplete" and leg.get("guard"):
            mark = f"{leg['campaign']}@{leg.get('job')}"
            if mark not in state.setdefault("reported_guard", []):
                state["reported_guard"].append(mark)
                events.append({"event": "guard_stopped", "campaign": leg["campaign"], "job": leg.get("job"),
                               "guard": leg["guard"].get("reason")})
    for name in campaigns:
        probe = HERE / "probes" / "kimi" / name
        result = HERE / "results" / f"kimi-{name}.json"
        entry = {"settings": None, "terminal": 0, "attempts": 0, "result": result.exists(), "jobs": None}
        if (probe / "settings.json").exists():
            settings = read_json(probe / "settings.json", {})
            entry["settings"] = {"models": settings.get("models"), "max": settings.get("max_concurrency_per_model")}
            if name not in state.setdefault("seen_campaigns", []):
                state["seen_campaigns"].append(name)
                expected = {"models": {"kimi": SPEC}, "max": caps.get(name)}
                if entry["settings"] != expected:
                    events.append({"event": "bad_settings", "campaign": name, "found": entry["settings"],
                                   "expected": expected})
                else:
                    state.setdefault("verified_campaigns", []).append(name)
            attempts = [a for d in campaign_dirs(name) for a in d.glob("models/*/attempts/*/attempt-*")]
            entry["attempts"] = len(attempts)
            entry["terminal"] = sum((a / "terminal.json").exists() for a in attempts)
            entry["recovery_campaigns"] = [d.name for d in campaign_dirs(name)[1:]]
            for attempt in attempts:
                http = attempt / "http.jsonl"
                key = str(http)
                if http.exists() and (not (attempt / "terminal.json").exists() or key not in sizes):
                    counts, bodies, size = tail_replies(http, sizes.get(key, 0))
                    sizes[key] = size
                    for code, count in counts.items():
                        statuses[code] = statuses.get(code, 0) + count
                    if bodies and key not in state.setdefault("reported_4xx", []):
                        state["reported_4xx"].append(key)
                        events.append({"event": "hub_4xx", "campaign": name, "attempt": str(attempt),
                                       "replies": bodies[:3]})
        if (HERE / "reruns" / f"{name}.json").exists() and name not in state.setdefault("reported_rerun_plan", []):
            state["reported_rerun_plan"].append(name)
            plan = read_json(HERE / "reruns" / f"{name}.json", {}) or {}
            events.append({"event": "rerun_planned", "campaign": name, "slots": len(plan.get("slots", []))})
        if entry["result"] and name in state.get("notify_done", []) and name not in state.setdefault("reported_done", []):
            state["reported_done"].append(name)
            events.append({"event": "campaign_done", "campaign": name})
        if jobs is not None:
            mine = jobs.get(f"kr5-{name}", [])
            entry["jobs"] = [f"{j['id']}:{j['state']}:{j['time']}:{j['part']}" for j in mine]
            active = any(j["state"] in ACTIVE for j in mine)
            if active and name in state.setdefault("reported_ended", []):  # resubmitted by hand: report its next end
                state["reported_ended"].remove(name)
            if not entry["result"] and not active and name not in state.setdefault("reported_ended", []):
                used = state.setdefault("auto_legs", {}).get(name, 0)
                if state.get("auto_leg_cap", 0) > used:
                    submitted = auto_leg(name, state["partition"])
                    state["auto_legs"][name] = used + 1
                    state.setdefault("auto_leg_log", []).append({"campaign": name, "at": now, **submitted})
                    if not submitted.get("ok"):
                        state["reported_ended"].append(name)
                        events.append({"event": "auto_leg_failed", "campaign": name, **submitted})
                else:
                    state["reported_ended"].append(name)
                    events.append({"event": "campaign_ended", "campaign": name, "terminal": entry["terminal"],
                                   "attempts": entry["attempts"], "auto_legs_used": used})
        entries[name] = entry
    if all(e["result"] for e in entries.values()) and not state.get("reported_all_done"):
        state["reported_all_done"] = True
        events.append({"event": "all_done"})
    progress = sum(e["terminal"] for e in entries.values()) + sum(sizes.values())
    if progress != state.get("progress"):
        state["progress"], state["progress_at"] = progress, now
    running = jobs is not None and any(j["state"] == "RUNNING" for js in jobs.values() for j in js)
    stall_h = (now - state.get("progress_at", now)) / 3600
    if running and stall_h >= state["stall_hours"]:
        events.append({"event": "stalled", "hours": round(stall_h, 2)})
        state["progress_at"] = now
    replies = sum(statuses.values())
    if replies >= 20 and statuses.get("429", 0) >= max(20, 0.1 * replies):
        events.append({"event": "rate_limited", "replies": replies, "statuses": statuses})
    free = quota_free_tib()
    if free is not None and free < state.get("quota_floor_tib", 2.0) and not state.get("reported_quota_low"):
        state["reported_quota_low"] = True
        events.append({"event": "quota_low", "free_tib": free})
    spend = budget("NVAPI_KEY_NEW_2")
    if spend.get("remaining") is not None and spend["remaining"] < 100:
        spend = budget("NVAPI_KEY_NEW_3")
    if spend.get("remaining") is None:
        state["budget_failures"] = state.get("budget_failures", 0) + 1
        if state["budget_failures"] >= 3:
            events.append({"event": "budget_unknown", "detail": spend})
            state["budget_failures"] = 0
    else:
        state["budget_failures"] = 0
        if spend["remaining"] < state["budget_floor"] and spend["key"] not in state.setdefault("reported_budget_low", []):
            state["reported_budget_low"].append(spend["key"])
            events.append({"event": "budget_low", "key": spend["key"], "remaining": spend["remaining"]})
    return {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "events": events, "budget": spend, "quota_free_tib": free,
            "reply_statuses_last_interval": statuses, "running_jobs": running,
            "campaigns": {k: dict(v) for k, v in entries.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaigns", nargs="+", required=True)
    parser.add_argument("--interval", type=float, default=600)
    parser.add_argument("--max-hours", type=float, default=10)
    parser.add_argument("--budget-floor", type=float, default=500)
    parser.add_argument("--stall-hours", type=float, default=2)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--auto-leg-cap", type=int, default=0, help="auto-submit up to N extra legs per campaign")
    parser.add_argument("--partition", default="cpu_interactive,cpu_dataprocessing,cpu_datamover")
    parser.add_argument("--notify-done", nargs="*", default=[], help="also exit when one of these campaigns completes")
    args = parser.parse_args()
    state_path = HERE / "watch-state.json"
    state = read_json(state_path, {}) or {}
    state.pop("progress", None)
    state["progress_at"] = time.time()  # the stall clock starts with this run (not a previous run's state)
    state.update(budget_floor=args.budget_floor, stall_hours=args.stall_hours, auto_leg_cap=args.auto_leg_cap,
                 partition=args.partition, notify_done=args.notify_done)
    deadline = time.time() + args.max_hours * 3600
    while True:
        snap = snapshot(state, args.campaigns)
        tmp = state_path.with_name("." + state_path.name + ".tmp")
        tmp.write_text(json.dumps(state) + "\n")
        os.replace(tmp, state_path)
        with open(HERE / "watch-log.jsonl", "a") as stream:
            stream.write(json.dumps({k: v for k, v in snap.items() if k != "campaigns"}) + "\n")
        if snap["events"] or args.once or time.time() >= deadline:
            if not snap["events"] and not args.once:
                snap["events"] = [{"event": "deadline"}]
            print(json.dumps(snap, indent=1))
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
