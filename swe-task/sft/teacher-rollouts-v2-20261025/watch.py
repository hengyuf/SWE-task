#!/usr/bin/env python3
"""teacher-rollouts-v2 (2026-10-25; rewritten 2026-10-26 at the takeover): background watcher of the full batch. Checks
every --interval seconds and EXITS (printing one compact JSON snapshot with "events") only when something needs
attention:
  group_done        a new results/<model>-group-<K>.json (each group reported once; state reported_done)
  all_done          every results/<model>-group-<K>.json exists
  group_ended       no queued/running job for tr-<model>-g<K> and no results: the agent submits that group's next leg
                    (since 2026-10-26 the watcher submits nothing; new legs use NVAPI_KEY_NEW_2/NEW_3 via submit.sh).
                    Reported once per latest job id of the group (logs/tr-<model>-g<K>-<job>.out)
  bad_settings      a new probes/<model>/group-<K>/settings.json whose caps differ from the launch (glm 6, deepseek 2;
                    context 1,048,576)
  budget_low        NVAPI_KEY_NEW_1 (pre-knob legs still running on it; also the r5 judge's key) below --budget-floor,
                    or the new-leg key (NEW_2; NEW_3 once NEW_2 < --switch-floor) below --new-key-floor
  budget_unknown    3 budget checks in a row failed
  stalled           running campaigns but no new terminal.json and no http.jsonl growth for --stall-hours
  rate_limited      >= --rate-share of the hub replies seen in the last interval were 429 (and >= 50 replies)
  deadline          --max-hours elapsed (re-arm)
State in watch-state.json; every check appends a line to watch-log.jsonl. Read-only on the campaigns; Slurm only via
squeue; budgets via key_budget.py (key NAMES and numbers only).

    /usr/bin/python3 -I -B watch.py [--interval 600] [--max-hours 12] [--budget-floor 600] [--new-key-floor 600]
                                    [--switch-floor 330] [--stall-hours 2] [--rate-share 0.2] [--once]
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
GROUPS = [(model, group) for model in ("glm", "deepseek") for group in range(8)]
EXPECTED_CAP = {"glm": 6, "deepseek": 2}
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "RESIZING", "SUSPENDED"}
OLD_KEY, NEW_KEYS = "NVAPI_KEY_NEW_1", ("NVAPI_KEY_NEW_2", "NVAPI_KEY_NEW_3")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def squeue():
    out = subprocess.run(["squeue", "--noheader", "-u", os.environ.get("USER", "hengyuf"), "--format=%i|%j|%T|%M"],
                         capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        return None
    jobs = {}
    for line in out.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 4 and parts[1].startswith("tr-"):
            jobs.setdefault(parts[1], []).append({"id": parts[0], "state": parts[2], "time": parts[3]})
    return jobs


def leg_keys():
    """{job id: key NAME} from legs.jsonl start records."""
    keys = {}
    try:
        lines = (HERE / "legs.jsonl").read_text().splitlines()
    except OSError:
        return keys
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("event") == "start":
            keys[str(entry.get("job"))] = entry.get("key")
    return keys


def latest_job(model, group):
    ids = [int(m.group(1)) for path in (HERE / "logs").glob(f"tr-{model}-g{group}-*.out")
           if (m := re.search(r"-(\d+)\.out$", path.name))]
    return max(ids) if ids else None


def budget(key_name):
    out = subprocess.run(["/usr/bin/python3", "-I", "-B", str(HERE / "key_budget.py"), "--key-name", key_name,
                          "--tag", "watch"], capture_output=True, text=True, timeout=300)
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"key": key_name, "error": (out.stdout + out.stderr)[-300:]}


def tail_statuses(path, since_size):
    """Count reply statuses in the part of an http.jsonl written since the last check (at most the last 64 MB)."""
    counts = {}
    try:
        size = path.stat().st_size
        start = max(since_size, size - (64 << 20))
        if size <= start:
            return counts, size
        with open(path, "rb") as stream:
            stream.seek(start)
            if start:
                stream.readline()
            for raw in stream:
                head = raw[:1500]
                if b'"event":"response"' not in head:
                    continue
                marker = raw.find(b'"status":')
                if marker < 0:
                    continue
                code = raw[marker + 9:marker + 12].decode(errors="replace")
                counts[code] = counts.get(code, 0) + 1
        return counts, size
    except OSError:
        return counts, since_size


def snapshot(state):
    now = time.time()
    events = []
    jobs = squeue()
    keys = leg_keys()
    campaigns = {}
    statuses = {}
    sizes = state.setdefault("http_sizes", {})
    for model, group in GROUPS:
        name = f"{model}/group-{group}"
        probe = HERE / "probes" / model / f"group-{group}"
        result = HERE / "results" / f"{model}-group-{group}.json"
        entry = {"slots": None, "terminal": 0, "attempts": 0, "result": result.exists(), "jobs": None}
        if (probe / "settings.json").exists():
            settings = read_json(probe / "settings.json", {})
            found = {"max": settings.get("max_concurrency_per_model"), "ctx": settings.get("context_window")}
            entry["slots"] = (settings.get("expected_tasks") or 0) * (settings.get("repeats") or 0)
            if name not in state.setdefault("seen_campaigns", []):
                state["seen_campaigns"].append(name)
                expected = {"max": EXPECTED_CAP[model], "ctx": 1048576}
                if found != expected:
                    events.append({"event": "bad_settings", "group": name, "found": found, "expected": expected})
                else:
                    state.setdefault("verified_campaigns", []).append(name)
            slots = list(probe.glob("models/*/attempts/*"))
            attempts = [attempt for slot in slots for attempt in slot.glob("attempt-*")]
            entry["attempts"] = len(attempts)
            entry["terminal"] = sum(any((attempt / "terminal.json").exists() for attempt in slot.glob("attempt-*"))
                                    for slot in slots)
            model_counts = statuses.setdefault(model, {})
            for attempt in attempts:
                http = attempt / "http.jsonl"
                key = str(http)
                if http.exists() and not (attempt / "terminal.json").exists() or key not in sizes:
                    counts, size = tail_statuses(http, sizes.get(key, 0))
                    sizes[key] = size
                    for code, count in counts.items():
                        model_counts[code] = model_counts.get(code, 0) + count
        if entry["result"] and name not in state.setdefault("reported_done", []):
            state["reported_done"].append(name)
            state.setdefault("done_log", []).append({"group": name, "at": now})
            events.append({"event": "group_done", "group": name})
        if jobs is not None:
            mine = jobs.get(f"tr-{model}-g{group}", [])
            entry["jobs"] = [f"{j['id']}:{j['state']}:{j['time']}:{keys.get(j['id'], '?')}" for j in mine]
            active = any(j["state"] in ACTIVE for j in mine)
            if not entry["result"] and not active:
                last = latest_job(model, group)
                if state.setdefault("ended_reported", {}).get(name) != last:
                    state["ended_reported"][name] = last
                    events.append({"event": "group_ended", "group": name, "last_job": last,
                                   "last_job_key": keys.get(str(last)), "terminal": entry["terminal"],
                                   "slots": entry["slots"], "attempts": entry["attempts"]})
        campaigns[name] = entry
    if all(e["result"] for e in campaigns.values()) and not state.get("reported_all_done"):
        state["reported_all_done"] = True
        events.append({"event": "all_done"})
    progress = sum(e["terminal"] for e in campaigns.values()) + sum(sizes.values())
    if progress != state.get("progress"):
        state["progress"], state["progress_at"] = progress, now
    running = jobs is not None and any(j["state"] == "RUNNING" for js in jobs.values() for j in js)
    stall_h = (now - state.get("progress_at", now)) / 3600
    if running and stall_h >= state["stall_hours"]:
        events.append({"event": "stalled", "hours": round(stall_h, 2)})
        state["progress_at"] = now
    total = {}
    for counts in statuses.values():
        for code, count in counts.items():
            total[code] = total.get(code, 0) + count
    replies = sum(total.values())
    if replies >= 50 and total.get("429", 0) >= state["rate_share"] * replies:
        events.append({"event": "rate_limited", "replies": replies, "statuses": statuses})
    spends = {}
    checks = [(OLD_KEY, state["budget_floor"])]
    for key_name in NEW_KEYS:  # NEW_2, and NEW_3 only once NEW_2 is below the driver's switch floor
        checks.append((key_name, state["new_key_floor"]))
        record = budget(key_name)
        spends[key_name] = record
        if record.get("remaining") is None or record["remaining"] >= state["switch_floor"]:
            break
    spends[OLD_KEY] = budget(OLD_KEY)
    failures = [name for name, record in spends.items() if record and record.get("remaining") is None]
    state["budget_failures"] = state.get("budget_failures", 0) + 1 if failures else 0
    if state["budget_failures"] >= 3:
        events.append({"event": "budget_unknown", "keys": failures})
        state["budget_failures"] = 0
    reported = state.setdefault("reported_budget_low", [])
    for key_name, floor in checks:
        record = spends.get(key_name) or {}
        if record.get("remaining") is not None and record["remaining"] < floor and key_name not in reported:
            reported.append(key_name)
            events.append({"event": "budget_low", "key": key_name, "remaining": record["remaining"], "floor": floor})
    return {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "events": events,
            "budget": {name: {k: record.get(k) for k in ("spend", "max_budget", "remaining", "error")}
                       for name, record in spends.items() if record},
            "reply_statuses_last_interval": statuses, "running_jobs": running,
            "campaigns": {name: f"{e['terminal']}/{e['slots']} att {e['attempts']}"
                                f"{' DONE' if e['result'] else ''} {' '.join(e['jobs'] or [])}".rstrip()
                          for name, e in campaigns.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=600)
    parser.add_argument("--max-hours", type=float, default=12)
    parser.add_argument("--budget-floor", type=float, default=600, help="NVAPI_KEY_NEW_1 alert floor")
    parser.add_argument("--new-key-floor", type=float, default=600, help="new-leg key (NEW_2/NEW_3) alert floor")
    parser.add_argument("--switch-floor", type=float, default=330, help="driver's TR_MIN_REMAINING: NEW_2 -> NEW_3")
    parser.add_argument("--stall-hours", type=float, default=2)
    parser.add_argument("--rate-share", type=float, default=0.2)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    state_path = HERE / "watch-state.json"
    state = read_json(state_path, {}) or {}
    for stale in ("auto_leg_cap", "reported_ended"):
        state.pop(stale, None)  # pre-2026-10-26 watcher fields (auto legs on NVAPI_KEY_NEW_1)
    if not isinstance(state.get("reported_budget_low"), list):
        state["reported_budget_low"] = []  # key NAMES already reported low (the old watcher kept one NEW_1 flag)
    state.update(budget_floor=args.budget_floor, new_key_floor=args.new_key_floor, switch_floor=args.switch_floor,
                 stall_hours=args.stall_hours, rate_share=args.rate_share)
    deadline = time.time() + args.max_hours * 3600
    while True:
        snap = snapshot(state)
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
