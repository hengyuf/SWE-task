#!/usr/bin/env python3
"""teacher-deepswe (2026-10-30): watcher of this campaign's legs (kimi-rollouts-v4's watch.py for 3 models: campaign ids
are <model>-<name>, e.g. glm-w1-g0 or kimi-pilot; jobs tdw-<model>-<name>; hub key NVAPI_KEY_NEW_2 from 2026-10-31
18:15 per OPERATOR_NOTE 18:15 (2026-10-30 21:45 .. the 10-31 10:30 floor stop: NVAPI_KEY_NEW_3; the three 21:34 pilot
legs: NVAPI_KEY_NEW_2); legs start at >= $1,700 left and stop below $1,500 because NEW_2 also carries the r6 RL rubric
judge). Checks every
--interval seconds and EXITS (printing one JSON snapshot with "events") only when something needs attention (--once:
one check, always prints):
  operator_note     OPERATOR_NOTE.md changed since the last check (re-read it: its instructions override the prompt)
  bad_settings      a new probes/<model>/<name>/settings.json whose model spec / concurrency cap differ from the launch
                    (the alias's hub id, reasoning_effort max, context_window 262144; cap = the leg's --max-concurrency)
  all_done          every watched campaign has results/<model>-<name>.json
  campaign_ended    no queued/running job tdw-<model>-<name> and no results (needs another leg); with --auto-leg-cap N
                    the watcher submits up to N legs per campaign itself (bash submit.sh ...; the count persists in
                    watch-state.json); submit.sh enforces the 9-campaign cap and the wave-2 gating
  w2_started        --auto-start-w2: a model's wave-1 campaign finished and one more of its wave-2 groups was submitted
  auto_leg_failed   such a submission failed (submit.sh's refusal included)
  guard_stopped     a leg stopped itself because its hub key fell below the guard floor (legs.jsonl)
  refused_start     a leg refused to start (NEW_2 budget unknown or below the start floor; legs.jsonl)
  budget_low        NVAPI_KEY_NEW_2 has < --budget-floor left (default 1,900: warning above the $1,700 start floor;
                    reported once per key)
  budget_start_floor NVAPI_KEY_NEW_2 has < $1,700 left: no new leg can start (reported once per key)
  budget_floor      NVAPI_KEY_NEW_2 has < $1,500 left (the guard floor; OPERATOR_NOTE 2026-10-31 18:15: stop the
                    campaign, never another key). With --stop-at-floor the watcher writes babysit/<model>-<name>/STOP
                    for every watched campaign without results (so no further leg is submitted) and lists them;
                    reported once per key (flags per key NAME in watch-state.json "budget_flags")
  budget_unknown    3 budget checks in a row failed
  hub_4xx           replies with status 400/401/403/404/422 (fatal for ATP: budget refusal, context overflow, ...),
                    reported once per new attempt with the reply body's first 300 characters (main campaign and its
                    <name>-rec<n> / <name>-brk<n> recovery campaigns)
  rerun_planned     a campaign wrote its broken-reply re-run plan (reruns/<model>-<name>.json; reported once)
  stalled           running jobs but no new terminal.json and no http.jsonl growth for --stall-hours
  rate_limited      >= 10% of the hub replies seen in the last interval were 429 (and >= 20 of them)
  campaign_done     a campaign named in --notify-done completed
  quota_low         hengyuf's Lustre user quota has < 2 TiB free
  inodes_low        hengyuf's user quota or project 10581 (nemotron_agents_dev) has < --inode-floor free files
                    (default 4,000,000; each reported once)
  deadline          --max-hours elapsed (re-arm)
State in watch-state.json; every check appends a line to watch-log.jsonl. Read-only on the campaigns; Slurm only via
squeue (and submit.sh for auto legs).

    /usr/bin/python3 -I -B watch.py --campaigns kimi-w1-g0 glm-w1-g0 ... [--interval 600] [--max-hours 10]
        [--budget-floor 1900] [--stall-hours 2] [--auto-leg-cap N] [--auto-start-w2] [--stop-at-floor]
        [--partition P] [--once]
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL_IDS = {"kimi": "nvidia/moonshotai/kimi-k3", "glm": "nvidia/zai-org/glm-5.3",
             "deepseek": "nvidia/deepseek-ai/deepseek-v4-pro"}
KEY = "NVAPI_KEY_NEW_2"  # every leg from 2026-10-31 18:15 (OPERATOR_NOTE 18:15; NEW_3 from 10-30 21:45 to its floor)
START_FLOOR = 1700.0      # driver.py: a leg starts only at >= this
GUARD_FLOOR = 1500.0      # driver.py: a running leg stops below this
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "RESIZING", "SUSPENDED"}
FATAL = {"400", "401", "403", "404", "422"}
PROJECT = "10581"


def spec(model):
    return {"id": MODEL_IDS[model], "sampling_overrides": {"reasoning_effort": "max"}, "context_window": 262144}


def split(campaign):
    """'glm-w1-g0' -> ('glm', 'w1-g0')."""
    model, _, name = campaign.partition("-")
    assert model in MODEL_IDS and re.fullmatch(r"pilot|w[12]-g\d", name), campaign
    return model, name


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
        if len(parts) == 5 and parts[1].startswith("tdw-"):
            jobs.setdefault(parts[1], []).append({"id": parts[0], "state": parts[2], "time": parts[3], "part": parts[4]})
    return jobs


def legs():
    path = HERE / "legs.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def campaign_dirs(model, name):
    """probes/<model>/<name> and its recovery campaigns <name>-rec<n> / <name>-brk<n> (driver.py)."""
    root = HERE / "probes" / model
    extra = sorted(p for p in root.glob(f"{name}-*") if re.fullmatch(re.escape(name) + r"-(rec|brk)\d+", p.name))
    return [root / name] + extra


def submit(args_list, partition, cap=None):
    env = dict(os.environ, TDW_PARTITION=partition)
    if cap:  # the campaign's own cap (fixed at creation), so legs.jsonl records what the campaign runs with
        env["TDW_MAX_CONCURRENCY_OVERRIDE"] = str(cap)
    out = subprocess.run(["bash", str(HERE / "submit.sh")] + args_list, capture_output=True, text=True, timeout=600,
                         env=env, cwd=str(HERE))
    return {"ok": out.returncode == 0, "args": args_list, "stdout": out.stdout.strip()[-300:],
            "stderr": out.stderr.strip()[-300:]}


def auto_leg(model, name, partition):
    cap = (read_json(HERE / "probes" / model / name / "settings.json", {}) or {}).get("max_concurrency_per_model")
    if name == "pilot":
        return submit(["pilot", model], partition)
    wave, _, group = name.partition("-g")
    return submit([model, wave, group], partition, cap)


def budget():
    out = subprocess.run(["/usr/bin/python3", "-I", "-B", str(HERE / "key_budget.py"), "--key-name", KEY, "--tag", "watch"],
                         capture_output=True, text=True, timeout=300)
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"key": KEY, "error": (out.stdout + out.stderr)[-300:]}


def lfs_quota(kind, who):
    """(free TiB, free files) from `lfs quota -u|-p` (KiB and file columns), or (None, None)."""
    try:
        out = subprocess.run(["lfs", "quota", f"-{kind}", who, "/lustre/fsw"], capture_output=True, text=True,
                             timeout=120).stdout.splitlines()
        fields = [line.split() for line in out if line.strip().startswith("/lustre/fsw")][0]
        used, limit = int(fields[1].rstrip("*")), int(fields[3])
        files, files_limit = int(fields[5].rstrip("*")), int(fields[7])
        return round((limit - used) / 2 ** 30, 2), files_limit - files
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None, None


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
    note = HERE / "OPERATOR_NOTE.md"
    digest = hashlib.sha256(note.read_bytes()).hexdigest() if note.exists() else None
    if digest != state.get("operator_note_sha256"):
        if state.get("operator_note_sha256") is not None or state.get("operator_note_seen"):
            events.append({"event": "operator_note", "sha256": digest})
        state["operator_note_sha256"] = digest
        state["operator_note_seen"] = True
    jobs = squeue()
    entries = {}
    statuses = {}
    sizes = state.setdefault("http_sizes", {})
    caps = {}
    for leg in legs():
        ident = f"{leg.get('model')}-{leg.get('campaign')}"
        mark = f"{ident}@{leg.get('job')}@{leg.get('at')}"
        if leg.get("event") == "start":
            caps[ident] = leg.get("max_concurrency")
        if leg.get("event") == "incomplete" and leg.get("guard") and mark not in state.setdefault("reported_guard", []):
            state["reported_guard"].append(mark)
            events.append({"event": "guard_stopped", "campaign": ident, "job": leg.get("job"),
                           "guard": leg["guard"].get("reason")})
        if leg.get("event") == "refused" and mark not in state.setdefault("reported_refused", []):
            state["reported_refused"].append(mark)
            events.append({"event": "refused_start", "campaign": ident, "job": leg.get("job"),
                           "reason": leg.get("reason"), "remaining": (leg.get("budget") or {}).get("remaining")})
    for ident in campaigns:
        model, name = split(ident)
        probe = HERE / "probes" / model / name
        result = HERE / "results" / f"{model}-{name}.json"
        entry = {"settings": None, "terminal": 0, "attempts": 0, "result": result.exists(), "jobs": None}
        if (probe / "settings.json").exists():
            settings = read_json(probe / "settings.json", {})
            entry["settings"] = {"models": settings.get("models"), "max": settings.get("max_concurrency_per_model")}
            if ident not in state.setdefault("seen_campaigns", []):
                state["seen_campaigns"].append(ident)
                expected = {"models": {model: spec(model)}, "max": caps.get(ident)}
                if entry["settings"] != expected:
                    events.append({"event": "bad_settings", "campaign": ident, "found": entry["settings"],
                                   "expected": expected})
                else:
                    state.setdefault("verified_campaigns", []).append(ident)
            attempts = [a for d in campaign_dirs(model, name) for a in d.glob("models/*/attempts/*/attempt-*")]
            entry["attempts"] = len(attempts)
            entry["terminal"] = sum((a / "terminal.json").exists() for a in attempts)
            entry["recovery_campaigns"] = [d.name for d in campaign_dirs(model, name)[1:]]
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
                        events.append({"event": "hub_4xx", "campaign": ident, "attempt": str(attempt),
                                       "replies": bodies[:3]})
        plan = HERE / "reruns" / f"{model}-{name}.json"
        if plan.exists() and ident not in state.setdefault("reported_rerun_plan", []):
            state["reported_rerun_plan"].append(ident)
            events.append({"event": "rerun_planned", "campaign": ident,
                           "slots": len((read_json(plan, {}) or {}).get("slots", []))})
        if entry["result"] and ident in state.get("notify_done", []) and ident not in state.setdefault("reported_done", []):
            state["reported_done"].append(ident)
            events.append({"event": "campaign_done", "campaign": ident})
        if jobs is not None:
            mine = jobs.get(f"tdw-{ident}", [])
            entry["jobs"] = [f"{j['id']}:{j['state']}:{j['time']}:{j['part']}" for j in mine]
            active = any(j["state"] in ACTIVE for j in mine)
            if active and ident in state.setdefault("reported_ended", []):  # resubmitted by hand: report its next end
                state["reported_ended"].remove(ident)
            started = probe.exists() or bool(mine)
            if started and not entry["result"] and not active and ident not in state.setdefault("reported_ended", []):
                used = state.setdefault("auto_legs", {}).get(ident, 0)
                if state.get("auto_leg_cap", 0) > used:
                    submitted = auto_leg(model, name, state["partition"])
                    state["auto_legs"][ident] = used + 1
                    state.setdefault("auto_leg_log", []).append({"campaign": ident, "at": now, **submitted})
                    if not submitted.get("ok"):
                        state["reported_ended"].append(ident)
                        events.append({"event": "auto_leg_failed", "campaign": ident, **submitted})
                else:
                    state["reported_ended"].append(ident)
                    events.append({"event": "campaign_ended", "campaign": ident, "terminal": entry["terminal"],
                                   "attempts": entry["attempts"], "auto_legs_used": used})
        entries[ident] = entry
    if state.get("auto_start_w2") and jobs is not None:
        for model in MODEL_IDS:
            done_w1 = len(list((HERE / "results").glob(f"{model}-w1-g*.json")))
            for group in (0, 1):
                ident = f"{model}-w2-g{group}"
                if (HERE / "probes" / model / f"w2-g{group}").exists() or jobs.get(f"tdw-{ident}"):
                    continue
                started = sum((HERE / "probes" / model / f"w2-g{g}").exists() or bool(jobs.get(f"tdw-{model}-w2-g{g}"))
                              for g in (0, 1))
                if started >= done_w1:
                    break
                submitted = submit([model, "w2", str(group)], state["partition"])  # submit.sh re-checks cap + gating
                if submitted["ok"]:
                    events.append({"event": "w2_started", "campaign": ident, **submitted})
                elif ident not in state.setdefault("w2_failed_reported", []):  # retried every check, reported once
                    state["w2_failed_reported"].append(ident)
                    events.append({"event": "auto_leg_failed", "campaign": ident, **submitted})
                break
    if entries and all(e["result"] for e in entries.values()) and not state.get("reported_all_done"):
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
    free_tib, user_free_files = lfs_quota("u", os.environ.get("USER", "hengyuf"))
    if free_tib is not None and free_tib < state.get("quota_floor_tib", 2.0) and not state.get("reported_quota_low"):
        state["reported_quota_low"] = True
        events.append({"event": "quota_low", "free_tib": free_tib})
    if user_free_files is not None and user_free_files < state["inode_floor"] and not state.get("reported_user_inodes_low"):
        state["reported_user_inodes_low"] = True  # hengyuf's user quota (25M files) is shared with every hengyuf job
        events.append({"event": "inodes_low", "quota": "user", "free_files": user_free_files})
    _, free_files = lfs_quota("p", PROJECT)
    if free_files is not None and free_files < state["inode_floor"] and not state.get("reported_inodes_low"):
        state["reported_inodes_low"] = True
        events.append({"event": "inodes_low", "quota": f"project {PROJECT}", "free_files": free_files})
    spend = budget()
    if spend.get("remaining") is None:
        state["budget_failures"] = state.get("budget_failures", 0) + 1
        if state["budget_failures"] >= 3:
            events.append({"event": "budget_unknown", "detail": spend})
            state["budget_failures"] = 0
    else:
        state["budget_failures"] = 0
        flags = state.setdefault("budget_flags", {}).setdefault(KEY, {})  # per key NAME (NEW_3's floor came first)
        if spend["remaining"] < state["budget_floor"] and not flags.get("low"):
            flags["low"] = True
            events.append({"event": "budget_low", "key": KEY, "remaining": spend["remaining"]})
        if spend["remaining"] < START_FLOOR and not flags.get("start_floor"):
            flags["start_floor"] = True
            events.append({"event": "budget_start_floor", "key": KEY, "remaining": spend["remaining"],
                           "start_floor": START_FLOOR})
        if spend["remaining"] < GUARD_FLOOR and not flags.get("floor"):
            flags["floor"] = True
            stopped = []
            if state.get("stop_at_floor"):
                for ident in campaigns:
                    model, name = split(ident)
                    if (HERE / "results" / f"{model}-{name}.json").exists():
                        continue
                    stop_file = HERE / "babysit" / f"{model}-{name}" / "STOP"
                    stop_file.parent.mkdir(parents=True, exist_ok=True)
                    if not stop_file.exists():
                        stop_file.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} watch.py: {KEY} remaining "
                                             f"${spend['remaining']:.2f} < ${GUARD_FLOOR:.0f} (OPERATOR_NOTE "
                                             "2026-10-31 18:15)\n")
                    stopped.append(ident)
            events.append({"event": "budget_floor", "key": KEY, "remaining": spend["remaining"],
                           "guard_floor": GUARD_FLOOR, "stop_files_written": stopped})
    return {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "events": events, "budget": spend,
            "quota_free_tib": free_tib, "user_free_files": user_free_files, "project_free_files": free_files,
            "reply_statuses_last_interval": statuses, "running_jobs": running,
            "campaigns": {k: dict(v) for k, v in entries.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaigns", nargs="+", required=True, help="<model>-<name>, e.g. glm-w1-g0, kimi-pilot")
    parser.add_argument("--interval", type=float, default=600)
    parser.add_argument("--max-hours", type=float, default=10)
    parser.add_argument("--budget-floor", type=float, default=1900)
    parser.add_argument("--stop-at-floor", action="store_true", help="write babysit/<model>-<name>/STOP for unfinished "
                                                                     "watched campaigns once NEW_2 < $1,500")
    parser.add_argument("--inode-floor", type=int, default=4_000_000)
    parser.add_argument("--stall-hours", type=float, default=2)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--auto-leg-cap", type=int, default=0, help="auto-submit up to N extra legs per campaign")
    parser.add_argument("--auto-start-w2", action="store_true", help="submit a model's next wave-2 group as its wave-1 "
                                                                     "campaigns finish (through submit.sh's gating)")
    parser.add_argument("--partition", default="cpu_interactive,cpu_dataprocessing,cpu_datamover")
    parser.add_argument("--notify-done", nargs="*", default=[], help="also exit when one of these campaigns completes")
    args = parser.parse_args()
    for ident in args.campaigns + args.notify_done:
        split(ident)
    state_path = HERE / "watch-state.json"
    state = read_json(state_path, {}) or {}
    state.pop("progress", None)
    state["progress_at"] = time.time()  # the stall clock starts with this run (not a previous run's state)
    state.update(budget_floor=args.budget_floor, stall_hours=args.stall_hours, auto_leg_cap=args.auto_leg_cap,
                 partition=args.partition, notify_done=args.notify_done, inode_floor=args.inode_floor,
                 auto_start_w2=args.auto_start_w2, stop_at_floor=args.stop_at_floor)
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
