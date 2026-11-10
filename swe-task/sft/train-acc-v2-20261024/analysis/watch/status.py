#!/usr/bin/env python3
"""train-acc-v2 watcher (2026-10-25): one file-based snapshot of the sftv2e10 / sftv2e20 test (no Slurm commands).

    python3 -I -B status.py --state STATE.json [--wake-at EPOCH] [--snooze FLEET=EPOCH ...]

Prints a compact snapshot; exit 0 = nothing needs the operator, 1 = attention (reasons printed as "ATTENTION: ...").
Attention when: a new worker registry record is failed/unhealthy; a driver log of these models shows a new
"driver exit" / CANCELLED / babysit failure; a new results file; all 16 results present; project 10581 has < 600k free
files; a running driver waits for workers while its fleet has < 8 routable for > 15 min; a fleet with unfinished
campaigns will have < 8 non-leaving workers 75 min from now (unless snoozed); no finished slot for 2 h while campaigns
run; --wake-at reached.
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

TA = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024")
FLEETS = {"sftv2e10": "sftv2e10", "sftv2e20": "sftv2e20"}  # model -> fleet dir
GROUPS = range(8)
LOOKAHEAD_S = 75 * 60
LEAVING_S = 600
MIN_WORKERS = 8


def registry(fleet):
    out = []
    for path in sorted((TA / "fleets" / fleet / "registry").glob("*.json")):
        try:
            out.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            pass
    return out


def alive(entry, at, now):
    """Worker serving (or starting) and not yet leaving at time `at`."""
    if entry.get("status") not in ("starting", "ready", "draining"):
        return False
    hb = entry.get("heartbeat_at") or 0
    if now - hb > 300:
        return False
    end = entry.get("end_epoch") or 0
    return end - LEAVING_S > at


def routable(entry, now):
    return (entry.get("status") in ("ready", "draining") and now - (entry.get("heartbeat_at") or 0) <= 180
            and (entry.get("end_epoch") or 0) > now)


def driver_logs(model, group):
    return sorted((TA / "logs").glob(f"ta-drv-{model}-g{group}-*.out"))


def driver_state(model, group):
    """(state, detail, job, events) of the most informative log of this group (latest job id first)."""
    states = []
    for path in driver_logs(model, group):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        job = path.stem.rsplit("-", 1)[-1]
        if "sibling candidate won" in text:
            continue
        events = []
        if re.search(r"driver exit \d+", text):
            events.append("exit:" + re.findall(r"driver exit (\d+)", text)[-1])
        if "CANCELLED" in text:
            events.append("cancelled")
        if "babysit_enter failed" in text:
            events.append("babysit_failed")
        if "Traceback" in text:
            events.append("traceback")
        lines = [line for line in text.splitlines() if line.strip()]
        last = lines[-1] if lines else ""
        if re.search(r"group \d+ done:", text):
            state = "done"
        elif "running campaign" in text or re.search(r"train-acc: \S+ group \d+: ", text):
            state = "campaign"
        elif "waiting for" in text:
            state = "waiting"
        elif "job " in text:
            state = "started"
        else:
            state = "empty"
        if events and state != "done":
            state = "ended"
        states.append((int(job), state, last[:160], job, events))
    if not states:
        return ("pending", "", None, [])
    states.sort()
    _, state, last, job, events = states[-1]
    return (state, last, job, events)


def finished_slots(model, group):
    root = TA / "probes" / model / f"group-{group}" / "models" / model / "attempts"
    done = total = 0
    if not root.is_dir():
        return 0, 0
    for slot in root.iterdir():
        if not slot.is_dir():
            continue
        total += 1
        if any(slot.glob("attempt-*/terminal.json")):
            done += 1
    return done, total


def quota_free():
    try:
        out = subprocess.run(["lfs", "quota", "-p", "10581", "/lustre/fsw"], capture_output=True, text=True,
                             timeout=120).stdout.splitlines()
        fields = out[-1].split()  # /lustre/fsw kbytes quota limit grace files quota limit grace
        return int(fields[7]) - int(fields[5])
    except Exception:  # noqa: BLE001
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--wake-at", type=float, default=0)
    parser.add_argument("--snooze", action="append", default=[])
    args = parser.parse_args()
    now = time.time()
    state_path = Path(args.state)
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    first = not state
    snooze = {item.split("=")[0]: float(item.split("=")[1]) for item in args.snooze}
    attention = []
    lines = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))]

    free = quota_free()
    lines.append(f"quota 10581 free files: {free}")
    if free is not None and free < 600_000:
        attention.append(f"project 10581 free files {free} < 600k")

    results = {f"{m}-group-{g}" for m in FLEETS for g in GROUPS if (TA / "results" / f"{m}-group-{g}.json").exists()}
    new_results = sorted(results - set(state.get("results", [])))
    if new_results and not first:
        lines.append(f"new results (info): {new_results}")
    if len(results) == 16:
        attention.append("all 16 results present")

    known_failed = set(state.get("failed", []))
    failed_now = set()
    progress_total = 0
    for model, fleet in FLEETS.items():
        entries = registry(fleet)
        for entry in entries:
            if entry.get("status") in ("failed", "unhealthy"):
                failed_now.add(str(entry.get("job_id")))
        new_failed = [e for e in entries if e.get("status") in ("failed", "unhealthy")
                      and str(e.get("job_id")) not in known_failed]
        if new_failed and not first:
            attention.append(f"{fleet}: new failed/unhealthy workers " + ", ".join(
                f"w{e.get('slot')} {e.get('job_id')} {e.get('status')} {e.get('node')} {e.get('reason') or ''}"
                for e in new_failed))
        n_route = sum(routable(e, now) for e in entries)
        n_alive_later = sum(alive(e, now + LOOKAHEAD_S, now) for e in entries)
        by_status = {}
        for e in entries:
            if now - (e.get("heartbeat_at") or 0) <= 300 or e.get("status") in ("failed", "unhealthy", "stopped"):
                by_status[e.get("status")] = by_status.get(e.get("status"), 0) + 1
        ends = sorted(time.strftime("%H:%M", time.localtime(e["end_epoch"])) for e in entries
                      if alive(e, now, now) and e.get("end_epoch"))
        lines.append(f"{fleet}: routable {n_route}, alive+75m {n_alive_later}, {by_status}, ends {ends}")

        unfinished = []
        waiting_short = []
        for group in GROUPS:
            dstate, last, job, events = driver_state(model, group)
            done, total = finished_slots(model, group)
            progress_total += done
            has_result = f"{model}-group-{group}" in results
            tag = f"g{group}:{dstate}{'(' + ','.join(events) + ')' if events else ''} {done}/{total}" + (
                " R" if has_result else "")
            lines.append(f"  {model} {tag} job={job} | {last[:110]}")
            key = f"{model}-g{group}"
            if not has_result:
                unfinished.append(group)
            seen = state.get("ended", {}).get(key)
            if dstate == "ended" and seen != job and not first:
                attention.append(f"driver {key} job {job} ended without results: {events} | {last[:120]}")
            if dstate == "waiting" and n_route < MIN_WORKERS:
                waiting_short.append(group)
        key = f"waitshort-{model}"
        if waiting_short:
            since = state.get(key) or now
            state[key] = since
            if now - since > 900:
                attention.append(f"{model}: drivers {waiting_short} waiting with only {n_route} routable workers "
                                 f"for {int((now - since) / 60)} min")
        else:
            state.pop(key, None)
        if unfinished and n_alive_later < MIN_WORKERS and now >= snooze.get(fleet, 0):
            attention.append(f"{fleet}: coverage: only {n_alive_later} workers alive (not leaving) at "
                             f"{time.strftime('%H:%M', time.localtime(now + LOOKAHEAD_S))}; unfinished groups "
                             f"{unfinished}")

    # driver-ended bookkeeping
    ended = state.get("ended", {})
    for model in FLEETS:
        for group in GROUPS:
            dstate, _, job, _ = driver_state(model, group)
            if dstate == "ended":
                ended[f"{model}-g{group}"] = job
    state["ended"] = ended

    if progress_total != state.get("progress_total"):
        state["progress_total"] = progress_total
        state["progress_at"] = now
    stalled_s = now - state.get("progress_at", now)
    lines.append(f"finished slots total {progress_total} (last change {int(stalled_s / 60)} min ago)")
    any_campaign = any(driver_state(m, g)[0] == "campaign" for m in FLEETS for g in GROUPS)
    if any_campaign and stalled_s > 7200:
        attention.append(f"no finished slot for {int(stalled_s / 60)} min while campaigns run")
    if args.wake_at and now >= args.wake_at:
        attention.append("scheduled wake-up reached")

    state["results"] = sorted(results)
    state["failed"] = sorted(known_failed | failed_now)
    tmp = state_path.with_name("." + state_path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(state_path)
    print("\n".join(lines))
    for item in attention:
        print("ATTENTION: " + item)
    return 1 if attention else 0


if __name__ == "__main__":
    raise SystemExit(main())
