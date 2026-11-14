#!/usr/bin/env python3
"""One tick of the read-only watcher for glm-rollouts-tasksall (tiers 3 / 4 resumed 2026-11-13). Deterministic: the
rules are code here, the workflow subagent only runs this and passes the JSON on (2026-11-13: agents can no longer
`sleep` between checks, so the wait lives in this script, like relay_tick.py).

    /usr/bin/python3 -I -B watch_tick.py [--wait 420] [--slow-every 3] [--reset-reported]

Each tick: for up to --wait s, every 30 s, the QUICK checks (squeue, keeper-status.json, new keeper-log.jsonl events,
server status files); it returns early as soon as a NEW alarm shows up. Every --slow-every-th tick, while campaigns are
running rollouts, also the SLOW checks (refresh progress.json with progress.py when it is older than 25 min, graded
stall, per-campaign infra / pass-rate rules, router errors, engine logs, MTP acceptance, PTY interruptions).
Prints ONE JSON line: {"state": "running" | "needs_attention" | "complete", "summary", "alarms": [{"key", "message"}],
"snapshot": {...}}. State between ticks (log offset, histories, which alarms were reported) is in watch-state.json next
to this file. An alarm key is reported once; it is reported again when its condition cleared and came back, or after
6 h while it lasts (log-event alarms: after 2 h). Never submits, cancels or edits anything except its own state file and
progress.py's own outputs.

Alarms (state needs_attention):
  keeper_missing            no gta-keeper job in squeue (a PENDING successor counts as present)
  status_stale              keeper-status.json older than 15 min
  keeper_halted / keeper_stopped   status halted / stop_reason other than "all campaigns done"
  parked:<campaign>         a campaign is parked (needs a human)
  log:<event>               new keeper-log events leg_submit_failed, keeper_exception, config_error,
                            config_unreadable, stale_pin_cancel_failed, halted, parked, fast_fail (>= 2 in 2 h)
  churn                     > 3 stale_pin_cancelled or > 8 leg_gate_cancel events within 30 min
  leg_wait                  campaigns wait for a CPU host node for > 2 h
  nodes:<partition>         own gta-* jobs run on > 2 distinct cpu nodes or > 4 distinct cpu_interactive nodes
  leg_pending:<job name>    a gta-t* job PENDING for > 60 min
  no_ready_server           0 ready servers for > 60 min although a glm53-serve-ta job has been RUNNING > 60 min
  server:<job>              a RUNNING server whose status file ends in failed / unhealthy
  overload                  working load > 1.75 x capacity for > 70 min (the keeper sheds after 45)
  first_ready               ONE-TIME notice: the first GLM server is ready (main agent: submit the tier-2 re-runs)
  squeue_failing            squeue failed 5 times in a row
  (slow) graded_stall       working legs older than 75 min exist and the graded count has not grown for 80 min
  (slow) infra:<campaign>   > 10% of >= 40 terminal slots sandbox-failed / without a grade
  (slow) passrate:<campaign>  >= 50 graded and pass rate < 15%
  (slow) router_errors      > 75 router errors in the last 30 min (progress.json)
  (slow) engine:<job>       Traceback / CUDA error / out of memory / multimem timeout in a RUNNING server's log tail
  (slow) mtp:<job>          Mean acceptance length <= 1.1 on a RUNNING server
  (slow) pty                > 10 new agent_interrupted events of the working campaigns since the last slow check
state complete: stop_reason says all campaigns are done.
"""
import argparse
import glob
import json
import os
from pathlib import Path
import re
import subprocess
import time

HERE = Path(__file__).resolve().parent
GR = Path(os.environ.get("WATCH_GR") or HERE.parent)                   # WATCH_* : offline tests only
G = Path(os.environ.get("WATCH_G") or "/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve")
STATE = Path(os.environ.get("WATCH_STATE") or HERE / "watch-state.json")
SQUEUE_FILE = os.environ.get("WATCH_SQUEUE_FILE")                      # a file with fake squeue output
LOG_ALARM_EVENTS = {"leg_submit_failed", "keeper_exception", "config_error", "config_unreadable",
                    "stale_pin_cancel_failed", "halted", "parked"}
NODE_LIMITS = {"cpu": 2, "cpu_interactive": 4}
REPEAT_S, LOG_REPEAT_S = 6 * 3600, 2 * 3600
SLOW_KEYS = ("graded_stall", "infra", "passrate", "router_errors", "engine", "mtp", "pty")  # re-armed by slow() only


def sh(args, timeout=60):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (subprocess.SubprocessError, OSError) as error:
        return subprocess.CompletedProcess(args, 99, "", str(error))


def load(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def save_state(state):
    tmp = STATE.with_name(".watch-state.json.tmp")
    tmp.write_text(json.dumps(state, indent=1) + "\n")
    os.replace(tmp, STATE)


def parse_time(text):
    """Local time of 'YYYY-MM-DDTHH:MM:SS[+-zzzz]' (keeper files, squeue %V) as a timestamp; None when unparsable."""
    try:
        return time.mktime(time.strptime((text or "")[:19], "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, TypeError):
        return None


def elapsed_s(text):
    """squeue %M (D-HH:MM:SS, HH:MM:SS, MM:SS) -> seconds."""
    days = 0
    if "-" in text:
        day, text = text.split("-", 1)
        days = int(day)
    parts = [int(p) for p in text.split(":") if p.isdigit()]
    while len(parts) < 3:
        parts.insert(0, 0)
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def squeue_rows():
    if SQUEUE_FILE:
        out = subprocess.CompletedProcess([], 0, Path(SQUEUE_FILE).read_text(), "")
    else:
        out = sh(["squeue", "--me", "-h", "-o", "%i|%j|%T|%P|%N|%M|%r|%V"])
    if out.returncode != 0:
        return None
    rows = []
    for line in out.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 8 and parts[1].startswith(("gta-", "glm53-serve-ta-")):
            rows.append(dict(zip(("job", "name", "state", "partition", "node", "elapsed", "reason", "submitted"), parts)))
    return rows


def server_status(job):
    try:
        lines = (G / "logs" / "serve" / f"{job}.status").read_text().splitlines()
    except OSError:
        return None
    return lines[-1].split(" ", 1)[0] if lines else None


def new_log_events(state):
    """Events appended to keeper-log.jsonl since the last tick (the offset restarts when the file shrank)."""
    path = GR / "keeper-log.jsonl"
    try:
        size = path.stat().st_size
    except OSError:
        return []
    offset = state.get("log_offset")
    if offset is None or offset > size:
        offset = size  # first tick: only what comes from now on
    events = []
    with open(path, "rb") as stream:
        stream.seek(offset)
        data = stream.read()
    end = data.rfind(b"\n") + 1
    for line in data[:end].splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    state["log_offset"] = offset + end
    return events


def quick(state, now):
    """(alarms {key: message}, snapshot, complete?) from squeue + keeper files. Conditions only; de-dup is the caller's."""
    alarms, snap = {}, {}
    rows = squeue_rows()
    if rows is None:
        state["squeue_fails"] = state.get("squeue_fails", 0) + 1
        if state["squeue_fails"] >= 5:
            alarms["squeue_failing"] = f"squeue failed {state['squeue_fails']} times in a row"
        rows = []
    else:
        state["squeue_fails"] = 0
        if not any(r["name"] == "gta-keeper" for r in rows):
            alarms["keeper_missing"] = "no gta-keeper job in squeue"
    status = load(GR / "keeper-status.json", {})
    at = parse_time(status.get("at") or "")
    if at is None or now - at > 900:
        alarms["status_stale"] = f"keeper-status.json is {round((now - at) / 60) if at else '?'} min old"
    sched = status.get("campaigns") or {}
    gate = sched.get("leg_gate") or {}
    ready = status.get("ready_servers", 0) or 0
    complete = "all" in (status.get("stop_reason") or "") and "campaigns done" in (status.get("stop_reason") or "") \
        and bool(sched.get("all_done"))
    if status.get("halted"):
        alarms["keeper_halted"] = f"keeper halted: {status['halted']}"
    if status.get("stop_reason") and not complete:
        alarms["keeper_stopped"] = f"keeper stop_reason: {status['stop_reason']}"
    for parked in sched.get("parked") or []:
        alarms[f"parked:{parked['name']}"] = (f"campaign {parked['name']} is parked (legs {parked.get('legs')}, failed "
                                              f"submits {parked.get('failed_submits')}, STOP file {parked.get('stop_file')})")
    # keeper-log events
    recent = [e for e in state.get("events", []) if now - e[0] < 2 * 3600]
    for event in new_log_events(state):
        name = event.get("event")
        when = parse_time(event.get("at") or "") or now
        if name in LOG_ALARM_EVENTS or name in ("fast_fail", "stale_pin_cancelled", "leg_gate_cancel"):
            recent.append([when, name, json.dumps(event)[:300]])
    state["events"] = recent
    for name in LOG_ALARM_EVENTS:
        hits = [e for e in recent if e[1] == name and now - e[0] < LOG_REPEAT_S]
        if hits:
            alarms[f"log:{name}"] = f"{len(hits)} {name} event(s) in keeper-log.jsonl within 2 h, last: {hits[-1][2]}"
    if sum(1 for e in recent if e[1] == "fast_fail") >= 2:
        alarms["log:fast_fail"] = "2 or more server fast-fails within 2 h (keeper-log.jsonl)"
    stale = sum(1 for e in recent if e[1] == "stale_pin_cancelled" and now - e[0] < 1800)
    cancels = sum(1 for e in recent if e[1] == "leg_gate_cancel" and now - e[0] < 1800)
    if stale > 3 or cancels > 8:
        alarms["churn"] = f"{stale} stale_pin_cancelled and {cancels} leg_gate_cancel events within 30 min"
    wait_since = parse_time(status.get("leg_wait_since") or "")
    if sched.get("leg_wait") and wait_since and now - wait_since > 2 * 3600:
        alarms["leg_wait"] = f"campaigns wait for a CPU host node since {status['leg_wait_since']}: {sched['leg_wait']}"
    # own jobs
    nodes = {}
    for row in rows:
        if row["name"].startswith("gta-") and row["state"] == "RUNNING" and row["node"]:
            nodes.setdefault(row["partition"], set()).add(row["node"])
        if row["name"].startswith("gta-t") and row["state"] == "PENDING":
            submitted = parse_time(row["submitted"])
            if submitted and now - submitted > 3600:
                alarms[f"leg_pending:{row['name']}"] = f"{row['name']} ({row['job']}) PENDING for {round((now - submitted) / 60)} min ({row['reason']})"
    for partition, limit in NODE_LIMITS.items():
        if len(nodes.get(partition, ())) > limit:
            alarms[f"nodes:{partition}"] = f"gta-* jobs run on {len(nodes[partition])} distinct {partition} nodes (> {limit}): {sorted(nodes[partition])}"
    servers = [r for r in rows if r["name"].startswith("glm53-serve-ta-")]
    running = [r for r in servers if r["state"] == "RUNNING"]
    for row in running:
        word = server_status(row["job"])
        if word in ("failed", "unhealthy"):
            alarms[f"server:{row['job']}"] = f"server {row['name']} ({row['job']}) status file ends in {word}"
    old_running = [r for r in running if elapsed_s(r["elapsed"]) > 3600]
    if ready == 0 and old_running:
        state.setdefault("zero_ready_since", now)
        if now - state["zero_ready_since"] > 3600:
            alarms["no_ready_server"] = f"0 ready servers for {round((now - state['zero_ready_since']) / 60)} min while {len(old_running)} server job(s) have been RUNNING > 60 min"
    else:
        state.pop("zero_ready_since", None)
    capacity = gate.get("capacity") or 0
    if ready >= 1 and (gate.get("working_load") or 0) > 1.75 * capacity:
        state.setdefault("overload_since", now)
        if now - state["overload_since"] > 70 * 60:
            alarms["overload"] = f"working load {gate.get('working_load')} > 1.75 x capacity {capacity} for {round((now - state['overload_since']) / 60)} min"
    else:
        state.pop("overload_since", None)
    if ready >= 1 and not state.get("first_ready_reported"):
        alarms["first_ready"] = f"first GLM server ready ({ready} ready): submit the pending tier-2 re-runs (dip2r1-h-n3 / n4)"
    snap.update({"ready": ready, "servers_running": len({r["name"] for r in running}),
                 "servers_pending": len({r["name"] for r in servers if r["state"] == "PENDING"}),
                 "working": len(gate.get("working") or []), "working_load": gate.get("working_load") or 0,
                 "held_back": len(gate.get("held_back") or []), "capacity": capacity,
                 "legs": sorted({r["name"][4:] for r in rows if r["name"].startswith("gta-t")}),
                 "done": sched.get("done"), "parked": len(sched.get("parked") or []),
                 "leg_nodes": {p: sorted(v) for p, v in nodes.items()}, "keeper_status_age_min": round((now - at) / 60) if at else None,
                 "working_names": gate.get("working") or [], "running_jobs": [r["job"] for r in running],
                 "oldest_leg_min": max([elapsed_s(r["elapsed"]) // 60 for r in rows if r["name"].startswith("gta-t")
                                        and r["state"] == "RUNNING"] or [0])})
    return alarms, snap, complete


def tail(path, size=1_000_000):
    try:
        with open(path, "rb") as stream:
            stream.seek(0, 2)
            end = stream.tell()
            stream.seek(max(0, end - size))
            return stream.read().decode("utf-8", "replace")
    except OSError:
        return ""


def slow(state, snap, now):
    """The expensive checks; only while campaigns are running rollouts."""
    alarms = {}
    progress = load(GR / "progress.json", {})
    try:
        age = now - (GR / "progress.json").stat().st_mtime
    except OSError:
        age = 1e9
    if age > 1500:
        sh(["/usr/bin/python3", "-I", "-B", str(GR / "progress.py"), "--stdout"], timeout=240)
        progress = load(GR / "progress.json", progress)
    total = progress.get("total") or {}
    graded = total.get("graded")
    history = [h for h in state.get("graded", []) if now - h[0] < 6 * 3600]
    if graded is not None:
        history.append([now, graded])
    state["graded"] = history
    snap.update({"graded": graded, "passed": total.get("passed"),
                 "gen_tok_s": (progress.get("router") or {}).get("gen_tok_per_s_last_30min")})
    if snap.get("working") and snap.get("oldest_leg_min", 0) > 75 and graded is not None:
        old = [h for h in history if now - h[0] >= 80 * 60]
        if old and old[-1][1] >= graded:
            alarms["graded_stall"] = f"graded count {graded} has not grown for >= 80 min while {snap['working']} campaign(s) run rollouts"
    for row in progress.get("rows") or []:
        name = row.get("campaign", "")
        if not row.get("started") or row.get("complete"):
            continue
        bad = (row.get("sandbox_failed") or 0) + (row.get("ungraded") or 0)
        terminal = (row.get("graded") or 0) + bad
        if terminal >= 40 and bad > 0.10 * terminal:
            alarms[f"infra:{name}"] = f"{name}: {bad} of {terminal} terminal slots sandbox-failed / without a grade"
        if (row.get("graded") or 0) >= 50 and (row.get("passed") or 0) < 0.15 * row["graded"]:
            alarms[f"passrate:{name}"] = f"{name}: pass rate {row.get('passed')}/{row['graded']} < 15%"
    errors = (progress.get("router") or {}).get("router_errors_last_30min") or {}
    count = sum(v for v in errors.values() if isinstance(v, (int, float))) if isinstance(errors, dict) else 0
    if count > 75:
        alarms["router_errors"] = f"{count} router errors in the last 30 min: {json.dumps(errors)[:200]}"
    for job in snap.get("running_jobs", []):
        text = tail(G / "logs" / "serve" / f"{job}_vllm_rank0.log")
        bad = re.findall(r"Traceback \(most recent call last\)|CUDA error|out of memory|OutOfMemoryError|multimem timeout", text)
        if bad:
            alarms[f"engine:{job}"] = f"server job {job}: {len(bad)} error marker(s) in the engine log tail ({sorted(set(bad))})"
        accept = re.findall(r"Mean acceptance length: ([0-9.]+)", text)
        if accept and float(accept[-1]) <= 1.1 and server_status(job) == "ready":
            alarms[f"mtp:{job}"] = f"server job {job}: MTP mean acceptance length {accept[-1]}"
    interrupted = 0
    for name in snap.get("working_names", []):
        pattern = str(GR / "probes" / "glmlocal" / f"{name}*" / "models" / "glmlocal" / "attempts" / "*" / "attempt-*" / "capture-events.jsonl")
        for path in glob.glob(pattern):
            try:
                with open(path, "rb") as stream:
                    interrupted += stream.read().count(b"agent_interrupted")
            except OSError:
                continue
    previous = state.get("pty", {})
    key = ",".join(sorted(snap.get("working_names", [])))
    if previous.get("key") == key and interrupted - previous.get("count", 0) > 10:
        alarms["pty"] = f"{interrupted - previous['count']} new PTY interruptions (agent_interrupted) since the last slow check"
    state["pty"] = {"key": key, "count": interrupted}
    snap["pty_interruptions"] = interrupted
    return alarms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", type=float, default=420)
    parser.add_argument("--slow-every", type=int, default=3)
    parser.add_argument("--reset-reported", action="store_true", help="forget which alarms were reported (report again)")
    args = parser.parse_args()
    state = load(STATE, {})
    if args.reset_reported:
        state["reported"] = {}
    state["ticks"] = state.get("ticks", 0) + 1
    reported = state.setdefault("reported", {})
    deadline = time.time() + args.wait
    new, alarms, snap, complete = {}, {}, {}, False
    while True:
        now = time.time()
        alarms, snap, complete = quick(state, now)
        if "first_ready" in alarms:
            state["first_ready_reported"] = True
        for key in list(reported):  # a cleared condition is re-armed (log events re-arm by time)
            if key not in alarms and not key.startswith("log:") and key.split(":")[0] not in SLOW_KEYS:
                del reported[key]
        new = {k: m for k, m in alarms.items()
               if k not in reported or now - reported[k] > (LOG_REPEAT_S if k.startswith("log:") else REPEAT_S)}
        if new or complete or now >= deadline:
            break
        save_state(state)
        time.sleep(min(30, max(1, deadline - now)))
    if not new and not complete and snap.get("working") and state["ticks"] % max(1, args.slow_every) == 0:
        now = time.time()
        slow_alarms = slow(state, snap, now)
        alarms.update(slow_alarms)
        for key in [k for k in reported if k.split(":")[0] in SLOW_KEYS and k not in slow_alarms]:
            del reported[key]
        new = {k: m for k, m in slow_alarms.items() if k not in reported or now - reported[k] > REPEAT_S}
    elif "graded" not in snap:
        total = load(GR / "progress.json", {}).get("total") or {}
        snap.update({"graded": total.get("graded"), "passed": total.get("passed")})
    for key in new:
        reported[key] = time.time()
    save_state(state)
    for drop in ("working_names", "running_jobs"):
        snap.pop(drop, None)
    state_word = "complete" if complete else "needs_attention" if new else "running"
    summary = (f"graded {snap.get('graded')} / pass {snap.get('passed')}; campaigns done {snap.get('done')}, parked "
               f"{snap.get('parked')}; servers {snap.get('ready')} ready / {snap.get('servers_running')} running / "
               f"{snap.get('servers_pending')} pending; legs working {snap.get('working')} (load {snap.get('working_load')} "
               f"on capacity {snap.get('capacity')}), held back {snap.get('held_back')}"
               + (f"; NEW ALARMS: {', '.join(new)}" if new else "")
               + (f"; known alarms still active: {', '.join(k for k in alarms if k not in new)}" if set(alarms) - set(new) else ""))
    print(json.dumps({"state": state_word, "summary": summary,
                      "alarms": [{"key": k, "message": m} for k, m in new.items()], "snapshot": snap}))


if __name__ == "__main__":
    main()
