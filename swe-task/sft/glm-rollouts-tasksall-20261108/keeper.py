#!/usr/bin/env python3
"""glm-rollouts-tasksall keeper (2026-11-08; derived from glm-rollouts-sftv3-20261107/keeper.py): keeps the GLM-5.3
server slots alive AND schedules the campaigns of tasks.json in priority order.

    /usr/bin/python3 -I -B keeper.py [--once]     # runs in a CPU job (keeper.sbatch); state in keeper-status.json

Every --interval s:
* Servers (glm53-serve 2-node jobs, submitted with glm53-serve/scripts/submit_serve.sh): pool "main" = slots ta-NN
  (batch, 4 h; target = int(servers-target.txt), replacement lead = lead_minutes) and pool "short" = ta-sNN
  (batch_short, 2 h; target = config short_target, lead = short_lead_minutes). A slot is LIVE when a candidate is
  PENDING, or RUNNING with more than its lead left; while live < target a new slot is submitted (at most max_per_hour
  per rolling hour; after max_fast_fails slots in a row that ended within fast-fail-minutes the keeper stops
  submitting servers: needs a human). Live > target: pending slots are cancelled only if config scale_down is true.
* Campaigns (tasks.json "campaigns", tier-major order): a campaign is DONE when results/glmlocal-<c>.json has the
  driver's sandbox_failure_rule, ACTIVE when released (GO-<c>) and not done. Its load = min(max_concurrency, slots
  not yet terminal) (terminal = the slot's latest attempt has terminal.json). With config "release": true the next
  campaigns in order are released (GO-<c>) while total load + the new campaign's load <= per_server x ready servers
  and the owner's load stays within owner_caps[owner] (a campaign blocked only by its owner's cap is skipped, so the
  other owner's next campaigns can go; within an owner the order is kept), at most max_active campaigns.
  Every active campaign without a driver job (gta-<c>, any state) gets a leg (bash submit.sh <c> k<NN>: a fresh babysit
  state dir per leg, so a lost sbatch response cannot wedge the campaign): at most max_legs successful submissions and
  max_submit_failures failed ones per campaign (keeper-legs.json, persists across keeper jobs), one per leg_gap_minutes;
  babysit/<c>/STOP blocks it. A campaign that hit a limit or has STOP and no driver job is PARKED: it holds no load and
  no max_active slot (needs a human; listed in keeper-status.json).
* keeper-config.json is re-read every pass (keys: lead_minutes, max_per_hour, max_fast_fails, release, per_server,
  owner_caps, max_active, max_legs, leg_gap_minutes, venv, short_target, short_lead_minutes, scale_down,
  cancel_running_when_done).
* STOP file (keeper.STOP) or every campaign done or parked with no driver job left: stop submitting, cancel pending
  server slots (and RUNNING ones too when cancel_running_when_done is true), write the status and exit 0.
* Child processes (submit.sh, submit_serve.sh) get a scrubbed environment: no BABYSIT_* / RESULTS_DIR / EXP_NAME /
  BATCH_SCRIPT of the keeper's own babysit job (they made the helper reuse the keeper's leg number) and none of its
  resource-shaping SLURM_* variables (2026-11-07: they confined server engine steps to 1 CPU / 2 GB).
* Circuit breaker: when breaker_parked campaigns got parked within breaker_hours, no new campaign is released (a
  shared cause would otherwise park the whole plan one campaign after another); running ones continue. Un-park: raise
  max_legs / max_submit_failures in keeper-config.json or remove babysit/<c>/STOP (the next leg tag is 1 + the highest
  babysit/<c>/kNN); parked_at resets when a campaign is no longer parked.
* Held leftovers: a gta-t* / glm53-serve-ta-* job PENDING with reason JobHeldUser for > 15 min (a babysit submit that
  failed half-way) is cancelled, so it cannot block its campaign or server slot forever.
* Servers: never more than max_servers server slots (pending + running, both pools) at once.
* GPU outage: no ready server for outage_minutes -> no releases, no new legs, and the RUNNING gta-t*/gta-dip* legs
  that got past their backend wait (their log has "probe: staging ATP": rollouts in flight) are cancelled once (their
  in-flight attempts become retryable for the next leg) instead of burning ATP's 3 h agent timeout; normal service
  resumes with the first ready server. (Legs still waiting for a backend: the leg capacity gate cancels them.)
* Tier pause: with pause_after_tier N, once every campaign of tier <= N is done the keeper writes TIER-PAUSE.json and
  from then on runs only tiers <= N (no releases or legs for the others; their RUNNING/PENDING legs are cancelled
  once; in-flight attempts become retryable). Delete TIER-PAUSE.json to continue all tiers.
* Idle GPUs: no campaign left to run (done or tier-paused) and no gta-dip* job -> after idle_release_minutes every
  server slot is cancelled and none is submitted until there is work again.
* Leg placement (2026-11-13; user 2026-11-10: use less of cpu_interactive). The CPU partition QoS limits count
  DISTINCT nodes per user (cpu: 2, cpu_interactive: 10; jobs sharing a node count once), so every driver leg is PINNED
  (--nodelist; leg_cpus CPUs, leg_mem_gb GB: legs measured <= 0.3 core, MaxRSS <= 15.75 GiB) to a "host" node: first
  the hosts of partition cpu (at most leg_cpu_nodes), then those of cpu_interactive (at most leg_host_nodes). A host is
  a node that already runs own gta-* jobs in that partition or has a pinned leg waiting; while a partition has fewer
  hosts than its cap, the usable node (idle / mixed) with the most free memory becomes one. No host with room: the leg
  waits ("leg_wait" in the status); every leg_wait_minutes of waiting allows one more cpu_interactive host, up to
  leg_host_nodes_max. A leg whose pinned candidates all stayed PENDING for pin_pending_minutes is cancelled (scancel
  must succeed), its node is avoided for an hour and the leg is placed again in a later pass; the first 3 such legs of
  a campaign do not count against max_legs.
* Leg capacity gate (2026-11-13, config leg_gate): a leg runs its campaign's rollouts as soon as ANY server is ready,
  so 16 released campaigns on 2 ready servers would put ~7x the designed load on them and the rollouts would run into
  ATP's 3 h agent timeout (the 2026-11-09 dip: 62 such fails). So legs follow the ready capacity: start capacity =
  per_server x ready servers x leg_start_factor. In campaign order, a campaign without a working leg gets / keeps a leg
  only while (load of the working legs + load of the legs already admitted + its load) fits the start capacity (the
  first one always fits while a server is ready, and so does a campaign with no slot left to run: its leg only
  writes results); a leg that has not staged ATP yet (pending, or waiting for a backend) and does not fit is cancelled
  at once (it has no rollout in flight; so with 0 ready servers there are no waiting legs). Working legs (log has
  "probe: staging ATP") are shed only after a LASTING overload: their load above per_server x ready servers x
  leg_shed_factor for leg_shed_minutes without a break, or above x leg_shed_hard_factor for leg_shed_hard_minutes
  (2026-11-13 simulation of a thin fleet: 1.75x for 15 min killed 384 in-flight rollouts for nothing, 45 min none,
  both without 3 h timeouts). Then exactly the working legs that the admission rule would not keep are cancelled, the
  last campaigns first (in-flight attempts become retryable, as with the outage pause). Legs cancelled by the gate do
  not count against max_legs ("free_legs", once per job).
* Its own 23 h limit: with less than successor_minutes left it submits a successor keeper (submit_keeper.sh with
  --dependency=afterany:<this job>) once, then keeps working until the end. A pass where squeue does not list this job
  is skipped (an empty squeue answer must not look like "nothing is running").
"""
import argparse
import collections
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

GR = Path(__file__).resolve().parent
G = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve")
SUBMIT = G / "scripts" / "submit_serve.sh"
ALIAS = "glmlocal"


def sh(args, timeout=120):
    """Read-only / cancel Slurm calls with this process's env; never raises."""
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (subprocess.SubprocessError, OSError) as error:
        return subprocess.CompletedProcess(args, 99, "", f"{type(error).__name__}: {error}")


def parse_left(text):
    text = text.strip()
    if not text or text in ("UNLIMITED", "NOT_SET", "INVALID"):
        return None
    days = 0
    if "-" in text:
        day, text = text.split("-", 1)
        days = int(day)
    parts = [int(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def my_jobs():
    out = sh(["squeue", "--me", "-h", "-o", "%i|%j|%T|%L|%M|%r|%V|%S"])
    if out.returncode != 0:
        raise RuntimeError(f"squeue failed: {out.stderr[:200]}")
    jobs = collections.defaultdict(list)
    for line in out.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 8:
            stamps = []
            for text in (parts[6], parts[7]):
                try:
                    stamps.append(time.mktime(time.strptime(text, "%Y-%m-%dT%H:%M:%S")))
                except ValueError:
                    stamps.append(None)
            jobs[parts[1]].append({"job": parts[0], "state": parts[2], "left": parse_left(parts[3]),
                                   "elapsed": parts[4], "reason": parts[5], "submitted": stamps[0],
                                   "started": stamps[1]})
    return dict(jobs)


def cancel_stuck_held(jobs, now, log_path, prefixes=("gta-t", "glm53-serve-ta-"), max_age_s=900):
    """A babysit submit that failed half-way leaves a user-HELD candidate that never runs; it would block the campaign
    (its gta-<c> job "exists") or the server slot forever. Cancel such own jobs after max_age_s; returns the names."""
    stuck = []
    for name, entries in jobs.items():
        if not name.startswith(prefixes):
            continue
        held = [j for j in entries if j["state"] == "PENDING" and j.get("reason") == "JobHeldUser"
                and j.get("submitted") and now - j["submitted"] > max_age_s]
        if held and all(j["state"] == "PENDING" for j in entries):
            sh(["scancel", *[j["job"] for j in held]])
            log(log_path, event="stuck_held_cancelled", name=name, jobs=[j["job"] for j in held])
            stuck.append(name)
            entries[:] = [j for j in entries if j not in held]
    for name in [n for n, e in jobs.items() if not e]:
        del jobs[name]
    return stuck


def ready(job):
    """Is this glm53-serve job's engine up (status file: a 'ready' line with nothing terminal after it)?"""
    try:
        lines = (G / "logs" / "serve" / f"{job}.status").read_text().splitlines()
    except OSError:
        return False
    state = False
    for line in lines:
        word = line.split(" ", 1)[0]
        if word == "ready":
            state = True
        elif word in ("exited", "failed", "unhealthy"):
            state = False
    return state


def log(path, **fields):
    with open(path, "a") as stream:
        stream.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields}) + "\n")


def write_json_atomic(path: Path, value) -> None:
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n")
    os.replace(tmp, path)


CHILD_DROP = {"SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_MEM_PER_GPU", "SLURM_CPUS_PER_TASK",
              "SLURM_CPUS_PER_GPU", "SLURM_TRES_PER_TASK", "RESULTS_DIR", "EXP_NAME", "BATCH_SCRIPT",
              "GR_NODELIST", "GR_CPUS", "GR_MEM", "GR_TIME"}


def child_env(**extra):
    """os.environ without this job's babysit identity and resource-shaping SLURM_* variables, plus extra."""
    env = {k: v for k, v in os.environ.items() if k not in CHILD_DROP and not k.startswith("BABYSIT_")}
    env.update({k: str(v) for k, v in extra.items()})
    return env


def run(args, timeout=300, **env_extra):
    """subprocess.run with a scrubbed env; never raises (returns an object with returncode/stdout/stderr)."""
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=child_env(**env_extra))
    except (subprocess.SubprocessError, OSError) as error:
        return subprocess.CompletedProcess(args, 99, "", f"{type(error).__name__}: {error}")


def checked_config(overrides, log_path, last_good):
    """DEFAULTS overlaid with type-checked overrides; a bad value keeps the last good one (logged once per pass)."""
    config = {}
    for key, default in DEFAULTS.items():
        value = overrides.get(key, default)
        ok = (isinstance(value, bool) if isinstance(default, bool) else
              isinstance(value, dict) and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                              for v in value.values()) if isinstance(default, dict) else
              isinstance(value, (int, float)) and not isinstance(value, bool))
        if not ok:
            log(log_path, event="config_error", key=key, value=repr(value)[:100])
            value = last_good.get(key, default)
        config[key] = value
    last_good.update(config)
    return config


_STARTED_ATP = set()  # (job name, job id) seen past the backend wait: it never goes back


def leg_started_atp(job_name, job_id, unknown=False):
    """Did this driver leg (log logs/<job name>-<job id>.out) get past its backend wait and stage/run ATP?
    unknown: the answer when the log exists but cannot be read (a missing log = the job has not written yet)."""
    if (job_name, job_id) in _STARTED_ATP:
        return True
    try:
        with open(GR / "logs" / f"{job_name}-{job_id}.out", "rb") as stream:
            started = b"probe: staging ATP" in stream.read()
    except FileNotFoundError:
        return False
    except OSError:
        return unknown
    if started:
        _STARTED_ATP.add((job_name, job_id))
    return started


def campaign_done(name):
    path = GR / "results" / f"{ALIAS}-{name}.json"
    try:
        return "sandbox_failure_rule" in json.loads(path.read_text())
    except (OSError, ValueError):
        return False


def terminal_slots(name):
    """Slots of the campaign whose latest attempt has terminal.json (ATP: models/<alias>/attempts/<task>-<rep>/)."""
    count = 0
    root = GR / "probes" / ALIAS / name / "models" / ALIAS / "attempts"
    try:
        slots = [p for p in root.iterdir() if p.is_dir()]
    except OSError:
        return 0
    for slot in slots:
        attempts = [p for p in slot.glob("attempt-*") if p.name.split("-", 1)[1].isdigit()]
        if attempts and (max(attempts, key=lambda p: int(p.name.split("-", 1)[1])) / "terminal.json").exists():
            count += 1
    return count


# Driver-leg placement: (partition, config key of its host-node cap), tried in this order. QoS per user: p_cpu 2
# DISTINCT nodes, p_cpu_short (cpu_interactive) 10 (2026-11-10: 11 own jobs ran on 10 nodes under that limit).
LEG_POOLS = (("cpu", "leg_cpu_nodes"), ("cpu_interactive", "leg_host_nodes"))
QOS_NODES = {"cpu": 2, "cpu_interactive": 10}
STALE_NODE_AVOID_S = 3600
STALE_FREE_LEGS = 3


def leg_rows():
    """ALL own jobs: [{job, name, state, partition, node, pinned, submitted}]; None when squeue fails. partition: where
    it runs, or the list it may run in while PENDING; pinned: its --nodelist. (Other sessions' jobs are listed too:
    their nodes already count against the per-user node limits, so a leg may share them.)"""
    out = sh(["squeue", "--me", "-h", "-o", "%i|%j|%T|%P|%N|%n|%V"])
    if out.returncode != 0:
        return None
    rows = []
    for line in out.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 7:
            try:
                submitted = time.mktime(time.strptime(parts[6], "%Y-%m-%dT%H:%M:%S"))
            except ValueError:
                submitted = None
            rows.append({"job": parts[0], "name": parts[1], "state": parts[2], "partition": parts[3],
                         "node": parts[4], "pinned": parts[5], "submitted": submitted})
    return rows


def pool_nodes(partition):
    """{node: (free CPUs, free memory in MB)} of the partition's nodes that take new jobs (state exactly idle or
    mixed: no drain / reservation / maintenance / planned / down marker); None when sinfo fails."""
    out = sh(["sinfo", "-h", "-p", partition, "-N", "-O",
              "NodeList:40,StateLong:40,CPUsState:30,Memory:15,AllocMem:15"], timeout=60)
    if out.returncode != 0:
        return None
    nodes = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) != 5 or parts[1] not in ("idle", "mixed"):
            continue
        try:
            nodes[parts[0]] = (int(parts[2].split("/")[1]), int(parts[3]) - int(parts[4]))
        except (ValueError, IndexError):
            continue
    return nodes


def leg_claims(rows):
    """Counter node -> pinned legs that hold no allocation there yet (campaigns whose candidates are all waiting)."""
    running = {r["name"] for r in rows if r["state"] == "RUNNING"}
    claims = collections.Counter()
    for name, node in {(r["name"], r["pinned"]) for r in rows
                       if r["name"].startswith("gta-t") and r["state"] != "RUNNING" and r["pinned"]}:
        if name not in running:
            claims[node] += 1
    return claims


def leg_hosts(rows, partition):
    """Counter host node -> gta-* jobs (keeper, legs, re-runs) running there in this partition (0 for a node that only
    has pinned legs waiting for it)."""
    hosts = collections.Counter()
    for row in rows:
        if row["partition"] != partition or not row["name"].startswith("gta-"):
            continue  # a pinned leg has exactly one partition; multi-partition lists never match
        if row["state"] == "RUNNING" and row["node"]:
            hosts[row["node"]] += 1
        elif row["pinned"] and row["name"].startswith("gta-t"):
            hosts[row["pinned"]] += 0
    return hosts


def own_nodes(rows, partition):
    """Nodes of this partition that count against the per-user node limit already: every own RUNNING job (any
    session's) plus the nodes pinned legs wait for."""
    return ({r["node"] for r in rows if r["partition"] == partition and r["state"] == "RUNNING" and r["node"]}
            | set(leg_hosts(rows, partition)))


def place_leg(rows, config, claims, pools, extra_hosts=0, avoid=()):
    """Where the next driver leg goes: ("pin", partition, node), or ("wait", None, reason).
    claims: Counter node -> pinned legs without an allocation yet (each takes one leg's CPUs and memory);
    pools: per-pass cache partition -> pool_nodes(); extra_hosts: additional cpu_interactive hosts allowed after a long
    wait; avoid: nodes that recently never started a pinned leg."""
    need_cpus, need_mem = max(1, int(config["leg_cpus"])), max(4, int(config["leg_mem_gb"])) * 1024
    seen = []
    for partition, key in LEG_POOLS:
        cap = int(config[key])
        if partition == "cpu_interactive":
            cap = min(cap + extra_hosts, int(config["leg_host_nodes_max"]))
        cap = max(0, min(cap, QOS_NODES[partition]))
        if cap == 0:
            continue
        if partition not in pools:
            pools[partition] = pool_nodes(partition)
        free = pools[partition]
        if free is None:
            seen.append(f"{partition}: sinfo failed")
            continue

        def fits(node):
            room = free.get(node)
            return (room is not None and node not in avoid and room[0] - claims[node] * need_cpus >= need_cpus
                    and room[1] - claims[node] * need_mem >= need_mem)

        def best(nodes):  # most free memory (after claims), then most free CPUs, then name
            return max(nodes, key=lambda n: (free[n][1] - claims[n] * need_mem, free[n][0], n))

        hosts = leg_hosts(rows, partition)
        for node, _ in sorted(hosts.items(), key=lambda item: (-item[1], item[0])):
            if fits(node):
                return "pin", partition, node
        mine = own_nodes(rows, partition)
        if len(hosts) < cap:
            shared = [n for n in mine if n not in hosts and fits(n)]  # other own jobs run there: no extra node
            if shared:
                return "pin", partition, best(shared)
            fresh = [n for n in free if n not in mine and fits(n)]
            if fresh and len(mine) < QOS_NODES[partition]:
                return "pin", partition, best(fresh)
        seen.append(f"{partition}: {len(hosts)}/{cap} hosts ({len(mine)}/{QOS_NODES[partition]} own nodes), "
                    f"none with room")
    return "wait", None, "; ".join(seen) or "no leg partition enabled"


def cancel_stale_pins(rows, legs_state, now, config, log_path):
    """A leg pinned to a node that never starts it would wait there for ever. When every candidate of a campaign is
    pinned and has been PENDING for pin_pending_minutes: scancel them and, only if scancel succeeded, give the leg back
    (it never ran; at most STALE_FREE_LEGS times per campaign). The campaign is NOT resubmitted in this pass (its jobs
    stay listed): a later pass, whose squeue no longer shows them, places it again.
    Returns [(job name, nodes, job ids)]."""
    by_name = collections.defaultdict(list)
    for row in rows:
        if row["name"].startswith("gta-t"):
            by_name[row["name"]].append(row)
    stale = []
    limit = max(5, config["pin_pending_minutes"]) * 60
    for name, entries in sorted(by_name.items()):
        if not all(r["state"] == "PENDING" and r["pinned"] and r["submitted"] and now - r["submitted"] > limit
                   for r in entries):
            continue
        result = sh(["scancel", *[r["job"] for r in entries]])
        nodes = sorted({r["pinned"] for r in entries})
        if result.returncode != 0:
            log(log_path, event="stale_pin_cancel_failed", name=name, jobs=[r["job"] for r in entries], nodes=nodes)
            continue
        entry = legs_state.setdefault(name[4:], {"submitted": [], "failed": []})
        entry["stale_pins"] = entry.get("stale_pins", 0) + 1
        first = min(r["submitted"] for r in entries)
        given_back = bool(entry["stale_pins"] <= STALE_FREE_LEGS and entry.get("submitted")
                          and abs(entry["submitted"][-1] - first) < 600)
        if given_back:
            entry["submitted"].pop()
        log(log_path, event="stale_pin_cancelled", name=name, jobs=[r["job"] for r in entries], nodes=nodes,
            given_back=given_back, stale_pins=entry["stale_pins"])
        stale.append((name, nodes, [r["job"] for r in entries]))
    if stale:
        write_json_atomic(Path(log_path).with_name("keeper-legs.json"), legs_state)
    return stale


POOLS = {
    # name -> slot-name regex (Slurm job name glm53-serve-<slot>), slot format, extra submit settings
    "main": {"regex": r"glm53-serve-ta-(\d+)", "slot": "ta-{:02d}", "partition": None, "extra": []},
    "short": {"regex": r"glm53-serve-ta-s(\d+)", "slot": "ta-s{:02d}", "partition": "batch_short",
              "extra": ["--time=02:00:00"]},
}
DEFAULTS = {"release": False, "per_server": 24, "owner_caps": {"hengyuf": 240, "baihe": 48}, "max_active": 16,
            "max_legs": 6, "max_submit_failures": 6, "leg_gap_minutes": 10, "short_target": 0,
            "short_lead_minutes": 25, "scale_down": False, "cancel_running_when_done": False, "successor_minutes": 30,
            "breaker_parked": 3, "breaker_hours": 2, "max_servers": 14, "outage_minutes": 10,
            "pause_after_tier": 0, "idle_release_minutes": 60, "leg_cpu_nodes": 2, "leg_host_nodes": 2,
            "leg_host_nodes_max": 5, "leg_wait_minutes": 30, "leg_cpus": 2, "leg_mem_gb": 24,
            "pin_pending_minutes": 20, "leg_gate": True, "leg_start_factor": 1.25, "leg_shed_factor": 1.75,
            "leg_shed_minutes": 45, "leg_shed_hard_factor": 3.0, "leg_shed_hard_minutes": 15}
def tier_pause_path():
    """Present = only its allowed_tiers run (user 2026-11-09: pause after tier 1). A function, not a constant, so
    that tests which point GR elsewhere never touch the live file."""
    return GR / "TIER-PAUSE.json"


def schedule_campaigns(campaigns, jobs, ready_servers, config, legs_state, now, log_path, stopping, paused=False,
                       allowed=None, rows="fetch", leg_extra_hosts=0, avoid=(), gate=None):
    """Release campaigns in order and keep one driver leg per active campaign; returns (actions, summary).
    allowed: set of tiers that may run (None = all); campaigns of other tiers are PAUSED (no release, no legs).
    rows: leg_rows() of this pass (None = squeue failed: no leg is placed; "fetch" = call it when a leg needs a place);
    leg_extra_hosts / avoid: see place_leg. gate: dict kept by the caller across passes (overload_since) for the
    leg capacity gate; None = no memory (waiting legs are still gated, working legs are never shed)."""
    actions = []
    done = {c["name"] for c in campaigns if campaign_done(c["name"])}
    released = {c["name"] for c in campaigns if (GR / f"GO-{c['name']}").exists()}
    tier_paused = {c["name"] for c in campaigns if allowed is not None and c["tier"] not in allowed}

    def has_job(name):
        return f"gta-{name}" in jobs

    def parked(name):
        entry = legs_state.get(name) or {}
        return not has_job(name) and ((GR / "babysit" / name / "STOP").exists()
                                      or len(entry.get("submitted", [])) - entry.get("free_legs", 0) >= config["max_legs"]
                                      or len(entry.get("failed", [])) >= config["max_submit_failures"])

    active = [c for c in campaigns if c["name"] in released and c["name"] not in done and not parked(c["name"])
              and c["name"] not in tier_paused]
    parked_names = [c["name"] for c in campaigns if c["name"] in released and c["name"] not in done
                    and parked(c["name"]) and c["name"] not in tier_paused]
    for name in parked_names:  # circuit breaker input: when each campaign got parked
        entry = legs_state.setdefault(name, {"submitted": [], "failed": []})
        if not entry.get("parked_at"):
            entry["parked_at"] = now
            log(log_path, event="parked", campaign=name, legs=len(entry.get("submitted", [])),
                failed_submits=len(entry.get("failed", [])), stop_file=(GR / "babysit" / name / "STOP").exists())
            write_json_atomic(Path(log_path).with_name("keeper-legs.json"), legs_state)
    for name, entry in legs_state.items():
        if entry.get("parked_at") and name not in parked_names:
            entry["parked_at"] = None  # un-parked by a human (limits raised / STOP removed)
    recent = [n for n, e in legs_state.items()
              if e.get("parked_at") and now - e["parked_at"] < config["breaker_hours"] * 3600]
    breaker = len(recent) >= config["breaker_parked"]
    loads = {c["name"]: max(0, min(c["max_concurrency"], c["slots"] - terminal_slots(c["name"]))) for c in active}
    total = sum(loads.values())
    owner_load = collections.Counter()
    for c in active:
        owner_load[c["sandbox"]] += loads[c["name"]]
    created = []
    if breaker and config["release"] and not stopping:
        actions.append({"breaker": f"{len(recent)} campaigns parked within {config['breaker_hours']} h: no new "
                                   f"releases until a human looks", "parked": recent})
    if config["release"] and not stopping and not paused and not breaker and ready_servers >= 1:
        capacity = config["per_server"] * ready_servers
        blocked_owners = set()
        for c in campaigns:
            name = c["name"]
            if name in released or name in done or name in tier_paused:
                continue
            if len(active) + len(created) >= config["max_active"]:
                break
            owner = c["sandbox"]
            if owner in blocked_owners:
                continue
            need = min(c["max_concurrency"], c["slots"])
            if owner_load[owner] + need > config["owner_caps"].get(owner, 10 ** 9):
                blocked_owners.add(owner)  # its own sandbox cap: the other owner's next campaigns may go
                continue
            if total + need > capacity and (active or created):
                break
            (GR / f"GO-{name}").write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                                       "ready_servers": ready_servers, "load_before": total,
                                                       "need": need, "capacity": capacity}) + "\n")
            created.append(name)
            released.add(name)
            total += need
            owner_load[owner] += need
            loads[name] = need
        if created:
            log(log_path, event="released", campaigns=created, ready_servers=ready_servers, load=total)
            actions.append({"released": created})
    active = [c for c in campaigns if c["name"] in released and c["name"] not in done and not parked(c["name"])
              and c["name"] not in tier_paused]
    for c in active:
        if c["name"] not in loads:
            loads[c["name"]] = max(0, min(c["max_concurrency"], c["slots"] - terminal_slots(c["name"])))
    leg_wait = None
    # ---- leg capacity gate: which campaigns may have a leg right now
    gate = {} if gate is None else gate
    if not (config["leg_gate"] and not stopping):
        gate.pop("overload_since", None)  # gate off: the overload clocks must not survive until it is on again
        gate.pop("hard_since", None)
    gate_info = None
    admitted = None  # campaigns without a working leg that may have one (None = gate off: every active campaign)
    if config["leg_gate"] and not stopping:
        capacity = config["per_server"] * ready_servers
        start_cap = capacity * config["leg_start_factor"]

        def leg_jobs(name):
            return jobs.get(f"gta-{name}", [])

        gone = gate.setdefault("cancelled_jobs", set())  # job ids this keeper process cancelled already
        gone &= {j["job"] for js in jobs.values() for j in js}

        def live_jobs(name):  # a COMPLETING job is over already (cancelled before, crashed, finished)
            return [j for j in leg_jobs(name) if j["state"] != "COMPLETING" and j["job"] not in gone]

        def cancel_leg(name, why):
            ids = [j["job"] for j in live_jobs(name)]
            if not ids:
                return False
            result = sh(["scancel", *ids])
            if result.returncode != 0:
                return False
            gone.update(ids)
            entry = legs_state.setdefault(name, {"submitted": [], "failed": []})
            entry["free_legs"] = entry.get("free_legs", 0) + 1
            log(log_path, event="leg_gate_cancel", campaign=name, jobs=ids, why=why, ready_servers=ready_servers)
            jobs.pop(f"gta-{name}", None)
            return True

        working = [c["name"] for c in active
                   if any(j["state"] == "RUNNING" and leg_started_atp(f"gta-{c['name']}", j["job"], unknown=True)
                          for j in live_jobs(c["name"]))]
        work_load = sum(loads[n] for n in working)
        admitted, planned, cancelled, held_back, shed = set(), 0, [], [], []
        for c in active:  # campaign order: waiting legs and campaigns without a leg compete for the room left
            name = c["name"]
            if name in working:
                continue
            fits = ready_servers >= 1 and (work_load + planned + loads[name] <= start_cap or work_load + planned == 0
                                           or loads[name] == 0)
            if fits:
                admitted.add(name)
                planned += loads[name]
            elif live_jobs(name):  # pending, or still waiting for a backend: nothing in flight, cancel it now
                if cancel_leg(name, "no capacity for a leg that has not started"):
                    cancelled.append(name)
            else:
                held_back.append(name)
        # working legs: shed the last ones only after a lasting overload (a short dip is ridden out)
        if now - gate.get("seen", now) > 300:  # "without a break": a gap in the observations restarts both clocks
            gate.pop("overload_since", None)
            gate.pop("hard_since", None)
        gate["seen"] = now
        if not (ready_servers >= 1 and work_load > capacity * config["leg_shed_hard_factor"]):
            gate.pop("hard_since", None)
        elif "hard_since" not in gate:
            gate["hard_since"] = now
        if ready_servers >= 1 and work_load > capacity * config["leg_shed_factor"]:
            gate.setdefault("overload_since", now)
            if (now - gate["overload_since"] >= config["leg_shed_minutes"] * 60
                    or now - gate.get("hard_since", now) >= config["leg_shed_hard_minutes"] * 60):
                keep, keep_load = [], 0
                for name in working:
                    if not keep or keep_load + loads[name] <= start_cap:
                        keep.append(name)
                        keep_load += loads[name]
                for name in reversed(working):
                    if name in keep:
                        continue
                    if cancel_leg(name, f"working load {work_load} on capacity {capacity} (start capacity "
                                        f"{start_cap:g}): overloaded since "
                                        f"{round((now - gate['overload_since']) / 60)} min"):
                        shed.append(name)
                        work_load -= loads[name]
                gate.pop("overload_since", None)
                gate.pop("hard_since", None)
        else:
            gate.pop("overload_since", None)
        if cancelled or shed:
            write_json_atomic(Path(log_path).with_name("keeper-legs.json"), legs_state)
            actions.append({"leg_gate": {"cancelled_waiting": cancelled, "shed": shed}})
        working = [n for n in working if n not in shed]
        gate_info = {"capacity": capacity, "start_capacity": start_cap, "working": working, "working_load": work_load,
                     "admitted": [c["name"] for c in active if c["name"] in admitted], "admitted_load": planned,
                     "held_back": held_back, "cancelled_waiting": cancelled, "shed": shed,
                     "overload_since": gate.get("overload_since"), "hard_overload_since": gate.get("hard_since")}
    if not stopping and not paused:
        claims, pools = None, {}
        for c in active:
            name = c["name"]
            if has_job(name) or (GR / "babysit" / name / "STOP").exists():
                continue
            if admitted is not None and name not in admitted:
                continue  # leg capacity gate: no ready capacity for this campaign yet
            entry = legs_state.setdefault(name, {"submitted": [], "failed": []})
            entry.setdefault("failed", [])
            last = max(entry["submitted"][-1:] + entry["failed"][-1:] or [0])
            if now - last < config["leg_gap_minutes"] * 60:
                continue
            if claims is None:
                if isinstance(rows, str):
                    rows = leg_rows()
                if rows is None:  # squeue failed: place nothing blind, try again next pass
                    leg_wait = "squeue failed"
                    break
                rows = list(rows)
                claims = leg_claims(rows)
            where, partition, node = place_leg(rows, config, claims, pools, leg_extra_hosts, avoid)
            if where == "wait":
                leg_wait = node
                break  # no room for this leg means none for the next ones either
            existing = [int(d.name[1:]) for d in (GR / "babysit" / name).glob("k*") if d.name[1:].isdigit()]
            tag = f"k{max(existing + [len(entry['submitted']) + len(entry['failed'])]) + 1:02d}"
            result = run(["bash", str(GR / "submit.sh"), name, tag], GR_PARTITION=partition, GR_NODELIST=node,
                         GR_CPUS=max(1, int(config["leg_cpus"])), GR_MEM=f"{max(4, int(config['leg_mem_gb']))}G")
            lines = result.stdout.strip().splitlines()
            ok = result.returncode == 0 and bool(lines) and bool(re.search(r" \d+,\d+$", lines[-1]))
            (entry["submitted"] if ok else entry["failed"]).append(now)
            if ok:  # the pinned leg claims its place for the rest of this pass
                claims[node] += 1
                rows.append({"job": "", "name": f"gta-{name}", "state": "PENDING", "partition": partition,
                             "node": "", "pinned": node, "submitted": now})
            action = {"leg": name, "tag": tag, "rc": result.returncode, "out": lines[-1:] or None,
                      "err": result.stderr.strip()[-300:] or None, "partition": partition, "node": node}
            actions.append(action)
            if not ok:
                log(log_path, event="leg_submit_failed", **action)
            write_json_atomic(Path(log_path).with_name("keeper-legs.json"), legs_state)
    finished = len(done) + len(parked_names)
    summary = {"campaigns": len(campaigns), "done": len(done), "released": len(released), "leg_wait": leg_wait,
               "leg_gate": gate_info,
               "active": [{"name": c["name"], "load": loads.get(c["name"]),
                           "driver": [j["state"] for j in jobs.get(f"gta-{c['name']}", [])]} for c in active],
               "parked": [{"name": n, "legs": len((legs_state.get(n) or {}).get("submitted", [])),
                           "failed_submits": len((legs_state.get(n) or {}).get("failed", [])),
                           "stop_file": (GR / "babysit" / n / "STOP").exists()} for n in parked_names],
               "load": sum(loads.get(c["name"], 0) for c in active), "owner_load": dict(owner_load),
               "next": next((c["name"] for c in campaigns if c["name"] not in released), None),
               "breaker": breaker, "recently_parked": recent,
               "all_done": len(done) == len(campaigns),
               "tier_paused": sorted(n for n in tier_paused if n not in done),
               "allowed_tiers": sorted(allowed) if allowed is not None else None,
               "work_left": any(c["name"] not in done and c["name"] not in tier_paused for c in campaigns),
               "all_finished": finished == len(campaigns) and not any(has_job(c["name"]) for c in campaigns)}
    return actions, summary


def successor_marker():
    return GR / "babysit" / "keeper" / f"successor-of-{os.environ.get('SLURM_JOB_ID', 'none')}.json"


def submit_successor(log_path):
    """Chain the next keeper job (afterany this one) through submit_keeper.sh, once per keeper job (marker file)."""
    job = os.environ.get("SLURM_JOB_ID", "")
    result = run(["bash", str(GR / "submit_keeper.sh")], GR_KEEPER_AFTER=job)
    action = {"successor": result.stdout.strip().splitlines()[-1:] or None, "rc": result.returncode,
              "err": result.stderr.strip()[-300:] or None, "after": job}
    log(log_path, event="successor", **action)
    if result.returncode == 0:
        successor_marker().parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(successor_marker(), action)
    return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="glm53-serve-ta-")
    parser.add_argument("--target-file", default=str(GR / "servers-target.txt"))
    parser.add_argument("--stop-file", default=str(GR / "keeper.STOP"))
    parser.add_argument("--status-file", default=str(GR / "keeper-status.json"))
    parser.add_argument("--log", default=str(GR / "keeper-log.jsonl"))
    parser.add_argument("--legs-file", default=str(GR / "keeper-legs.json"))
    parser.add_argument("--config-file", default=str(GR / "keeper-config.json"))
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--lead-minutes", type=float, default=30)
    parser.add_argument("--max-per-hour", type=int, default=12)
    parser.add_argument("--fast-fail-minutes", type=float, default=15)
    parser.add_argument("--max-fast-fails", type=int, default=4)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    campaigns = json.loads((GR / "tasks.json").read_text())["campaigns"]
    if (GR / "GO").exists():
        raise SystemExit(f"{GR / 'GO'} exists: campaigns are released one by one (GO-<campaign>); remove it")
    submissions = []          # server submissions (timestamps)
    seen_running = {}         # slot name -> start time (now - elapsed at first sighting)
    finished = set()
    fast_fails = 0
    halted = None
    successor = json.loads(successor_marker().read_text()) if successor_marker().exists() else None
    last_good = {}
    last_overrides = {}
    legs_path = Path(args.legs_file)
    legs_state = json.loads(legs_path.read_text()) if legs_path.exists() else {}
    my_job = os.environ.get("SLURM_JOB_ID")
    end_time = float(os.environ.get("SLURM_JOB_END_TIME") or 0)
    outage_since = None       # first pass with 0 ready servers (None while some server is ready)
    outage_cancelled = set()  # driver legs cancelled for the current outage
    tier_cancelled = set()    # legs cancelled by the tier pause
    idle_since, idle_released = None, False
    leg_wait_since = None     # first pass in which a leg found no host node (None while every leg has a place)
    leg_gate_state = {}       # leg capacity gate memory (overload_since)
    avoid_until = {}          # node -> time until which no leg is pinned there (it never started a pinned leg)
    while True:
        now = time.time()
        try:
            main_target = int(Path(args.target_file).read_text().strip())
        except (OSError, ValueError):
            main_target = 0
        try:  # live overrides (no restart needed); a half-written / broken file keeps the last good overrides
            overrides = json.loads(Path(args.config_file).read_text())
            if not isinstance(overrides, dict):
                raise ValueError("not an object")
            last_overrides = overrides
        except (OSError, ValueError) as error:
            log(args.log, event="config_unreadable", error=str(error)[:200])
            overrides = last_overrides
        for key in ("lead_minutes", "max_per_hour", "max_fast_fails"):
            if isinstance(overrides.get(key), (int, float)) and not isinstance(overrides.get(key), bool):
                setattr(args, key, overrides[key])
        config = checked_config(overrides, args.log, last_good)
        targets = {"main": main_target, "short": int(config["short_target"])}
        leads = {"main": args.lead_minutes, "short": float(config["short_lead_minutes"])}
        try:
            jobs = my_jobs()
            if my_job and my_job not in {j["job"] for js in jobs.values() for j in js}:
                raise RuntimeError(f"squeue does not list this keeper job {my_job}")
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            log(args.log, event="squeue_error", error=str(error)[:300])
            if args.once:
                return 1
            time.sleep(args.interval)
            continue
        stuck = cancel_stuck_held(jobs, now, args.log)
        rows = leg_rows()
        if rows is not None and my_job and my_job not in {r["job"] for r in rows}:
            rows = None  # an answer without this keeper job is not a real listing (as for my_jobs above)
        stale_pins = cancel_stale_pins(rows, legs_state, now, config, args.log) if rows else []
        for _, nodes, ids in stale_pins:
            for node in nodes:
                avoid_until[node] = now + STALE_NODE_AVOID_S
            leg_gate_state.setdefault("cancelled_jobs", set()).update(ids)  # the gate must not cancel them again
        avoid_until = {node: until for node, until in avoid_until.items() if until > now}
        leg_extra_hosts = (int((now - leg_wait_since) // (max(5, config["leg_wait_minutes"]) * 60))
                           if leg_wait_since is not None else 0)
        current = {name: js for name, js in jobs.items() if name.startswith(args.prefix)}
        stop_reason = "stop file" if Path(args.stop_file).exists() else None
        pools = {name: {"live": [], "pending": [], "running": []} for name in POOLS}
        ready_servers = 0
        for name, slot_jobs in sorted(current.items()):
            pool = next((p for p, spec in POOLS.items() if re.fullmatch(spec["regex"], name)), None)
            states = {j["state"] for j in slot_jobs}
            run_jobs = [j for j in slot_jobs if j["state"] == "RUNNING"]
            ready_servers += any(ready(j["job"]) for j in run_jobs)
            if run_jobs:
                elapsed = max((parse_left(j["elapsed"]) or 0) for j in run_jobs)
                seen_running.setdefault(name, now - elapsed)
            if pool is None:
                continue
            info = pools[pool]
            if run_jobs:
                lefts = [j["left"] for j in run_jobs if j["left"] is not None]
                left = min(lefts) if lefts else None
                info["running"].append((name, left))
                if left is None or left > leads[pool] * 60:
                    info["live"].append(name)
            elif states & {"PENDING", "CONFIGURING"}:
                info["pending"].append(name)
                info["live"].append(name)
        for name in list(seen_running):  # slots that ran and are gone now
            if name not in current and name not in finished:
                finished.add(name)
                ran = now - seen_running[name]
                if ran < args.fast_fail_minutes * 60:
                    fast_fails += 1
                    log(args.log, event="fast_fail", slot=name, ran_s=round(ran))
                else:
                    fast_fails = 0
                    log(args.log, event="slot_ended", slot=name, ran_s=round(ran))
        if fast_fails >= args.max_fast_fails and not halted:
            halted = f"{fast_fails} slots in a row ended within {args.fast_fail_minutes} min"
            log(args.log, event="halted", reason=halted)
        # GPU outage (no ready server for outage_minutes, e.g. every 4 h batch slot expired with the replacements still
        # queued: 2026-11-09 12:28 -> ~16:50): in-flight rollouts would only burn ATP's 3 h agent timeout and be graded
        # as fails. So: no new legs, and the driver legs that were running when the outage began are cancelled (their
        # in-flight attempts become retryable: the next leg's ATP recovery retries them); new legs start, and resume
        # the campaigns, once a server is ready again.
        if ready_servers >= 1:
            if outage_since is not None:
                log(args.log, event="outage_end", minutes=round((now - outage_since) / 60),
                    cancelled=sorted(outage_cancelled))
            outage_since, outage_cancelled = None, set()
        elif outage_since is None:
            outage_since = now
            log(args.log, event="outage_begin")
        paused = outage_since is not None and now - outage_since >= config["outage_minutes"] * 60
        if paused:
            # only legs whose driver got past its backend wait (ATP staged / campaign running): a leg still waiting
            # for a backend holds no sandbox and simply starts when a server is back
            victims = [j["job"] for name, entries in jobs.items() if name.startswith(("gta-t", "gta-dip"))
                       for j in entries if j["state"] == "RUNNING" and j["job"] not in outage_cancelled
                       and leg_started_atp(name, j["job"])]
            if victims:
                sh(["scancel", *victims])
                outage_cancelled.update(victims)
                log(args.log, event="outage_pause", cancelled=victims,
                    minutes=round((now - outage_since) / 60))
        # Tier pause (user 2026-11-09: "after tier 1 finishes, pause the other rollouts until I say continue"):
        # with pause_after_tier = N, once every campaign of tier <= N has its results, TIER-PAUSE.json is written
        # (allowed_tiers 1..N); while it exists only those tiers run and the RUNNING legs of other tiers are cancelled
        # once (their in-flight attempts become retryable). Delete the file (or set allowed_tiers) to continue.
        allowed = None
        if tier_pause_path().exists():
            try:
                allowed = set(json.loads(tier_pause_path().read_text())["allowed_tiers"])
            except (OSError, ValueError, KeyError, TypeError):
                allowed = {1}
        elif config["pause_after_tier"] and all(campaign_done(c["name"]) for c in campaigns
                                                 if c["tier"] <= config["pause_after_tier"]):
            allowed = set(range(1, int(config["pause_after_tier"]) + 1))
            write_json_atomic(tier_pause_path(), {"paused_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                           "allowed_tiers": sorted(allowed),
                                           "reason": f"all tier <= {config['pause_after_tier']} campaigns done; user "
                                                     f"2026-11-09: pause the other tiers until told to continue"})
            log(args.log, event="tier_pause", allowed_tiers=sorted(allowed))
        if allowed is not None:
            tiers = {c["name"]: c["tier"] for c in campaigns}
            victims = [j["job"] for name, entries in jobs.items() if name.startswith("gta-t")
                       and tiers.get(name[4:], 0) not in allowed for j in entries
                       if j["state"] in ("RUNNING", "PENDING", "CONFIGURING") and j["job"] not in tier_cancelled]
            if victims:
                sh(["scancel", *victims])
                tier_cancelled.update(victims)
                log(args.log, event="tier_pause_cancel", jobs=victims, allowed_tiers=sorted(allowed))
        actions, campaign_summary = schedule_campaigns(campaigns, jobs, ready_servers, config, legs_state, now,
                                                       args.log, bool(stop_reason), paused=paused, allowed=allowed,
                                                       rows=rows, leg_extra_hosts=leg_extra_hosts,
                                                       avoid=set(avoid_until), gate=leg_gate_state)
        if campaign_summary["leg_wait"]:
            if leg_wait_since is None:
                leg_wait_since = now
                log(args.log, event="leg_wait_begin", why=campaign_summary["leg_wait"])
        elif leg_wait_since is not None:
            log(args.log, event="leg_wait_end", minutes=round((now - leg_wait_since) / 60),
                extra_hosts=leg_extra_hosts)
            leg_wait_since = None
        # Idle GPUs: no campaign left to run (all done or tier-paused) and no gta-dip* re-run job -> after
        # idle_release_minutes cancel every server slot and submit none until there is work again.
        busy = campaign_summary["work_left"] or any(name.startswith("gta-dip") for name in jobs)
        if busy:
            idle_since = None
        elif idle_since is None:
            idle_since = now
            log(args.log, event="idle_begin")
        idle = idle_since is not None and now - idle_since >= config["idle_release_minutes"] * 60
        if idle and not idle_released:
            ids = [j["job"] for name, entries in current.items() for j in entries]
            if ids:
                sh(["scancel", *ids])
            idle_released = True
            log(args.log, event="idle_release", jobs=ids)
            actions.append({"idle_release": ids})
        if busy:
            idle_released = False
        if paused:
            actions.append({"outage_paused_min": round((now - outage_since) / 60)})
        if campaign_summary["all_finished"] and not stop_reason:
            stop_reason = (f"all {len(campaigns)} campaigns done" if campaign_summary["all_done"] else
                           f"all campaigns done or parked ({len(campaign_summary['parked'])} parked)")
        if (not stop_reason and not args.once and (successor is None or successor.get("rc") != 0) and end_time
                and end_time - now < config["successor_minutes"] * 60
                and (successor is None or now - successor.get("t", 0) > 600)):
            successor = dict(submit_successor(args.log), t=now)  # a failed submission is retried every 10 min
            actions.append(successor)
        submissions = [t for t in submissions if now - t < 3600]
        if stuck:
            actions.append({"stuck_held_cancelled": stuck})
        if stale_pins:
            actions.append({"stale_pin_cancelled": [name for name, _, _ in stale_pins]})
        server_slots = sum(1 for name in current if any(re.fullmatch(s["regex"], name) for s in POOLS.values()))
        for pool, spec in POOLS.items():
            info = pools[pool]
            if stop_reason:
                for name in info["pending"]:
                    ids = [j["job"] for j in current[name] if j["state"] != "RUNNING"]
                    sh(["scancel", *ids])
                    actions.append({"cancel_pending": name, "jobs": ids})
                if campaign_summary["all_finished"] and config["cancel_running_when_done"]:
                    for name, _ in info["running"]:
                        ids = [j["job"] for j in current[name]]
                        sh(["scancel", *ids])
                        actions.append({"cancel_running": name, "jobs": ids})
                continue
            if halted or idle:
                continue
            missing = targets[pool] - len(info["live"])
            if missing > 0:
                used = set()
                for n in list(current) + list(seen_running) + list(finished):
                    m = re.fullmatch(spec["regex"], n)
                    if m:
                        used.add(int(m.group(1)))
                for d in (G / "runs" / "serve").glob("ta-*"):
                    m = re.fullmatch(spec["regex"], "glm53-serve-" + d.name)
                    if m:
                        used.add(int(m.group(1)))
                next_id = max(used | {0}) + 1
                for _ in range(missing):
                    if len(submissions) >= args.max_per_hour:
                        actions.append({"rate_limited": True, "pool": pool})
                        break
                    if server_slots >= config["max_servers"]:  # owner caps use at most ~12 servers' worth of load
                        actions.append({"max_servers": config["max_servers"], "pool": pool})
                        break
                    server_slots += 1
                    slot = spec["slot"].format(next_id)
                    extra = {}
                    if overrides.get("venv"):  # pin the servers' vLLM env (else the sbatch default)
                        extra["VENV"] = str(overrides["venv"])
                    if spec["partition"]:
                        extra["PARTITION"] = spec["partition"]
                    result = run(["bash", str(SUBMIT), slot, *spec["extra"]], **extra)
                    submissions.append(now)
                    actions.append({"submitted": slot, "pool": pool, "rc": result.returncode,
                                    "out": result.stdout.strip().splitlines()[-1:] if result.stdout else None,
                                    "err": result.stderr.strip()[-300:] or None})
                    next_id += 1
            elif missing < 0 and config["scale_down"]:  # opt-in: never cancel queued servers unless asked
                newest = sorted(info["pending"], key=lambda n: int(re.fullmatch(spec["regex"], n).group(1)),
                                reverse=True)
                for name in newest[:-missing]:
                    ids = [j["job"] for j in current[name] if j["state"] != "RUNNING"]
                    sh(["scancel", *ids])
                    actions.append({"cancel_pending": name, "jobs": ids})
        status = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": targets, "leads_min": leads,
                  "pools": {p: {"live": i["live"], "pending": i["pending"],
                                "running": [{"slot": n, "left_min": round(l / 60) if l is not None else None}
                                            for n, l in i["running"]]} for p, i in pools.items()},
                  "running": [{"slot": n, "left_min": round(l / 60) if l is not None else None}
                              for i in pools.values() for n, l in i["running"]],
                  "pending": [n for i in pools.values() for n in i["pending"]],
                  "target": sum(targets.values()), "fast_fails_in_a_row": fast_fails, "halted": halted,
                  "stop_reason": stop_reason, "submissions_last_hour": len(submissions), "actions": actions,
                  "max_per_hour": args.max_per_hour, "ready_servers": ready_servers, "config": config,
                  "campaigns": campaign_summary, "successor": successor, "job": my_job,
                  "leg_hosts": {partition: dict(leg_hosts(rows or [], partition)) for partition, _ in LEG_POOLS},
                  "leg_wait_since": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(leg_wait_since))
                  if leg_wait_since else None,
                  "leg_extra_hosts": leg_extra_hosts, "avoid_nodes": sorted(avoid_until),
                  "outage_since": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(outage_since))
                  if outage_since else None,
                  "idle_since": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(idle_since)) if idle_since else None,
                  "idle_released": idle_released,
                  "job_left_min": round((end_time - now) / 60) if end_time else None}
        write_json_atomic(Path(args.status_file), status)
        if actions:
            log(args.log, event="actions", actions=actions, targets=targets)
        if stop_reason:
            log(args.log, event="stopped", reason=stop_reason)
            return 0
        if args.once:
            print(json.dumps(status, indent=1))
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    while True:
        try:
            sys.exit(main())
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:  # never let one bad pass end the keeper job: log and start over
            log(str(GR / "keeper-log.jsonl"), event="keeper_exception", error=f"{type(error).__name__}: {error}"[:500])
            if "--once" in sys.argv:
                raise
            time.sleep(60)
