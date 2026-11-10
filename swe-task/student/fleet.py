#!/usr/bin/env python3
"""Maintainer of the persistent student vLLM workers (stdlib only; Python 3.9+, runs with /usr/bin/python3).

Workers are 1-node 4h batch jobs named <job_prefix><NN> (worker.sbatch -> serve_student.sh); each publishes
registry/<job_id>.json for the routers (client.py). A batch job ends after 4h, so persistence = resubmission.
Fleet workers are exactly <job_prefix> + 2 digits of `user`; other jobs that merely share the prefix are not ours.
Slot NN holds one job, or with several worker accounts (below) up to one candidate job per account.
One reconcile pass:
  1. finalizes fleet jobs that left the queue (sacct): idle-reaper cancels are incidents (not worker failures),
     worker failures exclude the node, the registry record moves to registry/archive/;
  2. settles candidate pairs (below), then polices running fleet jobs: a failed/unhealthy/stopped worker that is still
     running, a startup without a ready record after stuck_startup_min, a stale heartbeat -> scancel (failures also
     exclude the node). A requeued job is judged by its current run only: registry records written before its
     current start are ignored. A pass that cannot list registry/ acts on nothing (RegistryError; the daemon skips
     it): an empty listing would make every running worker a stuck startup;
  3. tops coverage up to `target`: coverage = RUNNING fleet jobs in coverage_partitions with more than the horizon
     left + PENDING (not held) jobs there, the candidates of a pair once. horizon = horizon_min, raised
     (adaptive_horizon) to the expected queue wait + startup_s + horizon_margin_min, at most horizon_max_min; expected
     wait = the wait_quantile of the first-run submit->start waits of the last wait_window_h (state/waits.json, >= 3
     needed; a pair's wait counts from its first candidate's submission), raised to the median scheduler start
     estimate of the pending jobs (a pair's earliest candidate). A new worker takes the lowest free 2-digit slot; at
     most max_jobs nodes held or wanted by fleet jobs (a pending pair counts once) and max_submits_per_pass slots that
     get a candidate per pass, new slots and pair completions alike (one sbatch each with two worker accounts);
  4. writes state/status.json; every action goes to state/events.jsonl (incidents flagged).
Worker accounts (fleet.json worker_accounts, e.g. [nemotron_agents_dev, nemotron_rl_algo]; [] = [account], which is
the single-account fleet of before 2026-11-02). A new slot gets one job, on the first usable account (the next one
when that sbatch fails). A slot still pending (not held) pair_delay_min (15) after its first submission gets a
candidate on each account it lacks (pair completion): same name, exemption comment and exclusions; the first
candidate to start is the slot's worker. So a fast queue never starts two candidates, a slow one is hedged after
15 min, and a job queued before worker_accounts was set keeps its queue position and gains a sibling. Never more
candidates than accounts; completion pauses with target 0, the STOP flag or a non-closed breaker (the half-open probe
is one job). hengyuf's dual-account helper (schdule_helper.sh) cannot submit workers: it appends a tag to --comment,
which voids the idle-reaper exemption, and its loser exits by itself, which fleet.py would count as a worker failure
(no registry record). So the daemon settles pairs itself, by explicit job id. A pair = the jobs of one slot, all on
the fleet partition, with pairwise different squeue accounts (also a pair left over after a rollback to one account);
any other duplicate (same account or another partition: made by hand) keeps the single-account rules, in coverage
too. Once a candidate serves, its pending siblings are cancelled (scancel --state=PENDING, which leaves a job that
started meanwhile for the next pass); while the first one is still starting they stay queued as its backup (a failed
startup then costs no queue position). Of two started candidates one that is still starting goes (the later start, or
the one not serving when the other serves); one that reported ready/draining is never cancelled here (two serving
ones, only after a daemon outage, both stay), and one that police() is cancelling never keeps the slot. A candidate
cancelled here (sibling_cancel in state/jobs.json) is never a worker failure, exclusion or breaker entry, whatever its
registry says; one that is no longer a loser (e.g. its keeper failed) loses that mark and is an ordinary worker again.
An account whose sbatch fails is skipped for the rest of the pass and, while another account works, for
ACCOUNT_RETRY_S (breaker.json account_backoff; an incident at most every 30 min); completion adds its candidate later.
A new node exclusion is also applied to the pending fleet workers (scontrol update ExcNodeList); fleet.json is only
rewritten when it parses. Worker failures feed a circuit breaker: failures on breaker_failures distinct nodes within
breaker_window_min (one bad node is excluded instead) stop submissions and node exclusions. After breaker_cooldown_min
it is half-open: one probe worker is submitted, and the breaker closes once that worker serves (a failed probe doubles
the cooldown, at most 4 h); `fleet.py reset-breaker` closes it by hand. A systematic fault must not burn through the
cluster's nodes. Only jobs of `user` named <job_prefix><NN> or `daemon_job_name` are ever cancelled, always by
explicit job id (scancel is never run without one: `scancel --user` alone would cancel every job of the user).
`start-daemon --successor` (fleetd.sbatch) ignores its own job and daemon jobs that are no longer pending or running
(e.g. COMPLETING). A daemon exits with status 3 when state/RELOAD appears after its start (it removes the flag): its
batch loop (fleetd_babysit.sbatch, fleetd.sbatch) restarts it with the fleet.py on disk.
Review fixes 2026-11-02 (second round, tagged in the code): F1 settle() drops the sibling marks of a candidate that
is no longer a loser; F2 a job police() is cancelling never keeps its slot; F3 only account pairs settle or count
once; F5 a new slot's sibling only after pair_delay_min; F6 scancel never without job ids; F7 a settled candidate is
never a worker failure (ended(), and police() while it is still listed); F10 a registry/ that cannot be listed
aborts the pass instead of reading as empty.

usage: fleet.py status [--json] | reconcile [--dry-run] | daemon [--interval 60] [--until EPOCH] [--wait-lock]
       | submit --slot NN [--partition P] [--time T] [--account A] [--dry-run | --test-only] | stop [--yes]
       | start-daemon [--successor] | unexclude NODE... | reset-breaker | worker-accounts [ACCOUNT... | --clear]
"""
import sys

sys.dont_write_bytecode = True

import argparse
from collections import Counter
from contextlib import contextmanager
import errno
import fcntl
import json
import os
from pathlib import Path
import pwd
import shlex
import signal
import socket
import subprocess
import threading
import time
import traceback

ROOT = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/student")
SLURM_BIN = "/cm/shared/apps/slurm/current/bin"
# Idle-reaper exemption (docs/port-contract.md), passed as ONE argv element: --comment=<compact JSON>.
COMMENT = {"OccupiedIdleGPUsJobReaper": {
    "exemptIdleTimeMins": "240", "reason": "inference_server",
    "description": "Persistent 1-node 8-GPU vLLM inference worker for the swe-task synthetic-task student (4h job, "
                   "continuously resubmitted); GPUs can idle between on-demand rollout bursts"}}
DEFAULTS = {
    "target": 10,
    "horizon_min": 90,
    "max_jobs": 24,
    "max_submits_per_pass": 10,
    "account": "nemotron_agents_dev",
    "worker_accounts": [],  # one candidate per account for every slot, e.g. [agents_dev, rl_algo]; [] = [account]
    "pair_delay_min": 15,  # F5: a new slot's job queues alone this long before its siblings on the other accounts
    "partition": "batch",
    "time": "04:00:00",
    "job_prefix": "swe-task-student-w",
    "daemon_job_name": "swe-task-student-fleetd",
    "daemon_partition": "cpu_datamover",
    "daemon_time": "7-00:00:00",
    "user": "hengyuf",
    "exclude_nodes": ["pool0-00058", "pool0-00484", "pool0-00502", "pool0-01460", "pool0-00613"],
    "comment": COMMENT,
    "stuck_startup_min": 25,
    "terminal_grace_min": 5,
    "heartbeat_stale_min": 10,
    "exclude_on_unhealthy": True,
    "breaker_failures": 4,
    "breaker_window_min": 120,
    "breaker_cooldown_min": 60,
    "coverage_partitions": ["batch"],
    "adaptive_horizon": True,
    "horizon_max_min": 200,
    "horizon_margin_min": 20,
    "startup_s": 600,
    "wait_quantile": 0.9,
    "wait_window_h": 12,
    "stagger_min": 40,
}
RUNNING = ("RUNNING", "CONFIGURING")
PENDING = ("PENDING", "REQUEUED")
TERMINAL = ("BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED",
            "TIMEOUT")
SERVING = ("ready", "draining")
HEARTBEAT_TIMEOUT_S = 120  # contract: a worker is routable only with a heartbeat this fresh
REAPER = "occupied idle job reaper"  # sacct AdminComment of the cluster's idle-GPU reaper
UNCONFIRMED_GRACE_S = 1800  # a known job gone from squeue that sacct does not show ended is finalized after this
CANCEL_RETRY_S = 600  # a cancelled job still listed as running is cancelled again after this
PASS_TIMEOUT_S = 900  # daemon watchdog: a pass wedged this long (hung Lustre/Slurm call) exits the process
NOTE_EVERY_S = 1800  # repeated conditions (paused submissions, mass-stale heartbeats) are logged at most this often
RELOAD_EXIT = 3  # daemon exit status after state/RELOAD: the batch loop restarts it (any status but 0)
ACCOUNT_RETRY_S = 600  # a worker account whose sbatch failed is not tried again for this long while another works
SQUEUE_KEYS = ("job_id", "user", "state", "partition", "node", "elapsed", "time_left", "start", "end", "reason",
               "account", "name")
SQUEUE_FORMAT = "%i|%u|%T|%P|%N|%M|%L|%S|%e|%r|%a|%j"  # free-text job name last (split with maxsplit)
SACCT_KEYS = ("job_id", "user", "state", "exit_code", "start", "end", "elapsed", "node", "admin_comment")
SACCT_FORMAT = "JobIDRaw,User,State,ExitCode,Start,End,Elapsed,NodeList,AdminComment"


class SlurmError(RuntimeError):
    """A Slurm command failed or printed something unexpected; the pass then acts on nothing."""


class ConfigError(RuntimeError):
    """fleet.json is unreadable or invalid."""


class LockBusy(RuntimeError):
    """Another process holds the lock."""


class RegistryError(RuntimeError):
    """registry/ cannot be listed (F10): the pass then acts on nothing, like after a failed squeue."""


# ---- small helpers ---------------------------------------------------------------------------------------------------

def iso(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(ts)) if ts else None


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, value):
    """Atomic replace: temp file in the same directory (dot name, never *.json), then os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def append_jsonl(path, record):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def epoch(text):
    """Unix seconds as printed with SLURM_TIME_FORMAT=%s; None for N/A, Unknown, None."""
    text = (text or "").strip()
    return int(text) if text.isdigit() else None


def duration(text):
    """Seconds of a Slurm duration ([D-]HH:MM:SS, MM:SS); None for UNLIMITED, NOT_SET, INVALID."""
    days, _, clock = (text or "").strip().rpartition("-")
    parts = clock.split(":")
    if not 1 <= len(parts) <= 3 or not all(part.isdigit() for part in parts) or (days and not days.isdigit()):
        return None
    hours, minutes, seconds = [0] * (3 - len(parts)) + [int(part) for part in parts]
    return int(days or 0) * 86400 + hours * 3600 + minutes * 60 + seconds


def hms(seconds):
    if seconds is None:
        return "-"
    seconds = int(seconds)
    sign, seconds = ("-", -seconds) if seconds < 0 else ("", seconds)
    return "%s%d:%02d:%02d" % (sign, seconds // 3600, seconds // 60 % 60, seconds % 60)


def single_node(node):
    return bool(node) and not any(char in node for char in "[], ")


def comment_json(cfg):
    return json.dumps(cfg["comment"], separators=(",", ":"))


def worker_accounts(cfg):
    """The accounts a new slot gets one candidate on, in submission order: worker_accounts, else [account]."""
    return list(cfg.get("worker_accounts") or []) or [cfg["account"]]


def slot_of(cfg, name):
    """'03' for '<job_prefix>03', else None."""
    slot = name[len(cfg["job_prefix"]):] if name.startswith(cfg["job_prefix"]) else ""
    return slot if len(slot) == 2 and slot.isdigit() else None


def slot_key(cfg, job):
    """A job's slot; a job without one (not a fleet worker) is a group of its own."""
    return slot_of(cfg, job.get("name") or "") or "job " + str(job["job_id"])


def by_slot(cfg, jobs):
    """{slot: [jobs]}: the jobs of each slot (one job per slot with one worker account)."""
    slots = {}
    for job in jobs:
        slots.setdefault(slot_key(cfg, job), []).append(job)
    return slots


def candidate_pair(cfg, jobs):
    """The jobs of one slot race as candidates of one worker (F3): two or more jobs, all on the fleet partition, with
    known, pairwise different squeue accounts (fleet.py's pairs, also one left over after a rollback to one account).
    Any other duplicate of a slot name (same account or another partition: made by hand; or a record of the fleet.py
    before worker accounts, which kept no account) is not a pair: its jobs keep the single-account rules (each one
    counts, nothing settles), exactly as in the fleet.py before worker accounts."""
    accounts = [job.get("account") for job in jobs]
    return len(jobs) > 1 and all(job.get("partition") == cfg["partition"] for job in jobs) \
        and all(accounts) and len(set(accounts)) == len(jobs)


def pair_slots(cfg, jobs):
    """The slots whose jobs (among `jobs`) are a candidate pair."""
    return {slot for slot, group in by_slot(cfg, jobs).items() if candidate_pair(cfg, group)}


def race_key(cfg, job, pairs):
    """The group a job counts in (coverage, nodes, waits): its slot when the slot is a candidate pair (pairs =
    pair_slots of the same jobs), else the job alone (one job, one worker: the single-account rule)."""
    slot = slot_key(cfg, job)
    return slot if slot in pairs else "job " + str(job["job_id"])


def viable(job, registry):
    """A RUNNING job whose worker has not given up (registry status not failed/unhealthy/stopped)."""
    return job["state"] in RUNNING and (registry.get(job["job_id"]) or {}).get("status") not in (
        "failed", "unhealthy", "stopped")


def started_slots(cfg, jobs, registry, pairs=None):
    """Race groups (race_key) with a viable RUNNING job: the pending candidates of such a pair are backups that will
    be cancelled, so they neither cover nor take a node. A pair whose running candidate gave up keeps counting its
    pending ones."""
    pairs = pair_slots(cfg, jobs) if pairs is None else pairs
    return {race_key(cfg, job, pairs) for job in jobs if viable(job, registry)}


def police_cancelled(record):
    """A cancel request of police() (stuck, stale or failed worker) is under way: cancel_requested_at is set and is
    not the time of a settlement request (sibling_cancel.at). Such a job never keeps its slot in settle() (F2):
    keeping it would cancel its healthy sibling, and the slot would lose both."""
    asked = record.get("cancel_requested_at")
    return asked is not None and asked != (record.get("sibling_cancel") or {}).get("at")


def is_worker(cfg, job):
    """Exactly <job_prefix><2 digits> of cfg user: other jobs of the user that merely share the prefix are not ours."""
    return job["user"] == cfg["user"] and slot_of(cfg, job["name"]) is not None


def is_fleet_job(cfg, job):
    """The only jobs this tool may cancel: the user's workers and daemon jobs."""
    return is_worker(cfg, job) or (job["user"] == cfg["user"] and job["name"] == cfg["daemon_job_name"])


def held(job):
    reason = job.get("reason", "")
    return reason.startswith("JobHeld") or "held" in reason.lower()


def remaining(job, now):
    """Seconds left of a job (its end time, else squeue's time left); None when unknown."""
    if job.get("end"):
        return job["end"] - now
    return duration(job.get("time_left"))


def wait_estimate(cfg, waits, jobs, now):
    """Expected queue wait (s) of a worker submitted now, or None without data: the wait_quantile of the recent
    submit->start waits of fleet workers in the coverage partitions, raised to the scheduler's median expected start
    (squeue %S) of the workers pending there (a new job queues behind them). A candidate pair races, so its earliest
    estimate counts; backups of a pair that already runs are left out (race_key: any other job counts alone)."""
    counted = cfg.get("coverage_partitions") or None
    recent = sorted(item["wait_s"] for item in waits
                    if now - item["start"] <= cfg["wait_window_h"] * 3600
                    and (counted is None or item.get("partition") in counted))
    estimates = []
    if len(recent) >= 3:
        estimates.append(recent[min(len(recent) - 1, int(cfg["wait_quantile"] * len(recent)))])
    pairs = pair_slots(cfg, jobs)
    running = {race_key(cfg, job, pairs) for job in jobs if job["state"] in RUNNING}
    earliest = {}
    for job in jobs:
        key = race_key(cfg, job, pairs)
        if job["state"] in PENDING and not held(job) and isinstance(job.get("start"), (int, float)) \
                and job["start"] > now and (counted is None or job.get("partition") in counted) \
                and key not in running:
            earliest[key] = min(earliest.get(key, job["start"]), job["start"])
    pending = sorted(start - now for start in earliest.values())
    if pending:
        estimates.append(pending[len(pending) // 2])
    return max(estimates) if estimates else None


def horizon_cap_s(cfg):
    """The largest usable horizon: horizon_max_min, but never so long that horizon + the largest slot stagger reaches
    the worker walltime (10 min margin). Otherwise no running worker ever counts as coverage and the daemon tops up
    to max_jobs (2026-10-17: horizon_max_min 270 with 4 h workers held 24 nodes for a target of 10)."""
    walltime = duration(cfg["time"])
    if not walltime:
        return cfg["horizon_max_min"] * 60
    return min(cfg["horizon_max_min"] * 60, walltime - cfg.get("stagger_min", 0) * 60 - 600)


def horizon_s(cfg, waits, jobs, now):
    """Time left (s) below which a running worker stops counting as coverage: horizon_min, raised to the expected
    wait + startup + margin (adaptive_horizon), at most horizon_cap_s."""
    base = cfg["horizon_min"] * 60
    wait = wait_estimate(cfg, waits, jobs, now) if cfg["adaptive_horizon"] else None
    if wait is None:
        return base
    lead = wait + cfg["startup_s"] + cfg["horizon_margin_min"] * 60
    return int(min(max(base, lead), max(base, horizon_cap_s(cfg))))


def current_registry(registry, jobs):
    """Registry minus records written by an earlier run of a requeued job (started_at before the job's current
    start): until the new run reports, the job is starting, whatever the old record says."""
    starts = {job["job_id"]: job["start"] for job in jobs if job["state"] in RUNNING and job.get("start")}
    return {job_id: record for job_id, record in registry.items()
            if not (job_id in starts and isinstance(record.get("started_at"), (int, float))
                    and record["started_at"] < starts[job_id] - 60)}


def stagger_s(cfg, job):
    """Per-slot offset (s) added to the horizon: slot classes 0-9 (slot mod 10) spread a wave's replacements over
    stagger_min, so waves drift out of phase instead of expiring together (a sudden slow queue then hits only the
    few workers in handover, not the whole fleet)."""
    slot = slot_of(cfg, job.get("name") or "")
    if slot is None or not cfg.get("stagger_min"):
        return 0
    return int(cfg["stagger_min"] * 60 * (int(slot) % 10) / 9)


def coverage(cfg, jobs, registry, now, cancelled=(), horizon=None):
    """RUNNING workers with more than the horizon (+ their slot's stagger) left (and no terminal registry status)
    + PENDING (not held) workers: a candidate pair once (its candidates race for one node), not at all while a
    candidate of it runs viably (started_slots). Every other job counts alone (race_key), so with one worker account
    (and also with duplicates made by hand) this counts jobs, as before worker accounts."""
    horizon = cfg["horizon_min"] * 60 if horizon is None else horizon
    count = 0
    counted = cfg.get("coverage_partitions") or None  # e.g. ["batch"]: a manual interactive worker is extra capacity
    jobs = [job for job in jobs if job["job_id"] not in cancelled
            and not (counted and (job.get("partition") or cfg["partition"]) not in counted)]
    pairs = pair_slots(cfg, jobs)
    started, pending = started_slots(cfg, jobs, registry, pairs), set()
    for job in jobs:
        if job["state"] in PENDING:
            slot = race_key(cfg, job, pairs)
            if not held(job) and slot not in started and slot not in pending:
                pending.add(slot)
                count += 1
        elif job["state"] in RUNNING:
            status = (registry.get(job["job_id"]) or {}).get("status")
            left = remaining(job, now)
            count += status not in ("failed", "unhealthy", "stopped") and left is not None and \
                left > horizon + stagger_s(cfg, job)
    return count


def node_count(cfg, jobs, registry):
    """Nodes that fleet jobs hold or will hold (max_jobs): every job not pending + one per candidate pair with pending
    candidates and no viable running one + every other pending job. With one worker account: the number of jobs."""
    pairs = pair_slots(cfg, jobs)
    started = started_slots(cfg, jobs, registry, pairs)
    pending = {race_key(cfg, job, pairs) for job in jobs if job["state"] in PENDING} - started
    return len([job for job in jobs if job["state"] not in PENDING]) + len(pending)


def check_config(cfg):
    for key in ("target", "horizon_min", "max_jobs", "max_submits_per_pass", "stuck_startup_min", "terminal_grace_min",
                "heartbeat_stale_min", "breaker_failures", "breaker_window_min", "breaker_cooldown_min",
                "horizon_max_min", "horizon_margin_min", "startup_s", "wait_window_h", "stagger_min", "pair_delay_min"):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] < 0:
            raise ConfigError("fleet.json: %s must be a non-negative integer" % key)
    for key in ("account", "partition", "time", "job_prefix", "daemon_job_name", "daemon_partition", "daemon_time",
                "user"):
        if not isinstance(cfg[key], str) or not cfg[key] or any(char.isspace() for char in cfg[key]):
            raise ConfigError("fleet.json: %s must be a non-empty word" % key)
    partitions = cfg.get("coverage_partitions", [])
    if not isinstance(partitions, list) or not all(isinstance(name, str) and name for name in partitions):
        raise ConfigError("fleet.json: coverage_partitions must be a list of partition names ([] = all)")
    accounts = cfg.get("worker_accounts", [])
    if not isinstance(accounts, list) or len(set(map(str, accounts))) != len(accounts) or not all(
            isinstance(name, str) and name and not any(char.isspace() or char in ",|" for char in name)
            for name in accounts):
        raise ConfigError("fleet.json: worker_accounts must be a list of distinct account names ([] = account)")
    if not isinstance(cfg["wait_quantile"], (int, float)) or not 0 <= cfg["wait_quantile"] <= 1:
        raise ConfigError("fleet.json: wait_quantile must be within [0, 1]")
    if cfg["daemon_job_name"].startswith(cfg["job_prefix"]):
        raise ConfigError("fleet.json: daemon_job_name must not start with job_prefix")
    if not isinstance(cfg["exclude_nodes"], list) or not all(isinstance(node, str) and single_node(node)
                                                             for node in cfg["exclude_nodes"]):
        raise ConfigError("fleet.json: exclude_nodes must be a list of node names")
    reaper = cfg["comment"].get("OccupiedIdleGPUsJobReaper") if isinstance(cfg["comment"], dict) else None
    if not isinstance(reaper, dict) or not all(isinstance(reaper.get(key), str) and reaper[key]
                                               for key in ("exemptIdleTimeMins", "reason", "description")):
        raise ConfigError("fleet.json: comment must be the OccupiedIdleGPUsJobReaper exemption object")


def read_registry(directory):
    """{job_id: record} of registry/<job_id>.json; skips temp files and unreadable or partial records (one bad file
    costs only its own record). A directory that cannot be listed (missing, no permission, a Lustre error) raises
    RegistryError instead of reading as empty (F10): with no records every RUNNING worker would be a startup without
    a record, which police() cancels as stuck after stuck_startup_min (mass cancels, node exclusions, an open
    breaker). os.scandir, not Path.glob: glob returns nothing for a missing or unreadable directory."""
    try:
        with os.scandir(directory) as entries:
            names = sorted(entry.name for entry in entries)
    except OSError as error:
        raise RegistryError("cannot list %s: %s" % (directory, error))
    records = {}
    for name in names:
        stem = name[:-len(".json")]
        if not name.endswith(".json") or name.startswith(".") or not stem.isdigit():
            continue
        record = read_json(Path(directory) / name)
        if isinstance(record, dict) and str(record.get("job_id", "")) == stem:
            records[stem] = record
    return records


def worker_routing(record, now):
    """'ready' / 'draining' if the contract makes this registry record routable, else None."""
    if not isinstance(record, dict) or record.get("status") not in SERVING:
        return None
    heartbeat, end, drain = record.get("heartbeat_at"), record.get("end_epoch"), record.get("drain_at")
    if not isinstance(heartbeat, (int, float)) or now - heartbeat > HEARTBEAT_TIMEOUT_S:
        return None
    if isinstance(end, (int, float)) and now >= end:
        return None
    if record["status"] == "ready" and isinstance(drain, (int, float)) and now >= drain:
        return "draining"
    return record["status"]


# ---- Slurm -----------------------------------------------------------------------------------------------------------

def etc_environment():
    values = {}
    try:
        lines = Path("/etc/environment").read_text().splitlines()
    except OSError:
        return values
    for line in lines:
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            values[key] = value.strip().strip("\"'")
    return values


def slurm_env(epoch_times=False):
    """Minimal environment for every Slurm command. sbatch always exports the caller's SLURM_* variables, so a daemon
    running inside a job would leak its own allocation into new jobs; SQUEUE_*, SACCT_*, SCANCEL_* and
    SLURM_TIME_FORMAT would change what the commands select or print."""
    account = pwd.getpwuid(os.getuid())
    env = {"PATH": SLURM_BIN + ":/usr/local/bin:/usr/bin:/bin", "HOME": account.pw_dir, "USER": account.pw_name,
           "LOGNAME": account.pw_name, "LANG": "C.UTF-8"}
    conf = os.environ.get("SLURM_CONF") or etc_environment().get("SLURM_CONF")  # login nodes have no slurmd
    if conf:
        env["SLURM_CONF"] = conf
    if epoch_times:
        env["SLURM_TIME_FORMAT"] = "%s"
    return env


def parse_rows(text, keys, what):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", len(keys) - 1)
        if len(parts) != len(keys):
            raise SlurmError("unexpected %s line: %r" % (what, line[:200]))
        rows.append(dict(zip(keys, (part.strip() for part in parts))))
    return rows


def parse_squeue(text):
    jobs = parse_rows(text, SQUEUE_KEYS, "squeue")
    for job in jobs:
        job["start"], job["end"] = epoch(job["start"]), epoch(job["end"])
        if job["node"] in ("(null)", "None assigned"):
            job["node"] = ""
    return jobs


def parse_sacct(text):
    """{job_id: row}; a State like "CANCELLED by 146504" is split into state and cancelled_by."""
    rows = {}
    for row in parse_rows(text, SACCT_KEYS, "sacct"):
        state, _, by = row["state"].partition(" by ")
        row.update(state=state.strip(), cancelled_by=by.strip() or None, start=epoch(row["start"]),
                   end=epoch(row["end"]))
        if row["node"] in ("None assigned", "(null)"):
            row["node"] = ""
        rows[row["job_id"]] = row
    return rows


class Slurm:
    """Slurm CLI calls: argv lists (never a shell), scrubbed environment, bounded runtime."""

    def __init__(self, bin_dir=SLURM_BIN, timeout=120):
        self.bin_dir, self.timeout = bin_dir, timeout

    def run(self, argv, epoch_times=False):
        command = [os.path.join(self.bin_dir, argv[0])] + list(argv[1:])
        try:
            done = subprocess.run(command, env=slurm_env(epoch_times), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  universal_newlines=True, timeout=self.timeout)
        except (OSError, subprocess.SubprocessError) as error:
            raise SlurmError("%s: %s" % (argv[0], error))
        if done.returncode != 0:
            message = (done.stderr.strip() or done.stdout.strip())[:500]
            raise SlurmError("%s exited %d: %s" % (argv[0], done.returncode, message))
        return done

    def squeue(self, user):
        return parse_squeue(self.run(["squeue", "-h", "-u", user, "-o", SQUEUE_FORMAT], epoch_times=True).stdout)

    def sacct(self, job_ids):
        if not job_ids:
            return {}
        return parse_sacct(self.run(["sacct", "-n", "-P", "-X", "-j", ",".join(job_ids), "--format=" + SACCT_FORMAT],
                                    epoch_times=True).stdout)

    def sbatch(self, argv):
        """Submit (argv as built by Fleet.sbatch_argv, argv[0] == "sbatch"); returns the job id."""
        output = self.run(argv).stdout.strip().split(";", 1)[0]
        if not output.isdigit():
            raise SlurmError("sbatch printed no job id: %r" % output[:200])
        return output

    def sbatch_test(self, argv):
        """sbatch --test-only: validates the request and estimates a start; submits nothing."""
        done = self.run(argv[:1] + ["--test-only"] + argv[1:])
        return done.stderr.strip() or done.stdout.strip()

    def scancel(self, user, job_ids, pending_only=False):
        """scancel explicit job ids of user; pending_only (--state=PENDING) leaves alone a job that started since the
        caller's squeue (scancel signals only the jobs that match every filter). No job id, no call (F6): `scancel
        --user=<user>` alone would cancel every job of the user."""
        if not job_ids:
            return
        self.run(["scancel", "--user=" + user] + (["--state=PENDING"] if pending_only else []) + sorted(job_ids))

    def comment(self, job_id):
        return self.run(["squeue", "-h", "-j", job_id, "-o", "%k"]).stdout.strip()


# ---- fleet -----------------------------------------------------------------------------------------------------------

class Fleet:
    """One fleet directory (fleet.json, registry/, state/, logs/) and its Slurm access; tests use a temp root."""

    def __init__(self, root=ROOT, slurm=None, clock=time.time):
        self.root = Path(root)
        self.slurm = slurm if slurm is not None else Slurm()
        self.clock = clock
        self.config_path = self.root / "fleet.json"
        self.registry = self.root / "registry"
        self.state = self.root / "state"
        self.logs = self.root / "logs"
        self.worker_sbatch = self.root / "worker.sbatch"
        self.fleetd_sbatch = self.root / "fleetd.sbatch"
        self.stop_flag = self.state / "STOP"
        self.reload_flag = self.state / "RELOAD"
        self._lock_depth = 0

    def config(self, create=True):
        """fleet.json over DEFAULTS; written with the defaults when missing (unless create=False)."""
        if self.config_path.exists():
            stored = read_json(self.config_path)
            if not isinstance(stored, dict):
                raise ConfigError("unreadable %s" % self.config_path)
        else:
            stored = {}
            if create:
                write_json(self.config_path, DEFAULTS)
        cfg = json.loads(json.dumps(DEFAULTS))
        cfg.update(stored)
        check_config(cfg)
        return cfg

    def breaker(self):
        state = read_json(self.state / "breaker.json", {}) or {}
        state.setdefault("open", False)
        state.setdefault("failures", [])
        state.setdefault("noted", {})
        return state

    @contextmanager
    def locked(self, timeout=300):
        """Mutation lock (state/pass.lock) shared by daemon passes and manual commands; re-entrant."""
        if self._lock_depth:
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
            return
        self.state.mkdir(parents=True, exist_ok=True)
        stream = open(self.state / "pass.lock", "a")
        try:
            deadline = time.time() + timeout
            while True:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as error:
                    if error.errno not in (errno.EAGAIN, errno.EACCES):
                        raise
                    if time.time() > deadline:
                        raise LockBusy("state/pass.lock is busy (another fleet.py command?)")
                    time.sleep(1)
            self._lock_depth = 1
            try:
                yield
            finally:
                self._lock_depth = 0
                fcntl.flock(stream, fcntl.LOCK_UN)
        finally:
            stream.close()

    def event(self, kind, /, incident=False, **fields):
        now = self.clock()
        record = {"ts": round(now, 3), "time": iso(now), "event": kind}
        if incident:
            record["incident"] = True
        record.update(fields)
        append_jsonl(self.state / "events.jsonl", record)
        print("%s %s%s %s" % (iso(now), "INCIDENT " if incident else "", kind, json.dumps(fields, sort_keys=True)),
              flush=True)
        return record

    def safe_event(self, kind, /, incident=False, **fields):
        try:
            self.event(kind, incident=incident, **fields)
        except OSError as error:
            print("event %s not recorded: %s" % (kind, error), flush=True)

    def sbatch_argv(self, cfg, name, partition=None, walltime=None, account=None):
        """Every candidate of a slot gets the same argv but --account (default: the first worker account)."""
        argv = ["sbatch", "--parsable", "--export=NONE", "--job-name=" + name, "--comment=" + comment_json(cfg),
                "--account=" + (account or worker_accounts(cfg)[0]), "--partition=" + (partition or cfg["partition"]),
                "--time=" + (walltime or cfg["time"])]
        if cfg["exclude_nodes"]:
            argv.append("--exclude=" + ",".join(sorted(set(cfg["exclude_nodes"]))))
        return argv + [str(self.worker_sbatch)]

    def submit_worker(self, cfg, slot, known, partition=None, walltime=None, reason="coverage", account=None,
                      slot_submitted_at=None):
        """sbatch one worker (caller holds the lock); records it in `known` and verifies the exemption comment.
        slot_submitted_at: when the slot's first candidate was submitted (a slot's queue wait counts from then)."""
        name = cfg["job_prefix"] + slot
        account = account or worker_accounts(cfg)[0]
        self.logs.mkdir(parents=True, exist_ok=True)  # Slurm does not create the --output directory
        job_id = self.slurm.sbatch(self.sbatch_argv(cfg, name, partition, walltime, account))
        now = self.clock()
        known[job_id] = {"job_id": job_id, "name": name, "slot": slot, "partition": partition or cfg["partition"],
                         "account": account, "submit_account": account, "state": "PENDING", "node": "", "start": None,
                         "end": None, "submitted_at": now, "first_seen": now, "last_seen": now}
        if slot_submitted_at is not None:
            known[job_id]["slot_submitted_at"] = slot_submitted_at
        self.event("submit", job_id=job_id, slot=slot, name=name, partition=partition or cfg["partition"],
                   account=account, reason=reason, excluded_nodes=len(cfg["exclude_nodes"]))
        try:
            comment = self.slurm.comment(job_id)
        except SlurmError as error:
            comment = "unverified: %s" % error
        if comment != comment_json(cfg):
            self.event("comment_mismatch", incident=True, job_id=job_id, got=comment[:300])
        return job_id

    def reconcile(self, dry_run=False):
        """One maintenance pass. Returns the actions taken, or with dry_run planned (then nothing is written)."""
        if dry_run:
            return Pass(self, dry_run=True).run()
        with self.locked():
            return Pass(self, dry_run=False).run()

    def snapshot(self, cfg, queue, registry, now, known=None, unconfirmed=(), horizon=None):
        """Machine-readable fleet view (state/status.json, `status --json`)."""
        known = known or {}
        if horizon is None:
            horizon = horizon_s(cfg, read_json(self.state / "waits.json", []) or [],
                                [job for job in queue if is_worker(cfg, job)], now)
        registry = current_registry(registry, queue)
        workers = [job for job in queue if is_worker(cfg, job)]
        ghosts = [dict(known[job_id], user=cfg["user"], reason="", elapsed="", time_left="", ghost=True)
                  for job_id in unconfirmed if job_id in known]  # gone from squeue, end not confirmed yet
        rows, counts, reasons = [], Counter(), Counter()
        for job in sorted(workers + ghosts, key=lambda item: (slot_of(cfg, item["name"]) or "~", int(item["job_id"]))):
            ghost = bool(job.get("ghost"))
            record = registry.get(job["job_id"]) or {}
            running = job["state"] in RUNNING and not ghost
            status = record.get("status") if running else None
            routing = worker_routing(record, now) if running else None
            heartbeat = record.get("heartbeat_at")
            if ghost:
                bucket = "vanished"
            elif job["state"] in PENDING:
                bucket = "held" if held(job) else "pending"
                reasons[job["reason"]] += 1
            elif running:
                bucket = routing or (status if status in ("unhealthy", "failed", "stopped") else
                                     "stale" if status in SERVING else "starting")
            else:
                bucket = "other"
            counts[bucket] += 1
            counts["routable"] += bool(routing)
            rows.append({
                "job_id": job["job_id"], "slot": slot_of(cfg, job["name"]), "name": job["name"],
                "state": "VANISHED" if ghost else job["state"], "partition": job.get("partition"),
                "account": job.get("account") or (known.get(job["job_id"]) or {}).get("account"),
                "node": job.get("node") or "", "elapsed_s": duration(job.get("elapsed")),
                "left_s": remaining(job, now) if running else duration(job.get("time_left")),
                "start": job.get("start"), "end": job.get("end"),
                "reason": job["reason"] if job["state"] in PENDING and not ghost else "", "registry": status,
                "routing": routing, "ready_at": record.get("ready_at") if status else None,
                "heartbeat_age_s": round(now - heartbeat) if status and isinstance(heartbeat, (int, float)) else None,
                "base_url": record.get("base_url") if status else None, "parity": record.get("parity"),
                "registry_reason": record.get("reason") if status else None})
        counts["coverage"] = coverage(cfg, workers + ghosts, registry, now,
                                      {job_id for job_id, item in known.items() if item.get("cancel_requested_at")},
                                      horizon)
        counts["total"] = len(workers) + len(ghosts)
        # "pending" counts jobs (a slot can queue one candidate per worker account); pending_slots = workers waiting
        # for a node (pending, not held, no running candidate; a candidate pair once, race_key)
        pairs = pair_slots(cfg, workers)
        running = {race_key(cfg, job, pairs) for job in workers if job["state"] in RUNNING}
        counts["pending_slots"] = len({race_key(cfg, job, pairs) for job in workers
                                       if job["state"] in PENDING and not held(job)} - running)
        previous = read_json(self.state / "status.json") or {}
        # pair_delay_min only with several worker accounts: a single-account status.json stays as before
        paired = {"pair_delay_min": cfg["pair_delay_min"]} if len(worker_accounts(cfg)) > 1 else {}
        return {"time": iso(now), "ts": now,
                "config": dict({key: cfg[key] for key in ("target", "horizon_min", "max_jobs", "max_submits_per_pass",
                                                          "account", "partition", "time", "job_prefix",
                                                          "coverage_partitions", "stagger_min")},
                               worker_accounts=worker_accounts(cfg), horizon_effective_min=horizon // 60, **paired),
                "counts": dict(counts), "pending_reasons": dict(reasons), "jobs": rows,
                "exclude_nodes": sorted(cfg["exclude_nodes"]), "breaker": self.breaker(),
                "stop_flag": self.stop_flag.exists(), "daemon": read_json(self.state / "daemon.json"),
                "last_reconcile_at": previous.get("last_reconcile_at")}

    def daemon(self, interval=60, until=None, wait_lock=False):
        """Reconcile every `interval` s until `until` (epoch) or the STOP flag; one instance (state/fleet.lock,
        a cross-host flock on Lustre). Returns 75 when another daemon holds the lock (unless wait_lock), RELOAD_EXIT
        after state/RELOAD (see _daemon_loop), else 0."""
        self.state.mkdir(parents=True, exist_ok=True)
        with open(self.state / "fleet.lock", "a") as guard:
            while True:
                try:
                    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as error:
                    if error.errno not in (errno.EAGAIN, errno.EACCES):
                        raise
                    if not wait_lock:
                        print("another daemon holds state/fleet.lock: %s" % json.dumps(read_json(
                            self.state / "daemon.json")), flush=True)
                        return 75
                    if until and time.time() >= until:
                        return 0
                    time.sleep(30)
            stop = {"signal": None}

            def on_signal(number, _frame):
                stop["signal"] = number

            previous = {number: signal.signal(number, on_signal) for number in (signal.SIGTERM, signal.SIGINT)}
            try:
                return self._daemon_loop(interval, until, stop)
            finally:
                for number, handler in previous.items():
                    signal.signal(number, handler)

    def _daemon_loop(self, interval, until, stop):
        """The loop; returns RELOAD_EXIT for a state/RELOAD flag created after this daemon started (a code update:
        the batch loop restarts the daemon, which loads fleet.py again; an older flag is only removed)."""
        progress = {"pass_started": None}
        start_watchdog(progress, PASS_TIMEOUT_S)
        me = {"host": socket.gethostname(), "pid": os.getpid(), "job_id": os.environ.get("SLURM_JOB_ID"),
              "started_at": time.time(), "interval_s": interval, "until": until}
        self.safe_event("daemon_start", **me)
        skipped = {"count": 0, "noted": 0.0}  # passes skipped on Slurm/config/registry errors, logged every <= 10 min
        status = 0
        while True:
            if stop["signal"]:
                reason = "signal %d" % stop["signal"]
                break
            if until and time.time() >= until:
                reason = "until"
                break
            if self.stop_flag.exists():
                reason = "STOP flag"
                break
            try:
                flagged = self.reload_flag.stat().st_mtime
            except OSError:
                flagged = None
            if flagged is not None:
                try:
                    self.reload_flag.unlink()
                except OSError as error:  # cannot be removed: ignored, or every restarted daemon would exit again
                    print("state/RELOAD not removed (%s): ignored" % error, flush=True)
                else:
                    if flagged >= me["started_at"] - 1:
                        reason, status = "RELOAD flag", RELOAD_EXIT
                        break
                    self.safe_event("reload_flag_ignored", reason="older than this daemon", flag_mtime=flagged)
            began = time.time()
            progress["pass_started"] = began
            try:
                self.reconcile()
            except (SlurmError, ConfigError, LockBusy, RegistryError) as error:  # F10: registry/ unlistable too
                skipped["count"] += 1
                if time.time() - skipped["noted"] >= 600:
                    self.safe_event("pass_skipped", passes=skipped["count"],
                                    error="%s: %s" % (type(error).__name__, error))
                    skipped.update(count=0, noted=time.time())
            except Exception as error:  # never die on one bad pass
                traceback.print_exc()
                self.safe_event("daemon_error", incident=True, error="%s: %s" % (type(error).__name__, error))
            progress["pass_started"] = None
            try:
                write_json(self.state / "daemon.json",
                           dict(me, heartbeat_at=time.time(), pass_s=round(time.time() - began, 1)))
            except OSError as error:
                print("daemon.json write failed: %s" % error, flush=True)
            deadline = began + interval if not until else min(began + interval, until)
            while not stop["signal"] and time.time() < deadline:
                time.sleep(min(5.0, max(0.1, deadline - time.time())))
        self.safe_event("daemon_stop", reason=reason, **me)
        try:
            write_json(self.state / "daemon.json",
                       dict(me, heartbeat_at=time.time(), stopped_at=time.time(), stop_reason=reason))
        except OSError:
            pass
        return status


def start_watchdog(progress, limit_s):
    """Exit the process when one pass wedges (hung Lustre or Slurm call); fleetd.sbatch restarts the daemon."""
    def watch():
        while True:
            time.sleep(30)
            began = progress.get("pass_started")
            if began and time.time() - began > limit_s:
                sys.stderr.write("watchdog: pass running for %d s; exiting\n" % (time.time() - began))
                sys.stderr.flush()
                os._exit(70)
    threading.Thread(target=watch, daemon=True).start()


class Pass:
    """One reconcile pass; decisions are applied at once unless dry_run (then they are only listed)."""

    def __init__(self, fleet, dry_run):
        self.fleet, self.dry_run = fleet, dry_run
        self.actions = []
        self.config_changed = False
        self.failed_accounts = set()  # accounts whose sbatch failed in this pass: not tried again in it

    def note(self, kind, /, incident=False, **fields):
        action = dict(action=kind, **fields)
        if incident:
            action["incident"] = True
        self.actions.append(action)
        if not self.dry_run:
            self.fleet.event(kind, incident=incident, **fields)

    def note_throttled(self, kind, /, incident=False, key=None, **fields):
        """note() for a condition that persists across passes: logged at most every NOTE_EVERY_S (per key, default
        the kind)."""
        noted, key = self.breaker["noted"], key or kind
        if self.now - noted.get(key, 0) >= NOTE_EVERY_S:
            noted[key] = self.now
            self.note(kind, incident=incident, **fields)
        else:
            self.actions.append(dict(action=kind, repeated=True, **fields))

    def run(self):
        fleet = self.fleet
        cfg = self.cfg = fleet.config(create=not self.dry_run)
        now = self.now = fleet.clock()
        queue = fleet.slurm.squeue(cfg["user"])  # a failed query raises: no decision on partial information
        queued = {job["job_id"]: job for job in queue}
        workers = {job["job_id"]: job for job in queue if is_worker(cfg, job)}
        listed = read_registry(fleet.registry)  # F10: raises (the pass acts on nothing) when registry/ is unlistable
        registry = current_registry(listed, list(workers.values()))
        known = read_json(fleet.state / "jobs.json", {}) or {}
        self.breaker = fleet.breaker()
        waits = read_json(fleet.state / "waits.json", []) or []

        # A worker's queue wait is one sample: of a candidate pair, the first candidate to start, from the pair's first
        # submission (a sibling that starts later is a loser and is cancelled); any other job is its own (race_key)
        pairs, first = pair_slots(cfg, list(workers.values())), {}
        for job in workers.values():
            if job["state"] in RUNNING and job["start"]:
                key, group = (job["start"], int(job["job_id"])), race_key(cfg, job, pairs)
                first[group] = min(first.get(group, key), key)
        for job_id, job in workers.items():  # adopt / refresh
            record = known.setdefault(job_id, {"job_id": job_id, "first_seen": now, "submitted_at": None})
            if job["state"] in RUNNING and job["start"] and record.get("run_start") != job["start"]:
                if record.get("run_start") is not None:  # a new run of a requeued job: per-run flags start over
                    for key in ("failure_recorded", "terminal_seen_at", "cancel_requested_at"):
                        record.pop(key, None)
                elif record.get("submitted_at") and first.get(race_key(cfg, job, pairs)) == (job["start"],
                                                                                             int(job_id)):
                    # first run: its queue wait feeds the adaptive horizon
                    waits.append({"job_id": job_id, "partition": job["partition"], "start": job["start"],
                                  "account": job.get("account"),
                                  "wait_s": max(0, int(job["start"] - (record.get("slot_submitted_at")
                                                                       or record["submitted_at"])))})
                record["run_start"] = job["start"]
            record.update(name=job["name"], slot=slot_of(cfg, job["name"]), partition=job["partition"],
                          account=job.get("account") or record.get("account"), state=job["state"],
                          node=job["node"] or record.get("node", ""), start=job["start"], end=job["end"],
                          last_seen=now, vanished_at=None)
        self.waits = waits[-50:]
        self.horizon = horizon_s(cfg, self.waits, list(workers.values()), now)

        unconfirmed = self.finalize(known, registry, queued)
        losers = self.settle(workers, registry, known)  # candidates another candidate of their slot won against
        cancelled = self.police({job_id: job for job_id, job in workers.items() if job_id not in losers},
                                registry, known) | losers
        self.top_up(workers, registry, known, unconfirmed, cancelled)

        if not self.dry_run:
            write_json(fleet.state / "breaker.json", self.breaker)
            write_json(fleet.state / "jobs.json", known)
            write_json(fleet.state / "waits.json", self.waits)
            if self.config_changed:
                stored = read_json(fleet.config_path)  # re-read: keep other keys edited meanwhile
                if isinstance(stored, dict):
                    stored["exclude_nodes"] = cfg["exclude_nodes"]
                    write_json(fleet.config_path, stored)
                    self.exclude_pending({job_id: job for job_id, job in workers.items() if job_id not in cancelled})
                else:  # never replace a config that is unreadable (being edited) by a partial one
                    self.note("exclusion_not_saved", incident=True, exclude_nodes=cfg["exclude_nodes"])
            # The queue as the next pass will see it: cancelled jobs on their way out, this pass's submissions pending.
            view = [dict(job, state="COMPLETING") if job["job_id"] in cancelled else job for job in queue]
            view += [{"job_id": action["job_id"], "user": cfg["user"], "state": "PENDING",
                      "partition": known[action["job_id"]]["partition"], "node": "", "elapsed": "0:00",
                      "time_left": cfg["time"], "start": None, "end": None, "reason": "submitted",
                      "account": known[action["job_id"]].get("account"), "name": cfg["job_prefix"] + action["slot"]}
                     for action in self.actions if action["action"] == "submit" and action.get("job_id")]
            try:
                latest = read_registry(fleet.registry)
            except RegistryError:  # F10: the pass has acted already; the status shows this pass's own listing
                latest = listed
            status = fleet.snapshot(cfg, view, latest, now, known, unconfirmed, self.horizon)
            status["last_reconcile_at"] = now
            status["last_actions"] = self.actions
            write_json(fleet.state / "status.json", status)
        return self.actions

    def exclude_pending(self, workers):
        """A new exclusion also applies to fleet workers still pending (every candidate of a slot; the caller leaves
        out the ones being cancelled): they were submitted with the old list, and a node that fails fast would
        otherwise take them one after the other."""
        nodes = ",".join(sorted(set(self.cfg["exclude_nodes"])))
        for job_id, job in sorted(workers.items()):
            if job["state"] in PENDING and is_worker(self.cfg, job):
                try:
                    self.fleet.slurm.run(["scontrol", "update", "JobId=" + job_id, "ExcNodeList=" + nodes])
                except SlurmError as error:
                    self.note("exclude_update_failed", job_id=job_id, error=str(error))

    # 1. jobs that left the queue
    def finalize(self, known, registry, queued):
        now = self.now
        gone = sorted((set(known) | set(registry)) - set(queued), key=int)
        if not gone:
            return []
        try:
            accounting = self.fleet.slurm.sacct(gone)
        except SlurmError as error:
            self.note_throttled("sacct_failed", error=str(error))
            accounting = None
        unconfirmed = []
        for job_id in gone:
            info = accounting.get(job_id) if accounting is not None else None
            confirmed = info is not None and info["state"] in TERMINAL
            if not confirmed:
                if job_id in known:
                    record = known[job_id]
                    record["vanished_at"] = record.get("vanished_at") or now
                    if now - record["vanished_at"] < UNCONFIRMED_GRACE_S:
                        unconfirmed.append(job_id)  # keeps counting with its last known state: no mass resubmit
                        continue
                elif accounting is None or info is not None:
                    continue  # registry-only record of a job sacct shows active (or sacct failed): next pass
            # Accounting that does not show the end is not evidence of a failure: finalized as UNKNOWN.
            self.ended(job_id, known.pop(job_id, {"job_id": job_id}), registry.get(job_id), info if confirmed else None)
        return unconfirmed

    def ended(self, job_id, record, entry, info):
        cfg = self.cfg
        state = info["state"] if info else "UNKNOWN"
        status = (entry or {}).get("status")
        node = (entry or {}).get("node") or record.get("node") or (info or {}).get("node") or ""
        by = (info or {}).get("cancelled_by")
        ours = bool(by) and by.isdigit() and int(by) == os.getuid()
        started = bool(info and info.get("start"))
        reaped = bool(info and REAPER in (info.get("admin_comment") or "").lower())
        # F7: a candidate cancelled by settle() (sibling_cancel) never counts as a worker failure, exclusion or breaker
        # event, whatever its registry says: our scancel of a starting candidate can make vLLM exit before
        # serve_student.sh's TERM trap runs, which records "failed". (settle() drops the mark of a candidate that is
        # no longer a loser, F1, so a reprieved one is judged like any worker.)
        settled = bool(record.get("sibling_cancel"))
        if reaped:
            self.note("reaper_cancel", incident=True, job_id=job_id, slot=record.get("slot"), node=node,
                      admin_comment=info["admin_comment"][:300], registry_status=status)
        if self.breaker.get("probe_job") == job_id and (settled or status not in ("failed", "unhealthy")):
            self.breaker.update(probe_job=None, probe_after=self.now)
        if reaped or settled:
            pass
        elif state == "NODE_FAIL":
            self.failure(record, "node_fail", "Slurm NODE_FAIL", node, True)
        elif status == "failed":
            self.failure(record, "startup_failed", entry.get("reason"), node, entry.get("exclude_node", True))
        elif status == "unhealthy":
            self.failure(record, "unhealthy", entry.get("reason"), node,
                         cfg["exclude_on_unhealthy"] and entry.get("exclude_node", True))
        elif status == "starting" and started and not ours:
            self.failure(record, "ended_while_starting", "job %s while the worker was starting" % state, node, True)
        elif entry is None and started and not ours and state != "UNKNOWN":
            # serve_student.sh never wrote a record: the container/srun step failed (node suspect if the job FAILED)
            self.failure(record, "no_registry", "job %s without a registry record" % state, node,
                         state in ("FAILED", "BOOT_FAIL", "OUT_OF_MEMORY"))
        served = None
        if entry and isinstance(entry.get("ready_at"), (int, float)) and isinstance(entry.get("heartbeat_at"),
                                                                                    (int, float)):
            served = int(entry["heartbeat_at"] - entry["ready_at"])
        # sibling_of names the candidate that won the slot of a candidate cancelled by settle()
        extra = {"sibling_of": record["sibling_of"]} if record.get("sibling_of") else {}
        self.note("job_ended", job_id=job_id, slot=record.get("slot"), node=node, state=state,
                  exit_code=(info or {}).get("exit_code"), cancelled_by=by, registry_status=status,
                  registry_reason=(entry or {}).get("reason"), served_s=served, account=record.get("account"),
                  **extra)
        path = self.fleet.registry / ("%s.json" % job_id)
        if not self.dry_run and path.exists():
            try:
                (self.fleet.registry / "archive").mkdir(parents=True, exist_ok=True)
                os.replace(path, self.fleet.registry / "archive" / path.name)
            except OSError as error:
                self.note("archive_failed", job_id=job_id, error=str(error))

    # 2. candidate pairs, running jobs
    def settle(self, workers, registry, known):
        """The candidates of a slot (one per worker account) race for its node; this cancels the losers. Only a
        candidate pair (candidate_pair: all on the fleet partition, pairwise different accounts) is settled (F3):
        any other duplicate of a slot name keeps the single-account rules, so a single-account fleet behaves exactly
        as before worker accounts, also with same-name jobs made by hand. Once a candidate serves (routable:
        ready/draining with a fresh heartbeat), every candidate that is pending or still starting goes. Before that,
        of several started ones the earliest start (then the lowest job id) stays and the others still starting go
        at once (two startups would hold two nodes), while pending ones stay queued as the backup of a startup that
        may still fail (pending costs no node). Only pending or starting candidates are ever cancelled here: one that
        reported ready/draining stays (serving duplicates, only after a daemon outage, both serve out their time; a
        stale heartbeat is police()'s call), and one whose worker gave up (failed/unhealthy/stopped) is left to
        police(). A running job police() is cancelling (police_cancelled) neither keeps the slot nor is settled here
        (F2): it is on its way out, and keeping it would also cancel its healthy sibling. Pending losers go with
        --state=PENDING, so one that started since this pass's squeue is left alone and settled as a running one
        next pass. A request is repeated after CANCEL_RETRY_S, or at once when the job left PENDING since. A
        candidate with an earlier request that is no longer a loser and still pending or viable (e.g. its keeper
        failed, or the request failed and the keeper left) loses its sibling marks (F1): it is an ordinary worker
        again (counted, policed, judged by its own end), keeping only a police() cancel time. Returns {job_id} of
        the losers: no coverage, no policing, no exclusion update. With one worker account a slot holds one job and
        nothing happens here."""
        cfg, now = self.cfg, self.now
        losers = {}
        for _slot, jobs in sorted(by_slot(cfg, workers.values()).items()):
            if not candidate_pair(cfg, jobs):  # F3: one job, or a duplicate that is no account pair
                continue
            alive = sorted((job for job in jobs if viable(job, registry)
                            and not police_cancelled(known.get(job["job_id"]) or {})),  # F2
                           key=lambda job: (job["start"] or now, int(job["job_id"])))
            serving = [job for job in alive if worker_routing(registry.get(job["job_id"]), now)]
            starting = [job for job in alive if (registry.get(job["job_id"]) or {}).get("status") not in SERVING]
            if serving:
                keeper = serving[0]
                out = [job for job in jobs if job["state"] in PENDING] + starting
            elif len(alive) > 1:
                keeper = alive[0]
                out = [job for job in starting if job is not keeper]
            else:
                continue
            for job in out:
                losers[job["job_id"]] = keeper
        for pending_only in (True, False):
            fresh = []
            for job_id, keeper in sorted(losers.items()):
                job, record = workers[job_id], known[job_id]
                if (job["state"] in PENDING) != pending_only:
                    continue
                if not is_worker(cfg, job):
                    raise AssertionError("refusing to cancel non-fleet job %s" % job_id)
                record["sibling_of"] = keeper["job_id"]
                asked = record.get("sibling_cancel") or {}
                if asked.get("state") == job["state"] and now - asked.get("at", 0) < CANCEL_RETRY_S:
                    continue  # requested already; scancel takes a moment
                record["sibling_cancel"] = {"at": now, "state": job["state"]}
                record["cancel_requested_at"] = now
                fresh.append(job_id)
                self.note("sibling_cancel", job_id=job_id, slot=slot_of(cfg, job["name"]), state=job["state"],
                          account=job.get("account"), node=job["node"], keeper=keeper["job_id"],
                          keeper_account=keeper.get("account"), keeper_state=keeper["state"],
                          keeper_serving=bool(worker_routing(registry.get(keeper["job_id"]), now)))
            if fresh and not self.dry_run:
                try:
                    self.fleet.slurm.scancel(cfg["user"], fresh, pending_only=pending_only)
                except SlurmError as error:  # repeated after CANCEL_RETRY_S (at once if a job left PENDING)
                    self.note("sibling_cancel_failed", incident=True, job_ids=fresh, error=str(error))
        # F1: stale marks would keep such a job out of the status coverage (cancel_requested_at), delay a police()
        # cancel of it by CANCEL_RETRY_S, and hide a genuine failure of it at its end (F7)
        for job_id, job in sorted(workers.items()):
            record = known.get(job_id) or {}
            if job_id in losers or not record.get("sibling_cancel") \
                    or not (job["state"] in PENDING or viable(job, registry)):
                continue
            asked = record.pop("sibling_cancel")
            record.pop("sibling_of", None)
            if record.get("cancel_requested_at") == asked.get("at"):  # a police() cancel time stays
                record.pop("cancel_requested_at", None)
            self.note("sibling_cancel_dropped", job_id=job_id, slot=slot_of(cfg, job["name"]), state=job["state"],
                      account=job.get("account"), requested_at=asked.get("at"),
                      reason="no longer a loser (its keeper left or stopped serving, or it serves itself)")
        return set(losers)

    def police(self, workers, registry, known):
        cfg, now = self.cfg, self.now
        cancel, stale, serving = {}, [], 0
        for job_id, job in workers.items():
            if job["state"] not in RUNNING:
                continue
            record, entry = known[job_id], registry.get(job_id)
            status = (entry or {}).get("status")
            node = job["node"] or (entry or {}).get("node", "")
            if status in ("failed", "unhealthy", "stopped"):
                if record.get("sibling_cancel"):
                    pass  # F7: a candidate settle() cancelled, still listed: its registry status is our scancel's
                elif status == "failed":
                    self.failure(record, "startup_failed", entry.get("reason"), node, entry.get("exclude_node", True))
                elif status == "unhealthy":
                    self.failure(record, "unhealthy", entry.get("reason"), node,
                                 cfg["exclude_on_unhealthy"] and entry.get("exclude_node", True))
                record["terminal_seen_at"] = record.get("terminal_seen_at") or now
                if now - record["terminal_seen_at"] > cfg["terminal_grace_min"] * 60:
                    cancel[job_id] = "worker %s but the job is still running" % status
            elif status in SERVING:
                serving += 1
                if self.breaker["open"] and job_id == self.breaker.get("probe_job"):
                    self.breaker.update(open=False, failures=[], probe_job=None, closed_at=now)
                    self.note("breaker_closed", job_id=job_id, reason="half-open probe worker is serving")
                heartbeat = entry.get("heartbeat_at")
                if not isinstance(heartbeat, (int, float)) or now - heartbeat > cfg["heartbeat_stale_min"] * 60:
                    stale.append(job_id)
            else:  # no record yet, or still starting
                age = now - (job["start"] or now)
                if age > cfg["stuck_startup_min"] * 60:
                    reason = "not ready %d min after the job start (registry %s)" % (age // 60, status or "missing")
                    self.failure(record, "stuck_startup", reason, node, True)
                    cancel[job_id] = reason
        if stale and (len(stale) * 2 <= serving or len(stale) == 1):
            for job_id in stale:
                cancel[job_id] = "registry heartbeat older than %d min" % cfg["heartbeat_stale_min"]
        elif stale:  # most records stale at once: a Lustre or clock problem, not the workers
            self.note_throttled("mass_stale_heartbeats", incident=True, stale=stale, serving=serving)
        pending = {job_id for job_id in cancel if now - (known[job_id].get("cancel_requested_at") or 0) < CANCEL_RETRY_S}
        fresh = {job_id: reason for job_id, reason in cancel.items() if job_id not in pending}
        if fresh:
            self.cancel(fresh, workers, known)
        return set(cancel)  # on their way out: no coverage

    def cancel(self, reasons, workers, known=None):
        """scancel fleet worker jobs by explicit id; refuses anything that is not a worker of cfg['user']."""
        ids = []
        for job_id, reason in sorted(reasons.items()):
            job = workers.get(job_id)
            if job is None or not is_worker(self.cfg, job):
                raise AssertionError("refusing to cancel non-fleet job %s" % job_id)
            ids.append(job_id)
            if known is not None:
                known[job_id]["cancel_requested_at"] = self.now
            self.note("cancel", job_id=job_id, slot=slot_of(self.cfg, job["name"]), node=job["node"], reason=reason)
        if ids and not self.dry_run:
            try:
                self.fleet.slurm.scancel(self.cfg["user"], ids)
            except SlurmError as error:
                self.note("scancel_failed", incident=True, job_ids=ids, error=str(error))

    def failure(self, record, kind, reason, node, exclude):
        """Count a worker failure once (circuit breaker) and exclude its node when the node is suspect."""
        if record.get("failure_recorded"):
            return
        record["failure_recorded"] = True
        cfg, now, breaker = self.cfg, self.now, self.breaker
        job_id = record.get("job_id")
        breaker["failures"] = [item for item in breaker["failures"]
                               if now - item["ts"] < cfg["breaker_window_min"] * 60]
        breaker["failures"].append({"ts": now, "job_id": job_id, "node": node, "kind": kind})
        self.note("worker_failure", job_id=job_id, slot=record.get("slot"), node=node, kind=kind, reason=reason,
                  exclude=bool(exclude))
        if exclude and single_node(node):
            if breaker["open"]:
                self.note("exclude_skipped", job_id=job_id, node=node, reason="circuit breaker open")
            elif node not in cfg["exclude_nodes"]:
                cfg["exclude_nodes"].append(node)
                self.config_changed = True
                if not self.dry_run:
                    append_jsonl(self.fleet.state / "excluded_nodes.jsonl", {
                        "ts": now, "time": iso(now), "node": node, "job_id": job_id, "kind": kind, "reason": reason})
                self.note("exclude_node", job_id=job_id, node=node, kind=kind)
        if breaker["open"] and job_id and job_id == breaker.get("probe_job"):  # the half-open probe failed
            breaker["cooldown_s"] = min(4 * 3600, 2 * breaker.get("cooldown_s", cfg["breaker_cooldown_min"] * 60))
            breaker.update(probe_job=None, probe_after=now + breaker["cooldown_s"])
            self.note("breaker_probe_failed", incident=True, job_id=job_id, retry_at=iso(breaker["probe_after"]))
        sources = {item.get("node") or "job %s" % item.get("job_id") for item in breaker["failures"]}
        if not breaker["open"] and 0 < cfg["breaker_failures"] <= len(sources):  # one bad node is not systematic
            breaker.update(open=True, since=now, cooldown_s=cfg["breaker_cooldown_min"] * 60,
                           probe_after=now + cfg["breaker_cooldown_min"] * 60, probe_job=None,
                           reason="%d worker failures on %d nodes within %d min" % (
                               len(breaker["failures"]), len(sources), cfg["breaker_window_min"]))
            self.note("breaker_open", incident=True, reason=breaker["reason"], failures=breaker["failures"])

    # 3. coverage
    def usable_accounts(self, fallback=True):
        """Worker accounts to submit on now: not failed in this pass and, with several accounts, not in their
        ACCOUNT_RETRY_S backoff (breaker.json account_backoff), unless every one is (fallback: then all are tried,
        a new slot must not wait for a backoff)."""
        accounts = [account for account in worker_accounts(self.cfg) if account not in self.failed_accounts]
        if len(worker_accounts(self.cfg)) < 2:
            return accounts
        backoff = self.breaker.get("account_backoff") or {}
        usable = [account for account in accounts if self.now >= backoff.get(account, 0)]
        return usable or (accounts if fallback else [])

    def account_failed(self, account, slot, error):
        """An sbatch on `account` failed while the slot has (or may get) its candidate on another account: skip the
        account for the pass and, with several accounts, for ACCOUNT_RETRY_S; an incident at most every 30 min."""
        self.failed_accounts.add(account)
        if len(worker_accounts(self.cfg)) > 1:
            self.breaker.setdefault("account_backoff", {})[account] = self.now + ACCOUNT_RETRY_S
        self.note_throttled("account_submit_failed", incident=True, key="account_submit_failed " + account,
                            slot=slot, account=account, error=error, retry_after_s=ACCOUNT_RETRY_S)

    def account_worked(self, account):
        (self.breaker.get("account_backoff") or {}).pop(account, None)

    def top_up(self, workers, registry, known, unconfirmed, cancelled):
        """New slots up to the target, then pair completion. max_submits_per_pass bounds the slots that get a
        candidate in a pass, new slots and completions alike (completions use what new slots left of it); since F5
        each is one sbatch with two worker accounts, so it also bounds the sbatch calls that succeed."""
        cfg, now = self.cfg, self.now
        jobs = list(workers.values()) + [dict(known[job_id], user=cfg["user"], reason="", time_left="")
                                         for job_id in unconfirmed]
        leaving = set(cancelled) | {job_id for job_id in unconfirmed if known[job_id].get("cancel_requested_at")}
        covered = coverage(cfg, jobs, registry, now, leaving, self.horizon)
        total = node_count(cfg, [job for job in jobs if not (job["job_id"] in cancelled and job["job_id"] in workers)],
                           registry)
        wanted = min(cfg["target"] - covered, cfg["max_jobs"] - total, cfg["max_submits_per_pass"])
        slots = self.submit_slots(jobs, known, covered, wanted) if wanted > 0 else 0
        self.complete_pairs(workers, unconfirmed, known, leaving, cfg["max_submits_per_pass"] - slots)

    def submit_slots(self, jobs, known, covered, wanted):
        """New slots (lowest free), each with ONE job (F5): on the first usable worker account, or the next one when
        that sbatch fails (as the half-open probe always did). Its siblings on the other accounts come from pair
        completion once the slot has queued pair_delay_min: a fast queue then never starts two candidates. Returns
        how many slots got a job."""
        cfg, now = self.cfg, self.now
        probe = False
        if self.fleet.stop_flag.exists() or (self.breaker["open"] and (
                self.breaker.get("probe_job") or now < self.breaker.get("probe_after", float("inf")))):
            self.note_throttled("submit_paused", incident=self.breaker["open"], wanted=wanted, coverage=covered,
                                reason="STOP flag" if self.fleet.stop_flag.exists() else "circuit breaker open")
            return 0
        if self.breaker["open"]:  # half-open: one probe worker; it closes the breaker once it serves
            wanted, probe = 1, True
        dual = len(worker_accounts(cfg)) > 1
        used = {slot_of(cfg, job["name"]) for job in jobs}
        free = [slot for slot in ("%02d" % number for number in range(100)) if slot not in used]
        done = 0
        for slot in free[:wanted]:
            reason = "%scoverage %d/%d, horizon %d min" % ("breaker probe, " if probe else "", covered,
                                                           cfg["target"], self.horizon // 60)
            ids, errors = [], []
            for account in self.usable_accounts():
                if self.dry_run:
                    self.note("submit", slot=slot, name=cfg["job_prefix"] + slot, account=account, coverage=covered,
                              argv=self.fleet.sbatch_argv(cfg, cfg["job_prefix"] + slot, account=account))
                    ids.append(None)
                    break
                try:
                    job_id = self.fleet.submit_worker(cfg, slot, known, reason=reason, account=account)
                except SlurmError as error:
                    self.failed_accounts.add(account)
                    errors.append((account, str(error)))
                    continue
                self.account_worked(account)
                if dual:  # the slot's queue wait (and pair_delay_min) count from here
                    known[job_id]["slot_submitted_at"] = known[job_id]["submitted_at"]
                ids.append(job_id)
                self.actions.append({"action": "submit", "slot": slot, "job_id": job_id, "account": account})
                break  # F5: one job per new slot (also the half-open probe: breaker.probe_job tracks one id)
            if not ids:  # nothing accepted it: as with one account, an incident and no more slots this pass
                for account, error in errors or [(None, "no worker account left to try in this pass")]:
                    self.note("submit_failed", incident=True, slot=slot, error=error,
                              **({"account": account} if dual else {}))
                break
            for account, error in errors:  # the slot has its job on a later account; completion retries after backoff
                self.account_failed(account, slot, error)
            if probe and ids[0]:
                self.breaker["probe_job"] = ids[0]
            done += 1
        return done

    def complete_pairs(self, workers, unconfirmed, known, leaving, budget):
        """Pair completion: a slot whose candidates are all pending (not held, never run, on the fleet partition with
        the fleet walltime, none leaving, accounts among worker_accounts and pairwise different, no unconfirmed job
        in the slot) and that has queued pair_delay_min since its first submission (F5; slot_submitted_at, else
        submitted_at, else first seen) gets a candidate on each worker account it lacks, same name and argv. Its
        queued job keeps its queue position: this is how a new slot's job (after pair_delay_min) and workers queued
        on one account (before worker_accounts was set, or when the other sbatch failed) join the race. Never more
        candidates than accounts; `submit_account` guards against Slurm reporting an account name we did not ask
        for. At most `budget` slots per pass (what new slots left of max_submits_per_pass); nothing with target 0,
        the STOP flag or a breaker that is not closed (the half-open probe stays one job)."""
        cfg, now = self.cfg, self.now
        accounts = worker_accounts(cfg)
        if len(accounts) < 2 or budget <= 0 or cfg["target"] <= 0 or self.fleet.stop_flag.exists() \
                or self.breaker["open"]:
            return
        ghosts = {slot_key(cfg, known[job_id]) for job_id in unconfirmed if known[job_id].get("name")}
        for slot, jobs in sorted(by_slot(cfg, workers.values()).items()):
            if slot in ghosts or slot_of(cfg, jobs[0]["name"]) is None or len(jobs) >= len(accounts):
                continue
            records = [known.get(job["job_id"]) or {} for job in jobs]
            if any(job["state"] not in PENDING or held(job) or job["job_id"] in leaving
                   or job.get("partition") != cfg["partition"]
                   or duration(job.get("time_left")) != duration(cfg["time"])  # a pending job's time left = its limit
                   or job.get("account") not in accounts or record.get("run_start") is not None
                   for job, record in zip(jobs, records)) or len({job.get("account") for job in jobs}) < len(jobs):
                continue
            queued = min(record.get("slot_submitted_at") or record.get("submitted_at") or record.get("first_seen")
                         or now for record in records)
            if now - queued < cfg["pair_delay_min"] * 60:  # F5: a fast queue starts the slot's job alone
                continue
            have = {job.get("account") for job in jobs} | {record.get("submit_account") for record in records}
            missing = [account for account in accounts if account not in have][:len(accounts) - len(jobs)]
            missing = [account for account in missing if account in self.usable_accounts(fallback=False)]
            if not missing:
                continue
            first = min(jobs, key=lambda job: int(job["job_id"]))
            submitted = [value for value in (record.get("slot_submitted_at") or record.get("submitted_at")
                                             for record in records) if value]
            since = min(submitted) if submitted else None  # the slot's queue wait counts from its first candidate
            reason = "pair completion: sibling of %s (%s)" % (first["job_id"], first.get("account"))
            completed = False
            for account in missing:
                if self.dry_run:
                    self.note("submit", slot=slot, name=first["name"], account=account, reason=reason,
                              argv=self.fleet.sbatch_argv(cfg, first["name"], account=account))
                    completed = True
                    continue
                try:
                    job_id = self.fleet.submit_worker(cfg, slot, known, reason=reason, account=account,
                                                      slot_submitted_at=since)
                except SlurmError as error:  # e.g. the account lost access: backoff, incident every 30 min
                    self.account_failed(account, slot, str(error))
                    continue
                self.account_worked(account)
                completed = True
                self.actions.append({"action": "submit", "slot": slot, "job_id": job_id, "account": account})
            budget -= completed
            if budget <= 0 or not self.usable_accounts(fallback=False):
                return


# ---- commands --------------------------------------------------------------------------------------------------------

def short_account(name):
    """'agents_dev' for 'nemotron_agents_dev' (the status table's ACCOUNT column)."""
    name = name or "-"
    return name[len("nemotron_"):] if name.startswith("nemotron_") else name


def print_status(view):
    counts, cfg = view["counts"], view["config"]
    print("student fleet %s: target %d, horizon %d min (base %d, +0-%d stagger), max_jobs %d | coverage %d/%d (%s) "
          "| routable %d" % (
        view["time"], cfg["target"], cfg["horizon_effective_min"], cfg["horizon_min"], cfg.get("stagger_min", 0),
        cfg["max_jobs"],
        counts.get("coverage", 0), cfg["target"], ",".join(cfg["coverage_partitions"] or ["all partitions"]),
        counts.get("routable", 0)))
    daemon = view.get("daemon") or {}
    if not daemon:
        line = "daemon: never started"
    elif daemon.get("stopped_at"):
        line = "daemon: stopped %s (%s)" % (iso(daemon["stopped_at"]), daemon.get("stop_reason"))
    else:
        line = "daemon: job %s on %s pid %s, heartbeat %ds ago" % (
            daemon.get("job_id") or "-", daemon.get("host"), daemon.get("pid"), view["ts"] - daemon["heartbeat_at"])
    breaker = view["breaker"]
    line += " | breaker %s" % ("OPEN since %s (%s)" % (iso(breaker.get("since")), breaker.get("reason"))
                               if breaker.get("open") else "closed (%d recent failures)" % len(breaker["failures"]))
    line += " | %d excluded nodes" % len(view["exclude_nodes"])
    if view.get("stop_flag"):
        line += " | STOP flag set"
    if view.get("last_reconcile_at"):
        line += " | last pass %ds ago" % (view["ts"] - view["last_reconcile_at"])
    accounts = cfg.get("worker_accounts") or [cfg.get("account") or "?"]
    line += " | worker accounts %s" % "+".join(short_account(name) for name in accounts)
    if cfg.get("pair_delay_min") is not None:
        line += " (sibling after %d min)" % cfg["pair_delay_min"]
    if "last_pass_worker_accounts" in view and view["last_pass_worker_accounts"] != accounts:
        # the daemon's own view (its last state/status.json), when it differs from fleet.json as read here
        last = view["last_pass_worker_accounts"]
        line += " (daemon's last pass: %s)" % ("+".join(short_account(name) for name in last) if last else
                                               "no worker accounts, a fleet.py from before 2026-11-02")
    print(line)
    template = "%-9s %-4s %-10s %-9s %-12s %8s %8s %-9s %-8s %5s  %s"
    print(template % ("JOBID", "SLOT", "ACCOUNT", "STATE", "NODE", "ELAPSED", "LEFT", "REGISTRY", "READY", "HB",
                      "BASE_URL / REASON"))
    for row in view["jobs"]:
        detail = row["base_url"] or ""
        if row["state"] in PENDING:
            detail = "(%s)%s" % (row["reason"], " est. start %s" % iso(row["start"]) if row["start"] else "")
        elif row["registry"] in ("failed", "unhealthy", "stopped") and row.get("registry_reason"):
            detail = row["registry_reason"]
        print(template % (
            row["job_id"], row["slot"] or "?", short_account(row.get("account"))[:10], row["state"][:9],
            row["node"] or "-", hms(row["elapsed_s"]), hms(row["left_s"]),
            (row["routing"] or row["registry"] or "-")[:9],
            time.strftime("%H:%M:%S", time.localtime(row["ready_at"])) if row["ready_at"] else "-",
            "%ds" % row["heartbeat_age_s"] if row["heartbeat_age_s"] is not None else "-", detail))
    order = ("ready", "draining", "starting", "pending", "held", "stale", "unhealthy", "failed", "stopped", "vanished",
             "other")
    print("counts: " + ", ".join("%s %d" % (key, counts.get(key, 0)) for key in order) +
          " | total %d | pending slots %d" % (counts.get("total", 0), counts.get("pending_slots", 0)))
    if view["pending_reasons"]:
        print("pending reasons: " + ", ".join("%s %d" % item for item in sorted(view["pending_reasons"].items())))


def command_status(fleet, args):
    cfg = fleet.config(create=False)
    now = fleet.clock()
    queue = fleet.slurm.squeue(cfg["user"])
    known = read_json(fleet.state / "jobs.json", {}) or {}
    queued = {job["job_id"] for job in queue}
    view = fleet.snapshot(cfg, queue, read_registry(fleet.registry), now, known,
                          [job_id for job_id in known if job_id not in queued])
    # what the daemon's last completed pass used (None: a fleet.py without worker accounts wrote status.json)
    last = read_json(fleet.state / "status.json") or {}
    if isinstance(last.get("config"), dict):
        view["last_pass_worker_accounts"] = last["config"].get("worker_accounts")
    if args.json:
        print(json.dumps(view, indent=2, sort_keys=True))
    else:
        print_status(view)
    return 0


def command_reconcile(fleet, args):
    actions = fleet.reconcile(dry_run=args.dry_run)
    if args.dry_run:
        print("dry run (nothing submitted, cancelled or written):")
        for action in actions:
            argv = action.pop("argv", None)
            print("  " + json.dumps(action, sort_keys=True))
            if argv:
                print("    " + shlex.join(argv))
        if not actions:
            print("  no action")
    return 0


def command_submit(fleet, args):
    slot = "%02d" % args.slot
    probe = args.dry_run or args.test_only
    cfg = fleet.config(create=not probe)
    name = cfg["job_prefix"] + slot
    account = getattr(args, "account", None) or worker_accounts(cfg)[0]
    if not account or any(char.isspace() or char in ",|" for char in account):
        raise ConfigError("--account must be an account name")
    argv = fleet.sbatch_argv(cfg, name, args.partition, args.time, account)
    print(shlex.join(argv))
    print(json.dumps(argv))
    taken = [job["job_id"] for job in fleet.slurm.squeue(cfg["user"]) if is_worker(cfg, job) and job["name"] == name]
    if taken:
        print("slot %s is already used by job %s" % (slot, ",".join(taken)))
        if not probe:
            return 1
    if args.dry_run:
        return 0
    if args.test_only:
        print(fleet.slurm.sbatch_test(argv))
        return 0
    with fleet.locked():
        if fleet.stop_flag.exists():
            print("note: the STOP flag is set (no daemon); submitting this one worker anyway")
        known = read_json(fleet.state / "jobs.json", {}) or {}
        job_id = fleet.submit_worker(cfg, slot, known, args.partition, args.time, reason="manual submit",
                                     account=account)
        write_json(fleet.state / "jobs.json", known)
    print(job_id)
    return 0


def command_start_daemon(fleet, args):
    with fleet.locked():
        cfg = fleet.config()
        current = os.environ.get("SLURM_JOB_ID")
        if args.successor and fleet.stop_flag.exists():
            print("STOP flag set: no successor")
            return 0
        jobs = [job for job in fleet.slurm.squeue(cfg["user"])
                if job["name"] == cfg["daemon_job_name"] and job["state"] in RUNNING + PENDING
                and not (args.successor and job["job_id"] == current)]
        if jobs:
            print("daemon job already queued/running: %s" % ", ".join(
                "%s (%s)" % (job["job_id"], job["state"]) for job in jobs))
            return 0
        if fleet.stop_flag.exists():
            fleet.stop_flag.unlink()
        fleet.logs.mkdir(parents=True, exist_ok=True)
        argv = ["sbatch", "--parsable", "--export=NONE", "--job-name=" + cfg["daemon_job_name"],
                "--account=" + cfg["account"], "--partition=" + (args.partition or cfg["daemon_partition"]),
                "--time=" + (args.time or cfg["daemon_time"]), str(fleet.fleetd_sbatch)]
        job_id = fleet.slurm.sbatch(argv)
        fleet.event("daemon_submitted", job_id=job_id, partition=args.partition or cfg["daemon_partition"],
                    successor_of=current if args.successor else None)
    print(job_id)
    return 0


def command_stop(fleet, args):
    cfg = fleet.config(create=False)
    if args.yes:  # before listing: a running daemon must not resubmit what is being cancelled
        fleet.state.mkdir(parents=True, exist_ok=True)
        fleet.stop_flag.write_text("set by fleet.py stop at %s\n" % iso(time.time()))
    if not args.yes:
        targets = [job for job in fleet.slurm.squeue(cfg["user"]) if is_fleet_job(cfg, job)]
        for job in targets:
            print("%s %-26s %-10s %s" % (job["job_id"], job["name"], job["state"], job["node"] or "-"))
        print("%d fleet job(s); nothing cancelled (re-run with --yes to set the STOP flag and cancel them)"
              % len(targets))
        return 0
    with fleet.locked():
        targets = [job for job in fleet.slurm.squeue(cfg["user"]) if is_fleet_job(cfg, job)]
        for job in targets:
            print("%s %-26s %-10s %s" % (job["job_id"], job["name"], job["state"], job["node"] or "-"))
        daemons = [job["job_id"] for job in targets if job["name"] == cfg["daemon_job_name"]]
        workers = [job["job_id"] for job in targets if job["job_id"] not in daemons]
        for ids in (daemons, workers):  # the daemon first
            if ids:
                fleet.slurm.scancel(cfg["user"], ids)
        fleet.event("stop", daemons=daemons, workers=workers)
    print("cancelled %d daemon and %d worker job(s); STOP flag set (fleet.py start-daemon clears it)"
          % (len(daemons), len(workers)))
    return 0


def command_unexclude(fleet, args):
    with fleet.locked():
        stored = read_json(fleet.config_path)
        if not isinstance(stored, dict):
            raise ConfigError("unreadable %s" % fleet.config_path)
        before = list(stored.get("exclude_nodes", DEFAULTS["exclude_nodes"]))
        stored["exclude_nodes"] = [node for node in before if node not in args.nodes]
        write_json(fleet.config_path, stored)
        fleet.event("unexclude", nodes=[node for node in args.nodes if node in before])
    return 0


def command_worker_accounts(fleet, args):
    """Show or set fleet.json worker_accounts (read-modify-write under the pass lock, like unexclude; the daemon
    reads it on its next pass). --clear = [] = single-account on `account`. Candidates already queued are left
    alone: with fewer accounts, existing pairs still settle and no pair is completed."""
    if not args.accounts and not args.clear:
        cfg = fleet.config(create=False)
        print("worker_accounts %s -> candidates on %s%s" % (
            json.dumps(cfg.get("worker_accounts") or []), ", ".join(worker_accounts(cfg)),
            " (a new slot's siblings after pair_delay_min %d)" % cfg["pair_delay_min"]
            if len(worker_accounts(cfg)) > 1 else ""))
        return 0
    if args.accounts and args.clear:
        raise ConfigError("give accounts or --clear, not both")
    with fleet.locked():
        stored = read_json(fleet.config_path)
        if not isinstance(stored, dict):
            raise ConfigError("unreadable %s" % fleet.config_path)
        before = stored.get("worker_accounts", [])
        stored["worker_accounts"] = [] if args.clear else list(args.accounts)
        check_config(dict(json.loads(json.dumps(DEFAULTS)), **stored))
        write_json(fleet.config_path, stored)
        fleet.event("worker_accounts", before=before, after=stored["worker_accounts"])
    print("worker_accounts %s -> %s" % (json.dumps(before), json.dumps(stored["worker_accounts"])))
    return 0


def command_reset_breaker(fleet, args):
    with fleet.locked():
        previous = fleet.breaker()
        write_json(fleet.state / "breaker.json", {"open": False, "failures": [], "noted": {}, "reset_at": time.time()})
        fleet.event("breaker_reset", previous=previous)
    return 0


def command_daemon(fleet, args):
    return fleet.daemon(args.interval, args.until, args.wait_lock)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="fleet table (read-only)")
    status.add_argument("--json", action="store_true")
    reconcile = commands.add_parser("reconcile", help="one maintenance pass")
    reconcile.add_argument("--dry-run", action="store_true")
    daemon = commands.add_parser("daemon", help="reconcile in a loop (single instance)")
    daemon.add_argument("--interval", type=int, default=60)
    daemon.add_argument("--until", type=float, default=None, help="exit cleanly at this unix time")
    daemon.add_argument("--wait-lock", action="store_true", help="wait while another daemon holds the lock")
    submit = commands.add_parser("submit", help="submit one worker into a slot")
    submit.add_argument("--slot", type=int, required=True, choices=range(100), metavar="NN")
    submit.add_argument("--partition")
    submit.add_argument("--time", help="walltime (default: fleet.json time)")
    submit.add_argument("--account", help="this one job's account (default: the first worker account)")
    mode = submit.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the sbatch argv only")
    mode.add_argument("--test-only", action="store_true", help="sbatch --test-only (validates, submits nothing)")
    stop = commands.add_parser("stop", help="cancel every fleet job and the daemon (STOP flag)")
    stop.add_argument("--yes", action="store_true")
    start = commands.add_parser("start-daemon", help="submit fleetd.sbatch unless a daemon job is queued/running")
    start.add_argument("--successor", action="store_true", help="(fleetd.sbatch) queue the next daemon job")
    start.add_argument("--partition", help="this submission only (default: fleet.json daemon_partition)")
    start.add_argument("--time", help="this submission only (default: fleet.json daemon_time)")
    unexclude = commands.add_parser("unexclude", help="remove nodes from exclude_nodes")
    unexclude.add_argument("nodes", nargs="+")
    commands.add_parser("reset-breaker", help="close the circuit breaker")
    accounts = commands.add_parser("worker-accounts", help="show or set the accounts every slot gets a candidate on")
    accounts.add_argument("accounts", nargs="*", metavar="ACCOUNT")
    accounts.add_argument("--clear", action="store_true", help="[] = single-account on fleet.json account")
    args = parser.parse_args(argv)
    handler = {"status": command_status, "reconcile": command_reconcile, "daemon": command_daemon,
               "submit": command_submit, "stop": command_stop, "start-daemon": command_start_daemon,
               "unexclude": command_unexclude, "reset-breaker": command_reset_breaker,
               "worker-accounts": command_worker_accounts}[args.command]
    try:
        return handler(Fleet(), args)
    except (SlurmError, ConfigError, LockBusy, RegistryError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
