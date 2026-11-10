#!/usr/bin/env python3
"""official3 queue runner (hengyuf, 2026-10-26; a copy of synth/queue2/run_queue.py): build the group-A DeepSWE targets
(queue.json: the 21 unused targets with sc14 2-3/5 and kimi >= 3/5, the "reliability gap"), at most max_active at a
time, round 0 + round 1 (--rounds 1). These targets have no earlier failure analysis: an entry without "analysis_from"
skips the prep copy and task_builder runs the failure analyzer itself. Every Slurm job goes through
hengyuf's dual-account helper (schdule_helper.sh): the builds (synth/run_babysit.sbatch) and the student fleet
daemon (student/fleetd_babysit.sbatch). The GPU workers the daemon submits stay single-account (hengyuf, 2026-10-22).

Where it runs: as its own small CPU Slurm job (queue3/runner.sbatch, submitted through the helper with --export=NONE,
so it has the user's login environment, as run.sbatch builds had), or in a login-node shell (e.g. tmux). Commands:
    /usr/bin/python3 -I -B /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth/queue3/run_queue.py [run]
    ... run_queue.py status        # table of the queue (read-only)
    ... run_queue.py --dry-run     # one pass: print what it would do, change nothing
Only one runner may be live: `run` takes queue3/runner.lock (flock) and refuses to start while another runner's
heartbeat (queue3/heartbeat.json) is younger than HEARTBEAT_STALE_S (--force skips the heartbeat check). It is
restart-safe: the helper's per-run state (synth/runs/<run>/babysit/) is authoritative, state.json keeps the rest, and
launch.jsonl (read by the dashboard) is brought in line with the helper's current leg on every pass. Stopping the
runner never stops a job. A run parked as "stuck" or "failed" stays parked until a person edits state.json.

What one pass does (one squeue snapshot per pass; every part is isolated, so one failure never blocks the rest):
1. Prep (once per run): copy the earlier failure analysis into synth/runs/<run>/analysis/<task>/.
   task_builder reuses it, because analyze_target skips a result.json that validates.
2. Hub budget, every settings.budget_every_s, via tools/hub_budget.py (names only, never a key value). A key counts as
   low only after two low readings at least BUDGET_CONFIRM_S apart (the hub glitch can report ~2x the real spend
   for a minute). Low active key: switch synth/hub_key_active.txt to the next name in key_order if that one is above
   its threshold (running probes keep the key they started with); with no usable key, launch nothing new.
   stop_launching is recomputed on every check, so it clears when the readings recover.
3. Builds, in queue order:
   - done: the build root has summary.json;
   - a leg the helper left incomplete (a candidate's sbatch failed or its answer was lost, or the release did not
     finish) is completed by calling babysit_submit again; the helper never duplicates a submitted candidate. If it
     reports "submission outcome unknown" and squeue shows no job with that candidate's comment twice, a minute
     apart, the stale request is moved aside and the call is retried. After MAX_REPAIRS failures the run is
     parked as "stuck" (its held candidates are cancelled);
   - once a leg has a winner, a still-pending sibling is cancelled (the helper's own cancel can fail); released
     candidates that are still held are released again; held jobs of the run's name that are not in its current leg
     (orphans) are cancelled;
   - active (a candidate pending or running) counts toward max_active;
   - ended without summary.json: resumed with the same command (every stage is idempotent), up to
     max_launches_per_run legs in all; resumes go before new runs;
   - new runs start while fewer than max_active are active.
4. Serving: from the first launch on, fleet.json target := fleet_target while any run is pending or running, else 0
   (worker queue waits reach hours, so serving starts before the first probe). A daemon is kept in the queue through
   the helper unless student/state/STOP exists, at most DAEMON_LEGS_PER_WINDOW new daemon legs per
   DAEMON_WINDOW_S. The daemon job leases the target: fleetd_babysit.sbatch sets it to 0 when this runner's
   heartbeat is over an hour old.
5. When every run is done, failed or stuck: fleet target := 0, then `fleet.py stop --yes` (cancels the daemon and the
   workers, sets the STOP flag), heartbeat "stopped".
"""
import argparse
import fcntl
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
B = Path("/lustre/fsw/portfolios/nemotron/users/baiheh/swe-task-hengyu")
Q = S / "synth/queue3"
RUNS = S / "synth/runs"
HELPER = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh")
BUILD_SBATCH = S / "synth/run_babysit.sbatch"
FLEETD_SBATCH = S / "student/fleetd_babysit.sbatch"
FLEET_PY = S / "student/fleet.py"
FLEET_JSON = S / "student/fleet.json"
FLEETD_RESULTS = S / "student/state/fleetd-babysit"
STUDENT_STOP = S / "student/state/STOP"
ACTIVE_KEY = S / "synth/hub_key_active.txt"
SLURM_CONF = "/cm/shared/apps/slurm/var/etc/cw-dfw-cs-001/slurm.conf"
SLURM_BIN = "/cm/shared/apps/slurm/current/bin"
PROBE_ACCOUNT = "nemotron_agents_dev"  # job_submit.lua rejects a job (and a --test-only probe) without an account
ACCOUNTS = ["nemotron_agents_dev", "nemotron_rl_algo"]  # the helper's candidate order
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "RESIZING",
          "SUSPENDED", "STAGE_OUT", "SIGNALING"}
HELD = "JobHeldUser"
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL")  # = llm_engine.SECRET_NAME, on upper-cased names
DROP_VARS = {"SSH_AUTH_SOCK", "SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "TMUX", "TMUX_PANE", "DISPLAY",
             "XDG_RUNTIME_DIR", "XDG_SESSION_ID", "SYNTH_STUDENT_BASE_URL", "OLDPWD", "PWD", "SHLVL", "_", "HOSTNAME",
             "TMPDIR", "TMP", "TEMP", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"}
DROP_PREFIXES = ("VSCODE_", "CLAUDE", "ANTHROPIC", "TERM_PROGRAM", "BABYSIT_", "SLURM_", "SBATCH_", "SRUN_", "SALLOC_",
                 "PMI_", "PMIX_", "OMPI_")
USER = os.environ.get("USER") or os.environ.get("LOGNAME") or os.environ.get("SLURM_JOB_USER") or "hengyuf"
HOST = socket.gethostname()
HEARTBEAT_STALE_S = 900
MAX_REPAIRS = 3
BUDGET_CONFIRM_S = 540
DAEMON_LEGS_PER_WINDOW = 3
DAEMON_WINDOW_S = 6 * 3600
NO_JOB = ("", "", "")
DRY = False
ENV = {}
LOCK = None  # keeps the flock'ed file object alive for the process lifetime


def job_env():
    """The environment of every subprocess and, through the helper's --export=ALL, of every job it submits: this
    process's environment without secret-like names, session- and node-bound names and Slurm job settings (a nested
    sbatch must not inherit the runner job's SLURM_*/SBATCH_*), plus SLURM_CONF and the Slurm bin dir at the END of
    PATH (the login PATH keeps its order). Run as a Slurm job (--export=NONE), this process has the user's login
    environment, which is what run.sbatch builds got."""
    env = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if SECRET_NAME.search(upper) or key in DROP_VARS or upper.startswith(DROP_PREFIXES):
            continue
        env[key] = value
    parts = [p for p in (env.get("PATH") or "").split(":") if p]
    for extra in (SLURM_BIN, "/usr/local/bin", "/usr/bin", "/bin"):
        if extra not in parts:
            parts.append(extra)
    env.update(PATH=":".join(parts), SLURM_CONF=SLURM_CONF, USER=USER, LOGNAME=USER)
    env.setdefault("HOME", f"/home/{USER}")
    env.setdefault("LANG", "C.UTF-8")
    return env


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(message):
    line = f"[{now_iso()}] {message}"
    print(line, flush=True)
    if not DRY:
        with (Q / "runner.log").open("a") as stream:
            stream.write(line + "\n")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json_atomic(path, value):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n")
    tmp.replace(path)


def write_text_atomic(path, text):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def read_text(path, default=""):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def run(argv, *, cwd=None, check=True, timeout=600):
    result = subprocess.run(argv, env=ENV, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise RuntimeError(f"{argv[0]} failed ({result.returncode}): {(result.stderr or result.stdout).strip()[:400]}")
    return result


def resolve(spec):
    if spec.startswith("S/"):
        return S / spec[2:]
    if spec.startswith("B/"):
        return B / spec[2:]
    return Path(spec)


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


# ------------------------------------------------------------------------------------------------ Slurm
def my_jobs():
    """{job id: (name, state, reason)} of all of this user's jobs (one squeue call; raises if squeue fails, so a
    failed query never reads as "nothing is queued")."""
    result = run(["squeue", "--noheader", f"--user={USER}", "--format=%i|%T|%r|%j"], check=not DRY, timeout=120)
    if result.returncode != 0:  # dry run without Slurm access (e.g. main's sandbox): as if nothing were queued
        log(f"DRY: squeue unavailable ({(result.stderr or result.stdout).strip()[:120]}); assuming an empty queue")
        return {}
    jobs = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split("|", 3)
        if len(parts) == 4 and parts[0].isdigit():
            jobs[parts[0]] = (parts[3], parts[1], parts[2])
    return jobs


def jobs_with_comment(name, comment):
    result = run(["squeue", "--noheader", f"--user={USER}", f"--name={name}", "--format=%i|%k"], timeout=120)
    return [line.split("|", 1)[0].strip() for line in result.stdout.splitlines()
            if "|" in line and line.split("|", 1)[1].strip() == comment]


def scancel(ids, why):
    ids = [i for i in ids if str(i).isdigit()]
    if not ids:
        return
    log(f"scancel {','.join(ids)}: {why}")
    if not DRY:
        run(["scancel", *ids], check=False)


def pick_partition(job_args, partitions):
    result = run(["/usr/bin/python3", "-I", "-B", str(S / "tools/pick_partition.py"), "--partitions", partitions,
                  "--", f"--account={PROBE_ACCOUNT}", *job_args], cwd=S / "synth", timeout=900)
    pick = json.loads(result.stdout.strip().splitlines()[-1])
    return pick["partition"], pick


def babysit_submit(results_dir, batch_script, exp_name, sbatch_args):
    """The helper's babysit_submit (sourced in bash, environment ENV + its three variables); returns the leg's two
    candidate ids. Called for a new leg and, idempotently, to complete an incomplete one."""
    env = dict(ENV, RESULTS_DIR=str(results_dir), BATCH_SCRIPT=str(batch_script), EXP_NAME=exp_name)
    script = 'source "$0" && babysit_submit "$@"'
    result = subprocess.run(["bash", "-c", script, str(HELPER), *sbatch_args], env=env, capture_output=True,
                            text=True, timeout=900)
    for line in result.stderr.strip().splitlines()[-4:]:
        log(f"  {line}")
    if result.returncode != 0:
        raise RuntimeError(f"babysit_submit failed ({result.returncode}): {result.stderr.strip()[-300:]}")
    ids = result.stdout.strip().splitlines()[-1].split(",") if result.stdout.strip() else []
    if len(ids) != 2 or not all(i.isdigit() for i in ids):
        raise RuntimeError(f"unexpected babysit_submit output: {result.stdout.strip()[-200:]}")
    return ids


def leg_info(results_dir):
    """The helper's current leg under <results_dir>/babysit/: number, candidate ids, winner, and whether the leg is
    complete (both candidates recorded and released)."""
    state = Path(results_dir) / "babysit"
    current = read_text(state / "current", "0")
    leg = int(current) if current.isdigit() else 0
    if leg == 0:
        return {"leg": 0, "ids": [], "winner": None, "complete": True, "dir": None}
    legs = state / "legs" / str(leg)
    ids = [read_text(legs / f"job.{i}") for i in (0, 1)]
    complete = all(i.isdigit() for i in ids) and (legs / "released").is_file()
    return {"leg": leg, "ids": [i for i in ids if i.isdigit()], "winner": read_text(legs / "winner") or None,
            "complete": complete, "dir": legs}


def complete_leg(results_dir, batch_script, exp_name, sbatch_args, label):
    """Finish a leg the helper left incomplete (raises when it cannot)."""
    info = leg_info(results_dir)
    try:
        babysit_submit(results_dir, batch_script, exp_name, sbatch_args)
        log(f"{label}: completed leg {info['leg']}")
        return
    except RuntimeError as error:
        if "submission outcome unknown" not in str(error):
            raise
    # the helper found no held job carrying a recorded request's comment: confirm twice, then forget the request
    stale = []
    for index in (0, 1):
        request, job = info["dir"] / f"request.{index}", info["dir"] / f"job.{index}"
        if request.is_file() and not job.is_file():
            comment = read_text(request)
            if not jobs_with_comment(exp_name, comment):
                stale.append(request)
    if not stale:
        raise RuntimeError(f"{label}: leg {info['leg']} incomplete and the helper refuses; inspect {info['dir']}")
    time.sleep(60)
    for request in stale:
        if jobs_with_comment(exp_name, read_text(request)):
            raise RuntimeError(f"{label}: a job with comment {read_text(request)} appeared; inspect {info['dir']}")
    for request in stale:
        aside = request.with_name(f"{request.name}.failed-{int(time.time())}")
        request.rename(aside)
        log(f"{label}: no job carries {read_text(aside)}: moved {request.name} aside ({aside.name}) and retrying")
    babysit_submit(results_dir, batch_script, exp_name, sbatch_args)
    log(f"{label}: completed leg {info['leg']} after resubmitting the missing candidate")


def settle_candidates(label, name, info, jobs):
    """Cancel a pending sibling once a winner exists; release released-but-held candidates again; cancel held jobs
    of this name outside the current leg (e.g. a "lost" candidate that appeared after the leg was completed)."""
    if info["winner"]:
        losers = [job for job in info["ids"] if job != info["winner"] and jobs.get(job, NO_JOB)[1] == "PENDING"]
        scancel(losers, f"{label}: pending sibling of winner {info['winner']}")
    held = [job for job in info["ids"] if jobs.get(job, NO_JOB)[1] == "PENDING" and jobs[job][2] == HELD]
    if held and info["complete"]:
        log(f"{label}: released candidates still held ({','.join(held)}): releasing again")
        if not DRY:
            run(["scontrol", "release", ",".join(held)], check=False)
    orphans = [job for job, (job_name, job_state, reason) in jobs.items()
               if job_name == name and job_state == "PENDING" and reason == HELD and job not in info["ids"]]
    scancel(orphans, f"{label}: held job(s) outside the current leg {info['leg']}")


def leg_active(info, jobs):
    return any(jobs.get(job, NO_JOB)[1] in ACTIVE for job in info["ids"])


# ------------------------------------------------------------------------------------------------ queue
def build_root(entry):
    return RUNS / entry["run"] / "build" / entry["task"] / entry["repo"].replace("/", "__", 1)


def build_name(entry):
    return f"synth-build-{entry['run']}"


def build_extra(entry, settings):
    return ["--repo", entry["repo"], *settings["build_args"]]


def prep(entry):
    """Copy the earlier failure analysis into the new run (once). official3: an entry without "analysis_from" has none
    to copy; task_builder's analyze stage then writes analysis/<task>/ itself."""
    if not entry.get("analysis_from"):
        return
    dst = RUNS / entry["run"] / "analysis" / entry["task"]
    if (dst / "result.json").is_file():
        return
    src = resolve(entry["analysis_from"]) / "analysis" / entry["task"]
    if not (src / "result.json").is_file():
        raise RuntimeError(f"{entry['run']}: no failure analysis at {src}")
    log(f"{entry['run']}: copy failure analysis {src} -> {dst}")
    if DRY:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.copy")
    if tmp.exists():
        shutil.rmtree(tmp)
    skipped = []

    def copy_readable(source, target, *, follow_symlinks=True):
        try:
            return shutil.copy2(source, target, follow_symlinks=follow_symlinks)
        except PermissionError:  # e.g. an owner-only file in Baihe's tree; the pipeline needs none of them
            skipped.append(str(source))
            return target

    # The pipeline reads failure_analysis.md, result.json and inputs/ (stats, rollouts); scratch/ (the analyst's
    # work files) and bin/ (sandbox wrappers of the old analysis sandbox) are not needed.
    shutil.copytree(src, tmp, symlinks=False, ignore=shutil.ignore_patterns("scratch", "bin"),
                    copy_function=copy_readable)
    missing = [name for name in ("result.json", "failure_analysis.md", "inputs/stats.json") if not (tmp / name).is_file()]
    if missing:
        raise RuntimeError(f"{entry['run']}: copied analysis lacks {missing} (unreadable: {skipped[:5]})")
    tmp.rename(dst)
    if skipped:
        log(f"{entry['run']}: skipped {len(skipped)} unreadable file(s) of the old analysis, e.g. {skipped[0]}")


def launch_lines(entry):
    path = RUNS / entry["run"] / "launch.jsonl"
    lines = []
    if path.is_file():
        for raw in path.read_text().splitlines():
            try:
                value = json.loads(raw)
            except ValueError:
                continue  # a torn line never blocks the runner
            if isinstance(value, dict):
                lines.append(value)
    return lines


def append_launch(entry, line):
    path = RUNS / entry["run"] / "launch.jsonl"
    if DRY:
        log(f"  DRY launch.jsonl += {json.dumps(line)[:300]}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(line) + "\n")


def launch_line(entry, settings, info, partition, pick, reconstructed=False):
    babysit = {"leg": info["leg"], "candidates": info["ids"], "accounts": ACCOUNTS, "resolved": False}
    if reconstructed:
        babysit["reconstructed"] = True
    return {"submitted_at": now_iso(), "job_id": info["ids"][0], "run": entry["run"], "task": entry["task"],
            "time": settings["walltime"], "partition": partition, "extra": " ".join(build_extra(entry, settings)),
            "log": str(RUNS / "slurm" / f"{build_name(entry)}-{info['ids'][0]}.out"), "partition_probe": pick,
            "babysit": babysit}


def sync_launch(entry, settings, info):
    """launch.jsonl must name the current leg's job: append a line for a leg that has none (completed in a later
    pass, or the runner was killed between submission and writing), then the resolved line once there is a
    winner."""
    if not info["leg"] or not info["ids"]:
        return
    lines = launch_lines(entry)
    last = lines[-1] if lines else {}
    babysit = last.get("babysit") or {}
    if babysit.get("leg") != info["leg"]:
        args = read_json(RUNS / entry["run"] / "babysit/sbatch_args.json") or []
        partition = next((a.split("=", 1)[1] for a in args if str(a).startswith("--partition=")), None)
        last = launch_line(entry, settings, info, partition, None, reconstructed=True)
        append_launch(entry, last)
        log(f"{entry['run']}: launch.jsonl line added for leg {info['leg']} ({','.join(info['ids'])})")
        babysit = last["babysit"]
    if info["winner"] and not babysit.get("resolved"):
        line = dict(last, job_id=info["winner"],
                    log=str(RUNS / "slurm" / f"{build_name(entry)}-{info['winner']}.out"))
        line["babysit"] = dict(babysit, resolved=True, winner=info["winner"])
        append_launch(entry, line)
        log(f"{entry['run']}: leg {info['leg']} winner {info['winner']}")


def launch_build(entry, settings, leg_before):
    run_dir = RUNS / entry["run"]
    name = build_name(entry)
    argv = ["task_builder.py", "build", "--run", entry["run"], "--task", entry["task"], *build_extra(entry, settings)]
    job_args = [f"--job-name={name}", f"--time={settings['walltime']}", str(BUILD_SBATCH)]
    if DRY:
        log(f"{entry['run']}: DRY would launch (leg {leg_before + 1}): {' '.join(argv)}")
        return
    (run_dir / "babysit").mkdir(parents=True, exist_ok=True)
    write_text_atomic(run_dir / "babysit/argv", "\0".join(argv) + "\0")
    partition, pick = pick_partition(job_args, settings["partitions"])
    args = [f"--partition={partition}", f"--time={settings['walltime']}"]
    write_text_atomic(run_dir / "babysit/sbatch_args.json", json.dumps(args))
    ids = babysit_submit(run_dir, BUILD_SBATCH, name, args)
    info = leg_info(run_dir)
    if info["leg"] <= leg_before:  # the helper found the previous leg still active: nothing new was submitted
        log(f"{entry['run']}: the helper kept leg {info['leg']} (still active): {','.join(ids)}")
        return
    append_launch(entry, launch_line(entry, settings, info, partition, pick))
    log(f"{entry['run']}: launched leg {info['leg']} on {partition}: candidates {ids[0]},{ids[1]}")


# ------------------------------------------------------------------------------------------------ serving
def set_fleet_target(target):
    fleet = read_json(FLEET_JSON)
    if not isinstance(fleet, dict):
        raise RuntimeError(f"{FLEET_JSON} unreadable")
    if fleet.get("target") == target:
        return
    log(f"fleet.json target {fleet.get('target')} -> {target}")
    if DRY:
        return
    backup = FLEET_JSON.with_name("fleet.json.pre-queue3")
    if not backup.exists():
        shutil.copy2(FLEET_JSON, backup)
    fleet["target"] = target
    write_json_atomic(FLEET_JSON, fleet)


def fleetd_name():
    return (read_json(FLEET_JSON) or {}).get("daemon_job_name", "swe-task-student-fleetd")


def ensure_fleetd(jobs, state):
    name = fleetd_name()
    if state.get("fleetd_stuck"):
        return
    if STUDENT_STOP.exists():
        if not state.get("stop_flag_logged"):
            log(f"{STUDENT_STOP} exists: not submitting the fleet daemon (remove it to serve)")
            state["stop_flag_logged"] = True
        return
    state.pop("stop_flag_logged", None)
    fleet = read_json(FLEET_JSON) or {}
    walltime = fleet.get("daemon_time", "7-00:00:00")
    info = leg_info(FLEETD_RESULTS)
    if info["leg"] and not info["complete"]:
        args = read_json(FLEETD_RESULTS / "babysit/sbatch_args.json") or [f"--time={walltime}"]
        if DRY:
            return
        try:
            complete_leg(FLEETD_RESULTS, FLEETD_SBATCH, name, args, "fleetd")
            state.pop("fleetd_repairs", None)
        except Exception as error:  # noqa: BLE001 - counted; at MAX_REPAIRS the daemon is parked as stuck
            state["fleetd_repairs"] = state.get("fleetd_repairs", 0) + 1
            log(f"fleetd: could not complete leg {info['leg']} ({state['fleetd_repairs']}/{MAX_REPAIRS}): {error}")
            if state["fleetd_repairs"] >= MAX_REPAIRS:
                log("fleetd: STUCK (needs a person): its leg cannot be completed; clear state.json fleetd_stuck, "
                    "fleetd_repairs and daemon_legs to retry")
                state["fleetd_stuck"] = True
                now = leg_info(FLEETD_RESULTS)
                scancel([job for job in now["ids"] if jobs.get(job, NO_JOB)[1] == "PENDING" and jobs[job][2] == HELD],
                        "fleetd: held candidates of a stuck daemon leg")
        return
    if info["leg"]:
        settle_candidates("fleetd", name, info, jobs)
    if leg_active(info, jobs) or any(job_name == name and job_state in ACTIVE and reason != HELD
                                     for job_name, job_state, reason in jobs.values()):
        return
    recent = [t for t in state.get("daemon_legs", []) if time.time() - t < DAEMON_WINDOW_S]
    if len(recent) >= DAEMON_LEGS_PER_WINDOW:
        log(f"fleetd: STUCK: {len(recent)} daemon legs in {DAEMON_WINDOW_S // 3600} h and none stays up; "
            f"not submitting more (a person must look; clear state.json fleetd_stuck)")
        state["fleetd_stuck"] = True
        return
    if DRY:
        log(f"DRY would submit the fleet daemon ({name}) through the helper")
        return
    job_args = [f"--job-name={name}", f"--time={walltime}", str(FLEETD_SBATCH)]
    partition, _ = pick_partition(job_args, "cpu_datamover,cpu_dataprocessing")
    args = [f"--partition={partition}", f"--time={walltime}"]
    (FLEETD_RESULTS / "babysit").mkdir(parents=True, exist_ok=True)
    write_text_atomic(FLEETD_RESULTS / "babysit/sbatch_args.json", json.dumps(args))
    state["daemon_legs"] = recent + [time.time()]
    ids = babysit_submit(FLEETD_RESULTS, FLEETD_SBATCH, name, args)
    log(f"fleet daemon submitted on {partition}: candidates {ids[0]},{ids[1]}")


def stop_serving():
    """End of queue: no build is active, so the workers carry no request: fleet.py stop --yes cancels the daemon
    and every worker and sets the STOP flag (`fleet.py start-daemon`, or removing the flag, undoes it)."""
    set_fleet_target(0)
    if DRY:
        return
    try:
        result = run(["/usr/bin/python3", "-B", str(FLEET_PY), "stop", "--yes"], cwd=S / "student", check=False,
                     timeout=900)
        log(f"fleet.py stop --yes: rc {result.returncode}: {(result.stdout or result.stderr).strip()[-300:]}")
        if result.returncode == 0:
            return
    except Exception as error:  # noqa: BLE001 - e.g. the fleet lock was not free in time
        log(f"fleet.py stop --yes failed: {type(error).__name__}: {error}")
    prefix = (read_json(FLEET_JSON) or {}).get("job_prefix", "swe-task-student-w")
    jobs = my_jobs()
    scancel([job for job, (name, _, _) in jobs.items() if name == fleetd_name()], "end of queue: fleet daemon")
    scancel([job for job, (name, _, _) in jobs.items() if name.startswith(prefix)], "end of queue: student workers")


# ------------------------------------------------------------------------------------------------ budget
def budget(name):
    result = run(["/usr/bin/python3", "-I", "-B", str(S / "tools/hub_budget.py"), "--key-name", name],
                 cwd=S, check=False, timeout=180)
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"key": name, "remaining": None, "error": (result.stderr or result.stdout).strip()[-200:]}


def threshold(settings, name):
    order = settings["key_order"]
    last = name == order[-1] or name not in order
    return settings["stop_launching_below_usd"] if last else settings["switch_key_below_usd"]


def check_budget(settings, state):
    order = settings["key_order"]
    active = read_text(ACTIVE_KEY) or order[0]
    info = budget(active)
    remaining = number(info.get("remaining"))
    state["budget"] = {"at": now_iso(), "key": active, "remaining": remaining, "error": info.get("error")}
    history = state.setdefault("budget_history", {}).setdefault(active, [])
    history.append([time.time(), remaining])
    del history[:-3]
    log(f"budget: {active} remaining {remaining}")
    readings = [(at, value) for at, value in history if value is not None]
    if not readings:
        return  # no reading at all: keep the last decision
    limit = threshold(settings, active)
    if readings[-1][1] >= limit:
        if state.get("stop_launching"):
            log(f"budget: {active} back at {readings[-1][1]}: launching again")
        state["stop_launching"] = False
        return
    confirmed = len(readings) >= 2 and readings[-2][1] < limit and readings[-1][0] - readings[-2][0] >= BUDGET_CONFIRM_S
    if not confirmed:
        log(f"budget: {active} reads {readings[-1][1]} < {limit}; waiting for a second reading")
        return
    if active in order and order.index(active) + 1 < len(order):
        nxt = order[order.index(active) + 1]
        other = number(budget(nxt).get("remaining"))
        if other is not None and other >= threshold(settings, nxt):
            log(f"budget: switching the active hub key {active} -> {nxt} (remaining {other})")
            if not DRY:
                write_text_atomic(ACTIVE_KEY, nxt + "\n")
            state.setdefault("key_switches", []).append({"at": now_iso(), "from": active, "to": nxt})
            state["stop_launching"] = False
            return
        log(f"budget: next key {nxt} not usable (remaining {other})")
    if not state.get("stop_launching"):
        log(f"budget: no key above its threshold ({active} at {readings[-1][1]}): no new builds until one is")
    state["stop_launching"] = True


# ------------------------------------------------------------------------------------------------ main
def classify(entry, settings, info, leg, jobs, active_runs, ended, fresh):
    """One run's state for this pass (may complete an incomplete leg); appends it to one of the three lists."""
    if (build_root(entry) / "summary.json").is_file():
        if info.get("status") != "done":
            log(f"{entry['run']}: done (summary.json)")
        info["status"] = "done"
        return
    if info.get("status") in ("failed", "stuck"):
        return
    name = build_name(entry)
    if leg["leg"] and not leg["complete"]:
        if DRY:
            log(f"{entry['run']}: DRY leg {leg['leg']} incomplete; would complete it")
            active_runs.append(entry)
            return
        args = read_json(RUNS / entry["run"] / "babysit/sbatch_args.json") or [f"--time={settings['walltime']}"]
        try:
            complete_leg(RUNS / entry["run"], BUILD_SBATCH, name, args, entry["run"])
            info.pop("leg_repairs", None)
            info.pop("launch_errors", None)
        except Exception as error:  # noqa: BLE001 - counted, then the run is parked as stuck
            info["leg_repairs"] = info.get("leg_repairs", 0) + 1
            log(f"{entry['run']}: could not complete leg {leg['leg']} ({info['leg_repairs']}/{MAX_REPAIRS}): {error}")
            if info["leg_repairs"] >= MAX_REPAIRS:
                park_stuck(entry, info, leg_info(RUNS / entry["run"]), jobs, f"leg {leg['leg']} cannot be completed")
                return
        active_runs.append(entry)
        info["status"] = "pending"
        return
    if leg["leg"]:
        settle_candidates(entry["run"], name, leg, jobs)
        sync_launch(entry, settings, leg)
    if leg_active(leg, jobs):
        running = any(jobs.get(job, NO_JOB)[1] == "RUNNING" for job in leg["ids"])
        info["status"] = "running" if running else "pending"
        active_runs.append(entry)
    elif leg["leg"] == 0:
        fresh.append(entry)
    else:
        ended.append(entry)


def park_stuck(entry, info, leg, jobs, why):
    log(f"{entry['run']}: STUCK (needs a person): {why}; helper state {leg['dir']}")
    info["status"] = "stuck"
    held = [job for job in leg["ids"] if jobs.get(job, NO_JOB)[1] == "PENDING" and jobs[job][2] == HELD]
    scancel(held, f"{entry['run']}: held candidates of a stuck run")


def one_pass(queue, state):
    settings = queue["settings"]
    if time.time() - state.get("budget_checked", 0) >= settings["budget_every_s"]:
        try:
            check_budget(settings, state)
        except Exception as error:  # noqa: BLE001
            log(f"budget check failed: {type(error).__name__}: {error}")
        state["budget_checked"] = time.time()
    jobs = my_jobs()  # raises on failure: the pass is skipped, nothing is decided on a missing snapshot
    try:
        ensure_fleetd(jobs, state)
    except Exception as error:  # noqa: BLE001 - the daemon never blocks the builds
        log(f"fleetd: {type(error).__name__}: {error}")
    runs = state.setdefault("runs", {})
    active_runs, ended, fresh = [], [], []
    for entry in queue["runs"]:
        info = runs.setdefault(entry["run"], {"task": entry["task"], "repo": entry["repo"], "status": "queued"})
        try:
            leg = leg_info(RUNS / entry["run"])
            info.update(leg=leg["leg"], candidates=leg["ids"], winner=leg["winner"])
            classify(entry, settings, info, leg, jobs, active_runs, ended, fresh)
        except Exception as error:  # noqa: BLE001 - unknown state: count it as active (never over-launch)
            log(f"{entry['run']}: pass error: {type(error).__name__}: {error}")
            active_runs.append(entry)
    slots = settings["max_active"] - len(active_runs)
    # no student serving (daemon parked, or the STOP flag): a build would reach its probe with no worker, wait 3 h
    # and burn a leg, so nothing new starts until serving is back (running builds go on)
    no_serving = bool(state.get("fleetd_stuck")) or STUDENT_STOP.exists()
    if no_serving and not state.get("no_serving_logged"):
        log("no student serving (fleetd stuck or the STOP flag): no new launches or resumes until it is back")
        state["no_serving_logged"] = True
    elif not no_serving:
        state.pop("no_serving_logged", None)
    hold = bool(state.get("stop_launching")) or no_serving

    def try_launch(entry, leg_before):
        """One launch. Returns True when it used a slot: it launched, or it failed after the helper had already
        started the leg (that leg counts as active and is completed next pass). Errors are logged and counted, and a
        run that fails to launch MAX_REPAIRS times in a row is parked as stuck."""
        info = runs[entry["run"]]
        try:
            if leg_before == 0:
                prep(entry)
            launch_build(entry, settings, leg_before)
            info.pop("launch_errors", None)
            info["status"] = "pending"
            return True
        except Exception as error:  # noqa: BLE001
            info["launch_errors"] = info.get("launch_errors", 0) + 1
            log(f"{entry['run']}: launch failed ({info['launch_errors']}/{MAX_REPAIRS}): "
                f"{type(error).__name__}: {error}")
            advanced = leg_info(RUNS / entry["run"])["leg"] > leg_before
            if info["launch_errors"] >= MAX_REPAIRS and not advanced:
                park_stuck(entry, info, leg_info(RUNS / entry["run"]), jobs, "it fails to launch")
            if advanced:
                info["status"] = "pending"
            return advanced

    for entry in ended:  # resumes first: keep started builds moving
        info = runs[entry["run"]]
        if info["leg"] >= settings["max_launches_per_run"]:
            if info.get("status") != "failed":
                log(f"{entry['run']}: FAILED after {info['leg']} launches without summary.json")
            info["status"] = "failed"
            continue
        if slots <= 0 or hold:
            info["status"] = "waiting_to_resume"
            continue
        log(f"{entry['run']}: ended unfinished after leg {info['leg']}; resuming")
        if try_launch(entry, info["leg"]):
            slots -= 1
    for entry in fresh:
        info = runs[entry["run"]]
        if slots <= 0 or hold:
            info["status"] = "queued"
            continue
        if try_launch(entry, 0):
            slots -= 1
    busy = any(runs[e["run"]]["status"] in ("pending", "running") for e in queue["runs"])
    if busy and not state.get("serving"):
        log("first build launched: student serving on (worker queue waits reach hours)")
        state["serving"] = True
    if state.get("serving"):
        set_fleet_target(settings["fleet_target"] if busy else 0)
    return all(runs[e["run"]]["status"] in ("done", "failed", "stuck") for e in queue["runs"])


def heartbeat(note):
    if not DRY:
        write_json_atomic(Q / "heartbeat.json", {"at": time.time(), "at_iso": now_iso(), "host": HOST,
                                                 "pid": os.getpid(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                                                 "note": note})


def take_lock(force=False):
    """One live runner: an exclusive flock on queue3/runner.lock plus a fresh-heartbeat check (in case flock does not
    reach across nodes). The heartbeat of this same Slurm job (a requeued incarnation) never blocks it; --force skips
    the heartbeat check."""
    global LOCK
    LOCK = (Q / "runner.lock").open("a")
    try:
        fcntl.flock(LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another run_queue.py holds queue3/runner.lock: not starting")
    beat = read_json(Q / "heartbeat.json") or {}
    age = time.time() - float(beat.get("at") or 0)
    same_job = bool(os.environ.get("SLURM_JOB_ID")) and beat.get("slurm_job_id") == os.environ.get("SLURM_JOB_ID")
    if not force and not same_job and age < HEARTBEAT_STALE_S and beat.get("note") != "stopped" \
            and (beat.get("host"), beat.get("pid")) != (HOST, os.getpid()):
        sys.exit(f"another runner beat {age:.0f} s ago on {beat.get('host')} (pid {beat.get('pid')}, "
                 f"job {beat.get('slurm_job_id')}): not starting (wait {HEARTBEAT_STALE_S} s after it stops)")


def status(queue, state):
    beat = read_json(Q / "heartbeat.json") or {}
    if beat:
        print(f"runner: {beat.get('note', '-')} on {beat.get('host', '-')} pid {beat.get('pid', '-')} job "
              f"{beat.get('slurm_job_id') or '-'}, heartbeat {time.time() - float(beat.get('at') or 0):.0f} s ago")
    else:
        print("runner: never ran")
    print(f"{'run':30} {'task':44} {'repo':28} {'status':18} leg winner")
    for entry in queue["runs"]:
        info = (state.get("runs") or {}).get(entry["run"], {})
        print(f"{entry['run']:30} {entry['task']:44} {entry['repo']:28} {info.get('status', 'queued'):18} "
              f"{info.get('leg', 0)}   {info.get('winner') or '-'}")
    print("budget:", json.dumps(state.get("budget")), "| stop_launching:", bool(state.get("stop_launching")),
          "| serving:", bool(state.get("serving")), "| fleetd_stuck:", bool(state.get("fleetd_stuck")),
          "| STOP flag:", STUDENT_STOP.exists())
    print("unpark: a stuck/failed run -> set its status to \"queued\" (leg 0) or \"waiting_to_resume\" in state.json and "
          "drop leg_repairs/launch_errors; a stuck daemon -> drop fleetd_stuck, fleetd_repairs and daemon_legs. Edit "
          "state.json only while the runner is stopped (it rewrites the file every pass).")


def main():
    global DRY, ENV
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="run", choices=["run", "status", "prep"])
    parser.add_argument("--dry-run", action="store_true", help="one pass that prints actions and changes nothing")
    parser.add_argument("--force", action="store_true",
                        help="start although another runner's heartbeat is fresh (the flock still applies)")
    args = parser.parse_args()
    DRY = args.dry_run
    ENV = job_env()
    queue = read_json(Q / "queue.json")
    state_path = Q / "state.json"
    state = read_json(state_path, {}) or {}
    if args.command == "status":
        status(queue, state)
        return 0
    if args.command == "prep":
        for entry in queue["runs"]:
            prep(entry)
        return 0
    if not DRY:
        take_lock(force=args.force)
    log(f"official3 queue runner starting ({'dry run' if DRY else 'live'}) on {HOST} pid {os.getpid()} job "
        f"{os.environ.get('SLURM_JOB_ID') or '-'}: {len(queue['runs'])} runs, max {queue['settings']['max_active']} "
        f"active")
    for entry in queue["runs"]:  # fail fast on a missing analysis before any job is submitted
        prep(entry)
    while True:
        heartbeat("running")
        try:
            done = one_pass(queue, state)
        except Exception as error:  # noqa: BLE001 - e.g. squeue failed: skip this pass, decide nothing
            log(f"pass failed: {type(error).__name__}: {error}")
            done = False
        state["updated_at"] = now_iso()
        if not DRY:
            write_json_atomic(state_path, state)
        if DRY:
            status(queue, state)
            return 0
        if done:
            break
        time.sleep(queue["settings"]["poll_s"])
    log("queue finished: every run is done, failed or stuck")
    stop_serving()
    state["finished_at"] = now_iso()
    write_json_atomic(state_path, state)
    heartbeat("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
