#!/usr/bin/env python3
"""Keep N GLM-5.3 server slots (glm53-serve 2-node jobs, batch partition, 4 h limit) alive for the glm-rollouts campaigns.

    /usr/bin/python3 -I -B keeper.py [--once]     # runs in a CPU job (keeper.sbatch); state in keeper-status.json

Every --interval s:
* target = int(servers-target.txt) (edit the file to scale; 0 = stop submitting);
* slots = this user's jobs named <prefix>NN (each slot is one babysit pair of submit_serve.sh: both candidates share
  the name); a slot counts as LIVE when a candidate is PENDING, or RUNNING with more than --lead-minutes left;
* keeper-config.json (re-read every pass) may override lead_minutes, max_per_hour, max_fast_fails, "venv" (VENV of the
  servers it submits); with "release": true
  it also releases the campaigns (GO-<campaign> files, RELEASE_ORDER) while their summed concurrency stays within
  "per_server" (24) x ready servers, so the first servers are not flooded (driver.py waits for GO or GO-<campaign>);
* while live < target: submit a new slot (bash glm53-serve/scripts/submit_serve.sh roll-NN, next free NN), at most
  --max-per-hour per rolling hour; a slot whose winner ran < --fast-fail-minutes counts as a fast failure, and after
  --max-fast-fails in a row the keeper stops submitting (needs a human);
* two pools: "main" (roll-NN, batch, 4 h; target = servers-target.txt, lead = lead_minutes) and "short" (roll-sNN,
  batch_short, 2 h; target = config short_target, lead = config short_lead_minutes);
* live > target: cancel PENDING slots only if config "scale_down" is true (never a running server);
* STOP file (keeper.STOP) or every campaign results file present (--done-glob, --done-count): stop submitting, cancel
  pending slots, write the status and exit 0. Running servers are left to the operator.
"""
import argparse
import collections
import glob
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


def sh(args, timeout=120):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


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


def slots(prefix):
    out = sh(["squeue", "--me", "-h", "-o", "%i|%j|%T|%L|%M"])
    if out.returncode != 0:
        raise RuntimeError(f"squeue failed: {out.stderr[:200]}")
    found = collections.defaultdict(list)
    for line in out.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 5 and parts[1].startswith(prefix):
            found[parts[1]].append({"job": parts[0], "state": parts[2], "left": parse_left(parts[3]),
                                    "elapsed": parts[4]})
    return dict(found)


RELEASE_ORDER = ["hengyuf-g0", "baihe-g0", "hengyuf-g1", "baihe-g1", "hengyuf-g2", "baihe-g2", "hengyuf-g3", "baihe-g3"]
CONCURRENCY = {"hengyuf": 24, "baihe": 12}


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


def release(ready_servers, per_server, log_path):
    """Create GO-<campaign> files in RELEASE_ORDER while the released campaigns' concurrency stays within
    per_server x ready servers (the first campaign is released as soon as one server is ready). Never revoked."""
    if (GR / "GO").exists():
        return []
    released = [name for name in RELEASE_ORDER if (GR / f"GO-{name}").exists()]
    load = sum(CONCURRENCY[name.split("-")[0]] for name in released)
    created = []
    for name in RELEASE_ORDER:
        if name in released:
            continue
        need = CONCURRENCY[name.split("-")[0]]
        if ready_servers >= 1 and (load + need <= per_server * ready_servers or not released and not created):
            (GR / f"GO-{name}").write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                                       "ready_servers": ready_servers, "load_after": load + need}) + "\n")
            created.append(name)
            load += need
        else:
            break
    if created:
        log(log_path, event="released", campaigns=created, ready_servers=ready_servers, released_load=load)
    return created


def done_results(pattern):
    """Results files written by the driver with the sandbox-failure re-runs (hengyuf-g0's first leg predates them)."""
    count = 0
    for path in glob.glob(pattern):
        try:
            count += "sandbox_failure_rule" in json.loads(Path(path).read_text())
        except (OSError, ValueError):
            pass
    return count


def log(path, **fields):
    with open(path, "a") as stream:
        stream.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields}) + "\n")


POOLS = {
    # name -> slot-name regex (Slurm job name glm53-serve-<slot>), slot format, extra submit settings
    "main": {"regex": r"glm53-serve-roll-(\d+)", "slot": "roll-{:02d}", "partition": None, "extra": []},
    "short": {"regex": r"glm53-serve-roll-s(\d+)", "slot": "roll-s{:02d}", "partition": "batch_short",
              "extra": ["--time=02:00:00"]},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="glm53-serve-roll-")
    parser.add_argument("--target-file", default=str(GR / "servers-target.txt"))
    parser.add_argument("--stop-file", default=str(GR / "keeper.STOP"))
    parser.add_argument("--status-file", default=str(GR / "keeper-status.json"))
    parser.add_argument("--log", default=str(GR / "keeper-log.jsonl"))
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--lead-minutes", type=float, default=30)
    parser.add_argument("--max-per-hour", type=int, default=12)
    parser.add_argument("--fast-fail-minutes", type=float, default=15)
    parser.add_argument("--max-fast-fails", type=int, default=4)
    parser.add_argument("--done-glob", default=str(GR / "results" / "glmlocal-*-g*.json"))
    parser.add_argument("--done-count", type=int, default=8)
    parser.add_argument("--config-file", default=str(GR / "keeper-config.json"))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    submissions = []          # timestamps
    seen_running = {}         # slot name -> first time seen RUNNING
    finished = set()
    fast_fails = 0
    halted = None
    followups = set()  # campaigns given their one follow-up (recovery) leg by this keeper process
    while True:
        now = time.time()
        stop_reason = None
        if Path(args.stop_file).exists():
            stop_reason = "stop file"
        elif done_results(args.done_glob) >= args.done_count:
            stop_reason = f"{args.done_count} campaign results present"
        try:
            main_target = int(Path(args.target_file).read_text().strip())
        except (OSError, ValueError):
            main_target = 0
        try:  # live overrides (no restart needed)
            overrides = json.loads(Path(args.config_file).read_text())
        except (OSError, ValueError):
            overrides = {}
        for key in ("lead_minutes", "max_per_hour", "max_fast_fails"):
            if isinstance(overrides.get(key), (int, float)):
                setattr(args, key, overrides[key])
        do_release = bool(overrides.get("release", False))
        per_server = float(overrides.get("per_server", 24))
        targets = {"main": main_target, "short": int(overrides.get("short_target", 0))}
        leads = {"main": args.lead_minutes, "short": float(overrides.get("short_lead_minutes", 25))}
        scale_down = bool(overrides.get("scale_down", False))
        try:
            current = slots(args.prefix)
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            log(args.log, event="squeue_error", error=str(error)[:300])
            if args.once:
                return 1
            time.sleep(args.interval)
            continue
        pools = {name: {"live": [], "pending": [], "running": []} for name in POOLS}
        ready_servers = 0
        for name, jobs in sorted(current.items()):
            pool = next((p for p, spec in POOLS.items() if re.fullmatch(spec["regex"], name)), None)
            states = {j["state"] for j in jobs}
            run = [j for j in jobs if j["state"] == "RUNNING"]
            ready_servers += any(ready(j["job"]) for j in run)
            if run:
                seen_running.setdefault(name, now)
            if pool is None:
                continue
            info = pools[pool]
            if run:
                lefts = [j["left"] for j in run if j["left"] is not None]
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
        actions = []
        # A campaign leg that ended with a results file from the pre-recovery driver (hengyuf-g0's first leg) gets ONE
        # follow-up leg (bash submit.sh <taskset> <group>), which only runs the sandbox-failure re-runs.
        if not stop_reason:
            try:
                drivers = {line.split("|")[1] for line in sh(["squeue", "--me", "-h", "-o", "%i|%j"]).stdout.splitlines()
                           if "|" in line}
            except (OSError, subprocess.SubprocessError):
                drivers = None
            for name in RELEASE_ORDER if drivers is not None else []:
                result_path = GR / "results" / f"glmlocal-{name}.json"
                if name in followups or not result_path.exists() or f"gr-{name}" in drivers:
                    continue
                try:
                    has_recovery = "sandbox_failure_rule" in json.loads(result_path.read_text())
                except (OSError, ValueError):
                    continue
                if has_recovery or (GR / "babysit" / name / "STOP").exists():
                    continue
                taskset, group = name.rsplit("-g", 1)
                result = subprocess.run(["bash", str(GR / "submit.sh"), taskset, group], capture_output=True, text=True,
                                        timeout=300)
                followups.add(name)
                actions.append({"followup_leg": name, "rc": result.returncode,
                                 "out": result.stdout.strip().splitlines()[-1:] if result.stdout else None})
        if do_release and not stop_reason:
            created = release(ready_servers, per_server, args.log)
            if created:
                actions.append({"released": created, "ready_servers": ready_servers})
        submissions = [t for t in submissions if now - t < 3600]
        for pool, spec in POOLS.items():
            info = pools[pool]
            if stop_reason:
                for name in info["pending"]:
                    ids = [j["job"] for j in current[name] if j["state"] != "RUNNING"]
                    sh(["scancel", *ids])
                    actions.append({"cancel_pending": name, "jobs": ids})
                continue
            if halted:
                continue
            missing = targets[pool] - len(info["live"])
            if missing > 0:
                used = set()
                for n in list(current) + list(seen_running) + list(finished):
                    m = re.fullmatch(spec["regex"], n)
                    if m:
                        used.add(int(m.group(1)))
                for d in (G / "runs" / "serve").glob("roll-*"):
                    m = re.fullmatch(spec["regex"], "glm53-serve-" + d.name)
                    if m:
                        used.add(int(m.group(1)))
                next_id = max(used | {0}) + 1
                for _ in range(missing):
                    if len(submissions) >= args.max_per_hour:
                        actions.append({"rate_limited": True, "pool": pool})
                        break
                    slot = spec["slot"].format(next_id)
                    # Never hand this CPU job's SLURM_* (1 CPU / 2 GB) to the server jobs: through --export=ALL they
                    # confined the servers' engine steps to 1 CPU and 2 GB per node (2026-11-07 s03/s04/s05).
                    # (only the resource-shaping ones: sbatch itself needs e.g. SLURM_CONF)
                    drop = {"SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_MEM_PER_GPU", "SLURM_CPUS_PER_TASK",
                            "SLURM_CPUS_PER_GPU", "SLURM_TRES_PER_TASK"}
                    env = {k: v for k, v in os.environ.items() if k not in drop}
                    if overrides.get("venv"):  # pin the servers' vLLM env (else the sbatch default)
                        env["VENV"] = str(overrides["venv"])
                    if spec["partition"]:
                        env["PARTITION"] = spec["partition"]
                    result = subprocess.run(["bash", str(SUBMIT), slot, *spec["extra"]], capture_output=True,
                                            text=True, timeout=300, env=env)
                    submissions.append(now)
                    actions.append({"submitted": slot, "pool": pool, "rc": result.returncode,
                                    "out": result.stdout.strip().splitlines()[-1:] if result.stdout else None,
                                    "err": result.stderr.strip()[-300:] or None})
                    next_id += 1
            elif missing < 0 and scale_down:  # opt-in: never cancel queued servers unless asked
                for name in info["pending"][:-missing]:
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
                  "max_per_hour": args.max_per_hour, "ready_servers": ready_servers, "scale_down": scale_down,
                  "released": sorted(p.name for p in GR.glob("GO*")), "job": os.environ.get("SLURM_JOB_ID")}
        tmp = Path(args.status_file + ".tmp")
        tmp.write_text(json.dumps(status, indent=1) + "\n")
        os.replace(tmp, args.status_file)
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
    sys.exit(main())
