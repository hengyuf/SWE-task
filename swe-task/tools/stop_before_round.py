#!/usr/bin/env python3
"""Stop a running synth build the moment it starts round N, then let it finish without that round.

The official1 builds run an older task_builder process (started with --rounds 3, before max_rounds.json existed);
hengyuf decided on 2026-10-17 that they stop after round 2. Restarting them now would throw away in-flight probe
rollouts, so this waits instead. Every --interval s it lists the build root for the first round-N candidate
directory (cand-...-r<N>b<j>; evolve() creates it before its sandbox and codex agent start), then
  1. scancels the build job (the newest job id in the run's launch.jsonl),
  2. renames every round-N candidate directory to <name>.round<N>-cancelled (task_builder and the dashboard ignore
     dotted names), and lists the sandboxes the job brought up after the trigger (for cleanup; they expire with
     their 4 h TTL otherwise),
  3. relaunches the build: tools/launch_build.sh --run R --task T -- --rounds <N-1>. With max_rounds.json in the
     build root the new process re-reads the cached rounds and writes summary.json (no agent or hub calls).
A run whose job ends without reaching round N (finished, or crashed and left to the watchers) is dropped; a newer
job id in launch.jsonl (a watcher's resume) is followed instead.

    stop_before_round.py --run official1-bandit --run official1-narwhals --round 3 [--interval 5] [--dry-run]

Events: S/synth/runs/<run>/stop_before_round.log (JSON lines) and stdout.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

S = Path(__file__).resolve().parent.parent
RUNS = S / "synth/runs"
SLURM_BIN = "/cm/shared/apps/slurm/current/bin"
ENV = {"HOME": os.environ.get("HOME", "/root"), "USER": os.environ.get("USER", ""), "LANG": "C.UTF-8",
       "SLURM_CONF": os.environ.get("SLURM_CONF", "/cm/shared/apps/slurm/var/etc/cw-dfw-cs-001/slurm.conf"),
       "PATH": f"{SLURM_BIN}:/usr/local/bin:/usr/bin:/bin"}
JOB_CHECK_S = 60  # squeue at most once a minute per run; the directory listing is the fast trigger


def event(run, kind, **fields):
    line = json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "run": run, "event": kind, **fields})
    print(line, flush=True)
    with (RUNS / run / "stop_before_round.log").open("a") as stream:
        stream.write(line + "\n")


def latest_launch(run):
    lines = (RUNS / run / "launch.jsonl").read_text().splitlines()
    return json.loads(lines[-1])


def job_state(job_id):
    out = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%T"], capture_output=True, text=True, env=ENV,
                         timeout=120).stdout.strip()
    return out or None


class Watch:
    def __init__(self, run, round_index, dry_run):
        self.run, self.round, self.dry_run = run, round_index, dry_run
        launch = latest_launch(run)
        self.task, self.job = launch["task"], str(launch["job_id"])
        roots = [path for path in (RUNS / run / "build" / self.task).iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise SystemExit(f"{run}: expected one build root under build/{self.task}, found {roots}")
        self.root = roots[0]
        self.pattern = re.compile(rf"cand-[A-Za-z0-9-]*-r{round_index}b\d+")
        self.checked = 0.0
        self.done = False
        event(run, "watching", job=self.job, root=str(self.root), round=round_index, dry_run=dry_run)

    def round_dirs(self):
        return sorted(entry.name for entry in os.scandir(self.root)
                      if entry.is_dir(follow_symlinks=False) and self.pattern.fullmatch(entry.name))

    def poll(self):
        found = self.round_dirs()
        if found:
            return self.stop(found)
        if time.time() - self.checked >= JOB_CHECK_S:
            self.checked = time.time()
            if job_state(self.job) is None:
                launch = latest_launch(self.run)
                if str(launch["job_id"]) != self.job and job_state(launch["job_id"]):
                    event(self.run, "following_resume", old_job=self.job, job=str(launch["job_id"]))
                    self.job = str(launch["job_id"])
                elif not self.round_dirs():
                    event(self.run, "job_ended_before_round", job=self.job)
                    self.done = True

    def stop(self, found):
        trigger = time.time()
        event(self.run, "round_started", job=self.job, first_dirs=found[:5])
        if self.dry_run:
            self.done = True
            return
        subprocess.run(["scancel", self.job], env=ENV, timeout=120)
        for _ in range(60):  # up to 5 min for the job to leave the queue
            if job_state(self.job) is None:
                break
            time.sleep(5)
        moved = []
        for name in self.round_dirs():  # listed again: more may have appeared before the job died
            target = self.root / f"{name}.round{self.round}-cancelled"
            (self.root / name).rename(target)
            moved.append(name)
        log = Path(latest_launch(self.run)["log"])
        boxes = []
        if log.exists():
            for line in log.read_text(errors="replace").splitlines()[-400:]:
                match = re.match(r"(\d\d:\d\d:\d\d) sandbox ([0-9a-f-]{36}) up \(([^,]+)", line)
                if match and match.group(3) == "evolve-plan":
                    boxes.append(match.group(2))
        event(self.run, "cancelled", job=self.job, renamed=moved, evolve_plan_sandboxes_in_log_tail=boxes,
              seconds_after_trigger=round(time.time() - trigger, 1))
        relaunch = subprocess.run([str(S / "tools/launch_build.sh"), "--run", self.run, "--task", self.task, "--",
                                   "--rounds", str(self.round - 1)], capture_output=True, text=True, env=ENV,
                                  timeout=900)
        event(self.run, "relaunched", returncode=relaunch.returncode, output=relaunch.stdout.strip()[-600:],
              stderr=relaunch.stderr.strip()[-600:])
        self.done = True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true", help="report the trigger, do not cancel or relaunch")
    args = parser.parse_args()
    watches = [Watch(run, args.round, args.dry_run) for run in args.run]
    while any(not watch.done for watch in watches):
        for watch in watches:
            if not watch.done:
                try:
                    watch.poll()
                except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
                    event(watch.run, "poll_error", error=f"{type(error).__name__}: {error}"[:300])
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
