"""Offline tests of the student worker fleet: no Slurm job is submitted or cancelled, no vLLM or GPU is used.

    cd /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/student
    /usr/bin/python3 -B -m unittest tests.test_fleet -v

FleetTest         reconcile policy against a fake Slurm (canned squeue/sacct, recorded sbatch/scancel).
RegistryTest      serve_student.sh's atomic registry writer against fleet.py's reader (an unlistable registry/
                  raises, F10).
ParityTest        serve_student.sh's vLLM argv == EVAL/scripts/serve_vllm.sh's.
ServeStudentTest  serve_student.sh end to end with a fake vLLM server on this node (lifecycle, watchdogs, cleanup,
                  runtime env parity with serve_vllm.sh, hung registry writes).
"""
import sys

sys.dont_write_bytecode = True

import argparse
import contextlib
import errno
import glob
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock

STUDENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT))
import fleet  # noqa: E402

SERVE_STUDENT = STUDENT / "serve_student.sh"
MIG = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration")
SERVE_VLLM = MIG / "eval/deepswe-v1.1/scripts/serve_vllm.sh"
RUNTIME = MIG / "runtime"
CONTRACT_COMMENT = ('{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"inference_server",'
                    '"description":"Persistent 1-node 8-GPU vLLM inference worker for the swe-task synthetic-task '
                    'student (4h job, continuously resubmitted); GPUs can idle between on-demand rollout bursts"}}')
NOW = 1800000000
CONTRACT_KEYS = {"job_id": str, "slot": str, "node": str, "ip": str, "port": int, "base_url": str,
                 "served_model_name": str, "model_path": str, "max_model_len": int, "per_call_cap": int, "seed": int,
                 "started_at": int, "ready_at": (int, type(None)), "end_epoch": int, "drain_at": int,
                 "heartbeat_at": int, "status": str, "reason": (str, type(None))}


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class Done:
    def __init__(self, stdout="", stderr=""):
        self.stdout, self.stderr, self.returncode = stdout, stderr, 0


class FakeSlurm(fleet.Slurm):
    """Canned squeue/sacct output; records every call; sbatch hands out job ids and keeps the --comment."""

    def __init__(self, squeue="", sacct="", fail=()):
        super().__init__(bin_dir="/nonexistent")
        self.squeue_text, self.sacct_text, self.fail = squeue, sacct, set(fail)
        self.calls, self.comments, self.next_id = [], {}, 30000000

    def run(self, argv, epoch_times=False):
        self.calls.append(list(argv))
        if argv[0] in self.fail:
            raise fleet.SlurmError("%s failed (test)" % argv[0])
        if argv[0] == "squeue" and "-j" in argv:
            return Done(self.comments.get(argv[argv.index("-j") + 1], "(null)") + "\n")
        if argv[0] == "squeue":
            return Done(self.squeue_text)
        if argv[0] == "sacct":
            return Done(self.sacct_text)
        if argv[0] == "sbatch":
            self.next_id += 1
            comment = [arg[len("--comment="):] for arg in argv if arg.startswith("--comment=")]
            self.comments[str(self.next_id)] = comment[0] if comment else "(null)"
            return Done("%d\n" % self.next_id)
        if argv[0] in ("scancel", "scontrol"):
            return Done()
        raise AssertionError("unexpected Slurm command %r" % argv)

    def submitted(self):
        return [call for call in self.calls if call[0] == "sbatch"]

    def names(self):
        return [arg[len("--job-name="):] for call in self.submitted() for arg in call if arg.startswith("--job-name=")]

    def cancelled(self):
        return [arg for call in self.calls if call[0] == "scancel" for arg in call[1:] if not arg.startswith("--")]


def sq(job_id, name, state="RUNNING", node="pool0-00001", start=None, end=None, reason="None", user="hengyuf",
       partition="batch", account="nemotron_agents_dev"):
    """One squeue line in fleet's format (times as epoch, like SLURM_TIME_FORMAT=%s; %a before the name)."""
    if state in fleet.PENDING:
        node, end = "", None
    fields = [str(job_id), user, state, partition, node, "1:00:00", "2:00:00",
              str(start) if start else "N/A", str(end) if end else "N/A", reason, account, name]
    return "|".join(fields)


def lines(*rows):
    return "".join(row + "\n" for row in rows)


def entry(job_id, status, node="pool0-00001", heartbeat=NOW, ready_at=NOW - 600, end=NOW + 3 * 3600, **extra):
    record = {"job_id": str(job_id), "slot": "00", "node": node, "ip": "10.0.0.1", "port": 8000,
              "base_url": "http://10.0.0.1:8000/v1", "served_model_name": "policy", "model_path": "/m",
              "max_model_len": 262144, "per_call_cap": 131072, "seed": 1, "started_at": NOW - 3600,
              "ready_at": ready_at, "end_epoch": end, "drain_at": end - 2700, "heartbeat_at": heartbeat,
              "status": status, "reason": None}
    record.update(extra)
    return record


class FleetTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="fleet-test-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        (self.root / "registry").mkdir()
        self.now = NOW

    def fleet(self, squeue="", sacct="", fail=(), config=None):
        self.slurm = FakeSlurm(squeue, sacct, fail)
        if config is not None:
            cfg = json.loads(json.dumps(fleet.DEFAULTS))
            cfg.update(config)
            fleet.write_json(self.root / "fleet.json", cfg)
        return fleet.Fleet(self.root, self.slurm, clock=lambda: self.now)

    def reconcile(self, instance, dry_run=False):
        with quiet():
            return instance.reconcile(dry_run=dry_run)

    def register(self, record):
        fleet.write_json(self.root / "registry" / ("%s.json" % record["job_id"]), record)

    def remember(self, *records):
        fleet.write_json(self.root / "state/jobs.json", {record["job_id"]: record for record in records})

    def events(self, kind=None):
        path = self.root / "state/events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return [event for event in events if kind is None or event["event"] == kind]

    # -- submission argv --

    def test_comment_is_exactly_the_contract_string(self):
        self.assertEqual(fleet.comment_json(fleet.DEFAULTS), CONTRACT_COMMENT)
        deployed = json.loads((STUDENT / "fleet.json").read_text())
        self.assertEqual(fleet.comment_json(deployed), CONTRACT_COMMENT)

    def test_sbatch_argv(self):
        instance = self.fleet()
        cfg = instance.config()
        argv = instance.sbatch_argv(cfg, "swe-task-student-w07")
        self.assertEqual(argv[0], "sbatch")
        self.assertEqual(argv[-1], str(self.root / "worker.sbatch"))
        for expected in ("--parsable", "--export=NONE", "--job-name=swe-task-student-w07",
                         "--account=nemotron_agents_dev", "--partition=batch", "--time=04:00:00",
                         "--exclude=pool0-00058,pool0-00484,pool0-00502,pool0-00613,pool0-01460"):
            self.assertIn(expected, argv)
        self.assertEqual([arg for arg in argv if arg.startswith("--comment")], ["--comment=" + CONTRACT_COMMENT])
        argv = instance.sbatch_argv(cfg, "swe-task-student-w99", partition="interactive", walltime="01:00:00")
        self.assertIn("--partition=interactive", argv)
        self.assertIn("--time=01:00:00", argv)

    def test_slurm_calls_use_argv_and_a_scrubbed_environment(self):
        leaky = {"SLURM_JOB_ID": "1", "SLURM_CPUS_PER_TASK": "2", "SBATCH_PARTITION": "backfill", "SQUEUE_FORMAT": "%i",
                 "SCANCEL_NAME": "x", "SLURM_TIME_FORMAT": "relative", "SLURM_CONF": "/x/slurm.conf", "API_KEY": "s"}
        with mock.patch.dict(os.environ, leaky), mock.patch.object(fleet.subprocess, "run") as run:
            run.return_value = Done("123;cluster\n")
            job_id = fleet.Slurm().sbatch(["sbatch", "--parsable", "--comment=" + CONTRACT_COMMENT, "w.sbatch"])
        self.assertEqual(job_id, "123")
        args, kwargs = run.call_args
        self.assertEqual(args[0], [fleet.SLURM_BIN + "/sbatch", "--parsable", "--comment=" + CONTRACT_COMMENT,
                                   "w.sbatch"])
        self.assertFalse(kwargs.get("shell"))
        self.assertEqual(set(kwargs["env"]), {"PATH", "HOME", "USER", "LOGNAME", "LANG", "SLURM_CONF"})
        self.assertEqual(kwargs["env"]["SLURM_CONF"], "/x/slurm.conf")
        self.assertEqual(fleet.slurm_env(epoch_times=True)["SLURM_TIME_FORMAT"], "%s")

    # -- parsing --

    def test_parse_squeue_and_sacct(self):
        jobs = fleet.parse_squeue(
            "20366997|hengyuf|RUNNING|interactive|pool0-00028|13:27|1:46:33|1792196462|1792203662|None|"
            "nemotron_agents_dev|flres-debug\n"
            "20362017|hengyuf|PENDING|cpu_short||0:00|30:00|N/A|N/A|Dependency|nemotron_rl_algo|odd|name\n")
        self.assertEqual((jobs[0]["start"], jobs[0]["end"]), (1792196462, 1792203662))
        self.assertEqual((jobs[1]["name"], jobs[1]["start"], jobs[1]["node"]), ("odd|name", None, ""))
        with self.assertRaises(fleet.SlurmError):
            fleet.parse_squeue("not a squeue line\n")
        rows = fleet.parse_sacct(
            "20051394|hengyuf|CANCELLED by 146504|0:0|1792100000|1792103600|01:00:00|pool0-00012|Cancelled by "
            "occupied idle job reaper: 64/64 GPUs idle for 30m, 100% waste\n"
            "20051395|hengyuf|PENDING|0:0|Unknown|Unknown|00:00:00|None assigned|\n")
        self.assertEqual((rows["20051394"]["state"], rows["20051394"]["cancelled_by"]), ("CANCELLED", "146504"))
        self.assertEqual((rows["20051395"]["start"], rows["20051395"]["node"]), (None, ""))
        self.assertEqual([fleet.duration(text) for text in ("4:00:00", "30:00", "1-02:03:04", "0:00", "UNLIMITED")],
                         [14400, 1800, 93784, 0, None])

    # -- coverage, caps, slots --

    def test_coverage_counts_long_running_and_pending_workers_only(self):
        squeue = lines(
            sq(100, "swe-task-student-w00", start=NOW - 3600, end=NOW + 3 * 3600),  # covers
            sq(101, "swe-task-student-w01", start=NOW - 3 * 3600, end=NOW + 3600),  # < 90 min left
            sq(102, "swe-task-student-w02", state="PENDING", reason="Priority"),  # covers
            sq(103, "swe-task-student-w03", state="PENDING", reason="JobHeldUser"),  # held
            sq(104, "scale-rebench-super-sc-formal-20261013-r2", start=NOW - 3600, end=NOW + 3 * 3600),
            sq(105, "swe-task-student-w05", user="someoneelse", start=NOW - 60, end=NOW + 4 * 3600))
        self.register(entry(100, "ready"))
        self.register(entry(101, "draining"))
        instance = self.fleet(squeue)
        self.reconcile(instance)
        self.assertEqual(self.slurm.names(), ["swe-task-student-w%02d" % slot for slot in range(4, 12)])  # 10 - 2
        self.assertEqual(self.slurm.cancelled(), [])
        status = json.loads((self.root / "state/status.json").read_text())
        self.assertEqual(status["counts"]["coverage"], 10)  # this pass's submissions count as pending
        self.assertEqual(len(self.events("submit")), 8)
        known = json.loads((self.root / "state/jobs.json").read_text())
        self.assertEqual(len(known), 4 + 8)  # adopted workers + submitted (others' / non-fleet jobs never)

    def test_max_jobs_caps_submissions(self):
        squeue = lines(*[sq(200 + i, "swe-task-student-w%02d" % i, start=NOW - 3 * 3600, end=NOW + 1800)
                         for i in range(18)])
        for i in range(18):
            self.register(entry(200 + i, "draining", end=NOW + 1800))
        instance = self.fleet(squeue, config={"max_jobs": 20})
        self.reconcile(instance)
        self.assertEqual(self.slurm.names(), ["swe-task-student-w18", "swe-task-student-w19"])

    def test_max_submits_per_pass_and_lowest_free_slots(self):
        squeue = lines(*[sq(300 + i, "swe-task-student-w%02d" % i, start=NOW - 3 * 3600, end=NOW + 1800)
                         for i in (0, 1, 3)])
        instance = self.fleet(squeue, config={"max_submits_per_pass": 3})
        self.reconcile(instance)
        self.assertEqual(self.slurm.names(), ["swe-task-student-w02", "swe-task-student-w04", "swe-task-student-w05"])
        for call in self.slurm.submitted():  # each submission's exemption comment was verified
            self.assertIn("--comment=" + CONTRACT_COMMENT, call)
        self.assertEqual(self.events("comment_mismatch"), [])

    # -- policing --

    def test_stuck_startup_is_cancelled_and_its_node_excluded(self):
        squeue = lines(sq(400, "swe-task-student-w00", node="pool0-00777", start=NOW - 30 * 60, end=NOW + 3 * 3600),
                       sq(401, "swe-task-student-w01", node="pool0-00778", start=NOW - 20 * 60, end=NOW + 3 * 3600))
        self.register(entry(400, "starting", node="pool0-00777", ready_at=None))
        instance = self.fleet(squeue, config={"target": 2})
        self.reconcile(instance)
        self.assertIn(["scancel", "--user=hengyuf", "400"], self.slurm.calls)
        self.assertEqual(self.slurm.cancelled(), ["400"])  # 401 is only 20 min old
        self.assertIn("pool0-00777", instance.config()["exclude_nodes"])
        excluded = [json.loads(line) for line in (self.root / "state/excluded_nodes.jsonl").read_text().splitlines()]
        self.assertEqual([(item["node"], item["kind"], item["job_id"]) for item in excluded],
                         [("pool0-00777", "stuck_startup", "400")])
        self.assertEqual(self.slurm.names(), ["swe-task-student-w02"])  # replacement, avoiding the bad node
        self.assertTrue(any(arg.startswith("--exclude=") and "pool0-00777" in arg for arg in self.slurm.submitted()[0]))
        self.now += 6 * 60  # 401 still has no registry record after 26 min: stuck as well
        self.reconcile(self.fleet(squeue))
        self.assertEqual(self.slurm.cancelled(), ["401"])  # 400 is not counted or excluded twice
        self.assertEqual(len(self.events("worker_failure")), 2)

    def test_failed_worker_is_excluded_once_and_cancelled_after_the_grace(self):
        squeue = lines(sq(500, "swe-task-student-w00", node="pool0-00500", start=NOW - 600, end=NOW + 3 * 3600))
        self.register(entry(500, "failed", node="pool0-00500", ready_at=None, reason="startup hang", exclude_node=True,
                            started_at=NOW - 600))
        instance = self.fleet(squeue, config={"target": 0})
        self.reconcile(instance)
        self.assertEqual(self.slurm.cancelled(), [])  # the worker exits by itself; give it terminal_grace_min
        self.assertIn("pool0-00500", instance.config()["exclude_nodes"])
        self.now += 6 * 60
        self.reconcile(self.fleet(squeue))
        self.assertEqual(self.slurm.cancelled(), ["500"])
        self.assertEqual(len(self.events("worker_failure")), 1)
        self.assertEqual(len(self.events("exclude_node")), 1)

    def test_environment_failure_counts_but_keeps_the_node(self):
        squeue = lines(sq(510, "swe-task-student-w00", node="pool0-00510", start=NOW - 60, end=NOW + 4 * 3600))
        self.register(entry(510, "failed", node="pool0-00510", ready_at=None, exclude_node=False, started_at=NOW - 60))
        instance = self.fleet(squeue, config={"target": 0})
        self.reconcile(instance)
        self.assertNotIn("pool0-00510", instance.config()["exclude_nodes"])
        self.assertEqual(len(instance.breaker()["failures"]), 1)

    def test_stale_heartbeat_is_cancelled_unless_most_workers_are_stale(self):
        rows = [sq(520 + i, "swe-task-student-w%02d" % i, start=NOW - 3600, end=NOW + 3 * 3600) for i in range(4)]
        for i in range(4):
            self.register(entry(520 + i, "ready", heartbeat=NOW - (11 * 60 if i == 0 else 10)))
        instance = self.fleet(lines(*rows), config={"target": 4})
        self.reconcile(instance)
        self.assertEqual(self.slurm.cancelled(), ["520"])
        self.assertEqual(self.events("worker_failure"), [])  # a stale heartbeat does not blame the node
        for i in range(4):  # all stale at once: a Lustre/clock problem, cancel nothing
            self.register(entry(520 + i, "ready", heartbeat=NOW - 11 * 60))
        self.reconcile(self.fleet(lines(*rows), config={"target": 4}))
        self.assertEqual(self.slurm.cancelled(), [])
        self.assertEqual(len(self.events("mass_stale_heartbeats")), 1)

    def test_never_cancels_non_fleet_jobs(self):
        squeue = lines(
            sq(600, "scale-rebench-super-sc-formal-20261013-r2", start=NOW - 5 * 3600, end=NOW + 600),
            sq(601, "swe-task-student-w00", user="someoneelse", start=NOW - 5 * 3600, end=NOW + 3600),
            sq(602, "swe-task-student-fleetd", partition="cpu_long", start=NOW - 5 * 3600, end=NOW + 86400),
            sq(603, "ring-baseline-fullattn-norope-d768", start=NOW - 5 * 3600, end=NOW + 3600))
        instance = self.fleet(squeue, config={"target": 0})
        self.reconcile(instance)
        self.assertEqual(self.slurm.cancelled(), [])
        step = fleet.Pass(instance, dry_run=False)
        step.cfg, step.now = instance.config(), NOW
        jobs = {job["job_id"]: job for job in fleet.parse_squeue(squeue)}
        for job_id in jobs:
            with self.assertRaises(AssertionError):
                step.cancel({job_id: "test"}, jobs)
        self.assertEqual(self.slurm.cancelled(), [])

    def test_stop_cancels_only_fleet_and_daemon_jobs_daemon_first(self):
        squeue = lines(
            sq(700, "swe-task-student-w00", start=NOW - 3600, end=NOW + 3 * 3600),
            sq(701, "swe-task-student-w01", state="PENDING", reason="Priority"),
            sq(702, "swe-task-student-fleetd", partition="cpu_long", start=NOW - 3600, end=NOW + 86400),
            sq(703, "ring-baseline-fullattn-norope-d768", start=NOW - 3600, end=NOW + 3600),
            sq(704, "swe-task-student-w02", user="someoneelse", start=NOW - 3600, end=NOW + 3600))
        instance = self.fleet(squeue)
        with quiet():
            fleet.command_stop(instance, argparse.Namespace(yes=False))
        self.assertEqual((self.slurm.cancelled(), instance.stop_flag.exists()), ([], False))
        with quiet():
            fleet.command_stop(instance, argparse.Namespace(yes=True))
        self.assertEqual([call for call in self.slurm.calls if call[0] == "scancel"],
                         [["scancel", "--user=hengyuf", "702"], ["scancel", "--user=hengyuf", "700", "701"]])
        self.assertTrue(instance.stop_flag.exists())
        self.reconcile(self.fleet(""))  # STOP flag: no submissions
        self.assertEqual(self.slurm.submitted(), [])

    # -- jobs that left the queue --

    def test_reaper_cancel_is_an_incident_and_the_record_is_archived(self):
        self.remember({"job_id": "800", "name": "swe-task-student-w00", "slot": "00", "state": "RUNNING",
                       "node": "pool0-00800", "start": NOW - 7200, "end": NOW + 7200})
        self.register(entry(800, "ready", node="pool0-00800"))
        sacct = ("800|hengyuf|CANCELLED by 146504|0:0|%d|%d|02:00:00|pool0-00800|Cancelled by occupied idle job "
                 "reaper: 8/8 GPUs idle for 30m, 100%% waste\n" % (NOW - 7200, NOW - 60))
        instance = self.fleet("", sacct, config={"target": 0})
        self.reconcile(instance)
        self.assertEqual([event["event"] for event in self.events() if event.get("incident")], ["reaper_cancel"])
        self.assertFalse((self.root / "registry/800.json").exists())
        self.assertTrue((self.root / "registry/archive/800.json").exists())
        self.assertNotIn("800", json.loads((self.root / "state/jobs.json").read_text()))
        self.assertNotIn("pool0-00800", instance.config()["exclude_nodes"])

    def test_container_failure_without_registry_excludes_the_node(self):
        self.remember({"job_id": "810", "name": "swe-task-student-w00", "slot": "00", "state": "RUNNING",
                       "node": "pool0-00810", "start": NOW - 120, "end": NOW + 4 * 3600})
        sacct = "810|hengyuf|FAILED|1:0|%d|%d|00:01:30|pool0-00810|\n" % (NOW - 120, NOW - 30)
        instance = self.fleet("", sacct, config={"target": 1})
        self.reconcile(instance)
        self.assertIn("pool0-00810", instance.config()["exclude_nodes"])
        self.assertEqual(self.slurm.names(), ["swe-task-student-w00"])  # slot 00 is free again

    def test_vanished_job_keeps_counting_until_its_end_is_confirmed(self):
        record = {"job_id": "900", "name": "swe-task-student-w00", "slot": "00", "state": "RUNNING",
                  "node": "pool0-00900", "start": NOW - 3600, "end": NOW + 3 * 3600}
        self.remember(record)
        still_running = "900|hengyuf|RUNNING|0:0|%d|Unknown|01:00:00|pool0-00900|\n" % (NOW - 3600)
        self.reconcile(self.fleet("", still_running, config={"target": 1}))
        self.assertEqual(self.slurm.submitted(), [])  # an empty/odd squeue must not trigger a mass resubmission
        self.reconcile(self.fleet("", fail=("sacct",)))
        self.assertEqual(self.slurm.submitted(), [])
        self.now += fleet.UNCONFIRMED_GRACE_S + 1
        self.reconcile(self.fleet("", still_running))
        self.assertEqual(self.slurm.names(), ["swe-task-student-w00"])
        self.assertNotIn("900", json.loads((self.root / "state/jobs.json").read_text()))

    def test_failed_squeue_takes_no_action(self):
        instance = self.fleet(fail=("squeue",))
        with self.assertRaises(fleet.SlurmError):
            self.reconcile(instance)
        self.assertEqual([call[0] for call in self.slurm.calls], ["squeue"])
        self.assertFalse((self.root / "state/jobs.json").exists())

    # -- circuit breaker, dry run --

    def test_breaker_stops_submissions_and_exclusions_until_reset(self):
        rows = [sq(1000 + i, "swe-task-student-w%02d" % i, node="pool0-0100%d" % i, start=NOW - 40 * 60,
                   end=NOW + 3 * 3600) for i in range(5)]
        instance = self.fleet(lines(*rows), config={"target": 10})
        self.reconcile(instance)
        self.assertTrue(instance.breaker()["open"])
        self.assertEqual(sorted(self.slurm.cancelled()), [str(1000 + i) for i in range(5)])
        self.assertEqual(len([node for node in instance.config()["exclude_nodes"] if node.startswith("pool0-0100")]), 4)
        self.assertEqual(self.slurm.submitted(), [])
        self.assertEqual(len(self.events("breaker_open")), 1)
        self.reconcile(self.fleet(""))
        self.assertEqual(self.slurm.submitted(), [])
        self.assertEqual(len(self.events("submit_paused")), 1)  # throttled, not one per pass
        with quiet():
            fleet.command_reset_breaker(instance, None)
        self.reconcile(self.fleet(""))
        self.assertEqual(len(self.slurm.submitted()), 10)

    def test_dry_run_writes_nothing(self):
        squeue = lines(sq(1100, "swe-task-student-w00", node="pool0-01100", start=NOW - 40 * 60, end=NOW + 3 * 3600))
        before = sorted(self.root.rglob("*"))
        actions = self.fleet(squeue).reconcile(dry_run=True)
        self.assertEqual(sorted(self.root.rglob("*")), before)
        self.assertEqual([call[0] for call in self.slurm.calls], ["squeue"])
        kinds = [action["action"] for action in actions]
        self.assertEqual((kinds.count("cancel"), kinds.count("exclude_node"), kinds.count("submit")), (1, 1, 10))
        submit = [action for action in actions if action["action"] == "submit"][0]
        self.assertIn("--exclude=pool0-00058,pool0-00484,pool0-00502,pool0-00613,pool0-01100,pool0-01460",
                      submit["argv"])

    # -- status, daemon, start-daemon --

    def test_snapshot_routability_and_counts(self):
        squeue = lines(
            sq(1200, "swe-task-student-w00", start=NOW - 3600, end=NOW + 3 * 3600),
            sq(1201, "swe-task-student-w01", start=NOW - 3600, end=NOW + 3 * 3600),
            sq(1202, "swe-task-student-w02", start=NOW - 3 * 3600, end=NOW + 1800),
            sq(1203, "swe-task-student-w03", state="PENDING", reason="Resources"),
            sq(1204, "swe-task-student-w04", start=NOW - 300, end=NOW + 4 * 3600 - 300))
        self.register(entry(1200, "ready", heartbeat=NOW - 30))
        self.register(entry(1201, "ready", heartbeat=NOW - 300))  # routers skip it (> 120 s)
        self.register(entry(1202, "ready", heartbeat=NOW - 10, end=NOW + 1800))  # past drain_at
        self.register(entry(1204, "starting", ready_at=None))
        instance = self.fleet(squeue)
        view = instance.snapshot(instance.config(), fleet.parse_squeue(squeue),
                                 fleet.read_registry(self.root / "registry"), NOW)
        counts = view["counts"]
        self.assertEqual({key: counts.get(key) for key in ("ready", "draining", "stale", "starting", "pending",
                                                           "routable", "coverage", "total")},
                         {"ready": 1, "draining": 1, "stale": 1, "starting": 1, "pending": 1, "routable": 2,
                          "coverage": 4, "total": 5})
        self.assertEqual(view["pending_reasons"], {"Resources": 1})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            fleet.print_status(view)
        self.assertIn("http://10.0.0.1:8000/v1", output.getvalue())
        self.assertIn("pending reasons: Resources 1", output.getvalue())

    def test_daemon_runs_passes_until_and_is_single_instance(self):
        instance = self.fleet("", config={"target": 0})
        (self.root / "state").mkdir(exist_ok=True)
        with open(self.root / "state/fleet.lock", "a") as holder:
            import fcntl
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with quiet():
                self.assertEqual(instance.daemon(interval=1, until=time.time() + 5), 75)
        with quiet():
            self.assertEqual(instance.daemon(interval=1, until=time.time() + 2.5), 0)
        self.assertGreaterEqual(len([call for call in self.slurm.calls if call[0] == "squeue"]), 2)
        heartbeat = json.loads((self.root / "state/daemon.json").read_text())
        self.assertEqual(heartbeat["stop_reason"], "until")
        self.assertEqual([event["event"] for event in self.events() if event["event"].startswith("daemon")],
                         ["daemon_start", "daemon_stop"])
        self.assertIs(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)  # handlers restored

    def test_start_daemon_queues_one_successor(self):
        squeue = lines(sq(1300, "swe-task-student-fleetd", partition="cpu_long", start=NOW - 60, end=NOW + 7 * 86400))
        instance = self.fleet(squeue)
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "1300"}), quiet():
            fleet.command_start_daemon(instance, argparse.Namespace(successor=True, partition=None, time=None))
        argv = self.slurm.submitted()[0]
        self.assertEqual(argv, ["sbatch", "--parsable", "--export=NONE", "--job-name=swe-task-student-fleetd",
                                "--account=nemotron_agents_dev", "--partition=cpu_datamover", "--time=7-00:00:00",
                                str(self.root / "fleetd.sbatch")])
        both = squeue + sq(1301, "swe-task-student-fleetd", state="PENDING", reason="Dependency",
                           partition="cpu_long") + "\n"
        for successor in (True, False):
            instance = self.fleet(both)
            with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "1300"}), quiet():
                fleet.command_start_daemon(instance, argparse.Namespace(successor=successor, partition=None,
                                                                        time=None))
            self.assertEqual(self.slurm.submitted(), [])
        instance = self.fleet("")  # nothing queued: a manual start clears the STOP flag; CLI partition wins
        instance.state.mkdir(exist_ok=True)
        instance.stop_flag.write_text("x")
        with quiet():
            fleet.command_start_daemon(instance, argparse.Namespace(successor=False, partition="cpu_short",
                                                                    time="04:00:00"))
        self.assertFalse(instance.stop_flag.exists())
        self.assertIn("--partition=cpu_short", self.slurm.submitted()[0])

    def test_submit_dry_run_prints_the_argv_and_submits_nothing(self):
        instance = self.fleet("")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            fleet.command_submit(instance, argparse.Namespace(slot=0, partition=None, time=None, dry_run=True,
                                                              test_only=False))
        argv = json.loads(output.getvalue().splitlines()[1])
        self.assertEqual(argv, instance.sbatch_argv(instance.config(create=False), "swe-task-student-w00"))
        self.assertEqual(self.slurm.submitted(), [])
        self.assertFalse((self.root / "fleet.json").exists())


    # -- review regressions --

    def test_unrelated_jobs_sharing_the_prefix_are_never_touched(self):
        squeue = lines(sq(1400, "swe-task-student-webui", node="pool0-01400", start=NOW - 3600, end=NOW + 3 * 3600),
                       sq(1401, "swe-task-student-w05-debug", node="pool0-01401", start=NOW - 3600, end=NOW + 3600))
        instance = self.fleet(squeue, config={"target": 0})
        self.reconcile(instance)
        self.assertEqual(self.slurm.cancelled(), [])
        self.assertEqual(self.events("worker_failure"), [])
        with quiet():
            fleet.command_stop(instance, argparse.Namespace(yes=True))
        self.assertEqual(self.slurm.cancelled(), [])

    def test_successor_ignores_a_completing_daemon_job(self):
        squeue = lines(sq(1500, "swe-task-student-fleetd", state="COMPLETING", partition="cpu_datamover",
                          start=NOW - 7 * 86400, end=NOW),
                       sq(1501, "swe-task-student-fleetd", partition="cpu_datamover", start=NOW, end=NOW + 7 * 86400))
        instance = self.fleet(squeue)
        with quiet(), mock.patch.dict(os.environ, {"SLURM_JOB_ID": "1501"}):
            fleet.command_start_daemon(instance, argparse.Namespace(successor=True, partition=None, time=None))
        self.assertEqual(len(self.slurm.submitted()), 1)

    def test_requeued_worker_is_not_judged_by_its_previous_run(self):
        rows = [sq(1600, "swe-task-student-w00", node="pool0-01600", start=NOW - 90, end=NOW + 4 * 3600 - 90)]
        rows += [sq(1601 + i, "swe-task-student-w%02d" % (1 + i), start=NOW - 3600, end=NOW + 3 * 3600) for i in range(2)]
        self.register(entry(1600, "ready", node="pool0-09999", heartbeat=NOW - 3 * 3600, started_at=NOW - 6 * 3600))
        for i in range(2):
            self.register(entry(1601 + i, "ready", started_at=NOW - 3600))
        instance = self.fleet(lines(*rows), config={"target": 3})
        self.reconcile(instance)
        self.assertEqual(self.slurm.cancelled(), [])
        self.assertEqual(self.slurm.submitted(), [])  # the restarted run is starting: it covers its slot
        self.register(entry(1600, "unhealthy", node="pool0-09999", heartbeat=NOW - 3 * 3600, started_at=NOW - 6 * 3600))
        self.reconcile(self.fleet(lines(*rows), config={"target": 3}))
        self.assertEqual(self.events("worker_failure"), [])
        self.assertNotIn("pool0-01600", instance.config()["exclude_nodes"])

    def test_one_bad_node_neither_opens_the_breaker_nor_keeps_taking_pending_jobs(self):
        known = {str(1700 + i): {"job_id": str(1700 + i), "name": "swe-task-student-w%02d" % i, "slot": "%02d" % i,
                                 "partition": "batch", "state": "RUNNING", "node": "pool0-01700", "start": NOW - 60,
                                 "end": NOW + 4 * 3600, "submitted_at": NOW - 3600} for i in range(5)}
        self.remember(*known.values())
        sacct = "".join("%d|hengyuf|FAILED|1:0|%d|%d|00:01:00|pool0-01700|\n" % (1700 + i, NOW - 60, NOW - 5)
                        for i in range(5))
        pending = lines(sq(1710, "swe-task-student-w10", state="PENDING", reason="Priority"))
        instance = self.fleet(pending, sacct, config={"target": 1})
        self.reconcile(instance)
        self.assertFalse(instance.breaker()["open"])
        self.assertIn("pool0-01700", instance.config()["exclude_nodes"])
        updates = [call for call in self.slurm.calls if call[0] == "scontrol"]
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0][2], "JobId=1710")
        self.assertIn("pool0-01700", updates[0][3])

    def test_breaker_half_open_probe_closes_it(self):
        sacct = "".join("%d|hengyuf|FAILED|1:0|%d|%d|00:01:00|pool0-%05d|\n" % (1800 + i, NOW - 60, NOW - 5, 1800 + i)
                        for i in range(4))
        self.remember(*[{"job_id": str(1800 + i), "name": "swe-task-student-w%02d" % i, "slot": "%02d" % i,
                         "partition": "batch", "state": "RUNNING", "node": "pool0-%05d" % (1800 + i),
                         "start": NOW - 60, "end": NOW + 4 * 3600} for i in range(4)])
        instance = self.fleet("", sacct, config={"target": 2})
        self.reconcile(instance)
        self.assertTrue(instance.breaker()["open"])
        self.assertEqual(self.slurm.submitted(), [])
        self.now += 61 * 60  # cooldown over: exactly one probe
        self.reconcile(self.fleet("", sacct))
        self.assertEqual(len(self.slurm.submitted()), 1)
        probe = instance.breaker()["probe_job"]
        self.assertTrue(probe)
        self.reconcile(self.fleet(lines(sq(probe, "swe-task-student-w00", state="PENDING", reason="Priority")), sacct))
        self.assertEqual(self.slurm.submitted(), [])  # nothing more while the probe is out
        self.now += 600
        self.register(entry(probe, "ready", started_at=self.now - 540, heartbeat=self.now))
        self.reconcile(self.fleet(lines(sq(probe, "swe-task-student-w00", start=self.now - 540,
                                           end=self.now + 4 * 3600 - 540)), sacct))
        self.assertFalse(instance.breaker()["open"])
        self.assertEqual(len(self.slurm.submitted()), 1)  # target 2: normal top-up resumes

    def test_launch_failure_hold_is_not_coverage(self):
        squeue = lines(sq(1900, "swe-task-student-w00", state="PENDING", reason="launch failed requeued held"))
        instance = self.fleet(squeue, config={"target": 1})
        self.reconcile(instance)
        self.assertEqual(self.slurm.names(), ["swe-task-student-w01"])

    def test_single_stale_worker_is_cancelled(self):
        squeue = lines(sq(2000, "swe-task-student-w00", start=NOW - 7200, end=NOW + 2 * 3600))
        self.register(entry(2000, "ready", heartbeat=NOW - 1200, started_at=NOW - 7200))
        self.reconcile(self.fleet(squeue, config={"target": 1}))
        self.assertEqual(self.slurm.cancelled(), ["2000"])

    def test_adaptive_horizon(self):
        cfg = json.loads(json.dumps(fleet.DEFAULTS))
        self.assertEqual(fleet.horizon_s(cfg, [], [], NOW), 90 * 60)  # no data: horizon_min
        waits = [{"job_id": str(i), "partition": "batch", "start": NOW - 3600, "wait_s": w}
                 for i, w in enumerate((600, 4200, 8000, 9000))]
        self.assertEqual(fleet.horizon_s(cfg, waits, [], NOW), 9000 + 600 + 1200)  # p90 + startup + margin
        pending = [{"job_id": "1", "state": "PENDING", "reason": "Priority", "partition": "batch",
                    "start": NOW + 4 * 3600}]
        # scheduler estimate, capped: horizon_max_min 200, but 4 h walltime - 40 min stagger - 10 min = 190
        self.assertEqual(fleet.horizon_s(cfg, waits, pending, NOW), 190 * 60)
        self.assertEqual(fleet.horizon_s(dict(cfg, stagger_min=0), waits, pending, NOW), 200 * 60)
        self.assertEqual(fleet.horizon_s(dict(cfg, adaptive_horizon=False), waits, pending, NOW), 90 * 60)
        jobs = [{"job_id": "2", "user": "hengyuf", "state": "RUNNING", "partition": "batch", "end": NOW + 150 * 60,
                 "time_left": "2:30:00", "reason": "None"}]
        self.assertEqual(fleet.coverage(cfg, jobs, {}, NOW), 1)
        self.assertEqual(fleet.coverage(cfg, jobs, {}, NOW, horizon=fleet.horizon_s(cfg, waits, [], NOW)), 0)

    def test_horizon_never_hides_a_fresh_worker(self):
        # 2026-10-17: horizon_max_min 270 > the 240 min walltime: no worker counted, the daemon topped up to max_jobs
        cfg = dict(json.loads(json.dumps(fleet.DEFAULTS)), horizon_max_min=270, stagger_min=40, time="04:00:00")
        waits = [{"job_id": str(i), "partition": "batch", "start": NOW - 600, "wait_s": 6 * 3600} for i in range(10)]
        horizon = fleet.horizon_s(cfg, waits, [], NOW)
        self.assertEqual(horizon, 190 * 60)
        fresh = [{"job_id": "w%d" % c, "user": "hengyuf", "name": "swe-task-student-w%02d" % c, "state": "RUNNING",
                  "partition": "batch", "end": NOW + 235 * 60, "time_left": "", "reason": "None"} for c in range(10)]
        self.assertEqual(fleet.coverage(cfg, fresh, {}, NOW, horizon=horizon), 10)  # every stagger class counts
        self.assertEqual(fleet.horizon_cap_s(dict(cfg, time="UNLIMITED")), 270 * 60)  # no walltime: config cap

    def test_stagger_spreads_replacements(self):
        cfg = dict(json.loads(json.dumps(fleet.DEFAULTS)), stagger_min=40)

        def job(slot, left_min):
            return {"job_id": slot, "user": "hengyuf", "name": "swe-task-student-w%s" % slot, "state": "RUNNING",
                    "partition": "batch", "end": NOW + left_min * 60, "time_left": "", "reason": "None"}
        horizon = 45 * 60  # slot classes 0-9 add 0..40 min: a wave's replacements spread over 40 min
        self.assertEqual(fleet.coverage(cfg, [job("00", 60)], {}, NOW, horizon=horizon), 1)
        self.assertEqual(fleet.coverage(cfg, [job("09", 60)], {}, NOW, horizon=horizon), 0)
        self.assertEqual(fleet.coverage(cfg, [job("13", 60)], {}, NOW, horizon=horizon), 1)  # class 3: +13.3 min
        self.assertEqual(fleet.stagger_s(cfg, job("19", 0)), 40 * 60)
        self.assertEqual(fleet.stagger_s(dict(cfg, stagger_min=0), job("19", 0)), 0)
        self.assertEqual(fleet.stagger_s(cfg, {"name": "other-job"}), 0)

    def test_waits_are_recorded_once_per_job(self):
        self.remember({"job_id": "2100", "name": "swe-task-student-w00", "slot": "00", "partition": "batch",
                       "state": "PENDING", "submitted_at": NOW - 5000})
        squeue = lines(sq(2100, "swe-task-student-w00", start=NOW - 200, end=NOW + 4 * 3600 - 200))
        self.reconcile(self.fleet(squeue, config={"target": 1}))
        self.reconcile(self.fleet(squeue))
        waits = json.loads((self.root / "state/waits.json").read_text())
        self.assertEqual([item["wait_s"] for item in waits], [4800])

    def test_unreadable_config_is_never_replaced_by_exclusions(self):
        squeue = lines(sq(2200, "swe-task-student-w00", node="pool0-02200", start=NOW - 30 * 60, end=NOW + 3 * 3600))
        instance = self.fleet(squeue, config={"target": 1, "daemon_partition": "cpu_short"})
        original, calls = fleet.read_json, []

        def flaky(path, default=None):
            if str(path).endswith("fleet.json"):
                calls.append(path)
                if len(calls) > 1:
                    return default
            return original(path, default)
        with mock.patch.object(fleet, "read_json", flaky):
            self.reconcile(instance)
        self.assertEqual(json.loads((self.root / "fleet.json").read_text())["daemon_partition"], "cpu_short")
        self.assertEqual(len(self.events("exclusion_not_saved")), 1)

    # -- review fixes 2026-11-02 (F6, F10) --

    def test_scancel_never_runs_without_job_ids(self):
        """F6: `scancel --user=<user>` without job ids would cancel every job of the user."""
        slurm = FakeSlurm()
        for ids in ([], set(), ()):
            for pending_only in (False, True):
                slurm.scancel("hengyuf", ids, pending_only=pending_only)
        self.assertEqual(slurm.calls, [])
        slurm.scancel("hengyuf", {"124", "123"}, pending_only=True)
        self.assertEqual(slurm.calls, [["scancel", "--user=hengyuf", "--state=PENDING", "123", "124"]])

    def test_unlistable_registry_aborts_the_pass_instead_of_mass_cancelling(self):
        """F10: with registry/ read as empty, every running worker would be a startup without a record, cancelled as
        stuck after stuck_startup_min, its node excluded, and the breaker opened. The pass acts on nothing instead."""
        rows = lines(*[sq(950 + i, "swe-task-student-w%02d" % i, node="pool0-%05d" % (950 + i), start=NOW - 3600,
                          end=NOW + 3 * 3600) for i in range(4)])
        for i in range(4):
            self.register(entry(950 + i, "ready", node="pool0-%05d" % (950 + i)))
        self.reconcile(self.fleet(rows, config={"target": 4}))
        files = [self.root / "fleet.json"] + [self.root / "state" / name for name in
                                              ("jobs.json", "breaker.json", "waits.json", "status.json")]
        before = {path: path.read_bytes() for path in files}
        registry = self.root / "registry"
        os.chmod(registry, 0)
        self.addCleanup(os.chmod, registry, 0o755)
        if os.access(registry, os.R_OK):
            self.skipTest("the directory mode does not bind this user (root)")
        self.now += 30 * 60
        instance = self.fleet(rows)
        with self.assertRaises(fleet.RegistryError):
            self.reconcile(instance)
        self.assertEqual(self.slurm.calls, [["squeue", "-h", "-u", "hengyuf", "-o", fleet.SQUEUE_FORMAT]])
        self.assertEqual({path: path.read_bytes() for path in files}, before)
        with quiet():  # the daemon skips such passes and says so; it neither acts nor dies
            self.assertEqual(instance.daemon(interval=1, until=time.time() + 1.5), 0)
        self.assertEqual({call[0] for call in self.slurm.calls}, {"squeue"})
        skipped = self.events("pass_skipped")
        self.assertEqual(len(skipped), 1)
        self.assertTrue(skipped[0]["error"].startswith("RegistryError: cannot list"), skipped[0])
        self.assertEqual([self.events(kind) for kind in ("cancel", "worker_failure", "breaker_open")], [[], [], []])
        with self.assertRaises(fleet.RegistryError):  # `status` reports it (main: "error: ...", status 1)
            fleet.command_status(instance, argparse.Namespace(json=False))

    def test_a_registry_listing_failure_after_the_decisions_keeps_the_pass_listing(self):
        self.register(entry(960, "ready", node="pool0-00960"))
        rows = lines(sq(960, "swe-task-student-w00", node="pool0-00960", start=NOW - 3600, end=NOW + 3 * 3600))
        instance = self.fleet(rows, config={"target": 1})
        real, calls = fleet.read_registry, []

        def flaky(directory):
            calls.append(directory)
            if len(calls) > 1:
                raise fleet.RegistryError("cannot list %s: (test)" % directory)
            return real(directory)
        with mock.patch.object(fleet, "read_registry", flaky):
            self.reconcile(instance)
        self.assertEqual(len(calls), 2)
        status = json.loads((self.root / "state/status.json").read_text())
        self.assertEqual([(row["job_id"], row["registry"]) for row in status["jobs"]], [("960", "ready")])


def registry_writer_code():
    text = SERVE_STUDENT.read_text()
    return re.search(r"REGISTRY_PY=\$\(cat <<'PYEOF'\n(.*?)\nPYEOF\n\)", text, re.S).group(1)


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="registry-test-"))
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.code = registry_writer_code()

    def write(self, **overrides):
        env = {"W_JOB": "123", "W_SLOT": "04", "W_NODE": "pool0-00001", "W_IP": "10.1.2.3", "W_PORT": "8000",
               "W_SERVED": "policy", "W_MODEL": "/m", "W_MAX_LEN": "262144", "W_CAP": "131072", "W_SEED": "123",
               "W_STARTED": "1800000000", "W_READY_AT": "", "W_END": "1800014400", "W_DRAIN_AT": "1800011700",
               "W_HEARTBEAT": "1800000100", "W_STATUS": "starting", "W_REASON": "", "W_PARITY": "",
               "W_KV_TOKENS": "", "W_PARTITION": "batch", "W_VLLM_LOG": "/l", "W_EXCLUDE_NODE": ""}
        env.update(overrides)
        subprocess.run([sys.executable, "-I", "-S", "-c", self.code, str(self.dir / "123.json")], env=env, check=True)

    def test_record_schema(self):
        self.write()
        record = json.loads((self.dir / "123.json").read_text())
        for key, kind in CONTRACT_KEYS.items():
            self.assertIsInstance(record[key], kind, key)
        self.assertEqual(record["base_url"], "http://10.1.2.3:8000/v1")
        self.assertEqual((record["ready_at"], record["reason"]), (None, None))
        self.assertNotIn("exclude_node", record)
        self.write(W_STATUS="failed", W_REASON='quote " backslash \\ newline\n', W_EXCLUDE_NODE="0",
                   W_READY_AT="1800000300", W_KV_TOKENS="7156630", W_PARITY="PASS")
        record = json.loads((self.dir / "123.json").read_text())
        self.assertEqual((record["status"], record["reason"], record["exclude_node"], record["ready_at"],
                          record["kv_cache_tokens"]),
                         ("failed", 'quote " backslash \\ newline\n', False, 1800000300, 7156630))
        self.assertEqual([path.name for path in self.dir.iterdir()], ["123.json"])  # no temp file left

    def test_reader_skips_partial_temporary_and_foreign_files(self):
        self.write(W_STATUS="ready")
        (self.dir / "124.json").write_text('{"job_id": "124", "sta')  # partial
        (self.dir / ".125.json.77.tmp").write_text("{}")  # writer's temp file
        (self.dir / "abc.json").write_text('{"job_id": "abc"}')
        (self.dir / "126.json").write_text('{"job_id": "999"}')  # not <job_id>.json
        (self.dir / "127.json").write_text('[1, 2]')
        self.assertEqual(list(fleet.read_registry(self.dir)), ["123"])

    def test_unlistable_directory_raises_and_one_bad_record_is_skipped(self):
        """F10: a directory that cannot be listed raises RegistryError (it never reads as empty); a single unreadable
        record still costs only its own entry."""
        self.write(W_STATUS="ready")
        (self.dir / "128.json").mkdir()  # not a file
        (self.dir / "129.json").write_text(json.dumps({"job_id": "129", "status": "ready"}))
        os.chmod(self.dir / "129.json", 0)
        self.addCleanup(os.chmod, self.dir / "129.json", 0o644)
        expected = ["123"] + (["129"] if os.access(self.dir / "129.json", os.R_OK) else [])
        self.assertEqual(list(fleet.read_registry(self.dir)), expected)
        with self.assertRaises(fleet.RegistryError):
            fleet.read_registry(self.dir / "missing")
        with mock.patch.object(fleet.os, "scandir", side_effect=OSError(errno.EIO, "Input/output error")):
            with self.assertRaises(fleet.RegistryError) as caught:
                fleet.read_registry(self.dir)
        self.assertIn("Input/output error", str(caught.exception))

    def test_concurrent_rewrites_are_never_seen_partial(self):
        self.write()
        statuses = ("ready", "draining")
        done = threading.Event()

        def writer():
            for index in range(40):
                self.write(W_STATUS=statuses[index % 2], W_REASON="x" * (index * 500))
            done.set()

        thread = threading.Thread(target=writer)
        thread.start()
        reads = 0
        while not done.is_set():
            record = json.loads((self.dir / "123.json").read_text())  # raises on a partial file
            self.assertIn(record["status"], statuses + ("starting",))
            reads += 1
        thread.join()
        self.assertGreater(reads, 50)


def bash_array(path, name, variables):
    """The elements of the bash array `name=( ... )` in `path`, with the given variables set."""
    block = re.search(r"^%s=\(\n.*?^\)$" % name, Path(path).read_text(), re.S | re.M).group(0)
    prelude = "".join("%s=%s\n" % (key, shlex.quote(value)) for key, value in variables.items())
    script = prelude + block + '\nprintf "%%s\\n" "${%s[@]}"\n' % name
    return subprocess.run(["bash", "-c", script], stdout=subprocess.PIPE, universal_newlines=True,
                          check=True).stdout.splitlines()


def eval_runtime_env():
    """What serve_vllm.sh exports for vLLM (cache-dir exports left out: they would create EVAL's per-node dirs)."""
    exports = [line for line in SERVE_VLLM.read_text().splitlines() if line.startswith("export ") and "CACHE" not in line]
    script = "RUNTIME=%s\nPATH=/usr/bin:/bin\n%s\nenv -0\n" % (shlex.quote(str(RUNTIME)), "\n".join(exports))
    output = subprocess.run(["env", "-i", "bash", "-c", script], stdout=subprocess.PIPE, check=True).stdout
    return dict(item.split("=", 1) for item in output.decode().split("\0") if "=" in item)


class ParityTest(unittest.TestCase):
    def test_vllm_argv_equals_the_eval_serving_args(self):
        values = {"MODEL": "/models/x/hf", "SERVED_MODEL_NAME": "policy", "SEED": "20371234", "PORT": "8000"}
        eval_args = bash_array(SERVE_VLLM, "ARGS", values)
        ours = bash_array(SERVE_STUDENT, "VLLM_ARGS", dict(values, MAX_MODEL_LEN="262144"))
        self.assertGreater(len(eval_args), 40)
        self.assertEqual(ours, eval_args)


FAKE_SERVER = r'''
import http.server, json, os, subprocess, sys, time
port = int(sys.argv[1])
mode = os.environ.get("FAKE_MODE", "ok")
if os.environ.get("FAKE_ENV_DUMP"):
    with open(os.environ["FAKE_ENV_DUMP"], "w") as stream:
        json.dump(dict(os.environ), stream)
# A child that leaves the process group (like an orphaned worker): only the environment tag can find it.
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"], start_new_session=True)
print("(APIServer pid=1) INFO non-default args: {}", flush=True)
if mode == "crash":
    time.sleep(0.5)
    print("(Worker_TP3 pid=4) ERROR RuntimeError: CUDA error: an illegal memory access was encountered", flush=True)
    sys.exit(1)
while mode == "hang":
    time.sleep(1)
time.sleep(0.5)
print("(Worker_TP0 pid=2) INFO [default_loader.py:430] Loading weights took 0.50 seconds", flush=True)
print("(EngineCore pid=3) INFO [kv_cache_utils.py:2032] GPU KV cache size: 7,156,630 tokens, ...", flush=True)
time.sleep(1)
started = time.time()
sick_after = float(os.environ.get("FAKE_SICK_AFTER_S", "0"))


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        sick = sick_after and time.time() - started > sick_after
        ok = self.path == "/v1/models" or (self.path == "/health" and not sick)
        body = b'{"object": "list", "data": [{"id": "policy"}]}' if self.path == "/v1/models" else b""
        self.send_response(200 if ok else 500)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
'''


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def tagged_processes(tag):
    found = []
    for path in glob.glob("/proc/[0-9]*/environ"):
        try:
            with open(path, "rb") as stream:
                items = stream.read().split(b"\0")
        except OSError:
            continue
        if ("STUDENT_WORKER_TAG=" + tag).encode() in items:
            found.append(path.split("/")[2])
    return found


class ServeStudentTest(unittest.TestCase):
    """serve_student.sh with STUDENT_TEST_SERVER (a fake vLLM) and STUDENT_ROOT/STUDENT_PORT pointing at a temp dir."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="serve-student-test-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.fake = self.root / "fake_vllm.py"
        self.fake.write_text(FAKE_SERVER)

    def run_worker(self, job, end_in, timeout=120, term_on=None, **env):
        variables = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(self.root), "LANG": "C.UTF-8",
                     "SLURM_JOB_ID": job, "SLURM_JOB_NAME": "swe-task-student-w42", "SLURMD_NODENAME": "testnode",
                     "SLURM_JOB_PARTITION": "batch", "STUDENT_ROOT": str(self.root), "STUDENT_PORT": str(free_port()),
                     "STUDENT_TEST_SERVER": str(self.fake), "STUDENT_JOB_START": str(int(time.time())),
                     "STUDENT_END_EPOCH": str(int(time.time()) + end_in) if end_in else "",
                     "STUDENT_HEALTH_INTERVAL_S": "1", "STUDENT_STOP_MARGIN_S": "2"}
        variables.update(env)
        process = subprocess.Popen(["bash", str(SERVE_STUDENT)], env=variables, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        output = []
        reader = threading.Thread(target=lambda: output.extend(process.stdout))
        reader.start()
        path, seen, deadline, signalled = self.root / "registry" / (job + ".json"), [], time.time() + timeout, False
        while process.poll() is None and time.time() < deadline:
            record = fleet.read_json(path)
            if record and (not seen or seen[-1] != record["status"]):
                seen.append(record["status"])
            if term_on and not signalled and seen and seen[-1] == term_on:
                process.send_signal(signal.SIGTERM)
                signalled = True
            time.sleep(0.1)
        timed_out = process.poll() is None
        if timed_out:  # let the script clean up first; SIGKILL would orphan the fake server
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        reader.join(10)
        process.stdout.close()
        text = b"".join(output).decode(errors="replace")
        record = fleet.read_json(path)
        if record and (not seen or seen[-1] != record["status"]):
            seen.append(record["status"])
        time.sleep(0.5)
        leftovers = tagged_processes("swe-task-student-%s-0" % job)
        for pid in leftovers:  # never leave test processes behind on the login node
            os.kill(int(pid), signal.SIGKILL)
        self.assertFalse(timed_out, "worker did not exit within %d s:\n%s" % (timeout, text))
        self.assertEqual(leftovers, [], "leftover vLLM processes:\n" + text)
        return process.returncode, seen, record, text

    def test_lifecycle_ready_draining_stopped_and_runtime_env(self):
        dump = self.root / "env.json"
        code, seen, record, output = self.run_worker("990000001", end_in=22, STUDENT_DRAIN_S="10",
                                                     FAKE_ENV_DUMP=str(dump))
        self.assertEqual(code, 0, output)
        self.assertEqual(seen, ["starting", "ready", "draining", "stopped"], output)
        self.assertEqual(record["reason"], "walltime")
        for key, kind in CONTRACT_KEYS.items():
            self.assertIsInstance(record[key], kind, key)
        self.assertEqual((record["slot"], record["node"], record["seed"], record["served_model_name"],
                          record["max_model_len"], record["per_call_cap"], record["kv_cache_tokens"]),
                         ("42", "testnode", 990000001 % 2 ** 31, "policy", 262144, 131072, 7156630))
        self.assertEqual(record["base_url"], "http://%s:%d/v1" % (record["ip"], record["port"]))
        self.assertEqual(record["drain_at"], record["end_epoch"] - 10)
        self.assertIn(record["parity"], ("FAIL", "ERROR"))  # a fake log has no engine lines to compare
        self.assertTrue((self.root / "logs/parity-990000001.txt").exists())
        self.assertIn("Loading weights took", (self.root / "logs/vllm-990000001.log").read_text())
        env, expected = json.loads(dump.read_text()), eval_runtime_env()
        for key in ("VLLM_USE_V2_MODEL_RUNNER", "NRL_SC_IID_LOGPROBS", "VLLM_WORKER_MULTIPROC_METHOD",
                    "VLLM_ALLOW_INSECURE_SERIALIZATION", "OMP_NUM_THREADS", "PYTHONUNBUFFERED", "NCCL_DEBUG",
                    "LD_LIBRARY_PATH"):
            self.assertEqual(env.get(key), expected[key], key)
        self.assertTrue(env["PATH"].startswith("%s/bin:%s/envs/vllm/bin:" % (RUNTIME, RUNTIME)))
        compile_cache = self.root / "cache/compile" / socket.gethostname()
        self.assertEqual((env["HF_HOME"], env["VLLM_CACHE_ROOT"], env["TRITON_CACHE_DIR"]),
                         (str(self.root / "cache/hf_home"), str(compile_cache / "vllm"), str(compile_cache / "triton")))
        self.assertEqual(env["STUDENT_WORKER_TAG"], "swe-task-student-990000001-0")
        self.assertFalse([key for key in env if key.startswith("W_")])  # registry variables stay out of vLLM

    def test_startup_hang_fails_with_evidence_and_suspect_node(self):
        code, seen, record, output = self.run_worker("990000002", end_in=4000, FAKE_MODE="hang",
                                                     STUDENT_LOAD_TIMEOUT_S="3")
        self.assertEqual(code, 4, output)
        self.assertEqual(seen, ["starting", "failed"])
        self.assertIn("startup hang", record["reason"])
        self.assertIs(record["exclude_node"], True)
        self.assertTrue((self.root / "logs/hang-990000002.txt").exists())

    def test_crash_during_startup_records_the_error(self):
        code, seen, record, output = self.run_worker("990000003", end_in=4000, FAKE_MODE="crash")
        self.assertEqual(code, 3, output)
        self.assertEqual(seen, ["starting", "failed"])
        self.assertIn("illegal memory access", record["reason"])

    def test_health_failures_mark_unhealthy(self):
        code, seen, record, output = self.run_worker("990000004", end_in=4000, FAKE_SICK_AFTER_S="8")
        self.assertEqual(code, 6, output)
        self.assertEqual(seen, ["starting", "ready", "unhealthy"])
        self.assertIs(record["exclude_node"], True)
        self.assertLess(record["heartbeat_at"], time.time() - 2)  # heartbeat = last passing /health

    def test_sigterm_stops_cleanly(self):
        code, seen, record, output = self.run_worker("990000005", end_in=4000, term_on="ready")
        self.assertEqual(code, 143, output)
        self.assertEqual(seen, ["starting", "ready", "stopped"])
        self.assertIn("SIGTERM", record["reason"])

    def test_hung_registry_writes_time_out_and_are_retried(self):
        # 2026-10-20 15:22-15:26 a Lustre stall hung the registry writes of every bridge. Each attempt now gets
        # STUDENT_REG_TIMEOUT_S and a failed one is retried once; the worker keeps serving either way.
        shim, marks = self.root / "bin", self.root / "stalled"
        shim.mkdir()
        path = os.environ.get("PATH", "/usr/bin:/bin")
        (shim / "python3").write_text(
            "#!/bin/bash\n"
            "# the first 3 registry writes hang like a stalled filesystem; anything else runs the real python3\n"
            'if [[ "$*" == *"os.replace(tmp, path)"* ]]; then\n'
            '  for n in 1 2 3; do mkdir "%s.$n" 2>/dev/null && exec sleep 30; done\n'
            "fi\n"
            'exec %s "$@"\n' % (marks, shutil.which("python3", path=path)))
        (shim / "python3").chmod(0o755)
        code, seen, record, output = self.run_worker("990000007", end_in=26, STUDENT_REG_TIMEOUT_S="1",
                                                     PATH="%s:%s" % (shim, path))
        self.assertEqual(code, 0, output)
        # the startup write fails twice (both attempts hang), the first serving write succeeds on its retry
        failures = re.findall(r"WARNING: registry write failed \(status (\w+), attempt (\d)/2\)", output)
        self.assertEqual(failures, [("starting", "1"), ("starting", "2"), ("draining", "1")], output)
        self.assertEqual(seen, ["draining", "stopped"])
        self.assertEqual(record["reason"], "walltime")

    def test_unknown_end_time_is_an_environment_failure(self):
        code, seen, record, output = self.run_worker("990000006", end_in=None)
        self.assertEqual(code, 2, output)
        self.assertEqual(seen, ["failed"])
        self.assertIs(record["exclude_node"], False)


if __name__ == "__main__":
    unittest.main()
