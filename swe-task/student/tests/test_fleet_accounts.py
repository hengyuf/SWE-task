"""Offline tests of fleet.py's worker accounts (dual-account candidates, 2026-11-02): no Slurm job is submitted or
cancelled. Run by explicit path (TMPDIR = a private directory):

    TMPDIR=<private tmp> /usr/bin/python3 -I -B <student>/tests/test_fleet_accounts.py

AccountsTest      new slots (one job, its sibling after pair_delay_min), settlement, completion, coverage/max_jobs,
                  waits, breaker probe, failing account, exclusions, stop, status, dry run, rollback, RELOAD flag,
                  worker-accounts command.
ReviewFixesTest   second-round review fixes: F1 stale sibling marks, F2 a keeper police() is cancelling, F3 account
                  pairs only, F7 a settled candidate is never a failure (F6, F10: tests/test_fleet.py).
EquivalenceTest   with one worker account (worker_accounts absent, as in fleet.json before the switch), the new
                  fleet.py makes exactly the Slurm calls of the previous one (FLEET_ORIG, default
                  <student>/fleet.py.pre-dual-account) over randomized multi-pass histories of a modelled queue
                  (FLEET_EQUIV_SEEDS, default 300, ~4 min); skipped when that file is missing.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import collections
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
import unittest

STUDENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT))
import fleet  # noqa: E402

ORIG = Path(os.environ.get("FLEET_ORIG") or STUDENT / "fleet.py.pre-dual-account")
NOW = 1800000000
AD, RL = "nemotron_agents_dev", "nemotron_rl_algo"
DUAL = {"worker_accounts": [AD, RL]}
UID = os.getuid()
COMMENT = fleet.comment_json(fleet.DEFAULTS)


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class Done:
    def __init__(self, stdout="", stderr=""):
        self.stdout, self.stderr, self.returncode = stdout, stderr, 0


class FakeSlurm(fleet.Slurm):
    """Canned squeue/sacct; records every call; sbatch hands out ids (failing for accounts in fail_accounts). Ids
    never repeat across instances (a test makes one instance per pass)."""
    last_id = 30000000

    def __init__(self, squeue="", sacct="", fail=(), fail_accounts=()):
        super().__init__(bin_dir="/nonexistent")
        self.squeue_text, self.sacct_text, self.fail = squeue, sacct, set(fail)
        self.fail_accounts = set(fail_accounts)
        self.calls, self.comments, self.accepted = [], {}, []

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
            account = [arg[len("--account="):] for arg in argv if arg.startswith("--account=")][0]
            if account in self.fail_accounts:
                raise fleet.SlurmError("sbatch exited 1: Invalid account or account/partition combination (test)")
            FakeSlurm.last_id += 1
            comment = [arg[len("--comment="):] for arg in argv if arg.startswith("--comment=")]
            self.comments[str(FakeSlurm.last_id)] = comment[0] if comment else "(null)"
            self.accepted.append(account)
            return Done("%d\n" % FakeSlurm.last_id)
        if argv[0] in ("scancel", "scontrol"):
            return Done()
        raise AssertionError("unexpected Slurm command %r" % argv)

    def submitted(self):
        return [call for call in self.calls if call[0] == "sbatch"]

    def accounts(self):
        return [arg[len("--account="):] for call in self.submitted() for arg in call if arg.startswith("--account=")]

    def names(self):
        return [arg[len("--job-name="):] for call in self.submitted() for arg in call if arg.startswith("--job-name=")]

    def scancels(self):
        return [call for call in self.calls if call[0] == "scancel"]

    def cancelled(self):
        return [arg for call in self.scancels() for arg in call[1:] if not arg.startswith("--")]


def sq(job_id, name, state="RUNNING", node=None, start=None, end=None, reason="None", user="hengyuf",
       partition="batch", account=AD, time_left=None):
    """One squeue line in fleet's format; a pending job's time left is its time limit, as squeue prints it."""
    if state in fleet.PENDING:
        node, end = "", None
    node = "pool0-%05d" % (int(job_id) % 100000) if node is None else node
    left = time_left or ("4:00:00" if state in fleet.PENDING else "2:00:00")
    fields = [str(job_id), user, state, partition, node, "1:00:00", left,
              str(start) if start else "N/A", str(end) if end else "N/A", reason, account, name]
    return "|".join(fields)


def lines(*rows):
    return "".join(row + "\n" for row in rows)


def w(slot):
    return "swe-task-student-w%02d" % slot


def entry(job_id, status, node="pool0-00001", heartbeat=None, ready_at=None, end=None, started_at=None, **extra):
    heartbeat = NOW if heartbeat is None else heartbeat
    end = NOW + 3 * 3600 if end is None else end
    record = {"job_id": str(job_id), "slot": "00", "node": node, "ip": "10.0.0.1", "port": 8000,
              "base_url": "http://10.0.0.1:8000/v1", "served_model_name": "policy", "model_path": "/m",
              "max_model_len": 262144, "per_call_cap": 131072, "seed": 1,
              "started_at": NOW - 3600 if started_at is None else started_at,
              "ready_at": (NOW - 600 if status in ("ready", "draining") else None) if ready_at is None else ready_at,
              "end_epoch": end, "drain_at": end - 2700, "heartbeat_at": heartbeat, "status": status, "reason": None}
    record.update(extra)
    return record


def sacct_row(job_id, state, start=None, end=None, node="None assigned", exit_code="0:0", admin=""):
    return "%s|hengyuf|%s|%s|%s|%s|00:00:00|%s|%s\n" % (job_id, state, exit_code, start or "None", end or NOW,
                                                        node, admin)


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="fleet-accounts-test-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        (self.root / "registry").mkdir()
        self.now = NOW

    def fleet(self, squeue="", sacct="", fail=(), config=None, fail_accounts=(), module=fleet):
        self.slurm = (FakeSlurm if module is fleet else type("OrigFake", (FakeSlurm,), {}))(
            squeue, sacct, fail, fail_accounts)
        if config is not None:
            cfg = json.loads(json.dumps(module.DEFAULTS))
            cfg.update(config)
            module.write_json(self.root / "fleet.json", cfg)
        return module.Fleet(self.root, self.slurm, clock=lambda: self.now)

    def reconcile(self, instance, dry_run=False):
        with quiet():
            return instance.reconcile(dry_run=dry_run)

    def register(self, record):
        fleet.write_json(self.root / "registry" / ("%s.json" % record["job_id"]), record)

    def remember(self, *records):
        fleet.write_json(self.root / "state/jobs.json", {record["job_id"]: record for record in records})

    def known(self):
        return json.loads((self.root / "state/jobs.json").read_text())

    def status(self):
        return json.loads((self.root / "state/status.json").read_text())

    def events(self, kind=None):
        path = self.root / "state/events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return [event for event in events if kind is None or event["event"] == kind]

    def queue_of(self, *extra, state="PENDING", reason="Priority"):
        """squeue of every job in jobs.json (as submitted: pending) plus extra rows."""
        rows = [sq(job_id, item["name"], state=state, reason=reason, account=item["account"])
                for job_id, item in sorted(self.known().items())]
        return lines(*(rows + list(extra)))


class AccountsTest(Base):
    # -- configuration --

    def test_config_defaults_to_the_account_and_is_validated(self):
        cfg = json.loads(json.dumps(fleet.DEFAULTS))
        self.assertEqual(fleet.worker_accounts(cfg), [AD])
        self.assertEqual(fleet.worker_accounts(dict(cfg, **DUAL)), [AD, RL])
        self.assertEqual(fleet.worker_accounts(dict(cfg, worker_accounts=[RL])), [RL])
        for bad in (RL, [AD, AD], ["a b"], ["a,b"], [""], [3], {"a": 1}):
            with self.assertRaises(fleet.ConfigError, msg=repr(bad)):
                fleet.check_config(dict(cfg, worker_accounts=bad))
        self.assertEqual(cfg["pair_delay_min"], 15)  # F5
        fleet.check_config(dict(cfg, pair_delay_min=0))
        for bad in (-1, 1.5, "15", True, None):
            with self.assertRaises(fleet.ConfigError, msg=repr(bad)):
                fleet.check_config(dict(cfg, pair_delay_min=bad))
        # the deployed config is valid; it is single-account only while it has no worker_accounts (F4: it may have)
        live = json.loads((STUDENT / "fleet.json").read_text())
        fleet.check_config(dict(cfg, **live))
        if not live.get("worker_accounts"):
            self.assertEqual(fleet.worker_accounts(dict(cfg, **live)), [live["account"]])

    # -- new slots --

    def test_new_slot_gets_one_job_and_its_sibling_after_pair_delay_min(self):
        instance = self.fleet("", config=dict(DUAL, target=2))
        self.reconcile(instance)
        self.assertEqual((self.slurm.names(), self.slurm.accounts()), ([w(0), w(1)], [AD, AD]))  # F5: one job each
        firsts = self.slurm.submitted()
        known = self.known()
        self.assertEqual(sorted((item["slot"], item["account"], item["slot_submitted_at"] == item["submitted_at"])
                                for item in known.values()), [("00", AD, True), ("01", AD, True)])
        counts = self.status()["counts"]
        self.assertEqual((counts["coverage"], counts["pending_slots"], counts["pending"], counts["total"]),
                         (2, 2, 2, 2))
        self.assertEqual(self.status()["config"]["pair_delay_min"], 15)
        self.now += 14 * 60  # queued less than pair_delay_min: no sibling yet
        self.reconcile(self.fleet(self.queue_of()))
        self.assertEqual((self.slurm.submitted(), self.slurm.scancels()), ([], []))
        self.now += 60  # 15 min after the first submission: one sibling per slot on the other account
        self.reconcile(self.fleet(self.queue_of()))
        self.assertEqual((self.slurm.names(), self.slurm.accounts()), ([w(0), w(1)], [RL, RL]))
        for first, second in zip(firsts, self.slurm.submitted()):  # same name, comment, exclusions; account aside
            self.assertEqual([arg for arg in first if not arg.startswith("--account=")],
                             [arg for arg in second if not arg.startswith("--account=")])
        for call in firsts + self.slurm.submitted():  # no candidate tag: the idle-reaper exemption stays valid
            self.assertEqual([arg for arg in call if arg.startswith("--comment")], ["--comment=" + COMMENT])
            self.assertIn("--export=NONE", call)
        self.assertEqual(self.events("comment_mismatch"), [])
        known = self.known()
        self.assertEqual(sorted((item["slot"], item["account"], item["submit_account"]) for item in known.values()),
                         [("00", AD, AD), ("00", RL, RL), ("01", AD, AD), ("01", RL, RL)])
        for slot in ("00", "01"):  # a pair's wait counts from its first candidate
            pair = [item for item in known.values() if item["slot"] == slot]
            self.assertEqual({item["slot_submitted_at"] for item in pair}, {min(item["submitted_at"] for item in pair)})
        counts = self.status()["counts"]
        self.assertEqual((counts["coverage"], counts["pending_slots"], counts["pending"], counts["total"]),
                         (2, 2, 4, 4))
        self.assertEqual([event["account"] for event in self.events("submit")], [AD, AD, RL, RL])
        self.assertEqual([event["reason"][:15] for event in self.events("submit")][2:], ["pair completion"] * 2)
        self.reconcile(self.fleet(self.queue_of()))  # two pending pairs cover target 2: nothing more
        self.assertEqual((self.slurm.submitted(), self.slurm.scancels()), ([], []))

    def test_a_fast_queue_starts_the_slot_alone(self):
        self.reconcile(self.fleet("", config=dict(DUAL, target=1)))
        (job_id,) = self.known()
        start = NOW + 120  # starts 2 min after its submission: no sibling ever, so no duplicate start
        row = sq(job_id, w(0), node="pool0-00777", start=start, end=start + 4 * 3600)
        for minute in range(3, 40):
            self.now = NOW + 60 * minute
            if self.now >= start + 480:
                self.register(entry(job_id, "ready", node="pool0-00777", started_at=start, heartbeat=self.now))
            self.reconcile(self.fleet(lines(row)))
            self.assertEqual((self.slurm.submitted(), self.slurm.scancels()), ([], []), minute)
        self.assertEqual(len(self.events("submit")), 1)

    def test_pair_delay_counts_pending_slots_from_their_first_submission(self):
        self.remember({"job_id": "470", "name": w(0), "slot": "00", "partition": "batch", "account": AD,
                       "submit_account": AD, "state": "PENDING", "submitted_at": NOW - 1200,
                       "slot_submitted_at": NOW - 1200})
        held = sq(470, w(0), state="PENDING", reason="JobHeldUser")
        self.reconcile(self.fleet(lines(held), config=dict(DUAL, target=1)))
        self.assertEqual((self.slurm.names(), self.slurm.accounts()), ([w(1)], [AD]))  # held: no sibling (and no cover)
        self.now += 60  # released: 21 min after its submission, so its sibling comes at once; w01 queued 1 min only
        self.reconcile(self.fleet(self.queue_of()))
        self.assertEqual((self.slurm.names(), self.slurm.accounts()), ([w(0)], [RL]))
        # a job the daemon did not submit (submitted_at unknown) counts from when it was first seen
        shutil.rmtree(self.root / "state")
        rows = lines(sq(480, w(5), state="PENDING", reason="Priority"))
        self.reconcile(self.fleet(rows, config=dict(DUAL, target=1)))
        self.assertEqual(self.slurm.submitted(), [])
        self.now += 15 * 60
        self.reconcile(self.fleet(rows))
        self.assertEqual((self.slurm.names(), self.slurm.accounts()), ([w(5)], [RL]))
        # pair_delay_min 0: the sibling at the next pass (never in the pass that submits the slot)
        shutil.rmtree(self.root / "state")
        self.reconcile(self.fleet("", config=dict(DUAL, target=1, pair_delay_min=0)))
        self.assertEqual(self.slurm.accounts(), [AD])
        self.reconcile(self.fleet(self.queue_of()))
        self.assertEqual(self.slurm.accounts(), [RL])

    def test_pending_pairs_count_once_for_coverage_max_jobs_and_submissions_per_pass(self):
        rows = [sq(100 + 2 * i + k, w(i), state="PENDING", reason="Priority", account=account)
                for i in range(4) for k, account in enumerate((AD, RL))]  # 4 pending pairs = 4 slots, 8 jobs
        instance = self.fleet(lines(*rows), config=dict(DUAL, target=7, max_jobs=6, max_submits_per_pass=10))
        self.reconcile(instance)
        self.assertEqual(self.slurm.names(), [w(4), w(5)])  # max_jobs 6 nodes - 4 pending pairs = 2 (one job each)
        shutil.rmtree(self.root / "state")  # forget them (else they are unconfirmed jobs holding slots 04, 05)
        self.reconcile(self.fleet(lines(*rows), config=dict(DUAL, target=10, max_jobs=24, max_submits_per_pass=3)))
        self.assertEqual(self.slurm.names(), [w(4), w(5), w(6)])  # 3 slots per pass
        cfg = dict(json.loads(json.dumps(fleet.DEFAULTS)), **DUAL)
        jobs = fleet.parse_squeue(lines(*rows))
        self.assertEqual((fleet.coverage(cfg, jobs, {}, NOW), fleet.node_count(cfg, jobs, {})), (4, 4))

    def test_max_submits_per_pass_counts_new_slots_and_completions(self):
        self.remember(*[{"job_id": str(490 + i), "name": w(i), "slot": "%02d" % i, "partition": "batch",
                         "state": "PENDING", "submitted_at": NOW - 3600} for i in range(4)])
        rows = lines(*[sq(490 + i, w(i), state="PENDING", reason="Priority") for i in range(4)])
        self.reconcile(self.fleet(rows, config=dict(DUAL, target=6, max_submits_per_pass=3)))
        # coverage 4 of 6: two new slots, then one completion uses the rest of max_submits_per_pass
        self.assertEqual((self.slurm.names(), self.slurm.accounts()), ([w(4), w(5), w(0)], [AD, AD, RL]))

    # -- settlement --

    def test_backup_stays_while_the_winner_starts_then_goes_without_any_failure(self):
        self.remember(*[{"job_id": str(300 + i), "name": w(0), "slot": "00", "partition": "batch", "account": account,
                         "submit_account": account, "state": "PENDING", "submitted_at": NOW - 3600,
                         "slot_submitted_at": NOW - 3600} for i, account in enumerate((AD, RL))])
        winner = sq(301, w(0), account=RL, node="pool0-00301", start=NOW - 30, end=NOW + 4 * 3600 - 30)
        backup = sq(300, w(0), state="PENDING", reason="Priority")
        instance = self.fleet(lines(backup, winner), config=dict(DUAL, target=1))
        self.reconcile(instance)
        self.assertEqual(self.slurm.scancels(), [])  # the winner is still starting: the agents_dev job stays queued
        self.assertEqual(self.slurm.submitted(), [])  # the starting winner covers the slot
        self.assertEqual([(item["job_id"], item["wait_s"], item["account"])
                          for item in json.loads((self.root / "state/waits.json").read_text())],
                         [("301", 3570, RL)])  # from the slot's first submission
        self.assertEqual(self.status()["counts"]["coverage"], 1)
        self.now += 480  # serving now: the backup goes
        self.register(entry(301, "ready", node="pool0-00301", started_at=NOW - 30, ready_at=self.now - 10,
                            heartbeat=self.now - 5))
        self.reconcile(self.fleet(lines(backup, winner)))
        self.assertEqual(self.slurm.scancels(), [["scancel", "--user=hengyuf", "--state=PENDING", "300"]])
        self.assertEqual(self.slurm.submitted(), [])
        cancel = self.events("sibling_cancel")
        self.assertEqual([(event["job_id"], event["keeper"], event["keeper_account"], event["keeper_serving"])
                          for event in cancel], [("300", "301", RL, True)])
        self.reconcile(self.fleet(lines(backup, winner)))  # still listed a minute later: not cancelled again yet
        self.assertEqual(self.slurm.scancels(), [])
        self.assertEqual(self.status()["counts"]["coverage"], 1)
        self.now += 60  # gone: sacct says CANCELLED by us, never started
        self.reconcile(self.fleet(lines(winner), sacct_row(300, "CANCELLED by %d" % UID)))
        self.assertEqual([self.events(kind) for kind in ("worker_failure", "exclude_node", "breaker_open")],
                         [[], [], []])
        self.assertEqual([event["event"] for event in self.events() if event.get("incident")], [])
        self.assertEqual(instance.breaker()["failures"], [])
        self.assertEqual([(event["job_id"], event["sibling_of"], event["account"], event["cancelled_by"])
                          for event in self.events("job_ended")], [("300", "301", AD, str(UID))])
        self.assertEqual(self.slurm.submitted(), [])
        self.assertEqual(instance.config()["exclude_nodes"], fleet.DEFAULTS["exclude_nodes"])
        self.assertEqual(len(json.loads((self.root / "state/waits.json").read_text())), 1)

    def test_a_backup_survives_a_failed_startup_and_covers_the_slot(self):
        backup = sq(310, w(0), state="PENDING", reason="Priority")
        winner = sq(311, w(0), account=RL, node="pool0-00311", start=NOW - 300, end=NOW + 4 * 3600 - 300)
        self.register(entry(311, "failed", node="pool0-00311", started_at=NOW - 300, reason="startup hang",
                            exclude_node=True))
        instance = self.fleet(lines(backup, winner), config=dict(DUAL, target=1))
        self.reconcile(instance)
        self.assertEqual(self.slurm.scancels(), [])  # the failed one gets terminal_grace_min, the backup stays
        self.assertEqual(self.slurm.submitted(), [])  # the backup covers the slot
        self.assertIn("pool0-00311", instance.config()["exclude_nodes"])
        self.assertEqual([call[2] for call in self.slurm.calls if call[0] == "scontrol"], ["JobId=310"])
        self.assertEqual(self.status()["counts"]["coverage"], 1)
        self.now += 6 * 60
        self.reconcile(self.fleet(lines(backup, winner)))
        self.assertEqual(self.slurm.scancels(), [["scancel", "--user=hengyuf", "311"]])  # police, not settlement
        self.assertEqual((self.slurm.submitted(), self.events("sibling_cancel")), ([], []))
        # stuck startup of a winner: police cancels it (failure, exclusion); the backup still covers
        self.register(entry(321, "starting", node="pool0-00321", started_at=NOW - 1800))
        rows = [sq(320, w(1), state="PENDING", reason="Priority"),
                sq(321, w(1), account=RL, node="pool0-00321", start=self.now - 26 * 60, end=self.now + 3 * 3600)]
        self.reconcile(self.fleet(lines(*rows), config=dict(DUAL, target=1)))
        self.assertEqual(self.slurm.scancels(), [["scancel", "--user=hengyuf", "321"]])
        self.assertEqual(self.slurm.submitted(), [])
        self.assertIn("pool0-00321", instance.config()["exclude_nodes"])

    def test_two_starts_keep_the_earliest_and_a_serving_one_wins_over_a_starting_one(self):
        early = sq(330, w(0), node="pool0-00330", start=NOW - 120, end=NOW + 4 * 3600 - 120)
        late = sq(331, w(0), account=RL, node="pool0-00331", start=NOW - 60, end=NOW + 4 * 3600 - 60)
        instance = self.fleet(lines(early, late), config=dict(DUAL, target=1))
        self.reconcile(instance)
        self.assertEqual(self.slurm.scancels(), [["scancel", "--user=hengyuf", "331"]])
        self.assertEqual(self.slurm.submitted(), [])
        self.assertEqual([item["job_id"] for item in json.loads((self.root / "state/waits.json").read_text())],
                         [])  # no submitted_at known (adopted): no sample, and never one for the loser
        self.now += 60  # ended by our scancel while starting (SIGTERM -> "stopped"): no failure
        self.register(entry(331, "stopped", node="pool0-00331", started_at=NOW - 60))
        sacct = sacct_row(331, "CANCELLED by %d" % UID, start=NOW - 60, node="pool0-00331", exit_code="0:15")
        self.reconcile(self.fleet(lines(early), sacct))
        self.assertEqual(self.events("worker_failure"), [])
        self.assertTrue((self.root / "registry/archive/331.json").exists())
        # the same without any registry record (killed before serve_student.sh wrote one): no failure either
        self.remember({"job_id": "332", "name": w(2), "slot": "02", "account": RL, "state": "RUNNING",
                       "node": "pool0-00332", "start": NOW - 30, "sibling_of": "999"})
        self.reconcile(self.fleet("", sacct_row(332, "CANCELLED by %d" % UID, start=NOW - 30, node="pool0-00332"),
                                  config=dict(DUAL, target=0)))
        self.assertEqual(self.events("worker_failure"), [])
        # a serving candidate wins over an earlier start that is still starting
        self.register(entry(341, "ready", node="pool0-00341", started_at=NOW - 900))
        rows = [sq(340, w(1), node="pool0-00340", start=NOW - 1000, end=NOW + 3 * 3600),
                sq(341, w(1), account=RL, node="pool0-00341", start=NOW - 900, end=NOW + 3 * 3600)]
        self.reconcile(self.fleet(lines(*rows), config=dict(DUAL, target=1)))
        self.assertEqual(self.slurm.scancels(), [["scancel", "--user=hengyuf", "340"]])
        # two serving duplicates (a daemon outage): both stay and both cover
        self.register(entry(340, "ready", node="pool0-00340", started_at=NOW - 1000))
        self.reconcile(self.fleet(lines(*rows), config=dict(DUAL, target=2)))
        self.assertEqual((self.slurm.scancels(), self.slurm.submitted()), ([], []))

    def test_a_ready_duplicate_with_a_late_heartbeat_is_left_to_police(self):
        self.register(entry(380, "ready", node="pool0-00380", started_at=NOW - 1200, heartbeat=NOW - 5))
        self.register(entry(381, "ready", node="pool0-00381", started_at=NOW - 1100, heartbeat=NOW - 200))
        rows = lines(sq(380, w(0), node="pool0-00380", start=NOW - 1200, end=NOW + 3 * 3600),
                     sq(381, w(0), account=RL, node="pool0-00381", start=NOW - 1100, end=NOW + 3 * 3600))
        self.reconcile(self.fleet(rows, config=dict(DUAL, target=1)))
        self.assertEqual((self.slurm.scancels(), self.events("sibling_cancel")), ([], []))  # not routable, but ready
        self.now += 9 * 60 + 1  # stale for more than heartbeat_stale_min: police cancels it (no failure)
        self.register(entry(380, "ready", node="pool0-00380", started_at=NOW - 1200, heartbeat=self.now - 5))
        self.reconcile(self.fleet(rows))
        self.assertEqual(self.slurm.scancels(), [["scancel", "--user=hengyuf", "381"]])
        self.assertEqual(self.events("worker_failure"), [])

    def test_a_sibling_that_started_after_the_request_is_cancelled_at_once(self):
        self.register(entry(350, "ready", node="pool0-00350", started_at=NOW - 600))
        keeper = sq(350, w(0), node="pool0-00350", start=NOW - 600, end=NOW + 4 * 3600 - 600)
        self.reconcile(self.fleet(lines(keeper, sq(351, w(0), state="PENDING", reason="Priority", account=RL)),
                                  config=dict(DUAL, target=1)))
        self.assertEqual(self.slurm.scancels(), [["scancel", "--user=hengyuf", "--state=PENDING", "351"]])
        self.now += 60  # it started between squeue and scancel (--state=PENDING left it alone)
        self.reconcile(self.fleet(lines(keeper, sq(351, w(0), account=RL, node="pool0-00351", start=NOW + 30,
                                                   end=NOW + 30 + 4 * 3600))))
        self.assertEqual(self.slurm.scancels(), [["scancel", "--user=hengyuf", "351"]])
        self.assertEqual(self.slurm.submitted(), [])
        self.now += 60  # still running a minute later: not asked again before CANCEL_RETRY_S
        self.reconcile(self.fleet(lines(keeper, sq(351, w(0), account=RL, node="pool0-00351", start=NOW + 30,
                                                   end=NOW + 30 + 4 * 3600))))
        self.assertEqual(self.slurm.scancels(), [])
        self.assertEqual([event["state"] for event in self.events("sibling_cancel")], ["PENDING", "RUNNING"])

    def test_failed_sibling_scancel_is_an_incident_and_retried(self):
        self.register(entry(360, "ready", node="pool0-00360", started_at=NOW - 600))
        rows = lines(sq(360, w(0), node="pool0-00360", start=NOW - 600, end=NOW + 3 * 3600),
                     sq(361, w(0), state="PENDING", reason="Priority", account=RL))
        self.reconcile(self.fleet(rows, fail=("scancel",), config=dict(DUAL, target=1)))
        self.assertEqual(len(self.events("sibling_cancel_failed")), 1)
        self.assertTrue(self.events("sibling_cancel_failed")[0]["incident"])
        self.now += fleet.CANCEL_RETRY_S + 1
        self.register(entry(360, "ready", node="pool0-00360", started_at=NOW - 600, heartbeat=self.now - 5))
        self.reconcile(self.fleet(rows))
        self.assertEqual(self.slurm.cancelled(), ["361"])

    def test_settlement_refuses_foreign_jobs(self):
        step = fleet.Pass(self.fleet("", config=DUAL), dry_run=False)
        step.cfg, step.now, step.breaker = step.fleet.config(), NOW, step.fleet.breaker()
        jobs = {job["job_id"]: job for job in fleet.parse_squeue(lines(
            sq(370, w(0), user="someoneelse", start=NOW - 600, end=NOW + 3600),
            sq(371, w(0), user="someoneelse", state="PENDING", reason="Priority", account=RL)))}
        self.register(entry(370, "ready", started_at=NOW - 600))
        known = {job_id: {"job_id": job_id} for job_id in jobs}
        with self.assertRaises(AssertionError):
            step.settle(jobs, fleet.read_registry(self.root / "registry"), known)
        self.assertEqual(self.slurm.scancels(), [])
        # in a real pass such jobs are not workers at all: nothing is touched
        self.reconcile(self.fleet(lines(*[sq(job_id, w(0), user="someoneelse", state=job["state"],
                                             start=job["start"], end=job["end"], account=job["account"])
                                          for job_id, job in jobs.items()]), config=dict(DUAL, target=0)))
        self.assertEqual(self.slurm.scancels(), [])

    # -- pair completion (the transition of single-account workers) --

    def test_pending_single_gets_a_sibling_and_keeps_its_queue_position(self):
        self.remember(*[{"job_id": str(400 + i), "name": w(i), "slot": "%02d" % i, "partition": "batch",
                         "state": "PENDING", "submitted_at": NOW - 7200 + i} for i in range(3)])  # old-code records
        rows = [sq(400 + i, w(i), state="PENDING", reason="Priority", start=NOW + 3600) for i in range(3)]
        instance = self.fleet(lines(*rows), config=dict(DUAL, target=3))
        self.reconcile(instance)
        self.assertEqual((self.slurm.names(), self.slurm.accounts()), ([w(0), w(1), w(2)], [RL, RL, RL]))
        self.assertEqual([call for call in self.slurm.calls if call[0] in ("scancel", "scontrol")], [])  # untouched
        submits = self.events("submit")
        self.assertEqual([event["reason"] for event in submits],
                         ["pair completion: sibling of %d (%s)" % (400 + i, AD) for i in range(3)])
        known = self.known()
        siblings = [item for item in known.values() if item.get("account") == RL]
        self.assertEqual(sorted(item["slot_submitted_at"] for item in siblings), [NOW - 7200 + i for i in range(3)])
        counts = self.status()["counts"]
        self.assertEqual((counts["coverage"], counts["pending_slots"], counts["pending"]), (3, 3, 6))
        self.reconcile(self.fleet(self.queue_of()))  # complete now: nothing more, ever
        self.assertEqual(self.slurm.submitted(), [])
        # Slurm reporting an account name we did not ask for: submit_account still says the slot has its sibling
        odd = [sq(job_id, item["name"], state="PENDING", reason="Priority",
                  account="Nemotron_RL_Algo" if item["account"] == RL else AD)
               for job_id, item in sorted(known.items())]
        self.reconcile(self.fleet(lines(*odd)))
        self.assertEqual(self.slurm.submitted(), [])

    def test_completion_is_capped_and_skips_what_is_not_a_plain_pending_batch_slot(self):
        self.remember({"job_id": "450", "name": w(10), "slot": "10", "partition": "batch", "state": "RUNNING",
                       "submitted_at": NOW - 9000, "run_start": NOW - 600},  # requeued (ran before)
                      *[{"job_id": str(410 + i), "name": w(i), "slot": "%02d" % i, "partition": "batch",
                         "state": "PENDING", "submitted_at": NOW - 3600} for i in range(5)])  # queued for 1 h
        rows = [sq(410 + i, w(i), state="PENDING", reason="Priority") for i in range(5)]
        rows += [sq(420, w(20), state="PENDING", reason="JobHeldUser"),  # held
                 sq(421, w(21), state="PENDING", reason="Priority", partition="interactive"),  # manual bridge
                 sq(422, w(22), state="PENDING", reason="Priority", time_left="2:00:00"),  # other walltime
                 sq(423, w(23), state="PENDING", reason="Priority", account="nemotron_other"),  # foreign account
                 sq(450, w(10), state="PENDING", reason="BeginTime"),  # requeued
                 sq(424, w(24), start=NOW - 600, end=NOW + 3 * 3600)]  # running single: nothing to complete
        self.register(entry(424, "ready", started_at=NOW - 600))
        instance = self.fleet(lines(*rows), config=dict(DUAL, target=30, max_submits_per_pass=3))
        self.reconcile(instance)
        # top-up first (3 slots per pass, one job each, F5; here none left for completion)
        self.assertEqual(self.slurm.names(), [w(n) for n in (5, 6, 7)])
        self.reconcile(self.fleet(lines(*rows), config=dict(DUAL, target=11, max_submits_per_pass=3)))
        self.assertEqual(self.slurm.names(), [w(0), w(1), w(2)])  # coverage met: 3 completions, lowest slots
        self.assertEqual(self.slurm.accounts(), [RL] * 3)
        for config in ({"target": 0}, {"target": 11, "worker_accounts": []}):
            self.reconcile(self.fleet(lines(*rows), config=dict(DUAL, max_submits_per_pass=3, **config)))
            self.assertEqual(self.slurm.submitted(), [], config)
        instance.state.mkdir(exist_ok=True)
        instance.stop_flag.write_text("x")
        self.reconcile(self.fleet(lines(*rows), config=dict(DUAL, target=11)))
        self.assertEqual(self.slurm.submitted(), [])
        instance.stop_flag.unlink()
        fleet.write_json(self.root / "state/breaker.json", {"open": True, "failures": [], "noted": {},
                                                            "probe_after": NOW + 3600})
        self.reconcile(self.fleet(lines(*rows), config=dict(DUAL, target=11)))
        self.assertEqual(self.slurm.submitted(), [])

    def test_completion_skips_a_slot_with_an_unconfirmed_job(self):
        self.remember({"job_id": "460", "name": w(0), "slot": "00", "partition": "batch", "account": RL,
                       "state": "PENDING", "submitted_at": NOW - 600})
        rows = lines(sq(461, w(0), state="PENDING", reason="Priority"))
        sacct = "460|hengyuf|PENDING|0:0|Unknown|Unknown|00:00:00|None assigned|\n"  # squeue lost it, sacct did not
        self.reconcile(self.fleet(rows, sacct, config=dict(DUAL, target=1)))
        self.assertEqual(self.slurm.submitted(), [])

    # -- failing account --

    def test_a_failing_account_is_skipped_for_the_pass_and_completed_later(self):
        instance = self.fleet("", config=dict(DUAL, target=3), fail_accounts=(AD,))
        self.reconcile(instance)
        # F5: one job per new slot, on the first account that accepts it; agents_dev is tried once in the pass
        self.assertEqual(self.slurm.accounts(), [AD, RL, RL, RL])
        self.assertEqual(self.slurm.accepted, [RL, RL, RL])
        self.assertEqual(self.slurm.names(), [w(0), w(0), w(1), w(2)])
        failed = self.events("account_submit_failed")
        self.assertEqual([(event["account"], event["incident"]) for event in failed], [(AD, True)])
        self.assertEqual(self.events("submit_failed"), [])
        self.assertEqual(set(instance.breaker()["account_backoff"]), {AD})
        self.now += 60  # in its backoff (and the slots are not pair_delay_min old): not tried
        self.reconcile(self.fleet(self.queue_of(), fail_accounts=(AD,)))
        self.assertEqual(self.slurm.submitted(), [])
        self.now += 14 * 60  # due for completion, still failing: one attempt, the incident not repeated in 30 min
        self.reconcile(self.fleet(self.queue_of(), fail_accounts=(AD,)))
        self.assertEqual((self.slurm.accounts(), self.slurm.accepted), ([AD], []))
        self.assertEqual(len(self.events("account_submit_failed")), 1)
        self.now += 60  # in its backoff again: not tried
        self.reconcile(self.fleet(self.queue_of(), fail_accounts=(AD,)))
        self.assertEqual(self.slurm.submitted(), [])
        self.now += fleet.ACCOUNT_RETRY_S  # works again: completion adds the missing candidates, the backoff ends
        self.reconcile(self.fleet(self.queue_of()))
        self.assertEqual((self.slurm.names(), self.slurm.accounts()), ([w(0), w(1), w(2)], [AD, AD, AD]))
        self.assertEqual(instance.breaker()["account_backoff"], {})
        # a new slot while one account is in its backoff goes to the other one; completion follows later
        fleet.write_json(self.root / "state/breaker.json", dict(instance.breaker(),
                                                                account_backoff={AD: self.now + 300}))
        self.reconcile(self.fleet(self.queue_of(), config=dict(DUAL, target=4)))
        self.assertEqual((self.slurm.names(), self.slurm.accounts()), ([w(3)], [RL]))
        # every account in its backoff: a new slot is still tried, on the first account
        fleet.write_json(self.root / "state/breaker.json", dict(instance.breaker(),
                                                                account_backoff={AD: self.now + 300,
                                                                                 RL: self.now + 300}))
        self.reconcile(self.fleet(self.queue_of(), config=dict(DUAL, target=5)))
        self.assertEqual((self.slurm.names(), self.slurm.accounts()), ([w(4)], [AD]))

    def test_when_every_account_fails_it_is_a_submit_failure(self):
        instance = self.fleet("", config=dict(DUAL, target=2), fail_accounts=(AD, RL))
        self.reconcile(instance)
        self.assertEqual(len(self.slurm.submitted()), 2)  # one try per account, then the pass stops
        self.assertEqual(sorted(event["account"] for event in self.events("submit_failed")), [AD, RL])

    # -- breaker --

    def test_half_open_probe_is_one_job_on_the_first_account_that_accepts(self):
        breaker = {"open": True, "failures": [], "noted": {}, "probe_after": NOW - 1, "cooldown_s": 3600}
        fleet.write_json(self.root / "state/breaker.json", breaker)
        instance = self.fleet("", config=dict(DUAL, target=3))
        self.reconcile(instance)
        self.assertEqual(self.slurm.accounts(), [AD])
        probe = instance.breaker()["probe_job"]
        self.assertTrue(probe)
        self.reconcile(self.fleet(self.queue_of()))  # out: no completion, no more submissions
        self.assertEqual(self.slurm.submitted(), [])
        shutil.rmtree(self.root / "state")
        fleet.write_json(self.root / "state/breaker.json", breaker)
        self.reconcile(self.fleet("", config=dict(DUAL, target=3), fail_accounts=(AD,)))
        self.assertEqual((self.slurm.accounts(), self.slurm.accepted), ([AD, RL], [RL]))  # AD failed: RL probes
        self.assertTrue(instance.breaker()["probe_job"])

    # -- exclusions, stop --

    def test_new_exclusion_updates_both_pending_candidates_but_not_a_leaving_one(self):
        self.register(entry(500, "failed", node="pool0-00500", started_at=NOW - 300, exclude_node=True))
        self.register(entry(510, "ready", node="pool0-00510", started_at=NOW - 900))
        rows = lines(sq(500, w(0), node="pool0-00500", start=NOW - 300, end=NOW + 3 * 3600),  # failed
                     sq(501, w(1), state="PENDING", reason="Priority"),
                     sq(502, w(1), state="PENDING", reason="Priority", account=RL),
                     sq(510, w(2), node="pool0-00510", start=NOW - 900, end=NOW + 3 * 3600),
                     sq(511, w(2), state="PENDING", reason="Priority", account=RL))  # loser
        instance = self.fleet(rows, config=dict(DUAL, target=2))
        self.reconcile(instance)
        self.assertIn("pool0-00500", instance.config()["exclude_nodes"])
        updates = [call for call in self.slurm.calls if call[0] == "scontrol"]
        self.assertEqual([call[2] for call in updates], ["JobId=501", "JobId=502"])
        self.assertTrue(all("pool0-00500" in call[3] for call in updates))
        self.assertEqual(self.slurm.cancelled(), ["511"])

    def test_stop_cancels_every_candidate(self):
        rows = lines(sq(600, w(0), start=NOW - 600, end=NOW + 3 * 3600),
                     sq(601, w(0), state="PENDING", reason="Priority", account=RL),
                     sq(602, w(1), state="PENDING", reason="Priority"),
                     sq(603, w(1), state="PENDING", reason="Priority", account=RL),
                     sq(604, "swe-task-student-fleetd", partition="cpu_datamover", start=NOW - 60, end=NOW + 86400),
                     sq(605, "swe-task-student-webui", start=NOW - 60, end=NOW + 3600, account=RL))
        instance = self.fleet(rows, config=DUAL)
        with quiet():
            fleet.command_stop(instance, argparse.Namespace(yes=True))
        self.assertEqual(self.slurm.scancels(), [["scancel", "--user=hengyuf", "604"],
                                                 ["scancel", "--user=hengyuf", "600", "601", "602", "603"]])
        self.reconcile(self.fleet(""))  # STOP flag: no submissions, no completion
        self.assertEqual(self.slurm.submitted(), [])

    # -- status, dry run, rollback --

    def test_status_shows_accounts_and_pending_slots(self):
        self.register(entry(700, "ready", started_at=NOW - 600))
        rows = lines(sq(700, w(0), start=NOW - 600, end=NOW + 3 * 3600),
                     sq(701, w(1), state="PENDING", reason="Priority", start=NOW + 7200),
                     sq(702, w(1), state="PENDING", reason="Priority", account=RL, start=NOW + 80000))
        instance = self.fleet(rows, config=DUAL)
        with quiet():
            fleet.command_status(instance, argparse.Namespace(json=False))
        view = instance.snapshot(instance.config(), fleet.parse_squeue(rows),
                                 fleet.read_registry(self.root / "registry"), NOW)
        self.assertEqual([row["account"] for row in view["jobs"]], [AD, AD, RL])
        self.assertEqual((view["counts"]["pending"], view["counts"]["pending_slots"], view["counts"]["coverage"]),
                         (2, 1, 2))
        self.assertEqual(view["config"]["worker_accounts"], [AD, RL])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            fleet.print_status(view)
        text = output.getvalue()
        self.assertIn("worker accounts agents_dev+rl_algo (sibling after 15 min)", text)
        self.assertIn("ACCOUNT", text)
        self.assertIn("rl_algo", text.splitlines()[-2] + text)
        self.assertIn("pending slots 1", text)
        # the pending slot's earliest candidate is its scheduler estimate
        self.assertEqual(fleet.wait_estimate(dict(instance.config(), wait_window_h=1), [], fleet.parse_squeue(rows),
                                             NOW), 7200)

    def test_status_shows_the_worker_accounts_of_the_daemons_last_pass(self):
        """The apply checks: `status` says which worker accounts the daemon's last completed pass used when that
        differs from fleet.json (a daemon still on the old fleet.py, or no pass since a change)."""
        instance = self.fleet("", config={"target": 0})

        def status_line():
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                fleet.command_status(self.fleet(""), argparse.Namespace(json=False))
            return output.getvalue().splitlines()[1]
        self.assertNotIn("daemon's last pass", status_line())  # no status.json yet
        fleet.write_json(self.root / "state/status.json", {"config": {"target": 0}, "last_reconcile_at": NOW - 30})
        self.assertIn("| last pass 30s ago | worker accounts agents_dev (daemon's last pass: no worker accounts, "
                      "a fleet.py from before 2026-11-02)", status_line())  # written by the previous fleet.py
        self.reconcile(instance)  # a pass of this fleet.py
        self.assertTrue(status_line().endswith("| last pass 0s ago | worker accounts agents_dev"))
        with quiet():
            fleet.command_worker_accounts(instance, argparse.Namespace(accounts=[AD, RL], clear=False))
        self.assertTrue(status_line().endswith("worker accounts agents_dev+rl_algo (sibling after 15 min) "
                                               "(daemon's last pass: agents_dev)"))
        self.reconcile(self.fleet(""))
        self.assertTrue(status_line().endswith("worker accounts agents_dev+rl_algo (sibling after 15 min)"))
        self.assertEqual(self.status()["config"]["worker_accounts"], [AD, RL])

    def test_dry_run_writes_nothing_and_lists_every_candidate(self):
        rows = lines(sq(800, w(0), state="PENDING", reason="Priority"))
        self.remember({"job_id": "800", "name": w(0), "slot": "00", "partition": "batch", "state": "PENDING",
                       "submitted_at": NOW - 1800})
        self.fleet(rows, config=dict(DUAL, target=2))
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        actions = self.fleet(rows).reconcile(dry_run=True)
        self.assertEqual({path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}, before)
        self.assertEqual([call[0] for call in self.slurm.calls], ["squeue"])
        submits = [(action["slot"], action["account"], action.get("reason", "")[:15]) for action in actions
                   if action["action"] == "submit"]
        self.assertEqual(submits, [("01", AD, ""), ("00", RL, "pair completion")])  # F5: a new slot is one job
        self.assertTrue(all("--account=" + action["account"] in action["argv"] for action in actions))

    def test_rollback_to_one_account_still_settles_leftover_pairs(self):
        self.register(entry(900, "ready", started_at=NOW - 600))
        rows = lines(sq(900, w(0), start=NOW - 600, end=NOW + 3 * 3600),
                     sq(901, w(0), state="PENDING", reason="Priority", account=RL),
                     sq(902, w(1), state="PENDING", reason="Priority"),
                     sq(903, w(1), state="PENDING", reason="Priority", account=RL))
        instance = self.fleet(rows, config={"target": 3})  # worker_accounts absent again
        self.reconcile(instance)
        self.assertEqual(self.slurm.cancelled(), ["901"])
        self.assertEqual((self.slurm.names(), self.slurm.accounts()), ([w(2)], [AD]))  # one new single, slot 1 once

    # -- daemon reload, worker-accounts command --

    def test_reload_flag_restarts_the_daemon_once(self):
        instance = self.fleet("", config={"target": 0})
        instance.state.mkdir(exist_ok=True)
        instance.reload_flag.write_text("old")
        os.utime(instance.reload_flag, (time.time() - 3600, time.time() - 3600))  # older than the daemon: ignored
        with quiet():
            self.assertEqual(instance.daemon(interval=1, until=time.time() + 2.5), 0)
        self.assertFalse(instance.reload_flag.exists())
        self.assertEqual(len(self.events("reload_flag_ignored")), 1)

        def touch_later():
            time.sleep(1.5)
            instance.reload_flag.write_text("new")
        import threading
        threading.Thread(target=touch_later, daemon=True).start()
        with quiet():
            self.assertEqual(instance.daemon(interval=1, until=time.time() + 30), fleet.RELOAD_EXIT)
        self.assertFalse(instance.reload_flag.exists())
        self.assertEqual(json.loads((self.root / "state/daemon.json").read_text())["stop_reason"], "RELOAD flag")

    def test_worker_accounts_command(self):
        instance = self.fleet("", config={"target": 7, "exclude_nodes": ["pool0-00001"]})
        with quiet():
            fleet.command_worker_accounts(instance, argparse.Namespace(accounts=[AD, RL], clear=False))
        stored = json.loads((self.root / "fleet.json").read_text())
        self.assertEqual((stored["worker_accounts"], stored["target"], stored["exclude_nodes"]),
                         ([AD, RL], 7, ["pool0-00001"]))
        with self.assertRaises(fleet.ConfigError), quiet():
            fleet.command_worker_accounts(instance, argparse.Namespace(accounts=[AD, AD], clear=False))
        with quiet():
            fleet.command_worker_accounts(instance, argparse.Namespace(accounts=[], clear=True))
        self.assertEqual(json.loads((self.root / "fleet.json").read_text())["worker_accounts"], [])
        self.assertEqual([event["after"] for event in self.events("worker_accounts")], [[AD, RL], []])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            fleet.command_worker_accounts(instance, argparse.Namespace(accounts=[], clear=False))
        self.assertIn("candidates on nemotron_agents_dev", output.getvalue())

    def test_manual_submit_account(self):
        instance = self.fleet("", config=DUAL)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            fleet.command_submit(instance, argparse.Namespace(slot=5, partition=None, time=None, account=RL,
                                                              dry_run=True, test_only=False))
        self.assertIn("--account=" + RL, json.loads(output.getvalue().splitlines()[1]))
        with quiet():
            fleet.command_submit(instance, argparse.Namespace(slot=5, partition=None, time=None, account=None,
                                                              dry_run=False, test_only=False))
        self.assertEqual(self.slurm.accounts(), [AD])

    # -- the live transition, replayed --

    def test_transition_of_live_like_single_account_wave(self):
        """10 serving agents_dev workers + the old daemon's next wave of 10 pending agents_dev singles (2026-11-02
        20:57-21:37): enabling worker_accounts adds 10 rl_algo siblings in one pass, touches no existing job, and
        the first candidate to start wins each slot."""
        serving = [sq(1000 + i, w(i), node="pool0-%05d" % (1000 + i), start=NOW - 3600, end=NOW + 3 * 3600 - 600)
                   for i in range(10)]
        for i in range(10):
            self.register(entry(1000 + i, "ready", node="pool0-%05d" % (1000 + i), started_at=NOW - 3600))
        wave = [sq(1100 + i, w(10 + i), state="PENDING", reason="Priority", start=NOW + 7200) for i in range(10)]
        self.remember(*[{"job_id": str(1100 + i), "name": w(10 + i), "slot": "%02d" % (10 + i), "partition": "batch",
                         "state": "PENDING", "submitted_at": NOW - 1800 + 60 * i} for i in range(10)])
        instance = self.fleet(lines(*(serving + wave)), config=dict(DUAL, target=10, horizon_min=150))
        self.reconcile(instance)
        self.assertEqual((self.slurm.names(), self.slurm.accounts()), ([w(10 + i) for i in range(10)], [RL] * 10))
        self.assertEqual([call for call in self.slurm.calls if call[0] in ("scancel", "scontrol")], [])
        known = self.known()
        rl = {item["slot"]: job_id for job_id, item in known.items() if item.get("account") == RL}
        # agents_dev wins slot 10, rl_algo wins slot 11; the others stay pending
        self.now += 3600
        rows = list(serving)
        for i in range(10):
            slot = "%02d" % (10 + i)
            if i == 0:
                rows.append(sq(1100, w(10), node="pool0-01100", start=self.now - 30, end=self.now + 4 * 3600 - 30))
                rows.append(sq(rl[slot], w(10), state="PENDING", reason="Priority", account=RL))
            elif i == 1:
                rows.append(sq(1101, w(11), state="PENDING", reason="Priority"))
                rows.append(sq(rl[slot], w(11), account=RL, node="pool0-09011", start=self.now - 30,
                               end=self.now + 4 * 3600 - 30))
            else:
                rows.append(sq(1100 + i, w(10 + i), state="PENDING", reason="Priority"))
                rows.append(sq(rl[slot], w(10 + i), state="PENDING", reason="Priority", account=RL))
        self.reconcile(self.fleet(lines(*rows)))
        self.assertEqual((self.slurm.submitted(), self.slurm.scancels()), ([], []))  # both winners still starting
        waits = json.loads((self.root / "state/waits.json").read_text())
        self.assertEqual(sorted((item["job_id"], item["wait_s"]) for item in waits),
                         [("1100", 1800 + 3600 - 30), (rl["11"], 1800 - 60 + 3600 - 30)])
        self.now += 480
        self.register(entry(1100, "ready", node="pool0-01100", started_at=self.now - 510, heartbeat=self.now))
        self.register(entry(rl["11"], "ready", node="pool0-09011", started_at=self.now - 510, heartbeat=self.now))
        self.reconcile(self.fleet(lines(*rows)))
        self.assertEqual(self.slurm.scancels(), [["scancel", "--user=hengyuf", "--state=PENDING", "1101",
                                                  rl["10"]]])
        self.assertEqual(self.slurm.submitted(), [])


class ReviewFixesTest(Base):
    """Second-round review fixes (2026-11-02): F1 stale sibling marks, F2 a keeper police() is cancelling, F3 account
    pairs only, F7 a settled candidate is never a failure. (F5: AccountsTest; F6, F10: test_fleet.py.)"""

    def test_f1_a_candidate_that_is_no_longer_a_loser_loses_its_sibling_marks(self):
        # the agents_dev candidate serves, the scancel of its rl_algo backup fails, then the keeper dies serving
        self.register(entry(1300, "ready", node="pool0-01300", started_at=NOW - 600))
        keeper = sq(1300, w(0), node="pool0-01300", start=NOW - 600, end=NOW + 3 * 3600)
        backup = sq(1301, w(0), state="PENDING", reason="Priority", account=RL)
        self.reconcile(self.fleet(lines(keeper, backup), fail=("scancel",), config=dict(DUAL, target=1)))
        record = self.known()["1301"]
        self.assertEqual((record["sibling_of"], record["cancel_requested_at"], record["sibling_cancel"]["at"]),
                         ("1300", NOW, NOW))
        self.assertEqual(len(self.events("sibling_cancel_failed")), 1)
        self.now += 60
        self.register(entry(1300, "unhealthy", node="pool0-01300", started_at=NOW - 600, heartbeat=NOW + 30))
        sacct = sacct_row(1300, "FAILED", start=NOW - 600, node="pool0-01300", exit_code="6:0")
        self.reconcile(self.fleet(lines(backup), sacct))
        record = self.known()["1301"]
        self.assertEqual([key for key in ("sibling_of", "sibling_cancel", "cancel_requested_at") if key in record], [])
        self.assertEqual([(event["job_id"], event["requested_at"]) for event in self.events("sibling_cancel_dropped")],
                         [("1301", NOW)])
        self.assertEqual(self.status()["counts"]["coverage"], 1)  # the status counts it again (stale marks: 0)
        self.assertEqual((self.slurm.submitted(), self.slurm.scancels()), ([], []))  # it covers the slot
        # it starts and fails by itself: a genuine failure again, no longer hidden by the old mark (F7)
        self.now += 120
        self.register(entry(1301, "failed", node="pool0-01301", started_at=self.now - 60, exclude_node=True,
                            reason="vLLM exited during startup"))
        self.reconcile(self.fleet(lines(sq(1301, w(0), account=RL, node="pool0-01301", start=self.now - 60,
                                           end=self.now + 4 * 3600))))
        self.assertEqual([(event["job_id"], event["kind"]) for event in self.events("worker_failure")],
                         [("1300", "unhealthy"), ("1301", "startup_failed")])
        self.assertIn("pool0-01301", self.fleet().config()["exclude_nodes"])

    def test_f1_keeps_a_police_cancel_time(self):
        self.remember({"job_id": "1310", "name": w(0), "slot": "00", "partition": "batch", "account": RL,
                       "state": "RUNNING", "sibling_of": "1309", "submitted_at": NOW - 4000, "run_start": NOW - 400,
                       "sibling_cancel": {"at": NOW - 300, "state": "PENDING"}, "cancel_requested_at": NOW - 60})
        row = sq(1310, w(0), account=RL, node="pool0-01310", start=NOW - 400, end=NOW + 3 * 3600)
        self.reconcile(self.fleet(lines(row), config=dict(DUAL, target=1)))
        record = self.known()["1310"]
        self.assertEqual((record.get("sibling_of"), record.get("sibling_cancel"), record.get("cancel_requested_at")),
                         (None, None, NOW - 60))

    def test_f2_a_job_police_is_cancelling_never_keeps_its_slot(self):
        # the agents_dev candidate started first and never got ready: police() cancels it, but that scancel fails
        self.register(entry(1400, "starting", node="pool0-01400", started_at=NOW - 26 * 60))
        stuck = sq(1400, w(0), node="pool0-01400", start=NOW - 26 * 60, end=NOW + 3 * 3600)
        backup = sq(1401, w(0), state="PENDING", reason="Priority", account=RL)
        self.reconcile(self.fleet(lines(stuck, backup), fail=("scancel",), config=dict(DUAL, target=1)))
        self.assertEqual([event["kind"] for event in self.events("worker_failure")], ["stuck_startup"])
        self.assertEqual(len(self.events("scancel_failed")), 1)
        self.now += 60  # the backup starts while the stuck one is still listed: it must not be settled away
        started = sq(1401, w(0), account=RL, node="pool0-01401", start=self.now - 20, end=self.now + 4 * 3600)
        self.reconcile(self.fleet(lines(stuck, started)))
        self.assertEqual((self.slurm.scancels(), self.events("sibling_cancel")), ([], []))
        self.assertNotIn("sibling_cancel", self.known()["1401"])
        self.now += fleet.CANCEL_RETRY_S  # police() repeats its own request; the healthy one stays
        self.register(entry(1401, "starting", node="pool0-01401", started_at=NOW + 40))
        self.reconcile(self.fleet(lines(stuck, started)))
        self.assertEqual(self.slurm.scancels(), [["scancel", "--user=hengyuf", "1400"]])
        self.assertEqual(self.events("sibling_cancel"), [])

    def test_f3_only_account_pairs_settle_or_count_once(self):
        # one worker account (no worker_accounts): duplicates of a slot name made by hand are separate workers,
        # exactly as before worker accounts: none is settled, each one counts
        self.register(entry(1500, "ready", node="pool0-01500", started_at=NOW - 600))
        rows = [sq(1500, w(0), node="pool0-01500", start=NOW - 600, end=NOW + 3 * 3600),
                sq(1501, w(0), state="PENDING", reason="Priority"),  # same account as the running one
                sq(1502, w(1), state="PENDING", reason="Priority"),
                sq(1503, w(1), state="PENDING", reason="Priority"),  # same account, both pending
                sq(1504, w(2), state="PENDING", reason="Priority"),
                sq(1505, w(2), state="PENDING", reason="Priority", partition="interactive", account=RL)]  # 2 partitions
        instance = self.fleet(lines(*rows), config={"target": 6, "coverage_partitions": []})
        self.reconcile(instance)
        self.assertEqual((self.slurm.scancels(), self.events("sibling_cancel"), self.slurm.submitted()), ([], [], []))
        cfg, jobs = instance.config(), fleet.parse_squeue(lines(*rows))
        registry = fleet.read_registry(self.root / "registry")
        self.assertEqual(fleet.pair_slots(cfg, jobs), set())
        self.assertEqual((fleet.coverage(cfg, jobs, registry, NOW, (), 3600), fleet.node_count(cfg, jobs, registry)),
                         (6, 6))
        self.assertEqual(self.status()["counts"]["coverage"], 6)
        if ORIG.is_file():  # the fleet.py before worker accounts counts the same
            self.assertEqual(load_orig().coverage(cfg, jobs, registry, NOW, (), 3600), 6)
        # an account pair on the fleet partition is a pair, also with one worker account (left after a rollback)
        rows += [sq(1506, w(3), state="PENDING", reason="Priority"),
                 sq(1507, w(3), state="PENDING", reason="Priority", account=RL)]
        jobs = fleet.parse_squeue(lines(*rows))
        self.assertEqual(fleet.pair_slots(cfg, jobs), {"03"})
        self.assertEqual((fleet.coverage(cfg, jobs, registry, NOW, (), 3600), fleet.node_count(cfg, jobs, registry)),
                         (7, 7))

    def test_f7_a_settled_candidate_is_never_a_failure_whatever_its_registry_says(self):
        early = sq(1700, w(0), node="pool0-01700", start=NOW - 120, end=NOW + 4 * 3600 - 120)
        late = sq(1701, w(0), account=RL, node="pool0-01701", start=NOW - 60, end=NOW + 4 * 3600 - 60)
        instance = self.fleet(lines(early, late), config=dict(DUAL, target=1))
        self.reconcile(instance)
        self.assertEqual(self.slurm.scancels(), [["scancel", "--user=hengyuf", "1701"]])
        # our SIGTERM made vLLM exit before serve_student.sh's TERM trap: the registry says failed, node suspect
        self.register(entry(1701, "failed", node="pool0-01701", started_at=NOW - 60, exclude_node=True,
                            reason="vLLM exited during startup (status 143)"))
        self.now += 60  # still listed as running: police() must not count it either
        self.reconcile(self.fleet(lines(early, late)))
        self.assertEqual(self.events("worker_failure"), [])
        self.now += 60  # gone, CANCELLED by us
        self.reconcile(self.fleet(lines(early), sacct_row(1701, "CANCELLED by %d" % UID, start=NOW - 60,
                                                          node="pool0-01701", exit_code="0:15")))
        self.assertEqual([self.events(kind) for kind in ("worker_failure", "exclude_node", "breaker_open")],
                         [[], [], []])
        self.assertEqual(instance.breaker()["failures"], [])
        self.assertNotIn("pool0-01701", instance.config()["exclude_nodes"])
        self.assertEqual([(event["job_id"], event["sibling_of"], event["registry_status"])
                          for event in self.events("job_ended")], [("1701", "1700", "failed")])
        # the same end without the settlement mark is a failure
        known = self.known()
        known["1702"] = {"job_id": "1702", "name": w(1), "slot": "01", "partition": "batch", "state": "RUNNING",
                         "node": "pool0-01702", "start": NOW - 60}
        fleet.write_json(self.root / "state/jobs.json", known)
        self.register(entry(1702, "failed", node="pool0-01702", started_at=NOW - 60, exclude_node=True))
        self.reconcile(self.fleet(lines(early), sacct_row(1702, "FAILED", start=NOW - 60, node="pool0-01702",
                                                          exit_code="3:0")))
        self.assertEqual([event["job_id"] for event in self.events("worker_failure")], ["1702"])

    def test_f7_a_settled_probe_frees_the_half_open_breaker(self):
        fleet.write_json(self.root / "state/breaker.json", {
            "open": True, "failures": [], "noted": {}, "since": NOW - 4000, "cooldown_s": 3600,
            "probe_after": NOW - 60, "probe_job": "1801"})
        self.remember({"job_id": "1801", "name": w(0), "slot": "00", "partition": "batch", "account": RL,
                       "state": "RUNNING", "node": "pool0-01801", "start": NOW - 120, "sibling_of": "1800",
                       "sibling_cancel": {"at": NOW - 60, "state": "RUNNING"}, "cancel_requested_at": NOW - 60})
        self.register(entry(1801, "failed", node="pool0-01801", started_at=NOW - 120, exclude_node=True))
        instance = self.fleet(lines(sq(1800, w(0), node="pool0-01800", start=NOW - 180, end=NOW + 3 * 3600)),
                              sacct_row(1801, "CANCELLED by %d" % UID, start=NOW - 120, node="pool0-01801"),
                              config=dict(DUAL, target=1))
        self.reconcile(instance)
        breaker = instance.breaker()
        self.assertEqual((breaker["open"], breaker["probe_job"], breaker["probe_after"]), (True, None, NOW))
        self.assertEqual([self.events(kind) for kind in ("worker_failure", "breaker_probe_failed")], [[], []])


# ---- equivalence with the previous fleet.py (one worker account) ---------------------------------------------------

def load_orig():
    """The previous fleet.py under any file name (e.g. fleet.py.pre-dual-account): an explicit source loader."""
    loader = importlib.machinery.SourceFileLoader("fleet_orig", str(ORIG))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader("fleet_orig", loader))
    loader.exec_module(module)
    return module


class World(fleet.Slurm):
    """A modelled Slurm queue for one fleet root: sbatch queues a job that starts after a seeded wait, writes its
    registry record (starting, then ready or failed), and ends at its walltime; scancel removes it (sacct: CANCELLED
    by this uid). The rendering follows the squeue -o format the caller asks for, so an old and a new fleet.py can
    run against identical worlds. Every random draw is seeded by (seed, job id, purpose): equal histories stay
    equal."""

    def __init__(self, root, seed, now):
        super().__init__(bin_dir="/nonexistent")
        self.root, self.seed, self.now = Path(root), seed, now
        self.jobs, self.sacct_rows, self.calls, self.comments, self.next_id = {}, {}, [], {}, 50000000

    def rng(self, job_id, purpose):
        return random.Random("%s/%s/%s" % (self.seed, job_id, purpose))

    def add(self, job_id, name, state, account=AD, partition="batch", start=None, limit=4 * 3600, reason="Priority",
            node="", submitted=None, user="hengyuf"):
        self.jobs[str(job_id)] = {"job_id": str(job_id), "name": name, "state": state, "account": account,
                                  "partition": partition, "start": start, "limit": limit, "reason": reason,
                                  "node": node, "submitted": submitted if submitted is not None else self.now,
                                  "user": user}

    def render(self, job, fmt):
        pending = job["state"] in fleet.PENDING
        end = job["start"] + job["limit"] if job["start"] and not pending else None
        left = job["limit"] if pending else max(0, end - self.now)
        fields = [job["job_id"], job["user"], job["state"], job["partition"], "" if pending else job["node"],
                  "0:00", "%d:%02d:%02d" % (left // 3600, left // 60 % 60, left % 60),
                  str(job["start"]) if job["start"] and not pending else "N/A", str(end) if end else "N/A",
                  job["reason"] if pending else "None"]
        if "%a" in fmt:
            fields.append(job["account"])
        return "|".join(fields + [job["name"]])

    def run(self, argv, epoch_times=False):
        self.calls.append(list(argv))
        if argv[0] == "squeue" and "-j" in argv:
            return Done(self.comments.get(argv[argv.index("-j") + 1], "(null)") + "\n")
        if argv[0] == "squeue":
            fmt = argv[argv.index("-o") + 1]
            return Done("".join(self.render(job, fmt) + "\n" for job in self.jobs.values()))
        if argv[0] == "sacct":
            ids = argv[argv.index("-j") + 1].split(",")
            return Done("".join(self.sacct_rows[job_id] for job_id in ids if job_id in self.sacct_rows))
        if argv[0] == "sbatch":
            self.next_id += 1
            job_id = str(self.next_id)
            option = {arg.split("=", 1)[0]: arg.split("=", 1)[1] for arg in argv if arg.startswith("--") and "=" in arg}
            self.comments[job_id] = option.get("--comment", "(null)")
            self.add(job_id, option["--job-name"], "PENDING", account=option.get("--account", AD),
                     partition=option.get("--partition", "batch"), limit=fleet.duration(option.get("--time")))
            return Done(job_id + "\n")
        if argv[0] == "scancel":
            pending_only = "--state=PENDING" in argv
            for job_id in [arg for arg in argv[1:] if not arg.startswith("--")]:
                job = self.jobs.get(job_id)
                if job is None or (pending_only and job["state"] not in fleet.PENDING):
                    continue
                self.finish(job_id, "CANCELLED by %d" % UID)
            return Done()
        if argv[0] == "scontrol":
            return Done()
        raise AssertionError("unexpected Slurm command %r" % argv)

    def finish(self, job_id, state, exit_code="0:0"):
        job = self.jobs.pop(job_id)
        started = job["start"] if job["state"] in fleet.RUNNING else None
        self.sacct_rows[job_id] = "%s|%s|%s|%s|%s|%d|00:00:00|%s|\n" % (
            job_id, job["user"], state, exit_code, started or "None", self.now, job["node"] or "None assigned")
        path = self.root / "registry" / ("%s.json" % job_id)
        record = fleet.read_json(path)
        if started and isinstance(record, dict) and record.get("status") in ("starting", "ready", "draining"):
            record.update(status="stopped", reason="SIGTERM (test)")
            fleet.write_json(path, record)

    def advance(self, seconds):
        """Time passes: pending jobs start after their drawn wait, starting ones turn ready (or fail), heartbeats
        refresh (a few go stale), jobs end at their walltime."""
        self.now += seconds
        for job_id, job in sorted(self.jobs.items()):
            if job["user"] != "hengyuf" or not job["name"].startswith("swe-task-student-w"):
                continue
            draw = self.rng(job_id, "life")
            wait, startup, fate = draw.choice((0, 120, 900, 3600, 4 * 3600)), draw.randint(300, 900), draw.random()
            path = self.root / "registry" / ("%s.json" % job_id)
            if job["state"] == "PENDING" and "held" not in job["reason"].lower() \
                    and self.now >= job["submitted"] + wait:
                job.update(state="RUNNING", start=self.now - draw.randint(0, 50),
                           node="pool0-%05d" % draw.choice(range(1, 40)))
            if job["state"] in fleet.RUNNING:
                if self.now >= job["start"] + job["limit"]:
                    self.finish(job_id, "COMPLETED")
                    continue
                age = self.now - job["start"]
                status = "ready" if age >= startup else "starting"
                if fate < 0.07 and age >= startup / 2:
                    status = "failed"
                elif fate < 0.1:
                    status = "starting"  # stuck
                heartbeat = self.now - (900 if fate > 0.97 else 5)
                fleet.write_json(path, {"job_id": job_id, "slot": job["name"][-2:], "node": job["node"],
                                        "base_url": "http://10.0.0.1:8000/v1", "started_at": job["start"],
                                        "ready_at": job["start"] + startup if status == "ready" else None,
                                        "end_epoch": job["start"] + job["limit"],
                                        "drain_at": job["start"] + job["limit"] - 2700, "heartbeat_at": heartbeat,
                                        "status": status, "reason": "test" if status == "failed" else None,
                                        "exclude_node": fate < 0.04})
                if status == "failed" and fate < 0.02:
                    self.finish(job_id, "FAILED", "1:0")


def normalized_calls(calls, root):
    """Slurm calls with the squeue listing format (the new one adds %a) and the temp root left out."""
    return [["squeue", "-h", "-u", call[3], "-o", "<format>"] if call[0] == "squeue" and "-u" in call else
            [arg.replace(str(root), "<root>") for arg in call] for call in calls]


NEW_EVENT_KEYS = {"account", "sibling_of"}
NEW_RECORD_KEYS = {"account", "submit_account", "slot_submitted_at"}


def scrubbed_events(root):
    path = Path(root) / "state/events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [{key: value for key, value in event.items() if key not in NEW_EVENT_KEYS | {"ts", "time"}}
            for event in events]


def scrubbed_state(root):
    state = Path(root) / "state"
    jobs = fleet.read_json(state / "jobs.json", {}) or {}
    status = fleet.read_json(state / "status.json", {}) or {}
    return {
        "jobs": {job_id: {key: value for key, value in record.items() if key not in NEW_RECORD_KEYS}
                 for job_id, record in jobs.items()},
        "waits": [{key: value for key, value in item.items() if key != "account"}
                  for item in fleet.read_json(state / "waits.json", []) or []],
        "breaker": fleet.read_json(state / "breaker.json"),
        "exclude_nodes": (fleet.read_json(Path(root) / "fleet.json") or {}).get("exclude_nodes"),
        "counts": {key: value for key, value in (status.get("counts") or {}).items() if key != "pending_slots"},
        "rows": [{key: value for key, value in row.items() if key != "account"} for row in status.get("jobs") or []],
        "registry": sorted(path.name for path in (Path(root) / "registry").rglob("*.json")),
    }


class EquivalenceTest(Base):
    """One worker account: the new fleet.py must behave exactly as the previous one."""

    @classmethod
    def setUpClass(cls):
        if not ORIG.is_file():
            raise unittest.SkipTest("previous fleet.py not found at %s (set FLEET_ORIG)" % ORIG)
        cls.orig = load_orig()

    def scenario(self, seed, root, world):
        """A random fleet state: config, workers in the queue (unique slots, as fleet.py makes them), registry
        records, jobs.json (incl. ghosts with sacct rows), breaker, STOP flag, waits."""
        rng = random.Random(seed)
        cfg = json.loads(json.dumps(self.orig.DEFAULTS))
        cfg.update(target=rng.choice((0, 1, 3, 10, 12)), max_jobs=rng.choice((4, 12, 24)),
                   max_submits_per_pass=rng.choice((1, 3, 10)), horizon_min=rng.choice((45, 90)),
                   stagger_min=rng.choice((0, 40)), adaptive_horizon=rng.random() < 0.7,
                   horizon_max_min=150, wait_quantile=0.75, wait_window_h=3,
                   exclude_nodes=cfg["exclude_nodes"] + (["pool0-00003"] if rng.random() < 0.3 else []))
        fleet.write_json(Path(root) / "fleet.json", cfg)
        (Path(root) / "registry").mkdir(parents=True, exist_ok=True)
        known = {}
        slots = rng.sample(range(40), rng.randint(0, 14))
        for number, slot in enumerate(slots):
            job_id = str(40000000 + seed % 1000 * 100 + number)
            state = rng.choice(("RUNNING",) * 5 + ("PENDING",) * 3 + ("COMPLETING", "CONFIGURING"))
            partition = "batch" if rng.random() < 0.85 else "interactive"
            reason = rng.choice(("Priority", "Priority", "Resources", "JobHeldUser", "launch failed requeued held"))
            start = world.now - rng.randint(0, 4 * 3600 - 60) if state != "PENDING" else None
            world.add(job_id, "swe-task-student-w%02d" % slot, state, partition=partition, start=start,
                      reason=reason, node="pool0-%05d" % rng.randint(1, 40), submitted=world.now - rng.randint(0, 9000))
            if state != "PENDING" and rng.random() < 0.8:
                status = rng.choice(("ready", "ready", "draining", "starting", "failed", "unhealthy", "stopped"))
                fleet.write_json(Path(root) / "registry" / ("%s.json" % job_id), {
                    "job_id": job_id, "slot": "%02d" % slot, "node": world.jobs[job_id]["node"],
                    "base_url": "http://10.0.0.1:8000/v1",
                    "started_at": start - (rng.choice((0, 0, 0, 7200))), "ready_at": start + 400,
                    "end_epoch": start + 4 * 3600, "drain_at": start + 4 * 3600 - 2700,
                    "heartbeat_at": world.now - rng.choice((5, 30, 200, 700)), "status": status,
                    "reason": "x", "exclude_node": rng.random() < 0.7})
            if rng.random() < 0.8:
                known[job_id] = {"job_id": job_id, "name": "swe-task-student-w%02d" % slot, "slot": "%02d" % slot,
                                 "partition": partition, "state": state, "node": "",
                                 "submitted_at": world.jobs[job_id]["submitted"] if rng.random() < 0.8 else None,
                                 "first_seen": world.now - 9000}
                if start and rng.random() < 0.3:
                    known[job_id]["run_start"] = start - rng.choice((0, 60))
                if rng.random() < 0.15:
                    known[job_id]["cancel_requested_at"] = world.now - rng.choice((30, 900))
                if rng.random() < 0.1:
                    known[job_id]["failure_recorded"] = True
        for number in range(rng.randint(0, 4)):  # jobs gone from squeue
            job_id = str(41000000 + seed % 1000 * 100 + number)
            slot = rng.choice([value for value in range(40, 60)])
            known[job_id] = {"job_id": job_id, "name": "swe-task-student-w%02d" % slot, "slot": "%02d" % slot,
                             "partition": "batch", "state": rng.choice(("RUNNING", "PENDING")),
                             "node": "pool0-%05d" % rng.randint(1, 40), "start": world.now - 3600,
                             "end": world.now + rng.choice((-60, 3600, 3 * 3600)), "submitted_at": world.now - 9000,
                             "vanished_at": rng.choice((None, world.now - 2000))}
            if rng.random() < 0.3:
                known[job_id]["cancel_requested_at"] = world.now - 100
            kind = rng.choice(("ours", "other", "completed", "failed", "node_fail", "reaper", "none", "running"))
            if kind != "none":
                state = {"ours": "CANCELLED by %d" % UID, "other": "CANCELLED by 1", "completed": "COMPLETED",
                         "failed": "FAILED", "node_fail": "NODE_FAIL", "reaper": "CANCELLED by 146504",
                         "running": "RUNNING"}[kind]
                world.sacct_rows[job_id] = "%s|hengyuf|%s|0:0|%d|%s|01:00:00|%s|%s\n" % (
                    job_id, state, world.now - 3600, "Unknown" if kind == "running" else world.now - 30,
                    known[job_id]["node"], "Cancelled by occupied idle job reaper" if kind == "reaper" else "")
            if rng.random() < 0.5:
                fleet.write_json(Path(root) / "registry" / ("%s.json" % job_id), {
                    "job_id": job_id, "status": rng.choice(("ready", "starting", "failed", "unhealthy")),
                    "node": known[job_id]["node"], "ready_at": world.now - 3000, "heartbeat_at": world.now - 100,
                    "started_at": world.now - 3600, "reason": "y"})
        world.add(str(42000000 + seed), "some-other-job", "RUNNING", start=world.now - 60, node="pool0-00100")
        world.add(str(42000001 + seed), "swe-task-student-webui", "RUNNING", start=world.now - 60, node="pool0-00101")
        fleet.write_json(Path(root) / "state/jobs.json", known)
        roll = rng.random()
        if roll < 0.15:
            fleet.write_json(Path(root) / "state/breaker.json", {
                "open": True, "failures": [], "noted": {}, "since": world.now - 4000, "cooldown_s": 3600,
                "probe_after": world.now + rng.choice((-60, 600)), "probe_job": None})
        elif roll < 0.25:
            fleet.write_json(Path(root) / "state/breaker.json", {
                "open": False, "noted": {}, "failures": [{"ts": world.now - 600, "job_id": "1", "node": "pool0-00007",
                                                          "kind": "x"}] * rng.randint(1, 3)})
        if rng.random() < 0.05:
            (Path(root) / "state/STOP").write_text("x")
        fleet.write_json(Path(root) / "state/waits.json", [
            {"job_id": str(i), "partition": "batch", "start": world.now - rng.randint(0, 5 * 3600),
             "wait_s": rng.randint(0, 8 * 3600)} for i in range(rng.randint(0, 6))])

    def test_identical_slurm_calls_and_state_over_random_histories(self):
        seeds = range(int(os.environ.get("FLEET_EQUIV_SEEDS", "300")))
        compared, seen = 0, collections.Counter()
        for seed in seeds:
            roots, worlds, fleets = [], [], []
            for module in (self.orig, fleet):
                root = Path(tempfile.mkdtemp(prefix="fleet-equiv-"))
                self.addCleanup(shutil.rmtree, root, True)
                world = World(root, seed, NOW)
                (root / "state").mkdir(parents=True)
                roots.append(root)
                worlds.append(world)
            for root, world in zip(roots, worlds):
                self.scenario(seed, root, world)
            for root, world, module in zip(roots, worlds, (self.orig, fleet)):
                fleets.append(module.Fleet(root, world, clock=lambda world=world: world.now))
            passes = random.Random(seed).randint(1, 6)
            for number in range(passes):
                outcome = []
                for instance, world in zip(fleets, worlds):
                    before = len(world.calls)
                    try:
                        with quiet():
                            instance.reconcile()
                        error = None
                    except Exception as caught:  # both must fail the same way
                        error = type(caught).__name__
                    outcome.append((normalized_calls(world.calls[before:], world.root), error))
                self.assertEqual(outcome[0], outcome[1], "seed %d pass %d: Slurm calls differ" % (seed, number))
                self.assertEqual(scrubbed_events(roots[0]), scrubbed_events(roots[1]),
                                 "seed %d pass %d: events differ" % (seed, number))
                self.assertEqual(scrubbed_state(roots[0]), scrubbed_state(roots[1]),
                                 "seed %d pass %d: state differs" % (seed, number))
                compared += 1
                for call in outcome[1][0]:
                    seen[call[0] + (" --state=PENDING" if "--state=PENDING" in call else "")] += 1
                for world in worlds:
                    world.advance(random.Random("%d/%d" % (seed, number)).choice((60, 60, 600, 3600)))
            for event in scrubbed_events(roots[1]):
                seen["event " + event["event"]] += 1
        self.assertGreater(compared, len(seeds))
        sys.stderr.write("equivalence: %d passes over %d seeds; %s\n" % (compared, len(seeds),
                                                                           dict(sorted(seen.items()))))
        if len(seeds) >= 100:  # the histories reach every kind of decision (else the comparison proves little)
            for kind in ("sbatch", "scancel", "scontrol", "sacct", "event submit", "event cancel", "event job_ended",
                         "event worker_failure", "event exclude_node", "event breaker_open", "event reaper_cancel",
                         "event submit_paused", "event breaker_closed"):
                self.assertGreater(seen[kind], 0, kind)
            self.assertEqual(seen["scancel --state=PENDING"], 0)  # one account: no candidate pairs to settle


if __name__ == "__main__":
    unittest.main()
