"""The collector's output conforms to CONTRACT.md (tests/validate_contract.py), and the validator has teeth.

    cd /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/dashboard
    /usr/bin/python3 -I -B -m unittest discover -s tests -p test_contract.py -v

ConformTest   one collector cycle over test_collect's synthetic official build (every decision rule, a live probe,
              a scrape error, a hub sample) validates without a single problem; so do a degraded cycle and one with
              withdrawn tasks
MutationTest  copies of that output with one contract violation each: the validator names every one (also for the
              v1.2, v1.3 and v1.4 fields, which older output may lack)
SeedPrecedenceTest  v1.5, acceptance beats an invalid seed (official1-wasmi cand-c03-r1b2): the synthetic build with
              accepted tasks that seeds.json lists invalid conforms, also with --sources against its own files
              (accepted/, withdrawn/, seeds.json, metrics.json); the v1.4 collector's discard of such a task, a broken
              seed_invalid note and the round-0 rule are caught; withdrawn tasks stay not accepted, and a withdrawn
              task left in accepted/ is still an accepted-dir mismatch
BurnTest      api.json eta_exhaust at the higher of the 1 h and 6 h rates: the validator accepts the collector's own
              projection whichever rate is higher, and flags one made at the lower rate
BatchContractTest  v1.6, a batch from another tree (test_collect.BatchFixture): the cycle conforms, also with --sources
              against the batch's outputs/ and runs; copies breaking one batch rule each are caught (never audited,
              in_batch accepted and counted, the runs' and the batch's totals, the doc's own stamp, no job state or log,
              round statuses, the files), and --sources catches a provenance.json or accepted/tasks.jsonl that disagrees
ExplorerContractTest  v1.7: the cycle's explorer.json conforms (every row, run, target and total recomputed from the task
              and run files), also with --sources (candidate dirs, transcripts, patches); one planted violation per rule
              is caught (cells, rows, headers, runs, targets, totals, an audited batch task, stamps, the index entry)
OfficialQueueContractTest  v1.8: a cycle with two official runs of one target (test_collect.plant_official2) and the
              official2 queue (plant_queue) conforms, also with --sources (every official run dir with a launch or a
              build is one run, the latest is the last launch; the queue against queue.json, state.json,
              heartbeat.json and runner.log); one planted violation per rule is caught (latest, detail and its stamp,
              official_runs, the older run's file and stamp, the Explorer's runs, the queue's runner, budget, runs and
              log tail), and --sources catches an unlisted run dir and a queue that disagrees with its files
FileScanTest  next to a running collector: its temp files (".<name>.<pid>[.<thread>].tmp") are skipped unless left over
              (> 10 min), and an entry renamed away between the listing and its stat is absent, not a crash
The live data is checked by running the script itself: /usr/bin/python3 -I -B tests/validate_contract.py --sources
"""
import sys

sys.dont_write_bytecode = True

import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest import mock

import test_collect  # noqa: E402  (unittest discover puts tests/ on sys.path; it imports collect)
import validate_contract  # noqa: E402


def without_explorer(data):
    """Make a copy of the output look like an older collector's (before v1.7): no data/explorer.json and no index.json
    "explorer" (such a collector writes neither; v1.7 artifacts next to older NODE shapes would disagree with them)."""
    (Path(data) / "explorer.json").unlink(missing_ok=True)
    path = Path(data) / "index.json"
    doc = json.loads(path.read_text())
    doc.pop("explorer", None)
    path.write_text(json.dumps(doc))


def synthetic_output(cleanup):
    """One collector cycle over test_collect's synthetic tree (read-only afterwards; mutations work on copies)."""
    harness = test_collect.Harness(cleanup)
    summary = harness.cycle()
    assert summary["ok"], summary["errors"]
    return harness


class ConformTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = synthetic_output(cls.addClassCleanup)

    def test_synthetic_cycle_conforms(self):
        problems, checked = validate_contract.validate(self.h.data, self.h.state)
        self.assertEqual(problems, [])
        self.assertGreater(checked.get(f"{test_collect.TASK} nodes", 0), 10)
        self.assertGreater(checked.get("workers", 0), 0)

    def test_degraded_cycle_conforms(self):
        """Broken sources (no fleet status, no targets / three-way CSV, Slurm down): the collector's fallback output is
        still valid, its null sections explained by index.json errors."""
        h = test_collect.Harness(self.addCleanup)
        (h.fx.root / "student/state/status.json").unlink()
        shutil.rmtree(h.fx.root / "data/targets")
        h.fx.three_way.unlink()

        def broken_slurm(ids):
            raise RuntimeError("slurmctld down")
        sources = test_collect.collect.Sources(root=h.fx.root, three_way=h.fx.three_way)
        summary = test_collect.collect.run_cycle(h.data, h.state, sources=sources, clock=lambda: test_collect.NOW,
                                                 slurm=broken_slurm, scrape=h.scrape, hub=False)
        self.assertFalse(summary["ok"])
        problems, _ = validate_contract.validate(h.data, h.state)
        self.assertEqual(problems, [])
        # The run's task is still listed (no accuracy data), so the page can show the conversion.
        task = {t["task_id"]: t for t in json.loads((h.data / "index.json").read_text())["tasks"]}[test_collect.TASK]
        self.assertEqual((task["run"], task["accuracy"]), (test_collect.RUN, {"kimi": None, "super_vl": None,
                                                                              "sc14": None}))

    def test_withdrawn_cycle_conforms(self):
        """v1.4: tasks main withdrew (metrics.json "withdrawn", and a task dir moved to runs/<run>/withdrawn/) are
        decided as not accepted and left out of the audit's scope; the output still conforms."""
        h = test_collect.Harness(self.addCleanup)
        path = h.fx.build / "round-2/cand-c01-r1b0-r2b1/metrics.json"
        data = json.loads(path.read_text())
        data.update(accept=False, reason="withdrawn: grading artifact",
                    withdrawn={"at": "2026-10-18T23:03:14-0700", "by": "main", "why": "grading artifact",
                               "was": {"accept": True}})
        test_collect.write(path, data)
        test_collect.write(h.fx.runs / test_collect.RUN / "withdrawn/syn-cand-c03-r1b0/task.toml", "x")
        self.assertTrue(h.cycle(hub=False)["ok"])
        problems, checked = validate_contract.validate(h.data, h.state)
        self.assertEqual(problems, [])
        self.assertEqual(checked[f"{test_collect.TASK} eligible"], 6)


class MutationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = synthetic_output(cls.addClassCleanup)

    def mutated(self, mutate):
        """Problems of a copy of the synthetic output after mutate(data_dir)."""
        tmp = Path(tempfile.mkdtemp(prefix="dash-contract-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        shutil.copytree(self.h.data, tmp / "data")
        mutate(tmp / "data")
        return validate_contract.validate(tmp / "data", tmp / "no-state")[0]

    @staticmethod
    def edit(data, name, change):
        path = data / name
        doc = json.loads(path.read_text())
        change(doc)
        path.write_text(json.dumps(doc))

    def node_edit(self, decision, change):
        """Mutate the first node with `decision` of the synthetic task file."""
        def mutate(data):
            def apply(doc):
                node = next(n for n in doc["nodes"] if n["decision"] == decision)
                change(node, doc)
            self.edit(data, f"task/{test_collect.TASK}.json", apply)
        return self.mutated(mutate)

    def assertProblem(self, problems, needle):
        self.assertTrue(any(needle in problem for problem in problems), f"{needle!r} not in {problems}")

    def test_index_rules(self):
        def flip(doc):
            task = next(t for t in doc["tasks"] if t["eligible"])
            task["eligible"], task["queue_rank"] = False, None
        self.assertProblem(self.mutated(lambda d: self.edit(d, "index.json", flip)), "eligible must be True")

        def rank(doc):
            for task in doc["tasks"]:
                if task["eligible"]:
                    task["queue_rank"] = 1
        self.assertProblem(self.mutated(lambda d: self.edit(d, "index.json", rank)), "queue ranks must be 1..N")

        def card(doc):
            doc["tasks"][0]["accuracy"]["kimi"]["passed"] += 1
        self.assertProblem(self.mutated(lambda d: self.edit(d, "index.json", card)), "passed must count")

    def test_decision_rules(self):
        self.assertProblem(self.node_edit("discard", lambda n, d: n.update(decision="evolve")), "discard")
        self.assertProblem(self.node_edit("accepted_evolve", lambda n, d: n.update(decision="evolve")),
                           "accepted_evolve")
        self.assertProblem(self.node_edit("abandoned", lambda n, d: n.update(status="scored")), "abandoned")
        self.assertProblem(self.node_edit("pending", lambda n, d: n.update(decision="end")), "must be pending")

    def test_tree_structure(self):
        self.assertProblem(self.node_edit("evolve", lambda n, d: n.update(parent="cand-zz")), "parent must be")
        self.assertProblem(self.node_edit("evolve", lambda n, d: n["children"].pop()), "misses")
        self.assertProblem(self.node_edit("evolve", lambda n, d: n.update(summary="x" * 700)), "<= 600 chars")
        self.assertProblem(self.node_edit("evolve", lambda n, d: n["details"].pop("rubric_md")), "missing keys")

    def test_cycle_and_files(self):
        self.assertProblem(self.mutated(lambda d: self.edit(d, f"task/{test_collect.TASK}.json",
                                                            lambda doc: doc.update(generated_at=1.0))),
                           "not one cycle")
        self.assertProblem(self.mutated(lambda d: (d / "task/stray.json").write_text("{}")), "task files")
        self.assertProblem(self.mutated(lambda d: (d / "fleet.json").write_text('{"schema": NaN}')), "unreadable")
        self.assertProblem(self.mutated(lambda d: self.edit(d, "fleet.json", lambda doc: doc.pop("router"))),
                           "missing keys ['router']")

    def test_v12_fields(self):
        """Fields added in v1.2 are checked when present: focus patterns, live partial counts / own_done, the teacher
        audit list (the older JSON string still passes), busy workers, the prompt tok/s alias, the router scope, the
        burn windows, the ETA from the hub sample's time, the scan."""
        task = f"task/{test_collect.TASK}.json"

        def node(node_id, change):
            return self.mutated(lambda d: self.edit(d, task, lambda doc: change(
                next(n for n in doc["nodes"] if n["id"] == node_id))))
        self.assertProblem(node("cand-c01-r1b1", lambda n: n.update(focus_pattern="Scope creep")),
                           "focus_pattern must be 'Tests certify the bug'")
        self.assertProblem(node("cand-c01", lambda n: n.update(focus_pattern="Scope creep")), "a mask node has no")
        self.assertProblem(node("cand-c03-r1b1", lambda n: n["probe"]["partial"]["sc14"].update(done=3)),
                           "pass + fail + infra == done")
        self.assertProblem(node("cand-c03-r1b1", lambda n: n["probe"]["partial"]["kimi"].update(fail=1, infra=0)),
                           "differ from probe.teacher")
        self.assertProblem(node("cand-c03-r1b1", lambda n: n["probe"].update(own_done=True)), "must be False")
        self.assertProblem(node("cand-c01-r1b0", lambda n: n["probe"].update(own_done=True)), "probing node")
        self.assertProblem(node("cand-c01", lambda n: n["details"]["teacher_audit"][0].update(summary="x" * 401)),
                           "<= 400 chars")
        self.assertProblem(node("cand-c01", lambda n: n["details"]["teacher_audit"].pop()), "passing rollouts")
        self.assertEqual(self.mutated(lambda d: (self.edit(d, task, lambda doc: next(  # a v1.1 collector's string
            n for n in doc["nodes"] if n["id"] == "cand-c01")["details"].update(teacher_audit='{"kimi-r0": {}}')),
            without_explorer(d))), [])

        def fleet(change):
            return self.mutated(lambda d: self.edit(d, "fleet.json", change))
        self.assertProblem(fleet(lambda doc: doc["totals"].update(busy_workers=2)), "busy_workers")
        self.assertProblem(fleet(lambda doc: doc["totals"].update(prompt_tps=1.0)), "must equal prompt_tps_incl_cache")
        self.assertProblem(fleet(lambda doc: [doc["workers"][0]["metrics"].pop(key) for key in
                                              ("prompt_tps", "prompt_tps_incl_cache")]), "(or the old 'prompt_tps')")
        self.assertEqual(fleet(lambda doc: doc["totals"].pop("prompt_tps_incl_cache")), [])  # an older collector
        self.assertProblem(fleet(lambda doc: doc["router"].update(scope="all")), "official-run probes")

        def api(change):
            return self.mutated(lambda d: self.edit(d, "api.json", change))

        def projected(doc, at_sample):  # one hub sample, taken 5 min before the cycle; 900 $ left at 10 $/h
            doc["hub"]["sampled_at"] = doc["generated_at"] - 300
            eta = (doc["hub"]["sampled_at"] if at_sample else doc["generated_at"]) + 900 / 10.0 * 3600
            doc["burn"].update(per_hour_6h=10.0, window_min_6h=55.0, eta_exhaust=round(eta),
                               eta_exhaust_iso=test_collect.collect.iso(eta))
        self.assertEqual(api(lambda doc: projected(doc, True)), [])
        self.assertProblem(api(lambda doc: projected(doc, False)), "eta_exhaust must be hub.sampled_at")
        self.assertProblem(api(lambda doc: doc["burn"].update(per_hour_1h=5.0)), "null together")
        self.assertProblem(api(lambda doc: doc["burn"].update(per_hour_1h=5.0, window_min_1h=75.0)), "(0, 60]")
        self.assertProblem(api(lambda doc: doc["scan"].update(pending_bytes=5)), "no pending bytes")

    def test_v13_fields(self):
        """Failure-match audits (v1.3) are checked when present: each node's audit (verdict, limits, a built task) and
        index.json runs[].audit against the run's nodes; the output of an older collector (no such keys) still passes."""
        task = f"task/{test_collect.TASK}.json"
        _, checked = validate_contract.validate(self.h.data, self.h.state)
        self.assertEqual(checked[f"{test_collect.TASK} audited"], 2)

        def node(node_id, change):
            return self.mutated(lambda d: self.edit(d, task, lambda doc: change(
                next(n for n in doc["nodes"] if n["id"] == node_id))))
        self.assertProblem(node("cand-c03", lambda n: n["failure_match"].update(verdict="moderate")), "verdict")
        self.assertProblem(node("cand-c03", lambda n: n["failure_match"].update(summary="x" * 1001)), "<= 1000 chars")
        self.assertProblem(node("cand-c03", lambda n: n["failure_match"]["patterns"][0].update(reproduced="maybe")),
                           "yes, partial, no or null")
        self.assertProblem(node("cand-c03", lambda n: n["failure_match"]["student_rollouts"][1]["failing_tests"]
                                .extend(["t"] * 8)), "<= 8 test names")
        self.assertProblem(node("cand-c03", lambda n: n["failure_match"].pop("verdict")), "missing keys ['verdict']")

        def copy_audit(target):  # cand-c03's audit planted on another node
            def change(doc):
                nodes = {n["id"]: n for n in doc["nodes"]}
                nodes[target]["failure_match"] = dict(nodes["cand-c03"]["failure_match"])
            return self.mutated(lambda d: self.edit(d, task, change))
        self.assertProblem(copy_audit("cand-c04"), "only a built task")
        self.assertProblem(copy_audit("cand-c01-r1b0-r2b1"), "!= the counts of the accepted nodes")

        def index(change):
            return self.mutated(lambda d: self.edit(d, "index.json", lambda doc: change(doc["runs"][0]["audit"])))
        self.assertProblem(index(lambda audit: audit.update(strong=2)), "must equal audited")
        self.assertProblem(index(lambda audit: audit.update(accepted=4)), "accepted must equal runs[].accepted")
        self.assertProblem(index(lambda audit: audit.pop("none")), "missing keys ['none']")

        def older(data):  # a v1.2 collector: no failure_match on the nodes, no runs[].audit
            without_explorer(data)
            self.edit(data, "index.json", lambda doc: [run.pop("audit") for run in doc["runs"]])
            self.edit(data, task, lambda doc: [n.pop(key) for n in doc["nodes"]
                                               for key in ("failure_match", "audit_eligible", "withdrawn")])
        self.assertEqual(self.mutated(older), [])

    def test_v14_fields(self):
        """The failure-match audit's scope (v1.4) is checked when present: each node's audit_eligible against its score,
        its graded sc14 rollouts, its round's seeds.json and its withdrawal; NODE.withdrawn (shape, never accepted);
        index.json runs[].audit's eligible counts against the nodes. Output of an older collector still passes."""
        task = f"task/{test_collect.TASK}.json"
        _, checked = validate_contract.validate(self.h.data, self.h.state)
        self.assertEqual((checked[f"{test_collect.TASK} eligible"], checked[f"{test_collect.TASK} audited eligible"]),
                         (8, 3))

        def node(node_id, change):
            return self.mutated(lambda d: self.edit(d, task, lambda doc: change(
                next(n for n in doc["nodes"] if n["id"] == node_id))))
        self.assertProblem(node("cand-c01-r1b1", lambda n: n.update(audit_eligible=False)),
                           "audit_eligible must be True")
        self.assertProblem(node("cand-c02", lambda n: n.update(audit_eligible=True)),
                           "must be False (invalid in its round's seeds.json)")
        self.assertProblem(node("cand-c01-r1b0-r2b4", lambda n: n.update(audit_eligible=True)), "recomputed seed_check")
        self.assertProblem(node("cand-c03-r1b1", lambda n: n.update(audit_eligible=True)), "must be False (not scored)")
        self.assertProblem(node("cand-c01", lambda n: n.update(audit_eligible="yes")), "must be true or false")

        def ungraded(n):
            for row in n["probe"]["rollouts"]:
                if row["model"] == "sc14":
                    row["status"] = "infra"
        self.assertProblem(node("cand-c01-r1b1", ungraded), "must be False (no graded sc14 rollout)")
        withdrawn = {"at": None, "by": "main", "why": None}
        self.assertProblem(node("cand-c03-r1b0", lambda n: n.update(withdrawn=withdrawn)),
                           "a withdrawn task is never accepted")
        self.assertProblem(node("cand-c01-r1b1", lambda n: n.update(withdrawn=withdrawn)), "must be False (withdrawn)")
        self.assertProblem(node("cand-c01-r1b1", lambda n: n.update(withdrawn=dict(withdrawn, why="x" * 601))),
                           "<= 600 chars")
        self.assertProblem(node("cand-c01-r1b1", lambda n: n.update(withdrawn={"why": "x"})), "missing keys")

        def index(change):
            return self.mutated(lambda d: self.edit(d, "index.json", lambda doc: change(doc["runs"][0]["audit"])))
        self.assertProblem(index(lambda audit: audit.update(eligible=9)), "!= the counts of the eligible nodes")
        self.assertProblem(index(lambda audit: audit["eligible_verdicts"].update(none=1)),
                           "eligible_verdicts must sum to audited_eligible")
        self.assertProblem(index(lambda audit: audit.update(audited_eligible=9)), "audited_eligible must be <= eligible")
        self.assertProblem(index(lambda audit: audit.pop("eligible_verdicts")), "missing keys ['eligible_verdicts']")
        self.assertProblem(index(lambda audit: [audit.pop(key) for key in ("eligible", "audited_eligible",
                                                                           "eligible_verdicts")]),
                           "exactly when the nodes carry audit_eligible")

        def older(data):  # a v1.3 collector: no audit_eligible / withdrawn on the nodes, no eligible counts
            without_explorer(data)
            self.edit(data, "index.json", lambda doc: [run["audit"].pop(key) for run in doc["runs"]
                                                       for key in ("eligible", "audited_eligible", "eligible_verdicts")])
            self.edit(data, task, lambda doc: [n.pop(key) for n in doc["nodes"] for key in ("audit_eligible",
                                                                                            "withdrawn")])
        self.assertEqual(self.mutated(older), [])

    def test_secrets(self):
        planted = "nvapi-" + "K" * 30

        def plant(doc):
            doc["nodes"][0]["title"] = f"leak {planted}"
        self.assertProblem(self.mutated(lambda d: self.edit(d, f"task/{test_collect.TASK}.json", plant)),
                           "key-like text")
        self.assertProblem(self.mutated(lambda d: self.edit(d, "api.json",
                                                            lambda doc: doc["hub"].update(key=planted))),
                           "NAME of the key variable")


class SeedPrecedenceTest(unittest.TestCase):
    """v1.5 (main, 2026-10-21): acceptance beats an invalid seed. official1-wasmi round-1 cand-c03-r1b2 was accepted
    (runs/<run>/accepted/, metrics.json accept) and listed invalid in round-1/seeds.json (kimi-r3 never graded); the
    page showed it discarded and validate_contract.py --sources failed "accepted dir [6] != accepted nodes [5]". The
    synthetic build with the same case (test_collect.plant_accepted_invalid_seeds) conforms, also against its own files
    (--sources); copies with the old decision or a broken seed note are caught; withdrawn tasks stay not accepted, and a
    withdrawn task left in accepted/ is still a mismatch, as before."""

    ACCEPTED = ["cand-c03", "cand-c03-r1b0", "cand-c01-r1b0-r2b1", "cand-c02"]  # two of them invalid seeds
    TASK_FILE = f"task/{test_collect.TASK}.json"

    def harness(self):
        """The synthetic build with both accepted invalid seeds, and runs/<run>/accepted/ as task_builder fills it."""
        h = test_collect.Harness(self.addCleanup)
        test_collect.plant_accepted_invalid_seeds(h.fx)
        for name in self.ACCEPTED:
            self.task_dir(h, "accepted", name)
        return h

    @staticmethod
    def task_dir(h, where, name):
        """runs/<run>/<where>/syn-<name>/ and its .workspace archive, as task_builder accept() (or main's withdrawal)
        leaves them."""
        base = h.fx.runs / test_collect.RUN / where
        test_collect.write(base / f"syn-{name}/task.toml", "x")
        test_collect.write(base / f"syn-{name}.workspace/manifest.json", {})

    @staticmethod
    def validate(h, data=None, sources=True):
        """validate_contract.validate() with --sources against the synthetic tree (accuracy from the fixture's CSVs)."""
        with mock.patch.object(validate_contract, "THREE_WAY", h.fx.three_way), \
                mock.patch.object(validate_contract, "TARGETS", h.fx.root / "data/targets/sc14-out131k-kimi"):
            return validate_contract.validate(data or h.data, h.state, sources=sources)

    def copy(self, h, change):
        """A copy of the cycle's output whose task file went through change({node id: node})."""
        tmp = Path(tempfile.mkdtemp(prefix="dash-seed-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        shutil.copytree(h.data, tmp / "data")
        MutationTest.edit(tmp / "data", self.TASK_FILE, lambda doc: change({n["id"]: n for n in doc["nodes"]}))
        return tmp / "data"

    def assertProblem(self, problems, needle):
        self.assertTrue(any(needle in problem for problem in problems), f"{needle!r} not in {problems}")

    def test_conforms_with_sources(self):
        h = self.harness()
        self.assertTrue(h.cycle(hub=False)["ok"])
        problems, checked = self.validate(h)
        self.assertEqual(problems, [])
        self.assertEqual(checked[f"{test_collect.RUN} accepted invalid seeds"], 2)  # cand-c02 (round 0), cand-c03-r1b0
        self.assertEqual(checked[f"{test_collect.TASK} eligible"], 9)
        self.assertEqual(checked[f"{test_collect.RUN} cross-checked nodes"], checked[f"{test_collect.TASK} nodes"])
        self.assertEqual(self.validate(h, sources=False)[0], [])

    def test_old_decision_is_caught(self):
        """What a v1.4 collector wrote for such a task (discard with the seeds.json reason, outside the audit's scope,
        no note) is caught in the task file and against the pipeline's files, as on the live data on 2026-10-21."""
        h = self.harness()
        h.cycle(hub=False)

        def old(nodes):
            node = nodes["cand-c03-r1b0"]
            node.update(decision="discard", decision_reason=test_collect.SEED_INVALID, audit_eligible=False)
            del node["seed_invalid"]
        problems = self.validate(h, self.copy(h, old))[0]
        for needle in ("cand-c03-r1b0: listed invalid in seeds.json and accepted: must be accepted, got discard",
                       "cand-c03-r1b0: an accepted task is never discarded (v1.5: acceptance beats an invalid seed)",
                       "cand-c03-r1b0: audit_eligible must be True (scored, a graded sc14 rollout, accepted",
                       "seeds.json invalid cand-c03-r1b0, accepted by its metrics.json: expected accepted with its "
                       "reason as seed_invalid, got discard",
                       "accepted dir ['syn-cand-c01-r1b0-r2b1', 'syn-cand-c02', 'syn-cand-c03', "
                       "'syn-cand-c03-r1b0'] != accepted nodes ['syn-cand-c01-r1b0-r2b1', 'syn-cand-c02', "
                       "'syn-cand-c03']"):
            self.assertProblem(problems, needle)

    def test_seed_note_rules(self):
        """NODE.seed_invalid: the seeds.json reason (null for a valid seed), null for an unscored task or a failed
        build, one line, a discard's reason; round 0: an accepted invalid seed is accepted, not accepted_evolve."""
        h = self.harness()
        h.cycle(hub=False)

        def problems(node_id, **fields):
            return self.validate(h, self.copy(h, lambda nodes: nodes[node_id].update(fields)), sources=False)[0]
        self.assertProblem(problems("cand-c03-r1b0", seed_invalid=None),
                           f"seed_invalid must be its round's seeds.json reason {test_collect.SEED_INVALID!r}")
        self.assertProblem(problems("cand-c03-r1b0", seed_invalid="no grade\nfor kimi-r3"), "null or one line of text")
        self.assertProblem(problems("cand-c01", seed_invalid="x"), "seeds.json lists it as a valid seed")
        self.assertProblem(problems("cand-c03-r1b1", seed_invalid="x"), "null for a task never scored or not built")
        self.assertProblem(problems("cand-c04", seed_invalid="x"), "null for a task never scored or not built")
        self.assertProblem(problems("cand-c01-r1b0-r2b4", seed_invalid="x"), "its decision_reason is its seed_invalid")
        self.assertProblem(problems("cand-c01-r1b0-r2b0", seed_invalid="x"), "a task with seed_invalid is discard")
        self.assertProblem(problems("cand-c02", decision="accepted_evolve"),
                           "a round-0 node, accepted and an invalid seed, is accepted, got accepted_evolve")
        self.assertProblem(problems("cand-c02", decision="discard", decision_reason=test_collect.C02_INVALID),
                           "an accepted task is never discarded")
        # A collector that predates the note: the decisions alone still conform.
        older = self.copy(h, lambda nodes: [node.pop("seed_invalid") for node in nodes.values()])
        self.assertEqual(self.validate(h, older)[0], [])

    def test_withdrawn_stays_not_accepted(self):
        """Main withdraws cand-c03-r1b0 by moving its task dir to withdrawn/ (metrics.json still says accept: true) and
        cand-c02 by rewriting its metrics.json as well: both are discarded invalid seeds again, and accepted/ agrees. A
        withdrawn task whose dir is still in accepted/ is a mismatch, as before."""
        h = self.harness()
        runs = h.fx.runs / test_collect.RUN
        for name in ("cand-c03-r1b0", "cand-c02"):
            for suffix in ("", ".workspace"):
                shutil.move(runs / f"accepted/syn-{name}{suffix}", runs / f"withdrawn/syn-{name}{suffix}")
        path = h.fx.build / "round-0/cand-c02/metrics.json"
        data = json.loads(path.read_text())
        data.update(accept=False, reason="withdrawn: grading artifact", withdrawn={
            "at": "2026-10-21T09:00:00-0700", "by": "main", "why": "grading artifact", "was": {"accept": True}})
        test_collect.write(path, data)
        self.assertTrue(h.cycle(hub=False)["ok"])
        nodes = {n["id"]: n for n in json.loads((h.data / self.TASK_FILE).read_text())["nodes"]}
        self.assertEqual([nodes[name]["decision"] for name in ("cand-c03-r1b0", "cand-c02")], ["discard", "discard"])
        problems, checked = self.validate(h)
        self.assertEqual(problems, [])
        self.assertEqual(checked[f"{test_collect.RUN} withdrawn"], 2)
        self.assertNotIn(f"{test_collect.RUN} accepted invalid seeds", checked)
        # withdrawn (metrics.json) while its task dir is back in accepted/: never an accepted node, so a mismatch
        self.task_dir(h, "accepted", "cand-c02")
        self.assertProblem(self.validate(h)[0], "accepted dir ['syn-cand-c01-r1b0-r2b1', 'syn-cand-c02', "
                                                "'syn-cand-c03'] != accepted nodes ['syn-cand-c01-r1b0-r2b1', "
                                                "'syn-cand-c03']")


class BatchContractTest(unittest.TestCase):
    """v1.6: a cycle with a batch from another tree (test_collect.BatchFixture) conforms, also with --sources against
    the batch's own outputs/ and runs; copies with one violation each of the batch rules are caught (never audited,
    in_batch accepted and counted, the runs' and the batch's totals, the doc's own stamp, the empty job state, the round
    statuses, the files), and --sources catches a provenance.json or an accepted/tasks.jsonl that disagrees."""

    BATCH = test_collect.BATCH
    OFFICIAL_ACCEPTED = ["cand-c03", "cand-c03-r1b0", "cand-c01-r1b0-r2b1"]  # the official build's accepted tasks

    @classmethod
    def setUpClass(cls):
        cls.h = test_collect.Harness(cls.addClassCleanup, batch=True)
        for name in cls.OFFICIAL_ACCEPTED:  # runs/<run>/accepted/ as task_builder fills it (for --sources)
            SeedPrecedenceTest.task_dir(cls.h, "accepted", name)
        assert cls.h.cycle(hub=False)["ok"]
        cls.doc_a = f"run/{cls.BATCH}/pilotA-task-b.json"

    def copy(self, mutate):
        tmp = Path(tempfile.mkdtemp(prefix="dash-batch-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        shutil.copytree(self.h.data, tmp / "data")
        mutate(tmp / "data")
        return tmp / "data"

    def problems(self, mutate, sources=False):
        return SeedPrecedenceTest.validate(self.h, self.copy(mutate), sources=sources)[0]

    def node(self, node_id, change, doc=None):
        def mutate(data):
            MutationTest.edit(data, doc or self.doc_a, lambda d: change(next(n for n in d["nodes"] if n["id"] == node_id)))
        return mutate

    def index(self, change):
        return lambda data: MutationTest.edit(data, "index.json", change)

    @staticmethod
    def batch_run(doc, name="pilotA-task-b"):
        return next(run for run in doc["runs"] if run.get("run") == name)

    def assertProblem(self, problems, needle):
        self.assertTrue(any(needle in problem for problem in problems), f"{needle!r} not in {problems}")

    def test_conforms(self):
        problems, checked = validate_contract.validate(self.h.data, self.h.state)
        self.assertEqual(problems, [])
        self.assertEqual((checked[f"batch {self.BATCH} tasks"], checked[f"batch {self.BATCH} in_batch"],
                          checked["batch runs"]), (6, 6, 3))
        self.assertEqual(checked["pilotA-task-b in_batch"], 3)

    def test_conforms_with_sources(self):
        problems, checked = SeedPrecedenceTest.validate(self.h, sources=True)
        self.assertEqual(problems, [])
        self.assertEqual(checked[f"sources batch {self.BATCH} tasks vs provenance"], 6)
        self.assertEqual(checked[f"sources batch {self.BATCH} runs vs outputs"], 3)
        self.assertEqual(checked["pilotA-task-b cross-checked nodes"], 7)
        self.assertEqual(checked["pilotA-task-b focus patterns vs prompts"], 4)

    def test_batch_node_rules(self):
        self.assertProblem(self.problems(self.node("cand-c01-r1b1", lambda n: n.update(in_batch=True))),
                           "an in_batch node must be accepted (every task of the batch), got end")
        self.assertProblem(self.problems(self.node("cand-c01", lambda n: n.update(decision="evolve"))),
                           "an in_batch node must be accepted")
        self.assertProblem(self.problems(self.node("cand-c02", lambda n: n.update(audit_eligible=True))),
                           "audit_eligible must be False (a batch is never audited)")
        planted = {"verdict": "strong", "summary": "x", "patterns": [], "student_rollouts": [], "other_causes": None,
                   "similarity_judge": None, "audited_at": None}
        self.assertProblem(self.problems(self.node("cand-c01", lambda n: n.update(failure_match=planted))),
                           "failure_match must be null (a batch is never audited)")
        self.assertProblem(self.problems(self.node("cand-c01", lambda n: n.update(audit={"status": "audited"}))),
                           "audit must be {'status': 'not_audited'}")
        self.assertProblem(self.problems(self.node("cand-c01", lambda n: n.pop("in_batch"))), "missing keys ['in_batch']")
        self.assertProblem(self.problems(self.node("cand-c02-r1b1", lambda n: n.update(decision="end"))),
                           "listed invalid in seeds.json and not accepted: must be discard, got end")  # official rules

    def test_batch_doc_rules(self):
        def doc(change):
            return lambda data: MutationTest.edit(data, self.doc_a, change)
        self.assertProblem(self.problems(doc(lambda d: d.update(generated_at=d["generated_at"] - 60))), "not one cycle")
        self.assertProblem(self.problems(doc(lambda d: d["job"].update(slurm_state="RUNNING"))),
                           "slurm_state must be null")
        self.assertProblem(self.problems(doc(lambda d: d["job"].update(log_tail=["x"]))), "log_tail must be []")
        self.assertProblem(self.problems(doc(lambda d: d["rounds"][0].update(status="probing"))),
                           "one of ['done', 'pending', 'unfinished']")
        self.assertProblem(self.problems(doc(lambda d: d["rounds"][0].update(unfinished_probes=["round-0/probe-x"]))),
                           "unfinished_probes must be probe.dirs")
        self.assertProblem(self.problems(doc(lambda d: d.update(audit=None))), "a batch is never audited")
        self.assertProblem(self.problems(doc(lambda d: d.update(source="other"))), "source differs")
        self.assertProblem(self.problems(lambda data: (data / f"run/{self.BATCH}/stray.json").write_text("{}")),
                           "run files")
        self.assertProblem(self.problems(lambda data: (data / self.doc_a).unlink()), "run files")

    def test_index_rules(self):
        def run(change, name="pilotA-task-b"):
            return self.index(lambda doc: change(self.batch_run(doc, name)))
        self.assertProblem(self.problems(run(lambda r: r.update(batch=2))), "3 in_batch nodes, but index runs[].batch is 2")
        self.assertProblem(self.problems(run(lambda r: r.update(slurm_state="RUNNING"))),
                           "null (a batch's jobs are not polled)")
        self.assertProblem(self.problems(run(lambda r: r.update(status="running"))), "an ended run")
        self.assertProblem(self.problems(run(lambda r: r.update(audit={"accepted": 3, "audited": 0}))),
                           "a batch is never audited")
        self.assertProblem(self.problems(run(lambda r: r.update(detail_generated_at=1.0))), "not one cycle")
        self.assertProblem(self.problems(run(lambda r: r.update(root="/elsewhere/pilotA-task-b"))),
                           "<root>/synth/runs/<run>")

        def unnamed(doc):
            task = next(t for t in doc["tasks"] if t["task_id"] == "task-b")
            task["batch_runs"] = task["batch_runs"][:1]
        self.assertProblem(self.problems(self.index(unnamed)), "every batch run must be named once")

        def batches(change):
            return self.index(lambda doc: change(doc["batches"][0]))
        self.assertProblem(self.problems(batches(lambda b: b.update(tasks=7))), "differ")
        self.assertProblem(self.problems(batches(lambda b: b.update(accepted=5))), "accepted must sum its runs")
        self.assertProblem(self.problems(batches(lambda b: b.update(unmatched=["syn-x: no node"]))),
                           "outputs tasks that are no accepted node")
        self.assertProblem(self.problems(self.index(lambda doc: doc.pop("batches"))), "batch runs without index.json "
                                                                                       "batches")

    def test_sources_rules(self):
        bx = self.h.bx

        def unchanged(data):  # the output as written; the batch's own files change below (restored afterwards)
            pass
        path = bx.outputs / bx.tid("q1-task-c", "cand-c01") / "provenance.json"
        saved = path.read_text()
        self.addCleanup(path.write_text, saved)
        data = json.loads(saved)
        data["task_dir"] = data["task_dir"].replace("/cand-c01/", "/cand-c01-r1b1/")
        path.write_text(json.dumps(data))
        self.assertProblem(self.problems(unchanged, sources=True), "expected q1-task-c cand-c01-r1b1 accepted")
        path.write_text(saved)
        accepted = bx.runs / "pilotB-task-b/accepted/tasks.jsonl"
        saved_rows = accepted.read_text()
        self.addCleanup(accepted.write_text, saved_rows)
        accepted.write_text("")
        self.assertProblem(self.problems(unchanged, sources=True),
                           "run pilotB-task-b: accepted/tasks.jsonl [] != its outputs tasks")
        accepted.write_text(saved_rows)
        self.assertEqual(self.problems(unchanged, sources=True), [])


class ExplorerContractTest(unittest.TestCase):
    """v1.7: data/explorer.json of a cycle with the official build and a batch conforms (every row, run, target and
    total recomputed from the task and run files), also with --sources (candidate dirs, transcripts and patches on
    disk); a copy breaking one rule each is caught."""

    @classmethod
    def setUpClass(cls):
        cls.h = test_collect.Harness(cls.addClassCleanup, batch=True)
        for name in BatchContractTest.OFFICIAL_ACCEPTED:  # runs/<run>/accepted/ as task_builder fills it (for --sources)
            SeedPrecedenceTest.task_dir(cls.h, "accepted", name)
        assert cls.h.cycle(hub=False)["ok"]

    def problems(self, mutate, sources=False):
        tmp = Path(tempfile.mkdtemp(prefix="dash-explorer-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        shutil.copytree(self.h.data, tmp / "data")
        mutate(tmp / "data")
        return SeedPrecedenceTest.validate(self.h, tmp / "data", sources=sources)[0]

    @staticmethod
    def explorer(change):
        """A mutation of explorer.json; index.json's byte count follows, so only the planted violation is left."""
        def mutate(data):
            MutationTest.edit(data, "explorer.json", change)
            size = (data / "explorer.json").stat().st_size
            MutationTest.edit(data, "index.json", lambda doc: doc["explorer"].update(bytes=size))
        return mutate

    @staticmethod
    def cell(doc, kind, row, column, value):
        columns = doc["task_columns" if kind == "tasks" else "trajectory_columns"]
        doc[kind][row][columns.index(column)] = value

    def assertProblem(self, problems, needle):
        self.assertTrue(any(needle in problem for problem in problems), f"{needle!r} not in {problems}")

    def test_conforms(self):
        problems, checked = validate_contract.validate(self.h.data, self.h.state)
        self.assertEqual(problems, [])
        self.assertEqual((checked["explorer tasks"], checked["explorer trajectories"]), (28, 214))

    def test_conforms_with_sources(self):
        problems, checked = SeedPrecedenceTest.validate(self.h, sources=True)
        self.assertEqual(problems, [])
        # 28 task rows; 214 trajectories less the live probe's 4 (no transcript yet)
        self.assertEqual((checked["explorer cand dirs vs disk"], checked["explorer transcripts vs disk"]), (28, 210))

    def test_row_rules(self):
        """Each cell is its node's (the source of every column), and the row set is exactly the built nodes."""
        for kind, row, column, value, needle in [
                ("tasks", 0, "title", "another title", "title:"), ("tasks", 2, "similarity", 9, "similarity: 9 != 7"),
                ("tasks", 2, "student_rate", 0.4, "student_rate"), ("tasks", 0, "teacher_rate", 0.8, "teacher_rate"),
                ("tasks", 2, "audit", "weak", 'audit: "weak" != "strong"'),
                ("tasks", 2, "reproduced_yes", [1], "reproduced_yes"),
                ("tasks", 0, "decision", "accepted", "accepted must be decision"), ("tasks", 1, "scored", False, "scored"),
                ("tasks", 0, "run", 1, "run: 1 != 0"), ("tasks", 0, "kind", "evolved", "unexpected value"),
                ("tasks", 0, "rates_differ", True, "rates_differ: true != false"),
                ("tasks", 1, "rates_differ", False, "rates_differ: false != true"),   # cand-c02: valid 3 / 4, rows 5 / 5
                ("tasks", 0, "rates_differ", 0, "rates_differ"),
                ("trajectories", 0, "status", "fail", "status"),
                ("trajectories", 1, "clean", True, "clean: true != false"),
                ("trajectories", 0, "tool_calls", 13, "tool_calls"), ("trajectories", 0, "task", 5, "task: 5 != 0")]:
            self.assertProblem(self.problems(self.explorer(lambda doc: self.cell(doc, kind, row, column, value))), needle)
        self.assertProblem(self.problems(self.explorer(lambda doc: doc["trajectories"].pop())),
                           "rows, the task and run files")
        self.assertProblem(self.problems(self.explorer(lambda doc: doc["tasks"].pop(3))), "differs from its node")
        self.assertProblem(self.problems(self.explorer(lambda doc: doc["tasks"][0].pop())), "a list of 23 values")
        self.assertProblem(self.problems(self.explorer(
            lambda doc: doc["totals"]["hengyuf"].update(rates_differ=doc["totals"]["hengyuf"]["rates_differ"] + 1))), "totals")

    def test_batch_never_audited(self):
        def audit(doc):  # a batch task with a verdict and reproduced patterns
            row = next(i for i, r in enumerate(doc["tasks"]) if doc["runs"][r[0]]["source"] != "hengyuf")
            for column, value in (("audit", "strong"), ("reproduced_yes", [0]), ("reproduced_partial", [])):
                self.cell(doc, "tasks", row, column, value)
        self.assertProblem(self.problems(self.explorer(audit)), 'audit: "strong" != "not_audited"')
        self.assertProblem(self.problems(self.explorer(lambda doc: doc["sources"][1].update(audited=True))),
                           "only the official runs are audited")
        self.assertProblem(self.problems(self.explorer(lambda doc: doc["totals"][test_collect.BATCH].update(audited=1))),
                           "totals")

    def test_runs_targets_and_columns(self):
        self.assertProblem(self.problems(self.explorer(lambda doc: doc["runs"][1].update(link="#task=task-b"))),
                           "explorer.json runs[1]")
        self.assertProblem(self.problems(self.explorer(lambda doc: doc["runs"].reverse())), "explorer.json runs")
        self.assertProblem(self.problems(self.explorer(lambda doc: doc["targets"][0]["patterns"].reverse())),
                           "explorer.json targets[0]")
        self.assertProblem(self.problems(self.explorer(lambda doc: doc["task_columns"].reverse())), "task_columns must be")
        self.assertProblem(self.problems(self.explorer(lambda doc: doc["sources"].pop(0))), "must start with the official")

    def test_index_entry_and_stamps(self):
        self.assertProblem(self.problems(lambda data: MutationTest.edit(
            data, "index.json", lambda doc: doc["explorer"].update(bytes=1))), "bytes 1 != the file's")
        self.assertProblem(self.problems(lambda data: MutationTest.edit(
            data, "index.json", lambda doc: doc["explorer"].update(tasks=1))), "row counts")
        later = self.explorer(lambda doc: doc.update(generated_at=doc["generated_at"] + 60))
        self.assertProblem(self.problems(later), "later than index.json's")
        self.assertProblem(self.problems(later), "not one cycle")
        self.assertProblem(self.problems(lambda data: (data / "explorer.json").unlink()), "explorer.json: unreadable")
        self.assertProblem(self.problems(lambda data: MutationTest.edit(
            data, "index.json", lambda doc: doc.update(explorer=None))), "null with an 'explorer: ...' error")
        self.assertEqual(self.problems(lambda data: MutationTest.edit(  # not built this cycle, and said so: fine
            data, "index.json", lambda doc: doc.update(explorer=None, errors=["explorer: RuntimeError: x"]))), [])
        self.assertEqual(self.problems(without_explorer), [])  # an older collector writes neither

    def test_sources_rules(self):
        def drop(data):
            path = next(self.h.fx.build.rglob("round-0/cand-c01/rollouts/sc14-r0.patch"))
            path.rename(path.with_name("sc14-r0.patch.moved"))
            self.addCleanup(path.with_name("sc14-r0.patch.moved").rename, path)
        self.assertProblem(self.problems(drop, sources=True), "no transcript / patch")

    def test_secrets(self):
        planted = "nvapi-" + "K" * 30
        self.assertProblem(self.problems(self.explorer(lambda doc: self.cell(doc, "tasks", 0, "title", planted))),
                           "key-like text")


class BurnTest(unittest.TestCase):
    """eta_exhaust projects the hub sample's remaining budget at the HIGHER of the 1 h and 6 h rates (CONTRACT
    "Integration notes (v1.1)": conservative while probes burst; the page names the one used), as collect.py _burn()
    does. 2026-10-20: the validator expected the 6 h rate and flagged every projection made during a burst."""

    @staticmethod
    def problems(burn, remaining, sampled_at):
        r = validate_contract.Report()
        validate_contract.check_burn(r, burn, {"sampled_at": sampled_at, "remaining": remaining}, "api.json burn")
        return r.problems

    def test_collector_projection_passes(self):
        collect, at = test_collect.collect, test_collect.NOW + 6 * 3600
        burst, calm = test_collect.burn_histories()
        for history, remaining in ((burst, 700.0), (calm, 750.0)):
            burn = collect._burn(history, remaining, at)
            self.assertNotEqual(burn["per_hour_1h"], burn["per_hour_6h"])
            self.assertEqual(self.problems(burn, remaining, at), [])
        # The same burst projected at its lower (6 h) rate is a problem.
        burn = collect._burn(burst, 700.0, at)
        self.assertGreater(burn["per_hour_1h"], burn["per_hour_6h"])
        lower = at + 700.0 / burn["per_hour_6h"] * 3600
        burn.update(eta_exhaust=round(lower), eta_exhaust_iso=collect.iso(lower))
        self.assertTrue(any("eta_exhaust must be hub.sampled_at" in problem
                            for problem in self.problems(burn, 700.0, at)))

    def test_one_rate_or_none(self):
        """One measured window: its rate; a rate of 0 or below, or none: no projection is checked."""
        at = test_collect.NOW
        for rates, eta in (({"per_hour_1h": None, "per_hour_6h": 10.0}, at + 90 * 3600),
                           ({"per_hour_1h": 10.0, "per_hour_6h": None}, at + 90 * 3600),
                           ({"per_hour_1h": -3.6, "per_hour_6h": 10.0}, at + 90 * 3600),
                           ({"per_hour_1h": -3.6, "per_hour_6h": None}, None),
                           ({"per_hour_1h": None, "per_hour_6h": None}, None)):
            burn = dict(rates, window_min_1h=None if rates["per_hour_1h"] is None else 50.0,
                        window_min_6h=None if rates["per_hour_6h"] is None else 300.0, eta_exhaust=eta and round(eta),
                        eta_exhaust_iso=eta and test_collect.collect.iso(eta))
            self.assertEqual(self.problems(burn, 900.0, at), [], rates)


class FileScanTest(unittest.TestCase):
    """The validator's file scan next to a running collector, which replaces data/ and state/ files by renaming its
    temp files over them. 2026-10-20: a run crashed with FileNotFoundError on a temp file renamed away mid-scan."""

    @classmethod
    def setUpClass(cls):
        cls.h = synthetic_output(cls.addClassCleanup)

    def copy(self):
        """A private copy of the synthetic output and its state (modes kept)."""
        tmp = Path(tempfile.mkdtemp(prefix="dash-scan-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        shutil.copytree(self.h.data, tmp / "data")
        shutil.copytree(self.h.state, tmp / "state")
        return tmp / "data", tmp / "state"

    def test_copy_conforms(self):
        self.assertEqual(validate_contract.validate(*self.copy())[0], [])

    def test_collector_temp_files(self):
        """A temp file of a write in progress is skipped (not a data file, whatever its mode); one older than 10 min is
        a leftover."""
        data, state = self.copy()
        fresh = data / "task" / f".{test_collect.TASK}.json.4242.140000.tmp"
        fresh.write_text("{")
        fresh.chmod(0o666)
        self.assertEqual(validate_contract.validate(data, state)[0], [])
        stale = data / ".api.json.4242.140000.tmp"
        stale.write_text("{")
        os.utime(stale, (time.time() - 1200,) * 2)
        self.assertEqual(validate_contract.validate(data, state)[0], [f"{stale.name}: leftover temp file"])

    def test_vanished_entries_are_absent(self):
        """Entries listed but gone by the time they are looked at (the collector's rename landed in between) are
        absent: no crash, no problem, in data/ and in state/."""
        data, state = self.copy()
        ghosts = {data: [data / "task" / f".{test_collect.TASK}.json.4242.140000.tmp", data / "task" / "gone.json",
                         data / ".index.json.4242.140000.tmp"],
                  state: [state / ".collector.json.4242.140000.tmp", state / "collector.json.gone"]}
        rglob, iterdir = Path.rglob, Path.iterdir

        def then_gone(listing):
            def listed(self, *args):
                yield from listing(self, *args)
                yield from ghosts.get(self, [])
            return listed
        with mock.patch.object(Path, "rglob", then_gone(rglob)), mock.patch.object(Path, "iterdir", then_gone(iterdir)):
            self.assertTrue(any(path.name == "gone.json" for path in data.rglob("*")))  # the ghosts are listed
            problems, _ = validate_contract.validate(data, state)
        self.assertEqual(problems, [])


class OfficialQueueContractTest(unittest.TestCase):
    """v1.8: every official run has a document, and the official2 queue's runner state is in index.json."""

    @classmethod
    def setUpClass(cls):
        cls.h = test_collect.Harness(cls.addClassCleanup, batch=True, official2=True, queue=True)
        for name in BatchContractTest.OFFICIAL_ACCEPTED:  # runs/<run>/accepted/ as task_builder fills it (for --sources)
            SeedPrecedenceTest.task_dir(cls.h, "accepted", name)
        assert cls.h.cycle(hub=False)["ok"]

    def problems(self, mutate, sources=False):
        tmp = Path(tempfile.mkdtemp(prefix="dash-v18-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        shutil.copytree(self.h.data, tmp / "data")
        mutate(tmp / "data")
        return SeedPrecedenceTest.validate(self.h, tmp / "data", sources=sources)[0]

    @staticmethod
    def index(change):
        return lambda data: MutationTest.edit(data, "index.json", change)

    @staticmethod
    def run_entry(doc, name):
        return next(run for run in doc["runs"] if run["run"] == name)

    @staticmethod
    def queue_run(doc, name):
        return next(run for run in doc["queues"][0]["runs"] if run["run"] == name)

    def assertProblem(self, problems, needle):
        self.assertTrue(any(needle in problem for problem in problems), f"{needle!r} not in {problems}")

    def test_conforms(self):
        problems, checked = validate_contract.validate(self.h.data, self.h.state)
        self.assertEqual(problems, [])
        self.assertEqual((checked["official runs"], checked["older official runs"], checked["queue official2 runs"]),
                         (2, 1, 3))
        self.assertEqual(checked[f"{test_collect.RUN} nodes"], 19)  # the older run's document, checked as a task file

    def test_conforms_with_sources(self):
        problems, checked = SeedPrecedenceTest.validate(self.h, sources=True)
        self.assertEqual(problems, [])
        self.assertEqual((checked["sources official run dirs"], checked["sources official run dirs without launch or "
                                                                        "build"]), (2, 1))
        self.assertEqual((checked[f"{test_collect.RUN} cross-checked nodes"], checked[f"{test_collect.RUN2} cross-checked "
                                                                                       "nodes"]), (19, 3))
        self.assertEqual((checked["sources queue official2 runs vs queue.json"],
                          checked["sources queue official2 files compared"]), (3, 4))

    def test_official_run_rules(self):
        RUN, RUN2, TASK = test_collect.RUN, test_collect.RUN2, test_collect.TASK
        cases = [
            (lambda doc: self.run_entry(doc, RUN).update(latest=True), "latest official runs: expected exactly one"),
            (lambda doc: self.run_entry(doc, RUN).update(submitted_at=self.run_entry(doc, RUN2)["submitted_at"] + 60),
             "is not the most recently submitted"),
            (lambda doc: self.run_entry(doc, RUN).update(detail=f"run/hengyuf/{RUN}-x.json"), "an older official run"),
            (lambda doc: self.run_entry(doc, RUN).pop("detail_generated_at"), "detail_generated_at"),
            (lambda doc: self.run_entry(doc, RUN2).update(detail_generated_at=1.0), "no detail_generated_at"),
            (lambda doc: self.run_entry(doc, RUN2).update(detail=f"run/hengyuf/{RUN2}.json"), "task/<task_id>.json"),
            (lambda doc: next(t for t in doc["tasks"] if t["task_id"] == TASK)["official_runs"].reverse(), "official_runs"),
            (lambda doc: next(t for t in doc["tasks"] if t["task_id"] == TASK)["official_runs"].pop(), "official_runs"),
            (lambda doc: next(t for t in doc["tasks"] if t["task_id"] == "task-b").update(
                official_runs=[{"run": RUN, "latest": True, "detail": "task/task-b.json"}]), "without an official runs[]"),
        ]
        for change, needle in cases:
            self.assertProblem(self.problems(self.index(change)), needle)
        older = f"run/hengyuf/{RUN}.json"
        self.assertProblem(self.problems(lambda data: (data / older).unlink()), "run files")
        self.assertProblem(self.problems(lambda data: shutil.copy(data / older, data / "run/hengyuf/stray.json")),
                           "run files")
        self.assertProblem(self.problems(lambda data: MutationTest.edit(data, older, lambda doc: doc.update(
            generated_at=doc["generated_at"] - 60))), "detail_generated_at (not one cycle)")
        self.assertProblem(self.problems(lambda data: MutationTest.edit(data, older, lambda doc: doc.update(run=RUN2))),
                           "run differs from index runs[].run")
        self.assertProblem(self.problems(lambda data: MutationTest.edit(data, older, lambda doc: doc.update(
            source="baiheh"))), "carries no batch source")
        # v1.8: each launched job once in a document's job.jobs (a resolved babysit leg repeats its winner's line)
        self.assertProblem(self.problems(lambda data: MutationTest.edit(data, older, lambda doc: doc["job"].update(
            jobs=doc["job"]["jobs"] + doc["job"]["jobs"][-1:]))), "jobs repeat a job")
        # the older run's nodes are checked like a task file's (its audit counts against its runs[] entry)
        self.assertProblem(self.problems(self.index(lambda doc: self.run_entry(doc, RUN)["audit"].update(
            audited_eligible=0))), "!= the counts of the eligible nodes")

    def test_explorer_covers_every_official_run(self):
        def drop_older(doc):  # the explorer of a v1.7 collector: the latest runs only
            doc["runs"][0]["latest"] = True
        self.assertProblem(self.problems(ExplorerContractTest.explorer(drop_older)), "explorer.json runs[0]")
        self.assertProblem(self.problems(ExplorerContractTest.explorer(
            lambda doc: doc["runs"][0].update(link=f"#task={test_collect.TASK}"))), "explorer.json runs[0]")
        self.assertProblem(self.problems(ExplorerContractTest.explorer(
            lambda doc: doc["runs"][2].update(latest=False))), "explorer.json runs[2]")

    def test_queue_rules(self):
        RUN2 = test_collect.RUN2
        queue = lambda doc: doc["queues"][0]  # noqa: E731
        cases = [
            (lambda doc: queue(doc)["budget"].update(error="hub said no"), "null or exactly"),
            (lambda doc: queue(doc)["budget"].update(key="nvapi-abc"), "a key NAME or null"),
            (lambda doc: queue(doc)["runner"].update(alive=False), "alive must be True"),
            (lambda doc: queue(doc)["runner"].update(age_s=5.0), "age_s 5.0"),
            (lambda doc: queue(doc)["runner"].update(heartbeat_at=None), "both or neither"),
            (lambda doc: queue(doc)["runner"].update(pid=4242), "keys must be"),
            (lambda doc: self.queue_run(doc, RUN2).update(status="exploded"), "one of"),
            (lambda doc: self.queue_run(doc, RUN2).update(status=None), "unknown status' error"),
            (lambda doc: self.queue_run(doc, RUN2).update(winner="999"), "one of the candidates"),
            (lambda doc: self.queue_run(doc, RUN2).update(task="task-z"), "a DeepSWE task of tasks[]"),
            (lambda doc: self.queue_run(doc, RUN2).update(leg=-1), "an int >= 0"),
            (lambda doc: queue(doc)["runs"].append(dict(queue(doc)["runs"][0])), "a run name, once"),
            (lambda doc: queue(doc).update(log_tail=["x"] * 41), "<= 40 lines"),
            (lambda doc: queue(doc).update(serving="yes"), "a bool"),
            (lambda doc: queue(doc).pop("budget"), "keys must be"),
            (lambda doc: queue(doc).update(read_at=queue(doc)["read_at"] + 3600), "is not within the cycle"),
            (lambda doc: queue(doc)["log_tail"].append("[x] runner starting on h pid 4242 job 1"), "pid must be masked"),
        ]
        for change, needle in cases:
            self.assertProblem(self.problems(self.index(change)), needle)
        self.assertEqual(self.problems(self.index(lambda doc: queue(doc).update(budget=None))), [])  # never checked

    def test_sources_rules(self):
        # a launched run dir that is not shown, and a run listed that has neither launch nor build
        self.assertProblem(self.problems(self.index(lambda doc: doc["runs"].append(dict(
            self.run_entry(doc, test_collect.RUN2), run="official2-queued", latest=False,
            detail="run/hengyuf/official2-queued.json", detail_generated_at=1.0))), sources=True), "without a launch")
        q = self.h.queue
        state = json.loads((q / "state.json").read_text())
        try:
            changed = json.loads(json.dumps(state))
            changed["runs"][test_collect.RUN2]["status"] = "done"
            test_collect.write(q / "state.json", changed)
            self.assertProblem(self.problems(lambda data: None, sources=True), "!= state.json's")
            spec = json.loads((q / "queue.json").read_text())
            test_collect.write(q / "queue.json", dict(spec, runs=spec["runs"][:2]))
            self.assertProblem(self.problems(lambda data: None, sources=True), "differ from queue.json")
            test_collect.write(q / "queue.json", spec)
        finally:
            test_collect.write(q / "state.json", state)
        self.assertEqual(self.problems(lambda data: None, sources=True), [])


if __name__ == "__main__":
    unittest.main()
