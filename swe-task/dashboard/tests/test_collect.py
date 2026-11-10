"""Offline tests of the dashboard collector (collect.py) on synthetic fixture trees: no Slurm, hub, network or real
pipeline file is touched (Slurm, /metrics scrapes and the hub sample are fakes passed to run_cycle).

    cd /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/dashboard
    /usr/bin/python3 -I -B -m unittest discover -s tests -v

TreeTest       the generation tree of a synthetic official build: every decision rule (discard by build failure and
               by seeds.json, abandoned by an invalid parent / an unselected parent / a superseded probe,
               accepted_evolve, evolve, accepted, not_selected, end, pending), statuses, renamed files, rounds,
               live-probe progress with per-model partial counts and own_done, evolve focus patterns, teacher
               audits, the stage line, index.json accuracy / queue rank, smoke runs ignored
AuditTest      failure-match audits: attached to the node of their task (normalized, redacted, clipped), per-run
               counts, malformed / torn / misnamed / oversized / symlinked files ignored (and reported)
EligibilityTest  the audit's scope (v1.4, NODE.audit_eligible): built and probed with a graded sc14 rollout, not
               discarded (seeds.json / recomputed seed_check incl. uniform failure), not withdrawn (metrics.json
               "withdrawn" or runs/<run>/withdrawn/<task_id>/), not under an abandoned-* root; a withdrawn task is
               never accepted and still counts as accepted for the seed selection it took part in; the eligible
               counts of runs[].audit; one-line decision reasons
AcceptedInvalidSeedTest  v1.5, acceptance beats an invalid seed (official1-wasmi cand-c03-r1b2): an accepted task that
               seeds.json (or the recomputed seed_check) calls an invalid seed is accepted (round 0: not
               accepted_evolve; its children stay abandoned), keeps the reason as NODE.seed_invalid, is in the audit's
               scope and counted as accepted; the other nodes' notes; a withdrawn one stays not accepted (discard)
RedactTest     redact() / clip() and that no fixture secret reaches any output file
UsageTest      incremental offsets: kimi http.jsonl appends, a partial last line, a moved attempt dir (inode
               keyed), a rewritten codex events.jsonl; glm counts
FleetTest      vLLM /metrics parsing, tps from counter deltas across cycles (prompt tok/s incl. cache hits and its
               old alias), routing, busy workers, router stats and scope
HubTest        the hub sample interval guard (at most one per 10 min), burn rates over their measured windows (no
               rate from a single sample), the exhaustion ETA from the sample's time at the higher of the 1 h and
               6 h rates, history trimming
RobustTest     broken or missing sources become "errors", never exceptions; outputs are atomic
BatchTest      v1.6, a batch from another tree (BatchFixture, as Baihe's): its runs (the ones that built a task of its
               outputs/tasks.jsonl) built by the official code from their own trees (--rounds 1), never audited
               (failure_match null, audit_eligible false, audit not_audited; a planted audit file is never read), never
               polled, logs and usage never read, nothing outside outputs/ and those runs in the read roots; in_batch
               nodes, unfinished probes and the reasons of tasks an ended run never scored, round statuses, two runs on
               one target (batch_runs), the batch's own totals, mismatches reported, unchanged runs reused and their
               files rewritten only on a change (and never reused after a build with errors); a run dir that is a
               symlink out of synth/runs refused; a broken batch never stops the cycle; the official output
               byte-identical to a tree without the batch
OfficialRunsTest  v1.8, every official run has a document: a second official run of the same target (official2-test,
               launched later: the latest, data/task/<task>.json) and the older one (official1-test:
               data/run/hengyuf/official1-test.json, the same document, rewritten only on a change, removed once it
               is the latest again); runs[].latest / detail / detail_generated_at, tasks[].official_runs; audits per
               run dir (an official1 audit never reaches official2's node of the same task id; official2's eligible
               tasks not audited); a run dir with only analysis/ is no run; the Explorer's rows of both runs; every
               other output byte-identical to the v1.7 collector (collect.py.pre-v18) with one official run per task
QueueTest      v1.8, index.json queues: the official2 queue's runner state from exactly its four files (queue.json,
               state.json, heartbeat.json, runner.log): runs in queue order with their status / leg / winner /
               candidates, the runner's heartbeat and liveness, the budget's key NAME and number only, the redacted
               log tail; missing files read as a queue the runner never touched; nothing else of the directory read
               (a symlinked state.json refused); planted secrets never published; a broken queue.json never stops
               the cycle
ExplorerTest   v1.7, data/explorer.json: one row per built node of the task and run files and one per rollout of their
               probes, in order, every value from its node (the audit rule, reproduced patterns, kimi clean flags, live
               probes, withdrawn / abandoned tasks, the batch never audited, secrets masked), runs / targets / patterns;
               rewritten only on a change (also after a restart); a broken build never stops the cycle; every other
               output byte-identical to the v1.6 collector (collect.py.pre-explorer)
"""
import sys

sys.dont_write_bytecode = True

import contextlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import unittest
import unittest.mock

DASHBOARD = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DASHBOARD))
import collect  # noqa: E402

NOW = 1_800_000_000.0
RUN, TASK, REPO = "official1-test", "task-a", "owner__proj"
SECRET_KEY = "nvapi-" + "Q" * 40  # fake credentials planted in fixture files: they must never reach an output
SECRET_SK = "sk-proj-" + "Z" * 30
SECRET_TOKEN = "token=" + "ab12" * 10


def write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, (dict, list)):
        content = json.dumps(content, indent=2)
    path.write_text(content)
    return path


def append(path, text):
    with open(path, "a") as stream:
        stream.write(text)


def jsonl(path, rows, mode="w"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode) as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    return path


def burn_histories():
    """Two hub histories over 6 h (a sample every 30 min, budget 1000 $) whose 1 h and 6 h rates differ: (burst, calm).
    burst: 20 $/h for 5 h, then 100 $/h (1 h rate 100, 6 h rate 33.33); calm: 100 $/h for 1 h, then 10 $/h (1 h rate
    10, 6 h rate 25). Shared with test_contract.py: the validator must accept the collector's projection of both."""
    burst = [{"t": NOW + i * 1800, "spend": 100.0 + 10.0 * min(i, 10) + 50.0 * max(0, i - 10), "max_budget": 1000.0}
             for i in range(13)]
    calm = [{"t": NOW + i * 1800, "spend": 100.0 + 50.0 * min(i, 2) + 5.0 * max(0, i - 2), "max_budget": 1000.0}
            for i in range(13)]
    return burst, calm


def metrics(name, *, accept=False, student=0.6, teacher=1.0, similarity=None, valid=(5, 5), reason=None, **extra):
    out = {"name": name, "task_id": f"syn-{name}", "accept": accept, "student_rate": student,
           "teacher_clean_rate": teacher, "similarity": similarity, "valid": list(valid), "cheated": 0,
           "answer_retrieval": False, "target_student_rate": 0.2, "target_teacher_rate": 1.0, "ok": True,
           "reason": reason or ("similarity 7/10" if accept else "pass rates out of range"),
           "grading_failures": []}
    out.update(extra)
    if accept:
        out["similarity_detail"] = {"score": similarity or 7, "rationale": "close match"}
    return out


class Fixture:
    """A fake swe-task tree: targets, three-way CSV, one official build, a smoke run, the student fleet."""

    def __init__(self, root):
        self.root = Path(root)
        self.runs = self.root / "synth/runs"
        self.build = self.runs / RUN / "build" / TASK / REPO
        self.three_way = self.root / "three_way.csv"
        write(self.root / "synth/common.py", 'HUB_KEY = "NVAPI_KEY_NEW"  # name only\n')
        self.targets()
        self.official()
        self.smoke()
        self.fleet()

    # -- DeepSWE targets (sc14) and the three-way CSV (kimi, super VL)
    def targets(self):
        t = self.root / "data/targets/sc14-out131k-kimi"
        write(t / "task_ids.json", ["task-a", "task-b", "task-c", "task-d"])

        def records(statuses, f2ps, total):
            rows = [{"slot": f"sc14-r{i}", "model": "sc14", "status": status, "reward": float(status == "pass"),
                     "f2p": f"{f2p}/{total}" if status != "infra" else "0/0", "p2p": "4/4"}
                    for i, (status, f2p) in enumerate(zip(statuses, f2ps))]
            rows += [{"slot": f"kimi-r{i}", "model": "kimi", "status": "pass", "reward": 1.0,
                      "f2p": f"{total}/{total}", "p2p": "4/4"} for i in (1, 3, 5, 7, 9)]
            return rows

        write(t / "index.json", {
            "task-a": {"task_index": 0, "records": records(["pass", "fail", "fail", "fail", "fail"], [10, 9, 8, 9, 9], 10)},
            "task-b": {"task_index": 1, "records": records(["fail"] * 5, [19] * 5, 20)},
            "task-c": {"task_index": 2, "records": records(["pass", "pass", "pass", "fail", "infra"], [5, 5, 5, 1, 0], 5)}})
        write(t / "per_task.csv", "task_id,task_index,sc14_pass,sc14_fail,sc14_infra,kimi_pass,kimi_fail,kimi_grader,"
                                  "kimi_infra\ntask-a,0,1,4,0,5,0,0,0\ntask-b,1,0,5,0,5,0,0,0\ntask-c,2,3,1,1,5,0,0,0\n")
        header = ("task_id,f2p_total,kimi_correct,kimi_best_f2p,kimi_rollouts(f2p|p2p|score),super_correct,"
                  "super_best_f2p,super_rollouts(f2p|p2p|score),super_step0_correct,super_step0_best_f2p,"
                  "super_step0_rollouts(f2p|p2p|score)\n")
        rows = ["task-a,10,5,1.0," + " ".join(["10/10|4/4|1"] * 5) + ",2,1.0,10/10|4/4|1 10/10|4/4|1 9/10|4/4|0 "
                "0/0|0/0|U None/None|None/None|U,0,0,",
                "task-b,20,5,1.0," + " ".join(["20/20|4/4|1"] * 5) + ",0,0.9," + " ".join(["18/20|4/4|0"] * 5) + ",0,0,",
                "task-c,5,4,1.0," + " ".join(["5/5|4/4|1"] * 4) + " 4/5|4/4|0,1,1.0," + " ".join(["5/5|4/4|1"] * 5)
                + ",0,0,"]
        write(self.three_way, header + "\n".join(rows) + "\n")

    # -- one official build with every decision case (max_rounds 2, branches 6 from launch.jsonl)
    def official(self):
        run = self.runs / RUN
        jsonl(run / "launch.jsonl", [
            {"submitted_at": "2026-10-17T02:29:31-07:00", "job_id": "100", "run": RUN, "task": TASK, "extra": "",
             "log": str(self.runs / f"slurm/synth-build-{RUN}-100.out")},
            {"submitted_at": "2026-10-17T13:47:32-07:00", "job_id": "123", "run": RUN, "task": TASK,
             "partition": "cpu_interactive", "extra": "--rounds 2 --branches 6",
             "log": "/elsewhere/../../etc/passwd/synth-build-official1-test-123.out"}])  # only the name is used
        write(self.runs / f"slurm/synth-build-{RUN}-123.out",
              "".join(f"13:{i:02d}:00 step {i}\n" for i in range(60))
              + f"13:59:00 Authorization: Bearer abcdefghijklmnopqrstuvwxyz\n13:59:30 key {SECRET_KEY}\n")
        analysis = run / "analysis" / TASK
        write(analysis / "result.json", {
            "patterns": [{"name": "Scope creep", "category": "requirement_scope", "description": "broadens X",
                          "student_slots": ["sc14-r0", "sc14-r1"], "teacher_contrast": "keeps X",
                          "severity": "high"},
                         {"name": "Tests certify the bug", "category": "testing", "description": "asserts Y",
                          "student_slots": ["sc14-r2"], "teacher_contrast": "tests Y"}],  # evolve branch j: j % 2
            "transferable_signature": {"summary": "generalizes an exception"}, "excluded": []})
        write(analysis / "failure_analysis.md", f"# Failure analysis\n\nleaked {SECRET_SK} and {SECRET_TOKEN}\n")
        write(analysis / "inputs/stats.json", {"pass_stats": {"sc14": {"pass": 1, "fail": 4, "excluded": 0, "rate": 0.2},
                                                              "kimi": {"pass": 5, "fail": 0, "excluded": 0, "rate": 1.0}}})
        b = self.build
        for stage in ("env", "understanding"):
            write(b / stage / "stage.json", {"ok": True})
        candidates = [{"id": f"c0{i}", "title": f"Title {i}", "summary": f"Summary {i}", "files": ["a.py"],
                       "tests": ["t1", "t2", "t3"], "target_patterns": ["Scope creep"]} for i in range(1, 6)]
        write(b / "plan/stage.json", {"candidates": candidates})
        write(b / "plan/mask_plan.md", "# Mask plan\n" + "x" * 5000)

        # round 0 (mask)
        for name in ("cand-c01", "cand-c02", "cand-c03"):
            self.built(name)
        write(b / "cand-c01/stage.json.failed-materialize", {"ok": False, "problems": "old failure"})  # renamed
        self.node(0, "cand-c01", metrics("cand-c01", student=0.6, teacher=0.8, cheated=1), audit={
            "kimi-r1": {"categories": ["Memory narrated as lookup"], "cheated": True,
                        "evidence": ["[48] 'the upstream code looks like this' (no retrieval)"],
                        "notes": f"Recalled the upstream fix from memory (pasted {SECRET_KEY}) and narrated it as a "
                                 "lookup. " + "Otherwise rigorous. " * 40, "rubric_score": 9},
            "kimi-r2": {"categories": [], "cheated": False, "evidence": ["[1] read the spec", "[2] ran the tests"],
                        "rubric_score": 8.5}})                                    # no notes: the evidence summarizes
        self.node(0, "cand-c02", metrics("cand-c02", valid=(3, 4), grading_failures=["kimi-r1", "sc14-r0"],
                                         reason="inconclusive"))
        self.node(0, "cand-c03", metrics("cand-c03", accept=True, student=0.2, similarity=7))
        write(b / "cand-c04/stage.json", {"ok": False, "problems": "GateError: env broke\nsecond line"})
        for n in (0, 1):
            write(b / f"cand-c04/author-{n}/.agent/exit.json", {"exit_code": 0, "problems": []})
        write(b / "cand-c04/gates-0.json", {"f2p": ["t1"], "p2p": [], "problems": ["only 1 fail-to-pass tests"]})
        write(b / "round-0/seeds.json", {"valid": ["cand-c01", "cand-c03"],
                                         "invalid": {"cand-c02": "verifier produced no grade for 2 rollout(s): "
                                                                 "kimi-r1, sc14-r0"}})
        self.probe(0, "probe-r0", ["cand-c01", "cand-c02", "cand-c03"], 500)

        # round 1 (evolve)
        for name in ("cand-c01-r1b0", "cand-c01-r1b1", "cand-c01-r1b2", "cand-c02-r1b0", "cand-c03-r1b0",
                     "cand-c03-r1b1", "cand-c03-r1b2"):
            self.built(name, evolve=True)
        self.node(1, "cand-c01-r1b0", metrics("cand-c01-r1b0", student=0.2))   # rank (0, 0): the c01 seed
        self.node(1, "cand-c01-r1b1", metrics("cand-c01-r1b1", student=0.8))   # rank (0, -0.6): not selected
        self.node(1, "cand-c03-r1b0", metrics("cand-c03-r1b0", accept=True, student=0.0, similarity=8))
        self.node(1, "cand-c03-r1b2", metrics("cand-c03-r1b2", student=0.6))
        self.probe(1, "probe-old", ["cand-c01-r1b0", "cand-c01-r1b1", "cand-c01-r1b2", "cand-c02-r1b0"], 1000)
        self.probe(1, "probe-mid", ["cand-c01-r1b0", "cand-c01-r1b1", "cand-c03-r1b0", "cand-c03-r1b2"], 1500)
        live = self.probe(1, "probe-new", ["cand-c03-r1b1"], 2000)
        self.slot(live, "sc14", "0-0", [True], reward={"reward": 1, "f2p_passed": 3, "f2p_total": 3,
                                                       "p2p_passed": 4, "p2p_total": 4})
        self.slot(live, "sc14", "0-1", [True], reward={"reward": 0, "f2p_passed": 2, "f2p_total": 3,
                                                       "p2p_passed": 4, "p2p_total": 4})
        self.slot(live, "sc14", "0-2", [False])                       # still running
        self.slot(live, "kimi", "0-0", [True, True, True])            # 3 attempts, no grade: infra
        self.slot(live, "kimi", "0-1", [True], reward={"reward": 1, "f2p_passed": 3, "f2p_total": 3,
                                                       "p2p_passed": 4, "p2p_total": 4})

        # round 2 (= max_rounds)
        for name in ("cand-c01-r1b0-r2b0", "cand-c01-r1b0-r2b1", "cand-c01-r1b0-r2b4", "cand-c01-r1b0-r2b5",
                     "cand-c01-r1b1-r2b0"):
            self.built(name, evolve=True)
        self.node(2, "cand-c01-r1b0-r2b0", metrics("cand-c01-r1b0-r2b0", student=0.4))
        self.node(2, "cand-c01-r1b0-r2b1", metrics("cand-c01-r1b0-r2b1", accept=True, student=0.0, similarity=6))
        self.node(2, "cand-c01-r1b0-r2b4", metrics("cand-c01-r1b0-r2b4", valid=(2, 5)))
        m5 = metrics("cand-c01-r1b0-r2b5")
        del m5["grading_failures"]  # written before score() recorded them: read back from rollouts/<slot>.md
        self.node(2, "cand-c01-r1b0-r2b5", m5, infra={"sc14-r0": "test.patch failed to apply"})
        write(b / "cand-c01-r1b0-r2b2/evolve/inputs/metrics.json", {})            # evolve plan in progress
        write(b / "cand-c01-r1b0-r2b3/evolve/stage.json", {"title": "T r2b3", "summary": "S"})
        write(b / "cand-c01-r1b0-r2b3/built.json", {"attempt": 0, "gate": {"f2p": ["a", "b"], "p2p": ["c"],
                                                                           "problems": []}})
        write(b / "cand-c01-r1b0-r2b6/evolve/stage.json", {"title": "T r2b6", "summary": "S"})
        write(b / "cand-c01-r1b0-r2b6/stage.json.budget-422", {"ok": False, "problems": "AgentError: 422"})
        write(b / "cand-c01-r1b0-r2b6/author-0.budget-422/.agent/exit.json", {"exit_code": 1, "problems": ["x"]})
        self.probe(2, "probe-r2", ["cand-c01-r1b0-r2b0", "cand-c01-r1b0-r2b1", "cand-c01-r1b0-r2b4",
                                   "cand-c01-r1b0-r2b5"], 3000)
        write(b / "cand-weird.budget-422/stage.json", {"ok": True})  # renamed candidate dir: not a node

        # API usage: kimi http.jsonl (one attempt moved aside after a 422), codex events, router journal
        self.kimi = self.build / "round-1/probe-mid/models/kimi/attempts/0-0/attempt-0/http.jsonl"
        jsonl(self.kimi, [self.usage(0.5, 1000, 800, 20), {"event": "request", "payload": {"x": "y" * 5000}},
                          self.usage(0.25, 2000, 1000, 40)])
        jsonl(self.build / "round-1/probe-old/budget-422-moved/kimi-attempts/1-0/attempt-0/http.jsonl",
              [self.usage(1.0, 10, 0, 1), {"event": "response", "status": 422, "body": "budget_exceeded"}])
        self.codex = self.build / "env/.agent/events.jsonl"
        jsonl(self.codex, [{"type": "thread.started", "thread_id": "t-1"}, {"type": "turn.started"},
                           {"type": "item.completed", "item": {"text": '"type":"turn.completed"'}},
                           self.turn(100, 80, 10, 5)])
        jsonl(self.build / "cand-c01-r1b0-r2b6/author-0.budget-422/.agent/events.jsonl",
              [{"type": "thread.started", "thread_id": "t-2"}, {"type": "turn.failed", "error": {"message": "422"}}])
        # failure-match audits (auditor agents): two of the three accepted tasks (cand-c01-r1b0-r2b1 is not audited)
        # and a task that was not accepted (attached to its node, never counted)
        self.audit("syn-cand-c03", verdict="strong", round=0,
                   summary=f"sc14 broadens X in 4 of 5 rollouts, as the target did (it pasted {SECRET_KEY}).",
                   patterns=[{"name": "Scope creep", "reproduced": "yes", "slots": ["sc14-r1", "sc14-r2"],
                              "evidence": "sc14-r1 rewrote every caller of X: " + "x" * 700},
                             {"name": "Tests certify the bug", "reproduced": "No", "slots": [], "evidence": ""}],
                   student_rollouts=[{"slot": "sc14-r0", "status": "pass", "failing_tests": [], "cause": "",
                                      "pattern": None},
                                     {"slot": "sc14-r1", "status": "FAIL", "cause": "broadened X to every caller",
                                      "failing_tests": [f"tests/test_x.py::test_{i}" for i in range(12)],
                                      "pattern": "Scope creep"}],
                   other_causes="sc14-r4 ran out of steps before committing.",
                   similarity_judge={"score": 7, "agrees": True, "comment": "a fair 7"}, notes="not in the format")
        self.audit("syn-cand-c03-r1b0", verdict=" Partial ", summary="  Loose match: only sc14-r2.  ")
        self.audit("syn-cand-c01", verdict="weak", summary="One rollout only.")
        self.router = self.build / "round-0/probe-r0/student-router-requests.jsonl"
        rows = [self.request(NOW - 30 - i, worker="1001", duration=float(i + 1)) for i in range(10)]
        rows += [self.request(NOW - 100, worker="1001", status=500, error="HTTP 500"),
                 self.request(NOW - 100, worker="1002", status="client_disconnected"),
                 self.request(NOW - 120, worker="1002", clamps=[{"applied": True}]),
                 self.request(NOW - 2000, worker="1001"),           # last hour only
                 self.request(NOW - 7200, worker="1001")]           # older than an hour: dropped
        jsonl(self.router, rows)

    def built(self, name, evolve=False):
        b = self.build / name
        write(b / "stage.json", {"ok": True, "name": name, "task_id": f"syn-{name}", "f2p": 3, "p2p": 4,
                                 "kind": "evolve" if evolve else "mask", "title": f"Stage title {name}"})
        write(b / "built.json", {"attempt": 0, "gate": {"f2p": ["t1", "t2", "t3"], "p2p": ["p1", "p2", "p3", "p4"],
                                                        "problems": []}})
        write(b / "author-0/.agent/exit.json", {"exit_code": 0, "problems": []})
        write(b / "instruction/instruction.md", f"Implement {name}.\n")
        write(b / "instruction/rubric.md", "# rubric\n")
        if evolve:
            write(b / "evolve/stage.json", {"title": f"Evolved {name}", "summary": f"Plan of {name}"})
            write(b / "evolve/evolve_plan.md", f"# evolve plan {name}\n")

    def node(self, k, name, data, infra=None, audit=None):
        out = self.build / f"round-{k}" / name
        write(out / "metrics.json", data)
        rows = ["| slot | status | reward | F2P | P2P | tool calls | patch chars | agent error |",
                "|---|---|---|---|---|---|---|---|"]
        student_pass = round((data["student_rate"] or 0) * 5)
        verdicts = {}  # task_builder score(): one judge_pass verdict per passing teacher rollout
        for model in ("kimi", "sc14"):
            for i in range(5):
                slot = f"{model}-r{i}"
                if infra and slot in infra:
                    status, f2p, error = "infra", "0/0", infra[slot]
                else:
                    passed = i < (student_pass if model == "sc14" else 5)
                    status, f2p, error = ("pass" if passed else "fail"), ("3/3" if passed else "2/3"), None
                rows.append(f"| {slot} | {status} | {float(status == 'pass')} | {f2p} | 4/4 | 12 | 345 |  |")
                write(out / f"rollouts/{slot}.md", f"# {slot} — {status}\nreward 0.0; F2P {f2p}; P2P 4/4; "
                                                   f"agent_error None; verifier_error {error}\nfailing tests: none\n")
                # rollouts.write_rollouts() writes the patch next to every transcript (v1.7: the Explorer links both)
                write(out / f"rollouts/{slot}.patch", f"diff --git a/{slot}.py b/{slot}.py\n")
                if model == "kimi" and status == "pass":
                    verdicts[slot] = {"categories": [], "cheated": False, "evidence": [f"[3] {slot} read the spec"],
                                      "notes": f"No cheating: {slot} followed the spec.", "rubric_score": 9.5}
        write(out / "rollouts/index.md", "\n".join(rows) + "\n")
        write(out / "teacher_audit.json", {**verdicts, **(audit or {})})

    def audit(self, task_id, **fields):
        """An auditor agent's failure-match verdict on one task: runs/<run>/audit/failure_match/<task_id>.json."""
        data = {"schema": 1, "task_id": task_id, "run": RUN, "cand": task_id[len("syn-"):], "round": 1,
                "audited_at": "2026-10-18T20:00:00-07:00", "auditor": "claude-subagent",
                "target_patterns": [{"name": "Scope creep", "category": "requirement_scope"},
                                    {"name": "Tests certify the bug", "category": "testing"}]}
        data.update(fields)
        return write(self.runs / RUN / "audit/failure_match" / f"{task_id}.json", data)

    def probe(self, k, key, names, created):
        directory = self.build / f"round-{k}" / key
        write(directory / "manifest.json", {
            "version": 1, "created_at": created, "shards": 1,
            "settings": {"models": {"kimi": {}, "sc14": {"local": True}}, "repeats": 5, "max_attempts": 3},
            "tasks": [{"task_id": f"syn-{name}", "_ng_task_index": i, "responses_create_params": {"input": "x"}}
                      for i, name in enumerate(names)]})
        return directory

    def slot(self, probe, model, slot, terminals, reward=None):
        for n, terminal in enumerate(terminals):
            attempt = probe / "models" / model / "attempts" / slot / f"attempt-{n}"
            attempt.mkdir(parents=True, exist_ok=True)
            if terminal:
                write(attempt / "terminal.json", {"evaluation_completed": reward is not None})
            if reward is not None and n == len(terminals) - 1:
                write(attempt / "grading/attempt-0/reward.json", reward)

    @staticmethod
    def usage(cost, prompt, cached, completion):
        return {"event": "upstream_stream_completed", "attempt_dir": "/x", "url": "https://hub/v1/chat/completions",
                "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "cost": cost,
                          "prompt_tokens_details": {"cached_tokens": cached},
                          "completion_tokens_details": {"reasoning_tokens": completion // 2}}}

    @staticmethod
    def turn(prompt, cached, completion, reasoning):
        return {"type": "turn.completed", "usage": {"input_tokens": prompt, "cached_input_tokens": cached,
                                                    "output_tokens": completion, "reasoning_output_tokens": reasoning}}

    @staticmethod
    def request(end, worker, duration=2.0, status=200, error=None, clamps=()):
        return {"ts": end - duration, "path": "/v1/chat/completions", "worker": worker, "status": status,
                "duration_s": duration, "prompt_tokens": 100, "completion_tokens": 10, "clamps": list(clamps),
                "error": error}

    # -- a smoke run (never recorded)
    def smoke(self):
        smoke = self.runs / "smoke1"
        jsonl(smoke / "launch.jsonl", [{"job_id": "77", "run": "smoke1", "task": "task-c"}])
        write(smoke / "build/task-c/x__y/cand-c01/stage.json", {"ok": True, "task_id": "syn-smoke"})

    # -- the student fleet
    def fleet(self):
        student = self.root / "student"
        write(student / "state/status.json", {
            "ts": NOW - 30, "config": {"target": 2, "horizon_effective_min": 150, "horizon_min": 45, "stagger_min": 40,
                                       "max_jobs": 24, "time": "04:00:00", "coverage_partitions": ["batch"]},
            "counts": {"ready": 1, "pending": 1, "coverage": 2, "routable": 1, "total": 2},
            "breaker": {"open": False, "failures": []},
            "jobs": [{"job_id": "1001", "slot": "01", "state": "RUNNING", "partition": "batch", "node": "pool0-1",
                      "start": NOW - 1000, "end": NOW + 3600, "left_s": 3630, "reason": "", "registry": "ready",
                      "routing": "ready", "base_url": "http://10.0.0.1:8000/v1"},
                     {"job_id": "1002", "slot": "02", "state": "PENDING", "partition": "batch", "node": "",
                      "start": NOW + 600, "end": None, "left_s": 14400, "reason": "Priority", "registry": None},
                     {"job_id": "1003", "slot": "03", "state": "RUNNING", "partition": "batch", "node": "pool0-3",
                      "start": NOW - 500, "end": NOW + 9000, "reason": "", "base_url": "http://10.0.0.3:8000/v1"}]})
        write(student / "state/daemon.json", {"job_id": "900", "host": "cpu1-1", "heartbeat_at": NOW - 20})
        write(student / "registry/1001.json", {"job_id": "1001", "status": "ready", "heartbeat_at": NOW - 10,
                                               "end_epoch": NOW + 3600, "drain_at": NOW + 3000, "ready_at": NOW - 900,
                                               "base_url": "http://10.0.0.1:8000/v1"})
        write(student / "registry/1003.json", {"job_id": "1003", "status": "ready", "heartbeat_at": NOW - 500,
                                               "end_epoch": NOW + 9000, "base_url": "http://10.0.0.3:8000/v1"})


SEED_INVALID = "verifier produced no grade for 1 rollout(s): kimi-r3"  # task_builder seed_check()'s words
C02_INVALID = "verifier produced no grade for 2 rollout(s): kimi-r1, sc14-r0"  # the fixture's round-0/seeds.json


def plant_accepted_invalid_seeds(fx):
    """official1-wasmi, 2026-10-21: task_builder accepted round-1 cand-c03-r1b2 on its score (accepted/, metrics.json
    accept), then seed_check() listed it invalid in round-1/seeds.json (kimi-r3 was never graded), which only keeps a
    task from seeding evolution. The same twice in the synthetic build: round-1 cand-c03-r1b0 (accepted, kimi-r3
    ungraded) listed invalid in a round-1 seeds.json, and round-0 cand-c02 (invalid in round-0/seeds.json) accepted.
    Shared with test_contract.py."""
    fx.node(1, "cand-c03-r1b0", metrics("cand-c03-r1b0", accept=True, student=0.0, similarity=8, valid=(5, 4),
                                        reason="similarity 8/10", grading_failures=["kimi-r3"]),
            infra={"kimi-r3": "no reward.json"})
    write(fx.build / "round-1/seeds.json", {"valid": ["cand-c01-r1b0", "cand-c01-r1b1", "cand-c03-r1b2"],
                                            "invalid": {"cand-c03-r1b0": SEED_INVALID}})
    path = fx.build / "round-0/cand-c02/metrics.json"
    write(path, dict(json.loads(path.read_text()), accept=True, similarity=7, reason="similarity 7/10"))


RUN2, REPO2 = "official2-test", "other__lib"  # v1.8: a second official run of TASK (plant_official2)
QUEUED = "official2-queued"  # v1.8: a queued run: only its copied failure analysis, no launch, no build


def plant_official2(fx):
    """v1.8 (hengyuf's official2 queue, 2026-10-22: official2-etree alongside official1-etree, ...): a second official
    run of task-a, launched after official1-test, so it is the latest (data/task/task-a.json): --rounds 1, round 0 so
    far; cand-c01 accepted and eligible (no audit in its own run dir: not audited, although official1-test has an audit
    of a task with the same id), cand-c02 not accepted, cand-c03 built and still probing. Its failure analysis is
    official1-test's (the queue copies it, with its old mtimes). It was launched as the queue runner launches (the babysit
    helper's two candidate jobs 299 / 300, then the line naming the winner, 300). And a queued run with only analysis/
    (official2-queued, task-b): no run. Shared with test_contract.py."""
    run = fx.runs / RUN2
    leg = {"submitted_at": "2026-10-22T22:05:00-0700", "job_id": "299", "run": RUN2, "task": TASK, "time": "24:00:00",
           "partition": "cpu_datamover", "extra": "--repo other/lib --rounds 1",
           "log": str(fx.runs / f"slurm/synth-build-{RUN2}-299.out"),
           "babysit": {"leg": 1, "candidates": ["299", "300"], "accounts": ["a", "b"], "resolved": False}}
    jsonl(run / "launch.jsonl", [leg, dict(leg, job_id="300", log=str(fx.runs / f"slurm/synth-build-{RUN2}-300.out"),
                                           babysit=dict(leg["babysit"], resolved=True, winner="300"))])
    write(fx.runs / f"slurm/synth-build-{RUN2}-300.out", "22:05:00 official2 build started\n")
    shutil.copytree(fx.runs / RUN / "analysis", run / "analysis")
    saved, fx.build = fx.build, run / "build" / TASK / REPO2
    try:
        for stage in ("env", "understanding"):
            write(fx.build / stage / "stage.json", {"ok": True})
        write(fx.build / "plan/stage.json", {"candidates": [
            {"id": f"c0{i}", "title": f"Second title {i}", "summary": f"Second summary {i}", "files": ["b.py"],
             "tests": ["t1", "t2", "t3"], "target_patterns": ["Scope creep"]} for i in (1, 2, 3)]})
        for name in ("cand-c01", "cand-c02", "cand-c03"):
            fx.built(name)
        fx.node(0, "cand-c01", metrics("cand-c01", accept=True, student=0.2, similarity=8))
        fx.node(0, "cand-c02", metrics("cand-c02", student=0.8))
        fx.probe(0, "probe-a", ["cand-c01", "cand-c02"], 600)
        live = fx.probe(0, "probe-b", ["cand-c03"], 700)
        fx.slot(live, "sc14", "0-0", [False])  # still running
    finally:
        fx.build = saved
    write(run / f"accepted/syn-cand-c01/task.toml", "x")  # task_builder accept() of its accepted task
    write(fx.runs / QUEUED / "analysis/task-b/result.json", {"patterns": [{"name": "Queued pattern"}], "excluded": []})


def plant_queue(fx, state=True, heartbeat=True, root=None):
    """v1.8: the official2 queue's runner files (S/synth/queue2, run_queue.py's formats): queue.json with three runs
    (official2-test on task-a, launched; official2-queued on task-b and official2-later on task-c, not yet), state.json
    (official2-test running, leg 1, winner 300 of 300/301; official2-queued queued; nothing for official2-later yet; the
    budget with an error text that must never be published; serving on), heartbeat.json (30 s old), runner.log (51
    lines: the runner's start line with its pid, passes, an error, a key switch naming two key NAMES, another runner's
    pid, a launch, a line with a key). Planted and never to be read: runner.lock, notes.json. Shared with
    test_contract.py."""
    q = Path(root) if root else fx.root / "synth/queue2"
    write(q / "queue.json", {
        "created": "2026-10-22", "requested_by": f"hengyuf, 2026-10-22, request-marker-q2 ({SECRET_KEY})",
        "settings": {"max_active": 5, "max_launches_per_run": 6, "key_order": ["NVAPI_KEY_OFFICIAL_1",
                                                                               "NVAPI_KEY_OFFICIAL_2"],
                     "switch_key_below_usd": 2000, "stop_launching_below_usd": 2000, "fleet_target": 10},
        "runs": [{"run": RUN2, "task": TASK, "repo": "other/lib", "batch": "hengyuf",
                  "analysis_from": f"S/synth/runs/{RUN}"},
                 {"run": QUEUED, "task": "task-b", "repo": "queued/b", "batch": "baihe", "analysis_from": "B/x"},
                 {"run": "official2-later", "task": "task-c", "repo": "later/c", "batch": "baihe",
                  "analysis_from": "B/y"}],
        "settings_notes": {"keys": f"notes-marker-q2, see {SECRET_SK}"}})
    if state:
        write(q / "state.json", {
            "budget_checked": NOW - 100, "serving": True,
            "budget": {"at": "2026-10-22T22:10:00-0700", "key": "NVAPI_KEY_OFFICIAL_1", "remaining": 2345.678,
                       "error": f"budget-error-marker-q2: {SECRET_KEY}"},
            "key_switches": [{"at": "2026-10-22T21:00:00-0700", "from": "NVAPI_KEY_SWITCH_MARKER", "to": "NVAPI_KEY_OFFICIAL_1"}],
            "runs": {RUN2: {"task": TASK, "repo": "other/lib", "status": "running", "leg": 1,
                            "candidates": ["300", "301"], "winner": "300"},
                     QUEUED: {"task": "task-b", "repo": "queued/b", "status": "queued", "leg": 0, "candidates": [],
                              "winner": None, "launch_errors": 1}},
            "updated_at": "2026-10-22T22:12:00-0700"})
    if heartbeat:
        write(q / "heartbeat.json", {"at": NOW - 30, "at_iso": "2026-10-22T22:12:30-0700", "host": "cpu-node-7",
                                     "pid": 987654321, "slurm_job_id": "456", "note": "running", "extra": SECRET_TOKEN})
    write(q / "runner.log", "[2026-10-22T22:00:00-0700] official2 queue runner starting (live) on cpu-node-7 pid "
                            "987654321 job 456: 3 runs, max 5 active\n"
          + "".join(f"[2026-10-22T22:{i:02d}:00-0700] pass {i}\n" for i in range(1, 47))
          + "[2026-10-22T22:47:00-0700] pass failed: RuntimeError: squeue failed (1): slurm_load_jobs error: Socket timed "
            "out on send/recv operation\n"
          + "[2026-10-22T22:48:00-0700] budget: switching the active hub key NVAPI_KEY_OFFICIAL_1 -> NVAPI_KEY_OFFICIAL_2 "
            "(remaining 2100.5)\n"
          + "[2026-10-22T22:49:00-0700] another runner beat 30 s ago on cpu-node-8 (pid 1234, job 789): not starting\n"
          + "[2026-10-22T22:50:00-0700] official2-test: launched leg 1 on cpu_datamover: candidates 299,300\n"
          + f"[2026-10-22T22:51:00-0700] budget: NVAPI_KEY_OFFICIAL_1 remaining 2345.678 ({SECRET_KEY})\n")
    write(q / "runner.lock", "")
    write(q / "notes.json", {"never": "notes-json-marker-q2", "key": SECRET_KEY})
    return q


BATCH = "testbatch"  # the synthetic batch's source name (CONTRACT "v1.6"; Baihe's is "baiheh")


class BatchFixture:
    """A fake batch tree (CONTRACT "v1.6"), as Baihe's fork of the pipeline writes it: outputs/tasks.jsonl and
    outputs/<task_id>/provenance.json, and the runs that built those tasks (--rounds 1, every task probed alone): two
    finished runs on target task-b (as pilot7 / pilot10 on tengo-callable-instance-isolation) and a stopped one on task-c.
    Planted and never to be read: an audit file, the runs' Slurm logs, kimi usage logs, a run outside the batch, secrets.

      pilotA-task-b (finished): round 0 cand-c01 accepted (batch), cand-c02 not accepted, cand-c03 never scored (the
        round's one probe stays unfinished, as pilot2 cand-c02); round 1 cand-c01-r1b0 accepted (batch), cand-c01-r1b1
        not accepted (end), cand-c02-r1b0 accepted (batch), cand-c02-r1b1 invalid in seeds.json (discard)
      pilotB-task-b (finished, launched later): cand-c01 accepted (batch); cand-c01-r1b0 not accepted (end)
      q1-task-c (stopped: no summary.json): cand-c01 accepted (batch); cand-c01-r1b0 accepted (batch); cand-c01-r1b1
        built, its probe unfinished; cand-c01-r1b2 built, never probed"""

    RUNS = {"pilotA-task-b": ("task-b", "o__a", "2026-10-19T04:06:43-07:00", "201", True),
            "pilotB-task-b": ("task-b", "o__b", "2026-10-19T18:28:00-07:00", "202", True),
            "q1-task-c": ("task-c", "o__c", "2026-10-21T14:41:48-07:00", "203", False)}
    PATTERNS = ["Identity lost", "Tests miss composition"]

    def __init__(self, root):
        self.root = Path(root)
        self.outputs = self.root / "outputs"
        self.runs = self.root / "synth/runs"
        self.batch = collect.Batch(BATCH, "Testy", "Testy · fork", self.root, rounds=1, branches=5)
        self.tasks = []  # (run, target, repo, cand, task_id), in outputs/tasks.jsonl order
        a = self.run("pilotA-task-b")
        for name in ("cand-c01", "cand-c02", "cand-c03"):
            self.built(a, name)
        self.scored(a, 0, "cand-c01", accept=True, similarity=7)
        self.scored(a, 0, "cand-c02", student=0.6)
        write(a["build"] / "round-0/cand-c03/rollouts/index.md", "| slot | status |\n")  # its score() failed
        write(a["build"] / "round-0/seeds.json", {"valid": ["cand-c01", "cand-c02"], "invalid": {}})
        self.probe(a, 0, "probe-r0", ["cand-c01", "cand-c02", "cand-c03"], 100)  # one probe, an older fork
        for name in ("cand-c01-r1b0", "cand-c01-r1b1", "cand-c02-r1b0", "cand-c02-r1b1"):
            self.built(a, name, evolve=True)
            self.probe(a, 1, f"probe-{name[5:].replace('-', '')}", [name], 200)  # every child probed alone
        self.scored(a, 1, "cand-c01-r1b0", accept=True, student=0.0, similarity=8)
        self.scored(a, 1, "cand-c01-r1b1", student=0.8)
        self.scored(a, 1, "cand-c02-r1b0", accept=True, student=0.2, similarity=6)
        self.scored(a, 1, "cand-c02-r1b1", student=0.4, valid=(2, 5))
        write(a["build"] / "round-1/seeds.json", {"valid": ["cand-c01-r1b0", "cand-c01-r1b1", "cand-c02-r1b0"],
                                                  "invalid": {"cand-c02-r1b1": "fewer than 3 valid rollouts per "
                                                                               "model (valid [2, 5])"}})
        b = self.run("pilotB-task-b")
        self.built(b, "cand-c01")
        self.built(b, "cand-c01-r1b0", evolve=True)
        self.scored(b, 0, "cand-c01", accept=True, similarity=9)
        self.scored(b, 1, "cand-c01-r1b0", student=0.6)
        self.probe(b, 0, "probe-c01", ["cand-c01"], 100)
        self.probe(b, 1, "probe-c01r1b0", ["cand-c01-r1b0"], 200)
        c = self.run("q1-task-c")
        for name, evolve in (("cand-c01", False), ("cand-c01-r1b0", True), ("cand-c01-r1b1", True),
                             ("cand-c01-r1b2", True)):
            self.built(c, name, evolve=evolve)
        self.scored(c, 0, "cand-c01", accept=True, similarity=7)
        self.scored(c, 1, "cand-c01-r1b0", accept=True, student=0.0, similarity=8)
        self.probe(c, 0, "probe-c01", ["cand-c01"], 100)
        self.probe(c, 1, "probe-c01r1b0", ["cand-c01-r1b0"], 200)
        self.probe(c, 1, "probe-c01r1b1", ["cand-c01-r1b1"], 300)  # the job was cancelled while it ran
        write(c["build"] / "round-0/seeds.json", {"valid": ["cand-c01"], "invalid": {}})
        write(c["build"] / "round-1/seeds.json", {"valid": ["cand-c01-r1b0"], "invalid": {}})
        self.write_outputs()
        # never to be read: an audit, the Slurm logs, kimi usage, a run outside the batch, the tree's secrets
        write(self.runs / f"pilotA-task-b/audit/failure_match/{self.tid('pilotA-task-b', 'cand-c01')}.json",
              {"schema": 1, "verdict": "strong", "summary": "planted: never read"})
        write(self.runs / "slurm/synth-build-pilotA-task-b-201.out", f"log line {SECRET_KEY}\n")
        jsonl(a["build"] / "round-1/probe-c01r1b0/models/kimi/attempts/0-0/attempt-0/http.jsonl",
              [Fixture.usage(5.0, 1000, 0, 10)])
        jsonl(self.runs / "smoke9/launch.jsonl", [{"job_id": "999", "run": "smoke9", "task": "task-a"}])
        write(self.runs / "smoke9/build/task-a/x__y/cand-c01/stage.json", {"ok": True, "task_id": "syn-smoke9"})
        write(self.root / "secrets/hub.env", "NVAPI_KEY=never-read\n")

    @staticmethod
    def tid(run, name):
        return f"syn-{run.split('-')[0]}-{name}"

    def run(self, run):
        target, repo, submitted, job, finished = self.RUNS[run]
        d = self.runs / run
        jsonl(d / "launch.jsonl", [{"submitted_at": submitted, "job_id": job, "run": run, "task": target, "extra": "",
                                    "partition": "cpu_dataprocessing", "time": "24:00:00",
                                    "log": str(self.runs / f"slurm/synth-build-{run}-{job}.out")}])
        write(d / "analysis" / target / "result.json", {
            "patterns": [{"name": name, "category": "c", "description": f"{name} described", "student_slots": ["sc14-r0"],
                          "teacher_contrast": "does it right"} for name in self.PATTERNS],
            "transferable_signature": "identity vs equality", "excluded": []})
        write(d / "analysis" / target / "failure_analysis.md", f"# Failure analysis of {target}\n")
        write(d / "analysis" / target / "inputs/stats.json", {"pass_stats": {"sc14": {"pass": 0, "fail": 5, "rate": 0.0},
                                                                             "kimi": {"pass": 4, "fail": 1, "rate": 0.8}}})
        build = d / "build" / target / repo
        for stage in ("env", "understanding"):
            write(build / stage / "stage.json", {"ok": True})
        write(build / "plan/stage.json", {"candidates": [{"id": f"c0{i}", "title": f"Plan {i}", "summary": f"S{i}",
                                                          "files": ["a.go"], "tests": ["t1"]} for i in (1, 2, 3)]})
        write(build / "plan/mask_plan.md", "# Mask plan\n")
        if finished:
            write(build / "summary.json", {"target": target, "repo": repo, "built": [], "rejected": [], "accepted": []})
        return {"run": run, "dir": d, "build": build, "target": target, "repo": repo}

    def built(self, r, name, evolve=False):
        b = r["build"] / name
        write(b / "stage.json", {"ok": True, "name": name, "task_id": self.tid(r["run"], name), "f2p": 3, "p2p": 4,
                                 "kind": "evolve" if evolve else "mask", "title": f"Stage title {name}"})
        write(b / "built.json", {"attempt": 0, "gate": {"f2p": ["t1", "t2", "t3"], "p2p": ["p1"], "problems": []}})
        write(b / "author-0/.agent/exit.json", {"exit_code": 0, "problems": []})
        write(b / "instruction/instruction.md", f"Implement {name} of {r['run']}.\n")
        write(b / "instruction/rubric.md", "# rubric\n")
        if evolve:
            branch = int(name.rsplit("b", 1)[1])
            write(b / "evolve/stage.json", {"title": f"Evolved {name}", "summary": f"Plan of {name}"})
            write(b / "evolve/evolve_plan.md", f"# evolve plan {name}\n")
            write(b / "evolve/.agent/prompt.md",
                  f'Focus pattern for this branch: "{self.PATTERNS[branch % len(self.PATTERNS)]}".\n')

    def scored(self, r, k, name, accept=False, student=0.0, similarity=None, valid=(5, 5)):
        task_id = self.tid(r["run"], name)
        data = metrics(name, accept=accept, student=student, similarity=similarity, valid=valid, task_id=task_id,
                       kind="evolve" if k else "mask", lineage=name.split("-r")[0],
                       task_dir=str(r["build"] / name / "task" / task_id))
        Fixture.node(_At(r["build"]), k, name, data)
        if accept:
            self.tasks.append((r["run"], r["target"], r["repo"], name, task_id))
            accepted = r["dir"] / "accepted"
            write(accepted / task_id / "task.toml", "x")
            write(accepted / f"{task_id}.workspace/manifest.json", {})
            jsonl(accepted / "tasks.jsonl", [{"task_id": task_id}], mode="a")
            write(accepted / task_id / "provenance.json", data)

    def probe(self, r, k, key, names, created):
        write(r["build"] / f"round-{k}" / key / "manifest.json", {
            "version": 1, "created_at": created,
            "settings": {"models": {"kimi": {}, "sc14": {"local": True}}, "repeats": 5, "max_attempts": 3},
            "tasks": [{"task_id": self.tid(r["run"], name), "_ng_task_index": i} for i, name in enumerate(names)]})

    def write_outputs(self):
        rows = []
        for run, target, repo, cand, task_id in self.tasks:
            provenance = json.loads((self.runs / run / "accepted" / task_id / "provenance.json").read_text())
            write(self.outputs / task_id / "provenance.json", provenance)
            write(self.outputs / task_id / "instruction.md", f"Implement {cand}.\n")
            write(self.outputs / f"{task_id}.workspace/manifest.json", {})
            rows.append({"task_id": task_id, "subset": "deepswe-synth", "split": "test",
                         "responses_create_params": {"input": [{"role": "user", "content": f"Implement {cand}."}]}})
        jsonl(self.outputs / "tasks.jsonl", rows)

    def batch_task_ids(self, run=None):
        return [task_id for name, _, _, _, task_id in self.tasks if run in (None, name)]


class _At:
    """Fixture.node()'s view of a build root (it writes under self.build)."""

    def __init__(self, build):
        self.build = build


METRICS = """# HELP vllm:num_requests_running Number of requests running.
vllm:num_requests_running{{engine="0",model_name="policy"}} 2.0
vllm:num_requests_running{{engine="1",model_name="policy"}} 1.0
vllm:num_requests_waiting{{engine="0",model_name="policy"}} 1.0
vllm:num_requests_waiting_by_reason{{engine="0",model_name="policy",reason="capacity"}} 9.0
vllm:kv_cache_usage_perc{{engine="0",model_name="policy"}} 0.1
vllm:kv_cache_usage_perc{{engine="1",model_name="policy"}} 0.3
vllm:prompt_tokens_total{{engine="0",model_name="policy"}} {prompt}
vllm:generation_tokens_total{{engine="0",model_name="policy"}} {gen}
vllm:request_success_total{{engine="0",finished_reason="stop",model_name="policy"}} 5.0
"""


class Harness:
    """Runs collect cycles against a Fixture with fake Slurm, scrapes and hub."""

    def __init__(self, cleanup, batch=False, official2=False, queue=False):
        self.tmp = Path(tempfile.mkdtemp(prefix="dash-test-"))
        cleanup(shutil.rmtree, self.tmp, True)
        self.fx = Fixture(self.tmp / "S")
        self.bx = BatchFixture(self.tmp / "B") if batch else None  # v1.6: a batch from another tree
        if official2:  # v1.8: a second official run of task-a (the latest) and a queued run dir
            plant_official2(self.fx)
        self.queue = plant_queue(self.fx) if queue else None  # v1.8: the official2 queue's runner files
        self.data, self.state = self.tmp / "data", self.tmp / "state"
        self.counters = {"prompt": 1000.0, "gen": 100.0}
        self.hub_calls = []
        self.hub_samples = []
        self.slurm_calls = []

    def slurm(self, ids):
        self.slurm_calls.append(sorted(ids))
        return {job: {"state": "RUNNING", "partition": "cpu_interactive", "node": "cpu1-1", "submit": NOW - 5000,
                      "start": NOW - 4000, "end": NOW + 80000, "elapsed": "1:06:40", "time_left": "22:00:00",
                      "reason": None} for job in ids}

    def scrape(self, url):
        if url == "http://10.0.0.1:8000/metrics":
            return METRICS.format(**self.counters)
        raise OSError("connection refused")

    def hub_start(self, src):
        self.hub_calls.append(src.hub_budget)
        return self.hub_samples.pop(0) if self.hub_samples else {"status": 200, "spend": 100.0,
                                                                 "max_budget": 1000.0, "remaining": 900.0}

    @staticmethod
    def hub_finish(sample, timeout):
        return json.dumps(dict(sample, at="2026-10-17T15:00:00-0700", error=None)) + "\n", 0

    def cycle(self, now=NOW, hub=True):
        sources = collect.Sources(root=self.fx.root, three_way=self.fx.three_way,
                                  batches=[self.bx.batch] if self.bx else None,
                                  queues=[collect.Queue("official2", self.queue)] if self.queue else None)
        return collect.run_cycle(self.data, self.state, sources=sources, clock=lambda: now, slurm=self.slurm,
                                 scrape=self.scrape, hub=(self.hub_start, self.hub_finish) if hub else False)

    def load(self, name):
        return json.loads((self.data / name).read_text())


class TreeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = Harness(cls.addClassCleanup)
        cls.summary = cls.h.cycle()
        cls.doc = cls.h.load(f"task/{TASK}.json")
        cls.nodes = {node["id"]: node for node in cls.doc["nodes"]}
        cls.index = cls.h.load("index.json")

    def decision(self, name):
        node = self.nodes[name]
        return node["status"], node["decision"]

    def test_cycle_clean(self):
        self.assertEqual(self.summary["errors"], [])
        self.assertTrue(self.summary["ok"])
        self.assertEqual(self.doc["max_rounds"], 2)
        self.assertEqual(self.doc["branches"], 6)
        self.assertEqual(self.doc["repo"], "owner/proj")

    def test_round0(self):
        self.assertEqual(self.decision("cand-c01"), ("scored", "evolve"))
        self.assertEqual(self.decision("cand-c03"), ("scored", "accepted_evolve"))
        # seeds.json invalid (and not accepted: AcceptedInvalidSeedTest) -> discard with that reason
        self.assertEqual(self.decision("cand-c02"), ("scored", "discard"))
        self.assertIn("verifier produced no grade for 2 rollout(s)", self.nodes["cand-c02"]["decision_reason"])
        # stage.json {"ok": false} -> build_failed / discard, first problem line, no rollout
        node = self.nodes["cand-c04"]
        self.assertEqual((node["status"], node["decision"]), ("build_failed", "discard"))
        self.assertEqual(node["decision_reason"], "build failed: GateError: env broke")
        self.assertIsNone(node["probe"])
        self.assertEqual(node["build"]["attempts"], 2)
        self.assertIn("gates attempt 0: only 1 fail-to-pass tests", node["build"]["problems"])
        # a renamed stage.json.failed-materialize is not stage data
        self.assertTrue(self.nodes["cand-c01"]["build"]["ok"])

    def test_round1(self):
        self.assertEqual(self.decision("cand-c01-r1b0"), ("scored", "evolve"))
        self.assertEqual(self.decision("cand-c01-r1b1"), ("scored", "not_selected"))
        self.assertIn("from cand-c01-r1b0", self.nodes["cand-c01-r1b1"]["decision_reason"])
        self.assertEqual(self.decision("cand-c03-r1b0"), ("scored", "accepted"))
        self.assertEqual(self.decision("cand-c03-r1b1"), ("probing", "pending"))
        self.assertEqual(self.decision("cand-c03-r1b2"), ("scored", "pending"))  # lineage c03 not final yet

    def test_abandoned(self):
        # parent invalid in its round's seeds.json
        self.assertEqual(self.decision("cand-c02-r1b0"), ("abandoned", "abandoned"))
        self.assertIn("parent cand-c02 is not a valid seed", self.nodes["cand-c02-r1b0"]["decision_reason"])
        # only in a probe superseded by newer probes of the same round
        self.assertEqual(self.decision("cand-c01-r1b2"), ("abandoned", "abandoned"))
        self.assertIn("round-1/probe-old was superseded", self.nodes["cand-c01-r1b2"]["decision_reason"])
        # parent not its lineage's selected seed
        self.assertEqual(self.decision("cand-c01-r1b1-r2b0"), ("abandoned", "abandoned"))
        self.assertIn("(cand-c01-r1b0 was)", self.nodes["cand-c01-r1b1-r2b0"]["decision_reason"])
        rounds = {r["index"]: r for r in self.doc["rounds"]}
        self.assertEqual(rounds[1]["abandoned_probes"], ["round-1/probe-old"])

    def test_round2(self):
        self.assertEqual(self.decision("cand-c01-r1b0-r2b0"), ("scored", "end"))
        self.assertEqual(self.decision("cand-c01-r1b0-r2b1"), ("scored", "accepted"))
        self.assertEqual(self.decision("cand-c01-r1b0-r2b2"), ("planning", "pending"))
        self.assertEqual(self.decision("cand-c01-r1b0-r2b3"), ("building", "pending"))
        self.assertEqual(self.nodes["cand-c01-r1b0-r2b3"]["build"]["f2p"], 2)
        # no seeds.json in round 2: seed_check() recomputed
        self.assertEqual(self.decision("cand-c01-r1b0-r2b4"), ("scored", "discard"))
        self.assertIn("fewer than 3 valid rollouts", self.nodes["cand-c01-r1b0-r2b4"]["decision_reason"])
        self.assertEqual(self.decision("cand-c01-r1b0-r2b5"), ("scored", "discard"))
        self.assertEqual(self.nodes["cand-c01-r1b0-r2b5"]["decision_reason"],
                         "verifier produced no grade for 1 rollout(s): sc14-r0")
        self.assertEqual(self.nodes["cand-c01-r1b0-r2b5"]["score"]["grading_failures"], ["sc14-r0"])
        # stage.json.budget-422 / author-0.budget-422 are renamed: no stage data, no attempt
        node = self.nodes["cand-c01-r1b0-r2b6"]
        self.assertEqual((node["status"], node["decision"]), ("authoring", "pending"))
        self.assertEqual(node["build"]["attempts"], 0)
        self.assertIsNone(node["build"]["ok"])
        self.assertNotIn("cand-weird.budget-422", self.nodes)

    def test_node_schema(self):
        node = self.nodes["cand-c01-r1b0"]
        self.assertEqual((node["parent"], node["lineage"], node["round"], node["branch"], node["kind"]),
                         ("cand-c01", "cand-c01", 1, 0, "evolve"))
        self.assertEqual(node["title"], "Evolved cand-c01-r1b0")
        self.assertEqual(node["children"], [f"cand-c01-r1b0-r2b{j}" for j in range(7)])
        self.assertEqual(node["build"]["f2p_tests"], ["t1", "t2", "t3"])
        self.assertEqual(node["build"]["task_id"], "syn-cand-c01-r1b0")
        probe = node["probe"]
        self.assertEqual(probe["student"], {"pass": 1, "fail": 4, "infra": 0, "rate": 0.2})
        self.assertEqual(probe["teacher"]["pass"], 5)
        self.assertEqual(probe["progress"], {"kimi": [5, 5], "sc14": [5, 5]})
        self.assertEqual(len(probe["rollouts"]), 10)
        self.assertEqual(set(probe["rollouts"][0]), {"slot", "model", "status", "reward", "f2p", "p2p", "tool_calls",
                                                     "patch_chars", "agent_error", "verifier_error"})
        self.assertEqual(node["details"]["evolve_plan_md"], "# evolve plan cand-c01-r1b0\n")
        self.assertEqual([entry["slot"] for entry in node["details"]["teacher_audit"]],
                         [f"kimi-r{i}" for i in range(5)])
        root = self.nodes["cand-c03"]
        self.assertEqual(root["title"], "Title 3")  # mask nodes: the plan candidate's title
        self.assertEqual(json.loads(root["score"]["similarity_detail"])["score"], 7)
        self.assertNotIn("_", root)
        self.assertNotIn("partial", node["probe"])  # scored: the rollouts table, no live counts

    def test_teacher_audit(self):
        audit = self.nodes["cand-c01"]["details"]["teacher_audit"]  # a list, one verdict per teacher pass
        self.assertEqual(audit[0], {"slot": "kimi-r0", "cheated": False, "categories": [],
                                    "summary": "No cheating: kimi-r0 followed the spec.", "rubric_score": 9.5})
        cheat = audit[1]
        self.assertEqual((cheat["slot"], cheat["cheated"], cheat["categories"], cheat["rubric_score"]),
                         ("kimi-r1", True, ["Memory narrated as lookup"], 9.0))
        self.assertLessEqual(len(cheat["summary"]), 400)  # the judge's notes, redacted, then clipped
        self.assertTrue(cheat["summary"].startswith("Recalled the upstream fix from memory (pasted nvapi-<redacted>)"))
        self.assertTrue(cheat["summary"].endswith("chars omitted]…"))
        self.assertEqual(audit[2]["summary"], "[1] read the spec; [2] ran the tests")  # no notes: the evidence

    def test_audit_entries_shapes(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        collect.allow_reads(tmp)
        empty = write(tmp / "empty.json", {})  # no teacher pass: nothing audited
        self.assertEqual(collect._audit_entries(empty), [])
        listed = write(tmp / "listed.json", [{"slot": "kimi-r3", "cheated": 1, "notes": "n"}, {"slot": "kimi-r1"}])
        self.assertEqual(collect._audit_entries(listed), [
            {"slot": "kimi-r1", "cheated": None, "categories": [], "summary": None, "rubric_score": None},
            {"slot": "kimi-r3", "cheated": True, "categories": [], "summary": "n", "rubric_score": None}])
        odd = write(tmp / "odd.json", {"kimi-r0": "judge reply was not JSON"})
        self.assertEqual(collect._audit_entries(odd)[0]["summary"], "judge reply was not JSON")
        self.assertIsNone(collect._audit_entries(tmp / "missing.json"))
        self.assertIsNone(collect._audit_entries(write(tmp / "scalar.json", "3")))

    def test_focus_pattern(self):
        for name, node in self.nodes.items():  # evolve(): branch j aims at patterns[j % 2]; mask nodes at none
            want = ("Scope creep", "Tests certify the bug")[node["branch"] % 2] if node["kind"] == "evolve" else None
            self.assertEqual(node["focus_pattern"], want, name)
        self.assertEqual(self.nodes["cand-c01-r1b0-r2b5"]["focus_pattern"], "Tests certify the bug")
        # task_builder evolve(): a result.json without patterns gives "overall"; without result.json, unknown
        h = Harness(self.addCleanup)
        result = h.fx.runs / RUN / "analysis" / TASK / "result.json"
        write(result, {"patterns": [], "transferable_signature": None, "excluded": []})
        h.cycle(hub=False)
        nodes = {n["id"]: n for n in h.load(f"task/{TASK}.json")["nodes"]}
        self.assertEqual((nodes["cand-c01-r1b1"]["focus_pattern"], nodes["cand-c01"]["focus_pattern"]),
                         ("overall", None))
        result.unlink()
        h.cycle(NOW + 60, hub=False)
        self.assertIsNone({n["id"]: n for n in h.load(f"task/{TASK}.json")["nodes"]}["cand-c01-r1b1"]["focus_pattern"])
        self.assertIsNone(collect._focus_pattern({"patterns": [{"title": "no name"}]}, 0))
        self.assertEqual(collect._focus_pattern({"patterns": None}, 7), "overall")

    def test_live_probe(self):
        node = self.nodes["cand-c03-r1b1"]
        probe = node["probe"]
        self.assertTrue(probe["live"])
        self.assertEqual(probe["dir"], "round-1/probe-new")
        self.assertEqual(probe["progress"], {"kimi": [2, 5], "sc14": [2, 5]})
        statuses = {row["slot"]: row["status"] for row in probe["rollouts"]}
        self.assertEqual(statuses, {"kimi-r0": "infra", "kimi-r1": "pass", "sc14-r0": "pass", "sc14-r1": "fail"})
        self.assertEqual(probe["student"]["rate"], 0.5)
        # per-model counts of the graded / final slots so far; sc14-r2 is still running
        self.assertEqual(probe["partial"], {"kimi": {"done": 2, "total": 5, "pass": 1, "fail": 0, "infra": 1},
                                            "sc14": {"done": 2, "total": 5, "pass": 1, "fail": 1, "infra": 0}})
        self.assertFalse(probe["own_done"])
        rounds = {r["index"]: r for r in self.doc["rounds"]}
        self.assertEqual(rounds[1]["status"], "probing")
        self.assertEqual(rounds[1]["probe"]["dir"], "round-1/probe-new")
        self.assertEqual(rounds[1]["probe"]["dirs"], ["round-1/probe-mid", "round-1/probe-new"])
        self.assertEqual(self.doc["job"]["stage"], "round-1 probe (4/10 slots)")

    def test_own_done(self):
        """Every own slot final: own_done, while the node keeps probing until the campaign writes metrics.json."""
        h = Harness(self.addCleanup)
        live = h.fx.build / "round-1/probe-new"
        fail = {"reward": 0, "f2p_passed": 1, "f2p_total": 3, "p2p_passed": 4, "p2p_total": 4}
        for model in ("sc14", "kimi"):
            for slot in ("0-2", "0-3", "0-4"):  # sc14 0-2 was running (an attempt without terminal.json)
                h.fx.slot(live, model, slot, [True], reward=fail)
        h.cycle(hub=False)
        node = {n["id"]: n for n in h.load(f"task/{TASK}.json")["nodes"]}["cand-c03-r1b1"]
        self.assertEqual((node["status"], node["decision"]), ("probing", "pending"))
        self.assertTrue(node["probe"]["own_done"])
        self.assertEqual(node["probe"]["partial"], {"kimi": {"done": 5, "total": 5, "pass": 1, "fail": 3, "infra": 1},
                                                    "sc14": {"done": 5, "total": 5, "pass": 1, "fail": 4, "infra": 0}})
        # finished slots come from the state afterwards (incremental): the same counts without re-reading them
        state = json.loads((h.state / "collector.json").read_text())
        self.assertEqual(sum(key.startswith(str(live)) for key in state["slots"]), 10)
        h.cycle(NOW + 60, hub=False)
        node = {n["id"]: n for n in h.load(f"task/{TASK}.json")["nodes"]}["cand-c03-r1b1"]
        self.assertEqual(node["probe"]["partial"]["sc14"]["fail"], 4)

    def test_rounds(self):
        rounds = {r["index"]: r for r in self.doc["rounds"]}
        self.assertEqual([rounds[k]["status"] for k in (0, 1, 2)], ["done", "probing", "pending"])
        self.assertEqual(rounds[0]["seeds"]["valid"], ["cand-c01", "cand-c03"])
        self.assertEqual(rounds[0]["probe"]["slots_total"], 30)
        self.assertEqual(rounds[0]["probe"]["slots_done"], 30)
        self.assertEqual(rounds[0]["probe"]["by_model"], {"kimi": [15, 15], "sc14": [15, 15]})
        self.assertIsNone(rounds[1]["seeds"])

    def test_job_and_analysis(self):
        job = self.doc["job"]
        self.assertEqual((job["job_id"], job["slurm_state"], job["status"]), ("123", "RUNNING", "running"))
        self.assertTrue(job["log_path"].endswith("synth-build-official1-test-123.out"))
        self.assertLessEqual(len(job["log_tail"]), 40)
        self.assertTrue(all(len(line) <= 300 for line in job["log_tail"]))
        self.assertIn("Bearer <redacted>", job["log_tail"][-2])
        fa = self.doc["failure_analysis"]
        self.assertEqual(fa["patterns"][0]["student_slots"], ["sc14-r0", "sc14-r1"])
        self.assertEqual(fa["patterns"][0]["evidence"], "severity: high")
        self.assertEqual(json.loads(fa["transferable_signature"]), {"summary": "generalizes an exception"})
        self.assertEqual(fa["pass_stats"]["sc14"]["rate"], 0.2)
        plan = self.doc["plan"]
        self.assertEqual(len(plan["candidates"]), 5)
        self.assertEqual(plan["candidates"][0]["tests"], 3)
        self.assertTrue(plan["candidates"][0]["built"])
        self.assertFalse(plan["candidates"][3]["built"])  # c04 failed to build
        self.assertLessEqual(len(plan["mask_plan_excerpt"]), 4000)
        self.assertIn("chars omitted", plan["mask_plan_excerpt"])

    def test_index(self):
        index = self.index
        self.assertEqual(index["errors"], [])
        self.assertEqual(list(index["models"]), ["kimi", "super_vl", "sc14"])
        self.assertEqual([run["run"] for run in index["runs"]], [RUN])  # smoke1 is never recorded
        run = index["runs"][0]
        self.assertEqual((run["status"], run["stage"], run["accepted"]), ("running", "round-1 probe (4/10 slots)", 3))
        self.assertEqual(run["rounds_started"], 2)
        tasks = {task["task_id"]: task for task in index["tasks"]}
        a, b, c, d = (tasks[name] for name in ("task-a", "task-b", "task-c", "task-d"))
        self.assertEqual(a["accuracy"]["sc14"]["passed"], 1)
        self.assertEqual(a["accuracy"]["sc14"]["mean_f2p_ratio"], 0.9)
        self.assertEqual(a["accuracy"]["kimi"], {"passed": 5, "total": 5, "unscored": 0,
                                                 "rollouts": [{"f2p": "10/10", "p2p": "4/4", "score": 1}] * 5})
        self.assertEqual(a["accuracy"]["super_vl"]["unscored"], 2)
        self.assertIsNone(a["accuracy"]["super_vl"]["rollouts"][4]["f2p"])  # None/None
        self.assertEqual((a["eligible"], a["queue_rank"], a["run"]), (True, 2, RUN))
        self.assertEqual((b["eligible"], b["queue_rank"], b["run"]), (True, 1, None))
        self.assertEqual(c["accuracy"]["sc14"]["infra"], 1)
        self.assertFalse(c["eligible"])
        self.assertIsNone(c["queue_rank"])
        self.assertEqual(d["accuracy"], {"kimi": None, "super_vl": None, "sc14": None})
        self.assertFalse(d["eligible"])
        self.assertEqual(index["collect_seconds"] >= 0, True)


class AuditTest(unittest.TestCase):
    """Failure-match audits (CONTRACT "v1.3"): runs/<run>/audit/failure_match/<task_id>.json attached to the node whose
    build.task_id it names (validated, redacted, clipped), per-run counts in index.json, malformed files ignored."""

    @classmethod
    def setUpClass(cls):
        cls.h = Harness(cls.addClassCleanup)
        cls.summary = cls.h.cycle(hub=False)
        cls.nodes = {node["id"]: node for node in cls.h.load(f"task/{TASK}.json")["nodes"]}
        cls.index_run = cls.h.load("index.json")["runs"][0]

    def test_attached(self):
        self.assertEqual(self.summary["errors"], [])
        self.assertTrue(all("failure_match" in node for node in self.nodes.values()))  # v1.3: on every node
        fm = self.nodes["cand-c03"]["failure_match"]
        self.assertEqual(list(fm), ["verdict", "summary", "patterns", "student_rollouts", "other_causes",
                                    "similarity_judge", "audited_at"])  # target_patterns, auditor, notes: dropped
        self.assertEqual(fm["verdict"], "strong")
        self.assertEqual(fm["summary"], "sc14 broadens X in 4 of 5 rollouts, as the target did (it pasted "
                                        "nvapi-<redacted>).")
        scope, tests = fm["patterns"]
        self.assertEqual((scope["name"], scope["reproduced"], scope["slots"]), ("Scope creep", "yes", ["sc14-r1", "sc14-r2"]))
        self.assertTrue(560 < len(scope["evidence"]) <= 600)  # clipped to the format's limit
        self.assertTrue(scope["evidence"].startswith("sc14-r1 rewrote every caller of X: xxx"))
        self.assertTrue(scope["evidence"].endswith("chars omitted]…"))
        self.assertEqual(tests, {"name": "Tests certify the bug", "reproduced": "no", "slots": [], "evidence": None})
        passed, failed = fm["student_rollouts"]
        self.assertEqual(passed, {"slot": "sc14-r0", "status": "pass", "failing_tests": [], "cause": None,
                                  "pattern": None})
        self.assertEqual((failed["status"], failed["cause"], failed["pattern"]),
                         ("fail", "broadened X to every caller", "Scope creep"))
        self.assertEqual(failed["failing_tests"], [f"tests/test_x.py::test_{i}" for i in range(8)])  # at most 8
        self.assertEqual(fm["other_causes"], "sc14-r4 ran out of steps before committing.")
        self.assertEqual(fm["similarity_judge"], {"score": 7, "agrees": True, "comment": "a fair 7"})
        self.assertEqual(fm["audited_at"], "2026-10-18T20:00:00-07:00")
        partial = self.nodes["cand-c03-r1b0"]["failure_match"]  # enums and text normalized
        self.assertEqual((partial["verdict"], partial["summary"], partial["patterns"], partial["similarity_judge"]),
                         ("partial", "Loose match: only sc14-r2.", [], None))
        self.assertIsNone(self.nodes["cand-c01-r1b0-r2b1"]["failure_match"])  # accepted, not audited (yet)
        self.assertEqual(self.nodes["cand-c01"]["failure_match"]["verdict"], "weak")  # attached to any node it names
        self.assertIsNone(self.nodes["cand-c04"]["failure_match"])  # build failed: no task id to match

    def test_run_counts(self):
        # accepted: cand-c03, cand-c03-r1b0, cand-c01-r1b0-r2b1; cand-c01 (evolve) is audited but not counted there;
        # v1.4: the 8 eligible tasks (EligibilityTest) count all three audits, cand-c01's too
        self.assertEqual(self.index_run["audit"], {
            "accepted": 3, "audited": 2, "strong": 1, "partial": 1, "weak": 0, "none": 0,
            "eligible": 8, "audited_eligible": 3, "eligible_verdicts": {"strong": 1, "partial": 1, "weak": 1, "none": 0}})
        self.assertEqual(self.index_run["audit"]["accepted"], self.index_run["accepted"])

    def test_lenient_fields(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        collect.allow_reads(tmp)
        path = write(tmp / "syn-x.json", {  # no schema or task_id (the file's name), odd shapes everywhere
            "verdict": "NONE", "summary": 42, "patterns": [{"name": "P", "reproduced": "maybe", "slots": "sc14-r1, sc14-r3",
                                                            "evidence": ["a list"]}, "not an object"] * 15,
            "student_rollouts": [{"slot": "sc14-r0", "status": " Infra", "failing_tests": "tests/a.py::t1 tests/b.py::t2",
                                  "cause": None, "pattern": ""}] * 12,
            "other_causes": "  ", "similarity_judge": {"score": True, "agrees": "yes"}, "audited_at": ["x"]})
        entry = collect._failure_match(path)
        self.assertEqual(entry["task_id"], "syn-x")
        fm = entry["match"]
        self.assertEqual((fm["verdict"], fm["summary"], fm["other_causes"], fm["audited_at"]), ("none", "42", None, None))
        self.assertEqual(len(fm["patterns"]), 15)  # objects only, at most 20
        self.assertEqual(fm["patterns"][0], {"name": "P", "reproduced": None, "slots": ["sc14-r1", "sc14-r3"],
                                             "evidence": None})
        self.assertEqual(len(fm["student_rollouts"]), 10)  # at most 10
        self.assertEqual(fm["student_rollouts"][0], {"slot": "sc14-r0", "status": "infra",
                                                     "failing_tests": ["tests/a.py::t1", "tests/b.py::t2"],
                                                     "cause": None, "pattern": None})
        self.assertEqual(fm["similarity_judge"], {"score": None, "agrees": None, "comment": None})
        self.assertIsNone(collect._failure_match(tmp / "missing.json"))

    def test_malformed_ignored(self):
        h = Harness(self.addCleanup)
        audits = h.fx.runs / RUN / "audit/failure_match"
        write(audits / "syn-cand-c03.json", '{"schema": 1, "verdict": "str')  # torn (being written): skipped quietly
        write(audits / "syn-cand-c03-r1b0.json", {"task_id": "syn-cand-c03-r1b0", "verdict": "moderate"})
        write(audits / "syn-cand-c01-r1b0-r2b1.json", ["strong"])
        write(audits / "syn-cand-c01.json", {"task_id": "syn-cand-c01-r1b0", "verdict": "weak"})  # misnamed
        write(audits / "syn-cand-c01-r1b0.json", {"schema": 2, "verdict": "none"})
        write(audits / "syn-cand-c01-r1b1.json", {"verdict": "strong", "summary": "x" * (1 << 20)})  # > 1 MB
        write(audits / ".syn-cand-c01-r1b0-r2b0.json.tmp", {"verdict": "strong"})  # an atomic write in progress
        write(audits / "README.md", "notes")
        outside = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, outside, True)
        (audits / "syn-cand-c03-r1b2.json").symlink_to(write(outside / "planted.json", {"verdict": "strong"}))
        os.mkfifo(audits / "syn-cand-c01-r1b0-r2b4.json")  # never blocks the cycle
        summary = h.cycle(hub=False)
        errors = sorted(error for error in summary["errors"] if error.startswith(f"run {RUN} audit: "))
        self.assertEqual(errors, [
            f"run {RUN} audit: syn-cand-c01-r1b0-r2b1.json: not a JSON object; ignored",
            f"run {RUN} audit: syn-cand-c01-r1b0.json: schema 2 is not 1; ignored",
            f"run {RUN} audit: syn-cand-c01-r1b1.json: larger than 1024 KB; ignored",
            f"run {RUN} audit: syn-cand-c01.json: task_id \"syn-cand-c01-r1b0\" is not the file's name; ignored",
            f"run {RUN} audit: syn-cand-c03-r1b0.json: verdict \"moderate\" is not one of strong / partial / weak / "
            "none; ignored"])
        self.assertEqual(len(summary["errors"]), len(errors))  # every other source is unaffected
        doc = h.load(f"task/{TASK}.json")
        self.assertTrue(all(node["failure_match"] is None for node in doc["nodes"]))
        self.assertEqual(h.load("index.json")["runs"][0]["audit"],
                         {"accepted": 3, "audited": 0, "strong": 0, "partial": 0, "weak": 0, "none": 0,
                          "eligible": 8, "audited_eligible": 0,
                          "eligible_verdicts": {"strong": 0, "partial": 0, "weak": 0, "none": 0}})
        # a fixed file is picked up by the next cycle
        h.fx.audit("syn-cand-c03-r1b0", verdict="none", summary="Other causes: the environment broke.")
        summary = h.cycle(NOW + 60, hub=False)
        self.assertEqual(len(summary["errors"]), 4)
        nodes = {node["id"]: node for node in h.load(f"task/{TASK}.json")["nodes"]}
        self.assertEqual(nodes["cand-c03-r1b0"]["failure_match"]["verdict"], "none")
        self.assertEqual(h.load("index.json")["runs"][0]["audit"]["none"], 1)

    def test_no_audit_dir(self):
        h = Harness(self.addCleanup)
        shutil.rmtree(h.fx.runs / RUN / "audit")
        summary = h.cycle(hub=False)
        self.assertEqual(summary["errors"], [])
        self.assertTrue(all(node["failure_match"] is None for node in h.load(f"task/{TASK}.json")["nodes"]))
        self.assertEqual(h.load("index.json")["runs"][0]["audit"]["audited"], 0)


class EligibilityTest(unittest.TestCase):
    """The failure-match audit's scope (CONTRACT "v1.4"; hengyuf 2026-10-19: every task that is not discarded and on
    which the student has a rollout): NODE.audit_eligible, NODE.withdrawn and the eligible counts of runs[].audit."""

    # scored, a graded sc14 rollout, valid: accepted, rejected on pass rates, and the decisions in between
    ELIGIBLE = ["cand-c01", "cand-c03", "cand-c01-r1b0", "cand-c01-r1b1", "cand-c03-r1b0", "cand-c03-r1b2",
                "cand-c01-r1b0-r2b0", "cand-c01-r1b0-r2b1"]

    @classmethod
    def setUpClass(cls):
        cls.h = Harness(cls.addClassCleanup)
        cls.summary = cls.h.cycle(hub=False)
        cls.nodes = {node["id"]: node for node in cls.h.load(f"task/{TASK}.json")["nodes"]}

    @staticmethod
    def tree(h):
        return {node["id"]: node for node in h.load(f"task/{TASK}.json")["nodes"]}

    def eligible(self, nodes):
        return sorted(name for name, node in nodes.items() if node["audit_eligible"])

    def test_scope(self):
        self.assertEqual(self.summary["errors"], [])
        self.assertTrue(all(node["audit_eligible"] in (True, False) and node["withdrawn"] is None
                            for node in self.nodes.values()))
        self.assertEqual(self.eligible(self.nodes), sorted(self.ELIGIBLE))
        self.assertEqual({self.nodes[name]["decision"] for name in self.ELIGIBLE},
                         {"accepted_evolve", "accepted", "evolve", "not_selected", "end", "pending"})
        # out: invalid in seeds.json (c02); seed_check() recomputed where round 2 has no seeds.json (r2b4: < 3 valid
        # rollouts, r2b5: a grading failure); build failed (c04); never scored (probing, built, planning, authoring,
        # abandoned ones)
        for name in ("cand-c02", "cand-c01-r1b0-r2b4", "cand-c01-r1b0-r2b5", "cand-c04", "cand-c03-r1b1",
                     "cand-c01-r1b0-r2b2", "cand-c01-r1b0-r2b3", "cand-c01-r1b0-r2b6", "cand-c01-r1b2", "cand-c02-r1b0",
                     "cand-c01-r1b1-r2b0"):
            self.assertIs(self.nodes[name]["audit_eligible"], False, name)
        self.assertEqual(len(self.nodes), len(self.ELIGIBLE) + 11)

    def test_withdrawn(self):
        """Main withdraws an accepted task: metrics.json gets "withdrawn" (accept false, the reason rewritten), its task
        dir moves from accepted/ to withdrawn/; either is enough. A withdrawn task is discarded: never accepted, never
        eligible (its audit stays attached, uncounted)."""
        h = Harness(self.addCleanup)
        path = h.fx.build / "round-2/cand-c01-r1b0-r2b1/metrics.json"
        data = json.loads(path.read_text())
        why = "grading artifact: sc14-r2 only failed a P2P test\nthat test.patch never resets. " + "x" * 700
        data.update(accept=False, reason=f"withdrawn: {why}", withdrawn={
            "at": "2026-10-18T23:03:14-0700", "by": "main", "why": why,
            "was": {"accept": True, "reason": "similarity 6/10"}})
        write(path, data)
        write(h.fx.runs / RUN / "withdrawn/syn-cand-c03-r1b0/task.toml", "x")  # moved; its metrics.json untouched
        write(h.fx.runs / RUN / "withdrawn/syn-cand-c03-r1b0.workspace/manifest.json", {})
        write(h.fx.runs / RUN / "withdrawn/withdrawn.jsonl", "{}\n")
        self.assertEqual(h.cycle(hub=False)["errors"], [])
        nodes = self.tree(h)
        r2b1, r1b0 = nodes["cand-c01-r1b0-r2b1"], nodes["cand-c03-r1b0"]
        self.assertEqual((r2b1["decision"], r2b1["audit_eligible"], r2b1["score"]["accept"]), ("end", False, False))
        self.assertTrue(r2b1["decision_reason"].startswith(
            "not accepted (withdrawn: grading artifact: sc14-r2 only failed a P2P test that test.patch never resets."))
        self.assertNotIn("\n", r2b1["decision_reason"])
        withdrawn = r2b1["withdrawn"]
        self.assertEqual((withdrawn["at"], withdrawn["by"]), ("2026-10-18T23:03:14-0700", "main"))
        self.assertTrue(withdrawn["why"].startswith("grading artifact: sc14-r2") and len(withdrawn["why"]) <= 600)
        self.assertTrue(withdrawn["why"].endswith("chars omitted]…"))
        # metrics.json still says accept: true; the task dir in withdrawn/ alone withdraws it
        self.assertEqual(r1b0["withdrawn"], {"at": None, "by": None, "why": None})
        self.assertIs(r1b0["score"]["accept"], True)
        self.assertEqual((r1b0["decision"], r1b0["audit_eligible"]), ("pending", False))  # lineage c03 still probing
        self.assertEqual(r1b0["decision_reason"], f"not accepted (withdrawn (its task dir is in runs/{RUN}/withdrawn/)); "
                                                  "waiting for round 2")
        self.assertEqual(r1b0["failure_match"]["verdict"], "partial")  # still attached, no longer counted
        self.assertEqual([name for name, node in nodes.items() if node["withdrawn"]],
                         ["cand-c03-r1b0", "cand-c01-r1b0-r2b1"])
        self.assertEqual(self.eligible(nodes), sorted(set(self.ELIGIBLE) - {"cand-c03-r1b0", "cand-c01-r1b0-r2b1"}))
        run = h.load("index.json")["runs"][0]
        self.assertEqual(run["accepted"], 1)
        self.assertEqual(run["audit"], {"accepted": 1, "audited": 1, "strong": 1, "partial": 0, "weak": 0, "none": 0,
                                        "eligible": 6, "audited_eligible": 2,
                                        "eligible_verdicts": {"strong": 1, "partial": 0, "weak": 1, "none": 0}})

    def test_withdrawal_keeps_the_seed_selection(self):
        """official1-bandit, 2026-10-18: cand-c05-r1b1 was accepted in round 1, so task_builder seeded round 2 from
        cand-c05-r1b2; main withdrew cand-c05-r1b1 afterwards. The recomputed selection must still skip it, or the real
        seed would read as not selected and its scored (two accepted) children as abandoned."""
        h = Harness(self.addCleanup)
        path = h.fx.build / "round-1/cand-c01-r1b1/metrics.json"
        data = json.loads(path.read_text())
        data.update(accept=False, similarity=9, reason="withdrawn: grading artifact", withdrawn={
            "at": "2026-10-18T23:03:14-0700", "by": "main", "why": "grading artifact",
            "was": {"accept": True, "reason": "similarity 9/10"}})
        write(path, data)
        h.cycle(hub=False)
        nodes = self.tree(h)
        self.assertEqual(nodes["cand-c01-r1b0"]["decision"], "evolve")  # still round 2's seed
        self.assertEqual([nodes[f"cand-c01-r1b0-r2b{j}"]["decision"] for j in (0, 1)], ["end", "accepted"])
        self.assertEqual(nodes["cand-c01-r1b1"]["decision"], "not_selected")
        self.assertEqual(nodes["cand-c01-r1b1"]["decision_reason"], "not accepted (withdrawn: grading artifact); round 2 "
                                                                    "continues lineage cand-c01 from cand-c01-r1b0")
        self.assertIn("(cand-c01-r1b0 was)", nodes["cand-c01-r1b1-r2b0"]["decision_reason"])
        # a record of not being accepted then: it took part in the selection (and wins it on similarity)
        data["withdrawn"]["was"]["accept"] = False
        write(path, data)
        h.cycle(NOW + 60, hub=False)
        nodes = self.tree(h)
        self.assertEqual((nodes["cand-c01-r1b0"]["decision"], nodes["cand-c01-r1b0-r2b0"]["decision"]),
                         ("not_selected", "abandoned"))

    def test_discarded_and_ungraded(self):
        h = Harness(self.addCleanup)
        # round 2 has no seeds.json yet: seed_check() recomputed, uniform failure included
        path = h.fx.build / "round-2/cand-c01-r1b0-r2b0/metrics.json"
        write(path, dict(json.loads(path.read_text()), uniform_failure="all 10 graded rollouts with a patch score "
                                                                       "F2P 1/3, P2P 2/4"))
        # scored, but every sc14 rollout ended without a grade (agent errors): nothing of the student's to audit
        index = h.fx.build / "round-1/cand-c03-r1b2/rollouts/index.md"
        write(index, re.sub(r"^(\| sc14-r\d \| )(?:pass|fail)", r"\1infra", index.read_text(), flags=re.M))
        h.cycle(hub=False)
        nodes = self.tree(h)
        r2b0, r1b2 = nodes["cand-c01-r1b0-r2b0"], nodes["cand-c03-r1b2"]
        self.assertEqual((r2b0["decision"], r2b0["decision_reason"], r2b0["audit_eligible"]),
                         ("discard", "tests cannot run normally: all 10 graded rollouts with a patch score F2P 1/3, "
                                     "P2P 2/4", False))
        self.assertEqual((r1b2["status"], r1b2["decision"], r1b2["audit_eligible"]), ("scored", "pending", False))
        self.assertEqual(r1b2["probe"]["student"]["infra"], 5)
        self.assertEqual(self.eligible(nodes), sorted(set(self.ELIGIBLE) - {"cand-c01-r1b0-r2b0", "cand-c03-r1b2"}))
        self.assertEqual(h.load("index.json")["runs"][0]["audit"]["eligible"], 6)

    def test_abandoned_root(self):
        h = Harness(self.addCleanup)
        # main set an abandoned build aside next to the live one: the live root stays the build, newer or not
        write(h.fx.runs / RUN / "build" / TASK / "abandoned-other__repo/cand-c01/stage.json", {"ok": True})
        h.cycle(hub=False)
        doc = h.load(f"task/{TASK}.json")
        self.assertTrue(doc["build_root"].endswith(f"/build/{TASK}/{REPO}"))
        self.assertEqual(sum(node["audit_eligible"] for node in doc["nodes"]), len(self.ELIGIBLE))
        # a build whose only root is an abandoned one: still shown, nothing in the audit's scope
        shutil.rmtree(h.fx.runs / RUN / "build" / TASK / "abandoned-other__repo")
        os.rename(h.fx.build, h.fx.build.with_name(f"abandoned-{REPO}"))
        h.cycle(NOW + 60, hub=False)
        doc = h.load(f"task/{TASK}.json")
        self.assertTrue(doc["build_root"].endswith(f"/build/{TASK}/abandoned-{REPO}"))
        self.assertEqual(len(doc["nodes"]), len(self.nodes))
        self.assertEqual(self.tree(h)["cand-c03"]["decision"], "accepted_evolve")
        self.assertFalse(any(node["audit_eligible"] for node in doc["nodes"]))
        self.assertEqual(h.load("index.json")["runs"][0]["audit"]["eligible"], 0)

    def test_one_line_reasons(self):
        h = Harness(self.addCleanup)
        write(h.fx.build / "cand-c04/stage.json", {"ok": False, "problems": "GateError: " + "y" * 400 + "\nsecond"})
        h.cycle(hub=False)
        reason = self.tree(h)["cand-c04"]["decision_reason"]
        self.assertTrue(reason.startswith("build failed: GateError: yyy") and reason.endswith("chars omitted]…"))
        self.assertNotIn("\n", reason)


class AcceptedInvalidSeedTest(unittest.TestCase):
    """CONTRACT "v1.5" (main, 2026-10-21): acceptance beats an invalid seed. task_builder build() accepts a task on its
    score alone; seeds.json (seed_check) only picks the parents of the next round. An accepted task listed invalid there
    is decided accepted (never accepted_evolve: it seeds no branch), carries the reason as NODE.seed_invalid, is not
    discarded, so it is in the audit's scope; a withdrawn task stays not accepted, as before."""

    @classmethod
    def setUpClass(cls):
        cls.h = Harness(cls.addClassCleanup)
        plant_accepted_invalid_seeds(cls.h.fx)
        cls.summary = cls.h.cycle(hub=False)
        cls.nodes = EligibilityTest.tree(cls.h)
        cls.index_run = cls.h.load("index.json")["runs"][0]

    def test_round1_accepted_invalid_seed(self):
        """official1-wasmi cand-c03-r1b2's case: accepted, with the seeds.json reason as a note."""
        self.assertEqual(self.summary["errors"], [])
        node = self.nodes["cand-c03-r1b0"]
        self.assertEqual((node["status"], node["decision"], node["decision_reason"], node["seed_invalid"]),
                         ("scored", "accepted", "accepted (similarity 8/10)", SEED_INVALID))
        self.assertEqual((node["score"]["accept"], node["score"]["grading_failures"]), (True, ["kimi-r3"]))
        self.assertIs(node["audit_eligible"], True)  # not discarded: in the failure-match audit's scope
        self.assertEqual(node["failure_match"]["verdict"], "partial")
        rounds = {r["index"]: r for r in self.h.load(f"task/{TASK}.json")["rounds"]}
        self.assertEqual(rounds[1]["seeds"]["invalid"], {"cand-c03-r1b0": SEED_INVALID})  # the column head's count
        self.assertEqual(rounds[1]["accepted"], 1)

    def test_round0_accepted_invalid_seed(self):
        """Round 0: accepted, not accepted_evolve (task_builder evolves only the valid seeds); its children stay
        abandoned, since seed validity still governs evolution."""
        node = self.nodes["cand-c02"]
        self.assertEqual((node["decision"], node["decision_reason"], node["seed_invalid"], node["audit_eligible"]),
                         ("accepted", "accepted (similarity 7/10); an invalid seed, so not evolved", C02_INVALID, True))
        child = self.nodes["cand-c02-r1b0"]
        self.assertEqual(child["decision"], "abandoned")
        self.assertEqual(child["decision_reason"], f"parent cand-c02 is not a valid seed ({C02_INVALID})")

    def test_notes_of_the_other_nodes(self):
        """The note is the seed verdict of every scored task: a discard (seed_check() recomputed where round 2 has no
        seeds.json) carries it as its reason too; valid seeds, unscored tasks and failed builds have none."""
        for name in ("cand-c01-r1b0-r2b4", "cand-c01-r1b0-r2b5"):
            node = self.nodes[name]
            self.assertEqual(node["decision"], "discard", name)
            self.assertEqual(node["seed_invalid"], node["decision_reason"], name)
        self.assertEqual(self.nodes["cand-c01-r1b0-r2b5"]["seed_invalid"],
                         "verifier produced no grade for 1 rollout(s): sc14-r0")
        for name in ("cand-c01", "cand-c03", "cand-c01-r1b0", "cand-c01-r1b1", "cand-c03-r1b2", "cand-c01-r1b0-r2b0",
                     "cand-c01-r1b0-r2b1", "cand-c03-r1b1", "cand-c01-r1b2", "cand-c04", "cand-c01-r1b0-r2b6"):
            self.assertIsNone(self.nodes[name]["seed_invalid"], name)
        self.assertTrue(all("seed_invalid" in node for node in self.nodes.values()))

    def test_run_counts(self):
        """Four accepted tasks (cand-c03, cand-c03-r1b0, cand-c01-r1b0-r2b1 and now cand-c02) in runs[].accepted, the
        audit counts and a finished run's stage line ("finished (5 accepted)" read 5 of wasmi's 6); nine eligible."""
        self.assertEqual(self.index_run["accepted"], 4)
        doc = self.h.load(f"task/{TASK}.json")
        self.assertEqual(collect._stage("finished", {"nodes": doc["nodes"]}, None, None), "finished (4 accepted)")
        self.assertEqual(self.index_run["audit"], {"accepted": 4, "audited": 2, "strong": 1, "partial": 1, "weak": 0,
                                                   "none": 0, "eligible": 9, "audited_eligible": 3,
                                             "eligible_verdicts": {"strong": 1, "partial": 1, "weak": 1, "none": 0}})
        self.assertEqual(sorted(name for name, node in self.nodes.items() if node["audit_eligible"]),
                         sorted(EligibilityTest.ELIGIBLE + ["cand-c02"]))

    def test_recomputed_seed_check(self):
        """Before seeds.json lists it, seed_check() is recomputed (here: a grading failure): accepted all the same."""
        h = Harness(self.addCleanup)
        plant_accepted_invalid_seeds(h.fx)
        (h.fx.build / "round-1/seeds.json").unlink()
        h.cycle(hub=False)
        node = EligibilityTest.tree(h)["cand-c03-r1b0"]
        self.assertEqual((node["decision"], node["seed_invalid"], node["audit_eligible"]),
                         ("accepted", SEED_INVALID, True))

    def test_withdrawn_stays_not_accepted(self):
        """A withdrawn task stays not accepted, as before (CONTRACT "v1.4"): listed invalid, it is discarded with the
        seeds.json reason and leaves the audit's scope, whether main moved its task dir to withdrawn/ (metrics.json
        still saying accept: true) or rewrote its metrics.json."""
        h = Harness(self.addCleanup)
        plant_accepted_invalid_seeds(h.fx)
        write(h.fx.runs / RUN / "withdrawn/syn-cand-c03-r1b0/task.toml", "x")
        path = h.fx.build / "round-0/cand-c02/metrics.json"
        data = json.loads(path.read_text())
        data.update(accept=False, reason="withdrawn: grading artifact", withdrawn={
            "at": "2026-10-21T09:00:00-0700", "by": "main", "why": "grading artifact", "was": {"accept": True}})
        write(path, data)
        self.assertEqual(h.cycle(hub=False)["errors"], [])
        nodes = EligibilityTest.tree(h)
        for name, reason in (("cand-c03-r1b0", SEED_INVALID), ("cand-c02", C02_INVALID)):
            node = nodes[name]
            self.assertTrue(node["withdrawn"], name)
            self.assertEqual((node["decision"], node["decision_reason"], node["seed_invalid"], node["audit_eligible"]),
                             ("discard", reason, reason, False), name)
        self.assertEqual(h.load("index.json")["runs"][0]["accepted"], 2)  # cand-c03, cand-c01-r1b0-r2b1


class ReadSafetyTest(unittest.TestCase):
    """2026-10-17 security review: published files must not follow symlinks out of the read roots, and known secret
    values are masked even without a recognizable prefix."""

    def test_symlink_escape_and_fifo_are_refused(self):
        root, outside = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        self.addCleanup(shutil.rmtree, outside, True)
        collect.allow_reads(root)
        (outside / "secret.txt").write_text("TOP SECRET")
        (root / "inside.md").write_text("fine")
        (root / "evil.md").symlink_to(outside / "secret.txt")
        (root / "alias.md").symlink_to(root / "inside.md")
        os.mkfifo(root / "pipe.md")
        self.assertEqual(collect.read_text(root / "inside.md"), "fine")
        self.assertEqual(collect.read_text(root / "alias.md"), "fine")  # a symlink inside the roots is fine
        self.assertIsNone(collect.read_text(root / "evil.md"))
        self.assertIsNone(collect.read_text(root / "pipe.md"))  # never blocks on a FIFO
        self.assertIsNone(collect.read_text(outside / "secret.txt"))

    def test_known_secret_values_are_masked(self):
        saved = collect._SECRET_VALUES
        self.addCleanup(setattr, collect, "_SECRET_VALUES", saved)
        collect._SECRET_VALUES = ("plainvalue0123456789",)
        self.assertEqual(collect.redact("x plainvalue0123456789 y"), "x <redacted-secret> y")
        self.assertEqual(collect.redact("nothing here"), "nothing here")


class RedactTest(unittest.TestCase):
    def test_patterns(self):
        cases = [  # (text, the secret part that must disappear, what remains)
            (f"use {SECRET_KEY} now", SECRET_KEY[6:], "use nvapi-<redacted> now"),
            (f"OPENAI_API_KEY={SECRET_SK}", SECRET_SK[3:], "OPENAI_API_KEY=sk-<redacted>"),
            ("Authorization: Bearer abc.def-ghi_jkl", "abc.def-ghi_jkl", "Authorization: Bearer <redacted>"),
            ("x hf_" + "a1" * 10, "a1" * 10, "x hf_<redacted>"),
            ("ghp_" + "A" * 36, "A" * 36, "gh_<redacted>"),
            ("glpat-" + "x" * 20, "x" * 20, "glpat-<redacted>"),
            ("id AKIA" + "ABCDEFGHIJKLMNOP", "ABCDEFGHIJKLMNOP", "id AKIA<redacted>"),
            (SECRET_TOKEN, SECRET_TOKEN[6:], "token=<redacted>"),
            ('NVHUB_SECRET: "' + "Zm9v" * 9 + '"', "Zm9v" * 9, 'NVHUB_SECRET: "<redacted>"'),
        ]
        for text, secret, expected in cases:
            out = collect.redact(text)
            self.assertEqual(out, expected, text)
            self.assertNotIn(secret, out, text)
        for text in ("risk-assessment-framework-name", "total_tokens: 1234567", "prompt_sha256 " + "a" * 64,
                     "task-sk-learn", "the key is ready", "cached_tokens=" + "1" * 10):
            self.assertEqual(collect.redact(text), text)
        self.assertEqual(collect.redact(None), None)
        self.assertEqual(collect.redact(collect.redact(f"x {SECRET_KEY}")), collect.redact(f"x {SECRET_KEY}"))

    def test_linear_time_on_blobs(self):
        blob = "A" * 2_000_000 + "key" + "b" * 1_000_000  # a pathological pattern must stay fast
        import time
        started = time.monotonic()
        collect.redact(blob)
        self.assertLess(time.monotonic() - started, 2.0)

    def test_clip(self):
        self.assertEqual(collect.clip("short", 10), "short")
        text = "x" * 50000
        out = collect.clip(text, 20000)
        self.assertLessEqual(len(out), 20000)
        self.assertTrue(out.endswith("chars omitted]…"))
        omitted = int(out.rsplit("[", 1)[1].split()[0])
        self.assertEqual(len(out) - len(out.rsplit("\n", 1)[1]) - 1 + omitted, 50000)
        self.assertEqual(collect.clip(None, 5), None)

    def test_no_secret_in_outputs(self):
        h = Harness(self.addCleanup)
        h.cycle()
        for path in h.data.rglob("*.json*"):
            text = path.read_text()
            for secret in (SECRET_KEY, SECRET_SK, SECRET_TOKEN.split("=")[1], "abcdefghijklmnopqrstuvwxyz"):
                self.assertNotIn(secret, text, path)
        doc = h.load(f"task/{TASK}.json")
        self.assertIn("sk-<redacted>", doc["failure_analysis"]["markdown"])
        self.assertIn("token=<redacted>", doc["failure_analysis"]["markdown"])


class UsageTest(unittest.TestCase):
    def kimi(self, h):
        return {row["run"]: row for row in h.load("api.json")["by_run"]}[RUN]["kimi"]

    def test_incremental_offsets(self):
        h = Harness(self.addCleanup)
        h.cycle()
        kimi = self.kimi(h)
        self.assertEqual(kimi["calls"], 3)  # 2 in probe-mid + 1 in the moved attempt
        self.assertAlmostEqual(kimi["cost"], 1.75)
        self.assertEqual((kimi["input_tokens"], kimi["cached_tokens"], kimi["output_tokens"]), (3010, 1800, 61))
        self.assertEqual(kimi["http_errors"], 1)
        # append one event and half of the next line: only complete lines count
        append(h.fx.kimi, json.dumps(Fixture.usage(0.125, 10, 0, 2), separators=(",", ":")) + "\n"
               + '{"event":"upstream_stream_completed","usage":{"cost":')
        h.cycle(NOW + 60)
        self.assertEqual(self.kimi(h)["calls"], 4)
        append(h.fx.kimi, '0.0625,"prompt_tokens":1},"x":1}\n')
        h.cycle(NOW + 120)
        kimi = self.kimi(h)
        self.assertEqual(kimi["calls"], 5)
        self.assertAlmostEqual(kimi["cost"], 1.9375)
        # an attempt dir moved aside (budget-422-moved) is the same inode: never counted twice
        moved = h.fx.build / "round-1/probe-mid/budget-422-moved/kimi-attempts/0-0"
        moved.parent.mkdir(parents=True)
        os.rename(h.fx.kimi.parent.parent, moved)
        h.cycle(NOW + 180)
        self.assertEqual(self.kimi(h)["calls"], 5)
        state = json.loads((h.state / "collector.json").read_text())
        paths = [entry["path"] for entry in state["files"].values() if entry["kind"] == "kimi"]
        self.assertTrue(any("budget-422-moved/kimi-attempts/0-0" in path for path in paths))
        self.assertEqual(oct(os.stat(h.state).st_mode & 0o777), "0o700")
        self.assertEqual(oct(os.stat(h.state / "collector.json").st_mode & 0o777), "0o600")

    def test_codex_rewrite_and_glm(self):
        h = Harness(self.addCleanup)
        h.cycle()
        by_run = {row["run"]: row for row in h.load("api.json")["by_run"]}[RUN]
        codex = by_run["codex"]
        # env session + the renamed author-0.budget-422 session (its usage was real); a quoted
        # "type":"turn.completed" inside an item is not a turn
        self.assertEqual((codex["sessions"], codex["turns_completed"], codex["turns_failed"]), (2, 1, 1))
        self.assertEqual((codex["input_tokens"], codex["cached_tokens"], codex["output_tokens"],
                          codex["reasoning_tokens"]), (100, 80, 10, 5))
        # codex reruns the stage: events.jsonl is rewritten from scratch; the earlier session still counts
        jsonl(h.fx.codex, [{"type": "thread.started", "thread_id": "t-3-rerun"}, Fixture.turn(1000, 0, 1, 1),
                           {"type": "turn.started"}, Fixture.turn(1, 0, 0, 0)])
        h.cycle(NOW + 60)
        codex = {row["run"]: row for row in h.load("api.json")["by_run"]}[RUN]["codex"]
        self.assertEqual((codex["sessions"], codex["turns_completed"], codex["input_tokens"]), (3, 3, 1101))
        # glm: one audit per teacher pass (5 per scored node, 11 nodes) + one similarity call per accepted node (3)
        self.assertEqual(by_run["glm"]["calls"], 11 * 5 + 3)

    def test_scan_budget(self):
        h = Harness(self.addCleanup)
        saved = collect.SCAN_BUDGET_S, collect.SCAN_MIN_S
        collect.SCAN_BUDGET_S = collect.SCAN_MIN_S = -100  # deadline already passed: nothing is read
        try:
            summary = h.cycle()
        finally:
            collect.SCAN_BUDGET_S, collect.SCAN_MIN_S = saved
        self.assertFalse(summary["scan"]["complete"])
        self.assertGreater(summary["scan"]["pending_bytes"], 0)
        summary = h.cycle(NOW + 60)
        self.assertTrue(summary["scan"]["complete"])
        self.assertEqual(self.kimi(h)["calls"], 3)


class FleetTest(unittest.TestCase):
    def test_parse_metrics(self):
        out = collect.parse_metrics(METRICS.format(prompt=1000.0, gen=100.0) + "garbage line\nvllm:x NaN\n")
        self.assertEqual(out, {"running": 3.0, "waiting": 1.0, "kv_usage": 0.2, "prompt_tokens_total": 1000.0,
                               "generation_tokens_total": 100.0})
        self.assertEqual(collect.parse_metrics("")["running"], None)

    def test_workers_and_tps(self):
        h = Harness(self.addCleanup)
        h.cycle()
        fleet = h.load("fleet.json")
        workers = {w["job_id"]: w for w in fleet["workers"]}
        first = workers["1001"]["metrics"]
        self.assertEqual((first["running"], first["waiting"], first["kv_usage"]), (3.0, 1.0, 0.2))
        self.assertIsNone(first["gen_tps"])  # no previous scrape yet
        self.assertIsNotNone(workers["1003"]["metrics"]["error"])
        self.assertEqual({key: value for key, value in workers["1003"]["metrics"].items() if key != "scraped_at"
                          and key != "error"}, {"running": None, "waiting": None, "kv_usage": None,
                                                "prompt_tps_incl_cache": None, "prompt_tps": None, "gen_tps": None})
        self.assertIsNone(workers["1002"]["metrics"])  # pending: not scraped
        h.counters = {"prompt": 6000.0, "gen": 600.0}
        h.cycle(NOW + 100)
        fleet = h.load("fleet.json")
        workers = {w["job_id"]: w for w in fleet["workers"]}
        metrics_1001 = workers["1001"]["metrics"]
        # prompt tok/s includes prefix-cache hits (vLLM's prompt counter); prompt_tps is the old name, kept one release
        self.assertEqual((metrics_1001["prompt_tps_incl_cache"], metrics_1001["prompt_tps"], metrics_1001["gen_tps"]),
                         (50.0, 50.0, 5.0))
        self.assertEqual(fleet["totals"]["gen_tps"], 5.0)
        self.assertEqual((fleet["totals"]["prompt_tps_incl_cache"], fleet["totals"]["prompt_tps"]), (50.0, 50.0))
        self.assertEqual([point.get("prompt_tps_incl_cache") for point in fleet["history"]], [None, 50.0])
        # a counter reset (worker restarted under the same job id) gives no rate
        h.counters = {"prompt": 10.0, "gen": 1.0}
        h.cycle(NOW + 200)
        self.assertIsNone(h.load("fleet.json")["workers"][0]["metrics"]["gen_tps"])

    def test_fleet_doc(self):
        h = Harness(self.addCleanup)
        h.cycle()
        fleet = h.load("fleet.json")
        workers = {w["job_id"]: w for w in fleet["workers"]}
        self.assertEqual(workers["1001"]["routing"], "ready")
        self.assertIsNone(workers["1003"]["routing"])  # heartbeat 500 s old: not routable
        self.assertEqual(workers["1001"]["left_s"], 3600)
        self.assertEqual(workers["1001"]["heartbeat_age_s"], 10)
        self.assertEqual((workers["1002"]["pending_reason"], workers["1002"]["est_start"]), ("Priority", NOW + 600))
        self.assertEqual(fleet["counts"]["ready"], 1)
        self.assertEqual(fleet["counts"]["unhealthy"], 0)
        self.assertEqual(fleet["coverage"], {"value": 2, "target": 2})
        self.assertEqual(fleet["daemon"]["heartbeat_age_s"], 20)
        self.assertEqual(fleet["daemon"]["breaker"], "closed")
        self.assertEqual(fleet["totals"]["routable"], 1)
        self.assertEqual(fleet["totals"]["busy_workers"], 1)  # 1001: routable with 3 running requests
        self.assertEqual(len(fleet["history"]), 1)
        # busy = routable AND >= 1 running request: 1001 is routable but now idle, 1003 runs 3 but is not routable
        idle = re.sub(r"(vllm:num_requests_running\{[^}]*\}) \S+", r"\1 0.0", METRICS.format(**h.counters))
        h.scrape = lambda url: idle if url == "http://10.0.0.1:8000/metrics" else METRICS.format(**h.counters)
        h.cycle(NOW + 60)  # 1001's heartbeat (NOW - 10) is still fresh enough to route
        fleet = h.load("fleet.json")
        self.assertEqual((fleet["totals"]["routable"], fleet["totals"]["busy_workers"]), (1, 0))
        self.assertEqual(len(fleet["history"]), 2)

    def test_router_stats(self):
        h = Harness(self.addCleanup)
        h.cycle()
        router = h.load("fleet.json")["router"]
        self.assertEqual(router["scope"], "official-run probes")  # only the official runs' probe journals
        self.assertEqual(router["requests"], 13)
        self.assertEqual((router["errors"], router["disconnects"], router["clamps"]), (1, 1, 1))
        self.assertAlmostEqual(router["error_rate"], 1 / 13, places=4)
        self.assertEqual(router["p50_s"], 5.0)       # durations 1..10 plus one 2.0 -> 11 successes
        self.assertEqual(router["by_worker"]["1001"]["requests"], 11)
        self.assertEqual(router["by_worker"]["1001"]["errors"], 1)
        self.assertEqual(router["last_hour"], {"requests": 14, "errors": 1})
        # incremental: one more request later; the window moves
        jsonl(h.fx.router, [Fixture.request(NOW + 290, worker="1002")], mode="a")
        h.cycle(NOW + 300)
        router = h.load("fleet.json")["router"]
        self.assertEqual(router["requests"], 14)
        self.assertEqual(router["last_hour"]["requests"], 15)
        h.cycle(NOW + 800)
        router = h.load("fleet.json")["router"]
        self.assertEqual(router["requests"], 1)  # only the NOW+290 request is within the last 10 min
        self.assertIsNone(collect._router_stats([], NOW)["p90_s"])


class HubTest(unittest.TestCase):
    def test_interval_guard(self):
        h = Harness(self.addCleanup)
        h.hub_samples = [{"status": 200, "spend": 100.0, "max_budget": 1000.0, "remaining": 900.0},
                         {"status": 200, "spend": 160.0, "max_budget": 1000.0, "remaining": 840.0}]
        h.cycle(NOW)
        h.cycle(NOW + 300)  # within 10 min: no new sample
        self.assertEqual(len(h.hub_calls), 1)
        api = h.load("api.json")
        self.assertEqual((api["hub"]["spend"], api["hub"]["remaining"], api["hub"]["key"], api["hub"]["age_s"]),
                         (100.0, 900.0, "NVAPI_KEY_NEW", 300))  # the sample of 5 min ago, reused
        h.cycle(NOW + 3600)
        self.assertEqual(len(h.hub_calls), 2)
        self.assertTrue(str(h.hub_calls[0]).endswith("tools/hub_budget.py"))
        api = h.load("api.json")
        self.assertEqual(len(api["history"]), 2)
        self.assertEqual((api["burn"]["per_hour_1h"], api["burn"]["window_min_1h"]), (60.0, 60.0))
        self.assertIsNotNone(api["burn"]["eta_exhaust_iso"])
        self.assertEqual(api["burn"]["eta_exhaust"], round(NOW + 3600 + 840 / 60 * 3600))
        h.cycle(NOW + 3700, hub=False)
        self.assertEqual(len(h.hub_calls), 2)
        api = h.load("api.json")  # a later cycle reuses the sample: the ETA still counts from the sample's time
        self.assertEqual((api["hub"]["age_s"], api["burn"]["eta_exhaust"]), (100, round(NOW + 3600 + 840 / 60 * 3600)))

    def test_failed_sample_counts_as_attempt(self):
        h = Harness(self.addCleanup)

        def broken(src):
            h.hub_calls.append(1)
            raise OSError("no python")
        sources = collect.Sources(root=h.fx.root, three_way=h.fx.three_way)
        summaries = [collect.run_cycle(h.data, h.state, sources=sources, clock=lambda now=now: now, slurm=h.slurm,
                                       scrape=h.scrape, hub=(broken, h.hub_finish)) for now in (NOW, NOW + 60)]
        self.assertEqual(len(h.hub_calls), 1)  # the failed attempt still blocks a retry for 10 min
        self.assertTrue(any(error.startswith("hub:") for error in summaries[0]["errors"]))
        self.assertEqual(summaries[1]["errors"], [])

    def test_burn(self):
        history = [{"t": NOW, "spend": 10.0, "max_budget": 100.0},
                   {"t": NOW + 5 * 3600, "spend": 60.0, "max_budget": 100.0},
                   {"t": NOW + 6 * 3600, "spend": 70.0, "max_budget": 100.0}]
        burn = collect._burn(history, 30.0, NOW + 6 * 3600)
        self.assertEqual((burn["per_hour_1h"], burn["per_hour_6h"]), (10.0, 10.0))
        self.assertEqual((burn["window_min_1h"], burn["window_min_6h"]), (60.0, 360.0))
        self.assertEqual(burn["eta_exhaust"], round(NOW + 9 * 3600))
        # the remaining budget is projected from the time of the sample that measured it
        self.assertEqual(collect._burn(history, 30.0, NOW + 6 * 3600 - 300)["eta_exhaust"], round(NOW + 9 * 3600 - 300))
        # a window holding a single sample has no rate: never extrapolated from an older sample
        sparse = [{"t": NOW, "spend": 10.0, "max_budget": 100.0},
                  {"t": NOW + 3 * 3600, "spend": 40.0, "max_budget": 100.0}]
        burn = collect._burn(sparse, 60.0, NOW + 3 * 3600)
        self.assertEqual((burn["per_hour_1h"], burn["window_min_1h"]), (None, None))
        self.assertEqual((burn["per_hour_6h"], burn["window_min_6h"]), (10.0, 180.0))  # measured over 3 h
        self.assertEqual(burn["eta_exhaust"], round(NOW + 9 * 3600))  # at the 6 h rate
        uneven = [{"t": NOW, "spend": 0.0, "max_budget": 100.0}, {"t": NOW + 1500, "spend": 5.0, "max_budget": 100.0},
                  {"t": NOW + 3300, "spend": 20.0, "max_budget": 100.0}]
        burn = collect._burn(uneven, 80.0, NOW + 3300)
        self.assertEqual((burn["per_hour_1h"], burn["window_min_1h"]), (round(20 / 3300 * 3600, 2), 55.0))
        stale = [{"t": NOW, "spend": 1.0, "max_budget": 100.0},
                 {"t": NOW + 7 * 3600, "spend": 8.0, "max_budget": 100.0}]
        burn = collect._burn(stale, 92.0, NOW + 7 * 3600)  # the earlier sample lies outside both windows
        self.assertEqual([burn[key] for key in ("per_hour_1h", "per_hour_6h", "eta_exhaust")], [None, None, None])
        self.assertIsNone(collect._burn([], None, None)["per_hour_6h"])
        # a new key (spend restarts): only the newest stretch counts
        history.append({"t": NOW + 7 * 3600, "spend": 1.0, "max_budget": 500.0})
        self.assertIsNone(collect._burn(history, 499.0, NOW + 7 * 3600)["per_hour_1h"])

    def test_burn_ignores_small_counter_dips(self):
        history = [{"t": NOW + i * 600, "spend": spend, "max_budget": 10000.0}
                   for i, spend in enumerate((1549.0, 1600.0, 1650.0, 1788.0, 1785.0, 1924.0))]
        burn = collect._burn(history, 8076.0, NOW + 3000)
        self.assertEqual(burn["window_min_1h"], 50.0)  # the 3-dollar dip does not start a new stretch
        self.assertEqual(burn["per_hour_1h"], round((1924.0 - 1549.0) / 3000 * 3600, 2))

    def test_burn_projects_at_the_higher_rate(self):
        """The ETA uses the HIGHER of the 1 h and 6 h rates (CONTRACT v1.1 integration notes: conservative while probes
        burst; the page names the one used), whichever window it comes from."""
        burst, calm = burn_histories()
        burn = collect._burn(burst, 700.0, NOW + 6 * 3600)  # 20 $/h for 5 h, then 50 $ per half hour
        self.assertEqual((burn["per_hour_1h"], burn["per_hour_6h"]), (100.0, 33.33))
        self.assertEqual(burn["eta_exhaust"], round(NOW + 6 * 3600 + 700.0 / 100.0 * 3600))  # at the 1 h rate
        burn = collect._burn(calm, 750.0, NOW + 6 * 3600)  # 100 $/h for 1 h, then 10 $/h
        self.assertEqual((burn["per_hour_1h"], burn["per_hour_6h"]), (10.0, 25.0))
        self.assertEqual(burn["eta_exhaust"], round(NOW + 6 * 3600 + 750.0 / 25.0 * 3600))  # at the 6 h rate

    def test_trim(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        collect.allow_reads(tmp)  # a Cycle registers its data dir; this test calls trim_jsonl directly
        path = jsonl(tmp / "h.jsonl", [{"t": NOW - 10 * 86400}, {"t": NOW - 5 * 86400}, {"t": NOW}])
        rows = collect.trim_jsonl(path, NOW - 7 * 86400)
        self.assertEqual([row["t"] for row in rows], [NOW - 5 * 86400, NOW])
        self.assertEqual(len(path.read_text().splitlines()), 2)
        self.assertEqual(oct(os.stat(path).st_mode & 0o777), "0o600")  # rewritten private, like every data/ file


class RobustTest(unittest.TestCase):
    def test_broken_sources(self):
        h = Harness(self.addCleanup)
        os.unlink(h.fx.root / "student/state/status.json")
        write(h.fx.build / "round-0/cand-c01/metrics.json", "{not json")
        shutil.rmtree(h.fx.root / "data/targets")
        h.fx.three_way.unlink()

        def broken_slurm(ids):
            raise RuntimeError("slurmctld down")
        sources = collect.Sources(root=h.fx.root, three_way=h.fx.three_way)
        summary = collect.run_cycle(h.data, h.state, sources=sources, clock=lambda: NOW, slurm=broken_slurm,
                                    scrape=h.scrape, hub=False)
        self.assertFalse(summary["ok"])
        sources_with_errors = {error.split(":")[0] for error in summary["errors"]}
        self.assertTrue({"slurm", "fleet", "targets", "three-way"} <= sources_with_errors, summary["errors"])
        for name in ("index.json", "fleet.json", "api.json", f"task/{TASK}.json"):
            self.assertTrue((h.data / name).exists(), name)
        index = h.load("index.json")
        self.assertEqual(index["errors"], summary["errors"])
        node = {n["id"]: n for n in h.load(f"task/{TASK}.json")["nodes"]}["cand-c01"]
        self.assertIsNone(node["score"])  # its metrics.json is unreadable: not scored (its probe looks live)
        self.assertEqual(node["status"], "probing")
        leftovers = [p for p in h.data.rglob("*") if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_concurrent_calls_serialize(self):
        h = Harness(self.addCleanup)
        results = []
        threads = [threading.Thread(target=lambda: results.append(h.cycle(hub=False))) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual([r["errors"] for r in results], [[], [], []])
        self.assertEqual(h.load("index.json")["errors"], [])

    def test_state_lock_between_processes(self):
        h = Harness(self.addCleanup)
        h.state.mkdir(mode=0o700)
        import subprocess
        holder = subprocess.Popen([sys.executable, "-I", "-B", "-c", "import fcntl, sys, time; "
                                   "f = open(sys.argv[1], 'w'); fcntl.flock(f, fcntl.LOCK_EX); "
                                   "print('locked', flush=True); time.sleep(30)", str(h.state / "collector.lock")],
                                  stdout=subprocess.PIPE, text=True)
        self.addCleanup(holder.kill)
        self.addCleanup(holder.stdout.close)
        self.assertEqual(holder.stdout.readline().strip(), "locked")
        saved = collect.LOCK_WAIT_S
        collect.LOCK_WAIT_S = 0
        try:
            summary = h.cycle(hub=False)
        finally:
            collect.LOCK_WAIT_S = saved
        self.assertFalse(summary["ok"])
        self.assertIn("collector.lock", summary["errors"][0])
        self.assertFalse((h.data / "index.json").exists())  # nothing was written by the skipped cycle
        holder.kill()
        holder.wait()
        self.assertTrue(h.cycle(hub=False)["ok"])

    def test_stale_task_doc_removed(self):
        h = Harness(self.addCleanup)
        write(h.data / "task/old-task.json", {})
        h.cycle(hub=False)
        self.assertFalse((h.data / "task/old-task.json").exists())
        self.assertTrue((h.data / f"task/{TASK}.json").exists())

    def test_cli(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            collect.main(["bogus"])

    def test_malformed_metrics(self):
        h = Harness(self.addCleanup)
        bad = metrics("cand-c01-r1b0-r2b0", valid=[None, "x"], similarity="high", student_rate="?")
        write(h.fx.build / "round-2/cand-c01-r1b0-r2b0/metrics.json", bad)
        summary = h.cycle(hub=False)
        self.assertEqual(summary["errors"], [])
        node = {n["id"]: n for n in h.load(f"task/{TASK}.json")["nodes"]}["cand-c01-r1b0-r2b0"]
        self.assertEqual(node["decision"], "discard")
        self.assertIn("fewer than 3 valid rollouts", node["decision_reason"])

    def test_two_runs_same_task(self):
        h = Harness(self.addCleanup)
        run2 = h.fx.runs / "official2-test"
        jsonl(run2 / "launch.jsonl", [{"submitted_at": "2026-10-18T09:00:00-07:00", "job_id": "456", "run":
                                       "official2-test", "task": TASK, "extra": ""}])
        write(run2 / f"build/{TASK}/other__repo/env/stage.json", {"ok": True})
        h.cycle(hub=False)
        index = h.load("index.json")
        self.assertEqual(sorted(run["run"] for run in index["runs"]), [RUN, "official2-test"])
        self.assertEqual({t["task_id"]: t["run"] for t in index["tasks"]}[TASK], "official2-test")
        doc = h.load(f"task/{TASK}.json")
        self.assertEqual((doc["run"], doc["repo"], doc["max_rounds"]), ("official2-test", "other/repo", 3))
        self.assertEqual(doc["nodes"], [])


class BatchTest(unittest.TestCase):
    """CONTRACT "v1.6" (hengyuf 2026-10-22: add Baihe's batch to the page, do not audit it): a batch from another tree
    (BatchFixture) next to the official build. Its runs are the ones that built a task of outputs/tasks.jsonl, built by
    the official code from their own trees (--rounds 1), never audited, never polled, their logs and usage never read;
    every task of the batch is an accepted in_batch node; the official output stays exactly what it was."""

    @classmethod
    def setUpClass(cls):
        cls.h = Harness(cls.addClassCleanup, batch=True)
        cls.summary = cls.h.cycle(hub=False)
        cls.index = cls.h.load("index.json")
        cls.runs = {run["run"]: run for run in cls.index["runs"] if "source" in run}
        cls.docs = {name: cls.h.load(run["detail"]) for name, run in cls.runs.items()}
        cls.nodes = {name: {node["id"]: node for node in doc["nodes"]} for name, doc in cls.docs.items()}

    def decisions(self, run):
        return {node_id: node["decision"] for node_id, node in self.nodes[run].items()}

    def test_cycle_and_official_output_unchanged(self):
        self.assertEqual(self.summary["errors"], [])
        self.assertEqual(self.summary["batches"][BATCH], {"runs": 3, "nodes": 13, "tasks": 6, "in_batch": 6,
                                                         "accepted": 6, "unmatched": 0,
                                                         "seconds": self.summary["batches"][BATCH]["seconds"]})
        plain = Harness(self.addCleanup)  # the same official tree without the batch
        plain.cycle(hub=False)
        index = plain.load("index.json")
        self.assertEqual([run for run in self.index["runs"] if "source" not in run], index["runs"])
        self.assertEqual([{k: v for k, v in task.items() if k != "batch_runs"} for task in self.index["tasks"]],
                         index["tasks"])
        self.assertNotIn("batches", index)  # a tree without batches writes no batch keys at all

        def text(h, name):  # the official task file, its fixture paths aside
            return (h.data / name).read_text().replace(str(h.tmp), "<tmp>")
        self.assertEqual(text(self.h, f"task/{TASK}.json"), text(plain, f"task/{TASK}.json"))
        self.assertEqual(sorted(p.name for p in (self.h.data / "task").iterdir()), [f"{TASK}.json"])
        self.assertFalse((plain.data / "run").exists())

    def test_index_entries(self):
        self.assertEqual(list(self.runs), ["pilotA-task-b", "pilotB-task-b", "q1-task-c"])
        run = self.runs["pilotA-task-b"]
        self.assertEqual({key: run[key] for key in ("task_id", "repo", "job_id", "slurm_state", "partition", "started_at",
                                                    "ended_at", "status", "stage", "rounds_started", "accepted",
                                                    "tasks_built", "tasks_scored", "audit", "source", "source_label",
                                                    "batch", "detail", "detail_generated_at")},
                         {"task_id": "task-b", "repo": "o/a", "job_id": "201", "slurm_state": None,
                          "partition": "cpu_dataprocessing", "started_at": None, "ended_at": None, "status": "finished",
                          "stage": "finished (3 accepted)", "rounds_started": 2, "accepted": 3, "tasks_built": 7,
                          "tasks_scored": 6, "audit": {"status": "not_audited"}, "source": BATCH,
                          "source_label": "Testy · fork", "batch": 3, "detail": f"run/{BATCH}/pilotA-task-b.json",
                          "detail_generated_at": NOW})
        self.assertEqual(run["root"], str(self.h.bx.runs / "pilotA-task-b"))
        self.assertEqual((self.runs["q1-task-c"]["status"], self.runs["q1-task-c"]["stage"]),
                         ("stopped", "stopped: round-1 unfinished (1/3 tasks scored, 1 probe unfinished)"))
        tasks = {task["task_id"]: task for task in self.index["tasks"]}
        self.assertEqual(tasks["task-b"]["batch_runs"], [{"source": BATCH, "run": "pilotA-task-b"},
                                                         {"source": BATCH, "run": "pilotB-task-b"}])  # oldest first
        self.assertEqual((tasks["task-c"]["batch_runs"], tasks["task-c"]["run"]),
                         ([{"source": BATCH, "run": "q1-task-c"}], None))
        self.assertNotIn("batch_runs", tasks[TASK])
        self.assertEqual(tasks[TASK]["run"], RUN)
        self.assertEqual(self.index["batches"], [{
            "source": BATCH, "name": "Testy", "label": "Testy · fork", "root": str(self.h.bx.root),
            "outputs": str(self.h.bx.outputs), "tasks": 6, "runs": 3, "targets": 2, "in_batch": 6, "accepted": 6,
            "unmatched": [], "not_in_batch": [], "audit": {"status": "not_audited"}}])

    def test_documents(self):
        doc = self.docs["pilotA-task-b"]
        self.assertEqual((doc["generated_at"], doc["source"], doc["batch"], doc["audit"], doc["max_rounds"],
                          doc["branches"], doc["repo"], doc["root"], doc["outputs"]),
                         (NOW, BATCH, 3, {"status": "not_audited"}, 1, 5, "o/a",
                          str(self.h.bx.runs / "pilotA-task-b"), str(self.h.bx.outputs)))
        self.assertTrue(doc["build_root"].endswith("/synth/runs/pilotA-task-b/build/task-b/o__a"))
        job = doc["job"]  # launch.jsonl only: no Slurm, no log (its Slurm logs are outside the read roots)
        self.assertEqual({key: job[key] for key in ("job_id", "slurm_state", "partition", "node", "elapsed", "log_path",
                                                    "log_tail", "status", "jobs")},
                         {"job_id": "201", "slurm_state": None, "partition": "cpu_dataprocessing", "node": None,
                          "elapsed": None, "log_path": None, "log_tail": [], "status": "finished", "jobs": ["201"]})
        self.assertEqual([pattern["name"] for pattern in doc["failure_analysis"]["patterns"]], BatchFixture.PATTERNS)
        self.assertEqual(len(doc["plan"]["candidates"]), 3)

    def test_decisions(self):
        # --rounds 1: round 1 is the last; a valid round-1 task that was not accepted is end
        self.assertEqual(self.decisions("pilotA-task-b"), {
            "cand-c01": "accepted_evolve", "cand-c02": "evolve", "cand-c03": "pending", "cand-c01-r1b0": "accepted",
            "cand-c01-r1b1": "end", "cand-c02-r1b0": "accepted", "cand-c02-r1b1": "discard"})
        self.assertEqual(self.decisions("pilotB-task-b"), {"cand-c01": "accepted_evolve", "cand-c01-r1b0": "end"})
        self.assertEqual(self.decisions("q1-task-c"), {"cand-c01": "accepted_evolve", "cand-c01-r1b0": "accepted",
                                                       "cand-c01-r1b1": "pending", "cand-c01-r1b2": "pending"})
        a, q1 = self.nodes["pilotA-task-b"], self.nodes["q1-task-c"]
        self.assertEqual((a["cand-c03"]["status"], a["cand-c03"]["decision_reason"]),
                         ("built", "its probe round-0/probe-r0 never finished: the run finished before scoring it"))
        self.assertEqual(q1["cand-c01-r1b1"]["decision_reason"],
                         "its probe round-1/probe-c01r1b1 never finished: the run stopped before scoring it")
        self.assertEqual(q1["cand-c01-r1b2"]["decision_reason"], "built, never probed: the run stopped before probing it")
        self.assertEqual([a[f"cand-c0{i}-r1b{j}"]["focus_pattern"] for i, j in ((1, 0), (1, 1))], BatchFixture.PATTERNS)
        self.assertFalse(any(node["status"] in ("abandoned", "probing") for nodes in self.nodes.values()
                             for node in nodes.values()))  # an ended run: no live and no superseded probe

    def test_rounds(self):
        rounds = self.docs["pilotA-task-b"]["rounds"]
        self.assertEqual([(r["status"], r["unfinished_probes"], r["abandoned_probes"]) for r in rounds],
                         [("unfinished", ["round-0/probe-r0"], []), ("done", [], [])])
        self.assertEqual((rounds[0]["probe"]["slots_done"], rounds[0]["probe"]["slots_total"]), (20, 30))  # 2 scored
        rounds = self.docs["q1-task-c"]["rounds"]
        self.assertEqual([r["status"] for r in rounds], ["done", "unfinished"])
        self.assertEqual(rounds[1]["unfinished_probes"], ["round-1/probe-c01r1b1"])
        self.assertEqual(rounds[1]["probe"]["by_model"], {"kimi": [5, 10], "sc14": [5, 10]})
        self.assertFalse(rounds[1]["probe"]["live"])

    def test_never_audited_and_in_batch(self):
        for name, nodes in self.nodes.items():
            for node in nodes.values():
                self.assertEqual((node["failure_match"], node["audit_eligible"], node["audit"]),
                                 (None, False, {"status": "not_audited"}), (name, node["id"]))
                self.assertIs(node["in_batch"], node["build"]["task_id"] in self.h.bx.batch_task_ids(name))
        in_batch = sorted(node["build"]["task_id"] for nodes in self.nodes.values() for node in nodes.values()
                          if node["in_batch"])
        self.assertEqual(in_batch, sorted(self.h.bx.batch_task_ids()))
        self.assertTrue(all(node["decision"] in collect.ACCEPTED for nodes in self.nodes.values()
                            for node in nodes.values() if node["in_batch"]))
        # the planted audit of an accepted batch task is never read; the official audits are untouched
        self.assertIsNone(self.nodes["pilotA-task-b"]["cand-c01"]["failure_match"])
        official = {n["id"]: n for n in self.h.load(f"task/{TASK}.json")["nodes"]}
        self.assertEqual(official["cand-c03"]["failure_match"]["verdict"], "strong")
        self.assertEqual(self.index["runs"][0]["audit"]["eligible"], 8)  # the official counts alone
        self.assertNotIn("in_batch", official["cand-c03"])
        self.assertNotIn("audit", official["cand-c03"])

    def test_nothing_else_is_read_or_polled(self):
        self.assertEqual(self.h.slurm_calls, [["123"]])  # only the official job: a batch's jobs are never polled
        self.assertEqual([row["run"] for row in self.h.load("api.json")["by_run"]], [RUN])  # no usage of the batch
        self.assertEqual(self.h.load("fleet.json")["router"]["scope"], "official-run probes")
        roots = collect._READ_ROOTS
        bx = self.h.bx
        for path in [bx.outputs] + [bx.runs / name for name in bx.RUNS]:
            self.assertIn(os.path.realpath(path), roots)
        for path in (bx.root, bx.runs, bx.runs / "smoke9", bx.runs / "slurm", bx.root / "secrets"):
            self.assertNotIn(os.path.realpath(path), roots)
        self.assertIsNone(collect.read_text(bx.runs / "smoke9/launch.jsonl"))  # outside the read roots
        for path in self.h.data.rglob("*.json"):
            self.assertNotIn(SECRET_KEY, path.read_text(), path)  # the planted Slurm log line never surfaces

    def test_unchanged_runs_are_reused(self):
        h = Harness(self.addCleanup, batch=True)
        h.cycle(hub=False)
        files = {path.name: collect.file_signature(path) for path in (h.data / f"run/{BATCH}").iterdir()}
        summary = h.cycle(NOW + 60, hub=False)
        self.assertEqual(summary["errors"], [])
        self.assertFalse(any(name.startswith("run/") for name in summary["written_bytes"]))  # nothing rewritten
        self.assertEqual({path.name: collect.file_signature(path) for path in (h.data / f"run/{BATCH}").iterdir()},
                         files)
        index = h.load("index.json")
        self.assertEqual((index["generated_at"], {run["detail_generated_at"] for run in index["runs"]
                                                  if "source" in run}), (NOW + 60, {NOW}))
        # a change in a run's round dir: that run is rebuilt and its file rewritten with this cycle's stamp
        round_dir = h.bx.runs / "pilotB-task-b/build/task-b/o__b/round-1"
        write(round_dir / "seeds.json", {"valid": [], "invalid": {"cand-c01-r1b0": "verifier produced no grade for 1 "
                                                                                   "rollout(s): kimi-r0"}})
        os.utime(round_dir, (NOW, NOW))  # mtimes have 1 s steps: make the change visible to the fingerprint
        summary = h.cycle(NOW + 120, hub=False)
        self.assertEqual(sorted(name for name in summary["written_bytes"] if name.startswith("run/")),
                         [f"run/{BATCH}/pilotB-task-b.json"])
        doc = h.load(f"run/{BATCH}/pilotB-task-b.json")
        self.assertEqual((doc["generated_at"], {n["id"]: n["decision"] for n in doc["nodes"]}["cand-c01-r1b0"]),
                         (NOW + 120, "discard"))
        self.assertEqual({run["run"]: run["detail_generated_at"] for run in h.load("index.json")["runs"]
                          if "source" in run}, {"pilotA-task-b": NOW, "pilotB-task-b": NOW + 120, "q1-task-c": NOW})
        # past BATCH_REFRESH_S every run is rebuilt again, at most BATCH_REFRESHES per cycle; same content: no rewrite
        later = NOW + collect.BATCH_REFRESH_S + 200
        summary = h.cycle(later, hub=False)
        self.assertFalse(any(name.startswith("run/") for name in summary["written_bytes"]))
        rebuilt = {key[1]: entry["built_at"] for key, entry in collect._BATCH_CACHE.items() if key[0] == str(h.data)}
        self.assertEqual(sorted(at for at in rebuilt.values()), sorted([later, later, NOW]))
        # a file that is not the one written last (deleted here) is written again
        (h.data / f"run/{BATCH}/q1-task-c.json").unlink()
        summary = h.cycle(later + 60, hub=False)
        self.assertIn(f"run/{BATCH}/q1-task-c.json", summary["written_bytes"])
        self.assertEqual(h.load(f"run/{BATCH}/q1-task-c.json")["generated_at"], NOW)  # content unchanged

    def test_mismatches_are_reported(self):
        h = Harness(self.addCleanup, batch=True)
        bx = h.bx
        # a batch task that is not accepted, one without provenance.json, and an accepted task left out of the batch
        rows = [json.loads(line) for line in (bx.outputs / "tasks.jsonl").read_text().splitlines()]
        rows = [row for row in rows if row["task_id"] != bx.tid("pilotA-task-b", "cand-c02-r1b0")]
        not_accepted = bx.tid("pilotA-task-b", "cand-c01-r1b1")
        task_dir = bx.runs / "pilotA-task-b/build/task-b/o__a/cand-c01-r1b1/task" / not_accepted
        write(bx.outputs / not_accepted / "provenance.json", {"task_id": not_accepted, "task_dir": str(task_dir)})
        jsonl(bx.outputs / "tasks.jsonl", rows + [{"task_id": not_accepted}, {"task_id": "syn-orphan-cand-c09"},
                                                  {"task_id": "../escape"}])
        summary = h.cycle(hub=False)
        errors = sorted(error for error in summary["errors"] if error.startswith(f"batch {BATCH}:"))
        self.assertEqual(errors, sorted([
            f"batch {BATCH}: outputs task {not_accepted}: pilotA-task-b cand-c01-r1b1 is end, not accepted",
            f"batch {BATCH}: outputs task syn-orphan-cand-c09: outputs/<task_id>/provenance.json missing or unreadable",
            f"batch {BATCH}: outputs task ../escape: not a task id",
            f"batch {BATCH}: accepted node pilotA-task-b cand-c02-r1b0 ({bx.tid('pilotA-task-b', 'cand-c02-r1b0')}) is "
            "not in outputs/tasks.jsonl"]))
        batch = h.load("index.json")["batches"][0]
        self.assertEqual((batch["tasks"], batch["in_batch"], batch["accepted"], len(batch["unmatched"]),
                          len(batch["not_in_batch"])), (8, 6, 6, 3, 1))
        runs = {run["run"]: run for run in h.load("index.json")["runs"] if "source" in run}
        self.assertEqual({name: run["batch"] for name, run in runs.items()},
                         {"pilotA-task-b": 3, "pilotB-task-b": 1, "q1-task-c": 2})
        nodes = {n["id"]: n for n in h.load(runs["pilotA-task-b"]["detail"])["nodes"]}
        self.assertTrue(nodes["cand-c01-r1b1"]["in_batch"])  # flagged as in the batch, and reported: not accepted

    def test_symlinked_run_dir_is_refused(self):
        """A run dir that leaves synth/runs (a symlink planted there, e.g. into the tree's secrets/) never joins the read
        roots: its batch tasks are reported, the other runs are shown."""
        h = Harness(self.addCleanup, batch=True)
        bx = h.bx
        target = bx.runs / "q1-task-c"
        os.rename(target, bx.root / "secrets/q1-task-c")
        target.symlink_to(bx.root / "secrets/q1-task-c")
        summary = h.cycle(hub=False)
        self.assertEqual(sorted(error for error in summary["errors"]), sorted(
            f"batch {BATCH}: outputs task {task_id}: no run dir synth/runs/q1-task-c"
            for task_id in bx.batch_task_ids("q1-task-c")))
        self.assertEqual(sorted(run["run"] for run in h.load("index.json")["runs"] if "source" in run),
                         ["pilotA-task-b", "pilotB-task-b"])
        self.assertNotIn(os.path.realpath(bx.root / "secrets/q1-task-c"), collect._READ_ROOTS)
        self.assertNotIn(os.path.realpath(bx.root / "secrets"), collect._READ_ROOTS)

    def test_a_broken_batch_never_stops_the_cycle(self):
        """A bug or a broken source in the batch path costs the batch, never the official output."""
        h = Harness(self.addCleanup, batch=True)
        plain = Harness(self.addCleanup)
        plain.cycle(hub=False)

        def boom(*args):
            raise RuntimeError("batch bug")
        for method in ("batch_doc", "batch_index", "batch_docs", "write_batch_docs", "read_batch"):
            with unittest.mock.patch.object(collect.Cycle, method, boom):
                collect._BATCH_CACHE.clear()
                summary = h.cycle(hub=False)
            self.assertTrue(summary["errors"] and all(error.startswith("batch") for error in summary["errors"]),
                            (method, summary["errors"]))
            index = h.load("index.json")
            self.assertEqual([run for run in index["runs"] if "source" not in run], plain.load("index.json")["runs"],
                             method)
            self.assertTrue((h.data / f"task/{TASK}.json").exists(), method)
        self.assertEqual(h.cycle(hub=False)["errors"], [])  # and the next cycle is whole again

    def test_a_run_built_with_errors_is_built_again(self):
        """A source of a batch run that fails (its failure analysis here) is reported, and that run is not reused next
        cycle (unchanged fingerprint or not): the next cycle builds it again, whole."""
        h = Harness(self.addCleanup, batch=True)
        real = collect.Cycle.failure_analysis

        def flaky(cycle, directory):
            if "pilotB-task-b" in str(directory):
                raise OSError("flaky")
            return real(cycle, directory)
        with unittest.mock.patch.object(collect.Cycle, "failure_analysis", flaky):
            summary = h.cycle(hub=False)
        self.assertEqual(summary["errors"], [f"batch {BATCH} run pilotB-task-b failure analysis: OSError: flaky"])
        self.assertIsNone(h.load(f"run/{BATCH}/pilotB-task-b.json")["failure_analysis"])
        summary = h.cycle(NOW + 60, hub=False)
        self.assertEqual(summary["errors"], [])
        self.assertEqual(sorted(name for name in summary["written_bytes"] if name.startswith("run/")),
                         [f"run/{BATCH}/pilotB-task-b.json"])
        doc = h.load(f"run/{BATCH}/pilotB-task-b.json")
        self.assertEqual(([p["name"] for p in doc["failure_analysis"]["patterns"]], doc["generated_at"]),
                         (BatchFixture.PATTERNS, NOW + 60))

    def test_broken_batch_and_stub_target(self):
        h = Harness(self.addCleanup, batch=True)
        bx = h.bx
        bx.RUNS = dict(bx.RUNS, **{"q2-task-z": ("task-z", "o__z", "2026-10-22T01:00:00-07:00", "204", True)})
        z = bx.run("q2-task-z")  # a target the DeepSWE targets do not list: a stub task entry
        bx.built(z, "cand-c01")
        bx.scored(z, 0, "cand-c01", accept=True, similarity=7)
        bx.probe(z, 0, "probe-c01", ["cand-c01"], 100)
        bx.write_outputs()
        self.assertEqual(h.cycle(hub=False)["errors"], [])
        tasks = {task["task_id"]: task for task in h.load("index.json")["tasks"]}
        self.assertEqual((tasks["task-z"]["run"], tasks["task-z"]["batch_runs"], tasks["task-z"]["accuracy"],
                          tasks["task-z"]["eligible"]),
                         (None, [{"source": BATCH, "run": "q2-task-z"}], {"kimi": None, "super_vl": None, "sc14": None},
                          False))
        # no outputs/tasks.jsonl: an error, no batch runs, their files removed; the official output goes on
        (bx.outputs / "tasks.jsonl").unlink()
        summary = h.cycle(NOW + 60, hub=False)
        self.assertEqual([error.split(":")[0] for error in summary["errors"]], [f"batch {BATCH}"])
        index = h.load("index.json")
        self.assertEqual((index["batches"], [run["run"] for run in index["runs"]]), ([], [RUN]))
        self.assertEqual(list((h.data / f"run/{BATCH}").iterdir()), [])
        self.assertTrue((h.data / f"task/{TASK}.json").exists())


def explorer_rows(ex, kind="tasks"):
    """explorer.json rows as dicts (column name -> value)."""
    columns = ex["task_columns" if kind == "tasks" else "trajectory_columns"]
    return [dict(zip(columns, row)) for row in ex[kind]]


class ExplorerTest(unittest.TestCase):
    """v1.7 (hengyuf 2026-10-22: filter the synthetic tasks and their trajectories): data/explorer.json holds one row per
    built synthetic task of the task and run files and one per rollout their probes list, made from those very nodes;
    the batch is never audited; the file is rewritten only when its content changes; nothing else changes."""

    @classmethod
    def setUpClass(cls):
        cls.h = Harness(cls.addClassCleanup, batch=True)
        cls.summary = cls.h.cycle(hub=False)
        cls.ex = cls.h.load("explorer.json")
        cls.index = cls.h.load("index.json")
        cls.tasks, cls.traj = explorer_rows(cls.ex), explorer_rows(cls.ex, "trajectories")
        cls.docs = [cls.h.load(f"task/{TASK}.json")] + [cls.h.load(run["detail"]) for run in cls.index["runs"]
                                                        if "source" in run]

    def row(self, run, node):
        index = next(i for i, r in enumerate(self.ex["runs"]) if r["run"] == run)
        return next(row for row in self.tasks if row["run"] == index and row["node"] == node)

    def trajectories(self, run, node):
        row = self.row(run, node)
        position = self.tasks.index(row)
        return [t for t in self.traj if t["task"] == position]

    def test_cycle_index_and_summary(self):
        self.assertEqual(self.summary["errors"], [])
        path = self.h.data / "explorer.json"
        self.assertEqual(self.index["explorer"], {"path": "explorer.json", "generated_at": NOW,
                                                  "bytes": path.stat().st_size, "tasks": 28, "trajectories": 214})
        self.assertEqual(oct(path.stat().st_mode & 0o777), "0o600")
        self.assertEqual(self.summary["explorer"], {"bytes": path.stat().st_size, "written": True,
                                                    "tasks": {"hengyuf": 15, BATCH: 13},
                                                    "trajectories": {"hengyuf": 114, BATCH: 100},
                                                    "runs": {"hengyuf": 1, BATCH: 3},
                                                    "seconds": self.summary["explorer"]["seconds"]})
        self.assertEqual(self.summary["written_bytes"]["explorer.json"], path.stat().st_size)
        self.assertEqual((self.ex["schema"], self.ex["generated_at"], self.ex["task_columns"],
                          self.ex["trajectory_columns"]),
                         (1, NOW, list(collect.EXPLORER_TASK_COLUMNS), list(collect.EXPLORER_TRAJECTORY_COLUMNS)))
        self.assertEqual(self.ex["sources"], [collect.OFFICIAL_SOURCE, {"source": BATCH, "name": "Testy",
                                                                        "label": "Testy · fork", "audited": False}])

    def test_rows_are_the_documents_nodes(self):
        """Every built node (build.task_id) of every task / run file, in document order; every rollout of its probe."""
        want, want_traj = [], []
        for doc in self.docs:
            for node in doc["nodes"]:
                if node["build"]["task_id"]:
                    want.append((doc["run"], node["id"], node["build"]["task_id"], node["title"], node["decision"]))
                    want_traj += [(doc["run"], node["id"], r["slot"], r["status"]) for r in node["probe"]["rollouts"]] \
                        if node["probe"] else []
        runs = [r["run"] for r in self.ex["runs"]]
        self.assertEqual([(runs[r["run"]], r["node"], r["task_id"], r["title"], r["decision"]) for r in self.tasks], want)
        self.assertEqual([(runs[self.tasks[t["task"]]["run"]], self.tasks[t["task"]]["node"], t["slot"], t["status"])
                          for t in self.traj], want_traj)
        unbuilt = [node["id"] for node in self.docs[0]["nodes"] if not node["build"]["task_id"]]
        self.assertTrue(unbuilt and not {r["node"] for r in self.tasks if r["run"] == 0} & set(unbuilt))
        self.assertEqual(self.ex["totals"], {
            "hengyuf": {"runs": 1, "tasks": 15, "trajectories": 114, "accepted": 3, "audited": 3, "unmatched_patterns": 0,
                        "rates_differ": 3},
            BATCH: {"runs": 3, "tasks": 13, "trajectories": 100, "accepted": 6, "audited": 0, "unmatched_patterns": 0,
                    "rates_differ": 1}})

    def test_rates_differ(self):
        """v1.7 review (2026-10-22): task_builder's score() computes student_rate, teacher_clean_rate and valid from the
        rows it writes to rollouts/index.md, so a scored row whose metrics.json disagrees with them (a later regrade or
        a hand edit) keeps metrics.json's values and is flagged rates_differ; an unscored row never is."""
        flagged = [(self.ex["runs"][r["run"]]["run"], r["node"], r["student_rate"], r["valid"])
                   for r in self.tasks if r["rates_differ"]]
        self.assertEqual(flagged, [
            (RUN, "cand-c02", 0.6, [3, 4]),                    # valid 3 / 4; its rows grade 5 sc14 and 5 kimi
            (RUN, "cand-c01-r1b0-r2b4", 0.6, [2, 5]),          # valid 2 / 5; rows 5 / 5
            (RUN, "cand-c01-r1b0-r2b5", 0.6, [5, 5]),          # rows: sc14 2 of 4 graded (r0 infra), so 0.5 and [4, 5]
            ("pilotA-task-b", "cand-c02-r1b1", 0.4, [2, 5])])  # valid 2 / 5; rows 5 / 5
        self.assertTrue(all(r["rates_differ"] is False for r in self.tasks if not r["scored"]))
        self.assertTrue(all(type(r["rates_differ"]) is bool for r in self.tasks))
        # c01's clean 0.8 is 4 of its 5 kimi passes, one judged cheated: what the rows say, not flagged
        self.assertEqual((self.row(RUN, "cand-c01")["teacher_clean_rate"], self.row(RUN, "cand-c01")["rates_differ"]),
                         (0.8, False))
        # metrics.json rewritten after the probe: the row keeps the file's rates, flagged; nothing else changes
        h = Harness(self.addCleanup)
        path = h.fx.build / "round-0/cand-c03/metrics.json"
        data = json.loads(path.read_text())
        self.assertEqual((data["student_rate"], data["teacher_clean_rate"], data["valid"]), (0.2, 1.0, [5, 5]))
        write(path, dict(data, student_rate=0.8))
        h.cycle(hub=False)
        ex = h.load("explorer.json")
        row = next(r for r in explorer_rows(ex) if r["node"] == "cand-c03")
        self.assertEqual((row["student_rate"], row["teacher_clean_rate"], row["valid"], row["rates_differ"]),
                         (0.8, 1.0, [5, 5], True))
        self.assertEqual(ex["totals"]["hengyuf"]["rates_differ"], 4)
        write(path, dict(data, teacher_clean_rate=0.6))       # the clean rate alone: flagged too
        h.cycle(NOW + 60, hub=False)
        row = next(r for r in explorer_rows(h.load("explorer.json")) if r["node"] == "cand-c03")
        self.assertEqual((row["teacher_clean_rate"], row["rates_differ"]), (0.6, True))
        write(path, data)                                     # as score() wrote it: not flagged
        h.cycle(NOW + 120, hub=False)
        self.assertFalse(next(r for r in explorer_rows(h.load("explorer.json")) if r["node"] == "cand-c03")["rates_differ"])

    def test_runs_and_targets(self):
        b = f"run/{BATCH}"
        self.assertEqual([(r["run"], r["source"], r["deepswe_task"], r["detail"], r["link"], r["max_rounds"])
                          for r in self.ex["runs"]],
                         [(RUN, "hengyuf", TASK, f"task/{TASK}.json", f"#task={TASK}", 2),
                          ("pilotA-task-b", BATCH, "task-b", f"{b}/pilotA-task-b.json",
                           "#task=task-b&run=pilotA-task-b", 1),
                          ("pilotB-task-b", BATCH, "task-b", f"{b}/pilotB-task-b.json",
                           "#task=task-b&run=pilotB-task-b", 1),
                          ("q1-task-c", BATCH, "task-c", f"{b}/q1-task-c.json", "#task=task-c&run=q1-task-c", 1)])
        self.assertEqual([r["build_root"] for r in self.ex["runs"]], [doc["build_root"] for doc in self.docs])
        self.assertEqual([(t["deepswe_task"], t["runs"], [p["name"] for p in t["patterns"]]) for t in self.ex["targets"]],
                         [(TASK, [0], ["Scope creep", "Tests certify the bug"]),
                          ("task-b", [1, 2], BatchFixture.PATTERNS), ("task-c", [3], BatchFixture.PATTERNS)])
        self.assertEqual(self.ex["targets"][0]["patterns"][0], {"name": "Scope creep", "category": "requirement_scope"})

    def test_values_of_official_tasks(self):
        c03 = self.row(RUN, "cand-c03")
        self.assertEqual({k: c03[k] for k in ("round", "branch", "kind", "decision", "accepted", "withdrawn", "in_batch",
                                              "similarity", "student_rate", "teacher_clean_rate", "teacher_rate", "valid",
                                              "audit", "reproduced_yes", "reproduced_partial", "f2p", "p2p", "scored")},
                         {"round": 0, "branch": None, "kind": "mask", "decision": "accepted_evolve", "accepted": True,
                          "withdrawn": False, "in_batch": False, "similarity": 7, "student_rate": 0.2,
                          "teacher_clean_rate": 1.0, "teacher_rate": 1.0, "valid": [5, 5], "audit": "strong",
                          "reproduced_yes": [0], "reproduced_partial": [], "f2p": 3, "p2p": 4, "scored": True})
        self.assertEqual([self.row(RUN, n)["audit"] for n in ("cand-c01", "cand-c03-r1b0", "cand-c01-r1b0", "cand-c02",
                                                              "cand-c03-r1b1")],
                         ["weak", "partial", "not_audited", "not_eligible", "not_eligible"])
        self.assertEqual((self.row(RUN, "cand-c02")["reproduced_yes"], self.row(RUN, "cand-c01-r1b0")["reproduced_yes"]),
                         (None, None))
        # c01: metrics teacher_clean_rate 0.8 (one pass cheated) while the raw kimi rate over its rollouts is 1.0
        c01 = self.row(RUN, "cand-c01")
        self.assertEqual((c01["teacher_clean_rate"], c01["teacher_rate"], c01["similarity"]), (0.8, 1.0, None))
        # the live probe (cand-c03-r1b1): not scored, no rates, its graded slots so far as trajectories
        live = self.row(RUN, "cand-c03-r1b1")
        self.assertEqual((live["scored"], live["student_rate"], live["teacher_rate"], live["valid"], live["decision"]),
                         (False, None, None, None, "pending"))
        self.assertEqual(sorted((t["slot"], t["status"], t["tool_calls"])
                                for t in self.trajectories(RUN, "cand-c03-r1b1")),
                         [("kimi-r0", "infra", None), ("kimi-r1", "pass", None), ("sc14-r0", "pass", None),
                          ("sc14-r1", "fail", None)])
        abandoned = self.row(RUN, "cand-c01-r1b2")
        self.assertEqual((abandoned["decision"], abandoned["audit"], abandoned["scored"]),
                         ("abandoned", "not_eligible", False))

    def test_trajectory_values_and_clean_flags(self):
        rows = {t["slot"]: t for t in self.trajectories(RUN, "cand-c01")}
        self.assertEqual(rows["sc14-r0"], {"task": rows["sc14-r0"]["task"], "slot": "sc14-r0", "model": "sc14",
                                           "status": "pass", "reward": 1.0, "f2p": "3/3", "p2p": "4/4", "tool_calls": 12,
                                           "patch_chars": 345, "agent_error": False, "verifier_error": False,
                                           "clean": None})
        self.assertEqual({slot: t["clean"] for slot, t in rows.items() if t["model"] == "kimi"},
                         {"kimi-r0": True, "kimi-r1": False, "kimi-r2": True, "kimi-r3": True, "kimi-r4": True})
        r2b5 = {t["slot"]: t for t in self.trajectories(RUN, "cand-c01-r1b0-r2b5")}
        self.assertEqual((r2b5["sc14-r0"]["status"], r2b5["sc14-r0"]["verifier_error"], r2b5["sc14-r1"]["verifier_error"]),
                         ("infra", True, False))
        self.assertTrue(all(t["clean"] is None for t in self.traj if not (t["model"] == "kimi" and t["status"] == "pass")))

    def test_batch_tasks_are_never_audited(self):
        batch = [r for r in self.tasks if self.ex["runs"][r["run"]]["source"] == BATCH]
        self.assertEqual(len(batch), 13)
        self.assertEqual({(r["audit"], r["reproduced_yes"], r["reproduced_partial"]) for r in batch},
                         {("not_audited", None, None)})
        self.assertEqual(sorted(r["task_id"] for r in batch if r["in_batch"]), sorted(self.h.bx.batch_task_ids()))
        self.assertTrue(all(r["accepted"] for r in batch if r["in_batch"]))
        self.assertEqual(self.row("q1-task-c", "cand-c01-r1b1")["scored"], False)  # its probe never finished
        self.assertEqual(self.trajectories("q1-task-c", "cand-c01-r1b1"), [])

    def test_pattern_names_match_leniently(self):
        """An audit naming a target pattern with other case or spacing still counts; an unknown name is reported."""
        h = Harness(self.addCleanup)
        h.fx.audit("syn-cand-c03", verdict="strong", summary="x",
                   patterns=[{"name": "  scope   CREEP ", "reproduced": "partial", "slots": [], "evidence": ""},
                             {"name": "Tests certify the bug", "reproduced": "yes", "slots": [], "evidence": ""},
                             {"name": "a pattern of another target", "reproduced": "yes", "slots": [], "evidence": ""}])
        self.assertEqual(h.cycle(hub=False)["errors"], [])
        ex = h.load("explorer.json")
        c03 = next(r for r in explorer_rows(ex) if r["node"] == "cand-c03")
        self.assertEqual((c03["reproduced_yes"], c03["reproduced_partial"]), ([1], [0]))
        self.assertEqual(ex["totals"]["hengyuf"]["unmatched_patterns"], 1)

    def test_withdrawn_task(self):
        h = Harness(self.addCleanup)
        write(h.fx.runs / RUN / "withdrawn/syn-cand-c03-r1b0/task.toml", "x")
        self.assertEqual(h.cycle(hub=False)["errors"], [])
        row = next(r for r in explorer_rows(h.load("explorer.json")) if r["node"] == "cand-c03-r1b0")
        self.assertEqual((row["withdrawn"], row["accepted"], row["audit"], row["reproduced_yes"]),
                         (True, False, "not_eligible", None))  # its audit is attached, but no longer counted

    def test_rewritten_only_when_its_content_changes(self):
        h = Harness(self.addCleanup, batch=True)
        h.cycle(hub=False)
        path = h.data / "explorer.json"
        before = collect.file_signature(path)
        summary = h.cycle(NOW + 60, hub=False)  # nothing changed: neither the file nor its stamp
        self.assertEqual((summary["explorer"]["written"], collect.file_signature(path)), (False, before))
        self.assertNotIn("explorer.json", summary["written_bytes"])
        self.assertEqual((h.load("index.json")["explorer"]["generated_at"], h.load("explorer.json")["generated_at"]),
                         (NOW, NOW))
        h.fx.audit("syn-cand-c01-r1b1", verdict="none", summary="other causes")  # a new audit: a new row value
        summary = h.cycle(NOW + 120, hub=False)
        self.assertTrue(summary["explorer"]["written"])
        ex = h.load("explorer.json")
        self.assertEqual((ex["generated_at"], h.load("index.json")["explorer"]["generated_at"]), (NOW + 120, NOW + 120))
        self.assertEqual(next(r for r in explorer_rows(ex) if r["node"] == "cand-c01-r1b1")["audit"], "none")
        # a new collector process (no cache): the file on disk is read back once and kept when equal
        collect._EXPLORER_CACHE.clear()
        before = collect.file_signature(path)
        summary = h.cycle(NOW + 180, hub=False)
        self.assertEqual((summary["explorer"]["written"], collect.file_signature(path)), (False, before))
        self.assertEqual(h.load("index.json")["explorer"]["generated_at"], NOW + 120)
        # a file that is not the one written last (removed here) is written again, with the cycle's stamp
        path.unlink()
        summary = h.cycle(NOW + 240, hub=False)
        self.assertTrue(summary["explorer"]["written"] and path.exists())
        self.assertEqual(h.load("explorer.json")["generated_at"], NOW + 240)

    def test_a_broken_explorer_never_stops_the_cycle(self):
        h = Harness(self.addCleanup, batch=True)

        def boom(*args):
            raise RuntimeError("explorer bug")
        with unittest.mock.patch.object(collect.Cycle, "explorer_doc", boom):
            summary = h.cycle(hub=False)
        self.assertEqual(summary["errors"], ["explorer: RuntimeError: explorer bug"])
        self.assertIsNone(summary["explorer"])
        index = h.load("index.json")
        self.assertIsNone(index["explorer"])
        self.assertFalse((h.data / "explorer.json").exists())
        self.assertTrue((h.data / f"task/{TASK}.json").exists() and index["runs"])
        self.assertEqual(h.cycle(NOW + 60, hub=False)["errors"], [])  # the next cycle builds it
        self.assertEqual(h.load("index.json")["explorer"]["generated_at"], NOW + 60)

    def test_secrets_are_masked(self):
        h = Harness(self.addCleanup)
        write(h.fx.build / "cand-c01-r1b0/evolve/stage.json", {"title": f"Evolved, see {SECRET_KEY}", "summary": "S"})
        h.cycle(hub=False)
        text = (h.data / "explorer.json").read_text()
        self.assertNotIn(SECRET_KEY, text)
        self.assertIn("nvapi-<redacted>", text)

    def test_other_outputs_byte_identical_to_the_v1_6_collector(self):
        """Everything but index.json "explorer" (and the cycle's own timing) is what the collector wrote before v1.7
        (collect.py.pre-explorer, kept next to collect.py as the deploy's backup), on the same tree and clock."""
        previous = DASHBOARD / "collect.py.pre-explorer"
        if not previous.exists():
            self.skipTest("collect.py.pre-explorer is not there")
        import importlib.machinery
        import importlib.util
        loader = importlib.machinery.SourceFileLoader("collect_pre_explorer", str(previous))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        old = importlib.util.module_from_spec(spec)
        loader.exec_module(old)
        h = Harness(self.addCleanup, batch=True)
        h.cycle(hub=False)
        data, state = h.tmp / "old-data", h.tmp / "old-state"
        batch = old.Batch(BATCH, "Testy", "Testy · fork", h.bx.root, rounds=1, branches=5)
        sources = old.Sources(root=h.fx.root, three_way=h.fx.three_way, batches=[batch])
        summary = old.run_cycle(data, state, sources=sources, clock=lambda: NOW, slurm=h.slurm, scrape=h.scrape, hub=False)
        self.assertEqual(summary["errors"], [])
        new_files = sorted(str(p.relative_to(h.data)) for p in h.data.rglob("*") if p.is_file())
        old_files = sorted(str(p.relative_to(data)) for p in data.rglob("*") if p.is_file())
        self.assertEqual(new_files, sorted(old_files + ["explorer.json"]))
        for name in old_files:
            if name == "index.json":
                continue
            self.assertEqual((h.data / name).read_bytes(), (data / name).read_bytes(), name)
        new_index, old_index = h.load("index.json"), json.loads((data / "index.json").read_text())
        self.assertEqual(list(new_index), list(old_index) + ["explorer"])
        strip_v18(new_index)
        for key in old_index:
            if key != "collect_seconds":
                self.assertEqual(new_index[key], old_index[key], key)


def strip_v18(index):
    """index.json without its v1.8 additions (official runs[] latest / detail / detail_generated_at, tasks[]
    official_runs, queues): what a v1.7 collector writes for the same tree."""
    index.pop("queues", None)
    for run in index["runs"]:
        if "source" not in run:
            for key in ("latest", "detail", "detail_generated_at"):
                run.pop(key, None)
    for task in index["tasks"]:
        task.pop("official_runs", None)
    return index


def load_collector(path, name):
    """An earlier collect.py (kept next to collect.py as a deploy's backup) as a module."""
    import importlib.machinery
    import importlib.util
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class OfficialRunsTest(unittest.TestCase):
    """v1.8 (hengyuf 2026-10-22: the official2 queue builds six DeepSWE tasks again that official1 runs built; the
    latest official run of a task owned its only document, so official1-etree's tree and Explorer rows would have
    vanished once official2-etree launched): every official run has a document. official2-test (launched later) is
    task-a's latest: data/task/task-a.json; official1-test writes data/run/hengyuf/official1-test.json, the same
    document it wrote as the latest; official2-queued (only analysis/) is no run."""

    @classmethod
    def setUpClass(cls):
        cls.h = Harness(cls.addClassCleanup, official2=True)
        cls.summary = cls.h.cycle(hub=False)
        cls.index = cls.h.load("index.json")
        cls.task = cls.h.load(f"task/{TASK}.json")
        cls.older = cls.h.load(f"run/hengyuf/{RUN}.json")

    def test_latest_and_older_runs(self):
        self.assertEqual(self.summary["errors"], [])
        self.assertEqual(self.summary["official_runs"], {"latest": 1, "older": 1})
        runs = {run["run"]: run for run in self.index["runs"]}
        self.assertEqual(sorted(runs), [RUN, RUN2])  # official2-queued: neither launch.jsonl nor a build
        self.assertEqual((runs[RUN2]["latest"], runs[RUN2]["detail"], runs[RUN2]["job_id"]),
                         (True, f"task/{TASK}.json", "300"))
        self.assertNotIn("detail_generated_at", runs[RUN2])  # the task file: one cycle's
        self.assertEqual((runs[RUN]["latest"], runs[RUN]["detail"], runs[RUN]["detail_generated_at"]),
                         (False, f"run/hengyuf/{RUN}.json", NOW))
        task = next(task for task in self.index["tasks"] if task["task_id"] == TASK)
        self.assertEqual(task["run"], RUN2)
        self.assertEqual(task["official_runs"], [{"run": RUN2, "latest": True, "detail": f"task/{TASK}.json"},
                                                 {"run": RUN, "latest": False, "detail": f"run/hengyuf/{RUN}.json"}])
        self.assertTrue(all("official_runs" not in task for task in self.index["tasks"] if task["task_id"] != TASK))
        files = sorted(str(path.relative_to(self.h.data)) for path in self.h.data.rglob("*.json"))
        self.assertEqual([name for name in files if name.startswith(("task/", "run/"))],
                         [f"run/hengyuf/{RUN}.json", f"task/{TASK}.json"])
        self.assertEqual([oct(path.stat().st_mode & 0o777) for path in (self.h.data / "run", self.h.data / "run/hengyuf",
                                                                          self.h.data / f"run/hengyuf/{RUN}.json")],
                         ["0o700", "0o700", "0o600"])

    def test_the_older_document_is_the_one_it_wrote_as_latest(self):
        """official1-test's document is byte for byte what data/task/task-a.json was before official2-test launched
        (the same tree, the same rules), but for the stamp and the fixture's temporary path."""
        self.assertEqual((self.task["run"], self.older["run"], list(self.older)), (RUN2, RUN, list(self.task)))
        alone = Harness(self.addCleanup)
        alone.cycle(hub=False)
        before = (alone.data / f"task/{TASK}.json").read_text().replace(str(alone.tmp), "<tmp>")
        after = (self.h.data / f"run/hengyuf/{RUN}.json").read_text().replace(str(self.h.tmp), "<tmp>")
        self.assertEqual(json.loads(after), json.loads(before))

    def test_rewritten_only_on_a_change_and_removed_once_latest_again(self):
        h = Harness(self.addCleanup, official2=True)
        h.cycle(hub=False)
        detail, path = f"run/hengyuf/{RUN}.json", h.data / f"run/hengyuf/{RUN}.json"
        summary = h.cycle(NOW + 60, hub=False)
        self.assertNotIn(detail, summary["written_bytes"])
        self.assertIn(f"task/{TASK}.json", summary["written_bytes"])
        index = h.load("index.json")
        self.assertEqual((h.load(detail)["generated_at"], h.load(f"task/{TASK}.json")["generated_at"]), (NOW, NOW + 60))
        self.assertEqual(next(run for run in index["runs"] if run["run"] == RUN)["detail_generated_at"], NOW)
        h.fx.audit("syn-cand-c01-r1b0-r2b1", verdict="none", summary="Not the target's failure.")  # its run dir
        summary = h.cycle(NOW + 120, hub=False)
        self.assertIn(detail, summary["written_bytes"])
        older = h.load(detail)
        self.assertEqual(older["generated_at"], NOW + 120)
        self.assertEqual(next(node for node in older["nodes"] if node["id"] == "cand-c01-r1b0-r2b1")
                         ["failure_match"]["verdict"], "none")
        path.unlink()  # not the file written last: written again, with this cycle's stamp
        h.cycle(NOW + 180, hub=False)
        self.assertEqual(h.load(detail)["generated_at"], NOW + 180)
        shutil.rmtree(h.fx.runs / RUN2)  # official1-test is the latest again: its file goes, the task file is its
        h.cycle(NOW + 240, hub=False)
        self.assertFalse(path.exists())
        index = h.load("index.json")
        self.assertEqual((h.load(f"task/{TASK}.json")["run"], [run["latest"] for run in index["runs"]]), (RUN, [True]))
        self.assertEqual(next(task for task in index["tasks"] if task["task_id"] == TASK)["official_runs"],
                         [{"run": RUN, "latest": True, "detail": f"task/{TASK}.json"}])

    def test_audits_stay_per_run_dir(self):
        """official1-test's audits attach to its own nodes; official2-test has none yet, so its eligible tasks are not
        audited, although official1-test audited a task with the same id (syn-cand-c01); an audit in official2-test's
        dir attaches to its node alone."""
        older = {node["id"]: node for node in self.older["nodes"]}
        latest = {node["id"]: node for node in self.task["nodes"]}
        self.assertEqual((older["cand-c01"]["failure_match"]["verdict"], older["cand-c03"]["failure_match"]["verdict"]),
                         ("weak", "strong"))
        self.assertEqual([(node["id"], node["decision"], node["audit_eligible"], node["failure_match"])
                          for node in self.task["nodes"]],
                         [("cand-c01", "accepted_evolve", True, None), ("cand-c02", "evolve", True, None),
                          ("cand-c03", "pending", False, None)])
        runs = {run["run"]: run for run in self.index["runs"]}
        self.assertEqual({key: runs[RUN2]["audit"][key] for key in ("accepted", "audited", "eligible", "audited_eligible")},
                         {"accepted": 1, "audited": 0, "eligible": 2, "audited_eligible": 0})
        self.assertEqual(runs[RUN]["audit"]["audited_eligible"], 3)
        self.assertNotIn("failure_match", {key for node in latest.values() for key, value in node.items()
                                           if key == "failure_match" and value})
        h = Harness(self.addCleanup, official2=True)
        audit = json.loads((h.fx.runs / RUN / "audit/failure_match/syn-cand-c01.json").read_text())
        write(h.fx.runs / RUN2 / "audit/failure_match/syn-cand-c01.json", dict(audit, run=RUN2, verdict="strong"))
        h.cycle(hub=False)
        self.assertEqual([node["failure_match"]["verdict"] for name in (f"task/{TASK}.json", f"run/hengyuf/{RUN}.json")
                          for node in h.load(name)["nodes"] if node["id"] == "cand-c01"], ["strong", "weak"])

    def test_explorer_rows_of_every_official_run(self):
        ex = self.h.load("explorer.json")
        self.assertEqual([(run["run"], run["latest"], run["detail"], run["link"]) for run in ex["runs"]],
                         [(RUN, False, f"run/hengyuf/{RUN}.json", f"#task={TASK}&run={RUN}"),
                          (RUN2, True, f"task/{TASK}.json", f"#task={TASK}")])
        rows = explorer_rows(ex)
        built = lambda doc: [node["id"] for node in doc["nodes"] if node["build"]["task_id"]]  # noqa: E731
        self.assertEqual([row["node"] for row in rows if row["run"] == 0], built(self.older))
        self.assertEqual([row["node"] for row in rows if row["run"] == 1], built(self.task))
        self.assertEqual(ex["targets"], [{"deepswe_task": TASK, "runs": [0, 1], "patterns": [
            {"name": "Scope creep", "category": "requirement_scope"}, {"name": "Tests certify the bug", "category": "testing"}]}])
        audits = {(row["run"], row["node"]): row["audit"] for row in rows}
        self.assertEqual((audits[(1, "cand-c01")], audits[(0, "cand-c01")]), ("not_audited", "weak"))
        self.assertEqual((ex["totals"]["hengyuf"]["runs"], ex["totals"]["hengyuf"]["tasks"]),
                         (2, len(built(self.older)) + len(built(self.task))))
        self.assertEqual(self.index["explorer"]["tasks"], len(rows))

    def test_other_outputs_byte_identical_to_the_v1_7_collector(self):
        """With one official run per task (and a batch), everything is what the v1.7 collector (collect.py.pre-v18, the
        deploy's backup) writes, but for index.json's v1.8 additions (strip_v18) and explorer.json runs[].latest."""
        previous = DASHBOARD / "collect.py.pre-v18"
        if not previous.exists():
            self.skipTest("collect.py.pre-v18 is not there")
        old = load_collector(previous, "collect_pre_v18")
        h = Harness(self.addCleanup, batch=True)
        h.cycle(hub=False)
        data, state = h.tmp / "old-data", h.tmp / "old-state"
        batch = old.Batch(BATCH, "Testy", "Testy · fork", h.bx.root, rounds=1, branches=5)
        sources = old.Sources(root=h.fx.root, three_way=h.fx.three_way, batches=[batch])
        summary = old.run_cycle(data, state, sources=sources, clock=lambda: NOW, slurm=h.slurm, scrape=h.scrape, hub=False)
        self.assertEqual(summary["errors"], [])
        new_files = sorted(str(p.relative_to(h.data)) for p in h.data.rglob("*") if p.is_file())
        old_files = sorted(str(p.relative_to(data)) for p in data.rglob("*") if p.is_file())
        self.assertEqual(new_files, old_files)
        for name in old_files:
            if name not in ("index.json", "explorer.json"):
                self.assertEqual((h.data / name).read_bytes(), (data / name).read_bytes(), name)
        new_ex, old_ex = h.load("explorer.json"), json.loads((data / "explorer.json").read_text())
        self.assertEqual([run.pop("latest") for run in new_ex["runs"]], [True, None, None, None])
        self.assertEqual(new_ex, old_ex)
        new_index, old_index = h.load("index.json"), json.loads((data / "index.json").read_text())
        self.assertEqual(list(new_index), list(old_index))
        strip_v18(new_index)
        for key in old_index:
            if key not in ("collect_seconds", "explorer"):
                self.assertEqual(new_index[key], old_index[key], key)


class LaunchTest(unittest.TestCase):
    """v1.8, launch.jsonl as the official2 queue runner writes it (run_queue.py through the babysit helper): a leg's line
    naming its two candidate jobs (job_id the first), then, once one won, a line naming the winner (job_id the winner).
    The run's job is a resolved leg's winner; an unresolved leg polls every candidate and shows the most active one
    (RUNNING, then CONFIGURING / COMPLETING, then PENDING, then the first), so the run never reads "stopped" because its
    losing candidate just exited; job.jobs lists each job once, in launch order."""

    def cycle(self, states, extra=(), now=NOW):
        """A cycle over official2-test with launch.jsonl + `extra` lines, Slurm answering `states` ({job: state})."""
        h = self.h
        if extra:
            jsonl(h.fx.runs / RUN2 / "launch.jsonl", extra, mode="a")
        polled = []

        def slurm(ids):
            polled.append(sorted(ids))
            return {job: dict(h.slurm([job])[job], state=states[job]) for job in ids if job in states}
        sources = collect.Sources(root=h.fx.root, three_way=h.fx.three_way)
        summary = collect.run_cycle(h.data, h.state, sources=sources, clock=lambda: now, slurm=slurm, scrape=h.scrape,
                                    hub=False)
        self.assertEqual(summary["errors"], [])
        doc = h.load(f"task/{TASK}.json")
        run = next(run for run in h.load("index.json")["runs"] if run["run"] == RUN2)
        return doc["job"], run, polled[0]

    def leg(self, number, candidates, winner=None):
        line = {"submitted_at": f"2026-10-22T23:{number:02d}:00-0700", "job_id": winner or candidates[0], "run": RUN2,
                "task": TASK, "time": "24:00:00", "partition": "cpu_dataprocessing", "extra": "--repo other/lib --rounds 1",
                "log": str(self.h.fx.runs / f"slurm/synth-build-{RUN2}-{winner or candidates[0]}.out"),
                "babysit": {"leg": number, "candidates": list(candidates), "accounts": ["a", "b"], "resolved": bool(winner)}}
        if winner:
            line["babysit"]["winner"] = winner
        return line

    def setUp(self):
        self.h = Harness(self.addCleanup, official2=True)
        for job in ("310", "311"):
            write(self.h.fx.runs / f"slurm/synth-build-{RUN2}-{job}.out", f"23:00:00 the build in job {job}\n")

    def test_a_resolved_leg_is_its_winner(self):
        job, run, polled = self.cycle({"300": "RUNNING", "299": "CANCELLED"})
        self.assertEqual((job["job_id"], job["jobs"], job["status"], run["job_id"]), ("300", ["299", "300"], "running", "300"))
        self.assertIn("300", polled)
        self.assertNotIn("299", polled)                     # the loser of a resolved leg is not polled

    def test_an_unresolved_leg_shows_its_most_active_candidate(self):
        extra = [self.leg(2, ["310", "311"])]
        job, run, polled = self.cycle({"310": "CANCELLED", "311": "RUNNING"}, extra)
        self.assertEqual((job["job_id"], job["status"], run["job_id"], run["status"]), ("311", "running", "311", "running"))
        self.assertEqual(job["jobs"], ["299", "300", "310", "311"])
        self.assertTrue(job["log_path"].endswith(f"synth-build-{RUN2}-311.out"))
        self.assertEqual(job["log_tail"], ["23:00:00 the build in job 311"])
        self.assertTrue({"310", "311"} <= set(polled))
        for states, want in (({"310": "PENDING", "311": "CONFIGURING"}, "311"), ({"310": "COMPLETING", "311": "PENDING"}, "310"),
                             ({"310": "CANCELLED", "311": "PENDING"}, "311"), ({"310": "FAILED", "311": "CANCELLED"}, "310"),
                             ({}, "310")):              # Slurm silent: the line's own job
            self.assertEqual(self.cycle(states)[0]["job_id"], want, states)
        # the runner names the winner a pass later: that job, whatever Slurm says of the other
        job, run, polled = self.cycle({"310": "RUNNING", "311": "RUNNING"}, [self.leg(2, ["310", "311"], winner="311")])
        self.assertEqual((job["job_id"], job["jobs"]), ("311", ["299", "300", "310", "311"]))
        self.assertNotIn("310", polled)

    def test_jobs_listed_once(self):
        """official2-bandit-taint, 2026-10-22: leg 2's winner was its first candidate, so both lines name job 20720292."""
        job, run, _ = self.cycle({"320": "RUNNING"}, [self.leg(3, ["320", "321"]), self.leg(3, ["320", "321"], winner="320")])
        self.assertEqual((job["job_id"], job["jobs"]), ("320", ["299", "300", "320"]))


class QueueTest(unittest.TestCase):
    """v1.8 (hengyuf 2026-10-22: the new synthesis "同时更新到dash board"): index.json queues[] from the official2
    queue's four runner files and nothing else."""

    def test_the_queue_entry(self):
        h = Harness(self.addCleanup, official2=True, queue=True)
        summary = h.cycle(hub=False)
        self.assertEqual(summary["errors"], [])
        self.assertEqual(summary["queues"], {"official2": {"runs": 3, "alive": True}})
        index = h.load("index.json")
        queue = index["queues"][0]
        self.assertEqual(list(queue), ["name", "root", "read_at", "max_active", "runner", "budget", "serving",
                                       "stop_launching", "updated_at", "finished_at", "runs", "log_tail"])
        self.assertEqual(queue["read_at"], NOW)  # the fixture's clock: the files' read time
        self.assertEqual((queue["name"], queue["root"], queue["max_active"], queue["serving"], queue["stop_launching"]),
                         ("official2", str(h.queue), 5, True, False))
        self.assertEqual(queue["runner"], {"heartbeat_at": NOW - 30, "age_s": 30.0, "host": "cpu-node-7",
                                           "slurm_job_id": "456", "note": "running", "alive": True})
        self.assertEqual(queue["budget"], {"key": "NVAPI_KEY_OFFICIAL_1", "remaining": 2345.68,
                                           "at": collect.parse_time("2026-10-22T22:10:00-0700")})
        self.assertEqual((queue["updated_at"], queue["finished_at"]), (collect.parse_time("2026-10-22T22:12:00-0700"), None))
        self.assertEqual([(run["run"], run["task"], run["repo"], run["batch"], run["status"], run["leg"], run["winner"],
                           run["candidates"]) for run in queue["runs"]],
                         [(RUN2, TASK, "other/lib", "hengyuf", "running", 1, "300", ["300", "301"]),
                          (QUEUED, "task-b", "queued/b", "baihe", "queued", 0, None, []),
                          ("official2-later", "task-c", "later/c", "baihe", "queued", 0, None, [])])
        tail = queue["log_tail"]
        self.assertEqual((len(tail), tail[0]), (40, "[2026-10-22T22:12:00-0700] pass 12"))
        self.assertEqual(tail[-6:], [
            "[2026-10-22T22:46:00-0700] pass 46",
            "[2026-10-22T22:47:00-0700] pass failed: RuntimeError: squeue failed (1): slurm_load_jobs error: Socket timed "
            "out on send/recv operation",
            "[2026-10-22T22:48:00-0700] budget: switching the active hub key NVAPI_KEY_OFFICIAL_1 -> NVAPI_KEY_OFFICIAL_2 "
            "(remaining 2100.5)",
            "[2026-10-22T22:49:00-0700] another runner beat 30 s ago on cpu-node-8 (pid ⋯, job 789): not starting",
            "[2026-10-22T22:50:00-0700] official2-test: launched leg 1 on cpu_datamover: candidates 299,300",
            "[2026-10-22T22:51:00-0700] budget: NVAPI_KEY_OFFICIAL_1 remaining 2345.678 (nvapi-<redacted>)"])
        h.cycle(NOW + 60, hub=False)                         # the start line with the runner's pid, in a shorter log
        write(h.queue / "runner.log", (h.queue / "runner.log").read_text().splitlines(keepends=True)[0])
        h.cycle(NOW + 120, hub=False)
        self.assertEqual(h.load("index.json")["queues"][0]["log_tail"],
                         ["[2026-10-22T22:00:00-0700] official2 queue runner starting (live) on cpu-node-7 pid ⋯ job 456: "
                          "3 runs, max 5 active"])
        # nothing else of those files reaches any output: notes, requested_by, the budget's error, key switches, pid
        for path in h.data.rglob("*"):
            if path.is_file():
                text = path.read_text()
                for planted in (SECRET_KEY, SECRET_SK, SECRET_TOKEN, "budget-error-marker", "request-marker",
                                "notes-marker", "NVAPI_KEY_SWITCH_MARKER", "987654321", "launch_errors",
                                "notes-json-marker"):
                    self.assertNotIn(planted, text, f"{path.name}: {planted}")

    def test_files_the_runner_has_not_written_yet(self):
        h = Harness(self.addCleanup, official2=True)
        h.queue = plant_queue(h.fx, state=False, heartbeat=False)
        (h.queue / "runner.log").unlink()
        self.assertEqual(h.cycle(hub=False)["errors"], [])
        queue = h.load("index.json")["queues"][0]
        self.assertEqual(queue["runner"], {"heartbeat_at": None, "age_s": None, "host": None, "slurm_job_id": None,
                                           "note": None, "alive": False})
        self.assertEqual((queue["budget"], queue["serving"], queue["stop_launching"], queue["updated_at"],
                          queue["log_tail"]), (None, False, False, None, []))
        self.assertEqual({(run["status"], run["leg"], run["winner"]) for run in queue["runs"]}, {("queued", 0, None)})
        h.queue = h.tmp / "no-such-queue"  # a queue whose directory does not exist: no entry, no error
        summary = h.cycle(NOW + 60, hub=False)
        self.assertEqual((summary["errors"], h.load("index.json")["queues"]), ([], []))

    def test_liveness(self):
        h = Harness(self.addCleanup, queue=True)
        beat = json.loads((h.queue / "heartbeat.json").read_text())
        for at, note, alive in ((NOW - 900, "running", True), (NOW - 901, "running", False), (NOW - 10, "stopped", False),
                                (NOW - 10, "draining", True), (NOW + 5, "running", True)):
            write(h.queue / "heartbeat.json", dict(beat, at=at, note=note))
            h.cycle(hub=False)
            runner = h.load("index.json")["queues"][0]["runner"]
            self.assertEqual((runner["age_s"], runner["alive"]), (max(0.0, NOW - at), alive), (at, note))

    def test_only_the_four_files_are_read(self):
        """Exactly queue.json, state.json, heartbeat.json and runner.log may be opened (here the queue lives outside the
        fixture tree, as S/synth/queue2 lies outside every other read root): another file of the directory is refused,
        and a symlink at one of the four paths is never followed, wherever it points: out of every root, or into another
        read root (the fixture's synth/runs); the cycle names it."""
        h = Harness(self.addCleanup)
        h.queue = plant_queue(h.fx, root=h.tmp / "queue2")
        self.assertEqual(h.cycle(hub=False)["errors"], [])
        for name in ("notes.json", "runner.lock"):
            for exact in (False, True):
                with self.assertRaises(PermissionError):
                    collect._open_regular(h.queue / name, exact=exact)
        collect._open_regular(h.queue / "queue.json", exact=True).close()
        state = json.loads((h.queue / "state.json").read_text())
        inside = write(h.fx.runs / RUN / "state-copy.json", state)  # inside a read root (synth/runs)
        for target in (write(h.tmp / "elsewhere/state.json", state), inside):
            (h.queue / "state.json").unlink()
            (h.queue / "state.json").symlink_to(target)
            summary = h.cycle(NOW + 60, hub=False)
            self.assertEqual(summary["errors"], ["queue official2: state.json is a symlink: not read"], target)
            queue = h.load("index.json")["queues"][0]
            self.assertEqual(({run["status"] for run in queue["runs"]}, queue["budget"], queue["serving"]),
                             ({"queued"}, None, False), target)
        (h.queue / "runner.log").rename(h.queue / "runner.log.real")
        (h.queue / "runner.log").symlink_to(h.queue / "runner.log.real")  # even within the queue's directory
        h.cycle(NOW + 120, hub=False)
        self.assertEqual(h.load("index.json")["queues"][0]["log_tail"], [])

    def test_a_copied_analysis_is_not_the_runs_usage(self):
        """run_queue.py prep() copies the earlier run's analysis/<task>/ (its analyzer's codex session included) with its
        old mtimes into the queued run, and task_builder reuses it: a session older than the run's first launch is never
        the run's usage (counted where it ran), and one counted before is dropped; an analysis redone in the run (written
        after its launch) counts; a run dir without a launch has no usage. queue.json plays no part."""
        h = Harness(self.addCleanup)
        session = jsonl(h.fx.runs / RUN / f"analysis/{TASK}/.agent/events.jsonl",
                        [{"type": "thread.started", "thread_id": "analyzer-1"}, h.fx.turn(7000, 0, 300, 30)])
        first1 = collect.parse_time("2026-10-17T02:29:31-07:00")       # official1-test's first launch
        os.utime(session, (first1 + 600, first1 + 600))                 # analyzed in its own build
        plant_official2(h.fx)  # copies official1-test's analysis/ (copytree keeps the mtime)
        shutil.copytree(h.fx.runs / RUN / f"analysis/{TASK}", h.fx.runs / QUEUED / f"analysis/{TASK}")
        copy = h.fx.runs / RUN2 / f"analysis/{TASK}/.agent/events.jsonl"
        first2 = collect.parse_time("2026-10-22T22:05:00-0700")         # official2-test's first launch
        self.assertLess(copy.stat().st_mtime, first2)

        def usage(now, queue):
            h.queue = plant_queue(h.fx) if queue else None
            self.assertEqual(h.cycle(now, hub=False)["errors"], [])
            return {row["run"]: row for row in h.load("api.json")["by_run"]}
        for now, queue in ((NOW, False), (NOW + 60, True)):           # the same with and without queue.json
            by_run = usage(now, queue)
            self.assertEqual((by_run[RUN]["codex"]["input_tokens"] >= 7000, by_run[RUN2]["codex"], QUEUED in by_run),
                             (True, None, False), queue)
        # a copy counted before this rule (here: its mtime made late, then back): dropped from the state
        os.utime(copy, (first2 + 60, first2 + 60))
        self.assertEqual(usage(NOW + 120, True)[RUN2]["codex"]["input_tokens"], 7000)
        os.utime(copy, (first2 - 86400, first2 - 86400))
        self.assertEqual(usage(NOW + 180, True)[RUN2]["codex"], None)
        # an analysis redone in official2-test (after its launch): its own session counts
        jsonl(copy, [{"type": "thread.started", "thread_id": "analyzer-2"}, h.fx.turn(500, 0, 50, 5)])
        os.utime(copy, (first2 + 3600, first2 + 3600))
        by_run = usage(NOW + 240, False)
        self.assertEqual(by_run[RUN2]["codex"]["input_tokens"], 500)
        self.assertGreaterEqual(by_run[RUN]["codex"]["input_tokens"], 7000)

    def test_broken_files_never_stop_the_cycle(self):
        h = Harness(self.addCleanup, queue=True)
        state = json.loads((h.queue / "state.json").read_text())
        state["runs"][RUN2].update(status="exploded", leg="x", winner="abc", candidates=["300", "<b>", None])
        state["budget"].update(key="nvapi-not-a-name", remaining="lots")
        write(h.queue / "state.json", state)
        write(h.queue / "heartbeat.json", {"at": "yesterday", "slurm_job_id": "job-1", "note": "x" * 100})
        summary = h.cycle(hub=False)
        self.assertEqual(summary["errors"], [f"queue official2: {RUN2}: unknown status 'exploded'"])
        queue = h.load("index.json")["queues"][0]
        self.assertEqual({key: queue["runs"][0][key] for key in ("status", "leg", "winner", "candidates")},
                         {"status": None, "leg": 0, "winner": None, "candidates": ["300"]})
        self.assertEqual(queue["budget"], {"key": None, "remaining": None, "at": collect.parse_time("2026-10-22T22:10:00-0700")})
        self.assertEqual((queue["runner"]["heartbeat_at"], queue["runner"]["slurm_job_id"], queue["runner"]["alive"]),
                         (None, None, False))
        self.assertLessEqual(len(queue["runner"]["note"]), 40)
        write(h.queue / "queue.json", "{not json")
        summary = h.cycle(NOW + 60, hub=False)
        self.assertEqual(summary["errors"], ["queue official2: RuntimeError: queue.json missing, unreadable or without a "
                                             "run list"])
        index = h.load("index.json")
        self.assertEqual(index["queues"], [])
        self.assertTrue((h.data / f"task/{TASK}.json").exists() and index["runs"])


if __name__ == "__main__":
    unittest.main()
