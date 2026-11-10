"""Offline tests for tools/recover_slots.py: slot selection, loss classification, rate recomputation, views, the
recommendation, the retry/budget orchestration and the record (no sandbox, network, LLM or hub calls).
Run from S/synth:  ../atp/runtime/controller/bin/python3 -B -m unittest discover -s tests"""
import sys

sys.dont_write_bytecode = True
from pathlib import Path

SYNTH = Path(__file__).resolve().parents[1]
for _path in (SYNTH, SYNTH / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import asyncio  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402
from unittest import mock  # noqa: E402

import recover_slots as rs  # noqa: E402
import rollouts  # noqa: E402
from common import STUDENT, TEACHER, read_json, write_json  # noqa: E402


def slot(name, status, cheated=None, categories=None, edits=(), loss=None):
    model = name.rsplit("-r", 1)[0]
    if status not in rs.VALID and loss is None:
        loss = {"kind": "hub_422", "retry": True}
    return {"slot": name, "model": model, "status": status, "cheated": cheated if status == "pass" else None,
            "categories": categories, "surviving_test_edits": list(edits), "loss": loss,
            "lost": status not in rs.VALID}


def view(slots):
    return rs.view_slots(slots, {}, grader="recorded")


# The four tasks of the first recovery batch (2026-10-20), as recorded.
YTT = [slot("sc14-r0", "pass", edits=["a_test.go"]), slot("sc14-r1", "fail"), slot("sc14-r2", "fail"),
       slot("sc14-r3", "infra", loss={"kind": "student_abort", "retry": True}), slot("sc14-r4", "fail")] + \
      [slot(f"kimi-r{i}", "pass", cheated=False) for i in range(5)]
SCRIGGO = [slot(f"sc14-r{i}", "fail") for i in range(5)] + \
          [slot("kimi-r0", "fail", edits=["generics_test.go"]), slot("kimi-r1", "infra"), slot("kimi-r2", "infra"),
           slot("kimi-r3", "pass", cheated=False), slot("kimi-r4", "infra")]
DASEL = [slot(f"sc14-r{i}", "fail") for i in range(5)] + \
        [slot("kimi-r0", "infra", loss={"kind": "hub_401", "retry": True}), slot("kimi-r1", "fail"),
         slot("kimi-r2", "fail"), slot("kimi-r3", "infra"), slot("kimi-r4", "pass", cheated=False)]


class Selection(unittest.TestCase):
    def test_lost_slots_in_repeat_order(self):
        self.assertEqual(rs.select_replacements(SCRIGGO, TEACHER, 3), ["kimi-r1", "kimi-r2", "kimi-r4"])
        self.assertEqual(rs.select_replacements(SCRIGGO, TEACHER, 2), ["kimi-r1", "kimi-r2"])
        self.assertEqual(rs.select_replacements(YTT, STUDENT, 1), ["sc14-r3"])
        self.assertEqual(rs.select_replacements(DASEL, TEACHER, 2), ["kimi-r0", "kimi-r3"])

    def test_more_rollouts_than_lost_slots_is_refused(self):
        with self.assertRaises(rs.RecoveryError):
            rs.select_replacements(YTT, STUDENT, 2)  # would add a 6th sc14 slot
        with self.assertRaises(rs.RecoveryError):
            rs.select_replacements(YTT, TEACHER, 1)  # kimi lost nothing
        with self.assertRaises(rs.RecoveryError):
            rs.select_replacements(SCRIGGO, TEACHER, 0)

    def test_grading_failures_are_not_recoverable(self):
        slots = DASEL[:5] + [slot("kimi-r0", "infra", loss={"kind": "grading_failure", "retry": False}),
                             slot("kimi-r1", "infra"), slot("kimi-r2", "pass", cheated=False),
                             slot("kimi-r3", "pass", cheated=False), slot("kimi-r4", "fail")]
        self.assertEqual(rs.select_replacements(slots, TEACHER, 1), ["kimi-r1"])
        with self.assertRaises(rs.RecoveryError):
            rs.select_replacements(slots, TEACHER, 2)

    def test_slot_names(self):
        self.assertEqual(rs.parse_slot("kimi-r12"), ("kimi", 12))
        with self.assertRaises(rs.RecoveryError):
            rs.parse_slot("kimi_r1")
        self.assertEqual(sorted(["kimi-r10", "kimi-r2", "sc14-r1"], key=rs.slot_order), ["sc14-r1", "kimi-r2", "kimi-r10"])

    def test_probe_ids_are_deterministic_per_label_and_sequence(self):
        a = rs.probe_id("syn-x", TEACHER, 3, "L", 0)
        self.assertEqual(a, rs.probe_id("syn-x", TEACHER, 3, "L", 0))
        self.assertEqual(len({a, rs.probe_id("syn-x", TEACHER, 3, "L", 1), rs.probe_id("syn-x", TEACHER, 3, "M", 0),
                              rs.probe_id("syn-x", TEACHER, 2, "L", 0)}), 4)


class Rates(unittest.TestCase):
    def test_before_rates_match_score(self):
        ytt = rs.evaluate(view(YTT))
        self.assertEqual((ytt["valid"], ytt["sc14"], ytt["kimi_clean"], ytt["in_range"]), ([4, 5], 0.25, 1.0, False))
        scriggo = rs.evaluate(view(SCRIGGO))
        self.assertEqual((scriggo["valid"], scriggo["sc14"], scriggo["kimi_clean"], scriggo["in_range"]),
                         ([5, 2], 0.0, 0.5, None))
        self.assertTrue(scriggo["reason"].startswith("inconclusive"))
        dasel = rs.evaluate(view(DASEL))
        self.assertEqual((dasel["valid"], dasel["kimi_clean"], dasel["in_range"]), ([5, 3], 1 / 3, False))

    def test_outcome_tables_say_what_each_result_decides(self):
        ytt = rs.outcome_table(view(YTT), ["sc14-r3"], STUDENT)
        self.assertEqual([(r["sc14"], r["in_range"]) for r in ytt], [(0.2, True), (0.4, False)])  # FAIL: 1/5, PASS: 2/5
        scriggo = rs.outcome_table(view(SCRIGGO), ["kimi-r1", "kimi-r2", "kimi-r4"], TEACHER)
        self.assertEqual([(r["kimi_clean"], r["in_range"]) for r in scriggo],
                         [(0.2, False), (0.4, False), (0.6, True), (0.8, True)])
        dasel = rs.outcome_table(view(DASEL), ["kimi-r0", "kimi-r3"], TEACHER)
        self.assertEqual([(r["kimi_clean"], r["in_range"]) for r in dasel], [(0.2, False), (0.4, False), (0.6, True)])
        self.assertTrue(all(r["valid"] == [5, 5] for r in ytt + scriggo + dasel))

    def test_boundaries_use_score_arithmetic(self):
        slots = [slot("sc14-r0", "pass")] + [slot(f"sc14-r{i}", "fail") for i in range(1, 5)] + \
                [slot(f"kimi-r{i}", "pass", cheated=False) for i in range(3)] + \
                [slot("kimi-r3", "fail"), slot("kimi-r4", "fail")]
        self.assertTrue(rs.evaluate(view(slots))["in_range"])  # 0.2 is not > 0.2; 0.6 is not < 0.6

    def test_cheated_and_unaudited_passes(self):
        slots = [slot(f"sc14-r{i}", "fail") for i in range(5)] + \
                [slot("kimi-r0", "pass", cheated=True), slot("kimi-r1", "pass", cheated=False),
                 slot("kimi-r2", "pass", cheated=False), slot("kimi-r3", "pass"), slot("kimi-r4", "fail")]
        result = rs.evaluate(view(slots))
        self.assertEqual((result["kimi_clean_pass"], result["kimi_cheated_pass"], result["kimi_unaudited_pass"]),
                         (2, 1, ["kimi-r3"]))
        self.assertIsNone(result["in_range"])  # 2/5 or 3/5 depending on kimi-r3's audit
        self.assertEqual(result["bounds"]["best"]["kimi_clean"], 0.6)
        self.assertEqual(result["bounds"]["worst"]["kimi_clean"], 0.4)

    def test_answer_retrieval_rejects(self):
        slots = [slot(f"sc14-r{i}", "fail") for i in range(5)] + \
                [slot(f"kimi-r{i}", "pass", cheated=False) for i in range(4)] + \
                [slot("kimi-r4", "pass", cheated=True, categories=["Answer retrieval by action"])]
        result = rs.evaluate(rs.view_slots(slots, {}, grader="recorded"))
        self.assertIs(result["in_range"], False)
        self.assertIn("leaky", result["reason"])

    def test_unknown_outcomes_give_bounds(self):
        slots = [dict(s) for s in YTT]
        slots[3] = dict(slots[3], status="unknown")
        result = rs.evaluate(rs.view_slots(slots, {}, grader="recorded"))
        self.assertIsNone(result["in_range"])
        self.assertEqual((result["bounds"]["best"]["sc14"], result["bounds"]["worst"]["sc14"]), (0.2, 0.4))

    def test_recommendation(self):
        self.assertEqual(rs.recommend({"in_range": False}, {"in_range": True}), "newly_in_range")
        self.assertEqual(rs.recommend({"in_range": None}, {"in_range": True}), "newly_in_range")
        self.assertEqual(rs.recommend({"in_range": None}, {"in_range": False}), "still_out_of_range")
        self.assertEqual(rs.recommend({"in_range": False}, {"in_range": None}), "unclear")
        self.assertEqual(rs.recommend({"in_range": True}, {"in_range": True}), "unclear")  # nothing new to decide


class Views(unittest.TestCase):
    def test_recovered_slots_replace_lost_ones_per_grader(self):
        replacements = {"kimi-r1": {"status": "pass", "grades": {"new": "pass", "old": "pass"}, "cheated": False},
                        "kimi-r2": {"status": "fail", "grades": {"new": "pass", "old": "fail"}, "cheated": False},
                        "kimi-r4": {"status": "infra", "grades": {}, "cheated": None}}
        recorded = rs.view_slots(SCRIGGO, replacements, grader="recorded")
        sanitized = rs.view_slots(SCRIGGO, replacements, grader="new")
        statuses = lambda v: {x["slot"]: x["status"] for x in v if x["model"] == TEACHER}  # noqa: E731
        self.assertEqual(statuses(recorded), {"kimi-r0": "fail", "kimi-r1": "pass", "kimi-r2": "fail",
                                              "kimi-r3": "pass", "kimi-r4": "infra"})
        self.assertEqual(statuses(sanitized), {"kimi-r0": "fail", "kimi-r1": "pass", "kimi-r2": "pass",
                                               "kimi-r3": "pass", "kimi-r4": "infra"})
        self.assertEqual(rs.evaluate(recorded)["valid"], [5, 4])
        self.assertEqual(rs.evaluate(sanitized)["kimi_clean"], 0.75)

    def test_sanitizing_view_without_a_new_grade_is_unknown(self):
        replacements = {"sc14-r3": {"status": "fail", "grades": {"old": "fail"}}}
        sanitized = rs.view_slots(YTT, replacements, grader="new")
        self.assertEqual(next(x for x in sanitized if x["slot"] == "sc14-r3")["status"], "unknown")
        self.assertIsNone(rs.evaluate(sanitized)["in_range"])

    def test_recorded_slot_precedence_regrade_then_repair_then_recorded(self):
        regraded = {"kimi-r0": {"new": "pass", "old": "fail", "cheated": False, "categories": []}}
        repair = {"kimi-r0": "fail", "kimi-r3": "fail", "sc14-r0": "pass"}
        sanitized = {x["slot"]: x for x in rs.view_slots(SCRIGGO, {}, grader="new", regraded=regraded, repair=repair)}
        self.assertEqual((sanitized["kimi-r0"]["status"], sanitized["kimi-r0"]["source"]),
                         ("pass", "regraded here (sanitizing grader)"))
        self.assertIs(sanitized["kimi-r0"]["cheated"], False)  # a newly passing slot takes this tool's audit
        self.assertEqual((sanitized["kimi-r3"]["status"], sanitized["kimi-r3"]["source"]), ("fail", "grading_repair record"))
        self.assertEqual(sanitized["sc14-r0"]["status"], "pass")
        self.assertEqual(sanitized["kimi-r1"]["status"], "infra")  # lost slots stay lost
        recorded = {x["slot"]: x["status"] for x in rs.view_slots(SCRIGGO, {}, grader="recorded", regraded=regraded,
                                                                     repair=repair)}
        self.assertEqual((recorded["kimi-r0"], recorded["kimi-r3"]), ("fail", "pass"))  # the recorded grader's view

    def test_newly_passing_slot_without_audit_is_unaudited(self):
        sanitized = rs.view_slots(SCRIGGO, {}, grader="new", repair={"kimi-r0": "pass"})
        kimi_r0 = next(x for x in sanitized if x["slot"] == "kimi-r0")
        self.assertEqual((kimi_r0["status"], kimi_r0["cheated"]), ("pass", None))

    def test_scriggo_with_kimi_r0_regraded_and_three_recoveries(self):
        # main's addition (2026-10-20): kimi-r0 is a redeclared false FAIL; with it clean, 1 clean of 3 reruns suffices
        regraded = {"kimi-r0": {"new": "pass", "old": "fail", "cheated": False}}
        replacements = {s: {"status": "fail", "grades": {"new": "fail", "old": "fail"}} for s in ("kimi-r1", "kimi-r2")}
        replacements["kimi-r4"] = {"status": "pass", "grades": {"new": "pass", "old": "pass"}, "cheated": False}
        sanitized = rs.evaluate(rs.view_slots(SCRIGGO, replacements, grader="new", regraded=regraded))
        recorded = rs.evaluate(rs.view_slots(SCRIGGO, replacements, grader="recorded", regraded=regraded))
        self.assertEqual((sanitized["kimi_clean"], sanitized["in_range"]), (0.6, True))
        self.assertEqual((recorded["kimi_clean"], recorded["in_range"]), (0.4, False))

    def test_pending_repair_and_robustness(self):
        sanitized = rs.view_slots(YTT, {"sc14-r3": {"status": "fail", "grades": {"new": "fail"}}}, grader="new")
        pending = rs.pending_repair(sanitized, YTT)
        self.assertEqual(pending, ["sc14-r0"])  # the only valid slot with surviving test-path edits in the fixture
        result = rs.robustness(sanitized, pending)
        self.assertFalse(result["verdict_can_change"])  # sc14-r0 flipping to FAIL keeps sc14 <= 0.2
        flipped = [dict(s, surviving_test_edits=["x_test.go"]) if s["slot"] == "sc14-r1" else s for s in YTT]
        result = rs.robustness(sanitized, rs.pending_repair(sanitized, flipped))
        self.assertTrue(result["verdict_can_change"])  # sc14-r1 FAIL -> PASS would give 0.4

    def test_final_replacements_prefer_a_valid_retry(self):
        recovered = [{"slot": "kimi-r0", "replaces": "kimi-r1", "status": "infra", "seq": 0},
                     {"slot": "kimi-r1", "replaces": "kimi-r2", "status": "pass", "seq": 0},
                     {"slot": "kimi-r0", "replaces": "kimi-r1", "status": "fail", "seq": 1}]
        chosen = rs.final_replacements(recovered)
        self.assertEqual((chosen["kimi-r1"]["seq"], chosen["kimi-r1"]["status"]), (1, "fail"))
        self.assertEqual(chosen["kimi-r2"]["status"], "pass")
        lost_twice = rs.final_replacements([dict(recovered[0]), dict(recovered[0], seq=1, slot="kimi-r0")])
        self.assertEqual((lost_twice["kimi-r1"]["seq"], lost_twice["kimi-r1"]["status"]), (1, "infra"))


class RepairRecords(unittest.TestCase):
    def test_v2_statuses_after(self):
        statuses, field = rs.repair_statuses({"statuses_after": {"kimi-r0": "pass", "sc14-r0": "fail"},
                                              "unknown_slots": ["sc14-r0"]})
        self.assertEqual((statuses, field), ({"kimi-r0": "pass", "sc14-r0": "unknown"}, "statuses_after"))

    def test_v1_slot_entries(self):
        record = {"slots": [{"slot": "sc14-r0", "recorded": "pass", "new_effective": "fail", "verdict": "flip"},
                            {"slot": "kimi-r0", "recorded": "pass", "verdict": "not_regraded",
                             "surviving_test_files": ["a_test.go"]},
                            {"slot": "kimi-r1", "recorded": "pass", "verdict": "not_regraded",
                             "surviving_test_files": []},
                            {"slot": "kimi-r2", "recorded": "fail", "new": "pass", "verdict": "unchanged"}]}
        statuses, field = rs.repair_statuses(record)
        self.assertEqual(field, "slots")
        self.assertEqual(statuses, {"sc14-r0": "fail", "kimi-r0": "unknown", "kimi-r1": "pass", "kimi-r2": "pass"})

    def test_unknown_formats(self):
        self.assertEqual(rs.repair_statuses(None), ({}, "none"))
        self.assertEqual(rs.repair_statuses({"rates_after": {}}), ({}, "unrecognized"))


class Losses(unittest.TestCase):
    def attempt(self, root: Path, http=(), capture=(), files=None) -> Path:
        attempt = root / "attempt-0"
        attempt.mkdir(parents=True)
        (attempt / "http.jsonl").write_text("".join(json.dumps(e) + "\n" for e in http))
        (attempt / "capture-events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in capture))
        for name, value in (files or {}).items():
            (attempt / name).write_text(json.dumps(value))
        return attempt

    def test_hub_422_is_classified_and_redacted(self):
        body = json.dumps({"error": {"message": "Budget has been exceeded! Key=team-prod-svc-0485bc9c (sk-...zcrA) "
                                                "Current cost: 14839.9, Max budget: 10000.0", "type": "budget_exceeded"}})
        with tempfile.TemporaryDirectory() as directory:
            attempt = self.attempt(Path(directory), http=[
                {"event": "response", "status": 200, "timestamp": 1792521800.0, "body": "{}"},
                {"event": "response", "status": 422, "timestamp": 1792521888.4, "body": body}])
            evidence = rs.loss_evidence(attempt)
            loss = rs.classify_loss("fatal", evidence)
        self.assertEqual((loss["kind"], loss["retry"], loss["platform"]), ("hub_422", True, True))
        text = json.dumps(evidence) + json.dumps(loss)
        self.assertNotIn("sk-", text)
        self.assertNotIn("0485bc9c", text)
        self.assertIn("budget_exceeded", loss["detail"])
        self.assertIsNotNone(loss["at"])

    def test_hub_401_connection_slots(self):
        body = json.dumps({"error": {"message": "Authentication Error, Error in connector: Error querying the "
                                                "database: FATAL: remaining connection slots are reserved",
                                     "type": "auth_error"}})
        with tempfile.TemporaryDirectory() as directory:
            loss = rs.classify_loss("fatal", rs.loss_evidence(self.attempt(Path(directory), http=[
                {"event": "response", "status": 401, "timestamp": 1792471972.0, "body": body}])))
        self.assertEqual(loss["kind"], "hub_401")

    def test_student_abort(self):
        body = json.dumps({"choices": [{"finish_reason": "abort", "message": {"content": ""}}]})
        error = ("1 validation error for NeMoGymChatCompletion\nchoices.0.finish_reason\n  Input should be 'stop', "
                 "'length', 'tool_calls', 'content_filter' or 'function_call' [type=literal_error, input_value='abort'")
        with tempfile.TemporaryDirectory() as directory:
            evidence = rs.loss_evidence(self.attempt(Path(directory), http=[
                {"event": "response", "status": 200, "timestamp": 1792490643.2, "body": body},
                {"scope": "native_model_method", "event": "model_call_error", "timestamp": "1792490643.47",
                 "error_type": "ValidationError", "error": error}]))
            loss = rs.classify_loss("needs_review", evidence)
        self.assertEqual(len(evidence["aborts"]), 1)
        self.assertEqual((loss["kind"], loss["retry"]), ("student_abort", True))

    def test_pty_grading_failure_and_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            pty = rs.classify_loss("exhausted", rs.loss_evidence(self.attempt(
                Path(directory) / "a", capture=[{"time": 1.0, "event": "agent_interrupted", "error_type": "SandboxPtyError"}])))
            grading = rs.classify_loss("needs_review", rs.loss_evidence(self.attempt(
                Path(directory) / "b", files={"terminal.json": {"verifier_error": "test.patch failed to apply"}})))
            other = rs.classify_loss("needs_review", rs.loss_evidence(self.attempt(Path(directory) / "c")))
        self.assertEqual(pty["kind"], "pty_closed")
        self.assertEqual((grading["kind"], grading["retry"]), ("grading_failure", False))
        self.assertEqual((other["kind"], other["retry"]), ("atp_needs_review", True))


class Bookkeeping(unittest.TestCase):
    def test_budget_counts_slots_and_started_attempts(self):
        budget = rs.Budget({TEACHER: 10, STUDENT: 4})
        self.assertTrue(budget.reserve(TEACHER, 7, "initial"))
        self.assertTrue(budget.reserve(TEACHER, 3, "retry"))
        self.assertFalse(budget.reserve(TEACHER, 1, "retry 2"))
        self.assertTrue(budget.reserve(STUDENT, 1, "initial"))
        self.assertFalse(budget.reserve(STUDENT, 1, "retry", attempts_started=4))  # ATP already started 4 attempts
        self.assertEqual(budget.reserved, {TEACHER: 10, STUDENT: 1})
        self.assertEqual(rs.parse_budget("kimi=6"), {STUDENT: 4, TEACHER: 6})
        with self.assertRaises(rs.RecoveryError):
            rs.parse_budget("kimi=x")

    def test_limiter_caps_concurrent_rollouts_per_model(self):
        limiter, events = rs.Limiter(8), []

        async def hold(name, model, n, seconds):
            async with limiter.hold(model, n):
                events.append(("in", name, dict(limiter.used)))
                await asyncio.sleep(seconds)
            events.append(("out", name))

        async def main():
            await asyncio.gather(hold("a", TEACHER, 5, 0.05), hold("b", TEACHER, 5, 0.01), hold("c", STUDENT, 1, 0.01))

        asyncio.run(main())
        self.assertLess(events.index(("out", "a")), events.index(next(e for e in events if e[:2] == ("in", "b"))))
        self.assertTrue(all(e[2].get(TEACHER, 0) <= 8 for e in events if e[0] == "in"))
        with self.assertRaises(rs.RecoveryError):
            asyncio.run(self._too_many(limiter))

    async def _too_many(self, limiter):
        async with limiter.hold(TEACHER, 9):
            pass

    def test_spec_merges_items_and_checks_the_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec.json"
            write_json(path, {"label": "t1", "budget": {"kimi": 10, "sc14": 4}, "items": [
                {"run": "r", "task": "syn-a", "model": "kimi", "n": 3, "regrade_slots": ["kimi-r0"]},
                {"run": "r", "task": "syn-a", "model": "sc14", "n": 1},
                {"run": "r", "task": "syn-b", "model": "kimi", "n": 2}]})
            spec = rs.load_spec(path)
            self.assertEqual([t["recover"] for t in spec["tasks"]], [{"kimi": 3, "sc14": 1}, {"kimi": 2}])
            self.assertEqual(spec["tasks"][0]["regrade_slots"], ["kimi-r0"])
            write_json(path, {"label": "t2", "budget": {"kimi": 4}, "items": [
                {"run": "r", "task": "syn-a", "model": "kimi", "n": 3}, {"run": "r", "task": "syn-b", "model": "kimi", "n": 2}]})
            with self.assertRaises(rs.RecoveryError):
                rs.load_spec(path)
            write_json(path, {"label": "bad label", "items": []})
            with self.assertRaises(rs.RecoveryError):
                rs.load_spec(path)

    def test_concurrent_probes_get_their_own_atp_scratch(self):
        seen = {}

        async def probe(name, base):
            rs._ATP_SCRATCH.set(base)
            await asyncio.sleep(0.01)
            seen[name] = rollouts.scratch()

        async def main():
            await asyncio.gather(asyncio.create_task(probe("a", Path("/tmp/a"))),
                                 asyncio.create_task(probe("b", Path("/tmp/b"))))

        real = rollouts.scratch
        with rs.per_probe_scratch():
            asyncio.run(main())
        self.assertIs(rollouts.scratch, real)
        self.assertEqual(seen, {"a": Path("/tmp/a"), "b": Path("/tmp/b")})

    def test_redaction(self):
        text = rs.redact("Key=abc-123 (sk-...zcrA) Bearer abcdef sk-live_XYZ api_key=foo")
        for secret in ("abc-123", "zcrA", "abcdef", "XYZ", "foo"):
            self.assertNotIn(secret, text)


class Orchestration(unittest.TestCase):
    """recover_model's retry/budget logic and build_record on a fake task (no sandbox, hub or LLM)."""

    def make_task(self, root: Path) -> rs.Task:
        runs, scratch = root / "runs", root / "scratch"
        with mock.patch.object(rs, "RUNS", runs), mock.patch.object(rs, "SCRATCH", scratch):
            task = rs.Task("official1-test", "syn-test-c01", "unit")
        task.loc = {"metrics": {"student_rate": 0.0, "teacher_clean_rate": 1 / 3, "valid": [5, 3], "reason": "x",
                                "snapshot": "snapshot:s"},
                    "probe": root / "probe", "cand": "cand-c03-r1b4", "round": "round-1", "root": root,
                    "scored": root / "scored", "probed": root / "probed", "packaged": root / "packaged"}
        task.slots = [dict(s, atp_state="completed", attempts=1, f2p="1/1", p2p="1/1", agent_error=None,
                           verifier_error=None, patch_sha256="x", patch_chars=1, attempt_dirs=[]) for s in DASEL]
        task.stage_info = {"sanitizing": True, "packaged_differs": []}
        write_json(task.scratch / "validate.json", {"all_ok": True})
        return task

    def fake_probe(self, task, outcomes):
        """outcomes: per seq, the statuses of the probe's slots in order."""
        calls = []

        async def run_probe(model, replaces, seq, budget=None):
            calls.append((model, list(replaces), seq))
            statuses = outcomes[seq]
            entry = {"id": f"p{seq}", "dir": f"/x/probe-p{seq}", "model": model, "repeats": len(replaces), "seq": seq,
                     "replaces": replaces, "complete": True,
                     "slots": [{"slot": f"{model}-r{i}", "replaces": r, "status": statuses[i], "f2p": "1/1", "p2p": "1/1",
                                "loss": None if statuses[i] in rs.VALID else {"kind": "hub_422", "retry": True}}
                               for i, r in enumerate(replaces)]}
            async with task.edit() as state:
                state["probes"].append(entry)
            if budget is not None:
                budget.probes.append((model, Path(entry["dir"])))
            return entry

        async def grade_and_judge(probe, gate):
            async with task.edit() as state:
                for s in probe["slots"]:
                    if s["status"] in rs.VALID:
                        key = f"{probe['id']}:{s['slot']}"
                        state["grades"][key] = {g: {"status": s["status"]} for g in rs.GRADERS}
                        if s["status"] == "pass":
                            state["verdicts"][key] = {"cheated": False, "categories": []}
        return calls, run_probe, grade_and_judge

    def run_recovery(self, outcomes, limits, *, existing_retry=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = self.make_task(root)
            calls, run_probe, grade_and_judge = self.fake_probe(task, outcomes)
            budget = rs.JobBudget(limits)
            budget.reserve(TEACHER, 2, "initial")
            if existing_retry:  # an earlier run of this label (e.g. before a requeue) already created the retry probe
                retry_dir = task.home / f"probe-{rs.probe_id(task.task_id, TEACHER, 1, task.label, 1)}"
                write_json(retry_dir / "manifest.json", {"tasks": []})
            with mock.patch.object(task, "run_probe", run_probe), \
                    mock.patch.object(task, "grade_and_judge", grade_and_judge), \
                    mock.patch.object(rs, "read_repair", return_value={"present": False, "statuses": {}}), \
                    mock.patch.object(rs, "RUNS", root / "runs"):
                asyncio.run(task.recover_model(TEACHER, ["kimi-r0", "kimi-r3"], budget=budget, limiter=rs.Limiter(),
                                               gate=asyncio.Semaphore(2), retry=True))
                record = task.write_record()
                on_disk = read_json(task.record_path)
            return calls, budget, record, on_disk

    def test_lost_slot_is_retried_once_and_counted(self):
        calls, budget, record, on_disk = self.run_recovery({0: ["pass", "infra"], 1: ["pass"]}, {TEACHER: 10})
        self.assertEqual(calls, [(TEACHER, ["kimi-r0", "kimi-r3"], 0), (TEACHER, ["kimi-r3"], 1)])
        self.assertEqual(budget.reserved[TEACHER], 3)
        self.assertEqual(record["replacements"], {"kimi-r0": "p0:kimi-r0", "kimi-r3": "p1:kimi-r0"})
        self.assertEqual(record["after"]["sanitizing_grader"]["kimi_clean"], 0.6)
        self.assertEqual(record["after"]["recorded_grader"]["valid"], [5, 5])
        self.assertEqual(record["recommendation"], "newly_in_range")
        self.assertEqual(on_disk["recommendation"], "newly_in_range")
        self.assertEqual(record["before"]["kimi_clean"], 1 / 3)

    def test_no_second_retry_and_budget_stops_retries(self):
        calls, _, record, _ = self.run_recovery({0: ["infra", "fail"], 1: ["infra"]}, {TEACHER: 10})
        self.assertEqual(len(calls), 2)  # retried once, not again
        self.assertEqual(record["after"]["recorded_grader"]["valid"], [5, 4])
        self.assertEqual(record["recommendation"], "still_out_of_range")  # 1 clean of 4
        calls, budget, record, _ = self.run_recovery({0: ["pass", "infra"]}, {TEACHER: 2})
        self.assertEqual(len(calls), 1)
        self.assertFalse(budget.entries[-1]["granted"])
        self.assertTrue(any("budget" in note for note in record["notes"]))
        self.assertEqual(record["recommendation"], "still_out_of_range")  # 2 clean of 4 valid

    def test_an_existing_retry_probe_resumes_without_new_budget(self):
        calls, budget, record, _ = self.run_recovery({0: ["pass", "infra"], 1: ["pass"]}, {TEACHER: 2},
                                                     existing_retry=True)
        self.assertEqual(len(calls), 2)
        self.assertTrue(budget.entries[-1].get("resume"))
        self.assertEqual(budget.reserved[TEACHER], 2)
        self.assertTrue(any("one retry" in note for note in record["notes"]))  # persisted in state.json
        self.assertEqual(record["recommendation"], "newly_in_range")


class Export(unittest.TestCase):
    def test_recovered_rollouts_take_the_replaced_slot_names(self):
        recorded = [{"slot": "sc14-r3", "model": STUDENT, "status": "infra", "f2p": "0/17", "patch": ""},
                    {"slot": "sc14-r4", "model": STUDENT, "status": "fail", "f2p": "16/17", "patch": "x"}]
        recovered = {"sc14-r3": {"probe": "/p/probe-9af154c1",
                                 "record": {"slot": "sc14-r0", "model": STUDENT, "status": "fail", "f2p": "15/17",
                                            "patch": "y", "attempt_dir": "/p/a"}}}
        overrides = {"sc14-r4": {"status": "pass", "f2p": "17/17", "info": {"grader": "new"}}}
        out = rs.after_records(recorded, recovered, overrides)
        self.assertEqual([r["slot"] for r in out], ["sc14-r3", "sc14-r4"])
        self.assertEqual((out[0]["status"], out[0]["f2p"], out[0]["patch"]), ("fail", "15/17", "y"))
        self.assertEqual(out[0]["recovery"], {"probe": "/p/probe-9af154c1", "probe_slot": "sc14-r0",
                                              "attempt_dir": "/p/a", "replaces": "sc14-r3"})
        self.assertEqual((out[1]["status"], out[1]["f2p"], out[1]["regrade"]), ("pass", "17/17", {"grader": "new"}))
        self.assertEqual(recorded[1]["status"], "fail")  # inputs are not modified


class ProbeFiles(unittest.TestCase):
    def test_recovery_errors_of_the_last_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            launch = Path(directory) / "models/kimi/shards/0/launch-1-a"
            write_json(launch / "recovery-0.json", {"errors": [{"error": "old"}]})
            write_json(launch / "recovery-2.json", {"errors": [
                {"error": "Cannot recover attempt while owning job may still be active"},
                {"error": "Cannot recover attempt while owning job may still be active"}]})
            self.assertEqual(rs.recovery_errors(Path(directory), TEACHER),
                             ["Cannot recover attempt while owning job may still be active"])
            self.assertEqual(rs.recovery_errors(Path(directory), STUDENT), [])

    def test_owner_job_states(self):
        with tempfile.TemporaryDirectory() as directory:
            write_json(Path(directory) / "started.json", {"job_id": "20573337"})
            done = mock.Mock(stdout="20573337|NODE_FAIL|cpu1-00021\n20573337|RUNNING|cpu1-00135\n")
            with mock.patch.object(rs.os.path, "exists", return_value=True), \
                    mock.patch.object(rs.subprocess, "run", return_value=done):
                info = rs.owner_job(Path(directory))
        self.assertEqual(info, {"job_id": "20573337", "states": ["NODE_FAIL cpu1-00021", "RUNNING cpu1-00135"]})


if __name__ == "__main__":
    unittest.main()
