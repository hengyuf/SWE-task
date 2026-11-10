"""Offline tests of the 2026-11-02 target selection (hengyuf: "放所有正确的kimi rollouts 和所有错误的base rollout"):
the 10 + 10 target dataset (tools/build_targets_x10.py), rollouts.target_records(dataset=) / shown_records,
failure_analyzer.target_source / write_inputs / selection_note / _check, the failure_analysis prompt (unchanged for
"all"), and analyze_target end to end with the sandbox and codex mocked. No sandbox, network or LLM calls.

Run by explicit path from a private scratch directory (never test discovery in a shared directory):
    cd "$(mktemp -d)" && <atp controller python3> -B /lustre/fsw/.../swe-task/synth/tests/test_x10_selection.py -v
"""
import sys

sys.dont_write_bytecode = True
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402
import unittest  # noqa: E402
from unittest import mock  # noqa: E402

import common  # noqa: E402
import failure_analyzer as fa  # noqa: E402
import rollouts  # noqa: E402

X10 = common.S / "data/targets/sc14x10-kimix10-20261102"
ALL_SC14_FAIL = "textual-kitty-key-phases"     # sc14 0/10, kimi 9/10
HALF = "anko-typed-variable-bindings"          # sc14 5/10, kimi 10/10
SHOW = "teacher_pass_student_fail"


def by_model(records, model):
    return [r for r in records if r["model"] == model]


class Dataset(unittest.TestCase):
    def test_x10_records(self):
        for task in (ALL_SC14_FAIL, HALF):
            records = rollouts.target_records(task, dataset=X10)
            self.assertEqual(len(records), 20)
            self.assertEqual(sorted(r["slot"] for r in by_model(records, "sc14")), sorted(f"sc14-r{k}" for k in range(10)))
            kimi = sorted(int(r["slot"].rsplit("-r", 1)[1]) for r in by_model(records, "kimi"))
            self.assertEqual(len(kimi), 10)
            self.assertEqual(kimi[5:], [16, 17, 18, 19, 20])
            self.assertTrue(all(k < 16 for k in kimi[:5]))
            origins = {(r["model"], r["origin"]["name"]) for r in records}
            self.assertEqual(origins, {("sc14", "sc14-orig"), ("sc14", "sc14-rerun1"), ("kimi", "kimi-1m"),
                                       ("kimi", "kimi-256k")})

    def test_default_dataset_unchanged(self):
        records = rollouts.target_records(HALF)
        self.assertEqual(len(records), 10)
        self.assertTrue(all("origin" not in r for r in records))
        x10 = {r["slot"]: r for r in rollouts.target_records(HALF, dataset=X10)}
        for record in records:  # the old slots are the same rollouts, unchanged except "origin"
            other = dict(x10[record["slot"]])
            other.pop("origin")
            self.assertEqual(other, record)

    def test_counts_match_gap_table(self):
        stats = rollouts.pass_stats(rollouts.target_records(HALF, dataset=X10))
        self.assertEqual((stats["sc14"]["pass"], stats["sc14"]["fail"]), (5, 5))
        self.assertEqual((stats["kimi"]["pass"], stats["kimi"]["fail"]), (10, 0))


class Selection(unittest.TestCase):
    def test_shown_records(self):
        records = rollouts.target_records(HALF, dataset=X10)
        shown = rollouts.shown_records(records, SHOW)
        self.assertTrue(all((r["model"], r["status"]) in (("sc14", "fail"), ("kimi", "pass")) for r in shown))
        self.assertEqual(len(shown), 15)
        self.assertEqual(rollouts.shown_records(records, "all"), records)
        with self.assertRaises(ValueError):
            rollouts.shown_records(records, "teacher_pass")

    def test_check_uses_shown_slots(self):
        records = rollouts.target_records(HALF, dataset=X10)
        shown = rollouts.shown_records(records, SHOW)
        a_pass = next(r["slot"] for r in records if r["model"] == "sc14" and r["status"] == "pass")
        a_fail = next(r["slot"] for r in shown if r["model"] == "sc14")
        with tempfile.TemporaryDirectory() as out:
            (Path(out) / "failure_analysis.md").write_text("x" * 2000)
            check = fa._check(records, out, shown)
            pattern = {"name": "p", "category": "c", "description": "d", "teacher_contrast": "t"}
            good = {"patterns": [dict(pattern, student_slots=[a_fail])], "transferable_signature": "s",
                    "excluded": [{"slot": a_pass, "reason": "not shown, still a slot of the task"}]}
            self.assertEqual(check(good), [])
            bad = dict(good, patterns=[dict(pattern, student_slots=[a_pass])])
            self.assertTrue(any("unusable" in problem for problem in check(bad)))
            self.assertEqual(fa._check(records, out)(bad), [])  # "all": every graded slot is usable, as before


class Inputs(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_write_inputs_selection(self):
        records = rollouts.target_records(HALF, dataset=X10)
        shown = rollouts.shown_records(records, SHOW)
        source = {"dataset": str(X10), "show": SHOW}
        fa.write_inputs(self.tmp, common.DEEPSWE / HALF, records, shown, source)
        written = sorted(p.stem for p in (self.tmp / "inputs/rollouts").glob("*.md") if p.name != "index.md")
        self.assertEqual(written, sorted(r["slot"] for r in shown))
        stats = json.loads((self.tmp / "inputs/stats.json").read_text())
        self.assertEqual(stats["pass_stats"], json.loads(json.dumps(rollouts.pass_stats(records))))
        self.assertEqual(stats["pass_stats"]["sc14"]["rate"], 0.5)   # task_builder's target_student_rate
        self.assertEqual(stats["pass_stats"]["kimi"]["rate"], 1.0)   # task_builder's target_teacher_rate
        self.assertEqual(stats["selection"], {"show": SHOW})  # no dataset path for the analyst
        self.assertNotIn(str(X10), (self.tmp / "inputs/stats.json").read_text())
        self.assertEqual(sorted(stats["shown"]["slots"]), sorted(r["slot"] for r in shown))
        self.assertEqual(sorted(x["slot"] for x in stats["not_shown"]),
                         sorted(r["slot"] for r in records if r["model"] == "sc14" and r["status"] == "pass"))
        self.assertEqual(len(stats["origins"]), 20)

    def test_selection_clears_stale_rollouts(self):
        records = rollouts.target_records(HALF, dataset=X10)
        fa.write_inputs(self.tmp, common.DEEPSWE / HALF, records)  # e.g. an earlier, unfinished attempt of every slot
        self.assertEqual(len(list((self.tmp / "inputs/rollouts").glob("*.md"))), 21)
        shown = rollouts.shown_records(records, SHOW)
        fa.write_inputs(self.tmp, common.DEEPSWE / HALF, records, shown, {"dataset": str(X10), "show": SHOW})
        written = sorted(p.stem for p in (self.tmp / "inputs/rollouts").glob("*.md") if p.name != "index.md")
        self.assertEqual(written, sorted(r["slot"] for r in shown))
        self.assertEqual(len(list((self.tmp / "inputs/rollouts").glob("*.patch"))), len(shown))

    def test_render_escapes_nul_bytes(self):
        record = {"slot": "sc14-r0", "status": "fail", "reward": 0, "f2p": "0/1", "p2p": "0/0", "agent_error": None,
                  "verifier_error": None, "failing_tests": [], "patch": "",
                  "output": [{"type": "function_call_output", "output": "a\x00b"}]}
        text = rollouts.render(record)
        self.assertNotIn("\x00", text)
        self.assertIn("a\u2400b", text)

    def test_write_inputs_all_unchanged(self):
        records = rollouts.target_records(HALF)
        fa.write_inputs(self.tmp, common.DEEPSWE / HALF, records)
        stats = json.loads((self.tmp / "inputs/stats.json").read_text())
        self.assertEqual(set(stats), {"pass_stats", "excluded_by_construction"})
        self.assertEqual(len(list((self.tmp / "inputs/rollouts").glob("*.md"))), 11)

    def test_prompt_all_is_the_old_prompt(self):
        fields = dict(task_id=HALF, student="sc14", teacher="kimi", base_commit="abc", cheat_categories="CATS",
                      sandbox="SANDBOX")
        old = common.fill((common.PROMPTS / "failure_analysis.md.pre-x10-selection").read_text(), **fields)
        records = rollouts.target_records(HALF)
        new = common.render("failure_analysis", **fields,
                            rollout_selection=fa.selection_note(records, rollouts.shown_records(records), "all"))
        self.assertEqual(new, old)

    def test_selection_note(self):
        records = rollouts.target_records(ALL_SC14_FAIL, dataset=X10)
        shown = rollouts.shown_records(records, SHOW)
        note = fa.selection_note(records, shown, SHOW)
        self.assertIn("## Rollout selection", note)
        self.assertIn("the 9 kimi rollouts that PASSED (9 of its 10 graded rollouts passed)", note)
        self.assertIn("the 10 sc14 rollouts that FAILED (10 of its 10 graded rollouts failed)", note)
        self.assertIn("k of the 10 failing sc14 slots", note)
        self.assertIn("run base-step0-rep5-out131k-rerun1", note)
        self.assertIn("run teacher-deepswe-20261030, OpenCode context window 262,144 tokens, per-call output cap "
                      "131,072 tokens", note)
        self.assertIn("run partial-selected-super-kimi-20261015T1603Z, OpenCode context window 1,048,576 tokens", note)
        self.assertIn("run base-step0-rep5-out131k, OpenCode context window 262,144 tokens", note)
        self.assertIn("An OpenCode context compaction shows in a transcript as an empty `### [n] user` item", note)
        self.assertIn("starting `## Goal`", note)
        self.assertIn("A NUL byte in a tool output is shown as \u2400.", note)
        self.assertIn("Work only from ./inputs, your working directory and the sandbox", note)
        self.assertNotIn("{", note)
        self.assertNotIn("'pass'", note)
        prompt = common.render("failure_analysis", task_id=ALL_SC14_FAIL, student="sc14", teacher="kimi",
                               base_commit="abc", cheat_categories="CATS", sandbox="## Remote repository sandbox",
                               rollout_selection=note)
        self.assertLess(prompt.index("## Rollout selection"), prompt.index("## Remote repository sandbox"))
        self.assertLess(prompt.index("- `stats.json`: pass counts per model. They count every graded rollout"),
                        prompt.index("## Rollout selection"))


class TargetSource(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.patch = mock.patch.object(fa, "RUNS", self.tmp)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        shutil.rmtree(self.tmp)

    def write(self, value):
        (self.tmp / "r").mkdir(exist_ok=True)
        (self.tmp / "r" / fa.TARGET_SOURCE).write_text(value if isinstance(value, str) else json.dumps(value))

    def test_default(self):
        self.assertEqual(fa.target_source("r"), {"dataset": str(common.TARGETS), "show": "all"})

    def test_valid(self):
        self.write({"dataset": str(X10), "show": SHOW, "requested_by": "x"})
        self.assertEqual(fa.target_source("r")["show"], SHOW)

    def test_invalid_is_an_error(self):
        for bad in ({"dataset": str(X10), "show": "teacher_pass"}, {"show": SHOW},
                    {"dataset": str(self.tmp), "show": SHOW}, "not json", ["list"]):
            self.write(bad)
            with self.assertRaises(ValueError):
                fa.target_source("r")

    def test_analyze_target_end_to_end(self):
        """analyze_target with the sandbox and codex mocked: inputs, prompt and check follow target_source.json."""
        self.write({"dataset": str(X10), "show": SHOW})
        seen = {}

        @contextlib.asynccontextmanager
        async def owned(image, run, purpose):
            seen["owned"] = (image, run, purpose)
            yield types.SimpleNamespace(sid="sid-test")

        async def agent(prompt, out_dir, backend, check):
            seen["prompt"], seen["out"] = prompt, Path(out_dir)
            shown = rollouts.shown_records(rollouts.target_records(HALF, dataset=X10), SHOW)
            slot = next(r["slot"] for r in shown if r["model"] == "sc14")
            (Path(out_dir) / "failure_analysis.md").write_text("y" * 2000)
            result = {"patterns": [{"name": "p", "category": "c", "description": "d", "student_slots": [slot],
                                    "teacher_contrast": "t"}], "transferable_signature": "s", "excluded": []}
            self.assertEqual(check(result), [])
            (Path(out_dir) / "result.json").write_text(json.dumps(result))
            return result

        with mock.patch.object(fa, "owned", owned), mock.patch.object(fa, "agent", agent), \
                mock.patch.object(fa, "write_wrappers", lambda *a, **k: None):
            result = asyncio.run(fa.analyze_target(HALF, "r"))
        self.assertEqual(result["transferable_signature"], "s")
        self.assertEqual(seen["owned"][1:], ("r", "failure-analysis"))
        self.assertIn("## Rollout selection", seen["prompt"])
        self.assertIn("the 10 kimi rollouts that PASSED", seen["prompt"])
        self.assertIn("the 5 sc14 rollouts that FAILED", seen["prompt"])
        stats = json.loads((self.tmp / "r/analysis" / HALF / "inputs/stats.json").read_text())
        self.assertEqual(stats["pass_stats"]["sc14"]["rate"], 0.5)
        self.assertEqual(len(stats["shown"]["slots"]), 15)
        with mock.patch.object(fa, "agent", side_effect=AssertionError("must reuse the validated result")):
            again = asyncio.run(fa.analyze_target(HALF, "r"))
        self.assertEqual(again, result)

    def test_existing_analysis_of_another_selection_is_refused(self):
        """An analysis made without the selection (every slot) is never reused once target_source.json selects."""
        out = self.tmp / "r/analysis" / HALF
        records = rollouts.target_records(HALF)  # the old 5 + 5 analysis of every slot
        fa.write_inputs(out, common.DEEPSWE / HALF, records)
        slot = next(r["slot"] for r in records if r["model"] == "sc14" and r["status"] == "fail")
        (out / "failure_analysis.md").write_text("z" * 2000)
        (out / "result.json").write_text(json.dumps({"patterns": [{
            "name": "p", "category": "c", "description": "d", "student_slots": [slot], "teacher_contrast": "t"}],
            "transferable_signature": "s", "excluded": []}))
        with mock.patch.object(fa, "agent", side_effect=AssertionError("no new analysis")):
            self.assertEqual(asyncio.run(fa.analyze_target(HALF, "r"))["transferable_signature"], "s")  # no file
            self.write({"dataset": str(X10), "show": SHOW})
            with self.assertRaises(RuntimeError):
                asyncio.run(fa.analyze_target(HALF, "r"))
            (out / "result.json").write_text("{}")  # an invalid old result is not redone in place either
            with self.assertRaises(RuntimeError):
                asyncio.run(fa.analyze_target(HALF, "r"))
            (out / "inputs/stats.json").unlink()  # selection mode: leftovers without stats.json are refused too
            with self.assertRaises(RuntimeError):
                asyncio.run(fa.analyze_target(HALF, "r"))


if __name__ == "__main__":
    unittest.main()
