"""Offline checks (no sandbox, API or agent calls). Run in a CPU job:  python3 -B -m unittest discover -s tests -v"""
import sys

sys.dont_write_bytecode = True
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import base64
import contextlib
import hashlib
import json
import os
import shutil
import tempfile
import types
import unittest
from unittest import mock

import httpx

import llm_engine
import rollouts
from common import ATP, DEEPSWE, GYM, HELPER, IMPORTANT, JUDGE, S, SETTINGS, STUDENT, SYNTH, TARGETS, TEACHER, fill
from llm_engine import CODEX, CODEX_CONFIG, CURSOR_MODEL, SECRET_NAME, _agent_env, agent_argv, chat, parse_json
from rollouts import dataset_row, pass_stats, render, target_records, task_ids
from sandbox import parse_report
from task_builder import (removes_solution_code, DOCKERFILE, MATERIALIZE, RUN_END, RUN_START, SWE_RL_GYM, TASK_TOML, GateError, calibrate,
                          canonical_labels, unseen_paths, evolve_outcome, grading_failures, seed_check, round_cap,
                          uniform_failure,
                          hint_lines, leak_tokens, leak_problems, added_counts, dependency_leaks, ignore_rule_edits, rehomed_go_tests,
                          test_sh, validate_task)
from task_builder import (change_windows, densest, identifier_problems, name_match, parse_archive_scan, released_copies,
                          released_names, released_selection, solution_identifiers)
from task_builder import (GO_MODULES, STALE_PREFIX, VANISHED_CHECK, VANISHED_PREFIX, failure_evidence, go_changes,
                          go_modules, go_package_dir, group_summary, report_group, stale_problems, vanished_files,
                          vanished_groups, vanished_problems)

TARGET = "abs-stepped-slices"
ROUTER = "http://127.0.0.1:45678/v1"
HUB_KEY = "hub-key-for-tests"  # stands in for the real key: tests never read or print credentials


def fake_student_client(events: list) -> dict:
    """sys.modules entries for a stand-in `student.client` (the real one starts processes)."""
    client = types.ModuleType("student.client")

    def ensure_router(**kwargs):
        events.append(("ensure_router", kwargs))
        os.environ["SYNTH_STUDENT_BASE_URL"] = ROUTER
        return ROUTER

    def wait_for_workers(n=1, timeout_s=None, **kwargs):
        events.append(("wait_for_workers", n, timeout_s))
        return [{"job_id": "123", "status": "ready", "base_url": "http://10.0.0.1:8000/v1"}]

    client.ensure_router, client.wait_for_workers = ensure_router, wait_for_workers
    package = types.ModuleType("student")
    package.client = client
    return {"student": package, "student.client": client}


class TargetRecords(unittest.TestCase):
    def test_sc14_0_kimi5_task(self):
        records = target_records(TARGET)
        self.assertEqual([record["slot"] for record in records],  # student first, then by slot name
                         ["sc14-r0", "sc14-r1", "sc14-r2", "sc14-r3", "sc14-r4",
                          "kimi-r0", "kimi-r12", "kimi-r4", "kimi-r5", "kimi-r8"])
        stats = pass_stats(records)
        self.assertEqual((stats["sc14"]["pass"], stats["sc14"]["fail"], stats["kimi"]["pass"]), (0, 5, 5))
        self.assertTrue(all(record["output"] and record["status"] in ("pass", "fail") for record in records))
        self.assertEqual({record["task_id"] for record in records}, {TARGET})

    def test_adjudicated_and_infra_slots_are_excluded(self):
        records = {record["slot"]: record for record in target_records("koota-query-predicates")}
        self.assertEqual(records["kimi-r2"]["status"], "grader")  # verifier timeout, adjudicated zero
        stats = pass_stats(records.values())
        self.assertEqual((stats["kimi"]["pass"], stats["kimi"]["fail"], stats["kimi"]["excluded"]), (2, 2, 1))
        self.assertEqual((stats["sc14"]["pass"], stats["sc14"]["fail"], stats["sc14"]["excluded"]), (2, 3, 0))
        self.assertIn("] user\nContinue if you have next steps", render(records["sc14-r0"]))  # OpenCode compaction
        self.assertIn("failing tests: none", render(records["kimi-r1"]))  # kimi pass: nothing failed
        self.assertIn("failing tests: unavailable", render(records["kimi-r11"]))  # kimi fail: grade counts only
        stats = pass_stats(target_records("kombu-virtual-queue-dead-lettering"))
        self.assertEqual((stats["sc14"]["fail"], stats["sc14"]["excluded"]), (3, 2))  # sc14-r2, sc14-r3: infra

    def test_every_task_has_five_slots_per_model(self):
        ids = task_ids()
        self.assertEqual((len(ids), ids), (113, sorted(ids)))
        index = json.loads((TARGETS / "index.json").read_text())
        self.assertEqual(set(index), set(ids))
        for task_id in ids:
            models = [entry["model"] for entry in index[task_id]["records"]]
            self.assertEqual((models.count(STUDENT), models.count(TEACHER), len(models)), (5, 5, 10), task_id)

    def test_records_are_hash_checked_and_dataset_must_be_complete(self):
        records = [{"slot": "kimi-r0", "model": "kimi", "status": "pass"},
                   {"slot": "sc14-r1", "model": "sc14", "status": "fail"}]
        lines = [json.dumps(record, ensure_ascii=True).encode() for record in records]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "records").mkdir()
            path = root / "records/demo.jsonl"
            path.write_bytes(b"".join(line + b"\n" for line in lines))
            (root / "index.json").write_text(json.dumps({"demo": {"task_index": 0, "records": [
                {"slot": record["slot"], "record_sha256": hashlib.sha256(line).hexdigest()}
                for record, line in zip(records, lines)]}}))
            with mock.patch.object(rollouts, "TARGETS", root):
                with self.assertRaises(RuntimeError):  # no COMPLETE.json: the dataset build is unfinished
                    target_records("demo")
                # a finished dataset's COMPLETE.json pins index.json (checked once per process and dataset since
                # 2026-11-02; this test had passed only after another test had set the old process-wide flag)
                (root / "COMPLETE.json").write_text(json.dumps({"files": {"index.json": {
                    "sha256": hashlib.sha256((root / "index.json").read_bytes()).hexdigest()}}}))
                self.assertEqual([record["slot"] for record in target_records("demo")], ["sc14-r1", "kimi-r0"])
                path.write_bytes(path.read_bytes().replace(b'"pass"', b'"fail"'))
                with self.assertRaises(ValueError):
                    target_records("demo")


class Rendering(unittest.TestCase):
    def test_truncates_long_tool_output_head_and_tail(self):
        output = [{"type": "reasoning", "summary": [{"text": "plan"}]},
                  {"type": "function_call", "name": "bash", "arguments": '{"command": "go test ./..."}'},
                  {"type": "function_call_output", "output": "A" * 5000 + "TAIL"},
                  {"type": "message", "content": [{"text": "done"}]}]
        record = {"slot": "sc14-r0", "status": "fail", "reward": 0, "f2p": "0/6", "p2p": "6/6", "failing_tests": ["t1"],
                  "agent_error": None, "verifier_error": None, "patch": "diff", "output": output}
        text = render(record)
        self.assertIn("chars omitted", text)
        self.assertIn("TAIL", text)
        self.assertIn("call bash", text)
        self.assertIn("failing tests: t1", text)
        self.assertLess(len(text), 5000)

    def test_compaction_messages_and_unavailable_test_names(self):
        output = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "summary so far"}]},
                  {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}]
        record = {"slot": "kimi-r0", "status": "pass", "reward": 1, "f2p": "6/6", "p2p": "6/6", "failing_tests": [],
                  "failing_tests_available": False, "agent_error": None, "verifier_error": None, "patch": "",
                  "output": output}
        text = render(record)
        self.assertIn("### [1] user\nsummary so far", text)
        self.assertIn("### [2] assistant\ndone", text)
        self.assertIn("failing tests: unavailable", text)
        self.assertIn("failing tests: none", render(dict(record, failing_tests_available=True)))


class Gates(unittest.TestCase):
    reports = ["/logs/verifier/base-ctrf.json", "/logs/verifier/new-ctrf.json"]

    def test_leak_tokens_skip_formatting_lines(self):
        # litestar-style decorator: docstring, parameter list and imports repeat sibling decorators (2026-10-18)
        patch = "\n".join([
            "--- a/app/routes.py", "+++ b/app/routes.py",
            "+from litestar.handlers.http_handlers.decorators import HTTPRouteHandler",
            "+def options(path: str | None = None, *, after_request_hook_handler: AfterRequestHookHandler = None,",
            "+    before_request_hook_handler: BeforeRequestHookHandler | None = None,",
            "+    response_class_override_for_this_route: type[Response] | None = None,",
            '+    """Route handler decorator for OPTIONS requests that mirrors the get and post decorators.',
            "+    Args:",
            "+        before_request_hook_handler: A sync or async function called before the request is handled",
            '+    """',
            "+    # Answer every OPTIONS request with the allowed methods of the matching path",
            "+    allowed_methods = sorted({method for route in matching_routes for method in route.methods})",
            "+    return HTTPRouteHandler(path=path, http_method=HttpMethod.OPTIONS, allowed=allowed_methods)",
            ""])
        self.assertEqual(set(leak_tokens(patch)["app/routes.py"]), {  # the def line names the new function: kept
            "def options(path: str | None = None, *, after_request_hook_handler: AfterRequestHookHandler = None,",
            "return HTTPRouteHandler(path=path, http_method=HttpMethod.OPTIONS, allowed=allowed_methods)",
            "allowed_methods = sorted({method for route in matching_routes for method in route.methods})"})
        self.assertEqual(added_counts(patch), {"app/routes.py": 11})

    def test_leak_problems_need_one_concentrated_block(self):
        lines = [f"distinctive_line_{i} = compute_something_long({i}, other)" for i in range(6)]
        per_source, added = {"src/a.py": lines}, {"src/a.py": 30}
        block = {"/opt/vendor/a.py": [(100 + i, line) for i, line in enumerate(lines[:4])]}
        self.assertEqual(len(leak_problems(block, per_source, added)), 1)  # 4/6 within 60 lines: a leak
        scattered = {"/app/src/b.py": [(1 + 400 * i, line) for i, line in enumerate(lines)]}
        self.assertEqual(leak_problems(scattered, per_source, added), [])  # spread over 2000 lines: not a leak
        self.assertEqual(leak_problems({"/x": [(5, lines[0]), (6, lines[1])]}, per_source, added), [])  # < 3
        three = {"/x": [(5 + i, line) for i, line in enumerate(lines[:3])]}
        self.assertEqual(leak_problems(three, per_source, added), [])  # 3 lines: no complete implementation
        boiler = ["rows = await cursor.fetchall()  # three generic lines", "for key, job_bytes, status in rows:",
                  "job = self.deserialize(job_bytes, status)"]  # official1-aiomonitor c04, 2026-10-19
        self.assertEqual(leak_problems({"/app/saq/queue/postgres.py": [(417 + i, b) for i, b in enumerate(boiler)]},
                                       {"saq/queue/postgres.py": boiler}, {"saq/queue/postgres.py": 12}), [])

    def test_moved_code_is_not_distinctive(self):
        # official1-testem cand-c01-r1b1 (2026-10-18): a Farm.ts refactor moves and re-indents existing doWork code, so
        # the starting state "held" 10/20 of the solution's distinctive lines; lines the patch also removes do not count
        source = "packages/jest-worker/src/Farm.ts"
        moved = ["const computeWorkerKey = this._computeWorkerKey;",
                 "worker = hash == null ? null : this._cacheKeys[hash];",
                 "this._taskQueue.enqueue(task, worker.getWorkerId());"]
        new = ["const pendingTasksByWorker = this._pendingTasksFor(workerId);",
               "this._scheduleRetryWithBackoff(task, attempt + 1, maxRetries);",
               "return this._dispatchToLeastBusyWorker(pendingTasksByWorker);"]
        patch = "\n".join([f"--- a/{source}", f"+++ b/{source}", "@@ -10,6 +10,9 @@"] +
                          [f"-    {line}" for line in moved] + [f"+      {line}" for line in moved + new] + [""])
        tokens = leak_tokens(patch)[source]
        self.assertEqual(sorted(tokens), sorted(new))
        start_state = {f"/app/{source}": [(20 + i, line) for i, line in enumerate(moved)]}  # C keeps the moved code
        self.assertEqual(leak_problems(start_state, {source: tokens}, added_counts(patch)), [])
        copy = {"/opt/vendor/Farm.js": [(40 + i, line) for i, line in enumerate(moved + new)]}  # a genuine copy
        # 3 distinctive lines are below the line criterion's floor of 4 (2026-10-19); identifier_problems covers
        # such small solutions (its identifiers are absent from the starting state)
        self.assertEqual(leak_problems(copy, {source: tokens}, added_counts(patch)), [])
        bigger = new + ["const retriedTaskCount = this._retryCounts.get(task) ?? 0;"]
        copy4 = {"/opt/vendor/Farm.js": [(40 + i, line) for i, line in enumerate(moved + bigger)]}
        self.assertEqual(leak_problems(copy4, {source: bigger}, {source: 30}),
                         [f"/opt/vendor/Farm.js holds 4/4 distinctive lines of {source} within 60 lines"])
        # identifier evidence: only what the solution declares (hengyuf 2026-10-20); the methods it calls are declared
        # elsewhere (in C, or by a dependency)
        self.assertEqual(solution_identifiers(patch)[source], ["pendingTasksByWorker"])

    def test_dependency_tree_leaks_are_unfixable(self):
        problems = [
            "/app/node_modules/metro/node_modules/jest-worker/build/base/BaseWorkerPool.js holds 6/9 distinctive lines of "
            "packages/jest-worker/src/base/BaseWorkerPool.ts within 60 lines",
            "/usr/lib/python3/dist-packages/pip/_vendor/requests/models.py holds 12/13 distinctive lines of requests/models.py "
            "within 120 lines",
            "/app/src/other.py holds 4/6 distinctive lines of src/a.py within 60 lines",  # the repo itself: author can fix
            "base suite fails on the starting state: ['x']"]
        self.assertEqual(dependency_leaks(problems), problems[:2])

    def test_start_patch_may_not_edit_ignore_rules(self):
        patch = ("diff --git a/.gitignore b/.gitignore\n--- a/.gitignore\n+++ b/.gitignore\n@@ -1 +1 @@\n-/node_modules\n"
                 "+/node_modules/*\ndiff --git a/packages/x/.gitignore b/packages/x/.gitignore\n"
                 "diff --git a/src/a.ts b/src/a.ts\n")
        self.assertEqual(ignore_rule_edits(patch), [".gitignore", "packages/x/.gitignore"])
        self.assertEqual(ignore_rule_edits("diff --git a/src/a.ts b/src/a.ts\n"), [])

    def test_rehomed_go_tests_are_rejected(self):
        start = ("diff --git a/util/settings/settings_test.go b/util/settings/settings_test.go\n"
                 "-func TestGetIgnoreResourceUpdatesOverrides(t *testing.T) {\n-func TestKeep(t *testing.T) {\n")
        moved = ("diff --git a/util/settings/restored_test.go b/util/settings/restored_test.go\n"
                 "+func TestGetIgnoreResourceUpdatesOverrides(t *testing.T) {\n")
        self.assertEqual(rehomed_go_tests(start, moved), [("util/settings/settings_test.go", "util/settings/restored_test.go",
                                                           "TestGetIgnoreResourceUpdatesOverrides")])
        in_place = ("diff --git a/util/settings/settings_test.go b/util/settings/settings_test.go\n"
                    "+func TestGetIgnoreResourceUpdatesOverrides(t *testing.T) {\n")
        self.assertEqual(rehomed_go_tests(start, in_place), [])  # restored where it was: the grader resets that file
        both = moved + in_place  # test.patch touches the original file too: also fine
        self.assertEqual(rehomed_go_tests(start, both), [])

    def test_calibrate_f2p_p2p(self):
        empty = {"suites": {"base-ctrf.json": {"p1": "passed", "p2": "passed"}, "new-ctrf.json": {"n1": "failed"}}}
        solved = [{"suites": {"base-ctrf.json": {"p1": "passed", "p2": "passed"},
                              "new-ctrf.json": {"n1": "passed", "n2": "passed"}}}] * 3
        f2p, p2p, problems = calibrate(empty, solved, self.reports)
        self.assertEqual((f2p, p2p, problems), (["n1", "n2"], ["p1", "p2"], []))

    def test_calibrate_rejects_broken_base_and_flaky_solution(self):
        empty = {"suites": {"base-ctrf.json": {"p1": "failed"}, "new-ctrf.json": {}}}
        solved = [{"suites": {"base-ctrf.json": {"p1": "passed"}, "new-ctrf.json": {"n1": "passed"}}},
                  {"suites": {"base-ctrf.json": {"p1": "passed"}, "new-ctrf.json": {"n1": "failed"}}}]
        _f2p, _p2p, problems = calibrate(empty, solved, self.reports)
        self.assertTrue(any("base suite fails" in p for p in problems))
        self.assertTrue(any("not passing with the solution" in p for p in problems))

    def test_calibrate_skipped_tests_are_not_failures(self):
        # aiohttp on Python 3.12: environment skips in the base suite must not reject the task (2026-10-18)
        empty = {"suites": {"base-ctrf.json": {"p1": "passed", "s1": "skipped"},
                            "new-ctrf.json": {"n1": "failed", "k1": "skipped"}}}
        solved = [{"suites": {"base-ctrf.json": {"p1": "passed", "s1": "skipped"},
                              "new-ctrf.json": {"n1": "passed", "n2": "passed", "k1": "skipped"}}}] * 3
        f2p, p2p, problems = calibrate(empty, solved, self.reports)
        self.assertEqual((f2p, p2p, problems), (["n1", "n2"], ["p1"], []))  # skips enter neither list
        flaky = solved[:2] + [{"suites": {"base-ctrf.json": {"p1": "passed"},
                                          "new-ctrf.json": {"n1": "skipped", "n2": "passed"}}}]
        _f2p, _p2p, problems = calibrate(empty, flaky, self.reports)
        self.assertTrue(any("not passing with the solution" in p and "n1" in p for p in problems))

    def test_hints_and_leak_tokens(self):
        patch = ("--- a/x.go\n+++ b/x.go\n+\t// TODO: implement stepping\n+\treturn nil\n"
                 "+\tstep := computeStepFromSliceExpression(node, environment)\n")
        self.assertEqual(len(hint_lines(patch)), 1)
        self.assertEqual(leak_tokens(patch), {"x.go": ["step := computeStepFromSliceExpression(node, environment)"]})

    def test_reversed_patch_labels_are_normalized_for_the_grader(self):
        # `git diff -R` style headers (seen in official1-narwhals cand-c02): git apply takes them, but the DeepSWE
        # grader's patch_paths finds no file, so it never resets an agent's edits before applying test.patch.
        reversed_patch = ("diff --git b/pkg/t_test.py a/pkg/t_test.py\nindex 1..2 100644\n--- b/pkg/t_test.py\n"
                          "+++ a/pkg/t_test.py\n@@ -1,2 +1,3 @@\n x = 1\n--- b/not-a-header\n+def test_new():\n")
        self.assertEqual(unseen_paths(reversed_patch), ["pkg/t_test.py"])
        fixed = canonical_labels(reversed_patch)
        self.assertEqual(unseen_paths(fixed), [])
        self.assertEqual(fixed.splitlines()[:4], ["diff --git a/pkg/t_test.py b/pkg/t_test.py", "index 1..2 100644",
                                                  "--- a/pkg/t_test.py", "+++ b/pkg/t_test.py"])
        self.assertEqual(fixed.splitlines()[4:], reversed_patch.splitlines()[4:])  # hunks (incl. a "--- b/" line) kept
        normal = "diff --git a/x.go b/x.go\n--- a/x.go\n+++ b/x.go\n@@ -1 +1 @@\n-a\n+b\n"
        self.assertEqual(canonical_labels(normal), normal)

    def test_report_parsers_follow_grade_config(self):
        with tempfile.TemporaryDirectory() as directory:
            ctrf = Path(directory) / "r.json"
            ctrf.write_text(json.dumps({"results": {"tests": [
                {"name": "TestA", "suite": "example.com/pkg", "status": "passed"},
                {"name": "TestB", "suite": "example.com/pkg", "status": "failed"}]}}))
            self.assertEqual(parse_report(ctrf, {"format": "ctrf", "node_id": "suite.name"}),
                             {"example.com/pkg.TestA": "passed", "example.com/pkg.TestB": "failed"})
            junit = Path(directory) / "r.xml"
            junit.write_text('<testsuite><testcase classname="tests.test_x" name="test_ok"/>'
                             '<testcase classname="tests.test_x" name="test_bad"><failure message="x"/></testcase></testsuite>')
            self.assertEqual(parse_report(junit, {"format": "junit"}),
                             {"tests.test_x.test_ok": "passed", "tests.test_x.test_bad": "failed"})

    def test_verifier_frame_is_preserved(self):
        original = (DEEPSWE / TARGET / "tests/test.sh").read_text()
        spliced = test_sh("echo synthetic-block", DEEPSWE / TARGET)
        self.assertIn("echo synthetic-block", spliced)
        self.assertEqual(spliced.split(RUN_START)[0], original.split(RUN_START)[0])
        self.assertEqual(spliced.split(RUN_END)[1], original.split(RUN_END)[1])

    def test_templates(self):
        self.assertIn("-exec rm -rf {} +", MATERIALIZE)
        self.assertIn(r"\( -name __pycache__", MATERIALIZE)
        dockerfile = fill(DOCKERFILE, base_image="img", repo_url="https://github.com/o/r", sha="a" * 40)
        self.assertIn("--mount=type=bind,source=start.patch", dockerfile)
        self.assertEqual(fill("{{a}} {{b}}", a="{{b}}", b="x"), "{{b}} x")  # values are never re-scanned
        with self.assertRaises(ValueError):
            fill("{{missing}}")


class MaterializeGoModuleCache(unittest.TestCase):
    """MATERIALIZE purges the repo's own module versions but keeps nested modules (official1-opa, 2026-10-20)."""

    def test_nested_module_survives(self):
        import shutil, subprocess, tempfile
        if not shutil.which("git"):
            self.skipTest("git missing")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            app, cache, bin_dir = tmp / "app", tmp / "gomod", tmp / "bin"
            app.mkdir(); bin_dir.mkdir()
            run = lambda *a: subprocess.run(a, cwd=app, check=True, capture_output=True)
            run("git", "init", "-q", "-b", "main")
            (app / "go.mod").write_text("module github.com/acme/tool\n\ngo 1.22\n")
            run("git", "add", "-A")
            run("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
            own = cache / "cache/download/github.com/acme/tool/@v"
            own_v2 = cache / "cache/download/github.com/acme/tool/v2/@v"
            nested = cache / "cache/download/github.com/acme/tool/api/genpb/@v"
            for d in (own, own_v2, nested):
                d.mkdir(parents=True); (d / "v0.1.0.zip").write_text("x")
            (cache / "github.com/acme/tool@v0.1.0").mkdir(parents=True)
            (cache / "github.com/acme/tool/api/genpb@v0.1.0").mkdir(parents=True)
            go = bin_dir / "go"
            go.write_text(f"#!/bin/sh\n[ \"$1 $2\" = \"env GOMODCACHE\" ] && echo {cache}\nexit 0\n"); go.chmod(0o755)
            script = (MATERIALIZE.replace("/app", str(app)).replace("/root/.cache/go-build", str(tmp / "gb"))
                      .replace("/tmp/jest_* /tmp/vitest*", f"{tmp}/jest_* {tmp}/vitest*"))
            env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
            result = subprocess.run(["bash", "-c", script, "materialize", ""], capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(own.exists()); self.assertFalse(own_v2.exists())
            self.assertFalse((cache / "github.com/acme/tool@v0.1.0").exists())
            self.assertTrue(nested.exists())                                  # the nested module's download cache stays
            self.assertTrue((cache / "github.com/acme/tool/api/genpb@v0.1.0").exists())


class EvolveLeakHits(unittest.TestCase):
    """Leak-scan hits inside /app count only for mask tasks (official1-absslices c04-r1b4, 2026-10-19)."""

    def test_removes_solution_code(self):
        solution = "\n".join(["--- a/pkg/vm/vm.go", "+++ b/pkg/vm/vm.go", "@@ -1 +1,2 @@", " x", "+y", ""])
        mask = "\n".join(["--- a/pkg/vm/vm.go", "+++ b/pkg/vm/vm.go", "@@ -1,2 +1 @@", " x", "-y", ""])
        unrelated = "\n".join(["--- a/README.md", "+++ b/README.md", "@@ -1,2 +1 @@", " a", "-b", ""])
        self.assertTrue(removes_solution_code(mask, solution))
        self.assertFalse(removes_solution_code("", solution))              # evolve: empty start.patch
        self.assertFalse(removes_solution_code(unrelated, solution))       # deletions elsewhere only
        added = "\n".join(["--- /dev/null", "+++ b/pkg/new.go", "@@ -0,0 +1 @@", "+z", ""])
        self.assertFalse(removes_solution_code(added, solution))           # new files only


class MaterializeBinaries(unittest.TestCase):
    """MATERIALIZE deletes ignored executables built from the unmasked code (official1-ytt /app/yq, 2026-10-19)."""

    def test_ignored_binaries_and_bin_dirs_removed(self):
        import shutil, subprocess, tempfile
        if not shutil.which("git") or not shutil.which("od"):
            self.skipTest("git/od missing")
        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp) / "app"
            app.mkdir()
            elf = b"\x7fELF" + b"\0" * 60
            run = lambda *a: subprocess.run(a, cwd=app, check=True, capture_output=True)
            run("git", "init", "-q", "-b", "main")
            (app / ".gitignore").write_text("yq\nbin/\n*.so\nnode_modules/\nhelper.sh\n")
            (app / "main.go").write_text("package main\n")
            run("git", "add", "-A")
            run("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
            (app / "yq").write_bytes(elf)                       # ignored binary: removed
            (app / "bin").mkdir(); (app / "bin" / "tool").write_bytes(elf)   # ignored bin/ dir: removed
            (app / "libx.so").write_bytes(elf)                  # shared library: kept
            (app / "node_modules" / "x").mkdir(parents=True)
            (app / "node_modules" / "x" / "prog").write_bytes(elf)  # dependency tree: kept
            (app / "helper.sh").write_text("#!/bin/sh\n")      # ignored script, not ELF: kept
            script = (MATERIALIZE.replace("/app", str(app)).replace("/root/.cache/go-build", str(Path(tmp) / "gb"))
                      .replace("/tmp/jest_* /tmp/vitest*", f"{tmp}/jest_* {tmp}/vitest*")
                      .replace("command -v go", "false"))
            result = subprocess.run(["bash", "-c", script, "materialize", ""], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((app / "yq").exists())
            self.assertFalse((app / "bin").exists())
            self.assertTrue((app / "libx.so").exists())
            self.assertTrue((app / "node_modules" / "x" / "prog").exists())
            self.assertTrue((app / "helper.sh").exists())
            self.assertIn("removed ignored binary yq", result.stderr)


class ReleasedCopies(unittest.TestCase):
    """Released copies of removed code in archives and package caches (official1-testem c03, 2026-10-18)."""
    WORKER = "packages/jest-worker/src/base/BaseWorkerPool.ts"
    SOLUTION = "\n".join([
        f"diff --git a/{WORKER} b/{WORKER}", f"--- a/{WORKER}", f"+++ b/{WORKER}", "@@ -100,3 +100,12 @@",
        " export default class BaseWorkerPool {",
        "+  // Schedule a force exit in case the worker fails to exit gracefully",
        "+  const workerExitPromises = this._workers.map(async worker => {",
        "+    const forceExitTimeout = setTimeout(() => worker.forceExit(), FORCE_EXIT_DELAY);",
        "+    clearTimeout(forceExitTimeout);",
        "+  });",
        "+  const workerExits = await Promise.all(workerExitPromises);",
        "+  throw new Error(`Worker ${id} failed to exit gracefully`);",
        "+  return workerExits.reduce((result, forceExited) => result || forceExited, false);",
        " }",
        "@@ -300,2 +309,3 @@",
        " function tail() {",
        "+  const emptyMethod = () => {};",
        " }",
        "diff --git a/docs/README.md b/docs/README.md", "--- a/docs/README.md", "+++ b/docs/README.md", "@@ -1 +1,2 @@",
        " # Jest worker",
        "+Call workerExitPromises from the documentation to restartGracefully.",
        ""])

    def test_solution_identifiers_keep_names_and_messages_of_code_files(self):
        found = solution_identifiers(self.SOLUTION)
        self.assertEqual(list(found), [self.WORKER])  # the documentation file is not code
        self.assertEqual(found[self.WORKER], [  # declared names and messages (hengyuf 2026-10-20): no comment words,
            "emptyMethod", "failed to exit gracefully", "forceExitTimeout", "forceExited",  # no setTimeout, no name it
            "workerExitPromises", "workerExits"])  # only uses (FORCE_EXIT_DELAY, _workers, forceExit); forceExited is
        # a parameter of the arrow function (declared, 2026-10-21)
        python = ("--- a/m.py\n+++ b/m.py\n@@ -1 +1,6 @@\n+from mypkg.helpers import load_baseline_entries\n"
                  "+def __init__(self):  # remember_the_comment_word\n+    '''Docstring mentions docstring_only_name.'''\n"
                  "+    if isinstance(value, dict):\n+        raise ValueError(f\"cannot merge {value} into the baseline\")\n"
                  "+    return compute_baseline_digest(value)\n")
        self.assertEqual(solution_identifiers(python), {"m.py": ["cannot merge", "into the baseline"]})  # a call

    def test_change_windows_cover_the_changed_region(self):
        windows = change_windows(self.SOLUTION)
        self.assertEqual(windows[self.WORKER], 210)  # first added line 101, last 310
        self.assertEqual(change_windows("--- a/x.py\n+++ b/x.py\n@@ -1 +1,2 @@\n a\n+b\n"), {"x.py": 60})

    def test_identifier_signal_flags_one_concentrated_released_file(self):
        per_source = {self.WORKER: ["emptyMethod", "forceExitTimeout", "workerExitPromises", "workerExits"]}
        windows = {self.WORKER: 154}
        released = "/root/.yarn/berry/cache/jest-worker-npm-29.7.0-4d3567fed6-10.zip!/node_modules/jest-worker/" \
                   "build/base/BaseWorkerPool.js"
        hits = {released: [(31, "emptyMethod"), (128, "workerExitPromises"), (136, "forceExitTimeout"),
                           (141, "forceExitTimeout"), (145, "workerExits")],
                "/root/.cache/other/index.js": [(10, "workerExits"), (900, "emptyMethod")]}  # 1 per window: noise
        self.assertEqual(identifier_problems(hits, per_source, windows), [
            f"{released} holds 4/4 distinctive identifiers of {self.WORKER} within 154 lines"])
        scattered = {released: [(1, "emptyMethod"), (400, "workerExitPromises"), (800, "forceExitTimeout")]}
        self.assertEqual(identifier_problems(scattered, per_source, windows), [])  # never 3 inside one window
        minified = {"/root/.local/share/pnpm/store/v10/files/21/3b5f": [  # a bundled compiler: 2 lines, 900 KB apart
            (362, "emptyMethod", 10_000), (362, "forceExitTimeout", 400_000), (363, "workerExits", 910_000)]}
        self.assertEqual(identifier_problems(minified, per_source, windows), [])
        self.assertEqual(densest(minified, per_source, windows)[0]["best"], 1)
        compact = {released: [(31, "emptyMethod", 1240), (128, "workerExitPromises", 5100), (136, "forceExitTimeout",
                                                                                              5400)]}
        self.assertEqual(len(identifier_problems(compact, per_source, windows)), 1)  # 3/4 within 154 lines and 15 KB
        two = {released: [(128, "workerExitPromises"), (145, "workerExits")]}
        self.assertEqual(identifier_problems(two, per_source, windows), [])  # 2 < max(3, half of 4)
        self.assertEqual(densest(two, per_source, windows)[0]["best"], 2)

    def test_released_names_from_the_manifest(self):
        manifest = "\n".join([
            "F\tpackage.json\t{", 'F\tpackage.json\t  "name": "@jest/monorepo",', 'F\tpackage.json\t  "private": true,',
            'F\tpackage.json\t  "workspaces": ["packages/*", "examples/*", "!packages/skipped"]', "F\tpackage.json\t}",
            "J\tpackage.json\t@jest/monorepo\tprivate", "J\tpackages/jest-worker/package.json\tjest-worker\t",
            "J\tpackages/jest-get-type/package.json\t@jest/get-type\t", "J\tpackages/skipped/package.json\tskipped-pkg\t",
            "J\texamples/react/package.json\texample-react\tprivate", "J\te2e/fixture/package.json\tsome-fixture\t",
            "F\tpyproject.toml\t[project]", 'F\tpyproject.toml\tname = "ibis-framework"',
            "F\tpyproject.toml\t[[project.authors]]", 'F\tpyproject.toml\tname = "author-name"',
            "P\tibis/__init__.py", "P\ttests/__init__.py",
            "G\tgo.mod\tgithub.com/argoproj/argo-cd/v3", ""])
        released = released_names("jestjs/jest", manifest)
        self.assertEqual(released["names"], ["@jest/get-type", "argo-cd", "ibis", "ibis-framework", "jest",
                                             "jest-get-type", "jest-worker"])
        self.assertEqual(released["go_modules"], ["github.com/argoproj/argo-cd/v3"])
        self.assertEqual(released["workspace_dirs"], {"packages/jest-get-type": "@jest/get-type",
                                                      "packages/jest-worker": "jest-worker"})

    def test_released_selection_reads_archives_caches_and_package_dirs(self):
        released = {"names": ["@jest/types", "aiohttp", "jest-worker"], "go_modules": ["github.com/argoproj/argo-cd/v3"],
                    "workspace_dirs": {}}
        integrity = "sha512-" + base64.b64encode(bytes(range(64))).decode()
        discovered = "\n".join([
            "X\tunzip", "A\t1200\t/root/.yarn/berry/cache/jest-worker-npm-29.7.0-4d3567fed6-10.zip",
            "A\t900\t/root/.yarn/berry/cache/@jest-types-npm-29.6.3-a584ca999d-10.zip",
            "A\t50\t/root/.yarn/berry/cache/lodash-npm-4.17.21-6382451519-10.zip",
            "A\t70\t/root/go/pkg/mod/cache/download/github.com/argoproj/argo-cd/v3/@v/v3.1.0.zip",
            "A\t80\t/root/.cache/pip/wheels/ab/aiohttp-3.9.5-cp312-cp312-linux_x86_64.whl",
            "C\t/root/.npm/_cacache",
            f"N\tmake-fetch-happen:request-cache:https://registry.npmjs.org/jest-worker/-/jest-worker-27.5.1.tgz\t{integrity}",
            f"N\tmake-fetch-happen:request-cache:https://registry.npmjs.org/lodash/-/lodash-4.17.21.tgz\t{integrity}",
            "W\t/root/.cache/pip/http-v2/a/b/c.body\taiohttp,aiohttp-3.9.5.dist-info",
            "W\t/root/.cache/pip/http-v2/d/e/f.body\tyarl,yarl-1.9.4.dist-info",
            "R\t/root/.cache", "R\t/usr/lib/node_modules",
            "D\t/app/packages/jest-worker", "D\t/app/node_modules/metro/node_modules/jest-worker",
            "D\t/root/.cache/uv/archive-v0/x/aiohttp", "D\t/usr/local/lib/python3.12/site-packages/aiohttp", ""])
        selection = released_selection(discovered, released)
        content = "/root/.npm/_cacache/content-v2/sha512/00/01/" + bytes(range(64)).hex()[4:]
        self.assertEqual([(kind, path) for kind, path, _ in selection["unpack"]], [
            ("zip", "/root/.yarn/berry/cache/jest-worker-npm-29.7.0-4d3567fed6-10.zip"),
            ("zip", "/root/.yarn/berry/cache/@jest-types-npm-29.6.3-a584ca999d-10.zip"),
            ("zip", "/root/go/pkg/mod/cache/download/github.com/argoproj/argo-cd/v3/@v/v3.1.0.zip"),
            ("zip", "/root/.cache/pip/wheels/ab/aiohttp-3.9.5-cp312-cp312-linux_x86_64.whl"),
            ("blob", content), ("blob", "/root/.cache/pip/http-v2/a/b/c.body")])
        self.assertEqual(selection["roots"], ["/app/node_modules/metro/node_modules/jest-worker", "/root/.cache",
                                              "/usr/lib/node_modules", "/usr/local/lib/python3.12/site-packages/aiohttp"])
        self.assertEqual(name_match("@jest-types-npm-29.6.3-a584ca999d-10.zip", ["@jest/types"]), "@jest/types")
        self.assertEqual(name_match("lodash-npm-4.17.21-6382451519-10.zip", ["@jest/types", "jest-worker"]), "")

    def test_archive_scan_output_maps_members_to_their_archive(self):
        output = "\n".join([
            "U\t1\t/root/.yarn/berry/cache/jest-worker-npm-29.7.0-4d3567fed6-10.zip\t0", "P\tCHILD_MESSAGE_END",
            "H\t/tmp/leak-archives/1/node_modules/jest-worker/build/base/BaseWorkerPool.js:128:5120:const workerExits = 1;",
            "I\t/tmp/leak-archives/1/node_modules/jest-worker/build/base/BaseWorkerPool.js:31:1240:emptyMethod",
            "I\t/root/.cache/uv/archive-v0/x/pkg/mod.py:7:280:emptyMethod", "E\tdone"])
        scan = parse_archive_scan(output)
        member = "/root/.yarn/berry/cache/jest-worker-npm-29.7.0-4d3567fed6-10.zip!/node_modules/jest-worker/build/" \
                 "base/BaseWorkerPool.js"
        self.assertEqual(scan["unpacked"], {"1": ("/root/.yarn/berry/cache/jest-worker-npm-29.7.0-4d3567fed6-10.zip",
                                                  "0")})
        self.assertEqual(scan["present"], {"CHILD_MESSAGE_END"})
        self.assertEqual(scan["fixed"], {member: [(128, "const workerExits = 1;", 5120)]})
        self.assertEqual(scan["identifiers"], {member: [(31, "emptyMethod", 1240)],
                                               "/root/.cache/uv/archive-v0/x/pkg/mod.py": [(7, "emptyMethod", 280)]})

    def test_dependency_leaks_cover_archives_and_caches(self):
        source = "packages/jest-worker/src/workers/processChild.ts"
        released = [
            f"/root/.yarn/berry/cache/jest-worker-npm-29.7.0-4d3567fed6-10.zip!/node_modules/jest-worker/build/workers/"
            f"processChild.js holds 4/7 distinctive lines of {source} within 60 lines",
            f"/root/.yarn/berry/cache/jest-worker-npm-27.5.1-1c110b5894-10.zip!/node_modules/jest-worker/build/base/"
            f"BaseWorkerPool.js holds 4/4 distinctive identifiers of {source} within 154 lines",
            f"/root/.npm/_cacache/content-v2/sha512/ab/cd/ef01!/package/build/index.js holds 3/4 distinctive "
            f"identifiers of {source} within 60 lines",
            f"/root/.cache/pip/http-v2/a/b/c.body!/aiohttp/cookiejar.py holds 9/12 distinctive lines of {source} "
            f"within 80 lines",
            f"/root/.cache/uv/archive-v0/x/aiohttp/cookiejar.py holds 9/12 distinctive lines of {source} within 80 lines",
            f"/root/.local/share/pnpm/store/v10/files/ab/cdef holds 5/6 distinctive lines of {source} within 60 lines",
            f"/root/go/pkg/mod/github.com/o/r@v1.0.0/x.go holds 5/6 distinctive lines of {source} within 60 lines",
            f"/app/.yarn/cache/jest-worker-npm-29.7.0-4d3567fed6-10.zip!/build/index.js holds 4/7 distinctive lines "
            f"of {source} within 60 lines"]
        kept = [
            f"/app/.cache/generated.ts holds 4/7 distinctive lines of {source} within 60 lines",  # inside the repo
            f"/app/packages/jest-worker/build/index.js holds 4/7 distinctive identifiers of {source} within 60 lines",
            "archive leak scan incomplete: scan (exit 124, timed out True)"]
        self.assertEqual(dependency_leaks(released + kept), released)

    def test_released_copies_end_to_end(self):
        class Box:
            def __init__(self, outputs):
                self.outputs, self.uploads, self.commands = outputs, {}, []

            async def put_text(self, text, remote):
                self.uploads[remote] = text

            async def sh(self, command, *, cwd="/app", timeout=1800, check=True):
                self.commands.append((command, cwd, timeout))
                return types.SimpleNamespace(stdout=self.outputs[len(self.commands) - 1], exit_code=0, timed_out=False)

        zip_path = "/root/.yarn/berry/cache/jest-worker-npm-29.7.0-4d3567fed6-10.zip"
        member = "/tmp/leak-archives/1/node_modules/jest-worker/build/base/BaseWorkerPool.js"
        outputs = [
            "J\tpackages/jest-worker/package.json\tjest-worker\t\nE\tdone",
            f"X\tunzip\nA\t1000\t{zip_path}\nR\t/root/.cache\nE\tdone",
            "\n".join([f"U\t1\t{zip_path}\t0", "P\t_workers", "P\tforceExit", "P\tforceExited", "P\tFORCE_EXIT_DELAY",
                       f"H\t{member}:103:4100:const workerExitPromises = this._workers.map(async worker => {{",
                       f"H\t{member}:110:4400:const workerExits = await Promise.all(workerExitPromises);",
                       f"H\t{member}:120:4800:failed to exit gracefully",
                       f"I\t{member}:31:1240:emptyMethod", f"I\t{member}:103:4106:workerExitPromises",
                       f"I\t{member}:105:4200:forceExitTimeout", f"I\t{member}:110:4406:workerExits", "E\tdone"])]
        box = Box(outputs)
        report = asyncio.run(released_copies(box, "jestjs/jest", self.SOLUTION))
        self.assertEqual(box.uploads["/tmp/leak-select"], f"zip\t{zip_path}\n")
        self.assertEqual(box.uploads["/tmp/leak-roots"], "/root/.cache\n")
        self.assertIn("emptyMethod\n", box.uploads["/tmp/leak-idents"])
        self.assertEqual(box.uploads["/tmp/leak-literals"], "failed to exit gracefully\n")
        self.assertEqual([timeout for _, _, timeout in box.commands], [300, 900, 1500])
        shown = f"{zip_path}!/node_modules/jest-worker/build/base/BaseWorkerPool.js"
        self.assertEqual(report["problems"], [
            f"{shown} holds 5/5 distinctive identifiers of {self.WORKER} within 210 lines"])  # 2/5 lines: below need
        self.assertEqual(dependency_leaks(report["problems"]), report["problems"])
        box = Box(["J\tpackage.json\tx\t\nE\tdone", "A\t1\t/x.zip"])  # discovery killed before its end
        report = asyncio.run(released_copies(box, "o/x", self.SOLUTION))
        self.assertEqual(report["problems"], ["archive leak scan incomplete: discover (exit 0, timed out False)"])
        self.assertEqual(dependency_leaks(report["problems"]), [])

    def test_leak_scan_gate_adds_released_copies(self):
        import task_builder
        zip_path = "/root/.yarn/berry/cache/jest-worker-npm-29.7.0-4d3567fed6-10.zip"
        member = "/tmp/leak-archives/1/node_modules/jest-worker/build/base/BaseWorkerPool.js"
        outputs = {task_builder.ARCHIVE_MANIFEST: "E\tdone",
                   task_builder.ARCHIVE_DISCOVER: f"A\t1000\t{zip_path}\nE\tdone",
                   task_builder.ARCHIVE_SCAN: "\n".join([f"U\t1\t{zip_path}\t0", "P\t_workers", "P\tforceExit",
                                                          "P\tforceExited", "P\tFORCE_EXIT_DELAY"] + [
                       f"I\t{member}:{line}:{line * 40}:{name}" for line, name in (
                           (31, "emptyMethod"), (103, "workerExitPromises"), (105, "forceExitTimeout"),
                           (110, "workerExits"))]
                                                         + ["E\tdone"])}
        repo_copy = "/app/packages/jest-worker/build/base/BaseWorkerPool.js"
        commands = []

        class Box:
            async def put_text(self, text, remote):
                pass

            async def sh(self, command, *, cwd="/app", timeout=1800, check=True):
                commands.append(command)
                stdout = outputs.get(command) or "\n".join(  # LEAK_SCAN: the repo's own (ignored) build output
                    f"{repo_copy}:{100 + i}:{line}" for i, line in enumerate(leak_tokens(ReleasedCopies.SOLUTION)[
                        ReleasedCopies.WORKER][:4]))
                return types.SimpleNamespace(stdout=stdout, exit_code=0, timed_out=False)

        @contextlib.asynccontextmanager
        async def owned(snapshot, *, run, purpose):
            yield Box()

        build = types.SimpleNamespace(run="official1-testem", repo="jestjs/jest")
        with mock.patch.object(task_builder, "owned", owned):
            problems = asyncio.run(task_builder.leak_scan(build, "snapshot:x", self.SOLUTION))
        self.assertEqual(len(commands), 4)  # LEAK_SCAN, then the three archive-scan commands in the same sandbox
        released = (f"{zip_path}!/node_modules/jest-worker/build/base/BaseWorkerPool.js holds 4/5 distinctive "
                    f"identifiers of {self.WORKER} within 210 lines")
        self.assertEqual(problems, sorted([released, f"{repo_copy} holds 4/6 distinctive lines of {self.WORKER} "
                                                     f"within 60 lines"]))
        self.assertEqual(dependency_leaks(problems), [released])  # the repo's own copy stays an authoring problem


class StaleAndVanishedTests(unittest.TestCase):
    """stale_tests() on C (official1-opa, 2026-10-20): grouped feedback (c04 lost its retry to the 10-ID list) and test
    files/packages that pass on A but report nothing on C although they still exist (c01's index package)."""
    ENGINE = "github.com/cerbos/cerbos/internal/engine"
    SERVER = "github.com/cerbos/cerbos/internal/server"
    INDEX = "github.com/cerbos/cerbos/internal/ruletable/index"

    def test_report_group_on_real_ids(self):
        cases = {  # one ID per format seen in runs/*/build/*/*/env/baseline-* and cand-*/grade/c-baseline
            f"{self.ENGINE}.TestCheck/case_34/lenient_scope_search=false/schema_enforcement=ENFORCEMENT_NONE":
                ("go", self.ENGINE),
            "github.com/hashicorp/hcl.TestDecode_interface/escape_backslash.hcl": ("go", "github.com/hashicorp/hcl"),
            "github.com/evanphx/json-patch/v5.TestAllCases/Case_7": ("go", "github.com/evanphx/json-patch/v5"),
            "github.com/evanphx/json-patch/v5/internal/json.ExampleDecoder_Decode_stream":
                ("go", "github.com/evanphx/json-patch/v5/internal/json"),
            "github.com/deepnoodle-ai/risor/v2/pkg/vm.TestFunctions/function(x=\"a\",_y=\"b\")_{_x_+_y_+_\"c\"_}(\"W\",_\"X\")":
                ("go", "github.com/deepnoodle-ai/risor/v2/pkg/vm"),
            "tests.test_worker.TestHttpRedis.test_abort": ("py", "tests/test_worker.py"),
            "tests.test_config.test_parse_options": ("py", "tests/test_config.py"),
            "tests.unit.test_app.test_set_env_flags[None-None-False-LITESTAR_PDB-pdb_on_exception]":
                ("py", "tests/unit/test_app.py"),
            "tests.unit.test_cli.test_cli.test_info_command@cli_autodiscovery": ("py", "tests/unit/test_cli/test_cli.py"),
            "ibis.common.tests.test_numeric.test_normalize_failing[3.14-10-0]": ("py", "ibis/common/tests/test_numeric.py"),
            "tests.test_http_parser.test_c_parser_error_snippet_at_buffer_end_request[POST / HTTP/1.1\r\nHost: x]":
                ("py", "tests/test_http_parser.py"),
            "tests/test_x.py::TestA::test_b[1-2]": ("py", "tests/test_x.py"),
            "test/core/Relation.test.ts: Relation Tests > should init store with withStore":
                ("js", "test/core/Relation.test.ts"),
            "packages/jest-reporters/src/__tests__/SummaryReporter.test.js:  > snapshots all have results (no update)":
                ("js", "packages/jest-reporters/src/__tests__/SummaryReporter.test.js"),
            "packages/jest-haste-map/src/lib/__tests__/isIgnorableFileError.test.ts isIgnorableFileError does not ignore "
            "EACCES on win32": ("js", "packages/jest-haste-map/src/lib/__tests__/isIgnorableFileError.test.ts"),
            "packages/babel-jest/src/__tests__/index.ts Babel Jest transforms": ("js", "packages/babel-jest/src/__tests__/index.ts"),
            "packages/jest-mock/src/__tests__/index.test.ts suite execution": ("js", "packages/jest-mock/src/__tests__/index.test.ts"),
            "build: yarn build:js": None,  # vitest's build entry: no file, never guessed
            "environment build:js": None,
            "github.com/acme/bundler.TestBundle/entry.js": ("go", "github.com/acme/bundler"),  # not a JS file
        }
        for test_id, group in cases.items():
            self.assertEqual(report_group(test_id), group, test_id)

    def test_vanished_groups(self):
        index = {f"{self.INDEX}.TestActions/case_{i}" for i in range(3)}
        engine = {f"{self.ENGINE}.TestCheck/case_{i}" for i in range(5)}
        small = {"github.com/cerbos/cerbos/internal/namer.TestA", "github.com/cerbos/cerbos/internal/namer.TestB"}
        stable = index | engine | small
        now = {f"{self.ENGINE}.TestCheck/case_0": "failed", "github.com/cerbos/cerbos/internal/namer.TestA": "passed"}
        self.assertEqual(vanished_groups(stable, stable, now), ({("go", self.INDEX): sorted(index)}, False))
        vanished, silent = vanished_groups(stable, stable, {})  # nothing ran at all; namer has < 3 stable tests
        self.assertEqual((set(vanished), silent), ({("go", self.INDEX), ("go", self.ENGINE)}, True))
        jest = "packages/jest-mock/src/__tests__/index.test.ts"  # Jest: a file that fails to load
        tests = {f"{jest} moduleMocker getMetadata case {i}" for i in range(4)}
        broken = {f"{jest} suite execution": "failed"}
        self.assertEqual(vanished_groups(tests, tests, broken), ({("js", jest): sorted(tests)}, True))
        self.assertEqual(vanished_groups(tests, tests, dict(broken, **{f"{jest} new case": "passed"})), ({}, False))
        failing_on_a = f"{jest} flaky case"  # ran on A (failing): the file loads on C
        self.assertEqual(vanished_groups(tests, tests | {failing_on_a}, {failing_on_a: "failed"}), ({}, False))
        py = {f"tests.test_x.TestX.test_{i}" for i in range(3)}  # pytest collection error: classname "" + module
        other = {"tests.test_y.test_a"}
        self.assertEqual(vanished_groups(py | other, py | other, {"tests.test_x": "failed", "tests.test_y.test_a": "passed"}),
                         ({("py", "tests/test_x.py"): sorted(py)}, False))

    def test_grouped_stale_feedback_names_every_package(self):
        # official1-opa c04 attempt 0: 24 engine IDs sorted before 82 server IDs, so stale[:10] showed only the engine
        engine = [f"{self.ENGINE}.TestCheck", f"{self.ENGINE}.TestCheck/case_34"] + [
            f"{self.ENGINE}.TestCheck/case_34/lenient_scope_search={a}/schema_enforcement=ENFORCEMENT_{b}/strict={c}"
            for a in ("false", "true") for b in ("NONE", "WARN", "REJECT") for c in ("false", "true", "x", "y")][:22]
        server = [f"{self.SERVER}.TestServer/store=bundle_local/api/bundlev2/{tls}/{net}/{api}/test/pgt_case_{i:02d}"
                  for tls in ("with_tls", "without_tls") for net in ("tcp", "uds") for api in ("grpc", "grpc_over_http", "http")
                  for i in range(7)][:82]
        stable = set(engine + server) | {f"{self.INDEX}.TestOk"}
        now = {test_id: "failed" for test_id in engine + server} | {f"{self.INDEX}.TestOk": "passed"}
        [problem] = stale_problems(stable, now)
        self.assertTrue(problem.startswith(STALE_PREFIX + " 106 tests in 2 files/packages: "))
        self.assertIn(f"{self.SERVER} (82 tests, e.g. TestServer/store=bundle_local/", problem)
        self.assertIn(f"{self.ENGINE} (24 tests, e.g. TestCheck/case_34/lenient_scope_search=false/", problem)
        self.assertNotIn("TestCheck;", problem)  # parents (failing because a subtest fails) are not examples
        self.assertLess(len(problem), 2000)
        self.assertEqual(stale_problems(stable, {t: "passed" for t in stable}), [])
        many = {f"github.com/acme/tool/pkg/p{i:02d}.TestVeryLongName_{'x' * 80}/case_{j}" for i in range(15) for j in range(3)}
        [crowded] = stale_problems(many | {"build: yarn build:js"}, {t: "failed" for t in many} | {"build: yarn build:js": "failed"})
        self.assertLess(len(crowded), 2000)
        self.assertIn(" more (", crowded)  # the remaining groups are counted, never dropped silently
        [odd] = stale_problems({"build: yarn build:js", "x"}, {"build: yarn build:js": "failed", "x": "skipped"})
        self.assertEqual(odd, STALE_PREFIX + " 2 tests in 2 files/packages: build: yarn build:js; x")  # unknown: as is

    def test_group_summary_budget(self):
        groups = {("py", f"tests/test_{i}.py"): [f"tests.test_{i}.test_{j}" for j in range(i + 1)] for i in range(12)}
        text = group_summary(groups, budget=300)
        self.assertTrue(text.startswith("tests/test_11.py (12 tests, e.g. test_0; test_1)"))
        self.assertLess(len(text), 400)
        self.assertRegex(text, r"and \d+ more \(\d+ tests\)$")
        self.assertEqual(group_summary({("py", "tests/test_0.py"): ["tests.test_0.test_0"]}),
                         "tests/test_0.py (1 test, e.g. test_0)")

    def test_shell_scripts_find_modules_and_test_files(self):
        import subprocess
        if not shutil.which("bash") or not shutil.which("grep"):
            self.skipTest("bash/grep missing")
        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp) / "app"
            files = {
                "go.mod": "module github.com/acme/tool // root\n\ngo 1.22\n",
                "sub/go.mod": 'module "github.com/acme/tool/sub"\n',
                "vendor/x/go.mod": "module github.com/vendored/x\n",
                "pkg/a/a.go": "package a\n",
                "pkg/a/a_test.go": 'package a\n\nimport "testing"\n\nfunc TestA(t *testing.T) {}\n',
                "pkg/b/b_test.go": "package b\n\nfunc helper() {}\n",  # tests all removed: nothing to report
                "pkg/c/c.go": 'package c\n\nimport (\n\talias "github.com/acme/tool/pkg/a"\n)\n',
                "pkg/c/c_test.go": "package c\n\nfunc ExampleC() {}\n",
                "tests/test_x.py": "import os\n\n\nclass TestX:\n    async def test_one(self):\n        pass\n",
                "tests/test_empty.py": "import os\n\nFIXTURE = 1\n",
                "src/a.test.ts": "describe('a', () => {\n  it.each([1])('x', () => {});\n});\n",
                "src/helper.test.ts": "export const contest = 1;\n"}
            for name, text in files.items():
                (app / name).parent.mkdir(parents=True, exist_ok=True)
                (app / name).write_text(text)
            run = lambda script: subprocess.run(["bash", "-c", script.replace("cd /app", f"cd {app}").replace(
                "/tmp/vanished-", f"{tmp}/vanished-")], capture_output=True, text=True).stdout
            listing = run(GO_MODULES)
            self.assertEqual(sorted(listing.splitlines()), sorted(["M\tgithub.com/acme/tool\t.",
                                                                   "M\tgithub.com/acme/tool/sub\tsub", "E\tdone"]))
            self.assertEqual(go_modules(listing), {"github.com/acme/tool": ".", "github.com/acme/tool/sub": "sub"})
            Path(tmp, "vanished-groups").write_text("".join(f"{kind}\t{path}\n" for kind, path in [
                ("go", "pkg/a"), ("go", "pkg/b"), ("go", "pkg/c"), ("go", "pkg/gone"), ("py", "tests/test_x.py"),
                ("py", "tests/test_empty.py"), ("py", "tests/gone.py"), ("js", "src/a.test.ts"),
                ("js", "src/helper.test.ts")]))
            Path(tmp, "vanished-imports").write_text('"github.com/acme/tool/pkg/a"\n')
            self.assertEqual(run(VANISHED_CHECK).splitlines(), [
                "F\tpkg/a\tpkg/a/a_test.go", "F\tpkg/c\tpkg/c/c_test.go", "I\tpkg/c\tpkg/c/c.go",
                "F\ttests/test_x.py\ttests/test_x.py", "F\tsrc/a.test.ts\tsrc/a.test.ts", "E\tdone"])

    def test_go_paths_and_changes(self):
        modules = {"github.com/acme/tool": ".", "github.com/acme/tool/engine/v3": "engine"}
        self.assertEqual(go_package_dir("github.com/acme/tool", modules), ".")
        self.assertEqual(go_package_dir("github.com/acme/tool/pkg/index", modules), "pkg/index")
        self.assertEqual(go_package_dir("github.com/acme/tool/engine/v3/pkg/hook", modules), "engine/pkg/hook")
        self.assertIsNone(go_package_dir("github.com/hashicorp/hcl", modules))  # tested from the module cache
        patch = "".join(f"diff --git a/{path} b/{path}\n" for path in [
            "pkg/index/index_test.go", "pkg/core/core.go", "engine/pkg/sync/sync.go", "pkg/a/testdata/case.json",
            "main.go", "docs/README.md"])
        directories, imports, module_files = go_changes(patch, modules)
        self.assertEqual(directories, {"pkg/index", "pkg/core", "engine/pkg/sync", "pkg/a", ".", "docs"})
        self.assertEqual(imports, {'"github.com/acme/tool/pkg/core"', '"github.com/acme/tool/engine/v3/pkg/sync"',
                                   '"github.com/acme/tool"'})
        self.assertFalse(module_files)
        self.assertTrue(go_changes("diff --git a/engine/go.sum b/engine/go.sum\n", modules)[2])

    def test_vanished_files_need_existing_tests_and_for_go_a_link_to_start_patch(self):
        listing = "M\tgithub.com/acme/tool\t.\nM\tgithub.com/acme/tool/engine/v3\tengine\nE\tdone"
        check = "\n".join(["F\tpkg/index\tpkg/index/index_test.go", "F\tpkg/index\tpkg/index/bitmap_test.go",
                           "F\tengine/pkg/hook\tengine/pkg/hook/hook_test.go", "F\tpkg/user\tpkg/user/user_test.go",
                           "I\tpkg/user\tpkg/user/user.go", "F\ttests/test_broken.py\ttests/test_broken.py", "E\tdone"])

        class Box:
            def __init__(self, outputs):
                self.outputs, self.uploads, self.commands = outputs, {}, []

            async def put_text(self, text, remote):
                self.uploads[remote] = text

            async def sh(self, command, *, cwd="/app", timeout=1800, check=True):
                self.commands.append(command)
                return types.SimpleNamespace(stdout=self.outputs[command], exit_code=0, timed_out=False)

        groups = {("go", "github.com/acme/tool/pkg/index"): ["t"] * 3,      # its test file edited by start.patch
                  ("go", "github.com/acme/tool/engine/v3/pkg/hook"): ["t"] * 3,  # untouched: cut-off run
                  ("go", "github.com/acme/tool/pkg/user"): ["t"] * 3,       # imports a changed package
                  ("go", "github.com/other/dep"): ["t"] * 3,                # other module: not checked
                  ("py", "tests/test_gone.py"): ["t"] * 3,                  # deleted by start.patch
                  ("py", "tests/test_broken.py"): ["t"] * 3}
        patch = ("diff --git a/pkg/index/index_test.go b/pkg/index/index_test.go\n"
                 "diff --git a/pkg/core/core.go b/pkg/core/core.go\n"
                 "diff --git a/tests/test_gone.py b/tests/test_gone.py\ndeleted file mode 100644\n")
        box = Box({GO_MODULES: listing, VANISHED_CHECK: check})
        found = asyncio.run(vanished_files(box, groups, patch, silent=False))
        self.assertEqual(found, {
            ("go", "github.com/acme/tool/pkg/index"): ["pkg/index/index_test.go (edited by start.patch)",
                                                       "pkg/index/bitmap_test.go"],
            ("go", "github.com/acme/tool/pkg/user"): ["pkg/user/user_test.go"],
            ("py", "tests/test_broken.py"): ["tests/test_broken.py"]})
        self.assertEqual(sorted(box.uploads["/tmp/vanished-groups"].splitlines()), sorted([
            "go\tpkg/index", "go\tengine/pkg/hook", "go\tpkg/user", "py\ttests/test_gone.py", "py\ttests/test_broken.py"]))
        self.assertEqual(box.uploads["/tmp/vanished-imports"], '"github.com/acme/tool/pkg/core"\n')
        silent = asyncio.run(vanished_files(Box({GO_MODULES: listing, VANISHED_CHECK: check}), groups, patch, silent=True))
        self.assertIn(("go", "github.com/acme/tool/engine/v3/pkg/hook"), silent)  # nothing ran: no cut-off
        incomplete = Box({GO_MODULES: listing, VANISHED_CHECK: check.rsplit("\n", 1)[0]})  # killed: report nothing
        self.assertEqual(asyncio.run(vanished_files(incomplete, groups, patch, silent=False)), {})
        gopath = Box({GO_MODULES: "M\tgithub.com/evanphx/json-patch/v5\tv5\nE\tdone",
                      VANISHED_CHECK: "F\t.\tpatch_test.go\nE\tdone"})  # evanphx/json-patch v4: no root go.mod
        self.assertEqual(asyncio.run(vanished_files(gopath, {("go", "github.com/evanphx/json-patch"): ["t"] * 3},
                                                    "diff --git a/patch_test.go b/patch_test.go\n", silent=False,
                                                    repo="evanphx/json-patch")),
                         {("go", "github.com/evanphx/json-patch"): ["patch_test.go (edited by start.patch)"]})

    def test_vanished_problem_text_and_evidence(self):
        jest, py = "packages/jest-mock/src/__tests__/index.test.ts", ("py", "tests/test_x.py")
        groups = {("js", jest): [f"{jest} moduleMocker case {i}" for i in range(4)],
                  py: [f"tests.test_x.test_{i}" for i in range(3)], ("go", self.INDEX): [f"{self.INDEX}.TestA"] * 3}
        files = {("js", jest): [jest], py: ["tests/test_x.py (edited by start.patch)"]}
        report = {f"{jest} suite execution": ("failed", "\x1b[1m● \x1b[22mTest suite failed to run\n\n    Cannot find "
                                                        "resolver for: 5f8c\n      at getResolver (testWorker.js:452:11)"),
                  "tests.test_x": ("failed", "ImportError while importing test module '/app/tests/test_x.py'.\nTraceback:"
                                             "\ntests/test_x.py:3: in <module>\n    from lib.core import gone\n"
                                             "E   ImportError: cannot import name 'gone' from 'lib.core'"),
                  f"{jest} moduleMocker case 0": ("failed", "on A: not evidence")}
        evidence = failure_evidence(files, {f"{jest} moduleMocker case 0"}, report)
        self.assertEqual(evidence, {("js", jest): "C reports 'suite execution': Cannot find resolver for: 5f8c",
                                    py: "C reports 'tests.test_x': ImportError: cannot import name 'gone' from 'lib.core'"})
        [problem] = vanished_problems(groups, files, evidence)
        self.assertTrue(problem.startswith(VANISHED_PREFIX + ": "))
        self.assertIn(f"{jest} (4 pass on A, e.g. moduleMocker case 0; moduleMocker case 1; test files {jest}; C reports "
                      "'suite execution': Cannot find resolver for: 5f8c)", problem)
        self.assertIn("tests/test_x.py (3 pass on A, e.g. test_0; test_1; test files tests/test_x.py (edited by "
                      "start.patch); C reports 'tests.test_x': ImportError", problem)
        self.assertNotIn(self.INDEX, problem)  # no test files on C: start.patch deleted them, which is fine
        self.assertEqual(vanished_problems(groups, {}), [])
        [silent] = vanished_problems(groups, files, silent=True)
        self.assertIn("start.patch probably breaks the build or the test runner itself", silent)

    def test_stale_tests_end_to_end(self):
        import task_builder
        go_ctrf = {"format": "ctrf", "node_id": "suite.name"}

        def ctrf(rows):
            return json.dumps({"results": {"tests": [{"name": name, "suite": suite, "status": status}
                                                     for suite, name, status in rows]}})
        a_rows = [(self.INDEX, f"TestActions/case_{i}", "passed") for i in range(4)] + [
            (self.ENGINE, f"TestCheck/case_{i}", "passed") for i in range(3)]
        c_rows = [(self.ENGINE, "TestCheck/case_0", "failed"), (self.ENGINE, "TestCheck/case_1", "passed"),
                  (self.ENGINE, "TestCheck/case_2", "passed")]  # the index package did not compile
        start = ("diff --git a/internal/ruletable/index/index_test.go b/internal/ruletable/index/index_test.go\n"
                 "diff --git a/internal/engine/engine.go b/internal/engine/engine.go\n")
        commands = []

        class Box:
            async def put(self, local, remote):
                pass

            async def put_text(self, text, remote):
                pass

            async def get(self, remote, local):
                Path(local).parent.mkdir(parents=True, exist_ok=True)
                Path(local).write_text(ctrf(c_rows))

            async def sh(self, command, *, cwd="/app", timeout=1800, check=True):
                commands.append(command)
                stdout = {task_builder.GO_MODULES: "M\tgithub.com/cerbos/cerbos\t.\nE\tdone",
                          task_builder.VANISHED_CHECK: "F\tinternal/ruletable/index\tinternal/ruletable/index/index_test.go"
                                                       "\nE\tdone"}.get(command, "")
                return types.SimpleNamespace(stdout=stdout, exit_code=0, timed_out=False)

        @contextlib.asynccontextmanager
        async def owned(snapshot, *, run, purpose):
            yield Box()

        with tempfile.TemporaryDirectory() as tmp:
            root, cand = Path(tmp) / "build", Path(tmp) / "cand"
            (root / "env").mkdir(parents=True)
            for i in (0, 1):
                (root / f"env/baseline-{i}").write_text(ctrf(a_rows))
            (root / "env/run_tests.sh").write_text("exit 0\n")
            build = types.SimpleNamespace(root=root, run="official1-opa", repo="cerbos/cerbos", grade_cfg=go_ctrf)
            with mock.patch.object(task_builder, "owned", owned):
                problems = asyncio.run(task_builder.stale_tests(build, cand, {"snapshot": "snapshot:c"}, start))
            self.assertTrue((cand / "grade/c-baseline").exists())
        self.assertEqual(len(problems), 2)
        self.assertEqual(problems[0], f"{STALE_PREFIX} 1 test in 1 file/package: {self.ENGINE} (1 test, e.g. "
                                      f"TestCheck/case_0)")
        self.assertEqual(problems[1], f"{VANISHED_PREFIX}: {self.INDEX} (4 pass on A, e.g. TestActions/case_0; "
                                      f"TestActions/case_1; test files internal/ruletable/index/index_test.go (edited by "
                                      f"start.patch))")
        self.assertEqual(commands[-2:], [task_builder.GO_MODULES, task_builder.VANISHED_CHECK])  # same sandbox

    def test_gates_pass_start_patch_to_stale_tests(self):
        import task_builder
        build = task_builder.Build("unit-test-run", TARGET, "deepnoodle-ai/risor")
        seen = {}

        async def grade(source, task_dir, patch, *, run, out_dir):
            new = {"n1": "passed", "n2": "passed"} if patch else {"n1": "failed", "n2": "failed"}
            return {"suites": {"base-ctrf.json": {"p1": "passed"}, "new-ctrf.json": new}}

        async def stale_tests(b, cand, mat, start_patch=""):
            seen["start_patch"] = start_patch
            return ["stale!"]

        with tempfile.TemporaryDirectory() as tmp:
            author_dir, cand = Path(tmp) / "author-0", Path(tmp) / "cand"
            author_dir.mkdir()
            for name, text in {"run_tests.sh": "true\n", "test.patch": "", "solution.patch": "x",
                               "start.patch": "diff --git a/a.go b/a.go\n"}.items():
                (author_dir / name).write_text(text)
            with mock.patch.object(task_builder, "grade", grade), mock.patch.object(task_builder, "stale_tests", stale_tests):
                gate = asyncio.run(task_builder.gates(build, cand, author_dir, {"snapshot": "s", "base_commit": "c"}))
        self.assertEqual(seen["start_patch"], "diff --git a/a.go b/a.go\n")
        self.assertEqual(gate["problems"], ["stale!"])

    def test_real_opa_reports(self):
        root = S / "synth/runs/official1-opa/build/opa-rego-rule-profiling/cerbos__cerbos"
        if not (root / "cand-c01/grade/c-baseline").exists():
            self.skipTest("official1-opa reports not available")
        cfg = {"format": "ctrf", "node_id": "suite.name"}
        a = [parse_report(root / f"env/baseline-{i}", cfg) for i in (0, 1)]
        stable = {t for t, s in a[0].items() if s == "passed" and a[1].get(t) == "passed"}
        vanished, silent = vanished_groups(stable, set(a[0]) | set(a[1]), parse_report(root / "cand-c01/grade/c-baseline", cfg))
        self.assertEqual({group: len(tests) for group, tests in vanished.items()}, {("go", self.INDEX): 143})
        self.assertFalse(silent)


class Seeds(unittest.TestCase):
    """hengyuf 2026-10-17: evolution is seeded by task validity, not by pass rates."""

    @staticmethod
    def record(slot, status, verifier_error=None):
        return {"slot": slot, "model": slot.split("-")[0], "status": status, "reward": 1.0 if status == "pass" else 0.0,
                "f2p": "0/0", "p2p": "0/0", "failing_tests": [], "agent_error": None, "verifier_error": verifier_error,
                "patch": "", "output": [], "attempt_dir": None}

    def scored(self, records):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory)
        rollouts.write_rollouts(directory / "rollouts", records)
        return directory

    def test_pass_rates_and_acceptance_do_not_matter(self):
        good = self.scored([self.record(f"sc14-r{i}", "fail") for i in range(5)] +
                           [self.record(f"kimi-r{i}", "pass") for i in range(5)])
        for accept, rate in ((True, 0.2), (False, 1.0), (False, 0.0)):
            self.assertEqual(seed_check(good, {"valid": [5, 5], "accept": accept, "student_rate": rate}), "")

    def test_grading_failures_exclude_the_task(self):
        broken = "Verifier did not produce /logs/verifier/reward.json"  # narwhals cand-c02
        records = ([self.record("sc14-r0", "infra", broken), self.record("sc14-r1", "pass")] +
                   [self.record(f"sc14-r{i}", "fail") for i in range(2, 5)] +
                   [self.record(f"kimi-r{i}", "pass") for i in range(5)])
        scored = self.scored(records)
        self.assertEqual(grading_failures(scored, {}), ["sc14-r0"])  # read back from rollouts/<slot>.md
        self.assertIn("verifier produced no grade for 1 rollout(s): sc14-r0", seed_check(scored, {"valid": [4, 5]}))
        self.assertEqual(grading_failures(scored, {"grading_failures": []}), [])  # score() records them directly

    def test_platform_failures_count_only_toward_inconclusive(self):
        records = ([self.record("kimi-r0", "infra")] + [self.record(f"kimi-r{i}", "pass") for i in range(1, 5)] +
                   [self.record(f"sc14-r{i}", "fail") for i in range(5)])  # e.g. a hub 422: no verifier error
        scored = self.scored(records)
        self.assertEqual(seed_check(scored, {"valid": [5, 4]}), "")
        self.assertIn("fewer than 3 valid rollouts", seed_check(scored, {"valid": [1, 3]}))

    def test_uniform_failure_marks_broken_tests(self):
        def rec(slot, status, f2p, p2p, patch="diff"):
            return dict(self.record(slot, status), f2p=f2p, p2p=p2p, patch=patch)
        broken = [rec(f"sc14-r{i}", "fail", "1/22", "126/431") for i in range(4)] + \
                 [rec(f"kimi-r{i}", "fail", "1/22", "126/431") for i in range(3)] + [rec("kimi-r4", "fail", "0/22", "431/431", "")]
        self.assertIn("all 7 graded rollouts", uniform_failure(broken))
        self.assertIn("tests cannot run normally", seed_check(Path("/nonexistent"), {"valid": [4, 4], "grading_failures": [],
                                                                                        "uniform_failure": uniform_failure(broken)}))
        mixed = broken[:6] + [rec("kimi-r3", "pass", "22/22", "431/431")]
        self.assertIsNone(uniform_failure(mixed))
        full_p2p = [rec(f"sc14-r{i}", "fail", "0/22", "431/431") for i in range(5)]  # same F2P failure, P2P intact
        self.assertIsNone(uniform_failure(full_p2p))

    def test_evolve_prompt_keeps_the_original_wording_for_rejected_parents(self):
        metrics = {"target_student_rate": 0.2, "target_teacher_rate": 1.0}
        self.assertEqual(evolve_outcome(dict(metrics, accept=False)),
                         "It was not accepted: either the pass rates are off (target: student 0.2, teacher\n1.0) or "
                         "the student fails differently from how it fails the target.")
        self.assertIn("It was accepted", evolve_outcome(dict(metrics, accept=True)))
        template = (SYNTH / "prompts/evolve_plan.md").read_text()
        self.assertIn("({{teacher}}). {{outcome}}\n", template)
        self.assertNotIn("{{target_s}}", template)


class RoundCap(unittest.TestCase):
    def test_max_rounds_file_lowers_rounds(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        build = types.SimpleNamespace(root=root)
        self.assertEqual(round_cap(build, 3), 3)  # no file
        (root / "max_rounds.json").write_text('{"max_rounds": 2}')
        self.assertEqual(round_cap(build, 3), 2)
        self.assertEqual(round_cap(build, 1), 1)  # never raises --rounds
        for bad in ('{"max_rounds": -1}', '{"max_rounds": true}', '{"max_rounds": "2"}', "not json"):
            (root / "max_rounds.json").write_text(bad)
            self.assertEqual(round_cap(build, 3), 3)


class SandboxCapacity(unittest.TestCase):
    """A full OpenSandbox cluster (pods rejected for nemo.nvidia.com/sandbox-slot) is waited out, not fatal."""

    def run_create(self, errors):
        import sandbox
        calls, sleeps = [], []

        class Client:
            async def create(self, **kwargs):
                calls.append(kwargs)
                if errors:
                    raise errors.pop(0)
                return {"id": "sid"}

        async def sleep(seconds):
            sleeps.append(seconds)
        result = asyncio.run(sandbox._create(Client(), {"image": "img"}, ttl=60, network="deny",
                                             metadata={"purpose": "test"}, sleep=sleep))
        return result, calls, sleeps

    def test_capacity_rejections_are_retried_with_backoff(self):
        full = RuntimeError("SandboxApiException: Pod x entered Failed phase (OutOfnemo.nvidia.com/sandbox-slot): "
                            "Pod was rejected: Node didn't have enough resource: nemo.nvidia.com/sandbox-slot")
        result, calls, sleeps = self.run_create([full, full])
        self.assertEqual(result, {"id": "sid"})
        self.assertEqual((len(calls), sleeps), (3, [30, 60]))

    def test_other_errors_are_not_retried(self):
        with self.assertRaises(RuntimeError):
            self.run_create([RuntimeError("SandboxApiException: HTTP 401 unauthorized")])


class HubKeys(unittest.TestCase):
    """2026-10-18: NVAPI_KEY_NEW first, NVAPI_KEY_OFFICIAL once it is used up; the active NAME lives in a file."""

    def test_active_key_selection(self):
        import common
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        env = tmp / "env"
        env.write_text("NVAPI_KEY=old-spent-key-000000\nNVAPI_KEY_NEW='new-key-111111'\nNVAPI_KEY_OFFICIAL=official-key-222222\n"
                       "NVAPI_KEY_OFFICIAL_1=official1-key-333333\nNVAPI_KEY_OFFICIAL_2=official2-key-444444\n"
                       "NVAPI_KEY_NEW_3=new3-key-555555\n")
        active = tmp / "hub_key_active.txt"
        with mock.patch.object(common, "USER_ENV", env), mock.patch.object(common, "HUB_KEY_ACTIVE", active):
            self.assertEqual((common.hub_key_name(), common.api_keys()), ("NVAPI_KEY_NEW", ["new-key-111111"]))
            active.write_text("NVAPI_KEY_OFFICIAL\n")
            self.assertEqual((common.hub_key_name(), common.api_keys()), ("NVAPI_KEY_OFFICIAL", ["official-key-222222"]))
            active.write_text("NVAPI_KEY_OFFICIAL_1\n")  # 2026-10-20: once NVAPI_KEY_OFFICIAL is used up
            self.assertEqual((common.hub_key_name(), common.api_keys()), ("NVAPI_KEY_OFFICIAL_1", ["official1-key-333333"]))
            active.write_text("NVAPI_KEY_OFFICIAL_2\n")  # 2026-10-22: once NVAPI_KEY_OFFICIAL_1 is used up
            self.assertEqual((common.hub_key_name(), common.api_keys()), ("NVAPI_KEY_OFFICIAL_2", ["official2-key-444444"]))
            active.write_text("NVAPI_KEY_NEW_3\n")  # 2026-10-27: the official3 queue
            self.assertEqual((common.hub_key_name(), common.api_keys()), ("NVAPI_KEY_NEW_3", ["new3-key-555555"]))
            for bogus in ("NVAPI_KEY", "SOMETHING_ELSE", ""):  # the spent key and unknown names are never used
                active.write_text(bogus)
                self.assertEqual(common.hub_key_name(), "NVAPI_KEY_NEW")


class Agents(unittest.TestCase):
    def test_argv(self):
        self.assertEqual(CODEX, S / "tools/codex/rust-v0.163.0/bin/codex")
        self.assertTrue(os.access(CODEX, os.X_OK) and CODEX_CONFIG.is_file())
        codex = agent_argv("codex", "PROMPT", Path("/w"), ["/extra"], Path("/state"))
        self.assertEqual(codex, [str(CODEX), "exec", "--ephemeral", "--skip-git-repo-check", "-s", "workspace-write",
                                 "--json", "-C", "/w", "--add-dir", "/extra",
                                 "-c", "sandbox_workspace_write.network_access=true",
                                 "-c", 'sqlite_home="/state"', "-c", 'log_dir="/state"',
                                 "-o", "/w/.agent/last-message.txt", "PROMPT"])
        cursor = agent_argv("cursor", "PROMPT", Path("/w"))  # kept for later use; no stage runs Cursor now
        self.assertEqual(cursor[cursor.index("--model") + 1], CURSOR_MODEL)
        self.assertIn("-p", cursor)
        self.assertEqual(parse_json('```json\n{"ok": true}\n```'), {"ok": True})

    def test_every_stage_uses_codex(self):
        for name in ("task_builder.py", "failure_analyzer.py", "smoke.py"):
            self.assertNotIn('"cursor"', (SYNTH / name).read_text(), name)

    def test_agent_env_has_no_secrets_and_codex_gets_only_its_two_variables(self):
        planted = {"NVAPI_KEY": "x", "OPENSANDBOX_API_KEY": "x", "GITLAB_TOKEN": "x", "my_secret": "x",
                   "DB_PASSWORD": "x", "GOOGLE_APPLICATION_CREDENTIALS": "/x", "HF_TOKEN": "x", "KEYRING": "x",
                   "CODEX_HOME": "/lustre/shared/codex", "SAFE_SETTING": "kept"}
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, planted), \
                mock.patch.object(llm_engine, "scratch", return_value=Path(directory)), \
                mock.patch.object(llm_engine, "api_keys", return_value=[HUB_KEY]):
            plain, codex = _agent_env(Path("/w"), "cursor"), _agent_env(Path("/w"), "codex")
            home = Path(directory) / "codex-home"
            self.assertEqual((home / "config.toml").read_bytes(), CODEX_CONFIG.read_bytes())
            self.assertEqual((home.stat().st_mode & 0o777, (home / "config.toml").stat().st_mode & 0o777),
                             (0o700, 0o600))
        self.assertEqual([key for key in plain if SECRET_NAME.search(key.upper())], [])
        self.assertEqual(plain["SAFE_SETTING"], "kept")
        self.assertEqual(plain["PATH"].split(os.pathsep)[:2], ["/w/bin", str(CODEX.parent)])
        self.assertLessEqual(set(plain), set(codex))
        self.assertEqual({key for key in codex if codex[key] != plain.get(key)}, {"CODEX_HOME", "NVHUB_API_KEY"})
        self.assertEqual((codex["CODEX_HOME"], codex["NVHUB_API_KEY"]), (str(home), HUB_KEY))
        self.assertEqual([key for key in codex if SECRET_NAME.search(key.upper())], ["NVHUB_API_KEY"])


class Chat(unittest.TestCase):
    """chat(): the local student goes to this job's router without credentials; hosted models go to the hub."""

    def run_chat(self, model, environ=None):
        requests, events, options = [], [], []

        def answer(request):
            requests.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})

        def client(**kwargs):
            options.append(kwargs)
            return real_client(transport=httpx.MockTransport(answer), **kwargs)

        real_client = httpx.AsyncClient
        with mock.patch.dict(sys.modules, fake_student_client(events)), mock.patch.dict(os.environ, environ or {}), \
                mock.patch.object(llm_engine, "api_keys", return_value=[HUB_KEY]), \
                mock.patch.object(llm_engine.httpx, "AsyncClient", client):
            reply = asyncio.run(chat(model, [{"role": "user", "content": "hi"}], json_mode=True))
        self.assertEqual((reply, len(requests)), ({"ok": True}, 1))
        events.append(("trust_env", options[0]["trust_env"]))
        return requests[0], json.loads(requests[0].content), events

    def test_student_goes_to_the_router_without_credentials(self):
        request, body, events = self.run_chat(STUDENT, {"SYNTH_STUDENT_WAIT_S": "7"})
        self.assertEqual(str(request.url), ROUTER + "/chat/completions")
        self.assertNotIn("authorization", request.headers)
        self.assertEqual(body["model"], "policy")
        self.assertEqual((body["temperature"], body["top_p"], body["max_tokens"]), (1.0, 1.0, 32768))
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True, "truncate_history_thinking": False})
        self.assertEqual(events, [("ensure_router", {}), ("wait_for_workers", 1, 7.0), ("trust_env", False)])

    def test_hosted_models_go_to_the_hub_with_the_key(self):
        for model in (TEACHER, JUDGE):
            request, body, events = self.run_chat(model)
            self.assertEqual(str(request.url), "https://inference-api.nvidia.com/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer " + HUB_KEY)
            self.assertEqual(body["model"], SETTINGS["models"][model]["id"])
            self.assertEqual((body["temperature"], body["top_p"], body["reasoning_effort"]), (1.0, 0.95, "max"))
            self.assertNotIn("chat_template_kwargs", body)
            self.assertEqual(events, [("trust_env", True)])  # no router; environment proxies honoured as before


class Probe(unittest.TestCase):
    """probe(): a local student needs the router and a worker before ATP staging; settings keep ${oc.env:...}."""

    def run_probe(self, models):
        events = []

        async def run(argv, log_path, env):
            events.append(("run", Path(str(argv[2])).name, env.get("SYNTH_STUDENT_BASE_URL")))
            return 0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app/scripts").mkdir(parents=True)
            for name in rollouts.ATP_STAGE_SCRIPTS:
                (root / "app/scripts" / name).write_text("")
            with mock.patch.dict(sys.modules, fake_student_client(events)), \
                    mock.patch.dict(os.environ, {"SYNTH_STUDENT_WAIT_S": "9"}), \
                    mock.patch.object(rollouts, "ATP", root / "app"), \
                    mock.patch.object(rollouts, "scratch", return_value=root / "scratch"), \
                    mock.patch.object(rollouts, "_run", run), \
                    mock.patch.object(rollouts, "probe_complete", return_value=True), \
                    mock.patch.object(rollouts, "probe_records", return_value={}):
                os.environ.pop("SYNTH_STUDENT_BASE_URL", None)
                asyncio.run(rollouts.probe(root / "probe", [DEEPSWE / TARGET], models=models, repeats=1))
            return events, json.loads((root / "probe/settings.json").read_text()), root / "probe"

    def test_local_student_waits_for_router_and_worker_before_staging(self):
        events, settings, probe_dir = self.run_probe((STUDENT, TEACHER))
        self.assertEqual(events, [("ensure_router", {"log_dir": probe_dir}), ("wait_for_workers", 1, 9.0),
                                  ("run", "deepswe_stage.py", ROUTER), ("run", "probe_worker.py", ROUTER)])
        self.assertEqual(settings["models"], {STUDENT: SETTINGS["models"][STUDENT], TEACHER: SETTINGS["models"][TEACHER]})
        self.assertEqual(settings["models"][STUDENT]["base_url"], "${oc.env:SYNTH_STUDENT_BASE_URL}")
        self.assertEqual((settings["expected_tasks"], settings["repeats"], settings["max_concurrency_per_model"]),
                         (1, 1, 1))

    def test_probe_concurrency_is_capped(self):
        # 2026-10-19: a 25-task x 5-repeat probe ran 125 rollouts per model in one ATP process and lost its PTY sockets
        self.assertEqual(rollouts.probe_limits(1), (1, 1))
        self.assertEqual(rollouts.probe_limits(50), (min(50, rollouts.PROBE_MAX_CONCURRENCY),
                                                     SETTINGS["initial_concurrency_per_model"]))
        self.assertEqual(rollouts.probe_limits(125)[0], rollouts.PROBE_MAX_CONCURRENCY)

    def test_hosted_only_probe_starts_no_router(self):
        events, settings, _ = self.run_probe((TEACHER,))
        self.assertEqual(events, [("run", "deepswe_stage.py", None), ("run", "probe_worker.py", None)])
        self.assertEqual(list(settings["models"]), [TEACHER])


class Settings(unittest.TestCase):
    def test_models_and_atp_pins(self):
        self.assertEqual((STUDENT, TEACHER, JUDGE), ("sc14", "kimi", "glm"))
        self.assertEqual(SETTINGS["models"][STUDENT], {
            "id": "policy", "local": True, "base_url": "${oc.env:SYNTH_STUDENT_BASE_URL}", "api_key": "EMPTY",
            "context_window": 262144, "max_output_tokens": 131072,
            "chat_template_kwargs": {"enable_thinking": True, "truncate_history_thinking": False},
            "sampling_overrides": {"temperature": 1.0, "top_p": 1.0},
            "checkpoint": "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/models/swonlysc-step0014/hf"})
        pinned = json.loads((ATP / "configs/deepswe.json").read_text())  # the app's own, unchanged config
        self.assertEqual(list(SETTINGS["models"]), [STUDENT, TEACHER, JUDGE])
        self.assertEqual({model: SETTINGS["models"][model] for model in (TEACHER, JUDGE)},
                         {model: pinned["models"][model] for model in (TEACHER, JUDGE)})
        self.assertEqual({key: value for key, value in SETTINGS.items() if key != "models"},
                         {key: value for key, value in pinned.items() if key != "models"})

    def test_code_reads_nothing_from_the_original_tree(self):
        owner = "bai" + "heh"  # split so this file does not match itself
        for path in [*SYNTH.glob("*.py"), SYNTH / "settings.json", SYNTH / "run.sbatch", *HELPER.glob("*.py")]:
            text = path.read_text()
            self.assertFalse(owner in text or "nemotron_evals_research" in text, path)


def repair_tool():
    """tools/grading_repair.py: the repair of existing tasks, whose language mapping new builds must share."""
    tools = str(Path(__file__).resolve().parents[1] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import grading_repair
    return grading_repair


GO_GLOBS = ["**/*_test.go", "**/testdata/**"]
JS_GLOBS = ["**/*.test.*", "**/*.spec.*", "**/__tests__/**", "test/**", "tests/**"]
PY_GLOBS = ["tests/**", "test/**", "**/test_*.py", "**/*_test.py", "**/conftest.py"]


class NewBuildGrader(unittest.TestCase):
    """hengyuf 2026-10-20: new builds write the sanitizing grader (verifier/grader.py) with config.json "test_paths" and
    "test_paths_mode" chosen from the repository's top100.jsonl language exactly as tools/grading_repair.py chooses them
    for the repaired tasks (verifier/test_paths.py); gates() names a reference solution's edits of test paths."""

    def test_mapping_equals_grading_repair(self):
        import task_builder
        gr, tp = repair_tool(), task_builder.TEST_PATHS
        self.assertEqual(tp.GLOBS, gr.GLOBS)  # on a mismatch, update verifier/test_paths.py to grading_repair's
        self.assertEqual(tp.MODES, gr.MODES)
        for language in gr.GLOBS:
            self.assertEqual(tp.config_keys(language), {"test_paths": gr.GLOBS[language],
                                                         "test_paths_mode": gr.task_mode(language)}, language)
        self.assertEqual(tp.DEFAULT_MODE, gr.task_mode("a language without a mode"))

    def test_language_choice_equals_grading_repair(self):
        import task_builder
        gr, tp = repair_tool(), task_builder.TEST_PATHS
        rows = [{"task_id": "t1", "repo": "o/mixed", "language": "TypeScript"},
                {"task_id": "t2", "repo": "o/mixed", "language": "JavaScript"},
                {"task_id": "t1", "repo": "o/go", "language": "Go"},
                {"task_id": "t1", "repo": "o/cpp", "language": "C++"}]
        cases = {("t1", "o/mixed"): "TypeScript", ("t2", "O/Mixed"): "JavaScript",  # the target's own rows first
                 ("t3", "o/go"): "Go", ("t3", "o/mixed"): None,  # else all rows of the repo, which must agree
                 ("t1", "o/missing"): None, ("t1", "o/cpp"): "C++"}
        with tempfile.TemporaryDirectory() as directory:
            top100 = Path(directory) / "top100.jsonl"
            top100.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with mock.patch.object(gr, "TOP100", top100):
                for (target, repo), language in cases.items():
                    if language is None:
                        self.assertRaises(gr.RepairError, gr.repo_language, target, repo)
                        self.assertRaises(tp.UnknownLanguage, tp.repo_language, top100, target, repo)
                    else:
                        self.assertEqual((gr.repo_language(target, repo), tp.repo_language(top100, target, repo)),
                                         (language, language), (target, repo))
        with self.assertRaises(tp.UnknownLanguage):  # in top100.jsonl but without globs: no silent fallback
            tp.config_keys("C++")
        self.assertFalse(issubclass(tp.UnknownLanguage, RuntimeError))  # _make_task retries RuntimeErrors as authoring
        live = {("sql-formatter-bigquery-pipe-formatting", "mtxr/vscode-sqltools"): "TypeScript",  # 2026-10-20 builds
                ("dynamodb-toolbox-lazy-recursive-schemas", "IdoPesok/zsa"): "TypeScript",
                ("wasmi-trap-coredumps", "bytecodealliance/wasmtime"): "Rust",
                ("scriggo-method-declarations", "cosmos72/gomacro"): "Go",
                ("anko-default-function-arguments", "go-python/gpython"): "Go"}
        for (target, repo), language in live.items():
            self.assertEqual((tp.repo_language(task_builder.TOP100, target, repo), gr.repo_language(target, repo)),
                             (language, language), repo)
            self.assertEqual(task_builder.Build("unit-test-run", target, repo).grading, tp.config_keys(language))

    def test_write_tests_installs_the_sanitizing_grader(self):
        import task_builder
        for target, repo, globs, mode in ((TARGET, "deepnoodle-ai/risor", GO_GLOBS, "all"),
                                          ("testem-bail-on-test-failure", "jestjs/jest", JS_GLOBS, "non_additive"),
                                          ("aiomonitor-task-snapshots-diff", "tobymao/saq", PY_GLOBS, "all"),
                                          ("wasmi-trap-coredumps", "bytecodealliance/wasmtime", ["tests/**"], "all")):
            build = task_builder.Build("unit-test-run", target, repo)
            with tempfile.TemporaryDirectory() as directory:
                tests = Path(directory) / "tests"
                task_builder.write_tests(tests, build, "echo block", "", "c" * 40, ["f1"], ["p1"], solution="")
                self.assertEqual(sorted(p.name for p in tests.iterdir()), ["config.json", "grader.py", "test.patch",
                                                                         "test.sh"])
                self.assertEqual((tests / "grader.py").read_bytes(), (SYNTH / "verifier/grader.py").read_bytes())
                self.assertEqual(task_builder.SANITIZER, SYNTH / "verifier/grader.py")
                config = json.loads((tests / "config.json").read_text())
                self.assertEqual(config, {"base_commit": "c" * 40, "f2p_node_ids": ["f1"], "p2p_node_ids": ["p1"],
                                          "grade": dict(build.grade_cfg, reports=build.reports), "test_paths": globs,
                                          "test_paths_mode": mode}, repo)
                self.assertEqual((tests / "test.sh").read_text(), test_sh("echo block", DEEPSWE / target))

    def test_test_patch_directories_extend_the_globs(self):
        """official1-anko (main, 2026-10-20): gpython's TestVm/TestPy run every script in vm/tests/, py/tests/ and
        stdlib/*/tests/, which the Go globs miss, so test.patch's directories there join "test_paths"."""
        import task_builder

        def new(path, text="x"):
            return (f"diff --git a/{path} b/{path}\nnew file mode 100644\nindex 0000000..1111111\n--- /dev/null\n"
                    f"+++ b/{path}\n@@ -0,0 +1 @@\n+{text}\n")

        def modified(path):
            return (f"diff --git a/{path} b/{path}\nindex 1111111..2222222 100644\n--- a/{path}\n+++ b/{path}\n"
                    "@@ -1 +1,2 @@\n x\n+y\n")

        test_patch = "".join([
            new("vm/tests/defaults_hidden.py"), new("vm/vm_defaults_hidden_test.go", "package vm"),  # second: a Go glob
            modified("stdlib/math/tests/mathtests.py"), new("py/tests/function.py"), new("py/tests/sub/more.py"),
            modified("symtable/make_symtable_test.py"),  # a generator next to symtable.go: a source directory
            modified("go.mod"),  # the repository root
            new("examples/fixtures/config.json"),  # solution.patch changes examples/fixtures/loader.go
            "diff --git a/compile/tests/gone.py b/compile/tests/gone.py\ndeleted file mode 100644\n"
            "index 1111111..0000000\n--- a/compile/tests/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n",
            "diff --git a/old/tests/case.txt b/moved/tests/case.txt\nsimilarity index 100%\n"
            "rename from old/tests/case.txt\nrename to moved/tests/case.txt\n",
            new("data/tests/[v1]/case.txt"),
            "diff --git a/vm/tests/x.py b/vm/tests/x.py\n--- a/vm/tests/x.py\n+++ b/vm/tests/x.py\n@@ -1 +1 @@\n"
            "-a\n+diff --git a/hunk/tests/fake.py b/hunk/tests/fake.py\n"])  # a hunk line is no path
        solution = modified("vm/vm.go") + new("examples/fixtures/loader.go", "package fixtures")
        extra, skipped = task_builder.test_patch_globs(test_patch, solution, GO_GLOBS)
        self.assertEqual(extra, ["data/tests/[[]v1]/**", "moved/tests/**", "py/tests/**", "stdlib/math/tests/**",
                                 "vm/tests/**"])
        self.assertEqual(sorted(skipped), [".", "examples/fixtures", "symtable"])
        self.assertIn("go.mod", skipped["."])
        self.assertIn("examples/fixtures/loader.go", skipped["examples/fixtures"])
        self.assertIn("not a test directory", skipped["symtable"])
        grader = task_builder._load(task_builder.SANITIZER, "sanitizing_grader")
        self.assertTrue(grader.glob_match(extra[0], "data/tests/[v1]/other.txt"))
        self.assertFalse(grader.glob_match(extra[0], "data/tests/v/other.txt"))  # the brackets are literal
        build = task_builder.Build("unit-test-run", "anko-default-function-arguments", "go-python/gpython")
        logged = []
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(task_builder, "log",
                                                                           lambda *parts: logged.append(parts)):
            tests = Path(directory) / "tests"
            task_builder.write_tests(tests, build, "true", test_patch, "c" * 40, solution=solution)
            config = json.loads((tests / "config.json").read_text())
        self.assertEqual((config["test_paths"], config["test_paths_mode"]), (GO_GLOBS + extra, "all"))
        self.assertEqual(task_builder.restored_paths(solution, config["test_paths"]), [])  # never the solution's
        self.assertEqual(len(logged), 1)
        self.assertIn("symtable not added: not a test directory", " ".join(map(str, logged[0])))
        self.assertEqual(task_builder.test_patch_globs("", solution, GO_GLOBS), ([], {}))

    def test_unknown_language_fails_at_build_start(self):
        import task_builder
        unknown = task_builder.TEST_PATHS.UnknownLanguage
        build = task_builder.Build("unit-test-run", TARGET, "example/not-in-top100")
        stop = mock.AsyncMock(side_effect=AssertionError("the build went on without a grader config"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(unknown):
                task_builder.write_tests(Path(directory) / "tests", build, "true", "", "c" * 40, solution="")
            self.assertFalse((Path(directory) / "tests").exists())  # nothing written
            with mock.patch.object(task_builder, "RUNS", Path(directory) / "runs"), \
                    mock.patch.object(task_builder, "analyze_target", stop), \
                    mock.patch.object(task_builder, "build_env", stop):
                with self.assertRaises(unknown):
                    asyncio.run(task_builder.build("unit-test-run", TARGET, "example/not-in-top100", 1, 1, 0))
            stop.assert_not_called()
            self.assertFalse((Path(directory) / "runs").exists())  # not even the build root

    def test_gates_name_solution_edits_of_test_paths_first(self):
        import task_builder
        build = task_builder.Build("unit-test-run", TARGET, "deepnoodle-ai/risor")  # Go: *_test.go and testdata/
        graded = []

        async def grade(source, task_dir, patch, *, run, out_dir):
            graded.append(Path(task_dir))
            new = {"n1": "passed", "n2": "passed"} if patch else {"n1": "failed", "n2": "failed"}
            return {"suites": {"base-ctrf.json": {"p1": "passed"}, "new-ctrf.json": new}}

        async def stale_tests(b, cand, mat, start_patch=""):
            return []

        solution = ("diff --git a/vm/vm.go b/vm/vm.go\nindex 1..2 100644\n--- a/vm/vm.go\n+++ b/vm/vm.go\n"
                    "@@ -1 +1,2 @@\n x\n+// +++ b/vm/testdata/fake.txt\n"
                    "diff --git a/vm/testdata/slices.txt b/vm/testdata/slices.txt\nnew file mode 100644\n"
                    "index 0000000..1111111\n--- /dev/null\n+++ b/vm/testdata/slices.txt\n@@ -0,0 +1 @@\n+x\n"
                    "diff --git a/vm/old_test.go b/vm/helpers.go\nsimilarity index 100%\nrename from vm/old_test.go\n"
                    "rename to vm/helpers.go\n")
        self.assertEqual(task_builder.restored_paths(solution, GO_GLOBS), ["vm/testdata/slices.txt", "vm/old_test.go"])
        self.assertEqual(task_builder.restored_paths(solution, ["**/__tests__/**"]), [])
        with tempfile.TemporaryDirectory() as directory:
            author_dir, cand = Path(directory) / "author-0", Path(directory) / "cand"
            author_dir.mkdir()
            for name, text in {"run_tests.sh": "true\n", "test.patch": "", "solution.patch": solution,
                               "start.patch": "diff --git a/vm/vm.go b/vm/vm.go\n"}.items():
                (author_dir / name).write_text(text)
            with mock.patch.object(task_builder, "grade", grade), mock.patch.object(task_builder, "stale_tests",
                                                                                     stale_tests):
                gate = asyncio.run(task_builder.gates(build, cand, author_dir, {"snapshot": "s", "base_commit": "c"}))
            self.assertEqual(set(graded), {cand / "provisional"})  # the problem stops the gates before final grading
            tests = cand / "provisional/tests"
            self.assertEqual((tests / "grader.py").read_bytes(), (SYNTH / "verifier/grader.py").read_bytes())
            config = json.loads((tests / "config.json").read_text())
            self.assertEqual((config["test_paths"], config["test_paths_mode"]), (GO_GLOBS, "all"))
        self.assertEqual(gate["problems"], [
            f"solution.patch touches test paths the grader restores before grading (test_paths {GO_GLOBS}): "
            "['vm/testdata/slices.txt', 'vm/old_test.go']; move those changes to test.patch (solution.patch holds "
            "source files only)"])


class Harbor(unittest.TestCase):
    """A synthetic task assembled like task_builder.package() must load in Gym's DeepSWE task store."""

    def make_task(self, root: Path, collect_base: str) -> Path:
        task = root / "syn-demo-cand-c01-0123abcd"
        shutil.copytree(DEEPSWE / TARGET / "tests", task / "tests")
        (task / "solution").mkdir()
        shutil.copy2(DEEPSWE / TARGET / "solution/solution.patch", task / "solution/solution.patch")
        (task / "instruction.md").write_text("Implement stepped slices.\n\n" + IMPORTANT + "\n")
        base = "b" * 40
        collect = ("cd /app && mkdir -p /logs/artifacts && git config --global --add safe.directory /app "
                   f"&& git diff --binary {collect_base} HEAD > /logs/artifacts/model.patch")
        (task / "task.toml").write_text(fill(
            TASK_TOML, task_id=task.name, ext_id="e" * 32, title=json.dumps('Stepped "slices"'),
            summary=json.dumps("demo"), language="go", repo_url="https://github.com/o/r", base_commit=base,
            upstream="c" * 40, target=TARGET, kind="mask", collect=json.dumps(collect), image="snapshot:1234"))
        return task

    def test_both_gym_copies_are_present(self):  # validate_task() silently skips a missing copy
        for gym in (GYM, SWE_RL_GYM):
            self.assertTrue((gym / "resources_servers/deepswe/task_store.py").is_file(), gym)

    def test_gym_accepts_synthetic_task(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self.make_task(Path(directory), "b" * 40)
            validate_task(task)
            row = dataset_row(task)
            self.assertEqual((row["image"], row["task_id"]), ("snapshot:1234", task.name))

    def test_gym_rejects_wrong_collect_hook(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self.make_task(Path(directory), "d" * 40)
            with self.assertRaises(GateError):
                validate_task(task)

    def test_gym_accepts_tests_from_write_tests(self):  # the sanitizing grader and its config keys (2026-10-20)
        import task_builder
        with tempfile.TemporaryDirectory() as directory:
            task = self.make_task(Path(directory), "b" * 40)
            shutil.rmtree(task / "tests")
            build = task_builder.Build("unit-test-run", "testem-bail-on-test-failure", "jestjs/jest")
            test_patch = ("diff --git a/packages/x/test/data.json b/packages/x/test/data.json\nnew file mode 100644\n"
                          "index 0000000..1111111\n--- /dev/null\n+++ b/packages/x/test/data.json\n@@ -0,0 +1 @@\n+{}\n")
            task_builder.write_tests(task / "tests", build, "echo block", test_patch, "b" * 40, ["f1"], ["p1"],
                                     solution="")
            config = json.loads((task / "tests/config.json").read_text())
            self.assertEqual((config["test_paths"], config["test_paths_mode"]),
                             (JS_GLOBS + ["packages/x/test/**"], "non_additive"))
            validate_task(task)


if __name__ == "__main__":
    unittest.main()
