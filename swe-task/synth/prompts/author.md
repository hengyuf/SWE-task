# Author synthetic task: {{title}}

Repository: {{repo}} (kind: {{kind}}). {{kind_rule}}

## Inputs (./inputs)
- `plan.md`: the plan for this task. Follow it.
- `understanding.md`: map of the repository. `target_failure_analysis.md`: the student failure patterns this task
  must trigger.
- `reference/`: the target DeepSWE task's `test.sh`, `config.json` and `instruction.md` (conventions and style).
{{feedback}}

{{sandbox}}
The sandbox starts at state A: the repository at {{sha}} with its environment ready, offline. It is your scratch
space: develop and test there freely. The controller rebuilds everything from A using ONLY the files you write here,
so record every change in the patches. The student's starting state is A + start.patch re-initialised as a FRESH git
repository with a single commit: upstream history, commit {{sha}} and any other commit of A do not exist there, so
`run_tests.sh` and the tests must not depend on git history, upstream commits or files outside /app and the patches.
The controller also deletes git-ignored build outputs (directories named dist, build, out, lib, bin, coverage, htmlcov,
.turbo, .next, .svelte-kit, .parcel-cache, *.tsbuildinfo files, and git-ignored compiled executables such as a CLI
binary built into the repository root) and tool caches, so that no compiled copy of removed code survives. If the
tests need any of them (e.g. compiled packages of a monorepo, or the CLI binary for shell acceptance tests),
`run_tests.sh` must rebuild them first. Build artifacts COMMITTED to git are not deleted: if the repository tracks
build outputs that contain the code you remove (e.g. `dist/` bundles, minified files, source maps whose
`sourcesContent` embeds the sources, generated `.d.ts`), `start.patch` must delete or regenerate them so that no copy
of the removed code, verbatim or transpiled, is left anywhere in the starting state, and the tests must not depend on
them (or `run_tests.sh` rebuilds them). The same holds for committed files GENERATED FROM THE TESTS you remove, e.g.
documentation or examples that test scenarios regenerate (official1-ytt: yq's `pkg/yqlib/doc/**` pages list each
scenario's input and expected output): they spell out the hidden tests, so `start.patch` removes those sections (or
regenerates the pages from the remaining tests) like the stale tests themselves. Documentation of the feature that
the instruction itself states may stay.

## Write these files in this directory
1. `start.patch`: `git diff --binary` from A to the student's starting state ({{start_rule}}). It must apply on A
   with `git apply --index`, and it must not edit any `.gitignore` (the controller's cleanup follows those rules).
   When it removes behaviour, it also removes every existing test of that behaviour (into test.patch), including
   spec/conformance files and test data outside the usual suites (e.g. `.wast` files).
2. `solution.patch`: from the starting state to the solved state. Source files only, no test files: it must not
   touch test paths (e.g. Go `testdata/`, Python `tests/`, JS `__tests__/`), which the grader restores before grading.
3. `test.patch`: the hidden tests, applied on top of the starting state (and of the solution). Test files only, no
   path shared with `solution.patch`, and no edit of an existing non-test file (the grader resets every file test.patch
   touches, so a solution's own edits there would be lost): put hidden tests in test files, e.g. Rust integration tests
   under `tests/` or a new test-only module file (adding its `#[cfg(test)] mod <name>;` line is the only edit allowed in
   an existing source file; this also holds for tests start.patch removes from a source file's inline test module),
   Go `*_test.go`, Python `test_*.py`, JS/TS `*.test.*` files.
   Tests must compile/import against the starting state so that fail-to-pass
   tests fail by assertion or behaviour and the pass-to-pass tests of the same packages still run. Write plain unit
   tests in the repository's own test framework: no meta-verifiers, integrity or freeze checks, and no tests that
   re-run other suites (the grader already resets every file test.patch touches before applying it).
   Tests that `start.patch` removes go back into the SAME files they were removed from (the grader resets every file
   `test.patch` touches): moved to a new file, they would collide with a student's same-named test left in the
   original file (in Go, a redeclaration that breaks the whole package). For the same reason, give every NEW
   top-level name in the hidden test files (test functions and also helper types, funcs, vars, consts and fixtures)
   a name a student would not pick for their own tests (e.g. a task-specific suffix such as `_Hidden`, or a
   `hidden` prefix for helpers): in languages with package-level names (Go), a student's own new test or helper
   with the same name in another file of the package breaks the build (official1-arcane: a hidden `diagFixture`
   type), and the grader does not reset files test.patch does not touch.
   Every hidden test is one of two kinds. (a) Failure-pattern tests pin down the behaviour behind the target failure
   pattern(s) this task must trigger (`target_failure_analysis.md`, `plan.md`), i.e. what the student gets wrong;
   they belong in the hidden tests even where the instruction only implies that behaviour. (b) Every other hidden
   test checks only functionality that `instruction_draft.md` states. Both kinds go through the public interface:
   no test may depend on implementation details a correct solution may choose differently (private or internal
   names, helper functions, call sequences or file layout that the instruction does not name). In particular, do not
   inject or observe state by mocking an internal method (official1-aiomonitor c02-r1b2 patched `queue.job`, so
   solutions using the public bulk `queue.jobs` failed), and keep each hidden test to stated behaviour: a test that
   asserts an unstated detail first (per-app queue resolution in c02-r1b0) hides the stated checks after it. This also covers the
   EXISTING tests in every file test.patch touches: the grader resets those files, so their tests are graded (and
   must compile) exactly as you leave them. If one calls a private helper whose signature or existence the task
   leaves to the solver, remove or rewrite that test in test.patch (official1-ytt c04: an upstream test called the
   private `encodeTopLevelEntry` with 3 arguments; kimi refactored it to 4, and the reset file no longer compiled).
4. `run_tests.sh`: the body of the verifier's RUN TESTS block (bash, cwd /app, after the test patch is applied).
   Run the pass-to-pass suite writing `{{base_report}}` and the fail-to-pass suite writing `{{new_report}}`, with the
   same reporter and format as the reference (`grade` section: {{grade_cfg}}); mirror the reference block: `set +e`
   around the suites, never gate on reporter exit codes, append raw output to "$RUN_LOG". Keep both suites focused
   on the touched packages and under 10 minutes. "Touched" includes the packages that define the public types and
   APIs the feature extends or must preserve (e.g. the AST and parser for a syntax feature), even where the
   reference solution leaves them unchanged: a solution that changes those APIs and rewrites their tests must fail
   pass-to-pass (official1-absslices c02 graded only bytecode/compiler/vm, so sc14 changed the public
   ast.Func.Defaults, rewrote 5 parser tests and still passed). NEVER exit early: a failing rebuild, typecheck or install step must
   not stop the block (no `exit`, no `set -e`, no `|| exit`). Record each such step as a gate test in the pass-to-pass
   report, in the same format (e.g. a test case named `[gate] build` that passes only if the step succeeded, as the
   DeepSWE reference frame does), then still run both suites. A solution that does not build or typecheck then fails
   that pass-to-pass test and is graded FAILED, instead of leaving no grade at all. Gate only build/typecheck steps,
   never a suite's exit code and never linters (`go vet`, eslint, ruff, golangci-lint, ...): a linter also checks the
   student's own tests and style, so a correct solution could fail it. Do not wrap build or test commands in short `timeout`s: the starting state has cold
   build caches and probes run many verifiers at once, so a cold compile can take minutes. The verifier's overall time
   limit applies, and per-test limits (e.g. `go test -timeout`) are fine.
   A solution that hangs or leaves open handles must still be GRADED (as failed), not run into the verifier's 1800 s
   limit and get no grade: use the runner's own exit safeguards (e.g. Jest `--forceExit`; per-test timeouts as in the
   reference), and if a suite can still block after its tests finished, run it under one generous `timeout` (at least
   900 s and leaving room for both suites inside 1800 s; never around build/compile steps). Either way the block must
   still write both reports, even empty or partial ones: tests missing from a report count as failed. (official1-
   testem c05-r1b2: Jest without `--forceExit` never exited after a student's reporter left an open handle, so 3 of 5
   rollouts got no grade.)
5. `instruction_draft.md`: what the student must build, at the behaviour level, in the style of
   `reference/instruction.md`; name the public APIs and observable behaviour the tests rely on; no file paths of
   the implementation, no mention of tests, patches or removed code.
6. `result.json`: `{"status": "ok" | "failed", "summary": str, "f2p_tests": [test ids], "p2p_tests": [test ids], "notes": str}`.

## Self-check in the sandbox before finishing
Starting each time from a clean A (`git checkout -- . && git clean -fdq`, keeping /tmp/work) with the ignored build
outputs above deleted as the controller does (`git ls-files -o -i --exclude-standard --directory` lists them):
a) apply `start.patch` then `test.patch`, run `run_tests.sh`: every new-suite test fails, every base-suite test passes;
b) also apply `solution.patch`, rerun: every test passes;
c) the lines added by `start.patch` contain no TODO/TBD/FIXME/XXX, "not implemented" or unimplemented markers.
The controller re-runs these checks with the real grader (DeepSWE's, plus a step that restores every test path a
submission edits) and rejects the task if any fails.
