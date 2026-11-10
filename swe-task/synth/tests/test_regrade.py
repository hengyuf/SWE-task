"""Offline tests for tools/regrade.py (no sandbox, network or LLM calls; bash and git run locally on temp files).
Run from S/synth:  python3 -B -m unittest discover -s tests -v"""
import sys

sys.dont_write_bytecode = True
from pathlib import Path

SYNTH = Path(__file__).resolve().parents[1]
for _path in (SYNTH, SYNTH / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import contextlib  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tarfile  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402
import unittest  # noqa: E402
from unittest import mock  # noqa: E402

import regrade  # noqa: E402
import rollouts  # noqa: E402
import task_builder  # noqa: E402
from common import DEEPSWE  # noqa: E402
from task_builder import RUN_END, RUN_START  # noqa: E402

TARGET = "vitest-duration-sharding"  # a real DeepSWE task: its tests/test.sh is the shared verifier frame
BASH = shutil.which("bash")
GIT = shutil.which("git")

# ------------------------------------------------------------------------------------------------ fixture blocks
# Shapes seen in official1-vitest round 1 (2026-10-18), shortened.

TWO_GATES = """#!/usr/bin/env bash
set -uo pipefail
cd /app || exit 6
export YARN_ENABLE_NETWORK=0
export TMPDIR=/tmp/work
mkdir -p /tmp/work /logs/verifier
export RUN_LOG="${RUN_LOG:-/logs/verifier/run.log}"
touch "$RUN_LOG"
rm -f /logs/verifier/base.xml /logs/verifier/new.xml \\
  /logs/verifier/base-ctrf.json /logs/verifier/new-ctrf.json

# The CLI loads built entry points (a comment that says: exit 1 if the build fails).
set +e
yarn build:js 2>&1 | tee -a "$RUN_LOG"
build_rc=${PIPESTATUS[0]}
set -e
if [ "$build_rc" -ne 0 ]; then
  echo "JavaScript build failed: $build_rc" | tee -a "$RUN_LOG"
  exit "$build_rc"
fi

set +e
yarn tsc -b packages/test-globals packages/jest-worker/src/__tests__ \\
  2>&1 | tee -a "$RUN_LOG"
types_rc=${PIPESTATUS[0]}
set -e
if [ "$types_rc" -ne 0 ]; then
  echo "Package declaration/test typecheck failed: $types_rc" | tee -a "$RUN_LOG"
  exit "$types_rc"
fi

set +e
JEST_JUNIT_OUTPUT_DIR=/logs/verifier JEST_JUNIT_OUTPUT_NAME=base.xml \\
yarn jest --ci --runInBand --watchman=false --no-cache \\
  --reporters=default --reporters=jest-junit --runTestsByPath \\
  packages/jest-worker/src/__tests__/Farm.test.ts \\
  2>&1 | tee -a "$RUN_LOG"
echo "base suite rc=${PIPESTATUS[0]}" | tee -a "$RUN_LOG"
JEST_JUNIT_OUTPUT_DIR=/logs/verifier JEST_JUNIT_OUTPUT_NAME=new.xml \\
yarn jest --ci --runInBand --watchman=false --no-cache \\
  --reporters=default --reporters=jest-junit --runTestsByPath \\
  packages/jest-worker/src/__tests__/Farm.reservations.test.ts \\
  2>&1 | tee -a "$RUN_LOG"
echo "new suite rc=${PIPESTATUS[0]}" | tee -a "$RUN_LOG"
set -e

set +e
junit-to-ctrf /logs/verifier/base.xml -o /logs/verifier/base-ctrf.json \\
  -t jest --use-suite-name 2>&1 | tee -a "$RUN_LOG"
junit-to-ctrf /logs/verifier/new.xml -o /logs/verifier/new-ctrf.json \\
  -t jest --use-suite-name 2>&1 | tee -a "$RUN_LOG"
set -e"""

MULTI_RC = """#!/usr/bin/env bash
# Executed from /app after the verifier has applied its test files.
set -uo pipefail
cd /app || exit 6
mkdir -p /tmp/work /logs/verifier
export RUN_LOG=${RUN_LOG:-/logs/verifier/run.log}
run_log() {
  echo "+ $*" >> "$RUN_LOG"
  "$@" 2>&1 | tee -a "$RUN_LOG"
  return "${PIPESTATUS[0]}"
}
rm -f /logs/verifier/base.xml /logs/verifier/new-runtime.xml /logs/verifier/base-ctrf.json /logs/verifier/new-ctrf.json

set +e
run_log yarn build:js
js_rc=$?
run_log yarn build:ts
ts_rc=$?
run_log yarn bundle:ts
bundle_rc=$?
set -e
if [ "$js_rc" -ne 0 ] || [ "$ts_rc" -ne 0 ] || [ "$bundle_rc" -ne 0 ]; then
  echo "Required source build failed: js=$js_rc types=$ts_rc bundle=$bundle_rc" | tee -a "$RUN_LOG"
  exit 1
fi

export JEST_JUNIT_CLASSNAME='{filepath}'
jest_command=(node packages/jest-cli/bin/jest.js --config jest.config.mjs --ci --runInBand
  --watchman=false --cacheDirectory=/tmp/work/schedule-verifier-cache
  --reporters=default --reporters=jest-junit --runTestsByPath)

cat > /tmp/work/schedule-type-reporter.mjs <<'JS'
export default class {
  on([event]) {
    if (event === 'run:end') process.exit(1);
  }
}
exit 1
set -e
JS

set +e
JEST_JUNIT_OUTPUT_FILE=/logs/verifier/base.xml run_log "${jest_command[@]}" \\
  packages/jest-test-sequencer/src/__tests__/test_sequencer.test.ts
base_rc=$?
echo "base suite rc=$base_rc" | tee -a "$RUN_LOG"
JEST_JUNIT_OUTPUT_FILE=/logs/verifier/new-runtime.xml run_log "${jest_command[@]}" \\
  packages/jest-test-sequencer/src/__tests__/schedule-manifest.test.ts
echo "new runtime suite rc=$?" | tee -a "$RUN_LOG"
run_log junit-to-ctrf '/logs/verifier/base.xml' -o /logs/verifier/base-ctrf.json -t jest --use-suite-name
run_log junit-to-ctrf '/logs/verifier/new-*.xml' -o /logs/verifier/new-ctrf.json -t jest --use-suite-name
set -e"""

BARE_EXIT = """#!/usr/bin/env bash
set -uo pipefail
cd /app || exit 6
export RUN_LOG="${RUN_LOG:-/logs/verifier/run.log}"
set +e
yarn build:js 2>&1 | tee -a "$RUN_LOG"
build_rc=${PIPESTATUS[0]}
set -e
if [ "$build_rc" -ne 0 ]; then
  exit "$build_rc"
fi
jest_args=(
  --config jest.config.mjs --ci --runInBand --watchman=false --no-cache
  --reporters=default --reporters=jest-junit --runTestsByPath
)
set +e
JEST_JUNIT_OUTPUT_FILE=/logs/verifier/base.xml \\
  node packages/jest-cli/bin/jest.js "${jest_args[@]}" \\
  packages/jest-test-sequencer/src/__tests__/test_sequencer.test.ts \\
  2>&1 | tee -a "$RUN_LOG"
echo "base suite rc=${PIPESTATUS[0]}" | tee -a "$RUN_LOG"
JEST_JUNIT_OUTPUT_FILE=/logs/verifier/new.xml \\
  node packages/jest-cli/bin/jest.js "${jest_args[@]}" \\
  packages/jest-test-sequencer/src/__tests__/testSchedule.test.ts \\
  2>&1 | tee -a "$RUN_LOG"
echo "new suite rc=${PIPESTATUS[0]}" | tee -a "$RUN_LOG"
junit-to-ctrf /logs/verifier/base.xml -o /logs/verifier/base-ctrf.json -t jest --use-suite-name
junit-to-ctrf /logs/verifier/new.xml -o /logs/verifier/new-ctrf.json -t jest --use-suite-name
set -e"""

ERREXIT_TOP = """#!/bin/bash
set -euo pipefail
cd /app
mkdir -p /logs/verifier /tmp/work
export RUN_LOG=${RUN_LOG:-/logs/verifier/run.log}
rm -f /logs/verifier/base.xml /logs/verifier/new.xml

# Workspace imports and real worker backends must use the current source.
yarn build:js 2>&1 | tee -a "$RUN_LOG"

export JEST_JUNIT_SUITE_NAME='{filepath}'
set +e
JEST_JUNIT_OUTPUT_FILE=/logs/verifier/base.xml \\
node packages/jest-cli/bin/jest.js --config jest.config.mjs --ci --runTestsByPath \\
  packages/jest-worker/src/__tests__/Farm.test.ts 2>&1 | tee -a "$RUN_LOG"
echo "[verifier] base rc=${PIPESTATUS[0]}" | tee -a "$RUN_LOG"
JEST_JUNIT_OUTPUT_FILE=/logs/verifier/new.xml \\
node packages/jest-cli/bin/jest.js --config jest.config.mjs --ci --runTestsByPath \\
  packages/jest-worker/src/__tests__/reservations.test.ts 2>&1 | tee -a "$RUN_LOG"
echo "[verifier] new rc=${PIPESTATUS[0]}" | tee -a "$RUN_LOG"
junit-to-ctrf /logs/verifier/base.xml -o /logs/verifier/base-ctrf.json -t jest --use-suite-name
junit-to-ctrf /logs/verifier/new.xml -o /logs/verifier/new-ctrf.json -t jest --use-suite-name
set -e"""

ERREXIT_CHECK = """#!/usr/bin/env bash
set -euo pipefail
cd /app
mkdir -p /logs/verifier /tmp/work
export RUN_LOG=${RUN_LOG:-/logs/verifier/run.log}
touch "$RUN_LOG"
run_log() {
  echo "+ $*" >> "$RUN_LOG"
  "$@" 2>&1 | tee -a "$RUN_LOG"
  return "${PIPESTATUS[0]}"
}
run_log yarn build:js
runner=(node packages/jest-cli/bin/jest.js --config jest.config.mjs --ci
  --runInBand --watchman=false --no-cache --reporters=default --reporters=jest-junit --runTestsByPath)
set +e
JEST_JUNIT_OUTPUT_FILE=/logs/verifier/base.xml run_log "${runner[@]}" packages/jest-worker/src/__tests__/Farm.test.ts
echo "base suite rc=$?" | tee -a "$RUN_LOG"
JEST_JUNIT_OUTPUT_FILE=/logs/verifier/new.xml run_log "${runner[@]}" packages/jest-worker/src/__tests__/new.test.ts
echo "new suite rc=$?" | tee -a "$RUN_LOG"
run_log junit-to-ctrf /logs/verifier/base.xml -o /logs/verifier/base-ctrf.json -t jest --use-suite-name
run_log junit-to-ctrf /logs/verifier/new.xml -o /logs/verifier/new-ctrf.json -t jest --use-suite-name
set -e

# Missing or unreadable reports are execution errors, not behavioral results.
node - <<'NODE'
const fs = require('node:fs');
for (const suite of ['base', 'new']) {
  const report = JSON.parse(fs.readFileSync('/logs/verifier/' + suite + '-ctrf.json', 'utf8'));
  if (!report.results.tests.length) throw new Error('Empty report: ' + suite);
}
NODE"""

INLINE = """#!/bin/bash
set -uo pipefail
cd /app || { mkdir -p /logs/verifier; exit 6; }
export RUN_LOG=${RUN_LOG:-/logs/verifier/run.log}
die() { echo "verifier: $*" | tee -a "$RUN_LOG"; exit 3; }
node scripts/build.mjs 2>&1 | tee -a "$RUN_LOG" || exit 1
yarn install --immutable || { echo "install failed" | tee -a "$RUN_LOG"; exit 2; }
yarn tsc --noEmit -p . || die "typecheck failed"
JEST_JUNIT_OUTPUT_FILE=/logs/verifier/base.xml yarn jest --ci --runTestsByPath a.test.ts 2>&1 | tee -a "$RUN_LOG"
JEST_JUNIT_OUTPUT_FILE=/logs/verifier/new.xml yarn jest --ci --runTestsByPath b.test.ts 2>&1 | tee -a "$RUN_LOG"
junit-to-ctrf /logs/verifier/base.xml -o /logs/verifier/base-ctrf.json -t jest --use-suite-name
junit-to-ctrf /logs/verifier/new.xml -o /logs/verifier/new-ctrf.json -t jest --use-suite-name"""

RISK_ONLY = """#!/bin/bash
set -uo pipefail
cd /app || exit 6
mkdir -p /tmp/work /logs/verifier
export RUN_LOG=${RUN_LOG:-/logs/verifier/run.log}
set +e
yarn build:js 2>&1 | tee -a "$RUN_LOG"
build_rc=${PIPESTATUS[0]}
echo "build:js rc=$build_rc" | tee -a "$RUN_LOG"
base=(
  packages/jest-reporters/src/__tests__/CoverageReporter.test.js
)
for test_file in packages/jest-reporters/src/__tests__/*.test.ts; do
  [[ "$test_file" == */CoverageThresholdPolicy.test.ts ]] || base+=("$test_file")
done
JEST_JUNIT_OUTPUT_NAME=base.xml yarn jest --ci --runTestsByPath "${base[@]}" 2>&1 | tee -a "$RUN_LOG"
echo "base suite rc=${PIPESTATUS[0]}" | tee -a "$RUN_LOG"
JEST_JUNIT_OUTPUT_NAME=new.xml yarn jest --ci --runTestsByPath \\
  packages/jest-reporters/src/__tests__/CoverageThresholdPolicy.test.ts 2>&1 | tee -a "$RUN_LOG"
echo "new suite rc=${PIPESTATUS[0]}" | tee -a "$RUN_LOG"
junit-to-ctrf /logs/verifier/base.xml -o /logs/verifier/base-ctrf.json -t jest --use-suite-name 2>&1 | tee -a "$RUN_LOG"
junit-to-ctrf /logs/verifier/new.xml -o /logs/verifier/new-ctrf.json -t jest --use-suite-name 2>&1 | tee -a "$RUN_LOG"
set -e

# Record the prerequisite build in the base report, even if test discovery failed.
python3 - "$build_rc" <<'PY'
import json
import sys
from pathlib import Path
path = Path('/logs/verifier/base-ctrf.json')
report = json.loads(path.read_text()) if path.exists() else {'results': {'tests': []}}
report['results']['tests'].append({'name': '[gate] yarn build:js', 'status': 'passed' if sys.argv[1] == '0' else 'failed'})
path.write_text(json.dumps(report))
PY"""

CLEAN = """#!/bin/bash
# set -e is deliberately not used; the `exit 1` in this comment is only text.
set -uo pipefail
cd /app || exit 6
mkdir -p /tmp/work /logs/verifier
export RUN_LOG=${RUN_LOG:-/logs/verifier/run.log}
echo "never exit 1 here; set -e stays off" | tee -a "$RUN_LOG"
note='exit 1; set -e'
count=${#note}
files=()
for f in packages/*/src/__tests__/*.test.ts; do
  case "$f" in
    */skip.test.ts|*/other.test.ts) continue ;;
    *) files+=("$f") ;;
  esac
done
cat > /tmp/work/check.py <<'PY'
import sys
if len(sys.argv) > 9:
    sys.exit(1)
PY
cat > /tmp/work/inner.sh <<-EOF
\tset -e
\texit 2
\tEOF
set +e
run_log() { echo "+ $*" >> "$RUN_LOG"; "$@" 2>&1 | tee -a "$RUN_LOG"; return "${PIPESTATUS[0]}"; }
run_log yarn build:js
echo "build rc=$?" | tee -a "$RUN_LOG"
JEST_JUNIT_OUTPUT_FILE=/logs/verifier/base.xml run_log yarn jest --ci --runTestsByPath "${files[@]}" base.test.ts
JEST_JUNIT_OUTPUT_FILE=/logs/verifier/new.xml run_log yarn jest --ci --runTestsByPath new.test.ts
run_log junit-to-ctrf /logs/verifier/base.xml -o /logs/verifier/base-ctrf.json -t jest --use-suite-name
run_log junit-to-ctrf /logs/verifier/new.xml -o /logs/verifier/new-ctrf.json -t jest --use-suite-name
set -e"""

EARLY = {  # block: (expected early-exit findings as (kind, line), failure that the original block exits on)
    "TWO_GATES": (TWO_GATES, {("exit", 19), ("exit", 29)}, "tsc"),
    "MULTI_RC": (MULTI_RC, {("exit", 24)}, "build:ts"),
    "BARE_EXIT": (BARE_EXIT, {("exit", 10)}, "build:js"),
    "ERREXIT_TOP": (ERREXIT_TOP, {("errexit", 9)}, "build:js"),
    "ERREXIT_CHECK": (ERREXIT_CHECK, {("errexit", 12), ("errexit", 25)}, "build:js|node -"),
    "INLINE": (INLINE, {("exit", 5), ("exit", 6), ("exit", 7)}, "tsc"),
}
NOT_EARLY = {"RISK_ONLY": RISK_ONLY, "CLEAN": CLEAN}


def frame_test_sh(block: str) -> str:
    return task_builder.test_sh(block, DEEPSWE / TARGET)


# ------------------------------------------------------------------------------------------------ bash harness

STUB = """#!/bin/sh
line="$(basename "$0") $*"
printf '%s\\n' "$line" >> "$STUB_LOG"
cat > /dev/null
IFS='|'
for pat in $STUB_FAIL; do
  case "$line" in *"$pat"*) exit 1;; esac
done
exit 0
"""


def run_block(block: str, fail: str = "") -> dict:
    """Run a RUN TESTS block under bash with /app, /logs/verifier and /tmp/work moved into a temp dir and stub
    yarn/node/junit-to-ctrf/python3 that log their argv and fail when it contains a STUB_FAIL pattern (|-separated)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in ("app", "logs/verifier", "work", "bin"):
            (tmp / name).mkdir(parents=True)
        for name in ("yarn", "node", "npx", "junit-to-ctrf", "python3", "tstyche"):
            (tmp / "bin" / name).write_text(STUB)
            (tmp / "bin" / name).chmod(0o755)
        body = block.replace("/logs/verifier", str(tmp / "logs/verifier")).replace("/tmp/work", str(tmp / "work"))
        body = body.replace("/app", str(tmp / "app"))
        (tmp / "run.sh").write_text(f"set -uo pipefail\nexport RUN_LOG={tmp}/logs/verifier/run.log\n{body}\n"
                                    f"echo REACHED_END > {tmp}/end\n")
        proc = subprocess.run([BASH, str(tmp / "run.sh")], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=60, env={"PATH": f"{tmp}/bin:/usr/bin:/bin", "HOME": str(tmp),
                                               "STUB_LOG": str(tmp / "stub.log"), "STUB_FAIL": fail})
        stub = (tmp / "stub.log").read_text().replace(str(tmp), "<tmp>") if (tmp / "stub.log").exists() else ""
        log = (tmp / "logs/verifier/run.log")
        return {"rc": proc.returncode, "end": (tmp / "end").exists(), "stub": stub,
                "suites": sum("--runTestsByPath" in line for line in stub.splitlines()),
                "converters": sum(line.startswith("junit-to-ctrf") for line in stub.splitlines()),
                "run_log": log.read_text().replace(str(tmp), "<tmp>") if log.exists() else ""}


# ------------------------------------------------------------------------------------------------ analysis

class Scan(unittest.TestCase):
    def test_early_exit_shapes_are_found(self):
        for name, (block, expected, _) in EARLY.items():
            with self.subTest(name):
                found = {(f["kind"], f["line"]) for f in regrade.analyze(block) if f["severity"] == "early-exit"}
                self.assertEqual(found, expected)

    def test_gates_name_the_failing_step(self):
        findings = {f["line"]: f for f in regrade.analyze(TWO_GATES)}
        self.assertEqual([g["kind"] for g in findings[19]["gates"]], ["build"])
        self.assertEqual([(g["kind"], g["line"]) for g in findings[29]["gates"]], [("typecheck", 23)])
        self.assertEqual(findings[29]["condition"], '[ "$types_rc" -ne 0 ]')
        multi = [f for f in regrade.analyze(MULTI_RC) if f["kind"] == "exit"][0]
        self.assertEqual([(g["kind"], g["line"]) for g in multi["gates"]], [("build", 15), ("build", 17), ("build", 19)])
        inline = {f["line"]: f for f in regrade.analyze(INLINE)}
        self.assertEqual(inline[5]["function"], "die")
        self.assertEqual([g["kind"] for g in inline[5]["gates"]], ["typecheck"])  # `yarn tsc ... || die ...`
        self.assertEqual([g["kind"] for g in inline[6]["gates"]], ["build"])      # `node scripts/build.mjs | tee || exit 1`
        self.assertEqual([g["kind"] for g in inline[7]["gates"]], ["install"])    # `yarn install || { ...; exit 2; }`
        check = {f["line"]: f for f in regrade.analyze(ERREXIT_CHECK)}
        self.assertEqual((check[12]["step"], check[12]["set_line"]), ("build", 2))
        self.assertEqual((check[25]["step"], check[25]["set_line"]), ("check", 22))  # node script that throws

    def test_cd_guard_quotes_comments_and_heredocs_are_not_exits(self):
        self.assertEqual(regrade.analyze(CLEAN), [])
        lines = [f["line"] for f in regrade.analyze(TWO_GATES)]
        self.assertNotIn(3, lines)   # cd /app || exit 6
        self.assertNotIn(12, lines)  # comment mentioning exit 1
        multi = [(f["kind"], f["line"]) for f in regrade.analyze(MULTI_RC)]
        self.assertEqual(multi, [("exit", 24)])  # heredoc body with process.exit(1), exit 1 and set -e is data
        self.assertNotIn(3, [f["line"] for f in regrade.analyze(INLINE)])  # cd /app || { ...; exit 6; }

    def test_test_gates_resolve_their_status_variable_and_plain_exits_are_found(self):
        block = ('set -uo pipefail\nset -e\nyarn build:js\nrc=$?\n[ "$rc" -eq 0 ] || exit "$rc"\n'
                 'yarn jest --runTestsByPath a.test.ts\nexit "$new_rc"\n')
        found = [(f["kind"], f["line"], f.get("step"), [(g["kind"], g["line"]) for g in f.get("gates", [])])
                 for f in regrade.analyze(block)]
        self.assertEqual(found, [("errexit", 3, "build", []), ("exit", 5, None, [("build", 3)]),
                                 ("errexit", 6, "suite", []), ("exit", 7, None, [])])
        new, edits = regrade.repair_block(block)
        self.assertEqual([e["line"] for e in edits], [2, 5, 7])
        self.assertIn("build step failed (exit ${rc-} removed)", new.split("\n")[4])
        self.assertIn(f'echo "{regrade.MARK} exit ${{new_rc-}} removed; the grader scores this run"', new.split("\n")[6])

    def test_tolerant_script_under_set_e_is_a_risk_not_an_early_exit(self):
        findings = regrade.analyze(RISK_ONLY)
        self.assertEqual([(f["kind"], f["severity"], f["step"]) for f in findings], [("errexit", "risk", "script")])

    def test_set_arguments(self):
        cases = {("-euo", "pipefail"): ["-uo", "pipefail"], ("-e",): ["+e"], ("-eu",): ["-u"],
                 ("-e", "-o", "pipefail"): ["-o", "pipefail"], ("-o", "errexit"): ["+e"],
                 ("-Eeuxo", "pipefail"): ["-Euxo", "pipefail"], ("-o", "errexit", "-u"): ["-u"]}
        for args, expected in cases.items():
            with self.subTest(args):
                self.assertTrue(regrade._errexit_change(list(args)))
                self.assertEqual(regrade.neutralize_set(list(args)), expected)
                self.assertNotEqual(regrade._errexit_change(expected), True)
        self.assertIsNone(regrade._errexit_change(["-uo", "pipefail"]))
        self.assertFalse(regrade._errexit_change(["+e"]))

    def test_split_test_sh_round_trip(self):
        text = frame_test_sh(TWO_GATES)
        head, block, tail = regrade.split_test_sh(text)
        self.assertEqual((head + block + tail, block), (text, TWO_GATES))
        self.assertTrue(head.endswith(RUN_START + "\n") and tail.startswith("\n" + RUN_END))
        with self.assertRaises(regrade.FixError):
            regrade.split_test_sh(text + RUN_START)


# ------------------------------------------------------------------------------------------------ rewriting

class Rewrite(unittest.TestCase):
    def test_only_offending_lines_change_and_no_early_exit_is_left(self):
        for name, (block, expected, _) in EARLY.items():
            with self.subTest(name):
                new, edits = regrade.repair_block(block)
                self.assertEqual(len(new.split("\n")), len(block.split("\n")))
                changed = {n + 1 for n, (a, b) in enumerate(zip(block.split("\n"), new.split("\n"))) if a != b}
                self.assertEqual(changed, {e["line"] for e in edits})
                exits = {line for kind, line in expected if kind == "exit"}
                self.assertTrue(exits <= changed, (exits, changed))
                for line in changed:
                    after = new.split("\n")[line - 1]
                    self.assertTrue(regrade.MARK in after or after.lstrip().startswith("set "), after)
                self.assertEqual([f for f in regrade.analyze(new) if f["severity"] == "early-exit"], [])
                self.assertEqual(regrade.repair_block(new), (new, []))  # idempotent

    def test_exit_becomes_a_logged_line(self):
        new, edits = regrade.repair_block(TWO_GATES)
        lines = new.split("\n")
        self.assertEqual(lines[28], '  echo "[verifier] early-exit-fix: typecheck step failed (exit ${types_rc-} removed); '
                                    'the suites still run and the grader scores them" | tee -a '
                                    '"${RUN_LOG:-/logs/verifier/run.log}"')
        self.assertEqual(lines[27], TWO_GATES.split("\n")[27])  # the author's own failure message stays
        bare, _ = regrade.repair_block(BARE_EXIT)
        self.assertIn("build step failed (exit ${build_rc-} removed)", bare.split("\n")[9])

    def test_errexit_set_loses_only_e(self):
        new, edits = regrade.repair_block(ERREXIT_CHECK)
        self.assertEqual([(e["line"], e["before"], e["after"]) for e in edits], [
            (2, "set -euo pipefail", "set -uo pipefail  # early-exit-fix: was 'set -euo pipefail'"),
            (22, "set -e", "set +e  # early-exit-fix: was 'set -e'")])

    def test_blocks_without_early_exit_are_unchanged(self):
        for name, block in NOT_EARLY.items():
            with self.subTest(name):
                self.assertEqual(regrade.repair_block(block), (block, []))
        risk, edits = regrade.repair_block(RISK_ONLY, include_risks=True)
        self.assertEqual([e["after"] for e in edits], ["set +e  # early-exit-fix: was 'set -e'"])

    def test_test_sh_frame_stays_byte_identical(self):
        text = frame_test_sh(MULTI_RC)
        head, block, tail = regrade.split_test_sh(text)
        new, _ = regrade.repair_block(block)
        repaired = head + new + tail
        self.assertEqual(repaired, frame_test_sh(new))
        self.assertEqual((repaired[:len(head)], repaired[len(head) + len(new):]), (head, tail))

    @unittest.skipUnless(BASH, "bash not found")
    def test_repaired_blocks_run_the_suites_when_a_step_fails(self):
        for name, (block, _, fail) in EARLY.items():
            with self.subTest(name):
                new, _ = regrade.repair_block(block)
                before, after = run_block(block, fail), run_block(new, fail)
                self.assertFalse(before["end"], before)   # the original stops before grading...
                self.assertEqual(before["suites"], 0)
                self.assertTrue(after["end"], after)      # ...the repaired block reaches the end
                self.assertEqual((after["suites"], after["converters"]), (2, 2))

    @unittest.skipUnless(BASH, "bash not found")
    def test_repaired_blocks_behave_identically_when_nothing_fails(self):
        for name, block in [(n, b) for n, (b, _, _) in EARLY.items()] + list(NOT_EARLY.items()):
            with self.subTest(name):
                new, _ = regrade.repair_block(block)
                before, after = run_block(block), run_block(new)
                self.assertTrue(before["end"] and after["end"])
                self.assertEqual((before["stub"], before["run_log"], before["rc"]),
                                 (after["stub"], after["run_log"], after["rc"]))
                self.assertEqual(after["suites"], 2)


# ------------------------------------------------------------------------------------------------ fixture tree

SOLUTION = """diff --git a/packages/jest-worker/src/Farm.ts b/packages/jest-worker/src/Farm.ts
index 1111111..2222222 100644
--- a/packages/jest-worker/src/Farm.ts
+++ b/packages/jest-worker/src/Farm.ts
@@ -1,3 +1,6 @@
 import {FifoQueue} from './FifoQueue';
+import reserve from './reserve';
+
+export const RESERVED = reserve([]);

 export default class Farm {
"""
F2P = ["jest-worker: reservations > a", "jest-worker: reservations > b"]
P2P = ["jest-worker: Farm > c"]
LOST = "Verifier did not produce /logs/verifier/reward.json"


def snapshot(directory: Path) -> dict:
    return {str(p.relative_to(directory)): (p.read_bytes() if p.is_file() else None)
            for p in sorted(directory.rglob("*"))}


def make_record(model, repeat, status, *, patch="diff --git a/x b/x\n", error=None):
    record = {"slot": f"{model}-r{repeat}", "model": model, "status": status, "reward": 1.0 if status == "pass" else 0.0,
              "f2p": "2/2" if status == "pass" else ("1/2" if status == "fail" else "0/0"),
              "p2p": "1/1" if status in ("pass", "fail") else "0/0",
              "failing_tests": ["[f2p] " + F2P[1]] if status == "fail" else [], "agent_error": None,
              "verifier_error": error, "patch": patch, "output": [], "attempt_dir": f"/attempts/6-{repeat}/attempt-0"}
    return record


class Tree:
    """runs/fake-run/build/<TARGET>/jestjs__jest with built tasks, an accepted copy, a scored round and a probe."""

    def __init__(self, tmp: Path, blocks: dict, accepted=()):
        self.tmp = tmp
        self.root = tmp / "runs/fake-run/build" / TARGET / "jestjs__jest"
        self.accepted = tmp / "runs/fake-run/accepted"
        self.tasks = {}
        summary = {"target": TARGET, "repo": "jestjs/jest", "built": [], "rejected": [], "accepted": []}
        for name, block in blocks.items():
            cand = self.root / name
            task_id = f"syn-jest-{name}-0000abcd"
            task = cand / "task" / task_id
            text = frame_test_sh(block)
            config = {"base_commit": "0" * 40, "f2p_node_ids": F2P, "p2p_node_ids": P2P,
                      "grade": {"format": "ctrf", "node_id": "name", "tool_label": "junit-to-ctrf",
                                "reports": ["/logs/verifier/base-ctrf.json", "/logs/verifier/new-ctrf.json"]}}
            for tests in (cand / "final/tests", task / "tests", cand / "provisional/tests"):
                tests.mkdir(parents=True)
                (tests / "test.sh").write_text(text)
                (tests / "config.json").write_text(json.dumps(config))
            (task / "solution").mkdir()
            (task / "solution/solution.patch").write_text(SOLUTION)
            (task / "instruction.md").write_text("Add reservations.\n")
            (task / "rubric.md").write_text("- explores first\n")
            (cand / "author-0").mkdir()
            (cand / "author-0/run_tests.sh").write_text(block + "\n")
            (cand / "author-0/solution.patch").write_text(SOLUTION)
            stage = {"ok": True, "name": name, "task_id": task_id, "task_dir": str(task), "snapshot": "snapshot:fake",
                     "kind": "evolve", "title": "Reservations", "f2p": 2, "p2p": 1}
            (cand / "stage.json").write_text(json.dumps(stage))
            (cand / "built.json").write_text(json.dumps({"attempt": 0}))
            summary["built"].append(stage)
            if name in accepted:
                shutil.copytree(task, self.accepted / task_id)
                workspace = self.accepted / f"{task_id}.workspace"
                workspace.mkdir()
                with tarfile.open(workspace / "workspace.tar.gz", "w:gz") as tar:
                    data = b"export default class Farm {}\n"
                    info = tarfile.TarInfo("./packages/jest-worker/src/Farm.ts")
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
                summary["accepted"].append(dict(stage, accept=True))
            self.tasks[name] = stage
        (self.root / "summary.json").write_text(json.dumps(summary))

    def score_round(self, name, records, *, accept=False):
        """round-1/<name> as score() left it, plus the probe manifest (task index 6) and seeds.json."""
        stage = self.tasks[name]
        out = self.root / "round-1" / name
        rollouts.write_rollouts(out / "rollouts", records)
        lost = [r["slot"] for r in records if r["status"] == "infra" and r["verifier_error"]]
        student = [r for r in records if r["model"] == "sc14" and r["status"] in ("pass", "fail")]
        metrics = dict(stage, lineage="cand-c04", student_rate=sum(r["status"] == "pass" for r in student) / len(student),
                       teacher_clean_rate=1.0, valid=[len(student), 5], grading_failures=lost, cheated=0,
                       answer_retrieval=False, target_student_rate=0.25, target_teacher_rate=1.0, similarity=None,
                       accept=accept, reason="inconclusive: fewer than 3 valid rollouts per model")
        (out / "metrics.json").write_text(json.dumps(metrics))
        (out / "teacher_audit.json").write_text(json.dumps({f"kimi-r{i}": {"cheated": False} for i in range(5)}))
        probe = self.root / "round-1/probe-1234abcd"
        probe.mkdir(parents=True, exist_ok=True)
        (probe / "manifest.json").write_text(json.dumps({"tasks": [{"task_id": stage["task_id"], "_ng_task_index": 6}],
                                                         "settings": {"models": {"kimi": {}, "sc14": {}}, "repeats": 5}}))
        (probe / "tasks" / stage["task_id"] / "tests").mkdir(parents=True)
        shutil.copy2(Path(stage["task_dir"]) / "tests/test.sh", probe / "tasks" / stage["task_id"] / "tests/test.sh")
        (self.root / "round-1/seeds.json").write_text(json.dumps(
            {"valid": [], "invalid": {name: f"verifier produced no grade for {len(lost)} rollout(s)"}}))
        return out


def lost_slot_records():
    records = [make_record("kimi", i, "pass") for i in range(5)]
    records += [make_record("sc14", i, "infra", patch=f"diff --git a/s{i} b/s{i}\n", error=LOST) for i in range(4)]
    return records + [make_record("sc14", 4, "pass")]


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        result = function(*args, **kwargs)
    return result, out.getvalue()


class RepairCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.tree = Tree(self.tmp, {"cand-c04-r1b1": TWO_GATES, "cand-c04-r1b0": BARE_EXIT, "cand-c05-r1b3": RISK_ONLY},
                         accepted=("cand-c04-r1b0",))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_scan_lists_early_exits_with_lines(self):
        status, out = quiet(regrade.main, ["scan", "--run", str(self.tree.root), "--json"])
        tasks = {t["cand"]: t for t in json.loads(out)["tasks"]}
        self.assertEqual(status, 0)
        self.assertEqual({n for n, t in tasks.items() if t["early_exit"]}, {"cand-c04-r1b1", "cand-c04-r1b0"})
        self.assertTrue(tasks["cand-c05-r1b3"]["risk"] and not tasks["cand-c05-r1b3"]["early_exit"])
        offset = regrade.split_test_sh(frame_test_sh(TWO_GATES))[0].count("\n")
        lines = [f["test_sh_line"] for f in tasks["cand-c04-r1b1"]["findings"]]
        self.assertEqual(lines, [19 + offset, 29 + offset])
        self.assertTrue(tasks["cand-c04-r1b0"]["accepted"])
        _, text = quiet(regrade.main, ["scan", "--run", str(self.tree.root)])
        self.assertIn(f'test.sh:{29 + offset}: exit "$types_rc"', text)

    def test_repair_rewrites_every_live_copy_and_keeps_backups(self):
        name = "cand-c04-r1b0"
        stage = self.tree.tasks[name]
        cand = self.tree.root / name
        files = [cand / "final/tests/test.sh", Path(stage["task_dir"]) / "tests/test.sh",
                 self.tree.accepted / stage["task_id"] / "tests/test.sh", cand / "author-0/run_tests.sh"]
        before = {p: p.read_text() for p in files}
        history = [cand / "provisional/tests/test.sh"]
        status, out = quiet(regrade.main, ["repair", "--run", str(self.tree.root), "--cand", "c04-r1b0"])
        self.assertEqual(status, 0, out)
        self.assertIn("workspace archive holds no copy of the test script", out)
        new_block, _ = regrade.repair_block(BARE_EXIT)
        for path in files:
            self.assertEqual(path.with_name(path.name + regrade.BACKUP).read_text(), before[path])
            expected = new_block + "\n" if path.name == "run_tests.sh" else frame_test_sh(new_block)
            self.assertEqual(path.read_text(), expected)
        self.assertEqual(history[0].read_text(), before[files[0]])  # a record of what ran: untouched
        log = [json.loads(l) for l in (self.tree.root / regrade.LOG).read_text().splitlines()]
        self.assertEqual(sorted(e["copy"] for e in log), ["accepted", "final", "run_tests", "task"])
        self.assertTrue(all(e["action"] == "repair" and e["sha256_before"] != e["sha256_after"] for e in log))
        record = json.loads((cand / regrade.FIX / "repair.json").read_text())
        self.assertEqual([e["line"] for e in record["edits"]], [10])
        # a second run changes nothing
        snap = snapshot(self.tmp)
        status, out = quiet(regrade.main, ["repair", "--run", str(self.tree.root), "--cand", name])
        self.assertEqual(status, 0)
        self.assertIn("already repaired", out)
        self.assertEqual(snapshot(self.tmp), snap)

    def test_repair_refuses_diverging_copies(self):
        task_sh = Path(self.tree.tasks["cand-c04-r1b1"]["task_dir"]) / "tests/test.sh"
        task_sh.write_text(task_sh.read_text() + "# edited\n")
        status, out = quiet(regrade.main, ["repair", "--run", str(self.tree.root), "--cand", "cand-c04-r1b1"])
        self.assertEqual(status, 1)
        self.assertIn("NOT REPAIRED", out)
        self.assertFalse(task_sh.with_name("test.sh" + regrade.BACKUP).exists())

    def test_archive_copies_are_detected(self):
        archive = self.tmp / "w.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for member, data in (("./a.ts", b"x"), ("./scripts/verify.sh", (RUN_START + "\nexit 1\n").encode())):
                info = tarfile.TarInfo(member)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        found = regrade.archive_copies(archive, [RUN_START])
        self.assertEqual((found["files_checked"], found["copies"]), (2, ["./scripts/verify.sh"]))

    def test_dry_runs_write_nothing(self):
        self.tree.score_round("cand-c04-r1b1", lost_slot_records())
        quiet(regrade.main, ["repair", "--run", str(self.tree.root), "--cand", "cand-c04-r1b1"])
        before = snapshot(self.tmp)

        async def no_grade(*args, **kwargs):
            raise AssertionError("dry run graded")

        with mock.patch.object(regrade.sandbox, "grade", no_grade), \
                mock.patch.object(regrade.rollouts, "probe_records",
                                  lambda probe: {self.tree.tasks["cand-c04-r1b1"]["task_id"]: lost_slot_records()}), \
                mock.patch.object(regrade, "hub_budget_ok", side_effect=AssertionError("dry run hit the hub")):
            for argv in (["repair", "--cand", "cand-c04-r1b0", "c05-r1b3", "--include-risks"],
                         ["validate", "--cand", "cand-c04-r1b1", "cand-c05-r1b3"],
                         ["regrade", "--cand", "cand-c04-r1b1", "--slots", "6-0,6-1", "--no-validate-check"]):
                with self.subTest(argv[0]):
                    status, out = quiet(regrade.main, argv[:1] + ["--run", str(self.tree.root), "--dry-run"] + argv[1:])
                    self.assertEqual(status, 0, out)
                    self.assertIn("dry run", out)
                    self.assertNotIn("NOT ", out)
        self.assertEqual(snapshot(self.tmp), before)


# ------------------------------------------------------------------------------------------------ records

class Records(unittest.TestCase):
    CONFIG = {"f2p_node_ids": F2P + [F2P[0]], "p2p_node_ids": P2P}

    def test_regraded_record_mirrors_the_grader(self):
        old = make_record("sc14", 0, "infra", error=LOST)
        result = {"reward": 0, "exit_code": 0, "timed_out": False,
                  "reward_json": {"reward": 0, "f2p_passed": 1, "f2p_total": 2, "p2p_passed": 1, "p2p_total": 1},
                  "suites": {"base-ctrf.json": {P2P[0]: "passed", F2P[0]: "failed"},
                             "new-ctrf.json": {F2P[0]: "passed"}}}
        new = regrade.regraded_record(old, result, self.CONFIG, {"tool": "x"})
        self.assertEqual((new["status"], new["reward"], new["f2p"], new["p2p"], new["verifier_error"]),
                         ("fail", 0.0, "1/2", "1/1", None))
        self.assertEqual(new["failing_tests"], [f"[f2p] {F2P[0]}", f"[f2p] {F2P[1]}"])  # worst status wins; missing fails
        self.assertEqual((new["patch"], new["output"], new["attempt_dir"], new["agent_error"]),
                         (old["patch"], old["output"], old["attempt_dir"], None))
        self.assertEqual(old["status"], "infra")  # the old record is not modified
        passed = regrade.regraded_record(old, dict(result, reward=1, reward_json={"reward": 1, "f2p_passed": 2,
                                                                                  "f2p_total": 2, "p2p_passed": 1,
                                                                                  "p2p_total": 1}), self.CONFIG, {})
        self.assertEqual((passed["status"], passed["reward"], passed["f2p"]), ("pass", 1.0, "2/2"))
        missing = regrade.regraded_record(old, {"reward": None, "reward_json": None, "exit_code": 1}, self.CONFIG, {})
        self.assertEqual(missing["status"], "infra")
        self.assertIn("regrade produced no reward.json", missing["verifier_error"])

    def test_replace_records_keeps_order_and_other_slots(self):
        records = lost_slot_records()
        original = copy.deepcopy(records)
        new = dict(records[5], status="fail")
        out = regrade.replace_records(records, {"sc14-r0": new})
        self.assertEqual([r["slot"] for r in out], [r["slot"] for r in original])
        self.assertIs(out[5], new)
        self.assertEqual([r for i, r in enumerate(out) if i != 5], [r for i, r in enumerate(original) if i != 5])
        self.assertEqual(records, original)
        with self.assertRaises(regrade.FixError):
            regrade.replace_records(records, {"sc14-r9": new})

    def test_slots_by_attempt_dir_or_name(self):
        manifest = {"tasks": [{"task_id": "t", "_ng_task_index": 6}]}
        self.assertEqual(regrade.resolve_slots("6-0,6-3", model="sc14", manifest=manifest, task_id="t"),
                         ["sc14-r0", "sc14-r3"])
        self.assertEqual(regrade.resolve_slots("kimi-r2", model="sc14", manifest=manifest, task_id="t"), ["kimi-r2"])
        for bad in ("7-0", "6-0,6-0", "zero"):
            with self.assertRaises(regrade.FixError):
                regrade.resolve_slots(bad, model="sc14", manifest=manifest, task_id="t")


# ------------------------------------------------------------------------------------------------ broken solution

@unittest.skipUnless(GIT, "git not found")
class BrokenSolution(unittest.TestCase):
    def git(self, *args, check=True):
        return subprocess.run([GIT, *args], cwd=self.repo, capture_output=True, text=True, check=check)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = self.tmp / "repo"
        (self.repo / "src").mkdir(parents=True)
        (self.repo / "docs").mkdir()
        (self.repo / "src/farm.ts").write_text("import {a} from './a';\n\nexport class Farm {\n  run() {\n    return a;\n"
                                               "  }\n}\n\nexport const LAST = 1;\n")
        (self.repo / "docs/x.md").write_text("# x\n")
        self.git("init", "-q")
        self.git("add", "-A")
        self.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def solution(self, *, imported: bool, big_new_file: bool) -> str:
        farm = (self.repo / "src/farm.ts").read_text()
        farm = farm.replace("import {a} from './a';\n", "import {a} from './a';\n" +
                            ("import reserve from './reserve';\n" if imported else ""))
        farm = farm.replace("    return a;\n", "    const value = a;\n    return value;\n")
        farm = farm.replace("export const LAST = 1;\n", "export function reserved(keys: string[]) {\n  return keys;\n}\n\n"
                            "export const LAST = 1;\n")
        (self.repo / "src/farm.ts").write_text(farm)
        body = "".join(f"  const k{i} = keys[{i}];\n" for i in range(12 if big_new_file else 1))
        (self.repo / "src/reserve.ts").write_text(f"export default function reserve(keys: string[]) {{\n{body}"
                                                  "  return [...keys];\n}\n")
        (self.repo / "docs/x.md").write_text("# x\n\nReservations.\n")
        self.git("add", "-A")
        patch = self.git("diff", "--cached").stdout
        self.git("reset", "-q", "--hard")
        self.git("clean", "-qfd")
        return patch

    def apply(self, patch: str) -> dict:
        """{path: content} of the touched files after `git apply` (which must accept the patch), then reset."""
        (self.tmp / "p.patch").write_text(patch)
        self.git("apply", "--check", str(self.tmp / "p.patch"))
        self.git("apply", str(self.tmp / "p.patch"))
        files = {f: (self.repo / f).read_text() for f in ("src/farm.ts", "src/reserve.ts", "docs/x.md")
                 if (self.repo / f).exists()}
        self.git("reset", "-q", "--hard")
        self.git("clean", "-qfd")
        return files

    def test_largest_source_file_gets_the_error_before_a_top_level_declaration(self):
        solution = self.solution(imported=True, big_new_file=False)
        broken, info = regrade.break_patch(solution)
        self.assertEqual((info["file"], info["new_file"]), ("src/farm.ts", False))
        self.assertEqual(info["before_line"], "+import reserve from './reserve';")
        self.assertEqual(broken.count("\n") - solution.count("\n"), 1)
        good, bad = self.apply(solution), self.apply(broken)
        lines = bad["src/farm.ts"].splitlines()
        self.assertEqual(lines[lines.index(regrade.BROKEN_LINE) + 1], "import reserve from './reserve';")
        self.assertEqual(lines[:lines.index(regrade.BROKEN_LINE)] + lines[lines.index(regrade.BROKEN_LINE) + 1:],
                         good["src/farm.ts"].splitlines())  # one extra line, nothing else changes
        self.assertEqual({k: v for k, v in bad.items() if k != "src/farm.ts"},
                         {k: v for k, v in good.items() if k != "src/farm.ts"})

    def test_imported_new_file_gets_the_error_at_its_end(self):
        solution = self.solution(imported=True, big_new_file=True)
        broken, info = regrade.break_patch(solution)
        self.assertEqual((info["file"], info["new_file"], info["before_line"]), ("src/reserve.ts", True, "(end of file)"))
        bad = self.apply(broken)
        self.assertEqual(bad["src/reserve.ts"].splitlines()[-1], regrade.BROKEN_LINE)
        self.assertEqual(bad["src/reserve.ts"].splitlines()[:-1], self.apply(solution)["src/reserve.ts"].splitlines())

    def test_new_file_nothing_imports_is_not_chosen(self):
        solution = self.solution(imported=False, big_new_file=True)  # may be dead code: a syntax error there proves nothing
        broken, info = regrade.break_patch(solution)
        self.assertEqual(info["file"], "src/farm.ts")
        self.assertEqual(info["before_line"], "+export function reserved(keys: string[]) {")
        self.apply(broken)
        forced, info = regrade.break_patch(solution, "src/reserve.ts")
        self.assertEqual(info["file"], "src/reserve.ts")
        self.apply(forced)
        with self.assertRaises(regrade.FixError):
            regrade.break_patch(solution, "docs/x.md")


# ------------------------------------------------------------------------------------------------ validate + regrade

def fake_grader(rewards: dict, calls: list):
    """sandbox.grade stand-in: reward by patch (rewards[patch] or 0), output.log with a repair mark."""
    async def grade(source, task_dir, patch, *, run, out_dir):
        calls.append({"source": source, "task_dir": str(task_dir), "patch": patch, "run": run, "out_dir": str(out_dir)})
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        applied = "[verifier] model.patch applied (10 bytes)\n" if patch else "[verifier] no model.patch submitted\n"
        (out_dir / "output.log").write_text(applied + f"{regrade.MARK} build step failed (exit 2 removed); the suites "
                                                      "still run and the grader scores them\n")
        reward = rewards.get(patch, 0)
        passed = {**{t: "passed" for t in P2P}, F2P[0]: "passed", F2P[1]: "passed" if reward else "failed"}
        return {"reward": reward, "exit_code": 0, "timed_out": False, "suites": {"new-ctrf.json": passed},
                "reward_json": {"reward": reward, "f2p_total": 2, "f2p_passed": 2 if reward else 1, "p2p_total": 1,
                                "p2p_passed": 1}}
    return grade


class ValidateAndRegrade(unittest.TestCase):
    NAME = "cand-c04-r1b1"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.tree = Tree(self.tmp, {self.NAME: TWO_GATES})
        self.stage = self.tree.tasks[self.NAME]
        self.records = lost_slot_records()
        self.out = self.tree.score_round(self.NAME, self.records)
        self.calls = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_main(self, argv, rewards=None):
        with mock.patch.object(regrade.sandbox, "grade", fake_grader(rewards or {}, self.calls)):
            return quiet(regrade.main, argv[:1] + ["--run", str(self.tree.root)] + argv[1:])

    def test_validate_requires_the_repair_and_reports_three_grades(self):
        status, out = self.run_main(["validate", "--cand", self.NAME])
        self.assertEqual(status, 1)
        self.assertIn("repair first", out)
        quiet(regrade.main, ["repair", "--run", str(self.tree.root), "--cand", self.NAME])
        status, out = self.run_main(["validate", "--cand", self.NAME], {SOLUTION: 1})
        self.assertEqual(status, 0, out)
        record = json.loads((self.tree.root / self.NAME / regrade.FIX / "validate.json").read_text())
        self.assertEqual(record["checks"], {"solution_scores_1": True, "empty_scores_0": True, "broken_graded_0": True})
        self.assertEqual(record["test_sh_sha256"],
                         hashlib.sha256((self.tree.root / self.NAME / "final/tests/test.sh").read_bytes()).hexdigest())
        self.assertEqual({Path(c["task_dir"]) for c in self.calls}, {self.tree.root / self.NAME / "final"})
        self.assertEqual(sorted(len(c["patch"]) for c in self.calls)[0], 0)  # the empty patch
        broken = (self.tree.root / self.NAME / regrade.FIX / "validate/broken.patch").read_text()
        self.assertIn("+" + regrade.BROKEN_LINE, broken)
        self.assertIn("broken   reward 0", out)
        # a broken solution that is not graded is a failed check
        status, out = self.run_main(["validate", "--cand", self.NAME], {SOLUTION: 1, broken: 1})
        self.assertEqual(status, 1)

    def regrade(self, *extra):
        self.judged = []

        async def judge_pass(record, instruction, rubric):
            self.judged.append(record["slot"])
            return {"cheated": False, "categories": []}

        async def analyze(*args, **kwargs):
            return {}

        async def similarity(*args, **kwargs):
            return {"score": 7}

        accept = mock.AsyncMock()
        fa_dir = self.tmp / "analysis"
        build = types.SimpleNamespace(root=self.tree.root, run="fake-run", target=TARGET, repo="jestjs/jest",
                                      fa_dir=fa_dir)
        task_id = self.stage["task_id"]
        with mock.patch.object(regrade.rollouts, "probe_records", lambda probe: {task_id: copy.deepcopy(self.records)}), \
                mock.patch.object(regrade, "hub_budget_ok", return_value=True) as budget, \
                mock.patch.object(regrade, "make_build", return_value=build), \
                mock.patch.object(regrade.task_builder, "judge_pass", judge_pass), \
                mock.patch.object(regrade.task_builder, "analyze", analyze), \
                mock.patch.object(regrade.task_builder, "similarity", similarity), \
                mock.patch.object(regrade.task_builder, "accept", accept):
            status, out = self.run_main(["regrade", "--cand", self.NAME, "--slots", "6-0,6-1,6-2,6-3", *extra])
        return status, out, accept, budget

    def test_regrade_refuses_unvalidated_tasks_and_graded_slots(self):
        quiet(regrade.main, ["repair", "--run", str(self.tree.root), "--cand", self.NAME])
        status, out, _, _ = self.regrade()
        self.assertEqual(status, 1)
        self.assertIn("no passing validate.json", out)
        self.run_main(["validate", "--cand", self.NAME], {SOLUTION: 1})
        with mock.patch.object(regrade.rollouts, "probe_records",
                               lambda probe: {self.stage["task_id"]: copy.deepcopy(self.records)}):
            status, out = self.run_main(["regrade", "--cand", self.NAME, "--slots", "6-4"])
        self.assertEqual(status, 1)
        self.assertIn("not a slot whose grade was lost", out)
        self.assertEqual(self.calls[3:], [])

    def test_regrade_rescores_with_corrected_records(self):
        quiet(regrade.main, ["repair", "--run", str(self.tree.root), "--cand", self.NAME])
        self.run_main(["validate", "--cand", self.NAME], {SOLUTION: 1})
        self.calls.clear()
        old_metrics = (self.out / "metrics.json").read_text()
        old_index = (self.out / "rollouts/index.md").read_text()
        status, out, accept, budget = self.regrade()
        self.assertEqual(status, 0, out)
        # the four lost slots were graded with the packaged (repaired) task on its snapshot
        task_dir = Path(self.stage["task_dir"])
        self.assertEqual(sorted(c["patch"] for c in self.calls), [f"diff --git a/s{i} b/s{i}\n" for i in range(4)])
        self.assertTrue(all(Path(c["task_dir"]) == task_dir and c["source"] == "snapshot:fake" for c in self.calls))
        state = json.loads((self.out / "regrade.json").read_text())
        self.assertEqual({s: (v["old"]["status"], v["new"]["status"], v["new"]["f2p"]) for s, v in state["slots"].items()},
                         {f"sc14-r{i}": ("infra", "fail", "1/2") for i in range(4)})
        self.assertEqual(state["old_metrics"]["valid"], [1, 5])
        self.assertEqual(state["new_metrics"]["valid"], [5, 5])
        # score() ran on the corrected list: same slots, only the four changed
        metrics = json.loads((self.out / "metrics.json").read_text())
        self.assertEqual((metrics["student_rate"], metrics["teacher_clean_rate"], metrics["grading_failures"],
                          metrics["similarity"], metrics["accept"], metrics["lineage"]),
                         (0.2, 1.0, [], 7, True, "cand-c04"))
        self.assertEqual((self.out / f"metrics.json{regrade.ASIDE}").read_text(), old_metrics)
        self.assertEqual((self.out / f"rollouts{regrade.ASIDE}/index.md").read_text(), old_index)
        self.assertTrue((self.out / f"teacher_audit.json{regrade.ASIDE}").exists())
        self.assertTrue((self.out / "teacher_audit.json").exists())
        self.assertTrue((self.out / "rollouts/sc14-r0.md").read_text().startswith("# sc14-r0 — fail"))
        self.assertTrue((self.out / "rollouts/sc14-r4.md").read_text().startswith("# sc14-r4 — pass"))
        accept.assert_awaited_once()
        self.assertEqual(accept.await_args.args[1]["task_id"], self.stage["task_id"])
        budget.assert_called_once()
        # bookkeeping: seeds.json and summary.json follow the new metrics (backups kept)
        seeds = json.loads((self.tree.root / "round-1/seeds.json").read_text())
        self.assertEqual((seeds["valid"], seeds["invalid"]), ([self.NAME], {}))
        self.assertIn(self.NAME, json.loads((self.tree.root / f"round-1/seeds.json{regrade.ASIDE}").read_text())["invalid"])
        summary = json.loads((self.tree.root / "summary.json").read_text())
        self.assertEqual([m["name"] for m in summary["accepted"]], [self.NAME])
        self.assertTrue((self.tree.root / f"summary.json{regrade.ASIDE}").exists())
        actions = [json.loads(l)["action"] for l in (self.tree.root / regrade.LOG).read_text().splitlines()]
        for action in ("repair", "validate", "regrade", "move_aside", "score", "accept", "seeds", "summary"):
            self.assertIn(action, actions)
        # a second run does nothing
        self.calls.clear()
        status, out, accept, _ = self.regrade()
        self.assertIn("already re-scored", out)
        self.assertEqual(self.calls, [])
        accept.assert_not_awaited()

    def reset_scoring(self):
        """Back to the scored state before a re-score: its outputs removed, the originals moved back."""
        for name in ("metrics.json", "rollouts", "teacher_audit.json", "failure_analysis", "regrade.json"):
            path = self.out / name
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        for name in ("metrics.json", "rollouts", "teacher_audit.json"):
            (self.out / f"{name}{regrade.ASIDE}").rename(self.out / name)

    def test_rescore_keeps_the_verdicts_of_unchanged_teacher_passes(self):
        audit = {f"kimi-r{i}": {"cheated": i == 0, "categories": []} for i in range(5)}
        (self.out / "teacher_audit.json").write_text(json.dumps(audit))
        quiet(regrade.main, ["repair", "--run", str(self.tree.root), "--cand", self.NAME])
        self.run_main(["validate", "--cand", self.NAME], {SOLUTION: 1})
        status, out, _, _ = self.regrade()
        self.assertEqual(status, 0, out)
        self.assertEqual(self.judged, [])  # nothing judged again: kimi-r0 stays cheated
        self.assertEqual(json.loads((self.out / "teacher_audit.json").read_text()), audit)
        metrics = json.loads((self.out / "metrics.json").read_text())
        self.assertEqual((metrics["teacher_clean_rate"], metrics["cheated"]), (0.8, 1))
        state = json.loads((self.out / "regrade.json").read_text())
        self.assertEqual(state["teacher_verdicts"], {"kept": [f"kimi-r{i}" for i in range(5)], "judged": 0})
        # a teacher transcript that no longer renders as it was scored is judged again
        self.reset_scoring()
        with (self.out / "rollouts/kimi-r1.md").open("a") as stream:
            stream.write("changed\n")
        status, out, _, _ = self.regrade()
        self.assertEqual(status, 0, out)
        self.assertEqual(self.judged, ["kimi-r1"])
        # --rejudge judges every teacher pass
        self.reset_scoring()
        status, out, _, _ = self.regrade("--rejudge")
        self.assertEqual(status, 0, out)
        self.assertEqual(sorted(self.judged), [f"kimi-r{i}" for i in range(5)])
        self.assertEqual(json.loads((self.out / "metrics.json").read_text())["teacher_clean_rate"], 1.0)

    def test_summary_gets_a_new_acceptance_where_build_puts_it(self):
        path = self.tree.root / "summary.json"
        names = ["cand-c01", "cand-c01-r1b3", "cand-c04-r1b0", "cand-c05-r1b0", "cand-c05-r1b0-r2b1"]
        path.write_text(json.dumps({"accepted": [{"name": n, "task_id": f"t-{n}"} for n in names], "built": []}))
        regrade.refresh_summary(self.tree.root, {"name": "cand-c04-r1b1", "task_id": "t-new", "accept": True})
        self.assertEqual([m["name"] for m in json.loads(path.read_text())["accepted"]],
                         ["cand-c01", "cand-c01-r1b3", "cand-c04-r1b0", "cand-c04-r1b1", "cand-c05-r1b0",
                          "cand-c05-r1b0-r2b1"])
        regrade.refresh_summary(self.tree.root, {"name": "cand-c05-r1b10", "task_id": "t-new2", "accept": True})
        self.assertEqual([m["name"] for m in json.loads(path.read_text())["accepted"]][-2:],
                         ["cand-c05-r1b10", "cand-c05-r1b0-r2b1"])
        regrade.refresh_summary(self.tree.root, {"name": "cand-c04-r1b1", "task_id": "t-new", "accept": False})
        self.assertNotIn("cand-c04-r1b1", [m["name"] for m in json.loads(path.read_text())["accepted"]])
        self.assertEqual([m["name"] for m in json.loads((self.tree.root / f"summary.json{regrade.ASIDE}").read_text())
                          ["accepted"]], names)

    def test_regrade_resumes_after_score_fails(self):
        quiet(regrade.main, ["repair", "--run", str(self.tree.root), "--cand", self.NAME])
        self.run_main(["validate", "--cand", self.NAME], {SOLUTION: 1})

        async def broken_score(*args, **kwargs):
            raise RuntimeError("judge unavailable")

        with mock.patch.object(regrade.task_builder, "score", broken_score):
            status, out, accept, _ = self.regrade()
        self.assertEqual(status, 1)
        self.assertIn("rerun the same command to resume", out)
        self.assertFalse((self.out / "metrics.json").exists())
        for name in ("metrics.json", "rollouts", "teacher_audit.json"):
            self.assertTrue((self.out / f"{name}{regrade.ASIDE}").exists(), name)
        self.calls.clear()
        status, out, accept, _ = self.regrade()
        self.assertEqual(status, 0, out)
        self.assertIn("resuming an interrupted re-score", out)
        self.assertEqual(self.calls, [])  # the grades were kept
        self.assertTrue(json.loads((self.out / "metrics.json").read_text())["accept"])
        state = json.loads((self.out / "regrade.json").read_text())
        self.assertEqual(sorted(Path(p).name for p in state["moved_aside"]),
                         sorted(f"{n}{regrade.ASIDE}" for n in ("metrics.json", "rollouts", "teacher_audit.json")))

    def test_regrade_resumes_grades_and_stops_before_scoring_when_a_grade_is_missing(self):
        quiet(regrade.main, ["repair", "--run", str(self.tree.root), "--cand", self.NAME])
        self.run_main(["validate", "--cand", self.NAME], {SOLUTION: 1})
        self.calls.clear()
        real = fake_grader({}, self.calls)

        async def flaky(source, task_dir, patch, *, run, out_dir):
            if patch.startswith("diff --git a/s3"):
                return {"reward": None, "reward_json": None, "exit_code": 124, "timed_out": True, "suites": {}}
            return await real(source, task_dir, patch, run=run, out_dir=out_dir)

        with mock.patch.object(regrade.sandbox, "grade", flaky), \
                mock.patch.object(regrade.rollouts, "probe_records",
                                  lambda probe: {self.stage["task_id"]: copy.deepcopy(self.records)}), \
                mock.patch.object(regrade, "hub_budget_ok", side_effect=AssertionError("scored too early")):
            status, out = quiet(regrade.main, ["regrade", "--run", str(self.tree.root), "--cand", self.NAME,
                                               "--slots", "6-0,6-1,6-2,6-3"])
        self.assertEqual(status, 1)
        self.assertIn("stopping before the re-score", out)
        self.assertTrue((self.out / "metrics.json").exists())
        self.assertFalse((self.out / f"metrics.json{regrade.ASIDE}").exists())
        self.assertEqual(len(self.calls), 3)
        self.calls.clear()
        status, out, accept, _ = self.regrade()  # only sc14-r3 is graded again
        self.assertEqual(status, 0, out)
        self.assertEqual([c["patch"] for c in self.calls], ["diff --git a/s3 b/s3\n"])



# ------------------------------------------------------------------------------------------------ gate mode

GATE_REPORT = "/logs/verifier/base-ctrf.json"
TYPECHECK = "yarn build:ts"
GATED = {"TWO_GATES": TWO_GATES, "MULTI_RC": MULTI_RC, "BARE_EXIT": BARE_EXIT, "ERREXIT_TOP": ERREXIT_TOP,
         "ERREXIT_CHECK": ERREXIT_CHECK}
GATE_IDS = {"TWO_GATES": "[gate] yarn build:js && yarn tsc -b packages/test-globals packages/jest-worker/src/__tests__",
            "MULTI_RC": "[gate] yarn build:js && yarn build:ts && yarn bundle:ts",
            "BARE_EXIT": "[gate] yarn build:js && yarn build:ts", "ERREXIT_TOP": "[gate] yarn build:js && yarn build:ts",
            "ERREXIT_CHECK": "[gate] yarn build:js && yarn build:ts"}


def gated(block: str, add_typecheck=TYPECHECK):
    """(repaired block, gated block, info) for an unrepaired fixture."""
    repaired, _ = regrade.repair_block(block)
    new, info = regrade.gate_block(block, repaired, report=GATE_REPORT, add_typecheck=add_typecheck)
    return repaired, new, info


def added_lines(before: str, after: str) -> list:
    """The lines `after` adds to `before`, which must otherwise be unchanged (in order)."""
    old, new, extra, i = before.split("\n"), after.split("\n"), [], 0
    for line in new:
        if i < len(old) and line == old[i]:
            i += 1
        else:
            extra.append(line)
    if i != len(old):
        raise AssertionError(f"line {i + 1} of the original is missing: {old[i]!r}")
    return extra


class GateRewrite(unittest.TestCase):
    def test_gated_steps_are_the_build_and_typecheck_steps_the_exits_guarded(self):
        steps = {name: [(s["kind"], s["text"]) for s in regrade.gated_steps(block)] for name, block in GATED.items()}
        self.assertEqual(steps["TWO_GATES"], [("build", 'yarn build:js 2>&1 | tee -a "$RUN_LOG"'), (
            "typecheck", 'yarn tsc -b packages/test-globals packages/jest-worker/src/__tests__ 2>&1 | tee -a "$RUN_LOG"')])
        self.assertEqual([t for _, t in steps["MULTI_RC"]], ["run_log yarn build:js", "run_log yarn build:ts",
                                                            "run_log yarn bundle:ts"])
        self.assertEqual(steps["ERREXIT_TOP"], [("build", 'yarn build:js 2>&1 | tee -a "$RUN_LOG"')])
        self.assertEqual(steps["ERREXIT_CHECK"], [("build", "run_log yarn build:js")])  # not the report check (kind check)
        for name, block in NOT_EARLY.items():
            self.assertEqual(regrade.gated_steps(block), [], name)

    def test_only_lines_are_added_and_the_block_never_exits_early(self):
        for name, block in GATED.items():
            with self.subTest(name):
                repaired, new, info = gated(block)
                self.assertEqual(info["gate_id"], GATE_IDS[name])
                extra = added_lines(repaired, new)
                self.assertEqual(len(extra), info["added_lines"])
                recorder = regrade.gate_recorder(info["gate_id"], [s["status"] for s in info["steps"]] +
                                                 ([info["typecheck_added"]["status"]] if info["typecheck_added"] else []),
                                                 GATE_REPORT).split("\n")
                self.assertEqual(extra[-len(recorder):], recorder)  # one recorder, after every other addition
                self.assertTrue(all(regrade.GATE_TAG in line for line in extra[:-len(recorder)]), extra)
                self.assertEqual(new.count("<<'GATE'"), 1)
                self.assertEqual([f for f in regrade.analyze(new) if f["severity"] == "early-exit"], [])
                self.assertEqual(regrade.analyze(new), regrade.analyze(repaired)[:0] + [
                    f for f in regrade.analyze(new) if f["severity"] != "early-exit"])

    def test_statuses_come_from_the_blocks_own_captures_or_new_lines(self):
        _, new, info = gated(TWO_GATES)
        self.assertEqual([(s["status"], s["how"]) for s in info["steps"]],
                         [("build_rc", "captured by the block"), ("types_rc", "captured by the block")])
        self.assertIsNone(info["typecheck_added"])  # tsc already typechecks
        self.assertIn('"${build_rc-1}" "${types_rc-1}" <<\'GATE\'', new)
        _, new, info = gated(ERREXIT_TOP)
        lines = new.split("\n")
        build = lines.index('yarn build:js 2>&1 | tee -a "$RUN_LOG"')
        self.assertEqual(lines[build + 1], "gate_rc_1=${PIPESTATUS[0]}  # gate-fix: exit status of `yarn build:js`")
        self.assertTrue(lines[build + 2].startswith('yarn build:ts 2>&1 | tee -a "$RUN_LOG"  # gate-fix: typecheck'))
        self.assertEqual(lines[build + 3], "gate_rc_2=${PIPESTATUS[0]}  # gate-fix: exit status of `yarn build:ts`")
        _, new, info = gated(ERREXIT_CHECK)
        lines = new.split("\n")
        build = lines.index("run_log yarn build:js")
        self.assertEqual(lines[build + 1:build + 4], [
            "gate_rc_1=$?  # gate-fix: exit status of `yarn build:js`",
            "run_log yarn build:ts  # gate-fix: typecheck too: a solution that does not typecheck fails the gate",
            "gate_rc_2=$?  # gate-fix: exit status of `yarn build:ts`"])
        self.assertEqual(info["recorder"], "at the end")  # the block ends with its report check, not `set -e`
        _, new, info = gated(MULTI_RC)
        self.assertEqual([s["status"] for s in info["steps"]], ["js_rc", "ts_rc", "bundle_rc"])
        self.assertIsNone(info["typecheck_added"])  # yarn build:ts is gated already
        self.assertEqual(info["recorder"], "before the closing `set -e`")
        self.assertTrue(new.rstrip().endswith("GATE\nset -e"))
        _, _, info = gated(BARE_EXIT, add_typecheck=None)
        self.assertEqual(info["gate_id"], "[gate] yarn build:js")  # no --add-typecheck: the guarded steps only

    def test_a_reassigned_status_is_copied_right_after_its_capture(self):
        block = BARE_EXIT.replace("jest_args=(", "build_rc=0\njest_args=(")
        repaired, _ = regrade.repair_block(block)
        new, info = regrade.gate_block(block, repaired, report=GATE_REPORT)
        lines = new.split("\n")
        capture = lines.index("build_rc=${PIPESTATUS[0]}")
        self.assertEqual(lines[capture + 1], 'gate_rc_1="${build_rc-1}"  # gate-fix: exit status of `yarn build:js`')
        self.assertEqual(info["steps"][0]["status"], "gate_rc_1")

    def test_refusals(self):
        cases = {"status consumed by a list": (INLINE, None),
                 "nothing to gate": (CLEAN, None),
                 "reads $?": (ERREXIT_TOP.replace("\nexport JEST_JUNIT_SUITE_NAME", '\necho "rc=$?"\nexport '
                                                                                    "JEST_JUNIT_SUITE_NAME"), None)}
        for name, (block, _) in cases.items():
            with self.subTest(name):
                repaired, _ = regrade.repair_block(block)
                with self.assertRaises(regrade.FixError):
                    regrade.gate_block(block, repaired, report=GATE_REPORT, add_typecheck=TYPECHECK)
        with self.assertRaises(regrade.FixError):  # not repaired yet
            regrade.gate_block(BARE_EXIT, BARE_EXIT, report=GATE_REPORT)
        repaired, new, _ = gated(BARE_EXIT)
        with self.assertRaises(regrade.FixError):  # gated already
            regrade.gate_block(BARE_EXIT, new, report=GATE_REPORT)
        with self.assertRaises(regrade.FixError):  # the base report is never written
            regrade.gate_block(BARE_EXIT, repaired, report="/logs/verifier/other-ctrf.json")

    def test_test_sh_frame_stays_byte_identical(self):
        _, new, _ = gated(MULTI_RC)
        head, _, tail = regrade.split_test_sh(frame_test_sh(regrade.repair_block(MULTI_RC)[0]))
        self.assertEqual(head + new + tail, frame_test_sh(new))

    def test_config_gets_the_gate_id_among_the_p2p_ids(self):
        config = {"f2p_node_ids": F2P, "p2p_node_ids": ["b", "e2e/x: y", "jest-worker: Farm > c"], "grade": {}}
        new = regrade.gated_config(config, "[gate] yarn build:js")
        self.assertEqual(new["p2p_node_ids"], ["[gate] yarn build:js", "b", "e2e/x: y", "jest-worker: Farm > c"])
        self.assertEqual(config["p2p_node_ids"], ["b", "e2e/x: y", "jest-worker: Farm > c"])  # not modified
        self.assertIs(regrade.gated_config(new, "[gate] yarn build:js"), new)
        self.assertEqual(regrade.gate_ids(new), ["[gate] yarn build:js"])
        with self.assertRaises(regrade.FixError):
            regrade.gated_config(dict(config, f2p_node_ids=["[gate] x"]), "[gate] x")


# ------------------------------------------------------------------------------------------------ gate: bash + grader

CTRF_STUB = """#!/bin/sh
printf '%s\\n' "$(basename "$0") $*" >> "$STUB_LOG"
out=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-o" ]; then out="$2"; shift; fi
  shift
done
case "$out" in
  *base-ctrf.json) printf '%s' "$CTRF_BASE" > "$out" ;;
  *new-ctrf.json) printf '%s' "$CTRF_NEW" > "$out" ;;
esac
exit 0
"""
PYTHON3 = shutil.which("python3", path="/usr/bin:/bin")


def ctrf(tests: dict) -> str:
    return json.dumps({"reportFormat": "CTRF", "results": {"tool": {"name": "junit-to-ctrf"}, "tests": [
        {"name": name, "status": status} for name, status in tests.items()], "summary": {"tests": len(tests)}}})


def run_gated(block: str, fail: str = "", gate_id: str = None, base_report: bool = True) -> dict:
    """Run a gated block like run_block (stub yarn/node, a junit-to-ctrf stub that writes base/new CTRF reports with
    the fixture's P2P/F2P tests passing; the REAL python3 runs the recorder), then the DeepSWE grader on the reports
    with P2P + the gate id and F2P whitelisted."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in ("app", "logs/verifier", "work", "bin", "tests"):
            (tmp / name).mkdir(parents=True)
        for name in ("yarn", "node", "npx", "tstyche"):
            (tmp / "bin" / name).write_text(STUB)
            (tmp / "bin" / name).chmod(0o755)
        (tmp / "bin/junit-to-ctrf").write_text(CTRF_STUB)
        (tmp / "bin/junit-to-ctrf").chmod(0o755)
        verifier = tmp / "logs/verifier"
        body = block.replace("/logs/verifier", str(verifier)).replace("/tmp/work", str(tmp / "work"))
        body = body.replace("/app", str(tmp / "app"))
        (tmp / "run.sh").write_text(f"set -uo pipefail\nexport RUN_LOG={verifier}/run.log\n{body}\n"
                                    f"echo REACHED_END > {tmp}/end\n")
        proc = subprocess.run([BASH, str(tmp / "run.sh")], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=60, env={"PATH": f"{tmp}/bin:/usr/bin:/bin", "HOME": str(tmp),
                                               "STUB_LOG": str(tmp / "stub.log"), "STUB_FAIL": fail,
                                               "CTRF_BASE": ctrf({P2P[0]: "passed"}) if base_report else "not json",
                                               "CTRF_NEW": ctrf({t: "passed" for t in F2P})})
        stub = (tmp / "stub.log").read_text() if (tmp / "stub.log").exists() else ""
        base = read_report(verifier / "base-ctrf.json")
        reward = None
        if gate_id:
            (tmp / "tests/config.json").write_text(json.dumps({
                "base_commit": "0" * 40, "f2p_node_ids": F2P, "p2p_node_ids": sorted(P2P + [gate_id]),
                "grade": {"format": "ctrf", "node_id": "name", "tool_label": "junit-to-ctrf",
                          "reports": [str(verifier / "base-ctrf.json"), str(verifier / "new-ctrf.json")]}}))
            graded = subprocess.run([sys.executable, "-B", str(DEEPSWE / TARGET / "tests/grader.py"), "grade"],
                                    capture_output=True, text=True, timeout=60,
                                    env={"PATH": "/usr/bin:/bin", "TESTS_DIR": str(tmp / "tests"),
                                         "VERIFIER_DIR": str(verifier), "APP_DIR": str(tmp / "app")})
            reward = json.loads((verifier / "reward.json").read_text()) if (verifier / "reward.json").exists() \
                else {"error": graded.stderr[-500:]}
        log = (verifier / "run.log").read_text() if (verifier / "run.log").exists() else ""
        return {"rc": proc.returncode, "end": (tmp / "end").exists(), "stub": stub, "base": base, "reward": reward,
                "suites": sum("--runTestsByPath" in line for line in stub.splitlines()),
                "converters": sum(line.startswith("junit-to-ctrf") for line in stub.splitlines()),
                "gate_lines": [l for l in log.splitlines() if regrade.GATE_MARK in l], "stderr": proc.stderr}


def read_report(path: Path) -> dict:
    try:
        return {t["name"]: t["status"] for t in json.loads(path.read_text())["results"]["tests"]}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


@unittest.skipUnless(BASH and PYTHON3, "bash or /usr/bin/python3 not found")
class GateRuns(unittest.TestCase):
    def test_the_gate_passes_when_everything_builds_and_the_task_scores_1(self):
        for name, block in GATED.items():
            with self.subTest(name):
                _, new, info = gated(block)
                result = run_gated(new, "", info["gate_id"])
                self.assertTrue(result["end"], result)
                self.assertEqual(result["base"], {P2P[0]: "passed", info["gate_id"]: "passed"})
                self.assertEqual((result["reward"]["reward"], result["reward"]["p2p_passed"], result["reward"]["p2p_total"]),
                                 (1, 2, 2), result["reward"])
                self.assertEqual(result["gate_lines"], [f"[verifier] gate-fix: {info['gate_id']}: passed "
                                                        f"(exit statuses: {' '.join(['0'] * len(info['gate_id'].split(' && ')))})"])

    def test_a_failed_build_or_typecheck_fails_the_gate_and_the_run_is_graded_0(self):
        failures = {"TWO_GATES": ["build:js", "tsc"], "MULTI_RC": ["build:js", "build:ts", "bundle:ts"],
                    "BARE_EXIT": ["build:js", "build:ts"], "ERREXIT_TOP": ["build:js", "build:ts"],
                    "ERREXIT_CHECK": ["build:js", "build:ts"]}
        for name, block in GATED.items():
            _, new, info = gated(block)
            for fail in failures[name]:
                with self.subTest(name=name, fail=fail):
                    result = run_gated(new, fail, info["gate_id"])
                    self.assertTrue(result["end"], result)                       # never an early exit...
                    self.assertEqual((result["suites"], result["converters"]), (2, 2))  # ...the suites still ran
                    self.assertEqual(result["base"][info["gate_id"]], "failed")
                    self.assertEqual(result["base"][P2P[0]], "passed")
                    reward = result["reward"]
                    self.assertEqual((reward["reward"], reward["f2p_passed"], reward["p2p_passed"], reward["p2p_total"]),
                                     (0, 2, 1, 2), reward)                        # graded: only the gate failed

    def test_a_missing_base_report_still_gets_the_gate_test(self):
        _, new, info = gated(BARE_EXIT)
        result = run_gated(new, "build:js", info["gate_id"], base_report=False)
        self.assertEqual(result["base"], {info["gate_id"]: "failed"})
        self.assertEqual((result["reward"]["reward"], result["reward"]["p2p_passed"]), (0, 0))
        self.assertTrue(any("unusable" in line for line in result["gate_lines"]), result["gate_lines"])

    def test_the_repaired_block_without_the_gate_scores_a_broken_build_1(self):  # why gate mode exists
        repaired, new, info = gated(BARE_EXIT)
        before = run_gated(repaired, "build:js", info["gate_id"])
        self.assertEqual(before["reward"]["reward"], 0)  # the gate id is whitelisted but missing: absence is failure
        self.assertNotIn(info["gate_id"], before["base"])
        no_gate = run_gated(repaired, "build:js")
        self.assertEqual(no_gate["base"], {P2P[0]: "passed"})  # all suites pass: graded on them alone, the build lost


# ------------------------------------------------------------------------------------------------ gate: type-only patch

@unittest.skipUnless(GIT, "git not found")
class TypeOnlySolution(unittest.TestCase):
    setUp, tearDown, git = BrokenSolution.setUp, BrokenSolution.tearDown, BrokenSolution.git
    solution, apply = BrokenSolution.solution, BrokenSolution.apply

    def test_type_error_goes_to_the_top_level_of_the_largest_ts_file(self):
        solution = self.solution(imported=True, big_new_file=False)
        typed, info = regrade.type_break_patch(solution)
        self.assertEqual((info["file"], info["new_file"]), ("src/farm.ts", False))
        self.assertEqual(info["before_line"], "+import reserve from './reserve';")  # zero indentation: top level
        good, bad = self.apply(solution), self.apply(typed)
        lines = bad["src/farm.ts"].splitlines()
        at = lines.index(regrade.TYPE_ERROR_LINE)
        self.assertEqual(lines[:at] + lines[at + 1:], good["src/farm.ts"].splitlines())
        self.assertEqual({k: v for k, v in bad.items() if k != "src/farm.ts"},
                         {k: v for k, v in good.items() if k != "src/farm.ts"})
        self.assertTrue(regrade.TYPE_ERROR_LINE.startswith("void Number.parseInt(0);"))  # valid JS, TS2345 in TS

    def test_new_file_gets_it_at_its_end_and_indented_spots_are_never_used(self):
        solution = self.solution(imported=True, big_new_file=True)
        typed, info = regrade.type_break_patch(solution)
        self.assertEqual((info["file"], info["new_file"]), ("src/reserve.ts", True))
        self.assertEqual(self.apply(typed)["src/reserve.ts"].splitlines()[-1], regrade.TYPE_ERROR_LINE)
        # a modified file whose only added declaration is indented (inside a class body): the type error goes before
        # a top-level declaration among the context lines instead, never inside the body
        indented = self.solution(imported=False, big_new_file=True).replace("+export function reserved(",
                                                                            "+  static reserved(")
        self.assertNotIn("+import", indented)
        self.assertEqual(regrade.type_break_patch(indented, "src/farm.ts")[1]["before_line"], " export class Farm {")
        self.assertEqual(regrade.break_patch(indented, "src/farm.ts")[1]["before_line"], "+    const value = a;")
        # (a syntax error may go inside a body; the type error may not: it must stay valid JavaScript)

    def test_a_top_level_context_declaration_is_used_when_no_added_line_is_top_level(self):
        farm = (self.repo / "src/farm.ts").read_text()
        (self.repo / "src/farm.ts").write_text(farm.replace("    return a;\n", "    const value = a;\n    return value;\n"))
        self.git("add", "-A")
        patch = self.git("diff", "--cached").stdout  # only indented lines added; `export class Farm` is context
        self.git("reset", "-q", "--hard")
        self.assertNotIn("\n+export", patch)
        typed, info = regrade.type_break_patch(patch)
        self.assertEqual((info["file"], info["before_line"]), ("src/farm.ts", " export class Farm {"))
        lines = self.apply(typed)["src/farm.ts"].splitlines()
        at = lines.index(regrade.TYPE_ERROR_LINE)
        self.assertEqual(lines[at - 2:at + 2], ["import {a} from './a';", "", regrade.TYPE_ERROR_LINE,
                                                "export class Farm {"])
        # a context declaration whose preceding line does not end a statement (or is not shown) is not used
        dangling = patch.replace("\n \n export class Farm {", "\n x +\n export class Farm {")
        self.assertNotEqual(dangling, patch)
        with self.assertRaises(regrade.FixError):
            regrade.type_break_patch(dangling)

    def test_javascript_only_or_nocheck_solutions_have_no_type_error_spot(self):
        solution = self.solution(imported=True, big_new_file=False)
        js = solution.replace("src/farm.ts", "src/farm.js").replace("src/reserve.ts", "src/reserve.js")
        with self.assertRaises(regrade.FixError):
            regrade.type_break_patch(js)
        nocheck = solution.replace("+import reserve from './reserve';", "+// @ts-nocheck\n+import reserve from './reserve';")
        nocheck = nocheck.replace("@@ -1,3 +1,4 @@", "@@ -1,3 +1,5 @@")
        _, info = regrade.type_break_patch(nocheck)
        self.assertEqual(info["file"], "src/reserve.ts")


# ------------------------------------------------------------------------------------------------ gate: commands

def canonical_configs(directory: Path):
    """The fixture tree writes config.json compactly; the pipeline writes it with common.write_json."""
    for path in directory.rglob("config.json"):
        path.write_text(regrade.config_text(json.loads(path.read_text())))


def gate_grader(calls: list, gate_id: str, outcomes=None):
    """sandbox.grade stand-in for gated tasks: per patch (reward, gate status); default: the reference and the empty
    patch build (gate passed; reward 1 / 0), a patch with the syntax or type error fails the gate (reward 0, the
    type-only one with every other test passing), anything else scores 0 with the gate passed."""
    outcomes = dict(outcomes or {})

    async def grade(source, task_dir, patch, *, run, out_dir):
        calls.append({"source": source, "task_dir": str(task_dir), "patch": patch, "out_dir": str(out_dir)})
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if patch in outcomes:
            reward, gate = outcomes[patch]
        elif regrade.BROKEN_LINE in patch:
            reward, gate = 0, "failed"
        elif regrade.TYPE_ERROR_LINE in patch:
            reward, gate = 0, "failed"
        else:
            reward, gate = (1 if patch == SOLUTION else 0), "passed"
        new_ok = reward == 1 or regrade.TYPE_ERROR_LINE in patch or (patch in outcomes and gate == "failed")
        applied = "[verifier] model.patch applied (10 bytes)\n" if patch else "[verifier] no model.patch submitted\n"
        (out_dir / "output.log").write_text(applied + f"[verifier] gate-fix: {gate_id}: {gate} (exit statuses: "
                                                      f"{'0' if gate == 'passed' else '2'})\n")
        base = {**{t: "passed" for t in P2P}, gate_id: gate}
        new = {F2P[0]: "passed", F2P[1]: "passed" if new_ok else "failed"}
        p2p_passed = sum(s == "passed" for s in base.values())
        return {"reward": reward, "exit_code": 0, "timed_out": False,
                "suites": {"base-ctrf.json": base, "new-ctrf.json": new},
                "reward_json": {"reward": reward, "f2p_total": 2, "f2p_passed": sum(s == "passed" for s in new.values()),
                                "p2p_total": 2, "p2p_passed": p2p_passed}}
    return grade


class GateCommands(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.tree = Tree(self.tmp, {"cand-c04-r1b1": TWO_GATES, "cand-c04-r1b0": BARE_EXIT,
                                    "cand-c01-r1b4": ERREXIT_CHECK}, accepted=("cand-c04-r1b0",))
        canonical_configs(self.tmp)
        self.calls = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def main(self, *argv):
        return quiet(regrade.main, [argv[0], "--run", str(self.tree.root), *argv[1:]])

    def files(self, name):
        stage = self.tree.tasks[name]
        cand, task = self.tree.root / name, Path(stage["task_dir"])
        scripts = [cand / "final/tests/test.sh", task / "tests/test.sh", cand / "author-0/run_tests.sh"]
        configs = [cand / "final/tests/config.json", task / "tests/config.json"]
        accepted = self.tree.accepted / stage["task_id"] / "tests"
        if accepted.exists():
            scripts.append(accepted / "test.sh")
            configs.append(accepted / "config.json")
        return scripts, configs

    def test_gate_repair_rewrites_every_live_copy_once(self):
        name = "cand-c04-r1b0"
        self.assertEqual(self.main("repair", "--cand", name)[0], 0)
        scripts, configs = self.files(name)
        before = {p: p.read_text() for p in scripts + configs}
        history = [self.tree.root / name / "provisional/tests/test.sh", self.tree.root / name / "provisional/tests/config.json"]
        history_before = [p.read_text() for p in history]
        status, out = self.main("repair", "--gate", "--add-typecheck", TYPECHECK, "--cand", name)
        self.assertEqual(status, 0, out)
        _, new_block, info = gated(BARE_EXIT)
        for path in scripts:
            self.assertEqual(path.read_text(), new_block + "\n" if path.name == "run_tests.sh" else frame_test_sh(new_block))
            self.assertEqual(path.with_name(path.name + regrade.GATE_BACKUP).read_text(), before[path])
        for path in configs:
            config = json.loads(path.read_text())
            self.assertEqual(config["p2p_node_ids"], sorted(P2P + [info["gate_id"]]))
            self.assertEqual(config["f2p_node_ids"], F2P)
            self.assertEqual(path.read_text(), regrade.config_text(config))
            self.assertEqual(path.with_name(path.name + regrade.GATE_BACKUP).read_text(), before[path])
        self.assertEqual([p.read_text() for p in history], history_before)  # records of what ran: untouched
        log = [json.loads(l) for l in (self.tree.root / regrade.LOG).read_text().splitlines()]
        gate_log = [e for e in log if e["action"] == "gate_repair"]
        self.assertEqual(sorted(e["copy"] for e in gate_log), ["accepted", "config:accepted", "config:final",
                                                              "config:task", "final", "run_tests", "task"])
        self.assertTrue(all(e["mode"] == "gate" and e["gate_id"] == info["gate_id"] for e in gate_log))
        record = json.loads((self.tree.root / name / regrade.FIX / regrade.GATE_DIR / "repair.json").read_text())
        self.assertEqual((record["gate_id"], record["typecheck_added"]["command"]), (info["gate_id"], TYPECHECK))
        self.assertIn("workspace archive holds no copy of the test script", out)
        snap = snapshot(self.tmp)
        status, out = self.main("repair", "--gate", "--add-typecheck", TYPECHECK, "--cand", name)
        self.assertEqual(status, 0)
        self.assertIn("already gated", out)
        self.assertEqual(snapshot(self.tmp), snap)

    def test_gate_repair_needs_the_early_exit_repair_first(self):
        status, out = self.main("repair", "--gate", "--cand", "cand-c04-r1b1")
        self.assertEqual(status, 1)
        self.assertIn("NOT GATED", out)
        self.assertIn("run repair first", out)

    def test_restore_puts_the_pre_gate_copies_back(self):
        name = "cand-c01-r1b4"
        self.main("repair", "--cand", name)
        scripts, configs = self.files(name)
        before = {p: p.read_text() for p in scripts + configs}
        self.main("repair", "--gate", "--add-typecheck", TYPECHECK, "--cand", name)
        self.assertNotEqual({p: p.read_text() for p in before}, before)
        status, out = self.main("restore", "--gate", "--cand", name, "--reason", "test")
        self.assertEqual(status, 0, out)
        self.assertEqual({p: p.read_text() for p in before}, before)
        log = [json.loads(l) for l in (self.tree.root / regrade.LOG).read_text().splitlines()]
        self.assertEqual(len([e for e in log if e["action"] == "gate_restore"]), 5)

    def test_validate_gate_runs_four_checks_on_the_gate_test(self):
        name = "cand-c04-r1b1"
        self.main("repair", "--cand", name)
        status, out = self.main("validate", "--gate", "--cand", name)
        self.assertEqual(status, 1)
        self.assertIn("run repair --gate", out)  # not gated yet
        self.main("repair", "--gate", "--cand", name)
        gate_id = GATE_IDS["TWO_GATES"]
        with mock.patch.object(regrade.sandbox, "grade", gate_grader(self.calls, gate_id)):
            status, out = self.main("validate", "--gate", "--cand", name)
        self.assertEqual(status, 0, out)
        record = json.loads((self.tree.root / name / regrade.FIX / regrade.GATE_DIR / "validate.json").read_text())
        self.assertEqual(record["checks"], {"solution_scores_1": True, "empty_scores_0": True, "broken_graded_0": True,
                                            "typeonly_graded_0": True})
        self.assertEqual({n: r["gate"] for n, r in record["results"].items()},
                         {"solution": "passed", "empty": "passed", "broken": "failed", "typeonly": "failed"})
        self.assertTrue(record["results"]["typeonly"]["only_gate_failed"])
        self.assertEqual(record["typeonly"]["inserted"], regrade.TYPE_ERROR_LINE)
        self.assertEqual(len(self.calls), 4)
        self.assertIn("only the gate failed: yes", out)
        # a type error that does not fail the gate is a failed check
        typeonly = (self.tree.root / name / regrade.FIX / regrade.GATE_DIR / "validate/typeonly.patch").read_text()
        with mock.patch.object(regrade.sandbox, "grade", gate_grader(self.calls, gate_id, {typeonly: (1, "passed")})):
            status, out = self.main("validate", "--gate", "--cand", name)
        self.assertEqual(status, 1)
        self.assertIn("typeonly_graded_0: FAILED", out)

    def test_dry_runs_write_nothing(self):
        for name in ("cand-c04-r1b1", "cand-c04-r1b0"):
            self.main("repair", "--cand", name)
        before = snapshot(self.tmp)

        async def no_grade(*args, **kwargs):
            raise AssertionError("dry run graded")

        with mock.patch.object(regrade.sandbox, "grade", no_grade):
            status, out = self.main("repair", "--gate", "--add-typecheck", TYPECHECK, "--dry-run", "--cand",
                                    "cand-c04-r1b1", "cand-c04-r1b0")
            self.assertEqual(status, 0, out)
            self.assertIn("would rewrite", out)
            self.assertEqual(snapshot(self.tmp), before)
            self.main("repair", "--gate", "--add-typecheck", TYPECHECK, "--cand", "cand-c04-r1b1")
            before = snapshot(self.tmp)
            status, out = self.main("validate", "--gate", "--dry-run", "--cand", "cand-c04-r1b1")
            self.assertEqual(status, 0, out)
            self.assertIn("type-only broken solution", out)
            status, out = self.main("restore", "--gate", "--dry-run", "--cand", "cand-c04-r1b1")
            self.assertIn("would restore", out)
            status, out = self.main("tidy-accepted", "--dry-run")
            self.assertIn("would move", out)
        self.assertEqual(snapshot(self.tmp), before)


class GateRegrade(unittest.TestCase):
    NAME = "cand-c04-r1b1"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.tree = Tree(self.tmp, {self.NAME: TWO_GATES})
        canonical_configs(self.tmp)
        self.stage = self.tree.tasks[self.NAME]
        self.records = lost_slot_records()
        self.out = self.tree.score_round(self.NAME, self.records)
        self.audit = {f"kimi-r{i}": {"cheated": i == 0, "categories": []} for i in range(5)}
        (self.out / "teacher_audit.json").write_text(json.dumps(self.audit))
        self.calls = []
        self.gate_id = GATE_IDS["TWO_GATES"]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_regrade(self, argv, grader, *, accepted=False):
        self.judged = []

        async def judge_pass(record, instruction, rubric):
            self.judged.append(record["slot"])
            return {"cheated": False, "categories": []}

        async def analyze(*args, **kwargs):
            return {}

        async def similarity(*args, **kwargs):
            return {"score": 7}

        accept = mock.AsyncMock()
        build = types.SimpleNamespace(root=self.tree.root, run="fake-run", target=TARGET, repo="jestjs/jest",
                                      fa_dir=self.tmp / "analysis")
        with mock.patch.object(regrade.rollouts, "probe_records",
                               lambda probe: {self.stage["task_id"]: copy.deepcopy(self.records)}), \
                mock.patch.object(regrade, "hub_budget_ok", return_value=True), \
                mock.patch.object(regrade, "make_build", return_value=build), \
                mock.patch.object(regrade.task_builder, "judge_pass", judge_pass), \
                mock.patch.object(regrade.task_builder, "analyze", analyze), \
                mock.patch.object(regrade.task_builder, "similarity", similarity), \
                mock.patch.object(regrade.task_builder, "accept", accept), \
                mock.patch.object(regrade.sandbox, "grade", grader):
            status, out = quiet(regrade.main, [argv[0], "--run", str(self.tree.root), *argv[1:]])
        return status, out, accept

    def first_pass(self):
        """repair, validate and regrade (+ re-score) as the first pass did: the four lost slots pass the suites."""
        quiet(regrade.main, ["repair", "--run", str(self.tree.root), "--cand", self.NAME])
        rewards = {r["patch"]: 1 for r in self.records if r["status"] == "infra"}
        rewards[SOLUTION] = 1
        status, out, _ = self.run_regrade(["validate", "--cand", self.NAME], fake_grader(rewards, self.calls))
        self.assertEqual(status, 0, out)
        status, out, _ = self.run_regrade(["regrade", "--cand", self.NAME, "--slots", "6-0,6-1,6-2,6-3"],
                                          fake_grader(rewards, self.calls))
        self.assertEqual(status, 0, out)
        metrics = json.loads((self.out / "metrics.json").read_text())
        self.assertEqual((metrics["student_rate"], metrics["accept"]), (1.0, False))  # 5/5: too easy

    def test_gate_regrade_rescores_with_the_gate_and_keeps_the_first_pass_aside(self):
        self.first_pass()
        first_metrics = (self.out / "metrics.json").read_text()
        first_regrade = json.loads((self.out / "regrade.json").read_text())
        self.main_gate()
        lost = {r["patch"]: (0, "failed") for r in self.records if r["status"] == "infra"}
        status, out, accept = self.run_regrade(["regrade", "--gate", "--cand", self.NAME, "--slots", "6-0,6-1,6-2,6-3"],
                                               gate_grader(self.calls, self.gate_id, lost))
        self.assertEqual(status, 0, out)
        # the first pass's outputs are kept as *.pre-gate-fix, the original scoring stays *.pre-regrade
        self.assertEqual((self.out / f"metrics.json{regrade.GATE_BACKUP}").read_text(), first_metrics)
        self.assertEqual(json.loads((self.out / f"regrade.json{regrade.GATE_BACKUP}").read_text()), first_regrade)
        self.assertTrue((self.out / f"regrade{regrade.GATE_BACKUP}").is_dir())
        for name in ("rollouts", "teacher_audit.json"):
            self.assertTrue((self.out / f"{name}{regrade.GATE_BACKUP}").exists(), name)
            self.assertTrue((self.out / f"{name}{regrade.ASIDE}").exists(), name)
        state = json.loads((self.out / "regrade.json").read_text())
        self.assertEqual((state["mode"], state["gate_id"]), ("gate", self.gate_id))
        self.assertEqual({s: (v["old"]["status"], v["new"]["status"], v["grade"]["gate"]) for s, v in state["slots"].items()},
                         {f"sc14-r{i}": ("infra", "fail", "failed") for i in range(4)})
        self.assertEqual(state["old_metrics"]["student_rate"], 1.0)        # the first pass's re-score
        self.assertEqual(state["first_metrics"]["valid"], [1, 5])          # the original scoring
        metrics = json.loads((self.out / "metrics.json").read_text())
        self.assertEqual((metrics["student_rate"], metrics["teacher_clean_rate"], metrics["valid"], metrics["accept"],
                          metrics["similarity"]), (0.2, 0.8, [5, 5], True, 7))
        self.assertEqual(self.judged, [])  # every kimi verdict kept from the original audit (kimi-r0 cheated)
        self.assertEqual(state["teacher_verdicts"]["kept"], [f"kimi-r{i}" for i in range(5)])
        accept.assert_awaited_once()
        for name in ("round-1/seeds.json", "summary.json"):
            self.assertTrue((self.tree.root / f"{name}{regrade.GATE_BACKUP}").exists(), name)
        summary = json.loads((self.tree.root / "summary.json").read_text())
        self.assertEqual([m["name"] for m in summary["accepted"]], [self.NAME])
        log = [json.loads(l) for l in (self.tree.root / regrade.LOG).read_text().splitlines()]
        gate_actions = [e["action"] for e in log if e.get("mode") == "gate"]
        for action in ("gate_repair", "gate_validate", "move_aside", "regrade", "score", "accept", "seeds", "summary"):
            self.assertIn(action, gate_actions)
        self.calls.clear()
        status, out, accept = self.run_regrade(["regrade", "--gate", "--cand", self.NAME, "--slots", "6-0,6-1,6-2,6-3"],
                                               gate_grader(self.calls, self.gate_id, lost))
        self.assertIn("already re-scored", out)
        self.assertEqual(self.calls, [])

    def main_gate(self):
        status, out = quiet(regrade.main, ["repair", "--gate", "--run", str(self.tree.root), "--cand", self.NAME])
        self.assertEqual(status, 0, out)
        status, out, _ = self.run_regrade(["validate", "--gate", "--cand", self.NAME],
                                          gate_grader(self.calls, self.gate_id))
        self.assertEqual(status, 0, out)

    def test_gate_regrade_needs_a_gate_validation(self):
        self.first_pass()
        quiet(regrade.main, ["repair", "--gate", "--run", str(self.tree.root), "--cand", self.NAME])
        status, out, _ = self.run_regrade(["regrade", "--gate", "--cand", self.NAME, "--slots", "6-0"],
                                          gate_grader(self.calls, self.gate_id))
        self.assertEqual(status, 1)
        self.assertIn("no passing validate.json", out)

    def test_confirm_regrades_the_passes_and_rescores_only_on_a_change(self):
        self.records = [make_record("kimi", i, "pass") for i in range(5)] + \
                       [make_record("sc14", i, "fail" if i else "pass", patch=f"diff --git a/s{i} b/s{i}\n")
                        for i in range(5)]
        for i, record in enumerate(self.records):
            record["patch"] = f"diff --git a/{record['slot']} b/{record['slot']}\n"
        shutil.rmtree(self.out)
        shutil.rmtree(self.tree.root / "round-1/probe-1234abcd")
        self.out = self.tree.score_round(self.NAME, self.records, accept=True)
        (self.out / "teacher_audit.json").write_text(json.dumps(self.audit))
        quiet(regrade.main, ["repair", "--run", str(self.tree.root), "--cand", self.NAME])
        self.main_gate()
        passes = [r["slot"] for r in self.records if r["status"] == "pass"]
        metrics_before = (self.out / "metrics.json").read_text()
        # every pass still passes with the gate: confirmed, nothing re-scored
        status, out, accept = self.run_regrade(["regrade", "--gate", "--cand", self.NAME, "--slots", "passing",
                                                "--confirm"], gate_grader(self.calls, self.gate_id, {
                                                    r["patch"]: (1, "passed") for r in self.records}))
        self.assertEqual(status, 0, out)
        self.assertIn("confirmed", out)
        self.assertEqual(sorted(c["patch"] for c in self.calls if "regrade" in c["out_dir"]),
                         sorted(r["patch"] for r in self.records if r["status"] == "pass"))
        self.assertEqual((self.out / "metrics.json").read_text(), metrics_before)
        self.assertFalse((self.out / f"metrics.json{regrade.GATE_BACKUP}").exists())
        accept.assert_not_awaited()
        state = json.loads((self.out / "regrade.json").read_text())
        self.assertEqual(sorted(state["slots"]), sorted(passes))
        self.assertTrue(state["confirmed"]["unchanged"])
        # two kimi passes do not typecheck: re-scored with those two slots only; clean kimi 2/5 < 0.6 -> not accepted
        self.calls.clear()
        broken = {r["patch"]: (0, "failed") for r in self.records if r["slot"] in ("kimi-r1", "kimi-r2")}
        (self.out / "regrade.json").unlink()
        status, out, accept = self.run_regrade(["regrade", "--gate", "--cand", self.NAME, "--slots", "passing",
                                                "--confirm", "--unaccept"],
                                               gate_grader(self.calls, self.gate_id, {
                                                   **{r["patch"]: (1, "passed") for r in self.records}, **broken}))
        self.assertEqual(status, 0, out)
        state = json.loads((self.out / "regrade.json").read_text())
        self.assertEqual(state["replaced"], ["kimi-r1", "kimi-r2"])
        metrics = json.loads((self.out / "metrics.json").read_text())
        self.assertEqual((metrics["teacher_clean_rate"], metrics["accept"]), (0.4, False))  # kimi-r0 cheated + 2 fails
        self.assertEqual(self.judged, [])  # the unchanged kimi passes keep their verdicts
        accept.assert_not_awaited()
        self.assertIn("NO LONGER ACCEPTED", out)


class TidyAccepted(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.tree = Tree(self.tmp, {"cand-c04-r1b0": BARE_EXIT}, accepted=("cand-c04-r1b0",))
        canonical_configs(self.tmp)
        stage = self.tree.tasks["cand-c04-r1b0"]
        self.task_id = stage["task_id"]
        toml = (f'[metadata]\ntask_id = "{self.task_id}"\n[environment]\ndocker_image = "snapshot:fake"\n'
                '[verifier]\ntimeout_sec = 1800.0\n')
        for task in (Path(stage["task_dir"]), self.tree.accepted / self.task_id):
            (task / "task.toml").write_text(toml)
        workspace = self.tree.accepted / f"{self.task_id}.workspace"
        (workspace / "manifest.json").write_text("{}")
        (self.tree.accepted / "tasks.jsonl").write_text(
            json.dumps(rollouts.dataset_row(self.tree.accepted / self.task_id)) + "\n")
        metrics = dict(stage, accept=True)
        (self.tree.accepted / self.task_id / "provenance.json").write_text(json.dumps(metrics))
        (self.tree.root / "round-1" / "cand-c04-r1b0").mkdir(parents=True)
        (self.tree.root / "round-1/cand-c04-r1b0/metrics.json").write_text(json.dumps(metrics))
        quiet(regrade.main, ["repair", "--run", str(self.tree.root), "--cand", "cand-c04-r1b0"])
        quiet(regrade.main, ["repair", "--gate", "--run", str(self.tree.root), "--cand", "cand-c04-r1b0"])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_backups_leave_the_published_task_and_accepted_is_checked(self):
        tests = self.tree.accepted / self.task_id / "tests"
        backups = sorted(p.name for p in tests.iterdir() if p.name.endswith((regrade.BACKUP, regrade.GATE_BACKUP)))
        self.assertEqual(backups, ["config.json.pre-gate-fix", "test.sh.pre-early-exit-fix", "test.sh.pre-gate-fix"])
        contents = {name: (tests / name).read_bytes() for name in backups}
        problems = regrade.check_accepted(self.tree.root)
        self.assertTrue(any("stray backups" in p for p in problems), problems)
        status, out = quiet(regrade.main, ["tidy-accepted", "--run", str(self.tree.root)])
        self.assertEqual(status, 0, out)
        self.assertIn("accepted/ consistent", out)
        self.assertEqual(sorted(p.name for p in tests.iterdir()), ["config.json", "test.sh"])
        kept = self.tree.root / regrade.BACKUPS_DIR / "accepted" / self.task_id / "tests"
        self.assertEqual({name: (kept / name).read_bytes() for name in backups}, contents)
        self.assertEqual(regrade.check_accepted(self.tree.root), [])
        log = [json.loads(l) for l in (self.tree.root / regrade.LOG).read_text().splitlines()]
        self.assertEqual(len([e for e in log if e["action"] == "tidy_accepted"]), 3)
        # a published file that no longer matches its packaged task is a problem
        (tests / "test.sh").write_text("changed\n")
        self.assertTrue(any("test.sh differs" in p for p in regrade.check_accepted(self.tree.root)))

    def test_unaccept_moves_the_task_out_and_drops_its_row(self):
        rows = (self.tree.accepted / "tasks.jsonl").read_text()
        moved = regrade.unaccept(self.tree.root, self.task_id, {"mode": "gate"})
        backups = self.tree.root / regrade.BACKUPS_DIR
        self.assertFalse((self.tree.accepted / self.task_id).exists())
        self.assertFalse((self.tree.accepted / f"{self.task_id}.workspace").exists())
        self.assertTrue((backups / "unaccepted" / self.task_id / "tests/test.sh").exists())
        self.assertTrue((backups / "unaccepted" / f"{self.task_id}.workspace/workspace.tar.gz").exists())
        self.assertEqual((self.tree.accepted / "tasks.jsonl").read_text(), "")
        self.assertEqual((backups / "accepted" / f"tasks.jsonl{regrade.GATE_BACKUP}").read_text(), rows)
        self.assertEqual(len(moved), 3)
        entry = json.loads((self.tree.root / regrade.LOG).read_text().splitlines()[-1])
        self.assertEqual((entry["action"], entry["mode"], entry["task_id"]), ("unaccept", "gate", self.task_id))


if __name__ == "__main__":
    unittest.main()
