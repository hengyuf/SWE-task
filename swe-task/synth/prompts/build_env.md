# Build a DeepSWE-style environment for {{repo}}

The sandbox was started from the DeepSWE image of task {{target}} ({{language}}): it already contains the toolchain
and the test reporter DeepSWE uses for this language. The controller has replaced /app with {{repo}} at commit
{{sha}} on branch `{{branch}}` (other branches, unrelated tags and reflogs removed; `core.hooksPath` disabled).
Synthetic tasks will later be cut from this state and solved and graded OFFLINE, so everything needed to build and
test the repository must be installed now.

## Reference (read-only, in ./reference)
The target task's `Dockerfile`, `test.sh` and `config.json`. Follow the same conventions: repository at /app,
dependencies preinstalled, the same reporter and report format (`grade` section: {{grade_cfg}}), no git hooks.

{{sandbox}}

## Do
1. If the image carries an installation of the previous project (e.g. an installed package or linked module of
   the target repository), remove it. Leave system toolchains alone unless {{repo}} needs a different version.
2. Install everything needed to build and test {{repo}} offline (dependencies, tools, the {{report_format}}
   reporter as in the reference). Prefer in-place or editable installs; never install {{repo}} itself as a copied
   package outside /app.
3. Do not commit. Leave `git status --porcelain` empty in /app (add untracked build outputs to `.git/info/exclude`
   if needed). Do not rely on exported environment variables: anything the tests need must work from a fresh
   non-login shell (put tools on the default PATH, e.g. symlinks in /usr/local/bin).
4. Choose a test command that runs a broad, reliable part of the test suite (at least a few dozen tests) in under
   15 minutes and needs no downloads at test time (the controller re-runs it with networking disabled).

## Write here
- `setup.sh`: the exact idempotent commands you ran for steps 1-3 (bash, run as root with cwd /app).
- `environment/Dockerfile`: a deep-swe-style recipe: `FROM {{base_image}}`, replace /app with {{repo}} at {{sha}}
  (git time-travel as in the reference), run the setup steps, `WORKDIR /app`.
- `run_tests.sh`: a bash script run with cwd /app that runs your test command and writes ONE {{report_format}}
  report to the path given in `$REPORT`; it must exit 0 even when tests fail.
- `result.json`: `{"status": "ok" | "failed", "summary": str, "test_command": str, "approx_tests": int}`.
Before finishing, upload `run_tests.sh` and run `REPORT=/tmp/work/report bash /tmp/work/run_tests.sh` in the
sandbox twice; check the report lists the tests and that the same tests pass both times.
