"""Reviewer (round 1): run `tools/check_hengyuf_changes.py <check> [args]` of an app root with every tempfile.mkdtemp
(the `http` check hard-codes dir="/tmp") and tempfile's default dir forced into a private scratch dir.
Run: <atp>/runtime/controller/bin/python -I -B verify_run_tool.py <app root> <private scratch dir> <check> [args...]
"""
import sys

sys.dont_write_bytecode = True

import runpy  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

app, scratch, check = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3]
allowed = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-tasksall-20261108/atp-fix/"
               "scratch").resolve()
if allowed not in scratch.parents or allowed not in app.parents:
    raise SystemExit(f"app and scratch must be under {allowed}")
scratch.mkdir(parents=True, exist_ok=True)
_mkdtemp = tempfile.mkdtemp


def mkdtemp(suffix=None, prefix=None, dir=None):
    return _mkdtemp(suffix, prefix, str(scratch))


tempfile.mkdtemp = mkdtemp
tempfile.tempdir = str(scratch)
sys.argv = [str(app / "tools/check_hengyuf_changes.py"), check, *sys.argv[4:]]
runpy.run_path(sys.argv[0], run_name="__main__")
