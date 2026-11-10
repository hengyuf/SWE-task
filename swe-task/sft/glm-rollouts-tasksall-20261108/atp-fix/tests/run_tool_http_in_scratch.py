"""Runs `tools/check_hengyuf_changes.py http` (an unchanged subcommand) of the given app root with the temporary
directory it creates (hard-coded under /tmp) redirected into a private scratch dir. Offline: the subcommand uses a stub
`send` and a test-only credential string set in this throwaway process; it reads no .env/.api.
Run: <atp>/runtime/controller/bin/python -I -B <this file> <app root> <private scratch dir>
"""
import runpy
import sys
import tempfile
from pathlib import Path

app, scratch = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
_mkdtemp = tempfile.mkdtemp


def mkdtemp(suffix=None, prefix=None, dir=None):
    return _mkdtemp(suffix, prefix, str(scratch))


tempfile.mkdtemp = mkdtemp
sys.argv = [str(app / "tools/check_hengyuf_changes.py"), "http"]
runpy.run_path(sys.argv[0], run_name="__main__")
