"""Sensitivity check: plausible regressions of the capture fix must make the checks FAIL.

Each mutation is applied to a COPY of atp-fix/app/scripts/deepswe_capture.py inside a private mirror app under the
scratch dir (scripts/ = this app's scripts with the mutated file, upstream/, tools/check_hengyuf_changes.py, configs/).
The mirror's `tools/check_hengyuf_changes.py capture` (or tests/test_capture_fix.py unit) then has to exit non-zero.
Run: <atp>/runtime/controller/bin/python -I -B <this file> <private scratch dir> [mutation ...]
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FIX = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-tasksall-20261108/atp-fix")
ATP = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp")
PY = ATP / "runtime/controller/bin/python"
WAIT = """    cancelled = False
    while not call.done():
        try:
            await asyncio.wait((call,))
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        if not call.cancelled():
            call.exception()  # retrieved: a cancelled caller sees only the cancellation
        raise asyncio.CancelledError
    return call.result()
"""
MUTATIONS = {  # name: (old, new, checker)
    "no-wait-on-cancel": (WAIT, "    return await call\n", "tool"),
    "default-executor": ("run_in_executor(_STREAM_IO, function, *args)", "run_in_executor(None, function, *args)", "tool"),
    "no-fsync": ("                stream.write(data)\n                os.fsync(stream.fileno())\n",
                 "                stream.write(data)\n", "tool"),
    "write-on-loop": ("                await _in_stream_io(persist, chunk)\n", "                persist(chunk)\n", "tool"),
    "no-exception-retrieval": ("            call.exception()  # retrieved: a cancelled caller sees only the cancellation\n",
                               "            pass\n", "unit"),
}


def main():
    scratch = Path(sys.argv[1]).resolve()
    selected = sys.argv[2:] or list(MUTATIONS)
    source = (FIX / "app/scripts/deepswe_capture.py").read_text()
    failures = []
    for name in selected:
        old, new, checker = MUTATIONS[name]
        if source.count(old) != 1:
            raise SystemExit(f"{name}: anchor not found exactly once")
        mirror = Path(tempfile.mkdtemp(prefix=f"mutant-{name}-", dir=scratch)) / "app"
        for sub in ("scripts", "upstream", "tools", "configs"):
            (mirror / sub).mkdir(parents=True)
        for path in (ATP / "scripts").glob("*.py"):
            shutil.copyfile(path, mirror / "scripts" / path.name)
        for path in list((ATP / "upstream").glob("*.py")) + [FIX / "app/upstream/deepswe_capture.py"]:
            shutil.copyfile(path, mirror / "upstream" / path.name)
        shutil.copyfile(FIX / "app/tools/check_hengyuf_changes.py", mirror / "tools/check_hengyuf_changes.py")
        shutil.copyfile(ATP / "configs/deepswe.json", mirror / "configs/deepswe.json")
        mutant = mirror / "scripts/deepswe_capture.py"
        mutant.write_text(source.replace(old, new, 1))
        if checker == "tool":
            command = [str(PY), "-I", "-B", str(mirror / "tools/check_hengyuf_changes.py"), "capture",
                       "--scratch", str(mirror.parent)]
        else:
            command = [str(PY), "-I", "-B", str(FIX / "tests/test_capture_fix.py"), str(mirror.parent), "unit", str(mutant)]
        result = subprocess.run(command, cwd=mirror.parent, capture_output=True, text=True, timeout=100)
        failed_lines = [line for line in result.stdout.splitlines() if line.startswith("FAIL")]
        caught = result.returncode != 0
        print(("CAUGHT " if caught else "MISSED ") + f"{name}: exit {result.returncode}; first failure: "
              + (failed_lines[0] if failed_lines else (result.stderr.strip().splitlines() or ["-"])[-1]), flush=True)
        if not caught:
            failures.append(name)
    print("RESULT", "FAIL" if failures else "PASS", f"(missed: {failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
