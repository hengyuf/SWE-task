"""Explicit checks of the campaign-side patcher (atp-fix/capture_patch.py, next tasksall version) and of the read-only
glm-rollouts-sftv3 patcher (used by the live sftv3 re-run) against Baihe's drain() and the fixed app file.

Every patch_file() call runs on a COPY in the private scratch dir; no source file is ever patched.
Run: <atp>/runtime/controller/bin/python -I -B <this file> <private scratch dir>
         [--app-script <fixed deepswe_capture.py>] [--patcher <new capture_patch.py>]
Defaults: the staged atp-fix files; after the install pass the installed paths.
"""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile

FIX = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-tasksall-20261108/atp-fix")
BAIHE = Path("/lustre/fsw/portfolios/nemotron/projects/nemotron_evals_research/users/baiheh/nano-rl/"
             "agent-teacher-pipelines/scripts/deepswe_capture.py")
SFTV3 = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-sftv3-20261107/capture_patch.py")
TASKSALL_WRAPPER = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-tasksall-20261108/"
                        "probe_worker.py")
BAIHE_SHA256 = "b1babe29bfe228fbc3808bdc6bffb6daac475fe798e9dd17720744b186949fe8"
FAILED = []


def expect(condition, message):
    print(("PASS " if condition else "FAIL ") + message, flush=True)
    if not condition:
        FAILED.append(message)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scratch", type=Path)
    parser.add_argument("--app-script", type=Path, default=FIX / "app/scripts/deepswe_capture.py")
    parser.add_argument("--patcher", type=Path, default=FIX / "capture_patch.py")
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="capture-patch-", dir=args.scratch.resolve()))
    new, old = load(args.patcher, "new_capture_patch"), load(SFTV3, "sftv3_capture_patch")
    original, fixed = BAIHE.read_text(), args.app_script.read_text()
    expect(sha(BAIHE) == BAIHE_SHA256, "Baihe's deepswe_capture.py is still b1babe29")
    expect((new.OLD, new.NEW, new.MARK) == (old.OLD, old.NEW, old.MARK),
           "new patcher keeps OLD/NEW/MARK byte-identical to glm-rollouts-sftv3's (same fallback patch)")
    expect(callable(new.patch_file) and isinstance(new.UPSTREAM, tuple) and all(new.UPSTREAM), "API: patch_file, UPSTREAM")

    def copy(label, text, pycache=False):
        directory = root / label / "scripts"
        directory.mkdir(parents=True)
        path = directory / "deepswe_capture.py"
        path.write_text(text)
        if pycache:
            (directory / "__pycache__").mkdir()
            (directory / "__pycache__/deepswe_capture.cpython-313.pyc").write_bytes(b"stale")
        return path

    def outcome(patcher, path):
        before = path.read_bytes(), path.stat().st_mtime_ns
        try:
            result = patcher.patch_file(path)
        except RuntimeError as error:
            result = "raised: " + str(error).split(": ", 1)[-1]
        unchanged = (path.read_bytes(), path.stat().st_mtime_ns) == before
        leftovers = sorted(p.name for p in path.parent.iterdir() if p.name not in ("deepswe_capture.py", "__pycache__"))
        return result, unchanged, leftovers

    # 1. Baihe's drain(): patched to NEW (= sftv3's), stale bytecode removed, then "already".
    path = copy("original", original, pycache=True)
    result, unchanged, leftovers = outcome(new, path)
    expect(result == "patched" and path.read_text() == original.replace(new.OLD, new.NEW, 1) and not leftovers,
           f"original -> {result}; text == OLD replaced by NEW; no temp file left")
    expect(not (path.parent / "__pycache__/deepswe_capture.cpython-313.pyc").exists(), "stale __pycache__ removed")
    compile(path.read_text(), str(path), "exec")
    expect(outcome(new, path)[:2] == ("already", True), "patched copy -> already (no write)")

    # 2. The fixed app file: "upstream", no write; the sftv3 patcher (live re-run job) reports "already", no write.
    path = copy("fixed", fixed)
    result, unchanged, leftovers = outcome(new, path)
    expect(result == "upstream" and unchanged and not leftovers, f"fixed app file -> {result} (no write)")
    result, unchanged, leftovers = outcome(old, path)
    expect(result == "already" and unchanged and not leftovers,
           f"glm-rollouts-sftv3 patcher on the fixed app file -> {result} (no write; job 21541211 keeps working)")
    expect(fixed.count(new.MARK) == 1 and new.OLD not in fixed, "fixed app file: MARK once, Baihe's drain() gone")

    # 3. The old patcher's output is accepted by the new one, and vice versa.
    path = copy("by-sftv3", original)
    expect(outcome(old, path)[0] == "patched" and outcome(new, path)[:2] == ("already", True),
           "copy patched by the sftv3 patcher -> new patcher: already")

    # 4. Fail closed (raise, no write) on anything else.
    drain_call = "            while chunk := await reader():\n                await _in_stream_io(persist, chunk)\n"
    executor = '\n_STREAM_IO = ThreadPoolExecutor(max_workers=32, thread_name_prefix="atp-stream-io")\n'
    cases = {
        "partial fix (old drain loop, new helpers)": fixed.replace(
            drain_call, "            while chunk := await reader():\n                persist(chunk)\n"),
        "fix + Baihe's drain() block": fixed + "\n\ndef _unused(directory, os):\n" + new.OLD,
        "fix + sftv3 NEW block": fixed + "\n\ndef _unused(directory, os, asyncio):\n" + new.NEW,
        "fix with _STREAM_IO defined twice (adjacent lines)": fixed.replace(executor, executor + executor[1:]),
        "fix with _STREAM_IO defined twice (end of file)": fixed + executor,
        "fix with the executor line re-indented": fixed.replace(executor, executor.replace("\n_", "\n    _", 1)),
        "Baihe's drain() re-indented": original.replace(new.OLD, new.OLD.replace("    async def drain", "  async def drain")),
        "Baihe's drain() + MARK comment": original + "# " + new.MARK + "\n",
        "NEW twice": original.replace(new.OLD, new.NEW, 1) + "\n\ndef _unused(directory, os, asyncio):\n" + new.NEW,
        "empty file": "",
    }
    for index, (label, text) in enumerate(cases.items()):
        path = copy(f"bad-{index}", text)
        result, unchanged, leftovers = outcome(new, path)
        expect(result.startswith("raised") and unchanged and not leftovers, f"{label} -> {result}")

    # 5. The tasksall wrapper accepts every outcome the new patcher returns.
    wrapper = TASKSALL_WRAPPER.read_text()
    expect('outcome not in ("patched", "already", "upstream")' in wrapper,
           "tasksall probe_worker.py accepts patched/already/upstream")
    print("scratch:", root)
    print("RESULT", "FAIL" if FAILED else "PASS", f"({len(FAILED)} failed)")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
