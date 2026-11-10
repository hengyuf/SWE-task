"""Reviewer check (tracking/compat review, 2026-11-08): outcomes of the staged tasksall capture_patch.py and of the
read-only sftv3 capture_patch.py on scratch COPIES only.

Run: <atp>/runtime/controller/bin/python -I -B <this file> <private scratch dir>
Reads: Baihe's original deepswe_capture.py, the staged fixed copy, both patchers. Writes only under <scratch>.
"""
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
F = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-tasksall-20261108/atp-fix")
BAIHE = Path("/lustre/fsw/portfolios/nemotron/projects/nemotron_evals_research/users/baiheh/nano-rl/"
             "agent-teacher-pipelines/scripts/deepswe_capture.py")
FIXED = F / "app/scripts/deepswe_capture.py"
TASKSALL = F / "capture_patch.py"
SFTV3 = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-sftv3-20261107/capture_patch.py")
SCRATCH = Path(sys.argv[1]).resolve()
assert str(SCRATCH).startswith(str((F / "scratch").resolve()) + "/"), SCRATCH
failures = []


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(data):
    return hashlib.sha256(data).hexdigest()[:12]


def check(condition, message):
    print(("PASS " if condition else "FAIL ") + message, flush=True)
    if not condition:
        failures.append(message)


tasksall, sftv3 = load(TASKSALL, "review_tasksall_patch"), load(SFTV3, "review_sftv3_patch")
original, fixed = BAIHE.read_bytes(), FIXED.read_bytes()
check(sha(original) == "b1babe29bfe2" and sha(fixed) == "56f34ab96f42", "inputs have the expected sha256")
check(tasksall.OLD == sftv3.OLD and tasksall.NEW == sftv3.NEW and tasksall.MARK == sftv3.MARK,
      "tasksall OLD/NEW/MARK byte-identical to sftv3's")
counter = [0]


def place(data, label, pyc=True):
    counter[0] += 1
    directory = SCRATCH / f"{counter[0]:02d}-{label}" / "scripts"
    shutil.rmtree(directory.parent, ignore_errors=True)
    (directory / "__pycache__").mkdir(parents=True)
    path = directory / "deepswe_capture.py"
    path.write_bytes(data)
    if pyc:
        (directory / "__pycache__/deepswe_capture.cpython-313.pyc").write_bytes(b"stale")
    return path


def run(patcher, data, label, pyc=True):
    path = place(data, label, pyc)
    before = (path.read_bytes(), os.stat(path).st_ino, os.stat(path).st_mtime_ns)
    try:
        outcome = patcher.patch_file(path)
    except Exception as error:  # noqa: BLE001
        outcome = "raise:" + type(error).__name__ + ":" + str(error).split(": ", 1)[-1][:70]
    after = (path.read_bytes(), os.stat(path).st_ino, os.stat(path).st_mtime_ns)
    leftovers = sorted(p.name for p in path.parent.iterdir() if p.name not in ("deepswe_capture.py", "__pycache__"))
    pyc_left = (path.parent / "__pycache__/deepswe_capture.cpython-313.pyc").exists()
    return outcome, before, after, leftovers, pyc_left, path


# (i) Baihe's original -> patched (sftv3's NEW, byte-identical to sftv3's own result), stale pyc removed
out_t, before, after, left, pyc_left, path_t = run(tasksall, original, "orig-tasksall")
out_s, _, after_s, _, pyc_left_s, _ = run(sftv3, original, "orig-sftv3")
check(out_t == "patched" and after[0] == original.replace(tasksall.OLD.encode(), tasksall.NEW.encode(), 1)
      and not left and not pyc_left, f"(i) original -> {out_t}; text = original with OLD->NEW; pyc removed; no temp")
check(out_s == "patched" and after_s[0] == after[0], f"(i) sftv3 patcher on original -> {out_s}; same bytes as tasksall's")
compile(after[0], "patched", "exec")
# (ii) a file it already patched -> already, no write
patched = after[0]
for name, patcher in (("tasksall", tasksall), ("sftv3", sftv3)):
    out, before, after, left, pyc_left, _ = run(patcher, patched, f"already-{name}")
    check(out == "already" and before == after and not left and pyc_left,
          f"(ii) {name} on an already-patched copy -> {out}; no write (bytes, inode, mtime), pyc kept")
# (iii) the staged fixed file -> upstream (tasksall) / already (sftv3), no write
out, before, after, left, pyc_left, _ = run(tasksall, fixed, "fixed-tasksall")
check(out == "upstream" and before == after and not left and pyc_left,
      f"(iii) tasksall on the fixed file -> {out}; no write")
out, before, after, left, pyc_left, _ = run(sftv3, fixed, "fixed-sftv3")
check(out == "already" and before == after and not left and pyc_left,
      f"(iii) sftv3 on the fixed file -> {out}; no write")
# (iv) mangled files -> raise, no write
text_o, text_f = original.decode(), fixed.decode()
old_block = tasksall.OLD
mangled = {
    "empty": "",
    "orig-drain-reindented": text_o.replace("                stream.write(chunk)\n", "                stream.write(chunk)  # x\n"),
    "orig-no-asyncio-import": text_o.replace("import asyncio\n", "import trio\n", 1),
    "orig-old-twice": text_o + "\n\nasync def _x(reader, directory):\n" + old_block,
    "orig-mark-comment-only": text_o.replace(old_block, old_block.replace("stream.write(chunk)",
                                                                          "stream.write(chunk)  # glm-rollouts-sftv3 capture_patch.py")),
    "patched-new-twice": patched.decode() + "\n\nasync def _x(reader, directory):\n" + tasksall.NEW,
    "fixed-pool-16": text_f.replace("max_workers=32", "max_workers=16"),
    "fixed-helper-twice": text_f + "\n\nasync def _in_stream_io(function, *args):\n    pass\n",
    "fixed-plus-old": text_f + "\n\nasync def _x(reader, directory):\n" + old_block,
    "fixed-plus-new": text_f + "\n\nasync def _x(reader, directory):\n" + tasksall.NEW,
    "fixed-await-to_thread": text_f.replace("await _in_stream_io(persist, chunk)", "await asyncio.to_thread(persist, chunk)"),
    "fixed-no-pool": text_f.replace('_STREAM_IO = ThreadPoolExecutor(max_workers=32, thread_name_prefix="atp-stream-io")\n', ""),
}
for label, text in mangled.items():
    data = text.encode()
    out, before, after, left, pyc_left, _ = run(tasksall, data, "mangled-" + label)
    check(out.startswith("raise:") and before == after and not left and pyc_left,
          f"(iv) tasksall on {label} -> {out}; no write")
out, before, after, left, pyc_left, _ = run(sftv3, mangled["fixed-pool-16"].encode(), "sftv3-fixed-pool-16")
print(f"INFO sftv3 on fixed-pool-16 -> {out} (sftv3 keys on MARK only)")
print("FAILED: " + "; ".join(failures) if failures else "ALL PASS")
raise SystemExit(1 if failures else 0)
