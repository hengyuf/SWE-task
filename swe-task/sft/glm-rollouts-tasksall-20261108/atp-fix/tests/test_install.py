"""Exercises install.py on COPIES of the targets (never the live app): a fake ATP root (the live app's tracked files:
scripts/ upstream/ tools/ configs/ overlays/ deepswe.sbatch README.md PROVENANCE.json hengyuf-changes.diff) and a fake
tasksall root (capture_patch.py), both under the private scratch dir.
Checks: check/install/verify, idempotent re-install, tracking + patcher checks on the installed copy, refusal without
writes when a target changed, rollback to the byte-identical pre-install state, refusal of a rollback over a changed
target. Optionally (--with-capture) runs the installed copy's `check_hengyuf_changes.py capture` too.
Run: <atp>/runtime/controller/bin/python -I -B <this file> <private scratch dir> [--with-capture]
"""
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

FIX = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-tasksall-20261108/atp-fix")
A = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp")
T = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-tasksall-20261108")
PY = A / "runtime/controller/bin/python"
FAILED = []


def expect(condition, message):
    print(("PASS " if condition else "FAIL ") + message, flush=True)
    if not condition:
        FAILED.append(message)


def tree(root):
    return {str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mode & 0o777)
            for p in sorted(root.rglob("*")) if p.is_file()}


def fake_roots(base):
    atp, tasksall = base / "atp", base / "tasksall"
    for sub in ("scripts", "upstream", "tools", "configs", "overlays"):
        (atp / sub).mkdir(parents=True)
        for path in (A / sub).iterdir():
            if path.is_file():
                shutil.copy2(path, atp / sub / path.name)
    for name in ("deepswe.sbatch", "README.md", "PROVENANCE.json", "hengyuf-changes.diff"):
        shutil.copy2(A / name, atp / name)
    tasksall.mkdir()
    shutil.copy2(T / "capture_patch.py", tasksall / "capture_patch.py")
    return atp, tasksall


def run(*command, timeout=100):
    result = subprocess.run([str(c) for c in command], capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout + result.stderr


def main():
    scratch = Path(sys.argv[1]).resolve()
    base = Path(tempfile.mkdtemp(prefix="install-test-", dir=scratch))
    atp, tasksall = fake_roots(base / "one")
    roots = ["--atp-root", atp, "--tasksall-root", tasksall]
    before = tree(base / "one")
    expect(before == {**{f"atp/{k}": v for k, v in tree(atp).items()}, **{f"tasksall/{k}": v for k, v in tree(tasksall).items()}},
           f"fake roots built ({len(before)} files)")
    code, out = run(PY, "-I", "-B", FIX / "install.py", "check", *roots)
    expect(code == 0 and out.count("pre-install") == 6 and "absent" in out, "check: 6 targets pre-install, 1 absent")
    code, out = run(PY, "-I", "-B", FIX / "install.py", "install", *roots)
    expect(code == 0 and out.count("\ninstalled ") + out.startswith("installed ") == 7 and "OK install verified" in out,
           "install: 7 files placed, verified")
    leftovers = [p.name for p in base.rglob(".*atp-fix-new")]
    expect(not leftovers, f"no temporary files left {leftovers}")
    baihe = Path("/lustre/fsw/portfolios/nemotron/projects/nemotron_evals_research/users/baiheh/nano-rl/"
                 "agent-teacher-pipelines/scripts/deepswe_capture.py")
    copy = atp / "upstream/deepswe_capture.py"
    expect(copy.stat().st_mtime_ns == baihe.stat().st_mtime_ns and copy.stat().st_mode & 0o777 == 0o444
           and copy.read_bytes() == baihe.read_bytes(), "upstream/deepswe_capture.py: Baihe's bytes and mtime, 0444")
    code, out = run(PY, "-I", "-B", FIX / "install.py", "install", *roots)
    expect(code == 0 and out.count("installed ") == 7 and "\ninstalled /" not in out, "re-install: idempotent (skips all)")
    code, out = run(PY, "-I", "-B", FIX / "install.py", "verify", *roots)
    expect(code == 0, "verify after install")
    code, out = run(PY, "-I", "-B", FIX / "tests/check_tracking.py", atp, scratch)
    expect(code == 0, "check_tracking.py on the installed copy: " + out.strip().splitlines()[-1])
    code, out = run(PY, "-I", "-B", FIX / "tests/test_capture_patch.py", scratch, "--app-script",
                    atp / "scripts/deepswe_capture.py", "--patcher", tasksall / "capture_patch.py")
    expect(code == 0, "test_capture_patch.py with the installed copies: " + out.strip().splitlines()[-1])
    if "--with-capture" in sys.argv:
        code, out = run(PY, "-I", "-B", atp / "tools/check_hengyuf_changes.py", "capture", "--scratch", scratch)
        expect(code == 0, "installed tool: capture check: " + out.strip().splitlines()[-1])
    installed = tree(base / "one")
    # A target changed after the install: rollback refuses and writes nothing.
    readme = atp / "README.md"
    readme.write_text(readme.read_text() + "\nlocal edit\n")
    changed = tree(base / "one")
    code, out = run(PY, "-I", "-B", FIX / "install.py", "rollback", *roots)
    expect(code == 1 and "REFUSED" in out and tree(base / "one") == changed, "rollback over a changed target refused")
    shutil.copyfile(FIX / "app/README.md", readme)
    expect(tree(base / "one") == installed, "README restored to the installed bytes")
    code, out = run(PY, "-I", "-B", FIX / "install.py", "rollback", *roots)
    expect(code == 0 and tree(base / "one") == before, "rollback: byte-identical pre-install tree (upstream copy removed)")
    expect(all((atp / name).stat().st_mtime_ns == (A / name).stat().st_mtime_ns
               for name in ("scripts/deepswe_capture.py", "README.md", "PROVENANCE.json", "hengyuf-changes.diff",
                            "tools/check_hengyuf_changes.py")), "rollback restores the original mtimes too")
    code, out = run(PY, "-I", "-B", FIX / "install.py", "rollback", *roots)
    expect(code == 0 and tree(base / "one") == before, "second rollback: no-op")
    # A target changed before the install: install refuses and writes nothing.
    atp2, tasksall2 = fake_roots(base / "two")
    (atp2 / "PROVENANCE.json").write_text((atp2 / "PROVENANCE.json").read_text().replace("2026-10-16T19:47:02",
                                                                                          "2026-10-16T19:47:03"))
    pristine = tree(base / "two")
    code, out = run(PY, "-I", "-B", FIX / "install.py", "install", "--atp-root", atp2, "--tasksall-root", tasksall2)
    expect(code == 1 and "REFUSED" in out and "PROVENANCE.json" in out and tree(base / "two") == pristine,
           "install over a changed target refused, nothing written")
    print("scratch:", base)
    print("RESULT", "FAIL" if FAILED else "PASS", f"({len(FAILED)} failed)")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
