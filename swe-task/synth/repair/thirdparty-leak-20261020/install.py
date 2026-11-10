"""(Done 2026-10-20 19:57; the staging tree was removed afterwards, so this cannot run again.) Install the staged third-party leak rule (README "Third-party packages (2026-10-20)") atomically: per file, check the
live file still has its baseline checksum (no concurrent edit), keep a backup <file>.pre-thirdparty-leak, write the new
content to a temp file in the same directory, verify it (compile for .py, byte-equal to the staged copy), then
os.replace(). Order: task_builder.py, its tests, README. Run once with the pipeline venv:
  ../../../atp/runtime/controller/bin/python3 -B install.py"""
import hashlib
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
STAGE = HERE / "stage"
FILES = [(STAGE / "synth/task_builder.py", S / "synth/task_builder.py"),
         (STAGE / "synth/tests/test_gates_1020.py", S / "synth/tests/test_gates_1020.py"),
         (STAGE / "README.md", S / "README.md")]
SUFFIX = "pre-thirdparty-leak"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    baseline = {}
    for line in (HERE / "baseline-md5.txt").read_text().splitlines():
        digest, path = line.split(None, 1)
        baseline[Path(path).resolve()] = digest
    for source, target in FILES:
        if baseline.get(target.resolve()) != md5(target):
            sys.exit(f"{target} changed since the baseline: merge first")
    for source, target in FILES:
        backup = target.with_name(f"{target.name}.{SUFFIX}")
        if not backup.exists():
            shutil.copy2(target, backup)
        temp = target.with_name(f".{target.name}.tmp-thirdparty-leak")
        shutil.copyfile(source, temp)
        shutil.copymode(target, temp)
        if target.suffix == ".py":
            compile(temp.read_text(), str(target), "exec")  # a syntax error stops the install here
        if temp.read_bytes() != source.read_bytes():
            sys.exit(f"{temp}: copy differs from {source}")
        os.replace(temp, target)
        print(f"installed {target} ({md5(target)}), backup {backup.name}")


if __name__ == "__main__":
    main()
