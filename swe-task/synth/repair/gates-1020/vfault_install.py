"""Install main's verifier-fault seed decision (2026-10-20) atomically from vfault-stage/ (removed after the 17:1x install): per file, check the live
checksum against vfault-baseline-md5.txt (no concurrent edit), keep <file>.pre-vfault-seed, write a temp file in the
same directory, compile-check Python, os.replace()."""
import hashlib
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
STAGE = HERE / "vfault-stage"
FILES = [(STAGE / "rollouts.py", S / "synth/rollouts.py"), (STAGE / "task_builder.py", S / "synth/task_builder.py"),
         (STAGE / "tests/test_gates_1020.py", S / "synth/tests/test_gates_1020.py"), (STAGE / "README.md", S / "README.md")]


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    baseline = {}
    for line in (HERE / "vfault-baseline-md5.txt").read_text().splitlines():
        digest, path = line.split(None, 1)
        baseline[(S / path).resolve()] = digest
    for _, target in FILES:
        if baseline.get(target.resolve()) != md5(target):
            sys.exit(f"{target} changed since the baseline: merge first")
    for source, target in FILES:
        backup = target.with_name(target.name + ".pre-vfault-seed")
        if not backup.exists():
            shutil.copy2(target, backup)
        temp = target.with_name(f".{target.name}.tmp-vfault-seed")
        shutil.copyfile(source, temp)
        shutil.copymode(target, temp)
        if target.suffix == ".py":
            compile(temp.read_text(), str(target), "exec")
        if temp.read_bytes() != source.read_bytes():
            sys.exit(f"{temp}: copy differs from {source}")
        os.replace(temp, target)
        print(f"installed {target} ({md5(target)})")


if __name__ == "__main__":
    main()
