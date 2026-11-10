"""(Done 2026-10-20 16:57/17:2x; the staging tree was removed afterwards.) Install the staged gate fixes atomically: per file, check the live file still has its baseline checksum (no
concurrent edit), keep a backup <file>.pre-gates-1020, write the new content to a temp file in the same directory,
verify it (compile / prompt fields), then os.replace(). Order: rollouts.py before task_builder.py (it imports
verifier_fault). Run once from anywhere with the pipeline venv."""
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
STAGE = HERE / "stage"
FILES = [(STAGE / "rollouts.py", S / "synth/rollouts.py"), (STAGE / "task_builder.py", S / "synth/task_builder.py"),
         (STAGE / "prompts/author.md", S / "synth/prompts/author.md"),
         (STAGE / "tests/test_gates_1020.py", S / "synth/tests/test_gates_1020.py"), (STAGE / "README.md", S / "README.md")]
FIELDS = {"title", "repo", "kind", "kind_rule", "start_rule", "sha", "base_report", "new_report", "grade_cfg",
          "feedback", "sandbox"}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    baseline = {}
    for line in (HERE / "baseline-md5.txt").read_text().splitlines():
        digest, path = line.split(None, 1)
        baseline[Path(path).resolve()] = digest
    for source, target in FILES:
        if target.exists() and baseline.get(target.resolve()) not in (None, md5(target)):
            sys.exit(f"{target} changed since the baseline: merge first")
    for source, target in FILES:
        if target.exists():
            backup = target.with_name(target.name + ".pre-gates-1020")
            if not backup.exists():
                shutil.copy2(target, backup)
        temp = target.with_name(f".{target.name}.tmp-gates-1020")
        shutil.copyfile(source, temp)
        if target.exists():
            shutil.copymode(target, temp)
        if target.suffix == ".py":
            compile(temp.read_text(), str(target), "exec")  # a syntax error stops the install here
        if target.name == "author.md":
            found = set(re.findall(r"\{\{(\w+)\}\}", temp.read_text()))
            if found != FIELDS:
                sys.exit(f"author.md fields changed: {sorted(found ^ FIELDS)}")
        if temp.read_bytes() != source.read_bytes():
            sys.exit(f"{temp}: copy differs from {source}")
        os.replace(temp, target)
        print(f"installed {target} ({md5(target)})")


if __name__ == "__main__":
    main()
