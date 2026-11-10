"""Install grader v3 into the live synth/ tree atomically (2026-10-21). Adapted from ../grader-v3-20261021/install.py
(the verified stage) because two live files changed after staging: batch 3e (official1-anko, 12:32:20) edited
tools/grading_repair.py (ee993466 -> f0137efd) and tests/test_verifier.py (f708fca3 -> bcbd6b3c). The verifier's
3-way merge of the stage onto those files (verify-grader-v3-20261021/merge-check/*.merged, 0 conflicts, 291 tests OK)
was reproduced independently (git merge-file: identical bytes) and is what files/ holds:
  files/grader.py           = ../grader-v3-20261021/synth/verifier/grader.py (the verified v3, md5 a64eea49)
  files/grading_repair.py   = stage v3 changes + batch 3e (md5 1660ad45): v2 3936c324 joins PRIOR_GRADERS, installs
                              over an earlier sanitizer keep <file>.pre-grader-v3, scratch copies repair/tasks-v3/
  files/test_verifier.py    = stage v3 tests + batch 3e tests (md5 f0e1ccf6)
Same mechanics as the stage's install.py:
1. Every live file must hold the checksum pinned in FILES (no concurrent edit) and every source the merged checksum.
2. Backups <file>.pre-grader-v3 (copy2; never overwritten: an existing backup must equal the live file).
3. Temp files are written in the target directories (mode copied), checked byte-equal and compiled; only then is each
   live file re-checked and os.replace()d (grader.py first, the test last).
4. A failure after the first replace restores the replaced files from their backups (temp + os.replace).
Run once:  <S>/atp/runtime/controller/bin/python3 -B install.py [--dry-run]"""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
SUFFIX = "pre-grader-v3"
# (source, live file, live md5 now, md5 to install)
FILES = [(HERE / "files/grader.py", S / "synth/verifier/grader.py",
          "61db32599284425830bc37f7351587f2", "a64eea4931afb007c460f0069be1dbab"),
         (HERE / "files/grading_repair.py", S / "synth/tools/grading_repair.py",
          "f0137efd7b09b8b5d5dc216b2c2ea019", "1660ad45d6ab9f007c9d6bd7e077a383"),
         (HERE / "files/test_verifier.py", S / "synth/tests/test_verifier.py",
          "bcbd6b3c34085b8042c540dde9295e79", "f0e1ccf66fa3c2fedd07007c298443b0")]
RECORD = HERE / "logs/install.json"


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def fail(message: str):
    sys.exit(f"ABORT: {message}")


def write_temp(target: Path, content: bytes, mode_from: Path, tag: str) -> Path:
    temp = target.with_name(f".{target.name}.tmp-{tag}")
    with open(temp, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    shutil.copymode(mode_from, temp)
    return temp


def main():
    dry = "--dry-run" in sys.argv
    plan = []
    for source, target, live_md5, new_md5 in FILES:
        if md5(target.read_bytes()) != live_md5:
            fail(f"{target} is not {live_md5} (changed since the package was prepared): merge again")
        content = source.read_bytes()
        if md5(content) != new_md5:
            fail(f"{source} is not the merged file ({md5(content)} != {new_md5})")
        backup = target.with_name(f"{target.name}.{SUFFIX}")
        if backup.exists() and md5(backup.read_bytes()) != live_md5:
            fail(f"{backup} exists and differs from the live file: not overwriting it")
        compile(content.decode(), str(target), "exec")  # a syntax error stops here, before anything is written
        plan.append((source, target, live_md5, new_md5, content, backup))
    if dry:
        for _, target, live_md5, new_md5, content, backup in plan:
            print(f"would install {target}: {live_md5} -> {new_md5} ({len(content)} bytes), backup {backup.name}")
        return
    for _, target, live_md5, _, _, backup in plan:  # backups
        if not backup.exists():
            shutil.copy2(target, backup)
        if md5(backup.read_bytes()) != live_md5:
            fail(f"{backup}: backup checksum differs from {live_md5}")
        print(f"backup {backup} ({live_md5})")
    temps = []
    for _, target, _, new_md5, content, _ in plan:  # temp files, all written and checked before any replace
        temp = write_temp(target, content, target, SUFFIX)
        data = temp.read_bytes()
        if data != content or md5(data) != new_md5:
            fail(f"{temp}: written content differs from the source")
        compile(data.decode(), str(target), "exec")
        temps.append(temp)
        print(f"temp {temp} ok (compiled, {new_md5})")
    replaced = []
    try:
        for (_, target, live_md5, new_md5, _, _), temp in zip(plan, temps):
            if md5(target.read_bytes()) != live_md5:
                raise RuntimeError(f"{target} changed during the install (not {live_md5})")
            os.replace(temp, target)
            replaced.append(target)
            if md5(target.read_bytes()) != new_md5:
                raise RuntimeError(f"{target}: installed checksum is not {new_md5}")
            print(f"installed {target} ({new_md5})")
    except Exception as error:
        print(f"ERROR: {error}; rolling back {len(replaced)} file(s)")
        for _, target, live_md5, _, _, backup in plan:
            if target in replaced:
                restore = write_temp(target, backup.read_bytes(), backup, f"{SUFFIX}-rollback")
                os.replace(restore, target)
                print(f"restored {target} ({md5(target.read_bytes())}, expected {live_md5})")
        for temp in temps:
            if temp.exists():
                temp.unlink()
        sys.exit(1)
    record = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tool": str(Path(__file__)),
              "files": [{"path": str(target), "before_md5": live_md5, "after_md5": md5(target.read_bytes()),
                         "backup": str(backup)} for _, target, live_md5, _, _, backup in plan],
              "grader_sha256": hashlib.sha256((S / "synth/verifier/grader.py").read_bytes()).hexdigest()}
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(record, indent=2) + "\n")
    print("done: " + ", ".join(f"{target.name} {md5(target.read_bytes())}" for target in replaced))


if __name__ == "__main__":
    main()
