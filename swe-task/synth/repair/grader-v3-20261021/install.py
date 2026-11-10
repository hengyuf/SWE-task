"""Install grader v3 (staged here in synth/; NOT run by the stager: install only after the verifier confirms these exact
checksums) into the live synth/ tree atomically. Three files change; everything else in synth/ is byte-identical to the
live tree (stage-files.txt, baseline-md5.txt):
  verifier/grader.py       the sanitizer's v3 consistency rules (Cargo target tables of deleted test files; JS/TS
                           kept-additive files importing deleted/restored test paths). New builds copy it
                           (task_builder.write_tests); live builds keep the grader they imported.
  tests/test_verifier.py   offline tests for both rules (+ the repair tool's v3 behaviour)
  tools/grading_repair.py  v2 (3936c324) joins PRIOR_GRADERS, installs over a sanitizer keep <file>.pre-grader-v3,
                           scratch copies in repair/tasks-v3/, the new log actions in sanitize_marks
1. Every live file must still hold the checksum it had when v3 was staged (baseline-md5.txt: no concurrent edit) and
   every staged file the checksum the verifier confirmed (FILES below).
2. Backups <file>.pre-grader-v3 (copy2; never overwritten: an existing backup must equal the live file).
3. All temp files are written in their target directories (mode copied from the live file), checked byte-equal to the
   stage and compiled; only then is each live file re-checked and os.replace()d (grader.py first, the test last).
4. A failure after the first replace restores the replaced files from their backups (same temp + os.replace).
The accepted tasks' tests/grader.py are NOT touched here: after this install, `tools/grading_repair.py stage /
validate --tamper / install` per run puts v3 into runs/<R>/accepted/<T>/tests/ (backups <file>.pre-grader-v3).
Run once:  <S>/atp/runtime/controller/bin/python3 -B install.py [--dry-run]"""
import hashlib
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
STAGE = HERE / "synth"
SUFFIX = "pre-grader-v3"
# (staged file, live file, live md5 when v3 was staged, staged md5 to confirm)
FILES = [(STAGE / "verifier/grader.py", S / "synth/verifier/grader.py",
          "61db32599284425830bc37f7351587f2", "a64eea4931afb007c460f0069be1dbab"),
         (STAGE / "tools/grading_repair.py", S / "synth/tools/grading_repair.py",
          "ee993466bfe59eebc2e90e0276d1d45f", "ae06329ce10f6329eb80bdde9eff97ff"),
         (STAGE / "tests/test_verifier.py", S / "synth/tests/test_verifier.py",
          "f708fca3ae056beff7346a59a9b9df6a", "779952584960d341c3f35c0b308112e3")]


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
    baseline = {}
    for line in (HERE / "baseline-md5.txt").read_text().splitlines():
        digest, path = line.split(None, 1)
        baseline[(S / "synth" / path).resolve()] = digest
    plan = []
    for source, target, live_md5, stage_md5 in FILES:
        if baseline.get(target.resolve()) != live_md5:
            fail(f"{target}: baseline-md5.txt does not record {live_md5}")
        if md5(target.read_bytes()) != live_md5:
            fail(f"{target} changed since v3 was staged (not {live_md5}): merge first")
        content = source.read_bytes()
        if md5(content) != stage_md5:
            fail(f"{source} is not the verified stage ({md5(content)} != {stage_md5})")
        backup = target.with_name(f"{target.name}.{SUFFIX}")
        if backup.exists() and md5(backup.read_bytes()) != live_md5:
            fail(f"{backup} exists and differs from the live file: not overwriting it")
        compile(content.decode(), str(target), "exec")  # a syntax error stops here, before anything is written
        plan.append((source, target, live_md5, stage_md5, content, backup))
    if dry:
        for _, target, live_md5, stage_md5, content, backup in plan:
            print(f"would install {target}: {live_md5} -> {stage_md5} ({len(content)} bytes), backup {backup.name}")
        return
    for _, target, live_md5, _, _, backup in plan:  # backups
        if not backup.exists():
            shutil.copy2(target, backup)
        if md5(backup.read_bytes()) != live_md5:
            fail(f"{backup}: backup checksum differs from {live_md5}")
        print(f"backup {backup} ({live_md5})")
    temps = []
    for _, target, _, stage_md5, content, _ in plan:  # temp files, all written and checked before any replace
        temp = write_temp(target, content, target, SUFFIX)
        data = temp.read_bytes()
        if data != content or md5(data) != stage_md5:
            fail(f"{temp}: written content differs from the stage")
        compile(data.decode(), str(target), "exec")
        temps.append(temp)
        print(f"temp {temp} ok (compiled, {stage_md5})")
    replaced = []
    try:
        for (_, target, live_md5, stage_md5, _, _), temp in zip(plan, temps):
            if md5(target.read_bytes()) != live_md5:
                raise RuntimeError(f"{target} changed during the install (not {live_md5})")
            os.replace(temp, target)
            replaced.append(target)
            if md5(target.read_bytes()) != stage_md5:
                raise RuntimeError(f"{target}: installed checksum is not {stage_md5}")
            print(f"installed {target} ({stage_md5})")
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
    print("done: " + ", ".join(f"{target.name} {md5(target.read_bytes())}" for target in replaced))


if __name__ == "__main__":
    main()
