"""Installs (or rolls back) the 2026-11-08 ATP capture fix exactly as listed in install-manifest.json.

  check     staged files have their manifest sha256 and mode; every target is in its pre-install state (or already
            installed); rollback copies are intact. No writes.
  install   check, then per entry in manifest order: copy the staged file to a temporary name next to the target
            (".<name>.atp-fix-new", which cannot match scripts/*deepswe*.py), set its mode and the staged mtime (as
            cp -p), fsync, verify its sha256, rename it over the target, fsync the directory. Entries already installed
            are skipped; any other target state refuses before anything is written. Ends with verify.
  verify    every target has the staged sha256 and mode.
  rollback  in reverse order: an installed target gets its pre-install bytes back (from rollback/, same temp + rename;
            mode 0644) or, if it did not exist before, is removed; a target already back in its pre-install state is
            skipped; any other state refuses before anything is written.
Roots default to the manifest's; --atp-root/--tasksall-root redirect them (tests on a copy only).
Run: python3 -I -B install.py {check,install,verify,rollback} [--atp-root DIR] [--tasksall-root DIR]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

FIX = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state(entry, target):
    if not target.exists():
        return "absent"
    digest = sha(target)
    if digest == entry["sha256"]:
        return "installed"
    if digest == entry["pre_install_sha256"]:
        return "pre-install"
    return "changed:" + digest[:12]


def place(source, target, mode):
    temporary = target.with_name("." + target.name + ".atp-fix-new")
    if temporary.exists():
        temporary.chmod(0o600)
        temporary.unlink()
    shutil.copyfile(source, temporary)
    temporary.chmod(int(mode, 8))
    status = os.stat(source)  # keep the staged mtime, as cp -p (upstream/: Baihe's; rollback/: the original's)
    os.utime(temporary, ns=(status.st_atime_ns, status.st_mtime_ns))
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    if sha(temporary) != sha(source):
        temporary.unlink()
        raise SystemExit(f"copy of {source} is corrupt; nothing replaced at {target}")
    os.replace(temporary, target)
    descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("check", "install", "verify", "rollback"))
    parser.add_argument("--atp-root", type=Path)
    parser.add_argument("--tasksall-root", type=Path)
    args = parser.parse_args()
    manifest = json.loads((FIX / "install-manifest.json").read_text())
    roots = {"atp": args.atp_root or Path(manifest["roots"]["atp"]),
             "tasksall": args.tasksall_root or Path(manifest["roots"]["tasksall"])}
    entries = [(entry, roots[entry["root"]] / entry["target"]) for entry in manifest["entries"]]
    problems, states = [], []
    for entry, target in entries:
        staged = Path(entry["staged"])
        if sha(staged) != entry["sha256"] or oct(staged.stat().st_mode & 0o777)[2:] != entry["mode"].lstrip("0"):
            problems.append(f"staged {staged} differs from the manifest (rerun build_tracking.py and the tests)")
        if entry["rollback"] and sha(entry["rollback"]) != entry["pre_install_sha256"]:
            problems.append(f"rollback copy {entry['rollback']} differs from the pre-install sha256")
        current = state(entry, target)
        expected_pre = "absent" if entry["pre_install_sha256"] == "absent" else "pre-install"
        states.append(current)
        print(f"{current:20} {entry['root']}:{entry['target']}  (staged {entry['sha256'][:12]}, "
              f"pre {entry['pre_install_sha256'][:12]})")
        if args.action in ("check", "install") and current not in (expected_pre, "installed"):
            problems.append(f"{target}: {current}, expected {expected_pre} (or installed)")
        if args.action == "verify" and (current != "installed"
                                        or oct(target.stat().st_mode & 0o777)[2:] != entry["mode"].lstrip("0")):
            problems.append(f"{target}: {current}, mode {oct(target.stat().st_mode & 0o777) if target.exists() else '-'}")
        if args.action == "rollback" and current not in ("installed", expected_pre):
            problems.append(f"{target}: {current}; neither the installed nor the pre-install state, not touched")
    if problems:
        print("REFUSED" if args.action in ("install", "rollback") else "FAIL", *problems, sep="\n  ")
        return 1
    if args.action == "install":
        for (entry, target), current in zip(entries, states):
            if current != "installed":
                place(Path(entry["staged"]), target, entry["mode"])
                print(f"installed {target}")
        return main_verify(entries)
    if args.action == "rollback":
        for (entry, target), current in reversed(list(zip(entries, states))):
            if current != "installed":
                continue
            if entry["pre_install_sha256"] == "absent":
                target.chmod(0o600)
                target.unlink()
                print(f"removed {target}")
            else:
                place(Path(entry["rollback"]), target, "0644")
                print(f"restored {target}")
        for entry, target in entries:
            expected_pre = "absent" if entry["pre_install_sha256"] == "absent" else "pre-install"
            if state(entry, target) != expected_pre:
                print(f"FAIL {target} is not back in its pre-install state")
                return 1
        print("rolled back: every target in its pre-install state")
        return 0
    print("OK", args.action)
    return 0


def main_verify(entries):
    bad = [str(target) for entry, target in entries
           if not target.exists() or sha(target) != entry["sha256"]
           or oct(target.stat().st_mode & 0o777)[2:] != entry["mode"].lstrip("0")]
    print("FAIL verify: " + ", ".join(bad) if bad else "OK install verified: every target has the staged sha256 and mode")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
