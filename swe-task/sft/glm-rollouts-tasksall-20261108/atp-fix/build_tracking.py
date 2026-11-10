"""Builds the tracking files of the 2026-11-08 ATP capture fix from the staged sources, writing ONLY inside atp-fix:

  app/hengyuf-changes.diff  same method as the installed one: 3 comment lines + `diff -ru a/scripts b/scripts` over plain
                            copies of Baihe's 29 scripts/*.py (a) and this app's 29 (b = the live app's, with
                            app/scripts/deepswe_capture.py); verified with `patch -p1 --fuzz=0` on a copy of Baihe's
                            scripts, which must reproduce all 29 b files byte for byte
  app/PROVENANCE.json       the live PROVENANCE.json (must round-trip byte-identically) with the capture fields updated
  install-manifest.json     staged file -> target, mode, staged sha256, the target's current (pre-install) sha256
  APPLY.md                  its install table (between the <!-- install-table --> markers), from the manifest

Inputs that must already be final: app/scripts/deepswe_capture.py, app/upstream/deepswe_capture.py,
app/tools/check_hengyuf_changes.py, app/README.md (its "+P/-M lines" must match the diff), capture_patch.py.
Run: <atp>/runtime/controller/bin/python -I -B <this file>   (rerun whenever an input or a live target changes)
"""
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

FIX = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-tasksall-20261108/atp-fix")
A = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp")
B = Path("/lustre/fsw/portfolios/nemotron/projects/nemotron_evals_research/users/baiheh/nano-rl/agent-teacher-pipelines")
T = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/glm-rollouts-tasksall-20261108")
BAIHE_CAPTURE_SHA256 = "b1babe29bfe228fbc3808bdc6bffb6daac475fe798e9dd17720744b186949fe8"
NOTE = "; deepswe_capture.py changed 2026-11-08 (OpenCode stream write + fsync off the event loop, README item 6)"
# staged (relative to FIX) -> (root, target relative to root, mode); install order
INSTALL = [
    ("app/upstream/deepswe_capture.py", "atp", "upstream/deepswe_capture.py", "0444"),
    ("app/scripts/deepswe_capture.py", "atp", "scripts/deepswe_capture.py", "0644"),
    ("app/hengyuf-changes.diff", "atp", "hengyuf-changes.diff", "0644"),
    ("app/tools/check_hengyuf_changes.py", "atp", "tools/check_hengyuf_changes.py", "0644"),
    ("app/PROVENANCE.json", "atp", "PROVENANCE.json", "0644"),
    ("app/README.md", "atp", "README.md", "0644"),
    ("capture_patch.py", "tasksall", "capture_patch.py", "0644"),
]
ROOTS = {"atp": A, "tasksall": T}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counts(diff_text):
    body = diff_text.splitlines()[3:]
    return (sum(line.startswith("+") and not line.startswith("+++") for line in body),
            sum(line.startswith("-") and not line.startswith("---") for line in body))


def build_diff(work):
    for side in ("a", "b"):
        (work / side / "scripts").mkdir(parents=True)
    for path in sorted(B.glob("scripts/*.py")):
        shutil.copyfile(path, work / "a/scripts" / path.name)
    for path in sorted(A.glob("scripts/*.py")):
        shutil.copyfile(path, work / "b/scripts" / path.name)
    shutil.copyfile(FIX / "app/scripts/deepswe_capture.py", work / "b/scripts/deepswe_capture.py")
    names = sorted(p.name for p in (work / "a/scripts").iterdir())
    assert names == sorted(p.name for p in (work / "b/scripts").iterdir()) and len(names) == 29, names
    result = subprocess.run(["diff", "-ru", "a/scripts", "b/scripts"], cwd=work, capture_output=True)
    assert result.returncode == 1 and not result.stderr, (result.returncode, result.stderr)
    header = ("# hengyuf's changes to Baihe's ATP scripts: diff -ru <Baihe original> <this app>. Regenerated "
              f"{datetime.datetime.now().astimezone().isoformat(timespec='seconds')}.\n"
              f"# a/ = {B} (pristine copies: upstream/)\n# b/ = {A}\n")
    text = header + result.stdout.decode()
    temporary = FIX / "app/.hengyuf-changes.diff.new"
    temporary.write_text(text)
    temporary.chmod(0o644)
    temporary.replace(FIX / "app/hengyuf-changes.diff")
    # Verify: Baihe's scripts + this diff == b, all 29 files.
    verify = work / "verify"
    (verify / "scripts").mkdir(parents=True)
    for path in sorted(B.glob("scripts/*.py")):
        shutil.copyfile(path, verify / "scripts" / path.name)
    result = subprocess.run(["patch", "-p1", "--batch", "--forward", "--fuzz=0", "--no-backup-if-mismatch", "-d",
                             str(verify), "-i", str(FIX / "app/hengyuf-changes.diff")], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    same = [name for name in names if (verify / "scripts" / name).read_bytes() == (work / "b/scripts" / name).read_bytes()]
    assert len(same) == 29 and not list((verify / "scripts").glob("*.orig")) and not list((verify / "scripts").glob("*.rej"))
    changed = sorted(f"scripts/{line.split()[2].split('/', 2)[2]}" for line in text.splitlines() if line.startswith("diff -ru"))
    print(f"diff: {len(changed)} scripts {changed}, +{counts(text)[0]}/-{counts(text)[1]}; patch -p1 reproduces 29/29")
    return text, changed


def build_provenance(diff_text, changed):
    original = (A / "PROVENANCE.json").read_text()
    provenance = json.loads(original)
    assert json.dumps(provenance, indent=2) + "\n" == original, "live PROVENANCE.json does not round-trip"
    record = provenance["files"]["scripts/deepswe_capture.py"]
    assert record["sha256"] == BAIHE_CAPTURE_SHA256 == sha(B / "scripts/deepswe_capture.py")
    assert record["current_sha256"] == BAIHE_CAPTURE_SHA256 and record["modified_by_hengyuf"] is False, record
    record["current_sha256"] = sha(FIX / "app/scripts/deepswe_capture.py")
    record["modified_by_hengyuf"] = True
    copies = provenance["upstream_pristine_copies"]
    assert "upstream/deepswe_capture.py" not in copies
    assert sha(FIX / "app/upstream/deepswe_capture.py") == BAIHE_CAPTURE_SHA256
    copies["upstream/deepswe_capture.py"] = {"source": str(B / "scripts/deepswe_capture.py"),
                                             "sha256": BAIHE_CAPTURE_SHA256}
    provenance["upstream_pristine_copies"] = dict(sorted(copies.items()))
    changes = provenance["hengyuf_changes"]
    assert NOTE not in changes["note"]
    changes["diff_sha256"] = hashlib.sha256(diff_text.encode()).hexdigest()
    changes["scripts"] = sorted(set(changes["scripts"]) | {"scripts/deepswe_capture.py"})
    assert changes["scripts"] == changed, (changes["scripts"], changed)
    changes["note"] += NOTE
    provenance["tools"]["tools/check_hengyuf_changes.py"]["sha256"] = sha(FIX / "app/tools/check_hengyuf_changes.py")
    provenance["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    temporary = FIX / "app/.PROVENANCE.json.new"
    temporary.write_text(json.dumps(provenance, indent=2) + "\n")
    temporary.chmod(0o644)
    temporary.replace(FIX / "app/PROVENANCE.json")
    assert sha(FIX / "app/hengyuf-changes.diff") == changes["diff_sha256"]
    print("PROVENANCE.json: capture record, upstream copy, hengyuf_changes, tools sha, updated_at", provenance["updated_at"])


def check_readme(diff_text):
    plus, minus = counts(diff_text)
    readme = (FIX / "app/README.md").read_text()
    found = re.findall(r"Code \((\d) scripts: [^;]+; \+(\d+)/-(\d+) lines", readme)
    assert found == [("4", str(plus), str(minus))], (found, plus, minus)
    assert "29 scripts; 4 changed" in readme and "five Baihe scripts" in readme and "\n6. (2026-11-08)" in readme
    print(f"README: counts match the diff (+{plus}/-{minus})")


def build_manifest():
    entries = []
    for staged, root, target, mode in INSTALL:
        staged_path, target_path = FIX / staged, ROOTS[root] / target
        assert oct(staged_path.stat().st_mode & 0o777) == "0o" + mode.lstrip("0"), (staged, oct(staged_path.stat().st_mode))
        entries.append({"staged": str(staged_path), "root": root, "target": target, "target_path": str(target_path),
                        "mode": mode, "sha256": sha(staged_path), "bytes": staged_path.stat().st_size,
                        "pre_install_sha256": sha(target_path) if target_path.exists() else "absent",
                        "rollback": (str(FIX / "rollback" / ("atp" if root == "atp" else "tasksall") / target)
                                     if target_path.exists() else None)})
        if entries[-1]["rollback"]:
            assert sha(entries[-1]["rollback"]) == entries[-1]["pre_install_sha256"], f"stale rollback copy: {target}"
    manifest = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "roots": {key: str(value) for key, value in ROOTS.items()},
                "entries": entries}
    (FIX / "install-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for entry in entries:
        print(f"{entry['sha256'][:12]} {entry['mode']} {entry['root']}:{entry['target']} (pre {entry['pre_install_sha256'][:12]})")
    return manifest


def write_apply_table(manifest):
    """APPLY.md's install table (between its markers) from the manifest."""
    rows = [f"Manifest built {manifest['created_at']}.", "",
            "| # | staged (in F) | target | mode | bytes | staged sha256 | expected pre-install sha256 | rollback copy (in F) |",
            "|---|---|---|---|---|---|---|---|"]
    for index, entry in enumerate(manifest["entries"], 1):
        rollback = f"`{Path(entry['rollback']).relative_to(FIX)}`" if entry["rollback"] else "none (remove)"
        rows.append(f"| {index} | `{Path(entry['staged']).relative_to(FIX)}` | `{entry['target_path']}` | {entry['mode']} | "
                    f"{entry['bytes']} | `{entry['sha256']}` | `{entry['pre_install_sha256']}` | {rollback} |")
    path = FIX / "APPLY.md"
    text = path.read_text()
    start, end = "<!-- install-table -->\n", "<!-- /install-table -->"
    head, rest = text.split(start, 1)
    tail = rest.split(end, 1)[1]
    path.write_text(head + start + "\n".join(rows) + "\n" + end + tail)
    print("APPLY.md: install table rewritten from the manifest")


def main():
    with tempfile.TemporaryDirectory(prefix="build-tracking-", dir=FIX / "scratch") as work:
        diff_text, changed = build_diff(Path(work))
    build_provenance(diff_text, changed)
    check_readme(diff_text)
    write_apply_table(build_manifest())


if __name__ == "__main__":
    main()
