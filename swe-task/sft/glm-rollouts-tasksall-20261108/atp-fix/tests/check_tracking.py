"""Read-only consistency check of an ATP app root's tracking files (README.md, PROVENANCE.json, hengyuf-changes.diff,
upstream/, scripts/, tools/) against Baihe's original tree. Use on a full mirror before the install and on the live app
(/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp) after it; writes only a temporary dir under <scratch>.
Run: <atp>/runtime/controller/bin/python -I -B <this file> <app root> <private scratch dir>
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

B = Path("/lustre/fsw/portfolios/nemotron/projects/nemotron_evals_research/users/baiheh/nano-rl/agent-teacher-pipelines")
SFTV3_MARK = "glm-rollouts-sftv3 capture_patch.py"
FAILED = []


def expect(condition, message):
    print(("PASS " if condition else "FAIL ") + message, flush=True)
    if not condition:
        FAILED.append(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    app, scratch = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    text = (app / "PROVENANCE.json").read_text()
    provenance = json.loads(text)
    expect(json.dumps(provenance, indent=2) + "\n" == text, "PROVENANCE.json is json.dumps(indent=2) + newline")
    files = provenance["files"]
    bad = [name for name, record in files.items()
           if sha(app / name) != record["current_sha256"]
           or record["modified_by_hengyuf"] != (record["current_sha256"] != record["sha256"])
           or sha(record["source"]) != record["sha256"]]
    expect(not bad, f"{len(files)} file records: current_sha256 matches, modified flag consistent, Baihe's source "
                    f"unchanged {bad}")
    modified = sorted(name for name, record in files.items() if record["modified_by_hengyuf"])
    changes = provenance["hengyuf_changes"]
    expect(modified == changes["scripts"], f"modified files == hengyuf_changes.scripts {modified}")
    copies = provenance["upstream_pristine_copies"]
    expect(list(copies) == sorted(copies), "upstream_pristine_copies sorted")
    bad = [name for name, record in copies.items()
           if not (sha(app / name) == record["sha256"] == sha(record["source"])
                   and (app / name).stat().st_mode & 0o777 == 0o444)]
    expect(not bad, f"{len(copies)} upstream/ copies == record == Baihe's original, mode 0444 {bad}")
    expect(sorted(p.name for p in (app / "upstream").glob("*.py")) == sorted(Path(n).name for n in copies),
           "upstream/ holds exactly the recorded copies")
    expect(all(f"upstream/{Path(name).name}" in copies for name in modified), "every changed script has an upstream copy")
    bad = [name for name, record in provenance["tools"].items() if sha(app / name) != record["sha256"]]
    expect(not bad, f"tools records match {bad}")
    marker = changes["marker"]
    expect(all(marker in (app / name).read_text() for name in modified), f"every changed script carries '{marker}'")
    # The diff: sha, header, file list, and patch -p1 on Baihe's scripts reproduces this app's 29 scripts exactly.
    diff = app / changes["diff"]
    expect(sha(diff) == changes["diff_sha256"], "hengyuf-changes.diff sha256 == PROVENANCE diff_sha256")
    diff_text = diff.read_text()
    head = diff_text.splitlines()[:3]
    expect(re.fullmatch(r"# hengyuf's changes to Baihe's ATP scripts: diff -ru <Baihe original> <this app>\. "
                        r"Regenerated \d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d\.", head[0]) is not None
           and head[1] == f"# a/ = {B} (pristine copies: upstream/)"
           and head[2] == "# b/ = /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp", "diff header (3 lines)")
    listed = sorted("scripts/" + line.split()[2].split("/", 2)[2] for line in diff_text.splitlines()
                    if line.startswith("diff -ru "))
    expect(listed == modified, f"diff covers exactly the changed scripts {listed}")
    with tempfile.TemporaryDirectory(prefix="check-tracking-", dir=scratch) as work:
        (Path(work) / "scripts").mkdir()
        for path in B.glob("scripts/*.py"):
            shutil.copyfile(path, Path(work) / "scripts" / path.name)
        result = subprocess.run(["patch", "-p1", "--batch", "--forward", "--fuzz=0", "--no-backup-if-mismatch", "-d",
                                 work, "-i", str(diff)], capture_output=True, text=True)
        names = sorted(p.name for p in (app / "scripts").glob("*.py"))
        same = [n for n in names if (Path(work) / "scripts" / n).read_bytes() == (app / "scripts" / n).read_bytes()]
        expect(result.returncode == 0 and len(names) == 29 and same == names
               and sorted(p.name for p in (Path(work) / "scripts").iterdir()) == names,
               f"patch -p1 (fuzz 0) on Baihe's scripts reproduces {len(same)}/{len(names)} app scripts")
    body = diff_text.splitlines()[3:]
    plus = sum(line.startswith("+") and not line.startswith("+++") for line in body)
    minus = sum(line.startswith("-") and not line.startswith("---") for line in body)
    readme = (app / "README.md").read_text()
    found = re.findall(r"Code \((\d) scripts: ([^;]+); \+(\d+)/-(\d+) lines", readme)
    expect(found and found[0][0] == str(len(modified)) and (found[0][2], found[0][3]) == (str(plus), str(minus))
           and all(f"`{Path(name).name}`" in found[0][1] for name in modified),
           f"README 'Code (...)' line: {len(modified)} scripts, +{plus}/-{minus}")
    expect(f"29 scripts; {len(modified)} changed" in readme, "README layout row: changed-script count")
    words = {4: "four", 5: "five", 6: "six"}
    expect(f"copies of the {words.get(len(copies), len(copies))} Baihe scripts" in readme, "README upstream/ count")
    if "scripts/deepswe_capture.py" in modified:
        capture = (app / "scripts/deepswe_capture.py").read_text()
        expect(capture.count(SFTV3_MARK) == 1,
               "deepswe_capture.py names glm-rollouts-sftv3's patcher once (that patcher then reports 'already')")
    print("RESULT", "FAIL" if FAILED else "PASS", f"({len(FAILED)} failed)")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
