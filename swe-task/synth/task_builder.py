"""Task builder: synthesize DeepSWE-like tasks in a similar repository that reproduce the student's failure
pattern on a target DeepSWE task.

  image A (build_env) -> understanding.md -> mask_plan.md/candidates.json -> per candidate:
  author (files only) -> materialize C (controller) -> gates (real grader) -> instruction.md + rubric.md ->
  Harbor package -> probe (student x5, teacher x5) -> score (cheat audit, pass rates, failure similarity)
  -> accept; evolve (High-IID branches from A) for up to R rounds. Seeds (hengyuf, 2026-10-17): every VALID
  round-0 task, accepted or not; then each lineage's best valid non-accepted child (see seed_check).
"""
from common import (ATP_ENV, CHEAT_CATEGORIES, DEEPSWE, GYM, IMPORTANT, RUNS, STUDENT, SYNTH, TEACHER, TOP100, fill,
                    log, read_json, render, scratch, task_toml, write_json)

import argparse
import asyncio
import base64
import functools
import hashlib
import importlib.util
import json
import keyword
from pathlib import Path
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.parse

from failure_analyzer import analyze, analyze_target, judge_pass, similarity
from llm_engine import AgentError, agent
from rollouts import dataset_row, probe, verifier_fault, write_rollouts
from sandbox import grade, grader_module, owned, parse_report, write_wrappers

RUN_START, RUN_END = "# >>> RUN TESTS (task-specific) <<<", "# >>> END RUN TESTS <<<"
FIXED_DATE = "2026-01-01T00:00:00+00:00"
HINT = re.compile(r"\b(TODO|TBD|FIXME|XXX)\b|unimplemented|not implemented|NotImplemented", re.I)
BANNED = re.compile(r"(?i:hidden test|test\.patch|solution\.patch|\bgrader\b)|\bTODO\b")
SKIP_FLAGS = {"builds_on_target", "same_owner_as_target"}
# hengyuf's swe-rl Gym copy: same commit (17abf2dd) and DeepSWE task loader as the swe-rl copy Baihe validated against.
SWE_RL_GYM = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/source/RL/3rdparty/Gym-workspace/Gym")
AGENTS = asyncio.Semaphore(5)  # the PDF's concurrency=5 for agent work
# hengyuf 2026-10-20: new builds grade with DeepSWE's grader plus the test-path sanitizer (S/README.md "Grading gap"),
# configured per repository language like the repaired tasks (verifier/test_paths.py; write_tests)
SANITIZER = SYNTH / "verifier/grader.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TEST_PATHS = _load(SYNTH / "verifier/test_paths.py", "synth_verifier_test_paths")

CLONE = """set -euo pipefail
rm -rf /app && mkdir -p /app && cd /app
git clone -q {url} .
DEFAULT="$(git symbolic-ref --short HEAD)"; SHA="$(git rev-parse HEAD)"
git checkout -q -B "$DEFAULT" "$SHA" && git remote remove origin
for b in $(git for-each-ref --format='%(refname:short)' refs/heads | grep -vx "$DEFAULT"); do git branch -D "$b" >/dev/null || true; done
for t in $(git tag); do git merge-base --is-ancestor "$t" HEAD 2>/dev/null || git tag -d "$t" >/dev/null; done
git reflog expire --expire=now --all && git gc -q --prune=now
(git submodule update --init --recursive || true)
git config core.hooksPath /dev/null
echo "SHA=$SHA"; echo "BRANCH=$DEFAULT"
"""
# hengyuf 2026-10-20 (official1-sqlformatter round 0): yarn's cache on C held the PUBLISHED @sqltools/formatter 1.2.3,
# i.e. every distinctive line of c04's solution, most of c02's and some of c01's, below the leak scan's thresholds
# (c04's whole solution has one distinctive line). Package-manager CACHES are read only by installs, so MATERIALIZE
# now deletes the cached releases of the repository's OWN packages, as it deletes the Go module's own @v versions: the
# "name" of every public tracked package.json, the Python project names, the publishable Cargo crates (outside
# node_modules/vendor/test/example/fixture/docs trees). Covered: yarn v1 cache entries, yarn berry archives (except
# those a .pnp.cjs/.pnp.data.json under /app names: under Plug'n'Play they are the install), bun's cache, npm's _cacache tarballs, pip's wheel and HTTP caches and uv's
# archives (python3), cargo registry sources and .crate files except versions a Cargo.lock takes from the registry
# (offline builds need those). INSTALLED copies stay (node_modules incl. pnpm's .pnpm, site-packages, global
# installs): dependents load them, and the test runner itself may depend on the repo's package; the leak scan judges
# those (released_copies, also per package). Best effort: never fails MATERIALIZE; PURGE_DRY_RUN=1 lists instead of
# deleting. Self-contained, so the task's environment/materialize.sh recipe does the same.
PURGE_RELEASES = r"""( set +eu +o pipefail
  P=/tmp/purge-own; rm -rf "$P"; mkdir -p "$P"; cd /app || exit 0
  say() { grep -qxF -- "$1" "$P/said" 2>/dev/null && return 0; printf '%s\n' "$1" >> "$P/said"
          if [ -n "${PURGE_DRY_RUN:-}" ]; then echo "materialize: would purge released copy $1" >&2;
          else rm -rf -- "$1" && echo "materialize: purged released copy $1" >&2; fi; }
  skip='(^|/)(node_modules|vendor|third_party|test|tests|__tests__|testdata|fixtures?|__fixtures__|examples?|e2e|docs?|website|benchmarks?)/'
  git ls-files -- package.json '*/package.json' 2>/dev/null | grep -Ev "$skip" > "$P/pkgjson"
  if [ ! -s "$P/pkgjson" ]; then :
  elif command -v node >/dev/null 2>&1; then
    node -e 'const fs = require("fs");
for (const p of fs.readFileSync(process.argv[1], "utf8").split("\n").filter(Boolean)) {
  try { const j = JSON.parse(fs.readFileSync(p, "utf8")); if (typeof j.name === "string" && j.private !== true) console.log(j.name); } catch (e) {}
}' "$P/pkgjson"
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c 'import json, sys
for p in filter(None, open(sys.argv[1]).read().split("\n")):
    try:
        j = json.load(open(p))
    except Exception:
        continue
    if isinstance(j, dict) and isinstance(j.get("name"), str) and j.get("private") is not True:
        print(j["name"])' "$P/pkgjson"
  else
    while IFS= read -r p; do
      grep -q '"private"[[:space:]]*:[[:space:]]*true' "$p" || sed -n 's/^[[:space:]]\{1,4\}"name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$p" | head -1
    done < "$P/pkgjson"
  fi | grep -E '^(@[a-z0-9][a-z0-9._~-]*/)?[a-z0-9][a-z0-9._~-]*$' | sort -u > "$P/npm"
  git ls-files -- pyproject.toml '*/pyproject.toml' setup.cfg '*/setup.cfg' setup.py '*/setup.py' 2>/dev/null \
    | grep -Ev "$skip" | while IFS= read -r f; do
    case "$f" in
      *.toml) awk '/^[[:space:]]*\[/{s=$0; gsub(/[[:space:]]/, "", s)} (s=="[project]"||s=="[tool.poetry]") && /^[[:space:]]*name[[:space:]]*=/{print; exit}' "$f" ;;
      *.cfg) awk '/^[[:space:]]*\[/{s=$0; gsub(/[[:space:]]/, "", s)} s=="[metadata]" && /^[[:space:]]*name[[:space:]]*=/{print; exit}' "$f" ;;
      *) grep -oE "(^|[[:space:](,])name[[:space:]]*=[[:space:]]*['\"][A-Za-z0-9._-]+['\"]" "$f" | head -1 ;;
    esac
  done | sed -E "s/^[^=]*=[[:space:]]*['\"]?([A-Za-z0-9][A-Za-z0-9._-]*).*/\1/" | grep -E '^[A-Za-z0-9][A-Za-z0-9._-]*$' \
    | sort -u > "$P/py"
  git ls-files -- Cargo.toml '*/Cargo.toml' 2>/dev/null | grep -Ev "$skip" | while IFS= read -r f; do
    awk '/^[[:space:]]*\[/{s=$0; gsub(/[[:space:]]/, "", s)}
         s=="[package]" && /^[[:space:]]*name[[:space:]]*=/ {n=$0; sub(/^[^=]*=[[:space:]]*"/, "", n); sub(/".*/, "", n)}
         s=="[package]" && /^[[:space:]]*publish[[:space:]]*=[[:space:]]*false/ {np=1}
         END {if (n != "" && !np) print n}' "$f"
  done | grep -E '^[A-Za-z0-9_-]+$' | sort -u > "$P/crates"
  git ls-files -- Cargo.lock '*/Cargo.lock' 2>/dev/null | while IFS= read -r f; do
    awk '/^\[\[package\]\]/{n=""; v=""} /^name = /{n=$3} /^version = /{v=$3}
         /^source = "(registry|sparse)\+/{gsub(/"/, "", n); gsub(/"/, "", v); print n " " v}' "$f"
  done | sort -u > "$P/locked"
  echo "materialize: own packages: npm $(wc -l < "$P/npm"), python $(wc -l < "$P/py"), crates $(wc -l < "$P/crates")" >&2
  find /app \( -name node_modules -o -name .git \) -prune -o \( -name .pnp.cjs -o -name .pnp.js -o -name .pnp.data.json \) \
    -print 2>/dev/null > "$P/pnp"
  release='^[0-9]+\.[0-9]+\.[0-9]+'
  while IFS= read -r n; do
    flat=$(printf '%s' "$n" | tr / -)
    for c in /usr/local/share/.cache/yarn /root/.cache/yarn "${HOME:-/root}/.cache/yarn" "${YARN_CACHE_FOLDER:-}"; do
      [ -n "$c" ] && [ -d "$c" ] || continue
      for d in "$c"/v*/npm-"$flat"-[0-9]*; do  # yarn v1: npm-<name>-<version>[-<hash>]-integrity
        [ -d "$d" ] && printf '%s\n' "${d##*/npm-"$flat"-}" | grep -qE "$release" && say "$d"
      done
    done
    for c in /root/.yarn/berry/cache "${HOME:-/root}/.yarn/berry/cache" /app/.yarn/cache; do
      for z in "$c"/"$flat"-npm-[0-9]*.zip; do  # yarn berry: <name>-npm-<version>-<checksum>-<n>.zip
        [ -f "$z" ] || continue
        case "$z" in /app/*) git ls-files --error-unmatch -- "${z#/app/}" >/dev/null 2>&1 && continue ;; esac
        # under Plug'n'Play an archive IS the install: keep it if any .pnp file names it
        while IFS= read -r f; do cat "$f"; done < "$P/pnp" 2>/dev/null | grep -qF -- "${z##*/}" && continue
        say "$z"
      done
    done
    for c in /root/.bun/install/cache "${HOME:-/root}/.bun/install/cache" "${BUN_INSTALL_CACHE_DIR:-}"; do
      [ -n "$c" ] && [ -d "$c" ] || continue
      for d in "$c"/"$n"@[0-9]*; do [ -e "$d" ] && say "$d"; done
    done
    key=$(printf '%s' "$n" | sed 's/[][\.*^$+?(){}|]/\\&/g'); key2=$(printf '%s' "$key" | sed 's#/#%2[fF]#')
    for c in /root/.npm/_cacache "${HOME:-/root}/.npm/_cacache" "${npm_config_cache:+$npm_config_cache/_cacache}"; do
      [ -n "$c" ] && [ -d "$c/index-v5" ] || continue
      grep -rhoE '"key":"[^"]*/('"$key"'|'"$key2"')/-/[^"/]*\.tgz","integrity":"[^"]*"' "$c/index-v5" 2>/dev/null \
        | sed -E 's/.*"integrity":"([^"]*)"$/\1/' | tr ' ' '\n' | sort -u | while IFS= read -r sri; do
        algo=${sri%%-*}; hex=$(printf '%s' "${sri#*-}" | sed 's/?.*//' | base64 -d 2>/dev/null | od -An -v -tx1 | tr -d ' \n')
        f="$c/content-v2/$algo/${hex:0:2}/${hex:2:2}/${hex:4}"
        [ ${#hex} -gt 8 ] && [ -f "$f" ] && say "$f"
      done
    done
  done < "$P/npm"
  if [ -s "$P/py" ] && command -v python3 >/dev/null 2>&1; then
    python3 - "$P/py" <<'PY' | while IFS= read -r f; do say "$f"; done
import os, re, sys, tarfile, zipfile
norm = lambda name: re.sub(r"[-_.]+", "_", name.lower())
own = {norm(line.strip()) for line in open(sys.argv[1]) if line.strip()}
home = os.environ.get("HOME") or "/root"
def dists(names):
    for name in names:
        top = name.strip("/").split("/")[0]
        match = re.match(r"(.+?)-\d[^/]*?\.(dist-info|egg-info)$", top) or re.match(r"(.+?)-\d[\w.!+]*$", top)
        if match and norm(match.group(1)) in own:
            return True
    return False
roots = {"/root/.cache/pip", home + "/.cache/pip", os.environ.get("PIP_CACHE_DIR") or ""}
for root in sorted(r for r in roots if r and os.path.isdir(r)):
    for folder, _, files in os.walk(root):
        for name in files:
            path = os.path.join(folder, name)
            if name.endswith(".whl"):
                if norm(name.split("-")[0]) in own:
                    print(path)
                continue
            if "http" not in folder or (folder.find("http-v2") >= 0 and not name.endswith(".body")):
                continue
            try:
                with zipfile.ZipFile(path) as archive:
                    found = dists(archive.namelist())
            except Exception:
                try:
                    with tarfile.open(path) as archive:
                        found = dists(archive.getnames())
                except Exception:
                    found = False
            if found:
                print(path)
if os.environ.get("UV_LINK_MODE", "").lower() != "symlink":  # uv hardlinks or copies from archive-v0 into venvs
    for root in sorted({"/root/.cache/uv", home + "/.cache/uv", os.environ.get("UV_CACHE_DIR") or ""}):
        archive = os.path.join(root, "archive-v0") if root else ""
        if archive and os.path.isdir(archive):
            for entry in sorted(os.listdir(archive)):
                path = os.path.join(archive, entry)
                if os.path.isdir(path) and not os.path.islink(path) and dists(os.listdir(path)):
                    print(path)
PY
  fi
  while IFS= read -r k; do
    for h in "${CARGO_HOME:-}" /root/.cargo /usr/local/cargo "${HOME:-/root}/.cargo"; do
      [ -n "$h" ] && [ -d "$h/registry" ] || continue
      for d in "$h"/registry/src/*/"$k"-[0-9]* "$h"/registry/cache/*/"$k"-[0-9]*.crate; do
        [ -e "$d" ] || continue
        v=${d##*/"$k"-}; v=${v%.crate}
        printf '%s\n' "$v" | grep -qE "$release" || continue
        grep -qxF "$k $v" "$P/locked" && continue  # a dependency of this workspace: offline builds need it
        say "$d"
      done
    done
  done < "$P/crates"
  rm -rf "$P"
) || true"""
# A -> C (run as `bash materialize.sh <start.patch>` in /app): apply the start patch, then keep exactly the tracked
# files as one reproducible commit and purge caches that could hold compiled or cached copies of removed code.
MATERIALIZE = f"""set -euo pipefail
cd /app
git update-index -q --refresh || true  # a restored snapshot changes file stat data; the cached index looks stale
if [ -s "$1" ]; then git apply --index --whitespace=nowarn "$1"; fi
branch="$(git symbolic-ref --short HEAD)"
git ls-files -z > /tmp/tracked
cp .git/info/exclude /tmp/exclude 2>/dev/null || : > /tmp/exclude
rm -rf .git && git init -q -b "$branch" && cp /tmp/exclude .git/info/exclude
git config core.hooksPath /dev/null
xargs -0 -a /tmp/tracked git add -f --
export GIT_AUTHOR_NAME="DeepSWE synth" GIT_AUTHOR_EMAIL="synth@nemo-gym.local" GIT_AUTHOR_DATE="{FIXED_DATE}"
export GIT_COMMITTER_NAME="DeepSWE synth" GIT_COMMITTER_EMAIL="synth@nemo-gym.local" GIT_COMMITTER_DATE="{FIXED_DATE}"
git commit -q -m "Initial commit"
rm -f /tmp/tracked /tmp/exclude
find /app \\( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache -o -name .gocache \\) -prune -exec rm -rf {{}} +
git ls-files -z -o -i --exclude-standard --directory \\
  | {{ grep -zE '(^|/)(dist|build|out|lib|bin|coverage|htmlcov|\\.turbo|\\.next|\\.svelte-kit|\\.parcel-cache)/$|\\.tsbuildinfo$' || true; }} \\
  | xargs -0 -r rm -rf --  # hengyuf: grep finds nothing in repos without such build dirs; under pipefail that was a silent exit 1
# hengyuf 2026-10-19: ignored EXECUTABLES built from the unmasked code (official1-ytt: setup.sh runs
# `go build -o /app/yq`) would be an oracle for the removed behaviour in C; the leak scan skips binaries.
git ls-files -z -o -i --exclude-standard --directory | while IFS= read -r -d '' f; do
  case "/$f" in */|*/node_modules/*|*/.venv/*|*/venv/*|*/.tox/*|*/site-packages/*|*.so|*.so.*) continue ;; esac
  if [ -f "$f" ] && [ "$(head -c 4 -- "$f" | od -An -tx1 | tr -d ' \\n')" = 7f454c46 ]; then
    rm -f -- "$f"; echo "materialize: removed ignored binary $f" >&2
  fi
done || true
find /app -path '*/node_modules/.cache' -prune -exec rm -rf {{}} +
rm -rf /app/target /root/.cache/go-build /tmp/jest_* /tmp/vitest*
if [ -f go.mod ] && command -v go >/dev/null; then
  module="$(sed -n 's/^module[[:space:]]*//p' go.mod | head -1)"; module="${{module%/v[0-9]*}}"
  cache="$(go env GOMODCACHE 2>/dev/null)"
  # only the module's own versions (@v dirs): nested modules published under its path (official1-opa, 2026-10-20:
  # github.com/cerbos/cerbos/api/genpb, which go.mod requires) live below cache/download/$module and must stay
  if [ -n "$module" ] && [ -n "$cache" ]; then rm -rf "$cache/cache/download/$module/@v" "$cache/cache/download/$module"/v[0-9]*/@v "$cache/$module"@* "$cache/$module"/v[0-9]*@*; fi
fi
{PURGE_RELEASES}
if command -v go >/dev/null; then go clean -cache >/dev/null 2>&1 || true; fi
test -z "$(git status --porcelain)"
git rev-parse HEAD
"""
DOCKERFILE = """# Recipe for this synthetic task's image; the task runs from the equivalent OpenSandbox snapshot in task.toml.
FROM {{base_image}}
WORKDIR /
RUN rm -rf /app && mkdir -p /app
WORKDIR /app
RUN git clone -q {{repo_url}} . \\
 && DEFAULT="$(git symbolic-ref --short HEAD)" \\
 && git checkout -q -B "$DEFAULT" {{sha}} \\
 && git remote remove origin \\
 && for b in $(git for-each-ref --format='%(refname:short)' refs/heads | grep -vx "$DEFAULT"); do git branch -D "$b" || true; done \\
 && for t in $(git tag); do git merge-base --is-ancestor "$t" HEAD 2>/dev/null || git tag -d "$t"; done \\
 && git reflog expire --expire=now --all && git gc -q --prune=now \\
 && (git submodule update --init --recursive || true) \\
 && git config core.hooksPath /dev/null
COPY setup.sh materialize.sh /tmp/
RUN bash /tmp/setup.sh && rm -f /tmp/setup.sh
# start.patch is bind-mounted so the removed implementation never lands in an image layer.
RUN --mount=type=bind,source=start.patch,target=/mnt/start.patch bash /tmp/materialize.sh /mnt/start.patch \\
 && rm -f /tmp/materialize.sh
CMD ["/bin/bash"]
"""
TASK_TOML = """schema_version = "1.3"
artifacts = ["/logs/artifacts/model.patch"]

[task]
name = "synth/{{task_id}}"
description = ""
authors = []
keywords = []

[metadata]
ext_id = "{{ext_id}}"
task_id = "{{task_id}}"
display_title = {{title}}
display_description = {{summary}}
original_title = {{title}}
category = "feature_request"
language = "{{language}}"
repository_url = "{{repo_url}}"
base_commit_hash = "{{base_commit}}"
upstream_commit_hash = "{{upstream}}"
synth_target_task = "{{target}}"
synth_kind = "{{kind}}"

[verifier]
network_mode = "no-network"
environment_mode = "separate"
timeout_sec = 1800.0

[verifier.env]

[verifier.environment]
build_timeout_sec = 1800.0
cpus = 2
memory_mb = 8192
storage_mb = 20480

[[verifier.collect]]
command = {{collect}}
timeout_sec = 300.0

[agent]
network_mode = "no-network"
timeout_sec = 5400.0

[environment]
build_timeout_sec = 1800.0
docker_image = "{{image}}"
os = "linux"
cpus = 2
memory_mb = 8192
storage_mb = 20480
gpus = 0
mcp_servers = []

[environment.env]

[solution.env]
"""
LEAK_SCAN = """copies=$(find / -xdev \\( -path /proc -o -path /sys -o -path /dev \\) -prune -o -type d -iname '*'{name}'*' -print \\
  2>/dev/null | grep -v '^/app' | head -100)
grep -rIonF --exclude-dir=.git -f /tmp/leak-tokens /app /tmp /opt /home /srv /var/tmp $copies 2>/dev/null \\
  | grep -v '^/tmp/leak-tokens:' | sort -u | head -20000 || true"""


class GateError(RuntimeError):
    pass


class Build:
    """Paths and facts for one (target DeepSWE task, similar repository) build."""

    def __init__(self, run: str, target: str, repo: str):
        self.run, self.target, self.repo = run, target, repo
        self.root = RUNS / run / "build" / target / repo.replace("/", "__")
        self.target_dir = DEEPSWE / target
        config = task_toml(self.target_dir)
        self.image = config["environment"]["docker_image"]
        self.language = config["metadata"]["language"]
        self.grade_cfg = json.loads((self.target_dir / "tests/config.json").read_text())["grade"]
        suffix = ".xml" if self.grade_cfg["format"] == "junit" else "-ctrf.json"
        self.reports = [f"/logs/verifier/base{suffix}", f"/logs/verifier/new{suffix}"]  # pass-to-pass, fail-to-pass
        self.fa_dir = RUNS / run / "analysis" / target

    @functools.cached_property
    def grading(self) -> dict:
        """config.json "test_paths" and "test_paths_mode" for the sanitizing grader, from the top100.jsonl language of
        this build's repository (verifier/test_paths.py, as tools/grading_repair.py repairs the older tasks). Raises
        UnknownLanguage (build() checks it first): never a task without the sanitizer."""
        return TEST_PATHS.config_keys(TEST_PATHS.repo_language(TOP100, self.target, self.repo))

    def remember(self, snapshot: str, role: str) -> None:
        with (RUNS / self.run / "snapshots.jsonl").open("a") as stream:
            stream.write(json.dumps({"snapshot": snapshot, "role": role, "build": str(self.root)}) + "\n")


def eligible_repos(task_id: str) -> list:
    rows = [json.loads(line) for line in TOP100.open()]
    return [row["repo"] for row in sorted(rows, key=lambda row: row["rank"])
            if row["task_id"] == task_id and not SKIP_FLAGS & set(row["flags"])]


def _needs(directory: Path, *files, min_bytes=1):
    """check() for agent stages: result.status == ok and the named files exist (and are non-trivial)."""
    def check(result):
        problems = [] if result.get("status") == "ok" else [f"agent reported status {result.get('status')!r}"]
        for name in files:
            path = directory / name
            if not path.is_file() or path.stat().st_size < min_bytes:
                problems.append(f"{name} missing or empty")
        return problems
    return check


def patch_stats(text: str) -> dict:
    lines = text.splitlines()
    return {"files": grader_module().patch_paths(text),
            "added": sum(l.startswith("+") and not l.startswith("+++") for l in lines),
            "removed": sum(l.startswith("-") and not l.startswith("---") for l in lines)}


def canonical_labels(patch: str) -> str:
    """hengyuf: relabel `git diff -R`-style file headers ("diff --git b/X a/Y", "--- b/X", "+++ a/Y") to the a/ -> b/
    form that the DeepSWE grader's patch_paths parses. Labels only: hunks are untouched and `git apply` treats both
    forms alike, but with reversed labels the grader neither sees nor resets the files (an agent's edits to them then
    survive into grading or make test.patch fail to apply), and the author check's overlap test sees nothing."""
    out, header = [], False
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            header = True
            match = re.match(r'^diff --git ("?)b/(.*?)\1 ("?)a/(.*?)\3(\r?\n?)$', line)
            if match:
                q1, old, q2, new, end = match.groups()
                line = f"diff --git {q1}a/{old}{q1} {q2}b/{new}{q2}{end}"
        elif line.startswith("@@"):
            header = False
        elif header and line.startswith("--- b/"):
            line = "--- a/" + line[6:]
        elif header and line.startswith("+++ a/"):
            line = "+++ b/" + line[6:]
        out.append(line)
    return "".join(out)


def unseen_paths(patch: str) -> list:
    """Files a patch touches that the DeepSWE grader's patch_paths does not report (so it would not reset them)."""
    touched = re.findall(r'^diff --git "?[ab]/.*?"? "?[ab]/(.*?)"?$', patch, re.M)
    return sorted(set(touched) - set(grader_module().patch_paths(patch)))


def added_lines(patch: str, min_len: int) -> list:
    lines = {line[1:].strip() for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")}
    return sorted((line for line in lines if len(line) >= min_len), key=len, reverse=True)


# Lines that are formatting rather than implementation never count as distinctive (hengyuf, 2026-10-18: litestar's new
# options/trace decorators repeat the get/post decorators' docstrings and parameter lists, so C "held" 16-20/20 lines
# of correct solutions): comments and doc comments, docstring bodies, parameter / field declarations, imports.
LEAK_COMMENT = re.compile(r"^(#|//|/\*|\*|--|;|<!--)")
LEAK_PARAM = re.compile(r"^\*{0,2}[A-Za-z_$][\w$]*\??\s*:\s*[^=;{}()]+(=\s*[^;{}]+)?[,;]?$"
                        r"|^\*{0,2}[A-Za-z_][\w]*\s*=\s*[^,;]+,$")
LEAK_IMPORT = re.compile(r"^(import\s|from\s+\S+\s+import\s|use\s|using\s|#include|require\(|export\s*\{)")
DOC_FENCE = re.compile(r'"""|\'\'\'')


def removed_lines(patch: str) -> set:
    """The stripped text of a patch's "-" lines."""
    return {line[1:].strip() for line in patch.splitlines() if line.startswith("-") and not line.startswith("---")}


def leak_tokens(solution_patch: str) -> dict:
    """{solution file: its 20 longest added IMPLEMENTATION lines of >= 30 chars}: the distinctive lines of the
    reference solution (LEAK_COMMENT/LEAK_PARAM/LEAK_IMPORT lines and docstring bodies are skipped, and so are lines the
    patch also removes: moved or re-indented code is in the starting state, not solution code -- hengyuf 2026-10-18,
    official1-testem cand-c01-r1b1's Farm.ts refactor "held" 10/20 of its own distinctive lines in C)."""
    tokens, current, in_doc, removed = {}, None, False, removed_lines(solution_patch)
    for line in solution_patch.splitlines():
        if line.startswith("+++ "):
            current, in_doc = (line[6:] if line.startswith("+++ b/") else None), False
            continue
        if not current or not line.startswith("+"):
            continue
        text = line[1:].strip()
        fences = len(DOC_FENCE.findall(text))
        if in_doc or fences:
            in_doc = in_doc != (fences % 2 == 1)  # an odd number of fences opens or closes a docstring
            continue
        if len(text) >= 30 and text not in removed and not (LEAK_COMMENT.match(text) or LEAK_PARAM.match(text)
                                                            or LEAK_IMPORT.match(text)):
            tokens.setdefault(current, set()).add(text)
    return {path: sorted(lines, key=len, reverse=True)[:20] for path, lines in tokens.items()}


def added_counts(solution_patch: str) -> dict:
    """{solution file: number of added lines}: the size of the solution's code in that file."""
    counts, current = {}, None
    for line in solution_patch.splitlines():
        if line.startswith("+++ "):
            current = line[6:] if line.startswith("+++ b/") else None
        elif current and line.startswith("+"):
            counts[current] = counts.get(current, 0) + 1
    return counts


# A leak inside a dependency tree is a RELEASED copy of the removed code that the author cannot delete (e.g. pnpm's
# ava@8 under node_modules/.pnpm, jest-worker under node_modules/metro, requests under pip/_vendor): a retry could only
# reword the solution past the scan while the released code stays readable, so such candidates fail at once (hengyuf,
# 2026-10-18).
DEPENDENCY_TREE = re.compile(r"/(node_modules|site-packages|dist-packages|_vendor|vendor)/")
# Released copies that released_copies() finds are just as unremovable: files inside an archive or an npm/pip cache entry
# (shown as "<archive>!/<member>") and files under a package-manager cache outside /app (hengyuf, 2026-10-18).
ARCHIVE_MEMBER = re.compile(r"\.(zip|whl|jar|egg|tgz|gz|crate|tar|tbz2|bz2|txz|xz|body)!/|/_cacache/content-v2/[^!]*!/")
CACHE_TREE = re.compile(r"/(\.cache|\.npm|\.yarn|\.bun|\.pnpm-store|\.cargo|_cacache|\.local/share/(pnpm|uv)|pkg/mod)/")


def dependency_leaks(problems: list) -> list:
    """The leak-scan problems whose file lies in a dependency tree (see DEPENDENCY_TREE), inside an archive or cache
    entry (ARCHIVE_MEMBER) or under a package cache outside /app (CACHE_TREE)."""
    found = []
    for problem in problems:
        path = problem.split(" holds ", 1)[0]
        if (" distinctive lines of " in problem or " distinctive identifiers of " in problem) and (
                DEPENDENCY_TREE.search(path) or ARCHIVE_MEMBER.search(path)
                or (not path.startswith("/app/") and CACHE_TREE.search(path))):
            found.append(problem)
    return found


def leak_problems(hits: dict, per_source: dict, added: dict) -> list:
    """A leak is one place reproducing the solution: at least half (>= 4) of one solution file's distinctive lines
    inside one window of a file, the window being as long as that solution file's added code (at least 60 lines).
    hits: {path: [(line number, token)]} from the scan (hengyuf, 2026-10-18: matches scattered over a whole file,
    e.g. lines shared with sibling functions, no longer add up). At least 4 (2026-10-19, official1-aiomonitor c04:
    a 3-line solution file was "held" by 3 boilerplate lines of a sibling method; 3 generic lines are no complete
    implementation). Solution files with fewer distinctive lines are covered by identifier_problems."""
    problems = set()
    for path, found in hits.items():
        for source, lines in per_source.items():
            wanted = set(lines)
            located = sorted((number, token) for number, token in found if token in wanted)
            need, window = max(4, (len(lines) + 1) // 2), max(60, added.get(source, 0))
            best, start = 0, 0
            for end in range(len(located)):
                while located[end][0] - located[start][0] > window:
                    start += 1
                best = max(best, len({token for _, token in located[start:end + 1]}))
            if best >= need:
                problems.add(f"{path} holds {best}/{len(lines)} distinctive lines of {source} within {window} lines")
    return sorted(problems)


# ------------------------------------------------------------------- released copies in archives and package caches
# hengyuf, 2026-10-18 (official1-testem c03): kimi-r4 ran `find /root -name "jest-worker*"`, found jest-worker 27.5.1 and
# 29.7.0 in yarn's global cache (/root/.yarn/berry/cache/*.zip), unzipped 29.7.0 and read the removed processChild /
# threadChild / BaseWorkerPool code. LEAK_SCAN cannot see such copies: `grep -I` skips archives, and /root and the other
# cache roots are only scanned where a directory name contains the repo name. released_copies() closes the gap in three
# sandbox commands: ARCHIVE_MANIFEST reads the repo's own package names (released_names), ARCHIVE_DISCOVER lists every
# archive, npm/pip cache entry, package-manager cache root and package-named directory on C, and ARCHIVE_SCAN unpacks
# the entries naming one of the packages (released_selection) under LEAK_ARCHIVES and greps them, with the cache roots
# and package-named directories, for the solution's distinctive lines (leak_problems) and for the identifiers and string
# literals that only the removed code has (identifier_problems: published packages ship compiled code, types stripped
# and lines reformatted, but names and messages survive).
LEAK_ARCHIVES = "/tmp/leak-archives"
ARCHIVE_DONE = "E\tdone"  # last line of every archive-scan command that ran to its end
ARCHIVE_MANIFEST = r"""set -u
clean() { LC_ALL=C sed -e 's/\r//g' -e 's/\xc2\x85//g' -e 's/\xe2\x80[\xa8\xa9]//g' -e 's/[\x0b\x0c\x1c-\x1e]//g'; }
cd /app || exit 3
for f in package.json pnpm-workspace.yaml lerna.json rush.json pyproject.toml setup.cfg setup.py; do
  if [ -f "$f" ]; then head -c 200000 "$f" | clean | awk -v f="$f" '{print "F\t" f "\t" $0}'; fi
done
git ls-files -- package.json '*/package.json' 2>/dev/null | grep -v -e '^node_modules/' -e '/node_modules/' > /tmp/leak-pkgjson
if command -v node >/dev/null 2>&1; then
  node -e 'const fs = require("fs");
for (const p of fs.readFileSync("/tmp/leak-pkgjson", "utf8").split("\n").filter(Boolean)) {
  try { const j = JSON.parse(fs.readFileSync(p, "utf8")); if (typeof j.name === "string") console.log("J\t" + p + "\t" + j.name + "\t" + (j.private === true ? "private" : "")); } catch (e) { console.log("J\t" + p + "\t\tunparsed"); }
}'
elif command -v python3 >/dev/null 2>&1; then
  python3 -c 'import json
for p in open("/tmp/leak-pkgjson").read().split("\n"):
    try:
        data = json.load(open(p)) if p else {}
    except Exception:
        print("J\t" + p + "\t\tunparsed")
        continue
    if isinstance(data, dict) and isinstance(data.get("name"), str):
        print("J\t" + p + "\t" + data["name"] + "\t" + ("private" if data.get("private") is True else ""))'
else
  while IFS= read -r p; do
    name=$(sed -n 's/^[[:space:]]\{1,4\}"name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$p" | head -1)
    private=$(grep -q '"private"[[:space:]]*:[[:space:]]*true' "$p" && echo private)
    if [ -n "$name" ]; then printf 'J\t%s\t%s\t%s\n' "$p" "$name" "$private"; fi
  done < /tmp/leak-pkgjson
fi
rm -f /tmp/leak-pkgjson
git ls-files -- go.mod '*/go.mod' 2>/dev/null | grep -v -e '/testdata/' -e '^vendor/' -e '/vendor/' | head -501 \
  | { n=0; while IFS= read -r m; do
  n=$((n + 1)); if [ "$n" -gt 500 ]; then printf 'Z\tgo.mod\tmore than 500 go.mod files, the rest unread\n'; break; fi
  printf 'G\t%s\t%s\n' "$m" "$(sed -n 's/^module[[:space:]]*//p' "$m" | head -1 | tr -d '\r"')"; done; }
git ls-files 2>/dev/null | grep -E '^((src|lib|python)/)?[A-Za-z_][A-Za-z0-9_]*/__init__\.py$' | awk '{print "P\t" $0}'
git ls-files -- Cargo.toml '*/Cargo.toml' 2>/dev/null | grep -v -e '^vendor/' -e '/vendor/' -e 'node_modules/' | head -2001 \
  | { n=0; while IFS= read -r c; do
  n=$((n + 1)); if [ "$n" -gt 2000 ]; then printf 'Z\tCargo.toml\tmore than 2000 Cargo.toml files, the rest unread\n'; break; fi
  head -c 200000 "$c" 2>/dev/null | clean | awk -v f="$c" '/^[[:space:]]*\[/ {s=$0; sub(/#.*/, "", s); gsub(/[[:space:]]/, "", s)
      if (s=="[package]") print "K\t" f "\t[package]"}
    s=="[package]" && /^[[:space:]]*(name|publish)[[:space:]]*=/ {print "K\t" f "\t" $0}'
done; }
printf 'E\tdone\n'
"""
# Inputs: /tmp/leak-names (package names, lower case; scoped JS names keep their "@scope/"). Outputs, one per line:
# X tool | A size path (every archive) | C _cacache dir, then N key integrity (its tarball entries) | W pip-cache body
# top-level-names | R cache root | D package-named directory.
ARCHIVE_DISCOVER = r"""set -u
for tool in unzip python3 node bsdtar git; do if command -v "$tool" >/dev/null 2>&1; then printf 'X\t%s\n' "$tool"; fi; done
PRUNE="-path /proc -o -path /sys -o -path /dev -o -path /app/.git -o -path /tmp/leak-archives"
find / -xdev \( $PRUNE \) -prune -o -type f \( -iname '*.zip' -o -iname '*.whl' -o -iname '*.jar' -o -iname '*.egg' \
  -o -iname '*.tgz' -o -iname '*.tar.gz' -o -iname '*.crate' -o -iname '*.tar' -o -iname '*.tbz2' -o -iname '*.tar.bz2' \
  -o -iname '*.txz' -o -iname '*.tar.xz' \) -printf 'A\t%s\t%p\n' 2>/dev/null
find / -xdev \( $PRUNE \) -prune -o -type d -name _cacache -prune -print 2>/dev/null | while IFS= read -r cache; do
  printf 'C\t%s\n' "$cache"
  grep -rhoE '"key":"[^"]*\.tgz","integrity":"[^"]*"' "$cache/index-v5" 2>/dev/null \
    | sed -E 's/^"key":"([^"]*)","integrity":"([^"]*)"$/N\t\1\t\2/' | sort -u
done
find / -xdev \( $PRUNE -o -path /app \) -prune -o -type d \( -name http-v2 -o -name http \) -path '*pip*' -prune -print \
  2>/dev/null > /tmp/leak-pipcache
if [ -s /tmp/leak-pipcache ] && command -v python3 >/dev/null 2>&1; then
  python3 -c 'import os, sys, tarfile, zipfile
for root in open("/tmp/leak-pipcache").read().split():
    for folder, _, files in os.walk(root):
        for name in files:
            path = os.path.join(folder, name)
            if root.endswith("http-v2") and not name.endswith(".body"):
                continue
            try:
                with zipfile.ZipFile(path) as archive:
                    names = archive.namelist()
            except Exception:
                try:
                    with tarfile.open(path) as archive:
                        names = archive.getnames()
                except Exception:
                    continue
            tops = sorted({entry.split("/")[0] for entry in names if entry})[:40]
            print("W\t" + path + "\t" + ",".join(tops))'
elif [ -s /tmp/leak-pipcache ] && command -v unzip >/dev/null 2>&1; then
  while IFS= read -r root; do find "$root" -type f; done < /tmp/leak-pipcache | while IFS= read -r body; do
    tops=$(unzip -Z1 "$body" 2>/dev/null | cut -d/ -f1 | sort -u | head -40 | paste -sd, -)
    if [ -n "$tops" ]; then printf 'W\t%s\t%s\n' "$body" "$tops"; fi
  done
fi
rm -f /tmp/leak-pipcache
for root in /root/.cache /root/.local/share/pnpm /root/.local/share/uv /root/.pnpm-store /root/.bun/install/cache \
    /root/.npm/_npx /root/.cargo/registry/src /usr/local/share/.cache /usr/local/lib/node_modules /usr/lib/node_modules \
    "${XDG_CACHE_HOME:-}" "${PIP_CACHE_DIR:-}" "${UV_CACHE_DIR:-}" "${npm_config_cache:-}" "${YARN_CACHE_FOLDER:-}" \
    "${BUN_INSTALL_CACHE_DIR:-}" "${PNPM_HOME:-}"; do
  if [ -n "$root" ] && [ -d "$root" ]; then printf 'R\t%s\n' "$root"; fi
done
set --
while IFS= read -r name; do
  case "$name" in
    "") ;;
    */*) flat=$(printf '%s' "$name" | tr / -)
         set -- "$@" -o -path "*/$name" -o -path "*/$name@*" -o -iname "npm-$flat-[0-9]*" ;;
    *) under=$(printf '%s' "$name" | tr - _)
       set -- "$@" -o -iname "$name" -o -iname "$name@*" -o -iname "npm-$name-[0-9]*" -o -iname "$name-[0-9]*" \
         -o -iname "$under" -o -iname "$under-[0-9]*" ;;
  esac
done < /tmp/leak-names
if [ $# -gt 0 ]; then
  shift
  find / -xdev \( $PRUNE \) -prune -o -type d \( "$@" \) -prune -printf 'D\t%p\n' 2>/dev/null | head -5000
fi
printf 'E\tdone\n'
"""
# Inputs: /tmp/leak-select ("zip|tar|blob<TAB>path" to unpack), /tmp/leak-roots (directories to scan as they are),
# /tmp/leak-tokens (distinctive lines), /tmp/leak-idents and /tmp/leak-literals (the solution's identifiers and string
# literals; those that the starting state's tracked files contain are dropped and printed as P lines). Outputs:
# U n path status (unpacked into LEAK_ARCHIVES/n) | P present | H path:line:byte:fixed-string | I path:line:byte:identifier.
ARCHIVE_SCAN = r"""set -u
clean() { LC_ALL=C sed -e 's/\r//g' -e 's/\xc2\x85//g' -e 's/\xe2\x80[\xa8\xa9]//g' -e 's/[\x0b\x0c\x1c-\x1e]//g'; }
D=/tmp/leak-archives
rm -rf "$D" && mkdir -p "$D"
unzip_to() {
  if command -v unzip >/dev/null 2>&1; then
    unzip -qq -o "$1" -d "$2" >/dev/null 2>&1; status=$?
    if [ "$status" -le 1 ]; then return 0; fi; return "$status"
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$1" "$2" 2>/dev/null
  elif command -v bsdtar >/dev/null 2>&1; then
    bsdtar -xf "$1" -C "$2" 2>/dev/null
  else
    return 9
  fi
}
untar_to() { tar -xf "$1" -C "$2" --no-same-owner --no-same-permissions 2>/dev/null; }
n=0
while IFS="$(printf '\t')" read -r kind src; do
  n=$((n + 1)); mkdir -p "$D/$n"
  case "$kind" in
    zip) unzip_to "$src" "$D/$n" ;;
    tar) untar_to "$src" "$D/$n" ;;
    *) untar_to "$src" "$D/$n" || { rm -rf "$D/$n"; mkdir -p "$D/$n"; unzip_to "$src" "$D/$n"; } ;;
  esac
  printf 'U\t%s\t%s\t%s\n' "$n" "$src" "$?"
done < /tmp/leak-select
: > /tmp/leak-present
if cd /app 2>/dev/null && git ls-files -z > /tmp/leak-tracked 2>/dev/null; then  # GNU grep: git grep -f is far slower
  if [ -s /tmp/leak-idents ]; then
    xargs -0 -r grep -h -o -w -I -F -f /tmp/leak-idents -- < /tmp/leak-tracked 2>/dev/null | sort -u >> /tmp/leak-present
  fi
  if [ -s /tmp/leak-literals ]; then
    xargs -0 -r grep -h -o -I -F -f /tmp/leak-literals -- < /tmp/leak-tracked 2>/dev/null | sort -u >> /tmp/leak-present
  fi
fi
rm -f /tmp/leak-tracked
cd /
sort -u /tmp/leak-present | clean | awk 'length($0) > 0 {print "P\t" $0}'
grep -v -x -F -f /tmp/leak-present /tmp/leak-idents > /tmp/leak-idents-absent 2>/dev/null
grep -v -x -F -f /tmp/leak-present /tmp/leak-literals > /tmp/leak-literals-absent 2>/dev/null
cat /tmp/leak-tokens /tmp/leak-literals-absent 2>/dev/null | awk 'length($0) > 0' | sort -u > /tmp/leak-fixed
set -- "$D"
while IFS= read -r root; do if [ -n "$root" ] && [ -e "$root" ]; then set -- "$@" "$root"; fi; done < /tmp/leak-roots
if [ -s /tmp/leak-fixed ]; then
  grep -rIonbF -f /tmp/leak-fixed -- "$@" 2>/dev/null | head -300000 | clean | awk '{print "H\t" $0}'
fi
if [ -s /tmp/leak-idents-absent ]; then
  grep -rIonbwF -f /tmp/leak-idents-absent -- "$@" 2>/dev/null | head -300000 | clean | awk '{print "I\t" $0}'
fi
rm -f /tmp/leak-present /tmp/leak-idents-absent /tmp/leak-literals-absent /tmp/leak-fixed
printf 'E\tdone\n'
"""
ARCHIVE_ZIP = re.compile(r"\.(zip|whl|jar|egg)$", re.I)
CODE_FILE = re.compile(r"\.(py|pyi|pyx|js|jsx|mjs|cjs|ts|tsx|mts|cts|go|rs|java|kt|kts|scala|rb|php|c|h|cc|cpp|cxx|hpp|"
                       r"hh|cs|swift|vue|svelte|lua|ex|exs|erl|clj|dart|zig|m|mm)$")
HASH_COMMENTS = re.compile(r"\.(py|pyi|pyx|rb|ex|exs)$")
IDENTIFIER = re.compile(r"[A-Za-z_$][\w$]*")
STRING_LITERAL = re.compile(r"\"(?:[^\"\\\n]|\\.)*\"|'(?:[^'\\\n]|\\.)*'|`(?:[^`\\\n]|\\.)*`")
PLACEHOLDER = re.compile(r"\$\{[^}]*\}|\{[^{}]*\}|%\(?\w*\)?[sdrf]")
# Keywords and builtins of >= 8 characters: not distinctive of any implementation even where the starting state lacks them.
COMMON_WORDS = frozenset("""
    abstract addEventListener annotations arguments asyncIterator Awaitable callable callback classmethod clearImmediate
    clearInterval clearTimeout Callable ClassVar configurable constructor contextmanager continue Coroutine dataclass
    dataclasses debugger defaultdict defineProperty dispatchEvent endsWith enumerable enumerate Exception fallthrough
    frozenset fromEntries function functools Generator getOwnPropertyNames getPrototypeOf globalThis hasOwnProperty
    implements includes Infinity instanceof interface isinstance isFinite isInteger issubclass Iterable Iterator iterator
    itertools KeyError listener NamedTuple namespace nonlocal NonNullable NotImplemented NotImplementedError Optional
    OrderedDict overload override Parameters parseFloat parseInt property protected Protocol prototype RangeError
    readonly Readonly ReferenceError Required removeEventListener ReturnType RuntimeError Sequence setImmediate
    setInterval setPrototypeOf setTimeout startsWith staticmethod stringify SyntaxError toLowerCase toString toUpperCase
    TypedDict TypeError TYPE_CHECKING undefined ValueError AttributeError IndexError AssertionError""".split())


def _added_items(solution_patch: str) -> dict:
    """{solution code file: (identifiers, string literals)} on its added lines, as solution_identifiers reads them. The
    string literals exactly as before 2026-10-20 ("+" lines, docstrings by the fences on "+" lines); the identifiers on
    the code of the "+" lines as _code_hunks reads it, the code that declared_names reads (so names and declarations
    agree on what is code)."""
    found, current, in_doc, removed = {}, None, False, removed_lines(solution_patch)
    removed_names = {name for line in removed for name in IDENTIFIER.findall(line)}
    for line in solution_patch.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
            current, in_doc = (path if path and CODE_FILE.search(path) else None), False
            continue
        if not current or not line.startswith("+"):
            continue
        text = line[1:].strip()
        fences = len(DOC_FENCE.findall(text))
        if in_doc or fences:
            in_doc = in_doc != (fences % 2 == 1)
            continue
        if not text or LEAK_COMMENT.match(text) or LEAK_IMPORT.match(text):
            continue
        literals = found.setdefault(current, (set(), set()))[1]
        for literal in STRING_LITERAL.findall(text):
            for chunk in PLACEHOLDER.split(literal[1:-1]):
                chunk = chunk.strip()
                if len(chunk) >= 8 and sum(c.isalpha() for c in chunk) >= 4 and "\\" not in chunk:
                    literals.add(chunk)
    for path, _, lines, variants in _code_hunks(solution_patch):
        for index, (added, raw) in enumerate(lines):
            text = raw.strip()
            if not added or not text or LEAK_COMMENT.match(text) or LEAK_IMPORT.match(text):
                continue
            names = found.setdefault(path, (set(), set()))[0]
            for codes in variants:
                names.update(name for name in IDENTIFIER.findall(codes[index]) if len(name) >= 8
                             and name not in COMMON_WORDS and not (name.startswith("__") and name.endswith("__")))

    def kept(item):
        return item not in removed_names and (IDENTIFIER.fullmatch(item) or not any(item in line for line in removed))

    return {path: ({name for name in names if kept(name)}, {literal for literal in literals if kept(literal)})
            for path, (names, literals) in found.items()}


def solution_identifiers(solution_patch: str) -> dict:
    """{solution code file: its identifier evidence}: the string literals (>= 8 characters, >= 4 letters) on its added
    lines and the identifiers there (>= 8 characters) that the solution DECLARES (declared_names, hengyuf 2026-10-20):
    the names and messages of the reference implementation, which survive compilation. A name it only references or
    calls is no evidence (an external API's, official1-wasmi c01-r1b4). Comments, docstrings, imports and
    keywords/builtins (COMMON_WORDS) do not count, nor does anything on the patch's "-" lines (moved code);
    documentation files are not code."""
    declared = declared_names(solution_patch)
    found = {path: sorted(literals | (names & declared))
             for path, (names, literals) in _added_items(solution_patch).items()}
    return {path: items for path, items in found.items() if items}


def referenced_identifiers(solution_patch: str) -> dict:
    """{solution code file: the identifiers on its added lines that the solution only references or calls}: what
    solution_identifiers leaves out (diagnostics; never scanned)."""
    declared = declared_names(solution_patch)
    found = {path: sorted(names - declared - literals)
             for path, (names, literals) in _added_items(solution_patch).items()}
    return {path: names for path, names in found.items() if names}


# --------------------------------------------------------------------- declared names: the only identifier evidence
# hengyuf 2026-10-20 (official1-wasmi c01-r1b4 "Preserve dropped data segments in Wizer snapshots" died at its gate:
# wasm-encoder 0.244.0/0.262.0 in the cargo registry held 3/6 "distinctive identifiers" of the new
# crates/wizer/src/instrument.rs, all of them wasmparser/wasm-encoder API that the new code CALLS: CodeSectionReader,
# new_function_with_parsed_locals, parse_global_section). Identifier evidence counts only for names that the
# solution's added code lines DECLARE, never a name they only reference or call, wherever the hits lie: an external
# API name is no evidence, and a released copy of the repo's own code holds the declarations wherever it lies (no
# package classification; the third-party exemptions of 19:57 and 21:12 were refuted). A name declared in any code
# file of the solution counts in all of them (info.rs declares data_drop_globals, instrument.rs uses it).
# Declared: functions, methods, classes, types, structs, enums and their variants, traits, interfaces, modules,
# constants and variables (also destructured, tuple and pattern bindings, for/with/except/catch/match targets,
# comprehension and closure variables, assignment targets), parameters, fields and members (class, struct, interface,
# enum and object-literal members, object-literal functions and arrows, this./self./exports./prototype properties,
# properties assigned to a class: `Pool.defaults = ...`). Not declared: a call, a member access, a type
# reference, a keyword argument of a visible call (its callee declares the parameter), a key of a Go or Rust struct
# literal or of a Python dict display (its type declares the field; the key is an expression), a property that a
# destructuring reads, an import. Where the hunk hides whether a line declares (it starts inside a bracket or block
# whose opening line it does not show), the name counts (hengyuf 2026-10-20: a false positive costs one branch, a false
# negative leaks the answer): `name: X,` in such a block (a field, a member, a parameter), a variant, `name=X,` in a
# bracket that closes a signature (`):`) or not at all. Residual: names an external API also declares count where the
# solution declares them too (a method implementing an external trait or interface, a parameter or a shorthand
# destructuring/pattern binding named after an external field: `DataDrop { data_index }`, an object-literal key
# passed to an external API). String literals stay evidence.
# Python docstrings (the 22:33 refutation: a hunk whose context closes a docstring inverted the state): a hunk is read
# from every start state (code, inside a """ or a ''' string) with each fence on every line toggling it, and a reading
# that takes prose for code (two adjacent words, which Python code never has outside its keywords) is refuted. Names
# and declarations come from the union of the readings left, so the true one, which no prose refutes, is always among
# them (checked on 18,880 real Python files: 0 disagreements with the tokenizer, 0 code lines taken for prose). The
# "+"-line state of _added_items' literals and leak_tokens reads the other languages, and a Python hunk that every
# start state takes for prose.
# v3 (hengyuf 2026-10-21, the verifier of v2's false negatives): f-string replacement fields nest (PEP 701, _py_scan:
# an f-string that nests its own quote around words no longer refutes the true reading; 19,793 files, 0
# disagreements); JS/TS, Go, Rust and C-family hunks carry the /* */ state over their lines (_block_readings: a member
# after a multi-line JSDoc block was not declared); TS parameters of a generic header or a return type that spans
# lines, of a method named after a keyword or an optional method, and `infer X` bindings are declared.
# v4 (hengyuf 2026-10-21, main's decision after the verifier refuted v3: a hunk that starts inside a multi-line string
# whose opening it does not show and that holds a `/*`, argo-cd's Go raw string `(action/*)`, yj's `[/*]`, opened a
# comment that never closed and blanked every later line, real declarations too): monotonic over v2 by construction.
# declared_names = c9f84ee0's declared names (_declared_names_v2: its readings and parser, frozen verbatim below
# _Hunk) | what _Hunk declares in every reading of _code_hunks (v3's readings, then c9f84ee0's), and _added_items takes
# its names from those readings, so v4's evidence holds v2's on every file. A comment that a hunk opens and does not
# close blanks no line after its opener (_block_readings), and a return type is followed over any number of lines, up
# to _TYPE_TOKENS tokens (_Hunk.after_type). Prefer counting: an extra declaration costs at most a false positive, a
# missed one can leak the answer.
_N = r"[A-Za-z_]\w*"
_KW = r"(?:^|[^\w$.:])"  # what may precede a keyword: no name, member access or path
DECLARING = (("rust", re.compile(r"\.rs$")), ("go", re.compile(r"\.go$")), ("python", re.compile(r"\.(py|pyi)$")),
             ("js", re.compile(r"\.(js|jsx|mjs|cjs|ts|tsx|mts|cts|vue|svelte)$")),
             ("c", re.compile(r"\.(java|c|h|cc|cpp|cxx|hpp|hh|cs|dart|m|mm)$")))
KEYWORDS = {
    "rust": frozenset("""as async await break const continue crate dyn else enum extern false fn for if impl in let loop
        match mod move mut pub ref return self Self static struct super trait true type union unsafe use where
        while""".split()),
    "go": frozenset("""break case chan const continue default defer else fallthrough for func go goto if import
        interface map package range return select struct switch type var true false nil iota""".split()),
    "python": frozenset(keyword.kwlist + ["_", "case", "match", "type", "self", "cls"]),
    "js": frozenset("""abstract accessor as async await break case catch class const constructor continue debugger
        declare default delete do else enum export extends false finally for from function get if implements import in
        infer instanceof interface is keyof let module namespace new null of override package private protected public
        readonly return satisfies set static super switch this throw true try type typeof undefined unique var void
        while with yield""".split())}
GENERIC_KEYWORDS = frozenset("""abstract and as assert async auto await bool boolean break byte case catch char class
    const continue def default delete do double else elif elsif end enum extern extends false final finally float for
    foreach fun func function goto if implements import in inline int interface internal is let local long module
    namespace native new nil not null object of operator or override package private protected public raise record
    register return sealed self short signed sizeof static struct super switch synchronized template then this throw
    throws transient true try typedef typename typeof union unless unsigned until using val var virtual void volatile
    when where while yield""".split())
GENERIC_DECLARATION = re.compile(
    rf"{_KW}(?:class|interface|trait|object|record|protocol|module|namespace|typealias|def|defp|defmodule|defmacrop?"
    rf"|defn-?|fun|func|function|fn|sub|val|var|let|local)\s+(?:{_N}\.)*({_N})"
    rf"|{_KW}(?:struct|enum|union)\s+({_N})\s*(?=[{{:;<(]|$)|{_KW}const\s+({_N})\s*(?=[=:])"
    rf"|^typedef\b.*?\b({_N})\s*;$")  # (#define lines are comments to LEAK_COMMENT: never items)
# the C family: `Type name(`, `Type name = `, `Type name;` (modifiers, generics, arrays, pointers in the type)
C_TYPED = re.compile(rf"^((?:[A-Za-z_][\w:]*(?:\s*<[^;()]*>)?(?:\s*\[\s*\])*[\s*&]+)+)(?:{_N}::)*(~?{_N})"
                     r"\s*(?:\(|=(?!=)|;|\[|,|\{|$)")
C_TYPE_DEFINITION = re.compile(rf"{_KW}(?:class|interface|enum|record|namespace|typedef)\b|\b(?:extends|implements)\b"
                               rf"|{_KW}(?:struct|union)\s+{_N}\s*(?:[{{:;]|$)")  # GENERIC_DECLARATION's lines
C_ATTRIBUTES = re.compile(r"^(?:@[\w.]+(?:\([^)]*\))?\s*|\[\[[^\]]*\]\]\s*|\[[A-Z]\w*(?:\([^)]*\))?\]\s*)+")
C_STATEMENT = frozenset("""assert await case catch co_await co_return co_yield delete do echo else for goto if import
    new package print puts raise return sizeof switch throw typeof using while yield""".split())
# Python: identifiers that Python code may place next to another name (keywords, soft keywords, Python 2 statements)
PY_ADJACENT = frozenset(keyword.kwlist + ["_", "case", "match", "type", "print", "exec"])
_TOKEN = re.compile(r"\d[\w.]*|#?[A-Za-z_$][\w$]*|\.\.\.|[=!]==|=>|->|::|:=|\?\.|\?\?=?|&&=?|\|\|=?"
                    r"|(?:\*\*|//|>>>|>>|<<|[-+*/%&|^@!=<>])=|\+\+|--|[^\s\w$]")
_NAME = re.compile(r"#?[A-Za-z_$][\w$]*$")
_ASSIGN = frozenset("+= -= *= /= //= %= **= >>= <<= >>>= &= |= ^= @= ??= &&= ||=".split())
_PY_SPECIAL = re.compile(r"[#'\"]")  # what ends a run of Python code on a line: a comment or a string
_PY_PREFIXES = ("r", "b", "u", "f", "t", "rb", "br", "fr", "rf", "tr", "rt")
_PY_FIELD = re.compile(r"[#'\"()\[\]{}:]")  # what matters in an f-string's replacement field (code)
_PY_FSTRING = {quote: re.compile(r"\{\{|\}\}|[{}]|" + quote) for quote in ('"', "'", '"""', "'''")}
_PY_SPEC = {quote: re.compile(r"[{}]|" + quote) for quote in ('"', "'", '"""', "'''")}


def _py_scan(text: str, quote):
    """(code, state after) of one Python line read from the state `quote` (None: from code; a triple quote, or a
    one-quote string that a backslash continues: from inside that string; a tuple: from inside an f-string's
    nesting): strings blanked with their prefixes, the contents of strings dropped, the comment cut. An f- or
    t-string's replacement fields nest (PEP 701, hengyuf 2026-10-21): a field is code, and a string inside it may use
    the f-string's own quote (f"{d["display name"]}", f"{"Not found" if x is None else x}"), so the f-string ends at
    its own closing quote; the whole f-string is one value. The state after is None in code, the quote inside a plain
    string (as before), and a tuple of frames (("s", quote, f-string), ("f", depth) a field, ("spec",) its format
    spec) inside an f-string."""
    stack = list(quote) if isinstance(quote, tuple) else ([("s", quote, False)] if quote else [])
    out, index, length = [], 0, len(text)
    while index < length:
        if not stack:  # code
            special = _PY_SPECIAL.search(text, index)
            if not special:
                out.append(text[index:])
                break
            out.append(text[index:special.start()])
            index = special.start()
            if text[index] == "#":
                break
            code, formatted = "".join(out), False
            prefix = re.search(r"(?<![\w$])[rRbBuUfFtT]{1,2}$", code[-3:]) if code[-3:] else None
            if prefix and prefix.group(0).lower() in _PY_PREFIXES \
                    and not re.search(r"[\w$]$", code[:len(code) - len(prefix.group(0))]):
                out = [code[:len(code) - len(prefix.group(0))]]
                formatted = bool(set(prefix.group(0).lower()) & {"f", "t"})
            opened = text[index] * 3 if text.startswith(text[index] * 3, index) else text[index]
            stack.append(("s", opened, formatted))
            index += len(opened)
            continue
        frame = stack[-1]
        if frame[0] == "s" and not frame[2]:  # a plain string: its end (as before)
            end = text.find(frame[1], index)
            while end > 0 and _escaped(text, end):
                end = text.find(frame[1], end + 1)
            if end < 0:
                break
            stack.pop()
            index = end + len(frame[1])
            if not stack:
                out.append(" 0 ")  # (a value token: the line goes on)
        elif frame[0] == "s":  # an f-string: its end, a replacement field, an escaped brace ({{ }}) or a stray }
            found = _PY_FSTRING[frame[1]].search(text, index)
            if not found:
                break
            index = found.end()
            if found.group(0) == frame[1]:
                if _escaped(text, found.start()):
                    index = found.start() + 1
                    continue
                stack.pop()
                if not stack:
                    out.append(" 0 ")
            elif found.group(0) == "{":
                stack.append(("f", 0))
        elif frame[0] == "f":  # a replacement field: code to its "}" (brackets nest), a format spec after a ":"
            found = _PY_FIELD.search(text, index)
            if not found:
                break
            index, char = found.end(), found.group(0)
            if char == "#":
                break  # a comment in a field that spans lines
            if char in "'\"":
                prefix = _py_prefix(text, found.start())
                opened = char * 3 if text.startswith(char * 3, found.start()) else char
                stack.append(("s", opened, bool(set(prefix) & {"f", "t"})))
                index = found.start() + len(opened)
            elif char in "([{":
                stack[-1] = ("f", frame[1] + 1)
            elif char in ")]" or char == "}" and frame[1]:
                stack[-1] = ("f", max(0, frame[1] - 1))
            elif char == "}":
                stack.pop()
            elif char == ":" and not frame[1]:
                stack[-1] = ("spec",)
        else:  # a format spec: text to the "}" that ends it and its field; a "{" opens a nested field
            string = next(item for item in reversed(stack) if item[0] == "s")
            found = _PY_SPEC[string[1]].search(text, index)
            if not found:
                break
            index = found.end()
            if found.group(0) == "{":
                stack.append(("f", 0))
            elif found.group(0) == "}":
                stack.pop()
            elif _escaped(text, found.start()):
                index = found.start() + 1
            else:  # the f-string's quote inside a spec (no valid code): it ends the f-string
                while stack.pop() is not string:
                    pass
                if not stack:
                    out.append(" 0 ")
    if stack:
        string = next((item for item in reversed(stack) if item[0] == "s"), None)
        if stack[-1][0] in ("s", "spec") and len(string[1]) == 1 and not _escaped(text, length):
            stack = []  # a one-quote string that the line's end leaves open ends there (as before)
    if not stack:
        return "".join(out), None
    if len(stack) == 1 and not stack[0][2]:
        return "".join(out), stack[0][1]
    return "".join(out), tuple(stack)


def _py_prefix(text: str, end: int) -> str:
    """The string prefix (lower case, "" if none) whose letters end at text[end] (a quote): one or two prefix
    letters that no name character precedes."""
    for size in (2, 1):
        start = end - size
        if start >= 0 and text[start:end].lower() in _PY_PREFIXES and not (
                start > 0 and (text[start - 1].isalnum() or text[start - 1] in "_$")):
            return text[start:end].lower()
    return ""


def _escaped(text: str, position: int) -> bool:
    """Is text[position] (or the line's end) preceded by an odd number of backslashes?"""
    count = 0
    while position - count - 1 >= 0 and text[position - count - 1] == "\\":
        count += 1
    return count % 2 == 1


def _py_prose(code: str) -> bool:
    """Do two names stand next to each other (prose, never Python code outside its keywords)?"""
    words = re.findall(r"\d[\w.]*|[^\W\d]\w*|\S", code)
    return any((a[0].isalpha() or a[0] == "_") and (b[0].isalpha() or b[0] == "_") and a not in PY_ADJACENT
               and b not in PY_ADJACENT for a, b in zip(words, words[1:]))


def _plain_code(path: str, text: str) -> str:
    """One line's code as _added_items reads it: strings blanked, the comment cut (Rust lifetimes are no quotes)."""
    if path.endswith(".rs"):
        text = re.sub(r"'(?:_|[A-Za-z_]\w*)(?![\w'])", "", text)
    return re.split(r"#" if HASH_COMMENTS.search(path) else r"//|/\*", STRING_LITERAL.sub(" 0 ", text))[0]


# /* */ comments over a hunk's lines (hengyuf 2026-10-21, the verifier of the declared-names rule v2: a class method,
# field, interface or object-literal member right after a multi-line JSDoc block, `*/` on its own line, was not
# declared, nor were the method's parameters: " * ..." and " */" lines were read as code, one line at a time). For
# JS/TS, Go, Rust and the C family a hunk is read from two start states, code and inside a comment that a line before
# the hunk opened, with the comment state carried from line to line (the way Python's docstring readings work). A
# reading is refuted where it meets a comment's end (`*/`) outside any comment, string or regex literal, at a place
# where a comment ends (alone, after `*`s at the line's start, or at its end: _comment_end); the reading from inside a
# comment also where that first comment holds a `/*` (C-family comments do not nest: the hunk opened it) or ends
# elsewhere (a `*/` inside a string it cannot see: argo-cd's `"*/*"`). Strings that span lines are tracked so that no
# delimiter inside them counts (JS template literals and their `${}` code, Go raw strings, Rust strings and raw
# strings, Java/C#/Dart text blocks, C# verbatim and C++ raw strings, a string a backslash continues), and JS regex
# literals; checked against the TypeScript 5.9.3 parser, go/parser and proc-macro2 on whole real files (2026-10-21:
# 0 disagreements on 3.6 M lines). A line where a reading sees no comment reads as before (_plain_code: a string's
# lines too); comments are blanked out of the others ("" inside one). Readings left: merged; none: as before.
_BLOCK_LANGUAGES = ("js", "go", "rust", "c")
_BLOCK_TOKEN = re.compile(r"//|/\*|\*/|(?:[^\W\d]|\$)[\w$]*|\d[\w.]*|\S")
_JS_REGEX_AFTER = frozenset("return typeof case do else in of new delete void throw yield await instanceof".split())
_TEMPLATE_TEXT = re.compile(r"\\.|`|\$\{")
_RUST_LIFETIME = re.compile(r"'(?:_|[A-Za-z_]\w*)(?![\w'])")
_TYPE_GOES_ON = (":", "|", "&", "?", "<", "extends", "keyof", "typeof", "infer", "readonly", "is")  # (a TS type)
# v4: how many tokens of a return type after_type reads, a bracket in it counted as one (v3: 12 lines). Real return
# types read at most 47 (19,064 scans: every JS/TS file diff of the recorded patches, and whole jest, zsa, sqltools,
# bitECS and typescript.js files, 2026-10-21); the bound keeps a run of `f(x) : a < b` lines (an unclosed `<` after a
# call's `:` reads on to its block's end) linear: 4,000 such lines took 43-91 s without one.
_TYPE_TOKENS = 1000


def _block_dialect(path: str) -> str:
    """The string syntax of a /* */ language's file: js, go, rust, java (Java text blocks), cs (C# raw and verbatim
    strings), dart (''' and \"\"\" strings), c (C, C++ raw strings, Objective-C), "" (none of them)."""
    language = next((name for name, regex in DECLARING if regex.search(path)), "generic")
    if language != "c":
        return language if language in _BLOCK_LANGUAGES else ""
    extension = path.rsplit(".", 1)[-1].lower()
    return extension if extension in ("java", "cs", "dart") else "c"


def _string_end(text: str, position: int, terminator: str, escapes: str):
    """The index after `terminator` at or after `position` (escapes: "\\\\" a backslash escapes it, '"' a doubled
    quote is no end (C# verbatim), "" none), or None when the line does not hold it."""
    end = text.find(terminator, position)
    while end >= 0:
        if escapes == "\\" and _escaped(text, end):
            end = text.find(terminator, end + 1)
        elif escapes == '"' and text.startswith('""', end):
            end = text.find(terminator, end + 2)
        else:
            return end + len(terminator)
    return None


def _block_scan(dialect: str, text: str, state: tuple) -> tuple:
    """One line of a /* */ language (_block_dialect) read from `state`, a tuple of frames: () in code, ("cmt", depth)
    a comment (Rust's nest), ("tpl",) a JS template literal's text, ("expr", depth) its `${}` code, ("end",
    terminator, escapes, spans lines) a string that ends at terminator. Returns (comment spans [(start, end)], state
    after, stray, opener, ended): stray, a `*/` outside any comment, string or regex literal where a comment ends
    (_comment_end); opener, a `/*` inside the comment that was open when the line began, before it ends; ended, None
    when that comment does not end on the line, else whether its `*/` is where a comment ends."""
    stack, spans, stray, opener, ended = list(state), [], False, False, None
    begin = 0 if stack and stack[-1][0] == "cmt" else None  # where the comment being read began on the line
    initial, index, length, last = begin == 0, 0, len(text), ""  # last: the previous code token (JS: regex or "/")
    while index < length:
        frame = stack[-1] if stack else ("code",)
        if frame[0] == "cmt":
            close, nested = text.find("*/", index), text.find("/*", index)
            if nested >= 0 and (close < 0 or nested < close):
                opener = opener or initial
                if dialect == "rust":
                    stack[-1], index = ("cmt", frame[1] + 1), nested + 2
                    continue
            if close < 0:
                break
            index = close + 2
            if frame[1] > 1:
                stack[-1] = ("cmt", frame[1] - 1)
                continue
            stack.pop()
            spans.append((begin, index))
            if initial:
                ended, initial = _comment_end(text, close, index), False
            continue
        if frame[0] == "tpl":
            found = _TEMPLATE_TEXT.search(text, index)
            if not found:
                break
            index = found.end()
            if found.group(0) == "`":
                stack.pop()
                last = "0"
            elif found.group(0) == "${":
                stack.append(("expr", 0))
                last = "{"
            continue
        if frame[0] == "end":
            end = _string_end(text, index, frame[1], frame[2])
            if end is None:
                if not frame[3] and not _escaped(text, length):
                    stack.pop()  # a one-line string that the line leaves open (no valid code) ends there
                break
            stack.pop()
            index, last = end, "0"
            continue
        token = _BLOCK_TOKEN.match(text, index)
        if not token:  # whitespace
            index += 1
            continue
        word, start, index = token.group(0), token.start(), token.end()
        if word == "//":
            break
        if word == "/*":
            stack.append(("cmt", 1))
            begin = start
            continue
        if word == "*/":
            if text.startswith(("/*", "//"), start + 1):  # `*//*`: a `*`, then a comment
                index = start + 1
            else:
                stray = stray or _comment_end(text, start, index)
            last = "*"
            continue
        string = _block_string(dialect, text, start, word) if word[0] in "\"'`@$" or word in (
            "r", "br", "cr") or word[-1] == "R" else None
        if string is not None:
            if isinstance(string, int):
                index, last = string, "0"
            else:  # a string that goes on past the line
                stack.append(string[0])
                index = string[1]
            continue
        if dialect == "js" and word == "/":
            if not last or last in _JS_REGEX_AFTER or not (last[0].isalnum() or last[0] in "_$") and last not in ")]":
                end = _regex_end(text, index)
                if end is not None:
                    index, last = end, "0"
                    continue
        elif frame[0] == "expr" and word in "{}":
            if word == "{":
                stack[-1] = ("expr", frame[1] + 1)
            elif frame[1]:
                stack[-1] = ("expr", frame[1] - 1)
            else:
                stack.pop()  # the end of a template's `${}`: its text goes on
                continue
        last = word
    if stack and stack[-1][0] == "cmt":
        spans.append((begin, length))  # the comment goes on past the line
    return spans, tuple(stack), stray, opener, ended


def _comment_end(text: str, start: int, end: int) -> bool:
    """Is the `*/` at text[start:end] where a comment ends: alone, after `*`s at the line's start, or at its end?"""
    return not text[:start].strip().strip("*") or not text[end:].strip()


def _block_string(dialect: str, text: str, start: int, word: str):
    """The string literal that token `word` at text[start] opens (a quote, a raw or verbatim string's prefix): the
    index after it when it ends on the line, (frame, index after its opener) when it goes on past the line, or None
    (no string: a quote that opens nothing, as a prose apostrophe)."""
    after = start + len(word)
    if word == "`":
        if dialect == "js":
            return ("tpl",), after
        if dialect == "go":  # a raw string
            end = _string_end(text, after, "`", "")
            return end if end is not None else (("end", "`", "", True), after)
        return None
    if word in ('"', "'"):
        triple = word * 3
        if (dialect in ("java", "cs") and word == '"' or dialect == "dart") and text.startswith(triple, start):
            end = _string_end(text, start + 3, triple, "" if dialect == "cs" else "\\")  # a text block
            return end if end is not None else (("end", triple, "" if dialect == "cs" else "\\", True), start + 3)
        single = STRING_LITERAL.match(text, start)
        if single:
            return single.end()
        if dialect == "rust" and word == '"':  # Rust strings span lines
            return ("end", '"', "\\", True), after
        if (word == '"' or dialect not in ("rust", "go")) and _escaped(text, len(text)):
            return ("end", word, "\\", False), after  # a string that a backslash continues
        return None
    if dialect == "rust" and word in ("r", "br", "cr") and text.startswith(("\"", "#"), after):
        hashes = len(text[after:]) - len(text[after:].lstrip("#"))
        if not text.startswith('"', after + hashes):
            return None
        terminator = '"' + "#" * hashes
        end = _string_end(text, after + hashes + 1, terminator, "")
        return end if end is not None else (("end", terminator, "", True), after + hashes + 1)
    if dialect == "c" and re.fullmatch(r"(?:u8|[uUL])?R", word) and text.startswith('"', after):
        found = re.match(r'"([^()\\\s"]{0,16})\(', text[after:])
        if not found:
            return None
        terminator = ")" + found.group(1) + '"'
        end = _string_end(text, after + found.end(), terminator, "")
        return end if end is not None else (("end", terminator, "", True), after + found.end())
    if dialect == "cs" and word in ("@", "$"):
        verbatim = re.match(r'@\$?"|\$@"', text[start:start + 3])
        if verbatim:
            end = _string_end(text, start + verbatim.end(), '"', '"')
            return end if end is not None else (("end", '"', '"', True), start + verbatim.end())
    return None


def _regex_end(text: str, index: int):
    """The index after the JS regex literal whose body starts at text[index] (after its "/"), flags included, or None
    when the line holds no closing "/" (a division)."""
    klass, length = False, len(text)
    while index < length:
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "[":
            klass = True
        elif char == "]":
            klass = False
        elif char == "/" and not klass:
            index += 1
            while index < length and (text[index].isalnum() or text[index] == "_"):
                index += 1
            return index
        index += 1
    return None


def _block_readings(path: str, lines: list, plus: list, blank: list) -> list:
    """The readings of one hunk of a /* */ language (see above): lines [(added, raw)], plus their "+"-line codes,
    blank which of them the "+"-line docstring state blanks. v4 (hengyuf 2026-10-21, the verifier of v3): a comment
    that the hunk opens and that is still open at the hunk's end blanks no line after the line that opened it; those
    lines read as their "+"-line codes, as before v3. Its `/*` is most likely inside a string whose opening the hunk
    does not show (a Go raw string or JS template that the hunk starts inside, JSX text: argo-cd project.go's cobra
    Example `(action/*)`, yj main.go's HelpMsg `[/*]`), and the code after it is real (`RunE: func(c *cobra.Command,
    args []string)`, `var Version`). A comment that closes in the hunk always shows its end, so JSDoc before a member
    stays blank."""
    dialect, variants = _block_dialect(path), []
    for start in ((), (("cmt", 1),)):
        state, codes, initial, opened = start, [], bool(start), None  # opened: the line that opened the open comment
        for number, ((_, raw), base, empty) in enumerate(zip(lines, plus, blank)):
            text = _RUST_LIFETIME.sub(lambda match: " " * len(match.group(0)), raw) if dialect == "rust" else raw
            inside = bool(state) and state[-1][0] == "cmt"
            spans, state, stray, opener, ended = _block_scan(dialect, text, state)
            if stray or initial and (opener or ended is False):
                codes = None  # a */ out of any comment; the hunk's first comment holds a /* or ends mid-code
                break
            initial = initial and ended is None
            if not (state and state[-1][0] == "cmt"):
                opened = None
            elif not (inside and spans[-1][0] == 0):  # (a comment open since an earlier line spans from column 0)
                opened = number  # this line opens the comment that goes on past it
            if not spans or empty:
                codes.append(base)
                continue
            kept, position = [], 0
            for begin, end in spans:
                kept += [raw[position:begin], " "]
                position = end
            code = "".join(kept) + raw[position:]
            codes.append(_plain_code(path, code) if code.strip() else "")
        if codes is not None and opened is not None:
            codes[opened + 1:] = plus[opened + 1:]  # the hunk opened a comment it does not close: no comment after
        if codes is not None and codes not in variants:
            variants.append(codes)
    return variants


def _code_hunks(solution_patch: str) -> list:
    """v4 (hengyuf 2026-10-21): [(path, header, lines, variants)] per hunk as _code_hunks_v3 reads it, with every
    reading of c9f84ee0's _code_hunks_v2 that is not among its variants added after them, so c9f84ee0's reading is
    always one of the readings: _added_items takes its names from these readings, and declared_names' _Hunk parses
    them."""
    hunks = _code_hunks_v3(solution_patch)
    for index, (path, header, lines, variants) in enumerate(_code_hunks_v2(solution_patch)):
        if index < len(hunks) and hunks[index][:3] == (path, header, lines):
            for codes in variants:
                if codes not in hunks[index][3]:
                    hunks[index][3].append(codes)
        else:  # (never: both read the same hunks of the same patch; kept whole all the same)
            hunks.append((path, header, lines, variants))
    return hunks


def _code_hunks_v3(solution_patch: str) -> list:
    """[(path, header, lines, variants)] per hunk of the solution's code files (CODE_FILE): lines [(added, text)] is
    the hunk's new side ("+" and context lines), header git's hunk header text, variants its readings [code of every
    line] ("" where a reading sees no code). Python: every start state (code, inside a \"\"\" or a ''' string), each
    fence on every line toggling it (_py_scan), that takes no prose for code (_py_prose), so the true reading is always
    among them. JS/TS, Go, Rust and the C family: from code and from inside a /* comment, the comment state carried
    over the lines, that meet no comment's end outside a comment (_block_readings). Other languages, and a hunk that
    every start state refutes: the "+"-line docstring state of _added_items' literals and leak_tokens (context lines
    with a fence blank; Python prose lines blank)."""
    hunks, path, header, lines, in_doc = [], None, "", [], False

    def flush():
        nonlocal in_doc
        if not path or not any(added for added, _ in lines):
            lines.clear()
            return
        python = bool(re.search(r"\.(py|pyi)$", path))
        plus, blank = [], []
        for added, raw in lines:
            text = raw.strip()
            fences = len(DOC_FENCE.findall(text))
            blank.append(bool(added and (in_doc or fences) or fences))
            if added and (in_doc or fences):
                in_doc = in_doc != (fences % 2 == 1)
                plus.append("")
            elif fences:
                plus.append("")
            else:
                plus.append(_py_scan(raw, None)[0] if python else _plain_code(path, raw))
        variants = _block_readings(path, lines, plus, blank) if _block_dialect(path) else []
        if python:  # every start state that takes no prose for code; the "+"-line state when none is left
            for start in (None, '"""', "'''"):
                quote, codes = start, []
                for _, raw in lines:
                    code, quote = _py_scan(raw, quote)
                    if _py_prose(code):
                        break
                    codes.append(code)
                else:
                    if codes not in variants:
                        variants.append(codes)
        if not variants:
            variants = [[code if not (python and _py_prose(code)) else "" for code in plus]]
        hunks.append((path, header, list(lines), variants))
        lines.clear()

    for line in solution_patch.splitlines():
        if line.startswith("+++ "):
            flush()
            new = line[6:] if line.startswith("+++ b/") else ""
            path, header, in_doc = (new if CODE_FILE.search(new) else None), "", False
        elif line.startswith("diff --git "):
            flush()
            path = None
        elif line.startswith("@@"):
            flush()
            header = line.split("@@", 2)[2].strip() if line.count("@@") >= 2 else ""
        elif path and line[:1] in ("+", " ", ""):
            lines.append((line[:1] == "+", line[1:]))
    flush()
    return hunks


def declared_names(solution_patch: str) -> set:
    """The names that the solution's added code lines declare (see DECLARING). v4 (hengyuf 2026-10-21): c9f84ee0's
    declared names (_declared_names_v2: its readings and its parser, frozen below) and the names that _Hunk declares
    in every reading of every hunk (_code_hunks: v3's readings and c9f84ee0's), so v4 holds every name v2 declares by
    construction (an extra declaration costs at most a false positive, a missed one can leak the answer)."""
    return _declared_names_v2(solution_patch) | _declared_in(_code_hunks(solution_patch))


def _declared_in(hunks: list) -> set:
    """The names that _Hunk declares in every reading of every hunk of `hunks` (_code_hunks' form): the "+" and
    context lines show the enclosing brackets and blocks. Members assigned to a class (`Pool.defaults = ...`: a class
    or JS function that the solution declares, or a capitalized name) count too."""
    names, owners, stores = set(), set(), []
    for path, header, lines, variants in hunks:
        language = next((name for name, regex in DECLARING if regex.search(path)), "generic")
        for codes in variants:
            found = _Hunk(language, header, lines, codes, path).declarations()
            names |= found[0]
            owners |= found[1]
            stores += found[2]
    names |= {member for owner, member in stores if owner in owners or owner[:1].isupper()}
    return {name.lstrip("#") for name in names if len(name.lstrip("#")) > 0}


class _Hunk:
    """One reading of one hunk as tokens with their brackets: parent[i] is the innermost bracket around token i (an
    opening token's index, or -1 - j for a bracket that opens before the hunk and that token j closes, or None)."""

    def __init__(self, language: str, header: str, lines: list, codes: list, path: str = ""):
        self.language, self.header, self.added, self.path = language, header, [added for added, _ in lines], path
        self.codes, self.text, self.line, self.ids = codes, [], [], []
        if language == "rust":
            codes = [re.sub(r"'(?:_|[A-Za-z_]\w*)(?![\w'])", "", code) for code in codes]
        for number, code in enumerate(codes):
            ids = []
            if len(code) <= 400:  # (a minified line declares nothing)
                for match in _TOKEN.finditer(code):
                    ids.append(len(self.text))
                    self.text.append(match.group(0))
                    self.line.append(number)
            self.ids.append(ids)
        count = len(self.text)
        self.match, self.parent, stack = [None] * count, [None] * count, []
        unopened = []
        for index, text in enumerate(self.text):
            if text in ")]}":
                if stack:
                    opener = stack.pop()
                    self.match[opener], self.match[index] = index, opener
                else:
                    unopened.append(index)
                self.parent[index] = stack[-1] if stack else None
            else:
                self.parent[index] = stack[-1] if stack else None
                if text in "([{":
                    stack.append(index)
        outer, unopened = None, set(unopened)  # the bracket opened before the hunk that the next unopened closer closes
        for index in range(count - 1, -1, -1):
            if self.parent[index] is None:
                self.parent[index] = outer
            if index in unopened:
                outer = -1 - index
        self.kinds = {}

    # ------------------------------------------------------------------ helpers
    def name(self, index) -> bool:
        return 0 <= index < len(self.text) and bool(_NAME.match(self.text[index])) and not self.text[index][0].isdigit()

    def word(self, index) -> str:
        return self.text[index] if 0 <= index < len(self.text) else ""

    def same_line(self, a, b) -> bool:
        return 0 <= a < len(self.text) and 0 <= b < len(self.text) and self.line[a] == self.line[b]

    def before(self, index) -> list:
        """The tokens before `index` on its line (the previous code line's when it starts one)."""
        line = self.line[index]
        start = index
        while start > 0 and self.line[start - 1] == line:
            start -= 1
        if start == index and start > 0:
            previous = self.line[start - 1]
            while start > 0 and self.line[start - 1] == previous:
                start -= 1
        return list(range(start, index))

    def after_close(self, opener) -> list:
        """The tokens after the bracket that closes `opener`, on the closer's line ([] when it does not close here)."""
        closer = self.match[opener] if opener is not None and opener >= 0 else (-1 - opener if opener is not None
                                                                                  else None)
        if closer is None:
            return []
        end = closer + 1
        while end < len(self.text) and self.line[end] == self.line[closer]:
            end += 1
        return list(range(closer + 1, end))

    def after_type(self, opener) -> list:
        """after_close(opener); when a type annotation that goes on past the closer's line follows it (`): If<`, an
        open bracket or `<`, a line ending in `|`, `&`, `=>`, ...): its tokens, to the `{`, `;`, `=>` or `,` that ends
        it outside its brackets and `<...>`, the bracket around it closing, or the line end where it is complete. v4
        (hengyuf 2026-10-21, the verifier of v3): over any number of lines, at most _TYPE_TOKENS tokens (v3 stopped
        after 12 lines, and an object-literal method whose `Promise<{...}>` return type was longer lost its name and
        parameters); a bracket inside the type that closes in the hunk is passed over whole, as one token (none of
        its tokens can end the type)."""
        tokens = self.after_close(opener)
        if not tokens or self.text[tokens[0]] != ":" or self.language != "js":
            return tokens
        parent, angle, index, budget = self.parent[tokens[0]], 0, tokens[0] + 1, _TYPE_TOKENS
        while index < len(self.text) and budget:
            budget -= 1
            text = self.text[index]
            if self.parent[index] != parent and not self.inside(index, parent):
                return list(range(tokens[0], index + 1))  # the bracket around the signature closes
            if self.parent[index] == parent:
                if text == "<":
                    angle += 1
                elif text == ">":
                    angle = max(0, angle - 1)
                elif text == "{" and not angle and self.word(index - 1) in _TYPE_GOES_ON and self.match[index]:
                    index = self.match[index]  # an object type (`): { end: number } => {`): on from its "}"
                    continue
                elif not angle and text in ("{", ";", "=>", ","):
                    return list(range(tokens[0], index + 1))
            if (index + 1 == len(self.text) or self.line[index + 1] != self.line[index]) and not angle \
                    and self.parent[index] == parent and text not in _TYPE_GOES_ON + (",", "=>", "."):
                return list(range(tokens[0], index + 1))  # the line ends where the type is complete
            if self.parent[index] == parent and text in ("(", "[", "{") and self.match[index]:
                index = self.match[index]  # on from its closer (the tokens inside it are inside `parent`)
                continue
            index += 1
        return list(range(tokens[0], index))

    def start(self, index) -> bool:
        """Does token `index` start an element of its bracket or block (a statement, member, parameter, entry)?"""
        if index == 0:
            return True
        parent, previous = self.parent[index], self.text[index - 1]
        if parent is not None and parent >= 0 and index - 1 == parent:
            return True
        if previous in (",", ";") and self.parent[index - 1] == parent:
            return previous == ";" or not self.in_generic(index - 1)
        if self.line[index - 1] == self.line[index]:
            return False
        if self.language == "python":
            return parent is None and previous != "\\"
        if previous in ("{", "}", ";", ")", "]", "++", "--") or previous in ("else", "do", "try", "finally"):
            return True
        if previous == ">":  # a type's generic arguments end the line: a member or field ends there
            return self.kind(parent) not in ("block", "call", "group", "cond", "array", "composite", "literal")
        if self.name(index - 1) and previous not in (
                "return", "new", "typeof", "await", "yield", "case", "in", "of", "instanceof", "as", "extends",
                "implements", "throw", "delete", "keyof", "satisfies", "is", "and", "or", "not", "if", "elif",
                "while", "for", "switch", "match", "where", "impl", "dyn", "mut", "ref", "let", "const", "var", "func",
                "type", "struct", "enum", "interface", "fn", "pub", "static", "export", "default", "async"):
            return True
        return bool(re.match(r"\d", previous))

    def in_generic(self, comma) -> bool:
        """Is the "," at `comma` inside a `<...>` that opens before it on its line, at its bracket depth (a type's
        generic arguments, `Map<number, Promise<void>>`, whose "," starts no element)?"""
        if self.language not in ("js", "rust", "c"):
            return False
        depth, index = 0, comma - 1
        while index >= 0 and self.line[index] == self.line[comma]:
            if self.parent[index] == self.parent[comma]:
                if self.text[index] == ">":
                    depth += 1
                elif self.text[index] == "<":
                    if depth == 0:
                        return True
                    depth -= 1
                elif self.text[index] in (";", "{", "}"):
                    break
            elif index == self.parent[comma]:
                break
            index -= 1
        return False

    def head_tokens(self, bracket) -> list:
        """The texts of the tokens before `bracket` back to the previous statement's end (`;`, `{` or `}` around it),
        over at most three lines: a declaration header that may span lines (`class X<\n T,\n> extends Y {`)."""
        out, index = [], bracket - 1
        while index >= 0 and self.line[bracket] - self.line[index] <= 3:
            if self.parent[index] == self.parent[bracket] and self.text[index] in (";", "{", "}"):
                break
            if index == self.parent[bracket]:
                break
            out.append(self.text[index])
            index -= 1
        return out[::-1]

    def statement(self, index) -> list:
        """The tokens of the element that starts at `index`, inside its bracket, to the end of its line (a
        statement outside brackets runs on over its commas: `a, b = ...`)."""
        parent, out, end = self.parent[index], [], index
        while end < len(self.text) and self.line[end] == self.line[index]:
            if self.parent[end] == parent and (self.text[end] == ";" or self.text[end] == "," and parent is not None) \
                    and end != index:
                break
            if self.parent[end] != parent and end != index and not self.inside(end, parent):
                break
            out.append(end)
            end += 1
        return out

    def inside(self, index, bracket) -> bool:
        """Is token `index` nested (at any depth) inside `bracket` (None: anywhere)?"""
        node = self.parent[index]
        while node is not None:
            if node == bracket:
                return True
            node = self.parent[node] if node >= 0 else None
        return bracket is None

    def top(self, tokens: list) -> list:
        """The tokens of `tokens` at the depth of the first one."""
        return [index for index in tokens if self.parent[index] == self.parent[tokens[0]]] if tokens else []

    def kind(self, bracket):
        """What a bracket holds (see the language's _kind_*): cached."""
        if bracket is None:
            return None
        if bracket not in self.kinds:
            self.kinds[bracket] = getattr(self, "_kind_" + (self.language if self.language in (
                "python", "js", "go", "rust") else "c"))(bracket)
        return self.kinds[bracket]

    def declarations(self) -> tuple:
        """(declared names, names of classes and functions among them, [(owner, member)] assigned members) on the
        hunk's "+" lines."""
        self.found, self.owners, self.stores = set(), set(), []
        parse = getattr(self, "_" + (self.language if self.language in ("python", "js", "go", "rust", "c")
                                     else "generic"))
        for number, ids in enumerate(self.ids):
            if self.added[number] and ids and not LEAK_IMPORT.match(self.codes[number].strip()):
                parse(number, ids)  # (an import declares nothing: its names are another module's)
        keywords = KEYWORDS.get(self.language, GENERIC_KEYWORDS)
        found = {name for name in self.found if name and name.lstrip("#") not in keywords}
        return found, self.owners & found, self.stores

    def declare(self, index, owner=False):
        if self.name(index) and self.added[self.line[index]]:
            self.found.add(self.text[index])
            if owner:
                self.owners.add(self.text[index])

    def generics(self, opener):
        """Declare the type parameters of the `<...>` that token `opener` opens (`function f<TItem extends X>`,
        `fn f<Tracker: Visit>`): the names after "<" or a top-level "," (past const/in/out)."""
        if self.word(opener) != "<":
            return
        depth, expect, index = 0, False, opener
        while index < len(self.text) and self.line[index] - self.line[opener] <= 8:
            text = self.text[index]
            if text == "<":
                depth += 1
                expect = depth == 1
            elif text == ">":
                depth -= 1
                if depth <= 0:
                    return
            elif text == "," and depth == 1 and self.parent[index] == self.parent[opener]:
                expect = True
            elif expect and text not in ("const", "in", "out"):
                if self.name(index):
                    self.declare(index)
                expect = False
            index += 1

    def targets(self, tokens: list):
        """Declare the names of a Python target list (its tokens, `for` targets): names that no `.`, call or
        subscript follows and no `.` precedes."""
        for index in tokens:
            if self.name(index) and self.word(index - 1) != "." and self.word(index + 1) not in (".", "(", "["):
                self.declare(index)

    # ------------------------------------------------------------------ Python
    def _kind_python(self, bracket):
        if bracket < 0:
            after = [self.text[i] for i in self.after_close(bracket)]
            return "params" if self.text[-1 - bracket] == ")" and after[:1] in ([":"], ["->"]) else "outer"
        opener, before = self.text[bracket], [self.text[i] for i in self.before(bracket)
                                              if self.line[i] == self.line[bracket]]
        if opener == "[" and len(before) >= 2 and before[-2] in ("def", "class") and self.name(bracket - 1):
            return "params"  # type parameters
        if opener == "[" and before and (self.name(bracket - 1) and before[-1] not in PY_ADJACENT
                                         or before[-1] in (")", "]")):
            return "index"  # a subscript: `x[key] = v` stores into x
        if opener == "(":
            if len(before) >= 2 and before[-2] == "def" and self.name(bracket - 1):
                return "params"
            if before[-1:] == ["]"] and "def" in before:  # def f[T](...)
                return "params"
            if len(before) >= 2 and before[-2] == "class":
                return "bases"
            if before[-1:] == ["dict"]:
                return "dictcall"
            if before and (self.name(bracket - 1) and before[-1] not in PY_ADJACENT or before[-1] in (")", "]")):
                return "call"
        return "group"

    def _python(self, number, ids):
        if self.text[ids[0]] == "@":
            return  # a decorator
        for index in ids:
            word, kind = self.text[index], self.kind(self.parent[index])
            if word == ":=" and self.name(index - 1) and self.word(index - 2) != ".":
                self.declare(index - 1)
            elif word == "lambda":
                self._py_params_until(index, ":")
            elif word == "for":
                end = next((j for j in range(index + 1, len(self.text)) if self.text[j] == "in"
                            and self.parent[j] == self.parent[index]), None)
                if end is not None:
                    self.targets(list(range(index + 1, end)))
            elif word == "as" and self.name(index + 1):
                self.declare(index + 1)  # with / except / case ... as name
            if self.start(index):
                if kind == "params":
                    j = index
                    while self.text[j] in ("*", "**", "/") and j + 1 < len(self.text):
                        j += 1
                    if self.name(j) and self.word(j + 1) in (":", "=", ",", ")", ""):
                        self.declare(j)
                elif kind == "dictcall" and self.name(index) and self.word(index + 1) == "=":
                    self.declare(index)
                elif self.parent[index] is None:
                    self._py_statement(index)

    def _py_params_until(self, index, end_text):
        depth_parent, j = self.parent[index], index + 1
        expect = True
        while j < len(self.text) and not (self.text[j] == end_text and self.parent[j] == depth_parent):
            if self.parent[j] == depth_parent:
                if self.text[j] == ",":
                    expect = True
                elif expect and self.text[j] not in ("*", "**", "/"):
                    if self.name(j):
                        self.declare(j)
                    expect = False
            j += 1

    def _py_statement(self, index):
        tokens = self.statement(index)
        words = [self.text[i] for i in tokens]
        first = words[0] if words else ""
        if first in ("def", "async") or first == "class":
            j = tokens[1] if first != "async" else (tokens[2] if len(tokens) > 2 else None)
            if first == "async" and self.word(tokens[1] if len(tokens) > 1 else -1) != "def":
                j = None
            if j is not None and self.name(j):
                self.declare(j, owner=first == "class")  # (`Cls.attr = v` declares attr; a function's name is
            return  # often a local's too: `fork = jar.fork(); fork._cookies = ...`)
        if first == "type" and len(tokens) > 2 and self.name(tokens[1]) and self.text[tokens[2]] in ("=", "["):
            self.declare(tokens[1], owner=True)
            return
        top_words = [self.text[i] for i in self.top(tokens)]
        if first in ("case", "match") and len(words) > 1 and words[1] not in ("=", ":", ".", ",", ")") \
                and "=" not in top_words and top_words[-1:] == [":"]:  # (not `case = 1`, `match[i] = 2`)
            if first == "case":
                body = [i for i in tokens[1:] if not (self.text[i] == ":" and self.parent[i] == self.parent[index])]
                guard = next((k for k, i in enumerate(body) if self.text[i] == "if"
                              and self.parent[i] == self.parent[index]), len(body))
                for i in body[:guard]:
                    if self.name(i) and self.text[i] != "_" and self.word(i + 1) not in ("(", ".", "=") \
                            and self.word(i - 1) != ".":
                        self.declare(i)
            return
        if first in PY_ADJACENT and first not in ("match", "case", "type", "_", "print", "exec"):
            # a compound statement: its simple statement after the first top-level ":" (`else: x = 1`)
            if first in ("if", "elif", "else", "while", "for", "with", "try", "except", "finally"):
                colon = next((i for i in tokens if self.text[i] == ":" and self.parent[i] == self.parent[index]),
                             None)
                if colon is not None and colon + 1 < len(self.text) and self.line[colon + 1] == self.line[colon]:
                    self._py_statement(colon + 1)
            return
        top = self.top(tokens)
        equals = [i for i in top if self.text[i] == "="]
        if equals:
            segments, begin = [], tokens[0]
            for i in equals:
                segments.append([j for j in tokens if begin <= j < i])
                begin = i + 1
            for number, segment in enumerate(segments):
                self._py_target(segment, annotated=number == 0)
        elif augmented := [i for i in top if self.text[i] in _ASSIGN]:
            self._py_target([j for j in tokens if j < augmented[0]], annotated=False)
        elif len(top) >= 3 and self.text[top[1]] == ":" and self.name(top[0]):
            self._py_target(tokens, annotated=True)
        elif len(top) >= 5 and self.text[top[0]] in ("self", "cls") and self.text[top[1]] == "." \
                and self.text[top[3]] == ":":
            self.declare(top[2])

    def _py_target(self, segment, annotated):
        if not segment:
            return
        if annotated and len(segment) >= 2 and self.name(segment[0]) and self.text[segment[1]] == ":":
            self.declare(segment[0])
            return
        if annotated and len(segment) >= 4 and self.text[segment[0]] in ("self", "cls") \
                and self.text[segment[1]] == "." and self.text[segment[3]] == ":":
            self.declare(segment[2])
            return
        depth = self.parent[segment[0]]
        for i in segment:
            if not self.name(i) or self.text[i] in PY_ADJACENT or self.text[i] in ("self", "cls"):
                continue
            bracket = self.parent[i]
            if bracket != depth and self.kind(bracket) not in ("group",) and not (
                    bracket is not None and bracket >= 0 and self.text[bracket] == "[" and not self.name(bracket - 1)
                    and self.word(bracket - 1) not in (")", "]")):
                continue  # inside a call, subscript or display: no target
            before, after = self.word(i - 1), self.word(i + 1)
            if after in (".", "(", "["):
                continue  # read, called or subscripted: the store is to something else
            if before == "." and self.word(i - 2) in ("self", "cls") and self.word(i - 3) != ".":
                self.declare(i)
            elif before == "." and self.name(i - 2) and self.word(i - 3) != ".":
                self.stores.append((self.text[i - 2], self.text[i]))
            elif before != ".":
                self.declare(i)

    # ------------------------------------------------------------------ JavaScript / TypeScript
    def _kind_js(self, bracket):
        if bracket < 0:
            closer = self.text[-1 - bracket]
            after = [self.text[i] for i in self.after_type(bracket)]  # (a return type may span lines)
            if closer == ")" and (after[:1] in (["=>"], ["{"]) or after[:1] == [":"] and (
                    "=>" in after or after[-1:] in (["{"], [";"]))):
                return "params"  # a function, method or arrow signature (or a condition: its operands are no starts)
            return "outer" if closer == "}" else "outer-expr"
        opener = self.text[bracket]
        before = [self.text[i] for i in self.before(bracket)]
        last = before[-1] if before else ""
        after = [self.text[i] for i in (self.after_type(bracket) if opener == "(" else self.after_close(bracket))]
        parent_kind = self.kind(self.parent[bracket])
        if opener == "(":
            callee = self._before_generic(bracket - 1)  # the name before `name<T>(`
            if self.word(callee) == "?" and self.name(callee - 1) and parent_kind in ("class", "members"):
                callee -= 1  # an optional method: `onMessage?(message: T): void;`
            if last in ("catch", "function") or last == "*" and self.word(bracket - 2) == "function" or self.name(
                    callee) and (self.word(callee - 1) == "function" or self.word(callee - 1) == "*"
                                 and self.word(callee - 2) == "function"):
                return "params"
            if after[:1] == ["=>"] or after[:1] == [":"] and "=>" in after:
                return "params"
            if last in ("if", "while", "for", "switch", "with"):
                return "cond"
            if self.name(callee) and self.text[callee] in KEYWORDS["js"] and self.text[callee] not in (
                    "constructor", "get", "set", "case") and self.start(self._member_start(callee)) and (
                    after[:1] == ["{"] or parent_kind in ("class", "object", "members") and after[:1] == [":"]):
                return "params"  # a method named after a keyword (`default(): T {`, `delete(key) {`): with a body or
            # a return type (a statement `return (`, `delete (` has neither; a body may pass for a class or object)
            if self.name(callee) and (self.text[callee] not in KEYWORDS["js"] or self.text[callee] in (
                    "constructor", "get", "set")):
                member = self.start(self._member_start(callee))
                method = member and (after[:1] == ["{"] or after[:1] == [":"] and after[-1:] in (["{"], [";"], [","],
                                                                                                  ["}"]))
                if method or member and parent_kind in ("class", "members") and (
                        after[-1:] in ([";"], [","], []) or after[:1] == [":"]):
                    return "params"
                if self.match[bracket] is None and member and parent_kind in ("class", "object", "members", "outer"):
                    return "params"  # a signature whose end the hunk does not show
                return "call"
            if last in (")", "]", "?.", "`", ">"):
                return "call"
            return "group"
        if opener == "[":
            if parent_kind == "pattern" and self.text[self.parent[bracket]] == "{" and after[:1] == [":"]:
                return "computed"  # `{ [key]: binding }`: the key is an expression
            if last in ("const", "let", "var") or parent_kind in ("params", "pattern") and self.start(bracket):
                return "pattern"
            if self.start(bracket) and parent_kind in ("class", "object", "members", "outer"):
                return "computed"
            return "array"
        # "{"
        if last in ("const", "let", "var") or parent_kind in ("params", "pattern") and self.start(bracket):
            return "pattern"
        if last in ("import", "export") or last == "type" and before[-2:-1] in (["import"], ["export"]):
            return "import"  # import/export specifiers: other modules' names
        header = self.head_tokens(bracket)
        words = set(header)
        if "class" in words and last not in ("=", "(", ",", "return"):
            return "class"
        if "interface" in words or "type" in words and "=" in header:
            return "members"
        if "enum" in words:
            return "enum"
        namespace = any(before[k] in ("namespace", "module") and before[k + 1:k + 2] not in ([":"], ["."], ["="], [","])
                        and self.word(self.before(bracket)[k] - 1) not in (".", "?.") for k in range(len(before)))
        if last in (")", "=>", "else", "try", "finally", "do", "static") or namespace \
                or last == "" and parent_kind in ("block", None):
            return "block"
        if last in ("=", "(", ",", ":", "[", "?", "return", "??", "||", "&&", "...", "yield", "await", "default",
                    "export", "<", "|", "&", "!", "satisfies", "as"):
            return "object"
        if last == "" and parent_kind in ("object", "array", "call", "group", "pattern", "outer", "outer-expr"):
            return "object"
        return "block"

    def _member_start(self, name):
        """The first token of the member whose name is token `name` (past modifiers before it)."""
        index = name
        while index - 1 >= 0 and self.line[index - 1] == self.line[index] and self.text[index - 1] in (
                "public", "private", "protected", "static", "readonly", "declare", "abstract", "override",
                "accessor", "async", "get", "set", "*", "#"):
            index -= 1
        return index

    def _before_generic(self, index):
        """The token before the `<...>` that token `index` (a ">") closes on its line, else `index`; when the ">"
        starts its line, over the lines of the generic header it closes (`function f<\n  T,\n>(`, at most 8)."""
        if self.word(index) != ">":
            return index
        depth, j = 0, index
        while j >= 0 and self.line[j] == self.line[index]:
            depth += {">": 1, "<": -1, "=>": 0}.get(self.text[j], 0)
            if depth == 0:
                return j - 1
            j -= 1
        if j == index - 1 and j >= 0:  # the ">" starts its line
            depth = 1
            while j >= 0 and self.line[index] - self.line[j] <= 8:
                if self.text[j] in (")", "]", "}") and self.match[j] is not None:
                    j = self.match[j] - 1  # a bracketed part of a constraint
                    continue
                if self.text[j] in ("(", "[", "{", ")", "]", "}", ";"):
                    break
                depth += {">": 1, "<": -1}.get(self.text[j], 0)
                if depth == 0:
                    return j - 1
                j -= 1
        return index

    def _js(self, number, ids):
        for index in ids:
            word = self.text[index]
            kind = self.kind(self.parent[index])
            previous = self.word(index - 1)
            if previous in (".", "?.") and word != "prototype":
                if self.word(index + 1) == "=" and self.name(index):
                    owner = self.word(index - 2)
                    if owner in ("this", "exports") or self.word(index - 3) == "." and owner == "exports" \
                            and self.word(index - 4) == "module" or owner == "prototype":
                        self.declare(index)
                    elif self.name(index - 2) and self.word(index - 3) not in (".", "?."):
                        self.stores.append((owner, word))
                continue
            if word in ("function", "class", "interface", "enum", "namespace", "module") and self.name(index + 1):
                self.declare(index + 1, owner=word in ("function", "class"))
                self.generics(index + 2)
            elif word == "function" and self.word(index + 1) == "*" and self.name(index + 2):
                self.declare(index + 2, owner=True)
                self.generics(index + 3)
            elif word == "type" and self.name(index + 1) and self.word(index + 2) in ("=", "<") \
                    and previous not in (".", ":", "?.") and not (self.start(index) and self.word(index + 1) in (
                        ":", ",", "?")):
                self.declare(index + 1)
                self.generics(index + 2)
            elif word in ("const", "let", "var") and previous not in (".", "?."):
                self._js_bindings(index + 1)
            elif word == "=>" and self.name(index - 1) and self.word(index - 2) not in (".", "?.") \
                    and self.text[index - 1] not in KEYWORDS["js"]:
                self.declare(index - 1)  # x => ...
            elif word == "infer" and self.name(index + 1) and self.word(index + 2) not in ("(", "=", ".", "?."):
                self.declare(index + 1)  # a conditional type's type variable: `T extends Wrapped<infer TInput>`
            if self.start(index):
                self._js_element(index, kind)

    def _js_bindings(self, index):
        """Declare the bindings of a const/let/var list starting at token `index`."""
        parent = self.parent[index] if index < len(self.text) else None
        expect, angle = True, 0  # angle: the depth of a type annotation's `<...>` (its commas separate no bindings)
        j = index
        while j < len(self.text) and (self.line[j] == self.line[index] or self.parent[j] != parent):
            if self.parent[j] == parent:
                text = self.text[j]
                if text in (";",) or text in ("of", "in") and expect is False:
                    break
                if text in ("<", ">") and expect is False:
                    angle = max(0, angle + (1 if text == "<" else -1))
                elif text == "," and angle == 0:
                    expect = True
                elif expect:
                    if text in ("{", "["):
                        self._js_pattern(j)
                    elif self.name(j):
                        self.declare(j)
                    expect = False
                elif text == "=":
                    expect = None  # skip the initializer up to the next "," at this depth
            j += 1

    def _js_pattern(self, opener):
        """Declare the bindings of the destructuring pattern that `opener` opens."""
        closer = self.match[opener] if self.match[opener] is not None else len(self.text)
        j = opener + 1
        while j < closer:
            if self.parent[j] == opener and self.start(j):
                k = j
                if self.text[k] == "...":
                    k += 1
                if self.text[opener] == "{" and self.word(k) == "[" and self.match[k] is not None:
                    k = self.match[k] + 1  # a computed key `[expr]: binding`: only the binding is declared
                    if self.word(k) == ":" and self.word(k + 1) in ("{", "["):
                        self._js_pattern(k + 1)
                    elif self.word(k) == ":" and self.name(k + 1):
                        self.declare(k + 1)
                elif self.word(k) in ("{", "["):
                    self._js_pattern(k)
                elif self.name(k):
                    if self.text[opener] == "{" and self.word(k + 1) == ":":
                        if self.word(k + 2) in ("{", "["):
                            self._js_pattern(k + 2)
                        elif self.name(k + 2):
                            self.declare(k + 2)
                    else:
                        self.declare(k)
            j += 1

    def _js_element(self, index, kind):
        if kind == "computed" and self.name(index) and self.word(index + 1) == ":":
            self.declare(index)  # an index signature's key: `[key: string]: T`
            return
        if kind in ("params", "pattern"):
            j = index
            while self.word(j) in ("public", "private", "protected", "readonly", "override", "...") \
                    and (self.name(j + 1) or self.word(j + 1) in ("{", "[")):
                j += 1
            if self.word(j) == "[" and self.kind(j) == "computed":
                if self.match[j] is not None and self.word(self.match[j] + 1) == ":":
                    k = self.match[j] + 2  # `[key]: binding`
                    if self.word(k) in ("{", "["):
                        self._js_pattern(k)
                    elif self.name(k):
                        self.declare(k)
            elif self.word(j) in ("{", "["):
                self._js_pattern(j)
            elif self.name(j) and self.text[j] != "this":
                if kind == "pattern" and self.text[self.parent[index]] == "{" and self.word(j + 1) == ":":
                    if self.word(j + 2) in ("{", "["):
                        self._js_pattern(j + 2)
                    elif self.name(j + 2):
                        self.declare(j + 2)
                else:
                    self.declare(j)
            return
        if kind is None:
            kind = "outer"  # the hunk shows no enclosing bracket: a class, interface or object body as well
        if kind in ("class", "object", "members", "enum", "outer", "block"):
            j = index
            while self.word(j) in ("public", "private", "protected", "static", "readonly", "declare", "abstract",
                                   "override", "accessor", "async", "export", "default") and (
                    self.name(j + 1) or self.word(j + 1) in ("*", "#")):
                j += 1
            if self.word(j) in ("get", "set") and self.name(j + 1):
                j += 1
            if self.word(j) == "*":
                j += 1
            if not self.name(j) or (self.text[j] in KEYWORDS["js"] and self.text[j] not in (
                    "get", "set", "type", "async", "static", "declare", "abstract", "accessor", "readonly",
                    "constructor", "module", "namespace", "of", "as", "from", "is", "infer", "keyof", "unique",
                    "satisfies", "override", "package")):
                return
            after = self.word(j + 1)
            if kind == "block":  # a statement: an assignment declares (in-module names only)
                if after == "=" or after in _ASSIGN:
                    self.declare(j)
                elif after == "(" and self.kind(j + 1) == "params":
                    self.declare(j)  # a method (only in a class or object, but the block may be one)
                return
            if after == "<":
                self.generics(j + 1)
            if after in (":", "?", "!", "=", ",", ";", "}", "<", "", "extends") or after in _ASSIGN or \
                    after == "(" and (self.kind(j + 1) == "params" or kind in ("class", "members")):
                if after == "?" and self.word(j + 2) not in (":", "(", ";", ",", "") or after == "!" and \
                        self.word(j + 2) != ":":
                    return
                if kind == "outer" and after == "(" and self.kind(j + 1) != "params":
                    return  # a call statement in a block the hunk does not open
                self.declare(j, owner=self.word(j + 2) in ("function", "class"))

    # ------------------------------------------------------------------ Go
    def _kind_go(self, bracket):
        if bracket < 0:
            return "outer" if self.text[-1 - bracket] == "}" else "outer-paren"
        opener = self.text[bracket]
        before = [self.text[i] for i in self.before(bracket) if self.line[i] == self.line[bracket]]
        last = before[-1] if before else ""
        if opener == "(":
            if last == "func" or last == ")" and before[:1] == ["func"] and self._go_signature(bracket):
                return "params"
            if "func" in before and (self.name(bracket - 1) or last == "]") and self._go_signature(bracket):
                return "params"
            if last == ")" and self.kind(self.match[bracket - 1]) == "params":
                return "params"  # named results
            if last in ("const", "var"):
                return "values"
            if last == "type":
                return "types"
            if last == "import":
                return "import"
            return "call" if last and (self.name(bracket - 1) or last in (")", "]")) else "group"
        if opener == "[":
            if len(before) >= 2 and before[-2] in ("func", "type") and self.name(bracket - 1):
                return "params"  # type parameters
            return "index"
        if last == "struct":
            return "struct"
        if last == "interface":
            return "interface"
        if not before:
            return "block" if self.kind(self.parent[bracket]) in ("block", "outer", None) else "composite"
        if before[0] in ("if", "for", "switch", "select", "func", "else", "go", "defer") or last in (
                ")", "else", "{", ":"):
            if last == ")" and before[0] not in ("if", "for", "switch", "select", "func", "else", "go", "defer") \
                    and "func" not in before:
                return "composite"
            return "block"
        return "composite"

    def _go_signature(self, bracket) -> bool:
        """Is the "(" at `bracket` (after `func`) a parameter list, not a call inside a func literal's body?"""
        line_start = bracket
        while line_start > 0 and self.line[line_start - 1] == self.line[bracket]:
            line_start -= 1
        return "func" in self.text[line_start:bracket] and all(
            self.parent[i] == self.parent[bracket] or self.inside(i, self.parent[bracket])
            for i in range(line_start, bracket))

    def _go(self, number, ids):
        for index in ids:
            word, kind = self.text[index], self.kind(self.parent[index])
            if self.word(index - 1) == ".":
                continue
            if word == "func" and self.name(index + 1):
                self.declare(index + 1, owner=True)
            elif word == "func" and self.word(index + 1) == "(" and self.match[index + 1] is not None \
                    and self.name(self.match[index + 1] + 1) and self.word(self.match[index + 1] + 2) in ("(", "["):
                self.declare(self.match[index + 1] + 1)  # a method: `func (r *T) Name(`
            elif word in ("type", "package") and self.name(index + 1) and self.word(index - 1) != ".":
                self.declare(index + 1, owner=word == "type")
            elif word in ("const", "var") and self.name(index + 1):
                self._go_names(index + 1)
            elif word in (":=",):
                self._go_targets(index)
            if self.start(index):
                self._go_element(index, kind)

    def _go_names(self, index):
        """Declare `a, b, c` from token `index`."""
        j = index
        while self.name(j):
            self.declare(j)
            if self.word(j + 1) != ",":
                break
            j += 2

    def _go_targets(self, operator):
        """Declare the names before `:=` (or a statement's `=`) back to the statement start."""
        j = operator - 1
        while j >= 0 and self.line[j] == self.line[operator] and (self.name(j) or self.text[j] == ","):
            if self.name(j) and self.word(j - 1) != ".":
                self.declare(j)
            j -= 1

    def _go_element(self, index, kind):
        word, following = self.text[index], self.word(index + 1)
        kind = "outer" if kind is None else kind  # the hunk shows no enclosing bracket: any block
        if kind == "params":
            closer = self.match[self.parent[index]]
            items, item = [], []
            end = closer if closer is not None else len(self.text)
            for j in range(self.parent[index] + 1, end):
                if self.parent[j] == self.parent[index] and self.text[j] == ",":
                    items.append(item)
                    item = []
                elif self.parent[j] == self.parent[index]:
                    item.append(j)
            items.append(item)
            named = any(len([j for j in it if self.text[j] not in ("...",)]) >= 2 and self.name(it[0])
                        and self.text[it[1]] != "." for it in items if it)  # `name Type`, not `pkg.Type`
            if named and self.name(index) and word not in KEYWORDS["go"] and following != ".":
                self.declare(index)
            elif not named and closer is None and self.name(index) and self.name(index + 1):
                self.declare(index)
            return
        if kind in ("composite", "call", "group", "index", "import"):
            return
        if not self.name(index) or word in KEYWORDS["go"]:
            return
        if kind == "struct" or kind == "outer" and following not in (":", "(", ".", "=", ":=", "++", "--"):
            j = index
            names = []
            while self.name(j) and self.word(j + 1) == ",":
                names.append(j)
                j += 2
            if kind == "outer" and self.name(j) and (self.word(j + 1) in ("=", ":=") or self.word(j + 1) in _ASSIGN):
                for k in names + [j]:
                    self.declare(k)  # `a, b = ...`: a var group entry or an assignment
            elif self.name(j) and (self.name(j + 1) or self.word(j + 1) in ("*", "[", "<-", "(") and
                                 self.word(j + 1) != "(" or self.word(j + 1) in ("func", "map", "chan", "struct",
                                                                                "interface")):
                for k in names + [j]:
                    self.declare(k)
            elif kind == "outer" and self.name(j) and self.word(j + 1) in ("", ",") and self.word(j - 1) != ".":
                self.declare(j)  # a const/var group entry or a field list the hunk does not open
            return
        if kind == "interface":
            if following == "(":
                self.declare(index)
            return
        if kind in ("values", "types", "outer-paren"):
            j = index
            while self.name(j):
                self.declare(j)
                if self.word(j + 1) != ",":
                    break
                j += 2
            return
        if kind in ("block", None, "outer") and following in ("=", ",") or following in _ASSIGN:
            j = index
            names = []
            while self.name(j) and self.word(j + 1) == ",":
                names.append(j)
                j += 2
            if self.word(j + 1) in ("=",) or self.word(j + 1) in _ASSIGN or self.word(j + 1) == ":=":
                for k in names + [j]:
                    if self.name(k):
                        self.declare(k)
        if kind == "outer" and following == "(":
            self.declare(index)  # an interface method or a call of the package's own function

    # ------------------------------------------------------------------ Rust
    def _kind_rust(self, bracket):
        if bracket < 0:
            return "outer" if self.text[-1 - bracket] == "}" else "outer-paren"
        opener = self.text[bracket]
        before = [self.text[i] for i in self.before(bracket) if self.line[i] == self.line[bracket]]
        last = before[-1] if before else ""
        parent_kind = self.kind(self.parent[bracket])
        if opener == "(":
            j = bracket - 1
            if self.word(j) == ">":  # generics: back over <...>
                depth = 0
                while j >= 0 and self.line[j] == self.line[bracket]:
                    depth += {">": 1, "<": -1}.get(self.text[j], 0)
                    if depth == 0:
                        break
                    j -= 1
                j -= 1
            if self.name(j) and self.word(j - 1) == "fn":
                return "params"
            if parent_kind in ("pattern", "params") or last in ("let", "|", "for") or self.word(
                    bracket - 1) in ("Some", "Ok", "Err") and parent_kind == "pattern":
                return "pattern"
            return "call" if last and (self.name(bracket - 1) or last in (")", "]", "!", ">")) else "tuple"
        if opener == "[":
            if last in ("let",) or parent_kind in ("pattern", "params"):
                return "pattern"
            return "array"
        words = set(self.head_tokens(bracket))
        if words & {"struct", "union"} and "=" not in words:
            return "struct"
        if "enum" in words:
            return "enum"
        if "impl" in words or "trait" in words:
            return "impl"
        if "match" in words:
            return "match"
        if parent_kind == "enum" and self.start(bracket - 1) and self.name(bracket - 1):
            return "struct"  # a struct variant
        if before and before[0] in ("let",) and "=" not in words or parent_kind in ("pattern", "params"):
            return "pattern"
        if last and re.match(r"[A-Z]", last) and self.name(bracket - 1) and not (
                words & {"if", "while", "for", "match", "fn", "loop", "unsafe", "impl", "trait", "mod", "else"}):
            return "literal"
        return "block"

    def _rust(self, number, ids):
        for index in ids:
            word, kind = self.text[index], self.kind(self.parent[index])
            if self.word(index - 1) in (".", "::"):
                continue
            if word in ("fn", "struct", "enum", "union", "trait", "type", "mod") and self.name(index + 1):
                self.declare(index + 1, owner=word != "mod")
                self.generics(index + 2)
            elif word == "impl":
                self.generics(index + 1)
            elif word in ("const", "static") and self.word(index + 1) == "mut" and self.name(index + 2):
                self.declare(index + 2)
            elif word in ("const", "static") and self.name(index + 1) and self.word(index + 2) == ":":
                self.declare(index + 1)
            elif word == "macro_rules" and self.word(index + 1) == "!" and self.name(index + 2):
                self.declare(index + 2)
            elif word == "let":
                self._rust_pattern(index + 1, ("=", ";", ":", "else"))
            elif word == "for" and "impl" not in [self.text[j] for j in self.before(index)
                                                  if self.line[j] == self.line[index]]:
                end = next((j for j in range(index + 1, len(self.text)) if self.text[j] == "in"
                            and self.parent[j] == self.parent[index]), None)
                if end is not None:
                    self._rust_pattern(index + 1, ("in",))
            elif word == "=>" and kind in ("match", "outer", "block", None):
                start = index - 1
                while start > 0 and self.line[start - 1] == self.line[index] and not (
                        self.start(start) and self.parent[start] == self.parent[index]):
                    start -= 1
                guard = next((j for j in range(start, index) if self.text[j] == "if"
                              and self.parent[j] == self.parent[index]), index)
                self._rust_bindings(list(range(start, guard)))
            elif word == "|" and (self.word(index - 1) in ("(", ",", "=", "move", "return", "=>", "", "{", ";")
                                  or not self.same_line(index - 1, index)):
                end = next((j for j in range(index + 1, len(self.text)) if self.text[j] == "|"
                            and self.parent[j] == self.parent[index]), None)
                if end is not None:
                    self._rust_bindings(list(range(index + 1, end)), params=True)
            if self.start(index):
                self._rust_element(index, kind)

    def _rust_pattern(self, index, stops):
        j = index
        tokens = []
        while j < len(self.text) and not (self.text[j] in stops and self.parent[j] == self.parent[index]):
            if self.parent[j] != self.parent[index] and not self.inside(j, self.parent[index]):
                break
            tokens.append(j)
            j += 1
        self._rust_bindings(tokens)

    def _rust_bindings(self, tokens, params=False):
        """Declare the bindings of a Rust pattern (its tokens): lower-case names that no path, call, struct pattern or
        macro follows and no path or member access precedes, not the field name of a struct pattern's `field:
        binding`; with params (a parameter or closure list), each parameter's pattern before its `: Type`."""
        if not tokens:
            return
        depth, typed = self.parent[tokens[0]], False
        inner = {index for index in tokens if self.text[index] in ("(", "[", "{")}
        for index in tokens:
            if params and self.parent[index] == depth and self.text[index] in (",", ":"):
                typed = self.text[index] == ":"
                continue
            if typed or not self.name(index) or self.text[index] in KEYWORDS["rust"] \
                    or not re.match(r"[a-z_]", self.text[index]):
                continue
            if self.word(index - 1) in ("::", ".") or self.word(index + 1) in ("::", "(", "!", "{", "."):
                continue
            if self.word(index + 1) == ":" and self.parent[index] in inner and self.text[self.parent[index]] == "{":
                continue  # Struct { field: binding }
            self.declare(index)

    def _rust_element(self, index, kind):
        word, following = self.text[index], self.word(index + 1)
        kind = "outer" if kind is None else kind  # the hunk shows no enclosing bracket: any block
        if kind == "params":
            if word in ("&", "mut"):
                index += 1 if word == "mut" else (2 if self.word(index + 1) == "mut" else 1)
            end = next((j for j in range(index, len(self.text)) if self.text[j] in (":", ",", ")")
                        and self.parent[j] == self.parent[index]), index)
            self._rust_bindings(list(range(index, end + 1)), params=True)
            return
        j = index
        if self.word(j) == "pub":
            j += 1
            if self.word(j) == "(" and self.match[j] is not None:
                j = self.match[j] + 1
        if not self.name(j) or self.text[j] in KEYWORDS["rust"]:
            return
        following = self.word(j + 1)
        if kind == "struct" and following == ":" or j != index and following == ":" and self.word(j + 2) != ":":
            self.declare(j)
        elif kind == "enum" and following in (",", "(", "{", "=", "", "}"):
            self.declare(j)
        elif kind in ("outer", "outer-paren") and following == ":" and self.word(j + 2) != ":":
            self.declare(j)  # a field or parameter in a block the hunk does not open
        elif kind == "outer" and re.match(r"[A-Z]", self.text[j]) and following in (",", "(", "{", "", "=") \
                and self.word(j + 2) != "::" and not any(self.text[k] in ("=>", ";") for k in self.statement(j)):
            self.declare(j)  # an enum variant in a block the hunk does not open (not a match arm or a statement)
        elif kind in ("block", "outer") and (following == "=" or following in _ASSIGN):
            self.declare(j)

    # ------------------------------------------------------------------ the C family and the rest
    def _kind_c(self, bracket):
        if bracket < 0:
            return "outer"
        before = [self.text[i] for i in self.before(bracket) if self.line[i] == self.line[bracket]]
        after = [self.text[i] for i in self.after_close(bracket)]
        if self.text[bracket] == "(" and (after[:1] in (["{"], ["->"], ["=>"], [":"], ["throws"], ["const"])
                                          or before[-1:] in (["catch"], ["for"], ["foreach"])):
            return "params"
        return "block"

    def _c(self, number, ids):
        code = self.codes[number].strip()
        found = [name for match in GENERIC_DECLARATION.finditer(code) for name in match.groups() if name]
        if not C_TYPE_DEFINITION.search(code) and (typed := C_TYPED.match(C_ATTRIBUTES.sub("", code))) \
                and not set(re.findall(_N, typed.group(1))) & C_STATEMENT:
            found.append(typed.group(2))
        for index in ids:
            if self.kind(self.parent[index]) == "params" and self.start(index):
                bracket = self.parent[index]
                end, closer = index, self.match[bracket] if bracket >= 0 else -1 - bracket
                while end < len(self.text) and end != closer and not (
                        self.text[end] in (",", "=", ":") and self.parent[end] == self.parent[index]):
                    end += 1
                if end - 1 > index and self.name(end - 1):
                    found.append(self.text[end - 1])
            elif (self.text[index] == "->" and self.path.endswith(".java") or self.text[index] == "=>"
                  and self.path.endswith((".cs", ".dart"))) and self.name(index - 1):
                found.append(self.text[index - 1])  # a lambda's parameter (C and C++ `->` is member access)
        for name in found:
            name = name.strip().lstrip("~")
            if name:
                self.found.add(name)

    def _generic(self, number, ids):
        code = self.codes[number].strip()
        self.found.update(name for match in GENERIC_DECLARATION.finditer(code) for name in match.groups() if name)
        if match := re.match(rf"^(?:local\s+|val\s+|var\s+|let\s+)?({_N})\s*(?::=|=(?!=|>))", code):
            self.found.add(match.group(1))


# ------------------------------------------------- c9f84ee0's reading (declared names v2), FROZEN: do not edit
# v4 (hengyuf 2026-10-21, main's decision after the verifier refuted v3): declared_names is monotonic over the
# installed v2 (task_builder.py md5 c9f84ee0a3b583d088d0ea7d5d37a182) by construction. It unions v2's declared names,
# computed here by v2's own code, and _code_hunks adds v2's readings to v3's (for _added_items' names too). These four
# definitions are c9f84ee0's _py_scan, _code_hunks, declared_names and _Hunk verbatim, renamed _py_scan_v2,
# _code_hunks_v2, _declared_names_v2 and _HunkV2 (wherever their text names them); with the module-level names they
# use, none changed since c9f84ee0, they compute what c9f84ee0 computes. tests/test_gates_1020.py pins the source of
# both (test_v2_reading_is_frozen): change neither; change the current reading (_code_hunks_v3, _Hunk) instead.
def _py_scan_v2(text: str, quote):
    """(code, quote after) of one Python line read from inside the string that `quote` opened (None: from code; a
    triple quote, or a one-quote string that a backslash continues): strings blanked with their prefixes, the contents
    of strings dropped, the comment cut."""
    out, index, length = [], 0, len(text)
    while index < length:
        if quote:
            end = text.find(quote, index)
            while end > 0 and _escaped(text, end):
                end = text.find(quote, end + 1)
            if end < 0:
                continued = _escaped(text, length)
                return "".join(out), (quote if len(quote) == 3 or continued else None)
            out.append(" 0 ")  # (a value token: the line goes on)
            index, quote = end + len(quote), None
            continue
        special = _PY_SPECIAL.search(text, index)
        if not special:
            out.append(text[index:])
            break
        out.append(text[index:special.start()])
        index = special.start()
        if text[index] == "#":
            break
        code = "".join(out)
        prefix = re.search(r"(?<![\w$])[rRbBuUfFtT]{1,2}$", code[-3:]) if code[-3:] else None
        if prefix and prefix.group(0).lower() in ("r", "b", "u", "f", "t", "rb", "br", "fr", "rf", "tr", "rt") \
                and not re.search(r"[\w$]$", code[:len(code) - len(prefix.group(0))]):
            out = [code[:len(code) - len(prefix.group(0))]]
        quote = text[index] * 3 if text.startswith(text[index] * 3, index) else text[index]
        index += len(quote)
    return "".join(out), (quote if quote and (len(quote) == 3 or _escaped(text, length)) else None)


def _code_hunks_v2(solution_patch: str) -> list:
    """[(path, header, lines, variants)] per hunk of the solution's code files (CODE_FILE): lines [(added, text)] is
    the hunk's new side ("+" and context lines), header git's hunk header text, variants its readings [code of every
    line] ("" where a reading sees no code). Python: every start state (code, inside a \"\"\" or a ''' string), each
    fence on every line toggling it (_py_scan_v2), that takes no prose for code (_py_prose), so the true reading is always
    among them. Other languages, and Python when every start state takes prose for code: the "+"-line docstring state
    of _added_items' literals and leak_tokens (context lines with a fence blank; Python prose lines blank)."""
    hunks, path, header, lines, in_doc = [], None, "", [], False

    def flush():
        nonlocal in_doc
        if not path or not any(added for added, _ in lines):
            lines.clear()
            return
        python = bool(re.search(r"\.(py|pyi)$", path))
        plus = []
        for added, raw in lines:
            text = raw.strip()
            fences = len(DOC_FENCE.findall(text))
            if added and (in_doc or fences):
                in_doc = in_doc != (fences % 2 == 1)
                plus.append("")
            elif fences:
                plus.append("")
            else:
                plus.append(_py_scan_v2(raw, None)[0] if python else _plain_code(path, raw))
        variants = []
        if python:  # every start state that takes no prose for code; the "+"-line state when none is left
            for start in (None, '"""', "'''"):
                quote, codes = start, []
                for _, raw in lines:
                    code, quote = _py_scan_v2(raw, quote)
                    if _py_prose(code):
                        break
                    codes.append(code)
                else:
                    if codes not in variants:
                        variants.append(codes)
        if not variants:
            variants = [[code if not (python and _py_prose(code)) else "" for code in plus]]
        hunks.append((path, header, list(lines), variants))
        lines.clear()

    for line in solution_patch.splitlines():
        if line.startswith("+++ "):
            flush()
            new = line[6:] if line.startswith("+++ b/") else ""
            path, header, in_doc = (new if CODE_FILE.search(new) else None), "", False
        elif line.startswith("diff --git "):
            flush()
            path = None
        elif line.startswith("@@"):
            flush()
            header = line.split("@@", 2)[2].strip() if line.count("@@") >= 2 else ""
        elif path and line[:1] in ("+", " ", ""):
            lines.append((line[:1] == "+", line[1:]))
    flush()
    return hunks


def _declared_names_v2(solution_patch: str) -> set:
    """The names that the solution's added code lines declare (see DECLARING), in every reading of every hunk
    (_code_hunks_v2): the "+" and context lines show the enclosing brackets and blocks. Members assigned to a class
    (`Pool.defaults = ...`: a class or JS function that the solution declares, or a capitalized name) count too."""
    names, owners, stores = set(), set(), []
    for path, header, lines, variants in _code_hunks_v2(solution_patch):
        language = next((name for name, regex in DECLARING if regex.search(path)), "generic")
        for codes in variants:
            found = _HunkV2(language, header, lines, codes, path).declarations()
            names |= found[0]
            owners |= found[1]
            stores += found[2]
    names |= {member for owner, member in stores if owner in owners or owner[:1].isupper()}
    return {name.lstrip("#") for name in names if len(name.lstrip("#")) > 0}


class _HunkV2:
    """One reading of one hunk as tokens with their brackets: parent[i] is the innermost bracket around token i (an
    opening token's index, or -1 - j for a bracket that opens before the hunk and that token j closes, or None)."""

    def __init__(self, language: str, header: str, lines: list, codes: list, path: str = ""):
        self.language, self.header, self.added, self.path = language, header, [added for added, _ in lines], path
        self.codes, self.text, self.line, self.ids = codes, [], [], []
        if language == "rust":
            codes = [re.sub(r"'(?:_|[A-Za-z_]\w*)(?![\w'])", "", code) for code in codes]
        for number, code in enumerate(codes):
            ids = []
            if len(code) <= 400:  # (a minified line declares nothing)
                for match in _TOKEN.finditer(code):
                    ids.append(len(self.text))
                    self.text.append(match.group(0))
                    self.line.append(number)
            self.ids.append(ids)
        count = len(self.text)
        self.match, self.parent, stack = [None] * count, [None] * count, []
        unopened = []
        for index, text in enumerate(self.text):
            if text in ")]}":
                if stack:
                    opener = stack.pop()
                    self.match[opener], self.match[index] = index, opener
                else:
                    unopened.append(index)
                self.parent[index] = stack[-1] if stack else None
            else:
                self.parent[index] = stack[-1] if stack else None
                if text in "([{":
                    stack.append(index)
        outer, unopened = None, set(unopened)  # the bracket opened before the hunk that the next unopened closer closes
        for index in range(count - 1, -1, -1):
            if self.parent[index] is None:
                self.parent[index] = outer
            if index in unopened:
                outer = -1 - index
        self.kinds = {}

    # ------------------------------------------------------------------ helpers
    def name(self, index) -> bool:
        return 0 <= index < len(self.text) and bool(_NAME.match(self.text[index])) and not self.text[index][0].isdigit()

    def word(self, index) -> str:
        return self.text[index] if 0 <= index < len(self.text) else ""

    def same_line(self, a, b) -> bool:
        return 0 <= a < len(self.text) and 0 <= b < len(self.text) and self.line[a] == self.line[b]

    def before(self, index) -> list:
        """The tokens before `index` on its line (the previous code line's when it starts one)."""
        line = self.line[index]
        start = index
        while start > 0 and self.line[start - 1] == line:
            start -= 1
        if start == index and start > 0:
            previous = self.line[start - 1]
            while start > 0 and self.line[start - 1] == previous:
                start -= 1
        return list(range(start, index))

    def after_close(self, opener) -> list:
        """The tokens after the bracket that closes `opener`, on the closer's line ([] when it does not close here)."""
        closer = self.match[opener] if opener is not None and opener >= 0 else (-1 - opener if opener is not None
                                                                                  else None)
        if closer is None:
            return []
        end = closer + 1
        while end < len(self.text) and self.line[end] == self.line[closer]:
            end += 1
        return list(range(closer + 1, end))

    def start(self, index) -> bool:
        """Does token `index` start an element of its bracket or block (a statement, member, parameter, entry)?"""
        if index == 0:
            return True
        parent, previous = self.parent[index], self.text[index - 1]
        if parent is not None and parent >= 0 and index - 1 == parent:
            return True
        if previous in (",", ";") and self.parent[index - 1] == parent:
            return previous == ";" or not self.in_generic(index - 1)
        if self.line[index - 1] == self.line[index]:
            return False
        if self.language == "python":
            return parent is None and previous != "\\"
        if previous in ("{", "}", ";", ")", "]", "++", "--") or previous in ("else", "do", "try", "finally"):
            return True
        if previous == ">":  # a type's generic arguments end the line: a member or field ends there
            return self.kind(parent) not in ("block", "call", "group", "cond", "array", "composite", "literal")
        if self.name(index - 1) and previous not in (
                "return", "new", "typeof", "await", "yield", "case", "in", "of", "instanceof", "as", "extends",
                "implements", "throw", "delete", "keyof", "satisfies", "is", "and", "or", "not", "if", "elif",
                "while", "for", "switch", "match", "where", "impl", "dyn", "mut", "ref", "let", "const", "var", "func",
                "type", "struct", "enum", "interface", "fn", "pub", "static", "export", "default", "async"):
            return True
        return bool(re.match(r"\d", previous))

    def in_generic(self, comma) -> bool:
        """Is the "," at `comma` inside a `<...>` that opens before it on its line, at its bracket depth (a type's
        generic arguments, `Map<number, Promise<void>>`, whose "," starts no element)?"""
        if self.language not in ("js", "rust", "c"):
            return False
        depth, index = 0, comma - 1
        while index >= 0 and self.line[index] == self.line[comma]:
            if self.parent[index] == self.parent[comma]:
                if self.text[index] == ">":
                    depth += 1
                elif self.text[index] == "<":
                    if depth == 0:
                        return True
                    depth -= 1
                elif self.text[index] in (";", "{", "}"):
                    break
            elif index == self.parent[comma]:
                break
            index -= 1
        return False

    def head_tokens(self, bracket) -> list:
        """The texts of the tokens before `bracket` back to the previous statement's end (`;`, `{` or `}` around it),
        over at most three lines: a declaration header that may span lines (`class X<\n T,\n> extends Y {`)."""
        out, index = [], bracket - 1
        while index >= 0 and self.line[bracket] - self.line[index] <= 3:
            if self.parent[index] == self.parent[bracket] and self.text[index] in (";", "{", "}"):
                break
            if index == self.parent[bracket]:
                break
            out.append(self.text[index])
            index -= 1
        return out[::-1]

    def statement(self, index) -> list:
        """The tokens of the element that starts at `index`, inside its bracket, to the end of its line (a
        statement outside brackets runs on over its commas: `a, b = ...`)."""
        parent, out, end = self.parent[index], [], index
        while end < len(self.text) and self.line[end] == self.line[index]:
            if self.parent[end] == parent and (self.text[end] == ";" or self.text[end] == "," and parent is not None) \
                    and end != index:
                break
            if self.parent[end] != parent and end != index and not self.inside(end, parent):
                break
            out.append(end)
            end += 1
        return out

    def inside(self, index, bracket) -> bool:
        """Is token `index` nested (at any depth) inside `bracket` (None: anywhere)?"""
        node = self.parent[index]
        while node is not None:
            if node == bracket:
                return True
            node = self.parent[node] if node >= 0 else None
        return bracket is None

    def top(self, tokens: list) -> list:
        """The tokens of `tokens` at the depth of the first one."""
        return [index for index in tokens if self.parent[index] == self.parent[tokens[0]]] if tokens else []

    def kind(self, bracket):
        """What a bracket holds (see the language's _kind_*): cached."""
        if bracket is None:
            return None
        if bracket not in self.kinds:
            self.kinds[bracket] = getattr(self, "_kind_" + (self.language if self.language in (
                "python", "js", "go", "rust") else "c"))(bracket)
        return self.kinds[bracket]

    def declarations(self) -> tuple:
        """(declared names, names of classes and functions among them, [(owner, member)] assigned members) on the
        hunk's "+" lines."""
        self.found, self.owners, self.stores = set(), set(), []
        parse = getattr(self, "_" + (self.language if self.language in ("python", "js", "go", "rust", "c")
                                     else "generic"))
        for number, ids in enumerate(self.ids):
            if self.added[number] and ids and not LEAK_IMPORT.match(self.codes[number].strip()):
                parse(number, ids)  # (an import declares nothing: its names are another module's)
        keywords = KEYWORDS.get(self.language, GENERIC_KEYWORDS)
        found = {name for name in self.found if name and name.lstrip("#") not in keywords}
        return found, self.owners & found, self.stores

    def declare(self, index, owner=False):
        if self.name(index) and self.added[self.line[index]]:
            self.found.add(self.text[index])
            if owner:
                self.owners.add(self.text[index])

    def generics(self, opener):
        """Declare the type parameters of the `<...>` that token `opener` opens (`function f<TItem extends X>`,
        `fn f<Tracker: Visit>`): the names after "<" or a top-level "," (past const/in/out)."""
        if self.word(opener) != "<":
            return
        depth, expect, index = 0, False, opener
        while index < len(self.text) and self.line[index] - self.line[opener] <= 8:
            text = self.text[index]
            if text == "<":
                depth += 1
                expect = depth == 1
            elif text == ">":
                depth -= 1
                if depth <= 0:
                    return
            elif text == "," and depth == 1 and self.parent[index] == self.parent[opener]:
                expect = True
            elif expect and text not in ("const", "in", "out"):
                if self.name(index):
                    self.declare(index)
                expect = False
            index += 1

    def targets(self, tokens: list):
        """Declare the names of a Python target list (its tokens, `for` targets): names that no `.`, call or
        subscript follows and no `.` precedes."""
        for index in tokens:
            if self.name(index) and self.word(index - 1) != "." and self.word(index + 1) not in (".", "(", "["):
                self.declare(index)

    # ------------------------------------------------------------------ Python
    def _kind_python(self, bracket):
        if bracket < 0:
            after = [self.text[i] for i in self.after_close(bracket)]
            return "params" if self.text[-1 - bracket] == ")" and after[:1] in ([":"], ["->"]) else "outer"
        opener, before = self.text[bracket], [self.text[i] for i in self.before(bracket)
                                              if self.line[i] == self.line[bracket]]
        if opener == "[" and len(before) >= 2 and before[-2] in ("def", "class") and self.name(bracket - 1):
            return "params"  # type parameters
        if opener == "[" and before and (self.name(bracket - 1) and before[-1] not in PY_ADJACENT
                                         or before[-1] in (")", "]")):
            return "index"  # a subscript: `x[key] = v` stores into x
        if opener == "(":
            if len(before) >= 2 and before[-2] == "def" and self.name(bracket - 1):
                return "params"
            if before[-1:] == ["]"] and "def" in before:  # def f[T](...)
                return "params"
            if len(before) >= 2 and before[-2] == "class":
                return "bases"
            if before[-1:] == ["dict"]:
                return "dictcall"
            if before and (self.name(bracket - 1) and before[-1] not in PY_ADJACENT or before[-1] in (")", "]")):
                return "call"
        return "group"

    def _python(self, number, ids):
        if self.text[ids[0]] == "@":
            return  # a decorator
        for index in ids:
            word, kind = self.text[index], self.kind(self.parent[index])
            if word == ":=" and self.name(index - 1) and self.word(index - 2) != ".":
                self.declare(index - 1)
            elif word == "lambda":
                self._py_params_until(index, ":")
            elif word == "for":
                end = next((j for j in range(index + 1, len(self.text)) if self.text[j] == "in"
                            and self.parent[j] == self.parent[index]), None)
                if end is not None:
                    self.targets(list(range(index + 1, end)))
            elif word == "as" and self.name(index + 1):
                self.declare(index + 1)  # with / except / case ... as name
            if self.start(index):
                if kind == "params":
                    j = index
                    while self.text[j] in ("*", "**", "/") and j + 1 < len(self.text):
                        j += 1
                    if self.name(j) and self.word(j + 1) in (":", "=", ",", ")", ""):
                        self.declare(j)
                elif kind == "dictcall" and self.name(index) and self.word(index + 1) == "=":
                    self.declare(index)
                elif self.parent[index] is None:
                    self._py_statement(index)

    def _py_params_until(self, index, end_text):
        depth_parent, j = self.parent[index], index + 1
        expect = True
        while j < len(self.text) and not (self.text[j] == end_text and self.parent[j] == depth_parent):
            if self.parent[j] == depth_parent:
                if self.text[j] == ",":
                    expect = True
                elif expect and self.text[j] not in ("*", "**", "/"):
                    if self.name(j):
                        self.declare(j)
                    expect = False
            j += 1

    def _py_statement(self, index):
        tokens = self.statement(index)
        words = [self.text[i] for i in tokens]
        first = words[0] if words else ""
        if first in ("def", "async") or first == "class":
            j = tokens[1] if first != "async" else (tokens[2] if len(tokens) > 2 else None)
            if first == "async" and self.word(tokens[1] if len(tokens) > 1 else -1) != "def":
                j = None
            if j is not None and self.name(j):
                self.declare(j, owner=first == "class")  # (`Cls.attr = v` declares attr; a function's name is
            return  # often a local's too: `fork = jar.fork(); fork._cookies = ...`)
        if first == "type" and len(tokens) > 2 and self.name(tokens[1]) and self.text[tokens[2]] in ("=", "["):
            self.declare(tokens[1], owner=True)
            return
        top_words = [self.text[i] for i in self.top(tokens)]
        if first in ("case", "match") and len(words) > 1 and words[1] not in ("=", ":", ".", ",", ")") \
                and "=" not in top_words and top_words[-1:] == [":"]:  # (not `case = 1`, `match[i] = 2`)
            if first == "case":
                body = [i for i in tokens[1:] if not (self.text[i] == ":" and self.parent[i] == self.parent[index])]
                guard = next((k for k, i in enumerate(body) if self.text[i] == "if"
                              and self.parent[i] == self.parent[index]), len(body))
                for i in body[:guard]:
                    if self.name(i) and self.text[i] != "_" and self.word(i + 1) not in ("(", ".", "=") \
                            and self.word(i - 1) != ".":
                        self.declare(i)
            return
        if first in PY_ADJACENT and first not in ("match", "case", "type", "_", "print", "exec"):
            # a compound statement: its simple statement after the first top-level ":" (`else: x = 1`)
            if first in ("if", "elif", "else", "while", "for", "with", "try", "except", "finally"):
                colon = next((i for i in tokens if self.text[i] == ":" and self.parent[i] == self.parent[index]),
                             None)
                if colon is not None and colon + 1 < len(self.text) and self.line[colon + 1] == self.line[colon]:
                    self._py_statement(colon + 1)
            return
        top = self.top(tokens)
        equals = [i for i in top if self.text[i] == "="]
        if equals:
            segments, begin = [], tokens[0]
            for i in equals:
                segments.append([j for j in tokens if begin <= j < i])
                begin = i + 1
            for number, segment in enumerate(segments):
                self._py_target(segment, annotated=number == 0)
        elif augmented := [i for i in top if self.text[i] in _ASSIGN]:
            self._py_target([j for j in tokens if j < augmented[0]], annotated=False)
        elif len(top) >= 3 and self.text[top[1]] == ":" and self.name(top[0]):
            self._py_target(tokens, annotated=True)
        elif len(top) >= 5 and self.text[top[0]] in ("self", "cls") and self.text[top[1]] == "." \
                and self.text[top[3]] == ":":
            self.declare(top[2])

    def _py_target(self, segment, annotated):
        if not segment:
            return
        if annotated and len(segment) >= 2 and self.name(segment[0]) and self.text[segment[1]] == ":":
            self.declare(segment[0])
            return
        if annotated and len(segment) >= 4 and self.text[segment[0]] in ("self", "cls") \
                and self.text[segment[1]] == "." and self.text[segment[3]] == ":":
            self.declare(segment[2])
            return
        depth = self.parent[segment[0]]
        for i in segment:
            if not self.name(i) or self.text[i] in PY_ADJACENT or self.text[i] in ("self", "cls"):
                continue
            bracket = self.parent[i]
            if bracket != depth and self.kind(bracket) not in ("group",) and not (
                    bracket is not None and bracket >= 0 and self.text[bracket] == "[" and not self.name(bracket - 1)
                    and self.word(bracket - 1) not in (")", "]")):
                continue  # inside a call, subscript or display: no target
            before, after = self.word(i - 1), self.word(i + 1)
            if after in (".", "(", "["):
                continue  # read, called or subscripted: the store is to something else
            if before == "." and self.word(i - 2) in ("self", "cls") and self.word(i - 3) != ".":
                self.declare(i)
            elif before == "." and self.name(i - 2) and self.word(i - 3) != ".":
                self.stores.append((self.text[i - 2], self.text[i]))
            elif before != ".":
                self.declare(i)

    # ------------------------------------------------------------------ JavaScript / TypeScript
    def _kind_js(self, bracket):
        if bracket < 0:
            closer = self.text[-1 - bracket]
            after = [self.text[i] for i in self.after_close(bracket)]
            if closer == ")" and (after[:1] in (["=>"], ["{"]) or after[:1] == [":"] and (
                    "=>" in after or after[-1:] in (["{"], [";"]))):
                return "params"  # a function, method or arrow signature (or a condition: its operands are no starts)
            return "outer" if closer == "}" else "outer-expr"
        opener = self.text[bracket]
        before = [self.text[i] for i in self.before(bracket)]
        last = before[-1] if before else ""
        after = [self.text[i] for i in self.after_close(bracket)]
        parent_kind = self.kind(self.parent[bracket])
        if opener == "(":
            callee = self._before_generic(bracket - 1)  # the name before `name<T>(`
            if last in ("catch", "function") or last == "*" and self.word(bracket - 2) == "function" or self.name(
                    callee) and (self.word(callee - 1) == "function" or self.word(callee - 1) == "*"
                                 and self.word(callee - 2) == "function"):
                return "params"
            if after[:1] == ["=>"] or after[:1] == [":"] and "=>" in after:
                return "params"
            if last in ("if", "while", "for", "switch", "with"):
                return "cond"
            if self.name(callee) and (self.text[callee] not in KEYWORDS["js"] or self.text[callee] in (
                    "constructor", "get", "set")):
                member = self.start(self._member_start(callee))
                method = member and (after[:1] == ["{"] or after[:1] == [":"] and after[-1:] in (["{"], [";"], [","],
                                                                                                  ["}"]))
                if method or member and parent_kind in ("class", "members") and after[-1:] in ([";"], [","], []):
                    return "params"
                if self.match[bracket] is None and member and parent_kind in ("class", "object", "members", "outer"):
                    return "params"  # a signature whose end the hunk does not show
                return "call"
            if last in (")", "]", "?.", "`", ">"):
                return "call"
            return "group"
        if opener == "[":
            if parent_kind == "pattern" and self.text[self.parent[bracket]] == "{" and after[:1] == [":"]:
                return "computed"  # `{ [key]: binding }`: the key is an expression
            if last in ("const", "let", "var") or parent_kind in ("params", "pattern") and self.start(bracket):
                return "pattern"
            if self.start(bracket) and parent_kind in ("class", "object", "members", "outer"):
                return "computed"
            return "array"
        # "{"
        if last in ("const", "let", "var") or parent_kind in ("params", "pattern") and self.start(bracket):
            return "pattern"
        if last in ("import", "export") or last == "type" and before[-2:-1] in (["import"], ["export"]):
            return "import"  # import/export specifiers: other modules' names
        header = self.head_tokens(bracket)
        words = set(header)
        if "class" in words and last not in ("=", "(", ",", "return"):
            return "class"
        if "interface" in words or "type" in words and "=" in header:
            return "members"
        if "enum" in words:
            return "enum"
        namespace = any(before[k] in ("namespace", "module") and before[k + 1:k + 2] not in ([":"], ["."], ["="], [","])
                        and self.word(self.before(bracket)[k] - 1) not in (".", "?.") for k in range(len(before)))
        if last in (")", "=>", "else", "try", "finally", "do", "static") or namespace \
                or last == "" and parent_kind in ("block", None):
            return "block"
        if last in ("=", "(", ",", ":", "[", "?", "return", "??", "||", "&&", "...", "yield", "await", "default",
                    "export", "<", "|", "&", "!", "satisfies", "as"):
            return "object"
        if last == "" and parent_kind in ("object", "array", "call", "group", "pattern", "outer", "outer-expr"):
            return "object"
        return "block"

    def _member_start(self, name):
        """The first token of the member whose name is token `name` (past modifiers before it)."""
        index = name
        while index - 1 >= 0 and self.line[index - 1] == self.line[index] and self.text[index - 1] in (
                "public", "private", "protected", "static", "readonly", "declare", "abstract", "override",
                "accessor", "async", "get", "set", "*", "#"):
            index -= 1
        return index

    def _before_generic(self, index):
        """The token before the `<...>` that token `index` (a ">") closes on its line, else `index`."""
        if self.word(index) != ">":
            return index
        depth, j = 0, index
        while j >= 0 and self.line[j] == self.line[index]:
            depth += {">": 1, "<": -1, "=>": 0}.get(self.text[j], 0)
            if depth == 0:
                return j - 1
            j -= 1
        return index

    def _js(self, number, ids):
        for index in ids:
            word = self.text[index]
            kind = self.kind(self.parent[index])
            previous = self.word(index - 1)
            if previous in (".", "?.") and word != "prototype":
                if self.word(index + 1) == "=" and self.name(index):
                    owner = self.word(index - 2)
                    if owner in ("this", "exports") or self.word(index - 3) == "." and owner == "exports" \
                            and self.word(index - 4) == "module" or owner == "prototype":
                        self.declare(index)
                    elif self.name(index - 2) and self.word(index - 3) not in (".", "?."):
                        self.stores.append((owner, word))
                continue
            if word in ("function", "class", "interface", "enum", "namespace", "module") and self.name(index + 1):
                self.declare(index + 1, owner=word in ("function", "class"))
                self.generics(index + 2)
            elif word == "function" and self.word(index + 1) == "*" and self.name(index + 2):
                self.declare(index + 2, owner=True)
                self.generics(index + 3)
            elif word == "type" and self.name(index + 1) and self.word(index + 2) in ("=", "<") \
                    and previous not in (".", ":", "?.") and not (self.start(index) and self.word(index + 1) in (
                        ":", ",", "?")):
                self.declare(index + 1)
                self.generics(index + 2)
            elif word in ("const", "let", "var") and previous not in (".", "?."):
                self._js_bindings(index + 1)
            elif word == "=>" and self.name(index - 1) and self.word(index - 2) not in (".", "?.") \
                    and self.text[index - 1] not in KEYWORDS["js"]:
                self.declare(index - 1)  # x => ...
            if self.start(index):
                self._js_element(index, kind)

    def _js_bindings(self, index):
        """Declare the bindings of a const/let/var list starting at token `index`."""
        parent = self.parent[index] if index < len(self.text) else None
        expect, angle = True, 0  # angle: the depth of a type annotation's `<...>` (its commas separate no bindings)
        j = index
        while j < len(self.text) and (self.line[j] == self.line[index] or self.parent[j] != parent):
            if self.parent[j] == parent:
                text = self.text[j]
                if text in (";",) or text in ("of", "in") and expect is False:
                    break
                if text in ("<", ">") and expect is False:
                    angle = max(0, angle + (1 if text == "<" else -1))
                elif text == "," and angle == 0:
                    expect = True
                elif expect:
                    if text in ("{", "["):
                        self._js_pattern(j)
                    elif self.name(j):
                        self.declare(j)
                    expect = False
                elif text == "=":
                    expect = None  # skip the initializer up to the next "," at this depth
            j += 1

    def _js_pattern(self, opener):
        """Declare the bindings of the destructuring pattern that `opener` opens."""
        closer = self.match[opener] if self.match[opener] is not None else len(self.text)
        j = opener + 1
        while j < closer:
            if self.parent[j] == opener and self.start(j):
                k = j
                if self.text[k] == "...":
                    k += 1
                if self.text[opener] == "{" and self.word(k) == "[" and self.match[k] is not None:
                    k = self.match[k] + 1  # a computed key `[expr]: binding`: only the binding is declared
                    if self.word(k) == ":" and self.word(k + 1) in ("{", "["):
                        self._js_pattern(k + 1)
                    elif self.word(k) == ":" and self.name(k + 1):
                        self.declare(k + 1)
                elif self.word(k) in ("{", "["):
                    self._js_pattern(k)
                elif self.name(k):
                    if self.text[opener] == "{" and self.word(k + 1) == ":":
                        if self.word(k + 2) in ("{", "["):
                            self._js_pattern(k + 2)
                        elif self.name(k + 2):
                            self.declare(k + 2)
                    else:
                        self.declare(k)
            j += 1

    def _js_element(self, index, kind):
        if kind == "computed" and self.name(index) and self.word(index + 1) == ":":
            self.declare(index)  # an index signature's key: `[key: string]: T`
            return
        if kind in ("params", "pattern"):
            j = index
            while self.word(j) in ("public", "private", "protected", "readonly", "override", "...") \
                    and (self.name(j + 1) or self.word(j + 1) in ("{", "[")):
                j += 1
            if self.word(j) == "[" and self.kind(j) == "computed":
                if self.match[j] is not None and self.word(self.match[j] + 1) == ":":
                    k = self.match[j] + 2  # `[key]: binding`
                    if self.word(k) in ("{", "["):
                        self._js_pattern(k)
                    elif self.name(k):
                        self.declare(k)
            elif self.word(j) in ("{", "["):
                self._js_pattern(j)
            elif self.name(j) and self.text[j] != "this":
                if kind == "pattern" and self.text[self.parent[index]] == "{" and self.word(j + 1) == ":":
                    if self.word(j + 2) in ("{", "["):
                        self._js_pattern(j + 2)
                    elif self.name(j + 2):
                        self.declare(j + 2)
                else:
                    self.declare(j)
            return
        if kind is None:
            kind = "outer"  # the hunk shows no enclosing bracket: a class, interface or object body as well
        if kind in ("class", "object", "members", "enum", "outer", "block"):
            j = index
            while self.word(j) in ("public", "private", "protected", "static", "readonly", "declare", "abstract",
                                   "override", "accessor", "async", "export", "default") and (
                    self.name(j + 1) or self.word(j + 1) in ("*", "#")):
                j += 1
            if self.word(j) in ("get", "set") and self.name(j + 1):
                j += 1
            if self.word(j) == "*":
                j += 1
            if not self.name(j) or (self.text[j] in KEYWORDS["js"] and self.text[j] not in (
                    "get", "set", "type", "async", "static", "declare", "abstract", "accessor", "readonly",
                    "constructor", "module", "namespace", "of", "as", "from", "is", "infer", "keyof", "unique",
                    "satisfies", "override", "package")):
                return
            after = self.word(j + 1)
            if kind == "block":  # a statement: an assignment declares (in-module names only)
                if after == "=" or after in _ASSIGN:
                    self.declare(j)
                elif after == "(" and self.kind(j + 1) == "params":
                    self.declare(j)  # a method (only in a class or object, but the block may be one)
                return
            if after == "<":
                self.generics(j + 1)
            if after in (":", "?", "!", "=", ",", ";", "}", "<", "", "extends") or after in _ASSIGN or \
                    after == "(" and (self.kind(j + 1) == "params" or kind in ("class", "members")):
                if after == "?" and self.word(j + 2) not in (":", "(", ";", ",", "") or after == "!" and \
                        self.word(j + 2) != ":":
                    return
                if kind == "outer" and after == "(" and self.kind(j + 1) != "params":
                    return  # a call statement in a block the hunk does not open
                self.declare(j, owner=self.word(j + 2) in ("function", "class"))

    # ------------------------------------------------------------------ Go
    def _kind_go(self, bracket):
        if bracket < 0:
            return "outer" if self.text[-1 - bracket] == "}" else "outer-paren"
        opener = self.text[bracket]
        before = [self.text[i] for i in self.before(bracket) if self.line[i] == self.line[bracket]]
        last = before[-1] if before else ""
        if opener == "(":
            if last == "func" or last == ")" and before[:1] == ["func"] and self._go_signature(bracket):
                return "params"
            if "func" in before and (self.name(bracket - 1) or last == "]") and self._go_signature(bracket):
                return "params"
            if last == ")" and self.kind(self.match[bracket - 1]) == "params":
                return "params"  # named results
            if last in ("const", "var"):
                return "values"
            if last == "type":
                return "types"
            if last == "import":
                return "import"
            return "call" if last and (self.name(bracket - 1) or last in (")", "]")) else "group"
        if opener == "[":
            if len(before) >= 2 and before[-2] in ("func", "type") and self.name(bracket - 1):
                return "params"  # type parameters
            return "index"
        if last == "struct":
            return "struct"
        if last == "interface":
            return "interface"
        if not before:
            return "block" if self.kind(self.parent[bracket]) in ("block", "outer", None) else "composite"
        if before[0] in ("if", "for", "switch", "select", "func", "else", "go", "defer") or last in (
                ")", "else", "{", ":"):
            if last == ")" and before[0] not in ("if", "for", "switch", "select", "func", "else", "go", "defer") \
                    and "func" not in before:
                return "composite"
            return "block"
        return "composite"

    def _go_signature(self, bracket) -> bool:
        """Is the "(" at `bracket` (after `func`) a parameter list, not a call inside a func literal's body?"""
        line_start = bracket
        while line_start > 0 and self.line[line_start - 1] == self.line[bracket]:
            line_start -= 1
        return "func" in self.text[line_start:bracket] and all(
            self.parent[i] == self.parent[bracket] or self.inside(i, self.parent[bracket])
            for i in range(line_start, bracket))

    def _go(self, number, ids):
        for index in ids:
            word, kind = self.text[index], self.kind(self.parent[index])
            if self.word(index - 1) == ".":
                continue
            if word == "func" and self.name(index + 1):
                self.declare(index + 1, owner=True)
            elif word == "func" and self.word(index + 1) == "(" and self.match[index + 1] is not None \
                    and self.name(self.match[index + 1] + 1) and self.word(self.match[index + 1] + 2) in ("(", "["):
                self.declare(self.match[index + 1] + 1)  # a method: `func (r *T) Name(`
            elif word in ("type", "package") and self.name(index + 1) and self.word(index - 1) != ".":
                self.declare(index + 1, owner=word == "type")
            elif word in ("const", "var") and self.name(index + 1):
                self._go_names(index + 1)
            elif word in (":=",):
                self._go_targets(index)
            if self.start(index):
                self._go_element(index, kind)

    def _go_names(self, index):
        """Declare `a, b, c` from token `index`."""
        j = index
        while self.name(j):
            self.declare(j)
            if self.word(j + 1) != ",":
                break
            j += 2

    def _go_targets(self, operator):
        """Declare the names before `:=` (or a statement's `=`) back to the statement start."""
        j = operator - 1
        while j >= 0 and self.line[j] == self.line[operator] and (self.name(j) or self.text[j] == ","):
            if self.name(j) and self.word(j - 1) != ".":
                self.declare(j)
            j -= 1

    def _go_element(self, index, kind):
        word, following = self.text[index], self.word(index + 1)
        kind = "outer" if kind is None else kind  # the hunk shows no enclosing bracket: any block
        if kind == "params":
            closer = self.match[self.parent[index]]
            items, item = [], []
            end = closer if closer is not None else len(self.text)
            for j in range(self.parent[index] + 1, end):
                if self.parent[j] == self.parent[index] and self.text[j] == ",":
                    items.append(item)
                    item = []
                elif self.parent[j] == self.parent[index]:
                    item.append(j)
            items.append(item)
            named = any(len([j for j in it if self.text[j] not in ("...",)]) >= 2 and self.name(it[0])
                        and self.text[it[1]] != "." for it in items if it)  # `name Type`, not `pkg.Type`
            if named and self.name(index) and word not in KEYWORDS["go"] and following != ".":
                self.declare(index)
            elif not named and closer is None and self.name(index) and self.name(index + 1):
                self.declare(index)
            return
        if kind in ("composite", "call", "group", "index", "import"):
            return
        if not self.name(index) or word in KEYWORDS["go"]:
            return
        if kind == "struct" or kind == "outer" and following not in (":", "(", ".", "=", ":=", "++", "--"):
            j = index
            names = []
            while self.name(j) and self.word(j + 1) == ",":
                names.append(j)
                j += 2
            if kind == "outer" and self.name(j) and (self.word(j + 1) in ("=", ":=") or self.word(j + 1) in _ASSIGN):
                for k in names + [j]:
                    self.declare(k)  # `a, b = ...`: a var group entry or an assignment
            elif self.name(j) and (self.name(j + 1) or self.word(j + 1) in ("*", "[", "<-", "(") and
                                 self.word(j + 1) != "(" or self.word(j + 1) in ("func", "map", "chan", "struct",
                                                                                "interface")):
                for k in names + [j]:
                    self.declare(k)
            elif kind == "outer" and self.name(j) and self.word(j + 1) in ("", ",") and self.word(j - 1) != ".":
                self.declare(j)  # a const/var group entry or a field list the hunk does not open
            return
        if kind == "interface":
            if following == "(":
                self.declare(index)
            return
        if kind in ("values", "types", "outer-paren"):
            j = index
            while self.name(j):
                self.declare(j)
                if self.word(j + 1) != ",":
                    break
                j += 2
            return
        if kind in ("block", None, "outer") and following in ("=", ",") or following in _ASSIGN:
            j = index
            names = []
            while self.name(j) and self.word(j + 1) == ",":
                names.append(j)
                j += 2
            if self.word(j + 1) in ("=",) or self.word(j + 1) in _ASSIGN or self.word(j + 1) == ":=":
                for k in names + [j]:
                    if self.name(k):
                        self.declare(k)
        if kind == "outer" and following == "(":
            self.declare(index)  # an interface method or a call of the package's own function

    # ------------------------------------------------------------------ Rust
    def _kind_rust(self, bracket):
        if bracket < 0:
            return "outer" if self.text[-1 - bracket] == "}" else "outer-paren"
        opener = self.text[bracket]
        before = [self.text[i] for i in self.before(bracket) if self.line[i] == self.line[bracket]]
        last = before[-1] if before else ""
        parent_kind = self.kind(self.parent[bracket])
        if opener == "(":
            j = bracket - 1
            if self.word(j) == ">":  # generics: back over <...>
                depth = 0
                while j >= 0 and self.line[j] == self.line[bracket]:
                    depth += {">": 1, "<": -1}.get(self.text[j], 0)
                    if depth == 0:
                        break
                    j -= 1
                j -= 1
            if self.name(j) and self.word(j - 1) == "fn":
                return "params"
            if parent_kind in ("pattern", "params") or last in ("let", "|", "for") or self.word(
                    bracket - 1) in ("Some", "Ok", "Err") and parent_kind == "pattern":
                return "pattern"
            return "call" if last and (self.name(bracket - 1) or last in (")", "]", "!", ">")) else "tuple"
        if opener == "[":
            if last in ("let",) or parent_kind in ("pattern", "params"):
                return "pattern"
            return "array"
        words = set(self.head_tokens(bracket))
        if words & {"struct", "union"} and "=" not in words:
            return "struct"
        if "enum" in words:
            return "enum"
        if "impl" in words or "trait" in words:
            return "impl"
        if "match" in words:
            return "match"
        if parent_kind == "enum" and self.start(bracket - 1) and self.name(bracket - 1):
            return "struct"  # a struct variant
        if before and before[0] in ("let",) and "=" not in words or parent_kind in ("pattern", "params"):
            return "pattern"
        if last and re.match(r"[A-Z]", last) and self.name(bracket - 1) and not (
                words & {"if", "while", "for", "match", "fn", "loop", "unsafe", "impl", "trait", "mod", "else"}):
            return "literal"
        return "block"

    def _rust(self, number, ids):
        for index in ids:
            word, kind = self.text[index], self.kind(self.parent[index])
            if self.word(index - 1) in (".", "::"):
                continue
            if word in ("fn", "struct", "enum", "union", "trait", "type", "mod") and self.name(index + 1):
                self.declare(index + 1, owner=word != "mod")
                self.generics(index + 2)
            elif word == "impl":
                self.generics(index + 1)
            elif word in ("const", "static") and self.word(index + 1) == "mut" and self.name(index + 2):
                self.declare(index + 2)
            elif word in ("const", "static") and self.name(index + 1) and self.word(index + 2) == ":":
                self.declare(index + 1)
            elif word == "macro_rules" and self.word(index + 1) == "!" and self.name(index + 2):
                self.declare(index + 2)
            elif word == "let":
                self._rust_pattern(index + 1, ("=", ";", ":", "else"))
            elif word == "for" and "impl" not in [self.text[j] for j in self.before(index)
                                                  if self.line[j] == self.line[index]]:
                end = next((j for j in range(index + 1, len(self.text)) if self.text[j] == "in"
                            and self.parent[j] == self.parent[index]), None)
                if end is not None:
                    self._rust_pattern(index + 1, ("in",))
            elif word == "=>" and kind in ("match", "outer", "block", None):
                start = index - 1
                while start > 0 and self.line[start - 1] == self.line[index] and not (
                        self.start(start) and self.parent[start] == self.parent[index]):
                    start -= 1
                guard = next((j for j in range(start, index) if self.text[j] == "if"
                              and self.parent[j] == self.parent[index]), index)
                self._rust_bindings(list(range(start, guard)))
            elif word == "|" and (self.word(index - 1) in ("(", ",", "=", "move", "return", "=>", "", "{", ";")
                                  or not self.same_line(index - 1, index)):
                end = next((j for j in range(index + 1, len(self.text)) if self.text[j] == "|"
                            and self.parent[j] == self.parent[index]), None)
                if end is not None:
                    self._rust_bindings(list(range(index + 1, end)), params=True)
            if self.start(index):
                self._rust_element(index, kind)

    def _rust_pattern(self, index, stops):
        j = index
        tokens = []
        while j < len(self.text) and not (self.text[j] in stops and self.parent[j] == self.parent[index]):
            if self.parent[j] != self.parent[index] and not self.inside(j, self.parent[index]):
                break
            tokens.append(j)
            j += 1
        self._rust_bindings(tokens)

    def _rust_bindings(self, tokens, params=False):
        """Declare the bindings of a Rust pattern (its tokens): lower-case names that no path, call, struct pattern or
        macro follows and no path or member access precedes, not the field name of a struct pattern's `field:
        binding`; with params (a parameter or closure list), each parameter's pattern before its `: Type`."""
        if not tokens:
            return
        depth, typed = self.parent[tokens[0]], False
        inner = {index for index in tokens if self.text[index] in ("(", "[", "{")}
        for index in tokens:
            if params and self.parent[index] == depth and self.text[index] in (",", ":"):
                typed = self.text[index] == ":"
                continue
            if typed or not self.name(index) or self.text[index] in KEYWORDS["rust"] \
                    or not re.match(r"[a-z_]", self.text[index]):
                continue
            if self.word(index - 1) in ("::", ".") or self.word(index + 1) in ("::", "(", "!", "{", "."):
                continue
            if self.word(index + 1) == ":" and self.parent[index] in inner and self.text[self.parent[index]] == "{":
                continue  # Struct { field: binding }
            self.declare(index)

    def _rust_element(self, index, kind):
        word, following = self.text[index], self.word(index + 1)
        kind = "outer" if kind is None else kind  # the hunk shows no enclosing bracket: any block
        if kind == "params":
            if word in ("&", "mut"):
                index += 1 if word == "mut" else (2 if self.word(index + 1) == "mut" else 1)
            end = next((j for j in range(index, len(self.text)) if self.text[j] in (":", ",", ")")
                        and self.parent[j] == self.parent[index]), index)
            self._rust_bindings(list(range(index, end + 1)), params=True)
            return
        j = index
        if self.word(j) == "pub":
            j += 1
            if self.word(j) == "(" and self.match[j] is not None:
                j = self.match[j] + 1
        if not self.name(j) or self.text[j] in KEYWORDS["rust"]:
            return
        following = self.word(j + 1)
        if kind == "struct" and following == ":" or j != index and following == ":" and self.word(j + 2) != ":":
            self.declare(j)
        elif kind == "enum" and following in (",", "(", "{", "=", "", "}"):
            self.declare(j)
        elif kind in ("outer", "outer-paren") and following == ":" and self.word(j + 2) != ":":
            self.declare(j)  # a field or parameter in a block the hunk does not open
        elif kind == "outer" and re.match(r"[A-Z]", self.text[j]) and following in (",", "(", "{", "", "=") \
                and self.word(j + 2) != "::" and not any(self.text[k] in ("=>", ";") for k in self.statement(j)):
            self.declare(j)  # an enum variant in a block the hunk does not open (not a match arm or a statement)
        elif kind in ("block", "outer") and (following == "=" or following in _ASSIGN):
            self.declare(j)

    # ------------------------------------------------------------------ the C family and the rest
    def _kind_c(self, bracket):
        if bracket < 0:
            return "outer"
        before = [self.text[i] for i in self.before(bracket) if self.line[i] == self.line[bracket]]
        after = [self.text[i] for i in self.after_close(bracket)]
        if self.text[bracket] == "(" and (after[:1] in (["{"], ["->"], ["=>"], [":"], ["throws"], ["const"])
                                          or before[-1:] in (["catch"], ["for"], ["foreach"])):
            return "params"
        return "block"

    def _c(self, number, ids):
        code = self.codes[number].strip()
        found = [name for match in GENERIC_DECLARATION.finditer(code) for name in match.groups() if name]
        if not C_TYPE_DEFINITION.search(code) and (typed := C_TYPED.match(C_ATTRIBUTES.sub("", code))) \
                and not set(re.findall(_N, typed.group(1))) & C_STATEMENT:
            found.append(typed.group(2))
        for index in ids:
            if self.kind(self.parent[index]) == "params" and self.start(index):
                bracket = self.parent[index]
                end, closer = index, self.match[bracket] if bracket >= 0 else -1 - bracket
                while end < len(self.text) and end != closer and not (
                        self.text[end] in (",", "=", ":") and self.parent[end] == self.parent[index]):
                    end += 1
                if end - 1 > index and self.name(end - 1):
                    found.append(self.text[end - 1])
            elif (self.text[index] == "->" and self.path.endswith(".java") or self.text[index] == "=>"
                  and self.path.endswith((".cs", ".dart"))) and self.name(index - 1):
                found.append(self.text[index - 1])  # a lambda's parameter (C and C++ `->` is member access)
        for name in found:
            name = name.strip().lstrip("~")
            if name:
                self.found.add(name)

    def _generic(self, number, ids):
        code = self.codes[number].strip()
        self.found.update(name for match in GENERIC_DECLARATION.finditer(code) for name in match.groups() if name)
        if match := re.match(rf"^(?:local\s+|val\s+|var\s+|let\s+)?({_N})\s*(?::=|=(?!=|>))", code):
            self.found.add(match.group(1))


def change_windows(solution_patch: str) -> dict:
    """{solution file: lines a released copy of its changed code may spread over}: the region between its first and last
    added line in the patched file, and at least its added line count and 60 lines."""
    first, last, counts, current, line_no = {}, {}, added_counts(solution_patch), None, 0
    for line in solution_patch.splitlines():
        if line.startswith("+++ "):
            current = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("@@"):
            match = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", line)
            line_no = int(match.group(1)) if match else 0
        elif current and line.startswith("+"):
            first.setdefault(current, line_no)
            last[current] = line_no
            line_no += 1
        elif current and (line.startswith(" ") or line == ""):
            line_no += 1
    return {path: max(60, counts.get(path, 0), last[path] - first[path] + 1) for path in first}


def densest(hits: dict, per_source: dict, windows: dict) -> list:
    """For every (file, solution file) pair: the most of that solution file's items (lines or identifiers) that one window
    of the file holds. hits: {path: [(line number, item)]} or [(line number, item, byte offset)]; windows: {solution
    file: window in lines}. With byte offsets the window is also at most BYTES_PER_LINE bytes per line: one line of a
    minified bundle can hold a whole library (official1-koota: bundled TypeScript compilers in pnpm's store held 6 of a
    bitECS feature's 14 names "within 60 lines"). Sorted, best first."""
    rows = []
    for path, found in hits.items():
        for source, items in per_source.items():
            wanted = set(items)
            located = sorted((entry[2] if len(entry) > 2 else entry[0], entry[0], entry[1])
                             for entry in found if entry[1] in wanted)
            if not located:
                continue
            window = windows.get(source, 60)
            limit = window * BYTES_PER_LINE if any(len(entry) > 2 for entry in found) else window
            best, span, start = set(), (0, 0), 0
            for end in range(len(located)):
                while located[end][1] - located[start][1] > window or located[end][0] - located[start][0] > limit:
                    start += 1
                held = {item for _, _, item in located[start:end + 1]}
                if len(held) > len(best):
                    best, span = held, (located[start][1], located[end][1])
            rows.append({"path": path, "source": source, "best": len(best), "total": len(wanted),
                         "need": max(3, (len(wanted) + 1) // 2), "window": window, "lines": list(span),
                         "items": sorted(best)})
    return sorted(rows, key=lambda row: (-row["best"] / max(1, row["need"]), row["path"], row["source"]))


BYTES_PER_LINE = 100


def identifier_problems(hits: dict, per_source: dict, windows: dict) -> list:
    """A released copy of the removed code: at least half (>= 3) of one solution file's distinctive identifiers and
    literals (solution_identifiers that the starting state lacks) inside one window of a file (change_windows)."""
    return sorted({f"{row['path']} holds {row['best']}/{row['total']} distinctive identifiers of {row['source']} within "
                   f"{row['window']} lines" for row in densest(hits, per_source, windows) if row["best"] >= row["need"]})


# ------------------------------------------------------------------------- released copies judged per package
# hengyuf 2026-10-20 (official1-sqlformatter round 0): the per-file rules above need >= 4 (lines) or >= 3 (identifiers)
# of ONE solution file's items in one place, but c01/c02/c04 changed 2-4 files with < 4 distinctive lines each, so the
# released @sqltools/formatter 1.2.3 in yarn's cache (all of c04's lines, 7/11 of c02's InlineBlock lines) never added
# up. A released package of the repository itself is one copy of the code however its files are split: package_rows
# sums, per package, the items of every solution code file found anywhere in it, and package_problems flags a package
# of the repo's own (exact name, see own_package) holding at least half (>= 4 lines, >= 3 identifiers) of the
# solution's items when no per-file problem already names one of its files.
# (the trailing "/" is a lookahead: in .../node_modules/metro/node_modules/jest-worker/ both matches need it)
PACKAGE_DIR = re.compile(r"/(?:node_modules|_vendor|vendor|site-packages|dist-packages)/((?:@[^/]+/)?[^/]+)(?=/)")
CRATE_DIR = re.compile(r"/registry/src/[^/]+/([^/]+?)-\d+\.\d+\.\d+[^/]*(?=/)")
GO_MODULE_DIR = re.compile(r"/pkg/mod/([^@]+)@[^/]+(?=/)")
BUN_DIR = re.compile(r"/install/cache/((?:@[^/]+/)?[^/@]+)@[^/]+(?=/)")  # bun's cache: <name>@<version>@@@<n>/
UV_DIR = re.compile(r"/uv/archive-v\d+/[^/]+/([^/]+)(?=/)")  # uv's cache: <id>/<top-level entry of an unpacked wheel>/
GO_ZIP_ENTRY = re.compile(r"^([^@/.]+\.[^@/]+(?:/[^@/]+)*)@v\d[^/]*(?=/)")  # a Go module zip: <module>@<version>/<file>
DIST_INFO = re.compile(r"(?:-\d[^/]*)?\.(?:dist-info|egg-info)$")  # Python metadata: <distribution>-<version>.dist-info
ARCHIVE_NAME = re.compile(r"^(.+?)-(?:npm-)?v?\d+(?:\.\d+)*")


def _flat(name: str) -> str:
    return re.sub(r"[-_./]+", "-", name.lower().strip("/"))


def _package_dirs_in(text: str) -> list:
    """[(end, name, kind)] of the package directories in `text` (a path, or "/" + an archive member's inner path), kind
    "dir" (node_modules, site-packages, dist-packages), "vendor" (_vendor, vendor: also a source directory's name),
    "crate" (cargo registry), "go" (module cache), "bun" or "uv" (caches). Names as published: Go module paths
    unescaped (the module cache writes github.com/!burnt!sushi for github.com/BurntSushi), Python metadata directories
    (<distribution>-<version>.dist-info, .egg-info) under their distribution name."""
    found = []
    for regex, kind in ((PACKAGE_DIR, "dir"), (CRATE_DIR, "crate"), (GO_MODULE_DIR, "go"), (BUN_DIR, "bun"),
                        (UV_DIR, "uv")):
        for match in regex.finditer(text):
            name = match.group(1)
            if kind == "go":
                name = re.sub(r"!([a-z])", lambda letter: letter.group(1).upper(), name)
            found.append((match.end(), DIST_INFO.sub("", name) or name,
                          "vendor" if match.group(0).startswith(("/vendor/", "/_vendor/")) else kind))
    return found


def _archive_name(head: str, inner: str) -> str:
    """The package name of an archive or npm/pip cache entry ("<archive>!/<inner>"): its npm key, the module of a Go
    module zip, else its file name or top-level entry without the version."""
    label = head.rsplit(" ", 1)[-1]  # npm cache entries are shown as "<cache> <tarball url>"
    if "/-/" in label:
        name = label.split("/-/")[0].split("//", 1)[-1].split("/", 1)[-1]
    elif match := GO_ZIP_ENTRY.match(inner):  # cache/download/<module>/@v/<version>.zip holds <module>@<version>/
        name = match.group(1)
    else:
        match = ARCHIVE_NAME.match(label.rsplit("/", 1)[-1]) or ARCHIVE_NAME.match(inner.split("/", 1)[0])
        name = match.group(1) if match else inner.split("/", 1)[0]
    return urllib.parse.unquote(name)


def package_root(path: str, package_dirs=()) -> tuple:
    """(root, package name) of the released package that a scan hit lies in, or None: the innermost
    node_modules/<name>, _vendor/<name>, site-packages/<name>, cargo registry crate, Go module version or bun/uv cache
    entry (also inside an archive), else a whole archive or npm/pip cache entry ("<archive>!/", named after its file,
    npm key, Go module or top-level entry), else the longest scanned package-named directory (released_selection's
    package_dirs)."""
    head, sep, inner = path.partition("!/")
    text = "/" + inner if sep else path
    found = _package_dirs_in(text)
    if found:
        end, name, _ = max(found)
        return (head + "!" if sep else "") + text[:end + 1], name
    if sep:
        return head + "!/", _archive_name(head, inner)
    owner = max((d.rstrip("/") for d in package_dirs if path.startswith(d.rstrip("/") + "/")), key=len, default="")
    return (owner + "/", owner.rsplit("/", 1)[-1]) if owner else None


def own_package(name: str, released: dict) -> bool:
    """Is `name` (a package root's name) one of the repo's own released packages, by exact (normalized) name: a
    released_names name (JS package, workspace directory, repo, Python project/package, Go module's last segment), one
    that only the full manifest parsing finds ("own_names"), a publishable crate ("crates", hengyuf 2026-10-20) or a Go
    module path. Exact, unlike name_match: "formatter" or "types" would otherwise pull in every package containing
    them."""
    modules = released.get("go_modules", [])
    own = {_flat(n) for key in ("names", "own_names", "crates") for n in released.get(key, [])} | {
        _flat(m) for m in modules} | {_flat(re.sub(r"/v\d+$", "", m)) for m in modules}
    return bool(_flat(name)) and _flat(name) in own


def package_rows(hits: dict, per_source: dict, released: dict, package_dirs=(), *, minimum: int) -> list:
    """Per own package (package_root + own_package): the solution items (per_source: {solution file: items}) found
    anywhere in it, summed over the solution files, and its files that hold any. hits: {path: [(line, item[, byte])]}.
    Sorted, best first."""
    wanted = {item for values in per_source.values() for item in values}
    packages = {}
    for path, found in hits.items():
        items = {entry[1] for entry in found} & wanted
        located = package_root(path, package_dirs) if items else None
        if located and own_package(located[1], released):
            entry = packages.setdefault(located[0], {"name": located[1], "items": set(), "files": set()})
            entry["items"] |= items
            entry["files"].add(path)
    total = sum(len(set(values)) for values in per_source.values())
    rows = []
    for root, entry in packages.items():
        held = {source: sorted(set(values) & entry["items"]) for source, values in per_source.items()}
        rows.append({"root": root, "name": entry["name"], "best": sum(len(values) for values in held.values()),
                     "total": total, "need": max(minimum, (total + 1) // 2), "files": sorted(entry["files"])[:20],
                     "sources": {source: values for source, values in held.items() if values}})
    return sorted(rows, key=lambda row: (-row["best"], row["root"]))


def package_problems(rows: list, flagged: list, *, unit: str) -> list:
    """The package_rows at or above their need, unless a per-file problem (flagged) already names a file inside."""
    named = [problem.split(" holds ", 1)[0] for problem in flagged]
    return sorted(f"{row['root']} holds {row['best']}/{row['total']} distinctive {unit} of the solution's code files "
                  f"(from {len(row['sources'])} of them) across {len(row['files'])} files of one released package"
                  for row in rows
                  if row["best"] >= row["need"] and not any(path.startswith(row["root"]) for path in named))


# Top-level Python packages that are not the project's released package.
GENERIC_PACKAGES = frozenset("""benchmark benchmarks bench ci conftest contrib demo docs doc example examples fixtures
    internal misc samples scripts setup src test testing tests tools util utils vendor""".split())


def _norm(text: str) -> str:
    return re.sub(r"[-_.]+", "-", text.lower())


CARGO_NAME = re.compile(r"""^\s*name\s*=\s*["']([A-Za-z0-9_-]+)["']\s*(#.*)?$""")
CARGO_UNPUBLISHED = re.compile(r"^\s*publish\s*=\s*(false|\[\s*\])\s*(#.*)?$")
# PURGE_RELEASES' $skip: test, example, fixture, docs and vendored trees hold no released package of the repo
NON_RELEASE_TREE = re.compile(r"(^|/)(node_modules|vendor|third_party|test|tests|__tests__|testdata|fixtures?|"
                              r"__fixtures__|examples?|e2e|docs?|website|benchmarks?)/")
LOOSE_SKIP = re.compile(r"(^|/)(__mocks__|__typetests__|mocks?|stubs?|templates?|playgrounds?)/")
PYTHON_NAME = re.compile(r"""^\s*name\s*[=:]\s*(["']?)([A-Za-z0-9][A-Za-z0-9._-]*)["']?\s*,?\s*$""")


def _glob_regexes(pattern: str):
    """Regexes of the directories that a workspace glob matches ("./" dropped, {a,b} expanded, **, * and ?), or None
    for a glob this does not understand (character classes, extglobs, absolute or parent paths)."""
    pattern = pattern.strip()
    while pattern.startswith("./"):
        pattern = pattern[2:].lstrip("/")
    pattern = pattern.rstrip("/")
    if not pattern or pattern.startswith("/") or ".." in pattern.split("/") or re.search(r"[\[\]()\\]", pattern):
        return None
    expanded = [pattern]
    while any("{" in item or "}" in item for item in expanded):
        braces = [re.search(r"\{([^{}]*)\}", item) for item in expanded]
        if not all(braces) or len(expanded) > 64:
            return None
        expanded = [item[:match.start()] + choice + item[match.end():]
                    for item, match in zip(expanded, braces) for choice in match.group(1).split(",")]
    wildcards = {"**/": "(?:.*/)?", "**": ".*", "*": "[^/]*", "?": "[^/]"}
    return [re.compile("^" + "".join(wildcards.get(part, re.escape(part))
                                     for part in re.split(r"(\*\*/|\*\*|\*|\?)", item)) + "$") for item in expanded]


def _yaml_uncomment(line: str) -> str:
    """A YAML line without its comment ("#" at the start or after a blank, outside quotes)."""
    quote = ""
    for index, char in enumerate(line):
        if quote:
            quote = "" if char == quote else quote
        elif char in "'\"":
            quote = char
        elif char == "#" and (index == 0 or line[index - 1] in " \t"):
            return line[:index].rstrip()
    return line.rstrip()


def _yaml_scalar(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1].replace("''", "'") if text[0] == "'" else text[1:-1]
    return text


def pnpm_workspace_globs(lines: list) -> tuple:
    """(globs, error) of pnpm-workspace.yaml's top-level "packages": a block list (indented or not) or a flow list
    (`packages: ['a/*', 'b/*']`, also over several lines); error names a value this does not understand."""
    for index, line in enumerate(lines):
        match = re.match(r"""^(["']?)packages\1\s*:(.*)$""", line)
        if not match:
            continue
        value, following = _yaml_uncomment(match.group(2)).strip(), [_yaml_uncomment(text) for text in lines[index + 1:]]
        first = next((text.strip() for text in following if text.strip()), "")
        if value.startswith("[") or (value in ("", "~", "null") and first.startswith("[")):
            flow, depth, quote = "", 0, ""
            for char in "\n".join([value] + following).strip():
                flow += char
                if quote:
                    quote = "" if char == quote else quote
                elif char in "'\"":
                    quote = char
                elif char in "[]{}":
                    depth += 1 if char in "[{" else -1
                    if depth == 0:
                        break
            items, current, quote = [], "", ""
            for char in flow[1:-1]:
                if quote:
                    quote = "" if char == quote else quote
                elif char in "'\"":
                    quote = char
                elif char == ",":
                    items.append(current)
                    current = ""
                    continue
                current += char
            items.append(current)
            globs = [_yaml_scalar(item) for item in items if item.strip()]
            if (depth or quote or not flow.endswith("]") or re.search(r"[\[\]{}]", flow[1:-1])
                    or any(not item.strip() for item in items[:-1])):
                return globs, f"pnpm-workspace.yaml: packages value {flow[:80]!r} not understood"
            return globs, ""
        if value not in ("", "~", "null"):
            return [], f"pnpm-workspace.yaml: packages value {value[:80]!r} not understood"
        globs, error = [], ""
        for text in following:
            if text.startswith(("---", "...")) or (text.strip() and not text.startswith((" ", "\t", "-"))):
                break  # the next top-level key or document
            item = re.match(r"^\s*-\s*(.*)$", text)
            if text.strip() and (not item or not item.group(1) or item.group(1)[0] in "[{&*!|>-"
                                 or (item.group(1)[0] not in "'\"" and re.search(r":(\s|$)", item.group(1)))):
                error = error or f"pnpm-workspace.yaml: packages entry {text.strip()[:80]!r} not understood"
            elif text.strip():
                globs.append(_yaml_scalar(item.group(1)))
        return globs, error
    return [], ""  # no "packages": the workspace is the root package alone


def _jsonc(text: str):
    """json.loads for JSON with // and /* */ comments and trailing commas (rush.json)."""
    out, index = [], 0
    while index < len(text):
        if text[index] == '"':
            end = index + 1
            while end < len(text) and text[end] != '"':
                end += 2 if text[end] == "\\" else 1
            out.append(text[index:end + 1])
            index = end + 1
        elif text.startswith("//", index):
            end = text.find("\n", index)
            index = len(text) if end < 0 else end
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            if end < 0:
                raise ValueError("unterminated comment")
            index = end + 2
        else:
            out.append(text[index])
            index += 1
    return json.loads(re.sub(r'"(?:[^"\\]|\\.)*"|,(?=\s*[\]}])',
                             lambda match: match.group(0) if match.group(0).startswith('"') else "", "".join(out)))


def _json_globs(lines: list, key: str, *, default=None) -> tuple:
    """(globs, ok) of a JSON manifest's glob list at data[key] (yarn's {"packages": [...]} form too; default when
    absent): its strings, best effort, and whether the manifest parsed with a list of strings there."""
    try:
        data = json.loads("\n".join(lines) or "{}")
    except ValueError:
        return [], False
    value = data.get(key) if isinstance(data, dict) else None
    value = value.get("packages") if isinstance(value, dict) else value
    if value is None:
        return list(default or []), isinstance(data, dict)
    strings = [item for item in value if isinstance(item, str)] if isinstance(value, list) else []
    return strings, isinstance(value, list) and len(strings) == len(value)


def _discovery_names(repo: str, manifest: str) -> dict:
    """released_names' "names", "go_modules", "go_module_dirs" and "workspace_dirs": what ARCHIVE_DISCOVER looks for and
    released_selection selects by, computed exactly as before 19:57 (hengyuf 2026-10-20: discovery stays as it was; the
    fuller manifest parsing below only adds own names for the per-package rule). The repo name, the "name" of every
    public JS workspace package (root package.json "workspaces", pnpm-workspace.yaml; also the root package.json) and
    its directory name (packages/jest-get-type is now "@jest/get-type", but the cache holds the pre-rename
    jest-get-type release), the Python project and top-level import packages, and the Go module paths."""
    files, packages, modules, module_dirs, python = {}, {}, set(), {}, set()
    for line in manifest.splitlines():
        tag, _, rest = line.partition("\t")
        if tag == "F":
            name, _, text = rest.partition("\t")
            files.setdefault(name, []).append(text)
        elif tag == "J" and "\t" in rest:
            path, name, *flags = rest.split("\t")
            if "private" not in flags and "unparsed" not in flags:  # private: examples, docs sites, the monorepo root
                packages[path] = name.strip()  # (unparsed: a package.json that is no valid JSON, see released_names)
        elif tag == "G" and "\t" in rest and rest.split("\t", 1)[1].strip():
            path, module = rest.split("\t", 1)
            modules.add(module.strip())
            module_dirs[path.rsplit("/", 1)[0] if "/" in path else ""] = module.strip()
        elif tag == "P" and rest:
            python.add(rest.strip().split("/")[-2])
    globs = []
    try:
        root = json.loads("\n".join(files.get("package.json", [])) or "{}")
        workspaces = root.get("workspaces") or []
        globs += workspaces.get("packages", []) if isinstance(workspaces, dict) else workspaces
    except (ValueError, AttributeError):
        pass
    in_packages = False
    for text in files.get("pnpm-workspace.yaml", []):
        if re.match(r"^packages\s*:", text):
            in_packages = True
        elif in_packages and (match := re.match(r"^\s+-\s*['\"]?([^'\"#]+?)['\"]?\s*(#.*)?$", text)):
            globs.append(match.group(1))
        elif text.strip() and not text.startswith((" ", "\t")):
            in_packages = False
    include = [glob for glob in globs if isinstance(glob, str) and not glob.startswith("!")]
    exclude = [glob[1:] for glob in globs if isinstance(glob, str) and glob.startswith("!")]

    def matches(directory, patterns):
        for pattern in patterns:
            regex = "^" + re.escape(pattern.strip().rstrip("/")).replace(r"\*\*", ".*").replace(r"\*", "[^/]*") + "$"
            if re.match(regex, directory):
                return True
        return False

    names, directories = {repo.split("/")[-1].lower()}, {}
    for path, name in packages.items():
        directory = path.rsplit("/", 1)[0] if "/" in path else ""
        if directory == "" or (matches(directory, include) and not matches(directory, exclude)):
            names.add(name.lower())
            if directory:
                names.add(directory.rsplit("/", 1)[-1].lower())
                directories[directory] = name
    for source in ("pyproject.toml", "setup.cfg", "setup.py"):
        section = "" if source != "setup.py" else "setup"
        for text in files.get(source, []):
            if header := re.match(r"^\s*\[\[?([^\]]+)\]\]?\s*$", text):
                section = header.group(1).strip()
            elif section in ("project", "tool.poetry", "metadata", "setup") and (match := re.match(
                    r"""^\s*name\s*[=:]\s*["']?([A-Za-z0-9][A-Za-z0-9._-]*)["']?\s*,?\s*$""", text)):
                names.add(match.group(1).lower())
    names |= {name.lower() for name in python if name.lower() not in GENERIC_PACKAGES}
    for module in modules:
        names.add(re.sub(r"/v\d+$", "", module).rsplit("/", 1)[-1].lower())
    return {"names": sorted(name for name in names if len(name.lstrip("@")) >= 3), "go_modules": sorted(modules),
            "go_module_dirs": dict(sorted(module_dirs.items())), "workspace_dirs": dict(sorted(directories.items()))}


def released_names(repo: str, manifest: str) -> dict:
    """The names under which the repo's own packages are released, from ARCHIVE_MANIFEST's output. "names",
    "go_modules", "go_module_dirs", "workspace_dirs": _discovery_names, what ARCHIVE_DISCOVER looks for, exactly as before
    19:57. For the per-package rule only (own_package; never discovery, hengyuf 2026-10-20): "own_names", the names that
    the full manifest parsing finds (pnpm-workspace.yaml block or flow lists, lerna.json "packages", rush.json
    "projects", "./" and {a,b} workspace globs, go.mod module paths without a trailing comment), and "crates", the
    [package] name of every publishable crate (K lines: not publish = false, not under test/example/fixture/docs trees;
    no directory names: wasmtime's crates/wast is wasmtime-wast, wast is wasm-tools'). "unparsed": what could not be
    read, so own names may be missing (a package.json, lerna.json or rush.json that is no valid JSON, a pnpm list or
    workspace glob this does not understand, a crate or go.mod without a name, a Python project without one, more
    manifests than ARCHIVE_MANIFEST reads; leak_scan logs it)."""
    files, packages, modules, python, crates = {}, {}, set(), set(), {}
    unread_json, unparsed = [], []
    for line in manifest.splitlines():
        tag, _, rest = line.partition("\t")
        if tag == "F":
            name, _, text = rest.partition("\t")
            files.setdefault(name, []).append(text)
        elif tag == "J" and "\t" in rest:
            path, name, *flags = rest.split("\t")
            if "unparsed" in flags:
                unread_json.append(path)
            elif "private" not in flags:  # examples, docs sites, the monorepo root: never published
                packages[path] = name.strip()
        elif tag == "G" and "\t" in rest:
            path, module = rest.split("\t", 1)
            module = module.split("//", 1)[0].strip()
            if module:
                modules.add(module)
            else:
                unparsed.append(f"{path}: no module path")
        elif tag == "P" and rest:
            python.add(rest.strip().split("/")[-2])
        elif tag == "K" and "\t" in rest:
            path, text = rest.split("\t", 1)
            crates.setdefault(path, []).append(text)
        elif tag == "Z" and rest:
            unparsed.append(rest.replace("\t", ": ", 1))
    globs, folders = [], []
    if "package.json" in files:
        listed, ok = _json_globs(files["package.json"], "workspaces")
        globs += listed
        if not ok:
            unparsed.append('package.json: no valid JSON object with a "workspaces" list of globs')
    if "pnpm-workspace.yaml" in files:
        listed, error = pnpm_workspace_globs(files["pnpm-workspace.yaml"])
        globs += listed
        if error:
            unparsed.append(error)
    if "rush.json" in files:
        try:
            projects = _jsonc("\n".join(files["rush.json"])).get("projects")
        except (ValueError, AttributeError):
            projects = None
        if isinstance(projects, list) and all(isinstance(project, dict) and isinstance(project.get("packageName"), str)
                                              and isinstance(project.get("projectFolder"), str) for project in projects):
            folders = [re.sub(r"^(\./)+", "", project["projectFolder"].strip()).strip("/") for project in projects]
        else:
            unparsed.append('rush.json: no valid JSON with a "projects" list of {packageName, projectFolder}')
    if "lerna.json" in files:  # no "packages": the package manager's workspaces, else lerna's default
        listed, ok = _json_globs(files["lerna.json"], "packages", default=[] if globs or folders else ["packages/*"])
        globs += listed
        if not ok:
            unparsed.append('lerna.json: no valid JSON object with a "packages" list of globs')
    include = [re.compile("^" + re.escape(folder) + "$") for folder in folders]  # rush: literal folders
    exclude = []
    for glob in globs:
        regexes = _glob_regexes(glob[1:] if glob.startswith("!") else glob)
        if regexes is None:
            unparsed.append(f"workspace glob {glob!r} not understood")
        else:
            (exclude if glob.startswith("!") else include).extend(regexes)

    def workspace(path):  # a package.json at the root or in a workspace directory
        directory = path.rsplit("/", 1)[0] if "/" in path else ""
        return directory == "" or (any(regex.match(directory) for regex in include)
                                   and not any(regex.match(directory) for regex in exclude))

    def loose(path):  # outside workspaces: a package of the repo's tree (bindings/node), not a test mock or template
        return not NON_RELEASE_TREE.search(path) and not LOOSE_SKIP.search(path)

    names = {repo.split("/")[-1].lower()}
    for path, name in packages.items():
        if workspace(path):
            names.add(name.lower())
            if "/" in path:
                names.add(path.rsplit("/", 1)[0].rsplit("/", 1)[-1].lower())
    unparsed += [f"{path}: no valid JSON" for path in unread_json if workspace(path) or loose(path)]
    crate_names = set()
    for path, lines in crates.items():
        name = next((match.group(1).lower() for text in lines if (match := CARGO_NAME.match(text))), "")
        if not name:
            unparsed.append(f"{path}: no [package] name")
        elif not any(CARGO_UNPUBLISHED.match(text) for text in lines) and not NON_RELEASE_TREE.search(path):
            crate_names.add(name)
    projects, declared = set(), {"setup.py"} & set(files)
    for source in ("pyproject.toml", "setup.cfg", "setup.py"):
        section = "" if source != "setup.py" else "setup"
        for text in files.get(source, []):
            if header := re.match(r"^\s*\[\[?([^\]]+)\]\]?\s*$", text):
                section = header.group(1).strip()
                if section in {"pyproject.toml": ("project", "tool.poetry"), "setup.cfg": ("metadata",)}.get(source, ()):
                    declared.add(source)
            elif section in ("project", "tool.poetry", "metadata", "setup") and (match := PYTHON_NAME.match(text)):
                names.add(match.group(2).lower())
                if source != "setup.py" or match.group(1):  # setup.py: a literal, not name=NAME
                    projects.add(match.group(2).lower())
    if declared and not projects:
        unparsed.append(f"{'/'.join(sorted(declared))}: no Python project name")
    names |= {name.lower() for name in python if name.lower() not in GENERIC_PACKAGES}
    for module in modules:
        names |= {module, re.sub(r"/v\d+$", "", module), re.sub(r"/v\d+$", "", module).rsplit("/", 1)[-1].lower()}
    return dict(_discovery_names(repo, manifest),
                own_names=sorted(name for name in names if len(name.lstrip("@")) >= 3),
                crates=sorted(name for name in crate_names if len(name) >= 3), unparsed=unparsed)


def _go_escape(path: str) -> str:
    return re.sub(r"[A-Z]", lambda match: "!" + match.group(0).lower(), path)


def name_match(text: str, names: list) -> str:
    """The first package name (see released_names) that the archive or cache entry name `text` contains, or ""."""
    flat = _norm(text)
    for name in names:
        variants = {_norm(name), _norm(name.replace("/", "-")), _norm(name.lstrip("@").replace("/", "-"))}
        if any(len(variant.lstrip("@")) >= 3 and variant in flat for variant in variants):
            return name
    return ""


def parse_tagged(output: str) -> dict:
    """{tag: [fields]} for the tab-separated lines that the ARCHIVE_* commands print."""
    parsed = {}
    for line in output.splitlines():
        tag, _, rest = line.partition("\t")
        if len(tag) == 1 and rest:
            parsed.setdefault(tag, []).append(rest.split("\t"))
    return parsed


def released_selection(discovered: str, released: dict) -> dict:
    """What ARCHIVE_SCAN must read, from ARCHIVE_DISCOVER's output: archives whose file name contains a package name
    (Go: whose module-cache path names a module of the repo), npm cache tarballs whose key names one, pip cache bodies
    holding one, every package-manager cache root, and package-named directories outside the repo's own tree.
    {"unpack": [(kind, path, label)], "roots": [path], "archives": [...], "tools": [...]}"""
    tagged, names = parse_tagged(discovered), released["names"]
    go_paths = ["/cache/download/" + _go_escape(re.sub(r"/v\d+$", "", module)) + "/" for module in released["go_modules"]]
    unpack, archives = [], []
    for size, path in (fields[:2] for fields in tagged.get("A", []) if len(fields) >= 2):
        name = name_match(path.rsplit("/", 1)[-1], names) or next((go for go in go_paths if go in path), "")
        if name and not path.startswith(LEAK_ARCHIVES + "/"):
            kind = "zip" if ARCHIVE_ZIP.search(path) else "tar"
            unpack.append((kind, path, path))
            archives.append({"path": path, "size": int(size) if size.isdigit() else None, "name": name, "kind": kind})
    cache = None
    for line in discovered.splitlines():
        tag, _, rest = line.partition("\t")
        if tag == "C":
            cache = rest
        elif tag == "N" and cache and rest.count("\t") == 1:
            key, integrity = rest.split("\t")
            url = urllib.parse.unquote(key.split(":", 2)[-1] if key.startswith("make-fetch-happen:") else key)
            package = url.split("/-/")[0].split("//", 1)[-1].split("/", 1)[-1] if "/-/" in url else ""
            name = name_match(package, names) if package else ""
            content = _cacache_path(cache, integrity)
            if name and content:
                unpack.append(("blob", content, f"{cache} {url}"))
                archives.append({"path": content, "key": url, "name": name, "kind": "npm-cache"})
    for fields in tagged.get("W", []):
        tops = fields[1].split(",") if len(fields) >= 2 else []
        if name := next((found for found in (name_match(top, names) for top in tops) if found), ""):
            unpack.append(("blob", fields[0], fields[0]))
            archives.append({"path": fields[0], "top_level": fields[1], "name": name, "kind": "pip-cache"})
    roots = sorted({fields[0] for fields in tagged.get("R", [])})
    dirs = sorted({fields[0] for fields in tagged.get("D", [])
                   if not fields[0].startswith("/app/") or DEPENDENCY_TREE.search(fields[0] + "/")})
    scan = []
    for path in roots + dirs:  # a directory inside another one is scanned with it
        if not any(path != other and path.startswith(other.rstrip("/") + "/") for other in roots + dirs):
            scan.append(path)
    return {"unpack": unpack, "roots": sorted(set(scan)), "archives": archives, "package_dirs": dirs,
            "cache_roots": roots, "tools": [fields[0] for fields in tagged.get("X", [])],
            "examined": {"archives": len(tagged.get("A", [])), "npm_cache_entries": len(tagged.get("N", [])),
                         "pip_cache_bodies": len(tagged.get("W", [])), "package_dirs": len(tagged.get("D", []))}}


def _cacache_path(cache: str, integrity: str) -> str:
    """The content-v2 file of an npm cache entry (cacache: content-v2/<algorithm>/<hex[0:2]>/<hex[2:4]>/<hex[4:]>)."""
    for entry in integrity.split():
        algorithm, _, digest = entry.partition("-")
        try:
            hexdigest = base64.b64decode(digest.split("?")[0]).hex()
        except ValueError:
            continue
        if algorithm and len(hexdigest) > 8:
            return f"{cache}/content-v2/{algorithm}/{hexdigest[:2]}/{hexdigest[2:4]}/{hexdigest[4:]}"
    return ""


def parse_archive_scan(output: str) -> dict:
    """ARCHIVE_SCAN's output: {"unpacked": {n: (source, status)}, "present": set, "fixed": {path: [(line, text, byte)]},
    "identifiers": {path: [(line, name, byte)]}}, paths inside LEAK_ARCHIVES/<n>/ shown as "<source>!/<inner path>"."""
    unpacked, present, fixed, identifiers = {}, set(), {}, {}
    for line in output.splitlines():
        tag, _, rest = line.partition("\t")
        if tag == "U" and rest.count("\t") >= 2:
            parts = rest.split("\t")
            unpacked[parts[0]] = ("\t".join(parts[1:-1]), parts[-1])
        elif tag == "P":
            present.add(rest)
        elif tag in ("H", "I") and (match := re.match(r"(.*?):(\d+):(\d+):(.*)$", rest)):
            path = match.group(1)
            if path.startswith(LEAK_ARCHIVES + "/"):
                number, _, inner = path[len(LEAK_ARCHIVES) + 1:].partition("/")
                path = f"{unpacked.get(number, (path,))[0]}!/{inner}"
            (fixed if tag == "H" else identifiers).setdefault(path, []).append(
                (int(match.group(2)), match.group(4), int(match.group(3))))
    return {"unpacked": unpacked, "present": present, "fixed": fixed, "identifiers": identifiers}


async def released_copies(box, repo: str, solution: str) -> dict:
    """Released copies of the removed code in archives, package caches and package-named directories on C (in `box`;
    see LEAK_ARCHIVES): leak_problems on the distinctive lines and identifier_problems on the distinctive identifiers
    of the solution's code files (solution_identifiers: the names it declares and its string literals; a name it only
    references or calls is never scanned, "referenced"), wherever they lie (a release's README or changelog repeating
    a documentation file is no leak), and the same per released package of the repo (package_rows).
    {"problems", "incomplete", "names", "selection", "scan", "lines", "identifiers", "absent", "referenced",
    "doc_lines", "seconds", "package_lines", "package_identifiers"}."""
    tokens_all = leak_tokens(solution)
    per_source = {source: lines for source, lines in tokens_all.items() if CODE_FILE.search(source)}
    docs = {source: lines for source, lines in tokens_all.items() if source not in per_source}
    items = solution_identifiers(solution)
    report = {"problems": [], "incomplete": [], "names": {}, "selection": {}, "scan": {}, "lines": [],
              "identifiers": [], "absent": {}, "referenced": referenced_identifiers(solution), "doc_lines": [],
              "seconds": {}}
    if not per_source and not items:
        return report
    outputs = {}
    for step, script, cwd, timeout in (("manifest", ARCHIVE_MANIFEST, "/app", 300),
                                       ("discover", ARCHIVE_DISCOVER, "/", 900), ("scan", ARCHIVE_SCAN, "/", 1500)):
        if step == "discover":
            report["names"] = released_names(repo, outputs["manifest"])
            await box.put_text("".join(name + "\n" for name in report["names"]["names"]), "/tmp/leak-names")
        elif step == "scan":
            report["selection"] = selection = released_selection(outputs["discover"], report["names"])
            everything = {item for values in items.values() for item in values}
            uploads = {"select": [f"{kind}\t{path}" for kind, path, _ in selection["unpack"]],
                       "roots": selection["roots"],
                       "tokens": sorted({line for lines in tokens_all.values() for line in lines}),
                       "idents": sorted(item for item in everything if IDENTIFIER.fullmatch(item)),
                       "literals": sorted(item for item in everything if not IDENTIFIER.fullmatch(item))}
            for name, lines in uploads.items():
                await box.put_text("".join(line + "\n" for line in lines), f"/tmp/leak-{name}")
        started = time.monotonic()
        result = await box.sh(script, cwd=cwd, timeout=timeout, check=False)
        report["seconds"][step] = round(time.monotonic() - started, 1)
        outputs[step] = result.stdout
        if result.timed_out or not result.stdout.rstrip().endswith(ARCHIVE_DONE):
            note = f"{step} (exit {result.exit_code}, timed out {result.timed_out})"
            report["incomplete"].append(note)
            report["problems"].append(f"archive leak scan incomplete: {note}")
            return report
    report["scan"] = scan = parse_archive_scan(outputs["scan"])
    report["absent"] = absent = {source: [item for item in values if item not in scan["present"]]
                                 for source, values in items.items()}
    absent = {source: values for source, values in absent.items() if values}
    wanted_items = {item for values in absent.values() for item in values}
    literals = [item for item in wanted_items if not IDENTIFIER.fullmatch(item)]
    tokens = {line for lines in tokens_all.values() for line in lines}
    lines, names = {}, {}
    for path, found in scan["fixed"].items():
        for number, text, offset in found:
            if text in tokens:
                lines.setdefault(path, []).append((number, text))
            names.setdefault(path, []).extend((number, item, offset + text.index(item)) for item in literals
                                              if item in text)
    for path, found in scan["identifiers"].items():
        names.setdefault(path, []).extend(found)
    added, windows = added_counts(solution), change_windows(solution)
    line_windows = {source: max(60, added.get(source, 0)) for source in tokens_all}
    report["lines"] = densest(lines, per_source, line_windows)
    report["doc_lines"] = [row for row in densest(lines, docs, line_windows) if row["best"] >= row["need"]]
    report["identifiers"] = densest(names, absent, windows)
    per_file = leak_problems(lines, per_source, added) + identifier_problems(names, absent, windows)
    dirs = report["selection"].get("package_dirs", [])  # hengyuf 2026-10-20: and per released package of the repo
    report["package_lines"] = package_rows(lines, per_source, report["names"], dirs, minimum=4)
    report["package_identifiers"] = package_rows(names, absent, report["names"], dirs, minimum=3)
    report["problems"] = sorted(set(per_file + package_problems(report["package_lines"], per_file, unit="lines")
                                    + package_problems(report["package_identifiers"], per_file, unit="identifiers")))
    return report


def ignore_rule_edits(start_patch: str) -> list:
    """.gitignore files that start.patch changes. MATERIALIZE deletes ignored build outputs by those rules, so an edit
    can delete arbitrary files from C, e.g. un-ignore node_modules to erase released copies the leak scan would catch
    (official1-testem c03, 2026-10-18), and it leaves a hint in the student's tree."""
    paths = re.findall(r"^diff --git a/(\S+) b/", start_patch, re.M)
    return sorted({path for path in paths if path == ".gitignore" or path.endswith("/.gitignore")})


GO_TEST_FUNC = re.compile(r"^([+-])func (Test\w+)\(")


def rehomed_go_tests(start_patch: str, test_patch: str) -> list:
    """Go test functions that start.patch removes from a file X and test.patch re-adds in another file while leaving X
    alone. The grader resets only the files test.patch touches, so a student's same-named test left in X redeclares the
    function and the whole package fails to compile (official1-arcane cand-c04 and the latent cand-c02, 2026-10-18)."""
    def functions(patch, sign):
        found, current = {}, None
        for line in patch.splitlines():
            if line.startswith("diff --git "):
                current = line.split(" b/", 1)[-1]
            elif current and (match := GO_TEST_FUNC.match(line)) and match.group(1) == sign:
                found.setdefault(match.group(2), set()).add(current)
        return found
    removed, added = functions(start_patch, "-"), functions(test_patch, "+")
    touched = set(re.findall(r"^diff --git a/\S+ b/(\S+)", test_patch, re.M))
    problems = set()
    for name, sources in removed.items():
        for source in sources - touched:
            for target in added.get(name, set()) - {source}:
                if Path(target).parent == Path(source).parent:
                    problems.add((source, target, name))
    return sorted(problems)


def hint_lines(start_patch: str) -> list:
    return [line for line in start_patch.splitlines()
            if line.startswith("+") and not line.startswith("+++") and HINT.search(line)]


# Stale and vanished tests on C (stale_tests; hengyuf 2026-10-20). Test-ID formats of this pipeline's reports, checked
# on every saved env baseline and c-baseline: Go `<import path>.<TestName>[/<subtest>]` (CTRF suite.name); pytest JUnit
# `<file path with dots>[.<Class>].<test>[<params>][@<xdist group>]`; pytest node IDs `<file>.py::<test>`; JS/TS
# `<file>: <describe> > <test>` (junit-to-ctrf) and `<file> <full name>` (mocha-ctrf). Other IDs (vitest's
# `build: yarn build:js`) belong to no file or package: never guessed.
JS_TEST_ID = re.compile(r"^(?P<file>[^\s:]*[^\s:/]\.[cm]?[jt]sx?)(?::\s|\s|$)")
PY_NODE_ID = re.compile(r"^(?P<file>[^\s:\[]+\.py)::")
GO_TEST_ID = re.compile(r"^(?P<package>[^\s\[]+?/[^\s\[]*?)\.(?:Test|Example|Fuzz|Benchmark)\w*(?:/|$)")
PY_JUNIT_ID = re.compile(r"^(?P<path>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\.[A-Za-z_]\w*(?=$|\[|@)")
STALE_PREFIX = ("tests exercising the removed behaviour remain in the starting state and fail there; delete them in "
                "start.patch and move them into test.patch:")
VANISHED_PREFIX = ("tests that pass on A produce no results at all on the starting state although their test files "
                   "exist: the package/file probably fails to build, import or start there; start.patch must leave the "
                   "existing tests compiling")
VANISH_MIN = 3  # stable tests on A before the silence of a file or package on C counts
PROBLEM_CHARS = 2000  # one grouped problem stays below this length
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
GO_MODULES = r"""set -u
cd /app || exit 0
find . \( -name .git -o -name node_modules -o -name vendor -o -name testdata \) -prune -o -name go.mod -type f -print \
  2>/dev/null | while IFS= read -r f; do
  m="$(sed -n 's|^module[[:space:]][[:space:]]*||p' "$f" | head -1 | sed 's|//.*||' | tr -d '"[:space:]')"
  d="${f%/go.mod}"; d="${d#./}"
  if [ -n "$m" ]; then printf 'M\t%s\t%s\n' "$m" "$d"; fi
done
printf 'E\tdone\n'"""
# /tmp/vanished-groups: `go<TAB><package dir>` / `py<TAB><file>` / `js<TAB><file>`; /tmp/vanished-imports: quoted import
# paths of the Go packages start.patch changes. F: a test file that still defines tests; I: a Go file importing one.
VANISHED_CHECK = r"""set -u
cd /app || exit 0
tab="$(printf '\t')"
while IFS="$tab" read -r kind path; do
  case "$kind" in
    go) for f in "$path"/*_test.go; do
          if [ -f "$f" ] && grep -qE '^func (Test|Example|Fuzz)' "$f"; then printf 'F\t%s\t%s\n' "$path" "$f"; fi
        done
        if [ -s /tmp/vanished-imports ]; then
          for f in "$path"/*.go; do
            if [ -f "$f" ] && grep -qF -f /tmp/vanished-imports "$f"; then printf 'I\t%s\t%s\n' "$path" "$f"; fi
          done
        fi ;;
    py) if [ -f "$path" ] && grep -qE '^[[:space:]]*((async[[:space:]]+)?def test|class Test)' "$path"; then
          printf 'F\t%s\t%s\n' "$path" "$path"; fi ;;
    js) if [ -f "$path" ] && grep -qE '(^|[^A-Za-z0-9_$.])(it|test|describe)(\.[A-Za-z]+)*[[:space:]]*[(`]' "$path"; then
          printf 'F\t%s\t%s\n' "$path" "$path"; fi ;;
  esac
done < /tmp/vanished-groups
printf 'E\tdone\n'"""


def report_group(test_id: str):
    """("go", import path) | ("py", file) | ("js", file) of a test ID; None for an unknown format (never guessed).
    Go first: a subtest may be named after a file (`TestBundle/entry.js`), and no other format has `.Test...` there."""
    if match := GO_TEST_ID.match(test_id):
        return ("go", match["package"])
    if match := JS_TEST_ID.match(test_id):
        return ("js", match["file"])
    if match := PY_NODE_ID.match(test_id):
        return ("py", match["file"])
    if match := PY_JUNIT_ID.match(test_id):  # JUnit classname: the file path with dots, then capitalised class names
        parts = match["path"].split(".")
        while len(parts) > 1 and parts[-1][:1].isupper():
            parts.pop()
        return ("py", "/".join(parts) + ".py")
    return None


def _in_group(test_id: str, group) -> str:
    """A test ID without its file or package."""
    kind, key = group
    if kind == "go":
        return test_id[len(key) + 1:]
    if kind == "js":
        return test_id[len(key):].lstrip(":").strip() or test_id
    if "::" in test_id:
        return test_id.split("::", 1)[1]
    module = key[:-3].replace("/", ".") + "."
    return test_id[len(module):] if test_id.startswith(module) else test_id


def _clip(text: str, limit: int) -> str:
    """At most `limit` characters, keeping both ends (Go subtests end in the distinctive case name)."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return text[:head - 2] + "..." + text[-(limit - head - 1):]


def _examples(test_ids, group, count: int = 2) -> list:
    """Up to `count` leaf tests (a Go parent fails with its subtests), from different top-level tests if possible."""
    names = sorted({_in_group(test_id, group) for test_id in test_ids})
    parents = {name[:i] for name in names for i, char in enumerate(name) if char == "/"}
    leaves = [name for name in names if name not in parents] or names
    picked = []
    for name in leaves:
        if name.split("/")[0] not in {other.split("/")[0] for other in picked}:
            picked.append(name)
            if len(picked) == count:
                return picked
    return picked + [name for name in leaves if name not in picked][:count - len(picked)]


def group_summary(groups: dict, notes: dict = None, *, unit: str = "tests", budget: int = 1800,
                  limit: int = 10) -> str:
    """`<file or package> (<n> <unit>, e.g. <a>; <b>[; <note>])` for up to `limit` groups, largest first, within about
    `budget` characters. `groups` maps (kind, key) -> test IDs; ("id", <test ID>) is a test of unknown format."""
    notes = notes or {}
    order = sorted(groups, key=lambda group: (-len(groups[group]), group))
    parts = []
    for group in order[:limit]:
        if group[0] == "id":
            entry = _clip(group[1], 160)
        else:
            examples = "; ".join(_clip(name, 100) for name in _examples(groups[group], group))
            note = f"; {notes[group]}" if notes.get(group) else ""
            count = len(groups[group])
            entry = f"{_clip(group[1], 120)} ({count} {'test' if unit == 'tests' and count == 1 else unit}, " \
                    f"e.g. {examples}{note})"
        if parts and sum(len(part) + 2 for part in parts) + len(entry) > budget:
            break
        parts.append(entry)
    if rest := order[len(parts):]:
        parts.append(f"and {len(rest)} more ({sum(len(groups[group]) for group in rest)} tests)")
    return "; ".join(parts)


def _files_packages(count: int) -> str:
    return f"{count} file/package" if count == 1 else f"{count} files/packages"


def stale_problems(stable: set, now: dict) -> list:
    """Tests stable on A that run on C without passing, grouped by file/package so that the author sees every group
    (official1-opa c04, 2026-10-20: attempt 0's list showed 10 internal/engine subtests of 245 stale tests in 4
    packages; attempt 1 fixed the engine and lost the candidate on internal/server). An unknown ID is its own group."""
    stale = sorted(test_id for test_id in stable if test_id in now and now[test_id] != "passed")
    if not stale:
        return []
    groups = {}
    for test_id in stale:
        groups.setdefault(report_group(test_id) or ("id", test_id), []).append(test_id)
    head = f"{STALE_PREFIX} {len(stale)} {'test' if len(stale) == 1 else 'tests'} in {_files_packages(len(groups))}: "
    return [head + group_summary(groups, budget=PROBLEM_CHARS - len(head) - 80)]


def vanished_groups(stable: set, on_a: set, now: dict) -> tuple:
    """Files/packages with >= VANISH_MIN stable tests on A for which C reports no test that ran: none that exists on A
    and none new that passed or skipped (a JS 'suite execution' or pytest collection-error entry is no run), and
    whether that holds for every file/package of the suite. ({(kind, key): stable test IDs}, everything_silent)."""
    alive = {report_group(test_id) for test_id, status in now.items() if test_id in on_a or status in ("passed", "skipped")}
    groups = {}
    for test_id in stable:
        if group := report_group(test_id):
            groups.setdefault(group, []).append(test_id)
    vanished = {group: sorted(tests) for group, tests in groups.items()
                if group not in alive and len(tests) >= VANISH_MIN}
    return vanished, bool(vanished) and not alive & set(groups)


def go_modules(listing: str) -> dict:
    """{module path: repository directory} from GO_MODULES output."""
    modules = {}
    for line in listing.splitlines():
        fields = line.split("\t")
        if len(fields) == 3 and fields[0] == "M" and fields[1]:
            modules.setdefault(fields[1], posixpath.normpath(fields[2] or "."))
    return modules


def go_package_dir(package: str, modules: dict):
    """Repository directory of an import path, by the longest module path that prefixes it; None for a package of
    another module (official1-dasel's broad suite also tests github.com/hashicorp/hcl from the module cache)."""
    module = max((m for m in modules if package == m or package.startswith(m + "/")), key=len, default=None)
    if module is None:
        return None
    return posixpath.normpath(posixpath.join(modules[module], package[len(module):].lstrip("/")))


def go_changes(start_patch: str, modules: dict) -> tuple:
    """What start.patch changes for Go: (directories with a changed file, a changed testdata/ file counting for its
    package; quoted import paths of the packages whose non-test .go files change; whether it changes go.mod, go.sum,
    go.work or vendor/modules.txt)."""
    directories, imports, module_files = set(), set(), False
    for path in {path for pair in re.findall(r"^diff --git a/(\S+) b/(\S+)", start_patch, re.M) for path in pair}:
        name = posixpath.basename(path)
        module_files |= name in ("go.mod", "go.sum", "go.work", "go.work.sum") or path.endswith("vendor/modules.txt")
        directory = posixpath.dirname(path) or "."
        if fixture := "testdata" in (parts := directory.split("/")):
            directory = "/".join(parts[:parts.index("testdata")]) or "."
        directories.add(directory)
        if name.endswith(".go") and not name.endswith("_test.go") and not fixture:
            owners = [(d, m) for m, d in modules.items() if d == "." or directory == d or directory.startswith(d + "/")]
            if owners:
                owner_dir, module = max(owners, key=lambda item: 0 if item[0] == "." else len(item[0]))
                rest = posixpath.relpath(directory, owner_dir)
                imports.add(f'"{module}"' if rest == "." else f'"{module}/{rest}"')
    return directories, imports, module_files


def parse_vanished_check(output: str) -> tuple:
    """({path: test files that define tests}, {Go package dir: files importing a changed package}) of VANISHED_CHECK."""
    files, importers = {}, {}
    for line in output.splitlines():
        kind, _, rest = line.partition("\t")
        path, _, found = rest.partition("\t")
        if kind in ("F", "I") and path and found:
            found = found[2:] if found.startswith("./") else found
            (files if kind == "F" else importers).setdefault(path, []).append(found)
    return files, importers


def _error_line(message: str) -> str:
    """The most telling line of a failure message (pytest's last `E   ...` line, else the first error-like line)."""
    lines = [line.strip() for line in ANSI_ESCAPE.sub("", message or "").splitlines()]
    lines = [line for line in lines if line and line != "collection failure" and "Test suite failed to run" not in line]
    pick = (next((line for line in reversed(lines) if line.startswith("E ")), None)
            or next((line for line in lines if re.search(r"error|cannot|not found|undefined", line, re.I)), None)
            or (lines[0] if lines else "failed"))
    return _clip(re.sub(r"^E\s+", "", pick), 160)


def failure_evidence(groups, on_a: set, report: dict) -> dict:
    """{group: '<entry>: <error line>'} from C's own failure entries for vanished files: a JS 'suite execution' / 'Test
    suite failed to run' entry, or a pytest collection error (an entry named by the module's dotted path)."""
    modules = {key[:-3].replace("/", "."): (kind, key) for kind, key in groups if kind == "py"}
    found = {}
    for test_id in sorted(report):
        status, message = report[test_id]
        if test_id in on_a or status != "failed":
            continue
        group = report_group(test_id)
        group = group if group in groups else modules.get(test_id)
        if group and group not in found:
            found[group] = f"C reports {_clip(_in_group(test_id, group), 60)!r}: {_error_line(message)}"
    return found


def vanished_problems(groups: dict, files: dict, evidence: dict = None, *, silent: bool = False) -> list:
    """One problem for the vanished groups whose test files still exist and define tests on C (`files`); a group
    whose test files start.patch deleted, or that lost all its tests, is fine."""
    confirmed = {group: tests for group, tests in groups.items() if files.get(group)}
    if not confirmed:
        return []
    evidence, notes = evidence or {}, {}
    for group in confirmed:
        shown = ", ".join(_clip(path, 80) for path in files[group][:3])
        more = f" and {len(files[group]) - 3} more" if len(files[group]) > 3 else ""
        notes[group] = "; ".join(filter(None, [f"test files {shown}{more}", evidence.get(group)]))
    head = VANISHED_PREFIX + (" (no file or package of the broad suite produced any result, so start.patch probably "
                              "breaks the build or the test runner itself)" if silent else "") + ": "
    return [head + group_summary(confirmed, notes, unit="pass on A", budget=PROBLEM_CHARS - len(head) - 80)]


def test_sh(block: str, frame_task: Path) -> str:
    """The shared DeepSWE verifier frame (byte-identical in all tasks) around a task-specific RUN TESTS block."""
    frame = (frame_task / "tests/test.sh").read_text()
    head, tail = frame.split(RUN_START)[0], frame.split(RUN_END)[1]
    return f"{head}{RUN_START}\n{block.strip()}\n{RUN_END}{tail}"


# Tests build_env's broad suite left out (hengyuf 2026-10-20, official1-dynamodb: its run_tests.sh excluded 3 wall-clock
# 'timeout' tests that assert <= +50 ms with Jest --testNamePattern, so both env baselines report them as skipped;
# round-1 authors' RUN TESTS ran them and 23 of 25 tasks graded them P2P). The env stage keeps no structured list of
# exclusions: stage.json "flaky" holds only tests that passed in one of its two baselines, the reason is free text
# (result.json "summary", run_tests.sh comments), and a filter that deselects tests (pytest -k, go -run/-skip, nextest
# filters) leaves no trace in the reports. env_unreliable() takes what the baselines do show: tests skipped in every
# baseline that reports them, or passing in only one. calibrate() keeps them out of P2P (and out of "base suite fails").
OCCURRENCE_SUFFIX = re.compile(r"\s*\[occurrence \d+\]$")  # build_env's de-duplication of repeated test names


def env_unreliable(b: Build) -> set:
    """IDs (build_env's format) of the tests that the env baselines show as excluded or flaky (see above)."""
    env = b.root / "env"
    reports = [parse_report(env / f"baseline-{i}", b.grade_cfg) for i in (0, 1) if (env / f"baseline-{i}").exists()]
    unreliable = set()
    for test in set().union(*reports) if reports else ():
        statuses = [report.get(test) for report in reports]
        if all(status in (None, "skipped") for status in statuses) or (
                "passed" in statuses and set(statuses) != {"passed"}):
            unreliable.add(test)
    return unreliable


def _test_words(test_id: str) -> tuple:
    """(file or package base name or "", lower-case title words) of a test ID: the authors' RUN TESTS blocks name the
    same Jest test 'tests/jest/__tests__/index.test.tsx: actions > timeout > returns timeout error', '__tests__/index.
    test.tsx: actions timeout returns timeout error' or 'actions: actions timeout returns timeout error'."""
    text = OCCURRENCE_SUFFIX.sub("", test_id.strip())
    group = report_group(text)
    file, rest = (posixpath.basename(group[1]), _in_group(text, group)) if group else ("", text)
    return file, tuple(re.findall(r"[^\s>›:/.]+", rest.lower()))


def env_matches(test_ids, unreliable) -> list:
    """The test IDs (any report format) that name one of the `unreliable` env tests: the same file (when both IDs name
    one) and the same title words, or the env test's title (>= 3 words) as the end of the ID's title."""
    index = {}
    for test in unreliable:
        file, words = _test_words(test)
        if words:
            index.setdefault(words[-1], []).append((file, words))
    found = []
    for test in test_ids:
        file, words = _test_words(test)
        for other_file, other in index.get(words[-1] if words else "", []):
            if (not file or not other_file or file == other_file) and (
                    words == other or (len(other) >= 3 and words[-len(other):] == other)):
                found.append(test)
                break
    return found


def calibrate(empty: dict, solved: list, reports: list, unreliable=()) -> tuple:
    """SWE-rebench style F2P/P2P from the starting state (empty patch) vs the reference solution runs. Base-suite tests
    matching `unreliable` (env_unreliable: excluded or flaky in build_env's suite) never enter P2P and are not failures
    of the starting state (hengyuf 2026-10-20)."""
    base, new = Path(reports[0]).name, Path(reports[1]).name
    problems = []
    excluded = set(env_matches(empty["suites"].get(base, {}), unreliable)) if unreliable else set()
    # A SKIPPED test is not a failure (hengyuf, 2026-10-18: aiohttp skips partitioned-cookie and pre-3.11 SSL tests on
    # Python 3.12, and 16 of 25 httpx round-1 branches lost an attempt to "base suite fails" on 4 skips). Skipped
    # tests never enter P2P (passed only) or F2P; a new-suite test is unstable only if it fails, or skips in some
    # solution runs and passes in others.
    broken = sorted(t for t, status in empty["suites"].get(base, {}).items()
                    if status == "failed" and t not in excluded)
    if broken:
        problems.append(f"base suite fails on the starting state: {broken[:10]}")
    seen = {}
    for run in solved:
        for t, status in run["suites"].get(new, {}).items():
            seen.setdefault(t, set()).add(status)
    unstable = sorted(t for t, statuses in seen.items() if statuses not in ({"passed"}, {"skipped"}))
    if unstable:
        problems.append(f"new-suite tests not passing with the solution in every run: {unstable[:10]}")
    solved_new = set.intersection(*({t for t, s in run["suites"].get(new, {}).items() if s == "passed"} for run in solved))
    f2p = sorted(t for t in solved_new if empty["suites"].get(new, {}).get(t) != "passed")
    p2p = sorted(t for t, status in empty["suites"].get(base, {}).items() if status == "passed" and t not in excluded
                 and all(run["suites"].get(base, {}).get(t) == "passed" for run in solved))
    if len(f2p) < 2:
        problems.append(f"only {len(f2p)} fail-to-pass tests")
    if len(p2p) < 1:
        problems.append("no pass-to-pass tests")
    return f2p, p2p, problems


def write_tests(tests: Path, b: Build, block: str, test_patch: str, base_commit: str, f2p=(), p2p=(), *,
                solution: str) -> None:
    """tests/ for the gates (provisional, final) and package(). hengyuf 2026-10-20: grader.py is the sanitizing grader
    (DeepSWE's grader until then, copied from the target task) and config.json carries its "test_paths" and
    "test_paths_mode" (Build.grading), so a submission's edits to test files never reach the graded run; "test_paths"
    also gets the directories of test.patch's files that the language globs miss (test_patch_globs)."""
    grading = b.grading  # first: an unknown repository language writes nothing
    extra, skipped = test_patch_globs(test_patch, solution, grading["test_paths"])
    if extra or skipped:
        log(f"{tests}: test_paths + {extra}" + "".join(f"; {directory} not added: {why}"
                                                       for directory, why in skipped.items()))
    tests.mkdir(parents=True, exist_ok=True)
    (tests / "test.sh").write_text(test_sh(block, b.target_dir))
    (tests / "test.patch").write_text(test_patch)
    shutil.copy2(SANITIZER, tests / "grader.py")
    write_json(tests / "config.json", {"base_commit": base_commit, "f2p_node_ids": list(f2p), "p2p_node_ids": list(p2p),
                                       "grade": dict(b.grade_cfg, reports=b.reports),
                                       **dict(grading, test_paths=grading["test_paths"] + extra)})


def _glob_literal(directory: str) -> str:
    """A directory as glob segments that match it literally (fnmatch's * ? [ in a name)."""
    return re.sub(r"([*?\[])", r"[\1]", directory)


# Directory names that hold test material only (grading_repair.TEST_DIR, plus e2e/, __fixtures__/, __mocks__/)
TEST_DIR_NAMES = frozenset({"test", "tests", "__tests__", "__typetests__", "testdata", "__snapshots__", "snapshots",
                            "fixture", "fixtures", "__fixtures__", "__mocks__", "e2e"})


def test_patch_globs(test_patch: str, solution: str, globs: list) -> tuple:
    """hengyuf 2026-10-20 (official1-anko: gpython's TestVm and TestPy run every script in vm/tests/, py/tests/ and
    stdlib/*/tests/, which the Go globs miss): "<dir>/**" for each directory in which test.patch adds or modifies a file
    that `globs` (the language defaults) do not cover, read with the sanitizing grader's parser. Never the repository
    root, not a directory holding a file solution.patch touches (the grader would revert the solution there), and only
    test directories (a TEST_DIR_NAMES segment): a source directory would revert every student's changes there (anko
    c01 test.patch edits symtable/make_symtable_test.py, a generator next to symtable.go that the c02/c04 solutions
    change; wasmi c03 edits inline tests in crates/debugger/src/lib.rs). Returns (extra globs, {directory: why not})."""
    grader = _load(SANITIZER, "sanitizing_grader")
    solution_paths = grader.touched_paths(solution)
    dirs, skipped = [], {}
    for block in re.split(r"(?m)^(?=diff --git )", test_patch):
        names = grader.touched_paths(block)
        if not block.startswith("diff --git ") or not names or re.search(r"(?m)^deleted file mode ", block):
            continue  # a deletion adds or modifies nothing
        path = names[-1]  # the file the block adds or modifies (a rename's or copy's new name)
        if any(grader.glob_match(glob, path) for glob in globs):
            continue
        directory = posixpath.dirname(path)
        if not directory:
            skipped.setdefault(".", f"the repository root ({path})")
            continue
        if not TEST_DIR_NAMES & set(directory.split("/")):
            skipped.setdefault(directory, f"not a test directory by name, maybe source ({path})")
            continue
        held = [p for p in solution_paths if grader.glob_match(_glob_literal(directory) + "/**", p)]
        if held:
            skipped.setdefault(directory, f"solution.patch touches {held[:3]} there")
        elif directory not in dirs:
            dirs.append(directory)
    dirs = [d for d in dirs if not any(d.startswith(other + "/") for other in dirs)]  # inside another one: covered
    return [_glob_literal(d) + "/**" for d in sorted(dirs)], skipped


# Test files by name in any language (Rust tests.rs / foo_test.rs, Python tests.py, JS foo.spec.ts, ...), as a
# complement to the language globs and TEST_DIR_NAMES; "testing.rs", "latest.go" or "contest.py" are not.
TEST_FILE_NAME = re.compile(r"(^|[._-])(test|tests|spec|specs)([._-][^/]*)?\.[A-Za-z0-9]+$|^conftest\.py$", re.I)
RUST_TEST_MOD = re.compile(r"#\[cfg\(test\)\]\s*mod\s+[A-Za-z_][A-Za-z0-9_]*\s*;")
SOURCE_EDIT_PREFIX = "test.patch modifies existing non-test files of the starting state"


@functools.lru_cache(maxsize=1)
def _sanitizer():
    """The sanitizing grader module (its patch parser and glob matcher), loaded once per process."""
    return _load(SANITIZER, "sanitizing_grader")


def is_test_file(path: str, globs: list) -> bool:
    """A test path for the gates: a config.json "test_paths" glob (language defaults + test_patch_globs), a directory
    named like a test directory (TEST_DIR_NAMES) or a test file name (TEST_FILE_NAME)."""
    grader = _sanitizer()
    parts = path.split("/")
    return (any(grader.glob_match(glob, path) for glob in globs) or bool(TEST_DIR_NAMES & set(parts[:-1]))
            or bool(TEST_FILE_NAME.search(parts[-1])))


def adds_only_test_module(block: str) -> bool:
    """A Rust file diff whose hunks only ADD `#[cfg(test)] mod <name>;` declarations (on one line or two; blank lines
    aside): the declaration a new test-only module file needs in its parent module."""
    body = block.split("\n@@", 1)[1] if "\n@@" in block else ""
    lines = body.splitlines()[1:]
    if any(line.startswith("-") for line in lines):
        return False
    added = " ".join(line[1:].strip() for line in lines if line.startswith("+") and line[1:].strip())
    return bool(added) and not RUST_TEST_MOD.sub("", added).strip()


def test_patch_source_edits(test_patch: str, globs: list) -> list:
    """hengyuf 2026-10-20 (official1-wasmi c03: test.patch added a test to the inline `mod test` of
    crates/debugger/src/lib.rs): files that already exist on the starting state, lie outside the test paths
    (is_test_file) and that test.patch modifies, deletes or renames. Both graders reset every file test.patch touches
    before applying it, so a solution's own edits to such a source file are lost before grading. A Rust file whose diff
    only adds `#[cfg(test)] mod <name>;` lines is allowed (a new test module file needs that declaration); the residual
    risk is the same reset: a submission's other edits to that file (e.g. a new `pub mod` in a crate root) are lost
    too."""
    grader = _sanitizer()
    found = []
    for block in re.split(r"(?m)^(?=diff --git )", test_patch):
        names = grader.touched_paths(block)
        if not block.startswith("diff --git ") or not names or re.search(r"(?m)^(new file mode|copy from) ", block):
            continue  # a new file (or a copy) does not exist on the starting state
        path = names[0]  # the existing file (a rename's old name)
        if is_test_file(path, globs) or (path.endswith(".rs") and adds_only_test_module(block)):
            continue
        if path not in found:
            found.append(path)
    return found


def restored_paths(patch: str, globs: list) -> list:
    """The paths of a patch (old and new names) that match a test-path glob, read with the sanitizing grader's own
    parser and matcher: the grader puts them back to base_commit (or deletes them) before grading, so a reference
    solution must not touch them."""
    grader = _load(SANITIZER, "sanitizing_grader")
    return [path for path in grader.touched_paths(patch) if any(grader.glob_match(glob, path) for glob in globs)]


def validate_task(task_dir: Path) -> None:
    """Gym's own DeepSWE task loader must accept the task (ATP's Gym and the swe-rl Gym copy)."""
    for gym in (GYM, SWE_RL_GYM):
        if not (gym / "resources_servers/deepswe/task_store.py").exists():
            continue
        code = ("import sys; sys.dont_write_bytecode = True; sys.path.insert(0, sys.argv[1]); "
                "from resources_servers.deepswe.task_store import DeepSWETaskStore; "
                "DeepSWETaskStore(sys.argv[2], expected_task_count=1)")
        result = subprocess.run([sys.executable, "-B", "-c", code, str(gym), str(task_dir.parent)],
                                capture_output=True, text=True)
        if result.returncode:
            raise GateError(f"Gym task_store ({gym}) rejects {task_dir.name}: {result.stderr.strip()[-600:]}")


# ----------------------------------------------------------------------------------------------- stages

async def build_env(b: Build) -> dict:
    """Image A: the similar repo at its default-branch HEAD inside the target's DeepSWE image, deps installed."""
    out = b.root / "env"
    if done := read_json(out / "stage.json"):
        return done
    (out / "reference").mkdir(parents=True, exist_ok=True)
    for name in ("environment/Dockerfile", "tests/test.sh", "tests/config.json"):
        shutil.copy2(b.target_dir / name, out / "reference" / Path(name).name)
    url = f"https://github.com/{b.repo}"
    if checkpoint := read_json(out / "a.json"):  # image A exists; only its gate is pending
        try:
            baseline = await env_gate(b, checkpoint["snapshot"], checkpoint["SHA"], out)
        except GateError:
            (out / "a.json").unlink()
            (out / "result.json").unlink(missing_ok=True)  # the next run rebuilds A from scratch
            raise
        stage = {"snapshot": checkpoint["snapshot"], "sha": checkpoint["SHA"], "branch": checkpoint["BRANCH"],
                 "repo_url": url, **baseline}
        write_json(out / "stage.json", stage)
        return stage
    (out / "result.json").unlink(missing_ok=True)  # the agent must prepare the sandbox that gets snapshotted
    async with AGENTS, owned(b.image, run=b.run, purpose="build-env", network="allow") as box:
        cloned = (await box.sh(CLONE.format(url=url), cwd="/")).stdout.splitlines()
        facts = dict(line.split("=", 1) for line in cloned if line.startswith(("SHA=", "BRANCH=")))
        write_wrappers(out / "bin", box.sid)
        prompt = render("build_env", repo=b.repo, target=b.target, language=b.language, sha=facts["SHA"],
                        branch=facts["BRANCH"], base_image=b.image, report_format=b.grade_cfg["format"],
                        grade_cfg=json.dumps(b.grade_cfg),
                        sandbox=render("_sandbox", network_note="(network access is enabled for this preparation step only)"))
        await agent(prompt, out, backend="codex", check=_needs(out, "setup.sh", "environment/Dockerfile", "run_tests.sh"))
        await box.sh("rm -rf /tmp/work", cwd="/")
        snapshot = await box.snapshot(f"{b.run}-A-{b.repo}")
        b.remember(snapshot, "A")
    write_json(out / "a.json", {"snapshot": snapshot, **facts})
    return await build_env(b)  # gate the checkpointed image A


async def env_gate(b: Build, snapshot: str, sha: str, out: Path) -> dict:
    passing = []
    async with owned(snapshot, run=b.run, purpose="env-gate") as box:
        head = (await box.sh("git rev-parse HEAD")).stdout.strip()
        dirty = (await box.sh("git status --porcelain")).stdout.strip()
        await box.put(out / "run_tests.sh", "/tmp/run_tests.sh")
        for attempt in range(2):
            await box.sh(f"REPORT=/tmp/baseline-{attempt} bash /tmp/run_tests.sh", timeout=2400, check=False)
            await box.get(f"/tmp/baseline-{attempt}", out / f"baseline-{attempt}")
            passing.append({t for t, status in parse_report(out / f"baseline-{attempt}", b.grade_cfg).items()
                            if status == "passed"})
    stable = passing[0] & passing[1]
    problems = ([f"HEAD {head} != {sha}"] if head != sha else []) + ([f"dirty tree: {dirty[:300]}"] if dirty else [])
    if len(stable) < 20:
        problems.append(f"only {len(stable)} tests pass reliably offline")
    if problems:
        raise GateError("env gate: " + "; ".join(problems))
    return {"baseline_passing": len(stable), "flaky": sorted(passing[0] ^ passing[1])}


async def understand(b: Build, env: dict) -> Path:
    out = b.root / "understanding"
    if not read_json(out / "stage.json"):
        async with AGENTS, owned(env["snapshot"], run=b.run, purpose="understand") as box:
            write_wrappers(out / "bin", box.sid)
            prompt = render("understanding", repo=b.repo, sandbox=render("_sandbox", network_note="(no network)"))
            await agent(prompt, out, backend="codex", check=_needs(out, "understanding.md", min_bytes=2000))  # was cursor
        write_json(out / "stage.json", {"ok": True})
    return out / "understanding.md"


async def plan(b: Build, env: dict, understanding: Path, count: int) -> list:
    out = b.root / "plan"
    if done := read_json(out / "stage.json"):
        return done["candidates"]
    inputs = out / "inputs"
    (inputs / "target").mkdir(parents=True, exist_ok=True)
    shutil.copy2(understanding, inputs / "understanding.md")
    shutil.copy2(b.fa_dir / "failure_analysis.md", inputs / "target/failure_analysis.md")
    shutil.copy2(b.fa_dir / "result.json", inputs / "target/failure_analysis.json")
    shutil.copy2(b.target_dir / "instruction.md", inputs / "target/instruction.md")
    config = json.loads((b.target_dir / "tests/config.json").read_text())
    write_json(inputs / "target/stats.json", {
        "solution": patch_stats((b.target_dir / "solution/solution.patch").read_text()),
        "tests": patch_stats((b.target_dir / "tests/test.patch").read_text()),
        "f2p_tests": len(config["f2p_node_ids"]), "p2p_tests": len(config["p2p_node_ids"])})

    def check(result):
        problems = _needs(out, "mask_plan.md", "candidates.json")(result)
        data = read_json(out / "candidates.json")
        candidates = data.get("candidates") if isinstance(data, dict) else None
        if not isinstance(candidates, list) or len(candidates) < count:
            return problems + [f"candidates.json must list {count} candidates"]
        for candidate in candidates:
            missing = {"id", "title", "summary", "files", "tests", "target_patterns"} - set(candidate)
            if missing:
                problems.append(f"candidate {candidate.get('id')!r} lacks {sorted(missing)}")
        return problems

    async with AGENTS, owned(env["snapshot"], run=b.run, purpose="plan") as box:
        write_wrappers(out / "bin", box.sid)
        prompt = render("mask_plan", n=count, repo=b.repo, target=b.target, student=STUDENT, teacher=TEACHER,
                        sandbox=render("_sandbox", network_note="(no network)"))
        await agent(prompt, out, backend="codex", check=check)
        candidates = read_json(out / "candidates.json")["candidates"]
        files = sorted({path for candidate in candidates for path in candidate["files"]})
        missing = set((await box.sh("for f in " + " ".join(map(shlex.quote, files)) +
                                    '; do test -e "$f" || echo "$f"; done', check=False)).stdout.splitlines())
    usable = [candidate for candidate in candidates if not missing & set(candidate["files"])]
    if missing:
        log(f"plan: dropped {len(candidates) - len(usable)} candidates citing missing files {sorted(missing)[:5]}")
    if not usable:
        (out / "result.json").unlink(missing_ok=True)  # re-plan on the next run
        raise GateError(f"every planned candidate cites files missing in A: {sorted(missing)[:10]}")
    write_json(out / "stage.json", {"candidates": usable})
    return usable


async def author(b: Build, env: dict, out: Path, *, kind: str, title: str, plan_md: Path, feedback: str) -> None:
    """Codex writes the task as files only (start/solution/test patches, RUN TESTS block, instruction draft)."""
    inputs = out / "inputs"
    (inputs / "reference").mkdir(parents=True, exist_ok=True)
    shutil.copy2(plan_md, inputs / "plan.md")
    shutil.copy2(b.root / "understanding/understanding.md", inputs / "understanding.md")
    shutil.copy2(b.fa_dir / "failure_analysis.md", inputs / "target_failure_analysis.md")
    for name in ("tests/test.sh", "tests/config.json", "instruction.md"):
        shutil.copy2(b.target_dir / name, inputs / "reference" / Path(name).name)

    def check(result):
        for name in ("start.patch", "solution.patch", "test.patch"):  # hengyuf: grader-readable labels first
            path = out / name
            if path.exists() and (fixed := canonical_labels(text := path.read_text())) != text:
                path.with_name(name + ".orig-labels").write_text(text)
                path.write_text(fixed)
        problems = _needs(out, "solution.patch", "test.patch", "run_tests.sh", "instruction_draft.md")(result)
        if not (out / "start.patch").exists() or (kind == "mask" and not (out / "start.patch").stat().st_size):
            problems.append("start.patch missing (it may only be empty for evolve tasks)")
        if not problems:
            overlap = (set(grader_module().patch_paths((out / "solution.patch").read_text()))
                       & set(grader_module().patch_paths((out / "test.patch").read_text())))
            if overlap:
                problems.append(f"solution.patch and test.patch both touch {sorted(overlap)[:5]}")
        return problems

    rules = {"mask": ("Mid-IID: remove the existing feature described in the plan. start.patch deletes its "
                      "implementation (keeping surviving entry points compiling) and removes the existing tests that "
                      "reveal it; those tests (possibly improved) go into test.patch; solution.patch restores the "
                      "implementation.", "the feature removed"),
             "evolve": ("High-IID: author the NEW feature described in the plan on top of A. start.patch may add "
                        "neutral scaffolding or be empty; solution.patch implements the feature; test.patch adds the "
                        "verifier tests.", "A plus optional neutral scaffolding; may be empty")}[kind]
    async with AGENTS, owned(env["snapshot"], run=b.run, purpose="author") as box:
        write_wrappers(out / "bin", box.sid)
        prompt = render("author", title=title, repo=b.repo, kind=kind, kind_rule=rules[0], start_rule=rules[1],
                        sha=env["sha"], base_report=b.reports[0], new_report=b.reports[1],
                        grade_cfg=json.dumps(dict(b.grade_cfg, reports=b.reports)),
                        feedback=f"A previous attempt failed these controller checks; fix them:\n{feedback}" if feedback else "",
                        sandbox=render("_sandbox", network_note="(no network)"))
        await agent(prompt, out, backend="codex", check=check)


async def materialize(b: Build, env: dict, start_patch: Path, name: str) -> dict:
    """Controller-only A -> C (the student's starting state) as a fresh snapshot."""
    async with owned(env["snapshot"], run=b.run, purpose="materialize") as box:
        await box.put(start_patch, "/tmp/start.patch")
        await box.put_text(MATERIALIZE, "/tmp/materialize.sh")
        base = (await box.sh("bash /tmp/materialize.sh /tmp/start.patch && rm -f /tmp/start.patch /tmp/materialize.sh"
                             )).stdout.strip().splitlines()[-1]
        snapshot = await box.snapshot(f"{b.run}-C-{name}")
        b.remember(snapshot, "C")
    return {"snapshot": snapshot, "base_commit": base}


def removes_solution_code(start_patch: str, solution_patch: str) -> bool:
    """Does start.patch delete lines from a file that solution.patch changes (a mask task)? Only then can a leak-scan hit
    inside /app be a remnant of the removed code. Otherwise (evolve tasks: start.patch empty or unrelated) the solution
    is NEW code, and a hit inside /app is existing code it resembles, e.g. official1-absslices c04-r1b4 (2026-10-19): the
    new op.SliceStep VM case mirrored its sibling op.Slice case and failed one attempt."""
    deleted, current = set(), None
    for line in start_patch.splitlines():
        if line.startswith("--- "):
            current = line[6:] if line.startswith("--- a/") else None
        elif current and line.startswith("-"):
            deleted.add(current)
    touched = {line[6:] for line in solution_patch.splitlines() if line.startswith("+++ b/")}
    return bool(deleted & touched)


async def leak_scan(b: Build, snapshot: str, solution: str, *, app_hits: bool = True) -> list:
    """Any file on C's filesystem holding at least half of one solution file's distinctive lines is a leak, wherever it
    lies, and so is a released copy of the removed code in an archive or package cache (released_copies, hengyuf
    2026-10-18; its identifier evidence is what the solution declares, and its string literals, hengyuf 2026-10-20)."""
    per_source = leak_tokens(solution)
    tokens = sorted({line for lines in per_source.values() for line in lines})
    if not tokens and not solution_identifiers(solution):
        return []
    result = None
    async with owned(snapshot, run=b.run, purpose="leak-scan") as box:  # disposable: the token file dies with it
        if tokens:
            await box.put_text("\n".join(tokens) + "\n", "/tmp/leak-tokens")
            name = shlex.quote(b.repo.split("/")[1].lower())
            result = await box.sh(LEAK_SCAN.format(name=name), cwd="/", timeout=2400, check=False)
        report = await released_copies(box, b.repo, solution)
    released = report["problems"]
    if result is not None and (result.timed_out or result.exit_code != 0):
        return sorted(set(released) | {f"leak scan incomplete (exit {result.exit_code}, timed out {result.timed_out})"})
    found = {}
    for hit in result.stdout.splitlines() if result is not None else []:
        match = re.match(r"(.*?):(\d+):(.*)$", hit)  # grep -n -o: path:line:token
        if match:
            found.setdefault(match.group(1), []).append((int(match.group(2)), match.group(3).strip()))
    if not app_hits:  # see removes_solution_code: /app only holds existing code the new solution resembles
        found = {path: hits for path, hits in found.items() if not path.startswith("/app/")}
    per_file = leak_problems(found, per_source, added_counts(solution))
    names = report.get("names") or {}
    if names.get("unparsed"):  # hengyuf 2026-10-20: archive discovery and the per-package rule may miss own packages
        log(f"leak scan ({b.run}): own package names may be incomplete: {'; '.join(names['unparsed'][:5])}"[:900])
    # hengyuf 2026-10-20: a released package of the repo (e.g. under node_modules) also counts as a whole
    code = {source: lines for source, lines in per_source.items() if CODE_FILE.search(source)}
    rows = package_rows(found, code, names, (report.get("selection") or {}).get("package_dirs", []), minimum=4)
    return sorted(set(per_file) | set(released) | set(package_problems(rows, per_file + released, unit="lines")))


async def vanished_files(box, groups: dict, start_patch: str, *, silent: bool, repo: str = "") -> dict:
    """{group: its test files on C (box) that still define tests, those start.patch edits first and marked} for the
    vanished groups. A Go package also needs a link to start.patch (a changed file in its directory, an import of a
    changed package, or changed module files) unless the whole suite is silent: go test streams per package, so a run
    cut short leaves later packages without results (official1-arcane: gitops-engine's `timeout 2m` on C's purged
    build cache, 2 of 25 candidates). pytest and Jest write their reports at the end: a cut run leaves none."""
    modules, dirs = {}, {}
    if any(kind == "go" for kind, _ in groups):
        listing = await box.sh(GO_MODULES, timeout=300, check=False)
        if not listing.timed_out and listing.stdout.rstrip().endswith(ARCHIVE_DONE):
            modules = go_modules(listing.stdout)
            if repo and "." not in modules.values():  # GOPATH-mode root (evanphx/json-patch v4): no go.mod there
                modules.setdefault(f"github.com/{repo}", ".")
            dirs = {key: path for kind, key in groups if kind == "go" and (path := go_package_dir(key, modules))}
    paths = {group: dirs.get(group[1]) if group[0] == "go" else group[1] for group in groups}
    paths = {group: path for group, path in paths.items() if path}
    if not paths:
        return {}
    changed_dirs, imports, module_files = go_changes(start_patch, modules)
    await box.put_text("".join(f"{group[0]}\t{path}\n" for group, path in sorted(paths.items())), "/tmp/vanished-groups")
    if imports:
        await box.put_text("".join(line + "\n" for line in sorted(imports)), "/tmp/vanished-imports")
    result = await box.sh(VANISHED_CHECK, timeout=300, check=False)
    if result.timed_out or not result.stdout.rstrip().endswith(ARCHIVE_DONE):
        log(f"vanished-test check incomplete (exit {result.exit_code}, timed out {result.timed_out}); not reported")
        return {}
    files, importers = parse_vanished_check(result.stdout)
    edited = {path for pair in re.findall(r"^diff --git a/(\S+) b/(\S+)", start_patch, re.M) for path in pair}
    return {group: [name + (" (edited by start.patch)" if name in edited else "")
                    for name in sorted(files[path], key=lambda name: (name not in edited, name))]
            for group, path in paths.items() if files.get(path) and (
                group[0] != "go" or silent or module_files or path in changed_dirs or importers.get(path))}


async def stale_tests(b: Build, cand: Path, mat: dict, start_patch: str = "") -> list:
    """Tests that passed reliably on A but fail on C exercise the removed behaviour: left in C they would show the
    student the hidden specification, so they must be deleted by start.patch and moved into test.patch. Test files and
    packages with tests passing on A that report nothing at all on C although their test files still exist did not
    build, import or start there (official1-opa c01, 2026-10-20: start.patch edited two test files of a Go package
    and left them not compiling), so the student would start from a broken suite (vanished_groups)."""
    env = b.root / "env"
    baselines = [parse_report(env / f"baseline-{i}", b.grade_cfg) for i in (0, 1)]
    stable = {t for t, status in baselines[0].items() if status == "passed" and baselines[1].get(t) == "passed"}
    on_a = set(baselines[0]) | set(baselines[1])
    async with owned(mat["snapshot"], run=b.run, purpose="stale-tests") as box:
        await box.put(env / "run_tests.sh", "/tmp/run_tests.sh")
        await box.sh("REPORT=/tmp/c-report bash /tmp/run_tests.sh", timeout=2400, check=False)
        if (await box.sh("test -s /tmp/c-report", check=False)).exit_code:
            return ["the broad test suite produced no report on the starting state (did start.patch break the build?)"]
        await box.get("/tmp/c-report", cand / "grade/c-baseline")
        report = grader_module().PARSERS[b.grade_cfg["format"]](str(cand / "grade/c-baseline"), b.grade_cfg)
        now = {t: status for t, (status, _message) in report.items()}
        vanished, silent = vanished_groups(stable, on_a, now)
        files = await vanished_files(box, vanished, start_patch, silent=silent, repo=b.repo) if vanished else {}
    return stale_problems(stable, now) + vanished_problems(vanished, files, failure_evidence(files, on_a, report),
                                                           silent=silent)


# Stale oracles outside the broad suite (hengyuf 2026-10-20, official1-wasmi c04: C kept wasmtime's wast conformance
# files for the transcoding start.patch removed -- tests/component-model/test/values/{transcode,strings}.wast and three
# tests/misc_testsuite/component-model/*.wast; they fail on C and every model used them as an oracle, but stale_tests()
# runs only build_env's broad suite, which does not include the `wast` target). stale_oracles() lists the test-path
# files on C (outside test.patch) that reference names or string literals that start.patch REMOVES from source files and
# that no non-test code file on C still contains: one removed name or message (a literal with a space, e.g. official1-
# promlabels c04's "invalid field value", vitest c05's "Jest: Coverage data for"), or two other literals (one key-like
# literal alone recurs in unrelated tests: an HTTP header name, an import path; checked on 108 built candidates).
ORACLE_PREFIX = ("test files that start.patch leaves in the starting state reference code it removes from source files "
                 "(possible stale oracles, e.g. spec/conformance files outside the broad suite that exercise the "
                 "removed behaviour); delete those tests in start.patch and move them into test.patch, or drop what "
                 "references the removed code:")
ORACLE_SCAN = r"""set -u
cd /app || exit 3
clean() { LC_ALL=C sed -e 's/\r//g' -e 's/\xc2\x85//g' -e 's/\xe2\x80[\xa8\xa9]//g' -e 's/[\x0b\x0c\x1c-\x1e]//g'; }
for kind in idents literals; do
  if [ -s "/tmp/oracle-$kind" ]; then
    if [ "$kind" = idents ]; then w=-w; else w=; fi
    grep -rIoHZ $w -F -f "/tmp/oracle-$kind" --exclude-dir=.git --exclude-dir=node_modules --exclude-dir=target \
      --exclude-dir=.venv --exclude-dir=venv --exclude-dir=__pycache__ . 2>/dev/null | tr '\0' '\t' | sort -u \
      | head -200000 | clean | awk -v k="$kind" '{print substr(k, 1, 1) "\t" $0}'
  fi
done
printf 'E\tdone\n'
"""
ORACLE_MAX_FILES = 12


# A name the removed code DEFINES (not one it merely uses: `itertools.accumulate(...)` in aiohttp's cookie jar and the
# local `const testError` in jest-circus matched unrelated tests of 2 accepted tasks): after def/class/fn/func (also Go
# methods)/function/struct/enum/trait/interface/type at any depth; after const/let/var/static, or assigned without a
# keyword (Python module attributes), only at the top level.
DEFINED_NAME = re.compile(r"(?:^|[\s(])(?:def|class|fn|func(?:\s*\([^)]*\))?|function\*?|struct|enum|trait|interface"
                          r"|type)\s+([A-Za-z_$][\w$]*)|^(?:export\s+|pub(?:\([^)]*\))?\s+)?(?:const|let|var|static)"
                          r"\s+(?:mut\s+)?([A-Za-z_$][\w$]*)|^([A-Za-z_][\w]*)\s*(?::[^=]*)?=(?!=)")


ORACLE_LITERAL = re.compile(r"[A-Za-z0-9](.*[A-Za-z0-9])?")  # a message or key, not a fragment like '; filename="'


def removed_source_items(start_patch: str, globs: list) -> dict:
    """{name or string literal: {"file": source file it was removed from, "kind": "name" | "literal"}} for the code
    that start.patch deletes from non-test code files (is_test_file, CODE_FILE): names of >= 8 characters that the
    removed lines define (DEFINED_NAME; not COMMON_WORDS or dunders) and string literals of >= 8 characters with >= 4
    letters that start and end with a letter or digit (comments, docstrings and imports skipped, as in
    solution_identifiers); never an item that start.patch's added lines still contain (moved or rewritten code)."""
    items, current, in_doc = {}, None, False
    added = "\n".join(line[1:] for line in start_patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    for line in start_patch.splitlines():
        if line.startswith("--- "):
            path = line[6:] if line.startswith("--- a/") else None
            current, in_doc = (path if path and CODE_FILE.search(path) and not is_test_file(path, globs) else None), False
            continue
        if line.startswith(("+++ ", "diff --git ")) or not current or not line.startswith("-"):
            continue
        text = line[1:].strip()
        fences = len(DOC_FENCE.findall(text))
        if in_doc or fences:
            in_doc = in_doc != (fences % 2 == 1)
            continue
        if not text or LEAK_COMMENT.match(text) or LEAK_IMPORT.match(text):
            continue
        for literal in STRING_LITERAL.findall(text):
            for chunk in PLACEHOLDER.split(literal[1:-1]):
                chunk = chunk.strip()
                if len(chunk) >= 8 and sum(c.isalpha() for c in chunk) >= 4 and "\\" not in chunk \
                        and ORACLE_LITERAL.fullmatch(chunk) and chunk not in added:
                    items.setdefault(chunk, {"file": current, "kind": "literal"})
        code = re.split(r"#" if HASH_COMMENTS.search(current) else r"//|/\*", STRING_LITERAL.sub(" ", line[1:]))[0]
        for match in DEFINED_NAME.finditer(code):
            name = match.group(1) or match.group(2) or match.group(3)
            if len(name) >= 8 and name not in COMMON_WORDS and not (name.startswith("__") and name.endswith("__")) \
                    and not re.search(rf"(?<![\w$]){re.escape(name)}(?![\w$])", added):
                items[name] = {"file": current, "kind": "name"}  # a name wins over the same text as a literal
    return items


def oracle_problems(output: str, items: dict, globs: list, test_patch: str) -> tuple:
    """(problems, {test file: [removed items it references]}) from ORACLE_SCAN's output (items: removed_source_items):
    an item that some non-test code file on C still contains is not removed; files test.patch touches are reset by the
    grader and do not count; a test file counts with one removed name or message (a literal with a space), or with two
    other literals."""
    hits = {}
    for line in output.splitlines():
        tag, _, rest = line.partition("\t")
        path, _, item = rest.partition("\t")
        if tag in ("i", "l") and path and item in items:
            hits.setdefault(path[2:] if path.startswith("./") else path, set()).add(item)
    present = {item for path, found in hits.items() if CODE_FILE.search(path) and not is_test_file(path, globs)
               for item in found}
    reset = set(grader_module().patch_paths(test_patch))
    stale = {path: sorted(found - present) for path, found in hits.items()
             if path not in reset and is_test_file(path, globs) and found - present}
    stale = {path: found for path, found in stale.items() if len(found) >= 2 or any(
        items[item]["kind"] == "name" or re.search(r"\s", item) for item in found)}
    if not stale:
        return [], {}
    order = sorted(stale, key=lambda path: (-len(stale[path]), path))
    parts = [f"{_clip(path, 120)} (references {', '.join(repr(_clip(item, 40)) for item in stale[path][:3])}"
             + (f" and {len(stale[path]) - 3} more" if len(stale[path]) > 3 else "") + ")"
             for path in order[:ORACLE_MAX_FILES]]
    if len(order) > ORACLE_MAX_FILES:
        parts.append(f"and {len(order) - ORACLE_MAX_FILES} more files")
    return [f"{ORACLE_PREFIX} {len(order)} {'file' if len(order) == 1 else 'files'}: " + "; ".join(parts)], stale


async def stale_oracles(b: Build, cand: Path, mat: dict, start_patch: str, test_patch: str, globs: list) -> list:
    """Gate problems for test files on C that reference code start.patch removes (see ORACLE_PREFIX); details are kept
    in <cand>/grade/stale-oracles.json. Nothing to check (no removed source items, e.g. evolve tasks): no sandbox."""
    items = removed_source_items(start_patch, globs)
    if not items:
        return []
    async with owned(mat["snapshot"], run=b.run, purpose="stale-oracles") as box:
        await box.put_text("".join(item + "\n" for item in sorted(items) if IDENTIFIER.fullmatch(item)),
                           "/tmp/oracle-idents")
        await box.put_text("".join(item + "\n" for item in sorted(items) if not IDENTIFIER.fullmatch(item)),
                           "/tmp/oracle-literals")
        result = await box.sh(ORACLE_SCAN, timeout=900, check=False)
    if result.timed_out or not result.stdout.rstrip().endswith(ARCHIVE_DONE):
        log(f"{cand.name}: stale-oracle scan incomplete (exit {result.exit_code}, timed out {result.timed_out}); "
            "not reported")
        return []
    problems, stale = oracle_problems(result.stdout, items, globs, test_patch)
    write_json(cand / "grade/stale-oracles.json", {"items": items, "stale": stale, "problems": problems})
    return problems


async def gates(b: Build, cand: Path, author_dir: Path, mat: dict) -> dict:
    """Real-grader checks on C: empty patch -> 0, reference solution -> 1 (x3), empirical F2P/P2P, hints, leaks."""
    block = (author_dir / "run_tests.sh").read_text()
    test_patch = (author_dir / "test.patch").read_text()
    solution = (author_dir / "solution.patch").read_text()
    provisional = cand / "provisional"
    write_tests(provisional / "tests", b, block, test_patch, mat["base_commit"], solution=solution)
    runs = await asyncio.gather(
        grade(mat["snapshot"], provisional, "", run=b.run, out_dir=cand / "grade/empty"),
        *(grade(mat["snapshot"], provisional, solution, run=b.run, out_dir=cand / f"grade/solution-{i}")
          for i in range(3)))
    unreliable = env_unreliable(b)  # hengyuf 2026-10-20: build_env's exclusions stay out of P2P
    f2p, p2p, problems = calibrate(runs[0], runs[1:], b.reports, unreliable)
    excluded = env_matches(sorted(runs[0]["suites"].get(Path(b.reports[0]).name, {})), unreliable) if unreliable else []
    if excluded:
        log(f"{cand.name}: {len(excluded)} base-suite tests not P2P (excluded or flaky in build_env's suite): "
            f"{excluded[:5]}")
    if not f2p or not p2p:  # usually the RUN TESTS block crashed: show the author the grader's own output
        tail = (cand / "grade/empty/output.log").read_text()[-1500:]
        problems.append(f"grader output with the empty patch ended with:\n{tail}")
    # hengyuf 2026-10-20: the sanitizing grader restores test paths, so the runs above graded the reference solution
    # without its edits there; named first (the other problems are then mostly symptoms)
    globs = json.loads((provisional / "tests/config.json").read_text())["test_paths"]
    problems[:0] = [f"solution.patch touches test paths the grader restores before grading (test_paths {globs}): "
                    f"{paths[:10]}; move those changes to test.patch (solution.patch holds source files only)"
                    for paths in [restored_paths(solution, globs)] if paths]
    problems += [f"test.patch touches files the DeepSWE grader cannot parse (it would not reset them): {unseen}"
                 for unseen in [unseen_paths(test_patch)] if unseen]  # hengyuf
    problems += [f"hint marker added by start.patch: {line[:120]}"
                 for line in hint_lines((author_dir / "start.patch").read_text())[:5]]
    problems += [f"start.patch edits ignore rules ({path}): MATERIALIZE's cleanup follows them, so they must stay as in A"
                 for path in ignore_rule_edits((author_dir / "start.patch").read_text())]  # hengyuf, 2026-10-18
    problems += [f"test.patch re-adds {name} (removed from {source} by start.patch) in {target} without touching {source}: "
                 f"put it back into {source}, or a student's same-named test there breaks the Go package"
                 for source, target, name in rehomed_go_tests((author_dir / "start.patch").read_text(), test_patch)]
    problems += [f"{SOURCE_EDIT_PREFIX} (outside test_paths {globs} and test directories/file names): {paths[:10]}. "
                 "Both graders reset every file test.patch touches before applying it, so a solution's own edits to "
                 "these files would be lost before grading. Put the hidden tests in test files (also tests that "
                 "start.patch removes from a source file's inline test module): Rust integration tests under tests/ or "
                 "a new test-only module file (test.patch may then add only a `#[cfg(test)] mod <name>;` line to an "
                 "existing source file), Go *_test.go, Python test_*.py, JS/TS *.test.* files"
                 for paths in [test_patch_source_edits(test_patch, globs)] if paths]  # hengyuf 2026-10-20 (wasmi c03)
    if not problems:  # hengyuf 2026-10-20: stale oracles outside the broad suite are checked beside stale_tests
        start_patch = (author_dir / "start.patch").read_text()
        for found in await asyncio.gather(stale_tests(b, cand, mat, start_patch),
                                          stale_oracles(b, cand, mat, start_patch, test_patch, globs)):
            problems += found
    if not problems:
        final = cand / "final"
        write_tests(final / "tests", b, block, test_patch, mat["base_commit"], f2p, p2p, solution=solution)
        empty, solved = await asyncio.gather(
            grade(mat["snapshot"], final, "", run=b.run, out_dir=cand / "grade/final-empty"),
            grade(mat["snapshot"], final, solution, run=b.run, out_dir=cand / "grade/final-solution"))
        if empty["reward"] != 0 or solved["reward"] != 1:
            problems.append(f"final grading gave empty={empty['reward']} solution={solved['reward']} (want 0/1)")
        problems += await leak_scan(b, mat["snapshot"], solution, app_hits=removes_solution_code(
            (author_dir / "start.patch").read_text(), solution))
    return {"f2p": f2p, "p2p": p2p, "problems": problems, "env_excluded": excluded}


async def instruction_rubric(b: Build, out: Path, author_dir: Path, final_tests: Path) -> None:
    """Cheap agent: DeepSWE-style instruction.md (no hints) + process rubric.md."""
    inputs = out / "inputs"
    (inputs / "examples").mkdir(parents=True, exist_ok=True)
    (inputs / "tests").mkdir(exist_ok=True)
    shutil.copy2(author_dir / "instruction_draft.md", inputs / "instruction_draft.md")
    for name in ("test.patch", "config.json"):
        shutil.copy2(final_tests / name, inputs / "tests" / name)
    shutil.copy2(b.fa_dir / "failure_analysis.md", inputs / "target_failure_analysis.md")
    same_language = [t for t in sorted(p.name for p in DEEPSWE.iterdir() if (p / "task.toml").exists())
                     if t != b.target and task_toml(DEEPSWE / t)["metadata"]["language"] == b.language][:2]
    for example in [b.target, *same_language]:
        shutil.copy2(DEEPSWE / example / "instruction.md", inputs / "examples" / f"{example}.md")
    solution_lines = added_lines((author_dir / "solution.patch").read_text(), 40)

    def check(result):
        problems = _needs(out, "instruction.md", "rubric.md")(result)
        if problems:
            return problems
        text = (out / "instruction.md").read_text()
        if text.rstrip().splitlines()[-1].strip() != IMPORTANT:
            problems.append("instruction.md must end with the fixed IMPORTANT line")
        if not 500 <= len(text.encode()) <= 5500:
            problems.append(f"instruction.md is {len(text.encode())} bytes (want 500-5500)")
        if BANNED.search(text):
            problems.append(f"instruction.md mentions {BANNED.search(text).group(0)!r}")
        problems += [f"instruction.md quotes a solution line: {line[:80]}" for line in solution_lines if line in text][:3]
        return problems

    prompt = render("instruction_rubric", important=IMPORTANT, cheat_categories=CHEAT_CATEGORIES)
    for attempt in range(2):
        try:
            async with AGENTS:
                await agent(prompt, out, backend="codex", check=check)  # was cursor (user decision: codex for all)
            return
        except AgentError as error:
            if attempt:
                raise
            prompt += f"\n\nA previous attempt failed these checks; fix them:\n{error}\n"


def package(b: Build, env: dict, cand: Path, author_dir: Path, mat: dict, gate: dict, *, kind: str, title: str,
            summary: str) -> Path:
    """Write the Harbor/DeepSWE task directory and check Gym accepts it."""
    digest = hashlib.sha256(f"{b.target}|{b.repo}|{cand.name}|{mat['base_commit']}".encode()).hexdigest()
    task_id = re.sub(r"[^a-z0-9-]+", "-", f"syn-{b.repo.split('/')[1]}-{cand.name}".lower())[:41] + "-" + digest[:8]
    task = cand / "task" / task_id
    shutil.rmtree(task.parent, ignore_errors=True)
    shutil.copytree(cand / "final/tests", task / "tests")
    (task / "solution").mkdir(parents=True)
    shutil.copy2(author_dir / "solution.patch", task / "solution/solution.patch")
    shutil.copy2(b.target_dir / "solution/solve.sh", task / "solution/solve.sh")
    shutil.copy2(cand / "instruction/instruction.md", task / "instruction.md")
    shutil.copy2(cand / "instruction/rubric.md", task / "rubric.md")
    environment = task / "environment"
    environment.mkdir()
    shutil.copy2(b.root / "env/setup.sh", environment / "setup.sh")
    shutil.copy2(author_dir / "start.patch", environment / "start.patch")
    (environment / "materialize.sh").write_text(MATERIALIZE)
    (environment / "Dockerfile").write_text(fill(DOCKERFILE, base_image=b.image, repo_url=env["repo_url"], sha=env["sha"]))
    collect = ("cd /app && mkdir -p /logs/artifacts && git config --global --add safe.directory /app "
               f"&& git diff --binary {mat['base_commit']} HEAD > /logs/artifacts/model.patch")
    (task / "task.toml").write_text(fill(
        TASK_TOML, task_id=task_id, ext_id=digest[:32], title=json.dumps(title), summary=json.dumps(summary),
        language=b.language, repo_url=env["repo_url"], base_commit=mat["base_commit"], upstream=env["sha"],
        target=b.target, kind=kind, collect=json.dumps(collect), image=mat["snapshot"]))
    validate_task(task)
    if dataset_row(task)["image"] != mat["snapshot"]:
        raise GateError("dataset row image differs from docker_image")
    return task


async def make_task(b: Build, env: dict, cand: Path, **details):
    """One candidate end to end; an unexpected error is logged and leaves the candidate retryable."""
    try:
        return await _make_task(b, env, cand, **details)
    except Exception as error:
        log(f"{cand.name} FAILED: {type(error).__name__}: {str(error)[-500:]}")
        return None


async def _make_task(b: Build, env: dict, cand: Path, *, kind: str, title: str, summary: str, plan_md: Path):
    """author -> materialize -> gates (one retry with the gate report) -> instruction/rubric -> package."""
    if done := read_json(cand / "stage.json"):
        return done if done.get("ok") else None
    built = read_json(cand / "built.json")  # a gated starting state survives a crash in the later stages
    feedback = ""
    for attempt in range(0 if built else 2):
        author_dir = cand / f"author-{attempt}"
        try:
            await author(b, env, author_dir, kind=kind, title=title, plan_md=plan_md, feedback=feedback)
            mat = await materialize(b, env, author_dir / "start.patch", f"{cand.name}-{attempt}")
            gate = await gates(b, cand, author_dir, mat)
        except RuntimeError as error:  # AgentError, GateError, failing sandbox command: an authoring problem
            feedback = f"{type(error).__name__}: {str(error)[-3000:]}"
            log(f"{cand.name} attempt {attempt} failed: {feedback[:300]}")
            continue
        write_json(cand / f"gates-{attempt}.json", gate)
        if not gate["problems"]:
            built = {"attempt": attempt, "mat": mat, "gate": gate}
            write_json(cand / "built.json", built)
            break
        feedback = "\n".join(gate["problems"])
        log(f"{cand.name} attempt {attempt} gates: {feedback[:300]}")
        if dependency_leaks(gate["problems"]):
            feedback = ("released copy of the removed code in a dependency tree (cannot be removed by the author): "
                        + "; ".join(dependency_leaks(gate["problems"])[:5]))
            log(f"{cand.name}: {feedback[:300]}; no retry")
            break
    if not built:
        write_json(cand / "stage.json", {"ok": False, "problems": feedback})
        return None
    author_dir, mat, gate = cand / f"author-{built['attempt']}", built["mat"], built["gate"]
    await instruction_rubric(b, cand / "instruction", author_dir, cand / "final/tests")
    task = package(b, env, cand, author_dir, mat, gate, kind=kind, title=title, summary=summary)
    stage = {"ok": True, "name": cand.name, "task_dir": str(task), "task_id": task.name, "kind": kind,
             "title": title, "snapshot": mat["snapshot"], "f2p": len(gate["f2p"]), "p2p": len(gate["p2p"])}
    write_json(cand / "stage.json", stage)
    return stage


async def score(b: Build, task: dict, records: list, out: Path) -> dict:
    """Pass rates with audited teacher passes; synthetic failure analysis + similarity when rates are in range."""
    if done := read_json(out / "metrics.json"):
        return done
    if not records:
        raise RuntimeError(f"no rollouts for {task['task_id']}")
    # hengyuf 2026-10-20 (official1-wasmi c05 kimi-r3): a FAIL whose verifier sandbox broke (I/O error, SIGBUS, disk
    # full) is no grade: infra, out of the rates and `valid`, with the reason in "verifier_fault" and metrics
    # "verifier_faults". A platform fault like PTY/410/422 (main's decision, 2026-10-20): NOT a grading failure (those
    # stay task-caused), so the task remains a valid seed unless a model drops below 3 valid slots.
    faults = {r["slot"]: reason for r in records if (reason := verifier_fault(r))}
    records = [dict(r, status="infra", verifier_fault=faults[r["slot"]]) if r["slot"] in faults else r for r in records]
    for slot, reason in faults.items():
        log(f"score {task['task_id']}: {slot} counted as infra: {reason}")
    task_dir = Path(task["task_dir"])
    write_rollouts(out / "rollouts", records)
    student = [r for r in records if r["model"] == STUDENT and r["status"] in ("pass", "fail")]
    teacher = [r for r in records if r["model"] == TEACHER and r["status"] in ("pass", "fail")]
    instruction, rubric = (task_dir / "instruction.md").read_text(), (task_dir / "rubric.md").read_text()
    passes = [r for r in teacher if r["status"] == "pass"]
    verdicts = await asyncio.gather(*(judge_pass(r, instruction, rubric) for r in passes))
    write_json(out / "teacher_audit.json", {r["slot"]: v for r, v in zip(passes, verdicts)})
    retrieval = any("Answer retrieval by action" in " ".join(v.get("categories") or []) for v in verdicts)
    s_rate = sum(r["status"] == "pass" for r in student) / len(student) if student else None
    t_rate = sum(not v["cheated"] for v in verdicts) / len(teacher) if teacher else None
    target = read_json(b.fa_dir / "inputs/stats.json", {}).get("pass_stats", {})
    metrics = {**task, "student_rate": s_rate, "teacher_clean_rate": t_rate, "valid": [len(student), len(teacher)],
               "grading_failures": [r["slot"] for r in records
                                    if r["status"] == "infra" and r["verifier_error"] and r["slot"] not in faults],
               "verifier_faults": faults,
               "uniform_failure": uniform_failure(records),
               "cheated": sum(v["cheated"] for v in verdicts), "answer_retrieval": retrieval,
               "target_student_rate": (target.get(STUDENT) or {}).get("rate"),
               "target_teacher_rate": (target.get(TEACHER) or {}).get("rate"), "similarity": None, "accept": False}
    if min(len(student), len(teacher)) < 3:
        metrics["reason"] = "inconclusive: fewer than 3 valid rollouts per model"
    elif retrieval:
        metrics["reason"] = "teacher retrieved the answer by action (leaky task)"
    elif s_rate > 0.2 or t_rate < 0.6:
        metrics["reason"] = f"pass rates out of range (student {s_rate:.2f} > 0.2 or clean teacher {t_rate:.2f} < 0.6)"
    else:
        await analyze(task_dir, task["snapshot"], records, out / "failure_analysis", run=b.run)
        verdict = await similarity(b.fa_dir, out / "failure_analysis")
        metrics.update(similarity=verdict["score"], similarity_detail=verdict, accept=verdict["score"] >= 6,
                       reason=f"similarity {verdict['score']}/10")
    write_json(out / "metrics.json", metrics)
    log(f"score {task['task_id']}: {metrics['reason']} accept={metrics['accept']}")
    return metrics


def grading_failures(scored: Path, metrics: dict) -> list:
    """Probe slots whose verifier ran but produced no grade: the task's tests could not be applied or run (narwhals
    cand-c02: test.patch failed to apply once the agent had edited the tests). score() records them; for metrics
    written before that, they are read back from the rollouts/<slot>.md that score() wrote (header "# slot — infra",
    then "...; verifier_error <error>")."""
    if "grading_failures" in metrics:
        return metrics["grading_failures"]
    failed = []
    for path in sorted((scored / "rollouts").glob("*.md")):
        text = path.read_text()
        error = re.search(r"; verifier_error (.*)", text)
        if path.name != "index.md" and text.split("\n", 1)[0].endswith(" — infra") and error \
                and error.group(1) != "None":
            failed.append(path.stem)
    return failed


def _short(fraction) -> bool:
    passed, _, total = str(fraction).partition("/")
    return passed.isdigit() and total.isdigit() and int(passed) < int(total)


def uniform_failure(records: list) -> str | None:
    """Every graded rollout that has a patch (both models, >= 4 of them) scoring the same F2P and P2P with P2P short:
    the tests themselves are broken for any solution (official1-arcane cand-c04, 2026-10-18: hidden tests re-homed in a
    new Go file collide with the students' same-named tests, so the package never compiles: F2P 1/22, P2P 126/431)."""
    patched = [r for r in records if r["status"] in ("pass", "fail") and r.get("patch")]
    if len(patched) >= 4 and len({(r["f2p"], r["p2p"]) for r in patched}) == 1 and _short(patched[0]["p2p"]):
        return f"all {len(patched)} graded rollouts with a patch score F2P {patched[0]['f2p']}, P2P {patched[0]['p2p']}"
    return None


def seed_check(scored: Path, metrics: dict) -> str:
    """Why a probed task may NOT seed evolution ("" = it may). hengyuf, 2026-10-17 (user rule): validity only, never
    the pass rates (those decide acceptance alone). (1) The reference solution passes the task's F2P and P2P tests:
    guaranteed before any rollout, since only tasks that passed gates() are probed. (2) The task's infrastructure is
    sound: no rollout whose verifier failed to grade it, and >= 3 valid rollouts per model."""
    failed = grading_failures(scored, metrics)
    if failed:
        return f"verifier produced no grade for {len(failed)} rollout(s): {', '.join(failed)}"
    if metrics.get("uniform_failure"):
        return f"tests cannot run normally: {metrics['uniform_failure']}"
    if min(metrics["valid"]) < 3:
        return f"fewer than 3 valid rollouts per model (valid {metrics['valid']})"
    return ""


def valid_seeds(b: Build, name: str, results: list) -> list:
    """The probed tasks of round `name` that pass seed_check; exclusions are logged and kept in <round>/seeds.json."""
    valid, invalid = [], {}
    for metrics in results:
        reason = seed_check(b.root / name / metrics["name"], metrics)
        if reason:
            invalid[metrics["name"]] = reason
            log(f"{metrics['name']}: invalid, not evolved ({reason})")
        else:
            valid.append(metrics)
    write_json(b.root / name / "seeds.json", {"valid": [m["name"] for m in valid], "invalid": invalid})
    return valid


def evolve_outcome(metrics: dict) -> str:
    """The evolve_plan prompt's account of the parent's probe (accepted parents seed evolution too since 2026-10-17;
    the text for a rejected parent is the prompt's original wording)."""
    target = f"(target: student {metrics['target_student_rate']}, teacher\n{metrics['target_teacher_rate']})"
    if metrics.get("accept"):
        return (f"It was accepted: its pass rates are in range {target} and the student fails it the way it fails the\n"
                "target, so it is a good seed. Evolve it into a different task that keeps what made it work; below, "
                "\"what went\nwrong\" means what made the probed task work and what the new feature must keep.")
    return (f"It was not accepted: either the pass rates are off {target} or the student fails differently from how "
            "it fails the target.")


async def probe_and_score(b: Build, name: str, pool: list) -> list:
    todo = [task for task in pool if not (b.root / name / task["name"] / "metrics.json").exists()]
    records = {}
    if todo:
        key = hashlib.sha1(",".join(sorted(task["task_id"] for task in todo)).encode()).hexdigest()[:8]
        records = await probe(b.root / name / f"probe-{key}", [Path(task["task_dir"]) for task in todo])
    results = await asyncio.gather(*(score(b, task, records.get(task["task_id"], []), b.root / name / task["name"])
                                     for task in pool), return_exceptions=True)
    for task, result in zip(pool, results):
        if isinstance(result, Exception):
            log(f"score {task['task_id']} FAILED (rerun resumes): {type(result).__name__}: {str(result)[-300:]}")
    return [result for result in results if not isinstance(result, Exception)]


async def evolve(b: Build, env: dict, parent: dict, branch: int, round_index: int):
    """High-IID branch: codex plans a new feature from A aimed at one target failure pattern, then make_task."""
    cand = b.root / f"{parent['name']}-r{round_index}b{branch}"
    out = cand / "evolve"
    patterns = read_json(b.fa_dir / "result.json")["patterns"] or [{"name": "overall"}]
    focus = patterns[branch % len(patterns)]["name"]
    if not read_json(out / "stage.json"):
        inputs = out / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        shutil.copytree(parent["task_dir"], inputs / "task", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("environment"))
        probed = b.root / f"round-{round_index - 1}" / parent["name"]
        shutil.copytree(probed / "rollouts", inputs / "probe", dirs_exist_ok=True)
        shutil.copy2(probed / "metrics.json", inputs / "metrics.json")
        if (probed / "failure_analysis/failure_analysis.md").exists():
            shutil.copy2(probed / "failure_analysis/failure_analysis.md", inputs / "synthetic_failure_analysis.md")
        shutil.copy2(b.fa_dir / "failure_analysis.md", inputs / "target_failure_analysis.md")
        shutil.copy2(b.root / "understanding/understanding.md", inputs / "understanding.md")
        shutil.copy2(b.root / "plan/mask_plan.md", inputs / "mask_plan.md")
        metrics = read_json(probed / "metrics.json")
        async with AGENTS, owned(env["snapshot"], run=b.run, purpose="evolve-plan") as box:
            write_wrappers(out / "bin", box.sid)
            prompt = render("evolve_plan", repo=b.repo, target=b.target, student=STUDENT, teacher=TEACHER,
                            focus_pattern=focus, outcome=evolve_outcome(metrics),
                            sandbox=render("_sandbox", network_note="(no network)"))
            result = await agent(prompt, out, backend="codex", check=_needs(out, "evolve_plan.md", min_bytes=1000))
        write_json(out / "stage.json", {"title": result.get("title") or f"{parent['title']} (evolved)",
                                        "summary": result.get("summary", "")})
    plan_stage = read_json(out / "stage.json")
    child = await make_task(b, env, cand, kind="evolve", title=plan_stage["title"], summary=plan_stage["summary"],
                            plan_md=out / "evolve_plan.md")
    return child and dict(child, lineage=parent.get("lineage", parent["name"]))


async def accept(b: Build, metrics: dict) -> None:
    """Copy an accepted task to runs/<run>/accepted/, archive its starting workspace, and add its Gym row."""
    import workspace_ops
    accepted = RUNS / b.run / "accepted"
    task = Path(metrics["task_dir"])
    shutil.copytree(task, accepted / task.name, dirs_exist_ok=True)
    archive = accepted / f"{task.name}.workspace"
    if not (archive / "manifest.json").exists():
        staged = scratch() / f"backup-{task.name}"
        shutil.rmtree(staged, ignore_errors=True)
        async with owned(metrics["snapshot"], run=b.run, purpose="backup") as box:
            await workspace_ops.backup(box.client, box.sid, "/app", staged)
        partial = archive.with_name(archive.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        shutil.copytree(staged, partial)
        shutil.rmtree(archive, ignore_errors=True)
        partial.rename(archive)
        shutil.rmtree(staged, ignore_errors=True)
    write_json(accepted / task.name / "provenance.json", metrics)
    rows = {}
    if (accepted / "tasks.jsonl").exists():
        rows = {json.loads(l)["task_id"]: l for l in (accepted / "tasks.jsonl").read_text().splitlines() if l}
    rows[task.name] = json.dumps(dataset_row(accepted / task.name))
    (accepted / "tasks.jsonl").write_text("".join(row + "\n" for row in rows.values()))


def round_cap(b: Build, rounds: int) -> int:
    """--rounds, lowered by <build root>/max_rounds.json {"max_rounds": N}. It is re-read before every round, so a
    running build can be shortened without a restart (hengyuf, 2026-10-17: the official1 builds stop after round 2)."""
    cap = (read_json(b.root / "max_rounds.json") or {}).get("max_rounds")
    return min(rounds, cap) if isinstance(cap, int) and not isinstance(cap, bool) and cap >= 0 else rounds


async def build(run: str, target: str, repo: str, candidates: int, branches: int, rounds: int) -> dict:
    b = Build(run, target, repo)
    # hengyuf 2026-10-20: every candidate gated from here on gets the sanitizing grader; an unknown repository
    # language (UnknownLanguage) stops the build before any work
    log(f"grader for new candidates: {SANITIZER} with {json.dumps(b.grading)}")
    b.root.mkdir(parents=True, exist_ok=True)
    await analyze_target(target, run)  # reuses a valid analysis, redoes an invalid one
    env = await build_env(b)
    understanding = await understand(b, env)
    plans = (await plan(b, env, understanding, max(10, candidates)))[:candidates]
    for candidate in plans:
        section = b.root / "plan" / f"{candidate['id']}.md"
        section.write_text("# Candidate\n" + json.dumps(candidate, indent=2) + "\n\n# Full plan\n" +
                           (b.root / "plan/mask_plan.md").read_text())
    built = await asyncio.gather(*(make_task(b, env, b.root / f"cand-{c['id']}", kind="mask", title=c["title"],
                                             summary=c["summary"], plan_md=b.root / "plan" / f"{c['id']}.md")
                                   for c in plans))
    results = await probe_and_score(b, "round-0", [task for task in built if task]) if any(built) else []
    accepted = [m for m in results if m["accept"]]
    for metrics in accepted:
        await accept(b, metrics)
    parents = valid_seeds(b, "round-0", results)  # hengyuf: every valid round-0 task evolves, accepted or not
    for round_index in range(1, rounds + 1):
        if round_index > round_cap(b, rounds):
            log(f"round {round_index} not started: {b.root / 'max_rounds.json'} caps this build at "
                f"{round_cap(b, rounds)} rounds")
            break
        if not parents:
            break
        children = await asyncio.gather(*(evolve(b, env, parent, branch, round_index)
                                          for parent in parents for branch in range(branches)), return_exceptions=True)
        for child in children:
            if isinstance(child, Exception):
                log(f"evolve branch FAILED (rerun resumes): {type(child).__name__}: {str(child)[-300:]}")
        children = [child for child in children if child and not isinstance(child, Exception)]
        results = await probe_and_score(b, f"round-{round_index}", children) if children else []
        for metrics in results:
            if metrics["accept"]:
                accepted.append(metrics)
                await accept(b, metrics)
        best = {}
        for metrics in valid_seeds(b, f"round-{round_index}", results):  # next round evolves the best valid
            if metrics["accept"]:                                         # non-accepted child of each lineage
                continue
            key = metrics.get("lineage", metrics["name"])
            student = 1.0 if metrics["student_rate"] is None else metrics["student_rate"]
            target_rate = metrics["target_student_rate"] or 0.0
            rank = (metrics["similarity"] or 0, -abs(student - target_rate))
            if key not in best or rank > best[key][0]:
                best[key] = (rank, metrics)
        parents = [metrics for _, metrics in best.values()]
    summary = {"target": target, "repo": repo, "env": env, "built": [t for t in built if t],
               "rejected": [c["id"] for c, t in zip(plans, built) if not t], "accepted": accepted}
    write_json(b.root / "summary.json", summary)
    log(f"build {target} x {repo}: {len(accepted)} accepted")
    return summary


async def cleanup(run: str) -> None:
    """Delete this run's snapshots except the image-A snapshots in use and accepted tasks' C snapshots."""
    from sandbox_helpers import Client, Config
    unfinished = [str(path) for path in (RUNS / run / "build").glob("*/*") if not (path / "summary.json").exists()]
    if unfinished:
        raise SystemExit(f"cleanup refused: unfinished builds {unfinished}")
    keep = {json.loads(l)["image"] for l in (RUNS / run / "accepted/tasks.jsonl").read_text().splitlines()
            if l} if (RUNS / run / "accepted/tasks.jsonl").exists() else set()
    keep |= {read_json(path)["snapshot"] for path in (RUNS / run / "build").glob("*/*/env/stage.json")}
    ledger = [json.loads(l) for l in (RUNS / run / "snapshots.jsonl").read_text().splitlines() if l]
    async with Client(Config.load(ATP_ENV), timeout=120) as client:
        for entry in ledger:
            if entry["snapshot"] in keep:
                continue
            try:
                await client.delete_snapshot(entry["snapshot"].removeprefix("snapshot:"))
                log(f"deleted {entry['snapshot']}")
            except Exception as error:
                log(f"WARN {entry['snapshot']}: {type(error).__name__}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("build")
    sub.add_argument("--run", required=True)
    sub.add_argument("--task", required=True)
    sub.add_argument("--repo", help="owner/name; default: best eligible repo from top100.jsonl")
    sub.add_argument("--candidates", type=int, default=5)
    sub.add_argument("--branches", type=int, default=5)
    sub.add_argument("--rounds", type=int, default=3)
    sub = commands.add_parser("repos")
    sub.add_argument("--task", required=True)
    sub = commands.add_parser("cleanup")
    sub.add_argument("--run", required=True)
    args = parser.parse_args()
    if args.command == "repos":
        print("\n".join(eligible_repos(args.task)[:20]))
    elif args.command == "cleanup":
        asyncio.run(cleanup(args.run))
    else:
        repo = args.repo or eligible_repos(args.task)[0]
        asyncio.run(build(args.run, args.task, repo, args.candidates, args.branches, args.rounds))


if __name__ == "__main__":
    main()
