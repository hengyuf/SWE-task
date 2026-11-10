#!/usr/bin/env python3
"""Refresh this repo's code mirror from the live cluster trees (code and docs only; never data or secrets).

    python3 -I -B tools/sync_from_lustre.py [--write]     # default: dry run, prints the manifest summary

Mirrors (relative layout kept):
    /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task     -> swe-task/
    /lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve  -> glm53-serve/
Rules: only text source/config/doc files (by extension), at most MAX_BYTES each; data/state/secret directories and
names are skipped (EXCLUDE_DIRS / EXCLUDE_NAME); backups (*.pre-*, *.prev-*, *.bak) are skipped. Every selected file is
scanned for credential patterns (SECRET_PATTERNS) and the run refuses to write if one matches.
"""
import argparse
import fnmatch
import hashlib
import os
from pathlib import Path
import re
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
ROOTS = {"swe-task": Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task"),
         "glm53-serve": Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve")}
EXTENSIONS = {".py", ".sh", ".sbatch", ".md", ".json", ".yaml", ".yml", ".toml", ".patch", ".diff", ".cfg", ".ini",
              ".txt", ".js", ".ts", ".tsx", ".html", ".css", ".jinja", ".j2", ".tmpl", ".sql", ".fish"}
EXTRA_NAMES = {"Dockerfile", "Makefile"}
MAX_BYTES = 600_000
# directory names that are never walked (data / state / large artifacts / credentials / local agent settings)
EXCLUDE_DIRS = {
    "probes", "results", "logs", "babysit", "scratch", "runs", "cache", "vendor", "runtime", "venv", ".venv",
    "node_modules", "__pycache__", ".git", "tmp", "dummy", "models", "release-cache", "attempts", "workspace",
    "outputs", "captures", "export", "hsg-ref", "cache-0.29", "cache-0.33", "fleets", "rollback", "secrets", ".ssh",
    ".config", ".claude", ".agents", "followup", "dashboard-stage-baihe-20261022", "dashboard-stage-explorer-20261022",
    "dashboard-stage-v18-20261023", "repo_reselect_baihe",
}
EXCLUDE_NAME = ["*.pre-*", "*.prev-*", "*.bak", "*~", ".env", ".env.*", "*.env", "*secret*", "*token*", "*credential*",
                "*.pem", "*.key", "id_*", "hub_key_active.txt*", "keeper-status.json", "keeper-log.jsonl",
                "keeper-legs.json", "progress*.json", "PROGRESS.md", "legs.jsonl", "snapshots.json", "*.jsonl",
                ".mcp.json", "*.tmp", ".*.tmp", "*.log", "*.out", "GO", "GO-*", "*.STOP", "STOP", "watch-state.json",
                "PROGRESS-rows.json", "state.json", "TIER-PAUSE.json", "heartbeat.json", "audit_seen.json",
                "watch-last.json", "hub_smoke.json", "price_calib.json"]
# needed code / small configs that a name or directory rule would drop (still content-scanned; no secret VALUES)
ALLOW_FILES = [
    "swe-task/atp/scripts/deepswe_credentials.py",              # credential handling by NAME (imported by ATP)
    "swe-task/tools/make_secrets.py",                           # builds atp/.env + secrets/ from the user's .env
    "swe-task/sft/glm-rollouts-sftv3-20261107/make_baihe_secrets.py",
    "swe-task/tools/codex/home/config.toml",                    # codex provider config (env_key name only)
    "swe-task/tools/codex/catalog/gpt-6-astra-native.json",     # codex model catalog read by llm_engine
    "swe-task/sft/glm-rollouts-tasksall-20261108/release-cache/release-cache.json",  # stage_wrapper manifest
    "swe-task/sft/glm-rollouts-tasksall-20261108/release-cache/rg-15.1.0.json",
    "swe-task/sft/glm-rollouts-sftv3-20261107/tasks.json",      # frozen SFT-v3 plan (input of tasksall make_tasks)
    "glm53-serve/tools/replay/logs/build_all.sh",
    "swe-task/.swe-task-root",
    "swe-task/.agents/skills/karpathy-guidelines/SKILL.md",
    "swe-task/.agents/skills/repo-orient/SKILL.md",
    "swe-task/sft/train-acc-v2-20261024/assets/nano_v3_lenient_think_counted_parser.py",
] + [f"swe-task/sft/teacher-deepswe-20261030/final/{n}" for n in (
    "atp_states.py", "collect_final_20261102.py", "collect_report_wording_20261102.py", "independent_counts_20261102.py",
    "readme_collect.py", "readme_final_collect_20261102.py", "unit_check.py", "verify_collection.py",
    "verify_infra_rows_20261102.py")]
# job-submission helpers outside both roots, mirrored to tools/external/ (origin + sha256 in tools/external/README.md)
EXTERNAL = {
    "schdule_helper.sh": Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/schdule_helper.sh"),
    "schedule_helper_json_safe.sh": Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/recipes/"
                                         "scale-rebench-super-sc-length-formal-v1/fullscale-start-v1/overrides/"
                                         "r3-network-v1/launch/schedule_helper.sh"),
}
MAX_JSON_BYTES = 150_000  # bigger JSON files are data (task snapshots, inventories), not framework config
# files with internal operations notes about other people / clusters (kept on the cluster only)
EXCLUDE_PATHS = {"glm53-serve/notes/dfw-ops/digest.md"}
# fake secrets in tests (exact file, label)
ALLOW_HITS = {("swe-task/dashboard/tests/test_server.py", "assigned secret-looking value")}


def allowed(prefix: str, rel: Path) -> bool:
    """Is this file (relative to its root) part of the published framework (code, configs, docs)?"""
    parts, suffix = rel.parts, rel.suffix
    code_doc = suffix in (".py", ".sh", ".sbatch", ".md")
    if prefix == "glm53-serve":
        top = parts[0]
        if len(parts) == 1:
            return rel.name == "README.md"
        if top in ("scripts", "tools"):
            return True
        if top == "env":
            return len(parts) == 2 and suffix == ".sh"
        if top == "notes":
            return not any(p.startswith("dl.") for p in parts)
        if top == "e2e":
            return len(parts) == 2
        return False
    top = parts[0]  # swe-task
    if len(parts) == 1:
        return rel.name == "README.md"
    if top in ("docs", "opensandbox", "atp"):
        return True
    if top == "dashboard":
        return parts[1] not in ("state", "data")
    if top == "student":
        return parts[1] not in ("registry", "state")
    if top == "repo_reselect":
        return parts[1] != "out"
    if top == "tools":
        return len(parts) == 2
    if top == "synth":
        if len(parts) == 2:
            return True
        sub = parts[1]
        if sub in ("prompts", "tools", "tests", "verifier", "workflows", "queue", "queue2", "queue3", "queue4"):
            return True
        if sub == "analysis":
            return code_doc
        if sub == "repair":
            return len(parts) <= 4 and code_doc
        return False
    if top == "sft":
        if len(parts) == 2:
            return False
        campaign = parts[1]
        if campaign == "analysis":
            return code_doc
        if campaign == "corpus":
            return code_doc and len(parts) <= 4
        if campaign in ("hsg", "xfer"):
            return code_doc and "out" not in parts
        if len(parts) == 3:
            return True
        sub = parts[2]
        if sub in ("prompt", "tools", "setup"):
            return True
        if sub == "github":  # relay code + issue body; not its state.json or outbox messages
            return len(parts) == 4 and suffix in (".py", ".js", ".md") and not rel.name.startswith(".")
        if sub == "analysis":
            return code_doc
        if sub == "atp-fix":
            return "app" not in parts
        return False
    return False


# per-file content patterns that must never be published
SECRET_PATTERNS = [
    (r"nvapi-[A-Za-z0-9_-]{20,}", "NVIDIA API key"), (r"\bsk-[A-Za-z0-9_-]{20,}", "sk- key"),
    (r"ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gho_[A-Za-z0-9]{20,}", "GitHub token"),
    (r"\bhf_[A-Za-z0-9]{30,}", "HF token"), (r"AKIA[0-9A-Z]{16}", "AWS key"), (r"glpat-[A-Za-z0-9_-]{20,}", "GitLab token"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key"), (r"xox[baprs]-[A-Za-z0-9-]{10,}", "Slack token"),
    (r"(?i)(api[_-]?key|token|password|secret)\s*[=:]\s*[\"']?[A-Za-z0-9_\-]{32,}", "assigned secret-looking value"),
]


def selected_files():
    for prefix, root in ROOTS.items():
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = Path(dirpath).relative_to(root)
            dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(("synth-", "dl.", ".dl-"))
                                 and not (prefix == "glm53-serve" and rel_dir == Path("env"))      # env/vllm-* venvs
                                 and not (prefix == "glm53-serve" and rel_dir == Path("e2e"))      # e2e/root* data
                                 and not (prefix == "swe-task" and rel_dir == Path("tools"))       # codex/claude homes
                                 and not (prefix == "swe-task" and rel_dir == Path("synth") and d in ("runs", "repair"))
                                 and not (prefix == "swe-task" and rel_dir == Path("student") and d in ("logs", "registry", "state")))
            if prefix == "swe-task" and rel_dir == Path("synth"):  # only the top-level scripts of each repair dir
                for repair in sorted((root / "synth" / "repair").iterdir()):
                    if repair.is_dir():
                        for path in sorted(repair.iterdir()):
                            rel = path.relative_to(root)
                            if path.is_file() and allowed(prefix, rel) and path.stat().st_size <= MAX_BYTES:
                                yield path, Path(prefix) / rel
            for name in sorted(filenames):
                path = Path(dirpath) / name
                if any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDE_NAME):
                    continue
                if not allowed(prefix, rel_dir / name) or str(Path(prefix) / rel_dir / name) in EXCLUDE_PATHS:
                    continue
                if name.endswith(".json") and (Path(dirpath) / name).stat().st_size > MAX_JSON_BYTES:
                    continue
                if path.suffix not in EXTENSIONS and name not in EXTRA_NAMES:
                    continue
                try:
                    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
                        continue
                except OSError:
                    continue
                yield path, Path(prefix) / rel_dir / name


def explicit_files():
    """ALLOW_FILES (repo paths under the two roots) and EXTERNAL (-> tools/external/)."""
    for repo_path in ALLOW_FILES:
        prefix, _, rel = repo_path.partition("/")
        source = ROOTS[prefix] / rel
        if source.is_file():
            yield source, Path(repo_path)
    for name, source in EXTERNAL.items():
        if source.is_file():
            yield source, Path("tools/external") / name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    files, hits, total = [], [], 0
    seen = set()
    for source, target in list(selected_files()) + list(explicit_files()):
        if target in seen:
            continue
        seen.add(target)
        data = source.read_bytes()
        if b"\0" in data[:4096]:
            continue  # binary
        text = data.decode("utf-8", "replace")
        for pattern, label in SECRET_PATTERNS:
            if (str(target), label) in ALLOW_HITS:
                continue
            for match in re.finditer(pattern, text):
                hits.append((str(target), label, match.group(0)[:12] + "..."))
        files.append((source, target, len(data)))
        total += len(data)
    by_top = {}
    for _, target, size in files:
        key = "/".join(target.parts[:3])
        count, nbytes = by_top.get(key, (0, 0))
        by_top[key] = (count + 1, nbytes + size)
    for key, (count, nbytes) in sorted(by_top.items()):
        print(f"{count:6d} files {nbytes / 1e6:8.2f} MB  {key}")
    print(f"TOTAL {len(files)} files, {total / 1e6:.1f} MB; secret-pattern hits: {len(hits)}")
    for hit in hits[:50]:
        print("  HIT", *hit)
    if hits:
        raise SystemExit("refusing: secret-looking content (review the hits; fix the source or extend the excludes)")
    if args.write:
        for prefix in list(ROOTS) + ["tools/external"]:
            shutil.rmtree(REPO / prefix, ignore_errors=True)
        for source, target, _ in files:
            destination = REPO / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        (REPO / "tools" / "external").mkdir(parents=True, exist_ok=True)
        (REPO / "tools" / "external" / "README.md").write_text(
            "Job-submission helpers that live outside the two mirrored trees (sourced by the sbatch/submit scripts):\n\n"
            + "".join(f"- `{name}` = `{source}` (sha256 `{hashlib.sha256(source.read_bytes()).hexdigest()}`)\n"
                      for name, source in EXTERNAL.items() if source.is_file()))
        manifest = REPO / "tools" / "MANIFEST.txt"
        manifest.write_text("".join(f"{hashlib.sha256((REPO / t).read_bytes()).hexdigest()}  {t}\n" for _, t, _ in files))
        print(f"wrote {len(files)} files under {REPO}")


if __name__ == "__main__":
    main()
