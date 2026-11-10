"""config.json "test_paths" / "test_paths_mode" of the sanitizing grader (verifier/grader.py) per repository language.

New builds write both keys (task_builder.write_tests, hengyuf 2026-10-20) and choose them exactly as
tools/grading_repair.py chooses them for the repair of existing tasks: the top100.jsonl "language" of the build's
repository (repo_language), that language's default globs (GLOBS) and mode (MODES, "all" for the other languages).
tests/test_offline.py asserts that GLOBS and MODES equal grading_repair's, so the two copies cannot drift. A language
without globs raises UnknownLanguage: a build never silently goes without the sanitizer. Stdlib only.
"""
import json
from pathlib import Path

GLOBS = {  # hengyuf's defaults per repository language (= tools/grading_repair.py GLOBS)
    "Python": ["tests/**", "test/**", "**/test_*.py", "**/*_test.py", "**/conftest.py"],
    "Go": ["**/*_test.go", "**/testdata/**"],
    "JavaScript": ["**/*.test.*", "**/*.spec.*", "**/__tests__/**", "test/**", "tests/**"],
    "TypeScript": ["**/*.test.*", "**/*.spec.*", "**/__tests__/**", "test/**", "tests/**"],
    "Rust": ["tests/**"],
}
MODES = {"JavaScript": "non_additive", "TypeScript": "non_additive"}  # (= grading_repair.MODES)
DEFAULT_MODE = "all"  # every other language: restore every test path a submission touches


class UnknownLanguage(LookupError):
    """No test-path globs for the build's repository (not in top100.jsonl, ambiguous there, or a language without
    GLOBS). Deliberately not a RuntimeError: task_builder treats those as authoring problems to retry."""


def repo_language(top100, target: str, repo: str) -> str:
    """The top100.jsonl "language" of `repo`: from its rows for `target`, else from all of its rows; it must be unique
    (tools/grading_repair.repo_language)."""
    rows = [json.loads(line) for line in Path(top100).read_text().splitlines() if line.strip()]
    mine = [row for row in rows if row["repo"].lower() == repo.lower()]
    exact = sorted({row["language"] for row in mine if row["task_id"] == target})
    languages = exact or sorted({row["language"] for row in mine})
    if len(languages) != 1:
        raise UnknownLanguage(f"top100.jsonl gives {languages} for {repo}")
    return languages[0]


def config_keys(language: str) -> dict:
    """{"test_paths": globs, "test_paths_mode": mode} for a top100.jsonl repository language."""
    if language not in GLOBS:
        raise UnknownLanguage(f"no test-path globs for repository language {language!r} (verifier/test_paths.py)")
    return {"test_paths": list(GLOBS[language]), "test_paths_mode": MODES.get(language, DEFAULT_MODE)}
