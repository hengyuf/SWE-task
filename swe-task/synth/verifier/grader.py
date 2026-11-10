#!/usr/bin/env python3
"""DeepSWE v1.1 task verifier — one shared script, entered via tests/test.sh.

Shared verbatim by every task (canonical copy: tools/verifier/grader.py,
synced + CI-checked by tools/sync_verifier.py). All per-task data lives in
config.json next to this file:

  base_commit    str   the upstream commit the task is built at; preimage for
                       per-file resets when applying patches
  p2p_node_ids   [str] pass-to-pass whitelist (must keep passing)
  f2p_node_ids   [str] fail-to-pass whitelist (prove the task is solved);
                       both materialized from the oracle-vs-nop differential
  grade          {...} how to READ the reports test.sh produced (see below)

Subcommands:
  grader.py prepare                setup, apply model.patch + test.patch
  grader.py grade [--apply-failed] reports -> reward.json (+ ctrf.json)
  grader.py patch-paths <patch>    print unique file paths a diff touches

$TESTS_DIR (default /tests), $VERIFIER_DIR (default /logs/verifier),
$APP_DIR (default /app) and $ARTIFACTS_DIR (default /logs/artifacts) are
overridable for testing/replays.

== prepare ==

Runs in $APP_DIR (pristine repo at base_commit; image build steps may have
modified tracked files in-tree, so resets are per-file, never repo-wide):
  1. reset ONLY the files model.patch touches to base_commit, then apply it.
     No patch => the base state is graded (reward 0 by construction). A
     patch that fails to apply => reward.json written with apply_failed=1
     and exit 0 — test.sh sees reward.json and stops before running suites.
  2. reset the files test.patch touches, then apply it loudly (a failure
     here is an infrastructure error: nonzero exit, no reward.json, so the
     test.sh trap writes the reward.txt=-1 crash sentinel).

== grade: whitelisted node ids -> reward.json ==

An id missing from every report counts as FAILED (absence == failure), as
does a skipped test. Duplicate ids across/within reports merge
worst-status-wins (passed < skipped < failed). Whitelist ids and report
names are both whitespace-stripped; any further name canonicalization a
reporter needs is a task-local fixup in test.sh, BEFORE grade runs.

  reward    binary 0/1 (ranking): 1 iff |f2p| > 0, every f2p passes AND
            no p2p fails
  f2p_total / f2p_passed / p2p_total / p2p_passed   raw counts
  f2p       f2p_passed / f2p_total   (0.0 if the bucket is empty: no
                                      fail-to-pass evidence = nothing solved)
  p2p       p2p_passed / p2p_total   (1.0 vacuously if empty)
  partial   (f2p_passed + p2p_passed) / (f2p_total + p2p_total)
  apply_failed  (only with --apply-failed) the submitted patch did not
                apply; counts come from the whitelists with zero passes

  config keys (under "grade"):
    format      "ctrf" | "junit"     report parser
    node_id     "suite.name" | "name"  (ctrf only) id derivation; junit
                                     always derives classname.name
    tool_label  str                  tool.name written into the synthesized
                                     ctrf.json (required CTRF provenance)
    reports     [path...]            parsed in order
"""
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
VERIFIER_DIR = Path(os.environ.get("VERIFIER_DIR", "/logs/verifier"))
APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/logs/artifacts"))
RANK = {"passed": 0, "skipped": 1, "failed": 2}


def log(msg):
    print(f"[verifier] {msg}", flush=True)


def load_config():
    return json.loads((TESTS_DIR / "config.json").read_text())


# --- patch helpers ---------------------------------------------------------

def patch_paths(text):
    """unique file paths a unified diff touches, in order of appearance"""
    seen, out = set(), []
    for line in text.splitlines():
        path = None
        m = re.match(r'^diff --git (?:"?a/(.*?)"?) (?:"?b/(.*?)"?)$', line)
        if m:
            path = m.group(2)
        elif line.startswith('+++ b/'):
            path = line[6:]
        elif line.startswith('--- a/'):
            path = line[6:]
        if path and path != '/dev/null' and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def read_patch(path):
    p = Path(path)
    return p.read_text(errors="replace") if p.exists() else ""


# --- prepare ---------------------------------------------------------------

def git(*args, **kw):
    return subprocess.run(["git", *args], cwd=APP_DIR, **kw)


def reset_paths(paths, ref):
    # per-file reset to the patch's preimage; files the patch does not touch
    # keep their image state, exactly as the agent environment had them
    for f in paths:
        if not f:
            continue
        rc = git("checkout", "-q", ref, "--", f,
                 stderr=subprocess.DEVNULL).returncode
        if rc != 0 and ref == "HEAD" and (APP_DIR / f).exists():
            # path is new in the patch (no preimage): drop any leftover copy
            subprocess.run(["rm", "-rf", "--", f], cwd=APP_DIR)


# >>> swe-task addition: test-path sanitizer (hengyuf 2026-10-19, mode 2026-10-20, v3 2026-10-21; NOT DeepSWE v1.1) >>
# Step "1b" of prepare, run right after model.patch is applied and before the test.patch reset (the only call is
# marked in cmd_prepare). DeepSWE resets only the files test.patch touches, so a model's edits to OTHER test files
# reached the graded run: added parametrize rows renumbered frozen pytest ids, rewritten P2P assertions passed, a
# model test file redeclared a hidden Go helper, model-added test files were collected by the hidden suite.
# Here every path model.patch touches -- old AND new names: modifications, additions, deletions, renames, copies --
# that matches a glob of config.json's top-level "test_paths" goes back to base_commit:
# `git checkout -q <base_commit> -- <path>` when base_commit has the path, otherwise the path is deleted. Each
# sanitized path is logged ("[verifier] test-sanitize: ..."). Without a "test_paths" key this step does nothing.
# Globs: repo-relative POSIX paths, anchored at the repository root; inside one segment * ? [...] follow fnmatch and
# never cross "/"; a "**" segment matches any number of whole segments (a final "**" at least one, so "tests/**" is
# everything below tests/); "**/x" matches x at any depth. Stdlib only, Python 3.8 compatible.
# Mode (hengyuf 2026-10-20, for JS/TS tasks): the optional top-level config.json key "test_paths_mode" is "all" (the
# default, also when the key is absent: everything above) or "non_additive": then an EXISTING test file (one
# base_commit has) whose model.patch change only ADDS lines keeps the model's version, logged
# "[verifier] test-sanitize: kept-additive <path>". Why: jest suites mock internal modules with factories
# (jest.mock('../Farm', () => ({default: ...}))), a correct solution that imports a new helper from such a module has
# to add it to the factory, and restoring the test file turned that purely additive edit into a false FAIL
# (official1-vitest/testem). "Only adds lines" is read from model.patch itself (additive_paths): every file block that
# names the path is a plain text modification of that same path -- no new or deleted file, rename, copy, mode change
# or binary patch -- with at least one hunk, and no hunk has a "-" line. Anything else (one changed or deleted line, a
# malformed hunk, a header line this parser does not know) is restored exactly as in "all"; model-added test files
# are always deleted. A "test_paths_mode" other than "all"/"non_additive" is an infra error like a bad "test_paths".
# Consistency (v3, hengyuf 2026-10-21; the sanitizer must not fail a correct solution): sanitizing must not leave the
# graded tree referring to what it took away, i.e. to a test path it deletes or restores that test.patch does not put
# back (a path test.patch adds or resets ends as DeepSWE's grader leaves it anyway).
# (1) Cargo manifests, any mode (official1-wasmi: sc14 added crates/wasmtime/tests/coredump.rs and registered it as
# `[[test]] name = "coredump"`; the file was deleted, the table stayed, and every cargo command failed on the
# manifest). In every Cargo.toml that model.patch touches and that is no test path itself, a [[test]]/[[bench]]/
# [[example]]/[[bin]] table that base_commit's version of that manifest does not hold verbatim and whose target the
# sanitizer deleted -- its `path`, or by name <dir>/<name>.rs or <dir>/<name>/main.rs (tests/, benches/, examples/,
# src/bin/) when none of those exists -- is cut out of the model's manifest; everything else in it (dependencies,
# features) stays. Logged "[verifier] test-sanitize: dropped-manifest-target <manifest> [[<kind>]] <name> -> <target>
# ...". A table that is the model's edit of a base_commit table (same kind and name, the base table no longer in the
# model's manifest, its target present again) gets base_commit's table back instead ("restored-manifest-target"). A
# manifest the line scanner cannot follow (an unterminated string or array) is left as it is and logged.
# (2) Imports, mode "non_additive" (official1-dynamodb): a kept-additive file whose ADDED lines import a deleted or
# restored test path would keep a dangling import ('Cannot find module'; TS2305 for a symbol the restored file lacks).
# It is restored as in "all", logged "[verifier] test-sanitize: dangling-import <path>: ..." before its "restored"
# line, until no kept file imports such a path (a restored file counts for the others). Read from the JS/TS kept file
# itself (dangling_imports): every import/export ... from, import(), require(), require.resolve(), jest./vi. mock or
# requireActual-like call and /// <reference path> whose text includes an added line (a symbol added to an existing
# multi-line import counts). Relative specifiers resolve from the file's directory (extension and /index optional);
# bare ones of two or more segments match by path suffix ('server/actions' under a baseUrl, '@/lib/x'); a one-segment
# bare specifier is a package.
import fnmatch  # noqa: E402
import posixpath  # noqa: E402
import shutil  # noqa: E402

SANITIZE_EXIT = 7  # bad "test_paths" or a test path that cannot be put back: no reward.json, i.e. infra error
GIT_ESCAPES = {"a": 7, "b": 8, "t": 9, "n": 10, "v": 11, "f": 12, "r": 13, '"': 34, "\\": 92}
HUNK_HEADER = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
HUNK_RANGES = re.compile(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")  # old count, new start, new count
TEST_PATHS_MODES = ("all", "non_additive")


def glob_match(pattern, path):
    """True iff the repo-relative path matches the glob (rules above)."""
    pat = [s for s in pattern.split("/") if s not in ("", ".")]
    seg = [s for s in path.split("/") if s not in ("", ".")]
    memo = {}

    def match(i, j):
        if (i, j) not in memo:
            if i == len(pat):
                ok = j == len(seg)
            elif pat[i] == "**":
                if i == len(pat) - 1:
                    ok = j < len(seg)
                else:
                    ok = any(match(i + 1, k) for k in range(j, len(seg) + 1))
            else:
                ok = j < len(seg) and fnmatch.fnmatchcase(seg[j], pat[i]) and match(i + 1, j + 1)
            memo[(i, j)] = ok
        return memo[(i, j)]

    return bool(pat) and match(0, 0)


def git_unquote(name):
    """A name as git prints it ("C-quoted" when it holds special characters) -> the real path."""
    if len(name) < 2 or name[0] != '"' or name[-1] != '"':
        return name
    body, raw, i = name[1:-1], bytearray(), 0
    while i < len(body):
        if body[i] == "\\" and re.match(r"[0-3][0-7]{2}$", body[i + 1:i + 4]):
            raw.append(int(body[i + 1:i + 4], 8))
            i += 4
        elif body[i] == "\\" and body[i + 1:i + 2] in GIT_ESCAPES:
            raw.append(GIT_ESCAPES[body[i + 1]])
            i += 2
        else:
            raw += body[i].encode("utf-8", "surrogateescape")
            i += 1
    return raw.decode("utf-8", "surrogateescape")


def _header_names(rest):
    """The two names of `diff --git <old> <new>` (prefixes kept). Unquoted names are split where both halves name the
    same file, as git apply does; a rename/copy with unquoted names gets its names from its rename/copy lines."""
    if rest.startswith('"'):
        i = 1
        while i < len(rest) and rest[i] != '"':
            i += 2 if rest[i] == "\\" else 1
        return [rest[:i + 1], rest[i + 1:].lstrip(" ")]
    middle = len(rest) // 2
    if len(rest) % 2 == 1 and rest[middle] == " ":
        old, new = rest[:middle], rest[middle + 1:]
        if "/" in old and "/" in new and old.split("/", 1)[1] == new.split("/", 1)[1]:
            return [old, new]
    return []


def touched_paths(text):
    """Every path a git diff touches, old and new names (renames, copies, deletions included), in order of first
    appearance. Reads the header lines of each file (diff --git, ---/+++, rename/copy from/to) and skips hunk bodies
    by their line counts, so changed lines that merely look like headers are never taken for paths."""
    out = []

    def add(name, strip_prefix):
        name = git_unquote(name)
        if strip_prefix:  # a/<path>, b/<path>: model.patch is applied with git apply's default -p1
            if "/" not in name:
                return
            name = name.split("/", 1)[1]
        if name and not name.startswith("/") and ".." not in name.split("/") and name not in out:
            out.append(name)  # (git apply refuses paths outside the work tree, so ../x never got applied)

    old_left = new_left = 0
    binary = False
    for line in text.split("\n"):
        if old_left > 0 or new_left > 0:
            tag = line[:1]
            if tag in (" ", ""):
                old_left, new_left = old_left - 1, new_left - 1
                continue
            if tag == "-":
                old_left -= 1
                continue
            if tag == "+":
                new_left -= 1
                continue
            if tag == "\\":
                continue
            old_left = new_left = 0  # a malformed hunk ends here; read the line as a header line
        line = line.rstrip("\r")
        hunk = HUNK_HEADER.match(line)
        if line.startswith("diff --git "):
            binary = False
            for name in _header_names(line[len("diff --git "):]):
                add(name, True)
        elif binary:
            continue
        elif line.startswith("GIT binary patch"):
            binary = True
        elif hunk:
            old_left = int(hunk.group(1)) if hunk.group(1) is not None else 1
            new_left = int(hunk.group(2)) if hunk.group(2) is not None else 1
        elif line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
            add(line.split(" ", 2)[2], False)
        elif line.startswith(("--- ", "+++ ")):
            name = line[4:].split("\t", 1)[0]  # git ends a name holding a space with a TAB
            if name != "/dev/null":
                add(name, True)
    return out


def _only_adds_lines(block):
    """True iff one file block of a git diff (its "diff --git" line first) is a plain text modification whose hunks,
    at least one, hold only context and "+" lines. Any other header line (new/deleted file, old/new mode, rename,
    copy, similarity, binary) or a malformed hunk makes it False."""
    old_left = new_left = hunks = 0
    lines = block.split("\n")
    if lines[-1] == "":  # the text after the block's final newline is not a line (a truncated hunk stays truncated)
        lines.pop()
    for line in lines[1:]:
        if old_left > 0 or new_left > 0:
            tag = line[:1]
            if tag in (" ", ""):
                old_left, new_left = old_left - 1, new_left - 1
            elif tag == "+":
                new_left -= 1
            elif tag != "\\":
                return False  # a "-" line, or a line that does not belong to a hunk of these counts
            continue
        line = line.rstrip("\r")
        hunk = HUNK_HEADER.match(line)
        if hunk:
            hunks += 1
            old_left = int(hunk.group(1)) if hunk.group(1) is not None else 1
            new_left = int(hunk.group(2)) if hunk.group(2) is not None else 1
        elif line.startswith(("--- ", "+++ ")):
            if line[4:].split("\t", 1)[0] == "/dev/null":
                return False  # a new or a deleted file
        elif line and not line.startswith(("index ", "\\")):  # "\ No newline at end of file" after a hunk's last line
            return False
    return hunks > 0 and old_left <= 0 and new_left <= 0


def additive_paths(text):
    """The paths a git diff changes ONLY by adding lines: every file block naming the path (touched_paths of the
    block, old and new names) names just that path and _only_adds_lines. A path also named by any other block, or by
    text before the first "diff --git" line, is not additive."""
    verdict = {}
    for block in re.split(r"(?m)^(?=diff --git )", text):
        names = touched_paths(block)
        additive = block.startswith("diff --git ") and len(names) == 1 and _only_adds_lines(block)
        for name in names:
            verdict[name] = verdict.get(name, True) and additive
    return {name for name, additive in verdict.items() if additive}


def added_lines(text):
    """{path: the 1-based numbers, in the model's version, of the lines its file block adds} for the blocks of a git
    diff that name one path (plain modifications and new files)."""
    out = {}
    for block in re.split(r"(?m)^(?=diff --git )", text):
        names = touched_paths(block)
        if not block.startswith("diff --git ") or len(names) != 1:
            continue
        numbers = out.setdefault(names[0], set())
        old_left = new_left = number = 0
        for line in block.split("\n")[1:]:
            if old_left > 0 or new_left > 0:
                tag = line[:1]
                if tag in (" ", "", "+"):
                    if tag == "+":
                        numbers.add(number)
                    else:
                        old_left -= 1
                    new_left, number = new_left - 1, number + 1
                    continue
                if tag == "-":
                    old_left -= 1
                    continue
                if tag == "\\":
                    continue
                old_left = new_left = 0  # a malformed hunk ends here
            hunk = HUNK_RANGES.match(line.rstrip("\r"))
            if hunk:
                old_left = int(hunk.group(1)) if hunk.group(1) is not None else 1
                number = int(hunk.group(2))
                new_left = int(hunk.group(3)) if hunk.group(3) is not None else 1
    return out


JS_SOURCES = (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts")
JS_MODULE_EXTENSIONS = (".d.ts",) + JS_SOURCES + (".json",)
# import a, {b,\n c} from 'm' | export {a} from 'm' | const {a,\n b} = require('m') | require('m'), import('m'),
# jest.mock('m')... | import 'm' | /// <reference path="m" />. The specifier is the even-numbered group that matched.
JS_MODULE_REF = re.compile(
    r"\bimport\s+(?:type\s+)?[^'\"`;]*?\bfrom\s*(['\"])([^'\"\n]+)\1"
    r"|\bexport\s+(?:type\s+)?(?:\*\s*(?:as\s+[\w$]+\s*)?|\{[^}]*\}\s*)from\s*(['\"])([^'\"\n]+)\3"
    r"|\b(?:const|let|var)\s+\{[^}]*\}\s*=\s*require\s*\(\s*(['\"`])([^'\"`\n]+)\5"
    r"|\b(?:import|require(?:\.resolve)?|(?:jest|vi)\.(?:mock|doMock|unmock|dontMock|setMock|requireActual|"
    r"requireMock|importActual|importMock|createMockFromModule))\s*\(\s*(['\"`])([^'\"`\n]+)\7"
    r"|\bimport\s*(['\"])([^'\"\n]+)\9"
    r"|///\s*<reference\s+path\s*=\s*(['\"])([^'\"\n]+)\11")


def module_keys(path):
    """The names a module path answers to: itself, without a JS/TS/JSON extension, and its directory for index.*."""
    stem = next((path[:-len(ext)] for ext in JS_MODULE_EXTENSIONS if path.endswith(ext) and len(path) > len(ext)),
                path)
    keys = {path, stem}
    if stem == "index" or stem.endswith("/index"):
        keys.add(stem[:-len("index")].rstrip("/"))
    keys.discard("")
    return keys


def import_targets(spec, importer, paths):
    """The paths (of `paths`) a module specifier written in the file `importer` may resolve to. Relative specifiers
    resolve from the importer's directory; bare ones of two or more segments match by path suffix ('server/actions' ->
    tests/jest/server/actions.ts under a baseUrl; '@/x/y', '~/x/y', '#x/y' and '@alias/x/y' also without their first
    segment). A one-segment bare specifier ('zsa', '@scope/pkg' without its scope) is a package; URLs, node: and
    absolute specifiers match nothing."""
    spec = spec.split("?", 1)[0].strip()
    if not spec or spec.startswith("/") or ":" in spec:
        return []
    if spec in (".", "..") or spec.startswith(("./", "../")):
        joined = posixpath.normpath(posixpath.join(posixpath.dirname(importer), spec))
        if joined == ".." or joined.startswith("../"):
            return []
        wanted = module_keys(joined)
        return [p for p in paths if module_keys(p) & wanted]
    tails = {spec.lstrip("#~@/")}
    first, _, rest = spec.partition("/")
    if first[:1] in ("@", "~", "#"):
        tails.add(rest)
    wanted = set()
    for tail in tails:
        if "/" in tail.strip("/"):
            wanted |= module_keys(tail)
    return [p for p in paths if any(key == w or key.endswith("/" + w) for key in module_keys(p) for w in wanted)]


def dangling_ref(path, lines, gone):
    """(specifier, target) of the first module reference in the JS/TS file `path` (its version in APP_DIR) whose text
    includes one of `lines` and that may resolve to a path of `gone`; None if there is none."""
    if not lines or not path.endswith(JS_SOURCES):
        return None
    try:
        content = (APP_DIR / path).read_bytes().decode("utf-8", "surrogateescape")
    except OSError:
        return None
    number, position = 1, 0
    for match in JS_MODULE_REF.finditer(content):
        number += content.count("\n", position, match.start())
        position = match.start()
        if not any(n in lines for n in range(number, number + match.group(0).count("\n") + 1)):
            continue
        spec = next(group for group in match.groups()[1::2] if group is not None)
        targets = import_targets(spec, path, gone)
        if targets:
            return spec, targets[0]
    return None


def dangling_imports(text, kept, gone, skip=()):
    """(the kept-additive files still kept, {path: (specifier, target)} of those to restore): a file whose added lines
    import a path of `gone` (deleted or restored here) is restored, and is then a restored path for the others, until
    nothing changes. Files of `skip` (test.patch resets them anyway) are neither read nor restored."""
    added, kept, gone, found = added_lines(text), list(kept), list(gone), {}
    changed = True
    while changed:
        changed = False
        for path in [p for p in kept if p not in skip]:
            ref = dangling_ref(path, added.get(path, set()), gone)
            if ref:
                found[path] = ref
                kept.remove(path)
                gone.append(path)
                changed = True
    return kept, found


TARGET_HEADER = re.compile(r"^\s*\[\[\s*(test|bench|example|bin)\s*\]\]\s*(?:#.*)?$")
TARGET_KEY = re.compile(r"""^\s*(?:(name|path)|"(name|path)"|'(name|path)')\s*=\s*(.*?)\s*$""")
TARGET_DIRS = {"test": "tests", "bench": "benches", "example": "examples", "bin": "src/bin"}
TOML_ESCAPES = {"b": "\b", "t": "\t", "n": "\n", "f": "\f", "r": "\r", "e": "\x1b", '"': '"', "\\": "\\"}


def toml_string(value):
    """The text of the one-line TOML basic or literal string `value` starts with; None for anything else."""
    if value.startswith(("'''", '"""')):
        return None
    if value.startswith("'"):
        end = value.find("'", 1)
        return value[1:end] if end > 0 else None
    out, i = [], 1
    while value.startswith('"') and i < len(value):
        char, code = value[i], value[i + 1:i + 2]
        if char == '"':
            return "".join(out)
        if char != "\\":
            out.append(char)
            i += 1
        elif code in TOML_ESCAPES:
            out.append(TOML_ESCAPES[code])
            i += 2
        elif code in ("u", "U"):
            digits = value[i + 2:i + (6 if code == "u" else 10)]
            if len(digits) != (4 if code == "u" else 8) or not re.match(r"[0-9A-Fa-f]+$", digits) \
                    or int(digits, 16) > 0x10FFFF:
                return None
            out.append(chr(int(digits, 16)))
            i += 2 + len(digits)
        else:
            return None
    return None


def _toml_scan(line, state):
    """Carry (the delimiter of an open multi-line string or None, open [ and { count) over one line of TOML; None
    when a one-line string is not closed on it."""
    quote, depth = state
    i = 0
    while i < len(line):
        if quote:  # inside a multi-line string: find its end (a basic string's \" does not end it)
            end = line.find(quote, i)
            while quote == '"""' and end >= 0 and (len(line[i:end]) - len(line[i:end].rstrip("\\"))) % 2:
                end = line.find(quote, end + 1)
            if end < 0:
                return quote, depth
            i = end + 3
            while i < len(line) and line[i] == quote[0]:  # up to two quotes before the delimiter belong to the string
                i += 1
            quote = None
            continue
        char = line[i]
        if char == "#":
            break
        if line.startswith(('"""', "'''"), i):
            quote, i = line[i:i + 3], i + 3
            continue
        if char in "\"'":
            end = i + 1
            while end < len(line) and line[end] != char:
                end += 2 if char == '"' and line[end] == "\\" else 1
            if end >= len(line):
                return None
            i = end + 1
            continue
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth = max(depth - 1, 0)
        i += 1
    return quote, depth


def toml_target_tables(text):
    """[{kind, first, last, name, path, body}] for the [[test]]/[[bench]]/[[example]]/[[bin]] tables of a Cargo
    manifest: line indexes (of text.split("\\n")) of the header and of the table's last key/value line, the one-line
    string values of its name and path keys, and body = its stripped lines without blank and comment lines. None when
    the line scanner cannot follow the file (a one-line string not closed, a multi-line string or array still open at
    the end)."""
    tables, current, state = [], None, (None, 0)
    for index, line in enumerate(text.split("\n")):
        stripped = line.strip()
        if state == (None, 0):
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("["):
                kind = TARGET_HEADER.match(line)
                current = {"kind": kind.group(1), "first": index, "last": index, "name": None, "path": None,
                           "body": [stripped]} if kind else None
                if current:
                    tables.append(current)
                continue
            key = TARGET_KEY.match(line)
            if current and key:
                field = key.group(1) or key.group(2) or key.group(3)
                if current[field] is None:
                    current[field] = toml_string(key.group(4))
        state = _toml_scan(line, state)
        if state is None:
            return None
        if current:
            current["last"] = index
            if stripped and not stripped.startswith("#"):
                current["body"].append(stripped)
    if state != (None, 0):
        return None
    for table in tables:
        table["body"] = tuple(table["body"])
    return tables


def target_paths(manifest, table):
    """Where a Cargo target table points, repo-relative: [its `path`] or, by name, [<dir>/<name>.rs,
    <dir>/<name>/main.rs] (tests/, benches/, examples/, src/bin/ next to the manifest)."""
    if table["path"] is not None:
        names = [table["path"]]
    elif table["name"]:
        folder = TARGET_DIRS[table["kind"]]
        names = [f"{folder}/{table['name']}.rs", f"{folder}/{table['name']}/main.rs"]
    else:
        return []
    out = []
    for name in names:
        joined = posixpath.normpath(posixpath.join(posixpath.dirname(manifest), name))
        if name and not name.startswith("/") and joined != ".." and not joined.startswith("../"):
            out.append(joined)
    return out


def drop_manifest_targets(manifests, base, gone):
    """In these Cargo manifests (the model's versions), every target table that base_commit's manifest does not hold
    verbatim and that points at a path of `gone` (deleted here) while none of its target paths exists is cut out, or,
    when it is the model's edit of a base_commit table (same kind and name; that table is no longer in the model's
    manifest and its target exists), replaced by base_commit's table. Returns [(manifest, action)] per table."""
    done, gone = [], set(gone)
    for manifest in manifests:
        path = APP_DIR / manifest
        if path.is_symlink() or not path.is_file():
            continue
        try:
            text = path.read_bytes().decode("utf-8", "surrogateescape")
        except OSError as error:
            log(f"test-sanitize: manifest-unread {json.dumps(manifest)} ({error}): left as it is")
            continue
        tables = toml_target_tables(text)
        if tables is None:
            log(f"test-sanitize: manifest-unread {json.dumps(manifest)} (a string or array this scanner cannot "
                "follow): its target tables are left as they are")
            continue
        dangling = []
        for table in tables:
            targets = target_paths(manifest, table)
            if any(t in gone for t in targets) and not any((APP_DIR / t).exists() for t in targets):
                dangling.append((table, next(t for t in targets if t in gone)))
        if not dangling:
            continue
        shown = git("show", f"{base}:{manifest}", stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        base_text = shown.stdout.decode("utf-8", "surrogateescape") if shown.returncode == 0 else ""
        base_tables = toml_target_tables(base_text) or []
        lines, base_lines = text.split("\n"), base_text.split("\n")
        end = len(lines) - 1 if lines[-1] == "" else len(lines)  # lines[:end] are the real lines
        cut, replaced = set(), {}
        for table, target in dangling:
            if table["body"] in {t["body"] for t in base_tables}:  # base_commit's own table: never touched
                continue
            first, last = table["first"], table["last"]
            twins = [t for t in base_tables if t["kind"] == table["kind"] and t["name"] is not None
                     and t["name"] == table["name"] and t["body"] not in {m["body"] for m in tables}]
            if len(twins) == 1 and any((APP_DIR / t).exists() for t in target_paths(manifest, twins[0])):
                replaced[first] = (last, base_lines[twins[0]["first"]:twins[0]["last"] + 1])
                action, note = "restored-manifest-target", "base_commit's table is back"
            else:
                cut.update(range(first, last + 1))
                if first > 0 and not lines[first - 1].strip():  # one blank line of the two around it goes too
                    if last + 1 < end and not lines[last + 1].strip():
                        cut.add(last + 1)
                    elif last + 1 >= end:
                        cut.add(first - 1)
                action, note = "dropped-manifest-target", "cut out"
            log(f"test-sanitize: {action} {json.dumps(manifest)} [[{table['kind']}]] {json.dumps(table['name'] or '')}"
                f" -> {json.dumps(target)} (deleted above): {note}")
            done.append((manifest, action))
        if not cut and not replaced:
            continue
        out, index = [], 0
        while index < len(lines):
            if index in replaced:
                out.extend(replaced[index][1])
                index = replaced[index][0] + 1
                continue
            if index not in cut:
                out.append(lines[index])
            index += 1
        try:
            path.write_bytes("\n".join(out).encode("utf-8", "surrogateescape"))
        except OSError as error:
            log(f"ERROR: test-sanitize: could not rewrite {json.dumps(manifest)}: {error}")
            sys.exit(SANITIZE_EXIT)
    return done


def test_path_globs(config):
    """config.json's "test_paths" globs; None when the key is absent (the sanitizer is off)."""
    if "test_paths" not in config:
        return None
    globs = config["test_paths"]
    if not isinstance(globs, list) or not all(isinstance(g, str) and g.strip("/. ") for g in globs):
        log('ERROR: test-sanitize: config.json "test_paths" must be a list of glob strings')
        sys.exit(SANITIZE_EXIT)
    return globs


def sanitize_test_paths(patch_file, base, globs, mode="all"):
    """Put every test path model.patch touches back to base_commit (mode "non_additive": except the existing test
    files it only adds lines to, unless those lines import a path put back here), then drop the Cargo target tables
    that point at a deleted path; returns [(path, action)]."""
    if globs is None:
        return []
    if mode not in TEST_PATHS_MODES:
        log(f'ERROR: test-sanitize: config.json "test_paths_mode" must be one of {json.dumps(list(TEST_PATHS_MODES))}')
        sys.exit(SANITIZE_EXIT)
    text = Path(patch_file).read_bytes().decode("utf-8", "surrogateescape")
    touched = touched_paths(text)
    hits = [p for p in touched if any(glob_match(g, p) for g in globs)]
    log(f"test-sanitize: {len(hits)} path(s) of model.patch match test_paths {json.dumps(globs)}")
    if mode != "all":
        log(f"test-sanitize: test_paths_mode {mode}: existing test files that model.patch only adds lines to are kept")
    if not hits:
        return []
    if git("rev-parse", "-q", "--verify", f"{base}^{{commit}}", stdout=subprocess.DEVNULL,
           stderr=subprocess.DEVNULL).returncode != 0:  # never mistake an unknown base for "path not in base"
        log(f"ERROR: test-sanitize: base_commit {base} is not a commit of {APP_DIR}")
        sys.exit(SANITIZE_EXIT)
    at_base = {p: git("rev-parse", "-q", "--verify", f"{base}:{p}", stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL).returncode == 0 for p in hits}
    additive = additive_paths(text) if mode == "non_additive" else set()
    kept = [p for p in hits if at_base[p] and p in additive]
    # v3: what the graded tree lacks because of this step -- not the paths test.patch puts back (DeepSWE's grader and
    # the reference were graded with test.patch's versions of those)
    provided = set(patch_paths(read_patch(TESTS_DIR / "test.patch")))
    gone = [p for p in hits if p not in kept and p not in provided]
    kept, dangling = dangling_imports(text, kept, gone, provided) if kept else (kept, {})
    done = []
    for p in [p for p in hits if not at_base[p]]:  # deletions first: they may make room for a restored path
        target = APP_DIR / p
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(str(target))
        else:
            log(f"test-sanitize: absent {json.dumps(p)} (not in base_commit, nothing to delete)")
            continue
        if target.is_symlink() or target.exists():
            log(f"ERROR: test-sanitize: could not delete {json.dumps(p)}")
            sys.exit(SANITIZE_EXIT)
        log(f"test-sanitize: deleted {json.dumps(p)} (not in base_commit)")
        done.append((p, "deleted"))
    for p in kept:
        log(f"test-sanitize: kept-additive {json.dumps(p)}")
        done.append((p, "kept-additive"))
    for p, (spec, target) in dangling.items():
        how = "restored" if at_base[target] else "deleted"
        log(f"test-sanitize: dangling-import {json.dumps(p)}: its added lines import {json.dumps(spec)} -> "
            f"{json.dumps(target)} ({how} here): restored like a changed file")
    for p in [p for p in hits if at_base[p] and p not in kept]:
        r = git("--literal-pathspecs", "checkout", "-q", base, "--", p, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, universal_newlines=True)
        if r.returncode != 0:
            log(f"ERROR: test-sanitize: git checkout {base} -- {json.dumps(p)} failed (exit {r.returncode}): "
                f"{r.stdout.strip()[-300:]}")
            sys.exit(SANITIZE_EXIT)
        log(f"test-sanitize: restored {json.dumps(p)} from base_commit")
        done.append((p, "restored"))
    manifests = [p for p in touched if posixpath.basename(p) == "Cargo.toml" and p not in hits]
    return done + drop_manifest_targets(manifests, base, [p for p in gone if not at_base[p]])
# <<< swe-task addition: test-path sanitizer <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def cmd_prepare(argv):
    if not APP_DIR.is_dir():
        VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
        sys.exit(6)
    os.chdir(APP_DIR)
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "config", "--global", "--add", "safe.directory",
                    str(APP_DIR)], stderr=subprocess.DEVNULL)
    base = load_config()["base_commit"]
    model_patch = ARTIFACTS_DIR / "model.patch"
    if model_patch.exists() and model_patch.stat().st_size > 0:
        reset_paths(patch_paths(read_patch(model_patch)), base)
        rc = git("apply", "--whitespace=nowarn", str(model_patch)).returncode
        if rc != 0:
            log("ERROR: submitted model.patch failed to apply")
            cmd_grade(["--apply-failed"])
            sys.exit(0)
        log(f"model.patch applied ({model_patch.stat().st_size} bytes)")
        # >>> swe-task addition: step 1b, test-path sanitizer (see its section above cmd_prepare) >>>
        config = load_config()
        sanitize_test_paths(model_patch, base, test_path_globs(config), config.get("test_paths_mode", "all"))
        # <<< swe-task addition <<<
    else:
        log("no model.patch submitted — grading pristine base state")

    test_patch = TESTS_DIR / "test.patch"
    log("Resetting files touched by test.patch")
    reset_paths(patch_paths(read_patch(test_patch)), "HEAD")
    log("Applying test.patch")
    r = git("apply", "--whitespace=nowarn", "--allow-empty", str(test_patch),
            capture_output=True, text=True)
    if r.returncode != 0:
        log("ERROR: test.patch failed to apply")
        sys.stderr.write(r.stdout + r.stderr)
        sys.exit(r.returncode)
    try:
        inner = APP_DIR / "test.sh"
        inner.chmod(inner.stat().st_mode | 0o111)
    except OSError:
        pass


# --- grade -----------------------------------------------------------------

def norm_status(raw):
    raw = str(raw or "").strip().lower()
    if raw == "passed":
        return "passed"
    if raw in ("skipped", "pending", "other"):
        return "skipped"
    return "failed"


def add(res, nid, st, msg=""):
    # worst-status-wins: failed > skipped > passed; keep the failing entry's
    # full message. value is a (status, message) tuple.
    cur = res.get(nid)
    msg = msg or ""
    if cur is None or RANK[st] > RANK[cur[0]]:
        res[nid] = (st, msg if st != "passed" else "")
    elif RANK[st] == RANK[cur[0]] and st != "passed" and not cur[1] and msg:
        res[nid] = (st, msg)


def parse_ctrf(path, cfg):
    """report path -> {node_id: (status, failure_message)}"""
    res = {}
    try:
        doc = json.loads(Path(path).read_text())
        tests = (doc.get("results") or {}).get("tests") or []
        if not isinstance(tests, list):
            return res
    except Exception:
        return res
    for tc in tests:
        if not isinstance(tc, dict):
            continue
        nm = str(tc.get("name") or "").strip()
        if not nm:
            continue
        su_raw = tc.get("suite")
        if isinstance(su_raw, list) and su_raw:
            su = str(su_raw[0]).strip()
        elif isinstance(su_raw, str):
            su = su_raw.strip()
        else:
            su = ""
        nid = f"{su}.{nm}" if (cfg.get("node_id") == "suite.name" and su) else nm
        st = norm_status(tc.get("status"))
        msg = ""
        if st != "passed":
            msg = str(tc.get("message") or tc.get("trace") or "").strip()
        add(res, nid, st, msg)
    return res


def junit_status_msg(tc):
    st, msg = "passed", ""
    for ch in tc:
        tag = ch.tag.rsplit("}", 1)[-1]
        if tag in ("failure", "error"):
            parts = [(ch.get("message") or "").strip(), (ch.text or "").strip()]
            return "failed", "\n".join(p for p in parts if p).strip()
        if tag == "skipped":
            st = "skipped"
    return st, msg


def parse_junit(path, cfg):
    """report path -> {node_id: (status, failure_message)}"""
    res = {}
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return res
    for tc in root.iter("testcase"):
        cn = (tc.attrib.get("classname", "") or "").strip()
        nm = (tc.attrib.get("name", "") or "").strip()
        if not nm:
            continue
        nid = f"{cn}.{nm}" if cn else nm
        st, msg = junit_status_msg(tc)
        add(res, nid, st, msg)
    return res


PARSERS = {"ctrf": parse_ctrf, "junit": parse_junit}


def cmd_grade(argv):
    full = load_config()
    cfg = full.get("grade", {})
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)

    def load_ids(key):
        ids, seen = [], set()
        for line in full.get(key, []):
            s = str(line).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            ids.append(s)
        return ids

    p2p = load_ids("p2p_node_ids")
    f2p = load_ids("f2p_node_ids")

    def stats(fp, pp):
        total = len(f2p) + len(p2p)
        return {"f2p_total": len(f2p), "f2p_passed": fp,
                "p2p_total": len(p2p), "p2p_passed": pp,
                "f2p": fp / len(f2p) if f2p else 0.0,
                "p2p": pp / len(p2p) if p2p else 1.0,
                "partial": (fp + pp) / total if total else 0.0}

    if "--apply-failed" in argv:
        out = {"reward": 0, **stats(0, 0), "apply_failed": 1}
        (VERIFIER_DIR / "reward.json").write_text(json.dumps(out))
        print(f"[grade] model.patch failed to apply; reward.json={json.dumps(out)}")
        return
    parse = PARSERS[cfg.get("format", "ctrf")]
    seen = {}
    for rep in cfg["reports"]:
        for k, (st, msg) in parse(rep, cfg).items():
            add(seen, k, st, msg)

    def bucket(ids):
        p = f = 0
        rows = []
        for nid in ids:
            entry = seen.get(nid)
            if entry is None:
                rows.append({"name": nid, "status": "failed",
                             "message": "missing from report (test did not run "
                                        "or produced no result — see raw output)"})
                f += 1
            elif entry[0] == "passed":
                rows.append({"name": nid, "status": "passed"})
                p += 1
            else:
                rows.append({"name": nid, "status": entry[0], "message": entry[1]})
                f += 1
        return p, f, rows

    pp, pf, pr = bucket(p2p)
    fp, ff, fr = bucket(f2p)
    binary = 1 if (len(f2p) > 0 and ff == 0 and pf == 0) else 0

    def ctrf_test(t, b):
        d = {"name": f"[{b}] {t['name']}", "status": t["status"]}
        if t.get("message"):
            d["message"] = t["message"]
        return d

    ctrf = {"reportFormat": "CTRF", "specVersion": "1.0.0", "results": {
        "tool": {"name": cfg.get("tool_label", "unknown")},
        "summary": {"tests": len(p2p)+len(f2p), "passed": pp+fp,
                    "failed": pf+ff, "skipped": 0, "pending": 0, "other": 0},
        "tests": [ctrf_test(t, "p2p") for t in pr]
                + [ctrf_test(t, "f2p") for t in fr]}}
    (VERIFIER_DIR / "ctrf.json").write_text(json.dumps(ctrf, indent=2))

    out = {"reward": binary, **stats(fp, pp)}
    (VERIFIER_DIR / "reward.json").write_text(json.dumps(out))

    # Surface WHY each whitelisted test failed (lands in test-stdout.txt via the
    # harness capture). Reasons come from the report message; if absent, the raw
    # suite output catted by the frame is the fallback.
    fails = ([("p2p", t) for t in pr if t["status"] != "passed"]
             + [("f2p", t) for t in fr if t["status"] != "passed"])
    if fails:
        print(f"[verifier] ===== FAILURES ({len(fails)}) =====")
        for b, t in fails:
            print(f"[verifier] ✗ [{b}] {t['name']}")
            for line in (t.get("message") or "(no message)").splitlines():
                print(f"    {line}")
    print(f"P2P {pp}/{len(p2p)} pass {pf} fail; F2P {fp}/{len(f2p)} pass {ff} fail; "
          + f"PARTIAL {out['partial']}; BINARY {binary}")


def cmd_patch_paths(argv):
    for path in patch_paths(read_patch(argv[0])):
        print(path)


def main():
    cmds = {"prepare": cmd_prepare, "grade": cmd_grade,
            "patch-paths": cmd_patch_paths}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(f"usage: grader.py {{{'|'.join(cmds)}}} [args]", file=sys.stderr)
        sys.exit(2)
    cmds[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
