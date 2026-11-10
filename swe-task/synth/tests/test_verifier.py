"""Offline tests for verifier/grader.py: the DeepSWE v1.1 grader plus the swe-task test-path sanitizer (S/README.md,
"Known grading gap"; hengyuf 2026-10-19; v3 consistency rules 2026-10-21). `grader.py prepare` runs against temporary
git repositories (no sandbox, network or LLM calls).
Run from S/synth:  ../atp/runtime/controller/bin/python3 -B -m unittest discover -s tests"""
import sys

sys.dont_write_bytecode = True
from pathlib import Path

SYNTH = Path(__file__).resolve().parents[1]
if str(SYNTH) not in sys.path:
    sys.path.insert(0, str(SYNTH))

import ast  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402

GRADER = SYNTH / "verifier/grader.py"
ORIGINAL = SYNTH / "cache/deep-swe/tasks/abs-stepped-slices/tests/grader.py"  # DeepSWE's, shared by all 113 tasks
GIT = shutil.which("git") or "/usr/bin/git"
PY_GLOBS = ["tests/**", "test/**", "**/test_*.py", "**/*_test.py", "**/conftest.py"]
GO_GLOBS = ["**/*_test.go", "**/testdata/**"]
JS_GLOBS = ["**/*.test.*", "**/*.spec.*", "**/__tests__/**", "test/**", "tests/**"]
SECTION_START = "# >>> swe-task addition"
SECTION_END = "# <<< swe-task addition"

# git < 2.35 has no `git apply --allow-empty` (the grader always passes it for test.patch). Where the local git lacks
# it, the fixtures put this shim first in PATH: it drops the flag and applies an empty patch as a no-op.
GIT_SHIM = """#!/bin/bash
if [ "$1" = apply ]; then
  args=(); empty_ok=0
  for a in "$@"; do if [ "$a" = --allow-empty ]; then empty_ok=1; else args+=("$a"); fi; done
  if [ "$empty_ok" = 1 ] && [ -f "${args[-1]}" ] && ! grep -qE '^(diff --git |--- )' "${args[-1]}"; then exit 0; fi
  exec @GIT@ "${args[@]}"
fi
exec @GIT@ "$@"
"""


def _git_has_allow_empty() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        return subprocess.run([GIT, "apply", "--allow-empty", os.devnull], cwd=tmp, capture_output=True).returncode == 0


NEEDS_SHIM = not _git_has_allow_empty()


def load_grader(path=GRADER, name="sanitizing_grader"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def without_sections(text: str) -> str:
    """The grader with every marked swe-task section cut out: marker lines included, and for a top-level section the
    two blank lines that separate it from the next definition."""
    kept, inside, skip_blank = [], False, 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if skip_blank and not stripped:
            skip_blank -= 1
            continue
        skip_blank = 0
        if stripped.startswith(SECTION_START):
            inside = True
        elif stripped.startswith(SECTION_END):
            inside = False
            skip_blank = 2 if line.startswith(SECTION_END) else 0
        elif not inside:
            kept.append(line)
    return "".join(kept)


class Fixture:
    """A repository at base_commit in <tmp>/app (the verifier sandbox's /app), with /tests, /logs/verifier and
    /logs/artifacts moved into <tmp>. model.patch and test.patch are real `git diff --binary <base> HEAD` outputs of
    throw-away branches, so the tree is back at base_commit (clean) before prepare runs, as in a fresh verifier."""

    def __init__(self, case: unittest.TestCase, files: dict, *, test_paths=PY_GLOBS, hidden=None, mode=None):
        self.tmp = Path(tempfile.mkdtemp(prefix="test-verifier-"))
        case.addCleanup(shutil.rmtree, self.tmp, True)
        self.app, self.tests = self.tmp / "app", self.tmp / "tests"
        self.verifier, self.artifacts = self.tmp / "logs/verifier", self.tmp / "logs/artifacts"
        for directory in (self.app, self.tests, self.verifier, self.artifacts, self.tmp / "home", self.tmp / "bin"):
            directory.mkdir(parents=True)
        if NEEDS_SHIM:
            (self.tmp / "bin/git").write_text(GIT_SHIM.replace("@GIT@", GIT))
            (self.tmp / "bin/git").chmod(0o755)
        self.env = {"PATH": f"{self.tmp / 'bin'}:/usr/bin:/bin", "HOME": str(self.tmp / "home"), "LANG": "C.UTF-8",
                    "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
                    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+00:00", "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+00:00"}
        self.git("init", "-q")
        self.git("checkout", "-q", "-b", "main")
        self.git("config", "core.hooksPath", "/dev/null")
        self.write(files)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "base")
        self.base = self.git("rev-parse", "HEAD").stdout.decode().strip()
        self.files = dict(files)
        test_patch = self.diff(lambda: self.write(hidden or {"tests/test_hidden.py": "def test_hidden():\n    pass\n"}))
        (self.tests / "test.patch").write_bytes(test_patch)
        config = {"base_commit": self.base, "f2p_node_ids": ["tests/test_hidden.py::test_hidden"],
                  "p2p_node_ids": ["tests/test_core.py::test_add"],
                  "grade": {"format": "ctrf", "node_id": "name", "tool_label": "t",
                            "reports": [str(self.verifier / "base-ctrf.json"), str(self.verifier / "new-ctrf.json")]}}
        if test_paths is not None:
            config["test_paths"] = test_paths
        if mode is not None:  # config.json "test_paths_mode" (absent: the grader's default "all")
            config["test_paths_mode"] = mode
        (self.tests / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    def git(self, *args, check=True):
        return subprocess.run([GIT, *args], cwd=self.app, env=self.env, capture_output=True, check=check)

    def write(self, files: dict):
        for name, content in files.items():
            path = self.app / name
            if content is None:
                path.unlink()
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content if isinstance(content, bytes) else content.encode())

    def diff(self, change, *diff_args) -> bytes:
        self.git("checkout", "-q", "-b", "work")
        change()
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", "work")
        patch = self.git("diff", "--binary", *diff_args, self.base, "HEAD").stdout
        self.git("checkout", "-q", "main")
        self.git("branch", "-q", "-D", "work")
        assert not self.git("status", "--porcelain").stdout, "fixture tree not clean"
        return patch

    def model(self, files: dict = None, *, renames=(), diff_args=("-M",)) -> bytes:
        """model.patch for these writes (None deletes) and `git mv` renames."""
        def change():
            for old, new in renames:
                (self.app / new).parent.mkdir(parents=True, exist_ok=True)
                self.git("mv", old, new)
            self.write(files or {})
        patch = self.diff(change, *diff_args)
        (self.artifacts / "model.patch").write_bytes(patch)
        return patch

    def prepare(self, grader=GRADER):
        env = dict(self.env, TESTS_DIR=str(self.tests), VERIFIER_DIR=str(self.verifier), APP_DIR=str(self.app),
                   ARTIFACTS_DIR=str(self.artifacts))
        return subprocess.run([sys.executable, "-B", str(grader), "prepare"], cwd=self.tmp, env=env,
                              capture_output=True, text=True, timeout=120)

    def tree(self) -> dict:
        return {str(p.relative_to(self.app)): p.read_bytes() for p in sorted(self.app.rglob("*"))
                if p.is_file() and ".git" not in p.relative_to(self.app).parts}

    def base_tree_with_hidden(self) -> dict:
        tree = {name: content.encode() if isinstance(content, str) else content for name, content in self.files.items()}
        tree["tests/test_hidden.py"] = b"def test_hidden():\n    pass\n"
        return tree

    def base_tree_with_hidden_js(self) -> dict:
        tree = {name: content.encode() if isinstance(content, str) else content for name, content in self.files.items()}
        tree.update({name: content.encode() for name, content in JS_HIDDEN.items()})
        return tree


BASE_FILES = {
    "src/calc.py": "def add(a, b):\n    return a + b\n",
    "src/old.py": "OLD = 1\n",
    "src/testing_utils.py": "HELPER = 1\n",  # not a test path: no glob matches it
    "tests/test_core.py": "from src.calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
    "tests/test_old.py": "def test_old():\n    assert True\n",
    "tests/conftest.py": "import pytest\n",
    "README.md": "readme\n",
}


class Sanitizer(unittest.TestCase):
    def check_ok(self, proc):
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("model.patch applied", proc.stdout)
        self.assertIn("Applying test.patch", proc.stdout)

    def test_modified_test_file_is_restored(self):
        fx = Fixture(self, BASE_FILES)
        fx.model({"tests/test_core.py": BASE_FILES["tests/test_core.py"].replace("== 3", "== 4"),
                  "tests/conftest.py": "import pytest\n\n\n@pytest.fixture\ndef extra():\n    return 1\n",
                  "src/calc.py": "def add(a, b):\n    return a + b + 1\n"})
        proc = fx.prepare()
        self.check_ok(proc)
        tree = fx.tree()
        self.assertEqual(tree["tests/test_core.py"], BASE_FILES["tests/test_core.py"].encode())
        self.assertEqual(tree["tests/conftest.py"], BASE_FILES["tests/conftest.py"].encode())
        self.assertEqual(tree["src/calc.py"], b"def add(a, b):\n    return a + b + 1\n")  # the model's solution stays
        self.assertIn('[verifier] test-sanitize: restored "tests/test_core.py" from base_commit', proc.stdout)
        self.assertIn('[verifier] test-sanitize: restored "tests/conftest.py" from base_commit', proc.stdout)
        self.assertIn("test-sanitize: 2 path(s) of model.patch match test_paths", proc.stdout)
        # the old grader grades the model's edited assertion (the false PASS of testem c01 / arcane c02-r1b4)
        old = Fixture(self, BASE_FILES)
        old.model({"tests/test_core.py": BASE_FILES["tests/test_core.py"].replace("== 3", "== 4")})
        self.assertEqual(old.prepare(ORIGINAL).returncode, 0)
        self.assertIn(b"== 4", old.tree()["tests/test_core.py"])

    def test_added_test_file_is_deleted(self):
        fx = Fixture(self, BASE_FILES)
        fx.model({"tests/test_extra.py": "def test_extra():\n    assert True\n",
                  "tests/unit/deep/test_more.py": "def test_more():\n    pass\n",
                  "src/feature.py": "FEATURE = 1\n"})
        proc = fx.prepare()
        self.check_ok(proc)
        tree = fx.tree()
        self.assertNotIn("tests/test_extra.py", tree)
        self.assertNotIn("tests/unit/deep/test_more.py", tree)
        self.assertEqual(tree["src/feature.py"], b"FEATURE = 1\n")
        self.assertIn('test-sanitize: deleted "tests/test_extra.py" (not in base_commit)', proc.stdout)
        self.assertIn('test-sanitize: deleted "tests/unit/deep/test_more.py" (not in base_commit)', proc.stdout)

    def test_deleted_test_file_comes_back(self):
        fx = Fixture(self, BASE_FILES)
        fx.model({"tests/test_old.py": None, "src/old.py": None})
        proc = fx.prepare()
        self.check_ok(proc)
        tree = fx.tree()
        self.assertEqual(tree["tests/test_old.py"], BASE_FILES["tests/test_old.py"].encode())
        self.assertNotIn("src/old.py", tree)  # a deleted source file is the model's change
        self.assertIn('test-sanitize: restored "tests/test_old.py" from base_commit', proc.stdout)

    def test_pure_rename_is_handled(self):
        fx = Fixture(self, BASE_FILES)
        patch = fx.model(renames=[("tests/test_old.py", "tests/test_renamed.py")])
        self.assertIn(b"rename from tests/test_old.py", patch)
        self.assertNotIn(b"--- a/", patch)  # a pure rename has no ---/+++ lines ...
        # ... so DeepSWE's patch_paths sees only the new name; the sanitizer reads both
        self.assertEqual(load_grader(ORIGINAL, "deepswe_grader_ref").patch_paths(patch.decode()),
                         ["tests/test_renamed.py"])
        self.assertEqual(load_grader().touched_paths(patch.decode()), ["tests/test_old.py", "tests/test_renamed.py"])
        proc = fx.prepare()
        self.check_ok(proc)
        self.assertEqual(fx.tree(), fx.base_tree_with_hidden())
        self.assertIn('test-sanitize: restored "tests/test_old.py" from base_commit', proc.stdout)
        self.assertIn('test-sanitize: deleted "tests/test_renamed.py" (not in base_commit)', proc.stdout)

    def test_renames_with_edits_and_across_the_test_boundary(self):
        fx = Fixture(self, BASE_FILES)
        edited = "\n".join(f"# line {i}" for i in range(30)) + "\n"
        fx.write({"tests/test_old.py": edited})  # base needs a longer file for a rename-with-edit
        fx.git("commit", "-qam", "longer")
        fx.base = fx.git("rev-parse", "HEAD").stdout.decode().strip()
        config = json.loads((fx.tests / "config.json").read_text())
        (fx.tests / "config.json").write_text(json.dumps(dict(config, base_commit=fx.base)))
        patch = fx.model({"tests/test_moved.py": edited.replace("# line 3\n", "# line three\n")},
                         renames=[("tests/test_old.py", "tests/test_moved.py"),
                                  ("src/old.py", "tests/helpers_moved.py"),      # source -> test path
                                  ("tests/conftest.py", "src/conftest_moved.py")])  # test path -> source
        self.assertIn(b"rename from tests/test_old.py", patch)
        proc = fx.prepare()
        self.check_ok(proc)
        tree = fx.tree()
        self.assertEqual(tree["tests/test_old.py"], edited.encode())
        self.assertNotIn("tests/test_moved.py", tree)
        self.assertNotIn("tests/helpers_moved.py", tree)
        self.assertNotIn("src/old.py", tree)  # the old name is not a test path: the model's deletion stays
        self.assertEqual(tree["tests/conftest.py"], b"import pytest\n")
        self.assertEqual(tree["src/conftest_moved.py"], b"import pytest\n")  # a new non-test path stays

    def test_copy_is_handled(self):
        fx = Fixture(self, BASE_FILES)
        patch = fx.model({"tests/test_copy.py": BASE_FILES["tests/test_old.py"]},
                         diff_args=("-C", "--find-copies-harder"))
        self.assertIn(b"copy from tests/test_old.py", patch)
        proc = fx.prepare()
        self.check_ok(proc)
        self.assertEqual(fx.tree(), fx.base_tree_with_hidden())
        self.assertIn('test-sanitize: deleted "tests/test_copy.py" (not in base_commit)', proc.stdout)

    def test_non_test_files_are_untouched(self):
        fx = Fixture(self, BASE_FILES)
        model = {"src/calc.py": "def add(a, b):\n    return b + a\n", "src/testing_utils.py": "HELPER = 2\n",
                 "src/new_module.py": "NEW = 1\n", "README.md": None, "docs/test_guide.md": "guide\n"}
        fx.model(model)
        proc = fx.prepare()
        self.check_ok(proc)
        tree = fx.tree()
        for name, content in model.items():
            if content is None:
                self.assertNotIn(name, tree)
            else:
                self.assertEqual(tree[name], content.encode(), name)
        self.assertIn("test-sanitize: 0 path(s) of model.patch match test_paths", proc.stdout)
        self.assertNotIn("restored", proc.stdout)
        self.assertNotIn("deleted", proc.stdout)

    def test_missing_key_is_a_no_op(self):
        """Without "test_paths" the new grader leaves exactly the tree DeepSWE's grader leaves (model test edits
        included) and logs nothing of its own."""
        model = {"tests/test_core.py": BASE_FILES["tests/test_core.py"].replace("== 3", "== 4"),
                 "tests/test_extra.py": "def test_extra():\n    pass\n", "tests/test_old.py": None,
                 "src/calc.py": "def add(a, b):\n    return a - b\n"}
        new, old = Fixture(self, BASE_FILES, test_paths=None), Fixture(self, BASE_FILES, test_paths=None)
        new.model(model, renames=[("tests/conftest.py", "tests/conftest_renamed.py")])
        old.model(model, renames=[("tests/conftest.py", "tests/conftest_renamed.py")])
        proc, ref = new.prepare(), old.prepare(ORIGINAL)
        self.check_ok(proc)
        self.check_ok(ref)
        self.assertEqual(new.tree(), old.tree())
        self.assertIn(b"== 4", new.tree()["tests/test_core.py"])
        self.assertNotIn("test-sanitize", proc.stdout)
        self.assertEqual(proc.stdout, ref.stdout)

    def test_files_test_patch_touches_end_as_base_plus_test_patch(self):
        hidden = {"tests/test_core.py": BASE_FILES["tests/test_core.py"] + "\n\ndef test_add_more():\n"
                                                                            "    assert add(2, 2) == 4\n",
                  "tests/test_hidden.py": "def test_hidden():\n    pass\n"}
        fx = Fixture(self, BASE_FILES, hidden=hidden)
        fx.model({"tests/test_core.py": "def test_add():\n    assert True\n",
                  "tests/test_hidden.py": "def test_hidden():\n    assert False\n"})
        proc = fx.prepare()
        self.check_ok(proc)
        tree = fx.tree()
        self.assertEqual(tree["tests/test_core.py"], hidden["tests/test_core.py"].encode())
        self.assertEqual(tree["tests/test_hidden.py"], hidden["tests/test_hidden.py"].encode())

    def test_go_and_js_layouts(self):
        files = {"go.mod": "module example.com/x\n", "pkg/diag.go": "package pkg\n",
                 "pkg/diag_test.go": "package pkg\n\nfunc TestDiag(t *testing.T) {}\n",
                 "pkg/testdata/golden.txt": "golden\n", "web/src/__tests__/a.ts": "a\n", "web/src/b.ts": "b\n"}
        fx = Fixture(self, files, test_paths=GO_GLOBS + JS_GLOBS, hidden={"pkg/new_test.go": "package pkg\n"})
        fx.model({"pkg/diag_test.go": "package pkg\n// edited\n", "pkg/fixture_test.go": "package pkg\n\ntype diagFixture struct{}\n",
                  "pkg/testdata/golden.txt": "changed\n", "pkg/diag.go": "package pkg\n\nfunc Diag() {}\n",
                  "web/src/__tests__/a.ts": "a2\n", "web/src/b.spec.tsx": "spec\n", "web/src/b.ts": "b2\n"})
        proc = fx.prepare()
        self.check_ok(proc)
        tree = fx.tree()
        self.assertEqual(tree["pkg/diag_test.go"], files["pkg/diag_test.go"].encode())
        self.assertNotIn("pkg/fixture_test.go", tree)  # the model's own test file redeclaring a hidden helper
        self.assertEqual(tree["pkg/testdata/golden.txt"], b"golden\n")
        self.assertEqual(tree["web/src/__tests__/a.ts"], b"a\n")
        self.assertNotIn("web/src/b.spec.tsx", tree)
        self.assertEqual(tree["pkg/diag.go"], b"package pkg\n\nfunc Diag() {}\n")
        self.assertEqual(tree["web/src/b.ts"], b"b2\n")
        self.assertEqual(tree["pkg/new_test.go"], b"package pkg\n")  # test.patch still applies

    def test_quoted_spaced_and_binary_paths(self):
        files = dict(BASE_FILES, **{"tests/data dir/test my case.py": "x = 1\n", "tests/test_café.py": "y = 1\n",
                                    "tests/fixtures/blob.bin": bytes(range(256)) * 4})
        fx = Fixture(self, files)
        patch = fx.model({"tests/data dir/test my case.py": "x = 2\n", "tests/test_café.py": "y = 2\n",
                          "tests/fixtures/blob.bin": bytes(reversed(range(256))) * 4})
        self.assertIn(b'"a/tests/test_caf\\303\\251.py"', patch)  # git C-quotes non-ASCII names
        self.assertIn(b"GIT binary patch", patch)
        proc = fx.prepare()
        self.check_ok(proc)
        tree = fx.tree()
        self.assertEqual(tree["tests/data dir/test my case.py"], b"x = 1\n")
        self.assertEqual(tree["tests/test_café.py"], b"y = 1\n")
        self.assertEqual(tree["tests/fixtures/blob.bin"], bytes(range(256)) * 4)
        self.assertIn('restored "tests/data dir/test my case.py"', proc.stdout)

    def test_bad_test_paths_is_an_infra_error(self):
        for bad in ("tests/**", [], [""], [1]):
            fx = Fixture(self, BASE_FILES, test_paths=bad)
            fx.model({"tests/test_core.py": "changed\n"})
            proc = fx.prepare()
            if bad == []:  # an empty list is valid: nothing matches
                self.check_ok(proc)
                continue
            self.assertEqual(proc.returncode, 7, bad)
            self.assertIn('ERROR: test-sanitize: config.json "test_paths"', proc.stdout)
            self.assertFalse((fx.verifier / "reward.json").exists())  # test.sh's trap: reward.txt -1 (infra)

    def test_apply_failure_still_grades_zero_before_sanitizing(self):
        fx = Fixture(self, BASE_FILES)
        (fx.artifacts / "model.patch").write_text("diff --git a/src/calc.py b/src/calc.py\n--- a/src/calc.py\n"
                                                  "+++ b/src/calc.py\n@@ -1,1 +1,1 @@\n-nope\n+nah\n")
        proc = fx.prepare()
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(json.loads((fx.verifier / "reward.json").read_text())["apply_failed"], 1)
        self.assertNotIn("test-sanitize", proc.stdout)


# A jest-style layout (official1-vitest/testem, jest-worker): index.test.ts mocks the internal module ../Farm with a
# factory, so a solution that imports a new helper from Farm.ts must add it to the factory -- a purely additive edit.
JS_FILES = {
    "package.json": '{"name": "fixture"}\n',
    "src/Farm.ts": "export default class Farm {\n  run() {\n    return 1;\n  }\n}\n",
    "src/index.ts": "import Farm from './Farm';\n\nexport function make() {\n  return new Farm();\n}\n",
    "src/__tests__/index.test.ts": "jest.mock('../Farm', () => ({\n  __esModule: true,\n  default: jest.fn(),\n}));\n"
                                   "\ntest('make', () => {\n  expect(true).toBe(true);\n});\n",
    "src/__tests__/Farm.test.ts": "import Farm from '../Farm';\n\ntest('run', () => {\n"
                                  "  expect(new Farm().run()).toBe(1);\n});\n",
    "src/__tests__/__snapshots__/index.test.ts.snap": 'exports[`make 1`] = `\n{\n  "a": 1,\n}\n`;\n',
    "src/__tests__/old.test.ts": "test('old', () => {});\n",
    "src/__tests__/nonl.test.ts": "test('nonl', () => {});",  # no newline at the end of the file
    "src/__tests__/fixture.bin": bytes(range(256)) * 4,
}
JS_HIDDEN = {"src/__tests__/hidden.test.ts": "test('hidden', () => {});\n"}
FARM_HELPER = {  # the solution side: a new helper in Farm.ts, imported by index.ts (never test paths)
    "src/Farm.ts": JS_FILES["src/Farm.ts"] + "\nexport function validate() {\n  return true;\n}\n",
    "src/index.ts": "import Farm, {validate} from './Farm';\n\nexport function make() {\n  validate();\n"
                    "  return new Farm();\n}\n",
}
MOCK_EXTENDED = JS_FILES["src/__tests__/index.test.ts"].replace(
    "  default: jest.fn(),\n", "  default: jest.fn(),\n  validate: jest.fn(() => true),\n") + \
    "\ntest('validate is called', () => {\n  expect(true).toBe(true);\n});\n"
SNAP_EXTENDED = JS_FILES["src/__tests__/__snapshots__/index.test.ts.snap"].replace('  "a": 1,\n', '  "a": 1,\n  "b": 2,\n')


class NonAdditiveMode(unittest.TestCase):
    """config.json "test_paths_mode" (hengyuf 2026-10-20, JS/TS tasks): "non_additive" keeps an existing test file
    whose model.patch change only adds lines; everything else is sanitized as in "all" (the default)."""

    def fixture(self, mode, **kw):
        return Fixture(self, JS_FILES, test_paths=JS_GLOBS, hidden=kw.pop("hidden", JS_HIDDEN), mode=mode, **kw)

    def check_ok(self, proc):
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("model.patch applied", proc.stdout)
        self.assertIn("Applying test.patch", proc.stdout)

    def test_additive_edit_is_kept_in_non_additive_mode_and_restored_in_all_mode(self):
        model = dict(FARM_HELPER, **{"src/__tests__/index.test.ts": MOCK_EXTENDED,
                                     "src/__tests__/__snapshots__/index.test.ts.snap": SNAP_EXTENDED})
        for mode in ("non_additive", "all"):
            with self.subTest(mode=mode):
                fx = self.fixture(mode)
                patch = fx.model(model)
                for name in ("src/__tests__/index.test.ts", "src/__tests__/__snapshots__/index.test.ts.snap"):
                    block = next(b for b in patch.split(b"diff --git ")[1:] if b.startswith(f"a/{name} ".encode()))
                    body = block.split(b"\n@@", 1)[1].split(b"\n")[1:]
                    self.assertTrue(body and not any(line.startswith(b"-") for line in body), name)  # only "+" lines
                proc = fx.prepare()
                self.check_ok(proc)
                tree = fx.tree()
                for name in FARM_HELPER:  # the solution itself always stays
                    self.assertEqual(tree[name], FARM_HELPER[name].encode())
                self.assertEqual(tree["src/__tests__/hidden.test.ts"], JS_HIDDEN["src/__tests__/hidden.test.ts"].encode())
                kept = {"src/__tests__/index.test.ts": MOCK_EXTENDED,
                        "src/__tests__/__snapshots__/index.test.ts.snap": SNAP_EXTENDED}
                for name, content in kept.items():
                    if mode == "non_additive":
                        self.assertEqual(tree[name], content.encode(), name)
                        self.assertIn(f'[verifier] test-sanitize: kept-additive "{name}"\n', proc.stdout)
                    else:
                        self.assertEqual(tree[name], JS_FILES[name].encode(), name)
                        self.assertIn(f'[verifier] test-sanitize: restored "{name}" from base_commit', proc.stdout)
                        self.assertNotIn("kept-additive", proc.stdout)
                self.assertIn("test-sanitize: 2 path(s) of model.patch match test_paths", proc.stdout)
                self.assertEqual("test_paths_mode non_additive" in proc.stdout, mode == "non_additive")

    def test_a_changed_or_deleted_line_is_restored_in_both_modes(self):
        changed = {  # each edit adds lines AND changes or removes at least one existing line
            "src/__tests__/index.test.ts": MOCK_EXTENDED.replace("default: jest.fn(),", "default: jest.fn(() => 1),"),
            "src/__tests__/Farm.test.ts": JS_FILES["src/__tests__/Farm.test.ts"].replace("toBe(1)", "toBe(2)")
                                          + "\ntest('more', () => {});\n",
            "src/__tests__/__snapshots__/index.test.ts.snap": SNAP_EXTENDED.replace('"a": 1', '"a": 3'),
            "src/__tests__/old.test.ts": "test('old', () => {});\n".replace("old'", "old renamed'"),
            "src/__tests__/nonl.test.ts": JS_FILES["src/__tests__/nonl.test.ts"] + "\ntest('more', () => {});\n",
        }
        for mode in ("non_additive", "all"):
            with self.subTest(mode=mode):
                fx = self.fixture(mode)
                patch = fx.model(dict(FARM_HELPER, **changed))
                self.assertIn(b"\\ No newline at end of file", patch)  # appending to nonl.test.ts rewrites its last line
                proc = fx.prepare()
                self.check_ok(proc)
                tree = fx.tree()
                for name in changed:
                    self.assertEqual(tree[name], JS_FILES[name].encode(), name)
                    self.assertIn(f'test-sanitize: restored "{name}" from base_commit', proc.stdout)
                self.assertNotIn("kept-additive", proc.stdout)
                self.assertEqual(tree["src/index.ts"], FARM_HELPER["src/index.ts"].encode())

    def test_added_deleted_renamed_copied_binary_and_mode_changed_test_files_in_both_modes(self):
        for mode in ("non_additive", "all"):
            with self.subTest(mode=mode):
                fx = self.fixture(mode)
                patch = fx.model({"src/__tests__/extra.test.ts": "test('extra', () => {});\n",  # added: deleted
                                  "src/__tests__/deep/more.spec.ts": "it('more', () => {});\n",
                                  "src/__tests__/copy.test.ts": JS_FILES["src/__tests__/Farm.test.ts"],
                                  "src/__tests__/nonl.test.ts": None,  # deleted: comes back
                                  "src/__tests__/fixture.bin": bytes(reversed(range(256))) * 4,  # binary: restored
                                  "src/index.ts": FARM_HELPER["src/index.ts"]},
                                 renames=[("src/__tests__/old.test.ts", "src/__tests__/old2.test.ts")],
                                 diff_args=("-M", "-C", "--find-copies-harder"))
                for header in (b"copy from src/__tests__/Farm.test.ts", b"rename from src/__tests__/old.test.ts",
                               b"deleted file mode", b"GIT binary patch", b"new file mode"):
                    self.assertIn(header, patch)
                proc = fx.prepare()
                self.check_ok(proc)
                self.assertEqual(fx.tree(), dict(fx.base_tree_with_hidden_js(), **{
                    "src/index.ts": FARM_HELPER["src/index.ts"].encode()}))
                for name in ("src/__tests__/extra.test.ts", "src/__tests__/deep/more.spec.ts",
                             "src/__tests__/copy.test.ts", "src/__tests__/old2.test.ts"):
                    self.assertIn(f'test-sanitize: deleted "{name}" (not in base_commit)', proc.stdout)
                self.assertNotIn("kept-additive", proc.stdout)
        # a mode change is not "only adds lines" either, even with additive hunks
        fx = self.fixture("non_additive")
        def chmod_and_append():
            fx.write({"src/__tests__/old.test.ts": JS_FILES["src/__tests__/old.test.ts"] + "test('new', () => {});\n"})
            (fx.app / "src/__tests__/old.test.ts").chmod(0o755)
        patch = fx.diff(chmod_and_append)
        (fx.artifacts / "model.patch").write_bytes(patch)
        self.assertIn(b"new mode 100755", patch)
        proc = fx.prepare()
        self.check_ok(proc)
        self.assertEqual(fx.tree()["src/__tests__/old.test.ts"], JS_FILES["src/__tests__/old.test.ts"].encode())
        self.assertIn('restored "src/__tests__/old.test.ts" from base_commit', proc.stdout)

    def test_a_missing_mode_key_means_all(self):
        model = dict(FARM_HELPER, **{"src/__tests__/index.test.ts": MOCK_EXTENDED,
                                     "src/__tests__/extra.test.ts": "test('extra', () => {});\n"})
        default, explicit = self.fixture(None), self.fixture("all")
        self.assertNotIn("test_paths_mode", json.loads((default.tests / "config.json").read_text()))
        default.model(model)
        explicit.model(model)
        proc, ref = default.prepare(), explicit.prepare()
        self.check_ok(proc)
        self.check_ok(ref)
        self.assertEqual(default.tree(), explicit.tree())
        self.assertEqual(default.tree()["src/__tests__/index.test.ts"], JS_FILES["src/__tests__/index.test.ts"].encode())
        self.assertEqual(proc.stdout.replace(default.base, "BASE"), ref.stdout.replace(explicit.base, "BASE"))
        self.assertNotIn("test_paths_mode", proc.stdout)
        self.assertNotIn("kept-additive", proc.stdout)

    def test_bad_mode_is_an_infra_error_and_ignored_without_test_paths(self):
        for bad in ("additive", "ALL", "", None, ["all"], 1):
            with self.subTest(mode=bad):
                fx = Fixture(self, JS_FILES, test_paths=JS_GLOBS, hidden=JS_HIDDEN)
                config = json.loads((fx.tests / "config.json").read_text())
                (fx.tests / "config.json").write_text(json.dumps(dict(config, test_paths_mode=bad)))
                fx.model({"src/__tests__/index.test.ts": MOCK_EXTENDED})
                proc = fx.prepare()
                self.assertEqual(proc.returncode, 7, proc.stdout)
                self.assertIn('ERROR: test-sanitize: config.json "test_paths_mode" must be one of', proc.stdout)
                self.assertFalse((fx.verifier / "reward.json").exists())  # test.sh's trap: reward.txt -1 (infra)
        fx = Fixture(self, JS_FILES, test_paths=None, hidden=JS_HIDDEN, mode="bogus")  # sanitizer off: no check
        fx.model({"src/__tests__/index.test.ts": MOCK_EXTENDED, "src/__tests__/extra.test.ts": "x\n"})
        proc = fx.prepare()
        self.check_ok(proc)
        self.assertNotIn("test-sanitize", proc.stdout)
        self.assertEqual(fx.tree()["src/__tests__/index.test.ts"], MOCK_EXTENDED.encode())  # DeepSWE's behaviour

    def test_a_kept_file_that_test_patch_touches_ends_as_base_plus_test_patch(self):
        hidden = dict(JS_HIDDEN, **{"src/__tests__/index.test.ts": JS_FILES["src/__tests__/index.test.ts"]
                                    + "\ntest('hidden in index', () => {});\n"})
        fx = self.fixture("non_additive", hidden=hidden)
        fx.model({"src/__tests__/index.test.ts": MOCK_EXTENDED})
        proc = fx.prepare()
        self.check_ok(proc)
        self.assertIn('kept-additive "src/__tests__/index.test.ts"', proc.stdout)  # step 1b keeps it ...
        self.assertEqual(fx.tree()["src/__tests__/index.test.ts"], hidden["src/__tests__/index.test.ts"].encode())


# Grader v3 (hengyuf 2026-10-21): sanitizing must not leave the graded tree referring to what it took away.
# A Cargo workspace laid out like bytecodealliance/wasmtime (official1-wasmi): crate integration tests in
# crates/<crate>/tests/ (test paths there), registered targets next to auto-discovered ones, a banner before [features].
RUST_GLOBS = ["tests/**", "crates/*/tests/**"]
BANNER = "# " + "=" * 77
WASMTIME_TOML = ('[package]\nname = "wasmtime"\nversion = "1.0.0"\ndescription = """\nA runtime.\n[[test]]\n'
                 'name = "not-a-table"\n"""\n\n[dependencies]\nanyhow = "1"\n\n[[test]]\nname = "host_segfault"\n'
                 'path = "tests/host_segfault.rs"\nharness = false\n\n[[test]]\nname = "pooling_alloc_near_oom"\n'
                 f'harness = false\n\n{BANNER}\n#\n# Features for the Wasmtime crate.\n[features]\n'
                 'default = ["std"]\nstd = []\ncoredump = []\n')
RUST_FILES = {"Cargo.toml": '[workspace]\nmembers = ["crates/wasmtime"]\n',
              "crates/wasmtime/Cargo.toml": WASMTIME_TOML,
              "crates/wasmtime/src/lib.rs": "pub fn run() {}\n",
              "crates/wasmtime/tests/host_segfault.rs": "fn main() {}\n",
              "crates/wasmtime/tests/pooling_alloc_near_oom.rs": "#[test]\nfn oom() {}\n",
              "tests/all/main.rs": "mod traps;\n", "tests/all/traps.rs": "#[test]\nfn trap() {}\n"}
RUST_HIDDEN = {"crates/wasmtime/tests/coredump_snapshot_api.rs": "#[test]\nfn hidden() {}\n"}
COREDUMP_TABLE = '[[test]]\nname = "coredump"\nrequired-features = ["coredump"]\n'  # official1-wasmi c01-r1b1 sc14-r4


def registered(manifest, *tables, at_end=False):
    """The manifest with target tables added as sc14 added them: before the [features] banner (each followed by a blank
    line), or appended after a blank line each."""
    if at_end:
        return manifest + "".join("\n" + table for table in tables)
    return manifest.replace(BANNER, "".join(table + "\n" for table in tables) + BANNER, 1)


class ManifestTargets(unittest.TestCase):
    """v3 (1), any mode: a Cargo target table the model added for a test file the sanitizer deletes is cut out of the
    model's manifest (official1-wasmi: the kept `[[test]] name = "coredump"` made every cargo command fail on the
    manifest, F2P 0 / P2P 0); every other edit of that manifest stays."""

    def fixture(self, mode="all", hidden=None):
        return Fixture(self, RUST_FILES, test_paths=RUST_GLOBS, hidden=hidden or RUST_HIDDEN, mode=mode)

    def check_ok(self, proc):
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("model.patch applied", proc.stdout)
        self.assertIn("Applying test.patch", proc.stdout)

    def test_a_registered_model_test_goes_with_its_table_and_other_manifest_edits_stay(self):
        dependency = WASMTIME_TOML.replace('anyhow = "1"\n', 'anyhow = "1"\nlibc = "0.2"\n')
        for mode in ("all", "non_additive"):
            with self.subTest(mode=mode):
                fx = self.fixture(mode)
                patch = fx.model({"crates/wasmtime/tests/coredump.rs": "#[test]\nfn mine() {}\n",
                                  "crates/wasmtime/Cargo.toml": registered(dependency, COREDUMP_TABLE),
                                  "crates/wasmtime/src/lib.rs": "pub fn run() {}\npub fn coredump() {}\n"})
                self.assertIn(b'+[[test]]\n+name = "coredump"\n+required-features = ["coredump"]\n+\n', patch)
                proc = fx.prepare()
                self.check_ok(proc)
                tree = fx.tree()
                self.assertNotIn("crates/wasmtime/tests/coredump.rs", tree)
                self.assertEqual(tree["crates/wasmtime/Cargo.toml"].decode(), dependency)  # base + the new dependency
                self.assertEqual(tree["crates/wasmtime/src/lib.rs"], b"pub fn run() {}\npub fn coredump() {}\n")
                self.assertEqual(tree["crates/wasmtime/tests/coredump_snapshot_api.rs"],
                                 RUST_HIDDEN["crates/wasmtime/tests/coredump_snapshot_api.rs"].encode())
                self.assertIn('[verifier] test-sanitize: deleted "crates/wasmtime/tests/coredump.rs" (not in '
                              'base_commit)\n[verifier] test-sanitize: dropped-manifest-target '
                              '"crates/wasmtime/Cargo.toml" [[test]] "coredump" -> "crates/wasmtime/tests/coredump.rs" '
                              '(deleted above): cut out\n', proc.stdout)
        # DeepSWE's grader keeps both (no sanitizer): the registration is only a problem once the file is gone
        old = self.fixture()
        old.model({"crates/wasmtime/tests/coredump.rs": "x\n",
                   "crates/wasmtime/Cargo.toml": registered(WASMTIME_TOML, COREDUMP_TABLE)})
        self.assertEqual(old.prepare(ORIGINAL).returncode, 0)
        self.assertIn(b'name = "coredump"', old.tree()["crates/wasmtime/Cargo.toml"])
        self.assertIn("crates/wasmtime/tests/coredump.rs", old.tree())

    def test_a_table_for_a_path_test_patch_puts_back_stays(self):
        """test.patch adds a hidden test at the path the model's table names: nothing dangles in the graded tree, and
        the table stays as DeepSWE's grader leaves it (a reference solution may register that target too)."""
        hidden = {"crates/wasmtime/tests/coredump.rs": "#[test]\nfn hidden() {}\n"}
        fx = self.fixture(hidden=hidden)
        manifest = registered(WASMTIME_TOML, COREDUMP_TABLE)
        fx.model({"crates/wasmtime/tests/coredump.rs": "#[test]\nfn mine() {}\n",
                  "crates/wasmtime/Cargo.toml": manifest})
        proc = fx.prepare()
        self.check_ok(proc)
        tree = fx.tree()
        self.assertEqual(tree["crates/wasmtime/tests/coredump.rs"], b"#[test]\nfn hidden() {}\n")
        self.assertEqual(tree["crates/wasmtime/Cargo.toml"].decode(), manifest)
        self.assertIn('deleted "crates/wasmtime/tests/coredump.rs"', proc.stdout)
        self.assertNotIn("manifest-target", proc.stdout)

    def test_explicit_and_name_based_targets_of_every_kind(self):
        cut = ["[[test]]\nname = \"snap\"\npath = 'tests/model/snapshot.rs' # a literal string\n",
               '[[ test ]]\n"name" = "multi"\n',  # tests/multi/main.rs
               '[[example]]\nname = "ex"\npath = "tests/ex.rs"\n']
        stay = ['[[bench]]\nname = "speed"\nharness = false\n',  # benches/speed.rs: not a test path
                '[[bin]]\nname = "tool"\n',  # src/bin/tool/main.rs
                '[[test]]\nname = "ghost"\n',  # no file anywhere and nothing deleted for it: the model's own problem
                '[[test]]\nname = "again"\npath = "tests/pooling_alloc_near_oom.rs"\n']  # a base_commit file
        manifest = registered(registered(WASMTIME_TOML, cut[0], stay[0]), cut[1], stay[1], cut[2], stay[2], stay[3],
                              at_end=True)
        fx = self.fixture()
        fx.model({"crates/wasmtime/Cargo.toml": manifest, "crates/wasmtime/tests/model/snapshot.rs": "x\n",
                  "crates/wasmtime/tests/multi/main.rs": "mod helper;\n", "crates/wasmtime/tests/multi/helper.rs": "\n",
                  "crates/wasmtime/tests/ex.rs": "fn main() {}\n", "crates/wasmtime/benches/speed.rs": "fn main() {}\n",
                  "crates/wasmtime/src/bin/tool/main.rs": "fn main() {}\n"})
        proc = fx.prepare()
        self.check_ok(proc)
        tree = fx.tree()
        self.assertEqual(tree["crates/wasmtime/Cargo.toml"].decode(),
                         registered(registered(WASMTIME_TOML, stay[0]), stay[1], stay[2], stay[3], at_end=True))
        for name, target in (("snap", "tests/model/snapshot.rs"), ("multi", "tests/multi/main.rs"),
                             ("ex", "tests/ex.rs")):
            self.assertNotIn(f"crates/wasmtime/{target}", tree)
            self.assertRegex(proc.stdout, r'dropped-manifest-target "crates/wasmtime/Cargo\.toml" '
                                          rf'\[\[(test|example)\]\] "{name}" -> "crates/wasmtime/{target}" '
                                          r'\(deleted above\): cut out')
        self.assertEqual(proc.stdout.count("dropped-manifest-target"), 3)
        for kept in ("crates/wasmtime/benches/speed.rs", "crates/wasmtime/src/bin/tool/main.rs"):
            self.assertIn(kept, tree)

    def test_base_tables_stay_and_an_edited_base_table_comes_back(self):
        """The model moved a base test (harness = false) and repointed its table: the file comes back under its old
        name, so base_commit's table comes back too (dropping it would lose harness = false)."""
        repointed = WASMTIME_TOML.replace('path = "tests/host_segfault.rs"', 'path = "tests/segfault_host.rs"')
        fx = self.fixture()
        fx.model({"crates/wasmtime/Cargo.toml": repointed.replace('anyhow = "1"\n', 'anyhow = "1.0"\n'),
                  "crates/wasmtime/tests/pooling_alloc_near_oom.rs": None},  # a base test deleted: restored, table kept
                 renames=[("crates/wasmtime/tests/host_segfault.rs", "crates/wasmtime/tests/segfault_host.rs")])
        proc = fx.prepare()
        self.check_ok(proc)
        tree = fx.tree()
        self.assertEqual(tree["crates/wasmtime/tests/host_segfault.rs"], b"fn main() {}\n")
        self.assertEqual(tree["crates/wasmtime/tests/pooling_alloc_near_oom.rs"], b"#[test]\nfn oom() {}\n")
        self.assertNotIn("crates/wasmtime/tests/segfault_host.rs", tree)
        self.assertEqual(tree["crates/wasmtime/Cargo.toml"].decode(),
                         WASMTIME_TOML.replace('anyhow = "1"\n', 'anyhow = "1.0"\n'))
        self.assertIn('restored-manifest-target "crates/wasmtime/Cargo.toml" [[test]] "host_segfault" -> '
                      '"crates/wasmtime/tests/segfault_host.rs" (deleted above): base_commit\'s table is back',
                      proc.stdout)
        self.assertNotIn("dropped-manifest-target", proc.stdout)

    def test_test_path_manifests_and_unreadable_manifests_are_left_alone(self):
        unreadable = registered(WASMTIME_TOML, COREDUMP_TABLE) + 'notes = """\nnever closed\n'
        fx = self.fixture()
        fx.model({"crates/wasmtime/Cargo.toml": unreadable, "crates/wasmtime/tests/coredump.rs": "x\n",
                  "tests/fixture/Cargo.toml": '[package]\nname = "f"\n\n[[test]]\nname = "t"\n',  # a test path: deleted
                  "tests/fixture/tests/t.rs": "\n"})
        proc = fx.prepare()
        self.check_ok(proc)
        tree = fx.tree()
        self.assertEqual(tree["crates/wasmtime/Cargo.toml"].decode(), unreadable)
        self.assertIn('test-sanitize: manifest-unread "crates/wasmtime/Cargo.toml"', proc.stdout)
        self.assertNotIn("tests/fixture/Cargo.toml", tree)
        self.assertNotIn("manifest-target", proc.stdout)


# official1-dynamodb's layout (IdoPesok/zsa): tests/jest is a Next.js app whose suites import its modules by baseUrl
# ('server/actions'); server/router.ts imports './actions'. Everything under tests/ is a test path.
ZSA_FILES = {
    "package.json": '{"name": "zsa-monorepo"}\n',
    "packages/zsa/src/index.ts": "export const createAction = () => 1\n",
    "tests/jest/server/actions.ts": 'import { createAction } from "zsa"\n\nexport const helloAction = createAction()\n'
                                    "export const sumAction = createAction()\n",
    "tests/jest/server/router.ts": 'import {\n  helloAction,\n  sumAction,\n} from "./actions"\n\n'
                                   "export const router = [helloAction, sumAction]\n",
    "tests/jest/lib/utils.ts": "export const sleep = (ms: number) => ms\n",
    "tests/jest/__tests__/index.test.tsx": 'import {\n  helloAction,\n  sumAction,\n} from "server/actions"\n'
                                           'import { z } from "zod"\n\ntest("hello", () => {\n'
                                           "  expect(helloAction).toBe(1)\n})\n",
    "tests/jest/__tests__/openapi.test.tsx": 'import { router } from "../server/router"\n\ntest("router", () => {\n'
                                             "  expect(router.length).toBe(2)\n})\n",
}
ZSA_HIDDEN = {"tests/jest/__tests__/hidden.test.tsx": 'test("hidden", () => {})\n'}
LONG_IMPORT = 'import {\n' + "".join(f"  a{i},\n" for i in range(8)) + '  helloAction,\n} from "./actions"\n'


class DanglingImports(unittest.TestCase):
    """v3 (2), mode "non_additive": a kept-additive file whose added lines import a path the sanitizer deletes or
    restores is restored like a changed file, until no kept file imports one (official1-dynamodb c03-r1b4 kimi-r4:
    the kept index.test.tsx imported the deleted server/tracking and new symbols of the restored server/actions; P2P
    151 -> 72)."""

    def check_ok(self, proc):
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("model.patch applied", proc.stdout)
        self.assertIn("Applying test.patch", proc.stdout)

    def model_files(self):
        actions = ZSA_FILES["tests/jest/server/actions.ts"]
        return {
            # a changed line (restored) plus a new export
            "tests/jest/server/actions.ts": actions.replace("helloAction = createAction()",
                                                            "helloAction = createAction(1)")
                                            + "export const slowAction = createAction()\n",
            "tests/jest/server/tracking.ts": "export const tracking = []\n",  # model-added: deleted
            # additive: a symbol inside the multi-line import of the restored ./actions
            "tests/jest/server/router.ts": ZSA_FILES["tests/jest/server/router.ts"].replace(
                "  sumAction,\n", "  sumAction,\n  slowAction,\n") + "export const slow = slowAction\n",
            # additive: imports the deleted tracking and the restored actions
            "tests/jest/__tests__/index.test.tsx": ZSA_FILES["tests/jest/__tests__/index.test.tsx"].replace(
                "  sumAction,\n", "  sumAction,\n  slowAction,\n").replace(
                'import { z } from "zod"\n', 'import { z } from "zod"\nimport { tracking } from "server/tracking"\n')
                + 'test("slow", () => {\n  expect(slowAction).toBe(1)\n})\n',
            # additive: imports router.ts, which is restored in the first round -> restored in the second
            "tests/jest/__tests__/openapi.test.tsx": 'import { slow } from "../server/router"\n'
                                                     + ZSA_FILES["tests/jest/__tests__/openapi.test.tsx"],
            # additive without imports: kept
            "tests/jest/lib/utils.ts": ZSA_FILES["tests/jest/lib/utils.ts"] + "export const later = 2\n",
            "packages/zsa/src/index.ts": "export const createAction = (n = 1) => n\n",  # the solution
        }

    def test_kept_files_importing_deleted_or_restored_paths_are_restored_to_a_fixpoint(self):
        model = self.model_files()
        for mode in ("non_additive", "all"):
            with self.subTest(mode=mode):
                fx = Fixture(self, ZSA_FILES, test_paths=JS_GLOBS, hidden=ZSA_HIDDEN, mode=mode)
                fx.model(model)
                proc = fx.prepare()
                self.check_ok(proc)
                tree = fx.tree()
                self.assertEqual(tree["packages/zsa/src/index.ts"], model["packages/zsa/src/index.ts"].encode())
                self.assertNotIn("tests/jest/server/tracking.ts", tree)
                for name in ("tests/jest/server/actions.ts", "tests/jest/server/router.ts",
                             "tests/jest/__tests__/index.test.tsx", "tests/jest/__tests__/openapi.test.tsx"):
                    self.assertEqual(tree[name], ZSA_FILES[name].encode(), name)
                    self.assertIn(f'test-sanitize: restored "{name}" from base_commit', proc.stdout)
                if mode == "all":
                    self.assertEqual(tree["tests/jest/lib/utils.ts"], ZSA_FILES["tests/jest/lib/utils.ts"].encode())
                    self.assertNotIn("dangling-import", proc.stdout)
                    continue
                self.assertEqual(tree["tests/jest/lib/utils.ts"], model["tests/jest/lib/utils.ts"].encode())
                self.assertIn('kept-additive "tests/jest/lib/utils.ts"', proc.stdout)
                for name, spec, target, how in (
                        ("tests/jest/__tests__/index.test.tsx", "server/actions", "tests/jest/server/actions.ts",
                         "restored"),
                        ("tests/jest/server/router.ts", "./actions", "tests/jest/server/actions.ts", "restored"),
                        ("tests/jest/__tests__/openapi.test.tsx", "../server/router", "tests/jest/server/router.ts",
                         "restored")):
                    self.assertIn(f'[verifier] test-sanitize: dangling-import "{name}": its added lines import '
                                  f'"{spec}" -> "{target}" ({how} here): restored like a changed file\n', proc.stdout)
                self.assertEqual(proc.stdout.count("kept-additive"), 1)

    def test_an_added_import_of_a_deleted_file_alone(self):
        fx = Fixture(self, ZSA_FILES, test_paths=JS_GLOBS, hidden=ZSA_HIDDEN, mode="non_additive")
        fx.model({"tests/jest/server/tracking.ts": "export const tracking = []\n",
                  "tests/jest/__tests__/index.test.tsx": 'import { tracking } from "server/tracking"\n'
                                                         + ZSA_FILES["tests/jest/__tests__/index.test.tsx"]})
        proc = fx.prepare()
        self.check_ok(proc)
        self.assertEqual(fx.tree()["tests/jest/__tests__/index.test.tsx"],
                         ZSA_FILES["tests/jest/__tests__/index.test.tsx"].encode())
        self.assertIn('dangling-import "tests/jest/__tests__/index.test.tsx": its added lines import "server/tracking" '
                      '-> "tests/jest/server/tracking.ts" (deleted here)', proc.stdout)

    def test_paths_test_patch_puts_back_do_not_count(self):
        """A kept file importing a model-added helper at the path where test.patch adds the hidden one (and a restored
        file that test.patch resets) is left as DeepSWE's grader would leave it: kept."""
        hidden = dict(ZSA_HIDDEN, **{"tests/jest/server/tracking.ts": "export const tracking = [1]\n",
                                     "tests/jest/server/actions.ts": ZSA_FILES["tests/jest/server/actions.ts"]
                                     + "export const hiddenAction = 1\n"})
        fx = Fixture(self, ZSA_FILES, test_paths=JS_GLOBS, hidden=hidden, mode="non_additive")
        model = self.model_files()
        fx.model({name: model[name] for name in ("tests/jest/server/tracking.ts", "tests/jest/server/actions.ts",
                                                 "tests/jest/__tests__/index.test.tsx")})
        proc = fx.prepare()
        self.check_ok(proc)
        tree = fx.tree()
        self.assertEqual(tree["tests/jest/__tests__/index.test.tsx"],
                         model["tests/jest/__tests__/index.test.tsx"].encode())
        self.assertIn('kept-additive "tests/jest/__tests__/index.test.tsx"', proc.stdout)
        self.assertNotIn("dangling-import", proc.stdout)
        for name in ("tests/jest/server/tracking.ts", "tests/jest/server/actions.ts"):
            self.assertEqual(tree[name], hidden[name].encode())

    def test_only_import_statements_with_an_added_line_count(self):
        """A symbol added far above the `from` line counts (the whole file is read, not the hunk: official1-dynamodb
        c01-r1b0 sc14-r0); a new test that uses an unchanged import of a restored file does not (the import is
        base_commit's)."""
        files = dict(ZSA_FILES, **{"tests/jest/server/router.ts": LONG_IMPORT
                                   + "export const router = [helloAction]\n"})
        fx = Fixture(self, files, test_paths=JS_GLOBS, hidden=ZSA_HIDDEN, mode="non_additive")
        model = {"tests/jest/server/actions.ts": ZSA_FILES["tests/jest/server/actions.ts"].replace(
                     "sumAction = createAction()", "sumAction = createAction(2)") + "export const catAction = 3\n",
                 "tests/jest/server/router.ts": files["tests/jest/server/router.ts"].replace(
                     "import {\n", "import {\n  catAction,\n", 1),
                 "tests/jest/__tests__/index.test.tsx": ZSA_FILES["tests/jest/__tests__/index.test.tsx"]
                                                        + 'test("sum", () => {\n  expect(sumAction).toBe(1)\n})\n'}
        patch = fx.model(model).decode()
        router_hunk = patch.split("b/tests/jest/server/router.ts\n", 2)[2]
        self.assertNotIn('from "./actions"', router_hunk.split("\ndiff --git", 1)[0])  # not in the hunk
        proc = fx.prepare()
        self.check_ok(proc)
        tree = fx.tree()
        self.assertEqual(tree["tests/jest/server/router.ts"], files["tests/jest/server/router.ts"].encode())
        self.assertIn('dangling-import "tests/jest/server/router.ts": its added lines import "./actions"', proc.stdout)
        self.assertEqual(tree["tests/jest/__tests__/index.test.tsx"],
                         model["tests/jest/__tests__/index.test.tsx"].encode())
        self.assertIn('kept-additive "tests/jest/__tests__/index.test.tsx"', proc.stdout)

    def test_the_mock_factory_edit_stays_kept_beside_unrelated_sanitized_files(self):
        """The reason for non_additive (official1-vitest/testem): index.test.ts extends its jest.mock('../Farm')
        factory; a deleted model test file and a restored one that it does not import change nothing."""
        fx = Fixture(self, JS_FILES, test_paths=JS_GLOBS, hidden=JS_HIDDEN, mode="non_additive")
        fx.model(dict(FARM_HELPER, **{
            "src/__tests__/index.test.ts": MOCK_EXTENDED,
            "src/__tests__/extra.test.ts": "import Farm from '../Farm';\n\ntest('extra', () => {});\n",
            "src/__tests__/Farm.test.ts": JS_FILES["src/__tests__/Farm.test.ts"].replace("toBe(1)", "toBe(2)")}))
        proc = fx.prepare()
        self.check_ok(proc)
        tree = fx.tree()
        self.assertEqual(tree["src/__tests__/index.test.ts"], MOCK_EXTENDED.encode())
        self.assertIn('kept-additive "src/__tests__/index.test.ts"', proc.stdout)
        self.assertNotIn("dangling-import", proc.stdout)
        self.assertNotIn("src/__tests__/extra.test.ts", tree)
        self.assertEqual(tree["src/__tests__/Farm.test.ts"], JS_FILES["src/__tests__/Farm.test.ts"].encode())


class Parsing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grader = load_grader()

    def test_glob_semantics(self):
        m = self.grader.glob_match
        self.assertTrue(m("tests/**", "tests/a.py"))
        self.assertTrue(m("tests/**", "tests/x/y/z.json"))
        self.assertFalse(m("tests/**", "tests"))  # a final ** needs at least one segment
        self.assertFalse(m("tests/**", "pkg/tests/a.py"))  # anchored at the repository root
        self.assertTrue(m("**/test_*.py", "test_root.py"))
        self.assertTrue(m("**/test_*.py", "a/b/test_x.py"))
        self.assertFalse(m("**/test_*.py", "a/b/test_x.pyc"))
        self.assertFalse(m("**/test_*.py", "test_dir/x.py"))  # * never crosses /
        self.assertTrue(m("**/*_test.go", "pkg/a/b_test.go"))
        self.assertFalse(m("**/*_test.go", "pkg/a/b_test.go.orig"))
        self.assertTrue(m("**/testdata/**", "testdata/x"))
        self.assertTrue(m("**/testdata/**", "a/testdata/b/c"))
        self.assertFalse(m("**/testdata/**", "a/testdata"))
        self.assertTrue(m("**/__tests__/**", "packages/jest-circus/src/__tests__/run.test.ts"))
        self.assertTrue(m("**/*.test.*", "e2e/__tests__/bail.test.ts"))
        self.assertFalse(m("**/*.test.*", "packages/jest-test-result/src/index.ts"))
        self.assertTrue(m("**/*.spec.*", "src/a.spec.tsx"))
        self.assertTrue(m("**/conftest.py", "conftest.py"))
        self.assertTrue(m("a/**/b", "a/b"))
        self.assertTrue(m("a/**/b", "a/x/y/b"))
        self.assertTrue(m("src/[ab]?.py", "src/a1.py"))
        self.assertFalse(m("", "tests/a.py"))
        self.assertFalse(m("Tests/**", "tests/a.py"))  # case-sensitive

    def test_hunk_lines_that_look_like_headers_are_not_paths(self):
        patch = ("diff --git a/notes.sql b/notes.sql\nindex 1..2 100644\n--- a/notes.sql\n+++ b/notes.sql\n"
                 "@@ -1,3 +1,3 @@\n context\n--- a/tests/test_core.py\n+++ b/tests/test_fake.py\n"
                 "-rename from tests/x.py\n+copy to tests/y.py\n"
                 "diff --git a/tests/test_real.py b/tests/test_real.py\ndeleted file mode 100644\n"
                 "--- a/tests/test_real.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n")
        # one context line, two removed, two added: the hunk counts (3/3) cover exactly the five body lines
        self.assertEqual(self.grader.touched_paths(patch), ["notes.sql", "tests/test_real.py"])

    def test_header_only_files_and_quoting(self):
        patch = ('diff --git a/tests/run.sh b/tests/run.sh\nold mode 100644\nnew mode 100755\n'
                 'diff --git a/tests/empty b/tests/empty\nnew file mode 100644\nindex 0000000..e69de29\n'
                 'diff --git "a/tests/t\\303\\251\\"q\\".py" "b/tests/t\\303\\251\\"q\\".py"\nindex 1..2 100644\n'
                 '--- "a/tests/t\\303\\251\\"q\\".py"\n+++ "b/tests/t\\303\\251\\"q\\".py"\n@@ -1 +1 @@\n-a\n+b\n'
                 'diff --git a/tests/a b.py b/tests/a b.py\nindex 1..2 100644\n--- a/tests/a b.py\t\n'
                 '+++ b/tests/a b.py\t\n@@ -1 +1 @@\n-a\n+b\n')
        self.assertEqual(self.grader.touched_paths(patch),
                         ["tests/run.sh", "tests/empty", 'tests/té"q".py', "tests/a b.py"])

    def test_paths_outside_the_tree_are_ignored(self):
        patch = "diff --git a/../x_test.go b/../x_test.go\n--- a/../x_test.go\n+++ b//etc/y_test.go\n"
        self.assertEqual(self.grader.touched_paths(patch), [])

    def test_additive_paths(self):
        additive = self.grader.additive_paths

        def mod(path, body, header="@@ -1,2 +1,3 @@", extra=""):
            return (f"diff --git a/{path} b/{path}\n{extra}index 1111111..2222222 100644\n--- a/{path}\n+++ b/{path}\n"
                    f"{header}\n{body}")

        plain = mod("t/a.test.ts", " x\n+y\n z\n")
        self.assertEqual(additive(plain), {"t/a.test.ts"})
        self.assertEqual(additive(mod("t/a.test.ts", " x\n-y\n+z\n", "@@ -1,2 +1,2 @@")), set())  # a changed line
        self.assertEqual(additive(mod("t/a.test.ts", " x\n-y\n", "@@ -1,2 +1,1 @@")), set())  # a deleted line
        self.assertEqual(additive(mod("t/a.test.ts", "+-y\n+--- a/z\n+diff --git a/q b/q\n", "@@ -0,0 +1,3 @@")),
                         {"t/a.test.ts"})  # added lines that merely look like removals or headers
        self.assertEqual(additive(mod("t/a.test.ts", " x\n+y\n", "@@ -1 +1,2 @@")), {"t/a.test.ts"})  # count 1
        self.assertEqual(additive(mod("t/a.test.ts", " x\n+y\n z\n\\ No newline at end of file\n")), {"t/a.test.ts"})
        self.assertEqual(additive(mod("t/a.test.ts", " x\n-z\n\\ No newline at end of file\n+z\n+y\n",
                                      "@@ -1,2 +1,3 @@")), set())  # appending to a file without a final newline
        self.assertEqual(additive(plain.replace("\n", "\r\n")), {"t/a.test.ts"})  # CRLF patch
        self.assertEqual(additive(mod("t/a.test.ts", " x\n+y\n")), set())  # truncated hunk (z missing)
        self.assertEqual(additive(plain + mod("t/a.test.ts", " y\n-z\n+w\n", "@@ -2,2 +2,2 @@")), set())  # 2 blocks
        self.assertEqual(additive(plain + mod("t/b.test.ts", " x\n+y\n z\n")), {"t/a.test.ts", "t/b.test.ts"})
        self.assertEqual(additive(plain[plain.index("--- a/"):]), set())  # no "diff --git" header (preamble)
        self.assertEqual(additive(mod("t/a.test.ts", " x\n+y\n z\n", extra="old mode 100644\nnew mode 100755\n")),
                         set())  # a mode change
        self.assertEqual(additive("diff --git a/t/a.test.ts b/t/a.test.ts\nold mode 100644\nnew mode 100755\n"), set())
        new_file = ("diff --git a/t/n.test.ts b/t/n.test.ts\nnew file mode 100644\nindex 0000000..1111111\n"
                    "--- /dev/null\n+++ b/t/n.test.ts\n@@ -0,0 +1 @@\n+x\n")
        deleted = ("diff --git a/t/d.test.ts b/t/d.test.ts\ndeleted file mode 100644\nindex 1111111..0000000\n"
                   "--- a/t/d.test.ts\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n")
        renamed = ("diff --git a/t/r.test.ts b/t/s.test.ts\nsimilarity index 90%\nrename from t/r.test.ts\n"
                   "rename to t/s.test.ts\nindex 1111111..2222222 100644\n--- a/t/r.test.ts\n+++ b/t/s.test.ts\n"
                   "@@ -1,2 +1,3 @@\n x\n+y\n z\n")
        copied = ("diff --git a/t/c.test.ts b/t/k.test.ts\nsimilarity index 100%\ncopy from t/c.test.ts\n"
                  "copy to t/k.test.ts\n")
        binary = ("diff --git a/t/b.bin b/t/b.bin\nindex 1111111..2222222 100644\nGIT binary patch\nliteral 3\n"
                  "KcmZQzWB>pF00961\n\nliteral 3\nKcmZQzWB>pF00961\n\n")
        binary_short = ("diff --git a/t/b.bin b/t/b.bin\nindex 1111111..2222222 100644\n"
                        "Binary files a/t/b.bin and b/t/b.bin differ\n")
        for text in (new_file, deleted, renamed, copied, binary, binary_short):
            self.assertEqual(additive(text), set(), text.split("\n", 1)[0])
            self.assertEqual(additive(plain + text), {"t/a.test.ts"})  # the other blocks are unaffected
        self.assertEqual(additive(renamed + mod("t/r.test.ts", " x\n+y\n z\n")), set())  # also named by a rename
        quoted = ('diff --git "a/t/caf\\303\\251.test.ts" "b/t/caf\\303\\251.test.ts"\nindex 1..2 100644\n'
                  '--- "a/t/caf\\303\\251.test.ts"\n+++ "b/t/caf\\303\\251.test.ts"\n@@ -1 +1,2 @@\n x\n+y\n')
        self.assertEqual(additive(quoted), {"t/café.test.ts"})
        self.assertEqual(additive("diff --git a/t/a.test.ts b/t/a.test.ts\nindex 1..2 100644\n"), set())  # no hunk

    def test_added_lines(self):
        patch = ("diff --git a/a.ts b/a.ts\nindex 1..2 100644\n--- a/a.ts\n+++ b/a.ts\n"
                 "@@ -1,3 +1,4 @@\n x\n+y\n z\n w\n@@ -10,2 +11,3 @@\n p\n+q\n r\n"
                 "diff --git a/n.ts b/n.ts\nnew file mode 100644\n--- /dev/null\n+++ b/n.ts\n@@ -0,0 +1,2 @@\n+1\n+2\n"
                 "diff --git a/d.ts b/d.ts\ndeleted file mode 100644\n--- a/d.ts\n+++ /dev/null\n@@ -1 +0,0 @@\n-gone\n"
                 "diff --git a/m.ts b/m.ts\n--- a/m.ts\n+++ b/m.ts\n@@ -1,2 +1,2 @@\n a\n-b\n+c\n"
                 "\\ No newline at end of file\n")
        self.assertEqual(self.grader.added_lines(patch), {"a.ts": {2, 12}, "n.ts": {1, 2}, "d.ts": set(), "m.ts": {2}})
        self.assertEqual(self.grader.added_lines(patch.replace("\n", "\r\n"))["a.ts"], {2, 12})

    def test_import_targets(self):
        gone = ["tests/jest/server/actions.ts", "tests/jest/server/tracking.ts", "src/__tests__/helpers/index.ts",
                "src/__tests__/fixtures/data.json", "tests/jest/test-shims/zsa/index.d.ts",
                "src/__tests__/util.test.ts"]
        cases = {("server/actions", "tests/jest/__tests__/a.tsx"): ["tests/jest/server/actions.ts"],  # baseUrl
                 ("./actions", "tests/jest/server/router.ts"): ["tests/jest/server/actions.ts"],
                 ("../server/tracking.js", "tests/jest/__tests__/a.ts"): ["tests/jest/server/tracking.ts"],  # ESM .js
                 ("./helpers", "src/__tests__/a.test.ts"): ["src/__tests__/helpers/index.ts"],  # a directory's index
                 ("./helpers/index", "src/__tests__/a.test.ts"): ["src/__tests__/helpers/index.ts"],
                 ("./fixtures/data.json", "src/__tests__/a.test.ts"): ["src/__tests__/fixtures/data.json"],
                 ("./util.test?raw", "src/__tests__/b.test.ts"): ["src/__tests__/util.test.ts"],
                 ("@/server/actions", "x.ts"): ["tests/jest/server/actions.ts"],  # path aliases
                 ("~/jest/server/tracking", "x.ts"): ["tests/jest/server/tracking.ts"],
                 ("#server/actions", "x.ts"): ["tests/jest/server/actions.ts"],
                 ("zsa", "tests/jest/__tests__/a.tsx"): [],  # one segment: a package (dynamodb's test-shims/zsa/)
                 ("@scope/zsa", "x.ts"): [], ("lib/utils", "x.ts"): [], ("node:fs", "x.ts"): [],
                 ("https://example.com/server/actions.js", "x.ts"): [], ("/server/actions", "x.ts"): [],
                 ("../../../server/actions", "tests/jest/a.ts"): [],  # outside the repository
                 ("../Farm", "src/__tests__/index.test.ts"): [], ("", "x.ts"): []}
        for (spec, importer), expected in cases.items():
            self.assertEqual(self.grader.import_targets(spec, importer, gone), expected, spec)

    def test_module_references_by_statement(self):
        grader = self.grader
        content = ("import a, {\n  b,\n} from './m1'\n"  # 1-3
                   "export { c } from './m2'\n"  # 4
                   "export * as ns from './m3'\n"  # 5
                   "const {\n  d,\n} = require('./m4')\n"  # 6-8
                   "const e = require(`./m5`)\n"  # 9
                   "const f = await import('./m6')\n"  # 10
                   "jest.mock('./m7', () => ({}))\n"  # 11
                   "const g = jest.requireActual('./m8')\n"  # 12
                   "import './m9'\n"  # 13
                   '/// <reference path="./m10.d.ts" />\n'  # 14
                   "import type { H } from './m11'\n"  # 15
                   "vi.mock('./m12')\n"  # 16
                   "const r = require.resolve('./m13')\n"  # 17
                   "const plain = 'just a string ./m1'\n")  # 18
        expected = {1: "./m1", 2: "./m1", 3: "./m1", 4: "./m2", 5: "./m3", 6: "./m4", 7: "./m4", 8: "./m4", 9: "./m5",
                    10: "./m6", 11: "./m7", 12: "./m8", 13: "./m9", 14: "./m10.d.ts", 15: "./m11", 16: "./m12",
                    17: "./m13", 18: None}
        gone = [f"src/m{i}.ts" for i in range(1, 14) if i != 10] + ["src/m10.d.ts"]
        with tempfile.TemporaryDirectory() as tmp:
            saved = grader.APP_DIR
            self.addCleanup(setattr, grader, "APP_DIR", saved)
            grader.APP_DIR = Path(tmp)
            (Path(tmp) / "src").mkdir()
            (Path(tmp) / "src/x.test.ts").write_text(content)
            (Path(tmp) / "src/x.snap").write_text(content)
            for line, spec in expected.items():
                ref = grader.dangling_ref("src/x.test.ts", {line}, gone)
                self.assertEqual(ref[0] if ref else None, spec, line)
            self.assertIsNone(grader.dangling_ref("src/x.test.ts", {2}, ["src/other.ts"]))
            self.assertIsNone(grader.dangling_ref("src/x.test.ts", set(), gone))
            self.assertIsNone(grader.dangling_ref("src/x.snap", {2}, gone))  # JS/TS sources only
            self.assertIsNone(grader.dangling_ref("src/missing.ts", {2}, gone))

    def test_toml_target_tables(self):
        grader = self.grader
        tables = grader.toml_target_tables(WASMTIME_TOML)  # the [[test]] inside the description string is no table
        self.assertEqual([(t["kind"], t["name"], t["path"], t["first"], t["last"]) for t in tables],
                         [("test", "host_segfault", "tests/host_segfault.rs", 12, 15),
                          ("test", "pooling_alloc_near_oom", None, 17, 19)])
        self.assertEqual(tables[1]["body"], ("[[test]]", 'name = "pooling_alloc_near_oom"', "harness = false"))
        text = ('[[bench]] # a comment\nname = "b\\u0031"\nrequired-features = [\n  "x", # [[test]]\n  \'[y]\',\n]\n'
                "\n# [[test]]\n[[ bin ]]\n'name' = 'lit\\\\x'\npath = \"\"\"\n[[test]]\nnot read\"\"\"\"\n"
                '[lib]\nname = "nottarget"\npath = "src/lib.rs"\n[[test]]\n"path" = "t.rs" # c\nname = "t"\n'
                '[[example]]\npath = "e.rs"\nname = { x = "y" }\n')
        tables = grader.toml_target_tables(text)
        self.assertEqual([(t["kind"], t["name"], t["path"], t["first"], t["last"]) for t in tables],
                         [("bench", "b1", None, 0, 5), ("bin", "lit\\\\x", None, 8, 12), ("test", "t", "t.rs", 16, 18),
                          ("example", None, "e.rs", 19, 21)])
        self.assertEqual(tables[0]["body"], ("[[bench]] # a comment", 'name = "b\\u0031"', "required-features = [",
                                             '"x", # [[test]]', "'[y]',", "]"))
        for bad in ('[[test]]\nname = "not closed\n', 'a = """\nopen\n', "a = [\n1,\n", "a = '''\n"):
            self.assertIsNone(grader.toml_target_tables(bad), bad)
        self.assertEqual(grader.toml_target_tables(""), [])
        s = grader.toml_string
        self.assertEqual([s('"a\\"b" # x'), s("'c:\\\\d'"), s('"\\U0001F600"'), s('"x\\tq"')],
                         ['a"b', "c:\\\\d", "\U0001F600", "x\tq"])
        for bad in ('"""a"""', "'''a'''", '"open', '"\\q"', '"\\u12"', '"\\UFFFFFFFF"', "x", ""):
            self.assertIsNone(s(bad), bad)
        paths = grader.target_paths
        self.assertEqual(paths("crates/w/Cargo.toml", {"kind": "example", "name": "e", "path": None}),
                         ["crates/w/examples/e.rs", "crates/w/examples/e/main.rs"])
        self.assertEqual(paths("Cargo.toml", {"kind": "bin", "name": "b", "path": None}),
                         ["src/bin/b.rs", "src/bin/b/main.rs"])
        self.assertEqual(paths("crates/w/Cargo.toml", {"kind": "test", "name": "t", "path": "../x/./t.rs"}),
                         ["crates/x/t.rs"])
        for bad in ("/abs/t.rs", "../../../t.rs", ""):
            self.assertEqual(paths("crates/w/Cargo.toml", {"kind": "test", "name": "t", "path": bad}), [], bad)
        self.assertEqual(paths("Cargo.toml", {"kind": "test", "name": None, "path": None}), [])


def repair_tool():
    if str(SYNTH / "tools") not in sys.path:
        sys.path.insert(0, str(SYNTH / "tools"))
    import grading_repair
    return grading_repair


class RepairToolLogic(unittest.TestCase):
    """tools/grading_repair.py: the pure parts (no sandbox, no build tree writes)."""

    @classmethod
    def setUpClass(cls):
        cls.gr = repair_tool()

    def test_tamper_patches_apply_and_hit_their_globs(self):
        for language in ("Python", "Go", "TypeScript", "Rust"):
            path, patch = self.gr.tamper_patch(language)
            self.assertTrue(self.gr.matching([path], self.gr.GLOBS[language]), language)
            with tempfile.TemporaryDirectory() as tmp:
                subprocess.run([GIT, "init", "-q"], cwd=tmp, check=True)
                (Path(tmp) / "p.patch").write_text(patch)
                proc = subprocess.run([GIT, "apply", "p.patch"], cwd=tmp, capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertTrue((Path(tmp) / path).exists())

    def test_sanitized_config_is_write_json_format_plus_test_paths(self):
        original = json.dumps({"base_commit": "abc", "p2p_node_ids": ["b", "a"], "grade": {"format": "ctrf"}},
                              indent=2, sort_keys=True) + "\n"
        text = self.gr.sanitized_config(original, PY_GLOBS, "all")
        self.assertEqual(json.loads(text), dict(json.loads(original), test_paths=PY_GLOBS, test_paths_mode="all"))
        self.assertEqual(text, json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n")
        self.assertEqual(self.gr.sanitized_config(text, PY_GLOBS, "all"), text)  # idempotent
        with self.assertRaises(self.gr.RepairError):
            self.gr.sanitized_config(text, GO_GLOBS, "all")
        with self.assertRaises(self.gr.RepairError):  # a config that has the other mode already
            self.gr.sanitized_config(text, PY_GLOBS, "non_additive")
        self.assertEqual(self.gr.without_test_paths(text), json.loads(original))
        first = json.dumps(dict(json.loads(original), test_paths=JS_GLOBS), indent=2, sort_keys=True) + "\n"
        js = self.gr.sanitized_config(first, JS_GLOBS, "non_additive")  # the first sanitizer's config: mode added
        self.assertEqual(json.loads(js), dict(json.loads(original), test_paths=JS_GLOBS, test_paths_mode="non_additive"))
        self.assertEqual(self.gr.without_test_paths(js), json.loads(original))

    def test_mode_follows_the_repository_language(self):
        """non_additive for JavaScript/TypeScript repositories (top100.jsonl, like the globs), all for the rest."""
        self.assertEqual({lang: self.gr.task_mode(lang) for lang in self.gr.GLOBS},
                         {"Python": "all", "Go": "all", "JavaScript": "non_additive", "TypeScript": "non_additive",
                          "Rust": "all"})
        self.assertEqual(self.gr.task_mode("C++"), "all")
        self.assertEqual(set(self.gr.MODES.values()) | {"all"}, set(load_grader().TEST_PATHS_MODES))

    def test_run_globs_add_test_dirs_to_the_language_defaults(self):
        """RUN_GLOBS (batch 3b): a run's override is its language defaults plus test directories, for runs in scope;
        task_globs takes a task's GLOB_OVERRIDES entry first, then the run's RUN_GLOBS, then the defaults."""
        from unittest import mock
        gr = self.gr
        for run, globs in gr.RUN_GLOBS.items():
            self.assertIn(run, gr.RUN_ORDER)
            self.assertTrue(any(globs[:len(d)] == d and len(globs) > len(d) for d in gr.GLOBS.values()), run)
        self.assertEqual(gr.task_globs("t", "TypeScript", "official1-unknown"), gr.GLOBS["TypeScript"])
        sql = gr.task_globs("t", "TypeScript", "official1-sqlformatter")
        self.assertEqual(sql, gr.RUN_GLOBS["official1-sqlformatter"])
        self.assertIsNot(sql, gr.RUN_GLOBS["official1-sqlformatter"])  # a copy: plan entries never alias it
        for path in ("packages/formatter/test/behavesLikeSqlFormatter.ts", "packages/formatter/test/x.hidden.json",
                     "packages/formatter/test/fixtures/a.hidden.ts", "packages/extension/test/__mocks__/vscode.ts",
                     "packages/plugins/formatter/utils.test.ts", "test/config/baseConfig.js"):
            self.assertEqual(gr.matching([path], sql), [path], path)
        for path in ("packages/formatter/src/core/Formatter.ts", "packages/formatter/README.md",
                     "packages/types/index.d.ts", "packages/extension/package.json", "packages/util/query/index.ts",
                     "packages/formatter/tsconfig.check.json", "packages/formatter/testing/x.ts"):
            self.assertEqual(gr.matching([path], sql), [], path)
        with mock.patch.dict(gr.GLOB_OVERRIDES, {"t": ["only/**"]}):
            self.assertEqual(gr.task_globs("t", "TypeScript", "official1-sqlformatter"), ["only/**"])

    def test_run_globs_scriggo_cover_fixtures_not_source(self):
        """batch 3c: official1-scriggo (cosmos72/gomacro, Go) restores ast2/test/ and base/dep's TestSorter fixtures
        besides the Go defaults; base/dep's source files (a reference solution edits base/dep/scope.go), the .gomacro
        code templates and the _example programs stay outside."""
        gr = self.gr
        go = gr.task_globs("t", "Go", "official1-scriggo")
        self.assertIn("official1-scriggo", gr.FINISHED_LATER)
        self.assertEqual(go[:len(gr.GLOBS["Go"])], gr.GLOBS["Go"])
        for path in ("ast2/test/z_test_data_2.txt", "ast2/test/empty.go", "ast2/test/z_test.go",
                     "base/dep/z_test_data_1.txt", "base/dep/z_test_data_3.txt", "base/dep/z_test.go", "all_test.go",
                     "xreflect/z_test.go", "fast/method_api_hidden_test.go", "go/printer/testdata/slow.golden"):
            self.assertEqual(gr.matching([path], go), [path], path)
        for path in ("base/dep/scope.go", "base/dep/api.go", "base/dep/sorter.go", "fast/global.go", "ast2/ast.go",
                     "ast2/test.go", "fast/var_ops.gomacro", "_example/main.go", "README.md",
                     "base/dep/z_test_data_1.txt.go", "base/z_test_data_1.txt"):
            self.assertEqual(gr.matching([path], go), [], path)

    def test_run_globs_wasmi_cover_crate_tests_not_source(self):
        """batch 3d: official1-wasmi (bytecodealliance/wasmtime, Rust) restores every crate's integration tests
        (crates/<crate>/tests/: the hidden crates/wasmtime/tests/*.rs, crates/wizer/tests/all/) besides the root tests/;
        crate sources stay outside, among them crates/debugger/src/lib.rs (c03's test.patch edits its inline `mod test`,
        c03-r1b4's reference solution edits the file) and the guest programs of the test-programs crate."""
        gr = self.gr
        rust = gr.task_globs("t", "Rust", "official1-wasmi")
        self.assertIn("official1-wasmi", gr.FINISHED_LATER)
        self.assertEqual(rust, gr.GLOBS["Rust"] + ["crates/*/tests/**"])
        for path in ("crates/wasmtime/tests/coredump_snapshot_api.rs",
                     "crates/wasmtime/tests/evolve_coredump_options_support/mod.rs",
                     "crates/wizer/tests/all/component.rs", "crates/wizer/tests/all/main.rs",
                     "crates/c-api/tests/CMakeLists.txt", "tests/all/main.rs", "tests/all/coredump.rs",
                     "tests/all/component_model/strings.rs", "tests/frame_watch_hidden/Cargo.lock",
                     "tests/misc_testsuite/component-model/strings.wast"):
            self.assertEqual(gr.matching([path], rust), [path], path)
        for path in ("crates/debugger/src/lib.rs", "crates/wasmtime/src/runtime/coredump.rs",
                     "crates/wasmtime/src/runtime/debug/snapshot.rs", "crates/wizer/src/rewrite.rs",
                     "crates/wasmtime/Cargo.toml", "crates/test-programs/src/bin/wizer_component_interfaces.rs",
                     "examples/coredump.rs", "docs/examples-coredump.md", "Cargo.lock", "RELEASES.md",
                     "crates/wasmtime/build.rs"):
            self.assertEqual(gr.matching([path], rust), [], path)

    def test_run_globs_anko_cover_script_dirs_not_source(self):
        """batch 3e: official1-anko (go-python/gpython, Go) restores the script directories that vm.TestVm, py.TestPy
        and the stdlib packages' TestVm run (vm/tests/, py/tests/, stdlib/<module>/tests/) besides the Go defaults, as
        the test_paths of its round-1 tasks do; Go sources (py/*.go among them), the parser grammar and the
        make_*_test.py generators next to them stay outside."""
        gr = self.gr
        go = gr.task_globs("t", "Go", "official1-anko")
        self.assertIn("official1-anko", gr.FINISHED_LATER)
        self.assertEqual(go, gr.GLOBS["Go"] + ["vm/tests/**", "py/tests/**", "stdlib/*/tests/**"])
        for path in ("vm/tests/call.py", "vm/tests/libtest.py", "vm/tests/hidden_defaults_binding.py",
                     "py/tests/function.py", "py/tests/float.py", "stdlib/math/tests/mathtests.py",
                     "stdlib/builtin/tests/builtin.py", "vm/vm_test.go", "py/with_test.go",
                     "py/testdata/calltimedefaults/main.py", "compile/compile_data_test.go"):
            self.assertEqual(gr.matching([path], go), [path], path)
        for path in ("vm/eval.go", "vm/vm.go", "py/function.go", "py/code.go", "compile/compile.go",
                     "parser/grammar.y", "parser/y.go", "symtable/symtable.go", "stdlib/math/math.go",
                     "compile/make_compile_test.py", "parser/make_grammar_test.py", "symtable/make_symtable_test.py",
                     "pytest/pytest.go", "vm/testsuite.py", "py/tests.go", "stdlib/tests/x.py"):
            self.assertEqual(gr.matching([path], go), [], path)

    def test_tasks_probed_with_a_sanitizer_are_skipped(self):
        """batch 3e: a task whose probe copy (<probe>/tasks/<T>/tests/, what ATP graded) holds a sanitizing grader
        (verifier/grader.py or a PRIOR_GRADERS one; official1-anko round 1, gated after the build's resume) is not
        planned: skip_sanitized, with the status it would have been planned with and whether its test_paths and mode
        equal the run's. A probe without a copy falls back to the packaged task; DeepSWE's grader or an unknown file
        is planned as before."""
        from unittest import mock
        gr = self.gr
        globs = gr.GLOBS["Go"] + ["vm/tests/**"]
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            packaged = tmp / "cand-c01-r1b1/task/syn-x-cand-c01-r1b1-1234abcd"
            probe, scored = tmp / "round-1/probe-0001", tmp / "round-1/cand-c01-r1b1"
            copy = probe / "tasks" / packaged.name / "tests"
            metrics = {"task_id": packaged.name}

            def put(tests, grader: bytes, config: dict):
                tests.mkdir(parents=True, exist_ok=True)
                (tests / "grader.py").write_bytes(grader)
                (tests / "config.json").write_text(json.dumps(dict(config, base_commit="c" * 40)))

            def skip(status="accepted", run_globs=globs, mode="all"):
                return gr.sanitized_skip(metrics, scored, status, gr.probe_grader(probe, packaged), run_globs, mode)

            put(packaged / "tests", gr.OLD_GRADER.read_bytes(), {})
            graded = gr.probe_grader(probe, packaged)  # no probe copy: the packaged task's tests/
            self.assertEqual((graded["kind"], graded["tests"]), ("deepswe", str(packaged / "tests")))
            self.assertIsNone(skip())
            put(copy, gr.NEW_GRADER.read_bytes(), {"test_paths": globs, "test_paths_mode": "all"})
            graded = gr.probe_grader(probe, packaged)  # the probe's own copy wins
            self.assertEqual((graded["kind"], graded["tests"], graded["test_paths"]), ("current", str(copy), globs))
            self.assertEqual(graded["grader_sha256"], gr.file_sha(gr.NEW_GRADER))
            record = skip()
            self.assertEqual((record["task_id"], record["cand"], record["status"], record["decision"]),
                             (packaged.name, "cand-c01-r1b1", "skip_sanitized", "accepted"))
            self.assertTrue(record["matches_run_globs"])
            self.assertEqual(record["probe_grader"], graded)
            self.assertTrue(skip("rates", list(reversed(globs)))["matches_run_globs"])  # order does not matter
            self.assertFalse(skip("rates", gr.GLOBS["Go"])["matches_run_globs"])
            self.assertIn("NOT the run globs", skip("rates", gr.GLOBS["Go"])["why"])
            self.assertFalse(skip("rates", globs, "non_additive")["matches_run_globs"])
            put(copy, gr.NEW_GRADER.read_bytes(), {"test_paths": globs})  # no mode key: the grader's "all"
            self.assertTrue(skip("rates")["matches_run_globs"])
            prior = b"# an earlier sanitizer\n"
            put(copy, prior, {"test_paths": globs})
            self.assertIsNone(gr.probe_grader(probe, packaged)["kind"])
            self.assertIsNone(skip())  # an unknown grader is planned (and named as a problem by plan_run)
            with mock.patch.dict(gr.PRIOR_GRADERS, {gr.sha256(prior): "an earlier sanitizer (test)"}):
                self.assertEqual(gr.probe_grader(probe, packaged)["kind"], "prior")
                self.assertEqual(skip("recovered_in_range")["decision"], "recovered_in_range")
            put(copy, gr.OLD_GRADER.read_bytes(), {})
            self.assertIsNone(skip())

    def test_guard_rule_schedules_the_flips_that_could_end_an_acceptance(self):
        """accepted and recovered_in_range tasks (GUARDED): sc14 FAILs and clean kimi PASSes with surviving test paths
        are scheduled, everything else is listed; accepted tasks keep the 2026-10-19 wording."""
        gr = self.gr
        student, teacher = gr.STUDENT, gr.TEACHER
        cases = [({"model": student, "status": "fail"}, True), ({"model": student, "status": "pass"}, False),
                 ({"model": teacher, "status": "pass", "cheated": False}, True),
                 ({"model": teacher, "status": "pass", "cheated": None}, True),
                 ({"model": teacher, "status": "pass", "cheated": True}, False),
                 ({"model": teacher, "status": "fail"}, False), ({"model": teacher, "status": "infra"}, False),
                 ({"model": student, "status": "infra"}, False)]
        for slot, scheduled in cases:
            for label in ("accepted", "recovered in range"):
                why, schedule = gr.guard_rule(slot, label)
                self.assertEqual(schedule, scheduled, (slot, label))
                self.assertEqual(why.startswith(label + ": "), scheduled, why)
        self.assertEqual(gr.guard_rule({"model": student, "status": "fail"}, "accepted"),
                         ("accepted: sc14 FAIL touching test paths (could become a pass)", True))
        self.assertEqual(gr.guard_rule({"model": teacher, "status": "pass", "cheated": False}, "accepted"),
                         ("accepted: kimi clean PASS touching test paths (could become a fail)", True))
        self.assertEqual(gr.guard_rule({"model": teacher, "status": "fail"}, "accepted"),
                         (f"{teacher} FAIL: a flip can only help the acceptance", False))
        self.assertEqual(set(gr.GUARDED), {"accepted", "recovered_in_range"})

    def test_sanitize_marks_read_every_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "output.log").write_text(
                "[verifier] model.patch applied (10 bytes)\n"
                '[verifier] test-sanitize: 3 path(s) of model.patch match test_paths ["**/__tests__/**"]\n'
                "[verifier] test-sanitize: test_paths_mode non_additive: existing test files that model.patch only "
                "adds lines to are kept\n"
                '[verifier] test-sanitize: deleted "src/__tests__/x.test.ts" (not in base_commit)\n'
                '[verifier] test-sanitize: kept-additive "src/__tests__/index.test.ts"\n'
                '[verifier] test-sanitize: dangling-import "src/__tests__/b.test.ts": its added lines import "./x" -> '
                '"src/__tests__/x.test.ts" (deleted here): restored like a changed file\n'
                '[verifier] test-sanitize: restored "src/__tests__/a \\"q\\".test.ts" from base_commit\n'
                '[verifier] test-sanitize: restored "src/__tests__/b.test.ts" from base_commit\n'
                '[verifier] test-sanitize: dropped-manifest-target "crates/w/Cargo.toml" [[test]] "core" -> '
                '"crates/w/tests/core.rs" (deleted above): cut out\n'
                '[verifier] test-sanitize: restored-manifest-target "Cargo.toml" [[bench]] "b" -> "benches/b2.rs" '
                '(deleted above): base_commit\'s table is back\n')
            marks = self.gr.sanitize_marks(Path(tmp))
        self.assertEqual(marks["actions"], {"deleted": ["src/__tests__/x.test.ts"],
                                            "kept-additive": ["src/__tests__/index.test.ts"],
                                            "dangling-import": ["src/__tests__/b.test.ts"],
                                            "restored": ['src/__tests__/a "q".test.ts', "src/__tests__/b.test.ts"],
                                            "dropped-manifest-target": ["crates/w/Cargo.toml"],
                                            "restored-manifest-target": ["Cargo.toml"]})
        self.assertTrue(marks["applied"])
        self.assertEqual(len(marks["lines"]), 9)

    def test_atomic_write_keeps_the_first_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("v1")
            path.chmod(0o644)
            self.gr.atomic_write(path, "v2", dry_run=True)
            self.assertEqual(path.read_text(), "v1")
            self.assertFalse(path.with_name("config.json.pre-test-sanitize").exists())
            entry = self.gr.atomic_write(path, "v2")
            self.gr.atomic_write(path, "v3")
            self.assertEqual(path.read_text(), "v3")
            self.assertEqual(path.with_name("config.json.pre-test-sanitize").read_text(), "v1")
            self.assertEqual(oct(path.stat().st_mode & 0o777), "0o644")
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["config.json", "config.json.pre-test-sanitize"])
            self.assertTrue(entry["replaced"])

    def test_path_kinds_and_priorities(self):
        fx = Fixture(self, BASE_FILES)
        patch = fx.model({"tests/test_core.py": "changed\n", "tests/test_new.py": "x = 1\n", "tests/test_old.py": None,
                          "tests/sub/conftest.py": "y = 1\n"},
                         renames=[("tests/conftest.py", "tests/conftest_moved.py")]).decode()
        kinds = self.gr.path_kinds(patch)
        self.assertEqual(kinds, {"tests/test_core.py": "modified", "tests/test_new.py": "added",
                                 "tests/test_old.py": "deleted", "tests/sub/conftest.py": "added",
                                 "tests/conftest.py": "renamed-away", "tests/conftest_moved.py": "renamed-to"})
        p = self.gr.slot_priority
        self.assertEqual(p({"tests/test_core.py": "modified"}, "Python", None, [], False), "existing")
        self.assertEqual(p({"pkg/x_test.go": "added"}, "Go", None, [], False), "added_loaded")
        self.assertEqual(p({"tests/sub/conftest.py": "added"}, "Python", None, [], False), "added_loaded")
        report = '<testcase classname="tests.test_hidden" name="t"/><testcase classname="tests.test_new.T" name="u"/>'
        added = {"tests/test_new.py": "added"}
        self.assertEqual(p(added, "Python", report, ["tests/test_hidden.py"], False), "added_collected")
        other = '<testcase classname="tests.test_hidden" name="t"/><testcase classname="tests.test_newer" name="u"/>'
        self.assertEqual(p(added, "Python", other, ["tests/test_hidden.py"], False), "added_uncollected")
        self.assertEqual(p(added, "Python", other, ["tests/test_hidden.py"], True), "added_tree_checked")
        self.assertEqual(p(added, "Python", "<testsuite/>", ["tests/test_hidden.py"], False), "added_unknown")
        self.assertEqual(p(added, "Python", None, ["tests/test_hidden.py"], False), "added_unknown")
        self.assertIsNone(p({}, "Python", None, [], False))

    def test_in_range_is_scores_own_test(self):
        self.assertTrue(self.gr.in_range(1 / 5, 3 / 5))
        self.assertTrue(self.gr.in_range(0.0, 1.0))
        self.assertFalse(self.gr.in_range(2 / 5, 1.0))
        self.assertFalse(self.gr.in_range(0.0, 2 / 5))
        self.assertFalse(self.gr.in_range(None, 1.0))


def synthetic_accepted_task(case: unittest.TestCase, task_dir: Path, task_builder) -> None:
    """An accepted task as task_builder.package() lays it out, made from a Fixture repository (no live data):
    task.toml from task_builder's TASK_TOML template (base_commit = the fixture's real base commit), DeepSWE's own
    grader, config.json in write_json's format WITHOUT "test_paths" (as write_tests leaves it), the fixture's real
    test.patch (adds tests/test_hidden.py) and a reference solution.patch that touches no test path."""
    fx = Fixture(case, BASE_FILES, test_paths=None)
    solution = fx.model({"src/calc.py": "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"})
    tests, solution_dir, environment = task_dir / "tests", task_dir / "solution", task_dir / "environment"
    for directory in (tests, solution_dir, environment):
        directory.mkdir(parents=True)
    shutil.copyfile(ORIGINAL, tests / "grader.py")
    shutil.copyfile(fx.tests / "test.patch", tests / "test.patch")
    config = {"base_commit": fx.base, "f2p_node_ids": ["tests/test_hidden.py::test_hidden"],
              "p2p_node_ids": ["tests/test_core.py::test_add"],
              "grade": {"format": "junit", "reports": ["/logs/verifier/base.xml", "/logs/verifier/new.xml"],
                        "tool_label": "pytest"}}
    (tests / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (tests / "test.sh").write_text("#!/bin/bash\nset -uo pipefail\ncd /app || exit 6\n"
                                   "python3 /tests/grader.py prepare || exit $?\n"
                                   "[ -f /logs/verifier/reward.json ] && exit 0\n"
                                   "python3 -m pytest -q tests --junitxml=/logs/verifier/new.xml\n"
                                   "python3 /tests/grader.py grade\n")
    (solution_dir / "solution.patch").write_bytes(solution)
    (solution_dir / "solve.sh").write_text("#!/bin/bash\ncd /app && git apply /solution/solution.patch\n")
    (task_dir / "instruction.md").write_text("Add `sub(a, b)` next to `add` in `src/calc.py`; it returns `a - b`.\n")
    (task_dir / "rubric.md").write_text("- `sub` subtracts its second argument from its first.\n")
    (environment / "Dockerfile").write_text("FROM scratch\n")
    (environment / "start.patch").write_text("")
    collect = ("cd /app && mkdir -p /logs/artifacts && git config --global --add safe.directory /app "
               f"&& git diff --binary {fx.base} HEAD > /logs/artifacts/model.patch")  # the hook Gym's loader expects
    (task_dir / "task.toml").write_text(task_builder.fill(
        task_builder.TASK_TOML, task_id=task_dir.name, ext_id="0" * 32, title=json.dumps("Fixture: subtraction"),
        summary=json.dumps("Add sub() next to add()."), language="Python",
        repo_url="https://github.com/example/fixture", base_commit=fx.base, upstream=fx.base,
        target="fixture-target", kind="evolve", collect=json.dumps(collect),
        image="snapshot:00000000-0000-4000-8000-000000000000"))


class StageAndInstall(unittest.TestCase):
    """stage -> (validate.json) -> install on a synthetic accepted task (synthetic_accepted_task) in a temporary
    runs/ + repair/ tree; the module's RUNS/REPAIR/PLAN/LOG point there for the test (no sandbox, nothing under
    S/synth changes). It reads no live task: copying a real accepted one broke once the repair installed the new
    grader there (2026-10-20)."""

    def setUp(self):
        self.gr = gr = repair_tool()
        self.tmp = Path(tempfile.mkdtemp(prefix="test-grading-repair-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        saved = {name: getattr(gr, name) for name in ("RUNS", "REPAIR", "PLAN", "LOG")}
        self.addCleanup(lambda: [setattr(gr, k, v) for k, v in saved.items()])
        gr.RUNS, gr.REPAIR = self.tmp / "runs", self.tmp / "repair"
        gr.PLAN, gr.LOG = gr.REPAIR / "plan.json", gr.REPAIR / "test-sanitize.jsonl"
        self.run_name, self.task_id = "official1-fastapi", "syn-fixture-cand-c01-0a1b2c3d"  # an in-scope run name
        accepted = gr.RUNS / self.run_name / "accepted"
        synthetic_accepted_task(self, accepted / self.task_id, gr.task_builder)
        (accepted / "tasks.jsonl").write_text(json.dumps({"task_id": self.task_id}) + "\n")
        self.tests = accepted / self.task_id / "tests"
        self.assertEqual(gr.file_sha(self.tests / "grader.py"), gr.file_sha(gr.OLD_GRADER))  # DeepSWE's: not installed
        self.originals = {n: (self.tests / n).read_bytes() for n in ("grader.py", "config.json", "test.sh", "test.patch")}
        toml = gr.task_builder.task_toml(accepted / self.task_id)
        self.entry = {"run": self.run_name, "task_id": self.task_id, "status": "accepted", "language": "Python",
                      "task_dir": str(accepted / self.task_id), "globs": gr.GLOBS["Python"], "problems": [],
                      "snapshot": toml["environment"]["docker_image"], "scheduled": [], "listed": []}
        gr.REPAIR.mkdir(parents=True)
        gr.PLAN.write_text(json.dumps({"tasks": [self.entry]}))

    def fake_validation(self, ok=True):
        new_dir, _ = self.gr.staged_dirs(self.run_name, self.task_id)
        record = {"all_ok": ok, "grader_sha256": self.gr.file_sha(new_dir / "tests/grader.py"),
                  "config_sha256": self.gr.file_sha(new_dir / "tests/config.json")}
        self.gr.write_json_atomic(self.gr.task_home(self.run_name, self.task_id) / "validate.json", record)

    def test_install_needs_stage_and_validation_then_writes_with_backups(self):
        gr = self.gr
        with self.assertRaises(gr.RepairError):  # not staged
            gr.install_task(self.run_name, self.entry)
        record = gr.stage_task(self.run_name, self.entry)
        new_dir, old_dir = gr.staged_dirs(self.run_name, self.task_id)
        self.assertEqual((new_dir / "tests/grader.py").read_bytes(), GRADER.read_bytes())
        self.assertEqual((old_dir / "tests/grader.py").read_bytes(), ORIGINAL.read_bytes())
        self.assertEqual(json.loads((new_dir / "tests/config.json").read_text())["test_paths"], gr.GLOBS["Python"])
        self.assertEqual(json.loads((new_dir / "tests/config.json").read_text())["test_paths_mode"], "all")
        self.assertEqual(record["test_paths_mode"], "all")
        self.assertEqual(record["source_grader"], "deepswe")
        self.assertEqual((old_dir / "tests/config.json").read_bytes(), self.originals["config.json"])
        self.assertTrue(gr.stage_task(self.run_name, self.entry).get("unchanged"))  # idempotent
        with self.assertRaises(gr.RepairError):  # staged, not validated
            gr.install_task(self.run_name, self.entry)
        self.fake_validation(ok=False)
        with self.assertRaises(gr.RepairError):  # validation failed
            gr.install_task(self.run_name, self.entry)
        self.fake_validation()
        self.assertEqual(self.tests.joinpath("grader.py").read_bytes(), self.originals["grader.py"])  # untouched so far
        dry = gr.install_task(self.run_name, self.entry, dry_run=True)
        self.assertTrue(dry["dry_run"])
        self.assertEqual(self.tests.joinpath("grader.py").read_bytes(), self.originals["grader.py"])
        gr.install_task(self.run_name, self.entry)
        self.assertEqual((self.tests / "grader.py").read_bytes(), GRADER.read_bytes())
        config = json.loads((self.tests / "config.json").read_text())
        self.assertEqual(config.pop("test_paths"), gr.GLOBS["Python"])
        self.assertEqual(config.pop("test_paths_mode"), "all")  # Python keeps the full restore
        self.assertEqual(config, json.loads(self.originals["config.json"]))
        for name in ("grader.py", "config.json"):
            self.assertEqual((self.tests / f"{name}.pre-test-sanitize").read_bytes(), self.originals[name])
            self.assertFalse((self.tests / f"{name}.pre-non-additive").exists())
            self.assertFalse((self.tests / f"{name}.pre-grader-v3").exists())
        for name in ("test.sh", "test.patch"):
            self.assertEqual((self.tests / name).read_bytes(), self.originals[name])
        self.assertTrue(gr.install_task(self.run_name, self.entry).get("unchanged"))
        actions = [json.loads(line)["action"] for line in gr.LOG.read_text().splitlines()]
        self.assertEqual(actions, ["stage", "install"])
        self.assertEqual(record["source_sha256"]["tests/grader.py"], gr.file_sha(ORIGINAL))

    def test_install_refuses_drifted_tests_and_other_statuses(self):
        gr = self.gr
        gr.stage_task(self.run_name, self.entry)
        self.fake_validation()
        (self.tests / "test.sh").write_text(self.originals["test.sh"].decode() + "\n# edited after staging\n")
        with self.assertRaises(gr.RepairError):
            gr.install_task(self.run_name, self.entry)
        self.assertEqual((self.tests / "grader.py").read_bytes(), self.originals["grader.py"])
        with self.assertRaises(gr.RepairError):
            gr.install_task(self.run_name, dict(self.entry, status="near_bar"))


class StageAndInstallNonAdditive(StageAndInstall):
    """The 2026-10-20 refinement: a TypeScript task gets "test_paths_mode": "non_additive", and an accepted task that
    holds the FIRST sanitizer (installed 2026-10-19) is staged from it and upgraded with *.pre-non-additive backups;
    its *.pre-test-sanitize backups stay."""

    def setUp(self):
        super().setUp()
        gr = self.gr
        self.entry.update(language="TypeScript", globs=gr.GLOBS["TypeScript"])
        gr.PLAN.write_text(json.dumps({"tasks": [self.entry]}))

    def first_sanitizer_installed(self):
        """The packaged task (DeepSWE's grader) plus, in accepted/, what the first repair's install left: a known
        earlier sanitizer, config.json with "test_paths" but no mode, and the *.pre-test-sanitize backups."""
        gr = self.gr
        packaged = gr.RUNS / self.run_name / "build/fixture-target/example__fixture/cand-c01/task" / self.task_id
        shutil.copytree(self.tests.parent, packaged)
        prior = b"# the first sanitizer (a stand-in for its bytes)\n"
        saved = dict(gr.PRIOR_GRADERS)
        self.addCleanup(lambda: (gr.PRIOR_GRADERS.clear(), gr.PRIOR_GRADERS.update(saved)))
        gr.PRIOR_GRADERS[gr.sha256(prior)] = "test stand-in"
        first_config = json.dumps(dict(json.loads(self.originals["config.json"]), test_paths=gr.GLOBS["TypeScript"]),
                                  indent=2, sort_keys=True) + "\n"
        for name, data in (("grader.py", prior), ("config.json", first_config.encode())):
            (self.tests / f"{name}.pre-test-sanitize").write_bytes(self.originals[name])
            (self.tests / name).write_bytes(data)
        self.entry["packaged_task_dir"] = str(packaged)
        gr.PLAN.write_text(json.dumps({"tasks": [self.entry]}))
        return prior, first_config.encode(), packaged

    def test_typescript_tasks_are_staged_non_additive(self):
        gr = self.gr
        record = gr.stage_task(self.run_name, self.entry)
        new_dir, _ = gr.staged_dirs(self.run_name, self.task_id)
        config = json.loads((new_dir / "tests/config.json").read_text())
        self.assertEqual((config["test_paths"], config["test_paths_mode"]), (JS_GLOBS, "non_additive"))
        self.assertEqual(record["test_paths_mode"], "non_additive")
        self.assertEqual(gr.task_home(self.run_name, self.task_id).parent.parent.name, "tasks-v3")

    def test_upgrade_from_the_first_sanitizer(self):
        gr = self.gr
        prior, first_config, packaged = self.first_sanitizer_installed()
        record = gr.stage_task(self.run_name, self.entry)
        new_dir, old_dir = gr.staged_dirs(self.run_name, self.task_id)
        self.assertEqual(record["source_grader"], "prior")
        self.assertEqual((new_dir / "tests/grader.py").read_bytes(), GRADER.read_bytes())
        self.assertEqual(gr.without_test_paths((new_dir / "tests/config.json").read_text()),
                         json.loads(self.originals["config.json"]))
        for name in ("grader.py", "config.json", "test.sh", "test.patch"):  # the OLD control: DeepSWE's, as packaged
            self.assertEqual((old_dir / "tests" / name).read_bytes(), self.originals[name], name)
        self.assertFalse(list(new_dir.rglob("*.pre-test-sanitize")))  # backups never reach the scratch copy
        self.assertTrue(gr.staged_ok(self.run_name, self.task_id, record))
        self.fake_validation()
        gr.install_task(self.run_name, self.entry)
        self.assertEqual((self.tests / "grader.py").read_bytes(), GRADER.read_bytes())
        self.assertEqual(json.loads((self.tests / "config.json").read_text())["test_paths_mode"], "non_additive")
        self.assertEqual((self.tests / "grader.py.pre-grader-v3").read_bytes(), prior)
        self.assertEqual((self.tests / "config.json.pre-grader-v3").read_bytes(), first_config)
        self.assertFalse((self.tests / "grader.py.pre-non-additive").exists())
        for name in ("grader.py", "config.json"):  # the first install's backups are kept as they were
            self.assertEqual((self.tests / f"{name}.pre-test-sanitize").read_bytes(), self.originals[name])
        self.assertTrue(gr.install_task(self.run_name, self.entry).get("unchanged"))
        install = [json.loads(line) for line in gr.LOG.read_text().splitlines()][-1]
        self.assertEqual((install["action"], install["replaced_grader"], install["test_paths_mode"]),
                         ("install", "prior", "non_additive"))

    def test_upgrade_refuses_unknown_graders_and_drifted_configs(self):
        gr = self.gr
        prior, first_config, packaged = self.first_sanitizer_installed()
        config = json.loads((packaged / "tests/config.json").read_text())
        (packaged / "tests/config.json").write_text(json.dumps(dict(config, p2p_node_ids=["other"])))
        with self.assertRaises(gr.RepairError):  # accepted config != packaged config beyond the sanitizer keys
            gr.stage_task(self.run_name, self.entry)
        (self.tests / "grader.py").write_bytes(prior + b"# edited\n")
        with self.assertRaises(gr.RepairError):  # neither DeepSWE's, a known earlier sanitizer nor the new one
            gr.stage_task(self.run_name, self.entry)

    def test_install_needs_stage_and_validation_then_writes_with_backups(self):  # the TypeScript variant
        gr = self.gr
        gr.stage_task(self.run_name, self.entry)
        self.fake_validation()
        gr.install_task(self.run_name, self.entry)
        config = json.loads((self.tests / "config.json").read_text())
        self.assertEqual((config.pop("test_paths"), config.pop("test_paths_mode")), (JS_GLOBS, "non_additive"))
        self.assertEqual(config, json.loads(self.originals["config.json"]))
        for name in ("grader.py", "config.json"):  # DeepSWE's grader replaced: the first backups are .pre-test-sanitize
            self.assertEqual((self.tests / f"{name}.pre-test-sanitize").read_bytes(), self.originals[name])
            self.assertFalse((self.tests / f"{name}.pre-non-additive").exists())


    def test_upgrade_from_v2_keeps_its_config_and_backs_up_as_pre_grader_v3(self):
        """Grader v3 (2026-10-21): the test_paths_mode sanitizer (v2, sha 3936c324) is a prior grader; installing v3
        over it changes only grader.py (config.json already has both keys) and keeps <file>.pre-grader-v3 backups."""
        gr = self.gr
        v2 = "3936c324fb5d11e57c3e6d4397814808c6c660ef36ead93b58808adbaa4ac0ee"  # verifier/grader.py of 2026-10-20
        self.assertEqual((gr.grader_kind(v2), gr.grader_kind(gr.file_sha(GRADER))), ("prior", "current"))
        self.assertNotIn(gr.file_sha(GRADER), gr.PRIOR_GRADERS)
        prior, _, _ = self.first_sanitizer_installed()  # a stand-in for v2's bytes
        v2_config = gr.sanitized_config(self.originals["config.json"].decode(), JS_GLOBS, "non_additive")
        (self.tests / "config.json").write_text(v2_config)
        record = gr.stage_task(self.run_name, self.entry)
        new_dir, _ = gr.staged_dirs(self.run_name, self.task_id)
        self.assertEqual((record["source_grader"], (new_dir / "tests/config.json").read_text()), ("prior", v2_config))
        self.fake_validation()
        gr.install_task(self.run_name, self.entry)
        self.assertEqual((self.tests / "grader.py").read_bytes(), GRADER.read_bytes())
        self.assertEqual((self.tests / "config.json").read_text(), v2_config)
        self.assertEqual((self.tests / "grader.py.pre-grader-v3").read_bytes(), prior)
        self.assertEqual((self.tests / "config.json.pre-grader-v3").read_text(), v2_config)
        for name in ("grader.py", "config.json"):  # the first install's backups are kept as they were
            self.assertEqual((self.tests / f"{name}.pre-test-sanitize").read_bytes(), self.originals[name])


class RunScope(unittest.TestCase):
    """check_run (batch 2, 2026-10-20): the 11 runs of the first repair always; a FINISHED_LATER run only with its
    build's summary.json and no unfinished Slurm job of the run; any other run never. A temporary runs/ tree and a
    fake squeue (nothing is queried, nothing under S/synth changes)."""

    def setUp(self):
        self.gr = gr = repair_tool()
        self.tmp = Path(tempfile.mkdtemp(prefix="test-grading-repair-scope-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        saved, finished = {name: getattr(gr, name) for name in ("RUNS", "slurm_jobs")}, set(gr._FINISHED)

        def restore():
            for name, value in saved.items():
                setattr(gr, name, value)
            gr._FINISHED.clear()
            gr._FINISHED.update(finished)

        self.addCleanup(restore)
        gr.RUNS = self.tmp / "runs"
        gr._FINISHED.clear()
        self.real_slurm_jobs = saved["slurm_jobs"]
        self.jobs, self.queries = [], []

        def fake_squeue(users):
            self.queries.append(set(users))
            if isinstance(self.jobs, Exception):
                raise self.jobs
            return list(self.jobs)

        gr.slurm_jobs = fake_squeue
        self.run_name = gr.FINISHED_LATER[0]
        self.root = gr.RUNS / self.run_name / "build/some-target/owner__repo"
        self.root.mkdir(parents=True)
        (gr.RUNS / self.run_name / "launch.jsonl").write_text(
            json.dumps({"job_id": "4242", "run": self.run_name}) + "\nnot json\n")

    def test_first_repair_runs_need_no_check_and_others_are_refused(self):
        gr = self.gr
        self.assertEqual(len(gr.IN_SCOPE), 11)
        self.assertFalse(set(gr.IN_SCOPE) & set(gr.FINISHED_LATER))
        for run in gr.IN_SCOPE:
            self.assertEqual(gr.check_run(run), run)
        self.assertEqual(self.queries, [])  # no squeue for them
        for run in ("official1-live-build", "official1-unknown", "smoke1", "val1", self.run_name + "2", ""):
            with self.assertRaises(SystemExit, msg=run):
                gr.check_run(run)
        self.assertEqual(self.queries, [])

    def test_finished_later_needs_summary_json_and_no_unfinished_job(self):
        gr = self.gr
        with self.assertRaises(SystemExit):  # no summary.json yet: the build has not finished
            gr.check_run(self.run_name)
        self.assertEqual(self.queries, [])
        (self.root / "summary.json").write_text("{}\n")
        for jobs in ([{"id": "4242", "state": "RUNNING", "name": "renamed-job"}],  # a job id launch.jsonl recorded
                     [{"id": "4242_3", "state": "PENDING", "name": "array"}],
                     [{"id": "777", "state": "PENDING", "name": f"synth-build-{self.run_name}"}],
                     [{"id": "778+0", "state": "COMPLETING", "name": f"regrade-{self.run_name}-x"}],
                     gr.RepairError("squeue exit 1: Unable to contact slurm controller")):
            self.jobs = jobs
            with self.assertRaises(SystemExit, msg=str(jobs)):
                gr.check_run(self.run_name)
        self.assertNotIn(self.run_name, gr._FINISHED)
        self.jobs = [{"id": "779", "state": "RUNNING", "name": f"synth-build-{self.run_name}2"},  # another name
                     {"id": "780", "state": "RUNNING", "name": "synth-build-official1-anko"},
                     {"id": "42421", "state": "PENDING", "name": "swe-task-student-w01"}]
        self.assertEqual(gr.check_run(self.run_name), self.run_name)
        self.assertTrue(all(self.queries))  # every query named a user (the current one / the owner of runs/<R>)
        queried = len(self.queries)
        self.assertEqual(gr.check_run(self.run_name), self.run_name)  # one check per process
        self.assertEqual(len(self.queries), queried)

    def test_the_job_running_the_tool_does_not_count(self):
        gr = self.gr
        from unittest import mock
        (self.root / "summary.json").write_text("{}\n")
        own = {"id": "555", "state": "RUNNING", "name": f"repair-{self.run_name}"}
        environment = {k: v for k, v in os.environ.items() if k not in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID")}
        with mock.patch.dict(os.environ, environment, clear=True):
            self.jobs = [own]
            self.assertIn("555", gr.unfinished(self.run_name))  # outside that job it is a job of the run
            os.environ["SLURM_JOB_ID"] = "555"
            self.assertIsNone(gr.unfinished(self.run_name))
            self.jobs = [own, {"id": "556", "state": "PENDING", "name": f"synth-build-{self.run_name}"}]
            self.assertIn("556", gr.unfinished(self.run_name))

    def test_a_second_build_root_is_refused(self):
        gr = self.gr
        (self.root / "summary.json").write_text("{}\n")
        (self.root.parent.parent / "other-target/owner__repo").mkdir(parents=True)
        with self.assertRaises(SystemExit):
            gr.check_run(self.run_name)
        self.assertEqual(self.queries, [])

    def test_squeue_output_parsing_and_failures(self):
        gr = self.gr
        from unittest import mock
        calls = []

        def run(argv, **kw):
            calls.append((argv, kw))
            return subprocess.CompletedProcess(argv, 0, stdout="4242|RUNNING|synth-build-x\n\n"
                                               "4243_[1-3]|PENDING|odd|name\n", stderr="")

        with mock.patch.object(gr.subprocess, "run", run):
            jobs = self.real_slurm_jobs({"hengyuf", "other"})
        self.assertEqual(jobs, [{"id": "4242", "state": "RUNNING", "name": "synth-build-x"},
                                {"id": "4243_[1-3]", "state": "PENDING", "name": "odd|name"}])
        argv, kw = calls[0]
        self.assertEqual(argv[1:], ["-h", "-u", "hengyuf,other", "-o", "%i|%T|%j"])
        self.assertTrue(kw["env"]["SLURM_CONF"])
        with mock.patch.object(gr.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(
                argv, 1, stdout="", stderr="squeue: fatal: Could not establish a configuration source")):
            with self.assertRaises(gr.RepairError):
                self.real_slurm_jobs({"hengyuf"})

    def test_recovery_in_range_reads_recover_slots_records(self):
        """batch 3c: an inconclusive task is planned as recovered_in_range only while tools/recover_slots.py's record
        of it (runs/<R>/audit/recovery/<T>.json) recommends newly_in_range for that task and run."""
        gr = self.gr
        folder = gr.RUNS / self.run_name / "audit/recovery"
        folder.mkdir(parents=True)
        self.assertIsNone(gr.recovery_in_range(self.run_name, "t1"))  # no record
        record = {"run": self.run_name, "task_id": "t1", "recommendation": "newly_in_range", "at": "now",
                  "decision_view": "sanitizing_grader", "label": "x",
                  "after": {"sanitizing_grader": {"sc14": 0.0, "kimi_clean": 0.8, "valid": [5, 5], "in_range": True},
                            "recorded_grader": {"sc14": 0.0, "kimi_clean": 0.6, "valid": [5, 5], "in_range": True}},
                  "pending_repair": {"pending": ["sc14-r3", "kimi-r3"], "verdict_can_change": True}}
        (folder / "t1.json").write_text(json.dumps(record))
        got = gr.recovery_in_range(self.run_name, "t1")
        self.assertEqual((got["pending_repair"], got["rates_after"], got["verdict_can_change"], got["decision_view"]),
                         (["sc14-r3", "kimi-r3"], {"sc14": 0.0, "kimi_clean": 0.8, "valid": [5, 5], "in_range": True},
                          True, "sanitizing_grader"))
        self.assertEqual(got["sha256"], gr.file_sha(folder / "t1.json"))
        for change in ({"recommendation": "still_out_of_range"}, {"recommendation": "unclear"}, {"task_id": "t2"},
                       {"run": "official1-other"}):
            (folder / "t1.json").write_text(json.dumps(dict(record, **change)))
            self.assertIsNone(gr.recovery_in_range(self.run_name, "t1"), change)
        (folder / "t1.json").write_text("{not json")
        self.assertIsNone(gr.recovery_in_range(self.run_name, "t1"))
        self.assertIsNone(gr.recovery_in_range(self.run_name + "2", "t1"))  # another run's tree

    def test_plan_order(self):
        gr = self.gr
        self.assertEqual(gr.RUN_ORDER, gr.IN_SCOPE + gr.FINISHED_LATER)
        self.assertEqual(sorted(reversed(gr.RUN_ORDER), key=gr.run_rank), list(gr.RUN_ORDER))
        self.assertEqual(gr.run_rank("official1-unknown"), len(gr.RUN_ORDER))
        tasks = [{"run": run, "status": status, "task_id": task} for run in (self.run_name, gr.IN_SCOPE[-1])
                 for status, task in (("near_bar", "a"), ("accepted", "b"))]
        tasks.sort(key=lambda t: (gr.run_rank(t["run"]), t["run"], t["status"] != "accepted", t["task_id"]))
        self.assertEqual([(t["run"], t["task_id"]) for t in tasks],
                         [(gr.IN_SCOPE[-1], "b"), (gr.IN_SCOPE[-1], "a"), (self.run_name, "b"), (self.run_name, "a")])


class Verbatim(unittest.TestCase):
    def test_deepswe_grader_is_verbatim_outside_the_marked_sections(self):
        new, original = GRADER.read_text(), ORIGINAL.read_text()
        self.assertEqual(without_sections(new), original)
        self.assertEqual(new.count(SECTION_START), 2)  # the helper section and the one call in cmd_prepare
        section = new[new.index("def cmd_prepare"):]
        call = section[section.index(SECTION_START):section.index(SECTION_END)]
        self.assertIn("sanitize_test_paths(", call)
        self.assertLess(section.index('log(f"model.patch applied'), section.index(SECTION_START))
        self.assertLess(section.index(SECTION_END), section.index('log("Resetting files touched by test.patch")'))

    def test_python38_grammar_and_stdlib_only(self):
        tree = ast.parse(GRADER.read_text(), feature_version=(3, 8))
        imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                    for alias in node.names}
        imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertLessEqual(imported, {"json", "os", "re", "subprocess", "sys", "xml", "pathlib", "fnmatch", "shutil",
                                        "posixpath"})
        text = GRADER.read_text()
        for newer in ("removeprefix", "removesuffix", "functools.cache", "is_relative_to", "with_stem", "| None",
                      "missing_ok"):  # 3.9+ APIs, or kept out to stay obviously 3.8-safe
            self.assertNotIn(newer, text, newer)


if __name__ == "__main__":
    unittest.main()
