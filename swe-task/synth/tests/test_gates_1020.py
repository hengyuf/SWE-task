"""Offline tests of the 2026-10-20 gate fixes (S/README.md "2026-10-20" bullets): released copies below the leak floor
(MATERIALIZE's purge of the repo's own cached releases + a per-package rule in released_copies), identifier evidence
limited to the names a solution declares (official1-wasmi c01-r1b4; the 22:33 parser's docstring inversion and gaps
fixed 2026-10-21; v3: /* */ comments over a hunk's lines, PEP 701 f-strings, TS signatures over lines; v4: monotonic
over v2, c9f84ee0's reading frozen in the module), verifier infrastructure faults scored as infra,
test.patch edits of existing non-test files, stale oracles outside the broad suite, and build_env's test exclusions
kept out of P2P. No sandbox, network or LLM calls (bash, git and python3 run locally)."""
import sys

sys.dont_write_bytecode = True
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402
import base64  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402
import unittest  # noqa: E402
import zipfile  # noqa: E402
from unittest import mock  # noqa: E402

import rollouts  # noqa: E402
import task_builder  # noqa: E402
from task_builder import (ARCHIVE_MANIFEST, MATERIALIZE, PURGE_RELEASES, calibrate, dependency_leaks,  # noqa: E402
                          env_matches, env_unreliable, oracle_problems, own_package, package_problems, package_root,
                          package_rows, released_copies, released_names, removed_source_items, is_test_file,
                          test_patch_source_edits)

TARGET = "abs-stepped-slices"
SQLTOOLS = "/usr/local/share/.cache/yarn/v6/npm-@sqltools-formatter-1.2.3-1185726610acc37317ddab11c3c7f9066966bd20" \
           "-integrity/node_modules/@sqltools/formatter/"


def git(app: Path, *args):
    subprocess.run(["git", *args], cwd=app, check=True, capture_output=True)


class MaterializePurge(unittest.TestCase):
    """MATERIALIZE deletes cached releases of the repo's own packages (official1-sqlformatter c04: yarn's cache held the
    published @sqltools/formatter 1.2.3 with the removed CASE handling); installed copies and locked crates stay."""

    def setUp(self):
        if not shutil.which("git") or not shutil.which("od") or not shutil.which("base64"):
            self.skipTest("git/od/base64 missing")
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        t = self.tmp
        self.app, self.root, self.usr, self.home = t / "app", t / "root", t / "usr/local", t / "home"
        for directory in (self.app, self.root, self.usr, self.home):
            directory.mkdir(parents=True)
        files = {
            "package.json": json.dumps({"name": "vscode-sqltools", "private": True}),
            "packages/formatter/package.json": json.dumps({"name": "@sqltools/formatter"}),
            "packages/log/package.json": json.dumps({"name": "@sqltools/log"}),
            "packages/site/package.json": json.dumps({"name": "site", "private": True}),
            "test/fixtures/proj/package.json": json.dumps({"name": "lodash"}),  # a fixture, not a package of the repo
            "pyproject.toml": '[build-system]\nrequires = ["x"]\n\n[project]\nname = "My_Lib"\nversion = "1"\n',
            "Cargo.toml": '[workspace]\nmembers = ["crates/a"]\n',
            "crates/a/Cargo.toml": '[package]\nname = "wasm-thing"\nversion = "1.0.0"\n',
            "crates/b/Cargo.toml": '[package]\nname = "test-programs"\npublish = false\n',
            "Cargo.lock": '[[package]]\nname = "wasm-thing"\nversion = "0.9.0"\nsource = "registry+https://github.com/'
                          'rust-lang/crates.io-index"\n\n[[package]]\nname = "wasm-thing"\nversion = "1.0.0"\n',
            ".gitignore": "node_modules/\n",
        }
        for name, text in files.items():
            path = self.app / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        git(self.app, "init", "-q", "-b", "main")
        git(self.app, "add", "-A")
        git(self.app, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
        self.gone, self.kept = [], []
        yarn = self.usr / "share/.cache/yarn/v6"
        self.gone += [self.make(yarn / "npm-@sqltools-formatter-1.2.3-1185726610acc37317ddab11c3c7f9066966bd20-integrity"),
                      self.make(yarn / "npm-@sqltools-log-1.0.3-integrity")]  # older entries carry no hash
        self.kept += [self.make(yarn / "npm-lodash-4.17.21-cccccccccccccccccccccccccccccccccccccccc-integrity"),
                      self.make(yarn / "npm-site-1.0.0-dddddddddddddddddddddddddddddddddddddddd-integrity")]  # private
        berry = self.root / ".yarn/berry/cache"
        self.berry = self.make(berry / "@sqltools-formatter-npm-1.2.3-abcdef1234-10.zip", directory=False)
        self.kept.append(self.make(berry / "@sqltools-formatter-extra-npm-1.0.0-abcdef1234-10.zip", directory=False))
        self.gone.append(self.make(self.home / ".bun/install/cache/@sqltools/formatter@1.2.3@@@1"))
        tarball = b"tarball-bytes"
        own, other = hashlib.sha512(tarball).digest(), hashlib.sha512(b"other").digest()
        cacache = self.root / ".npm/_cacache"
        self.gone.append(self.make(cacache / f"content-v2/sha512/{own.hex()[:2]}/{own.hex()[2:4]}/{own.hex()[4:]}",
                                   directory=False))
        self.kept.append(self.make(cacache / f"content-v2/sha512/{other.hex()[:2]}/{other.hex()[2:4]}/{other.hex()[4:]}",
                                   directory=False))
        index = cacache / "index-v5/aa/bb/cc"
        index.parent.mkdir(parents=True)
        key = "make-fetch-happen:request-cache:https://registry.npmjs.org/"
        index.write_text("".join("h\t" + json.dumps({"key": key + url, "integrity": "sha512-" + base64.b64encode(
            digest).decode()}, separators=(",", ":")) + "\n" for url, digest in (
            ("@sqltools/formatter/-/formatter-1.2.3.tgz", own), ("lodash/-/lodash-4.17.21.tgz", other))))
        pip = self.root / ".cache/pip"
        self.gone.append(self.make(pip / "wheels/ab/cd/my_lib-1.0-py3-none-any.whl", directory=False))
        self.kept.append(self.make(pip / "wheels/ab/cd/requests-2.0-py3-none-any.whl", directory=False))
        for name, top in (("own.body", "my_lib"), ("other.body", "yarl")):
            body = pip / "http-v2/a/b" / name
            body.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(body, "w") as archive:
                archive.writestr(f"{top}/__init__.py", "x")
                archive.writestr(f"{top}-1.0.dist-info/METADATA", "x")
            (self.gone if top == "my_lib" else self.kept).append(body)
        for entry, dist in (("own", "my_lib-1.0.dist-info"), ("other", "yarl-1.0.dist-info")):
            (self.root / ".cache/uv/archive-v0" / entry / dist).mkdir(parents=True)
            (self.gone if entry == "own" else self.kept).append(self.root / ".cache/uv/archive-v0" / entry)
        cargo = self.root / ".cargo/registry"
        self.gone += [self.make(cargo / "src/index.crates.io-1/wasm-thing-1.1.0"),
                      self.make(cargo / "cache/index.crates.io-1/wasm-thing-1.1.0.crate", directory=False)]
        self.kept += [self.make(cargo / "src/index.crates.io-1/wasm-thing-0.9.0"),  # Cargo.lock takes it from crates.io
                      self.make(cargo / "cache/index.crates.io-1/wasm-thing-0.9.0.crate", directory=False),
                      self.make(cargo / "src/index.crates.io-1/wasm-thing-extra-1.0.0"),
                      self.make(cargo / "src/index.crates.io-1/test-programs-1.0.0")]  # publish = false
        self.kept.append(self.make(self.app / "node_modules/@sqltools/formatter"))  # installed copies stay

    @staticmethod
    def make(path: Path, directory=True) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if directory:
            path.mkdir()
            (path / "index.js").write_text("x")
        else:
            path.write_bytes(b"x")
        return path

    def run_script(self, script: str, **env) -> subprocess.CompletedProcess:
        script = (script.replace("/app", str(self.app)).replace("/root", str(self.root))
                  .replace("/usr/local", str(self.usr)).replace("/tmp/jest_* /tmp/vitest*", f"{self.tmp}/jest_*")
                  .replace("/tmp/purge-own", f"{self.tmp}/purge-own").replace("command -v go", "false"))
        variables = {k: v for k, v in os.environ.items() if k not in (
            "YARN_CACHE_FOLDER", "npm_config_cache", "BUN_INSTALL_CACHE_DIR", "PIP_CACHE_DIR", "UV_CACHE_DIR",
            "CARGO_HOME", "UV_LINK_MODE")}
        variables["HOME"] = str(self.home)
        return subprocess.run(["bash", "-c", script, "materialize", ""], capture_output=True, text=True,
                              env=dict(variables, **env), cwd=self.app)

    def test_materialize_purges_own_cached_releases_only(self):
        result = self.run_script(MATERIALIZE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([str(p) for p in self.gone + [self.berry] if p.exists()], [])
        self.assertEqual([str(p) for p in self.kept if not p.exists()], [])
        self.assertIn("own packages: npm 2, python 1, crates 1", result.stderr)
        self.assertIn(f"purged released copy {self.berry}", result.stderr)
        self.assertEqual(result.stderr.count(f"purged released copy {self.berry}"), 1)  # HOME=/root lists it once
        self.assertRegex(result.stdout.strip(), r"^[0-9a-f]{40}$")  # the base commit, still the last line

    def test_dry_run_lists_and_keeps(self):
        result = self.run_script(PURGE_RELEASES, PURGE_DRY_RUN="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(p.exists() for p in self.gone + self.kept + [self.berry]))
        listed = [line.split("would purge released copy ", 1)[1] for line in result.stderr.splitlines()
                  if "would purge released copy " in line]
        self.assertEqual(sorted(listed), sorted(str(p) for p in self.gone + [self.berry]))
        result = self.run_script(PURGE_RELEASES, PURGE_DRY_RUN="1", HOME=str(self.root))  # /root and $HOME: once
        self.assertEqual(result.stderr.count(f"would purge released copy {self.berry}\n"), 1)

    def test_plug_n_play_keeps_the_archives_it_loads(self):
        fixture = self.app / "e2e/pnp/.pnp.cjs"  # jest's PnP e2e fixture names no cache archive: no reason to keep
        fixture.parent.mkdir(parents=True)
        fixture.write_text("// pnp runtime: this.libzip.file.getExternalAttributes(this.zip, ...)\n")
        result = self.run_script(PURGE_RELEASES, PURGE_DRY_RUN="1")
        self.assertIn(f"would purge released copy {self.berry}", result.stderr)
        (self.app / ".pnp.cjs").write_text('packageLocation: "../root/.yarn/berry/cache/' + self.berry.name
                                           + '/node_modules/@sqltools/formatter/"\n')
        result = self.run_script(PURGE_RELEASES)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.berry.exists())  # with Plug'n'Play this archive is the install
        self.assertFalse(self.gone[0].exists())

    def test_never_fails_materialize(self):
        broken = PURGE_RELEASES.replace("cd /app || exit 0", "cd /app || exit 0\n  false; exit 7")
        result = self.run_script("set -euo pipefail\n" + broken + "\necho after")
        self.assertEqual((result.returncode, result.stdout.strip()), (0, "after"))


class PackageRule(unittest.TestCase):
    """Released copies judged per package of the repo (official1-sqlformatter c01: 10/20 lines in 3 files of the cached
    @sqltools/formatter, no single file reaching the per-file rule)."""
    RELEASED = {"names": ["@sqltools/formatter", "formatter", "vscode-sqltools", "jest-worker", "wasmtime"],
                "go_modules": ["github.com/cerbos/cerbos"]}

    def test_package_roots(self):
        cases = {
            SQLTOOLS + "lib/core/Formatter.js": (SQLTOOLS, "@sqltools/formatter"),
            "/root/.yarn/berry/cache/jest-worker-npm-29.7.0-4d3567fed6-10.zip!/node_modules/jest-worker/build/a.js":
                ("/root/.yarn/berry/cache/jest-worker-npm-29.7.0-4d3567fed6-10.zip!/node_modules/jest-worker/",
                 "jest-worker"),
            "/root/.npm/_cacache https://registry.npmjs.org/@sqltools/formatter/-/formatter-1.2.3.tgz!/package/lib/a.js":
                ("/root/.npm/_cacache https://registry.npmjs.org/@sqltools/formatter/-/formatter-1.2.3.tgz!/",
                 "@sqltools/formatter"),
            "/root/.cache/pip/http-v2/a/b/c.body!/aiohttp/cookiejar.py": ("/root/.cache/pip/http-v2/a/b/c.body!/", "aiohttp"),
            "/root/.cargo/registry/src/index.crates.io-1/wasmtime-30.0.0/src/lib.rs":
                ("/root/.cargo/registry/src/index.crates.io-1/wasmtime-30.0.0/", "wasmtime"),
            "/root/go/pkg/mod/github.com/cerbos/cerbos@v0.1.0/internal/x.go":
                ("/root/go/pkg/mod/github.com/cerbos/cerbos@v0.1.0/", "github.com/cerbos/cerbos"),
            "/usr/lib/python3/site-packages/pip/_vendor/requests/api.py":
                ("/usr/lib/python3/site-packages/pip/_vendor/requests/", "requests"),
            "/opt/released/formatter/lib/a.js": ("/opt/released/formatter/", "formatter"),
        }
        for path, expected in cases.items():
            self.assertEqual(package_root(path, ["/opt/released/formatter"]), expected, path)
        self.assertIsNone(package_root("/app/packages/formatter/src/a.ts"))
        self.assertTrue(own_package("@sqltools/formatter", self.RELEASED))
        self.assertTrue(own_package("@sqltools-formatter", self.RELEASED))  # berry's flat scoped name
        self.assertTrue(own_package("github.com/cerbos/cerbos", self.RELEASED))
        self.assertFalse(own_package("sql-formatter", self.RELEASED))  # exact names, unlike name_match
        self.assertFalse(own_package("@types/node", {"names": ["types"], "go_modules": []}))

    def test_lines_spread_over_a_package_add_up(self):
        per_source = {"src/Formatter.ts": [f"formatter line {i} of the removed code" for i in range(6)],
                      "src/Tokenizer.ts": [f"tokenizer line {i} of the removed code" for i in range(2)],
                      "src/PlSql.ts": [f"plsql line {i} of the removed code" for i in range(2)]}
        hits = {SQLTOOLS + "lib/core/Formatter.js": [(10 + i, f"formatter line {i} of the removed code") for i in (0, 1, 2)],
                SQLTOOLS + "lib/core/Tokenizer.js": [(5, "tokenizer line 0 of the removed code"),
                                                     (900, "tokenizer line 1 of the removed code")],
                "/usr/local/share/.cache/yarn/v6/npm-sql-formatter-15.0.0-integrity/node_modules/sql-formatter/x.js":
                    [(1, f"formatter line {i} of the removed code") for i in range(6)]}  # not the repo's package
        self.assertEqual(task_builder.leak_problems(hits, per_source, {}), [
            "/usr/local/share/.cache/yarn/v6/npm-sql-formatter-15.0.0-integrity/node_modules/sql-formatter/x.js holds "
            "6/6 distinctive lines of src/Formatter.ts within 60 lines"])  # the per-file rule (any path) is unchanged
        rows = package_rows(hits, per_source, self.RELEASED, minimum=4)
        self.assertEqual([(row["root"], row["best"], row["total"], row["need"]) for row in rows], [(SQLTOOLS, 5, 10, 5)])
        problems = package_problems(rows, [], unit="lines")
        self.assertEqual(problems, [f"{SQLTOOLS} holds 5/10 distinctive lines of the solution's code files (from 2 of "
                                    "them) across 2 files of one released package"])
        self.assertEqual(dependency_leaks(problems), problems)  # a released copy: the candidate fails at once
        named = [f"{SQLTOOLS}lib/core/Formatter.js holds 4/6 distinctive lines of src/Formatter.ts within 60 lines"]
        self.assertEqual(package_problems(rows, named, unit="lines"), [])  # already reported per file
        tiny = {"src/Formatter.ts": ["token.value = this.formatCase(token.value);"]}  # c04: 1 line in all
        rows = package_rows({SQLTOOLS + "lib/core/Formatter.js": [(3, tiny["src/Formatter.ts"][0])]}, tiny,
                            self.RELEASED, minimum=4)
        self.assertEqual((rows[0]["best"], rows[0]["need"]), (1, 4))
        self.assertEqual(package_problems(rows, [], unit="lines"), [])  # below the floor: the purge handles it

    def test_released_copies_add_package_problems(self):
        solution_files = ("Formatter", "Indentation", "InlineBlock")
        solution = "".join(
            f"diff --git a/packages/formatter/src/core/{name}.ts b/packages/formatter/src/core/{name}.ts\n"
            f"--- a/packages/formatter/src/core/{name}.ts\n+++ b/packages/formatter/src/core/{name}.ts\n@@ -1 +1,3 @@\n"
            f" x\n+    if (this.{name.lower()}Level === 0 && this.isInline(tokens, index)) {{\n"
            f"+      this.{name.lower()}Types.push(TOP_LEVEL_INDENT_MARKER);\n" for name in solution_files)
        lines = {name: task_builder.leak_tokens(solution)[f"packages/formatter/src/core/{name}.ts"]
                 for name in solution_files}
        cache = "/usr/local/share/.cache/yarn/v6/npm-@sqltools-formatter-1.2.3-1185726610acc37317ddab11c3c7f9066966bd" \
                "20-integrity/node_modules/@sqltools/formatter/lib/core"
        outputs = [
            'F\tpackage.json\t{"name": "vscode-sqltools", "private": true, "workspaces": ["packages/*"]}\n'
            "J\tpackages/formatter/package.json\t@sqltools/formatter\t\nE\tdone",
            "X\tunzip\nR\t/usr/local/share/.cache\nE\tdone",
            "\n".join([f"H\t{cache}/{name}.js:{10 + i}:{100 * i}:{line}" for name in solution_files[:2]
                       for i, line in enumerate(lines[name])] + ["E\tdone"])]

        class Box:
            def __init__(self):
                self.calls = 0

            async def put_text(self, text, remote):
                pass

            async def sh(self, command, *, cwd="/app", timeout=1800, check=True):
                self.calls += 1
                return types.SimpleNamespace(stdout=outputs[self.calls - 1], exit_code=0, timed_out=False)

        report = asyncio.run(released_copies(Box(), "mtxr/vscode-sqltools", solution))
        root = cache.rsplit("/lib/core", 1)[0] + "/"
        self.assertEqual(report["problems"], [f"{root} holds 4/6 distinctive lines of the solution's code files (from 2 "
                                              "of them) across 2 files of one released package"])
        self.assertEqual(report["package_lines"][0]["sources"], {
            f"packages/formatter/src/core/{name}.ts": sorted(lines[name]) for name in solution_files[:2]})


# released_names of task_builder.py before 19:57 (md5 f7147f64), verbatim apart from its name: the discovery
# names that OwnPackageNames.test_discovery_names_are_exactly_pre_1957 holds the stage to
def pre_1957_released_names(repo: str, manifest: str) -> dict:
    """The names under which the repo's own packages are released, from ARCHIVE_MANIFEST's output: the repo name, the
    "name" of every public JS workspace package (root package.json "workspaces", pnpm-workspace.yaml; also the root
    package.json) and its directory name (packages/jest-get-type is now "@jest/get-type", but the cache holds the
    pre-rename jest-get-type release), the Python project and top-level import packages, and the Go module paths."""
    files, packages, modules, module_dirs, python = {}, {}, set(), {}, set()
    for line in manifest.splitlines():
        tag, _, rest = line.partition("\t")
        if tag == "F":
            name, _, text = rest.partition("\t")
            files.setdefault(name, []).append(text)
        elif tag == "J" and "\t" in rest:
            path, name, *flags = rest.split("\t")
            if "private" not in flags and "unparsed" not in flags:  # (ARCHIVE_MANIFEST printed no line
                # for a package.json that is no valid JSON before 2026-10-20)
                packages[path] = name.strip()
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
    names |= {name.lower() for name in python if name.lower() not in task_builder.GENERIC_PACKAGES}
    for module in modules:
        names.add(re.sub(r"/v\d+$", "", module).rsplit("/", 1)[-1].lower())
    return {"names": sorted(name for name in names if len(name.lstrip("@")) >= 3), "go_modules": sorted(modules),
            "go_module_dirs": dict(sorted(module_dirs.items())), "workspace_dirs": dict(sorted(directories.items()))}


class OwnPackageNames(unittest.TestCase):
    """released_names beyond discovery (2026-10-20): what ARCHIVE_DISCOVER looks for ("names", "go_modules") stays
    exactly as before 19:57; the repo's crates and the names that the full manifest parsing finds (pnpm flow and
    unindented lists, lerna, rush, "./" and {a,b} globs, go.mod paths without comments: "own_names") serve the
    per-package rule only, and what could not be read is "unparsed". The third-party exemption these names served at
    19:57 and 21:12 is gone (DeclaredIdentifiers)."""
    MANIFESTS = {
        "bytecodealliance/wasmtime": [
            'K\tCargo.toml\tname = "wasmtime-cli"', 'K\tcrates/wasmtime/Cargo.toml\tname = "wasmtime"',
            'K\tcrates/wizer/Cargo.toml\tname = "wasmtime-wizer"',
            "K\tcranelift/codegen/Cargo.toml\tname = 'cranelift-codegen'  # a TOML literal string",
            'K\tcranelift/frontend/Cargo.toml\tname = "cranelift-frontend"',
            'K\tcrates/test-programs/Cargo.toml\tname = "test-programs"',
            "K\tcrates/test-programs/Cargo.toml\tpublish = false",
            'K\tcrates/fuzzing/Cargo.toml\tname = "wasmtime-fuzzing"', "K\tcrates/fuzzing/Cargo.toml\tpublish = [ ]",
            'K\texamples/min-platform/embedding/Cargo.toml\tname = "embedding"'],
        "mtxr/vscode-sqltools": [
            'F\tpackage.json\t{"name": "sqltools", "private": true, "workspaces": ["packages/*"]}',
            "J\tpackage.json\tsqltools\tprivate", "J\tpackages/formatter/package.json\t@sqltools/formatter\t",
            "J\tpackages/plugins/package.json\t@sqltools/plugins\tprivate",
            "J\ttest/package.json\t@sqltools/test\tprivate",
            "J\tpackages/driver.pg/package.json\tsqltools-driver-pg\tprivate"],
        "jestjs/jest": [
            'F\tpackage.json\t{"name": "@jest/monorepo", "private": true, "workspaces": ["packages/*"]}',
            "J\tpackage.json\t@jest/monorepo\tprivate", "J\tpackages/jest-worker/package.json\tjest-worker\t",
            "J\tpackages/pretty-format/package.json\tpretty-format\t",
            "J\tpackages/jest-resolve/src/__mocks__/foo/package.json\tfoo\t"],  # a test mock, not the repo's
        "open-telemetry/opentelemetry-collector": [
            "G\tgo.mod\tgo.opentelemetry.io/collector", "G\tpdata/go.mod\tgo.opentelemetry.io/collector/pdata",
            "G\tconfmap/go.mod\tgo.opentelemetry.io/collector/confmap"]}

    def released(self, repo, *extra):
        return released_names(repo, "\n".join(self.MANIFESTS[repo] + list(extra) + ["E\tdone"]))

    def test_released_names_keep_crates_out_of_discovery(self):
        wasmtime = self.released("bytecodealliance/wasmtime")
        self.assertEqual(wasmtime["names"], ["wasmtime"])  # what ARCHIVE_DISCOVER looks for: as before 19:57
        self.assertEqual(wasmtime["crates"], ["cranelift-codegen", "cranelift-frontend", "wasmtime", "wasmtime-cli",
                                              "wasmtime-wizer"])  # not test-programs, wasmtime-fuzzing, embedding
        self.assertEqual(wasmtime["unparsed"], [])
        self.assertTrue(own_package("cranelift-frontend", wasmtime))  # the per-package rule counts own crates
        self.assertFalse(own_package("test-programs", wasmtime))
        self.assertEqual(set(wasmtime), {"names", "go_modules", "go_module_dirs", "workspace_dirs", "own_names",
                                         "crates", "unparsed"})
        sqltools = self.released("mtxr/vscode-sqltools")
        self.assertEqual(sqltools["names"], ["@sqltools/formatter", "formatter", "vscode-sqltools"])
        jest = self.released("jestjs/jest")
        self.assertNotIn("foo", jest["names"])  # __mocks__/foo/package.json is no package of the repo

    def test_manifest_script_lists_crates(self):
        if not shutil.which("git"):
            self.skipTest("git missing")
        files = {
            "Cargo.toml": '[workspace]\nmembers = ["crates/*"]\n\n[package]\nname = "wasmtime-cli"\n',
            "crates/wizer/Cargo.toml": '[package] # the snapshotter\nname = "wasmtime-wizer"\n\n[[bin]]\n'
                                       'name = "wizer"\n\n[package.metadata.docs.rs]\nname = "not-a-crate"\n',
            "crates/test-programs/Cargo.toml": '[package]\nname = "test-programs"\npublish = false\n',
            "crates/plain/Cargo.toml": "[package]\nname = 'single-quoted'\n",
            "crates/no-package/Cargo.toml": '[dependencies]\nname = "x"\n',
            "examples/embedding/Cargo.toml": '[package]\nname = "embedding"\n',
            "vendor/wasm-encoder/Cargo.toml": '[package]\nname = "wasm-encoder"\n'}
        output = self.run_manifest(files)
        self.assertEqual(sorted(line for line in output.splitlines() if line.startswith("K\t")), sorted([
            "K\tCargo.toml\t[package]", 'K\tCargo.toml\tname = "wasmtime-cli"',
            "K\tcrates/plain/Cargo.toml\t[package]", "K\tcrates/plain/Cargo.toml\tname = 'single-quoted'",
            "K\tcrates/test-programs/Cargo.toml\t[package]",
            'K\tcrates/test-programs/Cargo.toml\tname = "test-programs"',
            "K\tcrates/test-programs/Cargo.toml\tpublish = false",
            "K\tcrates/wizer/Cargo.toml\t[package]", 'K\tcrates/wizer/Cargo.toml\tname = "wasmtime-wizer"',
            "K\texamples/embedding/Cargo.toml\t[package]",
            'K\texamples/embedding/Cargo.toml\tname = "embedding"']))  # no [[bin]] or metadata names, no vendor/
        released = released_names("bytecodealliance/wasmtime", output)
        self.assertEqual((released["names"], released["crates"], released["unparsed"]),
                         (["wasmtime"], ["single-quoted", "wasmtime-cli", "wasmtime-wizer"], []))

    def run_manifest(self, files, script=ARCHIVE_MANIFEST):
        """ARCHIVE_MANIFEST's output on a git repository holding `files` (python3, no node: both parsers are tried)."""
        if not shutil.which("git"):
            self.skipTest("git missing")
        outputs = []
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory) / "app"
            for name, text in files.items():
                path = app / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            git(app, "init", "-q", "-b", "main")
            git(app, "add", "-A")
            script = script.replace("cd /app", f"cd {app}").replace("/tmp/leak-pkgjson", f"{directory}/pkgjson")
            variants = [script] + ([script.replace("command -v node", "false")] if shutil.which("node") else [])
            for variant in variants:
                result = subprocess.run(["bash", "-c", variant], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(result.stdout.endswith("E\tdone\n"))
                outputs.append(result.stdout)
        self.assertEqual(len(set(outputs)), 1, outputs)  # node and python3 print the same
        return outputs[0]

    def test_manifest_script_reads_rush_and_marks_gaps(self):
        files = {"rush.json": '/* rush */ {"projects": [{"packageName": "@r/a", "projectFolder": "libs/a"},]}\n',
                 "libs/a/package.json": json.dumps({"name": "@r/a"}),
                 "libs/b/package.json": '{"name": "@r/b",}',  # no valid JSON
                 "test/fixture/package.json": "{",  # broken on purpose, in a test tree
                 "go.mod": "module example.com/m // the root\n", "tools/go.mod": "go 1.22\n",
                 "a/Cargo.toml": '[package]\n"name" = "quoted-key"\n'}
        output = self.run_manifest(files)
        self.assertIn('F\trush.json\t/* rush */ {"projects": [{"packageName": "@r/a", "projectFolder": "libs/a"},]}',
                      output)
        self.assertIn("J\tlibs/b/package.json\t\tunparsed", output)
        released = released_names("rush/repo", output)
        self.assertEqual(released["go_modules"], ["example.com/m // the root"])  # discovery: as before 19:57
        self.assertEqual(released["names"], [" the root", "repo"])  # (its junk name finds nothing, as before)
        self.assertEqual(released["own_names"], ["@r/a", "example.com/m", "repo"])  # the rush project's package
        self.assertTrue(own_package("@r/a", released) and own_package("example.com/m", released))
        self.assertEqual(released["unparsed"], ["tools/go.mod: no module path", "libs/b/package.json: no valid JSON",
                                                "a/Cargo.toml: no [package] name"])
        capped = self.run_manifest({f"m{i}/go.mod": f"module example.com/m{i}\n" for i in range(3)},
                                   ARCHIVE_MANIFEST.replace("head -501", "head -3").replace('-gt 500', '-gt 2'))
        self.assertIn("Z\tgo.mod\tmore than 500 go.mod files, the rest unread", capped)
        self.assertEqual(len(released_names("x/y", capped)["go_modules"]), 2)
        self.assertIn("go.mod: more than 500 go.mod files, the rest unread", released_names("x/y", capped)["unparsed"])

    def test_workspace_manifests_name_own_packages(self):
        packages = ["J\tpackages/alpha/package.json\t@x/alpha\t", "J\tapps/web/package.json\tweb-app\tprivate",
                    "J\tapps/web/test/package.json\tweb-test\t", "J\ttools/gen/package.json\tgen-tool\t",
                    "J\tlibs/beta/package.json\t@x/beta\t", "J\tmodules/gamma/package.json\tmod-gamma\t"]

        def names(*lines):  # (discovery names, own names, unparsed)
            released = released_names("x/mono", "\n".join(list(lines) + packages + ["E\tdone"]))
            return released["names"], released["own_names"], released["unparsed"]

        own = ["@x/alpha", "alpha", "mono"]  # web-app is private, the rest lie outside the workspaces
        for pnpm in (["packages: ['packages/*', \"apps/*\"]  # flow"],
                     ["packages: [", "  'packages/*',", '  "apps/*" # web', "]", "catalog:", "  x: 1"],
                     ["packages:", "  ['packages/*', 'apps/*']"],
                     ["packages:", "- packages/*", "- 'apps/*'", "- '!**/test/**'"]):
            # discovery reads only indented block lists, as before 19:57; the per-package rule reads them all
            self.assertEqual(names(*[f"F\tpnpm-workspace.yaml\t{line}" for line in pnpm]), (["mono"], own, []), pnpm)
        indented = ["catalog:", "  react: ^18", "packages:", "  # all", "  - 'packages/*'", "  - apps/* # web"]
        self.assertEqual(names(*[f"F\tpnpm-workspace.yaml\t{line}" for line in indented]), (own, own, []))
        rush = ("// rush.json\n{\n  /* projects */\n  \"projects\": [\n    {\"packageName\": \"@x/alpha\", "
                "\"projectFolder\": \"./packages/alpha\"},\n    {\"packageName\": \"web-app\", \"projectFolder\": "
                "\"apps/web\", }, // trailing comma\n  ],\n  \"url\": \"https://example.com/a//b\"\n}")
        self.assertEqual(names(*[f"F\trush.json\t{line}" for line in rush.splitlines()]), (["mono"], own, []))
        self.assertEqual(names('F\tlerna.json\t{"packages": ["modules/*", "./packages/*"], "version": "1.0.0"}'),
                         (["mono"], ["@x/alpha", "alpha", "gamma", "mod-gamma", "mono"], []))  # lerna, a "./" path
        self.assertEqual(names('F\tlerna.json\t{"version": "independent"}')[:2], (["mono"], own))  # packages/*
        self.assertEqual(names('F\tpackage.json\t{"workspaces": {"packages": ["./{packages,libs}/*"]}}'),
                         (["mono"], ["@x/alpha", "@x/beta", "alpha", "beta", "mono"], []))
        self.assertEqual(names('F\tpackage.json\t{"workspaces": {"packages": ["packages/*", "libs/*"]}}')[:2],
                         (["@x/alpha", "@x/beta", "alpha", "beta", "mono"],) * 2)

    def test_discovery_names_are_exactly_pre_1957(self):
        """released_names' discovery keys equal the pre-19:57 function's (frozen below, verbatim) on every manifest,
        the unparsed package.json lines that ARCHIVE_MANIFEST now prints included; only own_names grows."""
        packages = ["J\tpackages/alpha/package.json\t@x/alpha\t", "J\tapps/web/package.json\tweb-app\tprivate",
                    "J\tlibs/beta/package.json\t@x/beta\t", "J\tmodules/gamma/package.json\tmod-gamma\t",
                    "J\tbroken/package.json\t\tunparsed", "P\tsrc/mono_py/__init__.py", "P\ttests/__init__.py"]
        extras = [[], ["F\tpnpm-workspace.yaml\tpackages:", "F\tpnpm-workspace.yaml\t- packages/*"],
                  ["F\tpnpm-workspace.yaml\tpackages:", "F\tpnpm-workspace.yaml\t  - 'packages/*'",
                   "F\tpnpm-workspace.yaml\t  - '!**/test/**'"],
                  ['F\tlerna.json\t{"packages": ["modules/*"]}'], ['F\tpackage.json\t{"workspaces": ["./libs/*"]}'],
                  ['F\tpackage.json\t{"workspaces": ["broken/*", "packages/*"]}'],
                  ["G\tgo.mod\texample.com/mono // a comment", "G\tsub/go.mod\texample.com/mono/sub/v2",
                   "G\tempty/go.mod\t"],
                  ["F\tpyproject.toml\t[project]", 'F\tpyproject.toml\tname = "mono-dist"', "F\tsetup.py\tsetup(",
                   "F\tsetup.py\t    name='mono-setup',"],
                  ['K\tCargo.toml\tname = "mono-crate"', "Z\tCargo.toml\tmore than 2000 Cargo.toml files"]]
        manifests = [("x/mono", "\n".join(packages + extra + ["E\tdone"])) for extra in extras]
        manifests += [(repo, "\n".join(lines + ["E\tdone"])) for repo, lines in self.MANIFESTS.items()]
        for repo, manifest in manifests:
            released, before = released_names(repo, manifest), pre_1957_released_names(repo, manifest)
            self.assertEqual({key: released[key] for key in before}, before, manifest)
            self.assertTrue(set(released["names"]) <= set(released["own_names"]) | {" a comment"}, manifest)

    def test_unparsed_manifests_are_reported(self):
        gaps = {  # one more manifest line each: what it would add to the own names is unknown
            "K\tcrates/x/Cargo.toml\t[package]": "crates/x/Cargo.toml: no [package] name",
            "Z\tCargo.toml\tmore than 2000 Cargo.toml files, the rest unread":
                "Cargo.toml: more than 2000 Cargo.toml files, the rest unread",
            'F\tpackage.json\t{"workspaces": ["packages/*",]}':
                'package.json: no valid JSON object with a "workspaces" list of globs',
            'F\tpackage.json\t{"workspaces": "packages/*"}':
                'package.json: no valid JSON object with a "workspaces" list of globs',
            'F\tpackage.json\t{"workspaces": ["packages/[ab]*"]}': "workspace glob 'packages/[ab]*' not understood",
            "F\tpnpm-workspace.yaml\tpackages: *shared": "pnpm-workspace.yaml: packages value '*shared' not understood",
            "F\tpnpm-workspace.yaml\tpackages: ['a/*'": "pnpm-workspace.yaml: packages value \"['a/*'\" not understood",
            'F\tlerna.json\t{"packages": "packages/*"}': 'lerna.json: no valid JSON object with a "packages" list of '
                                                         'globs',
            'F\trush.json\t{"projects": [{"packageName": "@x/a"}]}':
                'rush.json: no valid JSON with a "projects" list of {packageName, projectFolder}',
            "J\tpackage.json\t\tunparsed": "package.json: no valid JSON",
            "J\tbindings/node/package.json\t\tunparsed": "bindings/node/package.json: no valid JSON",
            "G\tgo.mod\t": "go.mod: no module path",
            "F\tsetup.py\tsetup(name=NAME)": "setup.py: no Python project name",
            "F\tpyproject.toml\t[tool.poetry]": "pyproject.toml: no Python project name"}
        for line, reason in gaps.items():
            released = self.released("bytecodealliance/wasmtime", line)
            self.assertEqual(released["unparsed"], [reason], line)
            self.assertEqual(released["crates"][-1], "wasmtime-wizer", line)  # what could be read is kept
        self.assertEqual(self.released("bytecodealliance/wasmtime", "J\ttests/fixture/package.json\t\tunparsed",
                                       "F\tpyproject.toml\t[tool.black]")["unparsed"], [])  # a test fixture; no project
        partial = released_names("x/mono", 'F\tpackage.json\t{"workspaces": ["packages/*", 42]}\n'
                                           "J\tpackages/alpha/package.json\t@x/alpha\t\nE\tdone")
        self.assertEqual((partial["names"], partial["unparsed"]), (["@x/alpha", "alpha", "mono"], [
            'package.json: no valid JSON object with a "workspaces" list of globs']))  # discovery keeps what it can
        pnpm = released_names("x/mono", "F\tpnpm-workspace.yaml\tpackages:\nF\tpnpm-workspace.yaml\t  - packages/*\n"
                                        "F\tpnpm-workspace.yaml\t  - a: b\nJ\tpackages/alpha/package.json\t@x/alpha\t\n"
                                        "E\tdone")
        self.assertEqual((pnpm["names"], pnpm["unparsed"]), (["@x/alpha", "alpha", "mono"], [
            "pnpm-workspace.yaml: packages entry '- a: b' not understood"]))

    def test_package_roots_name_published_packages(self):
        otel, gomod = self.released("open-telemetry/opentelemetry-collector"), "/root/go/pkg/mod/"
        pdata_zip = (gomod + "cache/download/go.opentelemetry.io/collector/pdata/@v/v1.51.0.zip!/go.opentelemetry.io/"
                     "collector/pdata@v1.51.0/internal/json/iterator.go")
        # package_root names these as published, so the per-package rule finds the repo's own Go module zips too
        self.assertEqual(package_root(pdata_zip),
                         (pdata_zip.split("!/")[0] + "!/", "go.opentelemetry.io/collector/pdata"))
        self.assertEqual([row["name"] for row in package_rows({pdata_zip: [(1, "x")]}, {"pdata/x.go": ["x"]}, otel,
                                                              minimum=1)], ["go.opentelemetry.io/collector/pdata"])
        self.assertEqual(package_root(gomod + "github.com/!burnt!sushi/toml@v1.3.2/decode.go"),
                         (gomod + "github.com/!burnt!sushi/toml@v1.3.2/", "github.com/BurntSushi/toml"))
        self.assertEqual(package_root("/usr/lib/python3/site-packages/aiohttp-3.9.5.dist-info/METADATA"),
                         ("/usr/lib/python3/site-packages/aiohttp-3.9.5.dist-info/", "aiohttp"))
        self.assertEqual(package_root("/root/.bun/install/cache/@scope/pkg@1.0.0@@@1/index.js"),
                         ("/root/.bun/install/cache/@scope/pkg@1.0.0@@@1/", "@scope/pkg"))
        self.assertEqual(package_root("/root/.cache/uv/archive-v0/x1/yarl/_url.py"),
                         ("/root/.cache/uv/archive-v0/x1/yarl/", "yarl"))
        crate = "/root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/wasmtime-wizer-38.0.0/src/instrument.rs"
        wasmtime = self.released("bytecodealliance/wasmtime")
        self.assertEqual([row["name"] for row in package_rows({crate: [(1, "x")]}, {"a.rs": ["x"]}, wasmtime,
                                                              minimum=1)], ["wasmtime-wizer"])  # an own crate


def patch(path, *lines, header="", start=1):
    """A one-hunk diff of `path` (git's hunk header text `header`); lines: "+added" or " context"."""
    added = sum(1 for line in lines if line.startswith("+"))
    return "\n".join([f"diff --git a/{path} b/{path}", f"--- a/{path}", f"+++ b/{path}",
                      f"@@ -{start},{len(lines) - added} +{start},{len(lines)} @@{' ' + header if header else ''}",
                      *lines, ""])


# v4's frozen copy of c9f84ee0's reading: its name in task_builder -> its name in c9f84ee0 (the installed v2)
V2_FROZEN = {"_py_scan_v2": "_py_scan", "_code_hunks_v2": "_code_hunks", "_declared_names_v2": "declared_names",
             "_HunkV2": "_Hunk"}


def reading_digest(source: str, names: dict) -> str:
    """md5 of the source (a module's text) of the module-level definitions `names` (name -> the name to give it) and
    of every module-level name they use, directly or not (imports aside), each named as `names` names it."""
    import ast
    tree, lines, nodes = ast.parse(source), source.splitlines(keepends=True), {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            nodes[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                nodes.update((name.id, node) for name in ast.walk(target) if isinstance(name, ast.Name))
    used, todo = set(), list(names)
    while todo:
        name = todo.pop()
        if name in nodes and name not in used:
            used.add(name)
            todo += [node.id for node in ast.walk(nodes[name]) if isinstance(node, ast.Name)]
    rename = re.compile(r"(?<![\w$])(" + "|".join(map(re.escape, names)) + r")(?![\w$])")
    parts = []
    for name in sorted(used, key=lambda name: names.get(name, name)):
        node = nodes[name]
        text = "".join(lines[node.lineno - 1 - len(getattr(node, "decorator_list", [])):node.end_lineno])
        parts.append(f"{names.get(name, name)}\n{rename.sub(lambda match: names[match.group(1)], text)}")
    return hashlib.md5("\n".join(parts).encode()).hexdigest()


# official1-fastapi litestar cand-c03 (mask task): its solution.patch's last two hunks of litestar/handlers/base.py,
# verbatim; both start in a docstring that their first context lines close (the 22:33 parser's inversion)
LITESTAR_C03_HUNKS = (
    '@@ -416,9 +425,27 @@ class BaseRouteHandler:',
    '         Returns:',
    '             An optional :class:`DTO type <.dto.base_dto.AbstractDTO>`',
    '         """',
    '-        data_dto = None if self._dto is Empty else self._dto',
    '-        if data_dto is not None and self.parsed_data_field:',
    '-            data_dto.create_for_field_definition(self.parsed_data_field, self.handler_id)',
    '+        data_dto: type[AbstractDTO] | None = None',
    '+        if (_data_dto := self._dto) is not Empty:',
    '+            data_dto = _data_dto',
    '+        elif self.parsed_data_field and (',
    '+            plugin_for_data_type := next(',
    '+                (',
    '+                    plugin',
    '+                    for plugin in app.plugins.serialization',
    '+                    if self.parsed_data_field.match_predicate_recursively(plugin.supports_type)',
    '+                ),',
    '+                None,',
    '+            )',
    '+        ):',
    '+            data_dto = plugin_for_data_type.create_dto_for_type(self.parsed_data_field)',
    '+',
    '+        if self.parsed_data_field and data_dto:',
    '+            data_dto.create_for_field_definition(',
    '+                field_definition=self.parsed_data_field,',
    '+                handler_id=self.handler_id,',
    '+            )',
    '+',
    '         return data_dto',
    ' ',
    '     @property',
    '@@ -439,11 +466,32 @@ class BaseRouteHandler:',
    '         Returns:',
    '             An optional :class:`DTO type <.dto.base_dto.AbstractDTO>`',
    '         """',
    '-        return_dto = None if self._return_dto is Empty else self._return_dto',
    '-        if return_dto is not None and return_dto.is_supported_model_type_field(self.parsed_return_field):',
    '-            return_dto.create_for_field_definition(self.parsed_return_field, self.handler_id)',
    '-            return return_dto',
    '-        return None',
    '+        if (_return_dto := self._return_dto) is not Empty:',
    '+            return_dto: type[AbstractDTO] | None = _return_dto',
    '+        elif data_dto is not None and data_dto.is_supported_model_type_field(self.parsed_return_field):',
    '+            return_dto = data_dto',
    '+        elif plugin_for_return_type := next(',
    '+            (',
    '+                plugin',
    '+                for plugin in app.plugins.serialization',
    '+                if self.parsed_return_field.match_predicate_recursively(plugin.supports_type)',
    '+            ),',
    '+            None,',
    '+        ):',
    '+            return_dto = plugin_for_return_type.create_dto_for_type(self.parsed_return_field)',
    '+        else:',
    '+            return_dto = data_dto',
    '+',
    '+        if return_dto and return_dto.is_supported_model_type_field(self.parsed_return_field):',
    '+            return_dto.create_for_field_definition(',
    '+                field_definition=self.parsed_return_field,',
    '+                handler_id=self.handler_id,',
    '+            )',
    '+            resolved_return_dto = return_dto',
    '+        else:',
    '+            resolved_return_dto = None',
    '+',
    '+        return resolved_return_dto',
    ' ',
    '     async def authorize_connection(self, connection: ASGIConnection) -> None:',
    '         """Ensure the connection is authorized by running all the route guards in scope."""',
)


class DeclaredIdentifiers(unittest.TestCase):
    """Identifier evidence counts only for names the solution DECLARES (main's decision, 2026-10-20, after the
    third-party package exemptions of 19:57 and 21:12 were refuted): official1-wasmi c01-r1b4 died at its gate because
    wasm-encoder 0.244.0/0.262.0 in the cargo registry held 3/6 identifiers of the new crates/wizer/src/instrument.rs,
    the wasmparser/wasm-encoder API it calls. A copy of a solution that declares the matched names is a problem
    wherever it lies (node_modules/<third-party>, bundles, vendored single files); line and literal hits are
    unchanged. The 22:33 parser was refuted (a hunk whose context closes a docstring read its code as docstring and
    its next docstring as code; JS object-literal functions, destructuring, Rust tuple lets, parameters and targets
    were left out): Python docstrings are read from every start state, and parameters, targets, destructured and
    pattern bindings and object-literal members count, as does a name the hunk does not show to be a reference.
    v3 (the verifier of v2, 2026-10-21): the /* */ state is carried over a hunk's lines (JS/TS, Go, Rust, the C
    family: a member after a multi-line JSDoc block was lost), f-string replacement fields nest (PEP 701: one f-string
    nesting its own quote around words wiped a Python hunk), and TS signatures whose generic header or return type
    spans lines, keyword-named and optional methods, and `infer X` declare their names.
    v4 (main's decision after the verifier refuted v3, 2026-10-21): declared_names is c9f84ee0's declared names
    (frozen in the module) | the current reading's over v3's and c9f84ee0's readings, so it holds every name v2
    declares; a comment the hunk opens and leaves open blanks nothing after it; after_type is bounded by tokens, not
    12 lines. assertDeclares checks the current reading alone too, so the union hides no regression of it."""
    REGISTRY = "/root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/"
    CRATES = "/root/.cargo/registry/cache/index.crates.io-1949cf8c6b5b557f/"
    INFO, WIZER = "crates/wizer/src/info.rs", "crates/wizer/src/instrument.rs"
    WIZER_PATCH = patch(INFO, "     pub(crate) defined_global_exports: Option<Vec<(u32, String)>>,",
                        "+    pub(crate) data_drop_globals: BTreeMap<u32, (u32, String)>,", " }",
                        header="pub struct ModuleContext<'a> {", start=56) + patch(
        WIZER, " pub(crate) fn instrument(module: &mut ModuleContext<'_>) -> Vec<u8> {",
        "+    let mut globals_added = false;",
        "+    let mut exports_added = false;",
        "+    let reader = wasmparser::BinaryReader::new(section.data, 0);",
        "+    RoundtripReencoder.parse_global_section(&mut globals, wasmparser::GlobalSectionReader::new(reader)?)?;",
        "+    for body in wasmparser::CodeSectionReader::new(reader).unwrap() {",
        "+        let mut function = RoundtripReencoder.new_function_with_parsed_locals(&body)?;",
        "+        let mut ops = body.get_operators_reader().unwrap();",
        "+        if let Some((index, _)) = module.data_drop_globals.get(&data_index) {",
        "+            function.instruction(&wasm_encoder::Instruction::GlobalSet(*index));",
        "+        }",
        "+    }",
        "+    if !globals_added && !exports_added {",
        "+        encoder.section(&globals);",
        "+    }",
        " }", start=60)
    API = ["CodeSectionReader", "GlobalSectionReader", "get_operators_reader", "new_function_with_parsed_locals",
           "parse_global_section"]

    class Box:
        def __init__(self, outputs):
            self.outputs, self.uploads, self.calls = outputs, {}, 0

        async def put_text(self, text, remote):
            self.uploads[remote] = text

        async def sh(self, command, *, cwd="/app", timeout=1800, check=True):
            self.calls += 1
            return types.SimpleNamespace(stdout=self.outputs[self.calls - 1], exit_code=0, timed_out=False)

    def scan(self, repo, solution, manifest, discover, scan):
        box = self.Box(["\n".join(manifest + ["E\tdone"]), "\n".join(discover + ["E\tdone"]),
                        "\n".join(scan + ["E\tdone"])])
        return box, asyncio.run(released_copies(box, repo, solution))

    def assertDeclares(self, source, declared, references=(), v2_declares=()):
        """Every name of `declared` is declared in `source` (a patch), none of `references`. v4: by declared_names and
        by the current reading alone (_code_hunks_v3's readings parsed by _Hunk: v3 with v4's fixes), so that the union
        with c9f84ee0's reading hides no regression of the current one. `v2_declares`: the names of `references` that
        c9f84ee0's reading declares (it reads comment text as code); declared_names keeps them by construction."""
        current = task_builder._declared_in(task_builder._code_hunks_v3(source))
        self.assertEqual(([name for name in declared if name not in current],
                          [name for name in references if name in current]), ([], []),
                         f"the current reading declares {sorted(current)}")
        found = task_builder.declared_names(source)
        self.assertEqual(([name for name in declared if name not in found],
                          sorted(name for name in references if name in found),
                          [name for name in v2_declares if name not in task_builder._declared_names_v2(source)]),
                         ([], sorted(v2_declares), []), sorted(found))

    # ------------------------------------------------------------------------------------------------- Rust
    def test_rust_declarations(self):
        source = patch("src/segments.rs",
                       "     pending_segment_drops: Vec<u32>,",
                       "+    retained_segment_count: usize,",
                       "+    pub(crate) segment_lifetimes: BTreeMap<u32, SegmentLifetime>,",
                       " }",
                       "+pub enum SegmentLifetime {",
                       "+    RetainedSegment,",
                       "+    DroppedSegment {",
                       "+        dropped_at_offset: u32,",
                       "+    },",
                       "+    PartiallyCopied(u32),",
                       "+}",
                       "+const MAX_TRACKED_SEGMENTS: usize = 64;",
                       "+static mut SEGMENT_COUNTER: u32 = 0;",
                       "+pub(crate) struct SegmentBuilder<'a> {",
                       "+    segment_offset: u32,",
                       "+    retained_bytes: &'a [u8],",
                       "+}",
                       "+mod segment_tracking;",
                       "+macro_rules! track_segment_drop { () => {} }",
                       "+trait SegmentVisitor { type VisitOutput; }",
                       "+impl<'a> SegmentVisitor for SegmentBuilder<'a> {",
                       "+    fn visit_segment_drop(&mut self, segment_index: u32) -> Result<()> {",
                       "+        if let wasmparser::Operator::DataDrop { data_index } = self.current_operator {",
                       "+            let global_index = u32::try_from(self.globals.len()).unwrap();",
                       "+            RoundtripReencoder.parse_global_section(&mut globals, reader)?;",
                       "+            let global_type = wasm_encoder::GlobalType {",
                       "+                val_type: wasm_encoder::ValType::I32,",
                       "+                shared_flag: false,",
                       "+            };",
                       "+        }",
                       "+    }",
                       "+}", header="pub struct SegmentTracker {")
        self.assertDeclares(source, [
            "retained_segment_count", "segment_lifetimes", "SegmentLifetime", "RetainedSegment", "DroppedSegment",
            "dropped_at_offset", "PartiallyCopied", "MAX_TRACKED_SEGMENTS", "SEGMENT_COUNTER", "SegmentBuilder",
            "segment_offset", "retained_bytes", "segment_tracking", "track_segment_drop", "SegmentVisitor",
            "VisitOutput", "visit_segment_drop", "global_index", "global_type",
            "segment_index", "data_index"],  # a parameter; a binding (`DataDrop { data_index }` binds data_index)
            ["pending_segment_drops", "wasmparser", "Operator", "DataDrop", "current_operator", "try_from",
             "RoundtripReencoder", "parse_global_section", "GlobalType",  # context, paths, calls, a type
             "val_type", "shared_flag"])  # the fields of an external struct literal: its type declares them
        # the struct the hunk starts in is not shown (git's header names another block): `name: Type,` counts
        self.assertDeclares(source.replace("pub struct SegmentTracker {", "impl Tracker {"),
                            ["retained_segment_count", "segment_lifetimes"])

    def test_rust_patterns_closures_and_unshown_blocks(self):
        self.assertDeclares(patch(
            "crates/wasmi/src/engine/coredump.rs",
            "+    let (inner_entries, inner_count) = match &inner {",
            "+    let (local_types, num_locals) = if let FuncEntity::Wasm(wasm_func) = func_entity {",
            "+    if let Some((_, old_source)) = self.entries.iter_mut().find(|(entry_name, _)| *entry_name == label) {",
            "+    while let Some(stacked_item) = pending_stack.pop() {}",
            "+    for (segment_number, (_, segment_name)) in &module.data_drop_globals {}",
            "+    let PanelLayout { margin_size, padding: inner_padding } = layout;",
            "+    let doubled_values = items.iter().map(move |element_value| element_value * 2);",
            "+    match outcome_value {",
            "+        Ok(successful_result) if successful_result > 0 => {}",
            "+        Err(SnapshotError::Io { source: io_failure, .. }) => {}",
            "+    }"),
            ["inner_entries", "inner_count", "local_types", "num_locals", "wasm_func", "old_source", "entry_name",
             "stacked_item", "segment_number", "segment_name", "margin_size", "inner_padding", "doubled_values",
             "element_value", "successful_result", "io_failure"],
            ["FuncEntity", "func_entity", "iter_mut", "pending_stack", "data_drop_globals", "PanelLayout", "padding",
             "outcome_value", "SnapshotError", "source"])
        # a block whose opening line the hunk does not show: fields and variants count, match arms are patterns
        self.assertDeclares(patch("src/info.rs", "+    dropped_data_segments: Vec<u32>,", "+    RetainedSegment,",
                                  "+    PartiallyCopied(u32),", header="impl ModuleContext {"),
                            ["dropped_data_segments", "RetainedSegment", "PartiallyCopied"])
        self.assertDeclares(patch("crates/wizer/src/parse.rs",
                                  "             ExportSection(exports) => export_section(&mut module, exports)?,",
                                  "+            DataSection(section_data) => {",
                                  "+                for (index, data) in section_data.into_iter().enumerate() {}",
                                  "+            }",
                                  "             End { .. } => break,", "         }",
                                  header="pub(crate) fn parse_with<'a>("),
                            ["section_data"], ["DataSection", "ExportSection", "End"])  # wasmparser's variants

    # --------------------------------------------------------------------------------------------------- Go
    def test_go_declarations(self):
        source = patch("pkg/vm/vm.go",
                       " \tlocals []object.Object",
                       "+\tstepCounter int",
                       "+\tpendingSlices, finishedSlices []int",
                       "+\t*embeddedFrameBase",
                       "+\tframeMetadata map[string]string `json:\"frameMetadata\"`",
                       " }", header="type Frame struct {") + patch(
            "pkg/vm/opcodes.go", " \tOpLoad Opcode = iota", "+\tOpSliceStep", "+\tOpSliceRangeEnd = 99", " )",
            header="const (") + patch(
            "pkg/vm/slices.go",
            "+package slicestepping",
            "+type SliceStepper interface {",
            "+\tStepForward(count int) error",
            "+\tResetStepperState()",
            "+\tio.Closer",
            "+}",
            "+const maxSliceSteps = 64",
            "+var defaultStepper, fallbackStepper SliceStepper",
            "+func (vm *VirtualMachine) applySliceStep(node ast.Node) (object.Object, error) {",
            "+\tstepValue, convertErr := strconv.Atoi(node.Literal())",
            "+\tif remainderValue := stepValue % 2; remainderValue != 0 {",
            "+\t\tvm.evaluateExpression(node, environment)",
            "+\t}",
            "+\treturn object.NewSlice(sliceWindow{windowStart: stepValue}), convertErr",
            "+}",
            "+func computeSliceBounds[T any](length T) T {",
            "+\treturn length",
            "+}")
        self.assertDeclares(source, [
            "stepCounter", "pendingSlices", "finishedSlices", "frameMetadata", "OpSliceStep", "OpSliceRangeEnd",
            "slicestepping", "SliceStepper", "StepForward", "ResetStepperState", "maxSliceSteps", "defaultStepper",
            "fallbackStepper", "applySliceStep", "stepValue", "convertErr", "remainderValue", "computeSliceBounds"],
            ["embeddedFrameBase", "Closer", "VirtualMachine", "Atoi", "Literal", "evaluateExpression", "environment",
             "NewSlice", "sliceWindow", "windowStart"])  # types, calls, arguments, a struct literal's field
        self.assertDeclares(patch(
            "pkg/collector/expand.go",
            "+func expandEnvironmentVariables(rawConfiguration map[string]any) (expandedConfig map[string]any, err "
            "error) {",
            "+\texpanderOptions := ExpanderSettings{RecursionLimit: 10, AllowUndefined: false}",
            "+\tfor configurationKey, configurationValue := range rawConfiguration {}",
            "+\tgo func(shutdownSignal chan struct{}) {}(stopChannel)",
            "+\tswitch concreteValue := rawValue.(type) {",
            "+\t}",
            "+\treturn expandedConfig, nil",
            "+}",
            "+func (c *quantityFunc) CallTypeFunc(argumentName string, args []influxql.DataType) (influxql.DataType, "
            "error) {",
            "+}"),
            ["expandEnvironmentVariables", "rawConfiguration", "expandedConfig", "expanderOptions", "configurationKey",
             "configurationValue", "shutdownSignal", "concreteValue", "CallTypeFunc", "argumentName"],
            ["ExpanderSettings", "RecursionLimit", "AllowUndefined", "stopChannel", "rawValue", "influxql",
             "DataType"])  # the package of a result's type is no result name
        # a var group and a struct whose opening lines the hunk does not show
        self.assertDeclares(patch("cmd/prometheus/main.go",
                                  "+\tconfigRollbackSuccess = prometheus.NewGauge(prometheus.GaugeOpts{",
                                  "+\t\tNamespace: \"prometheus\",", "+\t})", "+\trecursionDepthLimit int"),
                            ["configRollbackSuccess", "recursionDepthLimit"], ["NewGauge", "GaugeOpts", "Namespace"])

    # ----------------------------------------------------------------------------------------------- Python
    def test_python_declarations(self):
        source = patch("saq/queue/snapshots.py",
                       "+SNAPSHOT_RETENTION = 25",
                       "+class SnapshotRegistry(BaseRegistry):",
                       "+    max_snapshots: int = 10",
                       "+    snapshot_label: str",
                       "+    def __init__(self, capture_limit=None):",
                       "+        self.captured_snapshots = []",
                       "+        self._capture_limit: int = capture_limit or SNAPSHOT_RETENTION",
                       "+    async def capture_snapshot(self, queue_state):",
                       "+        snapshot_items, skipped_items = self.partition_items(queue_state.list_jobs())",
                       "+        if (matching_count := len(snapshot_items)) > self.max_snapshots:",
                       "+            self.evict_oldest_snapshots(keep_newest=matching_count)",
                       "+        response = await self.redis_client.execute_command(",
                       "+            command_name=COMMAND_NAME,",
                       "+            timeout_seconds=30,",
                       "+        )",
                       "+        result_payload = {",
                       '+            "jobs": snapshot_items,',
                       "+            requested_keys: True,",
                       "+        }",
                       "+        registry.default_label = self.snapshot_label",
                       "+        return self.serialize_snapshot(result_payload, include_metadata=True)",
                       "+type SnapshotMapping = dict[str, int]")
        self.assertDeclares(source, [
            "SNAPSHOT_RETENTION", "SnapshotRegistry", "max_snapshots", "snapshot_label", "captured_snapshots",
            "_capture_limit", "capture_snapshot", "snapshot_items", "skipped_items", "matching_count", "response",
            "result_payload", "SnapshotMapping", "capture_limit", "queue_state"],  # (parameters)
            ["BaseRegistry", "partition_items", "list_jobs", "evict_oldest_snapshots", "keep_newest", "redis_client",
             "execute_command", "command_name", "timeout_seconds", "requested_keys", "default_label",
             "serialize_snapshot", "include_metadata"])  # bases, calls, keyword arguments, dict keys, other objects
        kwargs = patch("saq/worker.py", "+            retry_delay_seconds=backoff,", "+            jitter_seconds=0)",
                       header="def schedule(self, job):")
        self.assertEqual(task_builder.declared_names(kwargs), set())  # a call's arguments, its "(" before the hunk
        params = patch("saq/worker.py", "+        retry_delay_seconds: float = 1.0,", "+        jitter_seconds=0,",
                       "     ) -> None:", header="class Worker:")
        self.assertDeclares(params, ["retry_delay_seconds", "jitter_seconds"])  # a signature: `) -> None:` closes it
        self.assertDeclares(patch("pkg/conf.py", "+        connection_timeout_seconds=5,",
                                  "+        pool_recycle_seconds=30,"),
                            ["connection_timeout_seconds", "pool_recycle_seconds"])  # nothing shows a call: counted
        self.assertDeclares(patch("pkg/x.py", "+options = dict(retry_limit_value=3, backoff_factor=2)",
                                  "+class SessionRegistry:", "+    pass", "+SessionRegistry.default_capacity = 10",
                                  "+external_object.external_attribute = 3"),
                            ["options", "retry_limit_value", "backoff_factor", "SessionRegistry", "default_capacity"],
                            ["external_attribute"])  # dict(...) keys; a member of a class the solution declares

    def test_python_targets(self):
        self.assertDeclares(patch(
            "pkg/report.py",
            "+@register_handler(priority_level=3)",
            "+def render_failure_report(results, *pending_requests, include_skipped=False, **extra_options):",
            "+    for failing_testcase, (case_index, case_label) in enumerate(results):",
            "+        with open_report(failing_testcase) as report_stream, other_context() as second_context:",
            "+            first_part, *remaining_parts = failing_testcase.split()",
            "+    try:",
            "+        pass",
            "+    except LookupError as lookup_failure:",
            "+        handler_function = lambda event_payload, *extra_arguments: event_payload",
            "+    collected = [transform(record_value) for record_value in results if record_value]",
            "+    global module_level_counter",
            "+    module_level_counter += 1",
            "+    match incoming_event:",
            "+        case {\"kind\": event_kind_value, **remaining_fields}:",
            "+            pass",
            "+        case ResizeEvent(width=new_width_value) | [first_coordinate, *other_coordinates]:",
            "+            pass"),
            ["render_failure_report", "pending_requests", "include_skipped", "extra_options", "failing_testcase",
             "case_index", "case_label", "report_stream", "second_context", "first_part", "remaining_parts",
             "lookup_failure", "handler_function", "event_payload", "extra_arguments", "collected", "record_value",
             "module_level_counter", "event_kind_value", "remaining_fields", "new_width_value", "first_coordinate",
             "other_coordinates"],
            ["register_handler", "priority_level", "open_report", "other_context", "LookupError", "transform",
             "incoming_event", "ResizeEvent"])

    LITESTAR_C03 = "\n".join(["diff --git a/litestar/handlers/base.py b/litestar/handlers/base.py",
                              "--- a/litestar/handlers/base.py", "+++ b/litestar/handlers/base.py",
                              *LITESTAR_C03_HUNKS, ""])

    def test_python_docstrings_by_every_start_state(self):
        """The 22:33 refutation: official1-fastapi litestar cand-c03's handlers/base.py, whose hunks start in a
        docstring (context: "Returns:", ..., a closing \"\"\"), lost every declaration of its restored bodies."""
        source = "litestar/handlers/base.py"
        self.assertDeclares(self.LITESTAR_C03, ["plugin_for_data_type", "plugin_for_return_type",
                                                "resolved_return_dto", "_data_dto", "_return_dto", "data_dto",
                                                "return_dto"], ["Returns", "optional", "AbstractDTO"])
        self.assertEqual(task_builder.solution_identifiers(self.LITESTAR_C03)[source], [
            "_data_dto", "plugin_for_data_type", "plugin_for_return_type", "resolved_return_dto"])
        fence = '"' * 3
        closes = "\n".join([  # the verifier's minimal repro: context closes a docstring, an added one follows
            "diff --git a/pkg/mod.py b/pkg/mod.py", "--- a/pkg/mod.py", "+++ b/pkg/mod.py",
            "@@ -10,6 +10,14 @@ def resolve(self):", "         Returns:", "             The resolved value.",
            f"         {fence}", "+        resolved_settings = self.load_settings()", "+        return resolved_settings",
            "+", "+    def fetch_remote_record(self):", f"+        {fence}Fetch it.", "+", "+        Raises:",
            "+            ExternalServiceError: when the remote call fails.", f"+        {fence}",
            '+        raise ExternalServiceError("remote call failed")', "", "     def other(self):", ""])
        self.assertEqual(task_builder.declared_names(closes), {"resolved_settings", "fetch_remote_record"})
        self.assertEqual(task_builder.solution_identifiers(closes),
                         {"pkg/mod.py": ["fetch_remote_record", "remote call failed", "resolved_settings"]})
        opens = "\n".join([  # context opens a docstring that the added lines close
            "diff --git a/pkg/mod.py b/pkg/mod.py", "--- a/pkg/mod.py", "+++ b/pkg/mod.py",
            "@@ -10,4 +10,11 @@ class Client:", "     def fetch_remote_record(self):",
            f"         {fence}Fetch the record from the remote service.", "", "+        Raises:",
            "+            ExternalServiceError: when the remote call fails.", "+            ExternalTimeout: Raised.",
            f"+        {fence}", "+        resolved_settings = self.load_settings()", "+        return resolved_settings",
            "+", "     def other(self):", ""])
        self.assertEqual(task_builder.declared_names(opens), {"resolved_settings"})
        self.assertEqual(task_builder.solution_identifiers(opens), {"pkg/mod.py": ["resolved_settings"]})
        self.assertNotIn("ExternalServiceError", task_builder._added_items(opens)["pkg/mod.py"][0])  # no code
        closed_by_context = "\n".join([  # an added docstring that a context fence closes, code after it
            "diff --git a/pkg/mod.py b/pkg/mod.py", "--- a/pkg/mod.py", "+++ b/pkg/mod.py",
            "@@ -10,3 +10,9 @@ class Client:", "+    def compute_digest(self):", f"+        {fence}Summary.", "+",
            "+        More text about the digest.", f"         {fence}", "+        digest_accumulator = 1",
            "+    def refresh_snapshot_cache(self):", "         return 1", ""])
        self.assertDeclares(closed_by_context, ["compute_digest", "digest_accumulator", "refresh_snapshot_cache"])
        inside = "\n".join([  # added lines inside a docstring that context opens and closes: no code
            "diff --git a/pkg/mod.py b/pkg/mod.py", "--- a/pkg/mod.py", "+++ b/pkg/mod.py",
            "@@ -10,4 +10,6 @@ class Client:", f"         {fence}Fetch the record.", "",
            "+        Raises:", "+            ExternalServiceError: when the remote call fails.", f"         {fence}",
            "         return 1", ""])
        self.assertEqual((task_builder.declared_names(inside), task_builder.solution_identifiers(inside)),
                         (set(), {}))
        prose = patch("pkg/mod.py", "         Args:", "             retry_count: The number of retries.",
                      "+            ExternalServiceError: when the remote call fails.")  # no fence: prose only
        self.assertEqual(task_builder.declared_names(prose), set())
        attribute = patch("pkg/conf.py", "     timeout_seconds: int = 30", f"     {fence}", "+    Timeout in seconds.",
                          f"+    {fence}", "+    retries_allowed: int = 3")  # an attribute docstring
        self.assertEqual(task_builder.declared_names(attribute), {"retries_allowed"})
        strings = patch("pkg/sql.py", f"+QUERY_TEMPLATE = {fence}", "+SELECT identifier FROM snapshots",
                        f"+{fence}.strip()", f"+marker = '{fence}'", "+snapshot_identifier = 3",
                        '+formatted_message = f"part one {value} " f"part two"')
        self.assertDeclares(strings, ["QUERY_TEMPLATE", "snapshot_identifier", "formatted_message"],
                            ["SELECT", "identifier", "snapshots"])

    def test_python_readings_match_the_tokenizer(self):
        """The guarantee behind the readings, on standard-library modules: _py_scan, run over a file from code,
        starts a line inside a multi-line string exactly where the tokenizer does, and _py_prose calls no code
        line prose, so the true reading of a hunk is never refuted (2026-10-21, the stdlib, site-packages and
        litestar: 18,880 files, 0 disagreements, 0 code lines taken for prose)."""
        import inspect
        import io
        import tokenize
        starts = {getattr(tokenize, name, -1) for name in ("FSTRING_START", "TSTRING_START")}
        ends = {getattr(tokenize, name, -1) for name in ("FSTRING_END", "TSTRING_END")}
        layout = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                  tokenize.ENDMARKER, tokenize.STRING} | starts | ends | {
            getattr(tokenize, name, -1) for name in ("FSTRING_MIDDLE", "TSTRING_MIDDLE")}
        root, lines_seen = Path(inspect.__file__).parent, 0
        for name in ("inspect", "typing", "dataclasses", "argparse", "pathlib/__init__", "subprocess", "textwrap",
                     "asyncio/base_events", "json/decoder", "email/message", "http/client", "unittest/case",
                     "logging/__init__", "string/__init__", "re/_parser", "difflib", "pydoc", "doctest", "enum",
                     "functools", "string", "pathlib"):
            path = root / f"{name}.py"
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            inside, code, opened = set(), set(), []
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if token.type in starts:
                    opened.append(token.start[0])
                elif token.type in ends:
                    inside.update(range((opened.pop() if opened else token.start[0]) + 1, token.end[0] + 1))
                elif token.type == tokenize.STRING:
                    inside.update(range(token.start[0] + 1, token.end[0] + 1))
                elif token.type not in layout and not opened:
                    code.add(token.start[0])
            quote = None
            for number, line in enumerate(text.split("\n"), start=1):
                self.assertEqual(quote is not None, number in inside, f"{path}:{number}: {line}")
                scanned, quote = task_builder._py_scan(line, quote)
                if number in code:
                    self.assertFalse(task_builder._py_prose(scanned), f"{path}:{number}: {line}")
                lines_seen += 1
        self.assertGreater(lines_seen, 20000)

    PEP701 = "\n".join([  # Python 3.12+ f-strings: a replacement field may hold strings of the f-string's own quote
        "def render_summary_line(record_entry, width, items, entries):",
        '    label_text = f"{record_entry["display name"]}"',
        '    missing_text = f"{"Not found" if record_entry is None else record_entry}"',
        '    nested_text = f"{f"{label_text} {missing_text}"} trailing words"',
        '    aligned_text = f"{record_entry["display name"]!s:>{width}} and {"<" if width else ">"} words"',
        '    joined_text = f"{"\\n".join(items)} and more words"',
        '    escaped_text = f"{{literal words}} {label_text}"',
        "    quoted_text = f'{record_entry['display name']} and {'Not found' if record_entry is None else 0}'",
        '    multi_field = f"{',
        '        ", ".join(items)',
        '    } trailing words"',
        '    report_text = f"""',
        '{", ".join(',
        "    entry.name  # a comment in a field",
        "    for entry in entries",
        ')}',
        'done with "quoted words"',
        '"""',
        "    summary_value = compute_summary(record_entry, f\"{record_entry[\"display name\"]}\")",
        "    return label_text, summary_value",
        ""])

    def test_python_pep701_fstrings(self):
        """The verifier of v2: one f-string that nests its own quote around words (f"{d["display name"]}",
        f"{"Not found" if x is None else x}") made the true code reading look like prose, and the whole hunk
        declared nothing. _py_scan nests replacement fields: the f-string is one value, as the tokenizer reads it."""
        self.assertEqual(task_builder._py_scan('    label_text = f"{record_entry["display name"]}"', None),
                         ("    label_text =  0 ", None))
        self.assertEqual(task_builder._py_scan('x = f"{"Not found" if x is None else x}" + y', None),
                         ("x =  0  + y", None))
        self.assertEqual(task_builder._py_scan('x = rf"{"a b"}\\d" if f"{d["k"]!r:>{w}}" else 1', None),
                         ("x =  0  if  0  else 1", None))
        code, state = task_builder._py_scan('    multi_field = f"{', None)  # a field that spans lines (3.12+)
        self.assertEqual((code, state is not None), ("    multi_field = ", True))
        code, state = task_builder._py_scan('        ", ".join(items)', state)
        self.assertEqual((code, state is not None), ("", True))
        self.assertEqual(task_builder._py_scan('    } trailing words"', state), (" 0 ", None))
        self.assertEqual(task_builder._py_scan("x = 'it''s'", None), ("x =  0  0 ", None))  # plain strings as before
        self.assertEqual(task_builder._py_scan('doc = """Open', None), ("doc = ", '"""'))
        names = ["render_summary_line", "record_entry", "label_text", "missing_text", "nested_text", "aligned_text",
                 "joined_text", "escaped_text", "quoted_text", "multi_field", "report_text", "summary_value"]
        source = patch("pkg/render.py", *["+" + line for line in self.PEP701.splitlines()])
        self.assertDeclares(source, names, ["compute_summary", "display", "Found", "trailing", "quoted"])
        self.assertTrue(set(names) <= set(task_builder.solution_identifiers(source)["pkg/render.py"]))
        for form in ('f"{record_entry["display name"]}"', 'f"{"Not found" if record_entry is None else record_entry}"'):
            alone = patch("pkg/render.py", "+def render_summary_line(record_entry):", f"+    label_text = {form}",
                          "+    summary_value = compute_summary(record_entry)", "+    return label_text, summary_value")
            self.assertDeclares(alone, ["render_summary_line", "record_entry", "label_text", "summary_value"],
                                ["compute_summary"])  # (the 21:5x stage declared nothing in either hunk)
        context = patch("pkg/render.py", '     label_text = f"{record_entry["display name"]}"',
                        "+    summary_value = compute_summary(record_entry)", "+    return summary_value",
                        header="def render_summary_line(record_entry):")  # the f-string in context only
        self.assertDeclares(context, ["summary_value"], ["compute_summary"])

    def test_python_pep701_readings_match_the_tokenizer(self):
        """test_python_readings_match_the_tokenizer's guarantee on PEP 701 f-strings (the tokenizer of 3.12+)."""
        import io
        import tokenize
        if sys.version_info < (3, 12):
            self.skipTest("PEP 701 needs Python 3.12")
        starts = {getattr(tokenize, name, -1) for name in ("FSTRING_START", "TSTRING_START")}
        ends = {getattr(tokenize, name, -1) for name in ("FSTRING_END", "TSTRING_END")}
        layout = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                  tokenize.ENDMARKER, tokenize.STRING} | starts | ends | {
            getattr(tokenize, name, -1) for name in ("FSTRING_MIDDLE", "TSTRING_MIDDLE")}
        inside, code, opened = set(), set(), []
        for token in tokenize.generate_tokens(io.StringIO(self.PEP701).readline):
            if token.type in starts:
                opened.append(token.start[0])
            elif token.type in ends:
                inside.update(range(opened.pop() + 1, token.end[0] + 1))
            elif token.type == tokenize.STRING:
                inside.update(range(token.start[0] + 1, token.end[0] + 1))
            elif token.type not in layout and not opened:
                code.add(token.start[0])
        quote = None
        for number, line in enumerate(self.PEP701.split("\n"), start=1):
            self.assertEqual(quote is not None, number in inside, f"{number}: {line}")
            scanned, quote = task_builder._py_scan(line, quote)
            if number in code:
                self.assertFalse(task_builder._py_prose(scanned), f"{number}: {line}")
        self.assertIsNone(quote)
        self.assertGreater(len(inside), 6)

    # ----------------------------------------------------------------------------------------- JS / TS
    def test_js_ts_declarations(self):
        source = patch("packages/jest-worker/src/base/ExitTracker.ts",
                       "+export interface WorkerExitOptions {",
                       "+  forceExitDelay: number;",
                       "+  gracefulTimeout?: number",
                       "+  onWorkerExit(workerId: number): void;",
                       "+}",
                       "+export type ExitSummary = {",
                       "+  forcedExits: number",
                       "+};",
                       "+enum ExitReason {",
                       "+  GracefulExit = 0,",
                       "+  ForcedExit,",
                       "+}",
                       "+export default class WorkerExitTracker extends EventEmitter {",
                       "+  private readonly _pendingExits = new Map<number, Promise<void>>();",
                       "+  exitAttempts = 0",
                       "+  static createTracker(options: WorkerExitOptions): WorkerExitTracker {",
                       "+    return new WorkerExitTracker(options);",
                       "+  }",
                       "+  constructor(private readonly trackerOptions: WorkerExitOptions) {",
                       "+    super();",
                       "+    this.lastExitCode = null;",
                       "+  }",
                       "+  async waitForWorkerExit(",
                       "+    workerId: number,",
                       "+  ): Promise<boolean> {",
                       "+    const [firstAttempt, ...laterAttempts] = this.collectAttempts(workerId);",
                       "+    const { exitTimeoutMillis } = this.trackerOptions;",
                       "+    for (const exitPromise of this._pendingExits.values()) {",
                       "+      await exitPromise;",
                       "+    }",
                       "+    process.nextTick(() => this.emitWorkerExit(workerId));",
                       "+    this.scheduleRetries(",
                       "+      laterAttempts,",
                       "+    );",
                       "+    return this.onWorkerExit(workerId);",
                       "+  }",
                       "+}",
                       "+export function summarizeExits(tracker: WorkerExitTracker): ExitSummary {",
                       "+  return {forcedExits: tracker.exitAttempts, unmatchedExits: 0};",
                       "+}",
                       "+const reporterHooks = {",
                       "+  onRunComplete(results) {",
                       "+    reportExitSummary(results);",
                       "+  },",
                       "+  testTimeoutMillis: 5000,",
                       "+};",
                       "+module.exports.registerTracker = registerTracker;")
        self.assertDeclares(source, [
            "WorkerExitOptions", "forceExitDelay", "gracefulTimeout", "onWorkerExit", "ExitSummary", "forcedExits",
            "ExitReason", "GracefulExit", "ForcedExit", "WorkerExitTracker", "_pendingExits", "exitAttempts",
            "createTracker", "trackerOptions", "lastExitCode", "waitForWorkerExit", "firstAttempt", "laterAttempts",
            "exitPromise", "summarizeExits", "reporterHooks", "onRunComplete", "registerTracker",
            "workerId", "exitTimeoutMillis",  # a parameter; a destructured binding
            "unmatchedExits", "testTimeoutMillis"],  # object-literal keys define the solution's properties
            ["EventEmitter", "Promise", "collectAttempts", "nextTick", "emitWorkerExit", "scheduleRetries",
             "reportExitSummary"])  # a base class, a type, calls
        field = patch("src/Reporter.ts", "   private _bailed = false;", "+  pendingResults = new Set<string>();",
                      "+  private readonly _reportedFailures: number;", header="export default class Reporter {")
        self.assertEqual(task_builder.declared_names(field), {"pendingResults", "_reportedFailures"})  # a class the
        # hunk does not open: its members count with or without modifiers

    def test_js_object_literal_functions(self):
        self.assertDeclares(patch(
            "lib/worker_farm.js",
            "+module.exports = {",
            "+  createWorkerFarm: function (farmOptions) {",
            "+    return { pendingRequests: new Map() };",
            "+  },",
            "+  terminateWorkerFarm: async (workerFarm) => {",
            "+    await workerFarm.shutdownGracefully();",
            "+  },",
            "+  gracefulExitHandler: () => {},",
            "+  normalizeWorkerName: workerName => workerName.trim(),",
            "+  restartWorkerPool: async function* (poolOptions) {},",
            "+  defaultConcurrency: computeDefaultConcurrency(),",
            "+};",
            "+const testemClient = {",
            "+  handleAbortTests: function() {},",
            "+  onTestComplete(testResult) { return summarizeFailures(testResult); },",
            "+};"),
            ["createWorkerFarm", "farmOptions", "pendingRequests", "terminateWorkerFarm", "workerFarm",
             "gracefulExitHandler", "normalizeWorkerName", "workerName", "restartWorkerPool", "poolOptions",
             "defaultConcurrency", "testemClient", "handleAbortTests", "onTestComplete", "testResult"],
            ["shutdownGracefully", "computeDefaultConcurrency", "summarizeFailures"])
        self.assertDeclares(patch("packages/jest-worker/src/base/BaseWorkerPool.ts",
                                  "+      gracefulExit: () => {", "+        this.forceExitAll();", "+      },",
                                  header="  async end(): Promise<PoolExitResult> {"),
                            ["gracefulExit"], ["forceExitAll"])  # inside an object the hunk does not open

    def test_js_destructuring(self):
        self.assertDeclares(patch(
            "src/run.js",
            "+const { workerExitPromises, forceExitTimeout } = computeExitPlan(options);",
            "+let { retryAttempts = 3, backoffMultiplier } = settings;",
            "+const {channelId: incomingChannelId, type, ...remainingPayload} = message.ava;",
            "+const { masks, notMasks: { inner: deepBinding } } = query;",
            "+const [firstWorker, , [nestedWorker], ...otherWorkers] = workers;",
            "+const { [computedKeyName]: computedBinding } = term;",
            "+for (const { component, fieldsOfOwner } of owners) {}",
            "+function handleMessage({ requestPayload, replyChannel: responseChannel }, [firstArgument]) {}",
            "+const relayEvent = ({ eventName, ...eventDetails }) => eventName;",
            "+try {} catch ({ message: errorMessage }) {}",
            "+  const {",
            "+    assertCount,",
            "+    snapshotCount: totalSnapshots,",
            "+  } = await test.runAttempt(attemptTitle);"),
            ["workerExitPromises", "forceExitTimeout", "retryAttempts", "backoffMultiplier", "incomingChannelId",
             "remainingPayload", "deepBinding", "firstWorker", "nestedWorker", "otherWorkers", "computedBinding",
             "component", "fieldsOfOwner", "handleMessage", "requestPayload", "responseChannel", "firstArgument",
             "relayEvent", "eventName", "eventDetails", "errorMessage", "assertCount", "totalSnapshots"],
            ["computeExitPlan", "channelId", "notMasks", "computedKeyName", "replyChannel", "runAttempt",
             "attemptTitle", "snapshotCount"])  # the properties a destructuring reads, calls, computed keys

    def test_js_ts_members_after_multiline_jsdoc(self):
        """The verifier of v2: a class method, class field, interface member or object-literal member right after a
        multi-line JSDoc block (`*/` on its own line) was not declared, nor were the method's parameters: the " * ..."
        and " */" lines were read as code. The /* */ state is carried over the hunk's lines (added and context)."""
        source = patch("packages/jest-worker/src/base/BaseWorkerPool.ts",
                       "+export default class BaseWorkerPool {",
                       "+  private readonly _workers: Array<WorkerInterface>;",
                       "+",
                       "+  /**",
                       "+   * Returns the worker whose id is given.",
                       "+   *",
                       "+   * @param workerIdentifier the worker's id ({@link ExternalPoolScheduler})",
                       "+   */",
                       "+  getWorkerById(workerIdentifier: number): WorkerInterface {",
                       "+    return this._workers[workerIdentifier];",
                       "+  }",
                       "+",
                       "+  /**",
                       "+   * Promises of the workers that are exiting.",
                       "+   */",
                       "+  private readonly workerExitPromises: Array<Promise<void>> = [];",
                       "+  /**",
                       "+   * Kills the workers that do not exit in time.",
                       "+   */",
                       "+  forceExitTimeout = 500;",
                       "+",
                       "+  /**",
                       "+   * Ends all workers.",
                       "+   */",
                       "+  async endAllWorkers(forceExitDelay: number): Promise<PoolExitResult> {",
                       "+    return awaitWorkerExits(this.workerExitPromises, forceExitDelay);",
                       "+  }",
                       "+  /*",
                       "+    legacyWorkerOption: true,",
                       "+  */",
                       "+}",
                       "+export interface WorkerPoolOptions {",
                       "+  /**",
                       "+   * How many workers the pool keeps.",
                       "+   */",
                       "+  numberOfWorkers: number;",
                       "+  /**",
                       "+   * Called with every message a worker sends.",
                       "+   */",
                       "+  onCustomMessage?(workerMessage: unknown): void;",
                       "+}",
                       "+const reporterHooks = {",
                       "+  /**",
                       "+   * Called when a test completes.",
                       "+   */",
                       "+  onTestComplete(testResult) {",
                       "+    return summarizeFailures(testResult);",
                       "+  },",
                       "+  /**",
                       "+   * Called at the end of the run.",
                       "+   */",
                       "+  onRunComplete: async aggregatedResults => {},",
                       "+};")
        self.assertDeclares(source, [
            "BaseWorkerPool", "_workers", "getWorkerById", "workerIdentifier", "workerExitPromises",
            "forceExitTimeout", "endAllWorkers", "forceExitDelay", "WorkerPoolOptions", "numberOfWorkers",
            "onCustomMessage", "workerMessage", "reporterHooks", "onTestComplete", "testResult", "onRunComplete",
            "aggregatedResults"],
            ["ExternalPoolScheduler", "legacyWorkerOption", "Returns", "awaitWorkerExits", "summarizeFailures"],
            v2_declares=["legacyWorkerOption"])  # (c9f84ee0 reads the commented-out member as code; v4 keeps it)
        # a hunk that starts inside a JSDoc block (official1 jest-runtime CjsLoader.ts handleCjsParseError)
        inside = patch("packages/jest-runtime/src/internals/CjsLoader.ts",
                       "    * @param modulePath the path of the module whose parse failed",
                       "    * @returns never: it throws",
                       "    */",
                       "+  private handleCjsParseError(",
                       "+    parseError: SyntaxError,",
                       "+    modulePath: string,",
                       "+    moduleName: string | undefined,",
                       "+    isRequireActual: boolean,",
                       "+  ): never {",
                       "+    throw enrichParseError(parseError, modulePath);",
                       "+  }",
                       "   private _shouldMockCjs(", header="export default class CjsLoader {")
        self.assertDeclares(inside, ["handleCjsParseError", "parseError", "modulePath", "moduleName",
                                     "isRequireActual"], ["enrichParseError", "_shouldMockCjs"])
        self.assertEqual(task_builder.solution_identifiers(inside), {
            "packages/jest-runtime/src/internals/CjsLoader.ts": [
                "handleCjsParseError", "isRequireActual", "moduleName", "modulePath", "parseError"]})
        # official1-sqlformatter's @sqltools/formatter Indentation.ts: methods after JSDoc blocks, the class not shown
        methods = patch("packages/formatter/src/core/Indentation.ts",
                        "   }", "+", "+  /**", "+   * Increases indentation by one top-level indent.", "+   */",
                        "+  increaseTopLevel() {", "+    this.indentTypes.push(INDENT_TYPE_TOP_LEVEL);", "+  }",
                        "+", "+  /**", "+   * Decreases indentation by one block-level indent.",
                        "+   * If there are no block-level indents, does nothing.", "+   */",
                        "+  decreaseBlockLevel() {", "+    this.indentTypes.pop();", "+  }",
                        header="export default class Indentation {")
        self.assertDeclares(methods, ["increaseTopLevel", "decreaseBlockLevel"], ["INDENT_TYPE_TOP_LEVEL"])

    def test_block_comments_ignore_strings_and_regexes(self):
        """A /* or */ inside a string, a JS template literal or regex literal, a Go raw string or a Rust string
        neither opens nor closes a comment, also over lines (a glob, a cron spec); Go, Rust and the C family carry
        the comment state too."""
        self.assertDeclares(patch(
            "src/schedule.ts",
            "+const scheduleUsage = `",
            "+  Patterns: src/**/*.js and lib/*",
            "+  Schedule: */5 * * * * ${describeSchedule(\"/* here */\")}",
            "+`;",
            "+const blankLinePattern = /^\\s*$/;",
            "+const everythingPattern = /.*/g, slashedPattern = /\\/*/;",
            "+export function parseWorkerArguments(rawArguments: string): string[] {",
            "+  return rawArguments.split(/\\s*,\\s*/);",
            "+}",
            "+const dividedValue = totalDuration / 2; /* a comment */ const remainderValue = 1;"),
            ["scheduleUsage", "blankLinePattern", "everythingPattern", "slashedPattern", "parseWorkerArguments",
             "rawArguments", "dividedValue", "remainderValue"], ["describeSchedule"])  # (a template's text lines
        # read as before, line by line: only no comment opens or closes in them)
        self.assertDeclares(patch(
            "pkg/scheduler/run.go",
            "+/*",
            "+Package scheduler runs recurring jobs.",
            "+*/",
            "+package scheduler",
            "+const cronUsage = `",
            "+  */5 * * * *  runs every five minutes",
            "+  /* not a comment",
            "+`",
            "+/* deprecated: legacyScheduleInterval int */",
            "+type JobScheduler struct {",
            "+\t/*",
            "+\t   pendingJobQueue holds the jobs not yet started.",
            "+\t*/",
            "+\tpendingJobQueue []Job",
            "+}",
            "+func scheduleRecurringJob(jobIdentifier string, cronExpression string) error {",
            "+\treturn nil",
            "+}"),
            ["scheduler", "cronUsage", "JobScheduler", "pendingJobQueue", "scheduleRecurringJob", "jobIdentifier",
             "cronExpression"], ["legacyScheduleInterval", "Package"],
            v2_declares=["Package"])  # (c9f84ee0 reads the doc comment's line as code; v4 keeps it)
        self.assertDeclares(patch(
            "crates/wizer/src/snapshot.rs",
            "+pub struct SnapshotTracker {",
            "+    /*",
            "+     * Segments dropped before the snapshot.",
            "+     */",
            "+    dropped_segment_indices: Vec<u32>,",
            "+}",
            "+const USAGE_TEXT: &str = \"usage: wizer [options] <input.wasm>",
            "+    /* the options */ and globs like src/*.rs\";",
            "+fn describe_snapshot<'a>(snapshot_label: &'a str) -> String {",
            "+    let raw_template = r#\"/* not a comment\"#;",
            "+    /* outer /* nested */ still_commented_name */",
            "+    format!(\"{raw_template}{snapshot_label}\")",
            "+}"),
            ["SnapshotTracker", "dropped_segment_indices", "USAGE_TEXT", "describe_snapshot", "snapshot_label",
             "raw_template"], ["still_commented_name"])
        self.assertDeclares(patch(
            "src/main/java/org/example/RetryPolicy.java",
            "+public class RetryPolicy {",
            "+    /**",
            "+     * The attempts made so far.",
            "+     */",
            "+    private int attemptsMade;",
            "+    private static final String RETRY_HELP = \"\"\"",
            "+        Retries keep their files under logs/* for a day.",
            "+        \"\"\";",
            "+    public boolean shouldRetry(int attemptNumber) { return attemptNumber < 3; }",
            "+}"),
            ["RetryPolicy", "attemptsMade", "shouldRetry", "attemptNumber"], ["Retries"])  # (a text block's lines,
        # its opener's too, are blanked by the "+"-line docstring state as before; its /* opens no comment)
        # a reading that a */ outside any comment refutes: the hunk starts inside one (Go struct fields)
        self.assertDeclares(patch("pkg/vm/frame.go", "    The frame's registers.", "  */",
                                  "+\tregisterSnapshot []object.Object", "+\tstepCounter int", " }",
                                  header="type Frame struct {"), ["registerSnapshot", "stepCounter"])
        # the reading from inside a comment where the hunk's first */ is inside a string (official1-arcane argo-cd
        # cand-c04 settings.go: `"*/*"`): refuted, so its broken brackets declare no call
        self.assertDeclares(patch("util/settings/settings.go",
                                  " func (mgr *SettingsManager) GetIgnoreResourceUpdatesOverrides() (Overrides, error) {",
                                  "+\tif compareOptions.IgnoreDifferencesOnResourceUpdates {",
                                  "+\t\tlog.Info(\"Using diffing customizations\")", "+\t}",
                                  "+\taddIgnoreDiffItemOverrideToGK(resourceOverrides, \"*/*\", \"/metadata/generation\")",
                                  "+\taddIgnoreDiffItemOverrideToGK(resourceOverrides, \"*/*\", \"/metadata/managedFields\")",
                                  "+\treturn resourceOverrides, nil", " }",
                                  header="func (mgr *SettingsManager) GetEnabledSourceTypes() (map[string]bool, error) {"),
                            [], ["addIgnoreDiffItemOverrideToGK", "compareOptions", "resourceOverrides"])
        # a hunk that opens its own comment does not start inside one: the call after it stays a call
        self.assertDeclares(patch("pkg/scheduler/drain.go",
                                  "+func runPendingScheduler(pendingJobQueue chan Job) {",
                                  "+\t/**",
                                  "+\t * Drain the queue before the scheduler stops.",
                                  "+\t */",
                                  "+\tdrainPendingJobQueue(pendingJobQueue)",
                                  "+}"),
                            ["runPendingScheduler", "pendingJobQueue"], ["drainPendingJobQueue", "scheduler"])
        # prose in a Go doc comment is no field to the current reading (official1-anko gpython cand-c05,
        # official1-promlabels openGemini cand-c02: the installed v2 declared "Implements" and "AnalyzeTagSets" from
        # these comments, so v4, which holds every v2 declaration, declares them too)
        self.assertDeclares(patch(
            "compile/compile.go", "+/*", "+Implements the with statement from PEP 343.", "+", "+AnalyzeTagSets",
            "+eg, inputTags:", "+*/", "+func (c *compiler) withStatement(node *ast.With, position int) {", "+}"),
            ["withStatement", "position"], ["Implements", "statement", "AnalyzeTagSets", "inputTags"],
            v2_declares=["Implements", "AnalyzeTagSets"])

    def test_ts_signatures_spanning_lines(self):
        """The verifier of v2: a function whose generic header spans lines (`>(` on its own line), a method named
        after a keyword and a return type that spans lines (`): If<`) lost their parameters; `infer X` bindings are
        declared too (official1-dynamodb's zsa cand-c01-r1b3 bind.ts, its analysis patch's lazy schema)."""
        self.assertDeclares(patch(
            "packages/zsa/src/bind.ts",
            "+export function bindServerActionInput<",
            "+  T extends TAnyZodSafeFunctionHandler,",
            "+  const B extends Partial<RawInput<T>>,",
            "+>(",
            "+  serverAction: T & ObjectAction<T>,",
            "+  boundInput: B & {",
            "+    [K in keyof B]: K extends keyof RawInput<T> ? RawInput<T>[K] : never",
            "+  } & Record<Exclude<keyof B, keyof RawInput<T>>, never>",
            "+): BoundAction<T, B> {",
            "+  const snapshot = { ...boundInput }",
            "+  return snapshot",
            "+}",
            "+type InputPair<T> = T extends [infer FirstInput, infer SecondInput] ? [FirstInput, SecondInput] : never"),
            ["bindServerActionInput", "serverAction", "boundInput", "snapshot", "InputPair", "FirstInput",
             "SecondInput"],
            ["TAnyZodSafeFunctionHandler", "ObjectAction", "RawInput", "BoundAction", "Record", "Exclude"])
        self.assertDeclares(patch(
            "src/schema/lazy/schema.ts",
            "+export class LazySchema_<PROPS extends SchemaProps = SchemaProps> {",
            "+  default(",
            "+    nextDefault: ValueOrGetter<PROPS>",
            "+  ): If<",
            "+    PROPS['key'],",
            "+    LazySchema_<Overwrite<PROPS, KeyDefaultProps>>,",
            "+    LazySchema_<Overwrite<PROPS, PutDefaultProps>>",
            "+  > {",
            "+    return ifThenElse(this.props.key, nextDefault)",
            "+  }",
            "+",
            "+  validate(",
            "+    nextValidator: Validator<PROPS>",
            "+  ): If<",
            "+    PROPS['key'],",
            "+    LazySchema_<Overwrite<PROPS, KeyValidatorProps>>",
            "+  > {",
            "+    return ifThenElse(this.props.key, nextValidator)",
            "+  }",
            "+",
            "+  delete(recordIdentifier: string): boolean {",
            "+    return removeStoredRecord(recordIdentifier)",
            "+  }",
            "+}",
            "+interface RecordStore {",
            "+  delete(storedRecordKey: string): boolean;",
            "+}"),
            ["LazySchema_", "nextDefault", "nextValidator", "recordIdentifier", "RecordStore", "storedRecordKey"],
            ["ValueOrGetter", "KeyDefaultProps", "ifThenElse", "removeStoredRecord", "Validator"])
        # a return type over lines elsewhere: an arrow function, a method of a class the hunk does not show, a
        # parameter list the hunk starts inside
        self.assertDeclares(patch("src/schema/load.ts",
                                  "+export const loadSchemaDefinition = async (schemaIdentifier: string): Promise<",
                                  "+  LoadedSchema<SchemaProps>",
                                  "+> => {",
                                  "+  return fetchSchema(schemaIdentifier)",
                                  "+}",
                                  "+  resolveStoredRecord(",
                                  "+    storedRecordKey: string,",
                                  "+  ): If<",
                                  "+    RecordProps['key'],",
                                  "+    StoredRecord",
                                  "+  > {",
                                  "+    return lookupRecord(storedRecordKey)",
                                  "+  }", header="export class RecordStore {"),
                            ["loadSchemaDefinition", "schemaIdentifier", "resolveStoredRecord", "storedRecordKey"],
                            ["LoadedSchema", "fetchSchema", "lookupRecord", "StoredRecord"])
        self.assertDeclares(patch("src/schema/lazy/schema.ts",
                                  "+    nextDefaultValue: ValueOrGetter<PROPS>,",
                                  "+  ): If<",
                                  "+    PROPS['key'],",
                                  "+    LazySchema_<PROPS>",
                                  "+  > {",
                                  "+    return ifThenElse(this.props.key, nextDefaultValue)",
                                  header="export class LazySchema_<PROPS> {"),
                            ["nextDefaultValue"], ["ValueOrGetter", "ifThenElse"])
        # an object type is part of the return type, and a "," ends a member's type (as before)
        self.assertDeclares(patch("src/core/WindowLayout.ts",
                                  "+const computeBounds = (startOffset: number): { endOffset: number; words: string[] } => {",
                                  "+  return { endOffset: startOffset, words: [] }",
                                  "+}",
                                  "+export const separateMessageFromStack = (",
                                  "+  messageContent: string,",
                                  "+): {message: string; stack: string} => splitContent(messageContent)",
                                  "+type MemberPair = { firstMember(firstArgument: string): string, secondMember("
                                  "secondArgument: number): void }"),
                            ["computeBounds", "startOffset", "separateMessageFromStack", "messageContent",
                             "firstArgument", "secondArgument"], ["splitContent"])
        # `return (` over lines in a body that the hunk's header lines make look like a type (zsa's sonner.tsx: `type
        # ToasterProps = ...` above an arrow function, no semicolons): JSX attributes and calls stay undeclared
        self.assertDeclares(patch("examples/showcase/components/ui/sonner.tsx",
                                  "+type ToasterProps = React.ComponentProps<typeof Sonner>",
                                  "+",
                                  "+const Toaster = ({ ...props }: ToasterProps) => {",
                                  "+  return (",
                                  "+    <Sonner",
                                  "+      className=\"toaster group\"",
                                  "+      toastOptions={{ duration: 5 }}",
                                  "+      {...props}",
                                  "+    />",
                                  "+  )",
                                  "+}"),
                            ["ToasterProps", "Toaster"], ["className", "toastOptions"])
        # a statement is still no signature: a call after `return`, a parenthesized condition, a ternary
        self.assertDeclares(patch("src/run.ts", "+  if (isWorkerReady) {", "+    return (pendingResult)",
                                  "+  }", "+  const chosenValue = isWorkerReady ? computeValue(firstValue) : other;",
                                  header="  run() {"),
                            ["chosenValue"], ["isWorkerReady", "pendingResult", "computeValue", "firstValue"])

    def test_c_family_and_keyword_fallback_declarations(self):
        java = patch("src/main/java/org/example/RetryPolicy.java",
                     "+@Deprecated",
                     "+public final class RetryPolicy extends BasePolicy implements Serializable {",
                     "+    private static final int MAX_RETRY_ATTEMPTS = 5;",
                     "+    private final Map<String, Integer> attemptCounts = new HashMap<>();",
                     "+    @Override",
                     "+    public boolean shouldRetryRequest(Request request, int attemptNumber) {",
                     "+        int remainingAttempts = MAX_RETRY_ATTEMPTS - attemptNumber;",
                     "+        return backoffStrategy.computeDelay(remainingAttempts) > 0;",
                     "+    }",
                     "+    enum RetryOutcome { RETRY, ABORT }",
                     "+    record RetryDecision(boolean retry, long delayMillis) {}",
                     "+}")
        c = patch("src/retry.c",
                  "+typedef struct retry_state retry_state_t;",
                  "+struct retry_state {",
                  "+    int attempts_made;",
                  "+};",
                  "+static int compute_backoff_delay(const retry_state_t *state) {",
                  "+    unsigned long jitter_value = random_jitter(state->attempts_made);",
                  "+    return apply_backoff_cap(jitter_value);",
                  "+}")
        kotlin = patch("app/src/main/kotlin/RetryScheduler.kt",
                       "+class RetryScheduler(private val clock: Clock) {",
                       "+    val pendingRetries = mutableListOf<Retry>()",
                       "+    fun scheduleRetry(task: Task): Boolean {",
                       "+        var attemptsLeft = task.maxAttempts",
                       "+        return dispatcher.enqueueRetry(task, attemptsLeft)",
                       "+    }",
                       "+    fun String.toRetryKey() = this.lowercase()",
                       "+}")
        self.assertDeclares(java + c + kotlin, [
            "RetryPolicy", "MAX_RETRY_ATTEMPTS", "attemptCounts", "shouldRetryRequest", "remainingAttempts",
            "RetryOutcome", "RetryDecision", "retry_state_t", "retry_state", "attempts_made", "compute_backoff_delay",
            "jitter_value", "RetryScheduler", "pendingRetries", "scheduleRetry", "attemptsLeft", "toRetryKey",
            "attemptNumber"],  # a parameter
            ["BasePolicy", "Serializable", "HashMap", "Deprecated", "backoffStrategy", "computeDelay",
             "random_jitter", "apply_backoff_cap", "mutableListOf", "maxAttempts", "dispatcher", "enqueueRetry",
             "lowercase"])

    # -------------------------------------------------------------------- v4: monotonic over v2 (c9f84ee0)
    # The verifier of v3's real cases (repair/thirdparty-leak-20261020/verify-v3-adversarial): hunks that start inside
    # a multi-line string whose opening they do not show, holding a `/*`. argo-cd cmd/argocd/commands/admin/project.go,
    # a non-test source: the cobra Example raw string's `(action/*)`, with a RunE edit; yj main.go: the HelpMsg hunk
    # `[/*]` of official1-dasel's kimi attempt 14-2 model.patch (and 3 other model patches), two declarations added
    # after `var Version`. go/parser declares c, args, normalizedProjectGlob and bbcodeListItemKey,
    # decodeBBCodeDocument, sourceReader on added lines; c9f84ee0 found them, v3 found none.
    ARGO_CD = "\n".join([
        "diff --git a/cmd/argocd/commands/admin/project.go b/cmd/argocd/commands/admin/project.go",
        "--- a/cmd/argocd/commands/admin/project.go", "+++ b/cmd/argocd/commands/admin/project.go",
        "@@ -161,16 +161,16 @@ func NewUpdatePolicyRuleCommand() *cobra.Command {",
        "   # Remove policy that which manages running (action/*) from all roles which name matches *deployer* in all "
        "projects",
        "   argocd admin proj update-role-policy '*' remove override --role '*deployer*'",
        " `,",
        "-\t\tRun: func(c *cobra.Command, args []string) {",
        "+\t\tRunE: func(c *cobra.Command, args []string) error {",
        " \t\t\tctx := c.Context()",
        " ",
        " \t\t\tif len(args) != 3 {",
        "-\t\t\t\tc.HelpFunc()(c, args)",
        "-\t\t\t\tos.Exit(1)",
        "+\t\t\t\treturn fmt.Errorf(\"expected 3 arguments, got %d\", len(args))",
        " \t\t\t}",
        " \t\t\tprojectGlob := args[0]",
        " \t\t\tmodificationType := args[1]",
        " \t\t\taction := args[2]",
        "+\t\t\tnormalizedProjectGlob := strings.TrimSpace(projectGlob)",
        " ",
        " \t\t\tconfig, err := clientConfig.ClientConfig()",
        " \t\t\terrors.CheckError(err)", ""])
    YJ = "\n".join([
        "diff --git a/main.go b/main.go", "--- a/main.go", "+++ b/main.go", "@@ -35,9 +35,18 @@",
        " -h     Show this help message", " -v     Show version", " ",
        "+Supported BBCode dialect: [b], [i], [u], [quote=author], [url=href],",
        "+[code], [list] or [list=1|a] with [*] ... [/*] items, [br], and [[ for a",
        "+literal '['. Tag names are case-insensitive. BBCode maps to a tree of",
        "+ordered maps with #text for ordinary text, -type/-author/-href for tag",
        "+parameters, and item for list items.", "+", " `", " ", ' var Version = "0.0.0"', "+",
        '+var bbcodeListItemKey = "item"',
        "+func decodeBBCodeDocument(sourceReader io.Reader) (interface{}, error) { return nil, nil }", " ",
        " func main() {", " \tos.Exit(Run(os.Stdin, os.Stdout, os.Stderr, os.Args))", ""])
    # reading_digest(c9f84ee0a3b583d088d0ea7d5d37a182, its _py_scan, _code_hunks, declared_names, _Hunk): the four
    # definitions and the 23 module-level names they use (work/frozen_digest.py, 2026-10-21)
    V2_READING_MD5 = "5b8dd22c4f08d31f21e5f9072e439791"

    def assertHoldsV2(self, source):
        """v4 holds c9f84ee0's result on `source`: every name its declared_names declares, and every item of each code
        file's identifier evidence as c9f84ee0 computes it (solution_identifiers on its readings and declared names)."""
        self.assertEqual(sorted(task_builder._declared_names_v2(source) - task_builder.declared_names(source)), [])
        with mock.patch.object(task_builder, "_code_hunks", task_builder._code_hunks_v2), \
                mock.patch.object(task_builder, "declared_names", task_builder._declared_names_v2):
            v2 = task_builder.solution_identifiers(source)
        v4 = task_builder.solution_identifiers(source)
        self.assertEqual({path: sorted(set(items) - set(v4.get(path, []))) for path, items in v2.items()},
                         {path: [] for path in v2})

    def test_a_comment_the_hunk_does_not_close_blanks_nothing(self):
        """The verifier of v3: a `/*` inside a string whose opening the hunk does not show (a Go raw string, a JS
        template, JSX text) opened a comment that never closed, and v3 blanked every later line of the hunk, real
        declarations too. A comment that the hunk opens and leaves open blanks no line after its opener now: the
        current reading itself declares what c9f84ee0 declared (the union would hide a regression here), and a JSDoc
        block that closes in the hunk stays blank."""
        self.assertDeclares(self.ARGO_CD, ["c", "args", "normalizedProjectGlob"], ["TrimSpace", "Errorf"])
        self.assertIn("normalizedProjectGlob",
                      task_builder.solution_identifiers(self.ARGO_CD)["cmd/argocd/commands/admin/project.go"])
        self.assertDeclares(self.YJ, ["bbcodeListItemKey", "decodeBBCodeDocument", "sourceReader"], ["Reader"])
        (_, _, lines, variants), = task_builder._code_hunks_v3(self.YJ)
        version = [raw for _, raw in lines].index('var Version = "0.0.0"')  # read as its "+"-line code, not blank
        self.assertEqual([codes[version] for codes in variants],
                         [task_builder._plain_code("main.go", 'var Version = "0.0.0"')] * len(variants))
        self.assertDeclares(patch(
            "src/components/GlobHelp.tsx",
            "+export class GlobHelp extends Component {",
            "+  /**",
            "+   * Renders the hint below the glob field.",
            "+   */",
            "+  renderHint(hintLabel: string) {",
            "+    return <p className=\"hint\">{hintLabel} matches every src/*.ts file</p>",
            "+  }",
            "+  processRecords(inputItems: Item[], batchLimit: number) {",
            "+    const accumulatedResult = inputItems.slice(0, batchLimit)",
            "+    return accumulatedResult",
            "+  }",
            "+}"),
            ["GlobHelp", "renderHint", "hintLabel", "processRecords", "inputItems", "batchLimit", "accumulatedResult"],
            ["Renders", "Component", "slice"])  # (c9f84ee0 missed renderHint and hintLabel after the JSDoc)
        self.assertDeclares(patch(  # Rust: an unclosed nested comment
            "src/help.rs",
            "+    usage: lib/*/mod.rs and src/*",
            "+\"#;",
            "+pub fn render_usage_text(usage_width: usize) -> String {",
            "+    let wrapped_usage = wrap(USAGE, usage_width);",
            "+    wrapped_usage",
            "+}"),
            ["render_usage_text", "usage_width", "wrapped_usage"], ["wrap"])

    def test_return_types_over_any_number_of_lines(self):
        """The verifier of v3: after_type followed a return type over at most 12 lines, so an object-literal method
        whose `Promise<{...}>` return type was longer lost its name and parameters. It follows the type to its end
        over any number of lines, up to _TYPE_TOKENS of its tokens, a bracket inside it passed over whole as one (real
        return types read at most 47; the bound keeps a run of `f(x) : a < b` lines linear)."""
        fields = [f"+    accountField{number}: string" for number in range(14)]
        self.assertDeclares(patch(
            "src/api.ts", "+export const accountService = {",
            "+  async fetchAccountSummary(accountIdentifier: string, includeArchived: boolean): Promise<{", *fields,
            "+  }> {", "+    return loadSummary(accountIdentifier, includeArchived)", "+  },", "+}"),
            ["accountService", "fetchAccountSummary", "accountIdentifier", "includeArchived"], ["loadSummary"])
        long = patch(  # `Promise<` ends its line: c9f84ee0 (the closer's line only) and v3 (12 lines) both lost it
            "src/api.ts", "+export const accountService = {",
            "+  async fetchAccountSummary(accountIdentifier: string, includeArchived: boolean): Promise<", "+  {",
            *fields, "+  }", "+  > {", "+    return loadSummary(accountIdentifier, includeArchived)", "+  },", "+}")
        self.assertDeclares(long, ["fetchAccountSummary", "accountIdentifier", "includeArchived"], ["loadSummary"])
        self.assertFalse({"fetchAccountSummary", "accountIdentifier", "includeArchived"}
                         & task_builder._declared_names_v2(long))
        union = patch(  # 60 members, 120 of the type's tokens outside brackets: within _TYPE_TOKENS
            "src/icons.ts", "+export const iconService = {",
            "+  iconNameFor(iconCode: number, iconTheme: string): IconAlpha |",
            *[f"+    IconMember{number} |" for number in range(60)], "+    IconOmega {",
            "+    return lookupIcon(iconCode, iconTheme)", "+  },", "+}")
        self.assertDeclares(union, ["iconService", "iconNameFor", "iconCode", "iconTheme"], ["lookupIcon"])
        self.assertGreater(task_builder._TYPE_TOKENS, 120)
        self.assertDeclares(patch(  # an object type of 1,200 tokens: one token of the return type
            "src/api.ts", "+export const reportService = {",
            "+  async fetchReport(reportIdentifier: string): Promise<{",
            *[f"+    reportField{number}: string" for number in range(400)],
            "+  }> {", "+    return loadReport(reportIdentifier)", "+  },", "+}"),
            ["reportService", "fetchReport", "reportIdentifier"], ["loadReport"])
        self.assertDeclares(patch(  # a class method; a parameter list the hunk starts inside
            "src/store.ts", "+    recordIdentifier: string,", "+  ): Promise<", "+    Array<{",
            *[line.replace("+    ", "+      ") + "," for line in fields], "+    }>", "+  > {",
            "+    return this.loadAll(recordIdentifier)", "+  }", header="export class RecordStore {"),
            ["recordIdentifier"], ["loadAll"])

    def test_v4_holds_every_v2_declaration(self):
        """main's decision for v4: declared_names = c9f84ee0's declared names (_declared_names_v2, frozen) | what _Hunk
        declares in v3's readings and c9f84ee0's, and _added_items reads both, so no name v2 declares is lost and every
        file's evidence holds v2's. The routes by which the current reading alone still loses v2 names: a hunk that
        ends inside a return type (after_type sees no end), a `/*` in JSX text that a later JSDoc's `*/` closes (the
        code between is blanked), JSX text holding a lone `*/` (the reading from code is refuted; the one from inside
        a comment blanks the code before it), Go doc-comment prose (gpython cand-c05, openGemini cand-c02)."""
        records = ["processRecords", "inputItems", "batchLimit", "accumulatedResult"]
        cases = {
            "a return type past the hunk's end": (patch(
                "src/api.ts",
                "+  async fetchAccountSummary(accountIdentifier: string, includeArchived: boolean): Promise<{",
                "+    totalBalance: number", "+    openAccounts: string[]", "     closedAccounts: string[]",
                "     lastActivity: Date", header="export const accountService = {"),
                ["fetchAccountSummary", "accountIdentifier", "includeArchived"]),
            "a JSX glob that a later JSDoc closes": (patch(
                "src/Help.tsx", "+export function GlobHelp() {",
                "+  return <p className=\"hint\">Matches every src/*.ts file</p>", "+}",
                "+export function processRecords(inputItems: Item[], batchLimit: number) {",
                "+  const accumulatedResult = inputItems.slice(0, batchLimit)", "+  return accumulatedResult", "+}",
                "+/**", "+ * Formats one record.", "+ */", "+export function formatRecord(recordValue: Item) {",
                "+  return String(recordValue)", "+}"), records),
            "JSX text holding a lone */": (patch(
                "src/Help.tsx", "+export function processRecords(inputItems: Item[], batchLimit: number) {",
                "+  const accumulatedResult = inputItems.slice(0, batchLimit)", "+  return (", "+    <pre>",
                "+      */", "+    </pre>", "+  )", "+}"), records),
            "Go doc-comment prose": (patch(
                "compile/compile.go", "+/*", "+Implements the with statement from PEP 343.", "+", "+AnalyzeTagSets",
                "+*/", "+func (c *compiler) withStatement(node *ast.With, position int) {", "+}"),
                ["Implements", "AnalyzeTagSets"]),
        }
        for label, (source, names) in cases.items():
            with self.subTest(label):
                current = task_builder._declared_in(task_builder._code_hunks_v3(source))
                self.assertEqual(([name for name in names if name in current],
                                  [name for name in names if name not in task_builder._declared_names_v2(source)],
                                  [name for name in names if name not in task_builder.declared_names(source)]),
                                 ([], [], []))
                self.assertHoldsV2(source)
                path = re.search(r"^\+\+\+ b/(.+)$", source, re.M).group(1)
                self.assertTrue({name for name in names if len(name) >= 8}
                                <= set(task_builder.solution_identifiers(source)[path]))
        for source in (self.ARGO_CD, self.YJ, self.WIZER_PATCH, self.LITESTAR_C03):
            self.assertHoldsV2(source)

    def test_v4_holds_v2_on_corpus_patches(self):
        """The same on real recorded patches (read only; skipped where the runs are not): every 8th author
        solution.patch of the official1 runs in path order (a copy checked once), the two whose v3 evidence lost a v2
        item (gpython cand-c05 compile.go `Implements`, openGemini cand-c02 tag_array.go `AnalyzeTagSets`) and the yj
        model.patch whose HelpMsg hunk v3 blanked. work/monotonic_corpus.py checks every recorded patch, file by
        file."""
        runs = next((path for path in (Path(task_builder.__file__).resolve().parent / "runs",
                                       Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth/runs"))
                     if path.is_dir()), None)
        if runs is None:
            self.skipTest("no runs/ here")
        named = ["official1-anko/build/anko-default-function-arguments/go-python__gpython/cand-c05/author-0",
                 "official1-promlabels/build/prometheus-typed-label-sorting/openGemini__openGemini/cand-c02/author-0"]
        paths = [runs / name / "solution.patch" for name in named] + [
            runs / "official1-dasel/build/dasel-html-document-format/sclevine__yj/round-1/probe-bac27c9c/models/kimi"
                   "/attempts/14-2/attempt-1/model.patch"]
        seen, checked = set(), 0
        for path in [path for path in paths if path.is_file()] + sorted(
                runs.glob("official1-*/build/*/*/cand-*/author-*/solution.patch"))[::8]:
            text = path.read_text(errors="replace")
            if hashlib.md5(text.encode()).hexdigest() in seen:
                continue
            seen.add(hashlib.md5(text.encode()).hexdigest())
            with self.subTest(str(path.relative_to(runs))):
                self.assertHoldsV2(text)
                checked += 1
        self.assertGreater(checked, 20)
        lost = {"official1-anko": "Implements", "official1-promlabels": "AnalyzeTagSets"}
        for path in paths[:2]:
            if path.is_file():  # v3's reading drops these doc-comment words; v4 keeps them
                solution = path.read_text()
                found = {item for items in task_builder.solution_identifiers(solution).values() for item in items}
                self.assertIn(lost[path.relative_to(runs).parts[0]], found)

    def test_v2_reading_is_frozen(self):
        """v4's monotonicity rests on _declared_names_v2 computing what c9f84ee0 computes: its four definitions, named
        back, and the module-level names they use hold c9f84ee0's source. Changing either changes the v2 reading that
        v4 holds; change the current reading (_code_hunks_v3, _Hunk) instead."""
        self.assertEqual(reading_digest(Path(task_builder.__file__).read_text(), V2_FROZEN), self.V2_READING_MD5)
        self.assertIn("_declared_names_v2", task_builder.declared_names.__code__.co_names)  # (the union calls it)

    # ------------------------------------------------------------------------------- evidence and scans
    def test_identifier_evidence_is_declared_names_and_literals(self):
        self.assertEqual(task_builder.solution_identifiers(self.WIZER_PATCH), {
            self.INFO: ["data_drop_globals"], self.WIZER: ["data_drop_globals", "exports_added", "globals_added"]})
        referenced = task_builder.referenced_identifiers(self.WIZER_PATCH)[self.WIZER]
        self.assertTrue(set(self.API) <= set(referenced), referenced)
        literal = patch("src/messages.rs", '+    bail!("parse_global_section failed for the snapshot");',
                        '+    let label = "CodeSectionReader";')  # literals are evidence whatever they spell
        self.assertEqual(task_builder.solution_identifiers(literal),
                         {"src/messages.rs": ["CodeSectionReader", "parse_global_section failed for the snapshot"]})

    def wasmtime_scan(self, *lines, solution=None):
        """released_copies on the wasmi c01-r1b4 layout: the repo's own wasmtime-wizer .crate (unpacked as 1) and
        wasm-encoder in the registry; `lines` are extra ARCHIVE_SCAN output lines."""
        own = self.CRATES + "wasmtime-wizer-38.0.0.crate"
        return own, *self.scan("bytecodealliance/wasmtime", solution or self.WIZER_PATCH,
                               OwnPackageNames.MANIFESTS["bytecodealliance/wasmtime"],
                               ["X\tunzip", f"A\t4000\t{own}", f"A\t9000\t{self.CRATES}wasm-encoder-0.244.0.crate",
                                "R\t/root/.cargo/registry/src"], [f"U\t1\t{own}\t0", *lines])

    def test_wasm_encoder_api_names_are_no_leak(self):
        encoders = [self.REGISTRY + f"wasm-encoder-{version}/src/reencode.rs" for version in ("0.244.0", "0.262.0")]
        member = "/tmp/leak-archives/1/wasmtime-wizer-38.0.0/src/instrument.rs"
        api_hits = [f"I\t{path}:{1200 + 10 * i}:{51234 + 700 * i}:{name}" for path in encoders
                    for i, name in enumerate(self.API)]  # what a scan of the referenced names would print
        own, box, report = self.wasmtime_scan(*api_hits)
        self.assertEqual(box.uploads["/tmp/leak-names"], "wasmtime\n")  # discovery as before 19:57: no crate names
        self.assertEqual(box.uploads["/tmp/leak-select"], f"tar\t{own}\n")
        self.assertEqual(box.uploads["/tmp/leak-idents"], "data_drop_globals\nexports_added\nglobals_added\n")
        self.assertTrue(set(self.API) <= set(report["referenced"][self.WIZER]))  # listed, never scanned
        self.assertEqual(report["problems"], [])  # the API names are no evidence: c01-r1b4 passes
        own_hits = [f"I\t{member}:{41 + i}:{1200 + 90 * i}:{name}" for i, name in enumerate(
            ["globals_added", "exports_added", "data_drop_globals"])]
        _, _, report = self.wasmtime_scan(*api_hits, *own_hits)
        self.assertEqual(report["problems"], [  # a release of the repo's own crate holds the declared names
            f"{own}!/wasmtime-wizer-38.0.0/src/instrument.rs holds 3/3 distinctive identifiers of {self.WIZER} within "
            "60 lines"])
        self.assertEqual(dependency_leaks(report["problems"]), report["problems"])
        lines = task_builder.leak_tokens(self.WIZER_PATCH)[self.WIZER]
        self.assertGreaterEqual(len(lines), 8)
        _, _, report = self.wasmtime_scan(*api_hits, *[f"H\t{encoders[0]}:{300 + i}:{9000 + 80 * i}:{line}"
                                                       for i, line in enumerate(lines[:6])])
        self.assertEqual(report["problems"], [f"{encoders[0]} holds 6/{len(lines)} distinctive lines of {self.WIZER} "
                                              "within 60 lines"])  # a verbatim copy: a problem wherever it lies

    def test_declared_copies_are_problems_anywhere(self):
        worker = "packages/jest-worker/src/base/BaseWorkerPool.ts"
        solution = patch(worker, " export default class BaseWorkerPool {",
                         "+  async end(): Promise<PoolExitResult> {",
                         "+    const workerExitPromises = this._workers.map(async worker => {",
                         "+      const forceExitTimeout = setTimeout(() => worker.forceExit(), FORCE_EXIT_DELAY);",
                         "+      clearTimeout(forceExitTimeout);",
                         "+    });",
                         "+    const workerExits = await Promise.all(workerExitPromises);",
                         "+    return workerExits.reduce((result, forceExited) => result || forceExited, false);",
                         "+  }", " }", header="", start=100)
        self.assertEqual(task_builder.solution_identifiers(solution)[worker],
                         ["forceExitTimeout", "forceExited", "workerExitPromises", "workerExits"])  # (a parameter)
        places = ["/app/node_modules/p-limit/index.js",  # a third-party package under node_modules
                  "/usr/lib/node_modules/corepack/dist/lib/corepack.cjs",  # a bundle of other packages' code
                  "/root/.npm/_cacache https://registry.npmjs.org/left-pad/-/left-pad-1.3.0.tgz!/package/index.js"]
        declared = ["workerExitPromises", "forceExitTimeout", "workerExits"]
        scan = [f"I\t{path}:{10 + i}:{100 * i}:{name}" for path in places for i, name in enumerate(declared)]
        scan += [f"I\t/app/node_modules/lodash/lodash.js:{10 + i}:{100 * i}:{name}"  # references: never evidence
                 for i, name in enumerate(["FORCE_EXIT_DELAY", "forceExit", "_workers"])]
        box, report = self.scan("jestjs/jest", solution, OwnPackageNames.MANIFESTS["jestjs/jest"],
                                ["X\tunzip", "R\t/usr/lib/node_modules"], scan)
        self.assertEqual(box.uploads["/tmp/leak-idents"],
                         "forceExitTimeout\nforceExited\nworkerExitPromises\nworkerExits\n")
        self.assertEqual(report["problems"], sorted(f"{path} holds 3/4 distinctive identifiers of {worker} within "
                                                    "60 lines" for path in places))
        self.assertEqual(dependency_leaks(report["problems"]), report["problems"])
        six = "six.py"  # urllib3/packages/six.py: a single-file copy, no vendor segment
        python = patch(six, "+def ensure_binary_text(value, encoding='utf-8'):",
                       "+    converted_value = value.encode(encoding) if isinstance(value, str) else value",
                       "+    if not isinstance(converted_value, binary_type):",
                       "+        raise TypeError('not expecting type of the given value')",
                       "+    return converted_value")
        single = "/usr/lib/python3/dist-packages/urllib3/packages/six.py"
        box, report = self.scan("benjaminp/six", python, ["F\tsetup.py\tsetup(", "F\tsetup.py\t    name='six',"],
                                ["X\tunzip", "R\t/usr/lib/python3/dist-packages"],
                                [f"I\t{single}:{900 + i}:{40000 + 50 * i}:{name}" for i, name in enumerate(
                                    ["ensure_binary_text", "converted_value"])]
                                + [f"H\t{single}:905:40400:not expecting type of the given value"])
        self.assertEqual(report["problems"], [f"{single} holds 3/4 distinctive identifiers of {six} within 60 lines"])
        self.assertIn("encoding", task_builder.solution_identifiers(python)[six])  # (a parameter, not in the copy)

    def test_the_litestar_c03_copy_is_a_problem(self):
        """The verifier's end-to-end case: a release of litestar whose handlers/base.py re-wraps the distinctive
        lines (line evidence 4/20) still holds the restored bodies' declarations (pre-19:57 flagged it 3/3)."""
        source = "litestar/handlers/base.py"
        copy = "/usr/local/lib/python3.12/site-packages/litestar/handlers/base.py"
        hits = [f"I\t{copy}:{430 + 12 * i}:{18000 + 500 * i}:{name}" for i, name in enumerate(
            ["plugin_for_data_type", "plugin_for_return_type", "resolved_return_dto"])]
        box, report = self.scan("litestar-org/litestar", self.LITESTAR_C03,
                                ["F\tpyproject.toml\t[project]", 'F\tpyproject.toml\tname = "litestar"'],
                                ["X\tunzip", "R\t/usr/local/lib/python3.12/site-packages"],
                                ["P\tis not self", "P\t_data_dto", *hits])
        self.assertEqual(report["absent"][source], ["plugin_for_data_type", "plugin_for_return_type",
                                                    "resolved_return_dto"])
        window = task_builder.change_windows(self.LITESTAR_C03)[source]
        self.assertEqual(report["problems"], [f"{copy} holds 3/3 distinctive identifiers of {source} within {window} "
                                              "lines"])

    def test_line_and_literal_hits_stay_problems(self):
        snapshot = "crates/wizer/src/snapshot.rs"
        messages = ["data drop state must be an i32", "segment index out of bounds for the snapshot",
                    "cannot preserve a dropped passive segment"]
        solution = patch(snapshot, " fn f() {", *[f'+    bail!("{message}");' for message in messages])
        encoder = self.REGISTRY + "wasm-encoder-0.244.0/src/reencode.rs"
        _, report = self.scan("bytecodealliance/wasmtime", solution,
                              OwnPackageNames.MANIFESTS["bytecodealliance/wasmtime"],
                              ["X\tunzip", "R\t/root/.cargo/registry/src"],
                              [f"H\t{encoder}:{40 + i}:{700 + 90 * i}:{message}" for i, message in enumerate(messages)])
        self.assertEqual(report["problems"], [f"{encoder} holds 3/3 distinctive identifiers of {snapshot} within 60 "
                                              "lines"])  # string literals are verbatim text, wherever they lie

    def test_own_release_and_archives_outside_caches_stay_problems(self):
        tokenizer = "packages/formatter/src/core/Tokenizer.ts"
        solution = patch(tokenizer, "   tokenize(input, previousToken) {",
                         "+    const pipeMatch = this.getPipeOperatorToken(input, previousToken);",
                         "+    if (pipeMatch && this.options.dialect === 'bq') {",
                         "+      return this.createPipeToken(pipeMatch, TokenTypes.PIPE_OPERATOR);",
                         "+    }",
                         "+    const nextLineBreak = input.indexOf(NEWLINE, this.currentPosition);",
                         "+    this.currentPosition = nextLineBreak === -1 ? input.length : nextLineBreak;",
                         "+    tokens.push({ type: TokenTypes.PIPE_OPERATOR, value: pipeMatch[0] });",
                         "+    return this.tokenizeRemainder(input.slice(pipeMatch[0].length), tokens);",
                         "   }", start=90)
        lines = task_builder.leak_tokens(solution)[tokenizer]
        self.assertEqual(len(lines), 7)
        other = ("/usr/local/share/.cache/yarn/v6/npm-sql-formatter-15.0.0-0123456789abcdef0123456789abcdef01234567"
                 "-integrity/node_modules/sql-formatter/lib/core/Tokenizer.js")
        tarball = "/opt/dist/sql-formatter-15.0.0.tgz"
        hits = [(SQLTOOLS + "lib/core/Tokenizer.js", lines[:4]), (other, lines[:6]),
                ("/tmp/leak-archives/1/package/lib/core/Tokenizer.js", lines[:6])]
        _, report = self.scan("mtxr/vscode-sqltools", solution, OwnPackageNames.MANIFESTS["mtxr/vscode-sqltools"],
                              ["X\tunzip", f"A\t70000\t{tarball}", "R\t/usr/local/share/.cache"],
                              [f"U\t1\t{tarball}\t0"] + [f"H\t{path}:{10 + i}:{400 * i}:{line}" for path, found in hits
                                                         for i, line in enumerate(found)])
        self.assertEqual(report["problems"], [  # official1-sqlformatter c03's leak, a tarball outside the caches and
            # a verbatim copy inside another package (a fork published under another name): line hits all stay
            f"{tarball}!/package/lib/core/Tokenizer.js holds 6/7 distinctive lines of {tokenizer} within 60 lines",
            f"{SQLTOOLS}lib/core/Tokenizer.js holds 4/7 distinctive lines of {tokenizer} within 60 lines",
            f"{other} holds 6/7 distinctive lines of {tokenizer} within 60 lines"])
        self.assertEqual(dependency_leaks(report["problems"]), report["problems"])

    def test_leak_scan_gate_judges_declared_names_anywhere(self):
        worker = "packages/jest-worker/src/base/BaseWorkerPool.ts"
        solution = patch(worker, " export default class BaseWorkerPool {",
                         "+  async end(): Promise<PoolExitResult> {",
                         "+    const workerExitPromises = this._workers.map(async worker => {",
                         "+      const forceExitTimeout = setTimeout(() => worker.forceExit(), FORCE_EXIT_DELAY);",
                         "+      clearTimeout(forceExitTimeout);",
                         "+    });",
                         "+    const workerExits = await Promise.all(workerExitPromises);",
                         "+    return workerExits.reduce((result, forceExited) => result || forceExited, false);",
                         "+  }", " }", start=100)
        lines = task_builder.leak_tokens(solution)[worker]
        own = "/app/node_modules/metro/node_modules/jest-worker/build/base/BaseWorkerPool.js"
        other, api = "/app/node_modules/lodash/lodash.js", "/app/node_modules/p-limit/index.js"
        leak_scan = "\n".join(f"{path}:{100 + i}:{line}" for path in (own, other) for i, line in enumerate(lines[:4]))

        def run(manifest, names):
            scan = "\n".join(f"I\t{api}:{10 + i}:{100 * i}:{name}" for i, name in enumerate(names)) + "\nE\tdone"
            outputs = {task_builder.ARCHIVE_MANIFEST: "\n".join(manifest + ["E\tdone"]),
                       task_builder.ARCHIVE_DISCOVER: "E\tdone", task_builder.ARCHIVE_SCAN: scan}

            class Box:
                async def put_text(self, text, remote):
                    pass

                async def sh(self, command, *, cwd="/app", timeout=1800, check=True):
                    return types.SimpleNamespace(stdout=outputs.get(command, leak_scan), exit_code=0, timed_out=False)

            @contextlib.asynccontextmanager
            async def owned(snapshot, *, run, purpose):
                yield Box()

            logged = []
            build = types.SimpleNamespace(run="official1-testem", repo="jestjs/jest")
            with mock.patch.object(task_builder, "owned", owned), mock.patch.object(task_builder, "log", logged.append):
                return asyncio.run(task_builder.leak_scan(build, "snapshot:x", solution)), logged

        self.assertEqual(len(lines), 6)
        lines_found = [f"{other} holds 4/6 distinctive lines of {worker} within 60 lines",
                       f"{own} holds 4/6 distinctive lines of {worker} within 60 lines"]
        problems, logged = run(OwnPackageNames.MANIFESTS["jestjs/jest"], ["FORCE_EXIT_DELAY", "forceExit", "_workers"])
        self.assertEqual((problems, logged), (lines_found, []))  # line hits stay; the called API is no evidence
        self.assertEqual(dependency_leaks(problems), problems)
        broken = OwnPackageNames.MANIFESTS["jestjs/jest"] + ["F\tpnpm-workspace.yaml\tpackages: !!seq [a]"]
        problems, logged = run(broken, ["workerExitPromises", "forceExitTimeout", "workerExits"])
        self.assertEqual(problems, sorted(lines_found + [f"{api} holds 3/4 distinctive identifiers of {worker} within "
                                                         "60 lines"]))  # declared names inside a third-party package
        self.assertEqual(len(logged), 1)
        self.assertIn("leak scan (official1-testem): own package names may be incomplete: pnpm-workspace.yaml: "
                      "packages value", logged[0])


class VerifierFaults(unittest.TestCase):
    """A FAIL whose verifier sandbox broke is infra (official1-wasmi c05 kimi-r3: os error 5 + rustc SIGBUS, `[gate]
    build` failed, F2P 0/17, P2P 0/394)."""

    def attempt(self, logs: dict, grading="attempt-0") -> Path:
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory)
        (directory / "grading" / grading).mkdir(parents=True)
        for name, text in logs.items():
            (directory / "grading" / grading / name).write_text(text)
        return directory

    def record(self, attempt, status="fail", f2p="0/17", p2p="0/394", slot="kimi-r3"):
        return {"slot": slot, "model": slot.split("-")[0], "status": status, "reward": 0.0, "f2p": f2p, "p2p": p2p,
                "failing_tests": [], "agent_error": None, "verifier_error": None, "patch": "diff", "output": [],
                "attempt_dir": str(attempt) if attempt else None}

    def test_signatures_need_nothing_passing(self):
        broken = self.attempt({"run.log": "error: couldn't read `/root/.cargo/registry/src/x/lib.rs`: Input/output error "
                                          "(os error 5)\n(signal: 7, SIGBUS: access to undefined memory)\n",
                               "stderr.log": "/tests/test.sh: line 92: rm: command not found\n"})
        reason = rollouts.verifier_fault(self.record(broken))
        self.assertEqual(reason, "verifier infrastructure fault, not a grade: (os error 5), Input/output error, SIGBUS "
                                 "in attempt-0/{run.log} (F2P 0/17, P2P 0/394)")
        self.assertEqual(rollouts.verifier_fault(self.record(broken, p2p="3/394")), "")  # something was graded
        self.assertEqual(rollouts.verifier_fault(self.record(broken, status="pass")), "")
        self.assertEqual(rollouts.verifier_fault(self.record(None)), "")
        self.assertIn("(F2P None/None, P2P None/None)", rollouts.verifier_fault(self.record(broken, "fail", "None/None",
                                                                                            "None/None")))
        for text in ("read /app/x: input/output error", "write failed: No space left on device", "Bus error (core dumped)",
                     "Error: EIO: i/o error, read", "ENOSPC: no space left"):
            self.assertTrue(rollouts.verifier_fault(self.record(self.attempt({"stdout.log": text}))), text)
        for text in ("expected an EIOS frame", "bus errors are handled by the event bus", "panicked: assertion failed"):
            self.assertEqual(rollouts.verifier_fault(self.record(self.attempt({"stdout.log": text}))), "", text)

    def test_the_last_grading_attempt_counts(self):
        directory = self.attempt({"run.log": "Input/output error"}, grading="attempt-2")
        (directory / "grading/attempt-10").mkdir()
        (directory / "grading/attempt-10/run.log").write_text("ok\n")  # a later regrade without faults
        self.assertEqual(rollouts.verifier_fault(self.record(directory)), "")

    def test_score_counts_the_slot_as_infra(self):
        broken = self.attempt({"run.log": "Input/output error (os error 5)"})
        records = ([self.record(None, "pass" if i < 3 else "fail", "0/17", "0/394", f"sc14-r{i}") for i in range(5)]
                   + [self.record(None, "pass", "17/17", "394/394", f"kimi-r{i}") for i in range(3)]
                   + [self.record(broken), self.record(None, "fail", "0/17", "0/394", "kimi-r4")])
        for record in records:
            record["reward"] = 1.0 if record["status"] == "pass" else 0.0

        async def judge_pass(record, instruction, rubric):
            return {"cheated": False, "categories": []}

        async def analyze(*args, **kwargs):
            raise AssertionError("sc14 0.6 is out of range: no failure analysis")

        with tempfile.TemporaryDirectory() as tmp:
            task_dir, out = Path(tmp) / "task", Path(tmp) / "scored"
            task_dir.mkdir()
            (task_dir / "instruction.md").write_text("x")
            (task_dir / "rubric.md").write_text("x")
            build = types.SimpleNamespace(fa_dir=Path(tmp) / "fa", run="unit")
            task = {"task_id": "syn-x", "task_dir": str(task_dir), "snapshot": "s", "name": "cand-c05"}
            with mock.patch.object(task_builder, "judge_pass", judge_pass), \
                    mock.patch.object(task_builder, "analyze", analyze):
                metrics = asyncio.run(task_builder.score(build, task, records, out))
            header = (out / "rollouts/kimi-r3.md").read_text().split("\n")[:5]
            self.assertEqual(task_builder.grading_failures(out, {}), [])  # the md fallback agrees
            seed = task_builder.seed_check(out, metrics)
        self.assertEqual(metrics["valid"], [5, 4])  # kimi-r3 is no longer a FAIL
        self.assertEqual(metrics["teacher_clean_rate"], 0.75)  # 3/4 instead of 3/5
        reason = ("verifier infrastructure fault, not a grade: (os error 5), Input/output error in attempt-0/{run.log} "
                  "(F2P 0/17, P2P 0/394)")
        self.assertEqual(metrics["verifier_faults"], {"kimi-r3": reason})
        self.assertEqual(metrics["grading_failures"], [])  # a platform fault, not a task-caused grading failure
        self.assertEqual(seed, "")  # main 2026-10-20: the task stays a valid seed (4 valid kimi slots)
        self.assertEqual(header[0], "# kimi-r3 — infra")
        self.assertTrue(header[1].endswith("agent_error None; verifier_error None"))  # ATP's own fields unchanged
        self.assertEqual(header[3], f"counted as infra: {reason}")
        self.assertEqual(header[4], "")
        # the usual rule still applies: a model left with < 3 valid slots makes the task inconclusive
        self.assertIn("fewer than 3 valid rollouts per model", task_builder.seed_check(
            Path("/nonexistent"), dict(metrics, valid=[5, 2])))

    def test_only_fault_records_render_the_extra_line(self):
        record = self.record(None, slot="sc14-r0")
        plain = rollouts.render(record).split("\n")
        self.assertEqual(plain[3:5], ["", "## Transcript"])  # unchanged for every other record
        marked = rollouts.render(dict(record, status="infra", verifier_fault="verifier infrastructure fault, x")).split(
            "\n")
        self.assertEqual(marked[3:6], ["counted as infra: verifier infrastructure fault, x", "", "## Transcript"])
        self.assertEqual(plain[1:3], marked[1:3])


class TestPatchSourceEdits(unittest.TestCase):
    """test.patch may not modify existing non-test files (official1-wasmi c03 added a test to the inline `mod test` of
    crates/debugger/src/lib.rs, which both graders reset before grading)."""
    LIB = ("diff --git a/crates/debugger/src/lib.rs b/crates/debugger/src/lib.rs\nindex ebe4ccf..cac37a0 100644\n"
           "--- a/crates/debugger/src/lib.rs\n+++ b/crates/debugger/src/lib.rs\n@@ -524,6 +524,8 @@ mod test {\n"
           "     use super::*;\n+    #[tokio::test]\n+    async fn basic_debugger() {}\n")

    @staticmethod
    def module_line(*added, removed=()):
        body = "".join(f"+{line}\n" for line in added) + "".join(f"-{line}\n" for line in removed)
        return ("diff --git a/crates/debugger/src/lib.rs b/crates/debugger/src/lib.rs\nindex 1..2 100644\n"
                "--- a/crates/debugger/src/lib.rs\n+++ b/crates/debugger/src/lib.rs\n@@ -1,2 +1,4 @@\n pub mod x;\n"
                + body + " pub use x::*;\n")

    def test_existing_source_files_are_flagged(self):
        rust = ["tests/**"]
        self.assertEqual(test_patch_source_edits(self.LIB, rust), ["crates/debugger/src/lib.rs"])
        self.assertEqual(test_patch_source_edits(self.module_line("#[cfg(test)]", "mod debugger_hidden;"), rust), [])
        self.assertEqual(test_patch_source_edits(self.module_line("", "#[cfg(test)] mod hidden_a;",
                                                                  "#[cfg(test)]", "mod hidden_b;"), rust), [])
        self.assertEqual(test_patch_source_edits(self.module_line("mod debugger_hidden;"), rust),
                         ["crates/debugger/src/lib.rs"])  # a non-test module
        self.assertEqual(test_patch_source_edits(self.module_line("#[cfg(test)]", "mod hidden { }"), rust),
                         ["crates/debugger/src/lib.rs"])  # an inline module
        self.assertEqual(test_patch_source_edits(self.module_line("#[cfg(test)] mod hidden;", removed=["pub mod y;"]),
                                                 rust), ["crates/debugger/src/lib.rs"])
        new_file = ("diff --git a/crates/debugger/src/hidden.rs b/crates/debugger/src/hidden.rs\nnew file mode 100644\n"
                    "--- /dev/null\n+++ b/crates/debugger/src/hidden.rs\n@@ -0,0 +1 @@\n+fn t() {}\n")
        self.assertEqual(test_patch_source_edits(new_file, rust), [])  # absent from the starting state
        renamed = ("diff --git a/pkg/helpers.go b/pkg/helpers_test.go\nsimilarity index 90%\nrename from pkg/helpers.go\n"
                   "rename to pkg/helpers_test.go\n")
        self.assertEqual(test_patch_source_edits(renamed, ["**/*_test.go"]), ["pkg/helpers.go"])
        go_test = self.LIB.replace("crates/debugger/src/lib.rs", "pkg/vm/vm_test.go")
        self.assertEqual(test_patch_source_edits(go_test, ["**/*_test.go", "**/testdata/**"]), [])
        for path in ("crates/x/src/tests.rs", "src/foo/parser_tests.rs", "app/tests.py", "lib/x.spec.ts",
                     "crates/x/tests/all/main.rs", "pkg/testdata/in.txt", "src/__snapshots__/a.snap", "conftest.py"):
            self.assertTrue(is_test_file(path, rust), path)
        for path in ("crates/x/src/testing.rs", "src/latest.go", "src/contest.py", "Cargo.toml", "jest.config.js"):
            self.assertFalse(is_test_file(path, rust), path)

    def test_gates_report_the_edit(self):
        build = task_builder.Build("unit-test-run", TARGET, "deepnoodle-ai/risor")  # Go globs

        async def grade(source, task_dir, patch, *, run, out_dir):
            new = {"n1": "passed", "n2": "passed"} if patch else {"n1": "failed", "n2": "failed"}
            return {"suites": {"base-ctrf.json": {"p1": "passed"}, "new-ctrf.json": new}}

        async def never(*args, **kwargs):
            raise AssertionError("the static problem stops the gates first")

        test_patch = self.LIB.replace("crates/debugger/src/lib.rs", "vm/vm.go")
        with tempfile.TemporaryDirectory() as directory:
            author_dir, cand = Path(directory) / "author-0", Path(directory) / "cand"
            author_dir.mkdir()
            for name, text in {"run_tests.sh": "true\n", "test.patch": test_patch, "solution.patch": "x",
                               "start.patch": ""}.items():
                (author_dir / name).write_text(text)
            with mock.patch.object(task_builder, "grade", grade), mock.patch.object(task_builder, "stale_tests", never):
                gate = asyncio.run(task_builder.gates(build, cand, author_dir, {"snapshot": "s", "base_commit": "c"}))
        self.assertEqual(len(gate["problems"]), 1)
        self.assertTrue(gate["problems"][0].startswith(
            "test.patch modifies existing non-test files of the starting state (outside test_paths "
            "['**/*_test.go', '**/testdata/**'] and test directories/file names): ['vm/vm.go']. Both graders reset"))
        self.assertIn("`#[cfg(test)] mod <name>;`", gate["problems"][0])

    def test_author_prompt_states_the_rule_without_new_fields(self):
        text = (Path(__file__).resolve().parents[1] / "prompts/author.md").read_text()
        self.assertIn("no edit of an existing non-test file", text)
        self.assertIn("`#[cfg(test)] mod <name>;`", text)
        fields = dict.fromkeys(["title", "repo", "kind", "kind_rule", "start_rule", "sha", "base_report", "new_report",
                                "grade_cfg", "feedback", "sandbox"], "x")
        task_builder.fill(text, **fields)  # the running builds' render() supplies exactly these


class StaleOracles(unittest.TestCase):
    """Test files on C that reference code start.patch removes from source files (official1-wasmi c04 kept wast
    conformance files of the removed transcoding; they failed on C outside build_env's broad suite)."""
    START = "".join([
        "diff --git a/aiohttp/cookiejar.py b/aiohttp/cookiejar.py\n--- a/aiohttp/cookiejar.py\n+++ b/aiohttp/cookiejar.py\n",
        "@@ -1,9 +1,3 @@\n",
        "-def parse_cookie_partition(value):\n",
        "-    domains = itertools.accumulate(value)\n",            # a library call: not defined here
        "-    raise CookieError(\"invalid partition key in cookie\")\n",
        "-    header = '; filename=\"' + quote(name)\n",              # a fragment, not a message or key
        "-    # comment_only_identifier_here\n",
        "-        const_like_local = compute_partition(value)\n",   # a local: not defined at the top level
        "-PARTITION_SEPARATOR_TOKEN = \"%%\"\n",
        "-def kept_renamed_function(x):\n",
        "+def kept_renamed_function(y):\n",                         # still in the starting state
        " x\n",
        "diff --git a/tests/test_cookiejar.py b/tests/test_cookiejar.py\n--- a/tests/test_cookiejar.py\n",
        "+++ b/tests/test_cookiejar.py\n@@ -1,2 +1 @@\n-def test_removed_behaviour_helper(): pass\n x\n"])

    def test_removed_source_items(self):
        items = removed_source_items(self.START, ["tests/**", "**/test_*.py"])
        source = "aiohttp/cookiejar.py"
        self.assertEqual(items, {"parse_cookie_partition": {"file": source, "kind": "name"},
                                 "invalid partition key in cookie": {"file": source, "kind": "literal"},
                                 "PARTITION_SEPARATOR_TOKEN": {"file": source, "kind": "name"}})
        self.assertEqual(removed_source_items("", ["tests/**"]), {})  # evolve tasks: nothing removed

    def test_oracle_problems(self):
        def item(kind):
            return {"file": "aiohttp/cookiejar.py", "kind": kind}
        items = {"parse_cookie_partition": item("name"), "invalid partition key in cookie": item("literal"),
                 "partition cookies are rejected": item("literal"), "still_used_elsewhere": item("name"),
                 "content-encoding": item("literal")}
        output = "\n".join([
            "i\t./tests/test_partition_spec.py\tparse_cookie_partition",
            "l\t./tests/data/partition.txt\tinvalid partition key in cookie",
            "l\t./tests/data/partition.txt\tpartition cookies are rejected",
            "l\t./tests/test_compression.py\tcontent-encoding",  # one key-like literal: a header name, no oracle
            "l\t./tests/test_errors.py\tpartition cookies are rejected",  # one message suffices
            "i\t./tests/test_reset_by_grader.py\tparse_cookie_partition",
            "i\t./tests/test_other.py\tstill_used_elsewhere", "i\t./aiohttp/helpers.py\tstill_used_elsewhere",
            "i\t./docs/cookies.rst\tparse_cookie_partition",  # documentation is neither a test nor code
            "E\tdone"])
        test_patch = "diff --git a/tests/test_reset_by_grader.py b/tests/test_reset_by_grader.py\n"
        problems, stale = oracle_problems(output, items, ["tests/**"], test_patch)
        self.assertEqual(stale, {"tests/test_partition_spec.py": ["parse_cookie_partition"],
                                 "tests/data/partition.txt": ["invalid partition key in cookie",
                                                              "partition cookies are rejected"],
                                 "tests/test_errors.py": ["partition cookies are rejected"]})
        self.assertEqual(problems, [f"{task_builder.ORACLE_PREFIX} 3 files: tests/data/partition.txt (references "
                                    "'invalid partition key in cookie', 'partition cookies are rejected'); "
                                    "tests/test_errors.py (references 'partition cookies are rejected'); "
                                    "tests/test_partition_spec.py (references 'parse_cookie_partition')"])
        items["content-encoding-v2"] = item("literal")  # two key-like literals corroborate each other
        two = output.replace("E\tdone", "l\t./tests/test_compression.py\tcontent-encoding-v2\nE\tdone")
        self.assertIn("tests/test_compression.py", oracle_problems(two, items, ["tests/**"], test_patch)[1])
        self.assertEqual(oracle_problems("E\tdone", items, ["tests/**"], ""), ([], {}))

    def test_stale_oracles_sandbox_round_trip(self):
        uploads, commands = {}, []

        class Box:
            async def put_text(self, text, remote):
                uploads[remote] = text

            async def sh(self, command, *, cwd="/app", timeout=1800, check=True):
                commands.append(command)
                return types.SimpleNamespace(stdout="i\t./tests/spec/partition_spec.py\tparse_cookie_partition\nE\tdone",
                                             exit_code=0, timed_out=False)

        @contextlib.asynccontextmanager
        async def owned(snapshot, *, run, purpose):
            yield Box()

        build = types.SimpleNamespace(run="unit")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(task_builder, "owned", owned):
            cand = Path(tmp)
            problems = asyncio.run(task_builder.stale_oracles(build, cand, {"snapshot": "s"}, self.START, "",
                                                              ["tests/**"]))
            saved = json.loads((cand / "grade/stale-oracles.json").read_text())
            self.assertEqual(asyncio.run(task_builder.stale_oracles(build, cand, {"snapshot": "s"}, "", "",
                                                                    ["tests/**"])), [])
        self.assertEqual(commands, [task_builder.ORACLE_SCAN])  # the second call needed no sandbox
        self.assertEqual(uploads["/tmp/oracle-idents"], "PARTITION_SEPARATOR_TOKEN\nparse_cookie_partition\n")
        self.assertEqual(uploads["/tmp/oracle-literals"], "invalid partition key in cookie\n")
        self.assertEqual(len(problems), 1)
        self.assertEqual(saved["stale"], {"tests/spec/partition_spec.py": ["parse_cookie_partition"]})

    def test_scan_script_reports_files_and_items(self):
        if not shutil.which("grep"):
            self.skipTest("grep missing")
        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp) / "app"
            (app / "tests/spec").mkdir(parents=True)
            (app / "tests/spec/a.wast").write_text('(assert_trap (invoke "f") "invalid partition key in cookie")\n')
            (app / "tests/b.py").write_text("from x import parse_cookie_partition_v2, parse_cookie_partition\n")
            (app / "node_modules/x").mkdir(parents=True)
            (app / "node_modules/x/c.js").write_text("parse_cookie_partition\n")
            idents, literals = Path(tmp) / "idents", Path(tmp) / "literals"
            idents.write_text("parse_cookie_partition\n")
            literals.write_text("invalid partition key in cookie\n")
            script = (task_builder.ORACLE_SCAN.replace("/app", str(app)).replace("/tmp/oracle-$kind", f"{tmp}/$kind"))
            result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.stdout.splitlines(), ["i\t./tests/b.py\tparse_cookie_partition",
                                                      "l\t./tests/spec/a.wast\tinvalid partition key in cookie",
                                                      "E\tdone"])


class EnvExclusions(unittest.TestCase):
    """build_env's exclusions stay out of P2P (official1-dynamodb: 3 wall-clock 'timeout' tests excluded by the env's
    Jest --testNamePattern were graded P2P in 23 of 25 round-1 tasks)."""
    ENV = ["tests/jest/__tests__/index.test.tsx: actions > timeout > returns timeout error",
           "tests/jest/__tests__/index.test.tsx: actions > timeout > returns timeout error for procedure"]

    def test_env_unreliable_from_the_baselines(self):
        cfg = {"format": "ctrf", "node_id": "name"}

        def ctrf(rows):
            return json.dumps({"results": {"tests": [{"name": n, "status": s} for n, s in rows]}})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "env").mkdir()
            (root / "env/baseline-0").write_text(ctrf([("a: skipped both", "skipped"), ("a: flaky", "passed"),
                                                       ("a: broken", "failed"), ("a: fine", "passed"),
                                                       ("a: one skip", "skipped")]))
            (root / "env/baseline-1").write_text(ctrf([("a: skipped both", "skipped"), ("a: flaky", "failed"),
                                                       ("a: broken", "failed"), ("a: fine", "passed")]))
            build = types.SimpleNamespace(root=root, grade_cfg=cfg)
            self.assertEqual(env_unreliable(build), {"a: skipped both", "a: flaky", "a: one skip"})
            self.assertEqual(env_unreliable(types.SimpleNamespace(root=root / "missing", grade_cfg=cfg)), set())

    def test_ids_of_every_report_format_match(self):
        ids = ["tests/jest/__tests__/index.test.tsx: actions > timeout > returns timeout error",
               "tests/jest/__tests__/index.test.tsx: actions timeout returns timeout error",
               "__tests__/index.test.tsx: actions > timeout > returns timeout error",
               "index.test.tsx: actions timeout > returns timeout error",
               "tests/jest/__tests__/index.test.tsx: actions timeout returns timeout error [occurrence 1]",
               "actions: actions timeout returns timeout error",
               "__tests__/index.test.tsx: actions timeout returns timeout error for procedure"]
        others = ["__tests__/index.test.tsx: actions timeout returns timeout error for protected action",
                  "__tests__/client.test.tsx: actions > timeout > returns timeout error",  # another file
                  "__tests__/index.test.tsx: actions > retries > returns timeout error",
                  "__tests__/index.test.tsx: returns timeout error"]  # a title shorter than the env's
        self.assertEqual(env_matches(ids + others, self.ENV), ids)
        self.assertEqual(env_matches(["a.test.ts: parser works"], ["a.test.ts: works"]), [])  # short: exact only
        self.assertEqual(env_matches(["a.test.ts: works"], ["a.test.ts: works"]), ["a.test.ts: works"])
        go = ["github.com/o/r/pkg.TestTimeout/slow", "github.com/o/r/other.TestTimeout/slow"]
        self.assertEqual(env_matches(go, ["github.com/o/r/pkg.TestTimeout/slow"]), go[:1])
        py = ["tests/test_client.py::TestTimeout::test_slow_response_is_cut", "tests.test_client.TestTimeout"
              ".test_slow_response_is_cut"]
        self.assertEqual(env_matches(py, ["tests.test_client.TestTimeout.test_slow_response_is_cut"]), py)

    def test_calibrate_keeps_them_out_of_p2p(self):
        base = {"__tests__/index.test.tsx: actions timeout returns timeout error": "passed",
                "__tests__/index.test.tsx: actions timeout returns timeout error for procedure": "failed",
                "__tests__/index.test.tsx: other test": "passed"}
        empty = {"suites": {"base-ctrf.json": base, "new-ctrf.json": {"n1": "failed", "n2": "failed"}}}
        solved = [{"suites": {"base-ctrf.json": dict(base), "new-ctrf.json": {"n1": "passed", "n2": "passed"}}}
                  for _ in range(3)]
        reports = ["/logs/verifier/base-ctrf.json", "/logs/verifier/new-ctrf.json"]
        f2p, p2p, problems = calibrate(empty, solved, reports)
        self.assertEqual(p2p, ["__tests__/index.test.tsx: actions timeout returns timeout error",
                               "__tests__/index.test.tsx: other test"])
        self.assertIn("base suite fails on the starting state", problems[0])  # the flaky one failed once
        f2p, p2p, problems = calibrate(empty, solved, reports, set(self.ENV))
        self.assertEqual((f2p, p2p, problems), (["n1", "n2"], ["__tests__/index.test.tsx: other test"], []))

    def test_gates_record_the_exclusion(self):
        build = task_builder.Build("unit-test-run", TARGET, "deepnoodle-ai/risor")

        async def grade(source, task_dir, patch, *, run, out_dir):
            new = {"n1": "passed", "n2": "passed"} if patch else {"n1": "failed", "n2": "failed"}
            return {"reward": 1 if patch else 0,
                    "suites": {"base-ctrf.json": {"pkg.TestTimeout": "passed", "pkg.TestOther": "passed"},
                               "new-ctrf.json": new}}

        async def none(*args, **kwargs):
            return []

        with tempfile.TemporaryDirectory() as directory:
            author_dir, cand = Path(directory) / "author-0", Path(directory) / "cand"
            author_dir.mkdir()
            for name, text in {"run_tests.sh": "true\n", "test.patch": "", "solution.patch": "x",
                               "start.patch": ""}.items():
                (author_dir / name).write_text(text)
            with mock.patch.object(task_builder, "grade", grade), mock.patch.object(task_builder, "stale_tests", none), \
                    mock.patch.object(task_builder, "env_unreliable", lambda b: {"pkg.TestTimeout"}), \
                    mock.patch.object(task_builder, "leak_scan", none):
                gate = asyncio.run(task_builder.gates(build, cand, author_dir, {"snapshot": "s", "base_commit": "c"}))
        self.assertEqual(gate["p2p"], ["pkg.TestOther"])
        self.assertEqual(gate["env_excluded"], ["pkg.TestTimeout"])


if __name__ == "__main__":
    unittest.main()
