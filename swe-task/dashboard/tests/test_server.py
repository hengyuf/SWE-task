"""Offline tests of the dashboard server (server.py) and its Slurm wrappers (launch.sh, dashboard.sbatch).

Real server processes on 127.0.0.1 with port 0 in temporary directories, with dummy static/ and data/ files and
--no-collect or a fake collector; launch.sh runs against fake Slurm commands (DASHBOARD_SLURM_BIN). Nothing is
submitted or cancelled, no real pipeline file is read, and every server started here is stopped.

    cd /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/dashboard
    /usr/bin/python3 -I -B -m unittest discover -s tests -v

ResolveTest    App.resolve() (the path -> file mapping), the log redaction and secret masking, the session value, the
               conditional-request rules, in-process
AuthTest       403 without a token, the ?token= cookie flow and its redirect, the per-port HMAC session cookie (never
               the token; the old scheme's cookie expired), header auth, wrong tokens, /healthz
PathTest       traversal (plain, percent-encoded, double-encoded, backslash, NUL, symlinks), hidden and temp files,
               directory listings, unknown types, code and state never served, aliases and content types
ProtocolTest   methods (405 / 501), HEAD, security headers on every kind of response, malformed requests, the token
               never in the log, server.json mode and fields
CacheTest      ETag / Last-Modified / Cache-Control: no-cache on the files, 304 on If-None-Match / If-Modified-Since, a
               replaced file revalidating to 200, no-store and no validators elsewhere
LimitTest      the 32 handler slots: 503 at once beyond them, a slot back when its client leaves, the 15 s socket
               timeout
LifecycleTest  --reuse-token (and its port), SIGTERM, the token (whole, cut short, glued, malformed requests) never in
               the log, a non-loopback --host outside Slurm, a taken port, the collector thread (first cycle at
               startup, failing cycles, a broken then fixed collect.py, server_status.json)
ScriptsTest    launch.sh start / status / stop / local / header against fake Slurm commands (--reuse-token keeping the
               port, curl with the header file); dashboard.sbatch directives and its environment scrub
"""
import sys

sys.dont_write_bytecode = True

import email.utils
import gzip
import hashlib
import hmac
import http.client
import importlib.util
import io
import json
import os
from pathlib import Path
import pwd
import re
import secrets
import shutil
import socket
import stat
import subprocess
import tempfile
import time
import unittest
import urllib.parse

DASHBOARD = Path(__file__).resolve().parent.parent
SERVER = DASHBOARD / "server.py"
LAUNCH = DASHBOARD / "launch.sh"
SBATCH = DASHBOARD / "dashboard.sbatch"
PY = "/usr/bin/python3"
ME = pwd.getpwuid(os.getuid())
INDEX_HTML = '<!doctype html><title>dashboard</title><script src="static/app.js" defer></script>\n'
APP_JS = "'use strict'; console.log('app');\n"
STYLE_CSS = "body { margin: 0 }\n"
INDEX_JSON = {"schema": 1, "generated_at": 1800000000.0, "tasks": []}
BIG_JSON = json.dumps({"nodes": [{"id": f"cand-c{n:02d}", "title": "a node of the tree " * 4} for n in range(200)]})
SECRET = "TOP-SECRET-OUTSIDE-THE-SERVED-ROOTS"
FAKE_KEY = "nvapi-" + "Q" * 40  # must never reach a log or status file


def clean_env(**extra):
    """The environment of every child process: nothing inherited from the test runner."""
    env = {"PATH": "/usr/bin:/bin", "HOME": ME.pw_dir, "USER": ME.pw_name, "LOGNAME": ME.pw_name, "LANG": "C.UTF-8"}
    env.update(extra)
    return env


def read_json(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def session_value(token, port):
    """The session cookie's value as CONTRACT.md defines it (computed here independently of server.py)."""
    return hmac.new(token.encode(), f"session:{port}".encode(), hashlib.sha256).hexdigest()


def free_port(exclude=()):
    """A port nobody listens on right now (and not one of `exclude`)."""
    while True:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port not in exclude:
            return port


def wait_until(predicate, timeout=20.0, process=None, what="condition", output=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        if process is not None and process.poll() is not None:
            raise AssertionError(f"process exited ({process.returncode}) before {what}:\n{output() if output else ''}")
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}:\n{output() if output else ''}")


def make_root(root):
    """A dashboard root with dummy static/ and data/, plus files that must never be served."""
    root = Path(root)
    for relative, content in {
        "static/index.html": INDEX_HTML, "static/app.js": APP_JS, "static/style.css": STYLE_CSS,
        "static/logo.svg": "<svg xmlns='http://www.w3.org/2000/svg'/>\n", "static/notes.md": "# not a served type\n",
        "static/.hidden.js": "hidden\n", "data/index.json": json.dumps(INDEX_JSON), "data/big.json": BIG_JSON,
        "data/task/task-a.json": json.dumps({"task_id": "task-a"}), "data/task/v1.2-dotted.json": "{}",
        "data/history/fleet.jsonl": '{"t": 1}\n{"t": 2}\n', "data/notes.txt": "a text file in data/\n",
        "data/.index.json.123.140.tmp": "{}", "secret.json": json.dumps({"secret": SECRET}),
        "secret.txt": SECRET, "collect.py": "raise SystemExit('the real collector must not run in these tests')\n",
        "state/collector.json": json.dumps({"secret": SECRET}),
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    os.symlink(root / "secret.json", root / "data/escape.json")  # absolute symlink out of data/
    os.symlink("../secret.json", root / "static/escape.json")  # relative symlink out of static/
    os.symlink("index.json", root / "data/alias.json")  # a symlink that stays inside data/: served
    os.mkfifo(root / "data/pipe.json")  # not a regular file: never opened
    (root / "static/sub").mkdir()
    return root


class Response:
    def __init__(self, status, headers, body):
        self.status, self.headers, self.body = status, headers, body

    def header(self, name):
        values = self.all(name)
        return values[0] if values else None

    def all(self, name):
        return [value for key, value in self.headers if key.lower() == name.lower()]


def request(port, method, path, headers=None, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return Response(response.status, response.getheaders(), response.read())
    finally:
        connection.close()


def raw_request(port, data):
    """Bytes in, the whole answer out (for request lines http.client refuses to send)."""
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(data)
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)


class ServerProcess:
    """server.py (or a command that execs it) in a subprocess; stop() sends SIGTERM, then SIGKILL after 10 s."""

    def __init__(self, root, *args, env=None, command=None, info_name="server.json"):
        self.root = Path(root)
        self.info_file = self.root / "state" / info_name
        self.log_path = self.root / f"server-{time.monotonic_ns()}.log"
        self.log = open(self.log_path, "wb")
        argv = command or [PY, "-I", "-B", str(SERVER), "--root", str(self.root), "--host", "127.0.0.1",
                           "--port", "0", *args]
        self.process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT,
                                        env=clean_env() if env is None else env, cwd=str(self.root))
        try:
            self.info = wait_until(self.started, process=self.process, what=str(self.info_file), output=self.output)
        except BaseException:
            self.stop()
            raise
        self.port, self.token = self.info["port"], self.info["token"]

    def started(self):
        info = read_json(self.info_file)
        return info if info.get("pid") == self.process.pid and info.get("port") else None

    def output(self):
        try:
            return self.log_path.read_text(errors="replace")
        except OSError:
            return ""

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(10)
        self.log.close()
        return self.process.returncode

    def get(self, path, headers=None, auth=True, method="GET", body=None):
        headers = dict(headers or {})
        if auth:
            headers.setdefault("X-Dashboard-Token", self.token)
        return request(self.port, method, path, headers, body)

    def cookie(self):
        """The session cookie a login sets: named after the port, an HMAC of the token, never the token."""
        return f"swe_dash_{self.port}={session_value(self.token, self.port)}"


class ServerCase(unittest.TestCase):
    """One server per test class: a temporary root (make_root) and --no-collect."""

    server_args = ("--no-collect",)

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="dash-server-test-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.root = make_root(cls.tmp / "root")
        cls.server = ServerProcess(cls.root, *cls.server_args)
        cls.addClassCleanup(cls.server.stop)

    def assertSecurityHeaders(self, response, cache="no-store"):
        """The headers of every response; `cache`: no-cache for the files of static/ and data/, else no-store."""
        self.assertEqual(response.all("Cache-Control"), [cache])
        self.assertEqual(response.header("X-Content-Type-Options"), "nosniff")
        csp = response.header("Content-Security-Policy") or ""
        directives = {part.split()[0]: part.split()[1:] for part in csp.split(";") if part.strip()}
        self.assertEqual(directives.get("default-src"), ["'self'"])
        self.assertEqual(directives.get("script-src"), ["'self'"])  # no 'unsafe-inline' / 'unsafe-eval' scripts
        self.assertNotIn("'unsafe-eval'", csp)
        self.assertEqual(directives.get("object-src"), ["'none'"])
        self.assertEqual(directives.get("frame-ancestors"), ["'none'"])
        self.assertEqual(response.header("Referrer-Policy"), "no-referrer")

    def assertNoLeak(self, response):
        for needle in (SECRET.encode(), self.server.token.encode(), b"collector.json"):
            self.assertNotIn(needle, response.body)


# ------------------------------------------------------------------------------------------------------ in-process

class ResolveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("dashboard_server_under_test", SERVER)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.tmp = Path(tempfile.mkdtemp(prefix="dash-resolve-test-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.root = make_root(cls.tmp / "root")
        cls.app = cls.module.App(cls.root, "t" * 43)

    def test_served_files(self):
        real = os.path.realpath
        cases = {"/": "static/index.html", "/index.html": "static/index.html", "/app.js": "static/app.js",
                 "/style.css": "static/style.css", "/static/app.js": "static/app.js",
                 "/static/logo.svg": "static/logo.svg", "/data/index.json": "data/index.json",
                 "/data/task/task-a.json": "data/task/task-a.json", "/data/task/v1.2-dotted.json":
                 "data/task/v1.2-dotted.json", "/data/history/fleet.jsonl": "data/history/fleet.jsonl",
                 "/data/alias.json": "data/index.json"}
        for path, relative in cases.items():
            with self.subTest(path=path):
                self.assertEqual(self.app.resolve(path), real(self.root / relative))

    def test_refused_paths(self):
        for path in ["", "index.html", "/static", "/static/", "/data", "/data/", "/data/task", "/data/task/",
                     "/static/sub", "/static/sub/", "/static//app.js", "/static/./app.js", "/static/../secret.json",
                     "/data/../secret.json", "/data/task/../../secret.json", "/../secret.json", "//secret.json",
                     "/static/..\\secret.json", "/static/\\..\\secret.json", "/static/app.js\x00.json",
                     "/data/\nindex.json", "/static/.hidden.js", "/data/.index.json.123.140.tmp",
                     "/static/notes.md", "/data/notes.txt", "/data/escape.json", "/static/escape.json",
                     "/data/pipe.json", "/state/server.json", "/state/collector.json", "/collect.py", "/server.py",
                     "/secret.json", "/data/index.json/", "/data/missing.json", "/STATIC/app.js",
                     "/static/%2e%2e/secret.json"]:
            with self.subTest(path=path):
                self.assertIsNone(self.app.resolve(path))

    def test_check_is_exact(self):
        token = "t" * 43
        self.assertTrue(self.app.check(token))
        for offered in ["", token[:-1], token + "t", token.upper(), "t" * 42 + "é", None, 43, "t" * 10000]:
            with self.subTest(offered=offered if not isinstance(offered, str) else offered[:50]):
                self.assertFalse(self.app.check(offered))

    def test_accepts_gzip(self):
        for header, expected in [("gzip", True), ("gzip, deflate, br, zstd", True), ("br;q=1.0, gzip;q=0.5", True),
                                 ("*", True), ("x-gzip", True), ("GZIP ; Q=0.3", True), ("gzip;q=0", False),
                                 ("gzip;q=0.0, *", False), ("gzip;q=junk", False), ("identity", False),
                                 ("deflate, br", False), ("", False), (None, False)]:
            with self.subTest(header=header):
                self.assertEqual(self.module.accepts_gzip(header), expected)

    def test_redact_masks_tokens_and_keys(self):
        text = self.module.redact(f"GET /?token=abc123secret&x=1 {FAKE_KEY} Authorization: Bearer abcdefghijkl")
        self.assertNotIn("abc123secret", text)
        self.assertNotIn("Q" * 40, text)
        self.assertNotIn("abcdefghijkl", text)
        self.assertIn("x=1", text)

    def test_mask_hides_the_live_secrets(self):
        token, mask = secrets.token_urlsafe(32), self.module.SecretMask()
        session = session_value(token, 8765)
        mask.add(token, "<token>")
        mask.add(session, "<session>")
        for text, expected in [
            (f"GET /{token} -> 403", "GET /<token> -> 403"),  # typed as a path
            (f"GET /x{token}y.json -> 404", "GET /x<token>y.json -> 404"),  # glued to other text
            (f"/{token}/{token}", "/<token>/<token>"),  # every occurrence
            (f"/{token[:20]} -> 403", "/<token> -> 403"),  # cut short
            (f"/{token[5:]}", "/<token>"),
            (f"/{token[:15]}~{token[16:]}", "/<token>~<token>"),  # a typo in the middle
            (f"('GET /{token} x')", "('GET /<token> x')"),
            (f"cookie {session}", "cookie <session>"),
            (f"/{token[:7]}", f"/{token[:7]}"),  # 7 characters: no telling them from chance
            ("GET /data/task/narwhals-rolling-window-suite.json -> 404",
             "GET /data/task/narwhals-rolling-window-suite.json -> 404"),
        ]:
            with self.subTest(text=text.replace(token, "<t>")):
                self.assertEqual(mask(text), expected)
        # The module's own mask: in everything printed (scrub) and recorded (redact), with every token= value.
        self.module.MASK.add(token, "<token>")
        for function in (self.module.scrub, self.module.redact):
            text = function(f"GET /{token} ?a=1&token=other x Token=y")
            self.assertEqual(text, "GET /<token> ?a=1&token=<redacted> x Token=<redacted>")

    def test_session_value_is_per_port_and_never_the_token(self):
        token = "t" * 43
        app = self.module.App(self.root, token)
        app.set_port(8765)
        self.assertEqual(app.cookie_name, "swe_dash_8765")
        self.assertEqual(app.session(), session_value(token, 8765))
        self.assertTrue(app.check_session(session_value(token, 8765)))
        other = self.module.App(self.root, token)
        other.set_port(8766)
        self.assertEqual(other.cookie_name, "swe_dash_8766")
        for offered in [token, other.session(), session_value(token[:-1] + "u", 8765), app.session().upper(),
                        app.session()[:-1], app.session() + "0", "", None, 8765]:
            with self.subTest(offered=offered):
                self.assertFalse(app.check_session(offered))
        self.assertTrue(app.check(token))
        self.assertFalse(app.check(app.session()))  # the header takes the token, not the cookie value

    def test_conditional_requests(self):
        def headers(*lines):
            return http.client.parse_headers(io.BytesIO("".join(f"{line}\r\n" for line in lines + ("",)).encode()))

        mtime = 1792280000.7
        modified = email.utils.formatdate(mtime, usegmt=True)  # what Last-Modified said: whole seconds
        for lines, expected in [
            ((), False), (('If-None-Match: W/"a1-b2"',), True), (('If-None-Match: "a1-b2"',), True),
            (('If-None-Match: "x", W/"a1-b2" , "y"',), True),
            (('If-None-Match: "x"', 'If-None-Match: W/"a1-b2"'), True),  # two headers
            (("If-None-Match: *",), True), (('If-None-Match: W/"a1-b3"',), False), (("If-None-Match: a1-b2",), False),
            ((f"If-Modified-Since: {modified}",), True),
            ((f"If-Modified-Since: {email.utils.formatdate(mtime + 60, usegmt=True)}",), True),
            ((f"If-Modified-Since: {email.utils.formatdate(mtime - 1, usegmt=True)}",), False),
            (("If-Modified-Since: yesterday",), False), (("If-Modified-Since: ",), False),
            (('If-None-Match: "x"', f"If-Modified-Since: {modified}"), False),  # If-None-Match decides
            (('If-None-Match: "a1-b2"', "If-Modified-Since: Thu, 01 Jan 1970 00:00:00 GMT"), True),
        ]:
            with self.subTest(lines=lines):
                self.assertEqual(self.module.not_modified(headers(*lines), "a1-b2", mtime), expected)

    def test_limits(self):
        self.assertEqual(self.module.MAX_HANDLERS, 32)
        self.assertEqual(self.module.Handler.timeout, 15)  # the request socket's timeout


# ------------------------------------------------------------------------------------------------- a live server

class AuthTest(ServerCase):
    def test_healthz_is_open_and_holds_no_data(self):
        token = self.server.token
        for path, headers in [("/healthz", {}), (f"/healthz?token={token}", {}), ("/healthz?x=1", {}),
                              ("/healthz", {"Cookie": self.server.cookie(), "X-Dashboard-Token": token})]:
            for method in ("GET", "HEAD"):
                with self.subTest(path=path.replace(token, "<t>"), headers=bool(headers), method=method):
                    response = self.server.get(path, headers, auth=False, method=method)
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.header("Content-Type"), "application/json")
                    self.assertEqual(response.all("Set-Cookie"), [])
                    self.assertIsNone(response.header("ETag"))
                    self.assertSecurityHeaders(response)
                    self.assertEqual(response.header("Content-Length"), str(len(b'{"ok": true}')))
                    self.assertEqual(response.body, b"" if method == "HEAD" else b'{"ok": true}')  # that only

    def test_token_required(self):
        for path in ["/", "/index.html", "/app.js", "/static/app.js", "/static/style.css", "/data/index.json",
                     "/data/task/task-a.json", "/data/history/fleet.jsonl", "/missing", "/static/", "/data/"]:
            for method in ("GET", "HEAD"):
                with self.subTest(path=path, method=method):
                    response = self.server.get(path, auth=False, method=method)
                    self.assertEqual(response.status, 403)
                    self.assertSecurityHeaders(response)
                    self.assertEqual(response.all("Set-Cookie"), [])
                    self.assertNotIn(b"dashboard</title>", response.body)
                    self.assertNotIn(b"schema", response.body)
                    self.assertNoLeak(response)
        page = self.server.get("/", auth=False).body.decode()  # the 403 page says what to do
        self.assertIn("?token=", page)
        self.assertIn("curl -H @$HOME/.dashboard-header", page)  # tools: a header file, mode 600
        self.assertIn('"X-Dashboard-Token: <token>"', page)
        self.assertIn("never with the token on a command line", page)
        self.assertNotRegex(page, r"-H ['\"]X-Dashboard-Token")  # no example that puts it there

    def test_cookie_flow(self):
        token, port = self.server.token, self.server.port
        response = self.server.get(f"/?token={token}", auth=False)
        self.assertEqual(response.status, 303)
        self.assertEqual(response.header("Location"), "/")  # the token leaves the address bar
        self.assertEqual(response.header("Content-Length"), "0")
        self.assertSecurityHeaders(response)
        cookies = response.all("Set-Cookie")
        self.assertEqual(len(cookies), 1)
        pair, *attributes = [part.strip() for part in cookies[0].split(";")]
        # Per port, and the HMAC of the token: a browser sends it to every port of localhost, so never the token.
        self.assertEqual(pair, f"swe_dash_{port}={session_value(token, port)}")
        self.assertEqual(pair, self.server.cookie())
        self.assertEqual(self.server.info["cookie"], f"swe_dash_{port}")
        self.assertNotIn(token, cookies[0])
        self.assertIn("HttpOnly", attributes)
        self.assertIn("SameSite=Strict", attributes)
        self.assertIn("Path=/", attributes)
        self.assertTrue(any(re.fullmatch(r"Max-Age=\d+", item) for item in attributes))
        # The browser comes back with the cookie only.
        page = self.server.get("/", {"Cookie": pair}, auth=False)
        self.assertEqual(page.status, 200)
        self.assertEqual(page.body.decode(), INDEX_HTML)
        self.assertEqual(page.all("Set-Cookie"), [])
        data = self.server.get("/data/index.json", {"Cookie": f"other=1; {pair}; theme=dark"}, auth=False)
        self.assertEqual(data.status, 200)
        self.assertEqual(json.loads(data.body), INDEX_JSON)

    def test_token_redirect_keeps_path_and_other_parameters(self):
        response = self.server.get(f"/data/index.json?x=1&token={self.server.token}&y=a%20b", auth=False)
        self.assertEqual(response.status, 303)
        self.assertEqual(response.header("Location"), "/data/index.json?x=1&y=a+b")
        self.assertEqual(len(response.all("Set-Cookie")), 1)

    def test_redirect_never_leaves_the_origin(self):
        # (http.server itself folds a leading "//" into "/"; whatever arrives, Location stays a path here.)
        token = self.server.token
        for path in [f"//evil.example/?token={token}", f"/\\evil.example/?token={token}",
                     f"/\\\\evil.example?token={token}", f"///evil.example/%2e%2e?token={token}",
                     f"/%2F%2Fevil.example?token={token}"]:
            with self.subTest(path=path):
                response = self.server.get(path, auth=False)
                self.assertEqual(response.status, 303)
                location = response.header("Location")
                self.assertTrue(location.startswith("/") and not location.startswith("//"), location)
                self.assertNotIn("\\", location)
                self.assertEqual(urllib.parse.urlsplit(location).netloc, "")
        self.assertEqual(self.server.get(f"/\\evil.example/?token={self.server.token}", auth=False)
                         .header("Location"), "/")

    def test_wrong_tokens(self):
        token, port, name = self.server.token, self.server.port, self.server.info["cookie"]
        session = session_value(token, port)
        legacy = "swe_dash_" + hashlib.sha256(token.encode()).hexdigest()[:10]  # the old scheme: name and value
        for path, headers in [
            ("/?token=wrong", {}), (f"/?token={token[:-1]}", {}), (f"/?token={token}x", {}), ("/?token=", {}),
            (f"/?token={session}", {}),  # the cookie value is no token
            ("/", {"Cookie": f"{name}=wrong"}), ("/", {"Cookie": f"{name}={session[:-1]}"}),
            ("/", {"Cookie": f"{name}={session.upper()}"}),
            ("/", {"Cookie": f"{name}={token}"}),  # the token itself is no session
            ("/", {"Cookie": f"{name}={session_value(token, port + 1)}"}),  # another port's session
            ("/", {"Cookie": f"swe_dash_{port + 1}={session}"}),  # the right value under another name
            ("/", {"Cookie": f"{legacy}={token}"}),  # the old scheme's cookie
            ("/", {"Cookie": f"{name}"}), ("/", {"X-Dashboard-Token": "wrong"}), ("/", {"X-Dashboard-Token": ""}),
            ("/", {"X-Dashboard-Token": session}),  # the header takes the token itself
            ("/", {"Authorization": f"Bearer {token}"}),  # not a supported scheme
        ]:
            with self.subTest(path=path.replace(token, "<t>"), headers=str(headers).replace(token, "<t>")):
                response = self.server.get(path, headers, auth=False)
                self.assertEqual(response.status, 403)
                self.assertEqual(response.all("Set-Cookie"), [])

    def test_login_expires_the_old_scheme_cookie(self):
        token = self.server.token
        legacy = "swe_dash_" + hashlib.sha256(token.encode()).hexdigest()[:10]
        response = self.server.get(f"/?token={token}", {"Cookie": f"{legacy}={token}; theme=dark; swe_dash_8766=x; "
                                                                  f"swe_dash_0123456789=old; {legacy}={token}"},
                                   auth=False)  # (swe_dash_8766: another dashboard's session, left alone)
        self.assertEqual(response.status, 303)
        cookies = response.all("Set-Cookie")
        self.assertEqual(cookies[0].split(";")[0], self.server.cookie())
        self.assertEqual(sorted(cookies[1:]), sorted(f"{name}=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict"
                                                     for name in (legacy, "swe_dash_0123456789")))
        for cookie in cookies:
            self.assertNotIn(token, cookie)

    def test_header_auth(self):
        response = self.server.get("/data/index.json", {"X-Dashboard-Token": f"  {self.server.token} "}, auth=False)
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.body), INDEX_JSON)

    def test_stale_url_token_with_a_valid_cookie(self):
        response = self.server.get("/?token=stale", {"Cookie": self.server.cookie()}, auth=False)
        self.assertEqual(response.status, 303)
        self.assertEqual(response.header("Location"), "/")
        self.assertEqual(response.all("Set-Cookie"), [])  # the stale token is never stored

    def test_serve_only_writes_no_status_file(self):
        self.assertFalse((self.root / "data/server_status.json").exists())


class PathTest(ServerCase):
    def test_traversal_is_blocked(self):
        for path in [
            "/static/../secret.json", "/data/../secret.json", "/data/../state/collector.json",
            "/static/../state/server.json", "/data/task/../../secret.txt", "/../secret.txt", "/static/../../etc/passwd",
            "/static/%2e%2e/secret.json", "/static/%2E%2E/secret.json", "/data/%2e%2e%2fsecret.json",
            "/data/..%2fsecret.json", "/data%2f..%2fsecret.json", "/static/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "/static/%252e%252e/secret.json", "/static/..%5csecret.json", "/static/%5c..%5csecret.json",
            "/static/..\\secret.json", "/static/app.js%00.json", "/data/%00index.json", "/data/index.json%00",
            "/%2e%2e/secret.json", "/static/./app.js", "/static/%2e/app.js", "/static//app.js", "/data/%2findex.json",
            "/data/escape.json", "/static/escape.json", "/data/pipe.json", "/static/%c0%ae%c0%ae/secret.json",
        ]:
            with self.subTest(path=path):
                response = self.server.get(path)
                self.assertIn(response.status, (400, 404))
                self.assertSecurityHeaders(response)
                self.assertNoLeak(response)

    def test_state_code_and_hidden_files_are_not_served(self):
        for path in ["/state/server.json", "/state/collector.json", "/state/", "/server.py", "/collect.py",
                     "/secret.json", "/secret.txt", "/static/.hidden.js", "/data/.index.json.123.140.tmp",
                     "/data/%2eindex.json.123.140.tmp", "/static/notes.md", "/data/notes.txt", "/README.md",
                     "/favicon.ico", "/data/missing.json"]:
            with self.subTest(path=path):
                response = self.server.get(path)
                self.assertEqual(response.status, 404)
                self.assertNoLeak(response)

    def test_no_directory_listing(self):
        for path in ["/static/", "/static", "/static/sub", "/static/sub/", "/data/", "/data", "/data/task",
                     "/data/task/", "/data/history/", "/static/?C=M;O=D"]:
            with self.subTest(path=path):
                response = self.server.get(path)
                self.assertEqual(response.status, 404)
                for name in (b"app.js", b"index.json", b"task-a", b"fleet.jsonl"):
                    self.assertNotIn(name, response.body)

    def test_content_types_and_aliases(self):
        cases = {"/": ("text/html; charset=utf-8", INDEX_HTML), "/index.html": ("text/html; charset=utf-8", INDEX_HTML),
                 "/static/index.html": ("text/html; charset=utf-8", INDEX_HTML),
                 "/static/app.js": ("text/javascript; charset=utf-8", APP_JS),
                 "/app.js": ("text/javascript; charset=utf-8", APP_JS),
                 "/static/style.css": ("text/css; charset=utf-8", STYLE_CSS),
                 "/style.css": ("text/css; charset=utf-8", STYLE_CSS),
                 "/static/logo.svg": ("image/svg+xml", None),
                 "/data/index.json": ("application/json", json.dumps(INDEX_JSON)),
                 "/data/alias.json": ("application/json", json.dumps(INDEX_JSON)),
                 "/data/task/task-a.json": ("application/json", None),
                 "/data/task/task%2Da.json": ("application/json", None),
                 "/data/task/v1.2-dotted.json": ("application/json", "{}"),
                 "/data/history/fleet.jsonl": ("application/x-ndjson", None),
                 "/data/index.json?cache=1": ("application/json", None)}
        for path, (content_type, body) in cases.items():
            with self.subTest(path=path):
                response = self.server.get(path)
                self.assertEqual(response.status, 200)
                self.assertEqual(response.header("Content-Type"), content_type)
                self.assertEqual(int(response.header("Content-Length")), len(response.body))
                self.assertSecurityHeaders(response, cache="no-cache")
                if body is not None:
                    self.assertEqual(response.body.decode(), body)

    def test_gzip_for_clients_that_accept_it(self):
        original = BIG_JSON.encode()
        for encoding, compressed in [("gzip", True), ("gzip, deflate, br", True), ("*", True), ("gzip;q=0", False),
                                     ("identity", False), (None, False)]:
            with self.subTest(encoding=encoding):
                response = self.server.get("/data/big.json", {"Accept-Encoding": encoding} if encoding else {})
                self.assertEqual(response.status, 200)
                self.assertEqual(response.header("Vary"), "Accept-Encoding")
                self.assertEqual(response.header("Content-Type"), "application/json")
                self.assertEqual(int(response.header("Content-Length")), len(response.body))
                self.assertSecurityHeaders(response, cache="no-cache")
                if compressed:
                    self.assertEqual(response.header("Content-Encoding"), "gzip")
                    self.assertEqual(gzip.decompress(response.body), original)
                    self.assertLess(len(response.body), len(original) // 4)
                else:
                    self.assertIsNone(response.header("Content-Encoding"))
                    self.assertEqual(response.body, original)
        got = self.server.get("/data/big.json", {"Accept-Encoding": "gzip"})
        head = self.server.get("/data/big.json", {"Accept-Encoding": "gzip"}, method="HEAD")
        self.assertEqual(head.body, b"")
        self.assertEqual(head.header("Content-Length"), str(len(got.body)))
        self.assertEqual(head.header("Content-Encoding"), "gzip")
        small = self.server.get("/data/index.json", {"Accept-Encoding": "gzip"})  # under 2 KB: sent as is
        self.assertIsNone(small.header("Content-Encoding"))
        self.assertEqual(json.loads(small.body), INDEX_JSON)

    def test_absolute_form_request_is_refused(self):
        answer = raw_request(self.server.port, b"GET http://127.0.0.1/data/index.json HTTP/1.0\r\n"
                             + f"X-Dashboard-Token: {self.server.token}\r\n\r\n".encode())
        self.assertTrue(answer.startswith(b"HTTP/1.0 400"), answer[:80])
        self.assertNotIn(b"schema", answer)


class ProtocolTest(ServerCase):
    def test_only_get_and_head(self):
        for method in ("POST", "PUT", "DELETE", "PATCH", "OPTIONS"):
            for path, auth in (("/", True), ("/", False), ("/data/index.json", True), ("/healthz", False)):
                with self.subTest(method=method, path=path, auth=auth):
                    response = self.server.get(path, auth=auth, method=method, body=b'{"x": 1}')
                    self.assertEqual(response.status, 405)
                    self.assertEqual(response.header("Allow"), "GET, HEAD")
                    self.assertSecurityHeaders(response)
                    self.assertNotIn(b"schema", response.body)
        self.assertEqual(json.loads((self.root / "data/index.json").read_text()), INDEX_JSON)  # nothing written

    def test_unknown_method_gets_an_error_with_the_headers(self):
        answer = raw_request(self.server.port, b"FOO /data/index.json HTTP/1.0\r\n\r\n").decode("latin-1")
        head = answer.split("\r\n\r\n")[0]
        self.assertTrue(head.startswith("HTTP/1.0 501"), head)
        self.assertIn("Cache-Control: no-store", head)
        self.assertIn("X-Content-Type-Options: nosniff", head)
        self.assertIn("Content-Security-Policy: default-src 'self'", head)

    def test_head_matches_get(self):
        for path in ("/", "/data/index.json", "/static/app.js"):
            with self.subTest(path=path):
                got, head = self.server.get(path), self.server.get(path, method="HEAD")
                self.assertEqual(head.status, 200)
                self.assertEqual(head.body, b"")
                self.assertEqual(head.header("Content-Length"), str(len(got.body)))
                self.assertEqual(head.header("Content-Type"), got.header("Content-Type"))
                self.assertEqual(head.header("ETag"), got.header("ETag"))
                self.assertEqual(head.header("Last-Modified"), got.header("Last-Modified"))
                self.assertSecurityHeaders(head, cache="no-cache")

    def test_security_headers_on_every_kind_of_response(self):
        token = self.server.token
        for path, auth, status, cache in [
            ("/", True, 200, "no-cache"), ("/data/index.json", True, 200, "no-cache"),
            ("/healthz", False, 200, "no-store"),
            (f"/?token={token}", False, 303, "no-store"), ("/", False, 403, "no-store"),
            ("/missing.json", True, 404, "no-store"), ("/static/../secret.json", True, 404, "no-store")]:
            with self.subTest(path=path.replace(token, "<token>")):
                response = self.server.get(path, auth=auth)
                self.assertEqual(response.status, status)
                self.assertSecurityHeaders(response, cache=cache)
                self.assertEqual(response.header("Server"), "swe-task-dashboard")  # no Python version
                if cache == "no-store":  # only files have validators
                    self.assertIsNone(response.header("ETag"))
                    self.assertIsNone(response.header("Last-Modified"))

    def test_malformed_requests_do_not_hurt(self):
        for data, status in [(b"GARBAGE\r\n", b"400"), (b"GET /" + b"a" * 65537, b"414"),
                             (b"GET / HTTP/9.9\r\n\r\n", b"505")]:
            with self.subTest(status=status):
                answer = raw_request(self.server.port, data)
                self.assertIn(status, answer.split(b"\r\n")[0])
                self.assertIn(b"X-Content-Type-Options: nosniff", answer)
        self.assertEqual(self.server.get("/healthz", auth=False).status, 200)  # still serving

    def test_server_json(self):
        info_file = self.root / "state/server.json"
        self.assertEqual(stat.S_IMODE(info_file.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(info_file.parent.stat().st_mode), 0o700)
        info = read_json(info_file)
        self.assertEqual(info["bind"], "127.0.0.1")
        self.assertEqual(info["host"], socket.gethostname())
        self.assertEqual(info["pid"], self.server.process.pid)
        self.assertIsNone(info["slurm_job_id"])
        self.assertIsInstance(info["port"], int)
        self.assertGreaterEqual(len(info["token"]), 43)  # secrets.token_urlsafe(32)
        self.assertRegex(info["token"], r"^[A-Za-z0-9_-]+$")
        self.assertEqual(info["url"], f"http://127.0.0.1:{info['port']}/?token={info['token']}")
        self.assertEqual(info["local_url"], f"http://localhost:{info['port']}/?token={info['token']}")
        self.assertIsInstance(info["started_at"], float)
        self.assertFalse(info["collect"])
        self.assertEqual(info["cookie"], f"swe_dash_{info['port']}")  # the name only: the value derives from the token
        self.assertNotIn(session_value(info["token"], info["port"]), info_file.read_text())
        self.assertNotIn(info["token"], self.server.output())


class CacheTest(ServerCase):
    """The files of static/ and data/ may be kept, but are revalidated every time: an unchanged file costs a 304."""

    def etag(self, relative):
        st = (self.root / relative).stat()
        return f'W/"{st.st_size:x}-{st.st_mtime_ns:x}-{st.st_ino:x}"'

    def test_files_carry_validators(self):
        for path, relative in [("/", "static/index.html"), ("/static/app.js", "static/app.js"),
                               ("/style.css", "static/style.css"), ("/data/index.json", "data/index.json"),
                               ("/data/history/fleet.jsonl", "data/history/fleet.jsonl")]:
            for method in ("GET", "HEAD"):
                with self.subTest(path=path, method=method):
                    response = self.server.get(path, method=method)
                    self.assertEqual(response.status, 200)
                    self.assertSecurityHeaders(response, cache="no-cache")  # nosniff and the CSP stay
                    self.assertEqual(response.header("ETag"), self.etag(relative))  # size-mtime-inode
                    modified = email.utils.parsedate_to_datetime(response.header("Last-Modified"))
                    self.assertEqual(modified.timestamp(), int((self.root / relative).stat().st_mtime))
        # One weak tag for the plain and the gzipped body (the same content).
        plain = self.server.get("/data/big.json")
        packed = self.server.get("/data/big.json", {"Accept-Encoding": "gzip"})
        self.assertEqual(packed.header("Content-Encoding"), "gzip")
        self.assertEqual(packed.header("ETag"), plain.header("ETag"))
        self.assertEqual(self.server.get("/data/big.json", {"Accept-Encoding": "gzip",
                                                            "If-None-Match": plain.header("ETag")}).status, 304)

    def test_not_modified(self):
        first = self.server.get("/data/index.json")
        etag, modified = first.header("ETag"), first.header("Last-Modified")
        for headers in [{"If-None-Match": etag}, {"If-None-Match": etag[2:]}, {"If-None-Match": f'"other", {etag}'},
                        {"If-None-Match": "*"}, {"If-Modified-Since": modified},
                        {"If-None-Match": etag, "If-Modified-Since": "Thu, 01 Jan 1970 00:00:00 GMT"},
                        {"If-None-Match": etag, "Cookie": self.server.cookie(), "X-Dashboard-Token": ""}]:
            for method in ("GET", "HEAD"):
                with self.subTest(headers=headers, method=method):
                    response = self.server.get("/data/index.json", headers, method=method)
                    self.assertEqual(response.status, 304)
                    self.assertEqual(response.body, b"")
                    self.assertEqual(response.header("ETag"), etag)
                    self.assertEqual(response.header("Last-Modified"), modified)
                    self.assertEqual(response.header("Vary"), "Accept-Encoding")
                    self.assertSecurityHeaders(response, cache="no-cache")
        for headers in [{"If-None-Match": 'W/"other"'}, {"If-None-Match": "junk"},
                        {"If-None-Match": 'W/"other"', "If-Modified-Since": modified},  # If-None-Match decides
                        {"If-Modified-Since": "Thu, 01 Jan 1970 00:00:00 GMT"}, {"If-Modified-Since": "not a date"}]:
            with self.subTest(headers=headers):
                response = self.server.get("/data/index.json", headers)
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.body), INDEX_JSON)
                self.assertEqual(response.header("ETag"), etag)
        page = self.server.get("/")  # static files too, through their aliases
        for path in ("/", "/index.html", "/static/index.html"):
            with self.subTest(path=path):
                self.assertEqual(self.server.get(path, {"If-None-Match": page.header("ETag")}).status, 304)

    def test_a_replaced_file_is_sent_again(self):
        path = self.root / "data/cycle.json"
        path.write_text('{"cycle": 1}')
        first = self.server.get("/data/cycle.json")
        temporary = self.root / "data/.cycle.json.tmp"  # the collector's way: a new file renamed over the old one
        temporary.write_text('{"cycle": 2}')  # (the same size)
        os.replace(temporary, path)
        second = self.server.get("/data/cycle.json", {"If-None-Match": first.header("ETag")})
        self.assertEqual(second.status, 200)
        self.assertEqual(json.loads(second.body), {"cycle": 2})
        self.assertNotEqual(second.header("ETag"), first.header("ETag"))
        before = path.stat()
        path.write_text('{"cycle": 3}')  # rewritten in place: same inode and size, another mtime
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_007))
        third = self.server.get("/data/cycle.json", {"If-None-Match": second.header("ETag"),
                                                     "If-Modified-Since": second.header("Last-Modified")})
        self.assertEqual(third.status, 200)
        self.assertEqual(json.loads(third.body), {"cycle": 3})
        self.assertEqual(self.server.get("/data/cycle.json", {"If-None-Match": third.header("ETag")}).status, 304)

    def test_no_304_without_the_token(self):
        etag = self.server.get("/data/index.json").header("ETag")
        for headers in [{"If-None-Match": etag}, {"If-None-Match": "*"}, {"If-Modified-Since": "Fri, 01 Jan 2100 "
                                                                                                "00:00:00 GMT"}]:
            with self.subTest(headers=headers):
                response = self.server.get("/data/index.json", headers, auth=False)
                self.assertEqual(response.status, 403)
                self.assertIsNone(response.header("ETag"))
                self.assertSecurityHeaders(response)


class HotReloadTest(unittest.TestCase):
    """A changed collect.py is re-imported by the running collector; a broken one leaves the old code running."""

    def test_collector_reloads_changed_module_and_survives_a_broken_one(self):
        spec = importlib.util.spec_from_file_location("dashboard_server_reload_test", SERVER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        source = tmp / "collect.py"

        def write(body, bump):
            source.write_text(body)
            stamp = 1_700_000_000 + bump
            os.utime(source, (stamp, stamp))
        write("def run_cycle(data, state):\n    return {'v': 1}\n", 0)
        collector = module.Collector(source, tmp / "data", tmp / "state", 60, {})
        self.assertEqual(collector.cycle()[0], {"v": 1})
        write("def run_cycle(data, state):\n    return {'v': 2}\n", 10)
        self.assertEqual(collector.cycle()[0], {"v": 2})
        write("def run_cycle(data, state) return\n", 20)  # SyntaxError
        summary, failure, _ = collector.cycle()
        self.assertEqual(summary, {"v": 2})  # the previous module keeps collecting
        self.assertIn("SyntaxError", collector.status["import_error"])


class LimitTest(unittest.TestCase):
    """At most 32 connections are handled at once; a stalled one gives its slot back after the 15 s socket timeout."""

    def test_handler_slots_503_and_timeout(self):
        tmp = Path(tempfile.mkdtemp(prefix="dash-limit-test-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        server = ServerProcess(make_root(tmp / "root"), "--no-collect")
        self.addCleanup(server.stop)
        idle = []  # connections that never send a request: each holds a handler slot
        self.addCleanup(lambda: [sock.close() for sock in idle])
        started = time.monotonic()
        for _ in range(32):
            idle.append(socket.create_connection(("127.0.0.1", server.port), timeout=30))
        begin = time.monotonic()
        answer = raw_request(server.port, b"GET /healthz HTTP/1.0\r\n\r\n")  # the 33rd: refused at once
        self.assertLess(time.monotonic() - begin, 3)
        head, _, body = answer.partition(b"\r\n\r\n")
        head = head.decode("latin-1")
        self.assertTrue(head.startswith("HTTP/1.0 503 "), head)
        for line in ("Retry-After: 2", "Connection: close", "Cache-Control: no-store",
                     "X-Content-Type-Options: nosniff", "Content-Security-Policy: default-src 'self'",
                     f"Content-Length: {len(body)}"):
            self.assertIn(line, head)
        self.assertIn(b"503 busy", body)
        response = server.get("/data/index.json")  # the same for an authorized data request
        self.assertEqual(response.status, 503)
        self.assertNotIn(b"schema", response.body)
        idle.pop().close()  # a client goes away: its slot comes back
        wait_until(lambda: server.get("/data/index.json").status == 200, timeout=10, what="a free slot",
                   output=server.output)
        # The others never send anything: the socket timeout ends them after 15 s, and every slot is free again.
        self.assertEqual(idle[0].recv(1), b"")
        self.assertGreaterEqual(time.monotonic() - started, 14)
        self.assertLess(time.monotonic() - started, 30)
        wait_until(lambda: all(server.get("/healthz", auth=False).status == 200 for _ in range(32)), timeout=10,
                   what="all slots free", output=server.output)
        log = server.output()
        self.assertIn("busy: all 32 handler slots taken -> 503", log)
        self.assertIn("Request timed out", log)


# ------------------------------------------------------------------------------------------- start, stop, collect

FAKE_COLLECT = f'''
"""Fake collector: cycle 2 raises, cycle 3 reports a source error."""
import json
from pathlib import Path

CALLS = []


def run_cycle(data_dir, state_dir):
    CALLS.append(1)
    number = len(CALLS)
    Path(data_dir, "fake.json").write_text(json.dumps({{"cycle": number}}))
    if number == 2:
        raise RuntimeError("boom {FAKE_KEY}")
    errors = ["source x: broken"] if number == 3 else []
    return {{"ok": not errors, "seconds": 0.01, "errors": errors, "runs": 1, "tasks": 2}}
'''


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dash-life-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = make_root(self.tmp / "root")

    def start(self, *args, **options):
        server = ServerProcess(self.root, *args, **options)
        self.addCleanup(server.stop)
        return server

    def test_reuse_token_and_sigterm(self):
        first = self.start("--no-collect")
        self.assertEqual(first.stop(), 0)
        info = read_json(self.root / "state/server.json")
        self.assertIsInstance(info.get("stopped_at"), float)  # SIGTERM: marked stopped
        self.assertIn("swe-task dashboard: stopped", first.output())
        # The same token on the same port: the session cookie of the first server still works (only the token is
        # kept; the cookie derives from it and the port).
        second = self.start("--no-collect", "--reuse-token", "--port", str(first.port))
        self.assertEqual((second.token, second.port), (first.token, first.port))
        self.assertEqual(second.info["cookie"], first.info["cookie"])
        self.assertEqual(second.get("/", {"Cookie": first.cookie()}, auth=False).status, 200)  # old cookie works
        second.stop()
        # The same token on another port: a session is per port.
        third = self.start("--no-collect", "--reuse-token", "--port", str(free_port(exclude={first.port})))
        self.assertEqual(third.token, first.token)
        self.assertNotEqual(third.info["cookie"], first.info["cookie"])
        self.assertEqual(third.get("/", {"Cookie": first.cookie()}, auth=False).status, 403)
        self.assertEqual(third.get("/", {"Cookie": third.cookie()}, auth=False).status, 200)
        third.stop()
        fourth = self.start("--no-collect", "--port", str(first.port))
        self.assertNotEqual(fourth.token, first.token)
        self.assertEqual(fourth.get("/", {"Cookie": first.cookie()}, auth=False).status, 403)

    def test_token_never_logged(self):
        # A fresh server: its request log has its whole per-minute budget, so nothing here is suppressed.
        server = self.start("--no-collect")
        token, session = server.token, session_value(server.token, server.port)
        self.assertEqual(server.get(f"/?token={token}", auth=False).status, 303)
        self.assertEqual(server.get(f"/data/x.json?token={token[::-1]}", auth=False).status, 403)
        self.assertEqual(server.get(f"/nothing?a=1&token={token}", auth=False).status, 303)
        # The token mistyped into the path: whole, glued, cut short, with a typo, twice, as token=, with a valid
        # header (404); and one the 200-character path limit of the log line cuts in the middle.
        for path, auth, status in [
            (f"/{token}", False, 403), (f"/x{token}y.json", False, 403), (f"/{token[:20]}", False, 403),
            (f"/{token[:15]}~{token[16:]}", False, 403), (f"/{token}/{token}", False, 403),
            (f"/token={token}", False, 403), (f"/data/{token}.json", True, 404),
            ("/" + "a" * 185 + token, False, 403), (f"/{session}", False, 403),
        ]:
            self.assertEqual(server.get(path, auth=auth).status, status)
        raw_request(server.port, f"BAD?token={token} x\r\n".encode())
        raw_request(server.port, f"GET /?token={token} HTTP/9.9\r\n\r\n".encode())
        raw_request(server.port, f"GET /{token} x HTTP/1.0\r\n\r\n".encode())  # the request line is in the message
        raw_request(server.port, f"BAD{token} x\r\n".encode())
        log = server.output()
        for line in ("login: session cookie set", "GET /data/x.json -> 403", "Bad HTTP/0.9 request type",
                     "Invalid HTTP version", "GET /<token> -> 403", "GET /x<token>y.json -> 403",
                     "GET /<token>~<token> -> 403", "GET /token=<redacted> -> 403", "GET /data/<token>.json -> 404",
                     "GET /<session> -> 403", "Bad request syntax ('GET /<token> x HTTP/1.0')", "('BAD<token>')"):
            self.assertIn(line, log)
        self.assertNotIn("suppressed", log)
        for needle in (token, token[::-1], token[:8], token[-8:], token[17:25], token[:14], session, session[:8]):
            self.assertNotIn(needle, log)

    def test_request_log_is_rate_limited(self):
        server = self.start("--no-collect")
        for _ in range(45):
            self.assertEqual(server.get("/missing", auth=False).status, 403)
        self.assertEqual(server.output().count("GET /missing -> 403"), 30)  # LOG_LINES_PER_MIN
        self.assertEqual(server.get("/healthz", auth=False).status, 200)

    def test_reuse_token_without_a_previous_one(self):
        server = self.start("--no-collect", "--reuse-token")
        self.assertGreaterEqual(len(server.token), 43)
        self.assertIn("no usable token", server.output())

    def test_non_loopback_host_refused_outside_slurm(self):
        run = subprocess.run([PY, "-I", "-B", str(SERVER), "--root", str(self.root), "--host", "0.0.0.0", "--port",
                              "0", "--no-collect"], env=clean_env(), capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 2)
        self.assertIn("only inside the Slurm job", run.stderr)
        self.assertFalse((self.root / "state/server.json").exists())

    def test_taken_port_falls_back(self):
        blocker = socket.socket()
        self.addCleanup(blocker.close)
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        taken = blocker.getsockname()[1]
        server = self.start("--no-collect", "--port", str(taken))
        self.assertNotEqual(server.port, taken)
        self.assertIn(f"port {taken} is in use", server.output())
        self.assertEqual(server.get("/healthz", auth=False).status, 200)

    def test_collector_cycles_failures_and_status(self):
        (self.root / "collect.py").write_text(FAKE_COLLECT)
        server = self.start("--interval", "1")
        status_file = self.root / "data/server_status.json"
        wait_until(lambda: (self.root / "data/fake.json").exists(), timeout=10, what="first cycle at startup",
                   output=server.output)
        status = wait_until(lambda: (lambda doc: doc if (doc.get("collector") or {}).get("cycles", 0) >= 4 else None)(
            read_json(status_file)), timeout=30, what="4 collector cycles", output=server.output)
        collector = status["collector"]
        self.assertEqual(collector["failures"], 1)
        self.assertIn("run_cycle: RuntimeError: boom", collector["last_failure"])
        self.assertIsNone(collector["import_error"])
        self.assertEqual(status["server"]["pid"], server.process.pid)
        log = server.output()
        self.assertRegex(log, r"collect cycle 1: [\d.]+ s, 0 errors \(1 runs, 2 tasks\)")
        self.assertRegex(log, r"collect cycle 2 FAILED after [\d.]+ s, 0 errors: run_cycle: RuntimeError: boom")
        self.assertRegex(log, r"collect cycle 3: [\d.]+ s, 1 errors \(1 runs, 2 tasks\); first: source x: broken")
        self.assertRegex(log, r"collect cycle 4: [\d.]+ s, 0 errors")
        for text in (log, status_file.read_text()):
            self.assertNotIn("Q" * 40, text)  # the key in the exception message is masked
            self.assertNotIn(server.token, text)
        self.assertEqual(server.get("/healthz", auth=False).status, 200)  # serving throughout
        served = server.get("/data/server_status.json")
        self.assertEqual(served.status, 200)
        self.assertNotIn(server.token.encode(), served.body)
        self.assertEqual(server.stop(), 0)
        self.assertEqual(read_json(status_file)["collector"]["state"], "stopped")  # nobody collects any more

    def test_broken_collector_import_is_retried(self):
        (self.root / "collect.py").write_text("def run_cycle(:\n")
        server = self.start("--interval", "1")
        status_file = self.root / "data/server_status.json"
        status = wait_until(lambda: (lambda doc: doc if (doc.get("collector") or {}).get("import_error") else None)(
            read_json(status_file)), timeout=10, what="import error recorded", output=server.output)
        self.assertIn("SyntaxError", status["collector"]["import_error"])
        self.assertEqual(server.get("/data/index.json").status, 200)  # still serving
        (self.root / "collect.py").write_text(FAKE_COLLECT)  # fixed: the next cycle imports it
        status = wait_until(lambda: (lambda doc: doc if (doc.get("collector") or {}).get("last_ok") else None)(
            read_json(status_file)), timeout=15, what="a successful cycle after the fix", output=server.output)
        self.assertIsNone(status["collector"]["import_error"])
        self.assertGreaterEqual(status["collector"]["failures"], 1)
        self.assertIn("FAILED", server.output())

    def test_missing_collector_keeps_serving(self):
        server = self.start("--interval", "1", "--collector", str(self.tmp / "nowhere/collect.py"))
        status = wait_until(lambda: (lambda doc: doc if (doc.get("collector") or {}).get("import_error") else None)(
            read_json(self.root / "data/server_status.json")), timeout=10, what="import error", output=server.output)
        self.assertIn("FileNotFoundError", status["collector"]["import_error"])
        self.assertEqual(server.get("/", {"Cookie": server.cookie()}, auth=False).status, 200)


# ------------------------------------------------------------------------------------------ launch.sh, sbatch file

FAKE_SLURM = r'''#!/usr/bin/python3 -IB
"""Fake Slurm command for tests/test_server.py (one file, symlinked as each command): canned answers from the files
next to it, every call recorded in calls.jsonl with the names of its environment variables."""
import json
import os
from pathlib import Path
import sys

HERE = Path(sys.argv[0]).parent
NAME, ARGS = Path(sys.argv[0]).name, sys.argv[1:]
with open(HERE / "calls.jsonl", "a") as stream:
    stream.write(json.dumps({"cmd": NAME, "args": ARGS, "env": sorted(os.environ)}) + "\n")


def canned(name, default=""):
    path = HERE / name
    return path.read_text() if path.exists() else default


def option(flag):
    for index, arg in enumerate(ARGS):
        if arg == flag and index + 1 < len(ARGS):
            return ARGS[index + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


if NAME == "squeue":
    if option("-j"):
        line = canned("squeue_job.txt").strip()  # STATE|NODE|REASON|START of the submitted job
        if line == "INVALID":  # purged from slurmctld
            sys.exit("slurm_load_jobs error: Invalid job id specified")
        if line:
            print(line.split("|")[1] if option("-o") == "%N" else line)
    elif option("-n"):
        sys.stdout.write(canned("squeue_list.txt"))
elif NAME == "sinfo":
    if option("-t"):
        print("3")
    else:
        limit = json.loads(canned("limits.json", "{}")).get(option("-p"))
        if limit:
            print(limit)
elif NAME == "sbatch":
    if "--test-only" in ARGS:
        partition = option("--partition")
        start = json.loads(canned("starts.json", "{}")).get(partition, "2026-10-18T09:00:00")
        print(f"sbatch: Job 4 to start at {start} using 2 processors on nodes n1 in partition {partition}",
              file=sys.stderr)
    else:
        print(canned("sbatch_out.txt", "424242;cluster").strip())
elif NAME == "scontrol" and ARGS[:2] == ["show", "partition"]:
    for partition in ("cpu_datamover", "cpu_dataprocessing", "cpu_interactive"):
        print(f"PartitionName={partition} QoS=N/A MaxTime=UNLIMITED")
'''
LIMITS = {"cpu_datamover": "infinite", "cpu_dataprocessing": "infinite", "cpu_interactive": "1-00:00:00"}
STARTS = {"cpu_datamover": "2026-10-17T21:00:00", "cpu_dataprocessing": "2026-10-17T16:00:00",
          "cpu_interactive": "2026-10-17T15:00:00"}


class ScriptsTest(unittest.TestCase):
    """launch.sh is copied into a temporary dashboard directory, so it reads and writes only there."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dash-launch-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.dash = make_root(self.tmp / "dashboard")
        for source in (LAUNCH, SBATCH, SERVER):
            shutil.copy2(source, self.dash / source.name)
        self.bin = self.tmp / "fake-slurm-bin"
        self.bin.mkdir()
        (self.bin / "fake_slurm.py").write_text(FAKE_SLURM)
        (self.bin / "fake_slurm.py").chmod(0o755)
        for name in ("squeue", "sinfo", "sbatch", "scancel", "scontrol", "sacctmgr"):
            os.symlink("fake_slurm.py", self.bin / name)
        (self.bin / "limits.json").write_text(json.dumps(LIMITS))
        (self.bin / "starts.json").write_text(json.dumps(STARTS))

    def launch(self, *args, timeout=60, **env):
        env = clean_env(DASHBOARD_SLURM_BIN=str(self.bin), MY_API_KEY="leak-me", GH_TOKEN="leak-me-too", **env)
        return subprocess.run(["/bin/bash", str(self.dash / "launch.sh"), *args], env=env, capture_output=True,
                              text=True, timeout=timeout, stdin=subprocess.DEVNULL)

    def calls(self, command=None):
        try:
            rows = [json.loads(line) for line in (self.bin / "calls.jsonl").read_text().splitlines()]
        except OSError:
            rows = []
        return [row for row in rows if command is None or row["cmd"] == command]

    def submissions(self):
        return [row for row in self.calls("sbatch") if "--test-only" not in row["args"]]

    def running_server(self, job="424242"):
        """A real server in the temporary dashboard directory, as if job `job` ran it on node 127.0.0.1."""
        server = ServerProcess(self.dash, "--no-collect", env=clean_env(SLURM_JOB_ID=job))
        self.addCleanup(server.stop)
        (self.bin / "squeue_job.txt").write_text("RUNNING|127.0.0.1|None|2026-10-17T16:00:00\n")
        return server

    def test_dry_run_excludes_cpu_interactive_for_seven_days(self):
        run = self.launch("start", "--dry-run")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("skip cpu_interactive: MaxTime 1-00:00:00 < --time 7-00:00:00", run.stderr)
        probe = json.loads(run.stdout.split("probe: ", 1)[1].splitlines()[0])
        self.assertEqual([item["partition"] for item in probe["probes"]], ["cpu_datamover", "cpu_dataprocessing"])
        self.assertEqual(probe["partition"], "cpu_dataprocessing")  # earliest estimated start
        dry = [line for line in run.stdout.splitlines() if line.startswith("DRY: ")]
        self.assertEqual(len(dry), 1)
        for part in ("sbatch --parsable --partition=cpu_dataprocessing --job-name=swe-task-dashboard",
                     "--time=7-00:00:00", f"{self.dash}/dashboard.sbatch --port 8765"):
            self.assertIn(part, dry[0])
        self.assertEqual(self.submissions(), [])
        self.assertTrue(self.calls("sbatch"))  # the --test-only probes
        self.assertFalse((self.dash / "logs").exists())  # a dry run writes nothing

    def test_dry_run_short_walltime_probes_cpu_interactive(self):
        run = self.launch("--time", "12:00:00", "--dry-run", "--port", "9001", "--reuse-token")
        self.assertEqual(run.returncode, 0, run.stderr)
        probe = json.loads(run.stdout.split("probe: ", 1)[1].splitlines()[0])
        self.assertEqual(len(probe["probes"]), 3)
        self.assertEqual(probe["partition"], "cpu_interactive")
        self.assertIn("--partition=cpu_interactive", run.stdout)
        self.assertIn("dashboard.sbatch --port 9001 --reuse-token", run.stdout)

    def test_reuse_token_keeps_the_port(self):
        # Only the token is kept: the session cookie derives from it and the port, so --reuse-token asks for the
        # previous server's port unless --port says otherwise.
        (self.dash / "state/server.json").write_text(json.dumps({"port": 8799, "token": "t" * 43}))
        run = self.launch("start", "--dry-run", "--reuse-token")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("dashboard.sbatch --port 8799 --reuse-token", run.stdout)
        self.assertNotIn("note:", run.stderr)
        run = self.launch("start", "--dry-run", "--reuse-token", "--port", "9001")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("dashboard.sbatch --port 9001 --reuse-token", run.stdout)
        self.assertIn("note: the previous server listened on port 8799", run.stderr)
        run = self.launch("start", "--dry-run")  # a new token: the default port
        self.assertIn("dashboard.sbatch --port 8765 ", run.stdout)
        for run in (run, self.launch("start", "--dry-run", "--reuse-token")):
            self.assertNotIn("t" * 43, run.stdout + run.stderr)
        (self.dash / "state/server.json").unlink()
        run = self.launch("start", "--dry-run", "--reuse-token")  # nothing to reuse
        self.assertIn("dashboard.sbatch --port 8765 --reuse-token", run.stdout)

    def test_start_reuse_token_notes_a_changed_port(self):
        server = self.running_server()  # writes state/server.json first, as the previous server would have
        run = self.launch("start", "--reuse-token", "--port", "1")
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(self.submissions()[0]["args"][-3:], ["--port", "1", "--reuse-token"])
        self.assertIn(f"note: the previous server listened on port {server.port}", run.stderr)
        self.assertIn(f"note: port 1 was taken on the node, so the server took {server.port}", run.stderr)
        self.assertNotIn(server.token, run.stderr)

    def test_partition_override(self):
        run = self.launch("--partition", "cpu_interactive")
        self.assertEqual(run.returncode, 1)
        self.assertIn("no partition among cpu_interactive allows --time 7-00:00:00", run.stderr)
        run = self.launch("--partition", "cpu_interactive", "--time", "1-00:00:00", "--dry-run")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("--partition=cpu_interactive", run.stdout)
        self.assertEqual(self.calls("sbatch"), [])  # no probe when the partition is given

    def test_refuses_when_a_dashboard_job_exists(self):
        (self.bin / "squeue_list.txt").write_text("777|RUNNING|cpu_datamover|cpu1-00010|1:00|6-23:59:00|None\n")
        run = self.launch("start")
        self.assertEqual(run.returncode, 1)
        self.assertIn("already pending or running", run.stderr)
        self.assertIn("job 777  RUNNING", run.stderr)
        self.assertEqual(self.calls("sbatch"), [])
        run = self.launch("start", "--dry-run")  # a dry run says so and still shows the plan
        self.assertEqual(run.returncode, 0)
        self.assertIn("a real start would refuse", run.stderr)

    def test_bad_arguments(self):
        for args in (["--time", "soon"], ["--port", "70000"], ["--bogus"], ["stop", "--now"], ["status", "x"],
                     ["--time"]):
            with self.subTest(args=args):
                run = self.launch(*args)
                self.assertNotEqual(run.returncode, 0)
                self.assertEqual(self.submissions(), [])

    def test_start_submits_waits_and_prints_how_to_connect(self):
        server = self.running_server()
        run = self.launch("start")
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        submissions = self.submissions()
        self.assertEqual(len(submissions), 1)
        self.assertEqual(submissions[0]["args"], [
            "--parsable", "--partition=cpu_dataprocessing", "--job-name=swe-task-dashboard", "--time=7-00:00:00",
            f"{self.dash}/dashboard.sbatch", "--port", "8765"])
        for row in self.calls():  # scrubbed environment for every Slurm command
            self.assertEqual(set(row["env"]) - {"PWD", "SHLVL", "_"},
                             {"HOME", "USER", "LOGNAME", "LANG", "SLURM_CONF", "PATH"}, row)
        port, token = server.port, server.token
        self.assertIn("Submitted swe-task-dashboard job 424242 to cpu_dataprocessing", run.stdout)
        self.assertIn(f"serves on 127.0.0.1:{port} (/healthz from", run.stdout)
        self.assertIn(": ok)", run.stdout)
        self.assertIn(f"enter 127.0.0.1:{port}", run.stdout)
        self.assertIn(f"http://localhost:{port}/?token={token}", run.stdout)
        self.assertIn(f"ssh -L {port}:127.0.0.1:{port} {ME.pw_name}@", run.stdout)
        self.assertIn(f"curl -sS -H @{ME.pw_dir}/.dashboard-header http://localhost:{port}/data/index.json",
                      run.stdout)  # tools: the header file, never the token on a command line
        self.assertNotIn(token, run.stderr)
        self.assertNotIn("note:", run.stderr)  # no --reuse-token, no port note
        record = json.loads((self.dash / "logs/launch.jsonl").read_text().splitlines()[-1])
        self.assertEqual(record["job_id"], "424242")
        self.assertEqual(record["partition"], "cpu_dataprocessing")
        self.assertEqual(record["partition_probe"]["partition"], "cpu_dataprocessing")
        self.assertNotIn(token, (self.dash / "logs/launch.jsonl").read_text())

    def test_start_reports_a_job_that_died(self):
        (self.dash / "logs").mkdir()
        (self.dash / "logs/swe-task-dashboard-424242.out").write_text("Traceback: something broke\n")
        for listing, said in [("", "no longer queued"), ("FAILED|cpu1-00010|NonZeroExitCode|2026-10-17T16:00:00",
                                                         "ended (FAILED)"), ("INVALID", "Invalid job id")]:
            with self.subTest(listing=listing):
                (self.bin / "squeue_job.txt").write_text(listing)
                started = time.monotonic()
                run = self.launch("start")
                self.assertEqual(run.returncode, 1)
                self.assertIn(said, run.stderr)
                self.assertIn("something broke", run.stderr)
                self.assertLess(time.monotonic() - started, 30)  # at once, not after --wait

    def test_status(self):
        server = self.running_server()
        (self.bin / "squeue_list.txt").write_text(
            "424242|RUNNING|cpu_dataprocessing|127.0.0.1|0:10|6-23:59:50|None\n")
        run = self.launch("status")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("job 424242  RUNNING  partition cpu_dataprocessing", run.stdout)
        self.assertIn(f"http://localhost:{server.port}/?token={server.token}", run.stdout)
        self.assertIn("collector: no data/server_status.json yet", run.stdout)
        (self.bin / "squeue_list.txt").write_text("")
        run = self.launch("status")
        self.assertIn("No swe-task-dashboard job", run.stdout)
        self.assertNotIn(server.token, run.stdout)

    def test_stop(self):
        (self.bin / "squeue_list.txt").write_text("555|PENDING|cpu_datamover||0:00|7-00:00:00|Priority\n")
        run = self.launch("stop", "--dry-run")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn(f"DRY: scancel --user={ME.pw_name} --name=swe-task-dashboard", run.stdout)
        self.assertIn("reason Priority", run.stdout)
        self.assertEqual(self.calls("scancel"), [])
        run = self.launch("stop")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual([row["args"] for row in self.calls("scancel")],
                         [[f"--user={ME.pw_name}", "--name=swe-task-dashboard"]])
        (self.bin / "squeue_list.txt").write_text("")
        run = self.launch("stop")
        self.assertIn("No swe-task-dashboard job", run.stdout)
        self.assertEqual(len(self.calls("scancel")), 1)

    def local(self, *args):
        env = clean_env(DASHBOARD_SLURM_BIN=str(self.bin), MY_API_KEY="leak-me")
        server = ServerProcess(self.dash, env=env, info_name="server-local.json",
                               command=["/bin/bash", str(self.dash / "launch.sh"), "local", *args])
        self.addCleanup(server.stop)
        return server

    def test_local_serves_on_loopback(self):
        server = self.local("--no-collect", "--port", "0")
        self.assertEqual(server.info["bind"], "127.0.0.1")
        self.assertFalse(server.info["collect"])
        wait_until(lambda: "Dashboard: http://127.0.0.1:" in server.output(), timeout=10, what="the printed URL")
        self.assertIn(f"Dashboard: http://127.0.0.1:{server.port}/?token={server.token}", server.output())
        environment = Path(f"/proc/{server.process.pid}/environ").read_bytes().split(b"\0")
        names = {item.split(b"=", 1)[0].decode() for item in environment if item}
        self.assertNotIn("MY_API_KEY", names)
        self.assertNotIn("DASHBOARD_SLURM_BIN", names)
        self.assertEqual(server.get("/").status, 200)
        self.assertFalse((self.dash / "state/server.json").exists())  # the Slurm job's info file is untouched

    def test_local_is_serve_only_while_the_job_collects(self):
        (self.bin / "squeue_list.txt").write_text("424242|RUNNING|cpu_dataprocessing|cpu1-00010|1:00|6-23:00:00|None\n")
        server = self.local("--port", "0")
        self.assertFalse(server.info["collect"])
        self.assertIn("serving its data only", server.output())

    def test_local_reuse_token_keeps_the_port(self):
        first = self.local("--no-collect")
        port, token, cookie = first.port, first.token, first.cookie()
        self.assertEqual(first.stop(), 0)
        second = self.local("--no-collect", "--reuse-token")  # no --port: the previous one
        self.assertEqual((second.port, second.token), (port, token))
        self.assertEqual(second.get("/", {"Cookie": cookie}, auth=False).status, 200)  # the old session is valid

    def test_header_file_for_curl(self):
        home = self.tmp / "home"
        home.mkdir()
        default = home / ".dashboard-header"
        run = self.launch("header", HOME=str(home))  # no server has started yet
        self.assertEqual(run.returncode, 1)
        self.assertIn("no token in", run.stderr)
        self.assertFalse(default.exists())
        server = self.running_server()
        run = self.launch("header", HOME=str(home))
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(default.read_text(), f"X-Dashboard-Token: {server.token}\n")
        self.assertEqual(stat.S_IMODE(default.stat().st_mode), 0o600)
        self.assertIn(f"curl -sS -H @{default} http://localhost:{server.port}/data/index.json", run.stdout)
        self.assertNotIn(server.token, run.stdout + run.stderr)
        other = self.tmp / "other-header"
        other.write_text("stale\n")
        other.chmod(0o644)
        run = self.launch("header", "--file", str(other), HOME=str(home))
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(other.read_text(), default.read_text())  # replaced, and private
        self.assertEqual(stat.S_IMODE(other.stat().st_mode), 0o600)
        run = self.launch("header", "--local", "--file", str(other), HOME=str(home))  # no local server
        self.assertEqual(run.returncode, 1)
        self.assertEqual(other.read_text(), default.read_text())
        self.assertEqual(sorted(path.name for path in home.iterdir()), [".dashboard-header"])  # no temp file left
        # The README's way: curl reads the header from the file; the token is on no command line.
        if not os.path.exists("/usr/bin/curl"):
            self.skipTest("no /usr/bin/curl")
        for header, status in [(default, 200), (self.tmp / "no-header", 403)]:
            if not header.exists():
                header.write_text("X-Other: 1\n")
            curl = subprocess.run(["/usr/bin/curl", "-q", "-sS", "--noproxy", "*", "-o", "/dev/null", "-w",
                                   "%{http_code}", "-H", f"@{header}",
                                   f"http://127.0.0.1:{server.port}/data/index.json"],
                                  env=clean_env(), capture_output=True, text=True, timeout=30)
            self.assertEqual((curl.returncode, curl.stdout), (0, str(status)), curl.stderr)

    def test_sbatch_directives(self):
        text = SBATCH.read_text()
        directives = dict(re.findall(r"^#SBATCH (--[a-z-]+)=(\S+)", text, re.M))
        self.assertEqual(directives["--account"], "nemotron_agents_dev")
        self.assertEqual(directives["--job-name"], "swe-task-dashboard")
        self.assertEqual(directives["--cpus-per-task"], "2")
        self.assertEqual(directives["--mem"], "8G")
        self.assertEqual(directives["--time"], "7-00:00:00")
        self.assertEqual(directives["--export"], "NONE")
        output = Path(directives["--output"])
        self.assertEqual(output.name, "%x-%j.out")
        self.assertEqual(os.path.realpath(output.parent), os.path.realpath(DASHBOARD / "logs"))
        self.assertNotIn("--partition", directives)
        self.assertRegex(text, r'exec /usr/bin/python3 -I -B "\$D/server.py" --host 0\.0\.0\.0 '
                               r'--port "\$\{DASHBOARD_PORT:-8765\}" --interval 120 "\$@"')
        for script in (LAUNCH, SBATCH):
            check = subprocess.run(["/bin/bash", "-n", str(script)], capture_output=True, text=True)
            self.assertEqual(check.returncode, 0, check.stderr)

    def test_sbatch_scrubs_secret_like_variables(self):
        planted = {"MY_API_KEY": "k", "GITHUB_TOKEN": "t", "DB_PASSWORD": "p", "AWS_SECRET_ACCESS_KEY": "s",
                   "GOOGLE_APPLICATION_CREDENTIALS": "c", "hub_api_key": "l", "NVAPI_KEY_NEW": "n"}
        env = clean_env(SLURM_JOB_ID="999", HARMLESS="kept", **planted)
        # Runs the real dashboard.sbatch, whose extra arguments point server.py at the temporary root.
        server = ServerProcess(self.dash, env=env, command=["/bin/bash", str(SBATCH), "--root", str(self.dash),
                                                            "--host", "127.0.0.1", "--port", "0", "--no-collect"])
        self.addCleanup(server.stop)
        environment = dict(item.split(b"=", 1) for item in
                           Path(f"/proc/{server.process.pid}/environ").read_bytes().split(b"\0") if b"=" in item)
        names = {name.decode() for name in environment}
        self.assertFalse(names & set(planted), names & set(planted))
        self.assertEqual(environment[b"HARMLESS"], b"kept")
        self.assertTrue(environment[b"PATH"].startswith(b"/cm/shared/apps/slurm/current/bin:"))
        self.assertIn(b"SLURM_CONF", environment)
        self.assertEqual(server.info["slurm_job_id"], "999")
        self.assertIn("dashboard job 999 on", server.output())
        self.assertEqual(server.get("/healthz", auth=False).status, 200)


if __name__ == "__main__":
    unittest.main()
