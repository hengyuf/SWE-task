#!/usr/bin/env python3
"""HTTP server of the swe-task official-synthesis dashboard (stdlib only; /usr/bin/python3 -I -B, Python 3.10).

CONTRACT.md (same directory) is the source of truth; this file implements its "Security" section and runs the
collector (collect.py) in a background thread.
  - GET/HEAD only (anything else: 405). / and /index.html serve static/index.html, /static/<file> the single-page app
    (/app.js and /style.css are aliases of static/app.js and static/style.css), /data/<file> the collector's JSON.
    Regular files of known types only: no directory listings, no dot or empty path segments (percent-encoded ones
    included), nothing whose real path leaves static/ or data/ (symlinks included).
  - Token: secrets.token_urlsafe(32), new at every start, or the one in the info file with --reuse-token. Any URL with
    ?token=<t> sets an HttpOnly SameSite=Strict session cookie and redirects (303) to the same URL without the token;
    every other request needs that cookie or the X-Dashboard-Token header (the token itself), else 403. The cookie is
    per port and never holds the token (a browser sends the cookies of localhost to every port there): its name is
    swe_dash_<port>, its value HMAC-SHA256(token, "session:<port>") in hex, checked by recomputing it. Constant-time
    compares. /healthz answers {"ok": true} only, without a token.
  - Every response (errors included) carries X-Content-Type-Options: nosniff and a strict Content-Security-Policy:
    scripts only from this origin (so index.html must load app.js as a file; no inline script, no eval); inline style
    ATTRIBUTES are allowed because app.js positions the tree cards and sizes the bars with style="...". Responses are
    never stored (Cache-Control: no-store), except the files of static/ and data/: Cache-Control: no-cache with an
    ETag and Last-Modified, so the browser revalidates its copy every time and an unchanged file costs a 304
    (If-None-Match, else If-Modified-Since). Text files of 2 KB or more are gzipped for clients that accept it.
  - At most MAX_HANDLERS connections are handled at once (a thread each, socket timeout REQUEST_TIMEOUT_S); over the
    cap the accepting thread answers 503 at once.
  - The token never reaches a log: request lines are logged without their query string, everything printed has the
    token and the session value masked (also mistyped or truncated copies, e.g. a token typed as a path) and every
    token=... value too, and only errors and logins are logged (at most LOG_LINES_PER_MIN lines a minute).
  - Collector: collect.run_cycle(data, state) at startup and then every --interval s (start to start), one stdout line
    per cycle (duration, error count). A failed import (retried every cycle, e.g. collect.py mid-edit) or a failed
    cycle never stops the serving; data/server_status.json records the collector's health, "stopped" once the server
    exits (written only when collecting, so a serve-only instance never overwrites the status of the one that
    collects).
  - Connection info: state/server.json (file 600, dir 700) = host, bind, port, token, url, local_url, pid,
    slurm_job_id, started_at; launch.sh reads it. A bind address other than loopback is refused outside a Slurm job
    (CONTRACT: --host 0.0.0.0 only inside the job, which is reached by port forwarding).

    /usr/bin/python3 -I -B server.py [--host 127.0.0.1] [--port 0] [--interval 120] [--no-collect] [--reuse-token]
                                     [--print-url] [--root DIR] [--collector FILE] [--info-file FILE]
"""
import sys

sys.dont_write_bytecode = True

import argparse
from datetime import datetime, timezone
import email.utils
import errno
import gzip
import hashlib
import hmac
import http.server
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import socketserver
import threading
import time
import traceback
import urllib.parse

D = Path(__file__).resolve().parent
SCHEMA = 1
DEFAULT_INTERVAL_S = 120.0
MIN_GAP_S = 5.0  # pause between two collector cycles when one overran the interval
PORT_FALLBACKS = 20  # a taken --port: try the next 20 ports, then any free one (server.json has the actual port)
REQUEST_TIMEOUT_S = 15  # socket timeout of a request: a client that stalls this long gives its handler slot back
MAX_HANDLERS = 32  # connections handled at once (a thread each); the accepting thread answers the next ones 503
REFUSED_LINGER_S = 5.0  # a 503'd connection is closed once its client has closed it too, or after this long
MAX_REFUSED = 128  # 503'd connections waiting for that at once (beyond: the oldest is closed)
LOG_LINES_PER_MIN = 30  # request log lines (errors, logins) per minute; the rest is only counted
MAX_QUERY_FIELDS = 32
MAX_TOKEN_CHARS = 512  # longer offers are rejected before the compare
TOKEN_HEADER = "X-Dashboard-Token"
COOKIE_PREFIX = "swe_dash_"  # + the port: one session cookie per dashboard port, so two dashboards on localhost (the
#                              Slurm one and `launch.sh local`, forwarded to two ports) never clash
LEGACY_COOKIE = re.compile(r"swe_dash_[0-9a-f]{10}")  # the v1.1 cookie (the token itself, named after its hash):
#                                                       expired at the next login, so no token stays in the browser
COOKIE_MAX_AGE_S = 7 * 86400  # the Slurm job's walltime
MASK_RUN = 8  # this many consecutive characters of a live secret are masked wherever they appear in a log line

# Scripts, styles, images and fetches from this origin only; no inline or eval'd script, plugins, frames, <base> or
# forms. Style attributes are allowed (app.js sets left/top/width with style="..."); <style> elements are not where
# the browser knows style-src-elem (older ones fall back to style-src). index.html carries a meta CSP of its own; a
# browser enforces both.
CSP = "; ".join([
    "default-src 'self'", "script-src 'self'", "style-src 'self' 'unsafe-inline'", "style-src-elem 'self'",
    "style-src-attr 'unsafe-inline'", "img-src 'self' data:", "font-src 'self'", "connect-src 'self'",
    "object-src 'none'", "base-uri 'none'", "form-action 'none'", "frame-ancestors 'none'"])
# On every response, after its Cache-Control (no-store; no-cache for the files of static/ and data/: end_headers).
SECURITY_HEADERS = (
    ("X-Content-Type-Options", "nosniff"),
    ("Content-Security-Policy", CSP),
    ("Referrer-Policy", "no-referrer"),
    ("X-Frame-Options", "DENY"),
    ("Cross-Origin-Opener-Policy", "same-origin"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
)
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".json": "application/json", ".jsonl": "application/x-ndjson", ".map": "application/json",
    ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon", ".txt": "text/plain; charset=utf-8",
    ".woff2": "font/woff2",
}
AREA_TYPES = {"static": frozenset(CONTENT_TYPES), "data": frozenset({".json", ".jsonl"})}  # what each root serves
# Text is gzipped for clients that accept it: a task document is ~2 MB of JSON, refetched every 60 s through a port
# forward. (No secret and nothing from the request is ever in a body, so compression leaks nothing.)
COMPRESSIBLE = frozenset({".html", ".js", ".css", ".json", ".jsonl", ".map", ".svg", ".txt"})
GZIP_MIN_BYTES, GZIP_MAX_BYTES = 2048, 64 << 20
ALIASES = {"/": ("static", "index.html"), "/index.html": ("static", "index.html"),
           "/app.js": ("static", "app.js"), "/style.css": ("static", "style.css")}
BAD_SEGMENT = re.compile(r"[\x00-\x1f\x7f\\]")  # control characters and backslashes never name a file here
SAFE_LOCATION = re.compile(r"/(?!/)[A-Za-z0-9._~!$&'()*+,;=:@%/-]*")  # a redirect stays a path on this origin
TOKEN_SHAPE = re.compile(r"[A-Za-z0-9_-]{32,256}")  # what --reuse-token accepts from the info file
QUERY_TOKEN = re.compile(r"(?i)(token=)[^&\s'\"]*")
CONTROL = re.compile(r"[\x00-\x1f\x7f]")
ENTITY_TAG = re.compile(r'(?:W/)?"([^"]*)"')  # one entity tag of an If-None-Match list -> its opaque part
# Masking for everything this server logs or records (on top of collect.redact(), which may not be importable).
_SECRET_LIKE = [
    (re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{10,}"), "sk-<redacted>"),
    (re.compile(r"nvapi-[A-Za-z0-9_-]{10,}"), "nvapi-<redacted>"),
    (re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{6,}"), r"\1<redacted>"),
    (re.compile(r"(?i)((?:key|token|secret|password)[A-Za-z0-9_.-]{0,40}[\"']?\s{0,3}[=:]\s{0,3}[\"']?)"
                r"[^\s&'\",]{8,}"), r"\1<redacted>"),
]
FORBIDDEN = ("403 forbidden: missing or invalid dashboard token.\n"
             "Browser: open the URL printed by `launch.sh start` or `launch.sh status` (it ends in ?token=...) once: "
             "it sets a session cookie for this server in this browser.\n"
             "Tools: send the X-Dashboard-Token header from a file only you can read, never with the token on a "
             "command line (ps and the shell history keep it): curl -H @$HOME/.dashboard-header <url>, where the file "
             "(mode 600) holds the line \"X-Dashboard-Token: <token>\" (`launch.sh header` writes it).\n")
BUSY_BODY = b"503 busy: too many connections at once; try again in a moment\n"


# --------------------------------------------------------------------------------------------------------- helpers

class SecretMask:
    """Masks the live secrets (the token, the session cookie value) in a text: every occurrence of the exact value,
    and also every run of MASK_RUN or more characters copied from one, so a token that was mistyped, cut short or
    glued to other text (e.g. typed as a path: GET /<token>x) is masked too. (By chance, 8 characters of other text
    match a run of a random 43-character token with a probability of about 1e-13.)"""

    def __init__(self):
        self.secrets = ()  # (label, run length, the secret's substrings of that length); replaced, never mutated

    def add(self, secret, label):
        run = min(MASK_RUN, len(secret))
        if run:
            self.secrets += ((label, run, frozenset(secret[i:i + run] for i in range(len(secret) - run + 1))),)

    def __call__(self, text):
        for label, run, pieces in self.secrets:
            spans = []  # [start, end) of the masked stretches; overlapping or touching matches merge
            for i in range(len(text) - run + 1):
                if text[i:i + run] in pieces:
                    if spans and i <= spans[-1][1]:
                        spans[-1][1] = i + run
                    else:
                        spans.append([i, i + run])
            for start, end in reversed(spans):
                text = text[:start] + label + text[end:]
        return text


MASK = SecretMask()  # main() adds the token and the session value before anything can log them


def scrub(text):
    """What may be printed: the live secrets (MASK) and every token=... value masked."""
    return QUERY_TOKEN.sub(r"\1<redacted>", MASK(str(text)))


def log(message):
    """One line on stdout (the Slurm log): local time + message, scrubbed."""
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {scrub(message)}", flush=True)


def iso(ts):
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def redact(text, module=None):
    """Mask the live secrets (MASK), credentials (collect.redact() when the collector module is given, then our own
    minimal set) and every token= value."""
    text = MASK(str(text))
    masker = getattr(module, "redact", None)
    if callable(masker):
        try:
            text = str(masker(text))
        except Exception:  # a broken collector must not break the logging
            pass
    for pattern, replacement in _SECRET_LIKE:
        text = pattern.sub(replacement, text)
    return QUERY_TOKEN.sub(r"\1<redacted>", text)


def clip(text, limit):
    text = str(text)
    return text if len(text) <= limit else f"{text[:limit]}…[{len(text) - limit} chars omitted]…"


def write_json_atomic(path, value, mode):
    """temp file in the same directory + rename (readers never see a partial file), created with `mode`."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, mode)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=1)
            stream.write("\n")
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def read_json(path):
    try:
        with open(path) as stream:
            value = json.load(stream)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def private_dir(path):
    """mkdir -p with mode 700 (the state directory holds the token); an existing directory of ours is made 700."""
    path = Path(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.stat().st_uid == os.getuid():
        os.chmod(path, 0o700)


def accepts_gzip(header):
    """Does an Accept-Encoding header allow gzip (q > 0)? The first entry naming gzip (or "*") decides."""
    for item in (header or "").split(","):
        coding, _, params = item.partition(";")
        if coding.strip().lower() not in ("gzip", "x-gzip", "*"):
            continue
        quality = 1.0
        for param in params.split(";"):
            name, _, value = param.partition("=")
            if name.strip().lower() == "q":
                try:
                    quality = float(value)
                except ValueError:
                    quality = 0.0
        return quality > 0
    return False


def not_modified(headers, tag, mtime):
    """Does a conditional GET/HEAD of a file with this entity tag (its opaque part) and mtime get a 304? If-None-Match
    decides when present: "*" or one of its entity tags (weak comparison, W/ ignored). Otherwise If-Modified-Since,
    when it is a date not older than the mtime (HTTP dates are whole seconds, hence the ETag first)."""
    offered = headers.get_all("If-None-Match")
    if offered:
        offered = ",".join(offered)
        return any(item.strip() == "*" for item in offered.split(",")) or tag in ENTITY_TAG.findall(offered)
    since = headers.get("If-Modified-Since")
    if not since:
        return False
    try:
        date = email.utils.parsedate_to_datetime(since)
    except (TypeError, ValueError, IndexError, OverflowError):
        return False  # not a date: ignored
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    return int(mtime) <= date.timestamp()


def is_loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def load_module(path):
    """Import the collector from its file (python -I puts no script directory on sys.path)."""
    spec = importlib.util.spec_from_file_location("collect", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["collect"] = module  # as a normal import would have it while the module body runs
    try:
        spec.loader.exec_module(module)
        if not callable(getattr(module, "run_cycle", None)):
            raise ImportError(f"{path} has no run_cycle()")
    except BaseException:
        sys.modules.pop("collect", None)
        raise
    return module


# ------------------------------------------------------------------------------------------------- the web server

class App:
    """What the request handlers share: the served roots, the token, the session cookie and the request-log budget."""

    def __init__(self, root, token):
        self.roots = {area: os.path.realpath(Path(root) / area) for area in AREA_TYPES}
        self.token = token
        self._token = token.encode()
        self.port = self.cookie_name = None  # set_port(), once the server is bound
        self._log_lock = threading.Lock()
        self._window = [0.0, 0, 0]  # start of the current minute, lines printed, lines suppressed

    def set_port(self, port):
        """The session cookie belongs to the port this server listens on (Server calls this once bound)."""
        self.port, self.cookie_name = port, f"{COOKIE_PREFIX}{port}"

    def session(self):
        """The session cookie's value, HMAC-SHA256(token, "session:<port>") in hex: derived from the token, never
        stored. Not the token, because a browser sends the cookies of localhost to every port there; and a session of
        one port (one dashboard) is no good on another."""
        return hmac.new(self._token, f"session:{self.port}".encode(), hashlib.sha256).hexdigest()

    def check(self, offered):
        """Constant-time comparison of an offered token with ours."""
        if not isinstance(offered, str) or not offered or len(offered) > MAX_TOKEN_CHARS:
            return False
        return hmac.compare_digest(offered.encode("utf-8", "replace"), self._token)

    def check_session(self, offered):
        """Constant-time comparison of an offered session cookie value with ours, recomputed."""
        if not isinstance(offered, str) or not offered or len(offered) > MAX_TOKEN_CHARS:
            return False
        return hmac.compare_digest(offered.encode("utf-8", "replace"), self.session().encode())

    def resolve(self, path):
        """Percent-decoded URL path -> real path of a regular file inside static/ or data/, else None."""
        if path in ALIASES:
            area, relative = ALIASES[path]
        elif path.startswith("/"):
            area, _, relative = path[1:].partition("/")
        else:
            return None
        root = self.roots.get(area)
        if root is None or not relative:
            return None
        parts = relative.split("/")
        # Empty segments ("a//b", a trailing "/": directories), ".", ".." and hidden names (the collector's
        # ".<name>.<pid>.tmp" files) are refused outright; nothing is normalized away.
        if any(not part or part.startswith(".") or BAD_SEGMENT.search(part) for part in parts):
            return None
        if os.path.splitext(parts[-1])[1].lower() not in AREA_TYPES[area]:
            return None
        candidate = os.path.realpath(os.path.join(root, *parts))
        if os.path.commonpath([root, candidate]) != root or not os.path.isfile(candidate):
            return None  # a symlink out of the root, a directory, a FIFO, or nothing at all
        return candidate

    def request_log(self, message):
        """Rate-limited request log line (a scanner must not flood a 7-day Slurm log)."""
        now = time.monotonic()
        with self._log_lock:
            start, printed, suppressed = self._window
            if now - start >= 60:
                if suppressed:
                    log(f"({suppressed} more request log lines suppressed)")
                start, printed, suppressed = now, 0, 0
            if printed < LOG_LINES_PER_MIN:
                printed += 1
                self._window = [start, printed, suppressed]
            else:
                self._window = [start, printed, suppressed + 1]
                return
        log(clip(CONTROL.sub("?", redact(message)), 500))


class Handler(http.server.BaseHTTPRequestHandler):
    """GET/HEAD of / , /index.html, /app.js, /style.css, /static/*, /data/* (token required) and /healthz (open)."""

    server_version = "swe-task-dashboard"
    # Malformed request lines would otherwise get an HTTP/0.9 answer: a bare body without the security headers.
    default_request_version = "HTTP/1.0"
    timeout = REQUEST_TIMEOUT_S

    # -- plumbing
    def version_string(self):  # the Server header: no Python version
        return self.server_version

    def end_headers(self, cache="no-store"):
        """Every response, including BaseHTTPRequestHandler's own error pages, carries the security headers. Only the
        files of static/ and data/ may be kept by the browser, and revalidated every time (send_file: no-cache)."""
        self.send_header("Cache-Control", cache)
        for name, value in SECURITY_HEADERS:
            self.send_header(name, value)
        super().end_headers()

    def log_request(self, code="-", size="-"):
        """Routine requests are not logged; errors are, without the query string (it may hold the token)."""
        code = getattr(code, "value", code)
        if isinstance(code, int) and code >= 400:
            self.server.app.request_log(f"{self.client_address[0]} {self.command or '-'} {self.log_path()} -> {code}")

    def log_message(self, format, *args):  # BaseHTTPRequestHandler's own messages (malformed requests, timeouts)
        self.server.app.request_log(f"{self.client_address[0]} {format % args if args else format}")

    def log_path(self):
        path = getattr(self, "path", None)  # unset when the request line itself was rejected (400, 414)
        return clip(CONTROL.sub("?", path.partition("?")[0]), 200) if isinstance(path, str) else "-"

    def send_body(self, code, body, content_type, head, extra=(), cache="no-store"):
        self.send_response(code)
        for name, value in extra:
            self.send_header(name, value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers(cache)
        if not head:
            for start in range(0, len(body), 1 << 16):  # REQUEST_TIMEOUT_S per write, not for a whole gzipped file
                self.wfile.write(body[start:start + (1 << 16)])

    def send_text(self, code, text, head, extra=()):
        self.send_body(code, text.encode(), "text/plain; charset=utf-8", head, extra)

    def cookies(self):
        """(name, value) of every cookie the request carries."""
        for header in self.headers.get_all("Cookie") or []:
            for item in header.split(";"):
                name, sep, value = item.strip().partition("=")
                if sep:
                    yield name.strip(), value.strip().strip('"')

    def authorized(self):
        """The X-Dashboard-Token header holds the token, or our session cookie holds the session value."""
        app = self.server.app
        header = self.headers.get(TOKEN_HEADER)
        if header is not None and app.check(header.strip()):
            return True
        return any(name == app.cookie_name and app.check_session(value) for name, value in self.cookies())

    # -- methods
    def do_GET(self):
        self.respond(head=False)

    def do_HEAD(self):
        self.respond(head=True)

    def reject(self):
        """405 for every method but GET/HEAD (a small request body is read first, so the client sees the answer)."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if 0 < length <= 1 << 20:
            self.rfile.read(length)
        self.close_connection = True
        self.send_text(405, "405 method not allowed: GET and HEAD only\n", False, [("Allow", "GET, HEAD")])

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_TRACE = do_CONNECT = do_PROPFIND = reject

    def respond(self, head):
        app = self.server.app
        if not self.path.startswith("/"):  # absolute-form or "*" request targets: not for this server
            return self.send_text(400, "400 bad request\n", head)
        path, _, query = self.path.partition("?")
        path = path.partition("#")[0]
        if path == "/healthz":  # open, so it says nothing else
            return self.send_body(200, b'{"ok": true}', "application/json", head)
        try:
            params = urllib.parse.parse_qsl(query, keep_blank_values=True, max_num_fields=MAX_QUERY_FIELDS)
        except ValueError:
            return self.send_text(400, "400 bad request\n", head)
        offered = [value for name, value in params if name == "token"]
        if offered:
            valid = app.check(offered[0])
            if not (valid or self.authorized()):
                return self.send_text(403, FORBIDDEN, head)
            # The token leaves the address bar (and the browser history); the session cookie stands in for it.
            rest = urllib.parse.urlencode([(name, value) for name, value in params if name != "token"])
            location = (path if SAFE_LOCATION.fullmatch(path) else "/") + (f"?{rest}" if rest else "")
            extra = [("Location", location)]
            if valid:
                extra.insert(0, ("Set-Cookie", f"{app.cookie_name}={app.session()}; Path=/; "
                                               f"Max-Age={COOKIE_MAX_AGE_S}; HttpOnly; SameSite=Strict"))
                extra += [("Set-Cookie", f"{name}=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict")
                          for name in sorted({name for name, _ in self.cookies() if LEGACY_COOKIE.fullmatch(name)})]
                app.request_log(f"{self.client_address[0]} login: session cookie set")
            return self.send_text(303, "", head, extra)
        if not self.authorized():
            return self.send_text(403, FORBIDDEN, head)
        try:
            target = app.resolve(urllib.parse.unquote(path, errors="strict"))
        except UnicodeDecodeError:
            target = None
        if target is None:
            return self.send_text(404, "404 not found\n", head)
        self.send_file(target, head)

    def send_file(self, target, head):
        try:
            stream = open(target, "rb")
        except OSError:  # vanished since resolve() (the collector removes stale task files)
            return self.send_text(404, "404 not found\n", head)
        with stream:
            st = os.fstat(stream.fileno())
            size = st.st_size
            suffix = os.path.splitext(target)[1].lower()
            vary = [("Vary", "Accept-Encoding")] if suffix in COMPRESSIBLE else []
            # The browser keeps a copy and revalidates it every time (no-cache): an unchanged file costs a 304. The
            # collector replaces a file by renaming a new one over it, so the inode changes too. Weak: the plain and
            # the gzipped body are the same content.
            tag = f"{size:x}-{st.st_mtime_ns:x}-{st.st_ino:x}"
            validators = [("ETag", f'W/"{tag}"'), ("Last-Modified", self.date_time_string(st.st_mtime))]
            if not_modified(self.headers, tag, st.st_mtime):
                self.send_response(304)
                for name, value in validators + vary:
                    self.send_header(name, value)
                return self.end_headers("no-cache")
            if vary and GZIP_MIN_BYTES <= size <= GZIP_MAX_BYTES and accepts_gzip(self.headers.get("Accept-Encoding")):
                body = gzip.compress(stream.read(size), compresslevel=6, mtime=0)
                return self.send_body(200, body, CONTENT_TYPES[suffix], head,
                                      [("Content-Encoding", "gzip")] + vary + validators, "no-cache")
            self.send_response(200)
            for name, value in vary + validators:
                self.send_header(name, value)
            self.send_header("Content-Type", CONTENT_TYPES[suffix])
            self.send_header("Content-Length", str(size))
            self.end_headers("no-cache")
            remaining = size  # exactly Content-Length bytes, also if an append-only history grows meanwhile
            while not head and remaining > 0:
                chunk = stream.read(min(1 << 16, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def drain(sock):
    """Read and drop what a non-blocking socket has received so far (up to 1 MB): False once the peer has closed (or
    the socket failed), True while it is open."""
    try:
        for _ in range(16):
            if not sock.recv(1 << 16):
                return False
    except BlockingIOError:
        return True
    except OSError:
        return False
    return True


class Server(http.server.ThreadingHTTPServer):
    """A handler thread per connection, at most MAX_HANDLERS at once; the accepting thread answers the next ones 503."""

    daemon_threads = True  # a slow client never delays the shutdown
    request_queue_size = 64

    def __init__(self, address, app):
        self.app = app
        self.slots = threading.BoundedSemaphore(MAX_HANDLERS)
        self.refused = []  # (socket, close-by time) of the connections answered 503, until their client closes
        self.busy = ("HTTP/1.0 503 Service Unavailable\r\n" + "".join(f"{name}: {value}\r\n" for name, value in [
            ("Server", Handler.server_version), ("Content-Type", "text/plain; charset=utf-8"),
            ("Content-Length", len(BUSY_BODY)), ("Retry-After", 2), ("Connection", "close"),
            ("Cache-Control", "no-store"), *SECURITY_HEADERS]) + "\r\n").encode("latin-1") + BUSY_BODY
        self.address_family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
        super().__init__(address, Handler)
        app.set_port(self.server_address[1])  # the session cookie belongs to the port actually bound

    def server_bind(self):
        # HTTPServer.server_bind would also resolve socket.getfqdn(host): a reverse DNS lookup nobody needs here.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = str(self.server_address[0]), self.server_address[1]

    def process_request(self, request, client_address):
        """A handler thread while a slot is free, which it gives back when it ends; otherwise 503 at once."""
        if not self.slots.acquire(blocking=False):
            return self.refuse(request, client_address)
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()  # no thread started
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def refuse(self, request, client_address):
        """503 from the accepting thread, which must never wait: the socket is made non-blocking, and the short answer
        fits its empty send buffer. The socket stays open (service_actions) until the client has closed it too, or
        for REFUSED_LINGER_S: closed at once, with the request unread, it would be reset, which can destroy the answer
        before the client reads it."""
        try:
            request.setblocking(False)
            request.send(self.busy)
            request.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        self.refused.append((request, time.monotonic() + REFUSED_LINGER_S))
        while len(self.refused) > MAX_REFUSED:
            self.refused.pop(0)[0].close()
        self.app.request_log(f"{client_address[0]} busy: all {MAX_HANDLERS} handler slots taken -> 503")

    def service_actions(self):
        """serve_forever() calls this after every accepted connection and at least every poll interval: a refused
        connection is closed once its client has closed it (what it sent is read first) or its time is up."""
        if self.refused:
            now = time.monotonic()
            waiting = []
            for sock, deadline in self.refused:
                if now < deadline and drain(sock):
                    waiting.append((sock, deadline))
                else:
                    sock.close()
            self.refused = waiting

    def server_close(self):
        super().server_close()
        for sock, _ in self.refused:
            sock.close()
        self.refused = []

    def handle_error(self, request, client_address):
        """One line instead of a traceback; a client that went away is not an error."""
        error = sys.exc_info()[1]
        if isinstance(error, (BrokenPipeError, ConnectionResetError, socket.timeout)):
            return
        self.app.request_log(f"{client_address[0]} handler error {type(error).__name__}: {error}")


def bind(host, port, app):
    """Server on host:port; a taken non-zero port falls back to the next PORT_FALLBACKS ports, then to any free one."""
    candidates = [port]
    if port:
        candidates += [number for number in range(port + 1, port + 1 + PORT_FALLBACKS) if number < 65536] + [0]
    for number in candidates:
        try:
            return Server((host, number), app)
        except OSError as error:
            if error.errno != errno.EADDRINUSE or number == candidates[-1]:
                raise
            log(f"port {number} is in use on {host}; trying another")


# ---------------------------------------------------------------------------------------------------- collector

class Collector(threading.Thread):
    """collect.run_cycle(data, state) now and then every `interval` s, start to start (at least MIN_GAP_S apart when a
    cycle overran). Never raises: a failed import is retried at the next cycle, a failed cycle is logged, and both
    are recorded in data/server_status.json."""

    def __init__(self, module_path, data_dir, state_dir, interval, server_info):
        super().__init__(name="collector", daemon=True)
        self.module_path, self.data_dir, self.state_dir = Path(module_path), Path(data_dir), Path(state_dir)
        self.interval = interval
        self.server_info = server_info
        self.module, self.module_mtime = None, None
        self.stopping = threading.Event()
        self.publish_failed = False
        self.status = {"enabled": True, "module": str(self.module_path), "interval_s": interval, "state": "starting",
                       "cycles": 0, "failures": 0, "consecutive_failures": 0, "import_error": None,
                       "last_started_at": None, "last_finished_at": None, "last_seconds": None, "last_ok": None,
                       "last_error_count": None, "last_errors": [], "last_failure": None, "last_failure_at": None,
                       "next_at": None}

    def stop(self):
        self.stopping.set()

    def publish(self):
        """data/server_status.json: small, for the page and for people; never holds the token or the port."""
        now = time.time()
        doc = {"schema": SCHEMA, "generated_at": round(now, 3), "generated_at_iso": iso(now),
               "server": self.server_info, "collector": self.status}
        try:
            write_json_atomic(self.data_dir / "server_status.json", doc, 0o644)
            self.publish_failed = False
        except OSError as error:
            if not self.publish_failed:  # once per failure streak
                log(f"cannot write {self.data_dir / 'server_status.json'}: {error}")
            self.publish_failed = True

    def cycle(self):
        """One cycle -> (summary dict or None, failure text or None, traceback or None). collect.py is re-imported
        when it changed on disk (a collector fix needs no server restart, so the job keeps its node and URL); if the
        new code does not import, the previous module keeps running until the file changes again."""
        try:
            mtime = self.module_path.stat().st_mtime_ns
        except OSError:
            mtime = None
        if self.module is None or (mtime is not None and mtime != self.module_mtime):
            try:
                module = load_module(self.module_path)
                if self.module is not None:
                    log(f"reloaded {self.module_path.name} (changed on disk)")
                self.module, self.module_mtime = module, mtime
                self.status["import_error"] = None
            except BaseException as error:  # SyntaxError, ImportError, SystemExit, ...: keep serving
                failure = clip(redact(f"import {self.module_path.name}: {type(error).__name__}: {error}"), 600)
                self.status["import_error"] = failure
                self.module_mtime = mtime
                if self.module is None:
                    return None, failure, None
        try:
            summary = self.module.run_cycle(self.data_dir, self.state_dir)
        except BaseException as error:  # run_cycle reports broken sources itself; this is a bug or sys.exit()
            failure = clip(redact(f"run_cycle: {type(error).__name__}: {error}", self.module), 600)
            return None, failure, redact(traceback.format_exc(limit=8), self.module)
        return (summary if isinstance(summary, dict) else {}), None, None

    def run(self):
        while not self.stopping.is_set():
            started, started_at = time.monotonic(), time.time()
            number = self.status["cycles"] + 1
            self.status.update(state="running", last_started_at=round(started_at, 3), next_at=None)
            self.publish()
            summary, failure, trace = self.cycle()
            seconds = round(time.monotonic() - started, 2)
            errors = [clip(redact(item, self.module), 300) for item in (summary or {}).get("errors") or []]
            self.status.update(cycles=number, last_finished_at=round(time.time(), 3), last_seconds=seconds,
                               last_ok=failure is None and not errors, last_error_count=len(errors),
                               last_errors=errors[:5])
            if failure is None:
                self.status["consecutive_failures"] = 0
                detail = f" ({summary.get('runs')} runs, {summary.get('tasks')} tasks)" if "runs" in summary else ""
                first = f"; first: {clip(errors[0], 200)}" if errors else ""
                log(f"collect cycle {number}: {seconds} s, {len(errors)} errors{detail}{first}")
            else:
                self.status["failures"] += 1
                self.status["consecutive_failures"] += 1
                self.status.update(last_failure=failure, last_failure_at=round(time.time(), 3))
                log(f"collect cycle {number} FAILED after {seconds} s, {len(errors)} errors: {failure}")
                if trace:
                    print(scrub(trace.rstrip()), flush=True)
            delay = max(self.interval - (time.monotonic() - started), min(MIN_GAP_S, self.interval))
            self.status.update(state="idle", next_at=round(time.time() + delay, 3))
            self.publish()
            self.stopping.wait(delay)


# --------------------------------------------------------------------------------------------------- entry point

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (default 127.0.0.1; 0.0.0.0 only inside the Slurm job)")
    parser.add_argument("--port", type=int, default=0,
                        help="0 = any free port; a taken port falls back to a nearby free one (see server.json)")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S,
                        help="seconds between two collector cycles, start to start (default 120)")
    parser.add_argument("--no-collect", action="store_true",
                        help="serve only, no collector thread (another process collects)")
    parser.add_argument("--reuse-token", action="store_true",
                        help="keep the token of the existing info file, so browser sessions stay valid (on the same "
                             "port: the session cookie is per port)")
    parser.add_argument("--print-url", action="store_true",
                        help="print the URL WITH the token on stdout (launch.sh local: an interactive terminal, "
                             "never a Slurm log)")
    parser.add_argument("--root", default=str(D), help="directory with static/, data/ and state/ (default: here)")
    parser.add_argument("--collector", help="collector module file (default <root>/collect.py)")
    parser.add_argument("--info-file", help="connection info file (default <root>/state/server.json)")
    args = parser.parse_args(argv)
    if not 0 <= args.port < 65536:
        parser.error("--port must be within 0..65535")
    if not args.interval >= 1:
        parser.error("--interval must be at least 1 s")
    slurm_job = os.environ.get("SLURM_JOB_ID") or None
    if not is_loopback(args.host) and not slurm_job:
        parser.error(f"--host {args.host} would expose the dashboard beyond this host: only inside the Slurm job "
                     "(dashboard.sbatch); on a login node use 127.0.0.1 (launch.sh local)")

    root = Path(args.root).resolve()
    data_dir, state_dir = root / "data", root / "state"
    info_file = Path(args.info_file).resolve() if args.info_file else state_dir / "server.json"
    private_dir(info_file.parent)
    data_dir.mkdir(parents=True, exist_ok=True)

    token = None
    if args.reuse_token:
        previous = read_json(info_file).get("token")
        if isinstance(previous, str) and TOKEN_SHAPE.fullmatch(previous):
            token = previous
        else:
            log(f"--reuse-token: no usable token in {info_file}; generating a new one")
    token = token or secrets.token_urlsafe(32)
    MASK.add(token, "<token>")  # from here on nothing printed or recorded shows it
    app = App(root, token)
    httpd = bind(args.host, args.port, app)
    port = httpd.server_address[1]
    MASK.add(app.session(), "<session>")

    hostname = socket.gethostname()
    wildcard = args.host in ("0.0.0.0", "::", "")
    url_host = hostname if wildcard else (f"[{args.host}]" if ":" in args.host else args.host)
    started = time.time()
    info = {"host": hostname, "bind": args.host, "port": port, "token": token,
            "url": f"http://{url_host}:{port}/?token={token}", "local_url": f"http://localhost:{port}/?token={token}",
            "pid": os.getpid(), "slurm_job_id": slurm_job, "started_at": round(started, 3),
            "started_at_iso": iso(started), "root": str(root), "collect": not args.no_collect,
            "interval_s": args.interval, "cookie": app.cookie_name}  # the name; the value derives from the token
    write_json_atomic(info_file, info, 0o600)
    log(f"swe-task dashboard: serving {root} on {args.host}:{port} (host {hostname}, pid {os.getpid()}, "
        f"job {slurm_job or '-'}); collector {'off' if args.no_collect else f'every {args.interval:g} s'}; "
        f"URL with the token in {info_file} (mode 600)")
    if args.print_url:
        print(f"Dashboard: {info['url']}", flush=True)

    collector = None
    if not args.no_collect:
        server_info = {"host": hostname, "pid": os.getpid(), "slurm_job_id": slurm_job,
                       "started_at": round(started, 3), "started_at_iso": iso(started)}
        collector = Collector(Path(args.collector) if args.collector else root / "collect.py", data_dir, state_dir,
                              args.interval, server_info)
        collector.start()

    def terminate(signum, frame):  # scancel / the walltime send SIGTERM: leave serve_forever() through finally
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if collector:  # tell status readers (launch.sh status, the page) that nobody collects any more
            collector.stop()
            collector.join(2.0)  # between cycles it ends at once; a cycle in progress dies with the process
            collector.status.update(state="stopped", next_at=None)
            collector.publish()
        httpd.server_close()
        current = read_json(info_file)
        if current.get("pid") == os.getpid() and current.get("token") == token:  # still ours: mark it stopped
            current["stopped_at"] = round(time.time(), 3)
            try:
                write_json_atomic(info_file, current, 0o600)
            except OSError:
                pass
        log("swe-task dashboard: stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
