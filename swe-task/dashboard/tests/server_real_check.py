#!/usr/bin/env python3
"""End-to-end check of server.py on the REAL dashboard files (static/ and data/), read-only.

    /usr/bin/python3 -I -B tests/server_real_check.py

Starts server.py on 127.0.0.1, port 0, --no-collect, with its connection info file in a temporary directory (so
state/server.json is not touched and nothing is collected), then through the cookie flow (token read from that info
file, never printed): 403 without the token; ?token= -> 303 + HttpOnly SameSite=Strict cookie swe_dash_<port> holding
HMAC-SHA256(token, "session:<port>"), never the token; / , the assets that index.html references, and every data file
(index, each task file, each batch run's run/<source>/<run>.json, v1.7 explorer.json, v1.8 each older official run's
run/hengyuf/<run>.json, fleet, api, histories)
byte-identical to the files on disk, plain and gzipped, then revalidated to a 304 with its ETag; the security headers
on every response (Cache-Control: no-cache on files, no-store elsewhere); a CSP that lets index.html load app.js /
style.css and lets app.js lay out the tree with style="..." (no inline script needed, none allowed); HEAD, 405,
traversal. Always stops the server (SIGTERM) and removes the temporary directory. Exit status 0 when everything holds.
"""
import sys

sys.dont_write_bytecode = True

import gzip
import hashlib
import hmac
import html.parser
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time

D = Path(__file__).resolve().parent.parent
PROBLEMS = []


def check(condition, message):
    if not condition:
        PROBLEMS.append(message)
    return bool(condition)


class Refs(html.parser.HTMLParser):
    """<script>/<link>/<style>/on*= usage of index.html."""

    def __init__(self):
        super().__init__()
        self.scripts, self.styles, self.inline_scripts, self.style_elements, self.handlers = [], [], 0, 0, []
        self.meta_csp = None
        self._in_script = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.handlers += [name for name in attrs if name.startswith("on")]
        if tag == "script":
            if attrs.get("src"):
                self.scripts.append(attrs["src"])
            else:
                self._in_script = True
        elif tag == "link" and attrs.get("rel") == "stylesheet":
            self.styles.append(attrs.get("href"))
        elif tag == "style":
            self.style_elements += 1
        elif tag == "meta" and (attrs.get("http-equiv") or "").lower() == "content-security-policy":
            self.meta_csp = attrs.get("content")

    def handle_data(self, data):
        if self._in_script and data.strip():
            self.inline_scripts += 1

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_script = False


def csp_map(text):
    """'a b; c d' -> {a: [b], c: [d]}."""
    out = {}
    for part in (text or "").split(";"):
        words = part.split()
        if words:
            out[words[0].lower()] = words[1:]
    return out


def allows(policy, directive, source):
    """Does a CSP allow `source` for `directive` (with the usual fallbacks)?"""
    fallbacks = {"script-src-elem": ["script-src"], "style-src-elem": ["style-src"], "style-src-attr": ["style-src"],
                 "connect-src": [], "img-src": []}
    for name in [directive] + fallbacks.get(directive, []) + ["default-src"]:
        if name in policy:
            return source in policy[name]
    return True  # no directive at all: allowed


class Client:
    def __init__(self, port, token):
        self.port, self.token, self.cookie = port, token, None

    def get(self, path, method="GET", headers=None, cookie=True):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        sent = dict(headers or {})
        if cookie and self.cookie:
            sent["Cookie"] = self.cookie
        connection.request(method, path, headers=sent)
        response = connection.getresponse()
        body = response.read()
        connection.close()
        return response.status, {k.lower(): v for k, v in response.getheaders()}, body


def security_headers(path, headers, cache="no-store"):
    """cache: no-cache for the files of static/ and data/ (revalidated with their ETag), no-store for the rest."""
    check(headers.get("cache-control") == cache, f"{path}: Cache-Control {headers.get('cache-control')!r}")
    check(headers.get("x-content-type-options") == "nosniff", f"{path}: missing nosniff")
    check(bool(headers.get("content-security-policy")), f"{path}: missing CSP")
    if cache == "no-cache":
        check(bool(headers.get("etag")) and bool(headers.get("last-modified")), f"{path}: no ETag / Last-Modified")


def run_checks(client, token):
    # 1. Without the token: 403 everywhere but /healthz.
    for path in ("/", "/static/app.js", "/data/index.json"):
        status, headers, body = client.get(path, cookie=False)
        check(status == 403, f"{path} without token -> {status}")
        security_headers(path, headers)
    status, _, body = client.get("/healthz", cookie=False)
    check(status == 200 and json.loads(body) == {"ok": True}, f"/healthz -> {status} {body[:60]!r}")

    # 2. The cookie flow.
    status, headers, _ = client.get(f"/?token={token}", cookie=False)
    cookie = headers.get("set-cookie", "")
    check(status == 303 and headers.get("location") == "/", f"/?token= -> {status} Location {headers.get('location')}")
    check(all(flag in cookie for flag in ("HttpOnly", "SameSite=Strict", "Path=/")), "cookie flags missing")
    client.cookie = cookie.split(";", 1)[0]
    session = hmac.new(token.encode(), f"session:{client.port}".encode(), hashlib.sha256).hexdigest()
    check(client.cookie == f"swe_dash_{client.port}={session}", "the cookie is not the per-port session")
    check(token not in cookie, "the cookie holds the token")

    # 3. The page and what it references.
    status, headers, body = client.get("/")
    page = body.decode()
    check(status == 200 and headers.get("content-type", "").startswith("text/html"), f"/ -> {status}")
    check(page == (D / "static/index.html").read_text(), "/ is not static/index.html")
    security_headers("/", headers, "no-cache")
    refs = Refs()
    refs.feed(page)
    check(refs.inline_scripts == 0 and not refs.handlers, "index.html has inline script or on* handlers")
    check(refs.style_elements == 0, "index.html has <style> elements (style-src-elem 'self' blocks them)")
    header_csp = csp_map(headers.get("content-security-policy"))
    policies = [("header CSP", header_csp)] + ([("meta CSP", csp_map(refs.meta_csp))] if refs.meta_csp else [])
    app_js = (D / "static/app.js").read_text()
    uses_style_attr = 'style="' in app_js or "style=\\\"" in app_js
    for name, policy in policies:  # a browser enforces both: each must allow what the page does
        check(allows(policy, "script-src-elem", "'self'"), f"{name}: app.js would be blocked")
        check(not allows(policy, "script-src-elem", "'unsafe-inline'"), f"{name}: allows inline script")
        check("'unsafe-eval'" not in sum(policy.values(), []), f"{name}: allows eval")
        check(allows(policy, "style-src-elem", "'self'"), f"{name}: style.css would be blocked")
        check(allows(policy, "connect-src", "'self'"), f"{name}: fetch of data/*.json would be blocked")
        check(allows(policy, "img-src", "data:"), f"{name}: the data: favicon would be blocked")
        if uses_style_attr:
            check(allows(policy, "style-src-attr", "'unsafe-inline'"),
                  f"{name}: app.js sets style=\"...\" (tree layout, bars) but inline style attributes are blocked")
    check(re.search(r"\beval\(|new Function\(", app_js) is None, "app.js uses eval / new Function")
    assets = {"/static/app.js": ("text/javascript", D / "static/app.js"),
              "/static/style.css": ("text/css", D / "static/style.css")}
    for ref in refs.scripts + refs.styles:
        check(f"/{ref}" in assets, f"index.html references {ref}, which is not a served asset")
    for path, (kind, file) in assets.items():
        status, headers, body = client.get(path)
        check(status == 200 and headers.get("content-type", "").startswith(kind) and body == file.read_bytes(),
              f"{path} -> {status} {headers.get('content-type')}")
        security_headers(path, headers, "no-cache")
        status, again, body = client.get(path, headers={"If-None-Match": headers.get("etag", "")})
        check(status == 304 and not body and again.get("etag") == headers.get("etag"),
              f"{path} revalidated -> {status}")

    # 4. Every data file, plain and gzipped, byte-identical to disk.
    data = D / "data"
    index = json.loads((data / "index.json").read_text())
    files = ["index.json", "fleet.json", "api.json"]
    files += [name for name in ("history/fleet.jsonl", "history/hub.jsonl") if (data / name).exists()]  # (no hub
    # sample yet, e.g. a test cycle run with --no-hub)
    files += [f"task/{task['task_id']}.json" for task in index["tasks"] if task.get("run")]
    files += [run["detail"] for run in index["runs"] if run.get("source")]  # v1.6: a batch's run documents
    files += [run["detail"] for run in index["runs"]  # v1.8: an older official run's document
              if not run.get("source") and run.get("latest") is False]
    if isinstance(index.get("explorer"), dict):  # v1.7: the Explorer's tables
        files.append("explorer.json")
    if (data / "server_status.json").exists():
        files.append("server_status.json")
    served = 0
    for name in files:
        expect = (data / name).read_bytes()
        status, headers, body = client.get(f"/data/{name}", headers={"Accept": "application/json"})
        check(status == 200 and body == expect, f"/data/{name} -> {status}, {len(body)} vs {len(expect)} bytes")
        security_headers(f"/data/{name}", headers, "no-cache")
        # Revalidation: 304 unless the collector replaced the file meanwhile (then 200 with a new ETag).
        status, again, body = client.get(f"/data/{name}", headers={"If-None-Match": headers.get("etag", "")})
        check(status == 304 and not body if again.get("etag") == headers.get("etag") else status == 200,
              f"/data/{name} revalidated -> {status}")
        status, headers, body = client.get(f"/data/{name}", headers={"Accept-Encoding": "gzip"})
        if headers.get("content-encoding") == "gzip":
            body = gzip.decompress(body)
        check(status == 200 and body == expect, f"/data/{name} (gzip) differs")
        check(token.encode() not in body, f"/data/{name} contains the token")
        if name.endswith(".json"):
            json.loads(body)
        served += 1
    print(f"served {served} data files ({', '.join(files)})")

    # 5. Methods, HEAD, traversal, listings.
    status, headers, body = client.get("/data/index.json", method="HEAD")
    check(status == 200 and not body and int(headers.get("content-length", -1)) == (data / "index.json").stat().st_size,
          "HEAD /data/index.json")
    status, headers, _ = client.get("/data/index.json", method="POST")
    check(status == 405 and headers.get("allow") == "GET, HEAD", f"POST -> {status}")
    for path in ("/data/../state/collector.json", "/data/%2e%2e/state/collector.json", "/data/", "/data/task/",
                 "/collect.py", "/state/collector.json", "/static/../server.py"):
        status, _, _ = client.get(path)
        check(status == 404, f"{path} -> {status}")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="dash-server-check-"))
    info_file = tmp / "state" / "server.json"
    server = subprocess.Popen(["/usr/bin/python3", "-I", "-B", str(D / "server.py"), "--host", "127.0.0.1",
                               "--port", "0", "--no-collect", "--info-file", str(info_file)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
    try:
        info = None
        for _ in range(100):
            if info_file.exists():
                info = json.loads(info_file.read_text())
                break
            time.sleep(0.1)
        if not check(info and info.get("port"), "server.json never appeared"):
            return 1
        check(oct(info_file.stat().st_mode & 0o777) == "0o600", "server.json is not mode 600")
        print(f"server pid {info['pid']} on 127.0.0.1:{info['port']} (token read from the temp info file)")
        run_checks(Client(info["port"], info["token"]), info["token"])
    finally:
        server.send_signal(signal.SIGTERM)
        try:
            output, _ = server.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            output, _ = server.communicate()
        check(server.returncode == 0, f"server exit status {server.returncode}")
        check(info is None or info["token"] not in output, "the token reached the server log")
        print("server log:\n  " + "\n  ".join(output.strip().splitlines()[-6:]))
        shutil.rmtree(tmp, ignore_errors=True)
    for problem in PROBLEMS:
        print(f"PROBLEM {problem}")
    print(f"{len(PROBLEMS)} problem(s)" if PROBLEMS else "OK: server, cookie flow, assets, data files and headers")
    return 1 if PROBLEMS else 0


if __name__ == "__main__":
    sys.exit(main())
