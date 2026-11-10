#!/usr/bin/env python3
"""Closed-loop token-level replay of recorded GLM-5.3 OpenCode sessions against an OpenAI-compatible
/v1/completions endpoint (vLLM). Stdlib only (python >= 3.8): runs on the login node and in the container.

Each session issues its recorded calls in order. Call k sends the exact recorded prompt token ids
(prompt_k = prompt_{k-1}[:shared_prefix_len] + delta_k) with max_tokens = min_tokens = recorded completion
tokens (reasoning included) and ignore_eos, so the server does the recorded prefill + decode work; after the call
the session sleeps the recorded tool/agent gap (x --think-scale, capped by --think-cap) and issues the next call.
K = --concurrency session slots run at once; a slot that finishes its session starts the next one (--loop
recycles the session list with a fresh cache salt).

--prefix-mode splice (default) puts the token ids the server just generated (return_token_ids) into the next
prompt in place of the re-rendered assistant turn (same length), so the server's prefix cache can reuse the
generated KV as it would in production when the chat template re-renders the model's own output token-identically
(recorded completion_tokens == rendered assistant span + stop token on 83.5-84.3% of GLM-5.3 OpenCode turns,
within +-2 tokens on 99.9%; see README) -> upper bound on reuse. --prefix-mode splice-exact splices only the
exact-count turns. --prefix-mode recorded sends the recorded prompts byte-exactly (assistant turn always
re-prefilled, lower bound; also what a P/D prefill tier that never sees the decode KV experiences).

Outputs in --out: calls.jsonl (one line per call), summary.json (+ printed), client_args.json,
metrics.tsv (if --metrics-url), progress lines on stdout.
"""
import argparse
import array
import asyncio
import base64
import collections
import gzip
import hashlib
import json
import math
import os
import random
import signal
import ssl
import sys
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


# ----------------------------------------------------------------------------------------------------------------
# replay file loading
# ----------------------------------------------------------------------------------------------------------------

def unpack_ids(b64):
    a = array.array("I")
    a.frombytes(base64.b64decode(b64))
    if sys.byteorder != "little":
        a.byteswap()
    return a


def ids_sha1(ids):
    a = ids if isinstance(ids, array.array) else array.array("I", ids)
    if sys.byteorder != "little":
        a = array.array("I", a)
        a.byteswap()
    return hashlib.sha1(a.tobytes()).hexdigest()


def load_replay(paths, source_filter=None):
    headers, sessions = [], []
    for p in paths:
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("type") == "header":
                    headers.append(d)
                    continue
                if source_filter and d.get("source") not in source_filter:
                    continue
                for c in d["calls"]:
                    c["delta"] = unpack_ids(c.pop("delta_ids"))
                d["_file"] = os.path.basename(p)
                sessions.append(d)
    return headers, sessions


# ----------------------------------------------------------------------------------------------------------------
# minimal async HTTP/1.1 + SSE client
# ----------------------------------------------------------------------------------------------------------------

class Endpoint:
    def __init__(self, base_url):
        u = urllib.parse.urlsplit(base_url if "://" in base_url else "http://" + base_url)
        self.scheme = u.scheme
        self.host = u.hostname
        self.port = u.port or (443 if u.scheme == "https" else 80)
        path = (u.path or "").rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        self.prefix = path
        self.ssl = ssl.create_default_context() if u.scheme == "https" else None

    def url(self, path):
        return "%s://%s:%d%s%s" % (self.scheme, self.host, self.port, self.prefix, path)


class HTTPError(Exception):
    def __init__(self, status, body):
        super().__init__("HTTP %s: %s" % (status, body[:300]))
        self.status = status


async def _read_headers(reader):
    raw = await reader.readuntil(b"\r\n\r\n")
    head = raw.decode("latin-1")
    first, _, rest = head.partition("\r\n")
    parts = first.split(" ", 2)
    status = int(parts[1])
    hdrs = {}
    for line in rest.split("\r\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            hdrs[k.strip().lower()] = v.strip()
    return status, hdrs


async def _iter_body(reader, hdrs):
    if "chunked" in hdrs.get("transfer-encoding", "").lower():
        while True:
            line = await reader.readuntil(b"\r\n")
            size = int(line.strip().split(b";")[0], 16)
            if size == 0:
                try:
                    await reader.readuntil(b"\r\n")
                except Exception:  # noqa: BLE001
                    pass
                return
            yield await reader.readexactly(size)
            await reader.readexactly(2)
    elif "content-length" in hdrs:
        n = int(hdrs["content-length"])
        while n > 0:
            b = await reader.read(min(n, 1 << 20))
            if not b:
                return
            n -= len(b)
            yield b
    else:
        while True:
            b = await reader.read(1 << 20)
            if not b:
                return
            yield b


async def http_request(ep, method, path, body=None, extra_headers=None, timeout=60.0):
    """Non-streaming request -> (status, body bytes)."""
    async def go():
        reader, writer = await asyncio.open_connection(ep.host, ep.port, ssl=ep.ssl)
        try:
            data = body or b""
            h = "%s %s%s HTTP/1.1\r\nHost: %s:%d\r\nAccept: */*\r\nConnection: close\r\n" % (
                method, ep.prefix, path, ep.host, ep.port)
            if body is not None:
                h += "Content-Type: application/json\r\nContent-Length: %d\r\n" % len(data)
            for k, v in (extra_headers or {}).items():
                h += "%s: %s\r\n" % (k, v)
            writer.write(h.encode() + b"\r\n" + data)
            await writer.drain()
            status, hdrs = await _read_headers(reader)
            chunks = []
            async for b in _iter_body(reader, hdrs):
                chunks.append(b)
            return status, b"".join(chunks)
        finally:
            writer.close()
    return await asyncio.wait_for(go(), timeout)


class StreamResult:
    __slots__ = ("t_send", "t_headers", "t_first", "t_last", "n_tok", "n_chunks", "gaps", "gen_ids", "usage",
                 "finish_reason", "status", "error", "prompt_ids_echoed", "cum_usage", "first_chunk_bytes")

    def __init__(self):
        self.t_send = self.t_headers = self.t_first = self.t_last = None
        self.n_tok = 0
        self.n_chunks = 0
        self.gaps = []          # (gap_s, tokens_in_chunk) for token-bearing chunks after the first
        self.gen_ids = None
        self.usage = None
        self.finish_reason = None
        self.status = None
        self.error = None
        self.prompt_ids_echoed = 0
        self.cum_usage = 0
        self.first_chunk_bytes = 0


async def stream_completion(ep, body_bytes, headers, timeout, want_ids):
    """POST /v1/completions (stream=True) and time every token-bearing SSE chunk."""
    res = StreamResult()
    if want_ids:
        res.gen_ids = []
    pc = time.perf_counter

    async def go():
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(ep.host, ep.port, ssl=ep.ssl, limit=1 << 22), timeout=30.0)
        try:
            h = ("POST %s/v1/completions HTTP/1.1\r\nHost: %s:%d\r\nContent-Type: application/json\r\n"
                 "Accept: text/event-stream\r\nContent-Length: %d\r\nConnection: close\r\n" % (
                     ep.prefix, ep.host, ep.port, len(body_bytes)))
            for k, v in headers.items():
                h += "%s: %s\r\n" % (k, v)
            writer.write(h.encode() + b"\r\n")
            writer.write(body_bytes)
            await writer.drain()
            status, hdrs = await _read_headers(reader)
            res.t_headers = pc()
            res.status = status
            if status != 200:
                chunks = []
                async for b in _iter_body(reader, hdrs):
                    chunks.append(b)
                raise HTTPError(status, b"".join(chunks).decode("utf-8", "replace"))
            buf = b""
            scan = 0
            async for piece in _iter_body(reader, hdrs):
                now = pc()
                buf += piece
                if b"\r" in piece:  # tolerate CRLF-delimited SSE (vLLM/uvicorn use bare LF)
                    buf = buf.replace(b"\r\n", b"\n")
                    scan = 0
                while True:
                    i = buf.find(b"\n\n", scan)
                    if i < 0:
                        scan = max(0, len(buf) - 1)
                        break
                    ev = buf[:i]
                    buf = buf[i + 2:]
                    scan = 0
                    for line in ev.split(b"\n"):
                        line = line.rstrip(b"\r")
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == b"[DONE]":
                            continue
                        if res.n_chunks == 0:
                            res.first_chunk_bytes = len(data)
                        j = json.loads(data)
                        res.n_chunks += 1
                        if "error" in j and not j.get("choices"):
                            raise HTTPError(200, json.dumps(j["error"]))
                        u = j.get("usage")
                        n_new = None
                        chs = j.get("choices") or []
                        for ch in chs:
                            tids = ch.get("token_ids")
                            if tids is not None:
                                n_new = (n_new or 0) + len(tids)
                                if res.gen_ids is not None:
                                    res.gen_ids.extend(tids)
                            pti = ch.get("prompt_token_ids")
                            if pti:
                                res.prompt_ids_echoed = len(pti)
                            if ch.get("finish_reason"):
                                res.finish_reason = ch["finish_reason"]
                        if u:
                            ct = u.get("completion_tokens")
                            if ct is not None and n_new is None and chs:
                                n_new = max(0, ct - res.cum_usage)
                            if ct is not None:
                                res.cum_usage = max(res.cum_usage, ct)
                            if not chs:
                                res.usage = u
                        if n_new is None and chs:
                            n_new = 1 if any(ch.get("text") for ch in chs) else 0
                        if n_new:
                            if res.t_first is None:
                                res.t_first = now
                            else:
                                res.gaps.append((now - res.t_last, n_new))
                            res.t_last = now
                            res.n_tok += n_new
        finally:
            writer.close()

    res.t_send = pc()
    try:
        await asyncio.wait_for(go(), timeout)
    except HTTPError as e:
        res.error = str(e)
    except asyncio.TimeoutError:
        res.error = "timeout after %.0fs" % timeout
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001  (connection reset, bad chunk framing, bad JSON, ...)
        res.error = "%s: %s" % (type(e).__name__, str(e)[:200])
    return res


# ----------------------------------------------------------------------------------------------------------------
# stats helpers
# ----------------------------------------------------------------------------------------------------------------

def pct(vals, qs=(10, 50, 90, 99)):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return {}
    out = {}
    for q in qs:
        i = min(len(v) - 1, max(0, int(math.ceil(q / 100.0 * len(v))) - 1))
        out["p%g" % q] = round(v[i], 4)
    out["mean"] = round(sum(v) / len(v), 4)
    out["max"] = round(v[-1], 4)
    out["n"] = len(v)
    return out


class LogHist:
    """Fixed log-spaced histogram (1e-5 s .. 1e3 s) for chunk gaps; weighted by tokens or by chunks."""

    def __init__(self, lo=1e-5, hi=1e3, bins_per_decade=100):
        self.lo = math.log10(lo)
        self.bpd = bins_per_decade
        self.n = int((math.log10(hi) - self.lo) * bins_per_decade) + 2
        self.c = [0] * self.n
        self.total = 0

    def add(self, x, w=1):
        if x <= 0:
            i = 0
        else:
            i = int((math.log10(x) - self.lo) * self.bpd) + 1
            i = min(max(i, 0), self.n - 1)
        self.c[i] += w
        self.total += w

    def quantile(self, q):
        if not self.total:
            return None
        target = q * self.total
        acc = 0
        for i, n in enumerate(self.c):
            acc += n
            if acc >= target:
                return 10 ** (self.lo + (i - 0.5) / self.bpd) if i > 0 else 0.0
        return None

    def summary(self, scale=1000.0):
        if not self.total:
            return {}
        return {"p50": round(self.quantile(0.5) * scale, 3), "p90": round(self.quantile(0.9) * scale, 3),
                "p99": round(self.quantile(0.99) * scale, 3), "p999": round(self.quantile(0.999) * scale, 3),
                "n": self.total}


def overlap(a0, a1, w0, w1):
    return max(0.0, min(a1, w1) - max(a0, w0))


# ----------------------------------------------------------------------------------------------------------------
# replay engine
# ----------------------------------------------------------------------------------------------------------------

class Runner:
    def __init__(self, args, sessions, model, ep):
        self.args = args
        self.sessions = sessions
        self.model = model
        self.ep = ep
        self.rng = random.Random(args.seed)
        self.run_id = "%s-%04x" % (time.strftime("%Y%m%d%H%M%S"), random.SystemRandom().randrange(1 << 16))
        self.order = list(range(len(sessions)))
        if args.order == "shuffle":
            self.rng.shuffle(self.order)
        elif args.order == "longest-first":
            self.order.sort(key=lambda i: -sum(c["prompt_len"] for c in sessions[i]["calls"]))
        self.queue = collections.deque(self.order)
        self.epoch = 0
        self.stop = asyncio.Event()
        self.slots_done = asyncio.Event()
        self.fout = open(os.path.join(args.out, "calls.jsonl"), "a")
        self.cnt = collections.Counter()
        self.err_kinds = collections.Counter()
        self.sess_records = []
        self.calls = []        # compact per-call tuples for the summary
        self.gap_hist_chunk = LogHist()
        self.gap_hist_token = LogHist()
        self.inflight = 0
        self.active = 0
        self.req_counter = 0
        self.t_begin_wall = None
        self.t_begin_pc = None
        self.deadline_pc = None
        self.last_progress = None

    # ---- helpers
    def wall(self, pc_t):
        return self.t_begin_wall + (pc_t - self.t_begin_pc) if pc_t is not None else None

    def next_session(self):
        if self.stop.is_set():
            return None
        if not self.queue:
            if not self.args.loop:
                return None
            self.epoch += 1
            order = list(self.order)
            self.rng.shuffle(order)
            self.queue.extend(order)
        return self.queue.popleft(), self.epoch

    def salt(self, sidx, epoch):
        m = self.args.cache_salt
        if m == "none":
            return None
        if m == "session":
            return "%s-e%d-s%d" % (self.run_id, epoch, sidx)
        return "%s-e%d" % (self.run_id, epoch)

    def past_deadline(self):
        return self.stop.is_set() or (self.deadline_pc is not None and time.perf_counter() >= self.deadline_pc)

    async def sleep_until_or_stop(self, seconds):
        if seconds <= 0:
            return
        try:
            await asyncio.wait_for(self.stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    # ---- one session
    async def run_session(self, slot, sidx, epoch):
        a = self.args
        s = self.sessions[sidx]
        calls = s["calls"]
        if a.max_calls_per_session:
            calls = calls[:a.max_calls_per_session]
        cur_ids = array.array("I")
        cur_json = ""
        prev_gen = None
        salt = self.salt(sidx, epoch)
        skey = "%s:e%d:s%d" % (self.run_id, epoch, sidx)
        t_sess0 = time.perf_counter()
        end_reason = "completed"
        n_done = 0
        think_total = 0.0
        spliced_any = False
        self.active += 1
        try:
            for k, c in enumerate(calls):
                think = 0.0
                if k > 0:
                    rec_think = calls[k - 1].get("think_after_s") or 0.0
                    think = min(rec_think * a.think_scale, a.think_cap) if a.think_scale > 0 else 0.0
                    if think > 0:
                        await self.sleep_until_or_stop(think)
                        think_total += think
                if self.past_deadline():
                    end_reason = "deadline"
                    break
                # ---- rebuild the prompt
                shared = c["shared_prefix_len"]
                delta = c["delta"]
                full_ext = shared == len(cur_ids) and k > 0
                if shared == 0:
                    base_ids, base_json = array.array("I"), ""
                elif full_ext:
                    base_ids, base_json = cur_ids, cur_json
                else:
                    base_ids = cur_ids[:shared]
                    base_json = ",".join(map(str, base_ids))
                n_splice = 0
                if (a.prefix_mode != "recorded" and full_ext and prev_gen and c.get("splice_len") is not None
                        and len(delta) > 0):
                    # splice-exact: only where the recorded output length == rendered assistant span + stop token
                    # (token-count identity), i.e. where production reuse of the generated KV is near-certain
                    if a.prefix_mode == "splice" or calls[k - 1].get("output_len") == c["splice_len"] + 1:
                        n_splice = min(c["splice_len"] + 1, len(prev_gen), len(delta))
                if n_splice:
                    new_delta = array.array("I", prev_gen[:n_splice])
                    new_delta.extend(delta[n_splice:])
                    spliced_any = True
                else:
                    new_delta = delta
                ids = base_ids + new_delta
                dj = ",".join(map(str, new_delta))
                pjson = base_json + ("," if base_json and dj else "") + dj
                if a.verify_prompts and not spliced_any:
                    # once a session has been spliced its prompts differ from the recording by design
                    self.cnt["prompts_verified"] += 1
                    if len(ids) != c["prompt_len"] or ids_sha1(ids) != c["prompt_sha1"]:
                        self.cnt["prompt_sha1_mismatch"] += 1
                P = len(ids)
                out_target = c.get("output_len") or 1
                if a.max_output:
                    out_target = min(out_target, a.max_output)
                skip = None
                if a.max_context:
                    if P + 1 > a.max_context:
                        skip = "prompt_exceeds_max_context"
                    else:
                        out_target = max(1, min(out_target, a.max_context - P))
                if skip:
                    self.cnt["calls_skipped_context"] += 1
                    self._write_call(dict(session_idx=sidx, session_id=s["session_id"], epoch=epoch, call_idx=k,
                                          prompt_len=P, skipped=skip))
                    cur_ids, cur_json, prev_gen = ids, pjson, None
                    if a.over_context == "end":
                        end_reason = "context"
                        break
                    continue
                # ---- send
                self.req_counter += 1
                body = ('{"model":%s,"prompt":[%s],"max_tokens":%d,"min_tokens":%d,"ignore_eos":true,'
                        '"temperature":%s,"top_p":%s,"stream":true,'
                        '"stream_options":{"include_usage":true,"continuous_usage_stats":true}' % (
                            json.dumps(self.model), pjson, out_target, out_target,
                            json.dumps(a.temperature), json.dumps(a.top_p)))
                want_ids = a.prefix_mode != "recorded"
                if want_ids:
                    body += ',"return_token_ids":true'
                if salt:
                    body += ',"cache_salt":%s' % json.dumps(salt)
                if a.user_field:
                    body += ',"user":%s' % json.dumps(skey)
                if a.extra_body:
                    body += "," + a.extra_body_json
                body += "}"
                headers = {"X-Request-Id": "%s-%d" % (skey, k), "X-Replay-Session": skey, "X-Replay-Call": str(k)}
                if a.session_header:
                    headers[a.session_header] = skey
                for hk, hv in a.header_kv:
                    headers[hk] = hv
                self.inflight += 1
                try:
                    bb = body.encode()
                    for attempt in range(a.connect_retries + 1):
                        if attempt:
                            headers["X-Request-Id"] = "%s-%d-r%d" % (skey, k, attempt)
                        r = await stream_completion(self.ep, bb, headers, a.timeout, want_ids)
                        # retry only connection-level failures before any HTTP response (refused/reset, listen
                        # backlog); never timeouts or anything after response headers
                        if (r.error is None or r.status is not None or r.error.startswith("timeout")
                                or attempt == a.connect_retries):
                            break
                        self.cnt["connect_retries"] += 1
                        await asyncio.sleep(0.5 * (attempt + 1))
                finally:
                    self.inflight -= 1
                t_end = time.perf_counter()
                rec = self._record(s, sidx, epoch, k, len(calls), c, P, shared, n_splice, out_target, think, r, t_end)
                self._write_call(rec)
                if r.error:
                    self.cnt["calls_error"] += 1
                    kind = r.error.split(":")[0][:60]
                    self.err_kinds[kind] += 1
                    cur_ids, cur_json, prev_gen = ids, pjson, None
                    if a.on_error == "end":
                        end_reason = "error"
                        break
                    await self.sleep_until_or_stop(1.0)
                    continue
                self.cnt["calls_ok"] += 1
                n_done += 1
                if r.n_tok != out_target:
                    self.cnt["calls_output_len_mismatch"] += 1
                cur_ids, cur_json = ids, pjson
                if want_ids:
                    if r.gen_ids is not None and len(r.gen_ids) == r.n_tok and r.n_tok > 0:
                        prev_gen = r.gen_ids
                    else:
                        prev_gen = None
                        self.cnt["splice_unavailable_no_token_ids"] += 1
            else:
                end_reason = "completed"
        finally:
            self.active -= 1
        t_sess1 = time.perf_counter()
        rec_wall = s.get("recorded_wall_s")
        if a.max_calls_per_session and len(s["calls"]) > a.max_calls_per_session and calls:
            rec_wall = (calls[-1].get("t_done") or 0) - (calls[0].get("t_first") or 0)
        rec_model = sum((x.get("hub") or {}).get("latency_s") or 0.0 for x in calls)
        rec_think = sum(x.get("think_after_s") or 0.0 for x in calls[:-1])
        self.sess_records.append(dict(session_idx=sidx, session_id=s["session_id"], source=s.get("source"),
                                      epoch=epoch, slot=slot, end=end_reason, calls_done=n_done,
                                      n_calls=len(calls), t0=self.wall(t_sess0), wall_s=round(t_sess1 - t_sess0, 3),
                                      think_s=round(think_total, 3), model_s=round(t_sess1 - t_sess0 - think_total, 3),
                                      recorded_wall_s=rec_wall, recorded_hub_latency_s=round(rec_model, 3),
                                      recorded_think_s=round(rec_think, 3)))
        self.cnt["sessions_" + end_reason] += 1

    def _record(self, s, sidx, epoch, k, n_calls, c, P, shared, n_splice, out_target, think, r, t_end):
        ttft = (r.t_first - r.t_send) if r.t_first is not None else None
        dec_s = (r.t_last - r.t_first) if r.t_first is not None and r.t_last is not None else None
        dec_tps = (r.n_tok - 1) / dec_s if dec_s and r.n_tok > 1 else None
        g = r.gaps
        tok_gaps = []
        for gap, n in g:
            self.gap_hist_chunk.add(gap, 1)
            self.gap_hist_token.add(gap / n, n)
            tok_gaps.append(gap)
        tok_gaps.sort()

        def q(v, x):
            return round(v[min(len(v) - 1, int(x * len(v)))] * 1000, 2) if v else None
        u = r.usage or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") if u else None
        rec = dict(session_idx=sidx, session_id=s["session_id"], source=s.get("source"), epoch=epoch, call_idx=k,
                   n_calls=n_calls, prompt_len=P, shared_prefix_len=shared, new_tokens=P - shared, spliced=n_splice,
                   output_target=out_target, output_got=r.n_tok, t_send=round(self.wall(r.t_send), 4),
                   ttft_s=round(ttft, 4) if ttft is not None else None,
                   e2e_s=round(t_end - r.t_send, 4), decode_s=round(dec_s, 4) if dec_s is not None else None,
                   decode_tok_s=round(dec_tps, 2) if dec_tps else None, n_chunks=r.n_chunks,
                   chunk_gap_ms_p50=q(tok_gaps, 0.5), chunk_gap_ms_p90=q(tok_gaps, 0.9),
                   chunk_gap_ms_max=round(tok_gaps[-1] * 1000, 2) if tok_gaps else None,
                   think_before_s=round(think, 3), usage_prompt_tokens=u.get("prompt_tokens"),
                   usage_completion_tokens=u.get("completion_tokens"), cached_tokens=cached,
                   prompt_ids_echoed=r.prompt_ids_echoed or None, finish_reason=r.finish_reason,
                   status=r.status, error=r.error,
                   rec_output_len=c.get("output_len"), rec_hub_latency_s=(c.get("hub") or {}).get("latency_s"),
                   rec_hub_cached=(c.get("hub") or {}).get("cached_tokens"))
        self.calls.append((r.t_send, r.t_first, r.t_last, t_end, P, P - shared, r.n_tok, ttft, dec_tps,
                           t_end - r.t_send, cached, r.error is None, out_target))
        return rec

    def _write_call(self, rec):
        self.fout.write(json.dumps(rec, separators=(",", ":")) + "\n")

    # ---- slots / progress
    async def slot(self, i):
        a = self.args
        if a.ramp > 0:
            await self.sleep_until_or_stop(i * a.ramp / max(1, a.concurrency))
        while not self.past_deadline():
            item = self.next_session()
            if item is None:
                return
            await self.run_session(i, item[0], item[1])

    async def progress(self):
        a = self.args
        last_n = 0
        last_t = time.perf_counter()
        while not self.slots_done.is_set():
            try:
                await asyncio.wait_for(self.slots_done.wait(), timeout=a.print_every)
            except asyncio.TimeoutError:
                pass
            now = time.perf_counter()
            new = self.calls[last_n:]
            last_n = len(self.calls)
            gen = sum(x[6] for x in new)
            ttfts = sorted(x[7] for x in new if x[7] is not None)
            dts = sorted(x[8] for x in new if x[8] is not None)
            print("[%6.0fs] sessions active %d done %d | in-flight %d | calls ok %d err %d | last %.0fs: %d calls, "
                  "gen %.0f tok/s, ttft p50 %s s, decode p50 %s tok/s" % (
                      now - self.t_begin_pc, self.active, self.cnt["sessions_completed"], self.inflight,
                      self.cnt["calls_ok"], self.cnt["calls_error"], now - last_t, len(new), gen / max(1e-9, now - last_t),
                      "%.2f" % ttfts[len(ttfts) // 2] if ttfts else "-",
                      "%.1f" % dts[len(dts) // 2] if dts else "-"), flush=True)
            last_t = now

    async def run(self):
        a = self.args
        self.t_begin_wall = time.time()
        self.t_begin_pc = time.perf_counter()
        if a.duration:
            self.deadline_pc = self.t_begin_pc + a.duration
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._on_signal)
            except (NotImplementedError, RuntimeError):
                pass
        ptask = asyncio.ensure_future(self.progress())
        slots = [asyncio.ensure_future(self.slot(i)) for i in range(a.concurrency)]
        if a.duration:
            # at the deadline no new call starts (think sleeps wake up); in-flight calls may finish for up to
            # --drain-timeout, then they are cancelled
            loop.call_later(a.duration, self.stop.set)
            done, pending = await asyncio.wait(slots, timeout=a.duration + a.drain_timeout)
            for t in pending:
                t.cancel()
            if pending:
                self.cnt["slots_cancelled_after_drain"] += len(pending)
                await asyncio.gather(*pending, return_exceptions=True)
        else:
            await asyncio.gather(*slots)
        self.t_end_pc = time.perf_counter()
        self.stop.set()
        self.slots_done.set()
        await ptask
        self.fout.close()

    def _on_signal(self):
        if self.stop.is_set():
            print("second signal: exiting now", flush=True)
            os._exit(130)
        print("signal: no new calls; waiting for in-flight calls (signal again to abort)", flush=True)
        self.stop.set()

    # ---- summary
    def summary(self, metrics_tsv=None):
        a = self.args
        t0 = self.t_begin_pc + a.warmup
        t1 = min(self.t_end_pc, self.deadline_pc) if self.deadline_pc else self.t_end_pc
        win = max(1e-9, t1 - t0)
        ok = [x for x in self.calls if x[11]]
        # token attribution: prompt tokens at first token, generated tokens spread over [t_first, t_last]
        gen_w = prompt_w = new_w = 0.0
        for x in ok:
            tf, tl = x[1], x[2]
            if tf is None:
                continue
            if t0 <= tf <= t1:
                prompt_w += x[4]
                new_w += x[5]
            n = x[6]
            if tl is not None and tl > tf:
                gen_w += n * overlap(tf, tl, t0, t1) / (tl - tf)
            elif t0 <= tf <= t1:
                gen_w += n
        calls_completed_in_win = sum(1 for x in ok if t0 <= x[3] <= t1)
        inflight_area = sum(overlap(x[0], x[3], t0, t1) for x in self.calls)
        meas = [x for x in ok if x[0] >= t0]
        sess_done = [r for r in self.sess_records if r["end"] == "completed"]
        ratio = [r["wall_s"] / r["recorded_wall_s"] for r in sess_done if r.get("recorded_wall_s")]
        mratio = [r["model_s"] / r["recorded_hub_latency_s"] for r in sess_done if r.get("recorded_hub_latency_s")]
        cached = [x[10] for x in meas if x[10] is not None]
        out = {
            "run_id": self.run_id,
            "model": self.model,
            "base_url": a.base_url,
            "args": {k: v for k, v in vars(a).items() if k not in ("header_kv", "extra_body_json")},
            "wall_s": round(self.t_end_pc - self.t_begin_pc, 2),
            "window": {"t0_offset_s": a.warmup, "duration_s": round(win, 2),
                       "note": "calls sent after warmup; rates use token attribution inside [t0, t1]"},
            "counts": dict(self.cnt),
            "errors_by_kind": dict(self.err_kinds),
            "throughput": {
                "calls_completed_per_s": round(calls_completed_in_win / win, 4),
                "generated_tok_s": round(gen_w / win, 1),
                "prompt_tok_s": round(prompt_w / win, 1),
                "new_prompt_tok_s(vs prev call)": round(new_w / win, 1),
                "mean_inflight_requests": round(inflight_area / win, 2),
                "server_cached_prompt_frac": round(sum(cached) / max(1, sum(x[4] for x in meas if x[10] is not None)), 4)
                if cached else None,
            },
            "ttft_s": pct([x[7] for x in meas]),
            "e2e_s": pct([x[9] for x in meas]),
            "decode_tok_s_per_request": pct([x[8] for x in meas], qs=(10, 50, 90)),
            "decode_tok_s_per_request(out>=256)": pct([x[8] for x in meas if x[6] >= 256], qs=(10, 50, 90)),
            "tpot_ms_per_request": pct([1000.0 / x[8] for x in meas if x[8]], qs=(10, 50, 90, 99)),
            "chunk_gap_ms": self.gap_hist_chunk.summary(),
            "per_token_itl_ms(token-weighted)": self.gap_hist_token.summary(),
            "prompt_len": pct([x[4] for x in meas]),
            "output_len": pct([x[6] for x in meas]),
            "sessions": {
                "started": len(self.sess_records),
                "by_end": dict(collections.Counter(r["end"] for r in self.sess_records)),
                "completed_wall_s": pct([r["wall_s"] for r in sess_done]),
                "completed_wall_over_recorded_wall": pct(ratio, qs=(10, 50, 90)),
                "completed_model_time_over_recorded_hub_latency": pct(mratio, qs=(10, 50, 90)),
            },
        }
        if metrics_tsv and os.path.exists(metrics_tsv):
            try:
                import scrape_metrics
                rows = scrape_metrics.read_tsv(metrics_tsv)
                out["server_metrics_window"] = scrape_metrics.window_summary(
                    rows, self.wall(t0), self.wall(t1))
            except Exception as e:  # noqa: BLE001
                out["server_metrics_window"] = {"error": repr(e)}
        return out


def choose_sessions(sessions, args):
    rng = random.Random(args.seed)
    idx = list(range(len(sessions)))
    rng.shuffle(idx)
    if args.sessions and args.sessions < len(idx):
        idx = idx[:args.sessions]
    idx.sort()
    return [sessions[i] for i in idx]


def dry_run(sessions, args):
    """Rebuild every prompt in recorded mode and check it against the build-time sha1."""
    n = bad = 0
    tok = 0
    for s in sessions:
        cur = array.array("I")
        for c in s["calls"]:
            shared = c["shared_prefix_len"]
            ids = cur[:shared] + c["delta"]
            n += 1
            tok += len(ids)
            if len(ids) != c["prompt_len"] or ids_sha1(ids) != c["prompt_sha1"]:
                bad += 1
            cur = ids
    print(json.dumps({"sessions": len(sessions), "calls": n, "prompt_tokens": tok, "sha1_mismatch": bad}))
    return bad == 0


async def resolve_model(ep, timeout=30.0):
    st, body = await http_request(ep, "GET", "/v1/models", timeout=timeout)
    if st != 200:
        raise SystemExit("GET /v1/models -> HTTP %s: %s" % (st, body[:200]))
    return json.loads(body)["data"][0]["id"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", nargs="+", required=True, help="replay set(s) from build_replay_set.py")
    ap.add_argument("--source", action="append", default=None, help="only sessions of this source (repeatable)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", help="server root or .../v1")
    ap.add_argument("--model", default=None, help="model name (default: first id of GET /v1/models)")
    ap.add_argument("--sessions", type=int, default=0, help="use N sessions (seeded sample; 0 = all)")
    ap.add_argument("--concurrency", type=int, default=8, help="concurrent sessions (closed loop)")
    ap.add_argument("--duration", type=float, default=0, help="stop issuing calls after S seconds (0 = run all)")
    ap.add_argument("--drain-timeout", type=float, default=900, help="after --duration, wait this long for in-flight calls")
    ap.add_argument("--warmup", type=float, default=0, help="exclude calls sent in the first S seconds from summary")
    ap.add_argument("--ramp", type=float, default=0, help="stagger slot starts uniformly over S seconds")
    ap.add_argument("--loop", action="store_true", help="recycle sessions (new cache salt epoch) when the list runs out")
    ap.add_argument("--order", choices=["shuffle", "file", "longest-first"], default="shuffle")
    ap.add_argument("--think-scale", type=float, default=1.0, help="x recorded think time (0 = no think)")
    ap.add_argument("--think-cap", type=float, default=600.0, help="cap one think gap (s)")
    ap.add_argument("--max-context", type=int, default=0, help="server max_model_len: clip output, skip/end longer calls")
    ap.add_argument("--over-context", choices=["end", "skip"], default="end",
                    help="when a prompt alone exceeds --max-context: end the session or skip the call")
    ap.add_argument("--max-output", type=int, default=0, help="cap output tokens per call (0 = recorded)")
    ap.add_argument("--max-calls-per-session", type=int, default=0)
    ap.add_argument("--prefix-mode", choices=["splice", "splice-exact", "recorded"], default="splice",
                    help="splice: reuse the server's generated ids for the re-rendered assistant turn (upper bound on "
                         "prefix reuse); splice-exact: only where recorded output == rendered span + 1; recorded: "
                         "byte-exact recorded prompts (assistant turn always re-prefilled; lower bound)")
    ap.add_argument("--cache-salt", choices=["run", "session", "none"], default="run",
                    help="vLLM cache_salt: 'run' = one salt per run+epoch (sessions still share the system prompt), "
                         "'session' = no cross-session sharing, 'none' = do not send (reuses older runs' cache!)")
    ap.add_argument("--temperature", type=float, default=1.0, help="recorded harness used 1.0")
    ap.add_argument("--top-p", type=float, default=0.95, help="recorded harness used 0.95")
    ap.add_argument("--extra-body", default=None, help="JSON object merged into every request body")
    ap.add_argument("--header", action="append", default=[], help="extra HTTP header 'Name: value' (repeatable)")
    ap.add_argument("--session-header", default=None, help="send the session key in this header (router affinity)")
    ap.add_argument("--user-field", action="store_true", help="send the session key as the OpenAI 'user' field")
    ap.add_argument("--on-error", choices=["end", "continue"], default="end")
    ap.add_argument("--connect-retries", type=int, default=3,
                    help="retries for failures before any HTTP response (connection refused/reset)")
    ap.add_argument("--timeout", type=float, default=3600, help="per-request timeout (s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify-prompts", action="store_true", help="sha1-check recorded-mode prompts while sending")
    ap.add_argument("--dry-run", action="store_true", help="only rebuild prompts offline and check sha1s")
    ap.add_argument("--metrics-url", action="append", default=[], help="vLLM /metrics URL(s) to poll during the run")
    ap.add_argument("--metrics-interval", type=float, default=5.0)
    ap.add_argument("--print-every", type=float, default=30.0)
    ap.add_argument("--out", default=None, help="output dir (required unless --dry-run)")
    args = ap.parse_args()
    args.header_kv = []
    for h in args.header:
        k, _, v = h.partition(":")
        args.header_kv.append((k.strip(), v.strip()))
    args.extra_body_json = None
    if args.extra_body:
        eb = json.loads(args.extra_body)
        args.extra_body_json = json.dumps(eb)[1:-1]
        if not args.extra_body_json:
            args.extra_body = None

    t = time.time()
    headers, sessions = load_replay(args.replay, set(args.source) if args.source else None)
    sessions = choose_sessions(sessions, args)
    print("loaded %d sessions / %d calls from %s in %.1fs" % (
        len(sessions), sum(len(s["calls"]) for s in sessions), ", ".join(args.replay), time.time() - t), flush=True)
    if not sessions:
        raise SystemExit("no sessions")
    if args.dry_run:
        sys.exit(0 if dry_run(sessions, args) else 1)
    if not args.out:
        raise SystemExit("--out is required")
    os.makedirs(args.out, exist_ok=True)
    ep = Endpoint(args.base_url)

    async def amain():
        model = args.model or await resolve_model(ep)
        with open(os.path.join(args.out, "client_args.json"), "w") as f:
            json.dump(dict(vars(args), model_resolved=model,
                           replay_headers=[{k: h.get(k) for k in ("format", "created", "render", "sources")}
                                           for h in headers]), f, indent=1)
        poller = None
        mt = os.path.join(args.out, "metrics.tsv")
        if args.metrics_url:
            import scrape_metrics
            urls = [u if u.rstrip("/").endswith("/metrics") else u.rstrip("/") + "/metrics" for u in args.metrics_url]
            poller = scrape_metrics.Poller(urls, mt, args.metrics_interval).start()
        runner = Runner(args, sessions, model, ep)
        print("run %s: model=%s concurrency=%d prefix_mode=%s cache_salt=%s think_scale=%g duration=%s" % (
            runner.run_id, model, args.concurrency, args.prefix_mode, args.cache_salt, args.think_scale,
            args.duration or "all"), flush=True)
        try:
            await runner.run()
        finally:
            if poller:
                await asyncio.get_running_loop().run_in_executor(None, poller.stop)
        summ = runner.summary(mt if poller else None)
        with open(os.path.join(args.out, "sessions.jsonl"), "w") as f:
            for r in runner.sess_records:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summ, f, indent=1)
        print(json.dumps({k: summ[k] for k in ("counts", "throughput", "ttft_s", "e2e_s", "decode_tok_s_per_request",
                                               "chunk_gap_ms", "sessions")}, indent=1))
        if "server_metrics_window" in summ:
            print(json.dumps(summ["server_metrics_window"], indent=1))
    asyncio.run(amain())


if __name__ == "__main__":
    main()
