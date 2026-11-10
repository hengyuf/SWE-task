#!/usr/bin/env python3
"""Tiny OpenAI-compatible stub of a vLLM /v1/completions server for offline tests of replay_load.py.
Stdlib only (python >= 3.8).

  python3 stub_server.py --port 18080 --decode-tok-s 200 --prefill-tok-s 2e5 --ttft-base 0.05 --max-chunk 4 \
      --max-running 64 --log /tmp/xyz/stub_requests.jsonl

Emulates what the replay client depends on:
  * POST /v1/completions, stream=true (SSE over chunked HTTP/1.1): prompt = list of token ids, emits exactly
    max_tokens tokens (ignore_eos/min_tokens semantics) in chunks of 1..--max-chunk tokens (MTP-like bursts) at
    --decode-tok-s per request, after a TTFT of --ttft-base + uncached_prompt_tokens / --prefill-tok-s (+ queueing
    when more than --max-running requests are active). return_token_ids -> token_ids per chunk and
    prompt_token_ids in the first chunk (like vLLM); stream_options.include_usage / continuous_usage_stats.
  * a block-hash prefix cache (64-token blocks, chained hashes, cache_salt mixed into the first block, prompt
    blocks + generated blocks inserted, hits capped at prompt_len-1 like vLLM); usage.prompt_tokens_details.cached_tokens.
  * --engines N: sticky per-session engines (hash of X-Replay-Session) with their own caches and labeled metrics.
  * GET /metrics with vLLM v1 metric names (gauges, counters, histogram _sum/_count, MTP spec-decode counters).
  * GET /v1/models; HTTP 400 when prompt+max_tokens > --max-model-len (vLLM-like error body).
Every request is logged to --log (JSONL) with headers, prompt sha1 (little-endian uint32) and generated ids.
"""
import argparse
import array
import base64
import collections
import hashlib
import json
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BLOCK = 64


def ids_bytes(ids):
    a = array.array("I", ids)
    if sys.byteorder != "little":
        a.byteswap()
    return a.tobytes()


class Engine:
    def __init__(self, idx, cache_blocks):
        self.idx = idx
        self.cache = collections.OrderedDict()   # block hash -> None (LRU order)
        self.cache_blocks = cache_blocks
        self.running = 0
        self.waiting = 0
        self.c = collections.Counter()
        self.h = collections.defaultdict(lambda: [0.0, 0])  # hist name -> [sum, count]
        self.per_pos = collections.Counter()

    def block_hashes(self, ids, salt):
        hs = []
        prev = hashlib.sha1((salt or "").encode()).digest()
        nfull = len(ids) // BLOCK
        b = ids_bytes(ids[:nfull * BLOCK])
        for i in range(nfull):
            prev = hashlib.sha1(prev + b[i * BLOCK * 4:(i + 1) * BLOCK * 4]).digest()
            hs.append(prev)
        return hs

    def lookup(self, hs):
        n = 0
        for h in hs:
            if h in self.cache:
                self.cache.move_to_end(h)
                n += 1
            else:
                break
        return n

    def insert(self, hs):
        for h in hs:
            self.cache[h] = None
            self.cache.move_to_end(h)
        while len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)


class State:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.engines = [Engine(i, args.cache_blocks) for i in range(args.engines)]
        self.sem = threading.Semaphore(args.max_running)
        self.rid = 0
        self.log = open(args.log, "a") if args.log else None

    def engine_for(self, key):
        if len(self.engines) == 1 or not key:
            return self.engines[0]
        return self.engines[int(hashlib.md5(key.encode()).hexdigest(), 16) % len(self.engines)]


def prom(state, model):
    lines = []
    lab = lambda e, extra="": '{engine="%d",model_name="%s"%s}' % (e.idx, model, extra)  # noqa: E731
    with state.lock:
        for e in state.engines:
            lines.append("vllm:num_requests_running%s %d" % (lab(e), e.running))
            lines.append("vllm:num_requests_waiting%s %d" % (lab(e), e.waiting))
            lines.append("vllm:num_requests_waiting_by_reason%s %d" % (lab(e, ',reason="capacity"'), e.waiting))
            lines.append("vllm:num_requests_waiting_by_reason%s 0" % lab(e, ',reason="deferred"'))
            lines.append("vllm:kv_cache_usage_perc%s %.6f" % (lab(e), min(1.0, len(e.cache) / float(e.cache_blocks))))
            for name, key in (("prompt_tokens", "prompt"), ("generation_tokens", "gen"),
                              ("prompt_tokens_cached", "cached"), ("prefix_cache_queries", "pc_q"),
                              ("prefix_cache_hits", "pc_h"), ("num_preemptions", "preempt"),
                              ("spec_decode_num_drafts", "drafts"), ("spec_decode_num_draft_tokens", "draft_tok"),
                              ("spec_decode_num_accepted_tokens", "accepted")):
                lines.append("vllm:%s_total%s %d" % (name, lab(e), e.c[key]))
                lines.append("vllm:%s_created%s %.3f" % (name, lab(e), 1.7e9))
            lines.append("vllm:request_success_total%s %d" % (lab(e, ',finished_reason="length"'), e.c["req_ok"]))
            lines.append("vllm:request_success_total%s 0" % lab(e, ',finished_reason="stop"'))
            for pos in range(3):
                lines.append("vllm:spec_decode_num_accepted_tokens_per_pos_total%s %d" % (
                    lab(e, ',position="%d"' % pos), e.per_pos[pos]))
            for hname in ("time_to_first_token_seconds", "e2e_request_latency_seconds", "request_queue_time_seconds",
                          "request_prefill_time_seconds", "request_decode_time_seconds",
                          "inter_token_latency_seconds"):
                sm, cn = e.h[hname]
                lines.append('vllm:%s_bucket%s %d' % (hname, lab(e, ',le="+Inf"'), cn))
                lines.append("vllm:%s_sum%s %.6f" % (hname, lab(e), sm))
                lines.append("vllm:%s_count%s %d" % (hname, lab(e), cn))
    return "# stub vLLM metrics\n" + "\n".join(lines) + "\n"


def make_handler(state, args):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):  # quiet
            if args.verbose:
                sys.stderr.write("stub: " + (fmt % a) + "\n")

        def _send(self, code, body, ctype="application/json"):
            b = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b)
            self.close_connection = True

        def do_GET(self):
            if self.path.startswith("/v1/models"):
                self._send(200, json.dumps({"object": "list", "data": [{"id": args.model, "object": "model"}]}))
            elif self.path.startswith("/metrics"):
                self._send(200, prom(state, args.model), "text/plain; version=0.0.4")
            elif self.path.startswith("/health"):
                self._send(200, "")
            else:
                self._send(404, json.dumps({"error": "not found"}))

        def _chunk(self, data):
            self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
            self.wfile.flush()

        def do_POST(self):
            t_recv = time.time()
            if not self.path.startswith("/v1/completions"):
                return self._send(404, json.dumps({"error": "only /v1/completions"}))
            n = int(self.headers.get("Content-Length", "0"))
            try:
                req = json.loads(self.rfile.read(n))
            except Exception as e:  # noqa: BLE001
                return self._send(400, json.dumps({"error": {"message": "bad json: %r" % e, "code": 400}}))
            prompt = req.get("prompt")
            if not isinstance(prompt, list) or not all(isinstance(x, int) for x in prompt[:16]):
                return self._send(400, json.dumps({"error": {"message": "prompt must be token ids", "code": 400}}))
            mt = int(req.get("max_tokens") or 16)
            P = len(prompt)
            if args.max_model_len and P + mt > args.max_model_len:
                msg = ("This model's maximum context length is %d tokens. However, your request has %d input tokens "
                       "and requested %d output tokens." % (args.max_model_len, P, mt))
                return self._send(400, json.dumps({"error": {"message": msg, "type": "BadRequestError",
                                                             "code": 400}}))
            with state.lock:
                state.rid += 1
                rid = state.rid
            salt = req.get("cache_salt")
            skey = self.headers.get("X-Replay-Session", "")
            eng = state.engine_for(skey)
            want_ids = bool(req.get("return_token_ids"))
            so = req.get("stream_options") or {}
            inc_usage = bool(so.get("include_usage"))
            cont_usage = inc_usage and bool(so.get("continuous_usage_stats"))
            rng = random.Random(rid * 7919 + args.seed)
            hs = eng.block_hashes(prompt, salt)
            with state.lock:
                eng.waiting += 1
            t_q0 = time.time()
            state.sem.acquire()
            try:
                with state.lock:
                    eng.waiting -= 1
                    eng.running += 1
                    hit_blocks = eng.lookup(hs)
                    cached = min(hit_blocks * BLOCK, ((P - 1) // BLOCK) * BLOCK)
                    eng.insert(hs)
                    eng.c["prompt"] += P
                    eng.c["cached"] += cached
                    eng.c["pc_q"] += P
                    eng.c["pc_h"] += cached
                t_sched = time.time()
                time.sleep(args.ttft_base + (P - cached) / args.prefill_tok_s)
                gen = []
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "close")
                self.end_headers()
                created = int(t_recv)
                t_first = None
                first = True
                last_t = None
                while len(gen) < mt:
                    k = min(mt - len(gen), rng.randint(1, max(1, args.max_chunk)))
                    if not first:
                        time.sleep(k / args.decode_tok_s)
                    toks = [rng.randrange(1000, 150000) for _ in range(k)]
                    gen.extend(toks)
                    now = time.time()
                    if t_first is None:
                        t_first = now
                    ch = {"index": 0, "text": " tok" * k, "logprobs": None,
                          "finish_reason": "length" if len(gen) >= mt else None, "stop_reason": None}
                    if want_ids:
                        ch["token_ids"] = toks
                        if first:
                            ch["prompt_token_ids"] = prompt
                    obj = {"id": "cmpl-%d" % rid, "object": "text_completion", "created": created,
                           "model": args.model, "choices": [ch]}
                    if cont_usage:
                        obj["usage"] = {"prompt_tokens": P, "completion_tokens": len(gen), "total_tokens": P + len(gen)}
                    self._chunk(b"data: " + json.dumps(obj, separators=(",", ":")).encode() + b"\n\n")
                    with state.lock:
                        eng.c["gen"] += k
                        if not first:
                            eng.c["drafts"] += 1
                            eng.c["draft_tok"] += 3
                            eng.c["accepted"] += k - 1
                            for pos in range(min(3, k - 1)):
                                eng.per_pos[pos] += 1
                            h = eng.h["inter_token_latency_seconds"]
                            h[0] += now - last_t
                            h[1] += 1
                    last_t = now
                    first = False
                if inc_usage:
                    obj = {"id": "cmpl-%d" % rid, "object": "text_completion", "created": created,
                           "model": args.model, "choices": [],
                           "usage": {"prompt_tokens": P, "completion_tokens": len(gen), "total_tokens": P + len(gen),
                                     "prompt_tokens_details": {"cached_tokens": cached}}}
                    self._chunk(b"data: " + json.dumps(obj, separators=(",", ":")).encode() + b"\n\n")
                self._chunk(b"data: [DONE]\n\n")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                self.close_connection = True
                t_done = time.time()
                with state.lock:
                    eng.insert(eng.block_hashes(prompt + gen[:-1], salt))
                    eng.c["req_ok"] += 1
                    for hname, v in (("time_to_first_token_seconds", t_first - t_recv),
                                     ("e2e_request_latency_seconds", t_done - t_recv),
                                     ("request_queue_time_seconds", t_sched - t_q0),
                                     ("request_prefill_time_seconds", t_first - t_sched),
                                     ("request_decode_time_seconds", t_done - t_first)):
                        eng.h[hname][0] += v
                        eng.h[hname][1] += 1
            finally:
                with state.lock:
                    eng.running -= 1
                state.sem.release()
            if state.log:
                rec = {"rid": rid, "engine": eng.idx, "x_request_id": self.headers.get("X-Request-Id"),
                       "session": skey, "call": self.headers.get("X-Replay-Call"), "t_recv": round(t_recv, 4),
                       "t_first": round(t_first, 4), "t_done": round(t_done, 4), "prompt_len": P,
                       "prompt_sha1": hashlib.sha1(ids_bytes(prompt)).hexdigest(), "cache_salt": salt,
                       "max_tokens": mt, "min_tokens": req.get("min_tokens"), "ignore_eos": req.get("ignore_eos"),
                       "temperature": req.get("temperature"), "top_p": req.get("top_p"), "return_token_ids": want_ids,
                       "user": req.get("user"), "cached_tokens": cached,
                       "gen_ids_b64": base64.b64encode(ids_bytes(gen)).decode()}
                with state.lock:
                    state.log.write(json.dumps(rec) + "\n")
                    state.log.flush()

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="0 = pick a free port (printed on stdout)")
    ap.add_argument("--model", default="stub-glm-5.3")
    ap.add_argument("--decode-tok-s", type=float, default=200.0, help="per-request decode rate")
    ap.add_argument("--prefill-tok-s", type=float, default=200000.0, help="uncached prompt tokens per second")
    ap.add_argument("--ttft-base", type=float, default=0.03)
    ap.add_argument("--max-chunk", type=int, default=4, help="tokens per SSE chunk drawn from 1..N (MTP-like)")
    ap.add_argument("--max-running", type=int, default=256, help="concurrent requests before queueing")
    ap.add_argument("--max-model-len", type=int, default=0, help="reject prompt+max_tokens above this (0 = off)")
    ap.add_argument("--cache-blocks", type=int, default=1 << 20, help="prefix-cache capacity in 64-token blocks")
    ap.add_argument("--engines", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None, help="JSONL request log")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    state = State(args)
    class Server(ThreadingHTTPServer):
        request_queue_size = 4096   # listen backlog (default 5 drops bursts of concurrent connects)
        daemon_threads = True

    srv = Server((args.host, args.port), make_handler(state, args))
    srv.daemon_threads = True
    print("stub listening on http://%s:%d" % (args.host, srv.server_address[1]), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
