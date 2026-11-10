# GLM-5.3 agentic-SWE replay load generator (dfw)

Replays real hosted-GLM-5.3 OpenCode sessions at the token level against a vLLM endpoint. The goal is to profile
serving configs under production-like agentic SWE traffic: long multi-turn prompts, heavy prefix reuse, reasoning on,
and recorded tool gaps.

Everything here was built and tested offline on 2026-11-06. No GPU and no paid API were used.
Directory: `/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/tools/replay/`

| file | what | runs with |
|---|---|---|
| `build_replay_set.py` | ATP `http.jsonl` captures → replay set (`.jsonl.gz`). Renders each call with the GLM-5.3 chat template exactly as vLLM 0.29 does and tokenizes it | `.venv/bin/python` (transformers 5.12.1, tokenizers 0.22.2, jinja2 3.1.6, the same versions as the RL vLLM env), or any python with transformers or tokenizers+jinja2 (`--renderer builtin`) |
| `replay_load.py` | async closed-loop client for `/v1/completions` with prompt token ids, streaming. Writes per-call JSONL and a summary, and optionally polls `/metrics` | stdlib only, python ≥ 3.8 (login-node `python3` 3.9 or any container python) |
| `scrape_metrics.py` | polls one or more vLLM `/metrics` into a TSV and computes window means. Also used as a library by `replay_load.py` | stdlib only |
| `compare_runs.py` | markdown table comparing several run dirs (for sweeps and A/B) | stdlib only |
| `stub_server.py` | fake vLLM: SSE streaming, `return_token_ids`, block-hash prefix cache with salts, vLLM metric names. For offline tests | stdlib only |
| `tests/offline_test.py` | end-to-end test: build → stub → client, with exact checks | `python3` (client/stub) + `.venv` (build) |
| `data/deepswe_glm_64.jsonl.gz` (31.8 MB) | **64 DeepSWE sessions** (OpenCode + ATP, 256k context with compaction), 9,006 calls | |
| `data/synth_v2_glm_64.jsonl.gz` (28.8 MB) | **64 synthetic-task sessions** (50 tasks), 11,571 calls; prompts reach 542k | |
| `data/*.summary.json` | build config, validation counters, distributions (tables below) | |
| `logs/build_all.sh`, `logs/build_all.log` | exact commands and log that produced both sets (about 3 + 6 min on the login node, 8 workers) | |
| `.venv/` | private uv venv (python 3.12.13, 5.5k files, 153 MB). Only needed to build new sets. To recreate: `uv venv --python ~/.local/bin/python3.12 .venv && uv pip install --python .venv/bin/python 'transformers==5.12.1' 'tokenizers==0.22.2' 'jinja2==3.1.6'` | |

## Quick start on dfw

```bash
R=/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/tools/replay
# 0) sanity: rebuild every recorded prompt offline and check the build-time sha1s (both sets: 20,577 calls OK)
python3 $R/replay_load.py --replay $R/data/deepswe_glm_64.jsonl.gz --dry-run

# 1) functional smoke against a live server (4 sessions, 10 calls each, no think time)
python3 $R/replay_load.py --replay $R/data/deepswe_glm_64.jsonl.gz --base-url http://<node>:<port> \
    --sessions 4 --concurrency 4 --max-calls-per-session 10 --think-scale 0 --out runs/smoke

# 2) steady-state load point (closed loop, K sessions, recorded think time, server metrics every 5 s)
python3 $R/replay_load.py --replay $R/data/deepswe_glm_64.jsonl.gz --base-url http://<node>:<port> \
    --concurrency 32 --ramp 120 --warmup 300 --duration 1800 --loop --max-context 262144 \
    --metrics-url http://<node>:<port>/metrics --out runs/c32
# 3) sweep and compare
for K in 16 32 64 128; do python3 $R/replay_load.py ... --concurrency $K --out runs/c$K; done
python3 $R/compare_runs.py runs/c16 runs/c32 runs/c64 runs/c128

# metrics only (e.g. while real OpenCode traffic runs), then window means
python3 $R/scrape_metrics.py poll --url http://<node>:<port>/metrics --interval 5 --out m.tsv --duration 3600
python3 $R/scrape_metrics.py window m.tsv --calls runs/c32/calls.jsonl --warmup 300
```

The client is stdlib-only, so it runs on the login node. It opens raw sockets and ignores `http_proxy`;
`scrape_metrics` also bypasses env proxies. At high load, run it next to the server instead, inside the job's
container with `srun --jobid=<job> --overlap -w <node> ... python3 replay_load.py ...`.
One client process used 33% of a core to drive 27 calls/s of ~50k-token prompts plus 4.1k generated tok/s
(stub test below), so ~80 calls/s per process is feasible. For more load, split sessions across processes with
`--source` or `--seed`.

Server-side notes for vLLM:
- Add `--enable-prompt-tokens-details` if you want per-call `cached_tokens` in the client output
  (`completion/serving.py:451`). The `/metrics` prefix-hit columns work without it.
- `--max-model-len` must be ≥ 262144 to replay the DeepSWE set fully (see the context table below).
- Any served model name works; the client reads it from `GET /v1/models`.

## Input data and what was extracted

Sources (read-only):
- **DeepSWE GLM rows**: `swe-task/sft/teacher-deepswe-20261030/dataset/glm.jsonl.gz`. 264 rows; the campaign was
  cancelled partway, so this is partial. Each row has `attempt_dir`, `task_id`, `status`, `compactions`,
  `hub_calls_ok`, token totals, and `output`. `output` is the Responses-API item list
  (reasoning / function_call / function_call_output / message); it has no per-call usage or timestamps.
- **Synthetic-task GLM rows**: `swe-task/sft/teacher-rollouts-v2-20261025/features/glm.jsonl`. 252 rows covering
  50 tasks × 5 repeats; each row has `attempt_dir`.
- The per-call ground truth is `<attempt_dir>/http.jsonl` (31 MB to 1.5 GB per session; 52.6 GB for DeepSWE,
  74.2 GB for synthetic). This is the ATP harness's capture of every model call. Events per `model_call_id`:
  - `api_gate_wait_started` → `api_gate_acquired` → `request` → `response`, repeated per HTTP attempt.
    429 attempts are followed by `retry` / `rate_limit_wait`.
  - then `model_call_completed`.
  - `request.payload` is the exact body sent to `https://inference-api.nvidia.com/v1/chat/completions`:
    `model=nvidia/zai-org/glm-5.3`, `messages`, 10 OpenCode `tools`, `tool_choice=auto`, `temperature=1.0`,
    `top_p=0.95`, `max_tokens=131072`, `reasoning_effort="max"`.
  - Assistant history carries `reasoning_content`. The harness drops OpenCode's `reasoning` field; see
    `source_payload` versus `payload`.
  - `response.body` (status 200) is a non-streaming chat completion. Its `usage` has `prompt_tokens`,
    `completion_tokens` (reasoning included), `completion_tokens_details.{reasoning_tokens,
    accepted_prediction_tokens, rejected_prediction_tokens}` and `prompt_tokens_details.cached_tokens`.
- Within a session, calls are strictly sequential: there were 0 overlapping model calls in all 128 sessions.
- **Think time** = next call's first event − this call's `model_call_completed`. This is OpenCode plus tool
  execution in the sandbox. It excludes the hub's 429 waits, which happened inside the next call: 8,870 s for
  DeepSWE, 233 s for synthetic.

## How a replay set is built (`build_replay_set.py`)

1. **Select sessions.** Seeded shuffle with `--seed 0`, status pass/fail, one session per task first, then fill up
   to `--n-sessions`.
2. **Stream `http.jsonl`.** For every model call that got an HTTP 200, take the request payload, the usage, and the
   timestamps. Calls with no 200 are skipped: 0 for DeepSWE, 1 for synthetic.
3. **Render exactly like vLLM 0.29 `/v1/chat/completions`.**
   - Apply the vLLM message preprocessing:
     - `reasoning_content` is renamed to `reasoning` (`openai/chat_completion/protocol.py:524-547`).
     - It is rendered as `reasoning_content` (`chat_utils.py:1975-2004`).
     - Tool-call `arguments` are JSON-decoded to dicts; invalid JSON becomes `{}` (`chat_utils.py:2043-2100`).
   - Call `tokenizer.apply_chat_template(messages, tools=..., add_generation_prompt=True, tokenize=False,
     reasoning_effort=<payload value>)`. vLLM passes `reasoning_effort` too (`protocol.py:579`). The template maps
     anything other than low/high to `Max`.
   - Tokenize with `add_special_tokens=False`. The template already emits `[gMASK]<sop>`.
   - Template and tokenizer come from `/lustre/fsw/portfolios/nemotron/users/hengyuf/models/GLM-5.3` (HF revision
     ba32f8c; `chat_template.jinja` sha256 0e2b6268...). This is what the dfw server will load.
   - `clear_thinking` is false by default, so every past turn keeps `<think>reasoning</think>`.
4. **Store prompts as deltas.** Each prompt is `prompt_k = prompt_{k-1}[:shared_prefix_len] + delta_ids`
   (base64 little-endian uint32), plus `prompt_sha1`. Tokenization is incremental at the generation-prompt boundary
   `<|assistant|><think>`, which is exact because `<think>` is an added token. A full re-tokenization every 20 calls
   confirms it.
5. **Store per-call fields:** `output_len` (= hub `completion_tokens`), `reasoning_tokens`, `finish_reason`,
   `n_tool_calls`, `think_after_s`, `splice_len`, the original timing (`t_first/t_req/t_resp/t_done`), and the hub
   usage block (`hub.prompt_tokens/cached_tokens/latency_s/accepted|rejected_prediction_tokens/http_attempts/
   rate_limit_wait_s`). `splice_len` is the length of the re-rendered previous assistant turn at the head of the
   delta, i.e. the tokens before the first role token.

File format: line 1 is `{"type":"header",...}` (render config, template/tokenizer sha256, sources, build stats,
summary). Every other line is `{"type":"session","session_id","source","task_id","status","attempt_dir","meta",
"n_calls","sampling","recorded_wall_s","calls":[...]}`.

### Validation of the rendering (all 20,577 calls in both sets)

| check | result |
|---|---|
| hub `usage.prompt_tokens` − our rendered length | **+1 on 20,577 / 20,577 calls**, a constant offset. The hub's extra token is in a fixed part of its template or accounting (cause UNVERIFIED). Everything else matches token-for-token in count. |
| alternative hypothesis: drop history reasoning | wrong. Example: call 11 of the first DeepSWE session is 23,589 tokens without reasoning versus hub 29,742 = ours + 1 |
| incremental vs full tokenization | 956 full re-tokenizations, 0 mismatches |
| HF-revision template vs HSG `model_meta` template (Sep-1 snapshot, 10465 B) | rendered text identical on all 20,577 calls. The diff is a tool-result sort rewrite plus `content is not none`, and neither triggers on OpenCode payloads. |
| `--renderer builtin` (jinja2 + tokenizers, no transformers) vs transformers | identical token ids on the 80-call test set |
| client reconstruction (`replay_load.py --dry-run`) | 20,577 / 20,577 prompts rebuild to the build-time sha1 |

## Ready replay sets

| | DeepSWE (`deepswe_glm_64`) | synthetic tasks (`synth_v2_glm_64`) |
|---|---|---|
| sessions / tasks / pass-fail | 64 / 64 / 47-17 | 64 / 50 / 50-14 |
| calls; calls per session p10/p50/p90/max | 9,006; 71 / 129 / 219 / 305 | 11,571; 78 / 151 / 324 / 536 |
| prompt tokens per call p10/p50/p90/max (mean) | 33.8k / 133k / 213k / 249k (127k) | 63.2k / 165k / 288k / **542k** (173k) |
| new tokens vs previous call p10/p50/p90/p99 (mean) | 121 / 567 / 3.8k / 18.6k (2.05k) | 120 / 506 / 2.8k / 14.1k (1.45k) |
| output tokens per call p10/p50/p90/p99/max (mean) | 35 / 185 / 1,777 / 13.3k / 35.9k (882) | 34 / 177 / 1,447 / 11.2k / 54.2k (771) |
| reasoning share of output tokens; calls with ≥ 1k output | 72%; 16.1% | 69%; 14.6% |
| think gap s p10/p50/p90/p99/max (mean) | 0.24 / 0.39 / 3.1 / 74 / 1,201 (3.4) | 0.29 / 0.47 / 4.4 / 134 / 1,801 (6.9) |
| think share of recorded session wall | 14.6% | 43.8% |
| final context per session p50/max | 194k / 249k | 239k / 543k |
| recorded session wall s p50/p90 | 2,630 / 6,159 | 1,889 / 5,296 |
| hub call latency s p50/p90/p99 | 5.3 / 25.2 / 150 | 3.5 / 11.7 / 72 |
| hub output tok/s per call (output ≥ 200; lower bound, includes prefill and queue) p50 | 72 | 107 |
| finish reason | tool_calls 8,913, stop 93 | tool_calls 11,506, stop 65 |
| non-prefix calls (compaction, subagent prompts, date flip) | 60 (29 sessions) | 12 (8 sessions) |
| offline prefix-hit fraction: recorded / splice mode (sticky, infinite cache, 64-token blocks) | 0.9837 / 0.9904 | 0.9914 / 0.9958 |
| → uncached prefill tokens per call, recorded / splice | ~2.1k / ~1.2k | ~1.5k / ~0.73k |
| hub-reported cached fraction | 0.9794 | 0.9903 |

Calls that fit a given `--max-context`, with `--over-context end`, i.e. each session stops at its first call that
does not fit:

| max context | DeepSWE: calls kept / sessions untruncated | synthetic: calls kept / sessions untruncated |
|---|---|---|
| 131072 | 39.5% / 6 of 64 | 32.2% / 0 of 64 |
| 196608 | 72.9% / 33 | 64.3% / 24 |
| **262144** | **100% / 64** | 85.7% / 43 |
| 393216 | 100% / 64 | 98.1% / 63 |

The synthetic sessions ran without compaction, and 14% of their calls exceed 256k. For a 256k-context
deployment, **use the DeepSWE set as the primary profile**. Use the synthetic set with `--max-context 262144`
(sessions end at the limit) or on a server with a larger max length.

## Replay semantics (`replay_load.py`)

**Load model.**
- `--concurrency K` session slots. Each slot runs one session's calls strictly in order and then takes the next
  session; sessions are a seeded shuffle of `--sessions N`.
- `--loop` recycles the list with a new cache-salt epoch, so `--duration` runs stay saturated.
- `--ramp S` staggers slot starts; `--warmup S` excludes early calls from the summary.
- After call k the session sleeps `min(think_after_s × --think-scale, --think-cap)`. Defaults: scale 1, cap 600 s.
- At the `--duration` deadline no new call starts and think sleeps end. In-flight calls get `--drain-timeout`
  (900 s) to finish, then they are cancelled.

**Per call** it POSTs `/v1/completions` with:
- `prompt=<token ids>`, `max_tokens = min_tokens = recorded completion_tokens`, `ignore_eos: true`,
  `temperature 1.0`, `top_p 0.95` (as recorded).
- `stream: true` with `stream_options {include_usage, continuous_usage_stats}`, which gives exact cumulative
  token counts per SSE chunk, so MTP bursts are counted correctly.
- `cache_salt`; `X-Request-Id` / `X-Replay-Session` / `X-Replay-Call` headers.
- Optional: `--session-header NAME` and `--user-field` for router affinity, `--extra-body JSON`, `--header`.

`ignore_eos` + `min_tokens` make vLLM drop the generation-config EOS ids (154820/154827/154829) as stop tokens
(`sampling_params.py:664-692`), so the server decodes exactly the recorded length.

**`--prefix-mode`** controls how the next prompt treats the turn the server just generated:
- `splice` (default): requests `return_token_ids`. It overwrites the first `min(splice_len+1, generated)` tokens of
  the next delta, i.e. the re-rendered assistant turn plus the `<|observation|>` stop/role token, with the ids the
  server actually generated. Lengths are unchanged. The server's prefix cache can then reuse the generated KV, as it
  would in production when the template re-renders the model's own output token-identically.
- `splice-exact`: splices only where the recording shows count identity (`output_len == splice_len + 1`).
- `recorded`: sends the recorded prompts byte-exactly. The assistant turn is always re-prefilled; this is the lower
  bound, and also what a P/D prefill tier that never sees decode KV experiences.
- Evidence: the recorded `completion_tokens(k−1) == rendered span(k) + 1` exactly on 83.5% / 84.3% of extension
  calls (DeepSWE / synthetic) and within ±2 on 99.9%. Most mismatches are −1. Token identity itself cannot be
  checked because the hub returned text, not ids, so production reuse on an aggregated vLLM lies between
  `splice-exact`/`splice` and `recorded`.
- Cost of `splice`: vLLM echoes the full `prompt_token_ids` in the first streamed chunk when `return_token_ids` is
  set (`completion/serving.py:369`). That is about 1 MB of JSON per 130k-token call; the client parses it.
- The hub itself cached only previous **prompts**: on 4,931 of 4,951 decidable DeepSWE extension calls the hub's
  `cached_tokens` stops at the previous prompt, never covering its output. This is consistent with a P/D or
  similarly routed hosted deployment (UNVERIFIED).

**`--cache-salt`**:
- `run` (default): one salt per run and epoch. Sessions in the run still share the OpenCode system-prompt and tools
  prefix (~5-7k tokens), as in production, but never hit KV left by an earlier run.
- `session`: no cross-session sharing.
- `none`: plain vLLM. It will hit an earlier run's cache, so restart the server between runs if you use it.

**`--max-context N`**: clips `max_tokens` to `N − prompt_len`. A prompt that alone does not fit either ends the
session (`--over-context end`, default) or is skipped (`skip`); the client state still advances, so later prompts
rebuild correctly. Other knobs: `--max-output` caps output, `--max-calls-per-session`. Failures before any HTTP
response are retried `--connect-retries` (3) times; other errors end the session (`--on-error end`) or continue.

**Outputs** in `--out`:
- `calls.jsonl`, one line per call:
  - identity and sizes: `session_id/source/epoch/call_idx`, `prompt_len`, `shared_prefix_len`, `new_tokens`,
    `spliced`, `output_target/output_got`.
  - timing: `t_send`, `ttft_s`, `e2e_s`, `decode_s`, `decode_tok_s` = (n−1)/(t_last−t_first),
    `chunk_gap_ms_p50/p90/max`, `n_chunks`, `think_before_s`.
  - usage and status: `usage_*`, `cached_tokens` (if the server reports it), `finish_reason`, `status`, `error`.
  - recorded reference: `rec_output_len`, `rec_hub_latency_s`, `rec_hub_cached`.
- `sessions.jsonl`, per session: `end` (completed/deadline/context/error), `wall_s`, `model_s`, `think_s`,
  `recorded_wall_s`, `recorded_hub_latency_s`, `recorded_think_s`.
- `summary.json`, also printed:
  - throughput: calls/s, generated tok/s, prompt tok/s, new-prompt tok/s, mean in-flight. Rates use token
    attribution inside [warmup, deadline]: prompt tokens at the first token, generated tokens spread over
    [first, last].
  - latency: TTFT p10/p50/p90/p99, e2e per call, per-request decode tok/s p10/p50/p90 (also for output ≥ 256),
    TPOT ms. `chunk_gap_ms` and per-token ITL p50/p90/p99/p999 come from a log histogram.
  - prompt and output length distributions; session completion wall times; replay/recorded wall and model-time
    ratios.
  - errors by kind; `server_metrics_window` when `--metrics-url` is set.
- `metrics.tsv` (with `--metrics-url`); `client_args.json`.

## Server metrics (`scrape_metrics.py`)

One TSV row per poll per target (each URL, plus `all` when there are several URLs). Gauges are summed or averaged
over the `engine` labels; counters are stored cumulatively (`c_*`) and as per-interval rates. Metric names were
checked against vLLM 0.29 `v1/metrics/loggers.py` and `v1/spec_decode/metrics.py`.

| group | columns |
|---|---|
| load | `running`, `waiting` (+`waiting_capacity/deferred`), `kv_usage_mean/max`, `n_engines` |
| tokens | `prompt_tok_s`, `gen_tok_s`, `gen_tok_s_per_running` |
| cache | `cached_prompt_frac`, `prefix_hit_rate`, `ext_prefix_hit_rate` |
| health | `preemptions`, `finished_req_s` |
| latency | interval means of the TTFT / ITL / TPOT / e2e / queue / prefill / decode histograms |
| MTP | `spec_accept_rate` (accepted/draft tokens), `spec_mean_accept_len` (1 + accepted/drafts), `spec_pos_accept` (per position) |

`window` reports, per target:
- gauge means and maxima;
- exact rates from the counter deltas between the first and last row of the window;
- `gen_tok_s_per_running_sample_weighted`, the rp_window.py-style per-request decode speed.

## Offline test

Last run: 2026-11-06, `RESULT: PASS (0 failures)`.

```bash
python3 tests/offline_test.py            # builds a 4-session/15-call set with .venv, then stub + client on python3.9
python3 tests/offline_test.py --replay data/deepswe_glm_64.jsonl.gz --keep   # same checks on a ready set (PASS)
```

It uses a private `tempfile.mkdtemp` scratch dir (removed afterwards) and never runs test discovery. Checks
(all PASS):
1. Dry-run sha1s.
2. Recorded mode:
   - stub-received prompts equal the recorded ones (sha1 and length) on all 60 calls;
   - max_tokens == min_tokens == recorded output (capped) == tokens streamed; ignore_eos, temperature, top_p;
   - stub cache hit ≥ the previous prompt's blocks;
   - generated_tok_s × window = 9,893 = generated tokens; prompt_tok_s × window = prompt tokens (±0.01%);
   - per-request decode p50 297.1 tok/s for a stub rate of 300;
   - TTFT ≥ stub base;
   - client `server_cached_prompt_frac` == stub 0.8675;
   - `/metrics` window prefix hit 0.871 and gen 487 vs client 492 tok/s.
3. Splice and splice-exact:
   - prompts equal the recorded prompts with the stub's generated ids spliced in (sha1);
   - client splice lengths equal independently recomputed ones;
   - cache hits cover the spliced span;
   - first-chunk prompt echo parsed.
4. `--duration` deadline: no call sent after it; sessions end `deadline`.
5. `--max-context`: no request over the limit; sessions end `context`.
6. Stub `--max-model-len` → HTTP 400 recorded; sessions end `error`.

Load test against the stub, using the full DeepSWE set:
- Setup: 64 sessions, 40 calls each, `--duration 60`, splice mode.
- Result: 1,725 calls, 0 errors, 27 calls/s, 1.43M prompt tok/s, 4.1k generated tok/s, 51 mean in-flight.
- Client cost: 33% of one core, 310 MB RSS.
- A first attempt hit the stub's default listen backlog of 5 (38 ConnectionResetError). That was fixed in the stub
  (backlog 4096) and the client gained `--connect-retries`.

## Findings worth knowing when serving GLM-5.3 for this traffic

- **Decode work is small per call; context is huge.** Mean output is 0.77-0.88k tokens per call, but the mean
  prompt is 127k-173k tokens. Uncached prefill is only ~0.7-2.1k tokens per call with sticky routing. Expect decode
  over a long context, plus KV capacity, to dominate: tens of concurrent sessions × 130-240k tokens each. This is
  an expectation, not yet measured on H100. Prefix hit rate depends on routing stickiness. Hub cache misses (`less_than_prev_prompt`) were 246 of 9,006 DeepSWE calls.
- **MTP on the hub** (inferred from usage, UNVERIFIED config):
  - accepted+rejected draft tokens per decode step ≈ 3.07 (DeepSWE) / 2.85 (synthetic) → ~3 speculative tokens;
  - acceptance 50.9% / 53.3%;
  - ~2.55 output tokens per decode step on this OpenCode traffic.
- **Prefix-cache killers present in real sessions:**
  - OpenCode compaction (summary call with a different system prompt, then a fresh short prompt; shared prefix only
    10 tokens);
  - subagent or alternate system prompts (5,023 shared tokens);
  - the `Today's date` line in OpenCode's `<env>` block. A session running across midnight re-prefills its whole
    prompt; seen at 241k tokens.
  - The replay reproduces all of these at their recorded positions.
- **Interleaved thinking is preserved by vLLM 0.29.** It renames `reasoning_content`→`reasoning` and renders it as
  `<think>…</think>` in every past turn. So the dfw server will see the same history as the hub did. Reasoning is
  69-72% of all output tokens, and all of it stays in the context: at call 11 of the first DeepSWE session it is
  6.2k of 29.7k prompt tokens. Prompts without history reasoning would be much shorter, but that is not what vLLM
  does.

## Limitations and deviations

- **Bypasses the chat frontend.** The replay uses `/v1/completions` with token ids, so it skips server-side
  chat-template rendering and the reasoning/tool parsers. These are CPU costs on the API server (the HSG notes found
  the Rust frontend +14%). To measure them, an extra `--api chat` mode would be needed; it is not implemented.
  Messages are not stored in the replay file; they can be re-derived from `attempt_dir/http.jsonl`.
- **Output content is not the recorded content.** The server samples real GLM-5.3 continuations, forced to the
  recorded length (with `ignore_eos`, past natural EOS when shorter). MTP acceptance is therefore realistic in
  distribution but not identical per call.
- **Think gaps are recorded wall time.** They include OpenCode overhead and the sandbox speed of that day. Hub
  rate-limit waits are excluded. Tool time does not adapt to server speed.
- **The splice mode is an upper bound on generated-KV reuse; recorded mode is the lower bound** (see above). The
  splice applies only when a call extends the immediately previous call. Splicing into calls that return to an older
  branch, e.g. after a subagent prompt, is not modeled; 72 such non-prefix calls exist across both sets.
- **The +1 hub prompt-token offset** is constant and its cause is UNVERIFIED (hub template or accounting). The dfw
  vLLM with the HF-revision template renders our length.
- **Never run against a live GLM-5.3 server here.** No GPU job was used. The vLLM request and stream fields used
  were verified by reading the vLLM 0.29 source in the RL env: `ignore_eos`/`min_tokens` at
  `completion/protocol.py:80-81`, `return_token_ids` at :167, `cache_salt` at :198, `continuous_usage_stats` at
  `generate/base/protocol.py:221`, `X-Request-Id` at `serve/engine/serving.py:117`. Behavior on newer vLLM
  versions (0.30-0.33) is UNVERIFIED.
- **The synthetic-task set exceeds 256k context on 14% of calls** (see the context table).
