# GLM-5.3 on dfw H100: experiment log (main agent)

Hardware: 2x 8x H100 80GB (pool0-01560 head, pool0-01912), interactive hold 21414711. Container
`nemo-rl-cuda13.2-base-20260903-x86_64.sqsh` (CUDA 13.2 compat on driver 535). Weights: HF zai-org/GLM-5.3 @ ba32f8c
(`/lustre/fsw/portfolios/nemotron/users/hengyuf/models/GLM-5.3`, sha256-verified; staged to node-local
`/raid/scratch/hengyuf-glm53/GLM-5.3` for 31 s loads vs 391 s from Lustre).

Launchers: `scripts/dev_serve.sh` (one node rank, inside the container), `scripts/dev_launch2.sh` (2-node mp/dp).
Bench: `tools/decode_bench.py` (fixed-shape, unique prompts = worst-case prefill), `hsg-ref/serve_scripts/bench_endpoint.py`
(correctness), `tools/replay/replay_load.py` (closed-loop replay of real hub GLM-5.3 OpenCode sessions).

## Runs

| id | vLLM | layout | KV tokens | result |
|---|---|---|---|---|
| t01 | 0.32.0 | 1 node TP8, dummy weights, 8 layers | 10.3M (8 layers) | pipeline works on SM90: FLASHMLA_SPARSE + fp8_ds_mla, block 64, DSA indexer via vendored DeepGEMM, TRITON fp8 MoE; first start ~20 min of JIT (FlashInfer + DeepGEMM), cached afterwards |
| t02 | 0.29.0 (RL env, ro) | TP8xPP2 (39/39), no MTP, fp8 | 781,696 | correct (17*23, tools 16/16, roundtrip 16/16); plain 1k-token decode: c1 83 tok/s, c8 348, c32 891, c64 1,050 agg |
| t03 | 0.29.0 | TP8xPP2 + MTP3 | - | FAILS: "Pipeline parallelism is not supported for this model" (0.29's DeepseekV32MTP lacks SupportsPP) |
| t04 | 0.32.0 | TP8xPP2 (41/37) + MTP3, fp8, u 0.90, seqs 64 | 758,080 | correct (tools/roundtrip 16/16, needle 106k 4/4, 252k 2/2, multiturn 32/32); plain decode c1 187, c8 688, c64 2,481 tok/s agg (~2.2x t02); unique 35k prompts: c1 137 tok/s/req, c8 27, c16 22, c32 9.7 (prefill-bound, TTFT 39-72 s) |
| t05 | 0.32.0 | TP8xPP2 (41/37) + DCP8 (ag_rs) + MTP3 | **6,051,840** | DCP works on SM90 (8x KV). Correct (tools/roundtrip 16/16, needle 106k 8/8, multiturn 32/32). Prefill ~9.6k tok/s (vs ~13.5k); unique 35k prompts: c1 93 tok/s/req, c8 21.6, c16 14.3 |

## Correctness hazard found by the external-recipes agent (vllm #63170, fix #59961 not in any release <= 0.33.0)

Under PP, a stage that starts on an IndexCache "shared" layer reads a stale top-k buffer (silent quality loss, fluent
output). Indexer-owning layers are 0,1,2,6,10,...,74 (layer % 4 == 2, plus 0,1). t04/t05 used 41/37 (stage 2 starts
at 41 = shared) and the default 39/39 also starts on a shared layer. **Use `VLLM_PP_LAYER_PARTITION=42,36`** (stage 2
starts at 42; stage 1 holds 3 dense + 39 MoE layers, stage 2 holds 36 + MTP + lm_head, which balances weights).

## Runs, continued (2026-11-06 18:10-19:50)

| id | vLLM | change vs t05 | KV tokens | replay (64 DeepSWE GLM sessions, closed loop, 600 s) |
|---|---|---|---|---|
| t05 | 0.32.0 | PP 41/37, DCP8, MTP3, chunk 8192 | 6.05M | c16: 1,051 gen tok/s, decode p50 59.7 tok/s/req, TTFT p50 0.42 s, hit 97.2%, KV max 25% |
| t06 | 0.32.0 | PP **42/36** (correct) | 5.92M | c32: 1,123 gen tok/s, decode p50 25.8, TTFT p50 0.83 s, ITL gap p99 1.30 s |
| t07 | 0.32.0 | 42/36 + **chunk 4096** | 5.92M | c32: **1,264** gen tok/s (+13%), decode p50 30.6 (+19%), TTFT 0.69, gap p99 0.84 s -> new default |
| t08 | 0.32.0 | t07 + `--moe-backend deep_gemm` | 5.91M | c32: 1,018 gen tok/s, decode p50 21.1 -> Triton (default) is better |
| t09 | 0.32.0 | TP16 + DCP16 (no PP) | - | stopped during first-time JIT warmup to free the hold for t10 |
| t10 | **0.33.0** | t07 config | 5.94M | first start 20 min (JIT); gates 16/16 + 16/16 |

## MTP acceptance collapses on tool-enabled chat requests (found in the E2E, 19:05)

- Production server (0.32, t07 config) ran the swe-task E2E: MTP mean acceptance length fell from 2.5-3.3 to exactly
  1.00 (0 of ~10k drafted tokens accepted per 10 s) within a minute of the OpenCode sessions starting, and stayed there.
  Generation fell to ~20 tok/s per request at 16 in flight. Outputs stayed coherent (spec decode is verified).
- Reproduced on 0.33.0 (hold B): bench_endpoint `plain` (no tools) at c8 -> acceptance 2.6-2.9; `tools` at c8 ->
  1.00-1.01. The token-level replay never showed it because it sends /v1/completions without tools.
- Cause: the glm47 tool parser constrains every request that has tools with an xgrammar structural tag
  (`default_tool_strict_level = FUNCTION`, vllm/tool_parsers/glm47_moe_tool_parser.py); grammar-constrained decoding +
  MTP drafts -> drafts always rejected. Fix used: `VLLM_ENFORCE_STRICT_TOOL_CALLING=0` (disables the structural tags,
  as the hub does not constrain either), now the default in scripts/serve_glm53_rank.sh.
- Separately, 0.33.0 carries the MRV2 PP + spec-decode broadcast fixes (#62129, #64458; a race in
  pp_utils.broadcast_drafts in 0.32), so the production default moved to 0.33.0 (cache root cache-0.33).

## Decode-path quality problem under PP (found 20:20-21:05)

- E2E round 2 (0.33, PP2 42/36, DCP8, MTP3, fp8, tool tags off) ran at full speed (MTP len 2.8-2.9, ~850 gen tok/s at
  14 sessions) but GLM fell into degenerate reasoning loops ("Alright - I'm going to re-read the file" x1133, "Hmm -
  wait" x957) until the 131,072-token output cap: 7/407 replies finish=length, 3.9% of replies > 32k tokens; 8 of 14
  safe14 slots ended within 8-15 calls after ~131k generated tokens. Hosted GLM-5.3: 0 length finishes in 82k replies.
  "Hmm" per 1k reasoning words p50/p90: hosted 8.3/14.7, ours 19.9/189 (e2e/reasoning_stats.py).
- Teacher-forced NLL of hosted-GLM generations (prefill path, tools/nll_check.py, 24 calls / 84.6k tokens) is identical
  with and without DCP/MTP (0.7888 vs 0.7891) -> prefill path fine.
- Decode vs prefill parity (tools/greedy_parity.py: greedy 256 tokens, re-score in one prefill):
  | config | greedy argmax agreement | mean abs logprob diff (greedy) |
  |---|---|---|
  | B: TP8xPP2 42/36, fp8 KV, no DCP, no MTP (0.33) | 94% on 1.3k prompt; |d| 0.11 |  |
  | R3: TP8xPP2 42/36, bf16 KV (FA3 sparse), no DCP/MTP | 86.2% (74-91% on 52k real prompts) | 0.251 |
  | R1: TP16 (no PP), bf16 KV, no DCP/MTP | **95.9%** (94.5-97.3% on 52k real prompts) | **0.049** |
  -> pipeline parallelism (even with stage starts aligned to indexer layers) corrupts the decode path of GLM-5.3 on
  vLLM 0.33 (consistent with vllm #59961's GSM8K table: aligned PP2 0.882 vs fixed 0.908). Eager mode and KV dtype do
  not matter. Production must not use PP on released vLLM.

## Root cause: vLLM PR #60477 (found by the code-reading agents, confirmed 21:43)

- GLM-5.3's MoE is built with reduce_results=False; the last layer of a non-final PP stage returns a per-TP-rank
  PARTIAL sum, but the PP send path assumes TP-replicated tensors (each rank sends a 1/TP slice, the receiver
  all-gathers and then all-reduces again). Layer 41's output becomes a stitched mix of different ranks' partials x8,
  in a token-count-dependent pattern -> corrupted prefill AND decode under any TP>1 x PP>1 layout.
  Upstream fix: PR #60477 "[Bugfix] DSA: all-reduce partial hidden_states before PP send" (main 1aea05364,
  2026-10-29), in NO release <= 0.33.0. (The #63170/#59961 stale-top-k issue is separate; 42,36 avoids it.)
- Backport: env/vllm-0.33.0-pr60477 (fresh venv + notes/divergence/upstream/pr60477-nvidia-model.patch, built by
  env/build_patched_env.sh). serve_glm53_dfw.sbatch refuses PP>1 on an unpatched env.
- Teacher-forced NLL of 84,610 hosted-GLM generated tokens (24 calls, tools/nll_check.py):
  | config | NLL nats/token |
  |---|---|
  | unpatched TP8xPP2 (+/- DCP8/MTP3) | 0.789 |
  | patched TP8xPP2 + DCP8 + MTP3 | **0.514** |
  | TP16 + DCP8 + MTP3 (no PP, stock 0.33) | **0.514** |
- Greedy decode-vs-prefill parity: patched PP2 96.7% (also 96.7% with 6 concurrent generations = ragged MTP batches),
  TP16 96.6%, unpatched PP2 74-91%.
- Replay c16 (64 DeepSWE sessions): TP16+DCP8 823 gen tok/s, decode p50 47.8 tok/s/req; PP2 (unpatched, same speed as
  patched) 1,074 gen tok/s, decode p50 62.4.

## Correct layouts compared (replay of 64 DeepSWE sessions, 600 s windows; same seed per concurrency)

| layout (vLLM 0.33) | KV tokens | c16 gen tok/s | c16 decode p50 | c32 gen tok/s | c32 decode p50 | c32 TTFT p50 |
|---|---|---|---|---|---|---|
| TP16 + DCP8 + MTP3 (stock) | 2.89M | 823 | 47.8 | 626 | 26.1 | 0.63 s |
| TP8xPP2(42,36) + DCP8 + MTP3 (+ PR #60477 backport) | 5.94M | 983 | 59.9 | 1,087 | 32.0 | 0.48 s |

TP16 pays a cross-node all-reduce on every layer (and its prefill is slower: c32 TTFT p90 2.4 s vs 1.3 s), so the
patched PP2 layout is 1.2x (c16) to 1.7x (c32) faster with twice the KV, at identical model quality.

## E2E round 3 (TP16 + DCP8 + MTP3, stock 0.33; server 21427782; campaigns 21428665/6, e2e/root3)

Reply statistics after 1,171 replies match the hosted GLM's early-session distribution: >16k 1.37% (hub first-60
calls 1.34%), >32k 0.60% (hub 0.53%), max 66,542 (hub 54,239), never at the cap, 'hmm' per 1k words p50 9.5 (hub 8.3),
loopy replies 0 (round 2 with broken PP: p99 = 131,072, >32k 4.1%, loopy 52/725, safe14 0/11).

Final (22:49): safe14 **11/14 = 78.6%** (hub 64/70 = 91.4%), pilot 1/2 (aiohttp pass, argo-cd fail; hub 6/10), all 16
= 12/16 = 75.0% vs hub 70/80 = 87.5% -> not significant (Fisher one-sided p = 0.17-0.18; with Laplace-smoothed hub
per-task rates the expected safe14 score is 11.1, P(<= 11) = 0.57). The 3 safe14 misses are near-misses (f2p 0.957,
0.985, 0.976; p2p 1.0) on tasks the hub passes 5/5, 4/5, 3/5. 1,625 replies: p99 15,132, max 66,542, >32k 0.43%,
'hmm' p50/p90 8.7/16.5 (hub 8.3/14.7), loopy 0, 0 non-200, every session ended with a normal stop. Max prompt per
session 107k-187k (safe14), so no OpenCode compaction (only argo-cd reached 242k); hub ran with 1M context. MTP
acceptance length 2.9-3.45 throughout, prefix-cache hit 99.4%. 33-56 min per session at 16 concurrent sessions.

## E2E round 4 (TP8xPP2 + DCP8 + MTP3 on env/vllm-0.33.0-pr60477; server 21430755 batch_short; e2e/root4)

- Production sbatch, `TP=8 PP=2 VENV=env/vllm-0.33.0-pr60477 STAGE_LOCAL=1`: job start 22:45, staging 72-94 s,
  weights 21 s, ready 22:54:54 (9.7 min), KV 5,944,832 tokens.
- Gates on the production server: NLL 0.5138 (runs/nll/prod-pp2-pr60477-21430755.json), greedy parity 96.29% at 4
  concurrent (runs/parity/prod-pp2-21430755.json), roundtrip 8/8, tools 16/16 tool_calls.
- Campaigns: safe14 x1 (21432208/9, conc 14) + pilot x1 (21432210/1); watcher wf_d5368bcc-0ff.
- Final (00:18): safe14 **12/14**, pilot **2/2** (argo-cd passed; hub 1/5) -> 14/16 = 87.5% = hub 70/80. Failures:
  dacite c01 (f2p 0.970) and c02 (f2p 0.976), the same two that failed in round 3 (hub 4/5, 3/5). 1,694 replies: p99
  20,806, max 61,195, >32k 0.35%, 0 at the cap, 0 loopy, 0 non-200; MTP acceptance 2.55-2.99.
- Speed on the same 16 tasks: generation 849 tok/s p50 (831 mean) with >= 12 requests running vs 641 (587) on TP16 in
  round 3 -> 1.3-1.4x; mean safe14 session wall 33.8 min vs 46.0 min (sessions include tool/test execution time).

## Decision (2026-11-07 00:20): default = patched TP8xPP2 + DCP8 + MTP3

- Same quality as TP16 on every gate (NLL 0.514, greedy parity 96-97%, E2E 14/16 vs 12/16, hub 70/80), 2x KV,
  1.2-1.7x throughput. serve_glm53_dfw.sbatch defaults: TP=8, PP = 8 x nodes / TP (so `TP=16` alone gives PP1),
  VENV=env/vllm-0.33.0-pr60477, STAGE_LOCAL=1. Fallback: `TP=16 PP=1 VENV=env/vllm-0.33.0` (stock).
- serve_glm53_rank.sh now deletes the node-local weight copy (/raid/scratch/<user>-glm53/GLM-5.3, 704 GB) on exit
  (EXIT trap; SIGTERM handled promptly via `sleep & wait`; Slurm KillWait 120 s); STAGE_KEEP=1 keeps it.
  Previous versions: notes/serve_glm53_{dfw.sbatch,rank.sh}.prev-20261107.
- Validation of the new defaults (job 21436105, interactive, no env overrides): TP8xPP2 DCP8 MTP3 on the pr60477 env,
  staging 75-81 s, ready 6.7 min after job start, KV 5,944,832; NLL 0.5145, roundtrip 8/8, tools 32/32, MTP
  acceptance 3.11 under tool traffic. Cleanup on scancel proven with a 1-node mock (fake vllm + 1 MB fake model, job
  21436668): "removed staged weights" in the rank log in the same second as the cancel. The cleanup note now goes to
  the rank's engine log, because srun's stdout forwarding dies with the batch step on scancel.
- Leftover copies from earlier runs (before the cleanup existed), 704 GB each: /raid/scratch/hengyuf-glm53 on
  pool0-00786, 00791, 00835, 00837, 01560, 01654, 01664, 01731, 01758, 01891, 01912, 01916 and /raid/scratch/root-glm53
  on pool0-00082, 01472. scripts/cleanup_staged.sh queues pinned 1-GPU delete jobs (max 3 nodes per call): a first
  attempt with all nodes hit batch_short's 20-jobs-per-user submit limit (QOSMaxSubmitJobPerUserLimit) and would
  have blocked other batch_short submissions for hours, so it was cancelled before any job ran.
- Switch back to stock vLLM once a release contains PR #60477 (and #59961); re-run the gates (README) first.

## MTP + structured outputs under PP fixed (2026-11-07, env/vllm-0.33.0-glm53)

- Repro (dev hold 21455472, TP8xPP2 + DCP8 + MTP3, pr60477 env, `VLLM_ENFORCE_STRICT_TOOL_CALLING=1`): plain c8
  acceptance length 2.99 / 688 tok/s; tools c8 1.11 / 178 tok/s (32/32 tool calls); json_schema 1.36-1.40 (32/32 valid).
- Cause: async scheduling sends -1 placeholder drafts; with PP decode microbatching a structured-output request is not
  "pending" when rescheduled, so the grammar bitmask is built from placeholders, and the worker's DraftTokensHandler kept
  only the latest micro-batch's drafts; #54442's fail-closed guard then rejects every draft. Upstream: #63219 (main
  322f90a, 2026-10-30: per-request retention + revert of #54442), issue #65430 (open: the revert may accept
  grammar-invalid drafts).
- Fix: #63219's retention without the revert + engine fills retained drafts in the non-deferred branch
  (notes/divergence/upstream/mtp-so-pp-fix/, env/build_glm53_env.sh).
- Result (strict on): tools c8 3.13 / 340 tok/s, c16 3.15 / 533 tok/s 64/64; json_schema c8 3.25 13 s (was 42 s),
  c16 3.30 64/64 valid; no grammar-mismatch errors. Strict off on the same env: plain 3.00, tools c16 3.20 / 577 tok/s,
  json_schema c16 3.30 63/64 (1 reply kept its JSON inside the reasoning); NLL 0.5148. Default sbatch env switched to
  vllm-0.33.0-glm53 (the rollout fleet stays pinned to vllm-0.33.0-pr60477, strict off, as the pilots).
- TP16 + strict on stock 0.33 (to confirm the PP-only diagnosis) could not be measured: that start hit a CUDA
  "unspecified launch failure" during graph capture together with the multimem timeout below.

## Startup hang: "direct DCP q-gather multimem timeout" (2026-11-07)

- 2 of ~12 starts today hung in CUDA-graph capture (31/51) with this message (s5 TP16 stock on pool0-00570/00680 and the
  production PP2 job 21462196 on pool0-01226/01238); every successful start also uses the direct symmetric-memory DCP
  query gather (VLLM_USE_DIRECT_DCP_Q_GATHER auto). Not node-specific.
- serve_glm53_rank.sh now restarts vLLM inside the job on that message, on exit-before-ready, or after READY_TIMEOUT_S
  (1500 s), at most MAX_STARTS (3) starts (mock-tested: detection within 10 s, both ranks restart, ready on start 1),
  and fails the job when requests run without token progress for STALL_S (600 s).
