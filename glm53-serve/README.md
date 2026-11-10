# GLM-5.3 on dfw H100 (2 nodes x 8 H100 80GB)

Self-hosted `zai-org/GLM-5.3` (FP8, 756 GB, GlmMoeDsaForCausalLM: MLA + DeepSeek sparse attention + IndexCache, 256
routed experts, 1 MTP layer) with vLLM 0.33, ported from jkyi's HSG GB200 launchers
(`hsg-ref/serve_scripts/serve_glm_53*.{slurm,sh}`, tuning record `hsg-ref/notes/NOTES.md`) and re-tuned for H100.
Full experiment log: `notes/EXPERIMENTS.md`.

## Quick start

```bash
cd /lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve
sbatch --partition=interactive scripts/serve_glm53_dfw.sbatch      # debug / tests (interactive: 2 nodes per user)
bash scripts/submit_serve.sh glm53-a                                 # batch, dual-account babysit pair (helper rule)
TP=16 PP=1 VENV=$PWD/env/vllm-0.33.0 sbatch scripts/serve_glm53_dfw.sbatch   # stock-vLLM fallback layout
cat logs/serve/<jobid>.url        # http://<head-ip>:8000/v1   (written at job start, removed at exit)
cat logs/serve/<jobid>.status     # starting / ready <url> <KV size> / failed / unhealthy / exited
curl -s $(cat logs/serve/<jobid>.url)/models                         # served model name: glm-5.3
```

Ready 7-10 min after the job starts (container ~1.5 min, weight copy to node-local `/raid/scratch/<user>-glm53`
~1.5 min, vLLM 4-7 min from the warm JIT cache in `cache-0.33`); the weight copy is deleted when the job ends
(`STAGE_KEEP=1` keeps it, `STAGE_LOCAL=0` loads from Lustre, ~6 min). A new vLLM version or a new parallel layout
builds JIT caches once (~20 min).
Clients: OpenAI-compatible chat/completions; reasoning comes back in `message.reasoning` (the hosted NVIDIA endpoint
uses `reasoning_content`; read both). `reasoning_effort` per request = low | high | (anything else / unset = max).
Recommended sampling: temperature 1.0, top_p 0.95. No API key.

## Layouts

| | default: TP8 x PP2 + DCP8 + MTP3 | fallback: TP16 + DCP8 + MTP3 |
|---|---|---|
| vLLM | 0.33.0 + backport of PR #60477 + MTP/structured-output fix (`env/vllm-0.33.0-glm53`; `env/vllm-0.33.0-pr60477` = #60477 only) | stock 0.33.0 (`env/vllm-0.33.0`) |
| KV cache | 5.94M tokens | 2.89M tokens |
| replay, 16 agent sessions | 983 generated tok/s, 59.9 tok/s per request | 823 tok/s, 47.8 tok/s per request |
| replay, 32 agent sessions | 1,087 tok/s, 32.0 tok/s per request | 626 tok/s, 26.1 tok/s per request |
| teacher-forced NLL of hosted-GLM tokens | 0.514 (unpatched: 0.789 = broken) | 0.514 |
| greedy decode vs prefill agreement | 96.3-96.7% (unpatched: 74-91%) | 96.6% |
| swe-task E2E, 16 tasks / 16 concurrent OpenCode sessions | **14/16 passed**, 849 gen tok/s, 33.8 min per safe14 session | 12/16 passed, 641 gen tok/s, 46.0 min |

Hosted GLM-5.3 on the same 16 tasks: 70/80 (87.5%). Neither local result differs significantly from it (12/16 vs
70/80: Fisher one-sided p = 0.18).

**Do not run TP x PP on an unpatched vLLM <= 0.33.0** (`serve_glm53_dfw.sbatch` refuses). GLM-5.3's MoE output is a
TP-partial sum at the last layer of a pipeline stage and the PP send path re-gathers it as if it were replicated
(fixed upstream by PR #60477, main only). The model stays fluent but degrades badly: in real OpenCode sessions it
looped ("Hmm - wait, let me re-read the file" x1000) until the 131k output cap and passed 0 of 11 safe14 tasks.
Pipeline stages must also start on an indexer-owning layer (0,1,2,6,10,...,74; vLLM #63170, fix #59961 on main),
hence `VLLM_PP_LAYER_PARTITION=42,36`.

## Settings and why (all overridable via environment; see the sbatch header)

| setting | value | evidence |
|---|---|---|
| nodes | 2 x 8 H100 | 704 GiB of weights do not fit one node (TP8 = 88 GiB/GPU); DP16xEP16 needs 64.5 GiB/GPU of weights |
| parallelism | TP8 inside a node x PP2 across nodes, `VLLM_PP_LAYER_PARTITION=42,36` | TP16 pays a cross-node all-reduce in every layer: 16-42% fewer tok/s, half the KV. PP needs the PR #60477 backport (above) and stages that start on indexer layers |
| decode context parallel | DCP8 (`--dcp-comm-backend ag_rs`) | shards MLA KV inside a node instead of replicating it: PP2 0.76M -> 5.94M tokens (TP16: 0.36M -> 2.89M). Hosted-GLM OpenCode sessions keep ~125k live tokens each. DCP16 hung at model load |
| speculative decoding | MTP 3 | ~2.2x single-stream decode; acceptance length 2.4-3.3 on agent traffic |
| `VLLM_ENFORCE_STRICT_TOOL_CALLING=0` | set in `serve_glm53_rank.sh` | no grammar on tool calls, as the hosted GLM (tool calls 32/32 parse anyway). With `=1` the glm47 parser puts an xgrammar structural tag on every tool-enabled request; on stock 0.33 / `vllm-0.33.0-pr60477` that zeroes MTP under PP (acceptance length 1.1); `vllm-0.33.0-glm53` fixes it (3.1-3.3, ~8% fewer tok/s than `=0`) |
| MTP + structured outputs under PP | fixed in `vllm-0.33.0-glm53` | every grammar-constrained request (strict tool calls, `response_format` json_schema) lost all MTP drafts under PP: drafts were validated against -1 placeholders (async scheduling + PP decode microbatching; the draft handler kept only the latest micro-batch). Fix = backport of vLLM #63219 (per-request draft retention) WITHOUT its revert of the #54442 fail-closed guard (vllm issue #65430) + the engine fills retained drafts before building the grammar bitmask. json_schema c16: acceptance 1.4 -> 3.3, 64/64 valid, 3x faster; NLL 0.5148. `notes/divergence/upstream/mtp-so-pp-fix/` |
| startup self-healing | `serve_glm53_rank.sh` | vLLM 0.33 sometimes hangs in CUDA-graph capture ("direct DCP q-gather multimem timeout", 2 of ~12 starts); rank 0 detects it (or no readiness after `READY_TIMEOUT_S`=1500) and restarts vLLM on both nodes inside the job (`MAX_STARTS`=3), keeping the allocation |
| stall watchdog | `serve_glm53_rank.sh` | requests running but no token progress for `STALL_S`=600 s -> the job fails (status `unhealthy (stalled ...)`), so clients/routers fail over; `/health` alone does not see a hung engine |
| KV dtype | `fp8` (= fp8_ds_mla) | FLASHMLA_SPARSE on SM90 (DCP-capable); bf16 KV (FA3 sparse) halves capacity and has no DCP |
| `--max-num-batched-tokens` | 4096 | +13% tok/s and +19% per-request decode vs 8192 at 32 sessions; keep <= 8192 and util <= 0.90 (vLLM #59305) |
| `--max-model-len` | 262144 | OpenCode compacts at 242,144; vLLM >= 0.32 clamps max_tokens 131072 to the remaining context (<= 0.31 returned 400) |
| MoE backend | Triton (auto) | DeepGEMM MoE was 20-24% slower |
| weights | `/lustre/fsw/portfolios/nemotron/users/hengyuf/models/GLM-5.3` (HF rev ba32f8c, sha256-verified), copied to `/raid/scratch/<user>-glm53/` at start and deleted at exit (`STAGE_LOCAL=1`, `STAGE_KEEP=0`) | Lustre load 365-391 s vs 20-30 s from local NVMe after a 72-100 s parallel copy |
| nodes excluded | 43 known-bad dfw nodes | `notes/dfw-ops/digest.md` section 4 |
| reaper comment | `inference_server`, 240 min | required for >= 4-node jobs |

## Correctness gates (run on every new layout / version / patch)

1. `tools/nll_check.py --base-url <url> --n 24`: teacher-forced NLL of 84.6k hosted-GLM generated tokens; must be ~0.514
   (`--compare runs/nll/*.json`). This is the cheapest detector of silent corruption.
2. `tools/greedy_parity.py --base-url <url> [--concurrent 6]`: greedy decode vs prefill re-score; expect >= 95%.
3. `hsg-ref/serve_scripts/bench_endpoint.py --url <url> --model glm-5.3 --scenario {tools,roundtrip,multiturn,longctx}`.
4. `notes/swe-task-harness/replay_hub_requests.py --attempt <hub attempt> --base-url <url> --send`: same prompt tokens
   as the hub for identical payloads (constant -1), tool calls parse, the 242k-token call with max_tokens 131072 works.
5. MTP acceptance under chat + tools traffic: `grep "SpecDecoding metrics" logs/serve/<job>_vllm_rank0.log`
   (2-3.5, never stuck at 1.00). The token-level replay cannot see tool-related problems (no tools in /v1/completions).
6. Real agent traffic: `e2e/reasoning_stats.py` on the campaign journals; hosted-GLM reference: no reply at the
   131072 cap, >32k replies 0.15-0.34%, 'hmm' per 1k words p50 ~8.

## swe-task end-to-end test (`e2e/`)

`e2e/submit.sh <pilot|safe14|group> <base-url> [repeats] [concurrency] [group]` (env `E2E_ROOT`, `E2E_TAG`) runs the
synth probe harness (ATP + OpenCode + task graders) with GLM-5.3 served here as the agent (in-memory model spec
`glmlocal`, no hub key, no change to swe-task files). `e2e/summarize.py <root> [--live]` compares with the hosted-GLM
records of the same tasks (safe14 hub baseline 64/70 = 91.4%; 50 tasks 76.7%).

Results (safe14 + pilot = 16 tasks x 1, 16 concurrent sessions; `e2e/root3`, `e2e/root4`):

| server | passed | replies | >32k replies | at 131k cap | loopy | non-200 | MTP acceptance |
|---|---|---|---|---|---|---|---|
| TP8xPP2 patched (default) | 14/16 (safe14 12/14) | 1,694 | 0.35% | 0 | 0 | 0 | 2.55-2.99 |
| TP16 stock (fallback) | 12/16 (safe14 11/14) | 1,625 | 0.43% | 0 | 0 | 0 | 2.9-3.45 |
| TP8xPP2 unpatched (broken) | safe14 0/11 graded | 725 | 4.1% | 18 | 52 | - | 2.8-2.9 |
| hosted GLM-5.3 (5 runs per task) | 70/80 (safe14 64/70) | | 0.15-0.34% | 0 | | | |

Every failure on the two correct servers was a near-miss: 96-99% of the fail-to-pass tests passed (argo-cd on the
fallback: all target tests passed, 1 of 156 regression tests broken). The 2 default-layout failures (dacite c01/c02;
hosted GLM 4/5 and 3/5 on them) also failed on the fallback.

## Layout of this directory

- `scripts/serve_glm53_dfw.sbatch` + `scripts/serve_glm53_rank.sh` (production), `scripts/submit_serve.sh` (babysit),
  `scripts/dev_*.sh` (debug holds), `scripts/cleanup_staged.sh` (delete weight copies left on nodes by runs before
  2026-11-07; node list in its header)
- `env/` vLLM venvs (`build_vllm_env.sh <version>`, `build_patched_env.sh`), `cache-0.33/` warm JIT caches
- `tools/` quality gates (`nll_check.py`, `greedy_parity.py`, `decode_consistency.py`), `replay/` load generator,
  `decode_bench.py`
- `notes/` digests (HSG launcher, vLLM SM90, public recipes, swe-task harness, dfw ops, divergence root cause) and
  `EXPERIMENTS.md`; `hsg-ref/` read-only copies of the HSG material; `runs/`, `logs/`, `e2e/`
