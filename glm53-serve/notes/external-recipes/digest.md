# GLM-5.3 on Hopper: external serving recipes, measured numbers, known issues

Prepared 2026-11-06 for the dfw 2x8 H100 GLM-5.3 build. The scope is every public recommendation for serving GLM-5.3, and for the architecture-identical GLM-5 / 5.1 / 5.2 and DeepSeek-V3.2 (DSA) models, on H100/H200/H800/H20. Raw copies of every source are in `dl.2cEBnI/` next to this file (Appendix A). Helper scripts are in `tools/`.

Conventions used below:
- "Verified" means the source itself claims it ran the configuration.
- "Synthesized" means a generator emitted the command without a hardware run.
- UNVERIFIED marks anything this digest could not confirm.
- "0.29 env" is the production RL vLLM (`swe-rl-v2-sc-migration/runtime/envs/vllm`, v0.29.0 g98dff2a81), read-only.

---

## 0. TL;DR

1. **No public source verifies GLM-5.3 FP8 on H100 in any layout.**
   - vLLM's GLM-5.3 recipe marks only b300, h200, mi325x, mi350x and dgx_station_gb300 as `verified` (plus an Ascend overlay). For H100, recipes.vllm.ai *synthesizes* a 2-node `--tensor-parallel-size 16` command.
   - The SGLang GLM-5.3 cookbook has no H100 cells. TokenSpeed's GLM-5.3 command uses a Blackwell-only MoE backend.
   - The closest Hopper evidence:
     - **arXiv 2607.02518:** GLM-5 FP8 on **2x8 H100**, SGLang 0.5.9. TP4xPP4 was best: 9,993 total tok/s at 30k-in/500-out. TP8xPP2 was worst at 6,906.
     - **vLLM #46074:** GLM-5.2-FP8 on **4x8 H100**, DP4xTP8+EP, fp8_ds_mla, MTP 5.
     - **SGLang DeepSeek-V3.2 on 16x H800:** TP16 + attn-DP2 + EAGLE. 86 tok/s single stream; 2,531-3,422 output tok/s at concurrency 1,024. PP2xTP8 variants are described as "verified on Hopper".
     - **SGLang GLM-5/5.1 generators:** H100 FP8 = `--tp 16`, `--mem-fraction-static 0.85`. This is a config table, not a benchmark.

2. **Minimum GPU count.**
   - vLLM: FP8 needs `vram_minimum_gb: 893`, i.e. "8xH200/H20 (141GB x 8) single-node".
   - On 8xH200 the recipe caps context at `--max-model-len 262144`, because only ~23 GiB/GPU is left for KV (~450K tokens).
   - SGLang: H100 FP8 needs tp=16 (BF16 needs tp=32).
   - The vLLM DeepSeek-V3.2-Exp blog says "16xH100, 8xH200, or 8xB200".
   - The only recipe route onto a **single** 8xH100 node is the community `gpustack/GLM-5.3-W4A8` (INT4 routed experts, 372 GiB). It is listed in the vLLM recipe for h100/h200 only.

3. **Pipeline parallelism silently corrupts GLM-5.x in every released vLLM (v0.29.0 through v0.33.0).**
   - Under PP, layers that reuse indexer top-k at the start of a stage read a stale per-GPU buffer (vllm #63170). On H200 PP8, NLL rose +16.1% while answers stayed fluent.
   - The fix, vllm PR #59961, was merged to main on 2026-10-31 (`63d3d39c`). git shows it is **not** in v0.33.0.
   - The 0.29 env has the bug: `model_executor/models/deepseek_v2.py:1549-1552` passes only `hidden_states` and `residual` across stages.
   - Workaround: start every stage on an indexer-owning layer, i.e. one of `{0,1,2,6,10,...,74}`.
     - PP2: `VLLM_PP_LAYER_PARTITION=38,40`. The PR author says this "restored accuracy", but the numbers are ambiguous (§2.7, §7).
     - PP4: `18,20,20,20`.

4. **Plain TP is a bad fit for Hopper sparse MLA.**
   - FlashMLA's sparse FP8 decode supports only h_q = 64 or 128, so vLLM pads the per-rank heads up to 64 (`flashmla_sparse.py:256,545-547` in the 0.29 env).
   - GLM-5.3 has 64 heads. TP8 gives 8 heads per rank (8x padding); TP16 gives 4 (16x padding).
   - vLLM's DeepSeek-V3.2 recipe says "Avoid `-tp 8` with FlashMLA-Sparse… Prefer TP=2 (Hopper)".
   - SGLang GLM-5-FP8 at TP16 on H100 crashed with `Unsupported h_q: 4` (sglang #21934). The suggested remedies were head padding and DP-attention.
   - MLA's latent KV is replicated on every TP rank, so TP16 also gives the smallest KV pool. PP and DCP are the only ways to shrink KV per GPU.

5. **KV dtype on Hopper.**
   - Use `--kv-cache-dtype fp8`, which is an alias of `fp8_ds_mla` at 656 B per token per layer. This is the vLLM recipe's `hopper` override.
   - `fp8_e4m3` is rejected by FLASHMLA_SPARSE on SM90 (vllm-project/recipes#565).
   - In the 0.29 env, `--kv-cache-dtype auto` (bf16) on SM90 selects **FLASH_ATTN_MLA_SPARSE** (FA3, no head padding, BF16 KV only, no DCP). `fp8` selects **FLASHMLA_SPARSE** (padding, DCP-capable). See `platforms/cuda.py:135-143`.
   - PR #46189 (GLM-5.1-FP8, TP8, Hopper) measured the FA3 sparse path at BS1 4096/4096: TPOT 28.0 → 16.3 ms. At BS64 512/512 it was flat (50.7 → 51.9 ms). The cost is half the KV capacity.

6. **FlashMLA SM90 workspace regression in vLLM 0.32.0/0.33.0** (vllm #59305, open).
   - With fp8_ds_mla mixed batches, each call allocates `(b+num_sm_parts)*s_q*h_q*d_v*4` bytes outside the profiled budget: 2,052 MiB at 8,192 tokens and 8,208 MiB at 32,768.
   - On H100, `--gpu-memory-utilization 0.925` crashed and 0.90 passed.
   - git shows the regression came in with #56893, which is in v0.32.0 and v0.33.0. It is absent from v0.29.0, which carries the earlier fix #53755.
   - For a 0.32.0 build: keep `--max-num-batched-tokens <= 8192` and `--gpu-memory-utilization <= 0.90` when using fp8 KV.

7. **Version matrix** (git-verified, §2.7):
   - v0.29.0 lacks the GLM-5.3 performance PRs (e.g. #57443, +13.3% at concurrency 1) and the GLM-5.3 reasoning-parser fix #56994. v0.32.0 has both.
   - v0.33.0 adds the PP+spec-decode fixes #62129 and #64458.
   - **No release** has #59961 (PP top-k) or #62424 (`--tokenizer-mode glm5_next`, native GLM-5.3 rendering in the Rust frontend). On these builds use the default tokenizer mode and the HF `chat_template.jinja`.

8. **MTP evidence.**
   - vLLM GLM-5.3 recipe: 5 draft tokens. GLM-5 and 5.1 recipes: 3.
   - llm-d on GLM-5.2/H200 uses 3; MTP "halves median TBT" and gives 1.6-2.0x aggregate output at concurrency ≤128.
   - SGLang uses EAGLE 5-1-6 for low latency and 1-1-2 for throughput.
   - gpustack on H20, MTP=1: 1.56x speedup at concurrency 1, falling to 1.24x at 64.
   - Random-token benchmarks under-report MTP. GLM-5.1 on H200 measured AL 1.64 with 3 drafts on random data.

9. **Chat template.**
   - The HSG copy is byte-identical to HF revision 7cda819 (Aug 27, the initial release). It renders a literal `None` for an assistant turn with `content=None` (verified by rendering, §8.2). HF fixed this in aca966e (Sep 4).
   - ba32f8c (Oct 14) only speeds up the tool-result reorder; the output is the same.
   - Template contract:
     - `reasoning_effort` ∈ {low, high}; anything else means max.
     - `clear_thinking` defaults to false, which keeps reasoning from all earlier turns.
     - The generation prompt ends in `<|assistant|><think>`.
     - Tools are rendered before the system message.
     - Calls use the format `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`.
     - EOS ids are 154820, 154827 and 154829 (`<|endoftext|>`, `<|user|>`, `<|observation|>`).

---

## 1. Source inventory

| # | Source (date / commit) | Hardware covered | H100 status | Local copy |
|---|---|---|---|---|
| S1 | vLLM recipe `vllm-project/vllm:recipes/models/zai-org/GLM-5.3.yaml` @ e740bad (main 52e8596 identical), date_updated 2026-10-19 | B300 (default), H200, MI325X, MI350X, DGX Station GB300 verified | not listed; site synthesizes TP16 | `dl.2cEBnI/vllm-project_vllm_main_recipes_models_zai-org_GLM-5.3.yaml` |
| S2 | recipes.vllm.ai JSON API (rendered from S1, lock e740bad) | all taxonomy GPUs | **synthesized** H100 commands (§2.2) | `dl.2cEBnI/api/*.json` |
| S3 | vLLM recipes GLM-5.2 / GLM-5.1 / GLM-5 / GLM-5.3-Flash / DeepSeek-V3.2 YAML (vllm main 52e8596) | H200, H20, B200/B300, AMD; DSv3.2 h100+h200 "verified" | DSv3.2 only | `dl.2cEBnI/vllm-sparse/recipes/models/...` |
| S4 | Legacy MkDocs recipes `vllm-project/recipes` @ 2ee0cd8: `GLM/GLM5.md`, `DeepSeek/DeepSeek-V3_2.md`, `DeepSeek-V3_2-Exp.md`, `strategies/*.yaml`, `SCHEMA.md` | 8xH200, 8xH20 (measured) | Hopper TP/padding advice | `dl.2cEBnI/recipes-repo/` |
| S5 | SGLang cookbook GLM-5.3 (`docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3`) + source `sgl-project/sglang:docs/src/snippets/configs/zai-org/glm-5.3*.jsx` | H200, B200, B300, GB300, AMD, Ascend | none | `dl.2cEBnI/sglang/glm53_cookbook.md`, `cfg_glm-5.3.jsx`, `bench_glm-5.3.jsx`, `agentx_glm-5.3.jsx` |
| S6 | SGLang GLM-5 / 5.1 / 5.2 cookbooks + generators | H100 (tp=16), H200, B-series, AMD | generator only | `dl.2cEBnI/sglang/GLM-5*_cookbook.md`, `glm-5-deployment.jsx`, `glm-51-deployment.jsx` |
| S7 | SGLang DeepSeek-V3.2 cookbook | **16x H800 measured**, B200 | Hopper multi-node verified | `dl.2cEBnI/sglang/dsv32_cookbook.md` |
| S8 | TokenSpeed model recipes (`lightseek.org/tokenspeed/recipes/models#glm-5-3`) | Blackwell MoE backend | not applicable | `dl.2cEBnI/tokenspeed/models.txt` |
| S9 | zai-org/GLM-5 GitHub @ 83c03d3 (+ history) | links only; history has 8-GPU FP8 commands | none | `dl.2cEBnI/glm5-repo/` |
| S10 | HF zai-org/GLM-5.3 @ ba32f8c (README, config, generation_config, tokenizer_config, chat_template); `zai-org/GLM-5.3-FP8` does not exist (HTTP 401) | n/a | n/a | `dl.2cEBnI/hf/`, `dl.2cEBnI/chat_template_ba32f8c.jinja` |
| S11 | arXiv 2607.02518 (China Unicom), "GLM-5 Serving Parameter Tuning for OpenClaw" | **2x8 H100, GLM-5 FP8, SGLang 0.5.9, measured** | **yes** | `dl.2cEBnI/arxiv/paper.txt` |
| S12 | gpustack/GLM-5.3-W4A8 model card | 8x H20-3e measured | claims H100 support, untested | (WebFetch only) |
| S13 | CSDN "GLM-5.3/GLM-5.3-Flash 部署实践" (hwcomputing.csdn.net/6aa1254348977663a5de3dc3.html) | 8x H20-3e, vLLM 0.28.0 vs SGLang | n/a | (WebFetch only) |
| S14 | llm-d GLM-5.2 well-lit path (llm-d.ai/docs/well-lit-paths/models/glm-5-2) | H200 P/D, 16-64 GPUs | n/a | (WebFetch only) |
| S15 | LMSYS blog 2026-07-13 GLM-5.2 NVFP4 agentic | B300/GB300 | n/a | (WebFetch only) |
| S16 | vLLM docs (attention backends, EP deployment, parallelism scaling) | generic | multi-node launch | `dl.2cEBnI/vllmdocs/*.txt` |
| S17 | GitHub issues/PRs (§7) | H100/H200/H20/B300 | several | (WebFetch only; git log in `dl.2cEBnI/vllm-sparse`) |

---

## 2. vLLM

### 2.1 GLM-5.3 recipe (S1)

File line numbers refer to `dl.2cEBnI/vllm-project_vllm_main_recipes_models_zai-org_GLM-5.3.yaml`.

**Requirements**
- `min_vllm_version: "0.29.0"` (l.33). Transformers 5.15.0 or newer (l.217).
- "Native GLM5Next rendering requires a build containing vllm-project/vllm#62424, including its Rust binary. The minimum version alone does not include this support." (l.39)
- On older builds, replace `--tokenizer-mode glm5_next` with `--tokenizer-mode auto` (l.234-235).
- "FP8 performance: DeepGEMM is required — install via `install_deepgemm.sh`" (l.423).

**Base flags and parsers**
- Base args (l.46-50): `--tokenizer-mode glm5_next --kv-cache-dtype fp8_e4m3`.
- Tool calling: `--tool-call-parser glm47 --enable-auto-tool-choice`. Reasoning: `--reasoning-parser glm47` (l.53-64).
- The SMG gRPC router uses `reasoning_parser: glm45` and `tool_call_parser: glm47_moe` (l.29-32).

**Hopper override (l.181-187)**
```yaml
hopper:
  # H100/H200 use plain fp8 KV cache; Blackwell keeps base_args' fp8_e4m3.
  extra_args: ["--kv-cache-dtype", "fp8"]
```

**H200 override (l.106-112):** "8x H200 (1128 GB) leaves ~23 GiB/GPU for KV after the 704 GB FP8 weights — vLLM estimates ~450K max tokens, so the 1M default fails at startup ("KV cache is needed"). Cap at 256K." It adds `--max-model-len 262144`.

**Standard command, "FP8 on 8xH200" (l.241-252)**
```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve zai-org/GLM-5.3 \
  --tokenizer-mode glm5_next --kv-cache-dtype fp8 --tensor-parallel-size 8 \
  --speculative-config.method mtp --speculative-config.num_speculative_tokens 5 \
  --tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice \
  --served-model-name glm-5.3
```

**Other variants and settings**
- **8xB200, full 1M context:** `--kv-cache-dtype fp8_e4m3 --max-num-seqs 32`. "`--max-num-seqs 32` — the single knob for fitting 1M context" (l.285-307).
- **AMD:** `--gpu-memory-utilization 0.80 --max-model-len 524288 --max-num-seqs 32 --linear-backend aiter --moe-backend aiter`, with `VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1` (l.254-274).
- **Spec-decode modes (l.65-95):**
  - `mtp`: `{"method":"mtp","num_speculative_tokens":5}`. DGX Station uses 4 tokens with `"moe_backend":"flashinfer_trtllm"`.
  - `dflash`: `{"method":"dflash","model":"incoai/GLM-5.3-DFlash2","num_speculative_tokens":7}`. AMD is unsupported. A comment says FP8+DFlash2 "ran end-to-end on 8-GPU HGX B300 (TEP8) and on H200" (l.14). SGLang notes the DFlash2 draft is licensed CC BY-NC-ND 4.0.
  - `dspark`: `{"method":"dspark","model":"RedHatAI/GLM-5.3-speculator.dspark","num_speculative_tokens":8}`.
- **W4A8 variant (l.148-165), the only recipe path that targets H100 directly:**
  - `gpustack/GLM-5.3-W4A8`, `vram_minimum_gb: 447`, `supported_hardware: [h100, h200]`.
  - `extra_args: --enable-expert-parallel --trust-remote-code`. The hopper override adds `--kv-cache-dtype fp8_ds_mla`.
  - "Weights drop from 704 GB to 372 GB… CUTLASS W4A8 is SM90-only".
- **Compatible strategies (l.172-179):** single_node_tp, single_node_tep, multi_node_tp, **multi_node_tp_pp**, multi_node_tep, multi_node_dep, pd_cluster.
- **Reasoning (l.341-386):**
  - "Thinking is always on — the generation prompt opens a `<think>` block unconditionally."
  - `reasoning_effort` is `low`/`high`/`max`, default max; "any other value falls back to `max`".
  - It can be passed via `chat_template_kwargs` or the top-level OpenAI `reasoning_effort`.
- **Sampling in the examples:** `temperature 1` / `1.0`.
- **Measured numbers:** AMD only.
  - MI325X, `--max-model-len 524288`, util 0.80: 1,477,168-token KV cache, 2.82x max concurrency, MTP AL 2.77 with 5 drafts (~500-token greedy run).
  - MI350X: 2,013,072 tokens, 3.84x, AL 3.19.
- **Benchmark command:** `vllm bench serve --dataset-name random --random-input-len 8000 --random-output-len 1024 --request-rate 10 --num-prompts 32 --ignore-eos`, with `--no-enable-prefix-caching` on the server. "pure throughput benchmarks tend to under-report real speed, because MTP's acceptance rate is usually low in synthetic runs."

The `vllm-project/recipes:models/zai-org/GLM-5.3.yaml` copy (date_updated 10-15) differs from S1. It says "vLLM 0.28.0 or newer", has no Rust frontend or `glm5_next`, and adds Ascend 950PR overlays. It is not rendered.

### 2.2 recipes.vllm.ai renderings for H100 (S2): synthesized, not verified

`GET https://recipes.vllm.ai/zai-org/GLM-5.3/hw/h100.json` returns `strategy: multi_node_tp`, `node_count: 2`, and `env: GLOO_SOCKET_IFNAME=$IFACE_NAME NCCL_SOCKET_IFNAME=$IFACE_NAME VLLM_USE_RUST_FRONTEND=1`:
```bash
vllm serve zai-org/GLM-5.3 --tokenizer-mode glm5_next --tensor-parallel-size 16 \
  --nnodes 2 --node-rank 0 --master-addr $HEAD_IP --kv-cache-dtype fp8 \
  --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm47
# worker: same + --node-rank 1 --headless
```

Alternatives under `/zai-org/GLM-5.3/hw/h100/strategies/`:
- `multi_node_tp_pp`: `--tensor-parallel-size 8 --pipeline-parallel-size 2 --nnodes 2 --node-rank {0,1} --master-addr $HEAD_IP [--headless]`. It does **not** set `VLLM_PP_LAYER_PARTITION`, so it hits §0.3 on every release.
- `multi_node_dep`: `--enable-expert-parallel --data-parallel-hybrid-lb --data-parallel-size 16 --data-parallel-size-local 8 --data-parallel-address $HEAD_IP [--data-parallel-start-rank 8]`.
- `multi_node_tep`: TP16 plus `--enable-expert-parallel`.
- `pd_cluster`: prefill and decode each TP16 on 2 nodes, NixlConnector, decode `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`, `vllm-router --policy round_robin --vllm-pd-disaggregation`.
- `single_node_tp`: TP8. This does not fit the FP8 checkpoint on H100. It is still emitted.
- `gpustack/GLM-5.3-W4A8/hw/h100.json`: single node, `--enable-expert-parallel --trust-remote-code --tensor-parallel-size 8 --kv-cache-dtype fp8_ds_mla` plus the parsers.

Strategy semantics come from `dl.2cEBnI/recipes-repo/strategies/*.yaml`, e.g. `multi_node_tp_pp.yaml:17-21`: "tp_pp → emit --tensor-parallel-size <gpu_count> --pipeline-parallel-size <node_count>… mp backend". `SCHEMA.md:131` calls tp_pp "vLLM's recommended multi-node default". `SCHEMA.md:420` gives "multi-node pure-TP on Hopper, which has no cross-node NVLink" as an example of a strategy to mark unsupported. GLM-5.3's YAML does not do so.

### 2.3 GLM-5.2 / GLM-5.1 / GLM-5 recipes (S3, S4)

**GLM-5.2** (`GLM-5.2.yaml`, updated 2026-10-03)
- Same hopper override, `--kv-cache-dtype fp8` (l.145-151). The comment adds: "H20 do not use kvcache fp8".
- **H20 DSpark "best practices"** (l.232-245):
  ```bash
  vllm serve zai-org/GLM-5.2-FP8 --tensor-parallel-size 8 --tool-call-parser glm47 --reasoning-parser glm47 \
    --enable-auto-tool-choice --max-model-len 200000 --max-num-seqs 16 --max-num-batched-tokens 32768 \
    --speculative-config '{"model": "RedHatAI/GLM-5.2-speculator.dspark","num_speculative_tokens": 7,"method": "dspark"}'
  ```
  Reported result: "tps up to 70+ for single request".
- B200 command uses `VLLM_DEEP_GEMM_WARMUP=skip`: "skips DeepGEMM JIT warmup for a faster startup; the first few requests compile kernels on demand" (l.334-349).
- "MTP performance: We fixed some MTP acceptance rate issue in PR #45895" (l.464).
- "If you need tool calling and MTP at the same time, use the latest `main` branch" (l.180-181).
- `kv_offload_support: offloading_cpu: verified, offloading_fs: verified` (l.141-143).
- MI325X: 1,564,016-token KV, AL 4.00 of 5 drafts. MI350X: 2,205,424 tokens, AL 3.62.

**GLM-5.1 / GLM-5** (`GLM-5.1.yaml`, `GLM-5.yaml`)
- Base args `--trust-remote-code --chat-template-content-format=string`. MTP 3: `--speculative-config.method mtp --speculative-config.num_speculative_tokens 3`.
- Troubleshooting: "Accuracy drift: Prefer the 0.19.0 stable release", "Tool calling + MTP: If both are needed, use the latest vLLM main branch", "DeepGEMM is required for FP8 performance".
- AMD `VLLM_ENGINE_READY_TIMEOUT_S: "3600"`: "Loading the FP8 checkpoint on 8x MI325X can outlast the Rust frontend's 600 s default wait".
- MI325X: 1,292,544-token KV (6.38x at 202,752); AL 2.72 (5.1) and 2.74 (5) with 3 drafts.

**Legacy `GLM/GLM5.md` measured result, 8xH200, GLM-5.1-FP8, TP8, MTP 3, `--reasoning-parser glm45` (l.157-210)**
- Random 8000-in/1024-out, 32 prompts at RPS 10.
- Throughput: output 526.57 tok/s (peak 800), total 4,640.43 tok/s.
- TTFT mean/median/P99: 13,395 / 14,494 / 22,952 ms. TPOT mean/median/P99: 35.39 / 34.58 / 49.72 ms. Mean ITL 57.98 ms.
- **MTP AL 1.64**: acceptance 21.33%, per position 36.59 / 20.39 / 7.00%.

**zai-org/GLM-5 README history** (commits 810a47b, eddd4aa, Apr-May 2026)
- 8-GPU FP8 commands with `--gpu-memory-utilization 0.85` (vLLM) and `--mem-fraction-static 0.85` (SGLang, EAGLE 3/1/4, `SGLANG_ENABLE_SPEC_V2=1`). Parsers `glm47` (tool) and `glm45` (reasoning).
- "When encounter Tool Call Parse issue with MTP enabled, please turn to vllm main branch".

### 2.4 GLM-5.3-Flash (different architecture; Hopper notes for contrast)

`GLM-5.3-Flash.yaml` is a 321B hybrid KDA + NoPE sparse MLA model. It lists `h100: verified`.
- Hopper override: `--disable-flashinfer-autotune`.
- Blackwell override comment: "FP8 KV cache only works on Blackwell for this model; Hopper runs BF16 KV."
- Base env `VLLM_ENGINE_READY_TIMEOUT_S: "3600"`.

Do not carry these settings over to GLM-5.3. Its head size is 576, not 512, and the backends differ.

### 2.5 DeepSeek-V3.2 / V3.2-Exp (same DSA architecture; vLLM)

**`DeepSeek-V3.2.yaml`** (h100 and h200 `verified`)
- Base args `--trust-remote-code --disable-flashinfer-autotune`.
- `--tokenizer-mode deepseek_v32 --tool-call-parser deepseek_v32 --reasoning-parser deepseek_v3`. MTP 3.
- DeepGEMM pin: `uv pip install git+https://github.com/deepseek-ai/DeepGEMM.git@v2.1.1.post3 --no-build-isolation`. Set `VLLM_USE_DEEP_GEMM=0` to skip MoE DeepGEMM (recommended on H20; also skips the long warmup).
- "Hardware: Minimum 8x H100/H200 80GB GPUs (BF16) or 3x H200 (NVFP4 variant)". This contradicts its own 805 GB FP8 size; treat it as wrong.
- **l.165-168:** "Avoid `-tp 8` with FlashMLA-Sparse. Due to kernel restrictions, TP=8 yields only 16 heads per rank but is padded to 64, causing overhead. Prefer TP=2 (Hopper) or TP=1 (Blackwell) with DP/EP mode: `vllm serve deepseek-ai/DeepSeek-V3.2 -dp 8 --enable-expert-parallel`."
- CUDA 13 `ptxas sm_110a` error fix: `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` and put `/usr/local/cuda/bin` on PATH.

**Legacy `DeepSeek-V3_2.md`, 8xH20 TP8, random 2048/1024, 100 prompts**
- Output 791.73 tok/s (peak 1,300), total 2,375.18 tok/s.
- TTFT mean 21,147 ms; TPOT mean 99.71 ms.
- "EP/DP Mode… recommended… kernels are mainly optimized for TP=1."

**Legacy `DeepSeek-V3_2-Exp.md`**
- "default config uses a custom fp8 kvcache… we recommend using bfloat16 kvcache for short requests, and fp8 kvcache for long requests."
- If you hit `CUDA error (flashmla-src/csrc/smxx/mla_combine.cu:201): invalid configuration argument`, "Try with `--max-num-seqs 256` or smaller (the default is 1024)."
- "Only Hopper and Blackwell data center GPUs are supported."

**vLLM blog 2025-09-29 (V3.2-Exp)**
- "on 16×H100, 8×H200, or 8×B200, you can run the model with tensor parallelism (expert parallelism has a slight bug we are fixing)".
- KV is "656 bytes" per token: 512 B fp8 NoPE + 16 B scales + 128 B bf16 RoPE. "we only support block size 64 for this model."

### 2.6 vLLM docs (S16)

**Multi-node with multiprocessing** (`parallelism_scaling.txt:197-210`)
```bash
vllm serve /path --tensor-parallel-size 8 --pipeline-parallel-size 2 --nnodes 2 --node-rank 0 --master-addr <HEAD_NODE_IP>
vllm serve /path --tensor-parallel-size 8 --pipeline-parallel-size 2 --nnodes 2 --node-rank 1 --master-addr <HEAD_NODE_IP> --headless
```
- The docs say: "set the tensor parallel size to the number of GPUs in each node, and the pipeline parallel size to the number of nodes".
- Check `NCCL_DEBUG=TRACE` for `[send] via NET/IB/GDRDMA`.
- vllm #41864 (vLLM 0.20.0, 2x8 H200) failed with `collective_rpc should not be called on follower node`. That report did not use `--nnodes/--node-rank`, and it was auto-closed.

**EP deployment** (`expert_parallel_deployment.txt`)
- `--all2all-backend {allgather_reducescatter (default), deepep_high_throughput (multi-node prefill), deepep_low_latency (multi-node decode, CUDA graphs), flashinfer_nvlink_* (MNNVL only)}`.
- 2-node DP16 example: `--tensor-parallel-size 1 --enable-expert-parallel --data-parallel-size 16 --data-parallel-size-local 8 --data-parallel-address IP --data-parallel-rpc-port 13345 --api-server-count=8`; the worker adds `--data-parallel-start-rank 8 --headless`.
- "On InfiniBand networked clusters, set… `export GLOO_SOCKET_IFNAME=eth0`".
- EPLB: `--enable-eplb --eplb-config '{"num_redundant_experts":32}'` recommended for multi-node, but "may not be a good fit… when KV cache space is at a premium".
- DeepEP needs IBGDA (`tools/ep_kernels/configure_system_drivers.sh` + reboot). The doc also says "kernels are optimized for disaggregated serving and may show poor performance for mixed workloads".
- The doc notes that 8-GPU DP/EP serving of DeepSeek-V3 "will work on a H200… For H100… refer to the multi-node deployment section".

**Attention-backend table** (`attn_backends.txt:1140-1250`, latest docs)

| Backend | KV dtypes | Block | Head sizes | DCP | Compute capability |
|---|---|---|---|---|---|
| FLASHMLA_SPARSE | auto, bfloat16, fp8_ds_mla, nvfp4_ds_mla | %64 | 576, 512 | ✅ | 9.x-10.x |
| FLASH_ATTN_MLA_SPARSE | auto, fp16, bf16 | %64 | – | ❌ | 9.x |
| FLASHINFER_MLA_SPARSE_SM90 (newer than 0.29) | auto, bf16, fp8, fp8_e4m3 | %64 | 512, 576 | ❌ | 9.x |

### 2.7 vLLM versions for GLM-5.3 on Hopper (git-verified in `dl.2cEBnI/vllm-sparse`)

Release tags: v0.29.0 = 98dff2a8 (the production env), v0.32.0 = 5df63a72, v0.33.0 = 4ba787ec (tagged Nov 1).

| PR | What | v0.29.0 | v0.32.0 | v0.33.0 | main |
|---|---|---|---|---|---|
| #46189 (06-24) | FLASH_ATTN_MLA_SPARSE (Hopper, BF16 KV, no head padding) | ✓ | ✓ | ✓ | ✓ |
| #50302 (07-31) | align block table (fixes #46074 crash at ≥~325K ctx with fp8_ds_mla) | ✓ | ✓ | ✓ | ✓ |
| #53755 (08-27) | FlashMLA sparse decode workspace fix (#44545/#53413 OOM) | ✓ | ✓ | ✓ | ✓ |
| #56893 (09-14) | FlashMLA pin bump that **re-introduced SM90 split-KV workspace** (#59305) | ✗ | ✓ (bad) | ✓ (bad) | open fix FlashMLA#29 |
| #58450, #57443, #59464, #58845 | GLM-5.3 performance work: metadata op 1.6-4.8x; fused multi-step decode +13.3% E2E at concurrency 1; index conversion reuse 3.5-3.9x kernel; skip qlnorm 4.4-7.7% TTFT | ✗ | ✓ | ✓ | ✓ |
| #56994 (09-29) | "Force reasoning mode for GLM-5.3 chat templates in the GLM MoE parser" | ✗ | ✓ | ✓ | ✓ |
| #58594 (09-26) | "[GLM5.3 Bug] Fix sparse indexer attn topk backend selection" | ✗ | ✓ | ✓ | ✓ |
| #60140 + #60281 (10-06/07) | fp8 KV default for GLM, then limited to **SM100** (Hopper default stays bf16) | ✗ | ✓ | ✓ | ✓ |
| #56403 (10-02) | constrain non-strict GLM-4.7 tool calls with a structural tag | ✗ | ✓ | ✓ | ✓ |
| #61253 (10-12) | stop re-parsing the whole GLM tool argument per chunk (streaming) | ✗ | ✗ | ✓ | ✓ |
| #61149 (10-12) | DeepGEMM bump to 26/09/30 + sparse indexer adapt | ✗ | ✗ | ✓ | ✓ |
| #62129 (10-17), #64458 (10-30) | MRV2 PP + spec-decode warmup deadlock / broadcast fixes | ✗ | ✗ | ✓ | ✓ |
| #62377 (10-19) | decode-only FULL CUDA graphs for DSv3.2/GLM-5.x sparse MLA with DCP=1 | ✗ | ✗ | ✓ | ✓ |
| #62424 (10-30) | Rust frontend native GLM-5.3 rendering (`--tokenizer-mode glm5_next`) | ✗ | ✗ | ✗ | ✓ |
| #59961 (10-31) | **carry indexer top-k across PP stages** | ✗ | ✗ | ✗ | ✓ |

**Local cross-checks in the 0.29 env**
- `glm45` and `glm47` are aliases of the same parsers (`reasoning/__init__.py:55-62`, `tool_parsers/__init__.py:57-64` → `Glm47MoeParserReasoningAdapter` / `Glm47MoeModelToolParser`). There is no `glm5_next` tokenizer mode (`tokenizers/registry.py`).
- The GLM parser starts in REASONING state unless `chat_template_kwargs.thinking/enable_thinking` is falsy (`parser/glm47_moe.py:184-195`). Sending `enable_thinking: false` to GLM-5.3, which always thinks, would push the reasoning into `content`. #56994 fixes this.
- Inbound `reasoning_content` is renamed to `reasoning` (`entrypoints/openai/chat_completion/protocol.py:530-546`). Both are re-exported to the template as `reasoning_content` (`entrypoints/chat_utils.py:2000-2004`), so either client field works.
- `VLLM_ENGINE_READY_TIMEOUT_S` defaults to 600 and bounds the API server's wait for engine-core readiness. The error text reads "often caused by slow weight loading" (`v1/engine/core_client.py:655-668`). Recipes raise it to 3600 for slow loads.
- `--enable-flashinfer-autotune` is a boolean flag (`engine/arg_utils.py:1672-1675`). To disable it on 0.29, use `--no-enable-flashinfer-autotune`. `--disable-flashinfer-autotune` as used in the newer YAMLs is UNVERIFIED on 0.29.

---

## 3. SGLang

### 3.1 GLM-5.3 cookbook (S5)

**Hardware and checkpoints**
- Hardware: H200, B200, B300, GB300, MI300X/325X/355X, Ascend A3/A5. **No H100.**
- Checkpoints: `zai-org/GLM-5.3` (FP8), `-BF16`, and `RadixArk/GLM-5.3-NVFP4` (experimental).
- "Recommended generation: `temperature=1.0`, `top_p=0.95`" (`glm53_cookbook.md:255`).

**Kernels and KV dtype** (l.261)
- SGLang "auto-selects the DSA attention backends (`flashmla_sparse` prefill, `fa3` decode, `sgl-kernel` indexer topk)".
- It "auto-selects the KV-cache dtype for DSA models — `fp8_e4m3` on Blackwell… and `bf16` on Hopper (H200)".
- "On Hopper, pairing `--kv-cache-dtype fp8_e4m3` with `--dsa-prefill-backend flashmla_sparse_q8 --dsa-decode-backend flashmla_kv` selects the native FP8 sparse prefill kernel… GLM-5.3's 64 query heads match the kernel's native tile".
- Warning (l.86-88): all recipes use `--dsa-topk-backend sgl-kernel`; the other top-k backends are not fully validated.

**MTP** (l.263)
- `--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6` for low latency; `1-1-2` for balanced.
- "`index_share_for_mtp_iteration` reuses the DSA indexer's topk across draft steps (effective only at `--speculative-eagle-topk 1`)."

**Memory, chunked prefill, PD, CP**
- "Start around `--mem-fraction-static 0.8` on H200 (TP8)" (l.267).
- Chunked prefill: "For long-input balanced workloads, start with `--chunked-prefill-size 32768`" (l.275).
  - The GLM-5.2 page adds that 32768 gave "+34–78% output throughput and −39–59% TTFT on 8×H200 and 8×B200 (8K-in / 1K-out)".
  - It is neutral for high throughput. About 60-90 concurrent 8K+1K FP8 requests fit on one 8-GPU node.
- PD (l.273): Mooncake auto-detects the IB HCA. In Docker, expose IB devices or "Mooncake silently falls back to TCP".
- CP (l.400-431): `--attn-cp-size 8 --enable-prefill-cp --cp-strategy interleave`; zigzag is unavailable for GLM-5.x. CP is single-machine only (generator: "SGLang asserts tp_size <= 8; cross-machine CP has precision issues"). LayerSplit is for PD prefill workers with Mooncake.

**H200 FP8 deploy cells** (`cfg_glm-5.3.jsx:587-626`, `verified: true`)
- low-latency: `--tp 8 --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 --mem-fraction-static 0.8`
- high-throughput: `--tp 8 --attn-dp-size 8 --moe-a2a-backend deepep --speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 --mem-fraction-static 0.85 --chunked-prefill-size 32768 --max-running-requests 256`
- Parsers: `--reasoning-parser auto --tool-call-parser auto`. These resolve to glm45 and glm47 on ≥v0.5.20. "the model emits the newer `<tool_call>…<arg_key>…<arg_value>…` format, which the older `glm45` [tool] parser does not parse."

**H200 FP8 benchmarks** (`bench_glm-5.3.jsx:5-30`, SGLang main @ 20a491d1d311, random 8192-in/1024-out). The speed numbers ran under `SGLANG_SIMULATE_ACC_LEN`, so they are throughput-mechanism numbers only.

| Cell | Concurrency | TTFT ms | TPOT ms | tok/s/GPU | Accuracy |
|---|---|---|---|---|---|
| low-latency (EAGLE 5/1/6, AL pinned 3.5) | 1 | 780 | 3.71 | 252 | GSM8K 97.42 |
|  | 16 | 5,499 | 14.20 | 920 | |
| high-throughput (DP-attn 8 + DeepEP, EAGLE 1/1/2) | 64 | 7,790 | 21.90 | 2,440 | GSM8K 97.12, AIME25 91.88 |
|  | 256 | 57,607 | 33.39 | 2,558 | |

**Multi-node cells.** Only BF16 has them (`cfg_glm-5.3.jsx:809-860`, `verified: false`): "2x 8xH200 / 2x 8xB200 at TP16". The flags are `--tp 16`, EAGLE 5/1/6 or 1/1/2, `--mem-fraction-static 0.85`, `--max-running-requests 80/256`.

**Ascend A3 prefill uses PP4 with an indexer-aligned partition** (`cfg_glm-5.3.jsx:378,384`):
- `SGLANG_PP_LAYER_PARTITION=18,20,24,16` with `--pp-size 4 --tp-size 8`. The stage starts are 0, 18, 38 and 62, all indexer-owning layers.
- This is the only published GLM-5.3 PP partition found, and it is aligned with §0.3.

**Claude Code integration** (l.395)
- "GLM-5.3's chat template renders `tools` **before** `system`, so that per-request hash is the first token to diverge between turns and the radix prefix cache re-prefills the whole system + history every turn."
- The fix there is `CLAUDE_CODE_ATTRIBUTION_HEADER=0`. The general lesson: any per-turn change in the system prompt, or in the `Reasoning Effort` line, destroys prefix reuse.

**MI355X agentic recipes** (InferenceX). These are not Hopper, but the same rule applies: "Set `--max-running-requests` and `--cuda-graph-max-bs-decode` to twice the concurrency".

### 3.2 GLM-5 / GLM-5.1 / GLM-5.2 cookbooks (S6)

**Generator configs**
- `glm-5-deployment.jsx:78`: `h100: { fp8: { tp: 16, mem: 0.85 }, bf16: { tp: 32, mem: 0.85 } }`.
- `glm-51-deployment.jsx:69`: `h100: { fp8: { tp: 16, mem: 0.85 } }`.

**Emitted GLM-5.1 H100 command** (the user must add `--nnodes 2 --node-rank N --dist-init-addr`)
```bash
sglang serve --model-path zai-org/GLM-5.1-FP8 --tp 16 [--attn-dp-size 16] \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --mem-fraction-static 0.85
```

**Other notes**
- GLM-5 page: "DP Attention… trades off low-concurrency latency for high-concurrency throughput". It shows an H200 example with `--enable-flashinfer-allreduce-fusion`.
- IndexCache override for GLM-5 and 5.1: `--json-model-override-args '{"index_topk_pattern": "FFSFSSSF..."}'`. This is not needed for GLM-5.3, which ships `index_topk_freq=4` and `indexer_types` in its config.

### 3.3 DeepSeek-V3.2 cookbook: Hopper multi-node (S7)

Recipes and measurements are on 16x H800, which has 80 GB per GPU and slower NVLink than H100.

**Kernel facts** (l.67)
- `flashmla_sparse_q8` is "Hopper SM90 + `--kv-cache-dtype fp8_e4m3` only, prefill only".
- Defaults: "Hopper BF16 KV → `flashmla_sparse` prefill / `fa3` decode; Hopper FP8 KV → `flashmla_kv` both".
- MHA is used for prefill below 2048 tokens (`SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`).
- `--dist-timeout 3600` is suggested for slow loads.

**Recommended for "local deployment of agentic workflow… on Hopper"** (l.449-475), sglang 0.5.9:
```bash
export SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS=32 SGLANG_SET_CPU_AFFINITY=1
sglang serve --model-path $M --nccl-init $MASTER_ADDR:$MASTER_PORT --nnodes 2 --node-rank $RANK --tp 16 \
  --attn-dp-size 2 --page-size 64 --context-length 65536 --max-running-requests 128 \
  --speculative-algorithm EAGLE --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 \
  --allow-auto-truncate --enable-metrics --tool-call-parser deepseekv32 --reasoning-parser deepseek-v3
# 1k/1k comments: raw 4948 tok/s; dp2 5020; dp4 4943 (MAX ITL 35.7 s); dp2+mtp 6843 tok/s
```

**Measured on 16x H800** (l.848-1050)
- Concurrency 1: output 86.20 tok/s, TTFT 199.88 ms, TPOT 10.99 ms, AL 3.27.
- Concurrency 1,024 (2,048 prompts, 1k/1k): output 2,530.66 tok/s (total 5,101), TPOT 49.18 ms, TTFT mean 127.7 s, AL 3.26.
- With `--random-range-ratio 1`: output 3,421.84 tok/s, TPOT 36.31 ms.
- "when concurrency is greater than 128, the TTFT increase sharply".

**PP variants "verified on Hopper"** (l.480-534)
- `--tp 8 --pp-size 2 --moe-a2a-backend deepep --ep-size 16 --page-size 128 --chunked-prefill-size 16384 --attention-backend dsa --dsa-prefill-backend flashmla_sparse --dsa-decode-backend flashmla_sparse --enable-prefill-cp --cp-strategy interleave --cuda-graph-max-bs-decode 128 --max-running-requests 128`
- An fp8-KV variant: `--tp 8 --pp-size 2 --pp-async-batch-depth 1 --enable-prefill-cp --cp-strategy interleave --kv-cache-dtype fp8_e4m3`.

---

## 4. TokenSpeed (S8)

GLM-5.3 command:
```bash
ts serve zai-org/GLM-5.3 --served-model-name glm-5.3 --trust-remote-code --tensor-parallel-size 8 \
  --enable-expert-parallel --moe-backend flashinfer_trtllm --kv-cache-dtype fp8 --max-model-len 262144 \
  --max-num-seqs 128 --draft-model-path-use-base --speculative-algorithm MTP --speculative-num-steps 3
```
- GLM-5.2 adds `--chunked-prefill-size 8192`. TokenSpeed "defaults the reasoning parser to glm45".
- **Not usable on H100 as written.** The same page says of `flashinfer_trtllm`: "The fused MoE path needs a Blackwell GPU (B200/B300). On other NVIDIA platforms, use `--moe-backend triton`."

---

## 5. Z.ai: HF model card (S10) and the GLM-5 repo (S9)

**Notes in the HF model card** (ba32f8c README, identical to the HSG copy)
- "`reasoning_effort`… accepts three levels: `low`, `high`, and `max`. It defaults to `max` if not passed (or if set to any other value)… For benchmark and leaderboard reproduction, keep the default `max`."
- "In the chat template for GLM-5.3, `clear_thinking` defaults to `false`… For chat scenarios, explicitly pass `clear_thinking=true`." The default false (preserve reasoning) is therefore the agentic setting.
- In GLM-5.2, `reasoning_effort` accepted only high/max, and `enable_thinking=false` turned thinking off. GLM-5.3 has no off switch.

**Evaluation sampling** (model card footnotes)
- **DeepSWE: mini-swe-agent, `temperature=0.95`, `top_p=1.0`, `timeout=6h`, 400K context.**
- Terminal-Bench 2.1: Claude Code, `temperature=1.0, top_p=1, max_new_tokens=65536`. HLE: `temperature=1.0, top_p=0.95`, 163,840 max generation tokens. SWE-Marathon: `top_p=0.95`, 1M context.
- `generation_config.json`: `temperature 1.0, top_p 0.95`, eos `[154820, 154827, 154829]`.
- gpustack's card: "Without top-p, thinking mode can loop." Unsloth recommends `top_p=1.0` for "long agentic tasks".

**Serving links on the card:** SGLang cookbook, vLLM recipes, TokenSpeed, Transformers (`glm_moe_dsa`), KTransformers (CPU/GPU hybrid), Unsloth (GGUF), Ascend (vLLM-Ascend / xLLM / SGLang). None covers H100 multi-node.

**config.json facts relevant to serving**
- `indexer_types` marks 21 "full" (indexer-owning) layers: `0,1,2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74`. The other 57 are "shared".
- `index_topk_freq 4`, `index_skip_topk_offset 3`, `index_share_for_mtp_iteration true`, `num_attention_heads 64`, `kv_lora_rank 512`, `qk_rope_head_dim 64`, `moe_intermediate_size 2048`. FP8 block [128,128]; `scoring_func sigmoid`.

---

## 6. Third-party measured deployments (closest to dfw)

### 6.1 arXiv 2607.02518: GLM-5 FP8 on 2x8 H100 with SGLang (S11)

**Setup**
- 2 nodes x 8 H100, 2x Xeon 8468, 2 TiB RAM, ConnectX-7 IB.
- SGLang v0.5.9, "two-node distributed mode… flashinfer attention backend".
- Fixed flags: `--nnodes 2 --tool-call-parser glm47 --reasoning-parser glm45 --enable-reasoning-tokens --context-length 202752 --mem-fraction-static 0.85 --enable-metrics --enable-cache-report`. No MTP is mentioned.
- Workload: `bench_serving --dataset-name random --random-input-len 30000 --random-output-len 500 --random-range-ratio 0.2`. Synthetic, so no prefix reuse.

**Results** (chunk = `--chunked-prefill-size`, max-run = `--max-running-requests`)

| chunk | tp | pp | max-run | Out tok/s | Total tok/s | Req/s | TTFT avg s | TTFT P90 s | TPOT ms | Lat P90 s |
|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 4 | 4 | 16 | 215.17 | 9,029.64 | 0.43 | 8.98 | 14.18 | 54 | 40.23 |
| 3072 | 4 | 4 | 16 | 220.89 | 9,269.88 | 0.44 | 8.04 | 12.58 | 54 | 40.07 |
| 4096 | 4 | 4 | 16 | 187.90 | 7,885.54 | 0.38 | 9.73 | 17.39 | 63 | 51.98 |
| 6144 | 4 | 4 | 16 | 202.62 | 8,503.08 | 0.41 | 8.85 | 15.00 | 59 | 44.97 |
| **3072** | **4** | **4** | **24** | **238.13** | **9,993.23** | **0.48** | **6.69** | **11.61** | **50** | **32.64** |
| 3072 | 4 | 4 | 32 | 235.55 | 9,885.18 | 0.47 | 7.01 | 12.07 | 50 | 33.21 |
| 3072 | 4 | 4 | 48 | 231.16 | 9,701.02 | 0.46 | 7.39 | 12.87 | 50 | 34.03 |
| 4096 | 4 | 4 | 32 | 225.47 | 9,462.00 | 0.45 | 7.73 | 12.90 | 51 | 36.01 |
| 3072 | 2 | 8 | 24 | 207.43 | 8,705.06 | 0.41 | 7.80 | 12.96 | 57 | 37.41 |
| 3072 | 8 | 2 | 24 | 164.56 | 6,905.99 | 0.33 | 9.48 | 16.50 | 73 | 47.55 |

- TP16 was not tested.
- The authors' reading: "tp=4 / pp=4 appears to strike the right balance… per-device memory headroom for long-context KV cache, communication cost for TP collectives, and pipeline balance".
- Whether SGLang 0.5.9 PP handles the IndexCache stage-boundary issue is UNVERIFIED. GLM-5, unlike 5.2/5.3, had no built-in index_topk_freq, so it was unaffected unless IndexCache was enabled.

### 6.2 Other measured or real deployments

- **vLLM #46074, GLM-5.2-FP8 on 4x8 H100** (DP=4, TP=8, EP; vLLM 0.23)
  - Flags: `--kv-cache-dtype fp8_ds_mla --quantization fp8`, MTP 5 (also 2), `--max-num-seqs 32 --max-num-batched-tokens 8192 --no-enable-prefix-caching --no-async-scheduling --disable-custom-all-reduce`.
  - Stable at max_model_len 230,000 and 300,000; crashed at ≥325K (off-by-one, fixed by #50302). bf16 KV was stable at every size.
- **vLLM #59305 reporter ehfd: 8x H100, GLM-5.3-NVFP4, TP8 + DCP8, vLLM 0.30.0.**
  - `--gpu-memory-utilization 0.925` crashed on 13.6K-20K prompts; 0.90 passed repeated 20K recall checks.
  - Evidence that DCP8 runs on H100 for glm_moe_dsa (NVFP4 weights). FP8 weights are UNVERIFIED.
- **gpustack/GLM-5.3-W4A8 on 8x H20-3e, vLLM nightly 0.26.1rc1.dev229** (S12)

  | Setup | Weights/GPU | KV tokens | Prefill tok/s | Decode @128 | Decode @256 |
  |---|---|---|---|---|---|
  | **official FP8, TP8** | ~88 GiB | **608,128** | 3,675 | 1,686 | 2,094 |
  | W4A8 TP8 + EP | 52.6 GiB | 1,257,280 | 3,441 | 1,617 | 1,779 |
  | W4A8 2x(TP4+EP) | – | 638,336 | 5,993 | 2,256 | 3,249 |

  - Accuracy: GPQA-D pass@1 FP8 89.65 vs W4A8 90.28; NLL 0.8929 vs 0.8940.
  - Throughput replicas: `--kv-cache-dtype fp8_ds_mla --max-model-len 65536 --gpu-memory-utilization 0.92`. Long context: TP8 `--max-model-len 262144 --gpu-memory-utilization 0.90`. MTP 1.
  - "Avoid `--moe-backend marlin` or `triton`. Expert parallelism is required for W4A8." TP16 is impossible (moe_intermediate/TP must be divisible by 256).
- **CSDN, GLM-5.3 FP8 on 8x H20-3e, TP8, 131,072 context, no MTP, no prefix cache** (S13). SGLang uses BF16 KV; vLLM 0.28.0 uses KV "auto". Medians of 3 runs, SGLang / vLLM:
  - 128-in/64-out: concurrency 1: 90.41 / 69.27 tok/s; 32: 964.25 / 814.68 tok/s. P50 TTFT at concurrency 32: 444 / 821 ms.
  - 4K/128 at C8: 115.40 / 103.63 tok/s, TTFT 5.91 / 1.90 s.
  - 16K/256 at C8: 57.67 / 53.24 tok/s, TTFT 20.28 / 12.54 s.
  - vLLM had lower TTFT on long prompts; SGLang had higher throughput.
- **llm-d GLM-5.2-FP8 on H200, P/D** (S14)
  - Each replica DEP16 (TP1, DP16, EP16) over 2 nodes. Prefill `deepep_high_throughput`; decode `deepep_low_latency` with IBGDA/NVSHMEM. MoE backend DeepGemm. NixlConnector. `gpu-memory-utilization` 0.935 prefill / 0.95 decode. MTP 3. Parsers `glm45` (reasoning) and `glm47` (tool).
  - Env: `NCCL_IB_HCA=ibp NVSHMEM_HCA_PREFIX=ibp NVSHMEM_REMOTE_TRANSPORT=ibgda`. The startup probe allows 45 min.
  - Workload: median main-agent request ~195K input / 317 output; "96% of turns reuse at least 90% of input".
  - MTP halves median TBT; aggregate output 1.6-2.0x at concurrency ≤128.
  - CPU KV offload (OffloadingConnector, mmap in /dev/shm): +13% output, −32% median TTFT, −27% median turn time on 24 GPUs at concurrency 32. At concurrency 128: ~951 → 2,776 tok/s.
  - "DeepEP fails on rail-only networks".
- **LMSYS GLM-5.2 NVFP4 on B300/GB300** (S15). OpenHands replay: ~80k mean input, ~220 output per turn, 13 turns, ~92% prefix-cache hit. "GLM-5.2's MTP head frequently reaches accept lengths of 5+".
- **phala GLM-5.2 W4AFP8 on 8x H200** (SGLang 0.5.13.post1, TP8, `--kv-cache-dtype fp8_e4m3`, 1M context). Single stream 75 tok/s without EAGLE and 118 with EAGLE 1/1/2. 8K/1K at concurrency 16: 658 → 733 tok/s.
- **Miles v0.1** (arXiv 2609.08368; GLM-5.2 RL on 64 GB300). Rollouts use key-based session affinity, "so each turn prefills only the newly generated suffix".

---

## 7. Known issues relevant to Hopper and multi-node

| Issue | Status (2026-11-06) | Impact on dfw | Workaround / note |
|---|---|---|---|
| vllm #63170: GLM-5.x DSA under PP reads stale top-k | closed; fix PR #59961 on main only | **any PP>1 on vLLM ≤0.33.0** (0.29 env confirmed by code) | Align stage starts to indexer layers (`VLLM_PP_LAYER_PARTITION=38,40`), or build main ≥63d3d39c. H200 PP8 stock NLL 0.9478 vs TP8 0.8165 (+16.1%). In the PR's GSM8K table (4x B300, GLM-5.3-NVFP4): main 0.879 vs fix 0.908 at PP2xTP2; with DCP2, main 0.820 vs fix 0.908; aligned 38,40 without the fix, 0.882. The author says alignment "restored accuracy". Whether alignment alone is sufficient is UNVERIFIED |
| vllm #46262: PP2 GLM-5.2-FP8 tool calls garbled (ROCm, chunked prefill) | open | PP + tool calling | Reporter needed `--no-async-scheduling` ("required to avoid KeyError in PP mode with tool calling"). Root cause unclear; possibly #63170 |
| vllm #41864: multi-node PP `collective_rpc should not be called on follower node` (0.20.0) | auto-closed | multi-node PP with mp backend | Use the documented `--nnodes/--node-rank/--master-addr/--headless`; the reporter did not |
| sglang #21934: `Unsupported h_q: 4` at TP16 on H100 (GLM-5-FP8, fp8_e4m3 KV) | closed (head-padding PR #22372) | TP16 on SGLang | "The updated flashmla kernel only supports num_head = 64 or 128"; DP-attention suggested on SM90 |
| sglang #22511: 2x H800 TP16 GLM-5 `/health` failing ("could not get a response from detokenizer") | – | health checks under load | /health measures recent responses, not liveness |
| vllm-project/recipes#565: FLASHMLA_SPARSE `[kv_cache_dtype not supported]` with `fp8_e4m3` (H200, GLM-5.2) | closed | KV dtype choice | Use `--kv-cache-dtype fp8` / `fp8_ds_mla` on Hopper |
| vllm #46074: fp8_ds_mla indexer off-by-one at max_model_len ≥~325K (4x8 H100) | fixed by #50302 (in 0.29.0) | long context | Keep max_model_len ≤262,144 anyway |
| vllm #44545 / #53413: FlashMLA sparse decode `o_accum` 8 GiB at 32k-token mixed batch (fp8 KV, H200) | fixed by #53755 (in 0.29.0) | – | – |
| vllm #59305: SM90 split-KV workspace allocated again (2 GiB per 8k tokens, unprofiled) | **open**; regression in v0.32.0/v0.33.0 | fp8 KV on 0.32.0 | `--gpu-memory-utilization 0.90` (0.925 crashed on 8x H100), `--max-num-batched-tokens ≤8192`; FlashMLA#29 pending |
| vllm #57878: fp8 FlashMLA sparse decode crashes at init with `--enforce-eager` on SM90 (GLM-5.3-Flash, 16k batched tokens) | open | eager debugging | Do not combine fp8 KV with `--enforce-eager` and 16k batched tokens; 8,192 or CUDA graphs worked |
| vllm #65278: GLM-5.3 not batch invariant; refused under `VLLM_BATCH_INVARIANT=1` | open | deterministic replays | none |
| vllm #42400: GLM-5.1 tool-call parse failures with MTP 3 (Claude Code, `/v1/messages`, streaming) | closed, no fix linked | MTP + streaming tool calls | Reporter: the glm45 reasoning parser consumed tokens the tool parser needed; vLLM 0.19 worked |
| vllm #60703 / PR #60842: glm47 parser swallows the rest of the stream after a stray `<tool_call>` in prose | PR open | silent missing tool calls (`finish_reason=stop`) | Monitor tool-call rate; related fixes #54971 and #49249 |
| vllm #31473: DSv3.2 performance tracker ("BF16 KV, support TP without padding") | open/stale | head padding | Mentions #49357 FP8 KV OOM during long prefill in mixed batches on Hopper |
| vllm #58980: DCP + fp8 sparse attention processes other ranks' masked top-k (H20) | open | DCP perf | 24-28% prefill and 3.9-10.8% decode gains pending |
| DCP for glm_moe_dsa with fp8_ds_mla | conflicting | KV capacity lever | The dealignai/GLM-5.3-CYBERSECURITY-FP8 card says 1M via DCP is "closed on glm_moe_dsa… page size is not divisible…" for fp8_ds_mla. PR #59961 tested PP2xTP2xDCP2 on B300, and ehfd ran TP8 DCP8 on H100 (NVFP4). UNVERIFIED for FP8 on H100 |

---

## 8. Chat template, parsers, special tokens

### 8.1 What the GLM-5.3 template expects

Line numbers refer to `dl.2cEBnI/chat_template_ba32f8c.jinja`.

**Prefix**
- `[gMASK]<sop>` followed by `<|system|>Reasoning Effort: {Low|High|Max}` (l.1-3). The line is **always** emitted.
- `effective_reasoning_effort = reasoning_effort if … in ['low','high'] else 'max'`.

**Tools**
- If `tools` is set, a second `<|system|>` block holds "# Tools … <tools>…</tools>" (l.32-50).
- Each tool is printed as one JSON line. The `defer_loading` and `strict` keys are dropped, and tools with `defer_loading: true` are hidden.
- Instruction (l.49-50): "For each function call, output the function name and arguments within the following XML format: `<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...</tool_call>`".
- **The tools block precedes any system message** (l.245-246 render system messages in order).

**Assistant turns** (l.134-158)
- Rendered as `<|assistant|><think>{reasoning}</think>{content.strip()}` followed by tool calls `<tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v | tojson if not string}</arg_value>…</tool_call>`.
- Reasoning comes from `m.reasoning_content`, or from a `<think>…</think>` block inside `content`.
- Past reasoning is kept unless `clear_thinking=true`. In that case only turns after the last user message keep it; earlier turns get `<think></think>`.
- `clear_thinking` defaults to false (l.4).
- `tc.arguments` must be a mapping. vLLM's `_postprocess_messages` converts the OpenAI JSON string into a dict.

**Tool results** (l.159-244)
- Consecutive `role:tool` messages become one `<|observation|>` followed by `<tool_response>…</tool_response>` per result.
- Results are reordered to match the preceding assistant `tool_calls` order, but only if every result id and call id is non-empty and unique and every result id exists among the calls. Otherwise they keep message order.
- Supports list-of-outputs content (`{output: …}` entries) and `tool_reference` outputs. The latter expand to the referenced tool's JSON inside `<tool_response><tools>…</tools></tool_response>` (tool search).

**Other content**
- Multimodal parts are replaced by `<reminder>You are unable to process this {image|video|audio}…</reminder>` (l.60-62).

**Generation prompt:** `<|assistant|><think>` (l.249-251). Thinking cannot be disabled. The reasoning parser must start in reasoning mode; see §2.7 for the 0.29 parser behaviour with `enable_thinking: false`.

**Special token ids** (from `tokenizer.json`)
- `<|endoftext|>`=154820, `[gMASK]`=154822, `<sop>`=154824, `<|system|>`=154826, `<|user|>`=154827, `<|assistant|>`=154828, `<|observation|>`=154829.
- `<think>`=154841, `</think>`=154842, `<tool_call>`=154843, `</tool_call>`=154844, `<tool_response>`=154845, `</tool_response>`=154846, `<arg_key>`=154847, `</arg_key>`=154848, `<arg_value>`=154849, `</arg_value>`=154850.
- EOS (`generation_config.json`) = [154820, 154827, 154829]. A tool-calling turn therefore ends when the model emits `<|observation|>`.
- `<tools>` is plain text, not a special token.

### 8.2 Diff: HSG copy vs HF ba32f8c

HF history, via `huggingface.co/zai-org/GLM-5.3/commits/main` plus fetching each revision. Copies are in `dl.2cEBnI/hf/ct_*.jinja`.

| Revision | Date | Bytes | sha256 (first 16) | Note |
|---|---|---|---|---|
| 7cda819 "Initial commit 0828" | Aug 27 | 10,465 | 69bb3ab52067898e | **= HSG copy** (`hsg-ref/model_meta/chat_template.jinja`, mtime Sep 1) |
| 3033303 | Aug 28 | 10,465 | 69bb3ab52067898e | same |
| aca966e "chat template: early break in tool result reordering check" | Sep 4 | 10,734 | 3740abcea51c4583 | fix plus early breaks |
| ba32f8c "chat template: speed up tool-result validation and reordering (#31)" | Oct 14 (HEAD of main) | 10,614 | 0e2b6268ac54f04d | the revision being downloaded |

**Difference 1 (semantic, at line 65)**
- `visible_text()` changed `{%- else -%}{{- content }}` to `{%- elif content is not none -%}{{- content }}`.
- With the old template, a message whose `content` is Python `None` renders the literal string `None`.
- Verified with jinja2 3.1.6 in an ImmutableSandboxedEnvironment with loopcontrols (`tools/render_template_test.py`). An assistant tool-call turn with `content=None` renders as:
  - HSG/7cda819: `<|assistant|><think>I should look.</think>None<tool_call>read<arg_key>path</arg_key>…`
  - ba32f8c: `<|assistant|><think>I should look.</think><tool_call>read<arg_key>path</arg_key>…`
- In vLLM 0.29, `chat_utils._parse_chat_message_content` turns `content=None` into `[]` before templating (`chat_utils.py:1977-1978`), so vLLM serving masks the bug.
- It does bite any pipeline that applies the template itself, e.g. HF `apply_chat_template` for training, token replay, or SFT data.

**Difference 2 (performance only, lines 175-243)**
- Old: tool-result validation loops over every result, calling `has_dup_tool_result_id` (an O(B) scan) and `tc_id_exists`. Reordering is a nested loop over tool_calls x results x entries.
- New: one pass collects result ids and their positions. Validation is `'' in …`, `unique` length checks and membership. Reordering sorts both id lists, merge-joins them, and sorts the pairs by call index.
- The output is identical in the tested case (results reordered to call order: `print(1)` before `a.py b.py`) and by construction for valid inputs.
- aca966e added early `break`s to the old loops.
- Why it matters for serving: on long agentic histories the Python Jinja render runs per request in the API server. PR #62424 reports its Rust renderer is 1.77-1.86x faster than MiniJinja on "Long tool histories".

### 8.3 Parsers and kwargs across engines

| Engine | Tool parser | Reasoning parser | Notes |
|---|---|---|---|
| vLLM | `glm47` (+ `--enable-auto-tool-choice`) | `glm47` (recipes ≥5.2) / `glm45` (older docs, llm-d) | In the 0.29 env both names map to the same Glm47Moe parser |
| vLLM SMG gRPC router | `glm47_moe` | `glm45` | from S1 l.29-32 |
| SGLang | `glm47` (or `auto` on ≥v0.5.20) | `glm45` (or `auto`) | `glm45` *tool* parser cannot parse GLM-5.3's arg_key format |
| TokenSpeed | default | `glm45` default | – |

Template kwargs: `chat_template_kwargs: {"reasoning_effort": "low"|"high"|"max", "clear_thinking": bool}`. vLLM also accepts top-level `reasoning_effort`. Do not send `enable_thinking`/`thinking`; they are ignored by the template and mis-set the 0.29 parser.

---

## 9. Recommendations for the dfw 2x8 H100 GLM-5.3 build

These are derived from the sources above. None of them has been run on H100 by any source.

1. **First layout to bring up: aggregated PP2xTP8 over the 2 nodes in vLLM, with a mandatory aligned partition.** Draft command (UNVERIFIED composite of S1, S2 and S16):
   ```bash
   # both nodes
   export VLLM_PP_LAYER_PARTITION=38,40            # stage 2 starts on indexer layer 38 (sec 0.3, 7)
   export VLLM_ENGINE_READY_TIMEOUT_S=3600         # 756 GB from Lustre; default 600 s (sec 2.7)
   export GLOO_SOCKET_IFNAME=<eth-if> NCCL_SOCKET_IFNAME=<eth-if>   # recipes multi-node env; IB HCAs via NCCL_IB_HCA in the image
   vllm serve /lustre/fsw/portfolios/nemotron/users/hengyuf/models/GLM-5.3 --served-model-name glm-5.3 \
     --tensor-parallel-size 8 --pipeline-parallel-size 2 --nnodes 2 --node-rank <0|1> --master-addr <head-ip> [--headless on rank 1] \
     --kv-cache-dtype fp8 --max-model-len 262144 --max-num-batched-tokens 8192 --max-num-seqs 32 \
     --gpu-memory-utilization 0.90 --enable-prefix-caching \
     --tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice
   # then add: --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
   ```
   - Why this layout:
     - It has no cross-node all-to-all.
     - Each GPU holds about half the layers, so KV per token per GPU halves (MLA KV is replicated across TP).
     - It is vLLM's documented multi-node default, and SGLang published Hopper PP2xTP8 DSA recipes.
   - Caveats:
     - It was the *worst* of the three SGLang splits in arXiv 2607.02518.
     - MTP + PP fixes (#62129, #64458) exist only in 0.33.0.
     - Bring it up without MTP first, then add MTP.

2. **Next, A/B PP4xTP4.** Use `VLLM_PP_LAYER_PARTITION=18,20,20,20` (stage starts 0/18/38/58, all indexer layers) or the SGLang Ascend partition `18,20,24,16`. TP4xPP4 was the measured winner on 2x8 H100 for 30k-in agentic-shaped traffic in SGLang: +45% total tok/s over TP8xPP2. TP16 is only a sanity reference: 16x head padding, 2 cross-node all-reduces per layer, and the smallest KV pool.

3. **KV dtype A/B on the chosen layout.**
   - `fp8` (fp8_ds_mla, FLASHMLA_SPARSE): ~1.75x KV capacity versus bf16 (656 vs 1,152 B per token per layer), with head padding.
   - `auto` (bf16, FLASH_ATTN_MLA_SPARSE/FA3): no padding. PR #46189 shows up to 1.7x lower TPOT at small batch, but half the capacity.
   - For SWE traffic with many resident long sessions (prefix reuse needs KV to survive tool-execution gaps), capacity probably dominates, so start with fp8.
   - Never use `fp8_e4m3` on Hopper.

4. **vLLM version.**
   - 0.29 env: no FlashMLA workspace regression, but it lacks the GLM-5.3 performance PRs and the #56994 parser fix.
   - 0.32.0, which the main agent is building: has those, but you must apply the #59305 guard (`--max-num-batched-tokens ≤8192`, `--gpu-memory-utilization ≤0.90`).
   - Neither has #59961 or #62424, so keep the default tokenizer mode and the model-dir `chat_template.jinja` (ba32f8c).
   - If PP+MTP misbehaves, 0.33.0 (#62129, #64458) or a main build ≥63d3d39c (adds #59961) are the next candidates.

5. **MTP.**
   - Start at `num_speculative_tokens` 3. This is the HSG stable ceiling and what llm-d and the GLM-5/5.1 recipes use.
   - Sweep 1/3/5 under the agentic load and read acceptance from `/metrics`.
   - Do not judge MTP on random-token benchmarks (GLM-5.1 H200 gave AL 1.64).
   - At high concurrency MTP gains shrink (H20 MTP1: 1.56x at concurrency 1, 1.24x at 64). SGLang's throughput cells use 1 draft token.

6. **Chunking and batch limits.**
   - `--max-num-batched-tokens 8192`. This matches the HSG +19% lesson, the #59305 workspace bound and #57878. 3,072 was optimal in SGLang on H100 for 30k inputs, so sweep 4096/8192.
   - Bound `--max-num-seqs` by KV capacity. Read "Maximum concurrency for 262,144 tokens" from the startup log. SGLang's rule: max-running roughly 2x concurrency.

7. **Client and harness settings.**
   - Keep `reasoning_effort` at the default (max) for parity with Z.ai's DeepSWE numbers, and `clear_thinking=false`.
   - The harness must send back `reasoning` or `reasoning_content` on assistant turns. Otherwise the history differs from what was generated, losing model fidelity and the prefix-cache hit on the last turn.
   - Keep the system prompt and tool list byte-stable within a session: tools render before system, and the effort line is first.
   - Sampling: generation_config `temperature=1.0, top_p=0.95`. Z.ai's DeepSWE run used `temperature=0.95, top_p=1.0` (gpustack warns that thinking can loop without top_p).
   - Never send `enable_thinking=false`.

8. **Scale-out.** Run multiple 2-node replicas behind a session-affinity / prefix-cache-aware router (HSG vllm-router `cache_aware`, Miles key-based affinity, llm-d precise prefix routing). Evaluate CPU KV offload: llm-d measured +13% output and −32% TTFT at concurrency 32 on H200. dfw nodes have 2 TB RAM and 1 TB /dev/shm. Whether OffloadingConnector is in the 0.29/0.32 env is UNVERIFIED.

9. **Correctness gates for every layout.** PP corruption is silent: fluent output, +16% NLL. Run each of these against a reference before load tests:
   - teacher-forced NLL on a fixed prompt set, or GSM8K 5-shot (~0.95 expected);
   - an exact-copy test;
   - tool-call round-trip checks on `swe-task` samples.

   Possible references are the HSG endpoint or a TP16 run.

10. **Fallbacks.**
    - SGLang has the most Hopper multi-node DSA evidence: 16x H800 DSv3.2, and GLM-5 FP8 on 2x8 H100. Use `--nnodes 2 --tp 16 --attn-dp-size 2 …` or `--tp 4 --pp-size 4`; the IndexCache-under-PP behaviour there is UNVERIFIED.
    - Single-node 8x H100 is possible only with `gpustack/GLM-5.3-W4A8` (TP8+EP, fp8_ds_mla; not the official weights).

---

## 10. Open questions / UNVERIFIED

- Does `VLLM_PP_LAYER_PARTITION=38,40` fully restore accuracy without #59961? The PR table shows 0.882 aligned vs 0.908 fixed (with DCP2), yet the author says "restored accuracy". Measure NLL on dfw.
- Do multi-node mp-backend PP and MTP work together for GLM-5.3 on 0.29 and 0.32? The PP + spec-decode fixes (#62129, #64458) landed only in 0.33.0. MTP+PP is not listed in the 0.29 config checks (see the HSG digest §7.3).
- Which sparse backend does 0.32.0 pick on SM90 with `--kv-cache-dtype fp8`: FLASHMLA_SPARSE or FLASHINFER_MLA_SPARSE_SM90 (which accepts fp8)? Check the startup log line "Using … backend".
- FA3 sparse (`FLASH_ATTN_MLA_SPARSE`) at h_q=4 (TP16) or 16 (TP4): does it work, and how fast is it? PR #46189 cites 8/16/32/64.
- DCP with fp8_ds_mla for FP8 GLM-5.3 on H100: the reports conflict (§7, last row).
- Whether SGLang PP carries the IndexCache top-k across stages.
- Whether the arXiv 2x8 H100 ranking (TP4xPP4 > TP2xPP8 > TP8xPP2) transfers to vLLM and to real prefix-heavy traffic. Their benchmark was random 30k-in with no reuse and no MTP.
- `--disable-flashinfer-autotune` (newer YAML spelling) vs `--no-enable-flashinfer-autotune` (0.29 boolean flag): use the latter on 0.29.
- No source gives H100 GLM-5.3 KV capacity. Estimate: ~55 KB/token with fp8 KV at TP-only layouts, from the vLLM H200 note (23 GiB → ~450K tokens). Confirm from the startup log.

---

## Appendix A: local copies (under `/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/notes/external-recipes/dl.2cEBnI/`)

- `vllm-project_vllm_main_recipes_models_zai-org_GLM-5.3.yaml` (= vllm @ e740bad), `vllm-project_recipes_main_models_zai-org_GLM-5.3.yaml` (unrendered copy).
- `vllm-sparse/` (vllm main 52e8596, sparse `recipes/` plus commit-only history for tags v0.29.0 / v0.32.0 / v0.33.0 / main). Example: `git log --oneline v0.29.0..v0.33.0`.
- `recipes-repo/` (vllm-project/recipes @ 2ee0cd8): `GLM/GLM5.md`, `DeepSeek/DeepSeek-V3_2*.md`, `strategies/`, `SCHEMA.md`, `taxonomy.yaml`.
- `api/` (recipes.vllm.ai JSON: GLM-5.3 index, `hw/h100`, `hw/h200`, H100 strategies, W4A8 H100, DeepSeek-V3.2 H100).
- `sglang/` (GLM-5.3 cookbook md, mdx and jsx configs/benchmarks/agentx; GLM-5/5.1/5.2 cookbooks plus generators; DeepSeek-V3.2 cookbook).
- `tokenspeed/models.txt`, `glm5-repo/` (zai-org/GLM-5 full history), `arxiv/paper.txt` (2607.02518), `arxiv/miles.txt` (2609.08368).
- `hf/` (GLM-5.3 README, config, generation_config, tokenizer_config at ba32f8c, all identical to HSG; `ct_7cda819/3033303/aca966e/c5087a7.jinja`), `chat_template_ba32f8c.jinja`.
- `vllmdocs/` (attention backends, EP deployment, DP deployment, parallelism scaling as html+txt).
- Scripts: `../tools/render_template_test.py` (renders both templates on a sample agent conversation; run with `python3 -I` from a scratch dir), `../tools/html2txt.py`.
