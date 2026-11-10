# GLM-5.3 on H100 (SM90) with vLLM 0.29.0 / 0.32.0 / 0.33.0: source-level digest

Written 2026-11-06 by the `vllm-sm90` subagent. Everything below comes from reading Python source and config files. No GPU code was run.
Items marked **UNVERIFIED** could not be confirmed from source alone.

## Sources and citation convention

| tag | what | where |
|---|---|---|
| `[029]` | installed vLLM **0.29.0 g98dff2a81 + NeMo-RL patches** (the production RL env, read-only) | `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/runtime/envs/vllm/lib/python3.13/site-packages/` |
| `[029s]` | stock 0.29.0 wheel, used only to diff out the RL patches | `notes/vllm-sm90/wheels.gYkDVn/v029/vllm-0.29.0-cp38-abi3-manylinux_2_28_x86_64.whl` |
| `[032]` | stock 0.32.0 wheel (commit g5df63a720) | `notes/vllm-sm90/wheels.gYkDVn/v032/vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl` |
| `[033]` | stock 0.33.0 wheel (commit g4ba787ec9) | `notes/vllm-sm90/wheels.gYkDVn/v033/vllm-0.33.0-cp38-abi3-manylinux_2_28_x86_64.whl` |
| `[DG25]` | top-level `deep_gemm` 2.5.0+local (DeepGEMM commit 67fc648) installed in the RL env | `[029]` + `deep_gemm/` |

Paths such as `[032] vllm/platforms/cuda.py:137` are relative to the wheel or site-packages root, followed by the line number.
To re-extract a wheel: `unzip -q <whl> -x '*.so' -d $(mktemp -d)`. The 3,100-5,400 source files were unpacked into a private `/tmp` dir that is not kept, so the wheels are the durable copy.

Helper scripts are in `notes/vllm-sm90/tools/`. All are stdlib-only; run them with `python3 -I`:

- `weight_bytes.py`: exact weight bytes per category. It reproduces the index `total_size` byte-for-byte.
- `mem_math.py`, `mem_table_md.py`, `pp_balance.py` and `dcp_calc.py`: the capacity tables.
- `mem_math_out.txt` and `mem_tables.md`: the raw outputs.

Wheel dependency pins (METADATA `Requires-Dist`):

| | torch | flashinfer-python | vendored DeepGEMM | CUDA |
|---|---|---|---|---|
| 0.29.0 | 2.13.0 | 0.6.18 | 2.6.1 (`vllm/third_party/deep_gemm/__init__.py:129`). The RL env overrides it with top-level `deep_gemm` 2.5.0, see section 3. | cu13 (`_C_stable_libtorch` NEEDED `libcudart.so.13`) |
| 0.32.0 | 2.13.0 | 0.7.0.post1 | 2.8.0 (`[032] vllm/third_party/deep_gemm/__init__.py:115`) | cu13 |
| 0.33.0 | **2.14.0** | 0.7.2.post1 | **2.8.1** (`[033] .../deep_gemm/__init__.py:130`) | cu13 only. The 0.33 release notes drop CUDA 12.9 artifacts (#62575). |

All three need the existing cuda-compat (forward-compatibility library) approach on the dfw 535 driver. That is the same container the RL env runs in.
All three wheels ship the Rust frontend binary `vllm/vllm-rs`, enabled with `VLLM_USE_RUST_FRONTEND=1` (`[029] vllm/envs.py:1398`).

---------------------------------------------------------------------------------------------------

## 0. TL;DR

1. **The model path is the same in all three versions.** `GlmMoeDsaForCausalLM` maps to `vllm.models.deepseek_v32`, which on CUDA is `nvidia.model.DeepseekV32ForCausalLM`. It subclasses `DeepseekV2ForCausalLM`, which declares `SupportsPP`, MixtureOfExperts (so EP and EPLB) and `SupportsEagle`.
   MTP builds `DeepseekV32MTP`. IndexCache is native:
   - only 21 main-model layers build an indexer (layers 0,1,2,6,10,…,74), plus 1 for the MTP layer;
   - shared layers reuse the top-k buffer of the last full layer;
   - draft step 0 computes the MTP layer's own top-k, and steps 1+ reuse it (`index_share_for_mtp_iteration`).
2. **The sparse-MLA backend on SM90 is set by `--kv-cache-dtype`:**

   | kv dtype | backend | bytes/token/layer |
   |---|---|---|
   | `auto`/`bfloat16` | `FLASH_ATTN_MLA_SPARSE` (FA3) | 1152 |
   | `fp8`/`fp8_ds_mla` | `FLASHMLA_SPARSE`, block-scaled `fp8_ds_mla` | 656 |
   | `fp8_e4m3` | 0.32/0.33: `FLASHINFER_MLA_SPARSE_SM90` | 576 |

   **`fp8_e4m3` (HSG's choice) has no valid backend on SM90 in 0.29, so startup fails.** On 0.29 use `fp8` instead.
   All SM90 sparse backends force KV block size 64.
   FlashMLA sparse pads query heads to 64 for decode and for its BF16 prefill kernel. With TP8 (8 heads per rank) that is 8x wasted attention FLOPs; with TP1 (DP attention, 64 heads) there is no waste.
3. **Per-token KV.** 78 MLA layers + 21 indexer layers (132 B each); add 1 MLA + 1 indexer layer when MTP is on.
   With `fp8_e4m3` + MTP this is **48,408 B/token per GPU**, which matches HSG's measured 39.3 GiB / 872k tokens = 48.4 kB/token.
   MLA KV is **replicated on every TP rank** unless DCP is used.
4. **Indexer kernels.** DeepGEMM is required on CUDA:
   - prefill: `fp8_fp4_mqa_logits`;
   - decode: `fp8_fp4_paged_mqa_logits` + `get_paged_mqa_logits_metadata`;
   - both have SM90 kernels in every build.

   On SM90 the paged kernel natively takes next_n ∈ {1,2,4}, so MTP 1 or 3 are the efficient settings; 2, 4 and 5 fall back to flattened rows.
   vLLM imports a **top-level `deep_gemm` before the vendored copy**.
   **HSG's `clean_logits` error is a version mismatch.** vLLM 0.32 Python passes `clean_logits=`, which DeepGEMM 2.8.0 accepts. The 0.33-era DeepGEMM 2.8.1 removed `clean_logits` and made `max_seqlen_k` mandatory. HSG ran 0.32 Python on top of nightly (0.33-era) binaries.
   DeepGEMM JIT needs nvcc (`CUDA_HOME`). Its cache, `DG_JIT_CACHE_DIR`, defaults to `~/.cache/vllm/deep_gemm`, which sits on the 10 GB /home quota, so redirect it.
5. **MoE on Hopper with block FP8.**
   - TP only: `TRITON` (fused_moe).
   - EP>1 with a "standard" all2all (`allgather_reducescatter`, `deepep_high_throughput`, `deepep_v2`): `FLASHINFER_CUTLASS`.
   - EP with a "batched" all2all (`deepep_low_latency`, `nixl_ep`): `BATCHED_DEEPGEMM`.
   - Force DeepGEMM with `--moe-backend deep_gemm` or by explicitly setting `VLLM_USE_DEEP_GEMM=1`.

   `VLLM_USE_DEEP_GEMM_E8M0=1` is the default and **also requantizes weight scales to UE8M0 (powers of two) on Hopper** whenever DeepGEMM is used.
6. **all2all on x86 + InfiniBand.**
   - `allgather_reducescatter` (NCCL, the default) works anywhere.
   - DeepEP (HT and LL) needs `deep_ep` + NVSHMEM. DeepEP-LL internode and DeepEPv2 (NCCL GIN) need IBGDA-capable NICs and drivers, **UNVERIFIED on dfw**.
   - `nixl_ep` needs the `nixl_ep` package.
   - The flashinfer_nvlink_* backends need MNNVL, so on H100 they work inside one node only.
   - `pplx` and `naive` were removed and silently become AG-RS.

   **DBO** needs a deepep_* or nixl_ep all2all. On Model Runner V2 it is experimental and **not allowed together with MTP**, so a DBO+MTP config falls back to Model Runner V1.
7. **Memory.**
   - Weights total 755.6 GB: routed experts are 734.6 GB (97.2%) and everything else is 21.0 GB.
   - DP-attention replicates about 23 GB of non-expert weights per GPU (non-expert 21.0 GB + 2.3 GB of absorbed MLA weights). That makes **DP16xEP16 on 2 H100 nodes infeasible**: 64.5 GiB of weights per GPU leaves at most about 35k KV tokens per GPU.
   - **Best non-experimental 2-node layout: TP8xPP2** (`VLLM_PP_LAYER_PARTITION=41,37`), about **0.90M KV tokens per engine** at u=0.93 with `fp8_e4m3`. For comparison, TP16 gives 0.45M and DP2xTP8+EP16 gives 0.72M.
   - On 4 nodes, DP32xEP32 reaches about 16M tokens but has the EP-lockstep latency problem HSG measured.
   - TP8 + DCP8 (decode context parallel) would multiply TP-layout capacity by up to 8 (about 6M tokens on 2 nodes), but it is **UNVERIFIED** on SM90.
8. **Version recommendation:** build a fresh 0.32.0 env first, for parity with HSG's tuned GLM-5.3 stack. A/B it against 0.33.0. Use the RL 0.29 env only for a read-only fallback smoke test. Details in section 11.

---------------------------------------------------------------------------------------------------

## 1. Model class, PP / EP / MTP, IndexCache

| item | 0.29 `[029]` | 0.32 `[032]` | 0.33 `[033]` |
|---|---|---|---|
| registry entry `GlmMoeDsaForCausalLM -> ("vllm.models.deepseek_v32", "GlmMoeDsaForCausalLM")` | `vllm/model_executor/models/registry.py:122` | `:120` | `:126` |
| CUDA alias `from .nvidia.model import DeepseekV32ForCausalLM as GlmMoeDsaForCausalLM` | `vllm/models/deepseek_v32/__init__.py:17-21` | `:17-21` (ROCm has its own `.amd`, L23-26) | `:17-21` |
| `class DeepseekV32ForCausalLM(DeepseekV2ForCausalLM)` | `vllm/models/deepseek_v32/nvidia/model.py:415` | `:427` | `:432` |
| base class mixins: `SupportsPP`, `DeepseekV2MixtureOfExperts` (EPLB), `SupportsLoRA`, `SupportsEagle(3)` | `vllm/model_executor/models/deepseek_v2.py:1848-1854` | `:1872-1878` | `:1872-1878` |
| PP plumbing (`make_layers`, `PPMissingLayer`, `IntermediateTensors`) | `nvidia/model.py:192-221, 284-290` | same structure | same |
| SP-MoE decoder path (on when TP>1 & DP>1 & EP) | `nvidia/model.py:73-76, 178-181` | `:78, :184` | `:78, :184` |
| config hook `GlmMoeDsaForCausalLM.verify_and_update_config` | `vllm/model_executor/models/config.py:43-50`: only sets DCP defaults (`comm_backend="a2a", q_replicate=True`) | `:43-70`: same, **plus** `auto` becomes `fp8_e4m3` **only on SM10x** (L49-55) | `:89-116`: same as 0.32 |

**PP: supported.**
- Layers are split evenly by `get_pp_indices`, which `VLLM_PP_LAYER_PARTITION` overrides (`[032] vllm/distributed/utils.py:126-156`).
- The 78 layers split 39/39 by default. The last stage also holds lm_head and the MTP layer, so `41,37` balances KV and weights better (section 9).
- SP-MoE asserts `not use_sequence_parallel` under PP (`[029] nvidia/model.py:270, 285-287`). The SP path is skipped automatically when PP>1.

**EP: supported** (`--enable-expert-parallel`).
- Experts are mapped through `fused_moe_make_expert_params_mapping(... num_redundant_experts ...)` (`[029] nvidia/model.py:325-332`).
- 256 experts: HSG saw EP24 change outputs (`hsg-ref/notes/NOTES.md:348-352`). Use EP sizes that divide 256.

**MTP.**
- `--speculative-config '{"method":"mtp",...}'`.
- `hf_config_override` maps `glm_moe_dsa` to `deepseek_mtp` with `architectures=["DeepseekV32MTPModel"]`: `[029] vllm/config/speculative.py:645-676`, `[032] :663-694`, `[033] :733-764`.
- `DeepseekV32MTPModel` resolves to `vllm.models.deepseek_v32.DeepseekV32MTP` (`[029] registry.py:651`, `[033] :716`), implemented in `nvidia/mtp.py`:
  - `[029]`: L58-90 layer, L160-176 predictor, L263-273 model;
  - `[032]`/`[033]`: L199-200 is `set_skip_topk`.
- The MTP layer is a full DSA decoder layer: MLA, its own indexer (always built for MTP layers, `attention.py` condition `if not skip_topk or is_mtp_layer`, `[029] :185`, `[032]/[033] :190`), and an MoE block with 256 routed experts. Its experts are EP-sharded like the main model's.
- `eh_proj` is a plain `nn.Linear` (BF16 and replicated, 151 MB; `[029] nvidia/mtp.py:68`).
- n_predict = 1, so any `num_speculative_tokens` is legal (1 divides everything). A count above 1 re-runs the same MTP layer and logs a warning (`[029] speculative.py:1314-1323`).

**IndexCache.**
- `skip_topk = max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq != 0`, with offset 3 and freq 4 for GLM-5.3 (`[029] vllm/models/deepseek_v32/attention.py:161-172`; `[032]/[033] :166-177`).
- That gives indexers at layers 0,1,2,6,10,…,74: 21 layers, identical to `config.json` `indexer_types` "full".
- The MTP layer always gets one (`is_mtp_layer`, `[029] :174-175, 185`).
- All layers share a single `topk_indices_buffer [max_num_batched_tokens, 2048] int32` (`[029] nvidia/model.py:182-190`). A shared layer just reads the buffer left by the previous full layer.
- MTP uses the same buffer: `[029] vllm/v1/spec_decode/llm_base_proposer.py:1600-1614` shares the target's buffer with the drafter.

**`index_share_for_mtp_iteration` (MTP x IndexCache):**
- **Model Runner V2:** `vllm/v1/worker/gpu/spec_decode/mtp/speculator.py`, `[029] :27-58`, `[032] :27-59`, `[033] :27-59`. Step 0 runs the MTP layer's indexer (`set_skip_topk(False)`). It then compacts each request's top-k row to the front (`compact_topk_indices`), and draft steps 1+ skip the indexer (`set_skip_topk(True)`). In 0.32+ this is disabled under PCP (L31).
- **Model Runner V1:** `[029] llm_base_proposer.py:576-611, 1618-1627`; `[032] :578-613, 1592-1602`.
- The 0.32 release notes list "MTP index sharing restored (#61079)". **UNVERIFIED** which runner or version was broken.

**Embedding and lm_head sharing for MTP.** The checkpoint's MTP layer 78 has no `embed_tokens` or `shared_head.head` of its own.
- With PP=1 the target embedding and lm_head are shared: V1 `[029] llm_base_proposer.py:1438-1580`, V2 `[032] vllm/v1/worker/gpu/spec_decode/eagle/utils.py:42-74`.
- **With PP>1 the drafter keeps its own embedding copy** (`[032] eagle/utils.py:56-58`), about 1.9 GB / TP.

**RL-env patch to know about.** `[029] vllm/model_executor/models/deepseek_v2.py:1289-1294` disables SP-MoE for `glm_moe_dsa` ("vLLM 0.25.1's decoder-level SP-MoE path corrupts iterative decoding for GLM-5.1/5.2").
That patch lives in the **generic** `DeepseekV2DecoderLayer`. **The CUDA path GLM actually uses, `nvidia/model.py:73-76`, is not patched**, and stock 0.32/0.33 have no such exclusion.
So any DP>1 & TP>1 & EP layout (DP2xTP8+EP16, DP4xTP4+EP16, DP4xTP8+EP32…) runs the SP-MoE path. **UNVERIFIED whether that is correct for GLM-5.3: run a greedy parity check against a TP-only layout before benchmarking.**
The other RL patches (routed-expert capture, `shm_broadcast` port-retry on non-driver nodes, Ray `py_executable`, sampler and logprobs) do not change GLM serving semantics.

---------------------------------------------------------------------------------------------------

## 2. Attention backends for DSA / sparse MLA on SM90

### 2.1 Selection order (CUDA platform, `use_mla=True`, `use_sparse=True`)

- `[029] vllm/platforms/cuda.py:135-143`: SM9x list `[FLASH_ATTN_MLA, FLASHMLA, FLASHINFER_MLA, TRITON_MLA, FLASH_ATTN_MLA_SPARSE, FLASHMLA_SPARSE]`.
  The dense ones are rejected (`use_sparse != is_sparse()`, `vllm/v1/attention/backend.py` `validate_configuration`). The first valid sparse backend wins (`cuda.py:437-502`).
- `[032] cuda.py:137-153`, `[033] cuda.py:139-155`: `sparse_tail = [FLASH_ATTN_MLA_SPARSE, FLASHMLA_SPARSE]`, and `FLASHINFER_MLA_SPARSE_SM90` is **appended** (it is moved to the front only when head_size == 512, the NoPE GLM-5.3-Flash).
  GLM-5.3's head_size is 576 (512 latent + 64 rope), so the order is FA-sparse, then FlashMLA-sparse, then FlashInfer-SM90.
- `kv_cache_dtype` support is an **exact string match** (`backend.py` `supports_kv_cache_dtype`, `[029] :109-114`, `[033] :114-119`).
- After selection, `FLASHMLA_SPARSE` + any quantized dtype is canonicalized to `fp8_ds_mla`: `_canonicalize_sparse_mla_kv_cache_dtype`, `[029] vllm/model_executor/layers/attention/mla_attention.py:350-362`, `[032] :358-375`, `[033] :359-375`.

| `--kv-cache-dtype` | 0.29 on SM90 | 0.32 / 0.33 on SM90 |
|---|---|---|
| `auto` / `bfloat16` | FLASH_ATTN_MLA_SPARSE | FLASH_ATTN_MLA_SPARSE |
| `fp8` | FLASHMLA_SPARSE, which becomes `fp8_ds_mla` | FLASHMLA_SPARSE, which becomes `fp8_ds_mla` |
| `fp8_ds_mla` | FLASHMLA_SPARSE | FLASHMLA_SPARSE |
| `fp8_e4m3` | **no valid backend → `ValueError`** (FA-sparse is BF16-only; FlashMLA-sparse lists `fp8`/`fp8_ds_mla`, not `fp8_e4m3`; `FLASHINFER_MLA_SPARSE` is SM100-only) | **FLASHINFER_MLA_SPARSE_SM90** (per-tensor `k_scale`, 1.0 when uncalibrated) |
| `nvfp4_ds_mla` | n/a | SM100 only |

So **`fp8_e4m3` is legal on SM90 only from 0.32** (release note "FP8 KV on SM90 sparse MLA (#55222)").
`fp8_ds_mla` is the DeepSeek block-scaled format: 512 FP8 values, 4 FP32 scales, and the 64 rope dims kept in BF16. It is **more accurate than per-tensor `fp8_e4m3` with an uncalibrated scale of 1.0**, at about 14% more KV bytes.

### 2.2 Backend constraints

| backend | file and lines | KV dtypes | block size | heads / misc | CC |
|---|---|---|---|---|---|
| FLASH_ATTN_MLA_SPARSE (FA3, `flash_attn_varlen_func(..., q_v=..., fa_version=3)`, top-k slots used as a page_size=1 block table) | `[029] vllm/v1/attention/backends/mla/flashattn_mla_sparse.py:36-44, 71-105, 236-249`; `[032] :38-46, 78, 95, 106`; `[033] :43-51, 83, 100, 111` | auto / float16 / bfloat16 only | `[64]` (0.29), `MultipleOf(64)` (0.32+) | **no DCP** ("does not support DCP for now"); `supports_quant_query_input=False` | major == 9 |
| FLASHMLA_SPARSE (FlashMLA; FP8 decode kernel plus BF16 prefill kernel) | `[029] flashmla_sparse.py:63, 90-130, 300-330`; `[032] :75, 149-204, 717-760`; `[033] :74, 149-204, 711-754` | auto, bfloat16, `fp8_ds_mla`, `fp8` (alias), `nvfp4_ds_mla` (SM100 only) | `[64]` / 64-row pages on SM90 | head_size 576 (+512 in 0.32+). **FP8 decode kernel pads h_q to 64 or 128** (`_compute_fp8_decode_padded_heads`, `[032] :721-724`). **BF16 prefill kernel pads to 64 heads on Hopper** (`self.prefill_padding = 64`, `[032] :759-760`). Below 32 heads per rank (`MIN_HEADS_FOR_BF16_PREFILL`) it switches to "mixed batch": prefill also runs through the FP8 decode kernel. `supports_dcp=True` (`[032] :717`). | major in [9,10] |
| FLASHINFER_MLA_SPARSE_SM90 (0.32+; FlashInfer `BatchMLAPagedAttentionWrapper(backend="fa3")`, page_size=1, `causal=False`) | `[032] flashinfer_mla_sparse_sm90.py:146-224, 237-420, 552-646`; `[033]` same file | auto, bfloat16, `fp8`, `fp8_e4m3` (in-kernel dequant, BF16 queries) | `MultipleOf(64)` | head_size 512/576; requires `kv_lora_rank==512` and `qk_rope_head_dim ∈ {0,64}` (L218-223); needs FlashInfer with `ckv_scale_arr` (`vllm/utils/flashinfer.py:415-437`, FlashInfer ≥0.6.18); 128 MiB workspace; plan() runs on the host outside graph capture; DCP not supported (default `supports_dcp=False`, `backend.py:822`) | major == 9 |
| DeepSeek V3.2 indexer K cache (`DeepseekV32IndexerBackend`) | `[029] vllm/v1/attention/backends/mla/indexer.py:152-181`; `[032] :284-285`; `[033] :285-286` | uint8: 128 FP8 + 4 B FP32 scale | **`[64]`** on CUDA | n/a | n/a |

**Block size.** Each backend's preferred block size is applied unless `--block-size` is given (`[029] vllm/platforms/interface.py:609-640`).
All SM90 sparse MLA backends and the indexer accept only 64 (or multiples in 0.32+). So the block is 64, and **`--prefix-match-unit` must divide 64** (HSG note: "DSA backends force block size 64").

**Heads per rank.** `assert num_heads % tp_size == 0` (`[029] attention.py:141-143`), so TP ∈ {1,2,4,8,16,32,64}.
- 64 heads/TP: TP1 = 64 (no padding), TP2 = 32, TP4 = 16, TP8 = 8, TP16 = 4.
- With FLASHMLA_SPARSE: decode always computes 64 query heads, prefill BF16 always 64 (mixed-batch FP8 below 32 heads). **TP8 does 8x and TP16 16x the attention math actually needed.**
- FA3-sparse and FlashInfer-SM90 avoid explicit padding. **UNVERIFIED** how efficient they are at 4-8 heads.
- The indexer has 32 heads (`index_n_heads`). It is **not TP-sharded**: `wq_b` is a `ReplicatedLinear` (`[029] attention.py:68-74`), so every TP rank computes the full indexer.

### 2.3 Prefill vs decode paths (inside `DeepseekV32Attention.forward` and `_sparse_indexer_and_attn`, `[029] attention.py:278-577`)

**Prefill.**
- When `prefill_max_seq_len <= index_topk (2048)` the layer takes the **dense MHA** path (`_use_dense_mha_prefill`, `[032] vllm/model_executor/layers/attention/sparse_mla_attention.py:130-138`; `_use_sparse_mha`, `[032] mla_attention.py:1193-1218`). It skips indexer scoring (`enable_short_prefill_scoring_skip`, `[029] attention.py:220-230`).
- Otherwise the indexer runs:
  1. `ops.cp_gather_indexer_k_quant_cache` gathers K into a workspace;
  2. DeepGEMM `fp8_fp4_mqa_logits` computes logits;
  3. `ops.top_k_per_row_prefill` picks the top-k.

  These are chunked by `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` (default 512) and by the gather workspace.
- Sparse MQA attention then runs over the top-2048 via the backend's `forward_mqa`. FlashMLA-sparse uses its BF16 prefill kernel (FP8 cache upconverted into a workspace), or mixed-batch FP8 below 32 heads.
- `--attention-config.sparse_mla_force_mqa` (0.32+, `[032] vllm/config/attention.py:115-118`) forces MQA for everything.

**Decode.**
- The indexer runs `fp8_fp4_paged_mqa_logits`, then top-k:
  - `cooperative_topk` when ≤ 64 rows and SM ≥ 90 (not SM120);
  - else `persistent_topk`;
  - else `top_k_per_row_decode`.

  See `[029] vllm/model_executor/layers/sparse_attn_indexer.py:530-689`. 0.32+ routes this through `get_indexer_topk(topk_backend)` (`[032] vllm/model_executor/layers/indexer_topk.py:137-200`); DeepSelect is SM100-only.
- The backend's sparse decode kernel then runs.
- `_sparse_indexer_and_attn` is decorated `@eager_break_during_capture`. That only matters with `VLLM_USE_BREAKABLE_CUDAGRAPH=1` (default 0).
- Normal cudagraph support for every SM90 sparse backend and for the indexer (native next_n) is `UNIFORM_BATCH`:
  - `[029] flashattn_mla_sparse.py:135`, `flashmla_sparse.py:223`, `indexer.py:522-529`;
  - `[032] flashmla_sparse.py:350`, `flashinfer_mla_sparse.py:258`.

  So uniform decode and MTP-verify batches get FULL graphs, and mixed batches run piecewise.

### 2.4 Per-token KV bytes

The MLA spec page is `num_heads(1) × block_size × state_content_size_bytes`. For MLA that is `head_size (576) × dtype size`, except `fp8_ds_mla`, which is fixed at 656 B (`[029] mla_attention.py:1218-1220`; `[032] :1365-1367`; `[033] :1386-1388`).
The indexer cache is an `MLAAttentionSpec(head_size=132, dtype=uint8)` (`[029] attention.py:92-97`, `deepseek_v2.py:653-659`).
All layers live in one `UniformTypeKVCacheSpecs` group whose page size is the sum over layers (`[029] vllm/v1/kv_cache_interface.py:997-1025`). One block id covers all 79+22 layer pages, and alignment padding is negligible at block 64.

| kv dtype (SM90 backend) | MLA B/token/layer | indexer B/token/layer | per token, 78 MLA + 21 idx (no MTP) | per token, 79 + 22 (MTP on) |
|---|---|---|---|---|
| `auto`/bf16 (FA-sparse) | 1152 | 132 | 92,628 | **93,912** |
| `fp8_e4m3` (FlashInfer-SM90, 0.32+) | 576 | 132 | 47,700 | **48,408** |
| `fp8`/`fp8_ds_mla` (FlashMLA-sparse) | 656 | 132 | 53,940 | **54,728** |

- Under TP every rank stores these bytes for **all** tokens (MLA latent replicated).
- Under PP each stage stores only its layers:

  | partition | stage | MLA layers | indexer layers | `fp8_e4m3` B/token |
  |---|---|---|---|---|
  | 39/39 | stage0 (layers 0-38) | 39 | 12 | 24,048 |
  | 39/39 | stage1 | 39 + MTP | 9 + MTP | 24,360 |
  | 41/37 | stage0 (layers 0-40) | 41 | 12 | 25,200 |
  | 41/37 | stage1 (layers 41-77) | 37 + MTP | 9 + MTP | 23,208 |

- Under DCP=k the bytes divide by k.
- **Check against HSG:** the E13 prefill tier stored 39.3 GiB for 872k tokens, about 48,392 B/token, which matches 48,408 (`fp8_e4m3` + MTP).

---------------------------------------------------------------------------------------------------

## 3. Indexer kernels and DeepGEMM

**Calls.** All of these are wrappers in `vllm/utils/deep_gemm.py`, and they hit DeepGEMM symbols resolved by getattr.

| | quant + cache insert | prefill gather | prefill logits | prefill top-k | decode logits | requires DeepGEMM on CUDA |
|---|---|---|---|---|---|---|
| `[029]` | `ops.indexer_k_quant_and_cache` L400 | `ops.cp_gather_indexer_k_quant_cache` L456 | `fp8_fp4_mqa_logits(..., clean_logits=False)` L500-507 | `ops.top_k_per_row_prefill` L509 | `fp8_fp4_paged_mqa_logits(..., clean_logits=False, indices=...)` L603-612 | `SparseAttnIndexer.__init__` raises "requires DeepGEMM" L777-781 |
| `[032]` | L535 | L614 | L678-684 (`clean_logits=False`) | L711 | L825-833 | L1003-1005 |
| `[033]` | L573 | L652 | L728-737: `fp8_fp4_mqa_logits(q, kv, weights, ks, ke, chunk.max_seqlen_k)`, compressed `[M, max_seqlen_k]` logits | L763 | L880 (no `clean_logits`) | L1060-1062 |

(Line numbers in this table refer to `vllm/model_executor/layers/sparse_attn_indexer.py`.)
The decode schedule comes from `get_paged_mqa_logits_metadata` (`[029] deep_gemm.py:578-605`).

**SM90 support.**
- Every DeepGEMM build ships SM90 JIT kernels `impls/sm90_fp8_mqa_logits.cuh` and `impls/sm90_fp8_paged_mqa_logits.cuh`: `[DG25]`, vendored 2.6.1 (`[029s]`), 2.8.0 (`[032]`) and 2.8.1 (`[033]`), all under `deep_gemm/include/deep_gemm/impls/`.
- `native_next_n_supported`: "SM90 implements only {1, 2, 4}" (`[029] deep_gemm.py:555-563`, `[032] :640-648`, `[033] :684-692`).
- For next_n == 4 on SM90 the scheduler uses 2-CTA multicast and vLLM halves the SM count (`_paged_mqa_logits_schedule_slots`, `[029] :566-575`). `[DG25]` has the matching `kNumKVMulticast` template (`sm90_fp8_paged_mqa_logits.cuh:22-45`).
- Unsupported next_n values are flattened to one row per token (`_use_flattening`, `[029] indexer.py:491-512`; `[032] :957-968`; `[033] :958-969`). Flattening re-reads KV per row and switches indexer cudagraph support to `ALWAYS`.
- **So MTP 3 (next_n = 4) or MTP 1 (next_n = 2) are the native options on H100.**

**How deep_gemm is found.** `_import_deep_gemm`, `[029] deep_gemm.py:170-203`, `[032] :183-210`, `[033] :185-217`:
1. `importlib.import_module("deep_gemm")`, a top-level package, is tried **first**;
2. then `vllm.third_party.deep_gemm`, the vendored copy.

Consequences:
- The RL env runs `[DG25]` (2.5.0, JIT, pybind `_C.cpython-313…so`), not the vendored 2.6.1.
- In a 0.33 env, any older top-level deep_gemm is dangerous. Its `fp8_fp4_mqa_logits(q, kv, w, ks, ke, clean_logits=True, max_seqlen_k=0, ...)` signature means 0.33's 6th positional argument (`max_seqlen_k`) silently lands in `clean_logits`.
- **Keep exactly one DeepGEMM matching the vLLM Python**: the vendored copy, or a top-level copy of the same version.

**API history (the HSG `clean_logits` / `max_seqlen_k` incident).**

| | vLLM wrapper | DeepGEMM binding |
|---|---|---|
| 0.29 | `[029] deep_gemm.py:510-552, 608-662` passes `clean_logits` | `[DG25] deep_gemm/_C.pyi:289-320`: `fp8_fp4_mqa_logits(..., clean_logits=True, max_seqlen_k=0, logits_dtype=None)`, `fp8_fp4_paged_mqa_logits(..., clean_logits=False, logits_dtype=None, indices=None)` |
| 0.32 | `[032] deep_gemm.py:594-637, 694-755` passes `clean_logits=` | vendored 2.8.0 `[032] vllm/third_party/deep_gemm/_C.py:162-177` accepts `clean_logits` and an optional `max_seqlen_k=0` |
| 0.33 | `[033] deep_gemm.py:638-681` passes `max_seqlen_k` positionally and documents "compressed logits … tail is not cleaned"; `:738-799` has no `clean_logits` | vendored 2.8.1 `[033] _C.py:180-192`: `fp8_fp4_mqa_logits(q, kv, weights, ks, ke, max_seqlen_k, schedule_meta=None)` and `fp8_fp4_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len, indices=None)` |

The bxyu image was 0.32 Python with nightly (0.33-era) binaries, so vendored DG 2.8.1 rejected `clean_logits` and required `max_seqlen_k` (HSG `NOTES.md:53-62`).
HSG's fix (prepending the stock v0.32.0 `vllm/third_party/deep_gemm` to PYTHONPATH) works precisely because the top-level import wins.

**JIT requirements.**
- `[DG25] deep_gemm/__init__.py:103-124` calls `_C.init(lib_root, _find_cuda_home())`. `_find_cuda_home` uses `CUDA_HOME`, then `CUDA_PATH`, then `which nvcc`, then `/usr/local/cuda`, and asserts that one exists.
- The vendored 2.8.x calls `_C.init(lib_root)` (`[032] .../deep_gemm/__init__.py:113`). Its `_C_extension.abi3.so` still JITs: the binary contains `CUDA_HOME`, `CUDA_PATH`, `bin/nvcc`, `JIT_CACHE_DIR`, `JIT_NVCC_COMPILER`, … The env names are presumably `DG_`-prefixed (**UNVERIFIED** for 2.8.x).
- `[DG25]` strings show `DG_JIT_CACHE_DIR`, `DG_JIT_NVCC_COMPILER`, `DG_JIT_USE_NVRTC`, `DG_JIT_DEBUG`, `DG_JIT_PRINT_COMPILER_COMMAND`, …
- vLLM sets `DG_JIT_CACHE_DIR=$VLLM_CACHE_ROOT/deep_gemm` when unset (`[029] deep_gemm.py:260-265`, `[032] :279-283`, `[033] :281-285`). `VLLM_CACHE_ROOT` defaults to `~/.cache/vllm` (`[032] envs.py:711-715`). **Put both on Lustre or `/raid`**, because /home is a 10 GB quota.
- `VLLM_DEEP_GEMM_WARMUP` defaults to "relax" (`[032] envs.py:1645-1653`).
- The container already provides CUDA 13.2 nvcc at `/usr/local/cuda`, so set `CUDA_HOME=/usr/local/cuda`.
- FlashInfer (FA3 SM90 sparse MLA, all-reduce) also JITs unless `flashinfer-jit-cache` is installed or `vllm download-kernels` is used (0.32 notes #58765, #60052).

**Indexer memory.**
- The prefill K-gather buffer is `max_model_len × 8 × factor` entries of 132 B: `[029] indexer.py:462-472` uses ×40; `[032] :918-929` and `[033] :916-927` use `(576*2//132) × sparse_mla_prefill_workspace_factor`.
- The factor defaults to 5 for `max_model_len ≤ 256K`, else 2 (`[032] vllm/config/vllm.py:1790-1798`, `[033] :1873`; flag `--attention-config.sparse_mla_prefill_workspace_factor`).
- At 262,144 tokens with factor 5 that is 1.38 GB. FlashMLA-sparse's prefill workspace is `factor × max_model_len × 576 × 2 B` = 1.51 GB. Both come from the shared workspace manager.
- The logits reservation is 512 MB.
- Decode logits are `[B × next_n, max_model_len]` fp32. For example 128 seqs × 4 × 262,144 × 4 B ≈ 0.54 GB.

---------------------------------------------------------------------------------------------------

## 4. MoE kernels, all2all, DBO, EPLB

**FP8 block-scaled MoE oracle** (`vllm/model_executor/layers/fused_moe/oracle/fp8.py`).
- The default priority list (AITER, FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, DEEPGEMM, VLLM_CUTLASS, TRITON, MARLIN, HUMMING, BATCHED_*) is reordered on Hopper. For block FP8 the code says "prefer Triton for TP and FI CUTLASS for EP": `[029] :113-123`, `[032] :120-130`, `[033] :122-132`.
- `ep_size > 1` moves **FLASHINFER_CUTLASS** to the front. FlashInfer CUTLASS supports `(kFp8Static128BlockSym, kFp8Dynamic128Sym)` exactly on 9.0 (`fused_moe/experts/flashinfer_cutlass_moe.py`, `[029] :164-172`, `[032]/[033] :168`). Its activation format is Standard.
- TP-only moves **TRITON** to the front (`fused_moe.py` block-FP8 Triton kernel).
  - In-tree H100 tuning exists for `E=256,N=128` (TP16, the same expert shape as DeepSeek-V3) at `fused_moe/configs/E=256,N=128,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json`.
  - For TP8 (`N=256`) there is only an H200 file, so Triton falls back to defaults. Consider tuning, or copying the H200 json (**UNVERIFIED** benefit).
- With batched activation format (`use_batched_activation_format = use_deepep_ll_kernels or use_nixl_ep_kernels`, `fused_moe/config.py` `[029] :1091`, `[032] :1057`, `[033] :1066`), FlashInfer-CUTLASS and standard DeepGEMM are skipped, so the first match is **BATCHED_DEEPGEMM** (masked grouped GEMM, `sm90_fp8_gemm_1d2d`).
- Explicitly setting `VLLM_USE_DEEP_GEMM=1` or `VLLM_MOE_USE_DEEP_GEMM=1` forces DEEPGEMM / BATCHED_DEEPGEMM (`[029] oracle/fp8.py:392-405`, `[032] :480`, `[033] :484`). Both default to 1 but only count when set explicitly.
- `--moe-backend {triton,deep_gemm,cutlass,flashinfer_trtllm,flashinfer_cutlass,marlin,humming,...}` (`[029] arg_utils.py:1682`; mapping `oracle/fp8.py:251-269`).
- TP sharding of `moe_intermediate_size=2048` stays block-aligned for TP ≤ 16. At TP32 the scales are refined to a 64 block (Triton only, `refine_fp8_moe_block_shape`, `[029] oracle/fp8.py:272-302`).

**UE8M0 on Hopper.**
- `is_deep_gemm_e8m0_used()` returns True whenever DeepGEMM is usable and `VLLM_USE_DEEP_GEMM_E8M0=1` (the default; `[029] deep_gemm.py:114-135`, `[032] :120-141`, `[033] :122-143`). There is no architecture check.
- When a DeepGEMM MoE backend is chosen, block-FP8 weight scales are **requantized in place to powers of two** (`requant_weight_ue8m0_inplace`, `[029] quantization/utils/fp8_utils.py:1115`, `[032]/[033] :1214`).
- Activation scales are `FLOAT32_CEIL_UE8M0` on SM90 (`[029] deep_gemm.py:61-83`).
- HSG on GB200 necessarily ran UE8M0. On H100, `VLLM_USE_DEEP_GEMM_E8M0=0` keeps the checkpoint's fp32 scales, which is an accuracy A/B knob (**UNVERIFIED** effect).

**all2all backends.** `All2AllBackend` literal is at `[029] vllm/config/parallel.py:42-55`, `[032] :44-59`, `[033] :45-60`; managers are in `vllm/distributed/device_communicators/cuda_communicator.py` (`[029] :143-204`).

| backend | requirement on x86 + IB (H100) | notes |
|---|---|---|
| `allgather_reducescatter` (default) | NCCL only; works anywhere | Standard format, so FLASHINFER_CUTLASS. MoE input is the all-gathered token set, so the workspace grows with DP × max_num_batched_tokens. |
| `naive`, `pplx` | removed | warning, then falls back to AG-RS (`[029] parallel.py:488`, `[032] :533-538`) |
| `deepep_high_throughput` | `deep_ep` (RL env: 1.2.1+17cfb81) + NVSHMEM (`nvidia_nvshmem_cu13` 3.7.2) for the internode RDMA buffer (`VLLM_DEEPEP_BUFFER_SIZE_MB` = 1024) | `[029] all2all.py:210-268`. Internode NVSHMEM transport on dfw: **UNVERIFIED** (IBGDA or IBRC). |
| `deepep_low_latency` | DeepEP LL internode needs **NVSHMEM IBGDA**. `allow_mnnvl=VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL` (default 0; only meaningful on NVL72). | `[029] all2all.py:271-330`. Batched format, so BATCHED_DEEPGEMM. |
| `deepep_v2` | DeepEP ≥ 2.0 + NCCL ≥ 2.30.4 with **NCCL GIN** ("usually means IBGDA-capable InfiniBand NICs or drivers", `[029] all2all.py:1054-1062`, `[032] :1163`, `[033] :1097`) | Standard format. The RL env's deep_ep is 1.x. |
| `nixl_ep` | `nixl_ep` package (present in the RL env: `nixl_ep`, `nixl_ep_cu13`) | `[029] all2all.py:384-440`. Batched format. Supports elastic EP. |
| `flashinfer_nvlink_one_sided` / `flashinfer_nvlink_two_sided` (`flashinfer_all2allv`) | **MNNVL fabric**, so on H100 only EP within one node | HSG's `glm_5.3.sh` default (NVL72). Not usable for EP16 across H100 nodes. |
| `mori_*` | ROCm | n/a |
| `moonep` (0.32+) | NVLink | n/a |
| `passthrough` (0.32+) | bound automatically for MoE-EP megakernels; not CLI-selectable | n/a |

**SP-MoE.** `use_sequence_parallel_moe` is true for AG-RS / deepep_* / nixl_ep / flashinfer one-sided with EP, TP>1 and DP>1 (`[029] parallel.py:704-719`, `[032] :764-781`, `[033] :768-785`). Under it, the dense MLP and shared expert are **replicated** (`DeepseekV2MLP(..., disable_tp=is_sequence_parallel)`, `[032] vllm/model_executor/models/deepseek_v2.py:244-275`, with `disable_tp` at L266 and L275). See the correctness caveat in section 1.

**DBO (`--enable-dbo`, `--dbo-decode-token-threshold` 32, `--dbo-prefill-token-threshold` 512).** Fields are at `[029] parallel.py:218-228`.
- On CUDA it requires all2all ∈ {deepep_low_latency, deepep_high_throughput, nixl_ep}: `[029] vllm/config/vllm.py:1733-1745`, `[032] :2104-2118`, `[033] :2180-2203` (on ROCm 0.33 also allows AG-RS).
- Model Runner V2: in 0.29, DBO is in the "unsupported" list, so it falls back to V1 (`[029] vllm.py:2591-2592`).
- In 0.32/0.33, DBO on V2 is enabled only with an explicit `VLLM_USE_V2_MODEL_RUNNER`, and is not allowed with speculative decoding, PP, DCP/PCP or PIECEWISE graphs (`_get_dbo_unsupported_features`, `[032] :3085-3125`, `[033] :3222-3262`).
- **So DBO + MTP means Model Runner V1.** HSG did not use DBO for GLM.

**EPLB.**
- `--enable-eplb --eplb-config '{"num_redundant_experts":R, ...}'` (`[029] arg_utils.py:1235-1236`).
- The model supports it through `DeepseekV2MixtureOfExperts`. The EPLB communicator prefers NIXL (`[029] parallel.py:1005-1015`).
- Each redundant expert per rank costs 37.76 MB × 76 MoE layers ≈ 2.87 GB per GPU, so it is only affordable on wide EP (EP32).

---------------------------------------------------------------------------------------------------

## 5. Speculative decoding (MTP) constraints

- **Runner.** Model Runner V2 is the default on CUDA unless an unsupported feature is set (`use_v2_model_runner`, `[029] vllm.py:645-677`, `[032] :703-739`, `[033] :731-767`). `VLLM_USE_V2_MODEL_RUNNER=1` forces it. MTP is supported on V2 in all three versions.
- **MTP + PP.** Allowed on V2: the speculator is created only on the last PP rank (`[029] vllm/v1/worker/gpu/model_runner.py:256-275`; `[032] :292-310`; `[033] :301`). Only eagle3/dflash/dspark/extract_hidden_states are rejected with PP in 0.29 (L262-275).
  - Fixes landed later: "MTP sharing target embeddings under PP (#58648)" and "MRV2+PP+MTP align state (#60210)" in 0.32. **Prefer ≥ 0.32 for MTP+PP.**
  - With PP the drafter keeps its own embedding copy.
  - V1 + PP + async scheduling is disallowed from 0.32 (`[032] vllm.py:2970-2974`, `[033] :3104-3108`). V2 keeps `pp_size + 1` batches in flight (`[029] vllm.py:563-569`).
- **MTP + DP + EP.** Works (HSG ran DP8xEP8 + MTP3 on 0.32 and V2). 0.31 fixed a "KV connector + MTP deadlock under KV pressure (#57104)", so P/D + MTP on 0.29 carries that bug.
- **MTP + DBO.** Not on V2 (section 4).
- **CUDA graphs.**
  - Default `cudagraph_mode` is FULL_AND_PIECEWISE (HiSparse error text, `[032] vllm.py:1818-1822`). Decode and verify batches of uniform query length (1 + k) get FULL graphs because all SM90 sparse backends and the indexer report `UNIFORM_BATCH`. Flattened next_n (k = 2, 4, 5) makes the indexer `ALWAYS`.
  - HSG's decode tier used `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` with `--max-cudagraph-capture-size = max_num_seqs × (1+k)` (`hsg-ref/gym/vllm_configs/glm_5.3.sh` VLLM_DECODE_ARGS).
- **num_speculative_tokens.**
  - Any k ≥ 1 is legal (n_predict = 1).
  - SM90 indexer native next_n {1,2,4}, so k ∈ {1,3}.
  - HSG on GB200: k = 3 best; k = 4 segfaulted in CUDA-graph replay; k = 5 and 8 hit illegal memory access in DeepGEMM `fp8_gemm_nt`; k = 8 also ran 15% slower (`NOTES.md:137, 298-301, 537-545`). Acceptance per position was 0.80 / 0.64 / 0.53 (mean 2.97) on agentic traffic.
- **Per-batch-size schedule.** `num_speculative_tokens_per_batch_size: [[lo,hi,k],...]` exists in all three (`[029] speculative.py:473`, `[032] :491`, `[033] :524`). HSG uses `[[1,4096,0]]` on the prefill tier ("prefill must build the MTP KV but has nothing to draft").
- **0.33 additions.** Lookahead block hashes keep the last prefix-cache block on EAGLE/MTP hits; on by default, opt out with `disable_lookahead_block_hashes` (`[033] speculative.py:473-482`, #61589). `draft_lm_head_quantization` (V2 only).

---------------------------------------------------------------------------------------------------

## 6. Multi-node launch

**Flags.** All exist in all three versions; `vllm/engine/arg_utils.py` lines are `[029]` / `[032]` / `[033]`.

| flag | 0.29 | 0.32 | 0.33 |
|---|---|---|---|
| `--distributed-executor-backend` | 1066 | 1135 | 1167 |
| `--pipeline-parallel-size` | 1070 | 1139 | 1171 |
| `--master-addr` | 1074 | 1143 | 1175 |
| `--master-port` | 1075 | 1144 | 1176 |
| `--nnodes/-n` | 1076 | 1145 | 1177 |
| `--node-rank/-r` | 1077 | 1146 | 1178 |
| `--tensor-parallel-size` | 1108 | 1177 | 1209 |
| `--decode-context-parallel-size` | 1111 | 1180 | 1212 |
| `--dcp-comm-backend` | 1116 | 1185 | 1217 |
| `--dcp-q-replicate` | 1120 | 1189 | 1221 |
| `--data-parallel-size/-dp` | 1137 | 1209 | 1241 |
| `--data-parallel-rank` | 1140 | 1212 | 1244 |
| `--data-parallel-start-rank` | 1149 | 1221 | 1253 |
| `--data-parallel-size-local` | 1155 | 1227 | 1259 |
| `--data-parallel-address` | 1161 | 1233 | 1265 |
| `--data-parallel-rpc-port` | 1167 | 1239 | 1271 |
| `--data-parallel-backend` | 1174 | 1246 | 1278 |
| `--data-parallel-hybrid-lb` | 1181 | 1253 | 1285 |
| `--data-parallel-external-lb` | 1186 | 1258 | 1290 |
| `--data-parallel-multi-port-external-lb` | 1191 | 1263 | 1295 |
| `--enable-expert-parallel` | 1200 | 1273 | 1305 |
| `--all2all-backend` | 1213 | 1286 | 1318 |
| `--enable-dbo` | 1215 | 1288 | 1320 |

- `--headless` and `--api-server-count` live in `vllm/entrypoints/cli/serve.py`.
- `--data-parallel-rpc-port` defaults to 29550 (`[029] parallel.py:143`); `master_port` defaults to 29501 (`:280`).

**TP or PP across nodes with the mp backend: supported, no Ray needed.**
- `nnodes > 1` on CUDA defaults the executor to `mp` (`[029] parallel.py:953`, `[032] :1045`, `[033] :1064`). `nnodes > 1` is only valid with `mp` / `uni` / `external_launcher` (`[029] :997-1003`, `[032] :1093`, `[033] :1112`).
- Node 0 runs the API server and engine. Nodes with `node_rank_within_dp > 0` and `--headless` run a "headless multiproc executor … for multi-node PP/TP" (`[029] serve.py:178-226`, `[032] :181-226`, `[033] :181-226`).
- Rank order is pp × tp, so TP8 stays inside a node and PP2 crosses nodes.
- `nnodes_within_dp = nnodes // (dp / dp_local)` (`[029] parallel.py:742-756`).
- **Unset `VLLM_PORT`.** Upstream has a port race on non-driver nodes (EADDRINUSE) that the RL env patched in `[029] shm_broadcast.py`. HSG's launcher also runs `unset VLLM_PORT`.

**DP options.**
- `--data-parallel-size N --data-parallel-size-local n --data-parallel-address HEAD --data-parallel-rpc-port P`.
- Additional nodes add `--data-parallel-start-rank r --headless` (internal LB, one API server on head), or run without `--headless` and with `--data-parallel-hybrid-lb` (one API server per node, local ranks only).
- Since 0.29, `--nnodes N --node-rank r --master-addr HEAD` also infers DP rank and local size, and the DP address defaults to `--master-addr` (`[029] arg_utils.py:2161-2200`, `[032] :2349-2405, 2496-2512`, `[033] :2390-2445`).
- External LB (`--data-parallel-external-lb` / `--data-parallel-rank`) runs one process per DP rank, MoE models only (`[032] :2415-2424`).
- `api_server_count` defaults: internal LB uses `data_parallel_size`; hybrid uses `data_parallel_size_local`; external or Rust uses 1 (`[029] serve.py:104-133`).
- **The internal DP LB is load-based (`waiting + running` score), not prefix-aware** (`[033] vllm/v1/engine/core_client.py:1751-1790`). For multi-turn agents, pin each trajectory to one DP rank with the `X-data-parallel-rank` header (`[029] vllm/entrypoints/generate/base/serving.py:243`, `[032] :225`, `[033] :221`) or with vllm-router.
- With the Rust frontend the header means the *global* DP rank (HSG `NOTES.md:91-99`).

---------------------------------------------------------------------------------------------------

## 7. Prefix caching, chunked prefill, context length, KV connectors

**Prefix caching and chunked prefill.**
- Prefix caching (`enable_prefix_caching=True`, `[029] cache.py:138`) and chunked prefill (`[032] scheduler.py:134`) are on by default, with no DSA-specific restriction found.
- Hash granularity is the 64-token block. `--prefix-match-unit` must divide it (`[029] cache.py:99`).
- In ≤ 0.32, EAGLE/MTP drop the last matched block on a hit; 0.33 keeps it via lookahead hashes.
- Async scheduling is auto-on (`[029] vllm.py:1262`, `[032] :1585`, `[033] :1685`).
- H100 serve defaults: `max_num_batched_tokens` 8192, `max_num_seqs` 1024 (`[032] arg_utils.py:2933-2941`, the "≥ 70 GiB and not A100" branch).
- Breaking change in 0.32: "max_tokens no longer reserved from the context window (#57035)".

**Context length.** `max_position_embeddings` is 1,048,576; HSG used `--max-model-len 262144`. The sparse prefill workspace factor drops from 5 to 2 above 256K (#61023), and the KV per request is bounded by the per-pool KV (section 9).

**KV connectors registered** (`vllm/distributed/kv_transfer/kv_connector/factory.py`):

| version | connectors |
|---|---|
| 0.29 | NixlConnector, MooncakeConnector, MooncakeStoreConnector, OffloadingConnector, SimpleCPUOffloadConnector, LMCacheConnectorV1, MultiConnector |
| 0.32 / 0.33 | the 0.29 set **+ HiSparseConnector** (added in 0.30) |

**NixlConnector.**
- Supports MLA (layout stays default for MLA, `[032] nixl/connector.py:150-161`) and `UniformTypeKVCacheSpecs` (MLA + indexer, `[029] nixl/utils.py:54`). It is `SupportsHMA` (`[032] connector.py:78`).
- Heterogeneous P/D TP for MLA: `tp_mapping.py`, `[029] :87-139`.
- PP-aware metadata keyed by (pp_rank, tp_rank) in `[032] nixl/base_scheduler.py:216-376`. "NIXL pipeline-parallel push prefill … packed MLA layouts" landed in 0.31.
- `kv_lease_duration` defaults to 30 s (`[029] base_scheduler.py:75-81`, `[032] :82-88`); HSG used 600.
- `kv_load_failure_policy` ∈ {recompute, fail} (`[029] config/kv_transfer.py:69`).
- **0.33 enforces `check_nixl_version("1.5.0")`** (`[033] nixl/base_worker.py:613`). The RL env has nixl 1.3.0.

**CPU KV offload (native).**
- `--kv-offloading-size <GiB> --kv-offloading-backend native|lmcache` (`[029] cache.py:241-250`, `[032] :266-275`, `[033] :383-392`). With TP>1 the size is the total across TP ranks.
- Handles DSA mixed page sizes ("mixed page sizes for DSA models #54756", 0.30). The offloading config certifies `UniformTypeKVCacheSpecs` (MLA + indexer) per layer (`[032] offloading/config.py:216-240`).
- The deduplicated "replicated" MLA layout needs `nnodes_within_dp == 1`, so a TP8xPP2 engine spanning 2 nodes stores one copy per TP rank (**UNVERIFIED** capacity impact).
- This is the H100 analogue of HSG's Mooncake store: the hosts have 2 TB RAM. HSG found a CPU tier *required* for SWE P/D on GB200 (`NOTES.md:83`).

**MooncakeStoreConnector.** In all versions; the RL env has mooncake 0.3.13. It worked on HSG's IB and silently corrupted requests on AGA RoCE (`NOTES.md:446-450`).

**LMCache.** Via `LMCacheConnectorV1` or `--kv-offloading-backend lmcache`. **UNVERIFIED** for DSA.

**HiSparse (0.30+, experimental): a host-resident KV tier for DSA decode.**
- `--attention-config` `hisparse_config` (`[032] config/attention.py:17-33, 106-110`), or inferred from HiSparseConnector.
- Requires: Model Runner V2 (`[032] vllm.py:704-709`), no PP (`:1811`), no DCP, not `cudagraph_mode=FULL` (`:1818`), the hybrid KV cache manager, and a DSA model (`:1834`).
- Wired into FA-sparse and FlashMLA-sparse (`index_group.HiSparseMLAIndexGroup`) but not FlashInfer-SM90.
- Interesting for H100's small HBM but **not for first benchmarks**.

**DCP (decode context parallel, shards KV over the TP ranks; needs `tp % dcp == 0`, `[032] parallel.py:592-601`).**
- On SM90 only FLASHMLA_SPARSE supports it (`supports_dcp=True`). FA-sparse rejects it; FlashInfer-SM90 does not support it.
- Requires `fp8_ds_mla` and the mixed-batch path (heads per rank < 32) (`[029] flashmla_sparse.py:309-323`, `[032] :448-466`).
- In 0.29 and 0.32 it **requires `--dcp-comm-backend ag_rs`**, but GLM's config hook defaults to `a2a`, so that must be overridden explicitly. 0.33 removed this check.
- The indexer DCP merge requires CuteDSL and index_topk ∈ {512,1024,2048} (`[032] sparse_attn_indexer.py:67-73`).
- **UNVERIFIED end-to-end on H100.**

---------------------------------------------------------------------------------------------------

## 8. Changelog 0.29 → 0.33 items relevant to GLM-5.x / DSA / Hopper

Sources: GitHub release pages. The 0.30-0.32 pages were parsed in full. For 0.33 only the Highlights and Model Support sections were retrievable: the tag page errored and the API was rate-limited, so the 0.33 Engine Core / Hardware sections are **UNVERIFIED**.

**0.30.0**
- DeepGEMM pinned to the vLLM fork 2.8.0 "with SM120 and SM90 paged-MQA ports" (#56876).
- MRV2: DBO (#50945, #51700); MTP (#46994) and EAGLE3/DFlash/DSpark (#50514) under PP.
- HiSparse (#53781, #56629, #57041); PCP+DCP on sparse-MLA models (#56157).
- KV offloading: "mixed page sizes for DSA models" (#54756) and offload hits under MTP/EAGLE (#52771).
- DeepSelect top-k `--sparse-indexer-topk-backend` (SM100) (#56464); FlashInfer XQA decode on SM90 (#50439).
- GLM-5.3-Flash with EPLB (#55119).

**0.31.0**
- MRV2 counts MoE memory during profiling, "avoiding OOMs on WideEP" (#57270, #58411). HSG's agg32 OOM is that class of bug.
- KV connector + MTP deadlock under KV pressure fixed (#57104); prompt-tail hits with MTP restored (#58368).
- Sparse prefill buffers reserved before KV sizing (#57575); allocator fragmentation no longer shrinks the KV cache (#58430); DeepGEMM warmup reuses the MoE workspace (#57268).
- NIXL PP push prefill and packed MLA layouts (#50494, #50499); expired leases reaped (#58292).
- `--max-num-active-seqs` (#56758); adaptive `--long-prefill-token-threshold` (#57951); KV-holding waiting queue (#58947).
- FlashInfer 0.7.0.post1.
- GLM-5.3-Flash opt-in FA/FlashMLA sparse on SM90 (#55385).

**0.32.0** (HSG's version)
- **FP8 KV on SM90 sparse MLA (#55222)**, i.e. FLASHINFER_MLA_SPARSE_SM90.
- `fp8_ds_mla` on SM90: "up to 31% more output throughput" for GLM-5.3-Flash (#59246).
- GLM-5.x FP8 KV default on SM100 (#60140, #60281).
- GLM-5.3 sparse MLA index conversion reused across layers, 3.5-3.9x on that kernel (#59464); fused multi-step draft decode, +13.3% at concurrency 1 (#57443).
- qlnorm skipped for MHA, 4.4-7.7% TTFT (#58845); long-context indexer prefill sharded across TP, up to 25% lower TTFT at 1M (#54951).
- **MTP index sharing restored (#61079)**; MTP sharing target embeddings under PP (#58648); MRV2+PP+MTP align-state fix (#60210); DP startup fix for GLM (#56974).
- `sparse_mla_prefill_workspace_factor` = 2 above 256K (#61023); block-FP8 DeepGEMM experts skip padding (#59128).
- NIXL 1.5.0 (#56907); Mooncake coalesced MLA RDMA regions (#57952); `vllm download-kernels` (#58765).
- Breaking: Python 3.10 dropped; transformers ≥ 5.16.1, < 5.20.0; cascade attention removed; max_tokens no longer reserved from the context window (#57035).

**0.33.0** (2026-11-01)
- GLM-5.3 per-token-group quant, 2.8% ITL (#60927); in-place draft decode metadata (#61873); skipped concat_and_cache_mla scales, B300 (#62102).
- FP8 indexer WK pairs kept across `load_weights`, "fixing thinking loops with runai_streamer" (#59223).
- **Lookahead block hashes for MTP prefix hits, default on (#61589)**.
- "native small-head FlashMLA sparse prefill on SM90 (#61615)" (listed under DeepSeek-V4/V4.1; **UNVERIFIED** whether it reaches the GLM path; `[033] flashmla_sparse.py:754` still sets `prefill_padding = 64`).
- PyTorch 2.14; CUDA 12.9 artifacts dropped.
- Breaking: **NIXL connector needs NIXL ≥ 1.5 (#61032)**; Medusa/MLP speculator removed; legacy Ray compiled-graph executor removed.
- Vendored DeepGEMM 2.8.1 API change (section 3; found in the source, not in the notes).

---------------------------------------------------------------------------------------------------

## 9. Memory math

### 9.1 Weights (exact)

`tools/weight_bytes.py` derives every tensor's shape from `config.json` and its name from the index. FP8 tensors carry `[ceil(o/128), ceil(i/128)]` fp32 `weight_scale_inv`. The router gate, indexer `weights_proj` and `k_norm` are BF16; this was resolved by requiring the sum to equal `total_size` exactly, which it does: **755,617,140,416 B**.

| category | bytes | GiB |
|---|---|---|
| routed experts, 75 main MoE layers × 256 × 37,757,952 B | 724,952,678,400 | 675.2 |
| routed experts, MTP layer 78 | 9,666,035,712 | 9.0 |
| attention, 78 layers (q_a, q_b, kv_a, kv_b, o + scales + norms) | 12,877,011,072 | 12.0 |
| shared experts (75) | 2,831,846,400 | 2.64 |
| embed_tokens / lm_head (BF16, 154880 × 6144 each) | 1,903,165,440 each | 1.77 each |
| dense MLP (layers 0-2) | 679,643,136 | 0.63 |
| router gate + `e_score_correction_bias` (75) | 236,006,400 | 0.22 |
| indexers (21) | 200,991,168 | 0.19 |
| MTP non-expert (attention, indexer, shared expert, router, `eh_proj` BF16 151 MB, norms) | 366,597,248 | 0.34 |

Note: the index used is the HSG Sep-1 snapshot (141 shards). The HF revision being downloaded has 156 shards; **UNVERIFIED** that its total is identical.

**Per-GPU model** (`tools/mem_math.py`). The sharding follows the vLLM code:

| treatment | parts |
|---|---|
| replicated under TP | fused `q_a`+`kv_a` (`DeepSeekV2FusedQkvAProjLinear(disable_tp=True)`, `[032] deepseek_v2.py:953-967`); indexer `wq_b` (ReplicatedLinear) and `wk_weights_proj` (BF16 after load); norms; router; `eh_proj` |
| TP-sharded | `q_b`, `kv_b`, `o_proj`, vocab-parallel embed and lm_head; dense MLP and shared expert (replicated under SP-MoE) |
| experts | ÷ EP or ÷ TP |
| plus MLA absorbed BF16 copies | `W_UK_T + W_UV` = 29.4 MB/layer at TP1, ÷ TP |

Calibration against HSG DP8×EP8 (TP1): the model predicts **107.3 GiB/GPU**; HSG `NOTES.md:47` reports 106.6 GiB.

### 9.2 Reserve assumption (activations, workspaces, CUDA graphs, comm buffers)

vLLM sizes KV as `u × total − weights − profiled peak − non-torch`. H100 total is taken as 81,559 MiB = 79.65 GiB, as nvidia-smi reports for the 80GB HBM3 part (**UNVERIFIED** on the dfw nodes).

HSG calibration on GB200, assuming 189,471 MiB per GPU (**UNVERIFIED**), DP8×EP8:
- decode tier (u = 0.93, 256 tokens/step, MTP3): "other" = **9.1 GiB**;
- prefill tier (u = 0.90, 8192-token chunks): **20.0 GiB**;
- halving chunks from 16384 to 8192 freed 251k tokens × 48.4 kB = **11.3 GiB/rank** (E9c → E11c, E12c → E13). DP×EP activation memory therefore scales with DP × batched tokens (about 1.4 MB per batched token at DP8, flashinfer one-sided all2all).

The tables below use **R = 8 GiB/GPU**. That is plausible for TP8/TP16/PP layouts at 8192 batched tokens: about 3.4 GiB of sparse workspaces at max_model_len 262,144, plus MoE and activation, CUDA graphs and NCCL buffers.

For DP×EP layouts R = 8 GiB is only plausible at `--max-num-batched-tokens ≤ 2048` (or with DeepEP's fixed-size buffers). **Treat every number as ±1 GiB-sensitive. Read "Available KV cache memory" in the startup log.**

Sensitivity per 1 GiB of reserve:

| layout | tokens/GPU per GiB |
|---|---|
| non-PP, `fp8_e4m3` + MTP | 22.2k |
| PP2 (41/37) | about 42.6k |
| PP4 | about 82k |

### 9.3 Capacity tables

- KV tokens per GPU are computed after weights and R.
- "KV tok/engine" is the capacity of independent KV pools × DP ranks. MLA is replicated across TP, so a TP group is one pool; a PP engine is one pool.
- "100k seqs" is `floor(tokens per pool / 100k) × pools`, ignoring prefix sharing.
- The MTP layer is included in every row.
- Full output for all three dtypes is in `tools/mem_tables.md`.

**`fp8_e4m3`** (0.32/0.33 FlashInfer-SM90; 48,408 B/token non-PP):

| layout | nodes | weights GiB/GPU | u | free GiB (before R) | KV tok/GPU | KV tok/engine | 100k seqs | verdict |
|---|---|---|---|---|---|---|---|---|
| TP16 (MoE TP16 or EP16) | 2 | 45.8 | 0.85 / 0.90 / 0.93 | 21.9 / 25.9 / 28.3 | 309k / 397k / 450k | 0.31M / 0.40M / 0.45M | 3 / 3 / 4 | feasible; small KV; TP all-reduce over IB on every layer |
| TP8xPP2 (39/39) | 2 | 47.4 (stage1) | same | 20.3 / 24.3 / 26.7 | 541k / 717k / 822k | same as per-GPU | 5 / 7 / 8 | feasible |
| **TP8xPP2, `VLLM_PP_LAYER_PARTITION=41,37`** | 2 | 45.1 | same | 22.6 / 26.6 / 29.0 | **630k / 799k / 901k** | same | 6 / 7 / 9 | **best plain 2-node layout** |
| DP2xTP8 + EP16 (SP-MoE) | 2 | 49.9 | same | 17.8 / 21.8 / 24.2 | 217k / 305k / 358k | 0.43M / 0.61M / 0.72M | 4 / 6 / 6 | feasible; SP-MoE correctness check needed |
| DP4xTP4 + EP16 (SP-MoE) | 2 | 52.0 | same | 15.7 / 19.7 / 22.1 | 171k / 259k / 312k | 0.68M / 1.04M / 1.25M | 4 / 8 / 12 | feasible but tight; 16 heads/rank; SP-MoE |
| DP16xEP16 (TP1 attention) | 2 | **64.5** | same | 3.2 / 7.2 / 9.6 | 0 / 0 / 35k | 0 / 0 / 0.56M | 0 | **infeasible** for 256k context; even R=0 at u=0.93 gives only 213k/GPU |
| P/D tier = TP8xPP2 (41,37), per tier | 2 | 45.1 | same | as above | 630k / 799k / 901k | 0.63M / 0.80M / 0.90M | 6 / 7 / 9 | feasible |
| P/D tier = DP2xTP8 + EP16, per tier | 2 | 49.9 | same | as above | 217k / 305k / 358k | 0.43M / 0.61M / 0.72M | 4 / 6 / 6 | feasible |
| DP32xEP32 (TP1 attention) | 4 | 43.1 | same | 24.6 / 28.6 / 31.0 | 368k / 456k / 509k | 11.8M / 14.6M / 16.3M | 96 / 128 / 160 | feasible; EP lockstep across 4 nodes |
| TP8xPP4 (21,20,19,18) | 4 | 23.4 | same | 44.3 / 48.3 / 50.7 | 3.15M / 3.48M / 3.68M | same | 31 / 34 / 36 | feasible; 4-stage pipeline latency |
| DP4xTP8 + EP32 (SP-MoE) | 4 | 28.5 | same | 39.2 / 43.1 / 45.5 | 691k / 780k / 833k | 2.77M / 3.12M / 3.33M | 24 / 28 / 32 | feasible |
| DP8xTP4 + EP32 (SP-MoE) | 4 | 30.6 | same | 37.1 / 41.1 / 43.5 | 645k / 733k / 786k | 5.16M / 5.87M / 6.29M | 48 / 56 / 56 | feasible |
| DP16xTP2 + EP32 (SP-MoE) | 4 | 34.8 | same | 32.9 / 36.9 / 39.3 | 553k / 641k / 694k | 8.84M / 10.3M / 11.1M | 80 / 96 / 96 | feasible |

**Other KV dtypes at u=0.93** (non-PP bytes: 54,728 for `fp8`/`fp8_ds_mla`, 93,912 for BF16):

| layout | `fp8_ds_mla` (FlashMLA-sparse, any version) tok/engine | BF16 (FA3-sparse) tok/engine |
|---|---|---|
| TP16 | 0.40M | 0.23M |
| TP8xPP2 (41,37) | about 0.80M | 0.46M |
| DP2xTP8 + EP16 | 0.63M | 0.37M |
| DP4xTP4 + EP16 | 1.10M | 0.64M |
| DP32xEP32 | 14.4M | 8.4M |
| TP8xPP4 | 3.26M | 1.91M |

**P/D with 2 nodes per tier (4 nodes total).** Each tier has the capacity of the corresponding 2-node row:

| tier layout | KV per tier at u = 0.90 / 0.93 (`fp8_e4m3`) |
|---|---|
| TP8xPP2 (41,37) | 0.80M / 0.90M |
| DP2xTP8 + EP16 | 0.61M / 0.72M |

The decode tier can run u = 0.93 with small batches, as HSG did. The prefill tier also needs the MTP layer's KV (HSG `glm_5.3.sh`).

**Advanced lever: TP8xPP2 + DCP8.** With `fp8_ds_mla`, KV is sharded 8x (`tools/dcp_calc.py`):

| setting | KV tok/engine at u = 0.90 / 0.93 |
|---|---|
| without q-replicate | about 5.7M / 6.4M |
| with q-replicate (+1.2 GB/GPU) | about 5.3M / 6.0M |

That is about 7x TP8xPP2. **UNVERIFIED on SM90**, and DCP's own workspaces are not counted.

---------------------------------------------------------------------------------------------------

## 10. Ranked configs to benchmark, with exact flags

### 10.0 Shared setup (inside the existing `nemo-rl-cuda13.2` container, fresh venv with vLLM 0.32.0)

```bash
export CUDA_HOME=/usr/local/cuda                 # DeepGEMM / FlashInfer JIT need nvcc
export VLLM_CACHE_ROOT=/raid/$USER/vllm-cache    # node-local; never ~/.cache (10 GB /home quota)
export DG_JIT_CACHE_DIR=$VLLM_CACHE_ROOT/deep_gemm
export XDG_CACHE_HOME=/raid/$USER/xdg-cache TRITON_CACHE_DIR=/raid/$USER/triton-cache
export VLLM_ENGINE_READY_TIMEOUT_S=3600          # default 600 s ([029] envs.py:796)
unset VLLM_PORT
export GLOO_SOCKET_IFNAME=$IFACE NCCL_SOCKET_IFNAME=$IFACE   # the head-IP interface, as in serve_glm_53.slurm
# NCCL_IB_HCA / UCX_NET_DEVICES are preset by the image for mlx5_0,1,2,4,5,6,7,8
MODEL=/lustre/fsw/portfolios/nemotron/users/hengyuf/models/GLM-5.3
COMMON=(--served-model-name glm-5.3 --trust-remote-code
  --max-model-len 262144
  --kv-cache-dtype fp8_e4m3          # 0.29: use fp8 (fp8_ds_mla); A/B fp8 on 0.32+ too
  --enable-prefix-caching
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  --tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":64}'
  --disable-uvicorn-access-log)
```

Checks after startup:
- the log names the attention backend ("Using … attention backend out of potential backends");
- the FP8 MoE backend ("Using … Fp8 MoE backend");
- "Using fp8_ds_mla KV cache format…" when `fp8` is used;
- the KV capacity line.

Run `hsg-ref/notes/probe_tools.py` (glm47 tool and reasoning parsing) and a greedy parity prompt before any load test.

### 10.1 Two nodes (16 x H100)

**1. TP8 x PP2, MTP3.** The default pick: about 0.9M KV tokens; all collectives inside NVLink except the PP hop; no EP all2all; no SP-MoE.

```bash
# node 0
VLLM_PP_LAYER_PARTITION=41,37 vllm serve $MODEL "${COMMON[@]}" \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --distributed-executor-backend mp --nnodes 2 --node-rank 0 --master-addr $HEAD_IP --master-port 29501 \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.92 --max-num-batched-tokens 8192 --max-num-seqs 64
# node 1: identical, with --node-rank 1 --headless (no --host/--port)
```

A/B knobs, in order:
1. `--kv-cache-dtype fp8` (FlashMLA-sparse `fp8_ds_mla`, 8x head padding at TP8) vs `fp8_e4m3` (FlashInfer-SM90).
2. MoE `--moe-backend triton` (default at TP8; N=256 has no H100 tuning file) vs `deep_gemm` vs `--enable-expert-parallel` (EP8 per stage, so FLASHINFER_CUTLASS).
3. `--gpu-memory-utilization 0.93`, `--max-num-seqs` 64 vs 128.
4. CPU tier `--kv-offloading-size 400 --kv-offloading-backend native` (**UNVERIFIED** with PP across nodes).
5. MTP 1 vs 3.

**2. DP2 x TP8 + EP16.** About 0.72M tokens; MoE weights 16-way; two attention groups. Run a parity check first because of the SP-MoE path.

```bash
# node 0 (API server, internal LB)
vllm serve $MODEL "${COMMON[@]}" --tensor-parallel-size 8 \
  --data-parallel-size 2 --data-parallel-size-local 1 \
  --data-parallel-address $HEAD_IP --data-parallel-rpc-port 13345 \
  --enable-expert-parallel --all2all-backend allgather_reducescatter \
  --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 4096 --max-num-seqs 64
# node 1: same plus --data-parallel-start-rank 1 --headless
# then try --all2all-backend deepep_high_throughput (needs deep_ep + working NVSHMEM internode: UNVERIFIED)
```

**3. TP16 (single engine).** The simplest cross-node baseline (0.45M tokens); inter-node all-reduce on every layer. Optionally add `--enable-expert-parallel`.

```bash
vllm serve $MODEL "${COMMON[@]}" --tensor-parallel-size 16 --distributed-executor-backend mp \
  --nnodes 2 --node-rank 0 --master-addr $HEAD_IP --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.92 --max-num-batched-tokens 8192 --max-num-seqs 64
# node 1: --node-rank 1 --headless
```

**4. (Capacity alternative) DP4 x TP4 + EP16.** 1.25M tokens but 16 heads/rank, SP-MoE, and a 4-rank lockstep. Use `--tensor-parallel-size 4 --data-parallel-size 4 --data-parallel-size-local 2`; node 1 adds `--data-parallel-start-rank 2 --headless`. Same EP flags as config 2, with `--max-num-batched-tokens 2048`.

**5. (Advanced, after 1 works) TP8 x PP2 + DCP8.** Add to config 1:
`--kv-cache-dtype fp8_ds_mla --decode-context-parallel-size 8 --dcp-comm-backend ag_rs`.
Optionally add `--no-dcp-q-replicate`; boolean fields get `BooleanOptionalAction` (`[032] arg_utils.py:380`).
Potentially about 6M KV tokens. **UNVERIFIED**: FlashMLA-sparse DCP on SM90 with PP and MTP.

**Not recommended on 2 nodes:**
- DP16 x EP16: weights alone are 64.5 GiB/GPU.
- Any `flashinfer_nvlink_*` all2all: needs MNNVL.
- DBO: forces Model Runner V1 with MTP and needs DeepEP or nixl_ep.

### 10.2 Four nodes (32 x H100)

**1. Two independent TP8xPP2 replicas** (two copies of 10.1-1) behind a session-sticky router (vllm-router cache-aware, or client-side pinning). 2 x 0.9M tokens, no cross-replica lockstep, nothing new to debug.

**2. P/D p2d2 with TP8xPP2 tiers** (HSG's winning structure; the decode tier is insulated from prefill chunks).

Prefill tier, config 10.1-1 flags plus:
```bash
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"kv_lease_duration":600}}' \
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"num_speculative_tokens_per_batch_size":[[1,4096,0]]}' \
--max-num-batched-tokens 8192 --gpu-memory-utilization 0.90
```

Decode tier, config 10.1-1 flags plus:
```bash
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"kv_lease_duration":600}}' \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --max-num-seqs 64 --max-cudagraph-capture-size 256 \
--max-num-batched-tokens 2048 --gpu-memory-utilization 0.93
```

Requirements:
- vllm-router in front (`hsg-ref/gym/build_vllm_router_wheel.sh`).
- NIXL ≥ 1.5 in the env (0.33 hard-checks).
- UCX over the mlx5 IB NICs.
- PP on NIXL tiers is code-supported in 0.31+ (**UNVERIFIED** end-to-end). A fallback tier shape is DP2xTP8+EP16.
- Optional CPU tier on prefill: add MooncakeStoreConnector via MultiConnector, as in HSG.

**3. DP32 x EP32 aggregated.** 16.3M tokens and the best attention efficiency (64 heads/rank), but EP lockstep: HSG measured 136 s/GLM call for aggregated DP32 vs 20 s for P/D (`NOTES.md:318-366`). Only worth it for throughput-bound offline rollouts.

```bash
# node r = 0..3
vllm serve $MODEL "${COMMON[@]}" --tensor-parallel-size 1 \
  --data-parallel-size 32 --data-parallel-size-local 8 \
  --data-parallel-address $HEAD_IP --data-parallel-rpc-port 13345 \
  --enable-expert-parallel --all2all-backend allgather_reducescatter \
  --max-num-batched-tokens 2048 --max-num-seqs 32 --gpu-memory-utilization 0.90
# r > 0: add --data-parallel-start-rank $((8*r)) --headless
# route with X-data-parallel-rank per trajectory
# try --all2all-backend deepep_low_latency only if NVSHMEM IBGDA works on dfw
```

**4. DP4 x TP8 + EP32** (3.3M tokens, TP inside each node, SP-MoE) or **TP8 x PP4** (3.7M tokens, one engine):
- DP4xTP8+EP32: `--tensor-parallel-size 8 --data-parallel-size 4 --data-parallel-size-local 1 ...`.
- TP8xPP4: `VLLM_PP_LAYER_PARTITION=21,20,19,18 ... --tensor-parallel-size 8 --pipeline-parallel-size 4 --nnodes 4`.

---------------------------------------------------------------------------------------------------

## 11. Which vLLM version on H100

**Recommendation: build a fresh venv with vLLM 0.32.0 first, then A/B 0.33.0.** Keep the 0.29 RL env only for a read-only smoke test.

Reasons for 0.32.0 over the RL env's 0.29:
- the SM90 FP8-KV sparse backend (`fp8_e4m3`, FlashInfer-SM90), so HSG's KV dtype is usable at 576 B;
- GLM-5.3 index-conversion and draft-decode speedups;
- the KV-connector + MTP deadlock fix (0.31);
- MoE memory counted in MRV2 profiling (0.31; HSG's OOM class);
- sparse buffers reserved before KV sizing;
- MTP+PP fixes;
- the workspace factor;
- NIXL 1.5;
- the same DeepGEMM API as HSG's working `deep_gemm_v0320`, plus HSG's measured configs (`max-num-batched-tokens` 8192, MTP3, u = 0.90 / 0.93) to transfer.

0.29 in the RL env: `fp8_e4m3` is impossible on SM90, DeepGEMM 2.5.0 is top-level, nixl is 1.3.0, and the P/D+MTP deadlock is unfixed.

Reasons to A/B 0.33.0:
- lookahead hashes (better prefix hits with MTP on multi-turn traffic);
- GLM per-token-group quant;
- the DCP `ag_rs` check removed.

Its new DeepGEMM 2.8.1 API (no `clean_logits`), torch 2.14 and the NIXL ≥ 1.5 hard check mean **do not mix its wheel with any other `deep_gemm`**.

Build notes:
- `uv pip install vllm==0.32.0` with torch 2.13.0 cu130 and flashinfer-python 0.7.0.post1 (pulled by the wheel pins);
- `nixl>=1.5` for P/D;
- `deep_ep` only for DeepEP all2all;
- **no top-level `deep_gemm`**;
- run under the same cuda-compat container as the RL env.

---------------------------------------------------------------------------------------------------

## 12. UNVERIFIED / open items

1. NVSHMEM IBGDA / NCCL GIN on dfw IB (gates DeepEP-LL, DeepEPv2; DeepEP-HT internode transport).
2. Real activation reserve per layout on H100 (section 9.2 assumes 8 GiB).
3. FlashInfer-SM90 sparse vs FlashMLA-sparse throughput at TP8 (head padding) and accuracy (per-tensor `fp8_e4m3` scale 1.0 vs block-scaled `fp8_ds_mla`).
4. SP-MoE correctness for GLM on the CUDA path (DP>1 & TP>1 & EP).
5. MTP3 stability on SM90; HSG's k > 3 crashes were on SM100 DeepGEMM.
6. DCP + PP + MTP on SM90.
7. OffloadingConnector capacity and behaviour for an engine spanning 2 nodes.
8. Full 0.33 Engine Core / Hardware release notes.
9. DG 2.8.x JIT env-var names.
10. H100 / GB200 memory constants used in the math.
11. The HF revision's shard set vs the HSG index.
