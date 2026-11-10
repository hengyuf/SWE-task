# GLM-5.3 serving on HSG (GB200): engineering digest for the dfw H100 port

Author: workflow subagent `hsg-launcher`, 2026-11-06. Output dir: `/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/notes/hsg-launcher/`.

Conventions
- Relative paths are under `/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/hsg-ref/`. `file:N` / `file:N-M` are line numbers in those read-only copies.
- "dfw-vLLM" = the read-only production env `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/runtime/envs/vllm` (vLLM `0.29.0`, commit `g98dff2a81`). Paths like `dfw:vllm/...` are relative to its `lib/python3.13/site-packages/`. I only grepped/read it.
- "router-0.1.17" = the PyPI `vllm-router` 0.1.17 sdist (sha256 6c37f9a8...), and "router-0.1.15" = 0.1.15 (sha256 af8efa64...). Both were downloaded only to read the source. They are kept (tarballs only) in `.dl-router.m6YVgb/` in the output dir.
- UNVERIFIED means I could not confirm it from code or measurements. Numbers without that mark are quoted from the HSG files, as cited.
- Terms: "P/D" = prefill/decode disaggregation; "pXdY" = X prefill nodes + Y decode nodes; "tier" = the prefill or the decode engine group; "rank" = a vLLM data-parallel (DP) engine. On HSG a node has 4 GB200 GPUs; on dfw it has 8 H100s.

---

## 0. TL;DR

1. **The HSG deliverable is P/D, not TP8.** `serve_scripts/serve_glm_53_pd_swe.sh` is used for SWE / GLM as the policy. Shape: p2d2 = 4 GB200 nodes = 16 GPUs.
   - Each tier is ONE DP8×EP8 engine with TP1, spread over 2 nodes. It sits behind one Rust-frontend API server ("coupled" mode).
   - `vllm-router` (cache_aware on both tiers) pins each request to one DP rank via the `X-data-parallel-rank` header.
   - NIXL (UCX/IB) moves KV from prefill to decode. A Mooncake CPU KV store sits behind NIXL with 120GB segments.
   - Other settings: MTP 3, fp8 KV, 8k prefill chunks, GPU memory 0.90 (prefill) / 0.93 (decode), NIXL lease 600 s.
   - Measured on SWE-bench Verified ×3 (1500 rollouts): 62.9 rollouts/min at concurrency 512, and 1500 done in 54:32 at concurrency 768. Resolve rate stayed at 77-79% (`notes/NOTES.md:5-18`).
2. **Biggest measured lever: P/D vs aggregated DP×EP.** In an aggregated DP×EP engine every rank steps in lockstep. Any rank's 8k prefill chunk stalls every rank's decode step.
   - On 8 nodes with 512 requests in flight, P/D p4d4 took 20.2 s per GLM call. Aggregated DP32 took 136.5 s (with MTP) and 175.3 s (without MTP).
   - That is 3.1× more unmasked rollouts per 30 minutes (`notes/NOTES.md:358-372`).
3. **MTP 3 halves a GLM call:** 40.3 s → 20.3 s; per-request decode speed 33 → ~80 tok/s (`notes/NOTES.md:272-276, 283`).
   - MTP 4, 5 and 8 crashed on this image, via DeepGEMM illegal-address errors or CUDA-graph replay segfaults. MTP 8 was also 15% slower. MTP 3 is the ceiling (`notes/NOTES.md:137, 298-302, 537-545`).
4. **Prefill KV memory was the SWE bottleneck.** Its consequences:
   - Mooncake: about 2× (E1 → E2).
   - 16k → 8k prefill chunks: +19%.
   - More memory for KV (0.90 / 0.93): +6%.
   - Rust frontend in coupled mode: +14%.
   - Concurrency 512 → 768: +15%. Concurrency 1024 swamped prefill with no gain.
   - Adding nodes barely helps: p4d4 is +6% for 2× the GPUs.
5. **Correctness gotchas that are silent** (HTTP 200, no errors):
   - p4d4 P/D on GB300 returned garbage.
   - Mooncake over RoCE corrupted 7 of 16 answers.
   - fp8 KV under P/D degenerated on GB300 Nano.
   - EP24 (does not divide 256 experts) produced ~3.5× longer outputs and lower pass rates.

   Always gate a new layout on a distinct-prompt answer check, not on liveness.
6. **H100 differences that change the design:**
   - GLM-5.3 weighs 703.7 GiB, so it does not fit one 8×H100 node (TP8 = 88 GiB/GPU).
   - Per-GPU HBM is 2.3× smaller than GB200, and there is no NVL72 across nodes.
   - EP16/TP1 on 2 nodes costs ~64 GiB of weights per GPU (estimate, §7.2). That leaves too little room for even one 262k-token request on a TP1 rank (13.4 GiB at H100's fp8 KV format).
   - The HSG all2all backend (`flashinfer_nvlink_one_sided`) and NCCL MNNVL settings are NVL72-only.
   - The HSG KV dtype `fp8_e4m3` is not accepted by the SM90 sparse-MLA backend. dfw-vLLM `FLASHMLA_SPARSE` accepts `auto|bfloat16|fp8_ds_mla|fp8`, so use `--kv-cache-dtype fp8`.
7. **Methodology and tools carry over directly.** The vLLM Prometheus metrics used by `glm_sampler.py` all exist in dfw-vLLM. The helpers run on the login node's Python 3.9 (parse-checked). `bench_endpoint.py` / `probe_tools.py` work against any base URL. Watch the defaults: model, port, log-dir, and reasoning effort (§8).

---

## 1. What the copy contains, and what is missing

Read fully:
- `gym/sbatch_external_vllm.sh` (1363 lines)
- `gym/vllm_configs/glm_5.3.sh`
- `gym/README.md`
- every `gym/vllm_configs/*`
- `gym/custom_{vllm,mooncake,flashinfer}_version.sh`
- `gym/build_vllm_router_wheel.sh`, `gym/add_vllm_router.sh`
- `gym/diagnostics/*`
- `gym/submit_batch.sh`, `gym/batch_configs/swe.yaml`
- `serve_scripts/*` (GLM 5/5.1/5.3 and the P/D wrappers, Kimi K3 and K3 longctx, DSv4 Flash, Flash-0731 and Pro, Ultra and Ultra NVFP4, Super 3.5; the others were grepped)
- `serve_scripts/bench_endpoint.py`, `serve_scripts/README.md`
- `notes/NOTES.md`, `notes/ledger*.tsv`, `notes/train_exps.tsv`
- every helper script in `notes/`
- `model_meta/config.json`, `chat_template.jinja`, `model.safetensors.index.json`

Missing from the copy, but referenced:
- `gym/save_warmed_vllm_container.py`. **`sbatch_external_vllm.sh:78` reads it unconditionally with `cat` under `set -euo pipefail`.** The launcher as copied therefore aborts at line 78. Port it with that line removed or guarded.
- Gym files: `benchmarks/nemotron_3.5_super/{sandbox_utils.yaml,policy_model_override.yaml,export_to_csv.py,build_eval_container.sh,sbatch_eval_with_external_vllm.sh}`, `nemo_gym/sandbox/providers/opensandbox/cleanup_sandboxes.py`, `allocation_trace_worker.py`.
- JK's harness: `policy_runs/rp_glmpd_submit.sh`, `readout.py`, `env_smoke/run.sh`.
- The AGA launcher copy `aga/serve_glm_53_pd.sh`.
- `hsg-ref/deep_gemm/` and `hsg-ref/lvega/` are **empty directories**.

Software stack of the HSG GLM P/D runs (`notes/NOTES.md:38-39`, `serve_scripts/serve_glm_53_pd.sh:68-71`):
- Image: Brian Yu's `vllm-openai:v0.32.0___with_gym_vllmfixes___mooncakemetricstry2___fixduplicatemambaforward.sqsh`, aarch64. It is the bxyu vLLM v0.32.0 fork with precompiled binaries from a newer nightly.
- A stock v0.32.0 `deep_gemm` is put first on PYTHONPATH (§6, incident 1).
- Gym worktree branch `jk/glm53-pd` on top of `bxyu/harness-benchmark-agnostic` @ bdb2b64 (Gym PR 2175).

---

## 2. Process topology of the P/D deployment (item 1)

### 2.1 Launch chain

`serve_glm_53_pd_swe.sh` (pins p2d2, Mooncake on, 120GB segments) or `serve_glm_53_pd_ats.sh` (pins p4d4, Mooncake off) → exec `serve_glm_53_pd.sh`.

`serve_glm_53_pd.sh`:
- Sets Slurm env: `SBATCH_ACCOUNT=nemotron_n4_post`, `SBATCH_PARTITION=batch`, `SBATCH_QOS=normal`, `SBATCH_TIMELIMIT=04:00:00`, `SBATCH_GRES=gpu:4`, and the reaper comment (`:58-64`).
- Sets `MODEL`, `MODEL_NAME=glm-5.3`, `CONTAINER`, `MOUNTS="/lustre:/lustre,$GYM:/opt/Gym"`, `VLLM_CONFIG=.../glm_5.3.sh`, `GLM_DEEP_GEMM_PATH` (`:66-71`).
- Sets tier and mode knobs (`:73-102`) and `SERVE_URL_FILE=$LOG_DIR/%j.url` (`:105`).
- Runs `bash benchmarks/nemotron_3.5_super/sbatch_external_vllm.sh` with **no eval arguments**, which means serve-only (`:107-112`).

`sbatch_external_vllm.sh` builds three heredoc scripts (`eval_command` `:254-349`, `vllm_command` `:351-1117`, `batch_command` `:1119-1289`). It then submits once (`:1306-1325`):

`sbatch --parsable --nodes=$NUM_NODES --time=${SBATCH_TIMELIMIT:-04:00:00} --job-name=gym-$EXPERIMENT_NAME-$USER --output=slurm-logs/%j-%x.log --ntasks-per-node=1 --comment="$SLURM_COMMENT" --exclusive [--mem=0 if Mooncake] --segment=$SEGMENT [--nodelist] --wrap 'exec bash -c "$batch_command"'`

`batch_command`:
1. Resolves `nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)` (`:1122`).
2. Writes the URL file (`:1123-1129`).
3. Starts ONE srun step with 1 task per node (`:1139-1151`):
   `srun --nodes=N --ntasks=N --ntasks-per-node=1 --kill-on-bad-exit=1 --container-image=$CONTAINER --container-name=vllm-$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0} --container-writable --container-mounts=$MOUNTS --container-workdir=$SLURM_SUBMIT_DIR --no-container-mount-home bash -c "$vllm_command"`
4. Each task decides its role from `SLURM_PROCID`. Node order is the `scontrol` hostname order: `nodes[0..P-1]` are prefill, `nodes[P..P+D-1]` are decode.

### 2.2 The SWE deliverable: p2d2, coupled, with Mooncake (16 GB200 GPUs)

```
node0  (SLURM_PROCID 0, prefill head)
  mooncake_master  rpc :50051, metrics+/health :9003            (sbatch_external_vllm.sh:651-670)
  vllm serve ... --data-parallel-size 8 --data-parallel-size-local 4 --data-parallel-address node0
       --data-parallel-rpc-port 13345 --api-server-count 1 --host node0 --port 8001     (:784-796)
     -> 1 API server (Rust frontend) + 4 DP engine-core ranks (DP0-3, 1 GPU each, TP1)
     -> NIXL side channel host=node0, port 5600 (+DP index, see §4)
     -> each rank embeds a Mooncake store client with a 120GB host segment
  vllm-router  :8000 (OpenAI API), Prometheus :29000                (:812-823)
node1  (prefill, headless)   vllm serve ... --headless --data-parallel-start-rank 4 (DP4-7)   (:844-853)
node2  (decode head)         vllm serve ... --data-parallel-size 8 --data-parallel-size-local 4
                             --data-parallel-address node2 --data-parallel-rpc-port 13346
                             --api-server-count 1 --port 8001; NIXL side channel port 5700   (:854-865)
node3  (decode, headless)    --headless --data-parallel-start-rank 4 (DP4-7)                 (:866-875)
```

Per-tier details:
- **Head nodes:** the generated args come from `VLLM_COMMON_ARGS + VLLM_{PREFILL|DECODE}_ARGS` (`glm_5.3.sh:70-142`).
- **Headless nodes:** `set_headless_args` strips `--api-server-count` (`:768-782`), because headless ranks reject it.
- **Router (coupled mode)** (`:812-823`): `vllm-router --prefill-policy cache_aware --decode-policy cache_aware --vllm-pd-disaggregation --prefill http://node0:8001 --decode http://node2:8001 --host node0 --port 8000 --intra-node-data-parallel-size 8 --request-timeout-secs 86400 --worker-startup-timeout-secs 3600 --log-level error --prometheus-host 0.0.0.0 --prometheus-port 29000 [ROUTER_ARGS]`.
- **Mode settings:** the serve script sets `VLLM_PD_DEPLOYMENT_MODE=coupled`, `COUPLED_LOCAL_DP_SIZE=4`, and `ROUTER_INTRA_NODE_DATA_PARALLEL_SIZE = NUM_PREFILL_NODES*4 = 8` (`serve_glm_53_pd.sh:78-82`). Coupled mode requires equal tier sizes (`:79-80`; launcher check `sbatch_external_vllm.sh:197-204`).
- **p4d4 (ATS wrapper):** the same layout with 4 nodes per tier (DP16×EP16), `--intra-node-data-parallel-size 16`, Mooncake off (`serve_glm_53_pd_ats.sh:24-26`).

### 2.3 Deployment modes the launcher supports

| mode (env) | what each node runs | router workers | used for GLM? |
|---|---|---|---|
| `VLLM_PD_DEPLOYMENT_MODE=independent` (default), `VLLM_ENGINES_PER_NODE=1|2|4` | One full TP engine per node, or 2×TP2 or 4×TP1 engines on ports 8001+i, with GPU masks from `CUDA_VISIBLE_DEVICES` (`:901-973`). **TP is computed as `4 / VLLM_ENGINES_PER_NODE`, so it is hard-coded for 4-GPU nodes** (`:905-912`) | Every engine URL | No (GLM does not fit on one 4-GPU node) |
| `coupled` (P/D only) | One API server per tier on the tier's first node; other nodes headless (`:764-876`) | One URL per tier; `--intra-node-data-parallel-size` = tier DP size | **Yes: SWE p2d2 and ATS p4d4** |
| `hybrid` | Every node runs its own API server for its local ranks: `--data-parallel-size (tier_nodes*R) --data-parallel-size-local R --data-parallel-start-rank (i*R) --data-parallel-address <tier head> --data-parallel-rpc-port 13345|13346 --data-parallel-hybrid-lb` with R=`ROUTER_INTRA_NODE_DATA_PARALLEL_SIZE` (`:975-1003`) | Every node URL; `--intra-node-data-parallel-size R` (=4) | Yes, for unequal tiers (p2d4, p4d2, p4d8, p2d6). **Python frontend only** (§6, incident 3) |
| `VLLM_MODE=aggregated` (+`hybrid` + `VLLM_NODES_PER_REPLICA=k`) | Each k-node replica is one DP×EP engine, each node runs a hybrid-LB API server. `--kv-transfer-config` is stripped (`:1059-1089`). Uses `VLLM_AGGREGATED_ARGS` if defined, else the prefill args | `--policy cache_aware --worker-urls <all node URLs>` (`:1038-1045`) | Comparison runs only (E7/E7b, agg32, agg4x8) |

### 2.4 Ports

| port | process | source |
|---|---|---|
| 8000 | vllm-router OpenAI API (the endpoint clients use) | `sbatch_external_vllm.sh:227` |
| 29000 | router Prometheus `/metrics` | `:228`, `:758` |
| 8001 (+engine index in independent mode) | each vLLM API server (`/v1`, `/health`, `/metrics`) | `:229`, `:939`, `:1027` |
| 13345 / 13346 | DP RPC (`--data-parallel-rpc-port`) for prefill / decode | `:231-232` |
| 5600 / 5700 (+DP index; +engine index in independent mode) | NIXL side channel (`VLLM_NIXL_SIDE_CHANNEL_PORT`), prefill / decode. dfw-vLLM adds `data_parallel_index` to the base port | `:224-225`; `dfw:vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:69-72` |
| 50051 / 9003 | mooncake_master RPC / metrics and `/health` | `:624`, `:658-668`, `:672` |
| 50052 / 50053 | Mooncake disk-offload owner: client RPC / TransferEngine segment | `:717-722` |
| 63000-64000 | Gym server port range (eval mode only) | `:332-333` |

### 2.5 vllm-router: flags, policy, and P/D request flow

**Flags used.** Common to all modes (`:758-762`): `--log-level error --prometheus-host 0.0.0.0 --prometheus-port 29000`, plus any `ROUTER_ARGS` split on whitespace (`:245`, `:759`).
- P/D: `--prefill-policy $ROUTER_PREFILL_POLICY --decode-policy $ROUTER_DECODE_POLICY --vllm-pd-disaggregation`, one `--prefill URL` / `--decode URL` per API server. Defaults are `cache_aware` (`:234-235`, `:1019-1037`).
- Aggregated: `--policy cache_aware --worker-urls ...`.
- Always: `--intra-node-data-parallel-size $ROUTER_INTRA_NODE_DATA_PARALLEL_SIZE --request-timeout-secs 86400 --worker-startup-timeout-secs $ROUTER_WORKER_STARTUP_TIMEOUT_SECONDS` (1800 by default; 3600 from the GLM script, `serve_glm_53_pd.sh:87`).

**Request flow in P/D.** The router sends the request to prefill first, then to decode. One GLM call therefore costs prefill e2e + decode e2e (`notes/NOTES.md:227-228`).
- The prefill request carries `kv_transfer_params {"do_remote_decode": true, "do_remote_prefill": false}`, with `max_tokens` / `max_completion_tokens` / `min_tokens` patched (router-0.1.17 `src/routers/http/vllm_pd_router.rs:442-512, 829-831`).
- The decode request carries the `kv_transfer_params` returned by prefill (`remote_block_ids`, host:5600, pull mode; `notes/NOTES.md:403-404`) with `do_remote_prefill: true` (`:521-579`).

**cache_aware policy** (router-0.1.17 `src/policies/cache_aware.rs:1-60, 338-380, 432-446`; defaults from `py_src/vllm_router/router_args.py:40-44`):
- The router keeps an approximate radix tree per worker over prompt tokens or text.
- If the best prefix-match rate exceeds `cache_threshold` (default 0.3), it routes to the best-matching worker, choosing the least-loaded one on ties. Otherwise it picks the least-loaded worker.
- The whole policy flips to shortest-queue when the **fleet** is imbalanced: `(max_load-min_load) > balance_abs_threshold (64)` AND `max_load > balance_rel_threshold (1.5) × min_load`. NOTES says "cache_aware only rebalances at >64 load difference and >1.5x" (`notes/NOTES.md:88`).
- Other defaults: `eviction_interval_secs 120`, `max_tree_size 2^25`, `health_check_interval_secs 60`, `worker_startup_check_interval 30`, `request_timeout_secs 1800` (HSG passes 86400), `max_concurrent_requests 32768`, `queue_size 100`, retries 5 (`router_args.py:38-105`).
- Nano used `ROUTER_ARGS="--balance-abs-threshold 16 --balance-rel-threshold 1.2"` (`gym/README.md:47`).

**DP-rank pinning (`X-data-parallel-rank`).**
- Router side: with `--intra-node-data-parallel-size N` each worker URL is treated as N addressable DP engines. The policy picks a (url, rank) pair and sends `X-data-parallel-rank: <rank>` on both the prefill and the decode hop (router-0.1.17 `src/routers/http/dp_utils.rs:193-218`, `vllm_pd_router.rs:1771, 1931`). Router-0.1.17 can also discover each worker's DP size from the `engine="..."` labels on `/metrics` (`dp_utils.rs:9-31`); the HSG fork predates this (UNVERIFIED).
- vLLM side, Python frontend:
  - The API server reads the header (`dfw:vllm/entrypoints/generate/base/serving.py:243-250`).
  - `get_core_engine_for_request` uses it **as an index into that API server's own engine list** (`dfw:vllm/v1/engine/core_client.py:1472-1474`, "Engines are in rank order").
  - Without the header, vLLM's internal balancer scores (waiting, running, KV usage, in-flight) (`:1474-1500`). That scatters a trajectory's turns across ranks. Nano lost prefix hits ~98% → ~60% and halved per-request decode that way (`gym/README.md:57`). The memory note on Super SWE profiling reports the same pattern: 97% hits with sticky routing vs 7-19% with random routing.
- vLLM side, Rust frontend: it reads the header as a **global DP rank** (`notes/NOTES.md:93-97`).
- Consequence:
  - In coupled mode one API server owns ranks 0..7, so index and global rank coincide.
  - In hybrid mode each node's API server owns 4 ranks, and the router sends 0..3 to every node. With Rust, every request routed to a tier's second node fails with HTTP 400 "engine-core error".
  - Hybrid mode is therefore Python-only (`glm_5.3.sh:57-65`).

**Router bug, and why HSG builds a patched wheel** (`gym/README.md:188-270`; `gym/build_vllm_router_wheel.sh`):
- Stock router 0.1.15 resets every worker's in-flight load counter every 10 health-check cycles, i.e. 10 minutes at the default 60 s interval (issue vllm-project/router#197). Confirmed in router-0.1.15 `src/core/worker_registry.rs:369-399`.
- Under P/D with `--decode-policy cache_aware` this produced a runaway. One decode node sat at ~100% KV with 200+ requests queued while its peer idled.
  - Decode max/min running ratio: median 9.84, max 333. Run 6800138 stalled at 583/900 rollouts.
  - With the fix the ratio was 1.03-1.34 and KV stayed below 29% (`gym/README.md:253-266`).
- Fix #216 alone is worse for prefill-heavy loads. With honest counters, the fleet-wide imbalance gate stays open under P/D, routing degenerates to shortest-queue, and cached prompts get recomputed. PR #238 applies the load check per request.
- Pinned build: `bxyu-nvidia/router@e7dda77989fc75d878da8071031de58e519d2415` (#216 + #238), Rust 1.95.0 (`build_vllm_router_wheel.sh:32-35`). Build-time asserts (`:119-130, :152-164`):
  - no `LOAD_RESET_INTERVAL` in src
  - 4 unit tests
  - the .so must not contain "Resetting worker loads (cycle"
  - the .so must contain "Attempted to decrement load counter that is already at 0"
- **Router-0.1.17 status (PyPI latest):**
  - It no longer resets loads in `worker_registry.rs` (no `reset_load` call there).
  - It still uses the fleet-wide gate (`cache_aware.rs:346, 365, 443-445`).
  - That matches the "#216 only" state the README warns about (UNVERIFIED in practice).
  - It also lacks the clamp-warning string, so the HSG build check would reject it.
  - dfw-vLLM has no `vllm-router` installed (not found in its `bin/` or site-packages).

### 2.6 Health, readiness and supervision

- **Engine readiness timeout:** `VLLM_ENGINE_READY_TIMEOUT_S = ROUTER_WORKER_STARTUP_TIMEOUT_SECONDS` (`:238-239, :359`). The GLM script uses 3600; the dfw-vLLM default is 600 (`dfw:vllm/envs.py:796-797`).
- **Router:** waits up to `--worker-startup-timeout-secs` for the workers. The launcher only checks `kill -0` after `sleep 5` (`:1052-1056`).
- **Coupled head:** polls `kill -0` on both the prefill API server and the router every second. If either exits, even with status 0, it exits non-zero (`:827-843`); `cleanup_coupled_head` kills local services (`:799-810`).
- **Independent engines:** a 1 s poll loop over all engine pids plus the router (`:956-972`).
- **srun `--kill-on-bad-exit=1`:** any node's failure tears down the whole step (`:1140`).
- **Mooncake:** waits up to 120 s for the master's `/health` (`:672-687`) and 180 s for the disk owner's TCP 50052 (`:732-752`).
- **External readiness probe used by JK's drivers:** `serves() { [ -s "$L/$1.url" ] && curl -s -m 5 "$(cat "$L/$1.url")/models" | grep -q "\"$2\""; }`, i.e. GET `/v1/models` through the router and grep for `"glm-5.3"` (`notes/rp_chain_r3r4.sh:15`, `notes/rp_train_chain.sh:18`).
- **Time to ready:**
  - Router answered `/v1/models` about 13 min after job start (job 8496605, `notes/NOTES.md:155-156`). The script prints "ready ~15-20 min later" (`serve_glm_53_pd.sh:111`).
  - p2d4 and p4d4 served 7 min after start (`notes/NOTES.md:238, 243`); cause unknown (UNVERIFIED).
  - On AGA GB300, weight load took about 70 s (`:398-399`).
- **Correctness gates after "ready":**
  - `probe_tools.py` checks tool and reasoning parsing.
  - The env smoke test gave 10/10 reward on 10 synthetic rows across six harnesses in 204 s (`:155-159`).
  - On AGA, 16 distinct ~2.6k-token prompts were sent through the router and directly to each tier head, and the answers checked (`:400-419`). This is the only check that caught broken P/D KV transfer.

### 2.7 URL files, requeue and restarts

- **URL file:** `SERVE_URL_FILE` with `%j` replaced by the job id. It is written at **batch start** (before the server is up) as `http://<IP of nodes[0]>:8000/v1` (`:1123-1129`).
- **Requeue:** the file is rewritten with the new node0 IP. Clients re-read it on every use (`bench_endpoint.py --job`; the `serves()` probe).
- **Requeues do happen:** "The serve job requeues (Restarts=1 was observed at ~90 min uptime) and the endpoint IP changes" (`serve_scripts/README.md:72-74`).
- **Stale files:** the Gym launcher does not delete the file on exit. The standalone `.slurm` launchers do (`trap rm -f`, e.g. `serve_glm_53.slurm:132`).
- **Container naming:** the pyxis container name is scoped to the attempt, `vllm-$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}` (`:1130-1136`). The reason: "Pyxis ignores --container-image when a named container already exists". The Mooncake SSD dir also carries the restart count (`:705`).
- **No in-launcher restart loop.** A dead service fails the step and the job. Drivers detect it (`rp_train_chain.sh:82` "A GLM that dies mid-window ends that window early").
- **Gym eval resume:** reuse `EXPERIMENT_NAME` and `ROLLOUTS_FPATH` with `++resume_from_cache=true` (`gym/README.md:136`).
- **Sandbox cleanup:** an `afterany` CPU job reaps sandboxes (`:1328-1358`).

### 2.8 Slurm and container details

- `--exclusive` always. `--mem=0` only when Mooncake is on (`:1302-1305`).
- `--segment=$SEGMENT`, default `NUM_NODES`. Comment: "--segment > 0 otherwise the engine will hang on the second or third engine step" (`:1291-1292`). This is the NVL72 block-topology placement, so that MNNVL all2all spans one NVLink domain. AGA retried p4d4 with `SEGMENT=4` (two 4-node blocks, one per tier) to schedule faster (`notes/NOTES.md:433-434`).
  - **dfw runs Slurm 24.05.5 with `TopologyPlugin=topology/tree`** (`scontrol show config`), so `--segment` should be dropped on dfw. Whether dfw rejects it is UNVERIFIED.
- Reaper comment used by HSG:
  - serving: `{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"1440","reason":"other","description":"inference"}}` (`serve_glm_53_pd.sh:63`)
  - tuning: `"720"`, `"benchmarking"` (`notes/submit.sh:34`)

  Whether `other` and `benchmarking` are valid reason codes on dfw is UNVERIFIED.
- `ulimit -Sn 65535` (`:431-434`). `ulimit -c 0`, because a crashed rank dumped a 137 GB core into the Gym checkout (`glm_5.3.sh:53-54`).
- Eval mode only:
  - Gym runs on `nodes[1]`, "to relieve network traffic on the router node" (`:1246-1252`).
  - It needs `--cpus-per-task=$SLURM_CPUS_ON_NODE`, "otherwise we run into a lot of ServerDisconnectedError and ConnectionResetByPeer" (`:1254-1257`).
- Warmed image (`SAVE_WARMED_CONTAINER`, `:1177-1241`): boots once, collects FlashInfer autotune caches, JIT modules and vLLM startup plans from all nodes, and exports a new sqsh. Later runs set `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`, `VLLM_CACHE_ROOT`, `VLLM_ENABLE_STARTUP_PLAN=1` (`:387-409`). This needs the missing helper file (§1).

---

## 3. Every vLLM flag and env var used for GLM-5.3 (item 2)

"dfw?" = whether the flag, value or env var exists in dfw-vLLM 0.29.0, by grep. "fork" = bxyu-fork-only.

### 3.1 P/D engine flags (`gym/vllm_configs/glm_5.3.sh`)

| flag | value | tier | purpose / measured effect | dfw? |
|---|---|---|---|---|
| `--trust-remote-code` | | both | standard | yes |
| `--disable-uvicorn-access-log` | | both | less log spam | yes |
| `--gpu-memory-utilization` | 0.85 common (`:39, :73`); overridden to **0.90 prefill** (`:122`), **0.93 decode** (`:130`) | both | KV capacity. E13 vs E11c: +6% rollouts/min. E12c (prefill 0.90 alone): +14% over E9c. Prefill KV 668k → 872k tokens/rank (+9.2 GiB). Decode 908k → 1.24M (+14.9 GiB). Aggregated DP32 OOMed at 0.85 and needed 0.65 (`NOTES:314-317`) | yes |
| `--distributed-executor-backend mp`, `--data-parallel-backend mp` | | both | no Ray | yes |
| `--enable-auto-tool-choice`, `--tool-call-parser glm47`, `--reasoning-parser glm47` | | both | GLM parsers. The parser choice does not change the response key (`reasoning`). Tool calls 16/16 under every parser tried (`serve_glm_53.slurm:24-57`). **The Rust frontend ignores `--enable-auto-tool-choice`** (`NOTES:98-99`) | yes (glm47 in `tool_parsers/__init__.py:61`, `reasoning/__init__.py:59`) |
| `--enable-chunked-prefill`, `--enable-prefix-caching` | | both | required for agentic prefix reuse | yes |
| `--max-model-len` | 262144 (`GLM_MAX_MODEL_LEN`) | both | the model allows 1,048,576. Must fit one request on one TP1 rank (§7.2) | yes |
| `--kv-cache-dtype` | `fp8_e4m3` | both | 48.3 KB/token/rank measured. **On H100, `FLASHMLA_SPARSE` (SM90/100) accepts only `auto,bfloat16,fp8_ds_mla,fp8`. `FLASHINFER_MLA_SPARSE` (accepts fp8_e4m3) is SM100-only** (`dfw:vllm/v1/attention/backends/mla/flashmla_sparse.py:91-127`, `flashinfer_mla_sparse.py:72-94`). Use `fp8` (= fp8_ds_mla, 656 B/token/layer) | value differs on H100 |
| `--model-loader-extra-config` | `{"enable_multithread_load": true, "num_threads": 96}` | both | fast weight load (AGA: ~70 s). On dfw: 8 engine processes × 96 threads on 128 cores (UNVERIFIED impact) | yes |
| `--enable-expert-parallel` | | both | EP over the tier's DP ranks | yes |
| `--all2all-backend` | `flashinfer_nvlink_one_sided` (`GLM_ALL2ALL_BACKEND`) | both | MNNVL (NVL72) one-sided all2all. **Not usable across H100 nodes.** dfw choices: `naive, pplx, deepep_high_throughput, deepep_low_latency, deepep_v2, mori_*, nixl_ep, allgather_reducescatter (default), flashinfer_nvlink_two_sided, flashinfer_nvlink_one_sided` (`dfw:vllm/config/parallel.py:42-55, 195`) | present; MNNVL-only |
| `--data-parallel-size 4 --data-parallel-size-local 4 --tensor-parallel-size 1 --api-server-count 1` | | both | single-node defaults; the launcher's coupled or hybrid args override them (`:86-89`, launcher `:993-1001`) | yes |
| `--prefix-match-unit` | unset (`GLM_PREFIX_MATCH_UNIT`) | both | coarser prefix-hash unit; must divide the KV block size. "GLM's DSA backends force block size 64, so `--prefix-match-unit` cannot coarsen" (`NOTES:85-86`). Super uses 128 because the Mooncake lookup cost scales with key count (`nemotron_3.5_super.sh:28-29, 56`) | yes (`config/cache.py`) |
| `GLM_EXTRA_COMMON_ARGS` | unset | both | free-form extra flags | n/a |
| `--kv-transfer-config` | NIXL producer/consumer, or MultiConnector[NIXL + MooncakeStore] (§4) | P / D | KV transfer | yes |
| `--max-num-batched-tokens` | **8192** (`GLM_PREFILL_BATCHED_TOKENS`) | prefill | chunk size. 16k → 8k: **+19% rollouts/min** (E9c 49.9 → E11c 59.5), prefill KV 417k → 668k/rank (+11.3 GiB/GPU), local hit 29% → 58%, waiting 179 → 61. 4k: +3% (noise), kept 8k. 16k on the tuned base: −10%. 32k: does not fit (4.23 GiB KV/GPU < 11.82 GiB for one 262k request) (`NOTES:120, 136, 481-482, 499-501`) | yes |
| `--max-num-seqs` | 64 (`GLM_PREFILL_MAX_NUM_SEQS`) | prefill | per prefill rank | yes |
| `--enable-prefill-batching-metrics` | on (`GLM_PREFILL_BATCHING_METRICS=1`) | prefill | prefill fill-ratio and underfill-reason metrics (§5) | **fork only** (absent in dfw) |
| `--speculative-config` | `{"method":"mtp","num_speculative_tokens":3,"num_speculative_tokens_per_batch_size":[[1,4096,0]]}` | prefill | "Prefill must build the MTP layer's KV for decode, but has nothing to draft" (`:139-140`). Costs ~6.2 GiB/GPU of prefill KV (E2 555k → E3 417k) | yes (`config/speculative.py:473`: `(start,end,n)` inclusive batch-size ranges) |
| `--compilation-config` | `{"cudagraph_mode":"FULL_DECODE_ONLY"}` | decode | full CUDA graphs for decode steps. The memory note on Super SWE profiling also favours FULL_DECODE_ONLY | yes |
| `--max-cudagraph-capture-size` | `max_num_seqs*(1+MTP)` = 64×4 = **256** | decode | capture exactly the decode step size (`:111, :126`) | yes |
| `--max-num-batched-tokens` | `max(256, 2048)` = 2048 | decode | | yes |
| `--max-num-seqs` | 64 (`GLM_DECODE_MAX_NUM_SEQS`) | decode | per decode rank (512 per p2d2 tier). Observed ~25-30 running/rank at 64-94% KV (`NOTES:135, 483-487`) | yes |
| `--speculative-config` | `{"method":"mtp","num_speculative_tokens":3}` | decode | **2× per-request decode** (§7). Acceptance per position 0.80 / 0.64 / 0.53, mean length 2.97 (`NOTES:294-295`) | yes |
| `VLLM_AGGREGATED_ARGS` | `--max-num-batched-tokens ${GLM_AGGREGATED_BATCHED_TOKENS:-8192} --max-num-seqs 64` + MTP 3 | aggregated | used by the agg comparison runs | yes |
| launcher-added | `--served-model-name glm-5.3 --host <hostname> --port 8001`; coupled/hybrid DP args (§2.3); headless `--headless --data-parallel-start-rank`; profiling `--profiler-config '{"profiler":"cuda","max_iterations":20}'` (`:578`); FlashInfer dump forces `--enforce-eager` (`:526-533`) | | | yes |

### 3.2 Env vars (P/D path)

| var | value | set at | purpose / note | dfw? |
|---|---|---|---|---|
| `VLLM_USE_FASTOKENS` | 1 | launcher `:357` | fast tokenizer path | yes (default 0, `envs.py:723`) |
| `VLLM_CACHE_INVENTORY_INTERVAL` | 0 | `:42, :358` | debug cache inventory scans | fork only |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `ROUTER_WORKER_STARTUP_TIMEOUT_SECONDS` (1800; GLM 3600) | `:239, :359` | long weight load | yes (default 600) |
| `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` | 180 | `:361-362` | "reduce connection reset errors between vllm-router and the prefill/decode instances" | yes (default 5, `envs.py:1116-1117`) |
| `VLLM_USE_RUST_FRONTEND` | 1 in coupled mode, 0 in hybrid (`GLM_RUST_FRONTEND`) | `glm_5.3.sh:61-65`; `serve_glm_53_pd.sh:85, 94` | Rust API server. **+14% on coupled p2d2** (E9c 49.2 vs E10c 43.1 rollouts/min). No accuracy cost on the 0.32 fork (79.0% both). **Launcher comment: Rust frontend in vLLM 0.29.0 has a "1-2% delta on SWE Verified"** (`:364-365`) | yes (`vllm-rs` binary present in dfw env) |
| `VLLM_USE_V2_MODEL_RUNNER` | 1 (`GLM_V2_MODEL_RUNNER`) | `glm_5.3.sh:56` | GLM measured with V2. Several bxyu configs set 0 because "V2 model runner ... has quite a large speed regression" in 0.29.0 (`nemotron_3_ultra.sh:13-14`, `batched.sh:55-56`, `glm_5.3_flash.sh:13-14`). In dfw-vLLM, features the V2 runner does not support (e.g. DBO, `config/vllm.py:2539-2592`) silently fall back to the V1 runner with a warning (`:668-674`); grep the log for "Model Runner V2 does not yet support" | yes |
| `NVSHMEM_DISABLE_NCCL` | 1 | `glm_5.3.sh:67` | "Cross-node EP within one NVL72 rack" | n/a |
| `VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL` | 1 | `glm_5.3.sh:68` | DeepEP low-latency kernels over MNNVL (NVL72) | yes; GB200-only meaning |
| `NCCL_CUMEM_ENABLE`, `NCCL_MNNVL_ENABLE`, `NCCL_NVLS_ENABLE` | 1, 1, 1 | `:378-380` | "Helpful NCCL env vars to set on modern clusters". MNNVL is NVL72-only; NVLS needs NVSwitch | n/a |
| `UCX_TLS`, `UCX_RNDV_SCHEME`, `UCX_RNDV_THRESH`, `UCX_NET_DEVICES` | see §4 | `:367-375` | NIXL transport | n/a |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` / `_PORT` | hostname / 5600 (P), 5700 (D) | `:787-788` etc. | NIXL handshake | yes (default port 5600) |
| `PYTHONPATH` (prepend) | `$GLM_DEEP_GEMM_PATH` | `glm_5.3.sh:33-35` | stock v0.32.0 deep_gemm overrides the vendored copy | dfw has top-level `deep_gemm 2.5.0+local` with matching signatures (§6) |
| `ENABLE_MOONCAKE`, `MOONCAKE_*`, `MC_TE_FILTERS`, `WITH_NVIDIA_PEERMEM=0`, `MOONCAKE_CONFIG_PATH` | see §4 | | | yes (mooncake 0.3.13, `mooncake_master` in env bin) |
| `GLM_KV_LEASE_S` | 180 config default; **600** serve-script default | `glm_5.3.sh:101-102`; `serve_glm_53_pd.sh:100-102` | NIXL lease (§4) | `kv_lease_duration` yes (default 30 s!) |
| `ROUTER_WORKER_STARTUP_TIMEOUT_SECONDS` | 3600 | `serve_glm_53_pd.sh:87` | | n/a |
| `ulimit -c 0`, `ulimit -Sn 65535` | | `glm_5.3.sh:54`; launcher `:431-434` | core dumps, fd limit | n/a |

### 3.3 The aggregated TP8 launcher (`serve_scripts/serve_glm_53.slurm`, 2 GB200 nodes)

`vllm serve` with:
- `--served-model-name glm-5.3 --trust-remote-code`
- `--gpu-memory-utilization 0.90 --tensor-parallel-size 8 --nnodes 2 --node-rank $SLURM_PROCID --master-addr <head IP>`
- `--max-model-len 262144 --max-num-seqs 128 --kv-cache-dtype fp8_e4m3`
- `--speculative-config.method mtp --speculative-config.num_speculative_tokens 5`
- `--tool-call-parser glm47 --reasoning-parser glm47`
- `--model-loader-extra-config '{"enable_multithread_load":true,"num_threads":96}'`
- `--enable-auto-tool-choice --enable-prefix-caching --no-enable-flashinfer-autotune`
- rank 0: `--host 0.0.0.0 --port 8000`; other ranks: `--headless`

(`:94-113, 182-209`). Env: `VLLM_ALLREDUCE_USE_FLASHINFER=1`, `VLLM_ENGINE_READY_TIMEOUT_S=3600`, `VLLM_USE_V2_MODEL_RUNNER=1`, `VLLM_USE_RUST_FRONTEND=0` (`:115-118`). The image is `vllm-openai:v0.30.0___with_gym_mambamooncakefix.sqsh` (`:91`).

Multi-node mechanics:
- `srun --mpi=pmix --container-mounts=/lustre:/lustre --no-container-mount-home`.
- The head IP is handed over through a file in the log dir (`:156-166`).
- `GLOO_SOCKET_IFNAME`/`NCCL_SOCKET_IFNAME` are set to the interface carrying the head IP (`:168-179`).
- **`unset VLLM_PORT`** (`:180`).
- Readiness: grep the log for `Uvicorn running on|OpenAI server is ready to accept requests|Application startup complete` (`:215`).

Measured: 1.37 M KV tokens for the whole 8-GPU engine. MLA KV is replicated on every TP rank (`NOTES:46-47`). No load test is recorded. MTP 5 here contradicts the P/D finding (MTP>3 unstable), so its stability is UNVERIFIED.

Response facts (`serve_glm_53.slurm:28-57`):
- Reasoning comes back in `message.reasoning`, not `reasoning_content`. Read both.
- `completion_tokens_details.reasoning_tokens` is unreliable (reported 0 on the 0.30.0 image).

### 3.4 Client side (sampling and template)

- Gym sends `temperature=1.0`, `top_p=0.95` (from `generation_config.json`) and `chat_template_kwargs.reasoning_effort=high` (`glm_5.3.sh:37, 46-51`).
- **The chat template defaults to MAX effort when `reasoning_effort` is absent or not in {low, high}** (`model_meta/chat_template.jinja:2-3`: `effective_reasoning_effort = reasoning_effort if ... in ['low','high'] else 'max'`). It also always emits a `<|system|>Reasoning Effort: ...` line.
- `clear_thinking` defaults to false, so prior reasoning is kept in history when the client sends it back (`:4, :139-146`).
- dfw-vLLM supports `--default-chat-template-kwargs` (`entrypoints/launchers/cli_args.py:174`), which can set `{"reasoning_effort":"high"}` server-side.

---

## 4. NIXL / UCX / Mooncake configuration (item 3)

### NIXL (prefill → decode KV)

- Producer and consumer configs (`glm_5.3.sh:101-102`):
  `{"kv_connector":"NixlConnector","kv_role":"kv_producer|kv_consumer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"kv_lease_duration":<GLM_KV_LEASE_S>}}`
- Pull mode: decode READs prefill's blocks. The prefill response carries `remote_block_ids` plus `host:5600` (`NOTES:403-404`).
- `kv_load_failure_policy: fail` makes a failed load fail the request rather than silently recompute.
- **Lease semantics (dfw-vLLM):**
  - Prefill keeps blocks for a not-yet-pulled request for `kv_lease_duration` seconds. The default is **30** if unset (`nixl/base_scheduler.py:75-78`).
  - Decode heartbeats every lease/6 (`:81`). A late pull logs ERROR "Potentially invalid KV blocks for unrecognized request ... were retrieved by a decode worker. They may have expired." (`nixl/pull_worker.py:426-430`). That is a correctness hazard, not just a warning.
- HSG values: 180 s in Super ("to give nodes a chance to pick up the request. Helps with resilience", `nemotron_3.5_super.sh:60`); 600 s in the GLM serve script.
- Measured: expired-lease events E2 16 (5 in the run table), E3 1, 0 in every tuned run (`NOTES:69, 140-141`). 600 s was kept because simulator/judge traffic is more decode-heavy.
- Side channel: `VLLM_NIXL_SIDE_CHANNEL_HOST=$(hostname)`, `VLLM_NIXL_SIDE_CHANNEL_PORT=5600` (prefill) / `5700` (decode). dfw-vLLM adds the DP index (`base_scheduler.py:69-72`).
- **NIXL metrics are useless as proof:** `nixl_*` metrics were always 0 in the HSG image, including on working runs (`NOTES:412`). Decode counts transferred prompt tokens as *external prefix-cache hits* (953/978 queries, `NOTES:402`).

### UCX (`sbatch_external_vllm.sh:367-375`)

```
export UCX_TLS=rc_x,rc,dc_x,dc,cuda_copy,cuda_ipc
export UCX_RNDV_SCHEME=get_zcopy
export UCX_RNDV_THRESH=0
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-mlx5_0,mlx5_1,mlx5_3,mlx5_4}   # HSG's 4 IB NICs per 4-GPU tray
```

The comment explains why: "otherwise UCX treats VRAM as host memory and NIXL KV-cache registration fails with NIXL_ERR_BACKEND". `UCX_NET_DEVICES` became overridable in commit 576ac2dbe (`NOTES:385`).
- AGA used `rdma_vf_rail0..3` (RoCE). Bare names, `name:1`, and `all` all worked on p1d1 and p2d2 (`NOTES:413-414`).
- Per the launcher comment, NIXL used UCX over the GB200 IB NICs (4 per 4-GPU tray; link speed UNVERIFIED). Whether UCX instead used `cuda_ipc` over MNNVL between trays of one NVL72 rack is UNVERIFIED. Mooncake definitely used RDMA on the `mlx5_*` IB devices (~11 GB/s/node loads, `NOTES:84-85`). dfw has 1×400G NDR per GPU, so per-GPU IB bandwidth is likely comparable to HSG's IB path (UNVERIFIED).

### Mooncake (CPU KV store behind NIXL; `ENABLE_MOONCAKE=1`, P/D only, `:156-159`)

- **Connector configs** (`glm_5.3.sh:103-105`):
  - Prefill: `MultiConnector` with role `kv_producer` over [NIXL producer, `MooncakeStoreConnector` `kv_role:kv_both`, `load_async:true`, `lookup_async:true`].
  - Decode: [NIXL consumer, `MooncakeStoreConnector` `kv_role:kv_consumer`, `load_async`, `lookup_async`, `enable_lookup:false`].
  - In dfw-vLLM that decode config makes the store connector **capacity-only**: it contributes its segment to the pool but transfers no KV (`mooncake/store/connector.py:132-141`).
  - dfw-vLLM also has an untested `save_decode_cache` option. It would write decode-side KV, including generated tokens, to the store (`store/scheduler.py:67, 208, 329`). Unexplored lever; UNVERIFIED.
- **Store config** (`MOONCAKE_CONFIG_PATH`, default `/tmp/mooncake-$SLURM_JOB_ID/config.json`, `:618-631`):
  `{"mode":"embedded","metadata_server":"P2PHANDSHAKE","master_server_address":"<node0>:50051","global_segment_size":"<MOONCAKE_GLOBAL_SEGMENT_SIZE>","local_buffer_size":"4GB","protocol":"rdma","device_name":"<MOONCAKE_DEVICE_NAMES>","enable_offload":false}`
- **NIC selection:** `MOONCAKE_DEVICE_NAMES` (default `mlx5_0,mlx5_1,mlx5_3,mlx5_4`) feeds both `device_name` and `MC_TE_FILTERS` (`:108-110, :633-635`). Without the filter you get "Worker: Cannot make connection for endpoint: ...@mlx5_2, pausing peer rail and retrying through an alternate peer RNIC". Launcher commit 52d075299 (`NOTES:442`).
  - Bug: the disk-offload owner hard-codes `--device_names=mlx5_0,mlx5_1,mlx5_3,mlx5_4` (`:726`).
- **`WITH_NVIDIA_PEERMEM=0`**: "Need these for mooncake connector on GB200" (`:648-649`).
- **Master on node0** (`:651-670`): `mooncake_master -rpc_port=50051 -enable_offload=false -rpc_thread_num=4 -metrics_port=9003 -eviction_high_watermark_ratio=0.9 -eviction_ratio=0.1 -minloglevel=1 -enable_metric_reporting=false -default_kv_lease_ttl=180000 -logtostderr`.
  - `MOONCAKE_CACHE_TRACE=1` sets loglevel 0 (hash-removal summaries).
  - The Mooncake lease (`-default_kv_lease_ttl`, in **milliseconds**) is separate from the NIXL lease: 180000 without disk offload, 30000 with (`:23-24, :121, :149, :155`).
- **Install:** uses the image's baked build if `/opt/super-vl-evals/mooncake-build.env` exists. Otherwise it runs `uv pip install --system mooncake-transfer-engine-cuda13` at job start, then an import check (`:637-646`). `custom_mooncake_version.sh` builds `bxyu-nvidia/Mooncake` main with CUDA 13 (TE+Store, no EP/Go/Rust/P2P). It uninstalls the PyPI wheels and does a TCP put/get smoke test; RDMA needs a real P/D run (`:17-157`).
- **Host memory pinning and the host-OOM incident:**
  - Each engine process, decode included, pins `MOONCAKE_GLOBAL_SEGMENT_SIZE` of RDMA-registered host RAM (`:106-107`).
  - With 4 GPUs per node, the 150GB default gave 4×150GB (NOTES: "644 GB") plus vLLM. Step MaxRSS reached 853 GB on a 920 GB node: 18 oom_kill events at segment mount (job 8496487, `NOTES:147-151`).
  - Fix: `MOONCAKE_GLOBAL_SEGMENT_SIZE=120GB` (1.92 TB store over 16 ranks; MaxRSS 781 GB). SWE runs filled 1.80 TB of 2.58 TB at 150GB. "If a node still OOMs, use 100GB" (`serve_glm_53_pd.sh:89-92`). `--mem=0` is required (`:1302-1305`).
- **Performance:**
  - `batch_is_exist` takes ~1 ms per ~400 keys. Loads take ~40 ms at ~11 GB/s/node.
  - The 0.8-8 s "lookup latency" is queueing: lookups are serial per engine (one-thread executor + ZMQ REQ/REP to one lookup server), waiting behind long, DP-lockstep prefill steps (`NOTES:84-86`).
  - Feedback loop: every request waiting on a Mooncake load already holds GPU blocks for its whole prompt. With ~180 waiting × ~50k tokens > 3.3M tier tokens, those holds evict cached prefixes, local hits fall to ~29%, and more loads are needed (`NOTES:125-129`).
- **Disk offload** (`ENABLE_MOONCAKE_DISK_OFFLOAD=1`; never used for GLM):
  - One standalone `mooncake_client` owner per node, `--global_segment_size` = `MOONCAKE_CPU_CACHE_GIB` (600) GiB; engines get segment 0.
  - Node-local SSD only; tmpfs/overlay/nfs/lustre are rejected (`:689-695`). Quota `MOONCAKE_DISK_QUOTA_GIB` (512).
  - `MOONCAKE_OFFLOAD_*` env (`:705-718`): eviction watermark 0.7, LRU bucket, heartbeat 1 s, 1280 MiB DirectIO staging. `MOONCAKE_PREFERRED_SEGMENT=<ip>:50053` ("the TransferEngine address, NOT the client RPC port").
- **Where Mooncake helped and where it hurt:**
  - Required for SWE on HSG: E1 (no Mooncake) was half E2's pace (`NOTES:68-69, 83`).
  - No upside for short simulator/judge traffic (prefill KV 3% used; −1.1 s prefill without it), so off for ATS (`NOTES:307-313, 371`).
  - **Silently corrupted 7/16 answers over AGA RoCE.** AGA defaults to off (`NOTES:446-450`).
  - Doesn't help when local hits are already ~99% (Nano, `gym/README.md:61`).

---

## 5. Profiling methodology (item 4)

### 5.1 What is scraped

- **vLLM Prometheus on every API server, port 8001.** Coupled mode has one per tier; headless nodes refuse the connection and are skipped. Counters and gauges used by `notes/glm_sampler.py:16-30`:
  - `vllm:prompt_tokens_total`, `vllm:generation_tokens_total`
  - `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` (local)
  - `vllm:external_prefix_cache_hits_total` / `_queries_total` (Mooncake on prefill; NIXL-delivered tokens on decode)
  - `vllm:e2e_request_latency_seconds_{sum,count}`, `vllm:request_queue_time_seconds_{sum,count}`, `vllm:time_to_first_token_seconds_{sum,count}`
  - gauges `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`

  **All of these exist in dfw-vLLM**, along with `vllm:num_requests_waiting_by_reason{reason=capacity|deferred}`, `vllm:spec_decode_num_drafts`, `vllm:spec_decode_num_draft_tokens`, `vllm:spec_decode_num_accepted_tokens`, `vllm:spec_decode_num_accepted_tokens_per_pos`, `vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds`, `vllm:inter_token_latency_seconds`, `vllm:num_preemptions`, `vllm:nixl_xfer_time` (grep of `v1/metrics/loggers.py`, `v1/spec_decode/metrics.py`).
- **Gym → W&B**:
  - The launcher writes `inference-metrics.yaml` with `node{i}: http://<node>:8001/metrics`, the router at `:29000/metrics` and Mooncake at `:9003/metrics` (`:274-305`). "Scrape each API server directly; the router only exposes its own metrics" (`:274`).
  - W&B keys: `vllm/node{i}/{prompt_tokens_per_second, generation_tokens_per_second, kv_cache_usage_perc, num_requests_running, num_requests_waiting, num_requests_waiting_by_reason/reason/{capacity,deferred}, prefix_cache_{hits,queries}_per_second, external_prefix_cache_{hits,queries}_per_second}`, plus `progress/completion_pct`, `progress/total/rollouts_per_min`, `progress/<agent>/reward` (`notes/wandb_summary.py:45-71`). Project `nvidia/jkyi-gym-eval`.
- **Log stats lines, every 10 s.** Python: `(APIServer pid=N) INFO MM-DD HH:MM:SS ... Engine NNN: Avg prompt throughput: X tokens/s, Avg generation throughput: Y tokens/s, Running: R reqs, Waiting: W reqs ... GPU KV cache usage: K%`. Rust: `RustFrontend pid=N ... Avg prompt tput: X toks/s, Avg generation tput: Y toks/s, Reqs Running: R, Waiting: W ... GPU KV cache used: K%` (regexes in `notes/swe_tier_stats.py:16-17`).
  - Identify the tiers by which throughput is non-zero (`gym/README.md:151-152`; `swe_tier_stats.py` treats a source as decode if its mean generation rate exceeds 100 tok/s).
  - The SpecDecoding log lines give MTP acceptance per position (`NOTES:294`).
- **Fork-only prefill batching metrics** (`--enable-prefill-batching-metrics`), which drove the key diagnosis (`NOTES:102-114`):
  - new tokens per prefill forward vs budget: 1,829 / 16,384
  - batch fill ratio (11%) and underfilled batches (99.7%)
  - underfill reasons: `kv_load_pending` / `kv_lookup_pending` / `kv_capacity` (72% / 70% / 7%)
  - forwards per second per engine (~1.15)
  - mean queue vs prefill time per request (1.4 s / 0.7 s)

  Conclusion: "Prefill is not FLOP-bound". **Not available in dfw-vLLM.**
- **Startup memory:** "GPU KV cache size" and KV tokens per rank from the logs (§7.2). "Check 'GPU KV cache size' in the log before assuming a longer context fits" (`serve_glm_53.slurm:72-74`).
- **Error counters** (`notes/status.sh:20-27`): latest stats line per API server, counts of `Traceback`, `CUDA out of memory`, ` ERROR `, and the last `Hit N global ...` line.

### 5.2 How the derived numbers are computed

- **Rates:** counter deltas between samples, Δcounter/Δt, summed over all API servers of a tier (`glm_sampler.py:79-104`).
- **Prefix hit rates (token-weighted):** local = Δ`prefix_cache_hits`/Δ`prefix_cache_queries`; external = Δ`external_..._hits`/Δ`external_..._queries`.
- **Mean TTFT / e2e / queue:** Δsum/Δcount per sample. On the P/D prefill tier, TTFT ≈ prefill e2e, because the router sends prefill a 1-token request. **One GLM call = prefill e2e + decode e2e** (`NOTES:227-228`).
- **Per-request decode tok/s:** Σ`gen_tok_s` / Σ`running` over the window's samples (`notes/rp_window.py:29-32`). It is robust to the call mix, unlike e2e. Same idea as "engine throughput ÷ instantaneous batch" in `gym/README.md:169-173`.
  - In aggregated mode `running` includes requests still prefilling, which understates decode speed. Compare e2e there (`NOTES:332-333`).
  - Speed follows requests per decode GPU, not EP size: 92.3 tok/s at 5.0 per GPU, 80.7 at 9.8, 33.2 without MTP at 15.1 (`NOTES:303-306`).
- **Residency via Little's law** (`notes/swe_tier_stats.py`; `NOTES:506-522`):
  - Mean prefill (running + waiting) and decode running, summed over ranks, give the number of rollouts inside each stage.
  - Example e21: prefill 26 + 43, decode 231 → ~305 of 512 rollouts in the LLM. About 40% sit in the sandbox, so the LLM accounts for ~60% of rollout time.
  - This explains why decode speedups only partly reach rollouts/min.
- **rollouts/min:**
  - `notes/swe_rate.py` parses the tqdm "Collecting rollouts: N/T [mm:ss<" bar and reports (N(22 min) − N(5 min))/17, plus done@20/30/40, wall time to all done, and resolved/finished from the final metrics JSON.
  - The W&B `progress/total/rollouts_per_min` mean is another method that gives different numbers: E13 = 62.9 (W&B) vs 57.2 (tqdm) (`NOTES:488-490`). Compare only within one method and one wave, because shared OpenSandbox load moves results by ~12% (`NOTES:501-502`).
- **Training-scale metric:** **unmasked rollouts completed per fixed 30-minute window**, with per-call GLM latency as the primary comparison (env startup eats 2-4 min of each window) (`NOTES:300-302, 335-336`).
- **Accuracy parity:**
  - `notes/row_summary.py` writes per-row (rollouts, masked, passed) counts keyed by sha1 of the input messages. Only counts leave the cluster.
  - `notes/row_compare.py` computes the paired difference over rows judged in both runs, with a 2000-sample bootstrap 95% CI. Run-to-run noise was ±0.016 with CIs within ±0.05 (`NOTES:426-431`).
- **Router-imbalance fingerprint** (`gym/README.md:247-270`): decode max/min running per minute, over minutes where the busiest node has ≥100 running; plus "starved" minutes. An even start that diverges monotonically means the stock router.

### 5.3 Tools and their reuse against a plain vLLM server on dfw

| tool | what it does | reuse on dfw |
|---|---|---|
| `notes/glm_sampler.py <job> <out.tsv> [--interval 30] [--prefill-nodes P]` | Finds the job's nodes via `squeue`/`scontrol show hostnames`, scrapes `http://<node>:8001/metrics` on every node, sums per tier, writes TSV columns `ts tier running waiting kv_usage prompt_tok_s gen_tok_s local_hit external_hit e2e_s queue_s ttft_s`; stops when the job leaves the queue | Works as-is if the dfw API servers listen on **8001** (hard-coded, `:59-60`). For an aggregated server use `--prefill-nodes 0` (everything lands in the "decode" tier). Uses `urllib`, so clear `http_proxy` if set. Python 3.9 OK (parse-checked). Headless or TP-worker nodes are skipped automatically |
| `notes/rp_window.py <tsv> <startZ> <endZ>` | Per-tier window means plus per-request decode tok/s | as-is |
| `notes/swe_tier_stats.py --from 5 --to 15 <log>` | Little's-law residency from stats log lines | Needs the Gym tqdm "Collecting rollouts: 0%" start marker (`:15`) and `-gym-` in the filename. Adapt the start regex to the swe-task harness |
| `notes/swe_rate.py <log>` | rollouts/min from the tqdm bar, resolved/finished from Gym metrics | Gym-specific; adapt to the harness's progress output |
| `notes/wandb_summary.py <job>` | W&B window means (needs `wandb`, `numpy`) | Only if dfw runs log the same keys to W&B |
| `notes/status.sh` | Slurm state, progress, latest stats line per API server, error counts | Change the hard-coded log dir (`:5`) |
| `notes/probe_tools.py <base_url>` | tool-call and reasoning parse check (§8) | as-is (`--model glm-5.3` is the default) |
| `serve_scripts/bench_endpoint.py` | load + correctness smoke test (§8) | as-is, with `--url`/`--model` |
| `notes/row_summary.py` / `row_compare.py` | paired accuracy | Needs Gym-style rollouts JSONL (`responses_create_params.input`, `mask_sample`, `reward`) |
| Launcher nsys capture (`PROFILE_PREFILL=1`) | After `/health` plus 300 s, POST `/start_profile`; the profile records 20 worker steps under `nsys profile --trace=cuda,nvtx,osrt,nccl,cublas,cublas-verbose --nccl-trace=all --pytorch=functions-trace-shapes --gpu-metrics-devices=all --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop` with `--profiler-config '{"profiler":"cuda","max_iterations":20}'` (`:439-590`) | Same recipe works with dfw-vLLM (`profiler_config` present). "Detailed tracing adds overhead; use separate runs for throughput" (`:11`) |
| `gym/diagnostics/{run,collect,analyze,prepare_recipe}.py` | SGLang srt-slurm instrumentation (`sglang:` metric names, `/server_info`, `/start_profile` schema check) | Code is SGLang-only. Reusable ideas:<br>- metrics window at minutes 5-10<br>- 1 s `nvidia-smi` sidecar sampling<br>- **liveness judged by whether scheduler counters still move**: "A crashed scheduler can leave /metrics answering 200 with frozen counters" (`collect.py:148-151`)<br>- kernel-category bucketing of torch traces |

---

## 6. Incidents and gotchas (item 5)

| # | incident | root cause | fix / lesson | ref |
|---|---|---|---|---|
| 1 | **DeepGEMM API mismatch**. All engines died at the profile run: `fp8_fp4_paged_mqa_logits() got an unexpected keyword argument 'clean_logits'` (the prefill kernel also requires `max_seqlen_k`) | Fork image is `vLLM 0.32.0+precompiled`: v0.32.0 Python but binaries, including vendored `vllm/third_party/deep_gemm`, from a newer nightly. Only DSA (GLM's indexer) calls these kernels; Super doesn't | Copy `vllm/third_party/deep_gemm` from the stock v0.32.0 image and prepend it to PYTHONPATH (`GLM_DEEP_GEMM_PATH`); "vLLM prefers an importable top-level deep_gemm over its vendored one". `custom_vllm_version.sh` builds this way: `VLLM_USE_PRECOMPILED=1` + only .py/.txt/.yaml/rust changes allowed (`:32-53`). **dfw:** top-level `deep_gemm 2.5.0+local` signatures (`deep_gemm/_C.pyi:289-345`) match dfw-vLLM's wrappers (`vllm/utils/deep_gemm.py:510-551`, import order `:171-194`); still smoke-test a long prefill | `glm_5.3.sh:28-35`; `NOTES:53-62` |
| 2 | **MTP > 3 crashes.** E15 (MTP 5) crashed at ~13 min: CUDA 700 illegal address in DeepGEMM `fp8_gemm_nt` on DP2 (`deep_jit cuTensorMapEncodeTiled`). TL mtp4 segfaulted in `cudaGraphLaunch` / `CUDAGraph::replay` on a decode rank ~1 min after load. e26 (MTP 8) died at 33:19 with the same DeepGEMM panic on DP1/2/4/7, after running 15% slower (acceptance length 3.4 vs 2.7; 42 vs 62 tok/s/req; each step verifies 9 tokens/seq) | DeepGEMM / CUDA-graph instability on that stack; not root-caused | MTP 3 is the ceiling. MTP 3 runs with identical DeepGEMM and memory were fine. **H100 note:** dfw-vLLM's SM90 paged-MQA-logits indexer natively handles `next_n` ∈ {1, 2, 4} only (`vllm/utils/deep_gemm.py:555-563`). MTP 3 → next_n=4 is native; other values use a flattened path | `NOTES:137, 298-302, 537-545` |
| 3 | **Rust frontend + hybrid DP** (E7): every request routed to a tier's 2nd node got 400 "engine-core error" | Rust reads `X-data-parallel-rank` as a global rank, Python as a local index (§2.5). Probed: T03 accepts rank 5 and rejects 0; T02 the reverse | Use coupled mode (one API server per tier, router intra-DP = tier DP size) with Rust; keep Python for hybrid. Also check tool parsing under Rust (it ignores `--enable-auto-tool-choice`) | `NOTES:91-99`; `glm_5.3.sh:57-65` |
| 4 | **EP24 uneven experts** (p2d6, decode DP24×EP24 over 6 nodes): highest gen tok/s (15.4k) but ~5k generated tokens per call vs ~1.4k elsewhere; e2e 78.5 s; masked 37.8%; pass 0.37-0.61 | 256 routed experts / 24 = 10.67, so 10 or 11 experts per rank; outputs changed. Not investigated | Use EP sizes that divide 256 (8, 16, 32) | `NOTES:348-352, 369-370` |
| 5 | **Aggregated EP lockstep collapse.** Agg DP32×EP32, MTP 3: 4.4-5.4 tok/s per request, 136.5 s per GLM call. Agg DP32 without MTP: 175.3 s. Agg 4×DP8: 35.3 s. P/D: ~20 s | With EP every DP rank steps in lockstep (all2all). Any rank running an 8k prefill chunk (~0.3-0.6 s) holds every rank's decode step; with 32 ranks some rank is prefilling nearly every step. The in-job AGA GLM showed 1.3-6.6 tok/s per request | P/D separates the lockstep groups. Keep lockstep groups small. "Scale with more P/D endpoints, not a bigger EP group" | `NOTES:318-344, 358-372` |
| 6 | **Host OOM** at Mooncake segment mount (job 8496487): 18 oom_kills, MaxRSS 853 GB / 920 GB | 4 engine processes per node × 150GB RDMA-pinned segments (decode too, though capacity-only) | `MOONCAKE_GLOBAL_SEGMENT_SIZE=120GB` → 781 GB; 100GB if needed; `--mem=0` | `NOTES:147-151, 161-166`; launcher `:106-107` |
| 7 | **AGA p4d4 garbage** ("What is 17*23?" → "material. material. ..."): 0/16 correct; raw markup in content; runaway length | Unknown. GB300 kernels are fine: both tier heads answered correctly when bypassing the router. Decode marked prompts as loaded without pulling. Only DP0 or 12/16 decode ranks logged NIXL transfer plans, vs ~230 plans across 16 ranks on HSG's working p4d4. Refuted: UCX device naming; per-layer KV region > 2 GiB. Candidate: rank → NIXL agent mapping for DP ranks ≥ 8 | Tier width matters: DP16 over 4 nodes failed, DP8 over 2 nodes worked, on GB300 only. Use 2 × p2d2 there. **Always run a distinct-prompt answer check per layout** | `NOTES:400-418, 459-469` |
| 8 | **Mooncake on RoCE corruption** (AGA, `rdma_vf_rail0..3`): probe 4/4 tool calls OK, but 9/16 correct on distinct 2.6k-token prompts, twice; no Mooncake error logged. No-Mooncake p2d2 got 16/16 | Mooncake over RoCE (cause unknown) | `ENABLE_MOONCAKE=0` on AGA. GB300 has ~4× the KV per GPU, so the thrash matters less | `NOTES:446-450` |
| 9 | **GPU OOM from MoE workspace**: agg32_mtp3 at util 0.85 → "Tried to allocate 24.19 GiB" (24.03 free) ~7 min into load | Transient MoE workspace for DP32 × 8192 batched tokens (the AGA config leaves room via util 0.65) | Util 0.65 for that layout. Related: Kimi K3 at 0.95 OOMed in the TRT-LLM mxfp4 MoE kernel (2.17 GiB workspace that grows with batch; `--max-num-seqs` doesn't protect) → 0.90 (`serve_scripts/README.md:57-63`). e23: 32k prefill chunks infeasible on p2d2 (4.23 GiB KV/GPU < 11.82 GiB per 262k request) | `NOTES:314-317, 481-482` |
| 10 | Router load-reset runaway (stock router 0.1.15) | #197 (§2.5) | patched wheel e7dda77 | `gym/README.md:188-270` |
| 11 | Prefill thrash without Mooncake (E1): local hits 95% → 65% as contexts grow; each prefill step schedules ~2 long prompts and leaves ~54 unexamined; ~210 deferred per prefill node; decode idle (2-9 running per engine) | 8 × ~1M-token prefill caches too small for SWE's growing contexts (~1.0M input tokens per rollout) | Mooncake, more prefill KV, smaller chunks | `NOTES:68, 114` |
| 12 | Concurrency 1024 swamped prefill (~295 deferred per node, 7.5-8 s lookups), same pace as 512 | prefill + Mooncake lookup queueing | stay at 512-768 on p2d2 | `NOTES:73` |
| 13 | NIXL lease expiry (E2: 5-16 events) | 180 s lease ran out while decode was full | Lease 600 s; tuned (unsaturated) configs had 0 | `NOTES:69, 140-141` |
| 14 | 137 GB core dump into the Gym checkout | crashed rank | `ulimit -c 0` | `glm_5.3.sh:53-54` |
| 15 | Readiness grep never matched on vLLM 0.27.1 | that build prints "Application startup complete"/"HTTP server started", not "Uvicorn running on" | match any of the three, or poll `/health` / `/v1/models` | `serve_super_35.slurm:150-155` |
| 16 | `reasoning_content` empty | the build returns `reasoning` | read both keys; don't trust `reasoning_tokens` | `serve_glm_53.slurm:28-57` |
| 17 | fp8 KV under P/D gave degenerate output (Nano, GB300); P/D runaways filled decode slots | unknown | P/D not recommended for Nano | `gym/README.md:62`; `nemotron_3.5_lightning.sh:28` |
| 18 | DeepGEMM/Triton JIT cache on shared Lustre → `CUDA_ERROR_FILE_NOT_FOUND` when node B loads node A's kernel (`jit/handle.hpp:117`) | shared JIT cache across nodes | node-local `DEEPGEMM_CACHE_DIR`/`TRITON_CACHE_DIR` (e.g. `/tmp/vllm_cache_shared`). The DSv4 launcher persists caches on Lustre instead (`serve_deepseek_v4_flash_0731_vllm.slurm:141-146`), which conflicts; prefer node-local for multi-node DeepGEMM | `serve_glm_5.slurm:31-35` |
| 19 | Multi-node launchers `unset VLLM_PORT` | an inherited `VLLM_PORT` clashes with vLLM's internal port use (UNVERIFIED mechanism) | unset it in the worker shell | `serve_glm_53.slurm:180`; `serve_glm_51.slurm:61`; `serve_ultra_ga_nvfp4.slurm:48, 65` |
| 20 | Gym needed `--cpus-per-task=$SLURM_CPUS_ON_NODE`, else ServerDisconnected/ConnectionReset; `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=180` against router↔worker resets | client and server socket churn | as stated | `sbatch_external_vllm.sh:361-362, 1254` |
| 21 | Policy call timeout 1 h < 3 h harness timeout → long trajectories scored 0 | harness config | raise `opencode_model_call_timeout` to 3 h | `gym/README.md:58` |
| 22 | Sandbox (OpenSandbox pod) OOM kills in every run | harness noise, not serving | ignore when judging serving | `NOTES:89` |
| 23 | HSG login node lacked `cli_filter_config_defaults.lua` (sbatch died in the Lua filter) | HSG-specific | `SLURM_CONF` override | `serve_glm_53_pd.sh:53-56` |
| 24 | Copied launcher aborts at line 78 | missing `save_warmed_vllm_container.py` | remove or guard the line | §1 |

---

## 7. Quantitative lessons ranked by impact, and H100 transferability (item 6)

### 7.1 Ranked lessons

| rank | lesson (HSG, measured) | magnitude | transfers to dfw H100? |
|---|---|---|---|
| 1 | **P/D beats aggregated DP×EP** because aggregated EP steps all ranks in lockstep (`NOTES:358-372`) | Training-scale (8 nodes, 512 in flight): 20.2 vs 136.5-175.3 s per call; 867 vs 279-302 unmasked rollouts / 30 min. Agg 4×DP8 is 1.75× slower per call. SWE evidence is weaker: E7b (agg 2×DP8, Python) ran 44.4 rollouts/min over min 5-25, about the same as E10c (P/D coupled, Python) at 43.1. But E7b had only 869 done at 27 min, vs 1036 at 24:40 (E10c) and 1137 at 25 (E9c, Rust). NOTES concludes "P/D wins" (`NOTES:74, 78-79`) | **Mechanism transfers and likely gets worse.** On H100 every replica spans ≥2 nodes. EP all2all crosses IB per MoE layer (75/forward) instead of NVL72 NVLink, so lockstep steps are longer. P/D additionally needs ≥2 nodes per tier (≥4 nodes, 32 GPUs, per p2d2-like unit) |
| 2 | **Mooncake for SWE prefill-thrash** | E1 → E2 ≈ 2× rollouts/min; local hit collapse prevented | **Transfers, possibly more important.** H100 per-GPU KV is far smaller than GB200's 0.9-1.24M tokens/rank, so thrash starts earlier. dfw is IB (like HSG), not RoCE (AGA's corruption). Host RAM 2 TB/node with 8 engine processes → 8 × ≤120GB pinned. Must re-validate with distinct-prompt checks. Irrelevant for aggregated mode (launcher forbids it, `:156-159`) |
| 3 | **MTP 3** | 2× per-request decode (E2 31 → E3 60-70; sim/judge 33 → 80 tok/s); 40.3 → 20.3 s per call. SWE rollouts/min only +1-3% at first (prefill became the bottleneck) | **Transfers.** dfw-vLLM has GLM MTP (`index_share_for_mtp_iteration` handled in `models/deepseek_v32/nvidia/mtp.py`). SM90 indexer supports next_n=4 natively. Keep 3 and do not try 4/5/8 first. MTP costs ~6 GiB/GPU on a prefill engine (large on 80 GB) |
| 4 | **8k prefill chunks** (vs 16k) | +19% rollouts/min; +11.3 GiB/GPU KV; shorter lockstep steps. 4k ≈ same; 16k −10%; 32k infeasible | **Transfers.** Absolute activation savings matter more on 80 GB. My hypothesis (UNVERIFIED): DSA indexer prefill logits scale with chunk × context. Start at 8192 and try 4096 |
| 5 | **More KV: 0.90 prefill / 0.93 decode** | +6% (E11c → E13); prefill util alone +14% | **Partly.** On H100 the headroom for MoE/all2all workspaces is only a few GiB (OOM risk, incident 9). Start ~0.88-0.90 and raise while watching for OOM under peak load |
| 6 | **Concurrency** 512 → 768 | +15% rollouts/min (overlaps sandbox time); 1024 gave nothing (prefill swamped) | **Transfers** as a method: sweep client concurrency against sandbox capacity. Absolute numbers will differ |
| 7 | **Rust frontend in coupled mode** | +14% (one Python API server for 8 ranks is a bottleneck); no accuracy cost on the 0.32 fork | **Conditional.** Helps only when one API server fronts many DP ranks. dfw-vLLM is 0.29.0, which the launcher says had a 1-2% SWE-V accuracy delta with Rust (UNVERIFIED for this build). Start with Python; A/B later with `probe_tools.py` plus an accuracy check |
| 8 | **Node scaling within a P/D unit** | SWE: p4d4 +6% for 2× GPUs, p2d4 +4% for 1.5×, p4d2 +0%; sim/judge 4 → 8 nodes −5% per call | **Transfers:** add whole serving units (several endpoints, sessions pinned per endpoint) rather than growing tiers |
| 9 | **EP group size**: EP16 per tier is the sweet spot; EP32 decode slower (70 vs 83 tok/s); EP24 broken | | **Transfers, stronger on H100.** Cross-node all2all grows with EP; keep EP ∈ {8, 16, 32} dividing 256 |
| 10 | **Router**: cache-aware + DP-rank pinning + patched load accounting | Stock router: decode imbalance up to 333× and stalls. Nano: prefix hit 98% vs 60% unpinned | **Transfers fully.** Any multi-rank or multi-replica dfw deployment needs pinned routing. dfw-vLLM honors `X-data-parallel-rank` as a local index (Python). Use the e7dda77 fork wheel or raise `--balance-abs-threshold` |
| 11 | **NIXL lease 600 s, `kv_load_failure_policy: fail`** | expiries 16 → 0 | **Transfers.** Note dfw-vLLM's default lease is 30 s, so always set it |
| 12 | **fp8 KV, max_model_len 262144** | 48.3 KB/token/rank (GB200 fp8_e4m3) | **Changes:** H100 needs `fp8` = fp8_ds_mla (≈54.7 KB/token) or bf16 (≈93.9 KB/token) (§7.2) |
| 13 | **Bigger MTP (8) or bigger chunks (16k/32k)** | both worse | do not retry first |

Do not transfer (GB200/NVL72-specific):
- `--all2all-backend flashinfer_nvlink_one_sided`
- `NCCL_MNNVL_ENABLE=1`, `VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL=1`, `NVSHMEM_DISABLE_NCCL=1` (cross-node rationale)
- `--segment`
- `WITH_NVIDIA_PEERMEM=0` (GB200 Mooncake note; dfw GPUDirect-RDMA status UNVERIFIED)
- HSG NIC names `mlx5_0,mlx5_1,mlx5_3,mlx5_4`. dfw IB NICs are `mlx5_0,1,2,4,5,6,7,8`; **exclude `mlx5_3` (100 GbE)**.
- The fork-only `--enable-prefill-batching-metrics`
- 4-GPU assumptions:
  - `COUPLED_LOCAL_DP_SIZE=4` (`serve_glm_53_pd.sh:81`)
  - `ROUTER_INTRA_NODE_DATA_PARALLEL_SIZE=NUM_PREFILL_NODES*4` (`:82`)
  - `--data-parallel-size(-local) 4` (`glm_5.3.sh:86-87`)
  - independent-mode `TP = 4 / VLLM_ENGINES_PER_NODE` (`sbatch_external_vllm.sh:905-912`)
  - `SBATCH_GRES=gpu:4`

  On dfw these become 8.

### 7.2 H100 memory arithmetic (estimates; confirm from the engine's startup lines)

Basis:
- Weights total = 755,617,140,416 B = **703.7 GiB** (`model_meta/model.safetensors.index.json` metadata).
- Measured GB200 EP8/TP1 = 106.6 GiB per rank (`NOTES:47`). That implies replicated (non-expert) weights R ≈ 21.3 GiB and experts E ≈ 682.4 GiB. Cross-check: 76 MoE layers (75 plus the MTP layer) × 256 × 3 × 6144 × 2048 B ≈ 684 GiB of fp8.
- KV per token per rank:
  - MLA latent 576 B × 79 layers (78 + MTP) + indexer K 132 B × 22 indexer layers = 48,408 B. That matches the measured 48.2-48.4 KB from E13 (39.3 GiB / 872k, 55.7 GiB / 1.24M).
  - H100 `fp8_ds_mla` is 656 B/layer (512 fp8 + 16 B scales + 128 B bf16 RoPE; `flashmla_sparse.py:65-73`) → **≈54.7 KB/token**.
  - bf16 → ≈93.9 KB/token.
- One 262,144-token request needs **11.8 GiB** (GB200 fp8_e4m3; matches `NOTES:481-482`), **13.4 GiB** (H100 fp8), or 22.9 GiB (bf16) on the rank that holds it.
  - MLA KV is replicated across TP ranks (`NOTES:46-47`; Kimi K3 the same, `serve_kimi_k3_longctx.slurm:61-78`), so TP does not shrink this per GPU.
  - Only PP (layer split) or DCP (context split) does.
- H100 budget: ≈79.6 GiB per GPU (UNVERIFIED exact) → 71.6 GiB at util 0.90, 74.0 at 0.93.
- Measured non-KV memory costs on GB200 (absolute, per GPU):
  - 16k vs 8k prefill chunk: ≈11.3 GiB
  - MTP 3 on a prefill engine: ≈6.2 GiB
  - transient MoE workspace in aggregated DP32 × 8192 tokens: 24.2 GiB

| layout (per serving engine) | GPUs | weights / GPU (est.) | KV left per GPU at u=0.90 | fits a 262k request? |
|---|---|---|---|---|
| TP8, one node | 8 | 88.0 GiB | — | **does not fit at all** |
| DP16×EP16, TP1 (HSG-style tier, 2 nodes) | 16 | ≈63.9 GiB | ≈7.7 GiB minus activations/workspace (likely ≤3-5 GiB) | **No** (needs 13.4 GiB on one rank) |
| DP32×EP32, TP1 (4 nodes) | 32 | ≈42.6 GiB | ≈29 GiB minus overheads | yes (~1-2 requests per rank) |
| TP8 (attention) × DP2 × EP16 (2 nodes) | 16 | ≈45-48 GiB | ≈20-25 GiB, replicated over 8 TP GPUs | yes (~1-2 per DP rank) |
| PP2 × TP8 (2 nodes; no cross-node all2all) | 16 | ≈44 GiB | ≈25 GiB; each GPU holds half the layers (≈27.4 KB/token) | yes (~3 per pipeline) |
| + DCP8 on TP8 (shards MLA KV across TP ranks) | | | 8× effective KV vs plain TP8 | code paths exist in dfw-vLLM, UNVERIFIED (see below) |

DCP status: dfw-vLLM requires `dcp_comm_backend=ag_rs` and the fp8 mixed-batch path, i.e. heads per rank < 32 (`flashmla_sparse.py:309-322`). GLM's config hook sets the DCP default `a2a` with q_replicate (`model_executor/models/config.py:43-50`), so `--dcp-comm-backend ag_rs` would be needed. UNVERIFIED end to end.

Implications (UNVERIFIED until a bring-up prints "GPU KV cache size" and "Maximum concurrency for 262,144 tokens"):
- The HSG tier shape (TP1 DP-attention over 2 nodes) cannot serve 256k contexts on H100.
- On dfw it needs ≥4 nodes per engine, or TP8 inside the node with EP/PP/DP across nodes, or a lower `max_model_len`.
- SWE contexts averaged ~50k tokens per call on HSG ("~180 waiting × ~50k tokens", `NOTES:127`), but the tail reaches 256k.

### 7.3 Candidate dfw layouts to test (derived from the lessons; none measured on H100)

| layout | nodes | upside | risks |
|---|---|---|---|
| **A. Aggregated PP2×TP8**, MTP 3, fp8(ds_mla) KV, 8k chunks, prefix caching, one engine per 2 nodes; replicate units behind a cache-aware/sticky router | 2 per unit | No DP lockstep and no cross-node all2all (only activations cross IB between stages). Fits 256k. Simplest to bring up | PP bubbles at low batch; prefill/decode interference inside one engine (HSG lesson 1, though without EP-lockstep amplification). MTP + PP at runtime in 0.29.0 UNVERIFIED. In `config/vllm.py`, EAGLE3+PP only forces the V1 runner (`:2585-2589`), adaptive verification + PP raises (`:2669-2675`), and MTP+PP is not listed |
| **B. Aggregated TP8 × DP2 + EP16**, DeepEP (HT/LL) or `allgather_reducescatter`, MTP 3 | 2 per unit | Closest to vLLM's DSv3-style deployment; DP2 lockstep group is small | Cross-node all2all per MoE layer over IB; needs GPUDirect RDMA for DeepEP/NVSHMEM on driver 535 + compat (UNVERIFIED) |
| **C. P/D with TP8-based tiers** (e.g. prefill = TP8×DP2/EP16 on 2 nodes, decode = same on 2 nodes), NIXL over IB, optional Mooncake | 4 per unit | Removes prefill/decode interference (lesson 1); router pins sessions | Most moving parts. Prefill tier never sees decode output (previous-turn output must be re-prefilled each turn, `NOTES:87`). Validate KV-transfer correctness with distinct prompts (incidents 7, 8, 17) |
| **D. HSG-literal DP×EP TP1 tiers** | ≥8 per unit (4+4) to fit 256k | Same code path as HSG (coupled mode, router intra-DP) | EP32 per tier was slower on GB200; cross-IB all2all; most nodes |

---

## 8. bench_endpoint.py and probe_tools.py (item 7)

`serve_scripts/bench_endpoint.py` is stdlib only. It speaks raw HTTP/1.1 over asyncio, `Connection: close` per request, and ignores proxy env vars (`:221-274`). Each scenario has an answer verifier:

| scenario | request | pass condition |
|---|---|---|
| `plain` | "Explain <topic> in a few sentences." with a unique nonce | non-empty `content` or `reasoning`; liveness only (`:159-170`) |
| `tools` | "What is the weather in <city>? Use the tool." + `get_weather(city, unit)` tool, `tool_choice:auto` | `tool_calls[0].function.name == get_weather`, arguments parse as JSON, `city` matches (`:172-192`) |
| `roundtrip` | turn 1 as `tools`; then append the assistant tool call and a tool result `{"city", "temperature_c": 17+(i%12), "condition": "light rain"}`; turn 2 | final `content` contains the temperature and "rain"; **never streamed** (`:351-396`) |
| `longctx` | ~4 chars/token filler with a 6-digit code at 70% depth; `--prompt-tokens` (actual ≈ 0.73× requested; 175000 → ~127k) | code appears in `content` (`:107-120, :194-202`) |
| `multiturn` | `--turns` alternating messages; turn 1 plants home city + 4-digit traveler ID; last turn asks to repeat both | both appear in `content` (`:123-151, :204-212`) |

Run against any base URL:

```bash
python3 bench_endpoint.py --url http://<host>:<port>/v1 --model glm-5.3 --scenario tools -c 8 -n 16 --max-tokens 4096
python3 bench_endpoint.py --url http://<host>:<port>/v1 --model glm-5.3 --scenario plain -c 64 -n 192 --stream --max-tokens 2048
python3 bench_endpoint.py --url http://<host>:<port>/v1 --model glm-5.3 --scenario multiturn --turns 20 --max-tokens 4096
python3 bench_endpoint.py --url http://<host>:<port>/v1 --model glm-5.3 --scenario longctx --prompt-tokens 175000 -c 8 -n 16 --max-tokens 4096
python3 bench_endpoint.py --job <jobid> --log-dir <dir containing <jobid>.url> ...      # reads the URL file
```

Endpoint resolution order is `--job` (reads `<log-dir>/<job>.url`), then `--url`, then `--host/--port` (`:416-439`). The output reports:
- wall time, ok/failed, median prompt tokens
- input/output tok/s, requests/s
- latency p50/p90/max, TTFT p50/p90 (with `--stream`: first content/reasoning/tool delta, `:277-333`)
- CORRECT x/y, wrong-answer notes, finish_reasons

**The exit code is non-zero on any failure or wrong answer**, so it works as a gate (`:525`).

GLM-specific pitfalls:
- **Defaults are wrong for GLM.** `--model` defaults to `moonshotai/Kimi-K3` (`:452`), `--port` to 12500 (`:450`), `--log-dir` to an HSG path (`:54-55`), `--temperature` to 0.3 (`:461`), `--max-tokens` to 512 (`:457`).
- **No `chat_template_kwargs` is sent, so GLM runs at "Max" reasoning effort** (template default, §3.4). That means long reasoning, and `finish_reason=length` truncations show up as wrong answers for `longctx`/`multiturn` (the verifiers only read `content`). Use `--max-tokens` ≥ 4096, or set the server default `--default-chat-template-kwargs '{"reasoning_effort":"high"}'`.
- `--stream` disables tool-call verification (`:356-357`). The stream parser assumes chunked transfer-encoding.
- Roughly reproducing AGA's P/D correctness check (16 distinct ~2.6k-token prompts): `--scenario longctx --prompt-tokens 3600 -c 16 -n 16 --max-tokens 4096`. Each request gets its own needle. Run it once through the router and once directly against each tier/engine head.

`notes/probe_tools.py <base_url> [--model glm-5.3] [--n 4] [--effort high]`:
- Sends 4 tool requests (temperature 1.0, top_p 0.95, max_tokens 4096, `chat_template_kwargs.reasoning_effort`).
- Prints `finish_reason`, the parsed tool calls, reasoning/content length, and `markup_in_content` (raw `<tool_call>`, `<arg_key>`, `<think>`, `</think>` in content means the parser did not run).
- Ends with a bat-and-ball + arithmetic reasoning check (`:45-81`).
- Uses `urllib`, so honor `no_proxy`. The docstring mentions `--stream` but **argparse has no `--stream`**; passing it errors (`:39-43`).

---

## 9. Other HSG launchers: knobs worth knowing

**Super 3.5 P/D configs** (Gym `vllm_configs/nemotron_3.5_super*.sh`). They show which knobs exist; **no measured outcomes are recorded for these variants in the copy**.

Base (`nemotron_3.5_super.sh`):
- MTP 5 on both tiers; prefill uses `num_speculative_tokens_per_batch_size [[1,1024,0]]` and `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'`.
- Prefill: `--max-num-batched-tokens 67840` (TP4) or 16960 (DP4×TP1); `--max-cudagraph-capture-size 1200`; `--max-num-seqs 1024` or 256.
- Decode: `FULL_DECODE_ONLY` with explicit `cudagraph_capture_sizes` in multiples of 1+MTP=6 up to 6144 (TP4) or 1536 (DP4).
- `--prefix-match-unit 128` (Mooncake lookup cost); NIXL+Mooncake MultiConnector with lease 180.
- `VLLM_USE_RUST_FRONTEND=1`, `VLLM_USE_V2_MODEL_RUNNER=1`, `VLLM_SSM_CONV_STATE_LAYOUT=DS` (Mamba only).

Variants:
- `_dbo`: `--enable-dbo` on prefill, eager mode, `--all2all-backend allgather_reducescatter --moe-backend flashinfer_trtllm`, `FLASHINFER_TRTLLM_MOE_OVERLAP_RESERVED_SMS=8`, `--enable-dbo-metrics`, `TRACE_DBO`.
- `_deepep`: `--all2all-backend deepep_high_throughput --moe-backend flashinfer_trtllm` on prefill, util 0.8, `PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:8`, optional allocation tracing.
- `_eplb`: `--enable-eplb --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":0,"log_balancedness":true,"communicator":"nixl"}'`.
- `_decodedl`: MTP 8, decode capture/batched tokens 2304, prefill batching metrics.
- `_a2abfinvlinkonesided`: `--all2all-backend flashinfer_nvlink_one_sided` on prefill (the backend GLM adopted).
- `_mooncakeevictondirectdescendant`: Mamba-specific eviction.
- `_prefilldev`: prefill 33920 tokens + `PREFILL_FLASHINFER_DUMP_DIR`.
- `_mcfp16_nochunkedprefill`: no chunked prefill, no prefix cache, NIXL only, 131k context.
- `_activestatefp32`: max-num-seqs 512, because "1024 left no memory for KV blocks".

Super decode-speed table (GB200, Sep 2026; `gym/README.md:176-183`):

| batch | engine tok/s | tok/s/req |
|---|---|---|
| ≤16 | 2000 | 130 |
| 32 | 2500 | 80 |
| 64 | 3600 | 60 |
| 128 | 6000 | 45 |
| 256 | 8000 | 30 |
| 512 | 9000 | 15 |

Typical eval shapes (`:31-37`): SWE-V + Multilingual p2d2 at concurrency 1024; SWE Pro p4d6 at 1024; TB2.1 p2d8 at 512.

**Kimi K3** (`serve_kimi_k3*.slurm`):
- TP16 + EP over 4 nodes; `--load-format fastsafetensors`.
- `--attention-config '{"mla_prefill_backend":"TRTLLM_RAGGED","use_prefill_query_quantization":true}'`, `-cc.pass_config.fuse_allreduce_rms=False`, `--max-num-batched-tokens 32768`, `--kv-cache-dtype fp8`.
- `VLLM_ALLREDUCE_USE_FLASHINFER=1`, Rust frontend.
- Master port = `SLURM_JOB_ID % 16384 + 49152`; `GLOO/NCCL_SOCKET_IFNAME` derived from the master IP's interface.
- Util 0.90, not 0.95 (MoE workspace OOM).
- TP32 on 8 nodes gave only 1.29× KV, because MLA is replicated and weights barely shard.
- DCP asserts for K3.
- Ready ~17 min after start (4 min weights + 10 min init).

**DeepSeek V4 Flash 0731, vLLM** (`serve_deepseek_v4_flash_0731_vllm.slurm`):
- Single-node DP4+EP chosen for throughput per the vLLM recipe ("DP gives 4 independent schedulers, 4x the aggregate KV cache, and no attention collective").
- Speculative decoding off for throughput ("spending ~8x the compute per step ... wrong trade once the batch saturates").
- `MAX_NUM_SEQS 512` / `MAX_NUM_BATCHED_TOKENS 16384` are per DP rank; `--block-size 256`; util 0.90; `HF_HUB_OFFLINE=1`.

**GLM-5 FP8 and GLM-5.1 BF16 (TP8 / TP16 over Ray)**:
- `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm`, `--compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'`, node-local JIT caches.
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN=0`, `RAY_CGRAPH_get_timeout=600`, fixed Ray ports.
- Older `--reasoning-parser glm45`.

---

## 10. Recommendations for the dfw H100 build

1. **Runtime.** Start from the proven container plus a *copy* of the dfw-vLLM 0.29.0 env; never modify the RL env. That env already has:
   - GLM DSA/IndexCache: `GlmMoeDsaForCausalLM` → `models/deepseek_v32/nvidia/model.py` on CUDA
   - MTP, glm47 parsers, NIXL 1.3.0, Mooncake 0.3.13 + `mooncake_master`, `vllm-rs`
   - a top-level `deep_gemm` whose `clean_logits` API matches

   Smoke-test a long prefill plus MTP decode before trusting it (incident 1).
2. **KV dtype.** Use `--kv-cache-dtype fp8`, not `fp8_e4m3`. On SM90 the DSA backend is `FLASHMLA_SPARSE`, which takes `fp8_ds_mla`/`fp8`/`bf16`. Check the backend name in the startup log.
3. **Layout.** Do not copy HSG's TP1 DP×EP tiers on 2 nodes; they cannot hold a 262k request (estimate, §7.2).
   - First bring-up: one aggregated 2-node replica, either PP2×TP8 (no cross-node all2all) or TP8×DP2+EP16.
   - Then measure per-request decode tok/s, TTFT and KV capacity under SWE load before considering P/D (§7.3).
   - Use EP sizes that divide 256.
4. **Keep HSG's measured-good engine settings:**
   - MTP 3: `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
   - `--max-num-batched-tokens 8192` (try 4096 if memory-starved)
   - `--enable-chunked-prefill --enable-prefix-caching`
   - `--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47`
   - `--max-model-len 262144` if the startup log shows ≥1× concurrency at 262k; otherwise lower it explicitly
   - `--gpu-memory-utilization` ≈0.88-0.90 at first
   - `--disable-uvicorn-access-log`
   - `--model-loader-extra-config '{"enable_multithread_load":true,"num_threads":<16-32>}'`
   - decode-only engines: `{"cudagraph_mode":"FULL_DECODE_ONLY"}` with `--max-cudagraph-capture-size = max_num_seqs×4`
5. **Reasoning effort.** Set `--default-chat-template-kwargs '{"reasoning_effort":"high"}'`, or have the swe-task client send it. The GLM template defaults to Max. Sampling: temperature 1.0, top_p 0.95.
6. **Environment:**
   - `VLLM_ENGINE_READY_TIMEOUT_S=3600`; `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=180` behind a router
   - `ulimit -c 0`; `ulimit -Sn 65535`; `unset VLLM_PORT`
   - node-local DeepGEMM/Triton/FlashInfer JIT caches (`/raid` or `/tmp`, not Lustre)
   - `GLOO_SOCKET_IFNAME`/`NCCL_SOCKET_IFNAME` from the head IP's interface
   - IB device lists (UCX/NCCL/Mooncake) = `mlx5_0,1,2,4,5,6,7,8`, excluding the 100 GbE `mlx5_3`
   - drop GB200-only settings: `flashinfer_nvlink_one_sided`, `NCCL_MNNVL_ENABLE`, `VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL`, `--segment`
7. **Routing.** If there is more than one DP rank or replica:
   - Front it with `vllm-router --policy cache_aware` (P/D: `--prefill-policy/--decode-policy cache_aware --vllm-pd-disaggregation`), `--intra-node-data-parallel-size` = DP ranks per API server, `--request-timeout-secs 86400`.
   - Prefer the patched fork `bxyu-nvidia/router@e7dda779...`. Stock 0.1.17 keeps the fleet-wide imbalance gate; raise `--balance-abs-threshold` if you use it.
   - Keep the Python frontend; it reads the header as a local index. Consider Rust only for one-API-server-many-ranks layouts, and only after an accuracy A/B on 0.29.0.
8. **If P/D:**
   - NIXL with `"kv_load_failure_policy":"fail"` and an explicit `"kv_lease_duration": 600` (the dfw default is 30 s).
   - `UCX_TLS=rc_x,rc,dc_x,dc,cuda_copy,cuda_ipc UCX_RNDV_SCHEME=get_zcopy UCX_RNDV_THRESH=0 UCX_NET_DEVICES=<IB list>`.
   - Prefill MTP config with `"num_speculative_tokens_per_batch_size":[[1,4096,0]]`; same MTP count on both tiers (matching KV layouts).
   - NIXL side-channel base ports 5600/5700.
   - Mooncake only after the distinct-prompt check passes: ≤120GB segment per engine process, `--mem=0`, `MC_TE_FILTERS` = the IB list.
9. **Validation gates, in order:**
   1. `/v1/models` lists `glm-5.3`
   2. `probe_tools.py` (tool calls parsed, no markup leak, reasoning separated)
   3. `bench_endpoint.py` `tools`, `roundtrip`, `multiturn` with `--model glm-5.3 --max-tokens 4096`
   4. `longctx` with 16 distinct ~3.6k prompts (P/D correctness) and at ~175k (long context)
   5. swe-task synthetic tasks at rising concurrency
10. **Instrumentation:**
    - `glm_sampler.py` against port 8001 at 30 s (`--prefill-nodes 0` for aggregated).
    - Compute per-request decode tok/s, TTFT, local/external hit rates, Little's-law residency.
    - MTP acceptance from `vllm:spec_decode_*`.
    - grep for "may have expired", "CUDA out of memory", "Model Runner V2 does not yet support", preemptions.
    - Keep the startup "GPU KV cache size" / "Maximum concurrency" lines.
11. **Capacity.**
    - Scale by adding whole replicas with per-session pinning, not by growing EP groups.
    - Size client concurrency against sandbox capacity: on HSG p2d2, 768 > 512, and 1024 gained nothing.
12. **Slurm and launcher:**
    - Hold one allocation for debugging; `--exclusive`; reaper JSON with a valid reason code.
    - Write the URL file at job start with the router/head IP.
    - Per-attempt pyxis container names.
    - If porting `sbatch_external_vllm.sh`: fix line 78 (missing helper) and every 4-GPU assumption (§7.1 "do not transfer").

## 11. Open questions / UNVERIFIED

1. Real per-GPU weights, KV tokens and max concurrency at 262k on H100 for PP2×TP8, TP8×DP2×EP16 and DP32×EP32 under dfw-vLLM 0.29.0. Needs bring-up logs; §7.2 is arithmetic only.
2. Does GLM-5.3 MTP 3 run stably on SM90 (FlashMLA-sparse + DeepGEMM SM90 indexer) with PP>1, and with DP×EP? The GB200 crashes were all at MTP>3.
3. Is GPUDirect RDMA (nvidia-peermem or dma-buf) usable inside the container on host driver 535 + forward-compat libcuda 595? It matters for NIXL/UCX, DeepEP/NVSHMEM and Mooncake. HSG set `WITH_NVIDIA_PEERMEM=0` on GB200.
4. Does this 0.29.0 build carry the Rust-frontend SWE-V accuracy delta (1-2%) that the launcher comment describes? Does the V2 model runner regress speed for GLM, as bxyu observed for other models in 0.29.0?
5. Does DCP work for GLM sparse MLA on SM90? It needs `--dcp-comm-backend ag_rs` and the fp8 mixed-batch path; the GLM hook defaults to a2a. If it works, it would multiply KV capacity under TP8.
6. Router: does stock 0.1.17 degrade to shortest-queue under P/D prefill load (fleet-wide imbalance gate)? Can the bxyu fork (Rust 1.95 toolchain, network access) be built for x86 on dfw?
7. Does dfw Slurm 24.05 (`topology/tree`) reject or ignore `--segment`? Which `OccupiedIdleGPUsJobReaper` reason codes are valid (HSG used "other" and "benchmarking")?
8. Is P/D worth ≥4 nodes per unit on H100, compared with aggregated 2-node replicas? HSG's evidence is GB200/NVL72, where EP lockstep is cheaper than over IB.
9. Would Mooncake `save_decode_cache` (present in dfw-vLLM, never tested at HSG) remove the re-prefill of previous-turn output in P/D?
10. Root cause of the AGA p4d4 garbage (DP16 tier over 4 nodes; NIXL plans missing on high ranks). It could recur on dfw if a tier spans 4 nodes.
11. The downloaded HF revision's chat template differs from the HSG copy (10614 vs 10465 bytes). Re-check the `reasoning_effort` default and `clear_thinking` semantics before relying on them.
12. The swe-task synthetic suite's context-length distribution. It decides whether per-rank 262k capacity is actually required.
13. Why p2d4 and p4d4 were ready in 7 min vs ~13-20 min for p2d2 on HSG (warm node-local JIT caches?). This is relevant to how long dfw bring-up takes.
