# swe-task harness x our own GLM-5.3 endpoint: digest (2026-11-06)

Scope: how to run GLM-5.3 served from our own OpenAI-compatible endpoint (`http://<node>:<port>/v1`, served model
name `glm-5.3`, no API key) as the coding agent of the synthetic-SWE harness in
`/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task`, how to smoke-test it, what traffic the hub GLM-5.3 runs
produced (for sizing), the hub baselines, and the risks. Nothing was launched, no Slurm job was submitted, the hub was
never called. Everything below was read from the files cited (paths, line numbers) or computed by the scripts in this
directory over the existing hub journals. Abbreviations: S = `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task`;
TR2 = `S/sft/teacher-rollouts-v2-20261025` (hub GLM-5.3 on the 50 synthetic tasks x 5);
TDW = `S/sft/teacher-deepswe-20261030` (hub GLM-5.3 on DeepSWE v1.1, partial); V29 = the read-only RL vLLM env
`.../swe-rl-v2-sc-migration/runtime/envs/vllm/lib/python3.13/site-packages/vllm` (vLLM 0.29.0).

## 0. Bottom line

1. **The harness needs no code change, only an in-memory model spec.** The synth probe path (`S/synth/rollouts.py`
   `probe()` -> ATP campaign -> Gym -> OpenCode 1.17.11 -> each task's grader) already honours per-model `base_url`,
   `api_key`, `context_window`, `max_output_tokens`, `chat_template_kwargs`, `sampling_overrides` (hengyuf's ATP changes,
   `S/atp/scripts/deepswe_stage.py:296-366`). Spec for our server (no `"local"` flag, no router):
   `{"id": "glm-5.3", "base_url": "${oc.env:GLM53_BASE_URL}", "api_key": "EMPTY", "context_window": 262144,
   "max_output_tokens": 131072, "sampling_overrides": {"reasoning_effort": "max"}}` (+ settings.json's temperature 1.0,
   top_p 0.95). Ready-to-review driver/wrapper/sbatch templates: `template/` (not run).
2. **Correction to the brief:** only TDW ran at 262,144 context (compaction at 242,144). TR2 (the 50 synthetic tasks
   whose pass rates we want to compare with) ran at the global **1,048,576** context (every `probes/glm/*/settings.json`
   in TR2: `context_window 1048576`, no per-spec override), never compacted, and reached prompts of up to **557,931**
   tokens; 112/250 TR2 rollouts (44.8%) went past 242,144 and 79 (31.6%) past 262,144. 131,072 output, reasoning_effort
   max, temperature 1.0, top_p 0.95 hold for both (confirmed on every one of the 82,439 hub request payloads).
3. **Blocking risk: max_tokens 131,072 vs max_model_len.** vLLM <= 0.31 (incl. V29, `renderers/params.py:205-210,451-473`)
   answers 400 when prompt + max_tokens > max_model_len; Gym turns that 400 into an empty `finish_reason: length` reply
   (`vllm_model/app.py:909-916`), OpenCode receives an empty reply and the session ends, and ATP classifies the slot **fatal** (any unresolved 400/401/403/404/422,
   `deepswe_campaign.py:340-342`). On hub TDW traffic that is 19,370/37,731 calls (51.3%) and 239/264 rollouts (90.5%)
   at max_model_len 262,144. vLLM 0.32.0/0.33.0 only reject prompt > max_model_len and clamp max_tokens
   (upstream `vllm/renderers/params.py` at those tags; UNVERIFIED by execution). Use vLLM >= 0.32, or max_model_len
   >= 393,216 (TDW max prompt 249,185 + 131,072 = 380,257), or a clamp proxy (sc14 router style). Observed max
   prompt+output was 253,394, so a clamp would never have shortened a real hub generation.
4. **Capacity is about warm prefixes.** Hub calls were 98.1-99.0% cached prompt tokens; a call adds p50 ~600 new
   prompt tokens and generates p50 180 tokens, but each in-flight rollout keeps a live context of p50 124k (TDW) /
   146k (TR2) / 105k (safe-14 tasks) tokens. Useful rollouts in flight ~= KV-cache tokens / 125k; beyond that each turn
   re-prefills 100-250k tokens.
5. **Baselines:** TR2 hub GLM-5.3 pass@1 **76.7%** (191/249 graded, pass@5 100%); on the 14 tasks whose 5 hub rollouts
   all stayed <= 200k prompt tokens (context-window-insensitive): **91.4%** (64/70); TDW (DeepSWE, 256k, partial):
   **68.9%** (182/264), 74.3% (78/105) on the 21 tasks with all 5 graded.

## 1. How a model is wired into the harness

### 1.1 Call path (who talks to whom)
- Driver (CPU Slurm job) -> `rollouts.probe(probe_dir, task_dirs, models=(alias,), repeats=N)` (`S/synth/rollouts.py:268-328`):
  writes `probe_dir/settings.json` = settings.json + `models={alias: SETTINGS["models"][alias]}` (`:286-288`), stages ATP
  (`deepswe_stage.py --app S/atp --scratch <node-local>`), runs `SYNTH/probe_worker.py probe_dir` (`:319-325`).
- `S/synth/probe_worker.py:169-207`: `create_campaign(..., shards=1)`, `stage()`, sets
  `NVIDIA_API_KEY = api_keys()[0]` (the hub key, `:184`) in the staged env, runs ATP `run_deepswe.run_model` per model.
- ATP `run_model` (`S/atp/scripts/run_deepswe.py:272-353`): `build_config()` -> `gym.yaml`; `gym eval run`;
  `ATP_DEEPSWE_MAX_OUTPUT_TOKENS` = spec max_output_tokens (`:309-310`); `ATP_DEEPSWE_UPSTREAM_STREAM` = "1" only for
  alias `kimi` (`:313`), so every GLM call is non-streaming JSON (hub journals: `upstream_transport: json` on all calls).
- OpenCode runs inside the OpenSandbox agent sandbox: `opencode run --title "NG dummy title" --thinking -- <instruction>`
  with `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=1000000000` (`S/atp/vendor/Gym/responses_api_agents/opencode_sandboxed_agent/app.py:705-718`).
  Its only provider is `nemo_gym` (`@ai-sdk/openai-compatible`, model `nemo_gym/dummy_model`, baseURL = Gym's model
  server on the CPU node, apiKey `dummy_key`, `limit.context = limit.input = context_window`) (`app.py:535-567`);
  ATP's overlay sets `limit.output = $ATP_DEEPSWE_MAX_OUTPUT_TOKENS` and `chunkTimeout = (agent timeout + 300) s`
  (`S/atp/overlays/deepswe-capture.patch:24-39`). OpenCode never sees our server.
- Gym `vllm_model` server (CPU node) -> `policy_base_url + /chat/completions` with `Authorization: Bearer <policy_api_key>`
  (`vllm_model/configs/vllm_model.yaml`; `base_url: ${policy_base_url}`, `api_key: ${policy_api_key}`, `model: ${policy_model_name}`).
  All model traffic leaves from the CPU node; the sandboxes only reach Gym. The CPU node must reach `<gpu-node>:<port>`
  (proven for the sc14 workers, which the per-job router on the CPU node calls at `http://<ip>:8000`).

### 1.2 Spec fields and where they land (`S/atp/scripts/deepswe_stage.py:build_config`)
| spec field | effect | line |
|---|---|---|
| `id` | `policy_model_name` -> request `model` (must equal `--served-model-name`, else 404 -> fatal) | 331 |
| `base_url` | `policy_base_url` (default: settings `base_url` = hub) | 329 |
| `api_key` | `policy_api_key` (default `${oc.env:NVIDIA_API_KEY}` = hub key) | 330 |
| `chat_template_kwargs` | vllm_model `chat_template_kwargs` (sent on every call) | 335 |
| `local: true` | vllm_model `preserve_reasoning_in_assistant_content: true`, `sequential_reasoning_allowed: false`; AND `rollouts.probe()` starts the sc14 student router and waits for sc14 workers (`rollouts.py:295-311`) | 339-340 |
| `sampling_overrides` | merged over `{temperature, top_p, max_tokens}` and pinned on every request | 341-343 |
| `max_output_tokens` | `sampling_overrides.max_tokens` (every request) and OpenCode `limit.output` (`run_deepswe.py:309-310`) | 342 |
| `context_window` | OpenCode `limit.context` = `limit.input` -> compaction at context - 20,000 | 353 |
Staging strips per-model `base_url`/`api_key` (`deepswe_stage.py:262-268`), so `${oc.env:...}` only has to exist when
the campaign runs (as for sc14's `SYNTH_STUDENT_BASE_URL`).

### 1.3 Hosted GLM vs the local sc14 student vs our local GLM
| | hub GLM (TR2/TDW) | sc14 student (settings.json:7-22) | our GLM (recommended) |
|---|---|---|---|
| alias / id | `glm` / `nvidia/zai-org/glm-5.3` | `sc14` / `policy` | `glmlocal` / `glm-5.3` |
| base_url | settings `https://inference-api.nvidia.com/v1` | `${oc.env:SYNTH_STUDENT_BASE_URL}` (per-job router) | `${oc.env:GLM53_BASE_URL}` (our server) |
| api_key | hub key (bound by deepswe_http) | `EMPTY` | `EMPTY` |
| local flag | no | yes (router + `<think>`-in-content history) | **no** |
| context / output | TR2 1,048,576 / TDW 262,144; 131,072 | 262,144 / 131,072 | 262,144 / 131,072 |
| sampling | T 1.0, top_p 0.95, reasoning_effort max | T 1.0, top_p 1.0, `enable_thinking`, `truncate_history_thinking: false` | T 1.0, top_p 0.95, reasoning_effort max |
| router | none | `S/student/router.py` (affinity, normalisation, clamp, abort failover) | none for one endpoint |
| streaming | no (alias != kimi) | no | no |
Why not `local: true` for GLM: it would make `rollouts.probe()` start the sc14 router and block up to
`SYNTH_STUDENT_WAIT_S` (10,800 s) for sc14 workers, and the router would rewrite every call for sc14
(`router.py:88-93,115-150`: model `policy`, top_p 1.0, drops `reasoning_effort`, sc14 template kwargs).

### 1.4 Exact steps (no shared file is edited)
1. Server: `--served-model-name glm-5.3`, `--reasoning-parser glm45|glm47` (same class in V29: `reasoning/__init__.py:55-62`),
   `--tool-call-parser glm47 --enable-auto-tool-choice` (`tool_choice: "auto"` is on every request; without them vLLM
   400s -> fatal), prefix caching on, `--enable-prompt-tokens-details` (so cached tokens appear in usage for comparison),
   no `--api-key`, `--host 0.0.0.0`; max_model_len per section 0 item 3. Do not set default chat-template kwargs
   (jkyi's Gym config `hsg-ref/gym/vllm_configs/glm_5.3.sh:37,50` passes `reasoning_effort=high`; the hub runs used max).
2. A driver modelled on `TR2/driver.py` (`template/glm_local_driver.py` is that, reviewed but never run):
   `os.environ["GLM53_BASE_URL"] = <url>`; `SETTINGS["models"]["glmlocal"] = SPEC` (memory only);
   `rollouts.SYNTH = <dir of the wrapper>`; `rollouts.PROBE_MAX_CONCURRENCY = C`;
   `asyncio.run(rollouts.probe(probe_dir, task_dirs, models=("glmlocal",), repeats=N))`. Drop TR2's
   `key_budget.py` calls (they call the hub). Pre-flight `GET <url>/models` (id + `max_model_len`).
3. Wrapper `template/probe_worker.py`: rebinds `common.api_keys` to a placeholder, then runs `S/synth/probe_worker.py`
   unchanged (same trick as TR2/TDW's key-by-name wrappers, `TR2/probe_worker.py`, `TDW/probe_worker.py:53`).
4. CPU job (`template/glm_local_driver.sbatch`, through the schedule helper's babysit pair like TR2/TDW):
   `cd S/synth && S/atp/runtime/controller/bin/python3 -B <driver> --root <new dir> --base-url http://<ip>:8000/v1 ...`.
   `stage()` refuses non-`cpu*` partitions (`deepswe_stage.py:198-199`) and `run_deepswe.worker` refuses GPU
   allocations (`run_deepswe.py:363-364`), so the harness and the server are always two jobs.

### 1.5 Keeping the hub key out
- A call is "hosted" (hub key bound as Bearer, shared hub cooldown, gateway-capacity handling) only if its URL equals
  `ATP_DEEPSWE_POLICY_BASE_URL + "/chat/completions"`, and that variable is always settings `base_url` = the hub
  (`deepswe_http.py:264-277`, `deepswe_stage.py:131`). Our URL differs, so our calls carry Gym's `Bearer EMPTY` and only
  a model-scoped cooldown (`deepswe_http.py:292-293`). Never point settings `base_url` at our server: that would send
  the hub key to it.
- `probe_worker.py:184` still puts the hub key into `NVIDIA_API_KEY` of the campaign environment (Gym, OpenCode
  launcher); the template wrapper replaces it with a placeholder. `deepswe_stage.runtime_env` keeps reading the key from
  `S/atp/.env` (`:66-71`, required non-empty) for the staging subprocesses (uv / gym prefetch, no model calls).
- No `key_budget.py` / `hub_budget.py` (TDW's guard made "one 4-token GLM call each", `TDW/driver.py:GUARD_INTERVAL`).
- sbatch scrubs `*KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*CREDENTIAL*|OPENSANDBOX_*` (as `TDW/driver.sbatch`).

### 1.6 Is a router needed?
Not for one OpenAI endpoint (one vLLM API server, whatever its internal TP/PP/DP). The sc14 router exists for
multi-worker prefix affinity, sc14 request normalisation, the context clamp and abort failover
(`S/student/router.py:1-60`); it is not reusable as-is for GLM (hard-coded model/sampling, registry-based workers).
Needed only if (a) several replicas/DP ranks must keep a conversation's turns on one prefix cache (vLLM's internal DP
balancer scatters them: `hsg-ref/notes/NOTES.md:50-51`; Gym sends no conversation id upstream, so hash the
conversation opening as `router.py:affinity_key` does), or (b) vLLM <= 0.31 at max_model_len < 393,216 (clamp on 400:
`/tokenize`, retry with `max_tokens = max_model_len - prompt_tokens`, as `router.py:630-669`). Port both
`router.py:131-150` and `:630-669` if (b) is needed: V29 truncates an overflowing prompt at max_input_tokens + 1 before it
raises, so the 400 text only says "at least N" tokens, and vLLM's `/tokenize` chat request does NOT rename
`reasoning_content` -> `reasoning` (`router.py:140`), so the proxy must rename it first or it under-counts the prompt.

### 1.7 Timeouts, retries, failure classes
- Agent: `agent_timeout_seconds` 10,800 per attempt (deadline = min(start + 10,800, job end - 300 s));
  OpenCode chunkTimeout 11,100 s; Gym -> upstream aiohttp `ClientTimeout()` = no read timeout, TCP keepalive
  (`nemo_gym/server_utils.py:160-178`). A proxy in front of vLLM must not cut long non-streaming calls (hub had a
  LiteLLM 360 s cap: 96 + 140 hub 408s "Timeout passed=360.0", retried as fresh samples).
- ATP retries 408/429/500/502/503/504/520 (`deepswe_http.py:24,405-467`; 429 until the deadline, others <= 6 with
  backoff <= 60 s) and every such status halves the model's admission limit for >= 60 s (`deepswe_admission.py:242-262`).
- Unresolved 400/401/403/404/422 -> slot **fatal**, never retried (`deepswe_campaign.py:340-342,622-623`): wrong model
  name, missing `--enable-auto-tool-choice`, unsupported `reasoning_effort` value, context overflow.
- vLLM answers in-flight calls with 200 + `finish_reason: "abort"` when it is stopped; Gym rejects it and the slot is
  lost (`S/student/README.md:408-411`). Never restart the server under a running campaign.
- Verifier: task.toml `timeout_sec` 1,800. ATP admits a slot only with agent timeout + max verifier timeout + 600 s =
  13,200 s (3.67 h) left in the CPU job (`rollouts.py:302-307`): use 8-24 h CPU jobs (TR2 12 h, TDW 23 h).

### 1.8 Concurrency knobs
- Per campaign, fixed at creation: driver `--max-concurrency` -> `rollouts.PROBE_MAX_CONCURRENCY` ->
  `max_concurrency_per_model = min(slots, cap)`, `initial = min(16, max)` (`rollouts.py:256-265`). Pipeline rule cap 32
  (one ATP process stalled at ~125 rollouts: PTY waves, `S/README.md:416-423`). TR2 ran 6 per group x 8 groups = 48;
  TDW glm 8 (w1) / 4 (w2) (`TDW/submit.sh:32`).
- Live: `S/atp/runtime/controller/bin/python -B S/atp/scripts/run_deepswe.py set-concurrency --campaign <probe_dir> --value N`.
- ATP adaptive admission: API limit doubles after 16 successes when saturated, rollout limit doubles only after a
  completed rollout (`deepswe_admission.py:134-140,264-275`); a local server that queues instead of 429-ing lets it
  ramp to the cap. Server: `--max-num-seqs` >= ~0.7 x rollouts in flight (hub: 30 calls in flight at 39 rollouts).
- More throughput = more campaigns (CPU jobs) in parallel, each <= 32.

### 1.9 Output cap and context window (confirmed from files)
| | TR2 (50 synthetic tasks) | TDW (DeepSWE) |
|---|---|---|
| OpenCode context | 1,048,576 (global; `TR2/probes/glm/group-*/settings.json`) | 262,144 (`TDW/driver.py:98,414`; spec `context_window` in `TDW/probes/glm/*/settings.json`) |
| compaction point | never (0 compactions) | 262,144 - 20,000 = 242,144 (`TDW/driver.py` docstring); 71 compaction calls |
| per-call max_tokens / limit.output | 131,072 (all 44,708 payloads) | 131,072 (all 37,731 payloads; `TDW/driver.py:99`) |
| reasoning_effort / T / top_p | max / 1.0 / 0.95 (all payloads) | max / 1.0 / 0.95 (all payloads) |
| tool_choice, tools | auto, 10 tools (compaction calls: no tools) | same |

### 1.10 Reasoning field naming
- Hub replies carry `reasoning_content` (46,188 of 82,439 replies; the rest had no reasoning field = empty think),
  never `reasoning`. vLLM replies carry `reasoning` (V29 `ChatMessage.reasoning`, `chat_completion/protocol.py:61-71`; v0.30.0 measured by jkyi, `hsg-ref/serve_scripts/serve_glm_53.slurm:28-57`).
  Gym reads `reasoning_content or reasoning`, drops both and returns `<think>R</think>` + content to OpenCode
  (`vllm_model/app.py:952-966`): both work.
- History: for non-local specs Gym moves `<think>R</think>` out of assistant content into `reasoning_content` and
  `reasoning` (`app.py:587-617`); ATP's `canonicalize_reasoning` drops `reasoning` (`deepswe_http.py:231-245`); V29
  renames `reasoning_content` -> `reasoning` before validation (`entrypoints/openai/chat_completion/protocol.py:522-547`,
  also present in v0.33.0) and `chat_utils.py:1999-2004` gives the template both names; the GLM template reads
  `m.reasoning_content` (`hsg-ref/model_meta/chat_template.jinja:137-147`). The hub kept history reasoning:
  23,685 (TR2) + 20,239 (TDW) consecutive call pairs grew by the full previous completion, 0 grew by less.
- GLM template defaults = what the hub ran: reasoning effort `Max` unless `low`/`high` (`chat_template.jinja:2-3`),
  `clear_thinking` false (all history thinking kept, `:4,143`), generation prompt ends in `<think>` (`:249-251`). The
  downloaded revision ba32f8c differs from the HSG copy only in tool-response ordering code and one `visible_text` branch.
- `reasoning_effort: "max"` is in V29's Literal (`protocol.py:244-246`) and v0.33.0's `ReasoningEffort`; vLLM also passes
  `enable_thinking=True` to the template (ignored by GLM's).

## 2. Smoke-test plan

Task set: the 50 train-acc-v2 tasks (`S/sft/train-acc-v2-20261024/tasks.json`; 43 accepted + 7 audit-strong probe
copies; unchanged since 2026-10-21, i.e. before the hub runs of 10-25/26; last run successfully 2026-11-04,
`v3etp20`: 250/250 graded, so their OpenSandbox snapshots existed then). Per-task hub table: `per-task-tr2.tsv`.

| stage | what | rollouts / concurrency | hub reference | expected wall |
|---|---|---|---|---|
| 0 | endpoint checks with `replay_hub_requests.py` (below) | 4-5 replayed calls | hub usage of the same payloads | 10-30 min |
| 1 | TR2 pilot pair: `syn-aiohttp-cand-c05-r1b0-7d9ec1f5` (hub 5/5, wall p50 19.7 min) + `syn-argo-cd-cand-c05-cc3e1272` (hub 1/5, p50 38.6 min; reached 307k prompt on the hub, so it exercises compaction at 262k) | 2 x 1, C=2 | pilot: pass / fail | ~1-1.5 h (8 h CPU job) |
| 2 | `safe14`: 14 tasks whose 5 hub rollouts all stayed <= 200k prompt (`traffic-extra.json`) | 14 x 1 (C=14), then 14 x 5 (C=16-24) | 64/70 = 91.4% | x1 ~1-2 h; x5 ~2-3 h |
| 3 (optional load) | all 50 x 5 as 8 groups `tasks[K::8]` | 250, 8 x 6 = 48 in flight like TR2 (or 8 x 4) | 191/249 = 76.7% (1M context) | ~4-8 h |
| 3b (optional like-for-like 256k) | DeepSWE: the 21 tasks with all 5 hub slots graded (TDW per_task.csv / FINAL-REPORT; task dirs in `TDW/tasks/waves.json`; the template needs a taskset for them) | 105 | 78/105 = 74.3% | ~4-6 h |
Wall estimates: hub safe-14 rollouts took p50 18.2 / p90 34.8 / max 84.9 min (LLM-busy 82%), the pilot pair p50 38.2 /
max 80.2 min, all TR2 rollouts p50 35.7 / p90 85.6 / max 179 min; local wall scales with per-call latency (hub decode
p50 94-139 tok/s per request). Agent-timeout exposure if LLM time grows: x1.5 -> 1.6% of TR2 rollouts pass 180 min,
x2 -> 2.0%, x3 -> 6.0% (TDW 0.4 / 1.5 / 7.6%); 3 TR2 rollouts already hit the 10,800 s timeout on the hub.
Statistics: 70 rollouts at p = 0.914 have sd 3.4 pp (95% ~ +-6.6 pp): < 80% on safe14 x5 is a red flag; 250 at
0.767: sd 2.7 pp. Stage 3 is confounded by context (TR2 hub rollouts whose max prompt stayed <= 242,144: 79.7% of 138;
those above: 73.0% of 111; locally the latter compact).

Stage 0 (from a CPU node or the login node; stdlib only):
```
N=/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/notes/swe-task-harness
A=/lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/hengyuf/swe-task/sft/teacher-deepswe-20261030/probes/glm/w1-g0/models/glm/attempts/1-0/attempt-0
/usr/bin/python3 -I -B $N/replay_hub_requests.py --attempt $A                       # dry run: lists first/median/largest/compaction
/usr/bin/python3 -I -B $N/replay_hub_requests.py --attempt $A --base-url http://<ip>:8000/v1 --send --repeat 2 --out $N/replay-<server>.jsonl
```
That attempt's picks: first call 7,485 prompt tokens; median 159,338; largest 242,138 (+131,072 = 373,210 > 262,144:
the overflow test); compaction 202,247 (no tools, cold prefix on the hub: 0 cached, 2,795 output). Pass criteria: all
200; `prompt_tokens_delta_local_minus_hub` == 0 (or a constant few tokens; a large negative delta = history reasoning
dropped or a different template); replies carry `reasoning`; tool calls parse with typed args (hub: `bash.timeout`,
`read.offset`, `read.limit` int; `todowrite.todos` list; `edit.replaceAll` bool); no `<tool_call>`/`<arg_key>` in content;
second send shows cached prompt tokens. V29's glm47 parser coerces arg types from the tool schema
(`parser/glm47_moe.py:56-71`, `parser/engine/parser_engine.py:233-283`).

Sandbox capacity (OpenSandbox Cell-3, shared): every rollout holds one agent sandbox (4 CPU / 16 GiB guaranteed,
`S/README.md:474-476`) for its whole life plus one verifier sandbox at the end. Nominal 140 `sandbox-slot`s
(`S/README.md:467-468`), but creates failed at "used 86/86" and "93/93" on 2026-10-31 while r6 RL shared the cell
(`TDW/README.md:1795`). Right now 20 `deepswe-eval-r8-*` jobs run and 64-node r8 RL jobs are pending (squeue); keep the
smoke at <= 16-24 in flight; ATP halves admission on `sandbox_failed` and retries the slot.

How to read the results:
- Per slot: `<root>/results/glmlocal-<name>.json` (TR2 shape: status pass/fail/infra, reward, f2p, p2p, agent_error,
  verifier_error, attempt_dir) next to `hub_glm_tr2` (the hub's 5 records per task). ATP's own view:
  `run_deepswe.py status --campaign <probe_dir>` (writes rollouts.jsonl/attempts.jsonl/metrics.csv/status.json into the
  campaign). Slot states: completed (graded), exhausted, fatal (e.g. a 400), needs_review (unexplained transport loss).
- Traffic and failure modes, same definitions as the hub baseline:
  `/usr/bin/python3 -I -B $N/traffic_scan.py scan --label <L> --results <root>/results/glmlocal-<name>.json` (or
  `--probe-dir <probe_dir>`), then `traffic_scan.py report --labels tr2,tdw,<L>`. It reports `rollouts_with_context_length_400`
  (hub: 0), `non200_bodies_top`, `rollouts_budget_timeout` (agent timeouts = ATP's generation-validity `budget_timeout`;
  hub TR2 3, TDW 0), `content_leak_markers` (parser failures; hub: 0 in 82,439 replies),
  `tool_calls_with_unparsable_arguments` (hub 0), `tool_arg_error_tool_results` (OpenCode "invalid arguments"
  tool results; hub 63 + 26 = ~0.1% of tool calls), `tool_arg_types_top`, `finish_reasons` (hub: tool_calls + stop only,
  0 `length`), per-call/rollout token and latency distributions.
- Broken endings (last 200 reply empty / `[PAD]` / special token without a tool call):
  `/usr/bin/python3 -I -B /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-deepswe-20261030/brokenreply.py <attempt dirs>`
  (read-only; hub GLM: 0 of 252 TR2 endings, 1 of 264 TDW).
  A context-overflow 400 ends a session with an empty Gym-made reply that is NOT in http.jsonl as a 200, so check the
  400 count separately (scanner).

## 3. Hub GLM-5.3 traffic (from the journals; `traffic_scan.py`, `traffic_extra.py`)

Inputs: every graded slot's final attempt, `http.jsonl` request/response events paired by (call_id, http_attempt):
TR2 250 rollouts / 44,708 replies (127 GB of journals in total were read), TDW 264 rollouts / 37,731 replies.
Full numbers: `traffic-stats.json`, `traffic-extra.json`.

Per rollout (p50 / p90 / max):
| | TR2 (1M ctx) | TDW (256k ctx) |
|---|---|---|
| LLM calls (200 replies) | 154 / 324 / 536 (mean 179) | 134 / 220 / 443 (mean 143) |
| prompt tokens summed over calls | 22.5M / 59.6M / 173.8M | 17.3M / 30.6M / 52.4M |
| uncached prompt tokens summed | 252k / 500k / 1.93M | 252k / 676k / 1.24M |
| completion tokens | 135k / 198k / 346k (mean 143k) | 118k / 176k / 311k (mean 124k) |
| of which reasoning | 95k / 143k / 226k (69% of completion) | 84k / 127k / 241k (72%) |
| max prompt in the rollout | 229k / 334k / 558k | 199k / 242k / 249k |
| time-weighted live context | 146k / 230k / 355k | 124k / 156k / 194k |
| LLM-span wall (first request -> last reply) | 35.7 / 85.6 / 179.0 min | 42.6 / 79.5 / 177.7 min |
| slot wall started.json -> terminal.json | 39.0 / 88.0 / 182.1 min | 43.6 / 81.5 / 182.3 min |
| LLM-busy fraction (sum of call latencies / wall) | 0.66 / 0.89 | 0.74 / 0.90 |
| calls per minute | 4.55 / 7.44 | 3.45 / 4.95 |

Per call (p50 / p90 / p99 / max):
| | TR2 | TDW |
|---|---|---|
| prompt (input) tokens | 162,941 / 284,098 / 423,633 / 557,931 | 133,417 / 209,812 / 238,310 / 249,185 |
| new (uncached) prompt tokens | 576 / 3,036 / 16,013 / 556,596 | 607 / 4,163 / 24,989 / 234,368 |
| completion tokens | 180 / 1,526 / 11,329 / 54,239 | 180 / 1,701 / 13,612 / 39,353 |
| reasoning tokens | 36 / 1,025 / 9,671 / 54,130 | 33 / 1,079 / 13,020 / 38,341 |
| hub latency (s) | 3.5 / 12.2 / 72.2 / 362.6 | 5.2 / 23.6 / 147.5 / 358.7 |
| hub decode tok/s per request (calls >= 1k out) | 139 / 187 | 94 / 139 |
| messages per request | 189 / 473 / 742 / 1,072 | 133 / 290 / 418 / 552 |
| request body bytes | 1.6 MB / 2.7 MB / 3.9 MB / 4.9 MB | 1.3 / 2.1 / 2.5 / 2.7 MB |
Shares: calls with prompt > 131,072: TR2 66.2%, TDW 51.3%; > 200k: 34.1% / 13.9%; > 242,144: 19.4% / 0.06%;
> 262,144: 13.7% / 0. Completion > 4,096: 3.6% / 4.6%; > 16,384: 0.5% / 0.7%; > 32,768: 0.2% / 0.02%; > 65,536: 0.
Cached share of all prompt tokens 99.0% / 98.1%. Tool calls per reply 1.10 / 1.14 (max 14 / 13). Finish reasons:
tool_calls 99.4% / 99.0%, stop otherwise, never `length`. Call kinds: main agent; OpenCode `explore` subagent
("You are a file search specialist", 3.1% / 2.8% of calls); compaction (TDW only).
Compaction (TDW): 71 of 264 rollouts (26.9%) compacted exactly once; compaction calls = 0.19% of calls; their prompt
p50 179k / max 210k with ~0 cached tokens (summarizer system prompt -> cold prefix), output p50 4,120 / max 13,140,
hub latency p50 58 s. The next call restarts from ~17k tokens.
Concurrency (whole campaigns): TR2 rollouts in flight p50 5 / p90 25 / max 39 with LLM calls in flight p50 4 / p90 15 /
max 30; TDW 4 / 12 / 24 rollouts and 3 / 8 / 20 calls (~0.6-0.7 calls per rollout in flight). Gap between a reply and
the next call (tool time): p50 0.4-0.5 s, p90 6.2-6.5 s, p99 121-149 s, max 2,231-2,533 s.
Hub serving fingerprints: vLLM-style ids, block-64 prefix-cache counts, MTP (acceptance 52.9% / 51.1% of drafted
tokens, 0.60 accepted per output token ~ 2.5 tokens per step); rate limits 100M TPM model-wide (TDW: 3,062 429s, 140
408s; ATP throttled GLM to ~1 rollout per campaign and the user cancelled TDW-GLM, `TDW/OPERATOR_NOTE.md:29-50`).

Sizing implications (for the serving build):
- KV to keep prefixes warm ~ rollouts in flight x 125k tokens (safe-14 ~105k; TR2-like long tasks ~150-230k). At HSG's
  ~51 KB/token (fp8 MLA; derived from 1.37M tokens on GB200 TP8, UNVERIFIED on H100), 1M tokens ~ 51 GB of KV on
  every GPU of a TP group (TP replicates the MLA latent cache on every rank: `hsg-ref/gym/vllm_configs/glm_5.3.sh:6`).
- Decode ~0.65 x rollouts concurrent sequences; ~120-140k output tokens per rollout over ~35-45 min at hub speed
  (~45-65 tok/s averaged per rollout); prefill with warm caches ~250-350k new tokens per rollout plus one cold
  150-210k compaction prefill per ~4 rollouts; with cold caches 17-30M prompt tokens per rollout (50-100x more).
- max_model_len per section 0 item 3; `--max-num-seqs` >= expected calls in flight.

## 4. Hub GLM-5.3 baselines
| set | context | graded | pass@1 | other |
|---|---|---|---|---|
| TR2, 50 synthetic tasks x 5 | 1,048,576 | 249 (+1 infra: wasmtime-c03 r1) | **76.7%** (191) | pass@5 100%; per group 70.0-91.4%; 0 broken endings |
| TR2 rollouts with max prompt <= 242,144 / > 242,144 | | 138 / 111 | 79.7% / 73.0% | |
| `safe14` (all 5 hub rollouts <= 200k prompt) | | 70 | **91.4%** (64) | per task in `per-task-tr2.tsv` |
| TR2 pilot pair (1 each) | | 2 | aiohttp-c05-r1b0 pass, argo-cd-c05 fail | group runs: 5/5 and 1/5 |
| TDW DeepSWE (PARTIAL, cancelled) | 262,144 | 264 | **68.9%** (182) | pass@5 82.1% (lower bound); 1 broken ending; compaction 26.9% |
| TDW, 21 tasks with all 5 graded | 262,144 | 105 | 74.3% (78) | `TDW/FINAL-REPORT.md` "Matched task sets" |
Contrast: base sc14 17.2% on the 50 tasks (`TR2/comparison.md`); kimi-k3 77.4%; deepseek-v4-pro 53.2%.

## 5. Risks (most severe first)
1. **Context overflow 400 -> fatal slots** (section 0 item 3). Detect with the stage-0 `largest` replay; fix by vLLM
   >= 0.32, max_model_len >= 393,216, or a clamp proxy (section 1.6). Even with clamping, a prompt > max_model_len is a
   400; OpenCode can overshoot 242,144 by one step's output + tool result (TDW max 249,185), so keep max_model_len >=
   ~262,144 + margin. If the build stays on V29 (the proven runtime with GLM IndexCache), 393,216 also requires one
   393k-token sequence to fit in one KV group at startup, which may not fit a 2-node TP16 replica (UNVERIFIED): then
   the clamp proxy is the practical route.
2. **KV capacity / prefix thrash** (section 3): with ~125k live tokens per rollout, a 2-node replica with
   TP-replicated MLA KV may keep only a handful of rollouts warm; size the smoke's concurrency from vLLM's reported KV
   tokens, not from GPU count.
3. **Request-validation 400s are also fatal**: model id mismatch (404), `tool_choice: auto` without auto tool choice,
   a vLLM build whose `reasoning_effort` Literal lacks `max` (V29 and v0.33.0 accept it; other builds UNVERIFIED).
4. **Silent behaviour drift**: a server that ignored `reasoning_content` in history would render `<think></think>` for
   every past turn (hub kept them); a different chat template or `reasoning_effort=high` defaults (jkyi's Gym config)
   change behaviour. Check prompt-token parity with the replay tool.
5. **Tool-call parsing**: glm45 == glm47 in V29 (same parser class, `tool_parsers/__init__.py:57-64`), with schema-based
   type coercion; untested on our build until the replay/pilot. Hub baseline: 0 leaked markers, 0 unparsable args.
6. **Stream vs non-stream**: GLM calls are non-streaming; long generations (hub max 54k tokens) can take many minutes;
   any proxy/router read timeout (e.g. vllm-router, Rust frontend defaults: UNVERIFIED) would turn them into
   408/502/503 retries (halving ATP admission) or losses. The hub's 360 s cap does not exist locally (slight behaviour
   difference: the hub resampled such calls).
7. **Server restarts/aborts** lose slots (200 + `abort`); drain by ending the CPU job first. Connection errors are
   infra (retryable) but halve admission.
8. **Baseline comparability**: TR2 ran at 1M context; at 262,144 ~45% of TR2-like rollouts would compact. Use safe14
   (or DeepSWE 21 tasks at 256k) for like-for-like; per-group spread on the hub was 70-91%.
9. **Sandbox contention** with r8 RL / DeepSWE eval on Cell-3 (86-93 slots observed full); agent timeout 3 h with a
   slower server (2-8% of rollouts at 2-3x LLM time).
10. **Security**: our server has no API key and listens on 0.0.0.0 (same as the sc14 workers); the hub key never needs
   to reach it.

## 6. Files in this directory
| file | what |
|---|---|
| `digest.md` | this note |
| `traffic_scan.py` | journal scanner (hub baselines `scan`; local campaigns `scan --label L --results/--probe-dir`; `report --labels`) |
| `traffic_extra.py` | per-task table, tool gaps, in-flight timelines, safe-14 selection |
| `traffic-stats.json`, `traffic-extra.json`, `per-task-tr2.tsv` | results of the above (hub) |
| `cache/{tr2,tdw}/*.json` | per-attempt per-call rows (no message content), 49 MB |
| `replay_hub_requests.py` | replay hub payloads against our endpoint (dry run by default; refuses hub URLs) |
| `template/glm_local_driver.py`, `template/probe_worker.py`, `template/glm_local_driver.sbatch` | harness templates (never run) |

UNVERIFIED items: vLLM 0.30/0.31 vs 0.32/0.33 overflow behaviour (read from upstream source at those tags, not
executed); hub template identity with ours (replay checks it); ~51 KB/token KV on H100; current OpenSandbox free
slots; whether the DeepSWE eval jobs use Cell-3; proxy/router default timeouts; list-valued `base_url` in an ATP spec
(not needed, not relied on).
