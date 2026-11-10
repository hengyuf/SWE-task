# MTP drafts of structured-output requests rejected under pipeline parallelism (vLLM 0.33.0)

## Symptom (2026-11-06/07, GLM-5.3, TP8 x PP2 + DCP8 + MTP3, async scheduling, model runner V2)

| traffic (c8) | MTP mean acceptance length | output tok/s |
|---|---|---|
| plain chat | 2.99 | 688 |
| chat + tools, `VLLM_ENFORCE_STRICT_TOOL_CALLING=1` (vLLM default for glm47: xgrammar structural tag on every tool-enabled request) | **1.11** | 178 |
| chat + tools, strict tool calling off (our workaround until now) | 3.18 | 418 (c8 n16) |
| `response_format` json_schema (strict off makes no difference; schema requests are always grammar-constrained) | **1.36-1.40** | - |

Outputs stayed valid (32/32 tool calls parsed, 32/32 schema-valid JSON): drafts are rejected, not accepted wrongly.

## Root cause

* With async scheduling the scheduler never learns MTP draft ids; it schedules every decode with -1 placeholders
  (`AsyncScheduler._update_after_schedule`). For structured-output requests the engine defers sampling until the
  previous step's output is processed, fetches the real drafts from the worker (`take_draft_token_ids`), validates them
  against the grammar and only then builds the grammar bitmask (`EngineCore.step_with_batch_queue`, deferred branch).
* The deferral is triggered by `pending_structured_output_tokens`, i.e. the request still has output placeholders of an
  in-flight step. Under pipeline parallelism model runner V2 microbatches decodes (`decode_stagger = pp_size`): a request
  is scheduled every pp_size steps, so when it is rescheduled its previous step was already processed, it is not
  "pending", and the bitmask is built from the placeholders.
* The worker's `DraftTokensHandler` kept only the latest step's batch, so the request's drafts were overwritten by the
  other micro-batch anyway.
* vLLM #54442 (in 0.33.0) makes this fail closed: `GrammarOutput.num_acceptable_drafts` = number of non-placeholder
  drafts = 0, so every draft row is invalidated in the rejection sampler -> acceptance length 1.0.
* TP16 (no PP) is not affected in principle: the request is scheduled every step, its previous step is in flight, the
  deferred branch runs (to be confirmed by measurement).

## Upstream

* vLLM PR #63219 (merged on main 2026-10-30, 322f90a, NOT in 0.33.0): per-request draft retention in
  `DraftTokensHandler` (+ `begin_step`, `discard`) and a revert of #54442's fail-closed guard. Patch: `../pr63219/`.
* vLLM issue #65430 (open, 2026-11-03): the revert may reopen acceptance of grammar-invalid drafts under async
  scheduling (a draft verified against a permissive row after a placeholder); disputed, unresolved.
* vLLM PR #61326 (open): serialize the PP batch queue for spec decode without async scheduling (a different, non-async
  bug).

## Our fix (`mtp-so-pp-fix.patch`, built into `env/vllm-0.33.0-glm53` by `env/build_glm53_env.sh`)

1. Worker: #63219's per-request draft retention (`vllm/v1/worker/gpu/spec_decode/utils.py`, plus `begin_step` /
   `discard` calls in `vllm/v1/worker/gpu/model_runner.py`).
2. Keep #54442's fail-closed guard (no revert): a draft the scheduler could not validate is still rejected, so the
   grammar guarantee does not depend on the hand-off being right.
3. Engine (`vllm/v1/engine/core.py`, ours): in the non-deferred branch, if a structured-output request of the step was
   scheduled with placeholder drafts, fetch the retained drafts (`take_draft_token_ids`) and validate them
   (`update_draft_token_ids_in_output`) before `get_grammar_bitmask`. The fetch is an RPC queued behind the step's
   `execute_model`, exactly like the deferred branch's fetch.

## Validation on the dev hold (2026-11-07, TP8 x PP2 + DCP8 + MTP3, `VLLM_ENFORCE_STRICT_TOOL_CALLING=1`)

| traffic | 0.33.0 + #60477 only (s1) | + this fix (s3, env/vllm-0.33.0-glm53) |
|---|---|---|
| plain chat c8 n32: acceptance length / tok/s | 2.99 / 688 | 3.01 / 598 |
| chat + tools c8 n32: acceptance length / tok/s, tool calls | 1.11 / 178, 32/32 | **3.13** / 340, 32/32 |
| chat + tools c16 n64 | - | **3.15** / 533, 64/64 |
| json_schema c8 n32: acceptance length, valid, wall | 1.36-1.40, 32/32, 42 s (strict off, s2) | **3.25**, 32/32, 13 s |
| json_schema c16 n64 | - | **3.30**, 64/64 valid, 17 s |

No "grammar terminated or rejected draft" errors and no tracebacks on either rank (the scheduler's validated drafts
matched what the worker verified).
