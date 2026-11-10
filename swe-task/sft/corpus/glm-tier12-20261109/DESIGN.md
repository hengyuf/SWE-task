# GLM-5.3 tier-1/2 SFT corpus: design (main agent, 2026-11-09, rev 2)

User request (hengyuf, GitHub issue hengyuf/SWE-task#1, comment 6758873905, verbatim): "现在收集tier1/2 的 successful
traces （包括有compaction的）， 将其加工为类似SFT v3 一样的dataset， 这里对于一个rollout 的多个segment（有compaction的），
将其视为多条独立的traces，task ID 可以用一个用于标记，但训练时保留各自边界。标准 SFT 可以把 segment 当独立样本，做
packing时，用现有 packer 的边界/attention 语义，别让 loss 或上下文跨样本串起来。pack 好data 把data 用datamover 移动到 AGA
（oci-aga-slurm-1-login-01.nvidia.com）上。（AGA 访问密码与HSG 相同）"

Research: workflow glm-sft-understand (wf_bff6ad79-027); findings in `scratch/RESEARCH-converter-packing.md`,
`scratch/RESEARCH-glm-compaction.md`, `scratch/RESEARCH-aga-transfer.md`; prototypes in `scratch/<lens>/`.
rev 2 adds the compaction lens's findings (prefix chains, truncate_history_thinking, system prompt changes, edge
cases) and the old-grader rule.

## Paths and names
- S = `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task`.
- WORK dir (this dir) = `S/sft/corpus/glm-tier12-20261109`: DESIGN.md, scratch/, regrade/ (old-grader re-grade).
- CORPUS dir (clean, transferred as is) = `S/sft/corpus/sft-glm-tier12-20261109`: tools/, sources/, raw_sft/, blends/,
  selection/, metadata.jsonl, DATA_PREP_COMPLETE.json, README.md, SHA256SUMS (the sft-v3-20261029 layout). Nothing
  else may be in it (the transfer refuses files outside SHA256SUMS). Dry runs go to WORK/scratch/..., never CORPUS.
- AGA DATA_ROOT = `/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_post/users/hengyuf/sft-glm-tier12-20261109`
  (AGA probe 2026-11-09 22:09: /lustre -> /scratch, users/hengyuf -> projects/nemotron_n4_post/users/hengyuf, writable,
  12.9 of 100 TB used). The resolved /scratch path is what the AGA trainer sees: the SFT-v3 GB300 setup
  (sft-v3-b300-20261106: megatron-lm with the SFT loss-mask fixes, container mounts /scratch:/scratch, TP4 CP16 EP32
  PP1, seq 524,288, GBS 32, MBS 1, MTP off) rewrote its v3 blend to /scratch paths. Passed to the builder as
  `--data-root`; baked into `blends/raw_sft.json`, the receipt's shard paths and verify's path check.
- Note: the AGA v3 run trains on sft-v3-20261029-empty-think-prefix-v1 (an HSG transform that masks the `</think>` of
  empty-reasoning turns: 73,633 masked, text unchanged). This corpus is the raw (untransformed) format like v3; an etp
  variant is a separate follow-up if wanted.
- SCHEMA `sft-glm-tier12-hsg-raw-sft-data-v1`; selection file `selection/sft_glm_tier12.jsonl`; shuffle seed 20261109.

## Sources (read-only)
- **tasksall**: `S/sft/glm-rollouts-tasksall-20261108/analysis/rollouts.jsonl` (final records of final_report.py, every
  re-run applied), tier 1 and 2 records (`tier` field). Tier 2 still has 3 `infra` slots (their re-runs were cancelled by
  the tier pause): not passes, listed.
- **sftv3**: `S/sft/glm-rollouts-sftv3-20261107/analysis/final-rollouts.jsonl` (the 220 SFT-v3 tasks, all tier 1/2 by
  inventory label; infra re-run applied).
- Task facts: `S/sft/tasks-all-20261106/all_tasks.jsonl` (tier rule of glm-rollouts-tasksall-20261108/analysis/
  tier12_accept_compare.py: tier 1 = accepted_or_strong, tier 2 = audit partial) and that script's
  `tier12-accept-compare.json` verdict with GLM as the teacher (informational).
- Not included: the hosted GLM-5.3 rollouts (teacher-rollouts-v2-20261025) of the 2 official1-scriggo gomacro tasks
  (5 passes): hub serving, 1M context, 131k output cap, i.e. not the self-hosted 262,144 / compaction-at-242,144 setup.
- Student template and tokenizer: sc14 `.../swe-rl-v2-sc-migration/models/swonlysc-step0014/hf` (as v1-v3).
- Population (research): 1,542 passes = 657 tasksall + 885 sftv3; 1,032 without compaction, 510 with (501 once, 9 twice)
  -> 1,029 windows; 519 summary-writing calls.

## Rollout selection (per final record; the first failing rule is the exclusion reason, as v3's decide())
1. final status pass (final record, re-runs applied);
2. clean (v3's rule and code): evaluation completed, reward 1 (terminal.json), export.json + http.jsonl + model.patch
   present, no cached generation-validity failure;
3. old grader (NEW, see "Old-grader rule");
4. no degenerate reply anywhere (v3): degenerate.py classify() over EVERY status-200 reply of http.jsonl (task
   subagents and summary-writing calls included). GLM puts its reasoning in `message.reasoning` (Kimi:
   `reasoning_content`): a reply with reasoning only is NOT empty. Expected exclusion: the empty summary reply of sftv3
   hengyuf-g3 attempts/19-2 (its window 2 would have no summary, i.e. no task statement at all);
5. v1's harness base (system base, tool set) (v3);
6. two-source gate (NEW, hard): every window rebuilt from export.json (OpenCode + Gym semantics,
   scratch/glm-compaction/segments.py export_segments) equals the http.jsonl prefix chain (the window's last request +
   its response), message by message: role, content, reasoning, tool calls (name, parsed arguments), tool results.
   System messages are not compared (they change inside 47 of 510 compacted windows: date line, AGENTS.md edits).
   #windows == #compaction calls of the main session + 1 == #compaction parts + 1. Research: 1,029 / 1,029 windows
   and 160 / 160 sampled non-compacted passes equal; any difference excludes the rollout (`two_source_mismatch`).
Compaction is NOT an exclusion (v3 excluded it): such rollouts are cut into windows.

## Old-grader rule (NEW; commitment in glm-rollouts-tasksall-20261108/README.md, decision 2026-11-08)
Tasks whose tests/grader.py is DeepSWE's old grader (13,468 bytes: no test-path sanitizer) graded passes without
restoring the model's edits to test files. The current verifier (synth/verifier/grader.py) first restores / deletes
every model.patch path matching config.json "test_paths" (language globs of synth/verifier/test_paths.py;
JavaScript/TypeScript "non_additive": an existing test file the patch only adds lines to is kept).
704 of the 1,542 passes are on old-grader tasks (tasksall 274, sftv3 430). Classification of model.patch
(scratch/oldgrader/classify.py; to be moved into regrade/):
- **class A (kept)**: no test-path change outside the paths test.patch resets anyway, or only NEW test-path files that
  are not conftest / setup / config-like and whose name no non-test hunk mentions (the sanitizer only deletes them;
  the hidden tests never depend on them): 461 passes;
- **class B (needs the re-grade)**: an existing test file modified or deleted outside test.patch (other than the JS/TS
  add-only edits the grader keeps), or a risky new file: 243 passes on 90 tasks.
Class B passes enter only if a re-grade with the current verifier (regrade/, sandbox.grade on the task snapshot, the
current grader + the language test_paths) gives reward 1; otherwise excluded (`old_grader_regrade_fail`, or
`old_grader_not_regraded` when no result exists). A sample of class A is re-graded too, as a check of the static rule.
Every sample records `grader_kind` (current / old), `test_touch_class` (none / A / B) and the re-grade result.

## Samples (one raw_sft conversation each; the unit of packing and of the loss boundaries)
- **Non-compacted pass that ends with a final answer** -> 1 sample = v2's `convert()` unchanged (exec'd from
  kimi-pass-v2's build_corpus.py source text, whole-file and block sha256 asserted), exactly as v3.
- **Window samples** (every window of a compacted pass; also a non-compacted pass whose export ends in a message that
  is not a valid final answer, e.g. the errored empty final message of tasksall t1-h00 attempts/11-2): built by a
  segment converter that applies v2's per-turn rules:
  - **window 1**: v2's system piece and user piece (the task), then v2's assistant / tool pieces for the assistant
    messages of the window's calls; when the window's last call made tool calls, their results are not included (the
    sample ends on an assistant piece; the results appear in the next window's context when OpenCode keeps that turn
    in its tail, else in no sample: 6 of 39 such windows in the dry run).
  - **window k >= 2**: the system piece of the window's first request (render of [system] with its tools), then ONE
    non-target `user` piece = render(context, tools, add_generation_prompt=True) minus the system piece, where
    context = that request's messages[1:] (OpenCode's post-compaction history: user "What did we do so far?", the
    summary with the summary call's reasoning, the tail of 0-27 assistant / tool messages, the synthetic user
    "Continue if you have next steps…"), taken from http.jsonl and redacted like the export; then v2's assistant / tool
    pieces for the assistant messages of the window's own calls.
  - **System prompt** of every sample = the window's FIRST request (v2's harness rule, generalized); flag
    `system_changed_in_window` when later requests of the window differ.
  - **Rendering**: sc14 chat template via transformers apply_chat_template with `truncate_history_thinking=False` for
    window samples (GLM saw all earlier reasoning; the RL / eval serving of the student sets it false too). For
    non-compacted conversations both settings render identically (one user message), so v2's convert() output is
    unchanged.
- **Self-test (build-time assert)**: the segment converter run on every non-compacted pass as one window reproduces
  v2's `convert()` byte for byte (pieces, training_row_sha256, token_length, assistant_tokens).
- **v2 invariants per sample**: join(pieces) == the sc14 render of the sample's conversation; every prompt piece ends
  with GEN_PREFIX; concat(enc(piece)) == enc(join); exactly one system message, first; every assistant piece non-empty
  and right after a user / tool piece; ends with an assistant piece.
- **Turns that are never targets**: assistant messages with no text and no tool part (OpenCode never sends them back;
  skipped); an errored message (info.error) with no call in the chain (dropped; known: tasksall t1-h00 11-2 attempt-0,
  final); a reply with finish_reason "length" (dropped; it can only be the last call of a window — the window then
  ends at the previous call; known: tasksall t1-h01 13-4, window 1). Anything else unexpected: the rollout fails the
  gate.
- **Kept, tagged**: windows after "final answer, then compaction" (5 rollouts; 1-2 "already done" calls):
  `post_final_continue`.
- **Not samples**: OpenCode's summary-writing calls (519; no tools, other system prompt), task-subagent sessions (their
  reports are the parent's tool results, as v1-v3). Offered as options in the README.

## Sample-level filters (v3)
- literal generation stop text `</s>` in an assistant target -> sample dropped (v3 rule); literal `<s>` counted only;
- ceil32(tokens) > 524,288 -> sample dropped (v2 / v3);
- no 262,144 cap (v3 kept v4 / v5 conversations up to 524,288; all 510 first windows are 262,875-278,292 sc14 tokens);
  the count over 262,144 is reported.

## Ids and metadata
- `step_weight_agent_run_id` = selection `agent_run_id` = `glm53-<batch>/<campaign>/<task_id>/<slot>#seg<k>of<K>`
  (K = 1 for non-compacted passes): unique per sample; `run`, `task_id`, `slot` mark the task / rollout.
- selection rows: v3's fields (HSG fields as strings) + `rollout_batch` (tasksall / sftv3), `campaign`,
  `parent_trajectory_id`, `segment`, `n_segments`, `context_tokens`, `first_call_index`, `last_call_index`,
  `system_changed_in_window`, `post_final_continue`, `tier`, `task_status`, `audit`, `glm_verdict`, `grader_kind`,
  `test_touch_class`, `regrade`.
- metadata.jsonl: every considered final record of the tier-1/2 tasks (excluded ones with the reason), with a `corpus`
  block per sample.

## Packing (v2 / v3, unchanged)
FFD on ceil32 lengths into 524,288-token rows over all samples, rows shuffled with random.Random(20261109), 6 contiguous
near-equal shards, weight 0.16666667 each. Windows of one rollout may share a row: the trainer
(megatron-lm-ultra step-v10 sft_dataset.py, sha cadbab60) splits a row at system messages into THD segments
(cu_seqlens), and attention, Mamba conv / SSM state and the MTP roll reset per segment, so no loss or context crosses
samples (PP = 1, or 2 with MTP; CP <= 16).

## Verification (verify_glm.py, independent of the builder)
- v3's tc() / row() re-implementation of HSG's SFTDataset: loss-mask sum == sum(counts x weights) per row, unique ids,
  no stop-id targets, rows <= 524,288 with ceil32 padding per sample.
- selection == shards, per-sample sha / token_length, receipt fields, blend paths == DATA_ROOT.
- Per rollout, re-derived from the raw attempt files: window count, the two-source gate again (own code), each
  context piece == the independently located request rendered, coverage (every call of the main chain except
  summary-writing calls and dropped length / errored replies is a target exactly once across the rollout's samples),
  degenerate re-scan, old-grader class and re-grade result.
- scratch/packing-semantics/tools/sim_sft_getitem.py (the real pinned SFTDataset.__getitem__) on sample rows, including
  rows holding two windows of one rollout: cu_seqlens == cumulative ceil32 lengths, no loss position labelled from
  another sample, per-sample loss mass == counts x weights, Mamba seq_idx boundaries.

## Run and transfer
- CPU job through schdule_helper.sh (pattern of sft-v3's tools/sft_v3.sbatch): sources_glm.py -> build_glm.py ->
  verify_glm.py -> SHA256SUMS; `swe-rl-v2-sc-migration/runtime/envs/vllm/bin/python -I -B`, offline, private TMPDIR,
  *KEY / *TOKEN / *SECRET* unset.
- dm two-leg copy to AGA (xfer/hsg_step.sbatch + new actions from scratch/aga-transfer/: leg 1 dfw -> pdx-staging
  hengyuf-dm-staging/sft-glm-tier12-20261109, leg 2 on AGA over aga-ssh), then sha256 of every file on AGA against
  SHA256SUMS. Needs a fresh dm login on dfw (expired 2026-11-07) and a dm setup on AGA (probe).
