# sft-v3-20261029: SFT corpus v3 (v2 conversations <= 262,144 tokens + Kimi-K3 v4 / v5 passes, raw_sft scheme)

User request (hengyuf, 2026-10-29, verbatim): "v4 rollout 结束之后, 我想造一个新的SFT dataset (命名为SFT-v3), 他包含v2 所有长度在256k以内的rollout, 以及v4, v5 里成功且没有compaction 的rollout,帮我build 一个新的SFT dataset 并用data mover transfer 到HSG 上."

**Snapshot build.** hengyuf, 2026-10-29 ~16:25 (verbatim): 不等这些长尾rollout了,直接开始构建SFT-v3的 sft data 并传到HSG. So the v4 campaigns baihe-g0, baihe-g5, baihe-g7 (still running) were read at a snapshot (2026-10-29T17:03:47-0700, 2026-10-29T17:03:51-0700, 2026-10-29T17:03:54-0700): their terminal slots are included like the finished campaigns; 12 slots pending at the snapshot are excluded and listed (section "Snapshot" below, `metadata.jsonl`, `DATA_PREP_COMPLETE.json` `snapshot`).

**Literal stop-token rule** (added after the first build's verification, 2026-10-29). sc14's tokenizer parses special-token text inside the content, so the literal text `</s>` (generation_config.json eos id 2) in an assistant target would train the model to emit EOS mid-turn (vLLM stops there). **1 conversation(s) dropped**: `kr4-baihe-g2/syn-remark-lint-cand-c04-r1b3-93c239cc/kimi-r0` (13 x `</s>`, 14 x `<s>`). The literal BOS `<s>` (id 1, not a stop token) is kept as v2 did: 20 occurrences in 6 kept conversations (hengyuf 3, baihe 1, v4 2).

The on-disk format is v2's (`kimi-pass-v2-20261023`), i.e. zhenghaox's HSG Super SFT `raw_sft` DATA_ROOT: the same converter (v2's code, exec'd from its source text), the sc14 tokenizer and chat template, identity prompt format, the same `weight` / `step_weight_agent_run_id` / `assistant_weights` / `assistant_token_counts` fields in each system message, and the same packing into 524,288-token rows, so the HSG training recipe works unchanged. Data only.

- HSG DATA_ROOT: `/lustre/fsw/portfolios/nemotron/users/hengyuf/sft-v3-20261029`
- Built on cw-dfw with `swe-rl-v2-sc-migration/runtime/envs/vllm/bin/python -I -B`, CPU only, no network (2026-10-30T01:21:54.728433+00:00).

## Files
| path | what |
|---|---|
| `raw_sft/part_00{0..5}.mat.jsonl.packed.512k` | 6 shards, 453 packed rows (76/76/76/75/75/75). Each line is `{"messages": [...]}` |
| `blends/raw_sft.json` | `{"train": ["0.16666667", <HSG part_000>, ...], "valid": null, "test": null}` |
| `DATA_PREP_COMPLETE.json` | Receipt with the HSG fields plus `build_stats` (`per_source`, `v2_filter`, `v4v5_selection`, `epochs_gbs32`) |
| `selection/sft_v3.jsonl` | 1,594 rows: v2's HSG selection fields + `source` (hengyuf, baihe, v4, v5), grading provenance, `rollout_batch`, `campaign`, `rerun`, `flags_any`, `flags_strong` |
| `metadata.jsonl` | Every considered conversation's meta (v2's dropped ones included) with a `corpus` block (`dropped`, `dropped_reason`, shard/row, tokens, `flags`); v4 / v5 rows carry `campaign`, `record_source` (`results_file` / `snapshot`) and `snapshot_at`; the v4 / v5 slots pending at the snapshot are listed too (`corpus.dropped_reason` = `pending_at_snapshot`, `pending_state`) |
| `sources/v4/`, `sources/v5/` | `trajectories.jsonl.gz` (v1 batch line layout `{meta, trajectory, final_patch}`, redacted), `harness.json`, `selection_report.json` (every slot's exclusion with reason, independent counts, flags, the per-campaign record source and the snapshot with its pending list); `snapshot_records.json` (the snapshot campaigns' records, first-attempt records and pending slots, and the logic check) |
| `sources/task_tests.json` | Per task: test.patch / grader.py sha256 and the test files test.patch creates (flags) |
| `tools/sources_v4v5.py` | Selects and extracts the v4 / v5 passes |
| `tools/build_sft_v3.py` | v2's builder extended with the v2 filter and the v4 / v5 sources (converter unchanged). Deterministic |
| `tools/verify_sft_v3.py` | Independent re-check (v2's verify plus the v3 checks below) |
| `tools/sft_v3.sbatch` | The CPU job (schedule helper babysit pair) that runs the three tools and writes SHA256SUMS |
| `SHA256SUMS` | sha256 of every file above |

Training reads only `blends/` + `raw_sft/`; the v2 parts are read from v2's own sources (not copied here).

## Sources (read-only)
- **hengyuf** (v2): `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/export/kimi_pass_batch_20261021`.
- **baihe** (v2): `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/kimi-pass-v2-20261023/sources/baihe`.
- **v4**: `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v4-stock256k-20261028` (Kimi-K3 x5 on all 194 v2 tasks, OpenCode stock prompt, 262,144 window).
- **v5**: `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v5-official2-20261028` (Kimi-K3 x5 on the 33 official2 accepted + strong tasks, same settings).
- Student template and tokenizer: sc14 `.../swe-rl-v2-sc-migration/models/swonlysc-step0014/hf` (as v1 / v2).

## v2 part: conversations <= 262,144 tokens
Rule: v2 selection token_length <= 262144 (sc14 tokens of the rendered conversation, as the v2 builder recorded them). Of v2's 797 conversations 756 are kept and **41 dropped** (hengyuf 23, baihe 18); v2 itself had already dropped one over 524,288 (`v2_dropped_earlier`). Every kept conversation is re-rendered and checked byte-identical to v2 (training_row_sha256, token_length, assistant_tokens).

| dropped, by DeepSWE target | conversations |
|---|---|
| scriggo-method-declarations | 19 |
| boa-hierarchical-evaluation-cancellation | 14 |
| wasmi-trap-coredumps | 3 |
| tengo-destructuring-bindings | 3 |
| vitest-duration-sharding | 1 |
| tengo-callable-instance-isolation | 1 |

Tasks left without any v2 conversation: 7 (`official1-scriggo/syn-gomacro-cand-c01-c696c6ca`, `official1-scriggo/syn-gomacro-cand-c02-c3c4135e`, `official1-scriggo/syn-gomacro-cand-c02-r1b3-ed535f70`, `official1-scriggo/syn-gomacro-cand-c03-f73e50a8`, `official1-scriggo/syn-gomacro-cand-c04-r1b2-b79686cb`, `official1-wasmi/syn-wasmtime-cand-c03-r1b2-03550fa4`, `q21-boa-hierarchical-evaluation-cancellation/syn-nova-cand-c04-r1b0-afe620cd`).

## v4 / v5 part: successful, no compaction, no degenerate reply (`tools/sources_v4v5.py`)
Per slot of each campaign's results file (`records` = the final record; a slot ended by a broken reply carries its one approved re-run, which replaced it) or, for a campaign still running at the build, per TERMINAL slot of its snapshot (section Snapshot), the first failing rule is the exclusion reason: final status not pass (fail, or infra = ATP exhausted / fatal / needs_review) -> not clean (evaluation not completed, reward != 1, cached generation-validity failure, files missing) -> OpenCode compaction (assistant `mode: compaction` / `summary: true`, a compaction part, or a compacted tool output) -> not exactly one user message -> a degenerate reply anywhere (any status-200 reply of http.jsonl, task subagents included, that degenerate.py's `classify()` labels pad / special / empty) -> a harness base other than v1's. Trajectory = the attempt's `export.json`; harness = the first request BODY of its `http.jsonl`; redaction = build_batch.py's patterns (as v2).

| | v4 | v5 |
|---|---|---|
| final slots considered | 960 | 166 |
| kept | 700 | 139 |
| kept that are re-runs | 35 | 7 |
| tasks with >= 1 kept | 182 | 33 |
| excluded: compaction | 48 | 0 |
| excluded: degenerate_reply | 11 | 1 |
| excluded: final_status_fail | 201 | 26 |
| final passes with compaction (independent) | 48 | 0 |
| final passes with degenerate_reply (independent) | 19 | 1 |
| final passes with compaction_and_degenerate (independent) | 8 | 0 |
| slots pending at the snapshot (not considered, listed below) | 12 | 0 |
| kept from snapshot campaigns | 132 | 0 |

Pilot rollouts (`pilot`, `baihe-pilot` of v4, `pilot` of v5; outside the 970 / 165 slots) are included when they pass the same rules.

### Snapshot

Campaigns whose driver had not written `results/kimi-<campaign>.json` yet were read from their probe dirs, read-only, with the driver's record logic: ATP slot_state's `completed` test from ATP's own cached generation-validity assessment, `rollouts.probe_records` -> `driver.compact` (exec'd from their source texts, sha256 recorded), and the broken-reply re-run plan `reruns/<campaign>.json`. Only TERMINAL slots are considered; a graded slot that a broken reply ended counts only through its graded approved re-run. Before it is used, the same code replays every finished campaign and must reproduce its results file exactly.

**v4**: snapshot started 2026-10-29T17:03:47-0700; logic check: 787 of 787 records of 17 finished campaigns reproduced exactly (0 differences). Per-campaign record source:

| campaign | record source | finished / scanned at | terminal slots considered | pending at snapshot | kept |
|---|---|---|---|---|---|
| sel50-g0 | results file | 2026-10-29T10:25:36-0700 | 50 | 0 | 30 |
| sel50-g1 | results file | 2026-10-29T09:26:53-0700 | 50 | 0 | 34 |
| sel50-g2 | results file | 2026-10-29T06:42:20-0700 | 50 | 0 | 35 |
| sel50-g3 | results file | 2026-10-29T05:17:13-0700 | 50 | 0 | 29 |
| sel50-g4 | results file | 2026-10-29T03:58:50-0700 | 50 | 0 | 37 |
| rest47-g0 | results file | 2026-10-29T03:43:08-0700 | 50 | 0 | 41 |
| rest47-g1 | results file | 2026-10-29T14:43:02-0700 | 50 | 0 | 29 |
| rest47-g2 | results file | 2026-10-29T05:46:12-0700 | 45 | 0 | 33 |
| rest47-g3 | results file | 2026-10-29T04:14:49-0700 | 45 | 0 | 37 |
| rest47-g4 | results file | 2026-10-29T03:24:20-0700 | 45 | 0 | 28 |
| baihe-g0 | **snapshot** | 2026-10-29T17:03:47-0700 | 61 | 4 | 40 |
| baihe-g1 | results file | 2026-10-29T14:27:15-0700 | 60 | 0 | 43 |
| baihe-g2 | results file | 2026-10-29T16:13:14-0700 | 60 | 0 | 45 |
| baihe-g3 | results file | 2026-10-29T12:58:18-0700 | 60 | 0 | 41 |
| baihe-g4 | results file | 2026-10-29T13:05:50-0700 | 60 | 0 | 53 |
| baihe-g5 | **snapshot** | 2026-10-29T17:03:51-0700 | 58 | 2 | 49 |
| baihe-g6 | results file | 2026-10-29T13:20:44-0700 | 60 | 0 | 51 |
| baihe-g7 | **snapshot** | 2026-10-29T17:03:54-0700 | 54 | 6 | 43 |
| pilot | results file | 2026-10-28T14:56:15-0700 | 1 | 0 | 1 |
| baihe-pilot | results file | 2026-10-28T14:33:52-0700 | 1 | 0 | 1 |

12 slots pending at the snapshot (excluded; by state: broken_ended_rerun_not_planned 4, broken_ended_rerun_pending 6, in_flight 2):

| slot | state | detail |
|---|---|---|
| `kr4-baihe-g0/syn-nova-cand-c04-r1b0-afe620cd/kimi-r0` | broken_ended_rerun_pending | original_status fail; last_reply_kind pad; rerun_campaign baihe-g0-brk1; rerun_slot kimi-r0; rerun_state rerun_campaign_not_created |
| `kr4-baihe-g0/syn-risor-cand-c01-r1b4-bde52eba/kimi-r4` | broken_ended_rerun_pending | original_status fail; last_reply_kind pad; rerun_campaign baihe-g0-brk1; rerun_slot kimi-r0; rerun_state rerun_campaign_not_created |
| `kr4-baihe-g0/syn-risor-cand-c05-r1b2-e15a5178/kimi-r0` | broken_ended_rerun_pending | original_status fail; last_reply_kind pad; rerun_campaign baihe-g0-brk2; rerun_slot kimi-r0; rerun_state in_flight |
| `kr4-baihe-g0/syn-risor-cand-c05-r1b2-e15a5178/kimi-r3` | broken_ended_rerun_pending | original_status fail; last_reply_kind pad; rerun_campaign baihe-g0-brk2; rerun_slot kimi-r1; rerun_state in_flight |
| `kr4-baihe-g5/syn-pytest-cand-c01-r1b0-954445d0/kimi-r1` | broken_ended_rerun_pending | original_status fail; last_reply_kind pad; rerun_campaign baihe-g5-brk2; rerun_slot kimi-r0; rerun_state in_flight |
| `kr4-baihe-g5/syn-pytest-cand-c01-r1b0-954445d0/kimi-r3` | broken_ended_rerun_pending | original_status fail; last_reply_kind pad; rerun_campaign baihe-g5-brk2; rerun_slot kimi-r1; rerun_state in_flight |
| `kr4-baihe-g7/syn-go-jsonnet-cand-c04-r1b2-b0655666/kimi-r0` | in_flight | - |
| `kr4-baihe-g7/syn-nova-cand-c04-73057cd6/kimi-r2` | in_flight | - |
| `kr4-baihe-g7/syn-citty-cand-c04-r1b1-ec1c0299/kimi-r0` | broken_ended_rerun_not_planned | original_status fail; last_reply_kind pad |
| `kr4-baihe-g7/syn-inquirer-js-cand-c05-r1b0-a34e3650/kimi-r1` | broken_ended_rerun_not_planned | original_status fail; last_reply_kind pad |
| `kr4-baihe-g7/syn-nova-cand-c04-73057cd6/kimi-r4` | broken_ended_rerun_not_planned | original_status fail; last_reply_kind pad |
| `kr4-baihe-g7/syn-remark-lint-cand-c03-5e615329/kimi-r1` | broken_ended_rerun_not_planned | original_status fail; last_reply_kind pad |

States: in_flight = an attempt is running; not_started = no attempt yet; ended_ungraded = the latest attempt ended without a grade (awaiting ATP's retry or review); unassessed = graded but ATP has not cached its generation-validity assessment yet; broken_ended_rerun_pending = graded, ended by a broken reply, its one approved re-run has no grade yet; broken_ended_rerun_not_planned = the same, the campaign's main slots still running (the driver plans the re-runs once they are all terminal).

No LLM teacher audit exists for v4 / v5 (v2 required a clean glm verdict); none was run. Informational flags per conversation (`metadata.jsonl` corpus.flags, `selection` flags_any / flags_strong; definitions in `tools/sources_v4v5.py`), never used to exclude. The same export-based flags are computed for the v2 part as a baseline (v2 passed the glm audit):

| conversations with | hengyuf | baihe | v4 | v5 |
|---|---|---|---|---|
| any | 49 | 60 | 118 | 13 |
| strong | 0 | 0 | 0 | 0 |
| webfetch_calls | 23 | 27 | 62 | 4 |
| webfetch_completed | 0 | 0 | 0 | 0 |
| net_fetch_cmds | 27 | 37 | 53 | 7 |
| net_fetch_failed | 21 | 30 | 37 | 5 |
| net_fetch_ok | 6 | 9 | 7 | 2 |
| git_history_cmds | 20 | 14 | 50 | 3 |
| hidden_artifact_refs | 0 | 0 | 1 | 0 |
| hidden_test_seen_before_touch | 0 | 0 | 0 | 0 |
| hidden_test_first_touch_reads_ok | 0 | 0 | 0 | 0 |
| hidden_test_read_attempts | 0 | 0 | 0 | 0 |
| hidden_test_bash_first_mentions | 0 | 1 | 0 | 0 |
| subagent_calls | 31 | 28 | 63 | 2 |
| subagent_beyond_export | 0 | 0 | 0 | 0 |

## Packing
FFD of ceil32 lengths into 524,288-token rows (key: padded length desc, then trajectory_id) over all sources together; row order shuffled with `random.Random(20261029)`; 6 contiguous shards 76/76/76/75/75/75, weight 0.16666667 each. 453 rows, <= 7 conversations per row, 98.37% fill. Rows by batch mix: v2 33, v2+v4 267, v2+v4+v5 76, v2+v5 27, v4 32, v4+v5 17, v5 1.

## Stats
| | hengyuf | baihe | v4 | v5 | total |
|---|---|---|---|---|---|
| conversations | 356 | 400 | 699 | 139 | 1,594 |
| tasks | 91 | 96 | 182 | 33 |  |
| tokens | 51,693,077 | 59,665,298 | 104,380,414 | 17,865,609 | 233,604,398 |
| assistant (trained) tokens, all weight 1.0 | 25,566,018 | 28,626,688 | 51,770,390 | 8,340,959 | 114,304,055 |
| tokens/conversation min / median / p90 / max | 46,065 / 142,035 / 217,944 / 259,548 | 35,544 / 150,592 / 217,514 / 258,072 | 43,970 / 150,740 / 217,969 / 262,507 | 46,676 / 126,299 / 183,850 / 259,068 | 35,544 / 146,303 / 216,657 / 262,507 |
| conversations over 262,144 tokens | 0 | 0 | 1 | 0 | 1 |
| assistant actions / tool calls | 37,943 / 43,882 | 38,994 / 43,623 | 70,993 / 81,089 | 10,726 / 12,301 | 158,656 / 180,895 |
| packed rows containing | 249 | 290 | 392 | 121 | 453 |

Total: 1,594 conversations, 233,604,398 tokens (padded 233,629,088), 114,304,055 assistant tokens; dropped over 524,288: 0.

## Epochs at GBS 32 (HSG recipe: MBS 1, seq 524,288)
| epochs | target_iteration = ceil(E * rows / 32) | train_samples = iterations * 32 | nominal epochs |
|---|---|---|---|
| 5 | 71 | 2272 | 5.015 |
| 10 | 142 | 4544 | 10.031 |
| 20 | 284 | 9088 | 20.062 |

As for v1 / v2, the HSG sbatch hard-codes the corpus (train_samples, target_iteration, save_interval, the `iter_*` checks, `selection_sha256`, `selected_trajectories`): adapt them (here selected_trajectories 1594, selection_sha256 from DATA_PREP_COMPLETE.json).

## Verification (`tools/verify_sft_v3.py`)
Reloads the shards with `datasets.load_dataset("json")`, re-implements `SFTTokenizer.tokenize_conversation` (identity) and `SFTDataset.__getitem__` packing (32-token padding, 524,289 truncation guard, step-weight spans, loss_mask = weights[1:] with pad/IGNORE masking), then the v3 checks: shard conversations == selection rows (by source), training_row_sha256 of every conversation, every v2 conversation byte-identical to v2 and <= 262,144 tokens (and, for the real build, all of them present), receipt / blend consistency, and for every v4 / v5 conversation an independent re-read of its results record, export (compaction) and http.jsonl (degenerate.py's own scan()); a snapshot conversation's record is re-checked against its probe slot dir (the campaign manifest's slot, the re-run plan for a re-run, the first attempt with a full-reward completed grade and ATP's cached validity verdict None) and, if the campaign's results file has appeared since, against that file; every slot of a snapshot campaign is kept, excluded with a reason or pending, exactly once.

Result (2026-10-29 18:25 PDT): OK. 453 rows, 1594 conversations, no truncation, loss-mask sum 114,304,055 == sum(counts x weights) 114,304,055; selection == shards (hengyuf 356, baihe 400, v4 699, v5 139); every training_row_sha256 and token_length matches; v2: 756 conversations byte-identical to v2, all <= 262,144 (= all 756 such v2 conversations); v4/v5: 838 conversations re-checked (record pass, no compaction, one user message, 84,323 hub replies re-scanned with degenerate.py's scan(): none degenerate; 132 from snapshot campaigns re-checked against their probe slot dirs and re-run plans); completeness (every slot kept, excluded with a reason, or pending at the snapshot, exactly once): v4 699 kept / 261 excluded of 960 slots + 12 pending at the snapshot, v5 139 kept / 27 excluded of 166 slots; campaigns finished between the snapshot and this check: {"v4/baihe-g5": {"records_same": 58, "records_differ": 0, "pending_final_status": {"fail": 2}}}; assistant targets: 0 of 1594 conversations contain a generation stop id [2], literal BOS (id 1) targets 20 in 6 conversations (v4 2, hengyuf 3, baihe 1); build-stage drops for a literal stop token, confirmed in their export.json: kr4-baihe-g2/syn-remark-lint-cand-c04-r1b3-93c239cc/kimi-r0 (13 x </s>).

## Rebuild
```
PY=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/runtime/envs/vllm/bin/python
$PY -I -B tools/sources_v4v5.py --workers 16    # sources/v4, sources/v5, sources/task_tests.json
$PY -I -B tools/build_sft_v3.py --workers 16
$PY -I -B tools/verify_sft_v3.py --workers 16 --readme
```
(normally all three run in one CPU job, `tools/sft_v3.sbatch build`, which also writes SHA256SUMS).
