# kimi-pass-v1-20261023: SFT corpus v1 (Kimi-K3 passes, raw_sft scheme)

This is hengyuf's first SFT corpus for the student sc14 (`swonlysc-step0014`). It is laid out as a DATA_ROOT in the
exact on-disk format of zhenghaox's HSG Super SFT pipeline (`super-filtered-glm53-v5-20260922`, scheme `raw_sft`).
It is meant to be copied by Data Mover to the HSG path below. This directory holds data only: nothing here was
trained, converted or run with Megatron.

- HSG DATA_ROOT (the path used inside the blend and receipt): `/lustre/fsw/portfolios/nemotron/users/hengyuf/sft-kimi-pass-v1-20261023`
- Built on cw-dfw on 2026-10-23 by `tools/build_corpus.py`, using only `swe-rl-v2-sc-migration/runtime/envs/vllm/bin/python -I`
  (transformers 5.12.1, tokenizers 0.22.2). CPU only, no network.

## Files
| path | what |
|---|---|
| `raw_sft/part_00{0..5}.mat.jsonl.packed.512k` | 6 shards with 117 packed rows in total (20/20/20/19/19/19). Each line is `{"messages": [...]}` |
| `blends/raw_sft.json` | `{"train": ["0.16666667", <HSG part_000>, ...], "valid": null, "test": null}` |
| `DATA_PREP_COMPLETE.json` | Receipt with the same fields as the HSG one. It adds `build_stats` (all counts in this README) |
| `selection/kimi_pass_v1.jsonl` | 379 rows in the HSG selection schema plus our fields (see below) |
| `metadata.jsonl` | All 380 batch metas, in batch order, each with a `corpus` block (tokens, shard/row, stats, `dropped`) |
| `tools/build_corpus.py` | The builder. It is deterministic: a rerun gives the same shards. `--invalid-policy mask` is an option (see below) |
| `tools/verify_corpus.py` | An independent re-check (described below) |
| `SHA256SUMS` | sha256 of every file above |

## Sources (read-only)
- Batch: `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/export/kimi_pass_batch_20261021`. It holds 380 Kimi-K3 passes
  (`trajectories.jsonl.gz`), already redacted. The system prompt and tools of each trajectory come from `harness.json`, keyed by
  `meta.harness_system_sha256` and `harness_tools_sha256`. There are 20 system variants (one system message each) and one tool set.
- Student template and tokenizer: `.../swe-rl-v2-sc-migration/models/swonlysc-step0014/hf/{chat_template.jinja,tokenizer.json,tokenizer_config.json}`.
- Format semantics: `sft_dataset.py` (sha `cadbab60…`, the same as the HSG runtime) and `sft_tokenizer.py`. The prompt format
  is identity: has_bos False, assistant_prefix_len 0.

## Rendering rules
1. **Chat request.** For each trajectory the builder builds the request sc14 would have received. It has the system
   message from harness.json, the tools from harness.json (rendered by sc14's template as its `# Tools … </IMPORTANT>`
   section), the one user message (the task text exactly as Kimi got it, including opencode/Gym's surrounding `"`
   quotes, which the HSG rows also have), and then each opencode assistant message (one step each) as an OpenAI
   assistant message. That message carries `reasoning_content` (the `<think>…</think>` text), `content` (the text after
   `</think>`) and `tool_calls` (name and the `state.input` dict, in its original key order). After it come one `tool`
   message per call: `state.output` if the call completed, or `state.error` if it errored (this is opencode's
   `error-text`). Step, patch and snapshot parts are bookkeeping, and the builder drops them.
2. **Splitting.** The builder renders the request with `transformers.apply_chat_template(sc14 template, tools)` and cuts
   it into per-message pieces. The concatenated pieces are asserted to equal the full template render, byte for byte,
   for every conversation.
   - system = `<|im_start|>system\n<prompt>\n\n# Tools…</IMPORTANT><|im_end|>\n`. Extra fields: `weight` 1.0,
     `step_weight_agent_run_id` = trajectory_id, `assistant_weights` = [1.0]*n, `assistant_token_counts`.
   - user = `<|im_start|>user\n<task><|im_end|>\n<|im_start|>assistant\n<think>\n`. The tail is asserted to equal the
     template's `add_generation_prompt=True` output.
   - assistant = the continuation after `<think>\n`. With tool calls it is `<reasoning></think><visible>` (trimmed) + `\n` +
     `<tool_call>\n<function=…>\n<parameter=…>\n…\n</parameter>\n</function>\n</tool_call>\n`… + `<|im_end|>\n`. The final
     turn is `<reasoning></think><visible><|im_end|>\n`. The template puts no `\n` after `</think>`, and the visible text is
     concatenated directly; this is the same as the HSG rows (`</think>Let me…`, `</think>\n<tool_call>`).
   - tool = all results of one assistant turn in one message:
     `<|im_start|>user\n<tool_response>\n…\n</tool_response>\n…<|im_end|>\n<|im_start|>assistant\n<think>\n`.
3. **Reasoning is kept on every turn.** The template's `truncate_history_thinking` only strips turns before the last user
   message. There is a single user message, so the plain template render already keeps every turn's reasoning.
4. **Turns with empty reasoning** (20,351 of 43,856 actions: Kimi emitted no reasoning, and opencode `tokens.reasoning`
   is 0 for them). These are rendered as `<think>\n</think>…`, so the assistant piece starts with `</think>`. Every prompt
   then ends with the same generation prefix `<|im_start|>assistant\n<think>\n` that the student is served at inference
   (enable_thinking defaults to True), as in the HSG rows. The template's own no-reasoning form (`<think></think>`) would
   have put a different prefix in the prompt.

## Edge cases
| case | occurrences | handling |
|---|---|---|
| several system messages | 0 (each of the 20 variants has 1) | The builder would join them with `\n\n` into one system content. None exist |
| opencode-injected text parts (`synthetic`/`ignored`) | 0 | Would be dropped. None exist. The single user text is the exact first-request user message (harness.json check: 380/380 match) |
| reasoning parts / compaction / summary parts / extra user turns | 0 | No reasoning parts, no compaction parts, no `time.compacted` tool outputs, and no summary messages. Exactly 1 user message per trajectory. The builder asserts this; compacted outputs would be replayed as `[Old tool result content cleared]` |
| empty visible text | 30,101 actions | `</think>\n<tool_call>…` (template behaviour). Every assistant piece is non-empty |
| empty reasoning | 20,351 actions | rule 4 above |
| `invalid` tool | 3 actions in 3 trajectories (scriggo gomacro c04 kimi-r2, statemachine transitions c03 kimi-r0, ytt yq c02 kimi-r4) | Kimi's `write` call had truncated JSON arguments, and opencode rewrote it to the `invalid` tool. These are kept as opencode replayed them to Kimi (`<function=invalid>` with `tool`/`error` parameters, then the "arguments are invalid" tool response). The next turn reacts to it. Weight 1.0, as the uniform policy requires. `--invalid-policy mask` would give these 3 actions weight 0.0 (context only) |
| failed webfetch | 40 calls in 29 trajectories (all `Transport error`, no egress) | Kept: the call, then the error text as the tool response. `webfetch_calls` is in selection/metadata so they can be filtered |
| other tool errors | 80 (edit 50, bash 17, read 10, grep 2, write 1) | Tool response = `state.error` |
| `task` subagent calls | 34 | Tool response = the subagent's report. Subagent sessions are not included |
| lone UTF-16 surrogates | 0 | Would be replaced by U+FFFD and counted |
| literal special-token text in assistant targets | 4 × `<s>` (3 anko gpython trajectories, Go/Python source using `"<s>"` as a filename) | Kept. These are tokenized as BOS, exactly as the HSG tokenizer would tokenize them. No `<unk>` (pad id) appears in any target |
| non-string tool arguments | ints (timeout/offset/limit), the todowrite list, 2 bools | Rendered by the template's rules: list → `tojson`, other values → `|string` (so `True`) |

## Invariants (checked for every conversation; violations abort the build)
1. `encode(join(contents)) == concat(encode(content_i))` with `add_special_tokens=False`. All 379 conversations pass
   with **0 fixes**: no whitespace adjustment was needed at any boundary.
2. The first message is the only system message of the conversation. Every assistant message is non-empty and follows
   user or tool. Each conversation ends with an assistant message.
3. `assistant_token_counts[i] == len(encode(assistant_i))` and their sum == the number of non-masked target tokens.
4. Every conversation is at most 524,288 tokens. **Dropped: 1**: `official1-scriggo/syn-gomacro-cand-c01-c696c6ca/kimi-r1`
   (555,047 tokens; batch estimate chars/4 = 397,700; included_because both, grader sanitizing-regrade). It is still in
   `metadata.jsonl` with `corpus.dropped: true`. Real token counts are about 1.4x the chars/4 estimate.

`tools/verify_corpus.py` re-checks the written shards independently. It reloads them with `datasets.load_dataset("json")`
and re-implements `SFTTokenizer.tokenize_conversation` (identity) with the transformers tokenizer. It then replays the
`SFTDataset.__getitem__` packing (32-token padding, the 524,289 truncation guard, the step-weight span assignment, and
loss_mask = weights[1:] with pad/IGNORE masking). Result: 117 rows, 379 conversations, no truncation, and loss-mask sum
29,311,482 == sum of assistant token counts × weights.

## Packing
- Sequence length 524,288 with CP16. Each conversation occupies `ceil32(len)`, and a row is limited to
  `sum(ceil32(len_i)) <= 524288`. The builder uses deterministic first-fit-decreasing (key: padded length descending,
  then trajectory_id).
- Row order is shuffled with `random.Random(20261023)`, then split into 6 contiguous shards (20/20/20/19/19/19).
  Each shard has weight 0.16666667 in the blend.
- 117 rows, at most 6 conversations per row, 97.6% fill (59,891,232 padded tokens / 117×524,288).

## Stats (379 conversations)
| | |
|---|---|
| tokens total | 59,885,267 (padded 59,891,232) |
| assistant (trained) tokens | 29,311,482, all weight 1.0 |
| tokens per conversation | min 46,065 / median 148,743 / p90 234,418 / max 520,361 (scriggo gomacro c01 kimi-r2) |
| assistant actions / tool messages / tool calls | 43,856 / 43,477 / 50,050 |
| system tokens | 7,558–13,015 per conversation (prompt variant + tools section) |
| grader | sanitizing-regrade 267, deepswe-recorded 88, sanitizing-probe 14, sanitizing-inferred 10 |
| sanitizer_noop | True 49 / False 330; grading_note set on 1 |
| included_because | accepted 230, both 110, strong 39 |

## Five epochs at GBS 32 (HSG recipe: MBS 1, seq 524,288)
- `target_iteration = ceil(5·117/32) = 19`
- `train_samples = 19·32 = 608` (5.197 nominal epochs)
- The HSG 5-epoch sbatch hard-codes values for the 344-row corpus: train_samples 1728, target_iteration 54,
  save_interval 32, checks for `iter_0000032`/`iter_0000054`, `selection_sha256`, and `selected_trajectories == 3205`.
  These must be adapted. In particular, save_interval 32 is greater than 19, and lr_warmup_samples 100 is about 16% of
  608 samples.

## selection/kimi_pass_v1.jsonl
- HSG fields, written as strings as in `all_three_zero_3205.jsonl`:
  - `agent_run_id` = trajectory_id
  - `record_id` = `r_` + sha256(trajectory_id)[:24]
  - `source_line` = 0-based line in the batch's trajectories.jsonl.gz
  - `source_part` = `kimi_pass_batch_20261021/trajectories.jsonl.gz`
  - `token_length`
  - `training_row_sha256` = sha256 of `json.dumps({"messages": <this conversation's messages>}, ensure_ascii=False)`
- Added fields: `run`, `task_id`, `slot`, `included_because`, `grader`, `sanitizer_noop`, `grading_note`,
  `n_assistant`, `assistant_tokens`, `masked_assistant_turns`, `webfetch_calls`, `shard`, `shard_row` (0-based line in
  the shard), and `row_position` (index of the conversation inside the packed row).
