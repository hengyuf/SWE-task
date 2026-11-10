# kimi-pass-v2-20261023: SFT corpus v2 (Kimi-K3 passes from hengyuf + Baihe, raw_sft scheme)

v2 = all 379 conversations of v1 (`kimi-pass-v1-20261023`, hengyuf's Kimi-K3 passes) + 418 Kimi-K3 passes on Baihe's
97 accepted synthetic DeepSWE tasks, rendered with the same converter and re-packed together. The on-disk format is the
same as v1, i.e. zhenghaox's HSG Super SFT `raw_sft` DATA_ROOT. Data only: nothing here was trained, converted or run
with Megatron.

- HSG DATA_ROOT (used inside the blend and receipt): `/lustre/fsw/portfolios/nemotron/users/hengyuf/sft-kimi-pass-v2-20261023`
- Built on cw-dfw on 2026-10-23 with `swe-rl-v2-sc-migration/runtime/envs/vllm/bin/python -I` only (the v1 env). CPU only,
  no network, no Slurm.

## Files
| path | what |
|---|---|
| `raw_sft/part_00{0..5}.mat.jsonl.packed.512k` | 6 shards, 242 packed rows (41/41/40/40/40/40). Each line is `{"messages": [...]}` |
| `blends/raw_sft.json` | `{"train": ["0.16666667", <HSG part_000>, ...], "valid": null, "test": null}` |
| `DATA_PREP_COMPLETE.json` | Receipt with the HSG fields plus `build_stats` (incl. `per_source`) |
| `selection/kimi_pass_v2.jsonl` | 797 rows: HSG selection fields + `source` (hengyuf, baihe), run, task_id, slot, grading provenance, flags |
| `metadata.jsonl` | All 798 source metas (380 hengyuf + 418 baihe), with `source` and a `corpus` block (`dropped`, shard/row, tokens) |
| `sources/baihe/trajectories.jsonl.gz` | Baihe's 418 selected trajectories (redacted), in the v1 batch line layout `{meta, trajectory, final_patch}` |
| `sources/baihe/harness.json` | System prompts / tools / request params of those trajectories (keys = the batch's sha256 scheme) |
| `sources/baihe/selection_report.json` | Accepted-list cross-check, per-task scoring/probe, every exclusion with reason, redaction counts, harness comparison |
| `tools/baihe_source.py` | Selects and extracts Baihe's part into `sources/baihe/` |
| `tools/build_corpus.py` | v1's builder extended to two sources (converter unchanged). Deterministic |
| `tools/verify_corpus.py` | v1's independent re-check plus v2 checks (below) |
| `SHA256SUMS` | sha256 of every file above |

The `sources/baihe/` directory (~95 MB) is provenance only; training reads only `blends/` + `raw_sft/`.

## Sources (read-only)
- **hengyuf**: `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/export/kimi_pass_batch_20261021` (380 passes; see v1's
  README). Re-rendered here; all 379 kept conversations are byte-identical to v1 (`training_row_sha256` checked), and the same
  one is dropped.
- **baihe**: `B=/lustre/fsw/portfolios/nemotron/users/baiheh/swe-task-hengyu` (the original pipeline hengyuf's port came from).
- Student template and tokenizer: sc14 `.../swe-rl-v2-sc-migration/models/swonlysc-step0014/hf` (as v1).

## Baihe selection (`tools/baihe_source.py`)
1. **Tasks**: the 97 `B/outputs/syn-*` task dirs (`.workspace` siblings skipped). Cross-check: they match exactly the union
   of `B/synth/runs/*/accepted/tasks.jsonl` (97 rows, 19 runs; no task in two runs) and `B/outputs/tasks.jsonl` (97). Every
   `provenance.json` has `accept: true` and its `task_id`; its `task_dir` names the run/target/repo/cand, and it equals the
   scored `metrics.json` field by field. **No mismatches.**
2. **Scored candidate**: the unique `build/<target>/<repo>/round-<N>/cand-<name>/metrics.json` with that task_id
   (recover_slots.locate); it is always provenance's cand and has `accept: true`.
3. **Grading repair / regrade**: none. Baihe's tree has no `audit/`, `posthoc.jsonl`, `withdrawn/`, `rollouts.pre-regrade` /
   `.pre-gate-fix` or any regrade/repair file in any run, and no grading_repair/recover_slots/regrade tools. So the latest
   grading per slot is the candidate's own scoring (`rollouts/kimi-rK.md` header, as score() wrote it), graded by DeepSWE's
   grader at probe time: `status_source: candidate_scoring`, `grader: deepswe-recorded`. No task is withdrawn.
4. **Probe and attempt**: among the round's `probe-*` holding the task (manifest `_ng_task_index` = k, attempts dir `k-r`,
   slot `kimi-r<r>`), the one whose records reproduce the scored rollouts' first two `.md` lines (rollouts.render) and the
   `.patch` (grading_repair.scored_probe); exactly one matched for every task. The attempt is ATP slot_state's: the first
   attempt with `evaluation_completed` and a clean cached generation-validity verdict (ATP's `generation_failure` was not
   run, since it writes caches; the md/patch reproduction is the guard). 5 kept slots use attempt-1 after an invalid
   attempt-0. Trajectory = that attempt's opencode `export.json` (never the clipped `.md`); final patch = its `model.patch`,
   equal to both the record patch and `rollouts/kimi-rK.patch` for all 418.
5. **Keep rule**: final status pass with reward 1.0 (md header and terminal.json), and a clean glm verdict for the slot in
   the same scoring's `teacher_audit.json`: `cheated: false`, no "Answer retrieval by action" category, and
   `environment_leak.found` not true (the field is absent in 24 kept slots of older audits).

| | slots |
|---|---|
| considered (97 tasks × 5 kimi slots) | 485 |
| kept | **418** (all 97 tasks have >= 1) |
| excluded: final status fail | 62 |
| excluded: final status infra (score() counted the slot as lost: invalid generation or verifier timeout) | 5 |
| excluded: cheated / answer retrieval / environment leak / no verdict / files missing | 0 / 0 / 0 / 0 / 0 |

Every exclusion is listed with its detail in `sources/baihe/selection_report.json` (`exclusions`).

**Grading caveat.** All 418 Baihe passes are DeepSWE-grader passes only (Baihe never ran the sanitizing grader).
`sanitizer_noop` is `null` for them (it needs hengyuf's per-language test globs). As an informational flag,
`metadata.jsonl` carries `patch_test_like_paths_outside_test_patch` (a path-regex heuristic); 305 of the 418 patches touch
such paths. Three glm rubric scores are on a 0-100 scale (97, 92, 95) instead of 0-10, as in v1's data notes.

## Harness (system prompt, tools, params)
Taken from the BODY (`payload`) of the first `request` event of each attempt's `http.jsonl`, with build_batch.py's own
`first_request_payload`/`harness_of` (headers were never read). Hashes per trajectory are in the meta / selection
(`harness_system_sha256`, `harness_tools_sha256`, `harness_params_sha256`).
- Tool set: 1, **the same as v1's** (`88515fe6…`). Request params: 1, the same as v1's (`77ac9e3a…`; kimi-k3,
  reasoning_effort max, temperature 1.0, top_p 0.95, max_tokens 131072). opencode 1.17.11 for all.
- System prompts: 11 variants, 1 base (the same opencode base as v1). 4 variants are byte-identical to v1 variants (261
  trajectories); **7 are new** (157 trajectories). They differ from v1 only in the `<env>` date and the repository
  instruction-file block: 2 use the date `Thu Oct 22 2026` (not in v1), 4 carry instruction-file blocks v1 never saw
  (pilot8-optique on citty, 23 trajectories; q10-clack on inquirer-js, 28), and 1 is a new combination of a v1 date and a
  v1 block (q15 on risor, 47). No new tool, template field or message structure; the sc14 template renders them like v1's.
- First user message of the request == first user text of the export: 418/418.

## Redaction
build_batch.py's patterns (its source text is exec'd by baihe_source.py, so they are identical) were applied to every
Baihe export, patch and the harness. Hits: `key_or_token_assignment` 36 (in 18 trajectories); every other pattern 0;
harness 0. Values are never printed.

## Rendering
Unchanged from v1 (see v1's README): request = harness system message + tools + the single user message + opencode assistant
messages (reasoning_content/content/tool_calls) + tool results; empty reasoning rendered as `<think>\n</think>…`; the
`invalid` tool kept at weight 1.0 (Baihe: 1 action, `pilot1-katex-multicolumn/syn-mathjax-src-cand-c02-r1b4-da3483cd/kimi-r4`);
every per-conversation invariant (template render == concatenated pieces, token boundaries, assistant_token_counts, generation
prefix, roles) asserted for all 798 conversations with 0 fixes. One literal special token in a Baihe assistant target
(`q15-…/syn-risor-cand-c01-r1b4-bde52eba/kimi-r0`), kept as in v1.

**Dropped (> 524,288 tokens): 1**, `official1-scriggo/syn-gomacro-cand-c01-c696c6ca/kimi-r1` (hengyuf, 555,047 tokens; as in
v1). No Baihe conversation is dropped (max 377,226).

## Packing
FFD of ceil32 lengths into 524,288-token rows (key: padded length desc, then trajectory_id) over both sources together;
row order shuffled with `random.Random(20261023)`; 6 contiguous shards 41/41/40/40/40/40, weight 0.16666667 each.
242 rows, <= 7 conversations per row, 98.65% fill. 169 rows mix both sources, 33 are hengyuf-only, 40 baihe-only.

## Stats
| | hengyuf | baihe | total |
|---|---|---|---|
| conversations | 379 | 418 | **797** |
| tasks | 97 | 97 | 194 |
| tokens | 59,885,267 | 65,261,009 | 125,146,276 (padded 125,158,816) |
| assistant (trained) tokens, all weight 1.0 | 29,311,482 | 31,049,765 | 60,361,247 |
| tokens/conversation min / median / p90 / max | 46,065 / 148,743 / 234,418 / 520,361 | 35,544 / 153,412 / 231,971 / 377,226 | 35,544 / 152,326 / 232,820 / 520,361 |
| assistant actions / tool calls | 43,856 / 50,050 | 42,817 / 47,709 | 86,673 / 97,759 |
| empty-reasoning actions | 20,351 | 20,370 | 40,721 |
| `invalid` tool actions | 3 | 1 | 4 |
| failed webfetch calls (no egress) | 40 | 46 (29 trajectories) | 86 |
| grader | sanitizing-regrade 267, deepswe-recorded 88, sanitizing-probe 14, sanitizing-inferred 10 | deepswe-recorded 418 | |
| max conversation | official1-scriggo/syn-gomacro-cand-c01-c696c6ca/kimi-r2 | q21-boa-…/syn-nova-cand-c04-r1b0-afe620cd/kimi-r0 | |

Packed rows: 242 total (202 rows contain hengyuf conversations, 209 contain baihe conversations).

## Five epochs at GBS 32 (HSG recipe: MBS 1, seq 524,288)
- `target_iteration = ceil(5·242/32) = 38`
- `train_samples = 38·32 = 1216` (5.025 nominal epochs)
- As for v1, the HSG 5-epoch sbatch hard-codes the 344-row corpus (train_samples 1728, target_iteration 54, save_interval
  32, `iter_0000032`/`iter_0000054` checks, `selection_sha256`, `selected_trajectories == 3205`); adapt them (here
  selected_trajectories 797, selection_sha256 from DATA_PREP_COMPLETE.json).

## Verification (`tools/verify_corpus.py`)
Reloads the shards with `datasets.load_dataset("json")`, re-implements `SFTTokenizer.tokenize_conversation` (identity) and
`SFTDataset.__getitem__` packing (32-token padding, 524,289 truncation guard, step-weight spans, loss_mask = weights[1:] with
pad/IGNORE masking). Result: 242 rows, 797 conversations, no truncation, loss-mask sum 60,361,247 == sum(counts × weights).
v2 checks: shard conversations == selection rows (797; hengyuf 379, baihe 418), every `training_row_sha256` matches the
shard content, and v1's 379 conversations are byte-identical in v2.

## selection/kimi_pass_v2.jsonl
HSG fields (strings): `agent_run_id` (= trajectory_id `<run>/<task_id>/<slot>`), `record_id`, `source_line` (0-based line
in `source_part`), `source_part` (`kimi_pass_batch_20261021/trajectories.jsonl.gz` or `sources/baihe/trajectories.jsonl.gz`),
`token_length`, `training_row_sha256`. Added: `source`, `run`, `task_id`, `slot`, `included_because`, `grader`,
`grader_detail`, `status_source`, `f2p`, `p2p`, `attempt_dir`, `teacher_audit`, `harness_*_sha256`, `sanitizer_noop`,
`grading_note`, `n_assistant`, `assistant_tokens`, `masked_assistant_turns`, `webfetch_calls`, `shard`, `shard_row`,
`row_position`.

## Rebuild
```
PY=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/runtime/envs/vllm/bin/python
$PY -I tools/baihe_source.py          # sources/baihe/ (add --dry-run for the selection only)
$PY -I tools/build_corpus.py --workers 12
$PY -I tools/verify_corpus.py
```
