# Synthetic SWE tasks and Kimi K3 rollouts: full inventory (2026-11-06)

All accepted synthetic tasks (hengyuf queues official1-4 and val1, plus Baihe's fork) and every Kimi K3 rollout run on them. This merges the outputs of the six collectors (10 files in `parts/`). No further gap-filling builds are planned ("不补了"), so official3 and official4 are final as listed here.

## Totals

- **267 accepted tasks**: hengyuf 170 (official1 84, val1 2, official2 32, official3 14, official4 38) and Baihe 97. They cover 57 distinct DeepSWE targets and no task id appears twice across parts.
- **2741 Kimi K3 rollout slots** in 6 campaign labels: 2156 pass, 565 fail, 20 infra (pass rate over graded slots 79.2%), on 281 distinct task ids. Scope note (independent verifier): this headline includes the v3/v4/v5 slots of 14 non-accepted tasks but not their 70 build-probe slots (42 pass / 28 fail), and the v3 pilot but not the 3 v4/v5 pilots (all pass); the accepted-task line below is the consistent scope.
  - 2630 slots (2088 pass) are on the 267 accepted tasks. The other 111 slots (68 pass) are on 14 tasks that were never accepted: audit-strong probe copies that went into the SFT corpora. They get `source: "unknown"`; see the non-accepted section below.
  - Build probe (probe-1m, 1M context, 5 slots per task) on all 267 accepted tasks: 1335 slots, 1119 pass.
- **New since SFT v3** (accepted tasks with no conversation in SFT v1, v2 or v3): 54 tasks with 224 probe-1m Kimi passes (223 after dropping the 1 pass the teacher audit flagged as cheated). By queue: official3 14 tasks / 57 passes, official4 38 / 157 (156 clean), val1 2 / 10. No official1, official2 or Baihe task is new. However, official2's 136 probe-1m passes are in no corpus: SFT v3 took official2 from the 256k v5 campaign instead.

## Files

| file | rows | content |
|---|---|---|
| `tasks.jsonl` | 267 | One row per distinct accepted task. |
| `kimi_rollouts.jsonl` | 2741 | One row per Kimi rollout slot, all campaigns (every row of `parts/rollouts-*.jsonl`). |
| `summary.json` | - | All numbers below in machine-readable form, plus the input sha256s and the merge checks. |
| `README.md` | - | This file. |
| `parts/` | - | The collector outputs this merge reads (tasks-*, rollouts-*, sft-membership). They are left as written. |

JSON Lines, one object per line, UTF-8, sorted keys. Merge script: `/tmp/claude-158141/inventory-merge/merge.py` (stdlib only; it reads only `parts/` and writes these four files with an atomic rename).

### tasks.jsonl

- Collector fields: `task_id`, `source` (hengyuf/baihe), `queue`, `run`, `target` (the DeepSWE target), `repo` (the synthetic host repo, not the target's repo), `cand`, `round`, `accepted_dir`, `scored_dir`, `status_dir` (where the slot statuses were read: = `scored_dir`, except the 2 official1 post-hoc accepts), `similarity`, `student_rate`, `teacher_clean_rate`, `audit_verdict` (failure_match audit; null if there was none), `teacher_reprobe`, `probe_kimi` and `probe_sc14` (build-probe slot counts).
- `kimi`: `{campaign: {slots, pass, fail, infra}}` counted from the rollout rows. It lists only the campaigns that ran on the task, and `kimi["probe-1m"]` equals `probe_kimi` for every task.
- `sft`: `{v1, v2, v3}`, the number of conversations of this task in each SFT corpus. Pilot conversations are included.
- `probe_kimi_cheated` (derived): Kimi passes that the build's teacher audit flagged as cheated. It is computed as probe_kimi.pass - teacher_clean_rate * (pass + fail), which is exact on every task. The only non-zero value is official4-yaegi-embed syn-gomacro-cand-c05-1e4d0065 (1, kimi-r4).

### kimi_rollouts.jsonl

- Collector fields: `task_id`, `slot` (kimi-r0..r4), `campaign`, `status` (equal to `raw_status` on every row), `context_window`, and `path`. `path` is the per-slot transcript `.../rollouts/kimi-rN.md` for probe-1m and the final counted ATP attempt dir for v3, v4 and v5. v4/v5 rows also carry `usable`, `compactions`, `rerun`, `rerun_outcome`, `first_attempt_status`, `first_attempt_path`, `ended_by_broken_reply`, `atp_campaign`, `taskset` and `results_file`.
- `source`: hengyuf or baihe from tasks.jsonl; `"unknown"` if the task is not an accepted task.
- `queue`: from tasks.jsonl (null for unknown).
- `part_source` (v4/v5 rows only): the source label the collector wrote. It is kept because `source` overrides it; on the non-accepted tasks it still reads `hengyuf`.
- `in_sft`: the SFT corpora that contain this exact rollout. The join is on (task_id, slot, campaign) against `parts/sft-membership.jsonl`. For v4/v5 the attempt path must also match, so the 3 pilot conversations, which reuse kimi-r0 of a grid slot, do not attach to that slot. In all 3 cases the grid slot is itself in v3, so `in_sft` is the same either way.

## Definitions

### Tasks and queues

A task is **accepted** if it has a row in its run's `accepted/tasks.jsonl`, the pipeline's accept decision (its scored metrics.json says `accept: true`). 5 withdrawn official1 tasks are excluded.

| queue | source | what |
|---|---|---|
| official1 | hengyuf | hengyuf queue 1 (23 `official1-*` runs, 20 with accepts) |
| val1 | hengyuf | hengyuf pipeline validation run `val1` (target abs-stepped-slices) |
| official2 | hengyuf | hengyuf queue 2 (14 `official2-*` runs, 10 with accepts) |
| official3 | hengyuf | hengyuf queue 3 (11 `official3-*` runs, 5 with accepts) |
| official4 | hengyuf | hengyuf queue 4 (23 `official4-*` runs, 11 with accepts; includes the re-scored and teacher re-probed tasks) |
| baihe | baihe | Baihe's fork `baiheh/swe-task-hengyu` (19 `pilot*` / `q*` runs with accepts) |

### Kimi campaigns and status mapping

All campaigns use Kimi K3. A slot is one rollout of one task. `pass`, `fail` and `infra` partition the slots, and the pass rate below is pass / (pass + fail). **probe-1m** is the build probe: 1M context, 5 Kimi slots per task. **v3, v4 and v5** are later campaigns run for SFT data at 256k context (except the 1M v3 control arm): v3 and v4 on the v2 task set, v5 on official2.

- **probe-1m**. Context 1M (1,048,576). The task pipeline's build probe: Kimi K3 (nvidia/moonshotai/kimi-k3), 5 slots per task (kimi-r0..r4), run next to the 5 sc14 student slots. Both feed the accept decision. Tasks: every accepted task (267). Source: `<runs>/<run>/build/<target>/<owner__repo>/round-<k>/<cand>/rollouts/kimi-rN.md` (hengyuf runs: `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth/runs`, Baihe runs: `/lustre/fsw/portfolios/nemotron/users/baiheh/swe-task-hengyu/synth/runs`).
  Status: The label that `score()` wrote into the scored dir's `rollouts/index.md`. The collectors cross-checked it against each transcript header and against metrics.json. `pass` = verifier reward 1. `fail` = a valid graded attempt with reward 0. `infra` = the slot has no valid graded attempt (ATP slot not completed, invalid generation, sandbox 410 or verifier error/timeout). Infra slots count in `slots` only. Exceptions: the 2 official1 post-hoc accepts are read from their post-hoc score dir, and the 11 re-probed official4 tasks come from the 2026-11-05/06 teacher re-probe, which used the same 1M settings.
- **v3-256k-reason**. Context 256k (262,144). v3 phase 1 (sel50): 256k context, with the reason-before-action system sentence added to the prompt. Tasks: 50 hengyuf tasks from the v2 task set (train-acc-v2 sel50; 7 of them not accepted). Source: `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v3-20261026`.
  Status: The driver's results-file status (ATP terminal.json `evaluation_completed` + reward, plus the generation-validity cache). `infra` = an agent TimeoutError with no terminal.json, or a verifier failure. 11 slots were hit by the 10-26 quota incident: 6 count by their recovery re-run, 5 by an ATP in-slot retry (attempt-1). v3 did not re-run broken-reply endings: 70 of the 250 stay as `fail`.
- **v3-256k-reason-pilot**. Context 256k (262,144). A 1-rollout pilot with the same settings as sel50. It has its own label because its task and slot (kimi-r0) collide with a sel50 row. Tasks: 1 (syn-aiohttp-cand-c05-r1b0-7d9ec1f5). Source: same dir as v3-256k-reason.
  Status: same as v3-256k-reason
- **v3-ctrl-old-1m**. Context 1M (1,048,576). The v3 control arm: the previous Kimi settings (stock prompt, 1M window) on 4 of the sel50 tasks. Tasks: 4 (1 of them not accepted). Source: same dir as v3-256k-reason.
  Status: Same as v3-256k-reason. 4 of the 20 ended on a broken reply and are `fail`. syn-vscode-sqltools c05-r1b3 kimi-r4 is `infra` (verifier timed out), although its captured reward.json shows reward 1.
- **v4-stock-256k**. Context 256k (262,144). Stock OpenCode prompt at 256k, 5 slots on every task of the v2 task set. Tasks: 194 (hengyuf sel50 + rest47 = 97, of which 13 are not accepted; Baihe 97). Source: `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v4-stock256k-20261028`.
  Status: The final record's label. Only `pass` and `fail` occur (0 infra). Slots whose first attempt ended on a broken hub reply were re-run once (66) and count by the re-run; `first_attempt_status` and `first_attempt_path` keep the original. 8 agent timeouts with an empty patch are graded `fail`. `usable` = the slot is in the campaign's own SFT-usable list.
- **v5-official2-256k**. Context 256k (262,144). Stock OpenCode prompt at 256k on official2. Tasks: 33 (32 accepted official2 tasks + 1 audit-strong task that was not accepted). Source: `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v5-official2-20261028`.
  Status: Same as v4 (8 broken-reply re-runs, 0 infra).

### SFT corpora (`/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/`)

| corpus | dir | conversations | tasks | on non-accepted tasks | content |
|---|---|---|---|---|---|
| v1 | `kimi-pass-v1-20261023` | 379 | 97 | 39 | probe-1m Kimi passes on official1 tasks plus 13 audit-strong probe copies |
| v2 | `kimi-pass-v2-20261023` | 797 | 194 | 39 | v1 + all 418 Baihe probe-1m passes |
| v3 | `sft-v3-20261029` | 1594 | 220 | 78 | v2 conversations of at most 262,144 tokens (756) + v4 (699) + v5 (139) passes without compaction, broken replies or a literal `</s>` |

The corpus named sft-v3 has nothing to do with the v3 rollout campaign. No v3-campaign, official3, official4 or val1 rollout is in any corpus. Across the three corpora there are 1635 distinct conversations on 227 tasks (213 of them accepted).

## Numbers

### Tasks by queue

| queue | source | tasks | runs with accepts | targets | probe-1m Kimi slots | pass | fail | infra | pass rate | sc14 pass/slots | tasks in any SFT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| official1 | hengyuf | 84 | 20 | 20 | 420 | 341 | 74 | 5 | 82.2% | 34/420 | 84 |
| val1 | hengyuf | 2 | 1 | 1 | 10 | 10 | 0 | 0 | 100.0% | 1/10 | 0 |
| official2 | hengyuf | 32 | 10 | 10 | 160 | 136 | 24 | 0 | 85.0% | 15/160 | 32 |
| official3 | hengyuf | 14 | 5 | 5 | 70 | 57 | 13 | 0 | 81.4% | 8/70 | 0 |
| official4 | hengyuf | 38 | 11 | 11 | 190 | 157 | 27 | 6 | 85.3% | 14/190 | 0 |
| baihe | baihe | 97 | 19 | 18 | 485 | 418 | 62 | 5 | 87.1% | 31/485 | 97 |
| **total** |  | 267 | 66 | 57 | 1335 | 1119 | 200 | 16 | 84.8% | 103/1335 | 213 |

Targets per source: hengyuf 45, Baihe 18. 6 targets have tasks from both sources (adaptix-name-mapping-aliases, kysely-window-grouping-helpers, mobly-grouped-test-barriers, obsidian-linter-auto-table-of-contents, pebble-durability-wait-apis, tengo-destructuring-bindings); the task ids are still disjoint.

### Kimi rollouts by campaign

| campaign | context | tasks | slots | pass | fail | infra | pass rate | accepted-task slots / pass | rows in SFT v1 / v2 / v3 |
|---|---|---|---|---|---|---|---|---|---|
| probe-1m | 1M (1,048,576) | 267 | 1335 | 1119 | 200 | 16 | 84.8% | 1335 / 1119 | 340 / 758 / 719 |
| v3-256k-reason | 256k (262,144) | 50 | 250 | 121 | 127 | 2 | 48.8% | 215 / 105 | 0 / 0 / 0 |
| v3-256k-reason-pilot | 256k (262,144) | 1 | 1 | 1 | 0 | 0 | 100.0% | 0 / 0 | 0 / 0 / 0 |
| v3-ctrl-old-1m | 1M (1,048,576) | 4 | 20 | 12 | 6 | 2 | 66.7% | 15 / 10 | 0 / 0 / 0 |
| v4-stock-256k | 256k (262,144) | 194 | 970 | 764 | 206 | 0 | 78.8% | 905 / 720 | 0 / 0 / 697 |
| v5-official2-256k | 256k (262,144) | 33 | 165 | 139 | 26 | 0 | 84.2% | 160 / 134 | 0 / 0 / 138 |
| **total** |  | 281 | 2741 | 2156 | 565 | 20 | 79.2% | 2630 / 2088 | 340 / 758 / 1554 |

SFT-usable 256k passes by the campaigns' own lists: v4 745, v5 138. 1593 rollout rows are in at least one corpus.

### Kimi rollouts by source

| source | tasks | slots | pass | fail | infra | pass rate | by campaign (pass/slots) |
|---|---|---|---|---|---|---|---|
| hengyuf | 170 | 1660 | 1276 | 370 | 14 | 77.5% | probe-1m 701/850; v3-256k-reason 105/215; v3-ctrl-old-1m 10/15; v4-stock-256k 326/420; v5-official2-256k 134/160 |
| baihe | 97 | 970 | 812 | 153 | 5 | 84.2% | probe-1m 418/485; v4-stock-256k 394/485 |
| unknown | 14 | 111 | 68 | 42 | 1 | 61.8% | v3-256k-reason 16/35; v3-256k-reason-pilot 1/1; v3-ctrl-old-1m 2/5; v4-stock-256k 44/65; v5-official2-256k 5/5 |

### Kimi rollouts on accepted tasks, by queue and campaign (pass/slots)

| queue | probe-1m | v3-256k-reason | v3-256k-reason-pilot | v3-ctrl-old-1m | v4-stock-256k | v5-official2-256k |
|---|---|---|---|---|---|---|
| official1 | 341/420 | 105/215 | - | 10/15 | 326/420 | - |
| val1 | 10/10 | - | - | - | - | - |
| official2 | 136/160 | - | - | - | - | 134/160 |
| official3 | 57/70 | - | - | - | - | - |
| official4 | 157/190 | - | - | - | - | - |
| baihe | 418/485 | - | - | - | 394/485 | - |

### New since SFT v3

An accepted task is new if it has 0 conversations in v1, v2 and v3. Only the build probe (probe-1m) has run on these tasks; no 256k campaign has.

| queue | new tasks | targets | probe-1m Kimi slots | pass | clean pass | fail | infra | tasks with a pass |
|---|---|---|---|---|---|---|---|---|
| official1 | 0 of 84 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| val1 | 2 of 2 | 1 | 10 | 10 | 10 | 0 | 0 | 2 |
| official2 | 0 of 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| official3 | 14 of 14 | 5 | 70 | 57 | 57 | 13 | 0 | 14 |
| official4 | 38 of 38 | 11 | 190 | 157 | 156 | 27 | 6 | 38 |
| baihe | 0 of 97 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **total** | 54 |  | 270 | 224 | 223 | 40 | 6 | 54 |

Kimi passes that are in no SFT corpus, including those on tasks that are covered (by queue and campaign):

| queue | passes not in any corpus |
|---|---|
| official1 | probe-1m 1, v3-256k-reason 105, v3-ctrl-old-1m 10, v4-stock-256k 28 |
| val1 | probe-1m 10 |
| official2 | probe-1m 136, v5-official2-256k 1 |
| official3 | probe-1m 57 |
| official4 | probe-1m 157 |
| baihe | v4-stock-256k 30 |
| (non-accepted) | v3-256k-reason 16, v3-256k-reason-pilot 1, v3-ctrl-old-1m 2, v4-stock-256k 9 |

- official1 probe-1m: 1. This is scriggo c01 kimi-r1 (a recovery-probe run), which v1 marked dropped.
- official2 probe-1m: the tasks are in v3 through v5, but their 1M build-probe passes were never used.
- v4 / v5: passes left out of v3 for compaction, broken replies or a literal `</s>`, plus 7 v4 slots that passed after the v3 snapshot (2026-10-29 17:03).
- v3 campaign: none of it went into SFT.
- The new tasks' probe-1m passes are raw verifier passes at 1M context. No length (at most 256k), compaction or broken-reply filter has been applied to them. SFT v3 dropped 41 of 797 probe-1m conversations for length.

### Tasks per SFT corpus, by queue

| queue | tasks | in v1 | in v2 | in v3 | in any |
|---|---|---|---|---|---|
| official1 | 84 | 84 | 84 | 79 | 84 |
| val1 | 2 | 0 | 0 | 0 | 0 |
| official2 | 32 | 0 | 0 | 32 | 32 |
| official3 | 14 | 0 | 0 | 0 | 0 |
| official4 | 38 | 0 | 0 | 0 | 0 |
| baihe | 97 | 0 | 97 | 96 | 97 |

## Non-accepted tasks (`source: "unknown"`)

14 task ids have Kimi rollouts and SFT conversations but no accepted row. All were built by hengyuf: 13 are official1 audit-strong probe copies and 1 is official2 strong-but-not-accepted. The SFT corpora included them as `task_included_because: strong`. They are not in tasks.jsonl. Their own build-probe (probe-1m) rollouts are in no part, so the 115 probe-1m SFT conversations on them have no rollout row.

| task_id | origin run (sft-membership) | Kimi rows (pass/slots) | SFT v1 / v2 / v3 |
|---|---|---|---|
| syn-aiohttp-cand-c01-r1b4-9f77d81d | official1-httpx | v3-256k-reason 2/5; v4-stock-256k 4/5 | 5 / 5 / 9 |
| syn-aiohttp-cand-c05-r1b0-7d9ec1f5 | official1-httpx | v3-256k-reason 5/5; v3-256k-reason-pilot 1/1; v3-ctrl-old-1m 2/5; v4-stock-256k 5/5 | 4 / 4 / 10 |
| syn-gomacro-cand-c02-r1b0-e8804dd9 | official1-scriggo | v3-256k-reason 2/5; v4-stock-256k 1/5 | 1 / 1 / 1 |
| syn-gomacro-cand-c02-r1b3-ed535f70 | official1-scriggo | v3-256k-reason 1/5; v4-stock-256k 4/5 | 2 / 2 / 0 |
| syn-gomacro-cand-c03-r1b1-68058464 | official1-scriggo | v4-stock-256k 2/5 | 1 / 1 / 3 |
| syn-gomacro-cand-c03-r1b4-67b6a828 | official1-scriggo | v4-stock-256k 3/5 | 2 / 2 / 5 |
| syn-gomacro-cand-c05-r1b1-d80cf0a5 | official1-scriggo | v4-stock-256k 3/5 | 2 / 2 / 4 |
| syn-ibis-cand-c05-r1b3-9f713fee | official1-narwhals | v3-256k-reason 0/5; v4-stock-256k 4/5 | 2 / 2 / 5 |
| syn-pylama-cand-c01-r1b4-r2b4-68d85058 | official1-bandit | v3-256k-reason 4/5; v4-stock-256k 3/5 | 4 / 4 / 7 |
| syn-risor-cand-c01-r1b4-74844668 | official2-tengo-callable | v5-official2-256k 5/5 | 0 / 0 / 5 |
| syn-risor-cand-c03-b214824f | official1-absslices | v4-stock-256k 5/5 | 5 / 5 / 9 |
| syn-wasmtime-cand-c03-r1b4-4d3d43a0 | official1-wasmi | v4-stock-256k 4/5 | 5 / 5 / 8 |
| syn-yq-cand-c01-r1b3-b5f105ff | official1-ytt | v4-stock-256k 2/5 | 2 / 2 / 4 |
| syn-zsa-cand-c02-r1b1-ae1eb794 | official1-dynamodb | v3-256k-reason 2/5; v4-stock-256k 4/5 | 4 / 4 / 8 |

## Merge checks

All 25 checks passed (`summary.json` → `checks`). They cover:

- Row counts and pass/fail/infra totals per part, matching each collector's report.
- No duplicate task ids across the three task parts.
- No duplicate (task_id, slot, campaign) key or path across the rollout parts.
- Every accepted task has exactly kimi-r0..r4 in probe-1m, and `probe_kimi` equals those rows.
- The v4/v5 collector's `source` agrees with tasks.jsonl on every accepted task.
- Every SFT member that matches a rollout row is a `pass`.
- Membership rows that match no rollout are all explained: pilot conversation outside the 5-slot grid (v3 3); probe-1m conversation on a non-accepted task (v1 39, v2 39, v3 37).
- 21 probe-1m members point at attempts outside the rollout's scored round dir. They are the post-hoc and recovery re-scores of official1-scriggo c01 and official1-ytt c01-r1b4, and both sides say pass.

## Caveats (condensed from the collectors)

- **Statuses are the recorded grades.** The official1 grading-repair records (decision `keep`) changed no Kimi status. Their 3 confirmed flips are sc14-only and are not reflected in `probe_sc14`.
- **official1.**
  - 9 tasks are second-round evolutions (`-r2bN`, round 2).
  - scriggo c01 is read from its post-hoc score dir: Kimi 4 pass / 1 fail, with kimi-r1/r2/r4 from recovery probe-89823377. Read literally, the round-dir index.md would give official1 338/74/8 instead of 341/74/5 (official1 + val1: 348/74/8 instead of 351/74/5). tasks.jsonl `status_dir` points at the post-hoc score dir for this task and for official1-ytt syn-yq-cand-c01-r1b4 (its sc14-r3 is fail there, infra in the stale round-dir index.md); for every other task `status_dir` = `scored_dir`.
  - The 5 Kimi infra slots are absslices c03-r1b2 r3, absslices c05-r1b2 r4, scriggo c03 r3, wasmi c03 r1 and wasmi c03-r1b2 r3.
- **official4.**
  - 11 tasks were teacher re-probed on 2026-11-05/06 with the same 1M settings, and their Kimi rows come from the re-probe. 13 tasks were accepted by re-scoring. The 55 superseded Kimi slots are ignored.
  - 6 Kimi infra slots (empty patch) remain in 3 tasks: textual-richlog harlequin c03-r1b3 and sqlite sqlitebiter c05-r1b0 / c05-r1b1. Their cause was not verified.
  - 1 pass is flagged cheated (yaegi-embed gomacro c05 kimi-r4). Its status stays `pass`, so drop it in SFT selection.
- **Baihe.** No failure_match audit (audit_verdict null), no teacher re-probe, and grading by the DeepSWE grader only. The probe `path` is a clipped render; the full trajectory is the attempt's `export.json` under `round-<k>/probe-*/models/kimi/attempts/`.
- **v3 campaign.**
  - Only phase 1 ran: sel50, the pilot and the 1M control. Phase 2 (rest47 + Baihe) was replaced by v4.
  - 28% of the sel50 rollouts (70/250) were cut short by the 2026-10-26 hub broken-reply regression and stay `fail`. Its raw pass rate (48.8%) is therefore not comparable to v4's.
  - The pilot task appears under all three v3 labels.
- **v4/v5.**
  - Not in the rows: 3 pilot rollouts (2 v4, 1 v5; all pass, outside the 5-slot grids), 74 replaced first attempts, and 14 ATP in-slot retries (13 in the main campaigns, 1 inside the broken-reply re-run campaign rest47-g1-brk1).
  - 8 v4 agent timeouts with an empty patch are kept as `fail`; a stricter reading would call them infra.
  - The v5 `usable` rule would give v4 744 instead of 745 (the zsa kimi-r2 refinement).
- **SFT v3 snapshot.** v3 was built at 2026-10-29 17:03. The 3 non-compacted v4 passes that came in later are in no corpus. 41 over-length v2 conversations were dropped from v3, so 7 v2 tasks have no v3 conversation.
- **Paths.** `/lustre/fsw/portfolios/nemotron/users/hengyuf` is a symlink to `/lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/hengyuf`. Some source records use the long form; the parts use the short one.

## Provenance

- Date: 2026-11-06. Inputs were read from `parts/` at merge time:

| part | rows | sha256 |
|---|---|---|
| `rollouts-probe-baihe.jsonl` | 485 | `7b1cec92480ab941901b441d22afa8f49be747fd98cfa1977ddd63933c90f3eb` |
| `rollouts-probe-hengyuf-early.jsonl` | 430 | `935986beea358e58b11450f5cc778d73e88140aa3638a7c0f011a77b6d5ecd74` |
| `rollouts-probe-hengyuf-late.jsonl` | 420 | `563b1b34d7f76fefb20740e6d6c93c370af83dc3d9f222277375e70b593e5cb9` |
| `rollouts-v3.jsonl` | 271 | `496d68d4cb0777759da78ea5dd71d7659025c421391536440cc2247144dfd61f` |
| `rollouts-v4.jsonl` | 970 | `dedf7c4bbd711e7930db297d462a276d78a36b3ed5de0a219af70ab23919ffb3` |
| `rollouts-v5.jsonl` | 165 | `3a82007666fda6cbe839d8074d4d33abbca4cb98732507f42f901baf8aebb607` |
| `sft-membership.jsonl` | 2770 | `24127c2214abf0c46978290364f1dc438e60008d797da94ccf478969df3f8877` |
| `tasks-baihe.jsonl` | 97 | `1e8de25ae0ee8ad4bbbca1b1994edb9c693123502568e71e9198ef2d025161a6` |
| `tasks-hengyuf-early.jsonl` | 86 | `00b54ed1d056dee6fc96bdc736b38583bd7668ce2ffcc0b92db65dde90bd9e93` |
| `tasks-hengyuf-late.jsonl` | 84 | `22b86d83f8287d7bb6102b7f90e9289c12c08c9defcac22021393acf31f77ea1` |

- Collectors (scripts in mode-700 scratch dirs under `/tmp/claude-158141/`):
  - `inventory-hengyuf-early/build_parts.py`: official1 and val1 tasks plus their probe-1m Kimi rows.
  - `inventory-hengyuf-late/collect.py`: official2, official3 and official4.
  - `inventory-baihe/collect_baihe.py`: Baihe's 19 runs.
  - `inventory-v3/build_v3.py`: the v3 campaign.
  - `inventory-kimi-v4v5/build.py`: v4 and v5.
  - `inventory-sft-membership/build_membership.py`: SFT v1, v2 and v3 selections.
  - This merge: `inventory-merge/merge.py`, which renders this file with `readme.py`.
- Source trees (read-only):
  - hengyuf runs: `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth/runs/` (`official1-*`, `val1`, `official2-*`, `official3-*`, `official4-*`).
  - official4 teacher re-probe: `/lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/hengyuf/swe-task/synth/repair/reprobe-teacher-official4-20261105`.
  - Baihe runs: `/lustre/fsw/portfolios/nemotron/users/baiheh/swe-task-hengyu/synth/runs/`.
  - Campaigns: `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v3-20261026`, `kimi-rollouts-v4-stock256k-20261028` and `kimi-rollouts-v5-official2-20261028`.
  - Corpora: `/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/{kimi-pass-v1-20261023,kimi-pass-v2-20261023,sft-v3-20261029}`.
- No hub or network call was made, and no secrets, .env or credential files were opened. The collectors did not call ATP's slot_state (it writes caches); they re-derived statuses read-only.

## Compactions (added 2026-11-06, workflow wf_bebdd29a-091)

`parts/compactions.jsonl` (2,741 rows, sha256 eeab82a2...) gives every slot's OpenCode compaction count, counted with the exact
criterion of `sft/kimi-rollouts-v4-stock256k-20261028/features.py` (main-session export messages with info.mode "compaction"
or info.summary true). The scanner reproduced v4's kimi4-all (73), v3's kimi3 (9) and kimi-1m (0) slot by slot. An
independent verifier used other signals (ATP http.jsonl summarizer calls, the rebased "Continue ..." request,
message-count and prompt-token drops; about 338 GB of http logs) and agreed on all 2,741 rows.

| campaign | slots | compacted | compacted pass / fail |
|---|---|---|---|
| probe-1m (1M) | 1,335 | 0 | 0 / 0 |
| v3-256k-reason | 250 | 9 | 4 / 5 |
| v3 pilot, v3-ctrl-old-1m | 21 | 0 | 0 / 0 |
| v4-stock-256k | 970 | 73 | 52 / 21 |
| v5-official2-256k | 165 | 0 | 0 / 0 |
| **total** | 2,741 | **82** | 56 / 26 |

- Every compacted slot has exactly one automatic compaction (OpenCode 1.17.11 compacts when the last main step's
  prompt + output exceeds limit.input - 20,000 = 242,144), none ended on it, and none is in SFT v1, v2 or v3.
- probe-1m never reaches the 1M compaction point (largest main-session prompt 505,801). But 42 probe-1m slots
  (35 pass) have a prompt above 262,144 tokens; 4 of those passes are on tasks new since SFT v3
  (val1 syn-risor-cand-c02-r1b0 kimi-r0; official4 syn-harlequin-cand-c02-r1b3 kimi-r0/r1/r4).
- 5 infra slots have no export (compactions null; the http logs show no summarizer call).
- The 224 probe-1m passes of the 54 tasks new since SFT v3 have 0 compactions.

## All tasks list (added 2026-11-08)

User request (hengyuf, 2026-11-08): "给我一个所有task(包括没有accepted 或者标记为strong)的汇总list". `all_tasks.jsonl` / `all_tasks.tsv` list **every built synthetic task, accepted or not: 2203 rows** (hengyuf 1571, Baihe 632). A task is *built* when its cand dir `<runs>/<run>/build/<target>/<owner__repo>/<cand>/` has a `stage.json` with `ok: true`; the row's `task_id` is the one in that stage.json. tasks.jsonl above stays the accepted-only view; this list adds the 1936 tasks that were built but are not accepted (rejected, withdrawn or never scored), each with its failure-match audit label.

### Files

| file | rows | content |
|---|---|---|
| `all_tasks.jsonl` | 2203 | One JSON object per built task, keys in the column order below (UTF-8). |
| `all_tasks.tsv` | 2203 + header | Same columns, tab-separated. Lists are comma-joined, null is an empty cell, booleans are `true`/`false`. |
| `all_tasks_summary.json` | - | All counts below, the subsets with task lists, per-run counts, special cases, not-built cands, skipped dirs, cross-checks and the sha256 of every inventory input. |

Rows are sorted by source, queue, run, round, cand (plain ascending order, so `baihe` rows come first and `val1` follows `official4`). Builder: `/tmp/claude-158141/alltasks-build/build_all_tasks.py` with `readme_section.py` (stdlib only). It is read-only on every source, and it writes only these three files plus this section (the previous README is kept as `README.md.pre-all-tasks`).

### Columns

| column | meaning and source |
|---|---|
| `source`, `queue`, `run` | hengyuf (queues official1-4 and val1) or baihe (queue `baihe`, runs `pilot*` and `q*`); `run` is the run dir name. |
| `target`, `repo` | The DeepSWE target (`build/<target>`) and the synthetic host repo (`<owner__repo>` as `owner/repo`). |
| `task_id`, `cand`, `kind`, `title`, `task_dir` | From the cand's stage.json (`task_dir` is the packaged task, `<cand>/task/<task_id>`). |
| `round` | k of the `round-<k>/<cand>/` dir that holds the cand. A cand without any round dir gets the round its name implies (`-r1bN` 1, `-r2bN` 2, else 0) and the flag `round_from_name`. |
| `parent` | Evolve tasks: the cand it was evolved from, i.e. the name minus its last `-rKbN` (round-2 `cand-c01-r1b4-r2b4` has parent `cand-c01-r1b4`). It equals the `name` in the cand's `evolve/inputs/metrics.json` on every evolve cand that has that file. metrics.json `lineage` is the root mask cand instead. Empty for mask tasks. |
| `status` | `accepted`: metrics.json `accept: true`, and the task is in the run's `accepted/tasks.jsonl` (both directions checked). `withdrawn`: accepted first and withdrawn later (metrics.json `withdrawn` block, `withdrawn/withdrawn.jsonl`, `metrics.json.pre-withdraw`). `rejected`: scored with `accept: false`. `unscored`: built, but no `round-<k>/<cand>/metrics.json`. |
| `reject_reason` | For rejected and withdrawn tasks, from metrics.json `reason`: `out_of_range` ("pass rates out of range"), `similarity_below_6` ("similarity N/10" with N < 6, including the 2 official1 grading-repair re-scores), `inconclusive` (fewer than 3 valid rollouts per model), `leaky` ("teacher retrieved the answer by action", answer_retrieval), else `other:<reason>`. Null for accepted and unscored tasks. |
| `similarity`, `student_rate`, `teacher_clean_rate` | From the final metrics.json (null if unscored). |
| `sc14_pass`, `sc14_graded`, `kimi_pass`, `kimi_graded` | Build probe (sc14 student and Kimi K3 teacher slots; 2181 tasks with sc14 5 + kimi 5): pass and pass + fail slots (infra excluded) from the `rollouts/index.md` in `status_dir`. They match metrics.json `valid` and the rates on every scored task except 3 withdrawn ones (see special cases). `kimi_pass` counts raw passes, including any the teacher audit flagged (flag `kimi_cheated:N`). |
| `audit` | `verdict` of `<runs>/<run>/audit/failure_match/<task_id>.json` (strong, partial, weak, none), `not_audited` if the file is missing. The files already hold the final verdict: all 175 reviews in `audit/failure_match_review/` (68 changed the verdict; 94 files have a `.pre-review` backup) match the current file. Baihe's runs have no audit dir, so every Baihe row is `not_audited`. |
| `accepted_or_strong` | `status == accepted` or `audit == strong`. |
| `in_v2_taskset` | The task is one of the 194 tasks of `kimi-rollouts-v4-stock256k-20261028` (union of `tasks` in its `results/kimi-*.json`; = the v2 SFT task set). |
| `sft_v1`, `sft_v2`, `sft_v3` | Conversations of the task in each SFT corpus (rows of `parts/sft-membership.jsonl`). |
| `kimi_v3_pass`, `kimi_v4_pass`, `kimi_v5_pass` | Kimi passes in `kimi_rollouts.jsonl` for campaigns `v3-256k-reason` (the sel50 arm only: the v3 pilot slot and the 1M ctrl arm are not counted), `v4-stock-256k` and `v5-official2-256k`. Null if that campaign has no slot on the task. |
| `kimi_compacted` | Slots of the task with at least one OpenCode compaction in `parts/compactions.jsonl`, all campaigns. Null if the task has no row there. |
| `accepted_dir` | `<runs>/<run>/accepted/<task_id>` for accepted tasks, else null. |
| `scored_dir` | `round-<k>/<cand>/` with the metrics.json (null if unscored). |
| `status_dir` (extra) | The dir whose `rollouts/index.md` gave the probe counts: = `scored_dir`, except for the 2 post-hoc accepts (post-hoc score dir, as `status_dir` in tasks.jsonl), the 2 official1 grading repairs (the repair dir), and one unscored Baihe cand that has rollouts but no metrics.json. |
| `flags` (extra) | Special cases: `posthoc_accept`, `grading_repair`, `counts_from_metrics`, `rescored` (the task has a `score` event in the build root's `rescore.jsonl`), `teacher_reprobe` (`teacher_reprobe.json` in the scored dir), `answer_retrieval`, `kimi_cheated:N` (metrics.json `cheated`: normally the build's teacher audit; for the withdrawn official1-testem cand-c03 it was set by main when the task was withdrawn on 2026-10-18, its teacher_audit.json flags no slot), `rollouts_without_metrics`, `round_from_name`, `task_id_collision`. |

The inventory columns (`in_v2_taskset`, `sft_*`, `kimi_v*_pass`, `kimi_compacted`) are joined on (run, task_id), where the run is the task's run in tasks.jsonl or, for the 14 non-accepted inventory tasks, the `run` in sft-membership. The run is needed because task ids are not unique across runs (see special cases): a build that only shares the id gets 0 / false / null.

### Counts by source and status

| source | accepted | withdrawn | rejected | unscored | built |
|---|---|---|---|---|---|
| hengyuf | 170 | 5 | 1386 | 10 | 1571 |
| baihe | 97 | 0 | 522 | 13 | 632 |
| **total** | 267 | 5 | 1908 | 23 | 2203 |

### Counts by source and audit

| source | strong | partial | weak | none | not_audited | built |
|---|---|---|---|---|---|---|
| hengyuf | 70 | 224 | 397 | 853 | 27 | 1571 |
| baihe | 0 | 0 | 0 | 0 | 632 | 632 |
| **total** | 70 | 224 | 397 | 853 | 659 | 2203 |

Status by audit (all hengyuf; Baihe is all `not_audited`):

| status | strong | partial | weak | none | not_audited | total |
|---|---|---|---|---|---|---|
| accepted | 36 | 78 | 47 | 7 | 2 | 170 |
| withdrawn | 0 | 1 | 3 | 0 | 1 | 5 |
| rejected | 34 | 145 | 347 | 846 | 14 | 1386 |
| unscored | 0 | 0 | 0 | 0 | 10 | 10 |
| **total** | 70 | 224 | 397 | 853 | 27 | 1571 |

### By queue

| queue | runs with a built task | built | accepted | withdrawn | rejected | unscored | strong | strong, not accepted | not_audited | in v2 task set |
|---|---|---|---|---|---|---|---|---|---|---|
| official1 | 23 of 23 | 668 | 84 | 5 | 574 | 5 | 47 | 19 | 11 | 97 |
| val1 | 1 of 1 | 4 | 2 | 0 | 2 | 0 | 0 | 0 | 4 | 0 |
| official2 | 14 of 14 | 324 | 32 | 0 | 287 | 5 | 2 | 1 | 10 | 0 |
| official3 | 10 of 11 | 256 | 14 | 0 | 242 | 0 | 3 | 3 | 2 | 0 |
| official4 | 15 of 23 | 319 | 38 | 0 | 281 | 0 | 18 | 11 | 0 | 0 |
| baihe | 22 of 25 | 632 | 97 | 0 | 522 | 13 | 0 | 0 | 632 | 97 |
| **total** | 85 of 97 | 2203 | 267 | 5 | 1908 | 23 | 70 | 34 | 659 | 194 |

Per-run counts (built, status, audit, not-built cands) are in `all_tasks_summary.json` → `per_run`.

### Reject reasons

| status / source | out_of_range | similarity_below_6 | inconclusive | leaky | other | total |
|---|---|---|---|---|---|---|
| rejected / hengyuf | 1208 | 147 | 28 | 3 | 0 | 1386 |
| rejected / baihe | 484 | 29 | 6 | 3 | 0 | 522 |
| withdrawn / hengyuf | 0 | 0 | 0 | 1 | 4 | 5 |
| **total** | 1692 | 176 | 34 | 7 | 4 | 1913 |

- `other` occurs only on withdrawn tasks: 4 official1 tasks whose reason is "withdrawn: <why>" (grading artifacts and an over-specified hidden check; each correction put a rate out of range). The fifth withdrawn task, testem `syn-jest-cand-c03-3178414d`, counts as `leaky` (the teacher read jest-worker's code from yarn's cache). The full texts are in `all_tasks_summary.json` → `reject_reasons_other_texts`.
- `similarity_below_6` includes the 2 official1 tasks re-scored after the 2026-10-19 grading repair (fastapi `cand-c05-r1b3` 3/10, ytt `cand-c04` 5/10).

### Subsets

| subset | definition | tasks | by source | by status |
|---|---|---|---|---|
| `accepted` | status accepted | 267 | baihe 97, hengyuf 170 | accepted 267 |
| `strong` | audit strong, any status | 70 | hengyuf 70 | accepted 36, rejected 34 |
| `strong_not_accepted` | audit strong and not accepted | 34 | hengyuf 34 | rejected 34 |
| `accepted_or_strong` | accepted, or audit strong | 301 | baihe 97, hengyuf 204 | accepted 267, rejected 34 |
| `in_v2_taskset` | in the 194-task v2 set | 194 | baihe 97, hengyuf 97 | accepted 181, rejected 13 |
| `in_any_sft` | at least one conversation in SFT v1, v2 or v3 | 227 | baihe 97, hengyuf 130 | accepted 213, rejected 14 |
| `not_accepted_with_kimi_or_sft` | not accepted, but with v3/v4/v5 Kimi rows or SFT conversations | 14 | hengyuf 14 | rejected 14 |

- Strong: 70 tasks, hengyuf 70: 36 accepted and 34 not accepted (rejected 34).
- In the v2 task set: 194 = hengyuf 84 accepted + 13 not accepted (rejected 13; audit strong 13) + Baihe 97 accepted.

### Strong but not accepted (34)

| run | cand | task_id | reject_reason | sim | sc14 / clean kimi rate | sc14 probe | Kimi probe | v2 set | SFT v1 / v2 / v3 | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| official1-absslices | cand-c03 | `syn-risor-cand-c03-b214824f` | out_of_range | - | 0.6 / 1.0 | 3/5 | 5/5 | yes | 5 / 5 / 9 | - |
| official1-bandit | cand-c01-r1b4-r2b4 | `syn-pylama-cand-c01-r1b4-r2b4-68d85058` | out_of_range | - | 0.6 / 0.8 | 3/5 | 4/5 | yes | 4 / 4 / 7 | - |
| official1-dynamodb | cand-c01-r1b1 | `syn-zsa-cand-c01-r1b1-20700476` | out_of_range | - | 0.0 / 0.0 | 0/5 | 0/5 | no | 0 / 0 / 0 | - |
| official1-dynamodb | cand-c02-r1b1 | `syn-zsa-cand-c02-r1b1-ae1eb794` | similarity_below_6 | 5 | 0.2 / 0.8 | 1/5 | 4/5 | yes | 4 / 4 / 8 | - |
| official1-httpx | cand-c01-r1b4 | `syn-aiohttp-cand-c01-r1b4-9f77d81d` | out_of_range | - | 0.4 / 1.0 | 2/5 | 5/5 | yes | 5 / 5 / 9 | - |
| official1-httpx | cand-c05-r1b0 | `syn-aiohttp-cand-c05-r1b0-7d9ec1f5` | out_of_range | - | 0.4 / 0.8 | 2/5 | 4/5 | yes | 4 / 4 / 10 | - |
| official1-narwhals | cand-c05-r1b3 | `syn-ibis-cand-c05-r1b3-9f713fee` | out_of_range | - | 0.0 / 0.4 | 0/5 | 2/5 | yes | 2 / 2 / 5 | - |
| official1-scriggo | cand-c02-r1b0 | `syn-gomacro-cand-c02-r1b0-e8804dd9` | out_of_range | - | 0.0 / 0.2 | 0/5 | 1/5 | yes | 1 / 1 / 1 | - |
| official1-scriggo | cand-c02-r1b3 | `syn-gomacro-cand-c02-r1b3-ed535f70` | out_of_range | - | 0.0 / 0.4 | 0/5 | 2/5 | yes | 2 / 2 / 0 | - |
| official1-scriggo | cand-c03-r1b0 | `syn-gomacro-cand-c03-r1b0-b9f7941f` | out_of_range | - | 0.0 / 0.0 | 0/5 | 0/5 | no | 0 / 0 / 0 | - |
| official1-scriggo | cand-c03-r1b1 | `syn-gomacro-cand-c03-r1b1-68058464` | out_of_range | - | 0.0 / 0.2 | 0/5 | 1/5 | yes | 1 / 1 / 3 | - |
| official1-scriggo | cand-c03-r1b3 | `syn-gomacro-cand-c03-r1b3-81cf3c32` | out_of_range | - | 0.0 / 0.0 | 0/5 | 0/5 | no | 0 / 0 / 0 | - |
| official1-scriggo | cand-c03-r1b4 | `syn-gomacro-cand-c03-r1b4-67b6a828` | out_of_range | - | 0.0 / 0.4 | 0/5 | 2/5 | yes | 2 / 2 / 5 | - |
| official1-scriggo | cand-c04-r1b1 | `syn-gomacro-cand-c04-r1b1-fd7b9d3a` | out_of_range | - | 0.0 / 0.0 | 0/5 | 0/5 | no | 0 / 0 / 0 | - |
| official1-scriggo | cand-c05-r1b1 | `syn-gomacro-cand-c05-r1b1-d80cf0a5` | out_of_range | - | 0.0 / 0.4 | 0/5 | 2/5 | yes | 2 / 2 / 4 | - |
| official1-sqlformatter | cand-c02-r1b2 | `syn-vscode-sqltools-cand-c02-r1b2-c0449990` | out_of_range | - | 0.0 / 0.0 | 0/5 | 0/5 | no | 0 / 0 / 0 | - |
| official1-wasmi | cand-c03-r1b0 | `syn-wasmtime-cand-c03-r1b0-82ff5eca` | out_of_range | - | 0.0 / 0.0 | 0/5 | 0/5 | no | 0 / 0 / 0 | - |
| official1-wasmi | cand-c03-r1b4 | `syn-wasmtime-cand-c03-r1b4-4d3d43a0` | out_of_range | - | 0.4 / 1.0 | 2/5 | 5/5 | yes | 5 / 5 / 8 | - |
| official1-ytt | cand-c01-r1b3 | `syn-yq-cand-c01-r1b3-b5f105ff` | out_of_range | - | 0.4 / 0.4 | 2/5 | 2/5 | yes | 2 / 2 / 4 | - |
| official2-tengo-callable | cand-c01-r1b4 | `syn-risor-cand-c01-r1b4-74844668` | out_of_range | - | 0.4 / 0.6 | 2/5 | 3/5 | no | 0 / 0 / 5 | - |
| official3-obsidian-ignore | cand-c02-r1b1 | `syn-xo-cand-c02-r1b1-e3b67ccd` | out_of_range | - | 0.0 / 0.0 | 0/5 | 0/5 | no | 0 / 0 / 0 | - |
| official3-obsidian-ignore | cand-c04-r1b4 | `syn-xo-cand-c04-r1b4-f406c71d` | out_of_range | - | 0.0 / 0.4 | 0/5 | 2/5 | no | 0 / 0 / 0 | - |
| official3-obsidian-ignore | cand-c05-r1b0 | `syn-xo-cand-c05-r1b0-19ee58b3` | out_of_range | - | 0.0 / 0.0 | 0/5 | 0/5 | no | 0 / 0 / 0 | - |
| official4-anko-typed | cand-c01-r1b3 | `syn-starlark-go-cand-c01-r1b3-567dd577` | out_of_range | - | 0.6 / 0.8 | 3/5 | 4/5 | no | 0 / 0 / 0 | - |
| official4-anko-typed | cand-c03-r1b1 | `syn-starlark-go-cand-c03-r1b1-ad480218` | out_of_range | - | 0.4 / 1.0 | 2/5 | 5/5 | no | 0 / 0 / 0 | - |
| official4-anko-typed | cand-c05-r1b2 | `syn-starlark-go-cand-c05-r1b2-acc2f952` | out_of_range | - | 0.4 / 1.0 | 2/5 | 4/4 | no | 0 / 0 / 0 | - |
| official4-gocritic-doclink | cand-c05-r1b0 | `syn-tools-cand-c05-r1b0-349b07f8` | out_of_range | - | 0.4 / 0.8 | 2/5 | 4/5 | no | 0 / 0 / 0 | rescored |
| official4-igel-schema | cand-c04-r1b4 | `syn-ludwig-cand-c04-r1b4-6dfcb782` | out_of_range | - | 0.4 / 1.0 | 2/5 | 5/5 | no | 0 / 0 / 0 | - |
| official4-kgateway-hash | cand-c04-r1b4 | `syn-gateway-cand-c04-r1b4-ed3362d0` | out_of_range | - | 0.0 / 0.0 | 0/5 | 0/5 | no | 0 / 0 / 0 | - |
| official4-kgateway-hash | cand-c05-r1b1 | `syn-gateway-cand-c05-r1b1-51364292` | out_of_range | - | 0.0 / 0.2 | 0/5 | 1/5 | no | 0 / 0 / 0 | - |
| official4-koota-predicates | cand-c01-r1b3 | `syn-bitecs-cand-c01-r1b3-b5bf04a4` | out_of_range | - | 0.0 / 0.4 | 0/5 | 2/5 | no | 0 / 0 / 0 | rescored |
| official4-koota-predicates | cand-c04-r1b1 | `syn-bitecs-cand-c04-r1b1-37a2db26` | out_of_range | - | 0.6 / 0.6 | 3/5 | 3/5 | no | 0 / 0 / 0 | rescored |
| official4-textual-richlog | cand-c02-r1b4 | `syn-harlequin-cand-c02-r1b4-72e85a55` | out_of_range | - | 0.0 / 0.2 | 0/5 | 1/5 | no | 0 / 0 / 0 | teacher_reprobe |
| official4-valibot-recursive | cand-c03-r1b4 | `syn-zod-cand-c03-r1b4-bdd8fb70` | out_of_range | - | 0.0 / 0.0 | 0/5 | 0/5 | no | 0 / 0 / 0 | - |

- official1: 19 strong-not-accepted tasks. 13 are the audit-strong probe copies in SFT v1/v2 (each has at least one Kimi probe pass); the other 6 have no Kimi probe pass (Kimi 0/5), so there was nothing to copy.
- official2: official2-tengo-callable `cand-c01-r1b4` (SFT v1/v2/v3 0/0/5, v5 Kimi pass 5) is the strong task that v5 and SFT v3 took.
- official3 / official4: 14 strong-not-accepted tasks; none of them is in a corpus or a 256k campaign. 10 have Kimi probe passes (31 passes in total, none flagged cheated). They are the official3/4 counterpart of the official1 strong probe copies, and nothing has used them yet.

### The 14 non-accepted inventory tasks in this list

The 14 task ids above that have v3/v4/v5 Kimi rows and SFT conversations but no accepted row (`source: "unknown"` in kimi_rollouts.jsonl) are all here, each matched to exactly one built row:

| run | cand | task_id | status | reject_reason | sim | sc14 / clean kimi rate | sc14 probe | Kimi probe | audit | v2 set | SFT v1 / v2 / v3 | Kimi pass v3 / v4 / v5 | compacted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| official1-absslices | cand-c03 | `syn-risor-cand-c03-b214824f` | rejected | out_of_range | - | 0.6 / 1.0 | 3/5 | 5/5 | strong | yes | 5 / 5 / 9 | - / 5 / - | 0 |
| official1-bandit | cand-c01-r1b4-r2b4 | `syn-pylama-cand-c01-r1b4-r2b4-68d85058` | rejected | out_of_range | - | 0.6 / 0.8 | 3/5 | 4/5 | strong | yes | 4 / 4 / 7 | 4 / 3 / - | 0 |
| official1-dynamodb | cand-c02-r1b1 | `syn-zsa-cand-c02-r1b1-ae1eb794` | rejected | similarity_below_6 | 5 | 0.2 / 0.8 | 1/5 | 4/5 | strong | yes | 4 / 4 / 8 | 2 / 4 / - | 0 |
| official1-httpx | cand-c01-r1b4 | `syn-aiohttp-cand-c01-r1b4-9f77d81d` | rejected | out_of_range | - | 0.4 / 1.0 | 2/5 | 5/5 | strong | yes | 5 / 5 / 9 | 2 / 4 / - | 0 |
| official1-httpx | cand-c05-r1b0 | `syn-aiohttp-cand-c05-r1b0-7d9ec1f5` | rejected | out_of_range | - | 0.4 / 0.8 | 2/5 | 4/5 | strong | yes | 4 / 4 / 10 | 5 / 5 / - | 0 |
| official1-narwhals | cand-c05-r1b3 | `syn-ibis-cand-c05-r1b3-9f713fee` | rejected | out_of_range | - | 0.0 / 0.4 | 0/5 | 2/5 | strong | yes | 2 / 2 / 5 | 0 / 4 / - | 0 |
| official1-scriggo | cand-c02-r1b0 | `syn-gomacro-cand-c02-r1b0-e8804dd9` | rejected | out_of_range | - | 0.0 / 0.2 | 0/5 | 1/5 | strong | yes | 1 / 1 / 1 | 2 / 1 / - | 5 |
| official1-scriggo | cand-c02-r1b3 | `syn-gomacro-cand-c02-r1b3-ed535f70` | rejected | out_of_range | - | 0.0 / 0.4 | 0/5 | 2/5 | strong | yes | 2 / 2 / 0 | 1 / 4 / - | 9 |
| official1-scriggo | cand-c03-r1b1 | `syn-gomacro-cand-c03-r1b1-68058464` | rejected | out_of_range | - | 0.0 / 0.2 | 0/5 | 1/5 | strong | yes | 1 / 1 / 3 | - / 2 / - | 0 |
| official1-scriggo | cand-c03-r1b4 | `syn-gomacro-cand-c03-r1b4-67b6a828` | rejected | out_of_range | - | 0.0 / 0.4 | 0/5 | 2/5 | strong | yes | 2 / 2 / 5 | - / 3 / - | 0 |
| official1-scriggo | cand-c05-r1b1 | `syn-gomacro-cand-c05-r1b1-d80cf0a5` | rejected | out_of_range | - | 0.0 / 0.4 | 0/5 | 2/5 | strong | yes | 2 / 2 / 4 | - / 3 / - | 1 |
| official1-wasmi | cand-c03-r1b4 | `syn-wasmtime-cand-c03-r1b4-4d3d43a0` | rejected | out_of_range | - | 0.4 / 1.0 | 2/5 | 5/5 | strong | yes | 5 / 5 / 8 | - / 4 / - | 0 |
| official1-ytt | cand-c01-r1b3 | `syn-yq-cand-c01-r1b3-b5f105ff` | rejected | out_of_range | - | 0.4 / 0.4 | 2/5 | 2/5 | strong | yes | 2 / 2 / 4 | - / 2 / - | 0 |
| official2-tengo-callable | cand-c01-r1b4 | `syn-risor-cand-c01-r1b4-74844668` | rejected | out_of_range | - | 0.4 / 0.6 | 2/5 | 3/5 | strong | no | 0 / 0 / 5 | - / - / 5 | 0 |

- All 14 are `rejected` and audit `strong`; reasons: out_of_range 13, similarity_below_6 1. 13 are in the v2 task set; official2-tengo-callable `cand-c01-r1b4` is not (v5 only).
- They are 14 of the 34 strong-not-accepted tasks; the other 20 have no 256k rollout and no conversation.

### Special cases

- **Post-hoc accepts (2, official1)**: official1-scriggo `cand-c01`, official1-ytt `cand-c01-r1b4`. Accepted after recovery re-runs. Their probe counts come from the post-hoc score dir (`status_dir`, as in tasks.jsonl).
- **Withdrawn (5, official1)**: official1-bandit `cand-c05-r1b1`, official1-dasel `cand-c01-r1b4`, official1-fastapi `cand-c04-r1b2`, official1-testem `cand-c01`, official1-testem `cand-c03`. Each was accepted and later withdrawn (grading artifact 3, over-specified hidden check 1, teacher answer retrieval 1). The accepted dir was moved to `<run>/withdrawn/<task_id>`, and the reason is in metrics.json `withdrawn.why`.
- **Recorded grades versus corrected rates (3 withdrawn tasks)**: official1-bandit `cand-c05-r1b1` index.md sc14 1/5 kimi 5/5, metrics sc14 2/5 kimi 5/5; official1-fastapi `cand-c04-r1b2` index.md sc14 1/5 kimi 5/5, metrics sc14 4/5 kimi 5/5; official1-testem `cand-c01` index.md sc14 0/5 kimi 3/5, metrics sc14 0/5 kimi 2/5. The recheck corrected the rates, but the slot grades in index.md were left as recorded. The row takes its counts from metrics.json (`valid` x rate, flag `counts_from_metrics`) so that they match the row's rates. The recorded counts are in the summary.
- **Grading repairs (2, rejected 2)**: official1-fastapi `cand-c05-r1b3`, official1-ytt `cand-c04`. metrics.json was re-scored after a 2026-10-19 grading repair (`repaired` block, `.pre-repair` / `.pre-rescore` backups). The scored dir's index.md is stale, so the counts come from the repair dir's index.md.
- **Re-scored (official4, 71 tasks)**: these have a `score` event in their build root's `rescore.jsonl` (tools/rescore_unscored.py, 2026-11-04). Their metrics.json equals the last re-score. Statuses: accepted 13, rejected 58.
- **Teacher re-probed (official4, 19 tasks)**: these have `teacher_reprobe.json` (status done, 2026-11-06) in the scored dir. metrics.json and rollouts/ hold the re-probe (its `new_metrics` equal metrics.json), and the old ones are kept as `*.pre-teacher-reprobe`. Statuses: accepted 11, rejected 8.
- **Unscored (23)**: official1-narwhals 5, official2-ytt 5, pilot2-expr-try-catch 1, q21-boa-hierarchical-evaluation-cancellation 12. These were built but never got a metrics.json. pilot2-expr-try-catch `cand-c02` has probe rollouts (index.md: sc14 5/5, Kimi 4/5) but no metrics.json; its counts come from that index.md (flag `rollouts_without_metrics`). The other unscored rows have no round dir at all (flag `round_from_name`) and null counts.
- **Task id collisions (17 ids, 34 rows)**: two different builds (different cand dir, snapshot and title) share an id: pilot3-dateutil-rfc5545 / pilot4-dateutil-rfc5545-codex 15 ids; official1-absslices / val1 2 ids. Use (source, run, cand) or `task_dir` as the key, not `task_id`. 2 of the shared ids are accepted in one of their runs (pilot3-dateutil-rfc5545 `cand-c05-r1b3`, val1 `cand-c02-r1b0`); tasks.jsonl and the inventory refer to those builds.
- **Not built (144 cand dirs, not in the list)**: hengyuf stage.json ok: false 61, hengyuf round3-cancelled (no stage.json) 38, hengyuf no stage.json (build not finished) 27, baihe no stage.json (build not finished) 2, baihe stage.json ok: false 16. The list is in the summary (`not_built_cands`).
- **Skipped dirs**: smoke runs (hengyuf smoke1, baihe codex-check-20477274, baihe smoke1); abandoned builds (official1-httpx/abandoned-psf__requests-pip-vendor-leak 5 cands, 0 with a stage.json, official1-testem/abandoned-avajs__ava-released-copy-leak 5 cands, 0 with a stage.json; ABANDONED.txt: every round-0 candidate failed the leak scan and the run was relaunched on another repo); 7 official4 runs that hold only `target_source.json` and were never built (official4-clack-autocomplete, official4-expr-trycatch, official4-httpx-multipart, official4-katex-multicolumn, official4-opa-template, official4-optique-deps, official4-textual-kitty); and the `slurm` log dirs.

### Cross-checks (all passed)

- Accepted: hengyuf 170 and Baihe 97. These are exactly the task ids of tasks.jsonl (no id only on one side). On all 267 accepted tasks, run, queue, target, repo, round, cand, scored_dir, status_dir, accepted_dir, similarity, rates, audit, the probe pass/graded counts, the SFT counts and the teacher_reprobe flag equal tasks.jsonl (0 diffs).
- Every accepted task is in its run's `accepted/tasks.jsonl` and vice versa. Every withdrawn task is in `withdrawn/withdrawn.jsonl` and has a `metrics.json.pre-withdraw`. Every scored dir belongs to a built cand. Every failure_match audit file belongs to a built task of its run, with matching task_id, run and cand.
- `in_v2_taskset`: 194 rows. This is the same set as the v4 results' task lists and as the v4-stock-256k task ids of kimi_rollouts.jsonl.
- Column totals equal the inventory: Kimi passes v3 121, v4 764, v5 139, compacted slots 82; SFT conversations v1 379, v2 797, v3 1594. Every task id of kimi_rollouts.jsonl, compactions.jsonl, sft-membership.jsonl and the v4 task lists lands on exactly one row.
- Probe counts agree with metrics.json `valid` and the rates on every scored task (2176 from the scored dir, 2 post-hoc, 2 repair dirs), except the 3 withdrawn tasks above. Evolve parents agree with `evolve/inputs/metrics.json` on 1805 cands.

### Caveats

- `audit` is the failure-match verdict only. Baihe never ran that audit, so `strong` and `accepted_or_strong` can only come from hengyuf runs. The 27 hengyuf `not_audited` rows are 10 unscored tasks, the runs without any audit file (official2-bandit-taint 5, val1 4) and 8 scored tasks without an audit file (official1-fastapi `cand-c05-r1b1` (rejected), official1-httpx `cand-c02-r1b3` (rejected), official1-narwhals `cand-c02` (rejected), official1-sqlformatter `cand-c01-r1b2` (rejected), official1-testem `cand-c03` (withdrawn), official1-testem `cand-c05-r1b2` (rejected), official3-goreleaser-retry `cand-c01-r1b2` (rejected), official3-goreleaser-retry `cand-c02-r1b4` (rejected)).
- The probe counts are the recorded grades. The official1 grading-repair records (`audit/grading_repair/`) are advisory. They count here only where metrics.json itself was changed: the 2 repairs, the 2 post-hoc accepts and the withdrawals. 5 records recommend `newly_in_range`: official1-fastapi `cand-c05-r1b3` → rejected (similarity_below_6), official1-scriggo `cand-c01` → accepted, official1-vitest `cand-c05-r1b1` → rejected (out_of_range), official1-vitest `cand-c05-r1b4` → rejected (out_of_range), official1-ytt `cand-c01-r1b4` → accepted. `cand-c05-r1b1` and `cand-c05-r1b4` stay rejected; their repair notes say they are in range only through a jest-worker mock coupling and that a similarity judge was still needed.
- The 1M probe passes of non-accepted tasks are counted in `kimi_pass` but have no rows in kimi_rollouts.jsonl or compactions.jsonl, because the inventory collected probe-1m slots only for accepted tasks. So `kimi_compacted` of a non-accepted task covers only its v3/v4/v5 slots.
- `kimi_v3_pass` is the sel50 arm (`v3-256k-reason`) only. Not counted: the v3 pilot slot (`syn-aiohttp-cand-c05-r1b0-7d9ec1f5` 1/1) and the 1M ctrl arm (`v3-ctrl-old-1m`: 3 accepted tasks 10/15 pass, 1 non-accepted task 2/5).
