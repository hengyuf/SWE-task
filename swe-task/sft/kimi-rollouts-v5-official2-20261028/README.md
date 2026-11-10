# kimi-rollouts-v5-official2-20261028

Kimi-K3 rollouts on the second official batch: every accepted task and every task labelled failure-match "strong" in
the 14 `synth/runs/official2-*` runs (the Codex-reselected rank 2-5 host repos). 5 rollouts per task (33 tasks, 165
rollouts), with v4's settings: OpenCode's **stock** system prompt and a **262,144-token** context window.

User request (hengyuf, 2026-10-28, verbatim): "这组rollout 跑完之后,再给official的第二批(找rank2-5 repo 的那一批) 里的accepted
tasks 和标记为strong的task 每个task 额外做5个kimi 的256 k context rollout". "这组rollout" is kimi-rollouts-v4, so this batch
starts only after v4 has finished (`submit.sh` checks that, see Operating).

Broken-reply re-runs: the user approved one re-run per slot for v4 ("这次若遇到被坏回复终止的 slot, 可以重跑1遍",
2026-10-28). Main applies the same rule to this follow-on batch: a slot whose session a broken reply ENDED gets exactly
one re-run, which replaces it in the results with both records kept. There is never a third attempt.

STATUS 2026-10-28 ~17:10: prepared and validated offline. Nothing has been launched.

STATUS 2026-10-29 02:51: LAUNCHED while v4 still runs. User, 2026-10-29 ~02:00 (verbatim): "v5 可以直接开始跑了", so v5
no longer waits for kimi-rollouts-v4 (v4 untouched, ~60 in flight at its 01:31 row; v4 + v5 ~92 < the ~110 v4 started
with).
- 02:01-02:03 re-validation: `official2_tasks.py` exit 0 (same 33 tasks), `validate_offline.py` all_ok, `snapshot_check.py`
  33/33 images OK under CELL3 (the 10-28 result is kept as `analysis/snapshot-check-20261028.json`).
- Pilot (main asked for v4's 1-rollout pilot before the launch): `kr5-pilot` job 20989652 (cpu_interactive), 02:09-02:48,
  `syn-markdown-cand-c04-r1b0-3b72285b` -> pass (F2P 40/40, P2P 1747/1747), not broken-ended, 82 replies all 200, reported
  cost $2.36. `verify_pilot.py` (with `--previous` = a build-time 1M Kimi attempt of the same task): all 13 checks OK
  (`analysis/verify-pilot-0-0-attempt-0.json`).
- Campaigns: `kr5-official2-g0..g3` = jobs 20990515 / 20990517 / 20990519 / 20990521 (rl_algo candidates won; cpu_interactive),
  started 02:51:42 on NVAPI_KEY_NEW_2 ($6,414.84 left), CELL3, stock prompt, 262144, cap 8 each.

## Task list (`official2-tasks.json`, written by `official2_tasks.py --write`)

| | count |
|---|---|
| official2 runs | 14 (4 with no accepted task: arcane, bandit-taint, tengo-callable, ytt) |
| accepted (`<run>/accepted/tasks.jsonl`) | 32 |
| failure-match "strong" (`<run>/audit/failure_match/<id>.json` verdict, reviewed) | 2 |
| both | 1: `syn-knex-cand-c01-r1b3-a888f2f2` (kysely) |
| strong but not accepted | 1: `syn-risor-cand-c01-r1b4-74844668` (tengo-callable; student 0.4 > 0.2) |
| withdrawn (excluded) | 0 (no `withdrawn/` dir, no `withdrawn` field in any official2 metrics.json) |
| **tasks** | **33** (= main's count), 33 distinct snapshots |

Accepted per run: tengo-destructuring 7, pebble 5, aiomonitor 4, obsidian 4, adaptix 3, etree 3, kysely 2,
promreload 2, cattrs 1, mobly 1.

- An accepted task is rolled out from `<run>/accepted/<task_id>` (its installed grader).
- The strong non-accepted task is rolled out from its scored candidate dir: the `task_dir` of the
  `round-1/cand-c01-r1b4/metrics.json` with that task_id,
  `runs/official2-tengo-callable/build/tengo-callable-instance-isolation/deepnoodle-ai__risor/cand-c01-r1b4/task/syn-risor-cand-c01-r1b4-74844668`.
  It is byte-identical to the probe copy `round-1/probe-cc69cee8/tasks/<id>`.
- Every task dir has a non-empty instruction.md, and task.toml with metadata.task_id, `docker_image = "snapshot:<uuid>"`
  and the verifier / agent timeouts. For accepted tasks the image equals the tasks.jsonl row's. Every tests/ dir has
  grader.py, test.sh, config.json and test.patch.
- All 33 grader.py files are the pipeline's current `synth/verifier/grader.py` (sha256 `b61ee8ac...`).
- The reviewed labels in `audit/failure_match/` agree with every `audit/failure_match_review/` decision (64 files, 0
  mismatches). The 4 pre-review "strong" labels that the review lowered are not included.
- The file also stores per-task digests (instruction.md, task.toml, every tests/ file). driver.py re-checks them at
  every leg and stops with a `task_changed` leg record if a task changed. The same check runs on every `--dry-run`.
- Re-derive and compare: `<ATP controller python> -I -B official2_tasks.py` (exit 0 = the runs still give this list).

## Settings: identical to v4 except the tasks and the names

| | v4 (`../kimi-rollouts-v4-stock256k-20261028`, read-only) | **v5 (this dir)** |
|---|---|---|
| model | Kimi-K3 `nvidia/moonshotai/kimi-k3`, reasoning_effort max | same |
| system prompt | OpenCode 1.17.11 stock (`KR3_PROMPT=stock`) | same |
| OpenCode window | 262,144 (`limit.context = limit.input`), output 131,072 | same |
| harness, grading | synth `rollouts.probe` (ATP, OpenCode), each task's installed grader | same |
| hub key | NVAPI_KEY_NEW_2, then NVAPI_KEY_NEW_3 (guard) | same |
| sandbox credential | CELL3 (hengyuf tasks), BAIHE (Baihe tasks) | **CELL3 only** (S/atp/.env; official2 = hengyuf's pipeline) |
| tasks | 194 v2 tasks (sel50, rest47, baihe; pilots) | **33 official2 tasks** (`official2`, 4 groups) |
| broken-ended slots | re-run once (`<campaign>-brk<n>`) | same (main applies the user's v4 rule) |
| budget-refusal recovery | `<campaign>-rec<n>` | same |
| job / legs prefix | kr4- / kr4-legs.jsonl | **kr5- / kr5-legs.jsonl** |

Code: a copy of v4's `*.py`, `driver.sbatch`, `submit.sh`, `prices.json` and `prompt/`. v4's data, probes, results,
logs, babysit state, secrets, legs / budget / progress files and analysis were not copied. Changes:
- `driver.py`: taskset `official2` only (`official2-tasks.json`, `tasks[g::4]`); task digest check; `--dry-run`; hengyuf
  sandbox only; kr5 names; results add `task_list` (file sha256, why, task dirs).
- `probe_worker.py`: v4's wrapper without the Baihe sandbox override (`KR3_SANDBOX` must be `hengyuf`); `kr5-legs.jsonl`.
- `submit.sh`: `official2 GROUP...` only, prefix kr5-, and the after-v4 check. 2026-10-29: `pilot` added; the after-v4
  check (and KR5_SKIP_V4_CHECK) removed after the user's early-start approval, so manual and watch.py auto legs work.
- `watch.py`: kr5- prefix, no single campaigns.
- `progress.py`: total = 33 x 5 = 165; priors from v4's PROGRESS-rows.json.
- `verify_pilot.py`: no Baihe checks; works on an attempt of a running campaign. 2026-10-29: knows the `pilot` campaign.
- `driver.py` 2026-10-29: taskset `pilot` = 1 rollout of PILOT_TASK `syn-markdown-cand-c04-r1b0-3b72285b` (campaign
  `pilot`, outside the 165; build-time Kimi 5/5 pass, median 27.6 min = the median of the 33 tasks' medians).
- `progress.py` 2026-10-29: spend = the LiteLLM-reported cost (usage.cost) of v5's own replies (every v5 attempt's
  http.jsonl, read incrementally, by the hub key of the attempt's leg), not the key's spend delta, because NEW_2 is
  shared with v4 (during the pilot the key's spend rose $115.08 while the pilot's own replies cost $2.36); the pilot is a
  header line. Backups of the 10-28 versions: `backup-pre-pilot-20261029/`.
- `sources.py`: kimi5 sources; v4's are read from v4's dir.
- `costs.py`: kr5-legs.jsonl.
- Headers only: `brokenreply.py`, `key_budget.py`.
- Unchanged: `degenerate.py` (byte-identical: brokenreply.py uses its `classify()`), `features.py`, `prices.json`,
  `prompt/`.
- Kept from v3 / v4 but still v3-shaped (adapt before use): `compare.py`, `phase_report.py`, `stops.py`,
  `ratelimits.py`, `window.py` (default source kimi5).
- Dropped (Baihe only): `make_baihe_secrets.py`, `baihe_snapshot_check.py`.
- New: `official2_tasks.py`, `snapshot_check.py`, `validate_offline.py`.

The broken-reply detector, the re-run plan / merge, the budget guard (NEW_2 >= $100 to start, stop below $40, check
every 300 s) and the budget-refusal recovery are v4's code, unchanged (see v4's README "Broken-reply re-runs").

## Campaigns

| campaign | tasks | rollouts | sandbox credential | max in flight |
|---|---|---|---|---|
| official2-g0 | 9 | 45 | CELL3 | 8 |
| official2-g1 | 8 | 40 | CELL3 | 8 |
| official2-g2 | 8 | 40 | CELL3 | 8 |
| official2-g3 | 8 | 40 | CELL3 | 8 |

Concurrency: 32 in flight in total.
- v4 ran ~110 in flight on NEW_2 with no HTTP 429 (watch-log.jsonl: 0 of ~11k replies 10-28 15:00-16:33).
- v3 phase 1 measured p90 4.1M tokens/min at ~60 in flight, against hub limits of 50M tokens/min and 10k requests/min.
- 32 in flight after v4 has finished is well inside both limits. Since the 2026-10-29 early start, v4 (~60 in flight)
  and v5 (32) run together, ~92 in flight on NEW_2, still below v4's own ~110 at its start.
- Expected time: ~45 slots per campaign at 8 in flight, ~1-1.5 h per rollout, so about 6-8 h in one 23 h leg,
  re-runs included.
- Expected cost: ~165 x $3.6-4.1 = $0.6-0.7k, plus ~5-7% broken-ended re-runs, on NEW_2. NEW_2 had $8,736 left at 16:33
  10-28, and v4 projects ~$3.2k more.

## Validation (offline, 2026-10-28; nothing launched)

- `validate_offline.py` -> `analysis/validate-offline.json`: all checks OK.
  - `task_list_ok`: the frozen list equals a fresh derivation, with 33 tasks and no problems.
  - `groups_ok`: `driver.py --dry-run` resolves groups 0-3 (9/8/8/8 tasks, disjoint, covering all 33) with digests,
    dataset rows, stock prompt sha, CELL3 and this dir's probe_worker.py.
  - `settings_ok`: each would-be campaign settings.json equals v4's sel50-g0 except expected_tasks.
  - `atp_source_ok`: all 24 ATP files in v4's frozen campaign source have the same sha256 now.
  - `shared_code_ok`: synth probe_worker / rollouts / common / settings.json and Gym.tar are older than v4's first leg,
    and Gym.tar matches its pin.
  - `gym_config_ok`: ATP's own build_config on group 0's settings gives v4's sel50-g0 gym.yaml once hosts, ports and
    paths are dropped. That is kimi-k3, max_tokens 131072, reasoning_effort max, opencode_max_context_window 262144,
    OpenCode 1.17.11, and no opencode_config. So OpenCode's config is v4's (v4 pilot: limit 262144 / 262144 / 131072,
    stock prompt; verify_pilot all OK).
  - `prompt_ok`: prompt/ is identical to v4's.
- `snapshot_check.py` -> `analysis/snapshot-check.json`: snapshot_info, then one create / exec / delete per distinct
  image (33). It uses S/atp/.env's sandbox settings, i.e. CELL3_SANDBOX_API_KEY, which it verified by name.
- `progress.py` ran on the empty campaign (PROGRESS.md "not started"; the pre-launch row was then removed from
  PROGRESS-rows.json, so the table starts at launch).

## Files

| file | role |
|---|---|
| `official2_tasks.py` | derives / checks the task list; `--write` froze `official2-tasks.json` |
| `driver.py` | one campaign leg: task check, key choice + guard, main campaign, rec / brk recovery campaigns, results |
| `probe_worker.py` | ATP worker wrapper: hub key by name, stock-prompt check, hengyuf sandbox, `kr5-legs.jsonl` |
| `driver.sbatch` | the CPU job (babysit_enter; drops credential env vars; runs driver.py with the ATP controller python) |
| `submit.sh` | babysit pair (nemotron_agents_dev + nemotron_rl_algo) per campaign leg; prefix kr5-; after-v4 check |
| `brokenreply.py` | broken-ending detector (degenerate.py's definition) |
| `progress.py` | rewrites `PROGRESS.md` (header, one row per run kept in `PROGRESS-rows.json`, campaign table) |
| `watch.py` | watcher (`--once` for one check): settings, endings, auto legs (`--auto-leg-cap`), 4xx, stalls, budget, quota |
| `verify_pilot.py` | checks one attempt: limits, no prompt override, stock system prompt, key, sandbox, grading |
| `key_budget.py` | a hub key's budget by NAME (NEW_2 / NEW_3 only; appends `budget-log.jsonl`) |
| `snapshot_check.py` | sandbox check of every task image under hengyuf's credential (no hub call) |
| `validate_offline.py` | the offline validation above |
| `sources.py`, `features.py`, `window.py`, `costs.py`, `stops.py`, `degenerate.py`, `compare.py`, `phase_report.py`, `ratelimits.py` | analysis tools (see Code above) |

Run-time state (not code):
- `probes/kimi/official2-g<K>[-rec<n>|-brk<n>]/` (ATP campaigns; `kr5-legs.jsonl` per campaign)
- `results/kimi-official2-g<K>.json`, `reruns/official2-g<K>.json`
- `legs.jsonl`, `budget-log.jsonl`
- `babysit/official2-g<K>/` (helper state; a `STOP` file there refuses new legs)
- `logs/kr5-*.out`
- `watch-state.json`, `watch-log.jsonl`, `progress-cache.json`, `PROGRESS-rows.json`, `analysis/`

## Operating

Run Slurm commands as standalone Bash calls. Touch only jobs whose names start with `kr5-`. `bash submit.sh` runs sbatch
inside a script, so it only works while the Claude Code Bash sandbox is off, as in the 2026-10-28 session. With the
sandbox on, do the helper's babysit steps by hand (memory: feedback_slurm_in_sandbox).

```bash
KR=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v5-official2-20261028
# 0. (2026-10-29: the user approved starting while v4 runs; submit.sh no longer checks v4.) Re-check the task list
#    and the setup:
/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp/runtime/controller/bin/python3 -I -B $KR/official2_tasks.py
/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp/runtime/controller/bin/python3 -I -B $KR/validate_offline.py
# 1. probe partitions (one standalone call per partition and account), then launch the 4 campaigns:
#   sbatch --test-only --partition=<p> --account=<a> --cpus-per-task=2 --mem=16G --time=23:00:00 $KR/driver.sbatch
#    (2026-10-29: pilot first, `bash $KR/submit.sh pilot`; then the launch below, with cpu_datamover added because
#    cpu_dataprocessing would only have started at ~04:19)
KR_PARTITION=cpu_datamover,cpu_interactive,cpu_dataprocessing bash $KR/submit.sh official2 0 1 2 3
# 2. every 30 min: one watcher check (auto-submits up to 3 further legs per campaign) and the progress report
/usr/bin/python3 -I -B $KR/watch.py --once --auto-leg-cap 3 --partition cpu_datamover,cpu_interactive,cpu_dataprocessing \
  --campaigns official2-g0 official2-g1 official2-g2 official2-g3
/usr/bin/python3 -I -B $KR/progress.py
# 3. verify the first finished attempt of a campaign (stock prompt, 262144 limits, key, sandbox, grading)
/usr/bin/python3 -I -B $KR/verify_pilot.py $KR/probes/kimi/official2-g0/models/kimi/attempts/<task>-<repeat>/attempt-0
```

Job lengths and resubmission:
- Legs are 23 h jobs. ATP admits a new rollout only while the agent timeout (3 h) + the verifier timeout + 10 min
  still fit in the job.
- A campaign (40-45 slots at 8 in flight) should finish, re-runs included, in one leg.
- To resubmit a campaign by hand, run `bash $KR/submit.sh official2 <group>`; the helper starts its next leg and the
  driver resumes where the last leg stopped. Do this only when no `kr5-official2-g<group>` job is queued or running
  and `results/kimi-official2-g<group>.json` does not exist yet. A `STOP` file in `babysit/official2-g<group>/` blocks
  this.

Never edit in place a script that a queued or running job executes (`driver.sbatch`, `driver.py`, `probe_worker.py`,
`brokenreply.py`, `degenerate.py`, `key_budget.py`, `official2_tasks.py`, `official2-tasks.json`, the helper). Write a
temporary file in this directory and `mv -f` it over the path.
