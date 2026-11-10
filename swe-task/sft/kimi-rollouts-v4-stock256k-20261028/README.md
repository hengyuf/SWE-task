# kimi-rollouts-v4-stock256k-20261028

Kimi-K3 rollouts on every v2 SFT task (194 = 97 hengyuf + 97 Baihe), 5 per task (970), with OpenCode's
**stock** system prompt and a **262,144-token** context window.

User request (hengyuf, 2026-10-28, verbatim): "在之前v2 上的所有task 上,用kimi 在每个生成的task 上分别rollout 5遍,
context window 设置为256k, prompt 保持不变(不要加reason before action 之类的prompt)" and "每30min 汇报进度在一个markdown里".
Broken-reply rule (hengyuf, 2026-10-28, verbatim): "这次若遇到被坏回复终止的 slot, 可以重跑1遍".

This directory is a self-contained copy of the code of `../kimi-rollouts-v3-20261026` (v3, read-only; only its code was
copied: `*.py`, `driver.sbatch`, `submit.sh`, `prices.json`, `prompt/`). Harness, graders, model spec and sandboxes are
v3's. The setting differences are:

| | v3 (sel50 / rest47 / baihe arms) | v3 ctrl-old | **v4 (this dir)** |
|---|---|---|---|
| system prompt | stock + one reasoning sentence | stock | **stock** (`KR3_PROMPT=stock` everywhere) |
| OpenCode window | 262,144 (`limit.context = limit.input`) | 1,048,576 | **262,144** |
| output cap | 131,072 | 131,072 | 131,072 |
| tasksets | sel50 (+ planned rest47, baihe) | 4 tasks | pilot, baihe-pilot, sel50 g0-4, rest47 g0-4, baihe g0-7 |
| broken-ended slots | kept | kept | **re-run once** (`<campaign>-brk<n>`) |
| job / exp prefix | kr3- | kr3- | **kr4-** |

## Configuration (driver.py)

- Model alias `kimi` = synth/settings.json's spec (`nvidia/moonshotai/kimi-k3`, `reasoning_effort: max`) +
  `context_window: 262144`, set in memory only (`SPEC`); ATP's `build_config` turns it into OpenCode
  `limit.context = limit.input = 262144`, `limit.output = 131072`. OpenCode compacts at 262,144 - 20,000 = 242,144.
- Prompt: `probe_worker.py` refuses anything but `KR3_PROMPT=stock` and only *checks* that ATP's gym.yaml sets no
  `opencode_config` for the agent, so OpenCode sends `prompt/opencode-1.17.11-default.txt` (sha256 `962fbf3c...`)
  followed by its usual model line / `<env>` block / AGENTS.md / skills. `prompt/build-prompt.txt` and
  `prompt/sentence.txt` are v3 files kept only so that `verify_pilot.py` can assert the sentence is absent.
- Hub key by NAME (never a value; read from `/lustre/fsw/portfolios/nemotron/users/hengyuf/.env` inside the code):
  `NVAPI_KEY_NEW_2` first, `NVAPI_KEY_NEW_3` once NEW_2 is used up (user, 2026-10-26). A leg starts on NEW_2 while it
  has >= $100 left; a guard re-checks the leg's key every 300 s and stops the leg below $40 so that the next leg moves
  to NEW_3. Budgets via `key_budget.py` (names and numbers only; appends to `budget-log.jsonl`).
- Sandbox credential by NAME: hengyuf's tasks use S/atp/.env's `CELL3_SANDBOX_API_KEY`; Baihe's tasks (`baihe*`) use
  `BAIHE_SANDBOX_API_KEY` (user-authorised for Baihe's sandbox tasks only) through this directory's own mode-600
  `secrets/baihe-opensandbox.env`, written by `make_baihe_secrets.py` (never a copy of v3's file). A campaign never mixes
  credentials (checked on resume).
- Infra recovery (unchanged from v3): a slot whose final attempt got a hub budget refusal (ATP makes a 4xx fatal), or
  that is listed in a `quota-incident-slots.json` (none in v4; v3's file lists v3 slots and was not copied), is re-run
  once in `probes/kimi/<campaign>-rec<n>` and replaces the refused slot.

## Broken-reply re-runs (new in v4)

- Detector: `brokenreply.py`. It applies degenerate.py's own `classify()` (during the campaigns degenerate.py was v3's file, byte-identical; the final analysis extended it, keeping `classify()` / `scan()` byte-identical, see below)
  to the LAST status-200 reply of an attempt's `http.jsonl`. A slot is *ended by a broken reply* when that reply is
  degenerate (`[PAD]`, a `<|...|>` special token, or empty) **and** has no tool call, i.e. the session stopped there.
  A slot that merely contains a broken reply but went on is not re-run. The journal is read backwards (a few hundred
  KB per attempt, instead of ~100 MB).
- Offline validation on v3's recorded slots (read-only), `analysis/brokenreply-validate-v3.json`:
  - labels: v3's per-slot `last_reply_degenerate` (window-kimi3.jsonl and window-ctrl-old.jsonl, 270 slots; 70/250 + 4/20,
    as in degenerate-compare.json);
  - result: 74 true positives (68 `[PAD]`, 6 empty), 196 true negatives, 0 false positives, 0 false negatives, so
    precision 1.0 and recall 1.0;
  - a forward scan of every journal with `degenerate.scan()` found the same last reply and the same verdict on all 268
    journals.
  - Re-run the check with `/usr/bin/python3 -I -B brokenreply.py --validate-v3 --forward --out analysis/brokenreply-validate-v3.json`.
- Driver flow per campaign (one leg can stop anywhere; the next leg resumes):
  1. The main campaign `probes/kimi/<name>` runs until every slot is terminal.
  2. Infra recovery runs `<name>-rec<n>` (as v3).
  3. Every slot whose current record (original, or its rec recovery) was ended by a broken reply is planned in
     `reruns/<name>.json`. The plan is written once, before the first re-run starts, and later legs must find the same
     set (else the driver stops, without re-planning). Slots are grouped by the task's number of such slots `n`, which
     gives campaign `probes/kimi/<name>-brk<n>` with `n` repeats per task. The campaigns run one after another, largest
     first, with the same spec, prompt, concurrency cap, leg key and sandbox credential.
  4. Merge into `results/kimi-<name>.json`:
     - A graded re-run (pass or fail) replaces the slot. The record keeps `original_attempt` (the original compact
       record) and `rerun_reason: "ended_by_broken_reply"`, plus `rerun_campaign` and `rerun_slot`.
     - A re-run that ends on a broken reply again is kept as it is (outcome `broken_again`). There is never a third
       attempt.
     - A re-run that ends infra (no grade) leaves the original in place, with the re-run kept as `rerun_attempt` and
       `rerun_replaced: false`.
     - `<name>-brk<n>` slots are never re-run (no broken-reply re-run and no budget-refusal recovery).
- Results file fields added in v4:
  - per record: `ended_by_broken_reply`, `last_reply_kind`;
  - per file: `first_attempt_records` (before the re-runs), `broken_reply_reruns` (one entry per re-run: original /
    re-run attempt, statuses, `outcome` pass | fail | broken_again | infra, `replaced`), `broken_reply_rerun_outcomes`,
    `stats_first_attempts`, `stats_final`, `prompt: "stock"`, `context_window: 262144`, `system_prompt` (file and
    sha256), `hub_keys_by_campaign` (main, rec and brk legs).

## Campaigns

| campaign | tasks | rollouts | sandbox credential | max in flight (default) |
|---|---|---|---|---|
| pilot | `syn-aiohttp-cand-c05-r1b0-7d9ec1f5` | 1 | CELL3 | 1 |
| baihe-pilot | `syn-citty-cand-c01-r1b0-6fa08bef` | 1 | BAIHE | 1 |
| sel50-g0..g4 | the 50 train-acc-v2 tasks, `tasks[g::5]` (10 each) | 250 | CELL3 | 8 each |
| rest47-g0..g4 | the other 47 hengyuf v2 tasks, `tasks[g::5]` (10,10,9,9,9) | 235 | CELL3 | 8 each |
| baihe-g0..g7 | Baihe's 97 v2 tasks (`baihe_v2_tasks.tsv`), `tasks[g::8]` (13,12,...) | 485 | BAIHE | 4 each |

Concurrency: phase 1 of v3 (~60 rollouts in flight, `../kimi-rollouts-v3-20261026/analysis/ratelimits-phase1.json`)
used at most 9.1M tokens/min (p90 4.1M) and 218 calls/min, against hub limits of 50M tokens/min and 10k requests/min
(lower tier). The hub-reported remaining-token headroom never fell below 63%, and there were 7 replies with HTTP 429
out of 23k.

With the defaults, all 18 campaigns at once run 80 rollouts on hengyuf's sandbox account and 32 on Baihe's (~112 in
flight, 1.9x phase 1). Scaling phase 1 by 1.9 gives:
- tokens: p90 ~7.7M/min (~15% of 50M), worst minute ~17M (~35%);
- calls: ~410/min (~4% of 10k).

Baihe's account stays at v3's 4 per campaign.

## Files

| file | role |
|---|---|
| `driver.py` | one campaign leg: key choice + guard, main campaign, rec / brk recovery campaigns, results |
| `probe_worker.py` | ATP worker wrapper: hub key by name, stock-prompt check, Baihe sandbox credential, `kr4-legs.jsonl` |
| `driver.sbatch` | the CPU job (babysit_enter; drops credential env vars; runs driver.py with the ATP controller python) |
| `submit.sh` | submits a babysit pair (accounts nemotron_agents_dev + nemotron_rl_algo) per campaign leg; prefix kr4- |
| `brokenreply.py` | broken-ending detector (degenerate.py's definition) and the v3 validation |
| `progress.py` | rewrites `PROGRESS.md` (header, one appended row per run kept in `PROGRESS-rows.json`, campaign table) |
| `watch.py` | watcher (`--once` for one check): settings, endings, auto legs (`--auto-leg-cap`), 4xx, stalls, budget, quota |
| `verify_pilot.py` | checks one attempt: limits, no prompt override, stock system prompt byte-identical, key, sandbox, grading |
| `key_budget.py` | a hub key's budget by NAME (NEW_2 / NEW_3 only) |
| `make_baihe_secrets.py` | writes `secrets/baihe-opensandbox.env` (mode 600) from `BAIHE_SANDBOX_API_KEY` |
| `sources.py`, `features.py`, `window.py`, `costs.py`, `stops.py`, `degenerate.py`, `compare.py`, `phase_report.py`, `ratelimits.py`, `baihe_snapshot_check.py` | v3 analysis tools. `sources.py` adds the v4 sources `kimi4` (final), `kimi4-first`, `kimi4-pilot` and reads v3's `kimi3` and `ctrl-old` from the v3 directory. `window.py` defaults to `kimi4`. `costs.py` reads `kr4-legs.jsonl`. `compare.py` is still v3-shaped; `phase_report.py` was rewritten for the final report (see below). |

Run-time state (not code): `probes/kimi/<campaign>[-rec<n>|-brk<n>]/` (ATP campaigns; `kr4-legs.jsonl` per campaign),
`results/kimi-<campaign>.json`, `reruns/<campaign>.json`, `legs.jsonl` (driver leg starts / ends), `budget-log.jsonl`,
`babysit/<campaign>/` (helper state; a `STOP` file there refuses new legs), `logs/kr4-*.out`, `watch-state.json`,
`watch-log.jsonl`, `progress-cache.json`, `analysis/`.

## Operating

All Slurm commands run as standalone Bash calls. Touch only jobs named `kr4-*`.

```bash
KR=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v4-stock256k-20261028
# a campaign's first or next leg (babysit pair; resumes where the last leg stopped). Probe partitions first:
#   sbatch --test-only --partition=<p> --account=<a> --cpus-per-task=2 --mem=16G --time=23:00:00 $KR/driver.sbatch
KR_PARTITION=cpu_dataprocessing,cpu_interactive bash $KR/submit.sh sel50 0 1 2 3 4
# 30-minute check (auto-submits up to N further legs per campaign, counted across runs in watch-state.json)
/usr/bin/python3 -I -B $KR/watch.py --once --auto-leg-cap 3 --partition cpu_dataprocessing,cpu_interactive \
  --campaigns sel50-g0 sel50-g1 sel50-g2 sel50-g3 sel50-g4 rest47-g0 rest47-g1 rest47-g2 rest47-g3 rest47-g4 \
              baihe-g0 baihe-g1 baihe-g2 baihe-g3 baihe-g4 baihe-g5 baihe-g6 baihe-g7
/usr/bin/python3 -I -B $KR/progress.py
# verify an attempt (pilots: --previous = an earlier 1M stock Kimi attempt of the same task)
/usr/bin/python3 -I -B $KR/verify_pilot.py $KR/probes/kimi/pilot/models/kimi/attempts/0-0/attempt-0
```

Job lengths and resubmission:
- Main legs are 23 h jobs (cpu_interactive allows up to 1 day; the other CPU partitions have no limit). ATP admits a
  new rollout only while the agent timeout (3 h) + the verifier timeout + 10 min still fit in the job.
- A hengyuf campaign (45-50 slots at 8 in flight) usually finishes, re-runs included, in one leg. A Baihe campaign
  (60-65 slots at 4 in flight) needs about two legs.
- To resubmit a campaign by hand (no `kr4-<name>` job queued or running, and no `results/kimi-<name>.json`), run
  `bash $KR/submit.sh <taskset> <group>`; the helper starts its next leg. A `STOP` file in `babysit/<name>/` blocks
  this.

Never edit in place a script that a queued or running job executes (`driver.sbatch`, `driver.py`, `probe_worker.py`,
`brokenreply.py`, `degenerate.py`, `key_budget.py`, the helper). Write a temporary file in this directory and `mv -f` it
over the path.

## Final state (2026-10-29)
All 970 slots graded (campaigns 2026-10-28 14:57 to 2026-10-29 21:32); 66 broken-reply re-runs, no third attempt, no kr4- job left.
pass@1 first attempts 74.2%, after the one re-run 78.8%; 745 SFT-usable passes (701 without compaction); $3,959.12 on
NVAPI_KEY_NEW_2 (PROGRESS.md's cost columns also include v5's $390 on the same key). Report: `analysis/FINAL-REPORT.md`
(written by the main agent from the workflow's final analysis) and the script-generated `analysis/final-report.{md,json}`,
`analysis/sft-usable-passes.jsonl`. The final analysis extended `sources.py` (`kimi4-all`, `kimi-1m`), `features.py`
(output in `analysis/features/`), `degenerate.py` (new arms and per-attempt rows; `classify()` / `scan()` and the statistics
helpers byte-identical, checked with ast) and rewrote `phase_report.py`; the previous files are in `backup-pre-final-20261029/`.
