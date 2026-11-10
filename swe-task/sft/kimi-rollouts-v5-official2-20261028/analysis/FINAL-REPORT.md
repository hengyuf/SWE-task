# Kimi-K3 v5 final summary: official2 accepted + strong tasks, 5 rollouts each, 256k window, stock prompt

**Status: finished.** The four campaigns (official2-g0..g3, jobs 20990515 / 20990517 / 20990519 / 20990521) ran on 2026-10-29 from 02:51 to 10:32 PDT. They used NVAPI_KEY_NEW_2 and the CELL3 sandbox credential.
- All 165 slots were graded, with 0 infra slots.
- 8 broken-reply re-runs were done. No slot had a third attempt (only `-brk1` campaigns exist, one attempt per slot).
- No kr5- job is queued or running.
- `progress.py` ran one last time; the last PROGRESS.md row is 10-29 10:54.

## Headline
| | v5 (256k, 2026-10-29) | build-time probes (1M, 10-23/24) |
|---|---|---|
| pass@1, first attempts | **132/165 = 80.0%** (95% CI 72.7–86.7) | 139/165 = 84.2% (CI 79.4–89.7) |
| pass@1, after the one re-run | **139/165 = 84.2%** (CI 76.4–91.5) | no re-runs |
| pass@1, excluding broken-ended rollouts | 132/157 = 84.1% | 139/156 = 89.1% |
| rollouts ended by a broken reply | **8/165 (4.8%)**, all `[PAD]`, all failed | 9/165 (5.5%), all failed |
| broken replies per 1000 replies | 9/12,935 = 0.70 | 16/11,981 = 1.34 (Fisher p 0.16) |
| compacted rollouts | **0/173** | 0/165 |
| largest prompt in the main session | 235,336 (6 rollouts over 200k, none over 242,144) | 214,613 |
| % of action turns with reasoning | **54.0%** (CI 52.2–56.1) | 54.1% (CI 52.2–55.9) |
| tasks with at least one pass / 5 of 5 / none | 33 / 19 / 0 | 33 / 14 / 0 |
| cost per rollout (LiteLLM usage.cost) | mean $2.24, median $1.94 | $1.69 |

- **pass@1 vs build, paired by task:** v5 after re-run minus build is 0.0 points (CI −7.9 to +7.3). v5 first attempts minus build is −4.2 (CI −12.1 to +3.0). With broken-ended rollouts removed from both sides it is −5.0 (CI −11.6 to +1.3). None of these is significant.
- **The build numbers are biased upward:** a task was accepted only if Kimi's build-time clean pass rate was at least 0.6, so some drop on a re-run is expected (regression to the mean).
- **The 256k window never mattered:** no v5 or build rollout came near OpenCode's compaction point (242,144). There were 0 replies cut at the output length limit and 0 context-overflow errors; the only non-200 replies were four 408s, which were retried.
- **Reasoning is unchanged:** v5 minus build is +0.1 points (CI −1.1 to +1.4), and still +0.0 when rollouts with broken replies are dropped.
- **Wall time:** the v5 median agent wall time was 44 min vs 24 min at build time, with similar turn counts (69 vs 68). On a 20-rollout sample per arm, ATP's request-to-response time per call had a median of 31 s vs 6 s. I did not look into the cause (v4 running on NEW_2 at the same time, or ATP's admission gate).
- **Why v5 cost more per rollout:** the cache hit rate was 89.8% vs 95.0%, and prompt tokens per rollout were about 9% higher.

## Broken-reply re-runs (one per slot; both records kept in the results files)
The detector's verdict matches driver.py's recorded verdict on all 173 attempts. Every original below committed no patch (patch 0 chars).

| campaign | task | slot | original | re-run (`<campaign>-brk1` kimi-r0) |
|---|---|---|---|---|
| g0 | syn-otto-cand-c02-r1b1-a4d613c7 | kimi-r2 | fail, F2P 0/72 | pass (72/72) |
| g0 | syn-otto-cand-c05-r1b4-da7cd37c | kimi-r2 | fail, F2P 0/26 (broken at reply 2) | pass (26/26) |
| g1 | syn-markdown-cand-c04-r1b0-3b72285b | kimi-r0 | fail, F2P 0/40 | pass (40/40) |
| g1 | syn-pyserde-cand-c04-r1b4-443ff5f7 | kimi-r0 | fail, F2P 0/45 | pass (45/45) |
| g2 | syn-markdown-cand-c01-r1b0-ecd019d8 | kimi-r0 | fail, F2P 0/40 | pass (40/40) |
| g2 | syn-pyserde-cand-c05-r1b2-170c68fa | kimi-r4 | fail, F2P 0/53 | pass (53/53) |
| g3 | syn-django-silk-cand-c01-r1b1-ffe67368 | kimi-r0 | fail, F2P 0/144 | fail (141/144) |
| g3 | syn-otto-cand-c05-r1b1-661a75c4 | kimi-r3 | fail, F2P 0/88 | pass (88/88) |

- **Re-run outcomes:** 7 pass, 1 fail, 0 broken again, 0 infra.
- **The broken replies were long:** 7 of the 8 were `[PAD]` runs of 14.8k–29.5k completion tokens; the eighth was 79 tokens.
- **One more broken reply did not end its rollout:** in syn-otto-cand-c04-r1b1 kimi-r2, a task subagent's final reply was `[PAD]` (reply 37 of 153). The main session went on and passed.
- **The build had broken replies too:** 9 build rollouts were ended by one and counted as fails, never re-run. 7 of the 139 build passes contain broken text: 5 had `[PAD]` in a reply that still made a tool call, and 2 had a broken subagent. This matters if anyone reuses build-time Kimi passes for SFT.

## Cost by key (v5's own replies, LiteLLM usage.cost; each attempt mapped to its leg's key)
| key | batch | first attempts | re-runs | pilot | key remaining at last check |
|---|---|---|---|---|---|
| NVAPI_KEY_NEW_2 | **$387.68** | $371.25 | $16.43 | $2.36 | $5,103.16 (10-29 10:54; this key is shared with v4, which is still running) |
| NVAPI_KEY_NEW_3 | **$0** (never used) | – | – | – | – |

Total on NEW_2 is $390.04, which equals progress.py's figure. Keys were read by name only (through progress.py → key_budget.py); no key was printed.

## Per task vs the build-time 1M probes
Every task had 0 compactions and every rollout's cost went to NEW_2. "Build" = the synth pipeline's 5 Kimi slots for the task, as score() graded them: the build-time clean rate equals its raw pass rate here, since 0 passes were flagged as cheating. The number in brackets after the build pass count is how many build rollouts a broken reply ended.

| task | run | why | build pass (broken-ended) | v5 first | broken-ended → re-run | v5 after re-run | after − build | % reasoned v5 / build | max prompt v5 (k) | $/rollout v5 / build | SFT-usable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-pyserde-cand-c04-r1b1-2b74e357 | adaptix | accepted | 5/5 (0) | 4/5 | - | 4/5 | -1 | 49.7 / 53.9 | 132 | 2.61 / 1.88 | 4 |
| syn-pyserde-cand-c04-r1b4-443ff5f7 | adaptix | accepted | 5/5 (0) | 4/5 | r0 → pass | 5/5 | +0 | 47.1 / 54.2 | 114 | 2.05 / 2.39 | 5 |
| syn-pyserde-cand-c05-r1b2-170c68fa | adaptix | accepted | 5/5 (0) | 4/5 | r4 → pass | 5/5 | +0 | 48.3 / 43.5 | 131 | 2.29 / 2.66 | 5 |
| syn-django-silk-cand-c01-r1b1-ffe67368 | aiomonitor | accepted | 4/5 (0) | 2/5 | r0 → fail | 2/5 | -2 | 50.3 / 55.4 | 87 | 1.15 / 0.87 | 2 |
| syn-django-silk-cand-c02-r1b4-3a06fd8c | aiomonitor | accepted | 5/5 (0) | 3/5 | - | 3/5 | -2 | 50.4 / 43.9 | 104 | 1.13 / 0.72 | 3 |
| syn-django-silk-cand-c03-r1b0-9bf52923 | aiomonitor | accepted | 5/5 (0) | 5/5 | - | 5/5 | +0 | 44.4 / 48.6 | 70 | 0.80 / 0.58 | 5 |
| syn-django-silk-cand-c04-r1b2-2a7a6a57 | aiomonitor | accepted | 4/5 (0) | 4/5 | - | 4/5 | +0 | 54.3 / 49.5 | 82 | 0.82 / 0.79 | 4 |
| syn-pyserde-cand-c04-r1b2-14d40723 | cattrs | accepted | 5/5 (0) | 5/5 | - | 5/5 | +0 | 56.9 / 58.5 | 149 | 3.22 / 2.11 | 5 |
| syn-jd-cand-c01-r1b1-ff9da598 | etree | accepted | 5/5 (0) | 5/5 | - | 5/5 | +0 | 62.1 / 61.9 | 139 | 1.83 / 1.06 | 5 |
| syn-jd-cand-c02-r1b4-3aa68c3b | etree | accepted | 3/5 (0) | 3/5 | - | 3/5 | +0 | 68.3 / 66.5 | 147 | 1.74 / 0.84 | 3 |
| syn-jd-cand-c05-r1b0-7994aee3 | etree | accepted | 5/5 (0) | 3/5 | - | 3/5 | -2 | 67.2 / 61.7 | 132 | 1.62 / 1.34 | 3 |
| syn-knex-cand-c01-r1b3-a888f2f2 | kysely | accepted + strong | 3/5 (0) | 2/5 | - | 2/5 | -1 | 53.3 / 53.0 | 106 | 1.96 / 1.43 | 2 |
| syn-knex-cand-c04-r1b0-b21e1e98 | kysely | accepted | 5/5 (0) | 4/5 | - | 4/5 | -1 | 53.4 / 56.4 | 85 | 1.39 / 1.54 | 4 |
| syn-pabot-cand-c03-r1b0-5f252c24 | mobly | accepted | 4/5 (0) | 4/5 | - | 4/5 | +0 | 58.2 / 55.2 | 150 | 3.59 / 2.76 | 4 |
| syn-markdown-cand-c01-r1b0-ecd019d8 | obsidian | accepted | 4/5 (1) | 3/5 | r0 → pass | 4/5 | +0 | 60.4 / 54.5 | 117 | 1.25 / 0.98 | 4 |
| syn-markdown-cand-c01-r1b3-986c83f4 | obsidian | accepted | 4/5 (1) | 5/5 | - | 5/5 | +1 | 64.7 / 62.6 | 147 | 2.55 / 1.86 | 5 |
| syn-markdown-cand-c01-r1b4-7673d9a2 | obsidian | accepted | 4/5 (0) | 5/5 | - | 5/5 | +1 | 61.4 / 63.6 | 142 | 1.82 / 1.29 | 5 |
| syn-markdown-cand-c04-r1b0-3b72285b | obsidian | accepted | 5/5 (0) | 4/5 | r0 → pass | 5/5 | +0 | 58.6 / 60.1 | 165 | 2.31 / 2.04 | 5 |
| syn-goleveldb-cand-c01-r1b1-7523606b | pebble | accepted | 4/5 (1) | 5/5 | - | 5/5 | +1 | 53.8 / 54.0 | 107 | 1.35 / 0.83 | 5 |
| syn-goleveldb-cand-c01-r1b2-b09a37b8 | pebble | accepted | 5/5 (0) | 5/5 | - | 5/5 | +0 | 60.5 / 57.1 | 92 | 0.93 / 0.69 | 5 |
| syn-goleveldb-cand-c01-r1b4-958e653b | pebble | accepted | 3/5 (0) | 3/5 | - | 3/5 | +0 | 52.9 / 52.6 | 118 | 2.06 / 0.86 | 3 |
| syn-goleveldb-cand-c02-r1b0-e5f8b461 | pebble | accepted | 5/5 (0) | 5/5 | - | 5/5 | +0 | 56.2 / 60.4 | 113 | 1.20 / 0.87 | 5 |
| syn-goleveldb-cand-c03-r1b4-9045ebe3 | pebble | accepted | 4/5 (0) | 1/5 | - | 1/5 | -3 | 56.2 / 52.3 | 138 | 2.15 / 1.42 | 1 |
| syn-kapacitor-cand-c02-r1b1-27100f50 | promreload | accepted | 3/5 (1) | 3/5 | - | 3/5 | +0 | 50.1 / 49.2 | 230 | 4.42 / 2.97 | 3 |
| syn-kapacitor-cand-c05-r1b1-55fcf64d | promreload | accepted | 4/5 (0) | 4/5 | - | 4/5 | +0 | 55.6 / 55.1 | 108 | 1.59 / 2.10 | 4 |
| syn-risor-cand-c01-r1b4-74844668 | tengo-callable | strong | 3/5 (1) | 5/5 | - | 5/5 | +2 | 52.4 / 44.7 | 134 | 2.40 / 1.59 | 5 |
| syn-otto-cand-c01-r1b1-0f53e898 | tengo-destructuring | accepted | 4/5 (1) | 5/5 | - | 5/5 | +1 | 49.9 / 50.4 | 235 | 6.11 / 3.63 | 5 |
| syn-otto-cand-c02-c19003e2 | tengo-destructuring | accepted | 5/5 (0) | 5/5 | - | 5/5 | +0 | 63.6 / 60.1 | 166 | 2.64 / 1.89 | 5 |
| syn-otto-cand-c02-r1b1-a4d613c7 | tengo-destructuring | accepted | 5/5 (0) | 4/5 | r2 → pass | 5/5 | +0 | 50.5 / 56.6 | 178 | 2.36 / 2.41 | 5 |
| syn-otto-cand-c03-r1b3-07071613 | tengo-destructuring | accepted | 3/5 (1) | 5/5 | - | 5/5 | +2 | 60.0 / 57.1 | 108 | 1.57 / 1.41 | 5 |
| syn-otto-cand-c04-r1b1-8b8a5c41 | tengo-destructuring | accepted | 4/5 (0) | 5/5 | - | 5/5 | +1 | 51.4 / 53.6 | 186 | 4.16 / 3.26 | 4 |
| syn-otto-cand-c05-r1b1-661a75c4 | tengo-destructuring | accepted | 3/5 (1) | 4/5 | r3 → pass | 5/5 | +2 | 47.5 / 48.0 | 214 | 4.03 / 2.88 | 5 |
| syn-otto-cand-c05-r1b4-da7cd37c | tengo-destructuring | accepted | 4/5 (1) | 4/5 | r2 → pass | 5/5 | +1 | 51.0 / 51.6 | 186 | 2.78 / 2.18 | 5 |

- **Across tasks:** v5 after re-run was higher on 9 tasks, equal on 17 and lower on 7. With 5 rollouts per task each difference is within sampling noise.
- **Largest drops:** goleveldb-c03-r1b4 (4 → 1), django-silk-c01 (4 → 2), django-silk-c02 (5 → 3), jd-c05 (5 → 3).
- **Largest gains:** risor (3 → 5), otto-c03 (3 → 5), otto-c05-r1b1 (3 → 5).
- **By task type:**
  - accepted (31 tasks): 125/155 first → 132/155 after re-run, vs 133/155 at build;
  - kysely, accepted and strong (1 task): 2/5 vs 3/5;
  - risor, strong but not accepted (1 task): 5/5 vs 3/5.

## SFT-usable passes: 138 of 139 (131 first attempts + 7 re-runs, covering all 33 tasks)
**Criteria:**
- a graded pass with reward 1 and every F2P and P2P test passing;
- evaluation completed, no verifier or agent error;
- a non-empty patch and an exported session;
- no broken reply anywhere in the attempt (subagents included) and no `[PAD]` or special-token text in the main session.

No LLM cheating audit was run on v5; the build ran one only on its own passes.

**Excluded (1):** syn-otto-cand-c04-r1b1-8b8a5c41 kimi-r2 (g2). Its task subagent's final reply was `[PAD]`. The main agent then read the saved subagent output, so a `[PAD]` tool output sits in the main session.

Usable slots per task (`*` = the broken-reply re-run, which lives in `<campaign>-brk1`):
- pyserde-c04-r1b1: r0 r1 r2 r3 · pyserde-c04-r1b4: r0* r1 r2 r3 r4 · pyserde-c05-r1b2: r0 r1 r2 r3 r4*
- django-silk-c01-r1b1: r1 r2 · django-silk-c02-r1b4: r0 r1 r3 · django-silk-c03-r1b0: r0–r4 · django-silk-c04-r1b2: r1 r2 r3 r4
- pyserde-c04-r1b2 (cattrs): r0–r4
- jd-c01-r1b1: r0–r4 · jd-c02-r1b4: r0 r2 r3 · jd-c05-r1b0: r0 r1 r4
- knex-c01-r1b3: r2 r4 · knex-c04-r1b0: r0 r1 r3 r4
- pabot-c03-r1b0: r0 r1 r3 r4
- markdown-c01-r1b0: r0* r1 r2 r4 · markdown-c01-r1b3: r0–r4 · markdown-c01-r1b4: r0–r4 · markdown-c04-r1b0: r0* r1 r2 r3 r4
- goleveldb-c01-r1b1: r0–r4 · goleveldb-c01-r1b2: r0–r4 · goleveldb-c01-r1b4: r0 r1 r4 · goleveldb-c02-r1b0: r0–r4 · goleveldb-c03-r1b4: r2
- kapacitor-c02-r1b1: r0 r1 r4 · kapacitor-c05-r1b1: r1 r2 r3 r4
- risor-c01-r1b4: r0–r4
- otto-c01-r1b1: r0–r4 · otto-c02 (round-0): r0–r4 · otto-c02-r1b1: r0 r1 r2* r3 r4 · otto-c03-r1b3: r0–r4 · otto-c04-r1b1: r0 r1 r3 r4 · otto-c05-r1b1: r0 r1 r2 r3* r4 · otto-c05-r1b4: r0 r1 r2* r3 r4

Usable passes per task: 5 passes on 18 tasks, 4 on 7, 3 on 5, 2 on 2, 1 on 1. The full list, with each attempt dir and export.json path, is in `analysis/sft-usable-passes.jsonl` (138 lines).

The pilot rollout (syn-markdown-cand-c04-r1b0-3b72285b) sits outside the 165 and is not in the list. It also passed cleanly (F2P 40/40, 0 broken replies in 82), so it could be used as a 139th pass.

## What I ran and changed (all in /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/kimi-rollouts-v5-official2-20261028)
1. `progress.py`: the final run.
2. v5 versions of the scripts. The previous files are in `backup-pre-final-20261029/`. Each was installed by writing a temp file and running `mv -f` over it, with no kr5- job running.
   - `sources.py` has two new sources:
     - `kimi5-all`: all 173 graded rollouts (165 first attempts + 8 re-runs).
     - `kimi5-build`: the build-time Kimi slots. It reads each task's round probe campaign read-only, picking the first attempt ATP counted as completed. All 165 slots match score()'s `rollouts/index.md` on status, F2P, P2P and patch size. Kapacitor-c05 had two probe campaigns; the one matching index.md is used.
   - `features.py`:
     - It writes to `analysis/features/` and adds fields for broken text, the largest prompt and steps over the compaction point.
     - The turn and reasoning code is unchanged; it reproduces v3's stored rows exactly, token counts included.
     - It needs a python with `tokenizers`; I used `/lustre/fsw/portfolios/nemotron/users/hengyuf/july-ladder-runtime/envs/data-cpu-py311/bin/python`.
   - `degenerate.py`:
     - It now compares v5 with the build probes and writes one row per attempt with its cost.
     - classify(), scan() and the statistics helpers are byte-identical, which brokenreply.py and driver.py rely on; brokenreply.py still runs.
   - `phase_report.py`: rewritten as this final report.
3. Outputs in `analysis/`:
   - `final-report.json` and `final-report.md` (script-generated);
   - `sft-usable-passes.jsonl`;
   - `degenerate-v5-vs-build.json` and `degenerate-v5-vs-build-attempts.jsonl`;
   - `features/kimi5-all.jsonl` and `features/kimi5-build.jsonl`.

## Notes for main
- **README is now stale.** README.md still says degenerate.py is byte-identical to v4's and that phase_report.py is "v3-shaped". I did not edit README; you may want to update it.
- **Rules:**
  - No Slurm job was submitted. The only Slurm commands were standalone squeue calls, plus progress.py's own squeue.
  - No v4 file or job was touched; v4 files and its PROGRESS rows were only read.
  - No key value was printed or copied.
  - No permission check denied a command.