# kimi-rollouts-v5 final report (phase_report.py, 2026-10-29T11:12:23-0700)

33 official2 tasks x 5, Kimi-K3 (effort max), OpenCode stock prompt, 262,144 window; build-time comparison = the synth pipeline's 5 Kimi probe slots per task (1M window).

## pass@1

| set | v5 (256k) | build (1M) |
|---|---|---|
| first attempts / build slots | 132/165 = 80.0% (CI [72.7, 86.7]) | 139/165 = 84.2% (CI [79.4, 89.7]) |
| excluding broken-ended | 132/157 = 84.1% (CI [76.5, 91.0]) | 139/156 = 89.1% (CI [84.3, 93.6]) |
| after the one re-run (final slots) | 139/165 = 84.2% (CI [76.4, 91.5]) | (no re-runs) |
| tasks with >= 1 pass / all 5 / none | 33 / 19 / 0 | 33 / 14 / 0 |

paired task bootstrap, v5 final - build: {"diff_points": 0.0, "ci95": [-7.9, 7.3], "tasks": 33}; v5 first - build: {"diff_points": -4.2, "ci95": [-12.1, 3.0], "tasks": 33}; both excluding broken-ended: {"diff_points": -5.0, "ci95": [-11.6, 1.3], "tasks": 33}

## Broken replies and re-runs

broken-ended first attempts 8/165 (4.8%), kinds {"pad": 8}, all failed: 8; re-runs 8: {"pass": 7, "fail": 1, "broken_again": 0, "infra": 0}; detector check {"attempts": 173, "mismatches": []}

| campaign | task | slot | original | broken reply (completion tokens) | re-run | re-run status (F2P) |
|---|---|---|---|---|---|---|
| official2-g0 | syn-otto-cand-c02-r1b1-a4d613c7 | kimi-r2 | fail (F2P 0/72, patch 0 chars, 21 replies) | pad (27682) | official2-g0-brk1 kimi-r0 | pass (72/72) |
| official2-g0 | syn-otto-cand-c05-r1b4-da7cd37c | kimi-r2 | fail (F2P 0/26, patch 0 chars, 2 replies) | pad (79) | official2-g0-brk1 kimi-r0 | pass (26/26) |
| official2-g1 | syn-markdown-cand-c04-r1b0-3b72285b | kimi-r0 | fail (F2P 0/40, patch 0 chars, 34 replies) | pad (28788) | official2-g1-brk1 kimi-r0 | pass (40/40) |
| official2-g1 | syn-pyserde-cand-c04-r1b4-443ff5f7 | kimi-r0 | fail (F2P 0/45, patch 0 chars, 68 replies) | pad (29525) | official2-g1-brk1 kimi-r0 | pass (45/45) |
| official2-g2 | syn-markdown-cand-c01-r1b0-ecd019d8 | kimi-r0 | fail (F2P 0/40, patch 0 chars, 41 replies) | pad (21980) | official2-g2-brk1 kimi-r0 | pass (40/40) |
| official2-g2 | syn-pyserde-cand-c05-r1b2-170c68fa | kimi-r4 | fail (F2P 0/53, patch 0 chars, 57 replies) | pad (28199) | official2-g2-brk1 kimi-r0 | pass (53/53) |
| official2-g3 | syn-django-silk-cand-c01-r1b1-ffe67368 | kimi-r0 | fail (F2P 0/144, patch 0 chars, 22 replies) | pad (14776) | official2-g3-brk1 kimi-r0 | fail (141/144) |
| official2-g3 | syn-otto-cand-c05-r1b1-661a75c4 | kimi-r3 | fail (F2P 0/88, patch 0 chars, 118 replies) | pad (28894) | official2-g3-brk1 kimi-r0 | pass (88/88) |

per reply: v5 9/12935 = 0.7/1000; build 16/11981 = 1.34/1000; Fisher p 0.16, rate ratio 0.52 (exact 95% [0.2, 1.25]); rollouts ended {"control": "9/165", "new": "8/165", "fisher_p": 1.0}; with any {"control": "16/165", "new": "9/165", "fisher_p": 0.2112897386734357}

## 262,144 window

| | v5 all graded | build (1M) |
|---|---|---|
| rollouts | 173 | 165 |
| compacted_rollouts | 0 | 0 |
| main_prompt_max_median | 111792 | 110076 |
| main_prompt_max_p90 | 163596 | 165208 |
| main_prompt_max_max | 235336 | 214613 |
| rollouts_main_prompt_over_200k | 6 | 2 |
| rollouts_main_prompt_over_242144 | 0 | 0 |
| hub_calls_over_242144_any_session | 0 | 0 |
| hub_prompt_max_any_session | 235336 | 214613 |
| length_cut_replies | 0 | 0 |
| non200_statuses | {'408': 4} | {} |

## Reasoning before actions

| metric | v5 all graded | v5 first | v5 final | build (1M) |
|---|---|---|---|---|
| rollouts | 173 | 165 | 165 | 165 |
| action_turns | 12689 | 12001 | 12334 | 11687 |
| pct_action_turns_with_reasoning | 54.0 | 54.3 | 54.1 | 54.1 |
| ci95_pct_action_turns_with_reasoning | [52.2, 56.1] | [52.5, 56.4] | [52.4, 56.1] | [52.2, 55.9] |
| rollout_mean_pct_action_reasoned | 55.6 | 55.9 | 55.3 | 55.2 |
| pct_rollouts_all_actions_reasoned | 0.6 | 0.6 | 0.0 | 0.0 |
| pct_first_action_reasoned | 100.0 | 100.0 | 100.0 | 99.4 |
| pct_action_turns_with_visible_text | 30.6 | 30.6 | 30.7 | 29.8 |
| pct_action_turns_bare | 42.9 | 42.7 | 42.8 | 43.1 |
| reasoning_tokens_per_reasoning_turn_median | 114.0 | 114.0 | 114 | 110 |
| turns_per_rollout_median | 69 | 69 | 72 | 68 |
| tool_calls_per_rollout_median | 81 | 80 | 83 | 79 |
| agent_wall_min_median | 44.2 | 46.2 | 44.2 | 24.2 |
| pct_rollouts_with_task_subagent | 1.7 | 1.8 | 1.8 | 3.6 |

paired v5 final - build: {"diff_points": 0.1, "ci95": [-1.1, 1.4], "tasks": 33}; without degenerate rollouts: {"diff_points": -0.0, "ci95": [-1.4, 1.4], "tasks": 33}

## Cost (LiteLLM usage.cost of v5's own replies)

| key | batch | first attempts | re-runs | pilot | key remaining (last check) |
|---|---|---|---|---|---|
| NVAPI_KEY_NEW_2 | $387.68 | $371.25 | $16.43 | $2.36 | 5103.16 (2026-10-29T10:54:13-0700) |
| NVAPI_KEY_NEW_3 | $0 | $0 | $0 | $0 | None (None) |

v5 all attempts {"attempts": 173, "usd": 387.68, "usd_per_rollout_mean": 2.241, "usd_per_rollout_median": 1.944, "replies": 12935, "replies_with_cost": 12935, "prompt_Mtok": 1094.76, "cached_pct": 89.8, "completion_Mtok": 10.153}
build reference {"attempts": 166, "usd": 281.05, "usd_per_rollout_mean": 1.693, "usd_per_rollout_median": 1.483, "replies": 11981, "replies_with_cost": 11981, "prompt_Mtok": 968.5, "cached_pct": 95.0, "completion_Mtok": 9.138}
progress.py last row {"time": "10-29 10:54", "spend": {"NVAPI_KEY_NEW_2": 387.68, "NVAPI_KEY_NEW_3": 0.0}, "spent": 387.68, "pilot_cost": 2.36, "usd_per_rollout": 2.241}

## Per task (v5 vs build-time 1M probes)

| task | run | why | build pass | build clean rate | build broken-ended | v5 first pass | v5 broken-ended -> re-run | v5 final pass | final - build | reasoned % v5 / build | turns v5 / build | $/rollout v5 / build | max prompt v5 | SFT-usable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-pyserde-cand-c04-r1b1-2b74e357 | adaptix | accepted | 5/5 | 1.0 | 0 | 4/5 | - | 4/5 | -1 | 49.7 / 53.9 | 86 / 69 | 2.61 / 1.88 | 131716 | 4 |
| syn-pyserde-cand-c04-r1b4-443ff5f7 | adaptix | accepted | 5/5 | 1.0 | 0 | 4/5 | kimi-r0:pass | 5/5 | +0 | 47.1 / 54.2 | 73.5 / 69 | 2.05 / 2.39 | 114036 | 5 |
| syn-pyserde-cand-c05-r1b2-170c68fa | adaptix | accepted | 5/5 | 1.0 | 0 | 4/5 | kimi-r4:pass | 5/5 | +0 | 48.3 / 43.5 | 89.0 / 93 | 2.29 / 2.66 | 131196 | 5 |
| syn-django-silk-cand-c01-r1b1-ffe67368 | aiomonitor | accepted | 4/5 | 0.8 | 0 | 2/5 | kimi-r0:fail | 2/5 | -2 | 50.3 / 55.4 | 59.0 / 55 | 1.15 / 0.87 | 87102 | 2 |
| syn-django-silk-cand-c02-r1b4-3a06fd8c | aiomonitor | accepted | 5/5 | 1.0 | 0 | 3/5 | - | 3/5 | -2 | 50.4 / 43.9 | 49 / 52 | 1.13 / 0.72 | 104098 | 3 |
| syn-django-silk-cand-c03-r1b0-9bf52923 | aiomonitor | accepted | 5/5 | 1.0 | 0 | 5/5 | - | 5/5 | +0 | 44.4 / 48.6 | 49 / 46 | 0.8 / 0.58 | 69991 | 5 |
| syn-django-silk-cand-c04-r1b2-2a7a6a57 | aiomonitor | accepted | 4/5 | 0.8 | 0 | 4/5 | - | 4/5 | +0 | 54.3 / 49.5 | 49 / 57 | 0.82 / 0.79 | 81723 | 4 |
| syn-pyserde-cand-c04-r1b2-14d40723 | cattrs | accepted | 5/5 | 1.0 | 0 | 5/5 | - | 5/5 | +0 | 56.9 / 58.5 | 93 / 74 | 3.22 / 2.11 | 149201 | 5 |
| syn-jd-cand-c01-r1b1-ff9da598 | etree | accepted | 5/5 | 1.0 | 0 | 5/5 | - | 5/5 | +0 | 62.1 / 61.9 | 43 / 36 | 1.83 / 1.06 | 138713 | 5 |
| syn-jd-cand-c02-r1b4-3aa68c3b | etree | accepted | 3/5 | 0.6 | 0 | 3/5 | - | 3/5 | +0 | 68.3 / 66.5 | 42 / 40 | 1.74 / 0.84 | 146768 | 3 |
| syn-jd-cand-c05-r1b0-7994aee3 | etree | accepted | 5/5 | 1.0 | 0 | 3/5 | - | 3/5 | -2 | 67.2 / 61.7 | 61 / 58 | 1.62 / 1.34 | 132034 | 3 |
| syn-knex-cand-c01-r1b3-a888f2f2 | kysely | both | 3/5 | 0.6 | 0 | 2/5 | - | 2/5 | -1 | 53.3 / 53.0 | 101 / 101 | 1.96 / 1.43 | 105543 | 2 |
| syn-knex-cand-c04-r1b0-b21e1e98 | kysely | accepted | 5/5 | 1.0 | 0 | 4/5 | - | 4/5 | -1 | 53.4 / 56.4 | 100 / 120 | 1.39 / 1.54 | 85195 | 4 |
| syn-pabot-cand-c03-r1b0-5f252c24 | mobly | accepted | 4/5 | 0.8 | 0 | 4/5 | - | 4/5 | +0 | 58.2 / 55.2 | 96 / 97 | 3.59 / 2.76 | 150412 | 4 |
| syn-markdown-cand-c01-r1b0-ecd019d8 | obsidian | accepted | 4/5 | 0.8 | 1 | 3/5 | kimi-r0:pass | 4/5 | +0 | 60.4 / 54.5 | 50.5 / 64 | 1.25 / 0.98 | 117073 | 4 |
| syn-markdown-cand-c01-r1b3-986c83f4 | obsidian | accepted | 4/5 | 0.8 | 1 | 5/5 | - | 5/5 | +1 | 64.7 / 62.6 | 62 / 73 | 2.55 / 1.86 | 146572 | 5 |
| syn-markdown-cand-c01-r1b4-7673d9a2 | obsidian | accepted | 4/5 | 0.8 | 0 | 5/5 | - | 5/5 | +1 | 61.4 / 63.6 | 74 / 61 | 1.82 / 1.29 | 142073 | 5 |
| syn-markdown-cand-c04-r1b0-3b72285b | obsidian | accepted | 5/5 | 1.0 | 0 | 4/5 | kimi-r0:pass | 5/5 | +0 | 58.6 / 60.1 | 77.5 / 77 | 2.31 / 2.04 | 165499 | 5 |
| syn-goleveldb-cand-c01-r1b1-7523606b | pebble | accepted | 4/5 | 0.8 | 1 | 5/5 | - | 5/5 | +1 | 53.8 / 54.0 | 52 / 39 | 1.35 / 0.83 | 106803 | 5 |
| syn-goleveldb-cand-c01-r1b2-b09a37b8 | pebble | accepted | 5/5 | 1.0 | 0 | 5/5 | - | 5/5 | +0 | 60.5 / 57.1 | 32 / 36 | 0.93 / 0.69 | 92289 | 5 |
| syn-goleveldb-cand-c01-r1b4-958e653b | pebble | accepted | 3/5 | 0.6 | 0 | 3/5 | - | 3/5 | +0 | 52.9 / 52.6 | 59 / 47 | 2.06 / 0.86 | 118076 | 3 |
| syn-goleveldb-cand-c02-r1b0-e5f8b461 | pebble | accepted | 5/5 | 1.0 | 0 | 5/5 | - | 5/5 | +0 | 56.2 / 60.4 | 45 / 47 | 1.2 / 0.87 | 113028 | 5 |
| syn-goleveldb-cand-c03-r1b4-9045ebe3 | pebble | accepted | 4/5 | 0.8 | 0 | 1/5 | - | 1/5 | -3 | 56.2 / 52.3 | 48 / 69 | 2.15 / 1.42 | 138424 | 1 |
| syn-kapacitor-cand-c02-r1b1-27100f50 | promreload | accepted | 3/5 | 0.6 | 1 | 3/5 | - | 3/5 | +0 | 50.1 / 49.2 | 84 / 74 | 4.42 / 2.97 | 229729 | 3 |
| syn-kapacitor-cand-c05-r1b1-55fcf64d | promreload | accepted | 4/5 | 0.8 | 0 | 4/5 | - | 4/5 | +0 | 55.6 / 55.1 | 50 / 53 | 1.59 / 2.1 | 107826 | 4 |
| syn-risor-cand-c01-r1b4-74844668 | tengo-callable | strong | 3/5 | 0.6 | 1 | 5/5 | - | 5/5 | +2 | 52.4 / 44.7 | 72 / 73 | 2.4 / 1.59 | 134350 | 5 |
| syn-otto-cand-c01-r1b1-0f53e898 | tengo-destructuring | accepted | 4/5 | 0.8 | 1 | 5/5 | - | 5/5 | +1 | 49.9 / 50.4 | 138 / 148 | 6.11 / 3.63 | 235336 | 5 |
| syn-otto-cand-c02-c19003e2 | tengo-destructuring | accepted | 5/5 | 1.0 | 0 | 5/5 | - | 5/5 | +0 | 63.6 / 60.1 | 73 / 74 | 2.64 / 1.89 | 165600 | 5 |
| syn-otto-cand-c02-r1b1-a4d613c7 | tengo-destructuring | accepted | 5/5 | 1.0 | 0 | 4/5 | kimi-r2:pass | 5/5 | +0 | 50.5 / 56.6 | 81.5 / 75 | 2.36 / 2.41 | 178409 | 5 |
| syn-otto-cand-c03-r1b3-07071613 | tengo-destructuring | accepted | 3/5 | 0.6 | 1 | 5/5 | - | 5/5 | +2 | 60.0 / 57.1 | 69 / 66 | 1.57 / 1.41 | 108063 | 5 |
| syn-otto-cand-c04-r1b1-8b8a5c41 | tengo-destructuring | accepted | 4/5 | 0.8 | 0 | 5/5 | - | 5/5 | +1 | 51.4 / 53.6 | 119 / 128 | 4.16 / 3.26 | 186021 | 4 |
| syn-otto-cand-c05-r1b1-661a75c4 | tengo-destructuring | accepted | 3/5 | 0.6 | 1 | 4/5 | kimi-r3:pass | 5/5 | +2 | 47.5 / 48.0 | 121.5 / 129 | 4.03 / 2.88 | 214168 | 5 |
| syn-otto-cand-c05-r1b4-da7cd37c | tengo-destructuring | accepted | 4/5 | 0.8 | 1 | 4/5 | kimi-r2:pass | 5/5 | +1 | 51.0 / 51.6 | 138.5 / 120 | 2.78 / 2.18 | 185877 | 5 |

tasks: v5 final higher 9, equal 17, lower 7; diff histogram {"-3": 1, "-2": 3, "-1": 3, "0": 17, "1": 6, "2": 3}. accepted tasks needed teacher_clean_rate >= 0.6 at build time (task_builder.score), so the build-time Kimi rates of these tasks are selected upward (regression to the mean is expected in any re-run); the strong non-accepted task (risor) also had 0.6.

## SFT-usable passes: 138 of 139

graded pass; evaluation completed; reward 1; every F2P and P2P test passing; no verifier or agent error; non-empty patch; exported session; no degenerate hub reply anywhere in the attempt (sub-sessions included) and no [PAD] / special-token text in the main session. No LLM cheating audit.

- excluded: syn-otto-cand-c04-r1b1-8b8a5c41 kimi-r2 (first, official2-g2): broken reply in the attempt (pad@reply37/153 no-tool; ended a task subagent); [PAD]/special text in the main session (assistant parts 0, tool outputs 1)

| task | usable passes (slot, role) |
|---|---|
| syn-pyserde-cand-c04-r1b1-2b74e357 | kimi-r0, kimi-r1, kimi-r2, kimi-r3 |
| syn-pyserde-cand-c04-r1b4-443ff5f7 | kimi-r0 (rerun, official2-g1-brk1), kimi-r1, kimi-r2, kimi-r3, kimi-r4 |
| syn-pyserde-cand-c05-r1b2-170c68fa | kimi-r0, kimi-r1, kimi-r2, kimi-r3, kimi-r4 (rerun, official2-g2-brk1) |
| syn-django-silk-cand-c01-r1b1-ffe67368 | kimi-r1, kimi-r2 |
| syn-django-silk-cand-c02-r1b4-3a06fd8c | kimi-r0, kimi-r1, kimi-r3 |
| syn-django-silk-cand-c03-r1b0-9bf52923 | kimi-r0, kimi-r1, kimi-r2, kimi-r3, kimi-r4 |
| syn-django-silk-cand-c04-r1b2-2a7a6a57 | kimi-r1, kimi-r2, kimi-r3, kimi-r4 |
| syn-pyserde-cand-c04-r1b2-14d40723 | kimi-r0, kimi-r1, kimi-r2, kimi-r3, kimi-r4 |
| syn-jd-cand-c01-r1b1-ff9da598 | kimi-r0, kimi-r1, kimi-r2, kimi-r3, kimi-r4 |
| syn-jd-cand-c02-r1b4-3aa68c3b | kimi-r0, kimi-r2, kimi-r3 |
| syn-jd-cand-c05-r1b0-7994aee3 | kimi-r0, kimi-r1, kimi-r4 |
| syn-knex-cand-c01-r1b3-a888f2f2 | kimi-r2, kimi-r4 |
| syn-knex-cand-c04-r1b0-b21e1e98 | kimi-r0, kimi-r1, kimi-r3, kimi-r4 |
| syn-pabot-cand-c03-r1b0-5f252c24 | kimi-r0, kimi-r1, kimi-r3, kimi-r4 |
| syn-markdown-cand-c01-r1b0-ecd019d8 | kimi-r0 (rerun, official2-g2-brk1), kimi-r1, kimi-r2, kimi-r4 |
| syn-markdown-cand-c01-r1b3-986c83f4 | kimi-r0, kimi-r1, kimi-r2, kimi-r3, kimi-r4 |
| syn-markdown-cand-c01-r1b4-7673d9a2 | kimi-r0, kimi-r1, kimi-r2, kimi-r3, kimi-r4 |
| syn-markdown-cand-c04-r1b0-3b72285b | kimi-r0 (rerun, official2-g1-brk1), kimi-r1, kimi-r2, kimi-r3, kimi-r4 |
| syn-goleveldb-cand-c01-r1b1-7523606b | kimi-r0, kimi-r1, kimi-r2, kimi-r3, kimi-r4 |
| syn-goleveldb-cand-c01-r1b2-b09a37b8 | kimi-r0, kimi-r1, kimi-r2, kimi-r3, kimi-r4 |
| syn-goleveldb-cand-c01-r1b4-958e653b | kimi-r0, kimi-r1, kimi-r4 |
| syn-goleveldb-cand-c02-r1b0-e5f8b461 | kimi-r0, kimi-r1, kimi-r2, kimi-r3, kimi-r4 |
| syn-goleveldb-cand-c03-r1b4-9045ebe3 | kimi-r2 |
| syn-kapacitor-cand-c02-r1b1-27100f50 | kimi-r0, kimi-r1, kimi-r4 |
| syn-kapacitor-cand-c05-r1b1-55fcf64d | kimi-r1, kimi-r2, kimi-r3, kimi-r4 |
| syn-risor-cand-c01-r1b4-74844668 | kimi-r0, kimi-r1, kimi-r2, kimi-r3, kimi-r4 |
| syn-otto-cand-c01-r1b1-0f53e898 | kimi-r0, kimi-r1, kimi-r2, kimi-r3, kimi-r4 |
| syn-otto-cand-c02-c19003e2 | kimi-r0, kimi-r1, kimi-r2, kimi-r3, kimi-r4 |
| syn-otto-cand-c02-r1b1-a4d613c7 | kimi-r0, kimi-r1, kimi-r2 (rerun, official2-g0-brk1), kimi-r3, kimi-r4 |
| syn-otto-cand-c03-r1b3-07071613 | kimi-r0, kimi-r1, kimi-r2, kimi-r3, kimi-r4 |
| syn-otto-cand-c04-r1b1-8b8a5c41 | kimi-r0, kimi-r1, kimi-r3, kimi-r4 |
| syn-otto-cand-c05-r1b1-661a75c4 | kimi-r0, kimi-r1, kimi-r2, kimi-r3 (rerun, official2-g3-brk1), kimi-r4 |
| syn-otto-cand-c05-r1b4-da7cd37c | kimi-r0, kimi-r1, kimi-r2 (rerun, official2-g0-brk1), kimi-r3, kimi-r4 |

Full list with attempt dirs and export paths: analysis/sft-usable-passes.jsonl.
