# SFT v2 training accuracy vs base (2026-10-24)

50 tasks sampled (seed 20261024) from the 97 hengyuf tasks of the v2 SFT pool (Baihe's 97: snapshots not accessible); 5 rollouts per task per model, the synth pipeline's probe harness and each task's grader; both models served with the lenient-think parser.

| model | groups done | rollouts | graded | infra | pass@1 | task-mean | pass@5 | tasks 5/5 |
|---|---|---|---|---|---|---|---|---|
| v2 SFT (iter38) | 8/8 | 250 | 249 | 1 | 18.1% | 18.0% | 52.0% | 0 |
| base (sc14) | 8/8 | 250 | 250 | 0 | 17.2% | 17.2% | 46.0% | 0 |

| task | run | v2 SFT | base |
|---|---|---|---|
| syn-aiohttp-cand-c01-r1b4-9f77d81d | official1-httpx | ..P.P | P.... |
| syn-aiohttp-cand-c03-r1b1-b706aac1 | official1-httpx | .PPPP | .PPPP |
| syn-aiohttp-cand-c04-7f733601 | official1-httpx | ..... | ..... |
| syn-aiohttp-cand-c05-r1b0-7d9ec1f5 | official1-httpx | PPPP. | PP.PP |
| syn-argo-cd-cand-c05-cc3e1272 | official1-arcane | ..P.. | ..... |
| syn-bitecs-cand-c05-r1b0-85a91c3b | official1-koota | P.... | ..... |
| syn-bitecs-cand-c05-r1b4-bc99c654 | official1-koota | P..P. | .P.PP |
| syn-dacite-cand-c01-r1b3-84fe778b | official1-cattrs | .P... | ..... |
| syn-dacite-cand-c02-r1b4-325c5f9d | official1-cattrs | ..... | ..... |
| syn-dacite-cand-c04-r1b0-eeb19c3a | official1-cattrs | .PP.P | ..... |
| syn-dacite-cand-c05-r1b4-2e37bfb7 | official1-cattrs | .PPPP | ..P.P |
| syn-gomacro-cand-c01-c696c6ca | official1-scriggo | ..... | ..... |
| syn-gomacro-cand-c02-r1b0-e8804dd9 | official1-scriggo | ..... | ..... |
| syn-gomacro-cand-c02-r1b3-ed535f70 | official1-scriggo | ..... | ..... |
| syn-gomacro-cand-c04-r1b4-2d729ead | official1-scriggo | ..... | ..... |
| syn-gpython-cand-c04-r1b4-0a5ca664 | official1-anko | ..... | ..... |
| syn-ibis-cand-c03-r1b4-3d3b86b5 | official1-narwhals | ..... | ..... |
| syn-ibis-cand-c04-86a17614 | official1-narwhals | ....P | ..... |
| syn-ibis-cand-c04-r1b0-r2b0-35888795 | official1-narwhals | ...PP | P.... |
| syn-ibis-cand-c05-r1b2-r2b3-269e70d9 | official1-narwhals | .P... | .PP.. |
| syn-ibis-cand-c05-r1b3-9f713fee | official1-narwhals | ..... | ...P. |
| syn-jest-cand-c01-r1b3-08931258 | official1-vitest | ..... | ..... |
| syn-jest-cand-c04-r1b4-d8b1da89 | official1-testem | ..... | ..... |
| syn-jest-cand-c05-9e1caaee | official1-testem | .P... | ..... |
| syn-jest-cand-c05-r1b2-d6a136fe | official1-vitest | ..... | ..... |
| syn-pylama-cand-c01-r1b4-r2b4-68d85058 | official1-bandit | ...PP | PPPP. |
| syn-pylama-cand-c02-r1b0-r2b2-02688723 | official1-bandit | ..... | ..... |
| syn-pylama-cand-c03-r1b2-6182c7b3 | official1-bandit | ...P. | ..... |
| syn-pylama-cand-c03-r1b4-r2b4-58e91fc7 | official1-bandit | ..... | P.... |
| syn-pylama-cand-c05-r1b4-6eedf9a6 | official1-bandit | .P... | P.... |
| syn-risor-cand-c05-r1b2-3718460d | official1-absslices | .P.P. | ..P.. |
| syn-transitions-cand-c03-c910069f | official1-statemachine | x.... | ...P. |
| syn-transitions-cand-c04-r1b2-799317b1 | official1-statemachine | ..P.. | ...P. |
| syn-transitions-cand-c05-r1b2-6a1e0dca | official1-statemachine | P.... | P.P.. |
| syn-vscode-sqltools-cand-c01-r1b0-24c4a31b | official1-sqlformatter | P.P.P | ..... |
| syn-vscode-sqltools-cand-c02-r1b0-fb4b293a | official1-sqlformatter | ..... | ..... |
| syn-vscode-sqltools-cand-c02-r1b3-7119812a | official1-sqlformatter | .P... | ..... |
| syn-vscode-sqltools-cand-c04-r1b3-0c604520 | official1-sqlformatter | ..... | ..... |
| syn-vscode-sqltools-cand-c05-r1b3-911416cb | official1-sqlformatter | ..... | ..... |
| syn-wasmtime-cand-c01-r1b1-82c9e518 | official1-wasmi | ...P. | .PPP. |
| syn-wasmtime-cand-c03-353ec720 | official1-wasmi | ..... | ..P.. |
| syn-wasmtime-cand-c04-r1b1-bbb510ca | official1-wasmi | ..... | ..... |
| syn-wasmtime-cand-c04-r1b4-786cced5 | official1-wasmi | ..... | ..... |
| syn-yj-cand-c01-7e8e179e | official1-dasel | ...P. | .P..P |
| syn-yq-cand-c01-r1b4-c75b758f | official1-ytt | P...P | PP... |
| syn-yq-cand-c02-r1b0-0af03743 | official1-ytt | ..P.. | ..... |
| syn-yq-cand-c02-r1b1-2f81d284 | official1-ytt | ..... | .P... |
| syn-yq-cand-c02-r1b4-c30118dd | official1-ytt | ..... | .P.PP |
| syn-yq-cand-c04-r1b4-cae9dfb2 | official1-ytt | ..... | .P... |
| syn-zsa-cand-c02-r1b1-ae1eb794 | official1-dynamodb | ...P. | .P... |

P = pass, . = fail, x = infra (excluded).

## Empty patches on the training tasks (2026-10-25 01:30)

The DeepSWE empty-patch classifier (eval/deepswe-v1.1/analysis/empty-patch-20261018) was run on each slot's final
attempt; outputs in `analysis/empty-patch/{sftv2,sc14}/`.

| | v2 SFT | base sc14 |
|---|---|---|
| resolved | 45/250 | 43/250 |
| empty patch | 128 (51%) | 16 (6%) |
| resolved / rollouts with a patch | 45/122 = 37% | 43/234 = 18% |
| committed | 121 | 233 |
| 1-token stops (EOS right after `<think>`) | 26 | 0 |
| stopped right after the compaction "continue" | 79 | 10 |
| >= 1 compaction | 145 | 145 |
| empty, no compaction / compacted | 40/105, 88/145 | 1/105, 15/145 |

v2 empty-patch causes: edited but uncommitted 57 (deferred/asked to commit 15, mid-work status 9, other 33), premature end
56 (tool call in think 27, reasoning-only 25, malformed 3, cap 1), no edit 6, not finished 9.
Even on its own training tasks, v2 stops the agent loop as on DeepSWE. When it does produce a patch it resolves twice as
often as base (37% vs 18%), which the stops cancel out (pass@1 18.1% vs 17.2%).

## SFT v2 at 10 and 20 epochs (2026-10-25, final 21:30)

Checkpoints: super-kimi-raw-sft-v2-wsd-10ep-20261024 iter76 and -20ep-20261024 iter152 (the same v2 corpus).
- Same harness, tasks and grader as above.
- Served with `assets/nano_v3_lenient_think_counted_parser.py`: it parses like the lenient parser and also logs every missing-`</think>` case.
- All 250/250 rollouts per model graded, 0 infra.
- Analysis files: `analysis/{accuracy/paired.md, features/comparison.md, think/split_table.md, empty-patch/table.md}`.

### Accuracy

| | base | 5 ep | 10 ep | 20 ep |
|---|---|---|---|---|
| pass@1 | 17.2% (43/250) | 18.1% (45/249) | 36.0% (90/250) | 44.0% (110/250) |
| pass@5 | 46% | 52% | 74% | 74% |
| tasks 5/5 | 0 | 0 | 2 | 4 |

Paired comparisons over the 50 tasks (task bootstrap CI, sign-flip p):

| comparison | difference | 95% CI | p |
|---|---|---|---|
| 10 ep vs 5 ep | +17.9 pp | [+10.8, +25.6] | <1e-4 |
| 20 ep vs 5 ep | +25.9 pp | [+17.9, +34.2] | <1e-4 |
| 20 ep vs 10 ep | +8.0 pp | [+0.4, +15.6] | 0.06 |

Seven tasks are unsolved by all four models.

### Actions preceded by reasoning (the teacher-rollout extractor)

| | base | 5 ep | 10 ep | 20 ep |
|---|---|---|---|---|
| % action turns with reasoning | 100.0 | 64.7 | 59.3 | 63.1 |
| % rollouts with every action reasoned | 90.8 | 0.4 | 0.4 | 0.0 |
| longest unreasoned run (median) | 0 | 5 | 6 | 5 |
| bare action turns (no reasoning, no text) | 0% | 35.0% | 40.3% | 36.7% |
| git commit rate | 93.6% | 49.2% | 62.0% | 73.6% |

More epochs do not bring the reasoning-before-action pattern back toward base. The ~60% pattern is the teacher's (Kimi 54.9%) and is already learned by 5 epochs.

### Missing `</think>`

Model calls, then rollouts with at least one such call:

| | base | 5 ep | 10 ep | 20 ep |
|---|---|---|---|---|
| tool call without `</think>`, rescued by the lenient parser (logged) | n/a (>=1) | n/a (>=160 calls; >=108 rollouts) | 30 (0.044%); 29 (11.6%) | 9 (0.015%); 9 (3.6%) |
| unclosed think, no tool call | 11; 10 | 89; 84 | 43; 36 | 51; 42 |
| ... cut at the length cap (runaway thinking) | 2; 2 | 7; 5 | 21; 14 | 41; 33 |
| empty output (EOS right after `<think>`) | 0 | 25; 25 | 4; 4 | 0 |

- The rescued count cannot be measured for 5 ep and base, which were served with the plain lenient parser. Their lower bounds are the bare-rescue cases: no reasoning text, plus a tool call.
- The proposed "reasoning ends with a newline" signature failed validation: precision 5-10%.

### Empty patches (DeepSWE classifier)

| | base | 5 ep | 10 ep | 20 ep |
|---|---|---|---|---|
| empty patch | 6.4% | 51.2% | 38.0% | 26.4% |
| resolved / rollouts with a patch | 18% | 37% | 58% | 60% |
| stopped right after the compaction "continue" | 10 | 79 | 69 | 53 |
| edited but uncommitted | 7 | 57 | 65 | 34 |
| tool call inside think (headless call) | 3 | 27 | 16 | 7 |
| runaway generation to the 131k cap | 0 | 1 | 2 | 18 |

### What more epochs changed

- **Higher accuracy, for two reasons:** fewer empty patches, and a higher resolve rate among the rollouts that do produce a patch.
- **Fewer missing-`</think>` and empty-output failures.**
- **Reasoning before actions is unchanged.**
- **New failure at 20 epochs: runaway thinking.** Repetition loops run to the length cap; 33 rollouts have at least one, and 18 end on one. This looks like over-training.
- **Caveat:** these are training tasks. The DeepSWE (held-out) evals of both checkpoints were launched on 2026-10-25 and are the real test.

## Empty-think-prefix SFT, 5 and 10 epochs (2026-10-27)

Checkpoints: super-kimi-v2-empty-think-prefix-wsd-5ep-16n-ep32-20261027 iter38 ("etp 5 ep") and -10ep-16n-ep32-20261027
iter76 ("etp 10 ep"). They use the same v2 Kimi corpus as the raw runs, trained with an empty think prefix.
- Same 50 tasks x 5, harness, grader and counting lenient-think parser as the 10/20 ep runs. The vLLM command differs
  only in the model path; the chat template and tokenizer are byte-identical to raw v2.
- Finished 21:57; 249/250 graded per model (1 infra slot each, see Infra slots).
- Analysis files, regenerated with the etp columns (earlier columns unchanged): `analysis/{accuracy/paired.md,
  features/comparison.md, think/split_table.md, think/notool_kinds.json, think/reasoning_quality.md,
  empty-patch/table.md, cost/cost.md}`.
- These numbers supersede the live monitor's. Its empty patch 24.1 / 17.7% was over graded slots; the table below
  counts 61 / 45 of 250. Its missing `</think>` 0.055 / 0.036% came from parser counters; the final-attempt join gives
  0.059 / 0.038%.

### Accuracy (`analysis/accuracy/paired.md`)

| | base | 5 ep | 10 ep | 20 ep | etp 5 ep | etp 10 ep |
|---|---|---|---|---|---|---|
| pass@1 | 17.2% (43/250) | 18.1% (45/249) | 36.0% (90/250) | 44.0% (110/250) | 26.9% (67/249) | 39.4% (98/249) |
| pass@5 | 46% | 52% | 74% | 74% | 68% | 80% |
| tasks 5/5 | 0 | 0 | 2 | 4 | 0 | 2 |
| graded rollouts with a patch | 93.6% | 49.0% | 62.0% | 73.6% | 75.9% | 82.3% |
| resolved / rollouts with a patch | 18.4% | 36.9% | 58.1% | 59.8% | 35.4% | 47.8% |

Paired over the 50 tasks: task-bootstrap CI and sign-flip p. pass@1 = patched share x resolved/patched, so the
difference splits exactly into a patch-rate part and a resolve-rate part, each with a task-bootstrap CI.

| comparison | difference | 95% CI | p | tasks better/equal/worse | patch-rate part | resolve-rate part |
|---|---|---|---|---|---|---|
| etp 5 ep vs 5 ep | +8.8 pp | [+2.8, +14.8] | 0.010 | 23/20/7 | +9.7 [+6.8, +12.9] | -0.9 [-7.4, +5.5] |
| etp 10 ep vs 10 ep | +3.4 pp | [-3.5, +10.0] | 0.41 | 21/15/14 | +10.8 [+7.1, +14.6] | -7.4 [-13.1, -1.9] |
| etp 10 ep vs 20 ep | -4.6 pp | [-12.0, +2.6] | 0.25 | 14/17/19 | +4.7 [+1.1, +8.3] | -9.3 [-15.6, -3.0] |
| etp 10 ep vs etp 5 ep | +12.4 pp | [+5.9, +18.8] | 0.0007 | 26/19/5 | +2.7 [+0.2, +5.3] | +9.8 [+3.5, +15.9] |
| etp 5 ep vs base | +9.7 pp | [+2.1, +17.3] | 0.023 | 21/21/8 | -4.8 [-7.1, -2.8] | +14.5 [+7.0, +22.1] |
| etp 10 ep vs base | +22.2 pp | [+14.0, +30.6] | <1e-4 | 33/11/6 | -3.7 [-6.2, -1.7] | +25.9 [+17.7, +34.4] |

- **etp 5 ep vs raw 5 ep:** the gain is all fewer empty patches. Its patches resolve at the raw rate (35% vs 37%).
- **etp 10 ep vs raw 10 ep:** no significant gain. It has 50 more patched rollouts but only 8 more passes (16%, about
  base's 18%). Resolved/patched falls from 58% to 48% (-10.3 pp [-18.0, -2.7]).
- **10 vs 5 epochs:** significant (sign test p = 0.0002), mostly through the resolve rate.
- **etp 10 ep vs raw 20 ep:** within noise.
- pass@5: 34 vs 26 tasks at 5 ep (McNemar p 0.08); 40 vs 37 at 10 ep (p 0.58). Four tasks are unsolved by all six
  models.

### Thinking (`analysis/features/comparison.md`, `analysis/think/{split_table.md, notool_kinds.json, reasoning_quality.md}`)

| | base | 5 ep | 10 ep | 20 ep | etp 5 ep | etp 10 ep |
|---|---|---|---|---|---|---|
| % action turns with reasoning | 99.96 | 64.7 | 59.3 | 63.1 | 99.97 | 99.997 |
| % rollouts with every action reasoned | 90.8 | 0.4 | 0.4 | 0.0 | 89.2 | 99.2 |
| reasoning tokens per reasoning turn: median / mean | 126 / 659 | 210 / 757 | 241 / 875 | 196 / 853 | 137 / 568 | 130 / 565 |
| ... p10 / p90 | 20 / 1,479 | 33 / 1,616 | 40 / 1,776 | 35 / 1,569 | 26 / 1,235 | 26 / 1,130 |
| % reasoning turns under 50 tokens | 28.7 | 16.8 | 13.4 | 16.7 | 24.3 | 24.3 |
| reasoning tokens per rollout (mean) | 161k | 141k | 140k | 126k | 189k | 154k |
| KS distance to base, reasoning tokens per turn | - | 0.354 | 0.407 | 0.369 | 0.051 | 0.050 |
| % action turns with visible text | 5.1 | 31.9 | 32.3 | 28.5 | 39.0 | 40.0 |
| git commit rate | 93.6% | 49.2% | 62.0% | 73.6% | 75.9% | 82.0% |

- etp's unreasoned actions: 28 turns in 27 rollouts at 5 ep, 2 turns at 10 ep (base: 24 in 23).
- 17 of the 28 at 5 ep are `<think><tool_call>` with no reasoning text; the rest have an empty closed think.

Missing `</think>` (model calls, % of calls; rollouts with at least one):

| | base | 5 ep | 10 ep | 20 ep | etp 5 ep | etp 10 ep |
|---|---|---|---|---|---|---|
| model calls (final attempts) | 61,943 | 72,691 | 68,598 | 59,900 | 84,420 | 68,871 |
| all missing `</think>` | >=12 (>=0.019%); >=11 | >=249 (>=0.343%); >=159 | 73 (0.106%); 59 | 60 (0.100%); 47 | 50 (0.059%); 45 | 26 (0.038%); 22 |
| tool call without `</think>`, rescued by the parser | n/a (>=1) | n/a (>=160) | 30; 29 | 9; 9 | 19; 19 | 1; 1 |
| unclosed think, no tool call | 11; 10 | 89; 84 | 43; 36 | 51; 42 | 31; 27 | 25; 21 |
| ... ended by EOS | 9; 8 | 82; 81 | 22; 22 | 10; 9 | 14; 14 | 5; 5 |
| ... of which a headless tool call (closing tags only) | 2 | 27 | 16 | 9 | 4 | 3 |
| ... cut at the length limit (runaway thinking) | 2; 2 | 7; 5 | 21; 14 | 41; 33 | 17; 13 | 20; 16 |
| empty output (EOS right after `<think>`) | 0 | 25; 25 | 4; 4 | 0 | 0 | 0 |

Is the reasoning substantive?
- **Length:** the distribution is close to base's (KS 0.05 vs 0.35-0.41 for raw). Median 130-137 tokens; a quarter
  under 50 tokens, as in base.
- **No filler:** verbatim repeats of an earlier reasoning in the same rollout are 0.02 / 0.01% of reasoned actions (base
  0.01%). Short reasonings (< 100 chars) that name no file, identifier or number are 0.00%. The most frequent short
  reasoning, for every model, is the opener "Let me start by exploring the repository structure ...".
- **Degenerate loops persist.** These are runs like "hold on — hold on — ..." that the model escapes ("OK — I see I'm
  looping again"); measured as reasoning of 4,000+ chars with zlib ratio below 0.15.
  - Turns, then rollouts: base 5 in 2; raw 105 / 79 / 25 in 16 / 13 / 13; etp 126 / 95 in 28 / 13.
  - Runaway thinking to the length limit, as at raw 10 ep: 17 / 20 calls in 13 / 16 rollouts.

### Empty patches (DeepSWE classifier; `analysis/empty-patch/table.md`)

| of 250 rollouts | base | 5 ep | 10 ep | 20 ep | etp 5 ep | etp 10 ep |
|---|---|---|---|---|---|---|
| empty patch | 16 (6.4%) | 128 (51.2%) | 95 (38.0%) | 66 (26.4%) | 61 (24.4%) | 45 (18.0%) |
| 3 h deadline kill | 5 | 9 | 7 | 4 | 16 | 12 |
| edited but uncommitted | 7 | 57 | 65 | 34 | 30 | 20 |
| ... says it has not committed / asks to commit | 5 | 15 | 40 | 11 | 11 | 10 |
| tool call inside think (headless) | 3 | 27 | 16 | 7 | 4 | 4 |
| reasoning-only last turn | 1 | 25 | 2 | 1 | 7 | 1 |
| malformed tool call | 0 | 3 | 1 | 2 | 2 | 2 |
| runaway generation to the length limit | 0 | 0 | 2 | 18 | 1 | 5 |
| context overflow (HTTP 400) | 0 | 1 | 0 | 0 | 0 | 1 |
| no edit | 0 | 6 | 2 | 0 | 0 | 0 |
| sandbox lost | 0 | 0 | 0 | 0 | 1 | 0 |
| (overlapping) stopped right after the compaction "continue" | 10 | 79 | 69 | 53 | 35 | 24 |
| (overlapping) 1-token stops | 0 | 24 | 4 | 0 | 0 | 0 |

- **What fell, vs the raw run of the same epochs:**
  - Uncommitted edits: -27 at 5 ep, -45 at 10 ep.
  - Headless tool calls: -23 / -12.
  - Reasoning-only stops: -18 / -1.
  - 1-token stops: from 24 to 0 at 5 ep.
  - Stops after the compaction "continue": from 79 / 69 to 35 / 24.
- **Deadline kills rose:** +7 / +5. They are now the largest single cause, 16 of 61 and 12 of 45.
- **Correction to the 10/20-epoch table:**
  - The "1" runaway generation at 5 ep was its context-overflow slot (0 output tokens on the last turn), so the runaway
    counts are 0 / 0 / 2 / 18.
  - All "not finished" rollouts of the earlier runs were 3 h deadline kills (5 / 9 / 7 / 4).

### Cost: is reasoning before every action slower? (`analysis/cost/cost.md`)

| mean per rollout unless noted | base | 5 ep | 10 ep | 20 ep | etp 5 ep | etp 10 ep |
|---|---|---|---|---|---|---|
| model calls | 248 | 291 | 274 | 240 | 337 | 276 |
| completion tokens | 210k | 214k | 205k | 188k | 276k | 223k |
| completion tokens per call: median / mean | 281 / 848 | 243 / 736 | 224 / 749 | 225 / 782 | 318 / 821 | 299 / 808 |
| latency per call: median / mean (s) | 5.1 / 13.0 | 4.5 / 11.9 | 4.2 / 11.1 | 3.8 / 10.5 | 5.7 / 12.7 | 5.5 / 14.1 |
| model time (min) | 53.6 | 57.4 | 50.7 | 42.0 | 71.1 | 64.5 |
| wall time: mean / median / p90 (min) | 61.6 / 53.1 / 100.6 | 69.4 / 61.2 / 138.7 | 62.4 / 55.0 / 108.0 | 53.6 / 44.1 / 102.2 | 85.4 / 74.1 / 164.9 | 77.4 / 66.9 / 136.5 |
| rollouts over 2 h | 11 | 33 | 20 | 15 | 61 | 42 |
| 3 h deadline kills | 5 | 9 | 7 | 4 | 16 | 12 |
| vLLM per worker: running sequences; decode tok/s per sequence | 7.6; 70 | 5.6; 70 | 6.6; 73 | 5.9; 80 | 7.8; 70 | 8.1; 61 |

- **Per call:** reasoning on every action makes the typical call longer (median +31% vs raw). The long tail is shorter
  (reasoning p90 1.1-1.2k vs 1.6-1.8k tokens), so the mean rises only ~10%.
- **etp 5 ep vs raw 5 ep:** +46 calls (+16%), +29% completion tokens, wall +15.9 min [+10.0, +22.2]. At raw 5 ep's
  serving speed its model time would be the same (72.7 vs 71.1 min), so the extra time is all extra output. Part of it
  is selection: raw 5 ep ends half its rollouts early with no patch.
- **etp 10 ep vs raw 10 ep:**
  - Calls are the same (276 vs 274), completion tokens +8% (CI includes 0), and the very long rollouts are as many
    (500+ calls: 22 vs 22).
  - Wall time is still +15.0 min [+9.9, +20.5]. At raw 10 ep's serving speed its model time would be 54.3 min, vs 50.7
    for raw 10 ep and 64.5 actual.
  - So about 4 of the +14 model minutes are extra output. The rest is slower serving in this run (8.1 vs 6.6 sequences
    per worker, 61 vs 73 tok/s per sequence), which this comparison does not control.
- **Deadline-killed rollouts are long agent loops,** not single hung calls: 644 / 569 calls on average, all compacted,
  and 4 / 5 of them contain a runaway call.

### Infra slots
- **etp 5 ep, syn-yq-cand-c02-r1b0-0af03743 r0 (group 5, ATP needs_review): sandbox lost, true infra.**
  - At 2.3 h, with 561 calls done and the agent mid-task, the sandbox PTY failed (SandboxPtyError).
  - Every later OpenSandbox command timed out (snapshots 240 s, patch collection 360 s).
  - No export, no patch, no grading. The classifier says N4b_UNRESPONSIVE; nothing points to OOM.
- **etp 10 ep, syn-jest-cand-c05-9e1caaee r3 (group 7, ATP fatal): a model-side context overflow, not infra.**
  - The session never compacted (226 calls) and its prompt grew to 240,743 tokens. vLLM rejected the next call with
    HTTP 400: "your prompt contains at least 131073 input tokens" + 131,072 output tokens > 262,144.
  - The session ended on an empty "length" turn with nothing committed and was graded fail (f2p 0/28). ATP's
    generation-validity rule marks any unresolved 400 as fatal.
  - The raw 5 ep infra slot (syn-transitions-cand-c03 r0) is the same failure (400 after a ~240k-token prompt).
  - Counted as fails: etp 10 ep 98/250 = 39.2%, raw 5 ep 45/250 = 18.0%.

### Conclusions
- **The empty think prefix fixes the thinking pattern.** Reasoning precedes 99.97-100% of actions, with base-like
  lengths and no filler. Missing `</think>` falls to 0.06 / 0.04% of calls, and empty outputs to 0.
- **Empty patches halve at the same epoch count** (51 → 24%, 38 → 18%). The drop comes from fewer uncommitted edits
  and headless tool calls in think.
- **Accuracy:** +8.8 pp at 5 ep (significant). At 10 ep, +3.4 pp, not significant: the extra patches resolve at about
  base's rate. etp 10 ep is level with raw 20 ep. 10 ep beats 5 ep by +12.4 pp.
- **Cost:** rollouts take ~24% longer and hit the 3 h deadline about twice as often (16 / 12 vs 9 / 7). Deadline
  kills are now the main empty-patch cause.
- **Remaining defects:** degenerate reasoning loops and runaway thinking at raw-like rates.
- **Held-out DeepSWE** (`eval/deepswe-v1.1/analysis/etp-sft-20261027/REPORT.md`): resolved@1 is 14.87 / 22.12% vs raw
  10.44 / 13.98% and base 27.79%. There too, 3 h kills are the largest empty-patch cause (62 / 47).

### 20 epochs (etp20, 2026-10-28)

Checkpoint: super-kimi-v2-empty-think-prefix-wsd-20ep-16n-ep32-20261027 iter152 ("etp 20 ep").
- Same harness, parser and vLLM command as etp 5/10 ep; the chat template and tokenizer are byte-identical.
- Finished 02:35; 250/250 graded, 0 infra. There was no non-200 response, and all 38 parser log lines match
  final-attempt calls.
- Every analysis file above now has an etp 20 ep column; the other columns are unchanged.
- The live table (`analysis/watch/live_etp20_final_20261028.txt`) gave missing `</think>` 0.058% from parser
  counters; the final-attempt join gives 0.068% (37 calls).

| | base | 20 ep | etp 10 ep | etp 20 ep |
|---|---|---|---|---|
| pass@1 | 17.2% (43/250) | 44.0% (110/250) | 39.4% (98/249) | 52.8% (132/250) |
| pass@5 | 46% | 74% | 80% | 84% |
| tasks 5/5 | 0 | 4 | 2 | 7 |
| graded rollouts with a patch | 93.6% | 73.6% | 82.3% | 91.6% |
| resolved / rollouts with a patch | 18.4% | 59.8% | 47.8% | 57.6% |

Paired over the 50 tasks (`analysis/accuracy/paired.md`), with the patch-rate / resolve-rate split:

| comparison | difference | 95% CI | p | tasks better/equal/worse | patch-rate part | resolve-rate part |
|---|---|---|---|---|---|---|
| etp 20 ep vs 20 ep | +8.8 pp | [+2.0, +15.6] | 0.021 | 19/22/9 | +10.6 [+6.0, +15.5] | -1.8 [-7.8, +4.3] |
| etp 20 ep vs etp 10 ep | +13.4 pp | [+6.6, +20.1] | 0.0006 | 25/17/8 | +4.9 [+1.7, +8.4] | +8.6 [+2.7, +14.4] |
| etp 20 ep vs base | +35.6 pp | [+27.6, +43.6] | <1e-4 | 39/8/3 | -0.8 [-2.5, +0.8] | +36.4 [+28.1, +44.6] |

- **vs raw 20 ep:** the gain is all fewer empty patches; patches resolve at the raw rate (58% vs 60%). pass@5 is 42
  vs 37 tasks (McNemar p 0.13).
- **vs etp 10 ep:** significant, mostly through the resolve rate, which is back at the raw level (48% → 58%).

Empty patches (`analysis/empty-patch/table.md`):

| of 250 rollouts | base | 20 ep | etp 10 ep | etp 20 ep |
|---|---|---|---|---|
| empty patch | 16 (6.4%) | 66 (26.4%) | 45 (18.0%) | 21 (8.4%) |
| 3 h deadline kill | 5 | 4 | 12 | 3 (2 empty; 1 had committed a patch) |
| edited but uncommitted | 7 | 34 | 20 | 8 |
| runaway generation to the length limit | 0 | 18 | 5 | 7 |
| tool call inside think (headless) | 3 | 7 | 4 | 3 |
| malformed tool call / reasoning-only last turn | 0 / 1 | 2 / 1 | 2 / 1 | 1 / 0 |
| context overflow (HTTP 400) | 0 | 0 | 1 | 0 |
| (overlapping) stopped right after the compaction "continue" | 10 | 53 | 24 | 16 |

- Deadline kills fall back to the raw / base level (3 vs 4, base 5).
- The remaining causes are uncommitted edits (8) and runaway generation (7).

Thinking (`analysis/features/comparison.md`, `analysis/think/{split_table.md, notool_kinds.json, reasoning_quality.md}`):

| | base | 20 ep | etp 10 ep | etp 20 ep |
|---|---|---|---|---|
| % action turns with reasoning | 99.96 | 63.1 | 99.997 | 99.996 |
| % rollouts with every action reasoned | 90.8 | 0.0 | 99.2 | 99.2 |
| reasoning tokens per reasoning turn: median / mean / p90 | 126 / 659 / 1,479 | 196 / 853 / 1,569 | 130 / 565 / 1,130 | 135 / 662 / 1,302 |
| reasoning tokens per rollout (mean) | 161k | 126k | 154k | 140k |
| git commit rate | 93.6% | 73.6% | 82.0% | 91.2% |
| missing `</think>`: calls (% of calls); rollouts | >=12 (>=0.019%); >=11 | 60 (0.100%); 47 | 26 (0.038%); 22 | 37 (0.068%); 25 |
| ... rescued tool call / EOS inside think / cut at the length limit (calls) | n/a / 9 / 2 | 9 / 10 / 41 | 1 / 5 / 20 | 1 / 3 / 33 |
| length-limit cuts that end in a repetition loop: calls; rollouts | 2; 2 | 40; 32 | 19; 16 | 31; 22 |
| escaped "hold on" / "hmm" loops inside reasoning: turns; rollouts | 5; 2 | 25; 13 | 95; 13 | 1; 1 |

- **Missing `</think>` rose from 10 to 20 ep (26 → 37 calls).** The rise is all runaway thinking cut at the length
  limit (20 → 33 calls). 31 of the 33 end in a "hmm — hmm — ..." loop, as the live counters suggested late in the run.
- **At 20 ep the model almost never escapes such a loop:** 1 escaped loop, vs 95 at 10 ep. A loop now runs to the limit
  instead, and 7 rollouts end on one (raw 20 ep: 18).
- **Reasoning stays base-like and substantive:** KS distance to base 0.053; verbatim repeats 0.01%; generic short
  reasoning 0.00%.

Cost (`analysis/cost/cost.md`):

| mean per rollout unless noted | base | 20 ep | etp 10 ep | etp 20 ep |
|---|---|---|---|---|
| model calls | 248 | 240 | 276 | 218 |
| completion tokens | 210k | 188k | 223k | 200k |
| completion tokens per call: median / mean | 281 / 848 | 225 / 782 | 299 / 808 | 306 / 919 |
| latency per call: median / mean (s) | 5.1 / 13.0 | 3.8 / 10.5 | 5.5 / 14.1 | 5.3 / 13.9 |
| model time (min) | 53.6 | 42.0 | 64.5 | 50.7 |
| wall time: mean / median / p90 (min) | 61.6 / 53.1 / 100.6 | 53.6 / 44.1 / 102.2 | 77.4 / 66.9 / 136.5 | 61.6 / 51.8 / 106.2 |
| rollouts over 2 h | 11 | 15 | 42 | 18 |
| vLLM per worker: running sequences; decode tok/s per sequence | 7.6; 70 | 5.9; 80 | 8.1; 61 | 7.5; 70 |

- **vs raw 20 ep:** wall +7.9 min [+3.7, +12.1], with 9% fewer calls (-21.5 [-41.7, -2.9]) and +7% completion tokens
  (CI includes 0). At raw 20 ep's serving speed its model time would be 43.9 min (raw 42.0, actual 50.7), so most of
  the gap is serving load (7.5 vs 5.9 sequences per worker).
- **vs etp 10 ep:** 15.8 min faster [11.5, 20.1]. Mean wall time equals base's.

Infra slots: none (all 250 completed and graded; no non-200 response).

Conclusions at 20 epochs:
- **Best pass@1 of all runs:** 52.8%, which is +8.8 pp over raw 20 ep (p = 0.021), +13.4 pp over etp 10 ep and
  +35.6 pp over base.
- **Patches:** empty patches are 8.4% (base 6.4%), and the resolve rate among patched rollouts is back at the raw level.
- **Speed:** rollouts are as fast as base's, and deadline kills are back to 3.
- **Main remaining defect:** "hmm — hmm" loops that run to the length limit (33 calls in 22 rollouts; 7 rollouts end on
  one).
- **Caveat:** these are training tasks, and 20 epochs may fit them more closely; the held-out DeepSWE eval decides.


<!-- Section written by the main agent from workflow wf_f74e9ef0-498 (analyze-ta; two verifiers, 10 problems raised, reconciled). -->
## SFT-v3 empty-think-prefix, 5 and 10 epochs (2026-10-31)

Checkpoints: super-kimi-sft-v3-empty-think-prefix-wsd-5ep-16n-ep32-20261029 iter71 ("v3 etp 5 ep") and -10ep-16n-ep32-20261029 iter142 ("v3 etp 10 ep"). Tokenizer, chat template and configs are byte-identical to the v2 etp checkpoints and base.

- **Corpus:**
  - SFT-v3 has 1,594 conversations in 453 rows; v2 had 797 in 242.
  - On these 50 tasks SFT-v3 has 352 conversations; v2 had 191. The 352 are 186 of the v2 conversations plus 166 new Kimi K3 v4 passes on 46 tasks. The 5 v2 conversations over 262,144 tokens were dropped; all of them were gomacro c01 / c02-r1b3.
  - So one epoch is about 1.9x the optimizer steps: v3 5 ep (71 iterations) ≈ v2 10 ep (76), and v3 10 ep (142) ≈ v2 20 ep (152). The comparisons below are given at both the same epochs and the same steps.
- **Setup:** the same 50 tasks x 5, harness, graders and counting lenient-think parser as etp 5/10/20 ep. Campaigns ran 00:49 to 07:26.
- **Outcome:**
  - v3 etp 5 ep: 250/250 graded, 0 infra.
  - v3 etp 10 ep: 245 graded, 5 infra. Main closed it with 249/250 slots finished (see Closing decisions).
- **Files:**
  - Regenerated with v3 columns: `accuracy.md`, `summary.json`, and `analysis/{accuracy/paired.md, features/comparison.md, think/{split_table.md, notool_kinds.json, reasoning_quality.md}, empty-patch/table.md, cost/cost.md}`.
  - Each script got one-line entries, with backups as `*.pre-v3etp`. Every earlier column reproduces exactly. `paired.py` and `cost.py` also gained the v3 pairs.
  - New in `analysis/v3etp/`: `comparison_v3etp.{txt,csv,json}`, `serving_gaps.json`, `sensitivity_infra.json`, and `thinkx_ta.json` (the DeepSWE sftv3-etp `thinkx.py`, run read-only on these rollouts).
  - Reconcile fix: `compare_v3etp.py` now has the full deadline-kill definition in its docstring and corrected infra causes (the context-overflow mechanism and the ibis close time). `comparison_v3etp.json` was regenerated. Only those 5 cause strings changed; the `.txt` and `.csv` are byte-identical.
  - Final live table: `analysis/watch/live_v3etp_final_20261031.txt`.

### Comparison (`analysis/v3etp/comparison_v3etp.txt`)

| | base | raw 5 ep | raw 10 ep | raw 20 ep | etp 5 ep | etp 10 ep | etp 20 ep | v3 etp 5 ep | v3 etp 10 ep |
|---|---|---|---|---|---|---|---|---|---|
| **Accuracy** | | | | | | | | | |
| pass@1 [95% CI] | 17.2% (43/250) [10.8, 24.4] | 18.1% (45/249) [12.1, 24.5] | 36.0% (90/250) [28.0, 44.4] | 44.0% (110/250) [34.4, 53.6] | 26.9% (67/249) [20.5, 34.0] | 39.4% (98/249) [31.3, 47.2] | 52.8% (132/250) [43.6, 61.6] | 44.0% (110/250) [35.6, 52.8] | 56.7% (139/245) [47.8, 65.6] |
| pass@5 | 23/50 | 26/50 | 37/50 | 37/50 | 34/50 | 40/50 | 42/50 | 41/50 | 45/50 |
| tasks 5/5 | 0 | 0 | 2 | 4 | 0 | 2 | 7 | 5 | 9 |
| resolved / graded rollouts with a patch | 43/234 | 45/122 | 90/155 | 110/184 | 67/189 | 98/205 | 132/229 | 110/224 | 139/220 |
| **Empty patches** (of 250 slots) | | | | | | | | | |
| empty patch (among graded) | 16 (16/250) | 128 (127/249) | 95 (95/250) | 66 (66/250) | 61 (60/249) | 45 (44/249) | 21 (21/250) | 26 (26/250) | 30 (25/245) |
| ... 3 h deadline kill | 5 | 9 | 7 | 4 | 16 | 12 | 2 | 9 | 4 |
| ... edited, not committed | 7 | 57 | 65 | 34 | 30 | 20 | 8 | 11 | 9 |
| ... tool call inside think (headless) | 3 | 27 | 16 | 7 | 4 | 4 | 3 | 1 | 4 |
| ... reasoning-only last turn | 1 | 25 | 2 | 1 | 7 | 1 | 0 | 4 | 0 |
| ... malformed tool call | 0 | 3 | 1 | 2 | 2 | 2 | 1 | 0 | 1 |
| ... runaway to the length limit | 0 | 0 | 2 | 18 | 1 | 5 | 7 | 1 | 7 |
| ... context overflow (HTTP 400) | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 2 |
| ... no edit | 0 | 6 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| ... sandbox lost / self-kill | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 2 / 1 |
| (overlapping) stopped after the compaction "continue" | 10 | 79 | 69 | 53 | 35 | 24 | 16 | 13 | 17 |
| **Bad thinking** | | | | | | | | | |
| missing `</think>`: calls (% of calls); rollouts | n/a (>= 12) | n/a (>= 249) | 73 (0.106%); 59 | 60 (0.100%); 47 | 50 (0.059%); 45 | 26 (0.038%); 22 | 37 (0.068%); 25 | 21 (0.032%); 19 | 36 (0.068%); 28 |
| ... tool call rescued from an unclosed think / unclosed, no call (EOS; length) | n/a / 11 (9; 2) | n/a / 89 (82; 7) | 30 / 43 (22; 21) | 9 / 51 (10; 41) | 19 / 31 (14; 17) | 1 / 25 (5; 20) | 1 / 36 (3; 33) | 3 / 18 (6; 12) | 5 / 31 (4; 27) |
| model calls (final attempts) | 61,943 | 72,691 | 68,598 | 59,900 | 84,420 | 68,871 | 54,517 | 65,255 | 52,897 |
| empty outputs (EOS right after `<think>`) | 0 | 25 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| action turns without reasoning | 24/61,039 | 25,302/71,645 (35.3%) | 27,371/67,224 (40.7%) | 21,544/58,326 (36.9%) | 28/82,400 | 2/68,067 | 2/52,584 | 4/63,937 | 4/50,656 |
| rollouts with every action reasoned | 227/250 | 1/250 | 1/250 | 0/250 | 222/249 | 248/250 | 248/250 | 246/250 | 244/248 |
| reasoning with tool-call markup: turns / rollouts (headless turns) | 4/4 (3) | 60/39 (51) | 20/20 (20) | 8/8 (8) | 20/13 (18) | 25/19 (23) | 7/7 (6) | 16/8 (11) | 6/6 (6) |
| reasoning with special tokens: turns / rollouts | 5/4 | 4/3 | 2/2 | 3/3 | 3/3 | 2/2 | 0/0 | 3/3 | 2/2 |
| **Timeouts** | | | | | | | | | |
| 3 h deadline kills (with empty patch) | 5 (5) | 9 (9) | 7 (7) | 4 (4) | 16 (16) | 12 (12) | 3 (2) | 9 (9) | 4 (4) |
| thinking cut at the length limit: calls; rollouts | 2; 2 | 7; 5 | 21; 14 | 41; 33 | 17; 13 | 20; 16 | 33; 22 | 12; 11 | 27; 20 |
| ... ending in a repetition loop | 2; 2 | 7; 5 | 21; 14 | 40; 32 | 15; 13 | 19; 16 | 31; 22 | 11; 11 | 26; 20 |
| ... reaching the full 131,072-token cap | 1; 1 | 0; 0 | 2; 2 | 22; 21 | 1; 1 | 7; 7 | 8; 8 | 1; 1 | 8; 8 |
| rollouts that end on a runaway call | 0 | 0 | 2 | 18 | 1 | 5 | 7 | 1 | 7 |
| **Infra and time** | | | | | | | | | |
| infra slots (excluded from pass@1) | 0 | 1 (context overflow, fatal) | 0 | 0 | 1 (sandbox lost) | 1 (context overflow, fatal) | 0 | 0 | 5 (see Infra slots) |
| pass@1, model-side infra counted as fail | 17.2% | 45/250 = 18.0% | 36.0% | 44.0% | 26.9% | 98/250 = 39.2% | 52.8% | 44.0% | 139/248 = 56.0% |
| rollout wall time: median / mean / p90 (min) | 53.1 / 61.6 / 100.6 | 61.2 / 69.4 / 138.7 | 55.0 / 62.4 / 108.0 | 44.1 / 53.6 / 102.2 | 74.1 / 85.4 / 164.9 | 66.9 / 77.4 / 136.5 | 51.8 / 61.6 / 106.2 | 54.8 / 66.0 / 108.8 | 52.6 / 58.6 / 99.3 |
| rollouts over 2 h | 11 | 33 | 20 | 15 | 61 | 42 | 18 | 17 | 15 |

How the rows are measured:
- **pass@1** = passes / graded rollouts, with infra slots excluded. The CI is a 95% task bootstrap.
- **Empty-patch causes** come from the DeepSWE classifier, run on each slot's final attempt. They cover all 250 slots, including infra slots.
- **Missing `</think>`** joins the counting parser's log lines to the final-attempt calls.
  - It is n/a for base and raw 5 ep, which ran on the plain lenient parser. The bracketed numbers there are lower bounds.
- **Length-limit cuts** include calls that the router's context clamp stopped before 131,072 tokens. A repetition loop means the last 20,000 characters of the reasoning compress below 0.15 with zlib.
- **Deadline kill** means the session did not finish, was not sandbox-lost (N4), and either has agent_error TimeoutError or ran for at least 10,500 s.
  - In all nine runs this picks exactly the slots that ATP flags `budget_timeout` (5/9/7/4/16/12/3/9/4).
  - Sessions that finished on their own after 10,500 s are not counted. There are 7, at 10,554–10,740 s: 1 each in raw 5, 10 and 20 ep, 2 in etp 5 ep and 2 in etp 20 ep. The v3 runs have none.

### Paired tests over the 50 tasks (`analysis/accuracy/paired.md`)

| B vs A | pass@1 A → B | B − A [95% CI] | sign-flip p | tasks better/equal/worse | pass@5 A → B (McNemar p) | patch-rate part | resolve-rate part |
|---|---|---|---|---|---|---|---|
| v3 etp 5 ep vs etp 5 ep (same epochs) | 67/249 → 110/250 | +17.1 [+9.9, +24.4] | <1e-4 | 29/13/8 | 34 → 41 (0.09) | +5.8 [+3.3, +8.7] | +11.3 [+4.6, +18.4] |
| v3 etp 5 ep vs etp 10 ep (~same steps, 71 vs 76) | 98/249 → 110/250 | +4.6 [−2.2, +11.6] | 0.23 | 24/9/17 | 40 → 41 (1.0) | +3.5 [+0.9, +6.5] | +1.1 [−5.8, +7.9] |
| v3 etp 10 ep vs etp 10 ep (same epochs) | 98/249 → 139/245 | +17.4 [+11.9, +23.2] | <1e-4 | 29/18/3 | 40 → 45 (0.13) | +4.1 [+1.1, +7.7] | +13.2 [+8.5, +18.0] |
| v3 etp 10 ep vs etp 20 ep (~same steps, 142 vs 152) | 132/250 → 139/245 | +3.9 [−2.9, +10.8] | 0.29 | 17/21/12 | 42 → 45 (0.38) | −1.1 [−3.6, +1.5] | +5.0 [−1.9, +11.8] |
| v3 etp 10 ep vs v3 etp 5 ep | 110/250 → 139/245 | +12.7 [+6.1, +19.5] | 0.0008 | 27/15/8 | 41 → 45 (0.22) | +0.1 [−2.5, +2.8] | +12.6 [+6.5, +18.9] |
| v3 etp 5 ep vs base | 43/250 → 110/250 | +26.8 [+18.8, +35.2] | <1e-4 | 35/13/2 | 23 → 41 (<1e-4) | −1.3 [−3.1, +0.3] | +28.1 [+20.5, +36.1] |
| v3 etp 10 ep vs base | 43/250 → 139/245 | +39.5 [+31.5, +48.0] | <1e-4 | 40/10/0 | 23 → 45 (<1e-4) | −1.6 [−3.4, +0.2] | +41.1 [+33.1, +49.5] |

- **At the same epochs, v3 is +17 pp at both 5 and 10 epochs.**
  - Most of the gain is in the resolve rate: patched rollouts resolve at 49.1% / 63.2%, vs 35.4% / 47.8% for v2 etp.
- **At about the same optimizer steps, the difference is within noise:** +4.6 vs etp 10 ep, +3.9 vs etp 20 ep.
  - On these tasks the extra data acts mostly like extra epochs.
  - v3 5 ep still patches more often than etp 10 ep: patch-rate part +3.5 [+0.9, +6.5]; 26 vs 45 empty patches.
- **v3 10 vs 5 epochs is +12.7 pp (p 0.0008), all through the resolve rate.** The patched share is unchanged (89.6% / 89.8%).
- **Counting the model-side infra slots as fails changes no conclusion** (`analysis/v3etp/sensitivity_infra.json`).
  - Slots counted: v3 etp 10 ep's 2 context overflows and 1 self-kill, and etp 10 ep's 1 overflow.
  - Results: v3 10 ep vs etp 10 ep +16.8; vs etp 20 ep +3.2 [−3.8, +10.2] (p 0.41); vs v3 5 ep +12.0 (p 0.002).
- **Tasks v3 etp 10 ep never solves:** 5 tasks: 3 gomacro, aiohttp-c04 and pylama-c02-r1b0-r2b2.
  - 3 of them are unsolved by all nine runs.
  - gomacro-c01 was solved once each by etp 10 ep and etp 20 ep, but SFT-v3 has no conversation for it.

### Bad thinking, timeouts, speed
- **Reasoning before every action holds.**
  - Only 4 action turns are unreasoned, out of 63,937 (5 ep) and 50,656 (10 ep).
  - Every action is reasoned in 246/250 and 244/248 rollouts.
  - Reasoning stays base-like: median 130 / 134 tokens per reasoning turn, KS distance to base 0.053 / 0.055.
  - Verbatim repeats are 0.00 / 0.01%, and generic short reasoning is 0.00%.
- **Missing `</think>` is 0.032% / 0.068% of calls.** The v2 etp range runs from 0.038% (etp 10 ep, 26/68,871) to 0.068% (etp 20 ep, 37/54,517).
  - 5 ep (21/65,255) is below that range.
  - 10 ep (36/52,897) is at its top.
  - By kind: 3 / 5 rescued tool calls, 6 / 4 EOS inside think, 12 / 27 length cuts.
  - Empty outputs: 0 / 0.
  - Tool-call markup inside reasoning: 16 turns in 8 rollouts / 6 turns in 6 rollouts. Special tokens: 3 / 2 turns.
- **Runaway thinking at 10 epochs looks like etp 20 ep.**
  - There are 27 length cuts in 20 rollouts (etp 20 ep: 33 in 22).
  - 26 of the 27 end in a repetition loop: 23 on "hmm — hmm", 2 on "AH" / "AH WAIT!!!" and 1 on a zero-width-space "|" pattern. For comparison, 29 of etp 20 ep's 31 loops are "hmm".
  - The 27th cut (zsa-c02-r1b1 r1, 8,713 tokens) is just above the loop threshold (zlib 0.21), though its tail is also mostly "hmm".
  - 8 calls reached the full 131,072-token cap, and 7 rollouts end on a runaway call (etp 20 ep: 7).
    - Of these 7, 6 end on a "hmm" loop and 1 (jest-c05 r3) on a 131,072-token "AH WAIT!!!" loop.
    - etp 20 ep's 7 split the same way: 6 "hmm", 1 "hold on".
  - Unlike etp 20 ep, the model still escapes most loops: 68 escaped loop turns in 10 rollouts (etp 20 ep: 1).
  - At 5 ep there are 12 cuts in 11 rollouts, and 1 rollout ends on one.
- **3 h deadline kills fell to 9 / 4** (etp 16 / 12 / 3; base 5).
  - The 9 at 5 ep are 7 flagged TimeoutError plus 2 unfinished sessions without the flag, stopped at 10,777 and 10,788 s.
- **Rollouts take about as long as base.**
  - Median wall time is 54.8 / 52.6 min (base 53.1; etp 74.1 / 66.9 / 51.8).
  - Against etp of the same epochs, the mean drops by 19.4 [14.3, 24.9] and 18.8 [13.9, 23.9] min, with 22–24% fewer model calls.
  - Serving load was base-like: 7.3 / 7.1 running sequences per worker and 71 / 70 tok/s per sequence (etp 10 ep: 8.1 and 61).
  - v3 10 ep spent 16.4 fewer model minutes than etp 10 ep. About 10.6 of those come from less output and about 5.8 from faster serving.
- **Empty patches: 26/250 and 30/250** (25/245 among graded rollouts).
  - 5 ep: 9 deadline kills, 11 uncommitted, 4 reasoning-only, 1 headless call, 1 runaway.
  - 10 ep: 7 runaway, 9 uncommitted, 4 deadline kills, 4 headless calls, 1 malformed, plus the 5 infra rows.
  - Stops right after the compaction "continue": 13 / 17 (etp 35 / 24 / 16; base 10).
  - Git commit rate: 89.6 / 88.7% of rollouts.

### Infra slots
- v3 etp 5 ep: none.
- v3 etp 10 ep: 5 slots. The pipeline's rule excludes all of them from pass@1 because their ATP state is not "completed".
  - **2 fatal (model-side): syn-wasmtime-cand-c04-r1b1-bbb510ca r0 and r1.**
    - Neither session compacted.
    - Their last successful calls had prompts of 231,188 and 238,510 tokens. The router had already cut those calls' output budget to 262,144 minus the prompt (30,956 tokens for the 231,188-token call).
    - Then one `cargo test … | tail` result added 46,034 / 51,381 tokens in a single step. Its few lines were array dumps from a failing coredump assertion, 46k / 51k characters long.
    - The next prompt, 277,222 tokens (r0, 02:12:51) and 289,891 tokens (r1, 02:01:43), was over the 262,144-token window on its own. vLLM returned HTTP 400.
    - The router's clamp could not apply because the remaining budget was negative (−15,078 / −27,747).
    - Both ended with an empty patch, f2p 0/14.
    - The infra slots of raw 5 ep and etp 10 ep failed the same way: their next prompts were 269,887 and 269,992 tokens, after 240,324 and 240,743.
  - **1 needs_review (model-side): syn-gomacro-cand-c02-r1b0-e8804dd9 r4.**
    - `go run .` in /app started the gomacro REPL. The model's `pkill -f gomacro` then killed its own OpenCode process (exit −1) at 2 min.
  - **1 needs_review (true infra): syn-wasmtime-cand-c03-353ec720 r1.** The sandbox PTY was lost at about 2.2 h, and patch collection timed out.
  - **1 retryable (true infra): syn-ibis-cand-c03-r1b4-3d3b86b5 r3.** The sandbox was OOM-killed 16 min in and the slot was never retried (see Closing decisions).
- Counting the 3 model-side slots as fails gives 139/248 = 56.0%; counting all 5 gives 139/250 = 55.6%.
  - Under the same rule, raw 5 ep is 45/250 = 18.0% and etp 10 ep is 98/250 = 39.2%. The other runs have no model-side infra.

### Closing decisions and serving
- **v3 etp 10 ep group 0 was closed with one slot unfinished.**
  - The slot is syn-ibis-cand-c03-r1b4-3d3b86b5 rep 3, attempt 0, started 01:13. Its sandbox was OOM-killed (137) 16 min in.
  - ATP recorded the failure only at the slot's 3 h deadline (04:13–04:16) and set it retryable. The launch then ended, and the driver exited at 04:22 with "probe incomplete".
  - Main's rerun driver (21107074, from 07:13) waited for 8 routable workers.
    - Only 2 existed, and both were due to stop at 09:04. New batch workers were estimated after 12:40.
    - The 4 sibling ibis reps had taken 68–181 min.
  - Main cancelled the rerun at 07:29:59. It then wrote `results/v3etp10-group-0.json` with `analysis/finalize_group_v3etp10_g0.py`.
    - The file was written at 07:30:28 (both finished_at and the file time).
    - The finalize script's docstring says "07:4x" and the workflow notes say "~07:45"; both are wrong. That script is main's provenance record and was left unedited. The same wrong time ("07:45") in `compare_v3etp.py` is now corrected.
    - The records come from `rollouts.probe_records()`, the synth pipeline's own rule: any slot that is not "completed" counts as infra.
    - They are compacted the same way driver.py does it, and carry a `finalized_by` note.
  - Group 0 ended with 33 completed slots (21 pass, 12 fail) and 2 infra (1 needs_review, 1 retryable).
- **v3 etp 5 ep's last slot ran into the 3 h deadline twice.** This is group 0, syn-wasmtime-cand-c03 rep 0.
  - Attempt 0 (00:49–03:50) could not be finalized ("Cannot finalize a candidate before the agent process has stopped"), so ATP retried it.
  - Attempt 1 (from 04:21) was killed at about 07:21 and graded fail with an empty patch. The results file was written at 07:26.
  - The other 249 slots had finished by 04:29.
- **Workers:**
  - Leg-1 workers (10 per model) were ready at 00:48, drained from 03:55 and were gone by 04:40:22–28.
  - Main's tail workers (4 per model, submitted 02:14) were ready at 04:41:32–52.
  - Leg-2 workers were ready at 05:11–05:14: 10 for v3etp5 and 9 for v3etp10. One v3etp10 leg-2 worker failed vLLM startup on pool0-01888, which is now excluded.
  - After 04:41 these 27 workers served only 3 rollouts: the v3etp5 slot above, and two retried v3etp10 group-3 slots that ended at 05:10 and 05:19.
  - Cancellations, from the Slurm "CANCELLED AT" lines:
    - Main cancelled 23 workers at 07:13:02.
    - The finish-workflow watcher (wf_dd5aa996-0e7, its rule d) cancelled the last two v3etp5 workers, 21094268 and 21094276, at 07:26:45, after the last v3etp5 slot had ended.
    - Main cancelled the last two v3etp10 workers, 21094309 and 21094313, together with the rerun driver 21107074 at 07:29:59.
  - From ready to cancellation the 27 workers held 59.5 node-hours, almost all of it idle. This had no effect on any outcome.
- **No rollout outcome was affected by a serving gap** (`analysis/v3etp/serving_gaps.json`, from `routers/<model>-group-*/student-router-requests.jsonl`).
  - There were 65,968 / 53,119 routed calls and no 429s.
  - The only gap was the ~70 s handover at 04:40–04:41, when no worker was routable:
    - 1 / 2 calls waited: 69 s, and 41 and 70 s.
    - 3 / 1 router requests returned 502, each after 4 attempts on workers that were leaving. For v3etp5 these were HTTP attempts 1–3 of one model call; for v3etp10, attempt 1 of one call.
  - **The harness's HTTP client retried both calls in place, not OpenCode.** In http.jsonl each retry has the same call_id and model_call_id, plus `retry` events.
    - After router holds of 69 s and 70 s, the v3etp5 call got 200 on HTTP attempt 4 at 04:41:48, and the v3etp10 call on attempt 2 at 04:42:14.
    - Both sessions continued. The v3etp5 one ran to its deadline; the v3etp10 one (vscode-sqltools-c02-r1b0 r1) ended on a runaway length cut at 05:19.
  - The two HTTP 400s are the context-overflow slots.
  - Among the earlier runs, only raw 5 ep had a gap: 24 calls held for about 28 min at 20:04 on 10-24.
- **Live table** (07:46; values are 5 ep / 10 ep):
  - pass@1 44.0 / 56.7%.
  - pass@5 82.0 / 91.3%, over the 50 / 46 tasks whose 5 slots are all graded.
  - Empty patch 10.4 / 10.2% of graded rollouts.
  - Timeouts 2.8 / 1.6%, from the TimeoutError flag only (7 / 4).
  - Missing `</think>` 0.033 / 0.062%, from the parser counters. The final-attempt join gives 0.032 / 0.068%.
  - Every action reasoned in 98.4 / 98.4% of rollouts.
  - It shows v3 etp 10 ep at 249/250 slots finished because ATP never finished the retryable slot.

### Conclusions
- **Against v2 etp at the same epochs, SFT-v3 adds about 17 pp pass@1 at both 5 and 10 epochs.** It also has:
  - fewer empty patches: 26 vs 61 at 5 ep, and 25 vs 44 graded at 10 ep;
  - fewer deadline kills: 9 vs 16, and 4 vs 12;
  - rollouts about 19 min shorter.
- **Per optimizer step it is level with v2 etp: +4.6 and +3.9 pp, not significant.**
  - SFT-v3 has about 1.8x the trajectories per task here, so one epoch is about 1.9x the steps.
- **v3 etp 10 ep is the best of the nine runs on these training tasks:** 56.7% pass@1, pass@5 45/50, 9 tasks at 5/5. It is not significantly above etp 20 ep.
- **The thinking fix holds.** What remains at 10 epochs is repetition loops running to the length limit, mostly "hmm — hmm". 7 rollouts end on one (6 "hmm", 1 "AH WAIT!!!"), as in etp 20 ep (7: 6 "hmm", 1 "hold on").
- **Caveat: these are training tasks.** On held-out DeepSWE (analyzed separately), v3 etp 5 / 10 ep reach 17.35% / 23.54%, vs 22.12% / 24.42% for etp 10 / 20 ep and 27.79% for base.


<!-- Section written by the main agent from workflow wf_90eb6733-08a (analyze-trainacc; two independent verifiers + reconcile). Tables and scripts: analysis/v3etp20/. -->
## SFT-v3 empty-think-prefix, 20 epochs (v3etp20, 2026-11-04)

Checkpoint: super-kimi-sft-v3-empty-think-prefix-wsd-20ep-16n-ep32-20261029 iter284 ("v3 etp 20 ep"). Its tokenizer, tokenizer config, chat template and config.json are byte-identical (md5) to those of v3 etp 10 ep and base sc14.

- **Corpus and steps:**
  - Same SFT-v3 corpus as v3 etp 5 / 10 ep. 20 epochs = 284 iterations, 1.87x v2 etp 20 ep (152).
  - The same-epochs pair is v2 etp 20 ep. No v2 run is near 284 iterations, so the same-steps pair stays v3 etp 10 ep (142) vs v2 etp 20 ep (152), as in the previous section.
- **Setup:**
  - Same 50 tasks x 5, harness, graders and counting lenient-think parser.
  - All 24 vLLM command files match v3 etp 10 ep's arguments except model path and seed. The harness diff is the model entry plus 2 more excluded nodes.
  - Campaigns ran 01:59–06:39, and all 8 drivers exited 0.
- **Outcome:** 250/250 graded, 0 infra, 136 passes (54.4%). ATP re-ran 2 slots whose first attempt lost its sandbox (see Infra and serving).
- **Files** (`analysis/v3etp20/`; `reproduce.sh` lists every command):
  - **Copied tools with a v3etp20 entry:** `accuracy/paired.{md,json}`, `empty-patch/table.{md,json}` (+ classifier rows `empty-patch/v3etp20/`), `think/{split_table, notool_kinds, reasoning_quality}`, `features/comparison.{md,json}`, `cost/cost.{md,json}`, `compare/{comparison_v3etp20, serving_gaps, sensitivity_infra, thinkx_ta}`.
  - **Reproduction checks:** every copied tool gives the same outputs as before for the earlier models.
    - Classifier re-run on v3etp10: 0 rows differ. Extractor re-run on v3etp10: byte-identical.
    - All earlier columns or pairs match: split_table, notool_kinds, empty-patch table, run_compare, paired (22 pairs), cost, thinkx, serving_gaps, sensitivity_infra, comparison table.
  - **New outputs:**
    - `think/runaway_units.txt`, `compare/slots_v3etp20.txt`, `compare/stats_tests.txt`.
    - `degenerate/`: copies of the endings, text and format lenses of degenerate-behaviors-20261031. Their sc14 / etp10 / etp20 / v3etp5 / v3etp10 outputs are identical to the published ones. Also `compaction_v3etp20.txt` and a DeepSWE cross-check.
    - `coverage/coverage_v3etp20.txt`, `live/live_metrics_final.txt`.
  - **Reconcile re-checks** (`reconcile/`; commands in `reconcile/reproduce_reconcile.sh`, outputs in `reconcile/out/`): my own scans of the raw session exports, HTTP journals, router logs and DeepSWE runs for every point the two verifiers raised (see Verification changes).

### Comparison (`analysis/v3etp20/compare/comparison_v3etp20.txt`)

| | base | raw 5 ep | raw 10 ep | raw 20 ep | etp 5 ep | etp 10 ep | etp 20 ep | v3 etp 5 ep | v3 etp 10 ep | v3 etp 20 ep |
|---|---|---|---|---|---|---|---|---|---|---|
| **Accuracy** | | | | | | | | | | |
| pass@1 [95% CI] | 17.2% (43/250) [10.8, 24.4] | 18.1% (45/249) [12.1, 24.5] | 36.0% (90/250) [28.0, 44.4] | 44.0% (110/250) [34.4, 53.6] | 26.9% (67/249) [20.5, 34.0] | 39.4% (98/249) [31.3, 47.2] | 52.8% (132/250) [43.6, 61.6] | 44.0% (110/250) [35.6, 52.8] | 56.7% (139/245) [47.8, 65.6] | 54.4% (136/250) [46.0, 62.8] |
| pass@5 | 23/50 | 26/50 | 37/50 | 37/50 | 34/50 | 40/50 | 42/50 | 41/50 | 45/50 | 46/50 |
| tasks 5/5 | 0 | 0 | 2 | 4 | 0 | 2 | 7 | 5 | 9 | 5 |
| resolved / graded rollouts with a patch | 43/234 | 45/122 | 90/155 | 110/184 | 67/189 | 98/205 | 132/229 | 110/224 | 139/220 | 136/214 |
| **Empty patches** (of 250 slots) | | | | | | | | | | |
| empty patch (among graded) | 16 (16/250) | 128 (127/249) | 95 (95/250) | 66 (66/250) | 61 (60/249) | 45 (44/249) | 21 (21/250) | 26 (26/250) | 30 (25/245) | 36 (36/250) |
| ... 3 h deadline kill | 5 | 9 | 7 | 4 | 16 | 12 | 2 | 9 | 4 | 7 |
| ... edited, not committed | 7 | 57 | 65 | 34 | 30 | 20 | 8 | 11 | 9 | 16 |
| ... tool call inside think (headless) | 3 | 27 | 16 | 7 | 4 | 4 | 3 | 1 | 4 | 1 |
| ... reasoning-only last turn | 1 | 25 | 2 | 1 | 7 | 1 | 0 | 4 | 0 | 1 |
| ... malformed tool call | 0 | 3 | 1 | 2 | 2 | 2 | 1 | 0 | 1 | 3 |
| ... runaway to the length limit | 0 | 0 | 2 | 18 | 1 | 5 | 7 | 1 | 7 | 8 |
| ... context overflow (HTTP 400) | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 2 | 0 |
| ... no edit | 0 | 6 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ... sandbox lost / self-kill | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 2 / 1 | 0 / 0 |
| (overlapping) stopped after the compaction "continue" | 10 | 79 | 69 | 53 | 35 | 24 | 16 | 13 | 17 | 27 |
| **Bad thinking** | | | | | | | | | | |
| missing `</think>`: calls (% of calls); rollouts | n/a (>= 12) | n/a (>= 249) | 73 (0.106%); 59 | 60 (0.100%); 47 | 50 (0.059%); 45 | 26 (0.038%); 22 | 37 (0.068%); 25 | 21 (0.032%); 19 | 36 (0.068%); 28 | 36 (0.065%); 28 |
| ... rescued tool call / unclosed, no call (EOS; length) | n/a / 11 (9; 2) | n/a / 89 (82; 7) | 30 / 43 (22; 21) | 9 / 51 (10; 41) | 19 / 31 (14; 17) | 1 / 25 (5; 20) | 1 / 36 (3; 33) | 3 / 18 (6; 12) | 5 / 31 (4; 27) | 8 / 28 (3; 25) |
| model calls (final attempts) | 61,943 | 72,691 | 68,598 | 59,900 | 84,420 | 68,871 | 54,517 | 65,255 | 52,897 | 55,623 |
| empty outputs (EOS right after `<think>`) | 0 | 25 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| action turns without reasoning | 24/61,039 | 25,302/71,645 (35.3%) | 27,371/67,224 (40.7%) | 21,544/58,326 (36.9%) | 28/82,400 | 2/68,067 | 2/52,584 | 4/63,937 | 4/50,656 | 12/53,848 |
| rollouts with every action reasoned | 227/250 | 1/250 | 1/250 | 0/250 | 222/249 | 248/250 | 248/250 | 246/250 | 244/248 | 238/250 |
| reasoning with tool-call markup: turns / rollouts (headless turns) | 4/4 (3) | 60/39 (51) | 20/20 (20) | 8/8 (8) | 20/13 (18) | 25/19 (23) | 7/7 (6) | 16/8 (11) | 6/6 (6) | 5/4 (5) |
| reasoning with special tokens: turns / rollouts | 5/4 | 4/3 | 2/2 | 3/3 | 3/3 | 2/2 | 0/0 | 3/3 | 2/2 | 1/1 |
| **Timeouts** | | | | | | | | | | |
| 3 h deadline kills (with empty patch) | 5 (5) | 9 (9) | 7 (7) | 4 (4) | 16 (16) | 12 (12) | 3 (2) | 9 (9) | 4 (4) | 7 (7) |
| thinking cut at the length limit: calls; rollouts | 2; 2 | 7; 5 | 21; 14 | 41; 33 | 17; 13 | 20; 16 | 33; 22 | 12; 11 | 27; 20 | 25; 19 |
| ... ending in a repetition loop | 2; 2 | 7; 5 | 21; 14 | 40; 32 | 15; 13 | 19; 16 | 31; 22 | 11; 11 | 26; 20 | 24; 18 |
| ... reaching the full 131,072-token cap | 1; 1 | 0; 0 | 2; 2 | 22; 21 | 1; 1 | 7; 7 | 8; 8 | 1; 1 | 8; 8 | 10; 10 |
| rollouts that end on a runaway call | 0 | 0 | 2 | 18 | 1 | 5 | 7 | 1 | 7 | 8 |
| **Infra and time** | | | | | | | | | | |
| infra slots (excluded from pass@1) | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 5 | 0 |
| pass@1, model-side infra as fail | 17.2% | 18.0% | 36.0% | 44.0% | 26.9% | 39.2% | 52.8% | 44.0% | 56.0% | 54.4% |
| rollout wall time: median / mean / p90 (min) | 53.1 / 61.6 / 100.6 | 61.2 / 69.4 / 138.7 | 55.0 / 62.4 / 108.0 | 44.1 / 53.6 / 102.2 | 74.1 / 85.4 / 164.9 | 66.9 / 77.4 / 136.5 | 51.8 / 61.6 / 106.2 | 54.8 / 66.0 / 108.8 | 52.6 / 58.6 / 99.3 | 52.0 / 61.7 / 117.0 |
| rollouts over 2 h | 11 | 33 | 20 | 15 | 61 | 42 | 18 | 17 | 15 | 24 |

How the rows are measured: the same scripts and definitions as in the SFT-v3 5 / 10 ep section.
- The deadline-kill rule picks exactly ATP's 7 `budget_timeout` slots (`compare/slots_v3etp20.py`; checked against generation-validity).
- One v3 etp 20 ep session ended on its own after 10,500 s and is not counted: ibis-c05-r1b3 r0, 10,542 s.
- Rows from the session export (every action reasoned, markup, special tokens) cover rollouts that have an export. For v3 etp 10 ep that is 248; its 2 true-infra slots have none.
- "Headless" markup turns contain tool-call markup without a `<tool_call>` opener. Most of them are quoted markup, not calls (see Bad thinking).

### Paired tests over the 50 tasks (`analysis/v3etp20/accuracy/paired.md`)

| B vs A | pass@1 A → B | B − A [95% CI] | sign-flip p | tasks better/equal/worse | pass@5 A → B (McNemar p) | patch-rate part | resolve-rate part |
|---|---|---|---|---|---|---|---|
| v3 etp 20 ep vs v3 etp 10 ep | 139/245 → 136/250 | −2.3 [−9.7, +4.9] | 0.58 | 15/17/18 | 45 → 46 (1.0) | −2.7 [−6.3, +0.8] | +0.3 [−6.4, +7.1] |
| v3 etp 20 ep vs v3 etp 5 ep | 110/250 → 136/250 | +10.4 [+3.2, +17.6] | 0.009 | 24/16/10 | 41 → 46 (0.13) | −2.3 [−5.2, +0.4] | +12.7 [+6.5, +18.7] |
| v3 etp 20 ep vs etp 20 ep (same epochs; 284 vs 152 steps) | 132/250 → 136/250 | +1.6 [−5.6, +8.8] | 0.75 | 19/16/15 | 42 → 46 (0.29) | −3.6 [−7.1, −0.4] | +5.2 [−1.3, +11.9] |
| v3 etp 20 ep vs base | 43/250 → 136/250 | +37.2 [+30.0, +44.8] | <1e-4 | 40/10/0 | 23 → 46 (<1e-4) | −3.3 [−6.0, −0.9] | +40.5 [+33.4, +47.8] |
| v3 etp 20 ep vs etp 10 ep | 98/249 → 136/250 | +15.0 [+7.4, +22.7] | 0.0005 | 29/12/9 | 40 → 46 (0.11) | +1.8 [−1.1, +5.0] | +13.2 [+6.0, +20.4] |
| v3 etp 20 ep vs etp 5 ep | 67/249 → 136/250 | +27.5 [+19.9, +35.0] | <1e-4 | 38/7/5 | 34 → 46 (0.002) | +4.8 [+2.0, +7.8] | +22.7 [+15.0, +30.1] |
| v3 etp 20 ep vs raw 20 ep | 110/250 → 136/250 | +10.4 [+2.4, +18.8] | 0.023 | 26/14/10 | 37 → 46 (0.012) | +7.4 [+3.3, +11.6] | +3.0 [−4.7, +11.1] |
| v3 etp 20 ep vs raw 10 ep | 90/250 → 136/250 | +18.4 [+11.6, +24.8] | <1e-4 | 32/12/6 | 37 → 46 (0.012) | +14.4 [+10.5, +18.3] | +4.0 [−2.4, +10.5] |
| v3 etp 20 ep vs raw 5 ep | 45/249 → 136/250 | +36.3 [+28.4, +44.0] | <1e-4 | 41/7/2 | 26 → 46 (<1e-4) | +18.4 [+14.2, +22.8] | +17.9 [+9.9, +25.9] |

- **20 vs 10 epochs on SFT-v3: −2.3 pp, no gain.**
  - The patched share fell from 89.8% to 85.6%.
  - Patched rollouts resolve at the same rate: 63.2% → 63.6%.
  - 15 tasks got better and 18 got worse.
- **20 vs 5 epochs: +10.4 pp (p 0.009), all of it in the resolve rate.**
- **Same epochs on the v2 corpus: +1.6 pp, n.s.**
  - At 5 and 10 epochs SFT-v3 was +17 pp ahead of v2 etp. At 20 epochs that lead is gone: v2 etp climbs 39.4 → 52.8% while v3 goes 56.7 → 54.4%.
  - The patch-rate part is significantly negative (−3.6 [−7.1, −0.4]): v3 etp 20 ep leaves more empty patches than v2 etp 20 ep.
- **By optimizer step:**
  - v2 etp at 38 / 76 / 152 steps: 26.9 / 39.4 / 52.8%. v3 etp at 71 / 142 / 284 steps: 44.0 / 56.7 / 54.4%.
  - Up to ~150 steps the two corpora track each other (v3 at 142 vs v2 at 152: +3.9, n.s., previous section). Doubling v3 to 284 steps adds nothing.
- **pass@5 46/50.**
  - New: gomacro-c02-r1b0 at 1/5, the first pass on this task by any of the ten runs; and pylama-c02-r1b0-r2b2 at 1/5.
  - Lost: vscode-sqltools-c02-r1b3, 2/5 → 0/5.
- **Unsolved tasks:**
  - v3 etp 20 ep never solves 4 tasks: aiohttp-c04, gomacro-c01, gomacro-c02-r1b3 and sqltools-c02-r1b3. The two gomacro tasks are the only two with no SFT-v3 conversation.
  - Unsolved by all ten runs: aiohttp-c04 and gomacro-c02-r1b3.
- **Counting v3 etp 10 ep's 3 model-side infra slots as fails** gives −1.6 [−8.9, +5.6], p 0.70 (`compare/sensitivity_infra.json`). v3 etp 20 ep has no infra slot.

### Empty patches: causes, and why they rose (`empty-patch/table.md`, `degenerate/out/compaction_v3etp20.txt`, `degenerate/endings/out/tables.txt`, `compare/slots_v3etp20.txt`, `compare/stats_tests.txt`, `reconcile/out/`)

| of 250 rollouts | base | etp 20 ep | v3 etp 5 ep | v3 etp 10 ep | v3 etp 20 ep |
|---|---|---|---|---|---|
| empty patch | 16 | 21 | 26 | 30 (25 graded) | 36 |
| ... after a broken compaction summary (tool call or length-cut) | 2 | 9 | 10 | 9 | 28 |
| ... all other empty patches | 14 | 12 | 16 | 21 | 8 |
| ending: uncommitted (asks or defers to commit / claims done / other) | 7 (5/2/0) | 8 (5/3/0) | 11 (6/4/1) | 9 (4/5/0) | 16 (10/6/0) |
| ending: 3 h deadline kill / runaway / format (headless, reasoning-only, malformed) | 5 / 0 / 4 | 2 / 7 / 4 | 9 / 1 / 5 | 4 / 7 / 5 | 7 / 8 / 5 |
| ending: infra, context overflow, self-kill | 0 | 0 | 0 | 5 | 0 |
| empty among compacted / not compacted rollouts | 15/145, 1/105 | 18/127, 3/123 | 22/149, 4/101 | 21/131, 9/119 | 34/125, 2/125 |
| compaction summaries: tool call instead of the summary | 2/206 (1.0%) | 10/177 (5.6%) | 21/218 (9.6%) | 10/174 (5.7%) | 44/194 (22.7%) |
| ... on the session's first compaction | 0/145 | 3/127 | 6/149 | 5/131 | 23/125 |
| rollouts with >= 1 tool-call summary | 2 | 10 | 21 | 9 | 38 |
| summaries still carrying the commit instruction | 96.1% | 91.5% | 90.4% | 90.8% | 72.2% |
| compacted sessions that lost the commit instruction (via a tool-call summary) | 8 (2) | 12 (8) | 18 (17) | 13 (9) | 39 (34) |
| uncommitted: after a loss / never lost | 5/8, 2/137 | 7/12, 1/115 | 5/18, 5/131 | 7/13, 2/118 | 15/39, 1/86 |

- **On these tasks the rise is modest and not significant:** 36 vs 30 empty patches (graded: 36/250 vs 25/245).
  - Paired by task: +2.4 pp [−3.2, +8.0], p 0.48. On graded slots: +4.2 [−1.4, +9.8], p 0.17.
  - vs etp 20 ep: +6.0 [+0.4, +12.0], p 0.055.
  - Changes vs v3 etp 10 ep: uncommitted +7, deadline kills +3, runaway +1. The 5 infra / overflow / self-kill rows of v3 etp 10 ep are gone.
- **Why they rose: compaction breakage.**
  - When OpenCode asks for the compaction summary, v3 etp 20 ep writes a tool call in 22.7% of summaries, vs 5.7% for v3 etp 10 ep. Rollouts with >= 1 such summary: 38 vs 9, +11.6 pp [+5.6, +18.0], p 0.0006 (Fisher on summaries p = 3e-6). The HTTP journals of v3 etp 10 ep's 2 no-export slots contain no tool-call summary, so 9/250 is exact.
  - It happens already on the first compaction: 18.4% vs 3.8% (p 0.0002).
  - 42 of the 44 tool-call summaries are a plain continuation such as "Let me inspect the current state of the test file: `<tool_call>`...". They have no `## Goal`, no task and no commit instruction. The other 2 (zsa-c02-r1b1 r2, gomacro-c02-r1b3 r3) first write a full `## Goal` summary, commit instruction included, and then append a tool call.
  - 39 of 125 compacted sessions lose the commit instruction, 34 of them through a tool-call summary (v3 etp 10 ep: 13/131, p 2e-5).
  - 15 of those 39 end edited but uncommitted ("Shall I commit?", "Done ..."), vs 1 of the 86 sessions that kept the instruction.
  - 28 of the 36 empty patches follow a broken summary (v3 etp 10 ep: 9 of 30). That includes 15 of 16 uncommitted endings, 6 of 7 deadline kills and 5 of 8 runaway endings.
  - Only 8 empty patches have no compaction breakage, the fewest of any etp run.
  - The classifier's own measure, run independently, agrees: broken summaries (no `## Goal` or tool-call markup) 50 in 39 rollouts vs 15 in 13.
- **What compaction breakage costs (association, not cause):**
  - Compacted rollouts with a broken summary resolve 4/38. Those whose summaries were fine resolve 46/87 (52.9%), the same as v3 etp 10 ep's fine-summary rollouts (60/119, 50.4%).
  - The gap holds within tasks. Over the 19 tasks that have both kinds, tool-call-summary rollouts resolve 4/37 and fine-summary rollouts 28/50 (Mantel-Haenszel odds ratio 0.21; `reconcile/out/within_task.txt`). Session length is still a confounder: sessions that compact more often have more chances to write a bad summary.
  - The 4 largest per-task losses vs v3 etp 10 ep all gained tool-call summaries: jest-c05 0→2, risor-c05 0→2, transitions-c03 0→3, wasmtime-c03 1→3.
  - The rise is not confined to the tasks that got worse (`reconcile/out/analyze.txt`, C2). Rollouts with a tool-call summary went:
    - 3 → 16 over the 18 tasks that got worse;
    - 2 → 16 over the 17 unchanged tasks, 8 of them on the 3 tasks that neither checkpoint solves;
    - 4 → 6 over the 15 tasks that got better.
- **The same mechanism explains most of the DeepSWE rise (159 vs 106 empty patches), but not all of it.**
  - Sources: the endings lens run read-only on the DeepSWE v3etp20 classifier rows (`degenerate/endings-ds-crosscheck/out/ds_crosscheck.txt`; v3etp10's session file is byte-identical to the published one), and a recount from the raw runs (`reconcile/out/deepswe.txt`).
  - Tool-call summaries: 133/712 (18.7%) vs 45/593 (7.6%), p 4e-9. On the first compaction: 52/423 vs 16/372.
  - Compacted sessions that lost the commit instruction: 121/423 vs 51/372.
  - The empty patches rose by 53:
    - +33 uncommitted endings (51 vs 18, p 5e-5). 50 of the 51 came after a loss of the commit instruction.
    - +11 deadline kills (48 vs 37 with an empty patch). 31 of the 48 had a tool-call or length-cut summary, vs 19 of the 37. Kills without a broken summary are level (17 vs 18).
    - +10 format endings (24 vs 14: tool call inside think, reasoning-only, malformed). Only 5 of the 24 follow a broken summary, and these training tasks show no such rise (5 vs 5), so this part is a separate change.
    - The rest: −1 self-kill, −1 context overflow, +1 commit off HEAD.
  - DeepSWE sessions compact more often (423/565 vs 125/250 here), which is why the same failure costs more there.
- **Why at 20 epochs (hypothesis):**
  - The SFT-v3 corpus has no compaction or summary turn, and every non-final turn ends in a tool call (degenerate-behaviors-20261031, b2).
  - After 284 steps, the model answers the summary request with a tool call 4.8x as often on the first compaction.
  - v2 etp moved the other way with steps (14.3% at 10 ep → 5.6% at 20 ep), so step count alone does not explain it.

### Bad thinking (`think/{split_table, notool_kinds, reasoning_quality, runaway_units}`, `features/comparison.md`, `compare/thinkx_ta.json`, `reconcile/out/`)

| | base | etp 20 ep | v3 etp 5 ep | v3 etp 10 ep | v3 etp 20 ep |
|---|---|---|---|---|---|
| missing `</think>`: calls (% of calls); rollouts | >= 12 (>= 0.019%); >= 11 | 37 (0.068%); 25 | 21 (0.032%); 19 | 36 (0.068%); 28 | 36 (0.065%); 28 |
| ... rescued tool call / EOS inside think / length cut (calls) | n/a / 9 / 2 | 1 / 3 / 33 | 3 / 6 / 12 | 5 / 4 / 27 | 8 / 3 / 25 |
| action turns without reasoning; rollouts with every action reasoned | 24/61,039; 227/250 | 2/52,584; 248/250 | 4/63,937; 246/250 | 4/50,656; 244/248 | 12/53,848 (0.022%); 238/250 |
| sessions ended by a headless tool call in think; reasoning with call markup: turns/rollouts | 3; 4/4 | 3; 7/7 | 1; 16/8 | 4; 6/6 | 1; 5/4 |
| reasoning tokens per reasoning turn: median / mean / p90; % < 50 tokens | 126 / 659 / 1,479; 28.7 | 135 / 662 / 1,302; 23.7 | 130 / 585 / 1,218; 24.1 | 134 / 645 / 1,259; 23.6 | 147 / 713 / 1,463; 21.2 |
| KS distance to base, reasoning tokens per turn | - | 0.053 | 0.053 | 0.055 | 0.079 |
| verbatim / near repeats; generic short reasoning (% of reasoned actions) | 0.01 / 0.06; 0.01 | 0.01 / 0.06; 0.00 | 0.00 / 0.03; 0.00 | 0.01 / 0.05; 0.00 | 0.03 / 0.10; 0.00 |
| loop-like reasoning in action turns (>= 4,000 chars, zlib < 0.15; the model left the loop and acted): turns; rollouts | 5; 2 | 1; 1 | 59; 12 | 68; 10 | 14; 5 |
| % action turns with visible text; rollouts with a git commit | 5.1; 93.6% | 36.8; 91.2% | 38.8; 89.6% | 41.5; 88.7% | 42.3; 86.0% |

- **Missing `</think>`: 36 calls (0.065%) in 28 rollouts.** That is the same as v3 etp 10 ep (36, 0.068%) and etp 20 ep (37, 0.068%).
  - 8 are rescued tool calls. All 8 are a bare `<think><tool_call>` with 0 reasoning characters.
  - 3 end with EOS inside the think: 1 headless tool call, 2 text.
  - 25 are length cuts.
  - Empty outputs: 0.
- **Unreasoned actions rose but stay tiny:** 12 turns in 12 rollouts, 0.022% of action turns. That is 2.8x v3 etp 10 ep's rate (4 turns, 0.008%).
  - 8 are the rescued bare calls; 4 have an empty closed think. 9 of the 12 rollouts passed.
  - Every action is reasoned in 238/250 rollouts (95.2%), vs 244/248 for v3 etp 10 ep (rollouts with an export). Rollouts with an unreasoned action: +3.2 pp [0.0, +6.4], p 0.10.
  - The HTTP journal of v3 etp 10 ep's no-export infra slot wasmtime-c03 r1 has 1 more such call. Counting it, 5/250: +2.8 pp [−0.4, +6.4], p 0.17 (`reconcile/out/unreasoned.txt`).
- **Headless calls stay at the floor:**
  - 1 session ended on one (v3 etp 10 ep: 4).
  - 5 reasoning turns in 4 rollouts carry tool-call markup, none with a `<tool_call>` opener. These are the "headless" turns of the table.
  - Only 1 of them is a call written inside the reasoning: jest-c05 r0, the session that ended on it.
  - The other 4 quote markup (`<parameter=newString>`, `<function=edit>`) while discussing an edit, and then make a normal tool call. The format lens counts them the same way: 1 call in think, 4 quoted.
  - 1 special-token turn (`<|open|>`).
- **Reasoning got longer and drifts from base:**
  - Median 147 tokens per reasoning turn (v3 etp 10 ep: 134). Per-call completion tokens median / mean: 326 / 960 (v3 etp 10 ep: 309 / 895).
  - KS distance to base is 0.079, vs 0.053–0.055 for the other etp runs.
  - The token counts come from the extractor's tokenizer. A character count from the raw exports agrees: median 546 vs 498 characters per reasoned action turn, KS to base 0.089 vs 0.066–0.069 (`reconcile/out/reasoning_chars.txt`).
  - The reasoning stays substantive: verbatim repeats 0.03%, generic short reasoning 0.00%.
- **Language-switch glitches inside reasoning are back to base level.**
  - Rollouts with a non-Latin run in the reasoning that is not copied from the context: 81/250 vs 39/248 for v3 etp 10 ep (+16.7 pp [+9.2, +24.2], p 8e-5). Base: 95/250 (`degenerate/text`, recounted from the raw exports in `reconcile/out/analyze.txt` C6).
  - That count also catches some deliberate Unicode test values, such as `$[١٢]` and `URL("http://Βόος.com/")`.
  - A stricter count keeps only runs glued to an English word outside quotes and code, such as "It renders每一个 file" and "My期望 keeps being sloppy". These are unambiguous glitches: 38/250 vs 13/248 (+10.0 pp [+3.9, +16.3], p 0.004), and base 36/250.

### Timeouts and session length (`cost/cost.md`, `degenerate/endings/out/tables.txt` T3, `compare/slots_v3etp20.txt`)

| mean per rollout unless noted | base | etp 20 ep | v3 etp 5 ep | v3 etp 10 ep | v3 etp 20 ep |
|---|---|---|---|---|---|
| 3 h deadline kills (with the TimeoutError flag) | 5 (4) | 3 (3) | 9 (7) | 4 (4) | 7 (6) |
| model calls: mean / median; rollouts with >= 500 calls | 247.8 / 211; 14 | 218.1 / 176; 14 | 261.0 / 210; 19 | 210.7 / 175; 8 | 222.5 / 177; 16 |
| completion tokens per rollout | 210k | 200k | 219k | 189k | 214k |
| completion tokens per call: median / mean / p90 | 281 / 848 / 1,834 | 306 / 919 / 1,795 | 306 / 839 / 1,705 | 309 / 895 / 1,772 | 326 / 960 / 1,939 |
| model time (min) | 53.6 | 50.7 | 54.7 | 48.1 | 52.1 |
| wall time: median / mean / p90 (min); rollouts over 2 h | 53.1 / 61.6 / 100.6; 11 | 51.8 / 61.6 / 106.2; 18 | 54.8 / 66.0 / 108.8; 17 | 52.6 / 58.6 / 99.3; 15 | 52.0 / 61.7 / 117.0; 24 |
| vLLM per worker: running sequences; decode tok/s per sequence | 7.6; 70.1 | 7.5; 69.9 | 7.3; 71.3 | 7.1; 69.6 | 7.4; 71.8 |

- **7 deadline kills (v3 etp 10 ep: 4; +1.2 pp [−0.4, +3.6], n.s.).**
  - Slots: gomacro-c01 r1/r3, gomacro-c02-r1b0 r1/r4, wasmtime-c03 r1/r3, zsa-c02-r1b1 r3.
  - 6 carry the TimeoutError flag. zsa r3 has no flag and stopped at 10,791 s.
  - All were long sessions with repeated compaction: 410–965 model calls (median 816, compaction calls included), 3–6 compactions (median 5). 6 had a tool-call summary and 3 a runaway call.
  - At the kill, 4 had a model call in flight, 1 a compaction and 2 a test command (`go test`, `cargo test`).
- **Sessions are a little longer than with v3 etp 10 ep, mostly in the tail.** Paired by task:
  - Completion tokens +24.3k [+8.5, +40.9]; model time +4.0 min [+0.7, +7.6].
  - Calls +11.8 [−3.7, +30.1]; wall time +3.1 min [−0.9, +7.2].
  - Wall p90 is 117 vs 99 min, with 24 vs 15 rollouts over 2 h.
  - Serving was base-like. At v3 etp 10 ep's serving speed the model time would be 53.7 min (actual 52.1), so the extra time comes from extra output.
- **Against etp 20 ep, time is level:** wall +0.1 min [−4.6, +4.7].

### Degenerate behaviors vs v3 etp 5 / 10 ep: do they grow with epochs? (`degenerate/{text, endings, format}`, `think/runaway_units.txt`, `reconcile/out/`)

| behavior (train-acc rollouts unless noted) | v3 etp 5 ep | v3 etp 10 ep | v3 etp 20 ep | etp 20 ep | base | with epochs |
|---|---|---|---|---|---|---|
| hmm loops (b3): rollouts with an hmm loop region | 15/250 | 20/248 | 15/250 | 22/250 | 0 | flat |
| reasoning loop calls (any unit); of them cut at the length limit | 94; 11 (12%) | 117; 23 (20%) | 41; 21 (51%) | 31; 28 (90%) | 8; 2 | fewer loops; a growing share runs to the limit |
| length-cut calls ending in a loop: total (hmm) | 11 (4) | 26 (23) | 24 (17) | 31 (29) | 2 | flat |
| calls at the full 131,072 cap; rollouts ending on a runaway call | 1; 1 | 8; 7 | 10; 8 | 8; 7 | 1; 0 | up 5 → 10 ep, then flat |
| interjection / em-dash floods (rollouts) | 21 / 44 | 26 / 39 | 18 / 30 | 23 / 46 | 9 / 11 | down |
| compaction: tool call instead of the summary (b2) | 21/218 | 10/174 | 44/194 | 10/177 | 2/206 | up 4x at 20 ep |
| compaction: loop inside the summary call (b5, length-cut summaries) | 0/218 | 4/174 | 3/194 | 5/177 | 0/206 | flat |
| commit instruction lost → uncommitted (b4) | 5/18 | 7/13 | 15/39 | 7/12 | 5/8 | up (follows b2) |
| argument run-ons (b9): rollouts; written into a file | 12; 7 | 6; 5 | 3; 1 | 8; 4 | 7; 5 | down |
| headless call in think (b8) / malformed visible call (b10) | 1 / 1 | 4 / 1 | 1 / 3 | 3 / 1 | 2 / 0 | flat |
| hallucinated tool name (OpenCode "invalid" tool), rollouts | 1 | 1 | 9 | 3 | 1 | up (p 0.02) |
| language-switch glitches in reasoning (b21), rollouts: non-Latin run not in the context (glued to an English word, outside quotes/code) | 38 (11) | 39 (13) of 248 | 81 (38) | 61 (28) | 95 (36) | back up to base level |

- **Compaction breakage grows sharply at 20 epochs. It is the degenerate behavior that matters.**
  - From the rows above: 22.7% tool-call summaries, 18.4% even on the first compaction, 39 lost commit instructions.
  - In rollouts: 38 vs 9, 21 and 10 (p 0.0006 vs v3 etp 10 ep; p 0.007 vs v3 etp 5 ep).
  - Summary loops (b5) do not grow.
- **hmm loops are no more frequent, but harder to escape.**
  - Loop calls fall (117 → 41), and rollouts with an hmm loop are flat (20 → 15).
  - Of the loops that occur, 51% run to the length limit (v3 etp 10 ep 20%, v3 etp 5 ep 12%). This is the pattern etp 20 ep showed: 90% of its loop calls run away.
  - Loop-like reasoning that still ended in an action: 14 turns in 5 rollouts (v3 etp 10 ep: 68 in 10).
  - All 8 runaway endings are 131,072-token calls: 7 end in "hmm —" and 1 in a bare-dash loop.
  - The 24 length cuts that end in a loop break down as follows:
    - 17 are hmm.
    - 1 is a bare-dash loop ("— — — E"). Bare-dash loops already appeared at 5 epochs.
    - The other 6 use units not seen before: 3 "WAIT —", 1 ". . .", 1 "h —", and a 304,617-character "INCTINCT…" run with no spaces (76,154 tokens).
  - hmm loops still sit on the same tasks: gomacro 5 rollouts, sqltools 4, jest 3, zsa 2, yq 1.
- **Run-ons keep falling with steps:** 12 → 6 → 3 rollouts, 1 of them written into a file.
- **Small new regressions:** hallucinated tool names 9 vs 1, malformed visible calls 3 vs 1, one argument of 65,536+ characters (86,377), and the language-switch glitches above.
- **Lenses not re-run:** the actions lens (b13 huge tool outputs, b17 edit loops, b18 test re-runs).

### Coverage and the train vs held-out picture (`coverage/coverage_v3etp20.txt`)

- **Coverage:**
  - SFT-v2 has 191 conversations on these 50 tasks, covering all 50.
  - SFT-v3 has 352 (186 from v2, 166 from v4), covering 48. gomacro-c01 and gomacro-c02-r1b3 have none: their v2 conversations were over 262,144 tokens.

| SFT-v3 conversations per task | tasks | base | etp 20 ep | v3 etp 5 ep | v3 etp 10 ep | v3 etp 20 ep |
|---|---|---|---|---|---|---|
| 0 | 2 | 0/10 | 1/10 | 0/10 | 0/10 | 0/10 |
| 1–4 | 5 | 2/25 | 6/25 | 5/25 | 5/23 | 5/25 |
| 5–8 | 29 | 21/145 | 71/145 | 55/145 | 75/142 | 75/145 (51.7%) |
| 9–12 | 14 | 20/70 | 54/70 | 50/70 | 59/70 | 56/70 (80.0%) |

- **Accuracy follows coverage**, which is confounded with task ease (the teacher passes easier tasks more often).
  - Spearman of conversations vs the v3 etp 20 ep per-task rate: 0.57 (vs the base rate: 0.29).
  - The 10 → 20 epoch change is unrelated to coverage (Spearman −0.20).
  - The 2 uncovered tasks are never solved by any SFT-v3 checkpoint.

| | base | raw 5 / 10 / 20 ep | etp 5 / 10 / 20 ep | v3 etp 5 / 10 / 20 ep |
|---|---|---|---|---|
| train-acc pass@1 (50 training tasks x 5) | 17.2% | 18.1 / 36.0 / 44.0% | 26.9 / 39.4 / 52.8% | 44.0 / 56.7 / 54.4% |
| DeepSWE resolved@1 (113 held-out tasks x 5) | 27.79% | 10.44 / 13.98 / 15.75% | 14.87 / 22.12 / 24.42% | 17.35 / 23.54 / 21.77% |
| gap (pp) | −10.6 | +7.6 / +22.0 / +28.2 | +12.0 / +17.2 / +28.4 | +26.7 / +33.2 / +32.6 |

- **Train vs held-out:**
  - On its training tasks v3 etp 20 ep is +37.2 pp over base. On held-out DeepSWE it is 6.0 pp below base (123/565 vs 157/565).
  - From 10 to 20 epochs both numbers fall by about 2 pp, so the gap stays at about 33 pp (33.2 → 32.6). These are the two largest gaps of all runs.
  - More epochs on SFT-v3 buy nothing on either set. The compaction failure behind most of the DeepSWE empty-patch rise is the same one visible here.

### Infra and serving (`compare/serving_gaps.json`, `compare/slots_v3etp20.txt`, `reconcile/out/router.txt`, fleet and driver logs)

- **Infra slots: none.**
  - No vLLM response in the final attempts was non-200. All 36 parser log lines match final-attempt calls.
  - Two slots lost their first attempt to a sandbox PTY failure (SandboxPtyError, true infra). ATP re-ran both, and they are graded like any completed slot:
    - ibis-c04 r1 (group 1): attempt 0 ran 02:22–03:08 and attempt 1 ran 05:03:51–06:38:25. It failed with an empty patch, uncommitted after a tool-call summary. It was the last slot of the run, which is why group 1 ended at 06:39.
    - ibis-c04-r1b0-r2b0 r1 (group 2): attempt 0 ran 02:24–02:48 and attempt 1 ran 03:45–04:21. It passed.
- **Router (55,890 routed calls):**
  - 55,885 got 200. There were no waits, no 429 and no 502.
  - The 5 non-200 records are client disconnects. Each falls within 1 s of a deadline-killed slot's 3 h deadline: the 4 kills with a model call in flight and the 1 with a compaction in flight.
- **Workers:**
  - Leg 1, 10 workers: ready 01:56–02:00, draining from 05:04–05:08, exited 05:48–05:52.
  - Tail, 4 workers (submitted ~04:43): ready 04:55–05:21. They served 660 calls, including all 129 calls after 05:50 (group 1's last slot), so the leg-1 drain left no gap. 4 of the 5 deadline disconnects were also routed to them.
  - Leg 2, 10 workers: 9 ready 06:17–06:21. One (21274840 on pool0-00214) hung at vLLM startup after an Inductor compile error and failed at 06:32 (`fleets/v3etp20/logs/hang-21274840.txt`).
  - Leg 2 served 0 calls. Its workers and the tail workers were cancelled at 06:48:07, 9 minutes after the last group ended.
  - Node-hours: leg 2 5.7 (idle), tail 7.1, leg 1 39.9. No outcome was affected.
- **The parity check reports FAIL on every worker,** as it did for v3 etp 10 ep. The only mismatch is the intended one: the counting lenient-think parser instead of RL's nano_v3.

### Live table vs final (`live/live_metrics_final.txt`)

The live monitor's metrics, recomputed from the final data:
- pass@1 54.4%.
- pass@5 46/50 tasks with 5 graded slots (92.0%).
- Empty patch 14.4% of graded rollouts.
- Timeouts 2.4% (6 rollouts with the TimeoutError flag).
- Every action reasoned in 95.2% of rollouts.
- Missing `</think>` 0.046% from the parser counters, vs 0.065% (36/55,623) from the final-attempt join. 10 of 24 vLLM logs have no counts line, and counters are logged only every 200 calls.

### Conclusions

- **20 epochs on SFT-v3 add nothing on the training tasks.**
  - 54.4% pass@1: −2.3 pp vs 10 epochs (n.s.), +10.4 pp vs 5 epochs, +37.2 pp vs base.
  - pass@5 46/50, but only 5 tasks at 5/5 (10 epochs: 9).
- **Against v2 etp at the same 20 epochs it is level** (+1.6, n.s.), so the +17 pp SFT-v3 lead at 5 and 10 epochs is gone.
  - Per optimizer step the two corpora track each other up to ~150 steps. v3's extra 142 steps do not help.
- **Empty patches rose to 36 (14.4%), almost all from compaction breakage.**
  - The model writes a tool call instead of the compaction summary 4x as often (22.7% of summaries; 18.4% on the first compaction). In 42 of 44 cases the compacted context then loses the task and the commit instruction.
  - 28 of 36 empty patches follow such a summary. Other empty patches fell to 8.
  - On these tasks the empty-patch rise is not significant (+2.4 pp, p 0.48). The mechanism is significant (p 0.0006), and broken-summary rollouts resolve far less often even within the same task (4/37 vs 28/50).
  - The same cross-check on DeepSWE explains most of its 159 vs 106 empty patches:
    - 18.7% vs 7.6% tool-call summaries;
    - the +33 uncommitted endings, 50 of the 51 after a lost instruction;
    - the +11 deadline kills (31 vs 19 after a broken summary).
    - The +10 format endings there are a separate change.
- **Thinking stays fixed:**
  - Missing `</think>` 0.065% (level with v3 etp 10 ep), 0 empty outputs, 1 headless ending.
  - Unreasoned actions rose to 12 turns (0.022%; 238/250 rollouts fully reasoned; n.s. vs v3 etp 10 ep).
  - Reasoning is longer and less base-like (KS 0.079). Language-switch glitches are back to base level, also when counting only runs glued to English words.
- **Degenerate behaviors with epochs:**
  - Compaction breakage grows sharply.
  - hmm loops are no more frequent, but half of them now run to the length limit (8 rollouts end on a 131,072-token loop).
  - Run-ons keep shrinking (3 rollouts).
  - There are 7 deadline kills, all long sessions with repeated compaction, and rollouts over 2 h rose to 24.
- **Train vs held-out:** the gap to DeepSWE stays at about 33 pp (54.4 vs 21.77%), and both fall by about 2 pp from 10 to 20 epochs.
- **Recommendations:**
  - Prefer the 10-epoch checkpoint over the 20-epoch one.
  - Fix compaction rather than train longer: put summary examples in the data, or have the harness re-inject the task and commit instruction.
  - The empty-patch fixes in the degenerate-behaviors report apply here directly.

### Caveats

- **Python environment.** The steps that need `tokenizers`, `orjson` or `scipy` ran on the vLLM env python with the same -I -B flags, as the original tools did: run_features.py (TR/features.py), the text lens's scan_models.py, fmt_detect.py and format_counts.py. /usr/bin/python3 has no `tokenizers`. Every other script, including all of `reconcile/`, ran with /usr/bin/python3 -I -B, and no __pycache__ was written. The token-level reasoning lengths were therefore not re-tokenized in the reconcile; a character-level check agrees with them.
- **DeepSWE classifier rows.** The DeepSWE v3etp20 numbers come from the parallel DeepSWE v3etp20 analysis's classifier rows (`sftv3-etp20-20261104/classifier/v3etp20/all.jsonl`, read-only).
  - That file was first written at 06:51 and rewritten at 07:17:34, when that analysis re-ran its classifier.
  - `coverage_v3etp20.py` (07:14) read the first version, and the DeepSWE cross-check (07:18) read the second.
  - Both versions give 123/565 resolved and 159 empty patches. These match the raw run's metrics.json (21.77%) and a recount from the raw rollouts (`reconcile/out/deepswe.txt`), so no number changed.
- **Statistical tests.** On the training tasks the empty-patch rise vs v3 etp 10 ep is not significant (+2.4 pp, p 0.48; graded only +4.2, p 0.17), while the compaction-breakage mechanism is significant. Fisher tests treat the 250 slots as independent. The task-paired bootstrap and sign-flip tests are the primary ones.
- **Association only.** The link between broken summaries and lower accuracy (4/38 vs 46/87 resolved; 4/37 vs 28/50 within the same tasks) is an association. Sessions that compact more often are longer, and they have more chances to write a bad summary.
- **Per-task view.** Per task, the rise in tool-call summaries is not confined to the tasks that got worse (unchanged tasks 2 → 16).
- **Hypothesis.** The "why at 20 epochs" explanation is a hypothesis: the SFT-v3 corpus has no compaction or summary turns (degenerate-behaviors-20261031). v2 etp moved the opposite way with steps.
- **Lenses re-run.** Only the endings, text and format lenses of degenerate-behaviors-20261031 were re-run. The actions lens (huge tool outputs, edit-failure loops, test re-runs) was not. The text lens takes its corpus-derived thresholds from the published corpus_summary.json; the SFT corpus scan was not re-run.
- **Lens-dependent splits.** The text lens's split of non-runaway loops into "self-ended" and "escaped" depends on the definition (18/2 by the lens, 15/5 by a region-only rule), so the report gives only total loop calls and the share cut at the length limit.
- **Language-switch count.** The lens's language-switch count includes a few deliberate Unicode test values. The stricter count (runs glued to an English word, outside quotes and code) is a heuristic too, but both counts point the same way.
- **Export basis.** Export-based measures use 248 rollouts for v3 etp 10 ep: its 2 true-infra slots have no session export. `compare/stats_tests.txt` puts those 2 slots in a /250 denominator as clean. For unreasoned actions and language-switch glitches the report uses the 248 basis (`reconcile/out/`) instead.
- **sc14 length cuts.** The 'length cut' flag in stats_tests is n/a for sc14, which was served without the counting parser (it shows 0/250 there).
- **Live script.** Main's live script (live_v3etp20.py) was not run, because it imports ATP slot-state code and writes a cache under analysis/watch. Its metrics were recomputed from the final data in live/live_metrics_final.py.
- **Same-steps comparison.** No v2 run exists near 284 steps, so a same-steps comparison for v3etp20 is not possible. The same-steps statement relies on the existing v3etp10 (142) vs etp20 (152) pair.
- **Worker cancellation.** The logs I read do not record who cancelled the leg-2 and tail workers at 06:48:07. The leg-2 workers held 5.7 node-hours idle.
- **Disk use.** `empty-patch/v3etp{10,20}/run/` hold ~690 MB of classifier inputs each. The v3etp10 copy exists only for the reproduction check and can be rebuilt with build_run.py.

### Verification changes

I re-checked every point the two verifiers raised against the raw data: session exports, HTTP journals, router logs and the raw DeepSWE runs. The scripts are in `analysis/v3etp20/reconcile/`, commands in `reproduce_reconcile.sh` and outputs in `reconcile/out/`. All of them are /usr/bin/python3 -I -B, stdlib only.

1. **DeepSWE gap to base "5.9 pp" (verifiers 1 and 2): corrected to 6.0 pp.**
   - From the raw rollouts and base's metrics.json: 123/565 vs 157/565 = −34/565 = −6.02 pp (`r_deepswe.py`).
2. **Deadline-kill "median 811" calls (verifiers 1 and 2): corrected to 816.**
   - The HTTP journals of the 7 kills hold 410, 630, 747, 816, 859, 860 and 965 completed model calls (`r_http_targeted.py`).
   - 811 is the median without the compaction-summary calls, which the stated range includes.
3. **DeepSWE "32 of the 48 deadline kills had a broken summary" (verifier 1): corrected.**
   - 32 is over all 49 N1 kills.
   - Over the 48 kills with an empty patch it is 31, both by "tool-call or length-cut summary" and by the endings lens's broader rule. v3 etp 10 ep: 19 of 37.
   - Kills without a broken summary are level (17 vs 18), and the report now says so (`r_deepswe.py`).
4. **Loop-unit list summing to 23 (verifier 1; verifier 2 noted the same): corrected.**
   - The 24th loop-ended length cut is a bare-dash loop ("— — — E", gomacro-c02-r1b0 r0, 131,072 tokens).
   - Bare-dash loops already occurred at 5 epochs, so the sentence now separates it from the 6 new units. Checked on the reasoning tails in the raw exports (`r_analyze.py`, C5).
5. **Classifier timestamp in the caveat (verifiers 1 and 2): corrected.**
   - The DeepSWE v3etp20 `all.jsonl` was rewritten at 07:17:34. Coverage (07:14) read the first version and the cross-check (07:18) the second.
   - Both give 123/565 resolved and 159 empty patches, equal to the raw metrics.json and my raw recount. No number changes.
6. **"The tool-call summary has no `## Goal`" (verifiers 1 and 2): corrected to 42 of 44.**
   - zsa-c02-r1b1 r2 and gomacro-c02-r1b3 r3 write a full `## Goal` summary, commit instruction included, before the tool call.
   - The export scan and the HTTP journal agree: 193 completed compaction responses (plus 1 in flight at a kill), 44 with markup, 0 parsed tool calls, 2 with `## Goal`.
   - 34 of the 39 first instruction losses are via a tool-call summary; the report now says so (`r_analyze.py` C1, `r_http_targeted.py`).
7. **"The same mechanism explains the DeepSWE rise" (verifier 1): narrowed to "most of".**
   - It explains the +33 uncommitted endings (50 of 51 after a lost instruction) and the +11 deadline kills (31 vs 19 with a broken summary, 17 vs 18 without).
   - It does not explain the +10 format endings: only 5 of 24 follow a broken summary, and there is no train-acc counterpart (5 vs 5) (`r_deepswe.py`).
   - The lens's uncommitted count for v3 etp 10 ep is 18. The classifier's E1 code gives 19, because the lens files one of those rollouts as a context overflow. This does not change the conclusion.
8. **Per-task evidence "18 worse tasks: 3 → 16 tool-call-summary rollouts" (verifier 2, medium): confirmed and rewritten.**
   - Unchanged tasks went 2 → 16 as well, 8 of them on the 3 tasks that neither checkpoint solves. Better tasks went 4 → 6. So the per-task split does not single out the worse tasks.
   - The report now gives all three groups.
   - It adds a within-task comparison as the stronger evidence: over the 19 tasks with both kinds, broken-summary rollouts resolve 4/37 and fine-summary rollouts 28/50 (Mantel-Haenszel OR 0.21) (`r_analyze.py` C2, `r_within_task.py`).
9. **Tail workers "served 664 calls" (verifier 2): corrected to 660.**
   - The router log has 660 status-200 records served by the 4 tail workers.
   - The 4 other records routed to them are client disconnects at deadline kills. Verifier 2 said they were served elsewhere; none was. All 129 calls after 05:50 are confirmed (`r_router.py`).
10. **v3 etp 10 ep denominators of 248 vs 250 (verifier 2): unified on the export basis, with the all-slot variant added.**
    - Every action reasoned: 244/248 vs 238/250, +3.2 pp [0.0, +6.4], p 0.10. This is numerically identical to the old /250 computation, which counted the 2 no-export slots as clean.
    - wasmtime-c03 r1, one of those slots, has 1 action call with no reasoning in its HTTP journal. Counting it gives 5/250: +2.8 pp [−0.4, +6.4], p 0.17 (`r_http_targeted.py`, `r_unreasoned.py`).
    - "3x" became "2.8x the per-action rate".
    - The language-switch test was recomputed on 39/248: +16.7 pp [+9.2, +24.2], p 8e-5 (previously +16.8 [+9.6, +24.4] on /250).
    - The tool-call-summary test keeps 9/250: neither no-export slot has a tool-call compaction response in its journal.
11. **"5 markup turns, all headless" (verifier 2): clarified.**
    - None of the 5 has a `<tool_call>` opener; that is the table's "headless".
    - Only jest-c05 r0's turn is a call written inside the reasoning, and it ended the session.
    - The other 4 quote `<parameter=…>` or `<function=edit>` while discussing an edit, then make a normal call. This matches the format lens: 1 call in think, 4 quoted turns (`r_analyze.py` C4).
12. **The language-switch metric counts deliberate Unicode test values (verifier 2): confirmed; stricter counts added.**
    - Examples: `$[١٢]`, `URL("http://Βόος.com/")`, `"unicode: €α日本語"`.
    - The lens's count reproduces exactly from the raw exports: 95 / 61 / 38 / 39 / 81.
    - Runs outside quotes and code: 56/250 vs 19/248 (p 0.0005; base 72).
    - Runs glued to an English word, outside quotes and code: 38/250 vs 13/248, +10.0 pp [+3.9, +16.3], p 0.004 (base 36/250).
    - The conclusion is unchanged (`r_analyze.py` C6).
13. **Lens-specific items that verifier 2 could not reproduce:**
    - **Run-ons, tool names, largest argument:** recounted from the raw exports and identical: argument run-ons in 12 / 6 / 3 rollouts (1 written into a file), hallucinated tool names 9 vs 1, a largest argument of 86,377 characters (`r_runons.py`).
    - **Uncommitted 10 / 6 split:** I read all 16 final texts and the split holds. One case is borderline: wasmtime-c01-r1b1 r4 says no commit was made "per the repo's AI tool policy", and the lens files it under claims done (`r_uncommitted.py`).
    - **Token-level reasoning lengths:** not re-tokenized. The character-level check agrees: median 546 vs 498 characters, KS to base 0.089 vs 0.066–0.069 (`r_reasoning_chars.py`).
    - **Loop outcomes:** the self-ended / escaped split depends on the definition (lens 18/2, region-only rule 15/5, verifier 2's detector 10/9), so it was removed from the table. What remains is robust: 41 loop calls (vs 117 and 94), of which 21–22 (51–54%) are cut at the length limit (`r_loops.py`).
    - **"Escaped":** the word named two different measures. The reasoning_quality measure is now called "loop-like reasoning in action turns".
    - **Floods and repeats / generic reasoning:** verifier 1 reproduced these, and they are kept as lens outputs.
