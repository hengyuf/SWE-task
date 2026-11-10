# kimi-rollouts-v4 final report (phase_report.py, 2026-10-29T21:57:34-0700)

Request (hengyuf, 2026-10-28): 在之前v2 上的所有task 上,用kimi 在每个生成的task 上分别rollout 5遍, context window 设置为256k, prompt 保持不变(不要加reason before action 之类的prompt). Re-run rule: 这次若遇到被坏回复终止的 slot, 可以重跑1遍.

## 1. The 50 sel50 tasks: v4: stock prompt, 256k (2026-10-28/29), v3 sentence arm: 256k + reasoning sentence (2026-10-26), earlier Kimi-K3: 1M, stock prompt (2026-10-17/21)

| metric | v4 first attempts | v4 after re-run | v3 sentence (256k) | earlier Kimi (1M) |
|---|---|---|---|---|
| pass@1 raw | 170/250 = 68.0% (CI [60.4, 75.2]) | 182/250 = 72.8% (CI [65.2, 80.0]) | 121/248 = 48.8% (CI [40.5, 57.4]) | 192/248 = 77.4% (CI [72.0, 82.6]) |
| pass@1 excl. broken-ended | 170/233 = 73.0% (CI [65.1, 80.3]) | 182/249 = 73.1% (CI [65.5, 80.3]) | 121/178 = 68.0% (CI [58.6, 77.3]) | 192/248 = 77.4% (CI [72.0, 82.6]) |
| tasks >=1 pass / 5 of 5 / none | 50 / 12 / 0 | 50 / 17 / 0 | 45 / 7 / 5 | 50 / 16 / 0 |
| % action turns with reasoning | 56.4 (CI [54.7, 58.1]) | 56.2 (CI [54.4, 57.9]) | 58.0 (CI [56.2, 59.7]) | 54.9 (CI [53.5, 56.4]) |
|   same, rollouts without broken replies | 56.4 | 56.2 | 58.0 | 54.9 |
| compacted rollouts | 20/250 (8.0%) | 22/250 (8.8%) | 9/248 (3.6%) | 0/250 (0.0%) |
| main prompt > 242,144 (rollouts) | 6 (2.4%) | 6 (2.4%) | 2 (0.8%) | 12 (4.8%) |
| largest main-session prompt | 253134 | 253134 | 243395 | 505801 |
| replies cut at output cap / context-overflow refusals | 1 / 0 | 1 / 0 | 0 / 0 | 0 / 0 |
| broken replies / replies (final attempts) | 26/26664 = 0.98 per 1000 | 10/27892 = 0.36 per 1000 | 77/20518 = 3.75 per 1000 | 0/27154 = 0.0 per 1000 |
| rollouts ended by a broken reply | 17/250 (6.8%) | 1/250 (0.4%) | 70/250 (28.0%) | 0/250 (0.0%) |
| rollouts with any broken reply | 24 (9.6%) | 8 (3.2%) | 75 (30.0%) | 0 (0.0%) |
| turns per rollout (median) | 93.0 | 96.0 | 73.0 | 93.5 |
| agent wall min (median) | 57.8 | 60.7 | 55.1 | 35.5 |

Paired task bootstrap (points): pass@1 v4_first_minus_1m -9.4 (CI -16.6 to -2.2, 50 tasks); v4_after_rerun_minus_1m -4.6 (CI -11.5 to +2.3, 50 tasks); v4_first_minus_v3 +19.2 (CI +10.0 to +28.0, 50 tasks); v4_after_rerun_minus_v3 +24.0 (CI +14.4 to +33.2, 50 tasks); v4_first_excl_minus_v3_excl +5.0 (CI -4.3 to +14.9, 50 tasks); v4_first_excl_minus_1m -4.5 (CI -12.0 to +3.2, 50 tasks); v3_minus_1m -28.6 (CI -36.6 to -19.8, 50 tasks)

% action turns with reasoning: v4_first_minus_1m +1.5 (CI +0.5 to +2.6, 50 tasks); v4_first_minus_v3 -1.6 (CI -2.9 to -0.2, 50 tasks); v3_minus_1m +3.1 (CI +1.7 to +4.4, 50 tasks); v4_after_rerun_minus_1m +1.3 (CI +0.2 to +2.4, 50 tasks); v4_minus_1m_both_without_degenerate +1.6 (CI +0.5 to +2.6, 50 tasks); v4_minus_v3_both_without_degenerate -1.5 (CI -3.1 to -0.0, 50 tasks)

Broken replies, every attempt (per 1000 replies): v4_originals_no_reruns 27/27240 = 0.99; v4_all_attempts 28/29617 = 0.95; v3_all_attempts 78/21538 = 3.62; 1m_all_attempts 0/27354 = 0.0

Tests: {"sel50_v4_vs_v3": {"fisher_p_per_reply": 5.882389998851767e-10, "bootstrap95_new_minus_control_per_1000": [-3.56, -1.78], "rate_ratio_new_over_control": 0.27, "rate_ratio_exact95": [0.17, 0.43], "rollouts_with_degenerate_reply": {"control": "75/250", "new": "24/250", "fisher_p": 1.0947592400573139e-08}, "rollouts_ended_by_degenerate_reply": {"control": "70/250", "new": "17/250", "fisher_p": 2.8611433674570666e-10}}, "sel50_v4_vs_1m": {"fisher_p_per_reply": 6.996386970413485e-09, "bootstrap95_new_minus_control_per_1000": [0.63, 1.4], "rollouts_with_degenerate_reply": {"control": "0/250", "new": "24/250", "fisher_p": 6.678128609274843e-08}, "rollouts_ended_by_degenerate_reply": {"control": "0/250", "new": "17/250", "fisher_p": 1.151694100025862e-05}}, "sel50_v3_vs_1m": {"fisher_p_per_reply": 1.5686779967849628e-28, "bootstrap95_new_minus_control_per_1000": [2.86, 4.49], "rollouts_with_degenerate_reply": {"control": "0/250", "new": "75/250", "fisher_p": 7.529704298868077e-26}, "rollouts_ended_by_degenerate_reply": {"control": "0/250", "new": "70/250", "fisher_p": 6.052759800328065e-24}}}

Cost on sel50 (LiteLLM usage.cost of the analysed attempts): v4_first $910.12 (3.64/rollout); v3 $748.58 (2.994/rollout); 1m $705.03 (2.82/rollout)

## 2. v4 full set by taskset

| | sel50 | rest47 | baihe | all |
|---|---|---|---|---|
| tasks / slots | 50 / 250 | 47 / 235 | 97 / 485 | 194 / 970 |
| pass@1 first attempts | 170/250 = 68.0% (CI [60.4, 75.2]) | 180/235 = 76.6% (CI [69.4, 83.0]) | 370/485 = 76.3% (CI [71.8, 80.8]) | 720/970 = 74.2% (CI [70.7, 77.6]) |
| pass@1 first, excl. broken-ended | 170/233 = 73.0% (CI [65.1, 80.3]) | 180/225 = 80.0% (CI [72.7, 86.4]) | 369/446 = 82.7% (CI [78.3, 87.1]) | 719/904 = 79.5% (CI [76.0, 83.0]) |
| pass@1 after the one re-run | 182/250 = 72.8% (CI [65.2, 80.0]) | 188/235 = 80.0% (CI [72.8, 86.4]) | 394/485 = 81.2% (CI [76.7, 85.6]) | 764/970 = 78.8% (CI [75.4, 82.1]) |
| tasks >=1 pass / 5 of 5 / none (after re-run) | 50 / 17 / 0 | 46 / 22 / 1 | 97 / 43 / 0 | 193 / 82 / 1 |
| broken-ended first attempts | 17 (6.8%) {"pad": 16, "empty": 1} | 10 (4.3%) {"pad": 9, "empty": 1} | 39 (8.0%) {"pad": 36, "special": 1, "empty": 2} | 66 (6.8%) {"pad": 61, "special": 1, "empty": 4} |
| re-runs: pass / fail / broken again / infra | 17: 12 / 4 / 1 / 0 | 10: 8 / 1 / 1 / 0 | 39: 25 / 7 / 7 / 0 | 66: 45 / 12 / 9 / 0 |
| broken replies per 1000 (first attempts) | 26/26664 = 0.98 | 14/30112 = 0.46 | 50/48309 = 1.04 | 90/105085 = 0.86 |
| rollouts with any broken reply (first) | 24 (9.6%) | 14 (6.0%) | 48 (9.9%) | 86 (8.9%) |
| % action turns with reasoning (first) | 56.4 (CI [54.7, 58.1]) | 53.2 (CI [51.8, 54.6]) | 54.2 (CI [53.1, 55.2]) | 54.5 (CI [53.7, 55.2]) |
| compacted (first attempts) | 20 (8.0%) | 21 (8.9%) | 23 (4.7%) | 64 (6.6%) |
| pass@1 compacted / not (first) | 60.0 / 68.7 | 66.7 / 77.6 | 82.6 / 76.0 | 70.3 / 74.5 |
| length cuts / context overflows (all graded) | 1 / 0 | 0 / 0 | 1 / 0 | 2 / 0 |
| turns / agent wall min (median, first) | 93.0 / 57.8 | 120 / 67.3 | 90 / 52.1 | 95.0 / 57.7 |

## 3. Broken-reply re-runs: 66 ({"pass": 45, "fail": 12, "broken_again": 9, "infra": 0}), replaced 66, third attempts 0; detector check {"attempts": 1036, "mismatches": []}

| campaign | task | slot | original (F2P, kind) | re-run | outcome (F2P) |
|---|---|---|---|---|---|
| sel50-g0 | syn-risor-cand-c05-r1b2-3718460d | kimi-r0 | fail (0/19, pad) | sel50-g0-brk1 kimi-r0 | pass (19/19) |
| sel50-g0 | syn-vscode-sqltools-cand-c02-r1b0-fb4b293a | kimi-r0 | fail (0/49, pad) | sel50-g0-brk1 kimi-r0 | pass (49/49) |
| sel50-g0 | syn-wasmtime-cand-c03-353ec720 | kimi-r1 | fail (0/14, pad) | sel50-g0-brk1 kimi-r0 | pass (14/14) |
| sel50-g0 | syn-yq-cand-c02-r1b0-0af03743 | kimi-r1 | fail (0/19, pad) | sel50-g0-brk1 kimi-r0 | pass (19/19) |
| sel50-g1 | syn-aiohttp-cand-c03-r1b1-b706aac1 | kimi-r1 | fail (0/129, pad) | sel50-g1-brk2 kimi-r0 | pass (129/129) |
| sel50-g1 | syn-aiohttp-cand-c03-r1b1-b706aac1 | kimi-r4 | fail (0/129, pad) | sel50-g1-brk2 kimi-r1 | pass (129/129) |
| sel50-g1 | syn-ibis-cand-c03-r1b4-3d3b86b5 | kimi-r1 | fail (0/570, pad) | sel50-g1-brk1 kimi-r0 | fail (534/570) |
| sel50-g1 | syn-jest-cand-c01-r1b3-08931258 | kimi-r0 | fail (0/81, empty) | sel50-g1-brk1 kimi-r0 | fail (77/81) |
| sel50-g1 | syn-pylama-cand-c02-r1b0-r2b2-02688723 | kimi-r3 | fail (0/37, pad) | sel50-g1-brk1 kimi-r0 | pass (37/37) |
| sel50-g1 | syn-vscode-sqltools-cand-c02-r1b3-7119812a | kimi-r3 | fail (0/26, pad) | sel50-g1-brk1 kimi-r0 | pass (26/26) |
| sel50-g2 | syn-aiohttp-cand-c04-7f733601 | kimi-r0 | fail (0/75, pad) | sel50-g2-brk1 kimi-r0 | fail (74/75) |
| sel50-g2 | syn-jest-cand-c04-r1b4-d8b1da89 | kimi-r3 | fail (0/50, pad) | sel50-g2-brk1 kimi-r0 | pass (50/50) |
| sel50-g3 | syn-gomacro-cand-c02-r1b3-ed535f70 | kimi-r0 | fail (0/18, pad) | sel50-g3-brk1 kimi-r0 | fail (15/18) |
| sel50-g3 | syn-jest-cand-c05-9e1caaee | kimi-r1 | fail (0/28, pad) | sel50-g3-brk1 kimi-r0 | pass (28/28) |
| sel50-g4 | syn-jest-cand-c05-r1b2-d6a136fe | kimi-r0 | fail (0/48, pad) | sel50-g4-brk1 kimi-r0 | pass (48/48) |
| sel50-g4 | syn-zsa-cand-c02-r1b1-ae1eb794 | kimi-r0 | fail (0/26, pad) | sel50-g4-brk2 kimi-r0 | broken_again (0/26) |
| sel50-g4 | syn-zsa-cand-c02-r1b1-ae1eb794 | kimi-r1 | fail (0/26, pad) | sel50-g4-brk2 kimi-r1 | pass (26/26) |
| rest47-g0 | syn-jest-cand-c05-r1b0-5ec91bba | kimi-r4 | fail (0/72, pad) | rest47-g0-brk1 kimi-r0 | pass (72/72) |
| rest47-g1 | syn-gomacro-cand-c04-r1b2-b79686cb | kimi-r0 | fail (0/31, pad) | rest47-g1-brk1 kimi-r0 | pass (31/31) |
| rest47-g1 | syn-wasmtime-cand-c03-r1b2-03550fa4 | kimi-r0 | fail (0/16, empty) | rest47-g1-brk1 kimi-r0 | broken_again (0/16) |
| rest47-g2 | syn-gomacro-cand-c05-r1b1-d80cf0a5 | kimi-r1 | fail (0/18, pad) | rest47-g2-brk1 kimi-r0 | pass (18/18) |
| rest47-g2 | syn-gpython-cand-c01-r1b1-765875d9 | kimi-r1 | fail (0/68, pad) | rest47-g2-brk1 kimi-r0 | pass (68/68) |
| rest47-g2 | syn-litestar-cand-c01-r1b1-8f1b686c | kimi-r0 | fail (0/32, pad) | rest47-g2-brk1 kimi-r0 | pass (32/32) |
| rest47-g2 | syn-wasmtime-cand-c03-r1b4-4d3d43a0 | kimi-r4 | fail (0/18, pad) | rest47-g2-brk1 kimi-r0 | pass (18/18) |
| rest47-g3 | syn-cerbos-cand-c02-r1b4-9bbe0541 | kimi-r4 | fail (0/18, pad) | rest47-g3-brk1 kimi-r0 | pass (18/18) |
| rest47-g3 | syn-gpython-cand-c04-r1b1-e954ad27 | kimi-r4 | fail (0/76, pad) | rest47-g3-brk1 kimi-r0 | pass (76/76) |
| rest47-g4 | syn-yq-cand-c01-r1b3-b5f105ff | kimi-r4 | fail (0/22, pad) | rest47-g4-brk1 kimi-r0 | fail (21/22) |
| baihe-g0 | syn-mathjax-src-cand-c05-r1b0-bf6190e1 | kimi-r0 | fail (0/16, pad) | baihe-g0-brk2 kimi-r0 | pass (16/16) |
| baihe-g0 | syn-mathjax-src-cand-c05-r1b0-bf6190e1 | kimi-r4 | fail (0/16, pad) | baihe-g0-brk2 kimi-r1 | broken_again (0/16) |
| baihe-g0 | syn-nova-cand-c04-r1b0-afe620cd | kimi-r0 | fail (0/32, pad) | baihe-g0-brk1 kimi-r0 | pass (32/32) |
| baihe-g0 | syn-risor-cand-c01-r1b4-bde52eba | kimi-r4 | fail (0/218, pad) | baihe-g0-brk1 kimi-r0 | pass (218/218) |
| baihe-g0 | syn-risor-cand-c05-r1b2-e15a5178 | kimi-r0 | fail (0/26, pad) | baihe-g0-brk2 kimi-r0 | pass (26/26) |
| baihe-g0 | syn-risor-cand-c05-r1b2-e15a5178 | kimi-r3 | fail (0/26, pad) | baihe-g0-brk2 kimi-r1 | fail (0/26) |
| baihe-g1 | syn-dacite-cand-c02-r1b0-a325e28a | kimi-r1 | fail (0/40, pad) | baihe-g1-brk2 kimi-r0 | fail (39/40) |
| baihe-g1 | syn-dacite-cand-c02-r1b0-a325e28a | kimi-r4 | fail (0/40, special) | baihe-g1-brk2 kimi-r1 | pass (40/40) |
| baihe-g1 | syn-lucid-cand-c04-r1b1-44b0d57a | kimi-r3 | fail (0/117, pad) | baihe-g1-brk1 kimi-r0 | fail (116/117) |
| baihe-g1 | syn-pyserde-cand-c01-aed9b543 | kimi-r2 | fail (0/19, pad) | baihe-g1-brk1 kimi-r0 | pass (19/19) |
| baihe-g1 | syn-python-prompt-toolkit-cand-c04-63029829 | kimi-r3 | fail (0/3, pad) | baihe-g1-brk1 kimi-r0 | pass (3/3) |
| baihe-g1 | syn-remark-lint-cand-c03-r1b4-a0f28434 | kimi-r1 | fail (0/30, pad) | baihe-g1-brk1 kimi-r0 | broken_again (0/30) |
| baihe-g1 | syn-risor-cand-c02-r1b1-0ccf905c | kimi-r2 | fail (0/137, pad) | baihe-g1-brk1 kimi-r0 | pass (137/137) |
| baihe-g1 | syn-risor-cand-c05-r1b3-e0a6aa76 | kimi-r0 | fail (0/180, pad) | baihe-g1-brk1 kimi-r0 | pass (180/180) |
| baihe-g2 | syn-go-jsonnet-cand-c05-r1b4-6d66c2db | kimi-r1 | fail (0/47, pad) | baihe-g2-brk1 kimi-r0 | broken_again (0/47) |
| baihe-g2 | syn-risor-cand-c05-r1b4-1ee392ce | kimi-r1 | fail (0/74, pad) | baihe-g2-brk1 kimi-r0 | pass (74/74) |
| baihe-g3 | syn-mathjax-src-cand-c02-r1b0-3bf1ffa4 | kimi-r3 | fail (0/23, pad) | baihe-g3-brk1 kimi-r0 | pass (23/23) |
| baihe-g3 | syn-remark-lint-cand-c01-r1b0-a55630d0 | kimi-r0 | fail (0/28, pad) | baihe-g3-brk1 kimi-r0 | broken_again (0/28) |
| baihe-g4 | syn-go-lua-cand-c01-r1b3-c9a601af | kimi-r3 | fail (0/12, pad) | baihe-g4-brk1 kimi-r0 | pass (12/12) |
| baihe-g4 | syn-gval-cand-c01-r1b0-b0a98783 | kimi-r3 | fail (0/119, pad) | baihe-g4-brk1 kimi-r0 | pass (119/119) |
| baihe-g4 | syn-inquirer-js-cand-c03-r1b0-0548d9f2 | kimi-r1 | pass (33/33, pad) | baihe-g4-brk1 kimi-r0 | fail (32/33) |
| baihe-g4 | syn-nova-cand-c02-r1b0-40b314b3 | kimi-r2 | fail (0/24, pad) | baihe-g4-brk1 kimi-r0 | broken_again (0/24) |
| baihe-g4 | syn-pyserde-cand-c02-r1b1-e4743457 | kimi-r0 | fail (0/48, pad) | baihe-g4-brk1 kimi-r0 | pass (48/48) |
| baihe-g4 | syn-risor-cand-c03-r1b3-3380be16 | kimi-r0 | fail (0/197, pad) | baihe-g4-brk1 kimi-r0 | fail (165/197) |
| baihe-g5 | syn-go-lua-cand-c02-41f65269 | kimi-r3 | fail (0/8, pad) | baihe-g5-brk1 kimi-r0 | pass (8/8) |
| baihe-g5 | syn-mathjax-src-cand-c02-r1b4-da3483cd | kimi-r0 | fail (0/20, pad) | baihe-g5-brk1 kimi-r0 | pass (20/20) |
| baihe-g5 | syn-pytest-cand-c01-r1b0-954445d0 | kimi-r1 | fail (0/45, pad) | baihe-g5-brk2 kimi-r0 | fail (0/45) |
| baihe-g5 | syn-pytest-cand-c01-r1b0-954445d0 | kimi-r3 | fail (0/45, pad) | baihe-g5-brk2 kimi-r1 | fail (0/45) |
| baihe-g5 | syn-risor-cand-c04-r1b1-eab3d6e4 | kimi-r2 | fail (0/224, pad) | baihe-g5-brk1 kimi-r0 | broken_again (0/224) |
| baihe-g6 | syn-mathjax-src-cand-c03-r1b0-4e89d622 | kimi-r2 | fail (0/23, pad) | baihe-g6-brk1 kimi-r0 | pass (23/23) |
| baihe-g6 | syn-nova-cand-c03-r1b0-cb0355d9 | kimi-r4 | fail (0/28, empty) | baihe-g6-brk1 kimi-r0 | pass (28/28) |
| baihe-g6 | syn-pytest-cand-c02-69e6073d | kimi-r2 | fail (0/13, pad) | baihe-g6-brk2 kimi-r0 | pass (13/13) |
| baihe-g6 | syn-pytest-cand-c02-69e6073d | kimi-r4 | fail (0/13, pad) | baihe-g6-brk2 kimi-r1 | pass (13/13) |
| baihe-g6 | syn-remark-lint-cand-c01-r1b4-b6f41331 | kimi-r4 | fail (0/24, empty) | baihe-g6-brk1 kimi-r0 | pass (24/24) |
| baihe-g6 | syn-risor-cand-c01-r1b0-e54193d4 | kimi-r4 | fail (0/179, pad) | baihe-g6-brk1 kimi-r0 | pass (179/179) |
| baihe-g7 | syn-citty-cand-c04-r1b1-ec1c0299 | kimi-r0 | fail (0/23, pad) | baihe-g7-brk1 kimi-r0 | pass (23/23) |
| baihe-g7 | syn-inquirer-js-cand-c05-r1b0-a34e3650 | kimi-r1 | fail (0/38, pad) | baihe-g7-brk1 kimi-r0 | pass (38/38) |
| baihe-g7 | syn-nova-cand-c04-73057cd6 | kimi-r4 | fail (0/38, pad) | baihe-g7-brk1 kimi-r0 | pass (38/38) |
| baihe-g7 | syn-remark-lint-cand-c03-5e615329 | kimi-r1 | fail (0/13, pad) | baihe-g7-brk1 kimi-r0 | broken_again (0/13) |

## 4. Cost (LiteLLM usage.cost of v4's own replies, by the hub key of each attempt's leg)

| key | total | first attempts | re-runs | ATP-retried attempts | pilots | graded rollouts | $/graded rollout |
|---|---|---|---|---|---|---|---|
| NVAPI_KEY_NEW_2 | $3959.12 | $3629.4 | $280.4 | $46.04 | $3.27 | 1036 | 3.818 |
| NVAPI_KEY_NEW_3 | $0 | $0 | $0 | $0 | $0 | 0 | None |

batch $3955.84 + pilots $3.27 = $3959.12; per graded rollout $3.818; per final slot $4.078
by taskset: {"sel50": {"usd": 1011.23, "usd_per_final_slot": 4.045, "usd_per_first_attempt_mean": 3.64, "cached_pct": 89.0}, "rest47": {"usd": 1117.46, "usd_per_final_slot": 4.755, "usd_per_first_attempt_mean": 4.475, "cached_pct": 89.6}, "baihe": {"usd": 1827.15, "usd_per_final_slot": 3.767, "usd_per_first_attempt_mean": 3.439, "cached_pct": 90.6}}
key spend delta: {"NVAPI_KEY_NEW_2": {"from": "2026-10-28T14:14:11-0700", "spend_from": 846.11, "to": "2026-10-29T21:41:17-0700", "spend_to": 5210.41, "delta": 4364.3, "remaining": 4789.59}, "NVAPI_KEY_NEW_3": {"delta": 0.0, "note": "no v4 leg started on this key", "remaining_last_check": 4000.69}}
reconciliation NEW_2: {"key_delta": 4364.3, "v4_journal_usd_incl_pilots": 3959.12, "v5_journal_usd_incl_pilot": 390.04, "residual": 15.14, "note": "kimi-rollouts-v5 (official2, 2026-10-29 02:09-10:32) used the same key; its journal cost is from its analysis/final-report.json (read-only)"}
progress.py last row: {"time": "10-29 21:41", "spend": {"NVAPI_KEY_NEW_2": 4364.3, "NVAPI_KEY_NEW_3": 0.0}, "spent": 4364.3, "usd_per_rollout": 4.213}

## 5. SFT-usable passes: 745 of 765 graded passes

graded pass; evaluation completed; reward 1; every F2P and P2P test passing; no verifier or agent error; non-empty patch; exported session; no degenerate hub reply anywhere in the attempt (sub-sessions included) and no [PAD] / special-token text in the main session's assistant parts (nor in its tool outputs when the attempt had a degenerate reply). Tool-output matches without any degenerate reply are repository text: kept, flagged in notes. No LLM cheating audit.

| taskset | tasks | final passes | usable (first / re-run) | usable w/o compaction | excluded | tasks with usable | usable per task |
|---|---|---|---|---|---|---|---|
| sel50 | 50 | 182 | 175 (163 / 12) | 165 | 7 | 49 | {"0": 1, "1": 5, "2": 6, "3": 10, "4": 12, "5": 16} |
| rest47 | 47 | 188 | 184 (176 / 8) | 168 | 4 | 46 | {"0": 1, "2": 6, "3": 8, "4": 12, "5": 20} |
| baihe | 97 | 394 | 386 (361 / 25) | 368 | 9 | 97 | {"1": 6, "2": 6, "3": 11, "4": 35, "5": 39} |

excluded reasons: {"broken reply in the attempt": 20, "[PAD]/special text in the main session": 19, "ended by a broken reply": 1}

- excluded: baihe syn-nova-cand-c04-r1b0-afe620cd kimi-r4 (first, baihe-g0): broken reply in the attempt (pad@reply35/275 no-tool,pad@reply59/275; ended a sub-session); [PAD]/special text in the main session (assistant parts 1, tool outputs 1)
- excluded: baihe syn-risor-cand-c05-r1b2-e15a5178 kimi-r1 (first, baihe-g0): broken reply in the attempt (special@reply49/270; main session)
- excluded: baihe syn-mathjax-src-cand-c05-r1b1-68a272b0 kimi-r3 (first, baihe-g1): broken reply in the attempt (pad@reply180/183 no-tool; ended a sub-session); [PAD]/special text in the main session (assistant parts 1, tool outputs 0)
- excluded: baihe syn-python-prompt-toolkit-cand-c05-r1b0-2bee84d9 kimi-r2 (first, baihe-g2): broken reply in the attempt (pad@reply18/59; main session); [PAD]/special text in the main session (assistant parts 1, tool outputs 0)
- excluded: baihe syn-nova-cand-c02-78764589 kimi-r4 (first, baihe-g3): broken reply in the attempt (pad@reply40/179 no-tool; ended a sub-session); [PAD]/special text in the main session (assistant parts 0, tool outputs 1)
- excluded: baihe syn-risor-cand-c04-r1b3-767bbc8d kimi-r2 (first, baihe-g6): broken reply in the attempt (pad@reply2/112; main session); [PAD]/special text in the main session (assistant parts 1, tool outputs 0)
- excluded: baihe syn-risor-cand-c04-r1b3-767bbc8d kimi-r3 (first, baihe-g6): broken reply in the attempt (special@reply73/160; main session); [PAD]/special text in the main session (assistant parts 1, tool outputs 0)
- excluded: baihe syn-nova-cand-c04-73057cd6 kimi-r1 (first, baihe-g7): broken reply in the attempt (pad@reply215/328,pad@reply216/328; main session); [PAD]/special text in the main session (assistant parts 2, tool outputs 0)
- excluded: rest47 syn-risor-cand-c03-b214824f kimi-r1 (first, rest47-g0): broken reply in the attempt (pad@reply54/158 no-tool; ended a sub-session); [PAD]/special text in the main session (assistant parts 0, tool outputs 1)
- excluded: rest47 syn-transitions-cand-c04-r1b3-272dacdb kimi-r0 (first, rest47-g1): broken reply in the attempt (pad@reply78/126; main session); [PAD]/special text in the main session (assistant parts 1, tool outputs 1)
- excluded: rest47 syn-wasmtime-cand-c03-r1b4-4d3d43a0 kimi-r1 (first, rest47-g2): broken reply in the attempt (pad@reply22/170 no-tool; ended a sub-session); [PAD]/special text in the main session (assistant parts 0, tool outputs 1)
- excluded: rest47 syn-cerbos-cand-c05-r1b1-b283c14b kimi-r3 (first, rest47-g4): broken reply in the attempt (pad@reply84/89; main session); [PAD]/special text in the main session (assistant parts 1, tool outputs 0)
- excluded: sel50 syn-ibis-cand-c05-r1b3-9f713fee kimi-r2 (first, sel50-g0): broken reply in the attempt (pad@reply36/103; main session); [PAD]/special text in the main session (assistant parts 1, tool outputs 0)
- excluded: sel50 syn-wasmtime-cand-c03-353ec720 kimi-r3 (first, sel50-g0): broken reply in the attempt (pad@reply210/231,pad@reply211/231; main session); [PAD]/special text in the main session (assistant parts 2, tool outputs 0)
- excluded: sel50 syn-wasmtime-cand-c03-353ec720 kimi-r4 (first, sel50-g0): broken reply in the attempt (pad@reply175/213 no-tool; ended a sub-session); [PAD]/special text in the main session (assistant parts 1, tool outputs 0)
- excluded: sel50 syn-gomacro-cand-c01-c696c6ca kimi-r0 (first, sel50-g1): broken reply in the attempt (pad@reply126/286 no-tool; ended a sub-session); [PAD]/special text in the main session (assistant parts 1, tool outputs 0)
- excluded: sel50 syn-wasmtime-cand-c04-r1b1-bbb510ca kimi-r3 (first, sel50-g1): broken reply in the attempt (special@reply34/119; main session); [PAD]/special text in the main session (assistant parts 1, tool outputs 0)
- excluded: sel50 syn-yq-cand-c02-r1b4-c30118dd kimi-r2 (first, sel50-g2): broken reply in the attempt (pad@reply37/128,pad@reply38/128; main session); [PAD]/special text in the main session (assistant parts 2, tool outputs 0)
- excluded: sel50 syn-jest-cand-c05-r1b2-d6a136fe kimi-r3 (first, sel50-g4): broken reply in the attempt (pad@reply151/164; main session); [PAD]/special text in the main session (assistant parts 1, tool outputs 1)
- kept (note): rest47 syn-zsa-cand-c02-r1b3-38afb932 kimi-r2 (first, rest47-g1): [PAD]/special-token text in 1 tool output(s) of the main session, but no degenerate hub reply in the attempt: repository / command text, kept

## 6. Per task

| task | taskset | first pass | broken-ended | re-runs | final pass | v3 | 1M | reasoned % | compacted | max prompt | $/rollout | SFT-usable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-aiohttp-cand-c01-r1b4-9f77d81d | sel50 | 4/5 | - | - | 4/5 | 2/5 | 5/5 | 55.6 | 0 | 115197 | 1.96 | 4 |
| syn-bitecs-cand-c05-r1b0-85a91c3b | sel50 | 3/5 | - | - | 3/5 | 2/5 | 5/5 | 43.1 | 0 | 96962 | 1.37 | 3 |
| syn-dacite-cand-c05-r1b4-2e37bfb7 | sel50 | 5/5 | - | - | 5/5 | 4/5 | 4/5 | 53.0 | 0 | 76846 | 1.04 | 5 |
| syn-gpython-cand-c04-r1b4-0a5ca664 | sel50 | 5/5 | - | - | 5/5 | 1/5 | 5/5 | 54.2 | 3 | 242252 | 8.66 | 5 |
| syn-ibis-cand-c05-r1b3-9f713fee | sel50 | 4/5 | - | - | 4/5 | 0/5 | 2/5 | 58.0 | 0 | 156219 | 3.09 | 3 |
| syn-pylama-cand-c01-r1b4-r2b4-68d85058 | sel50 | 3/5 | - | - | 3/5 | 4/5 | 4/5 | 54.6 | 0 | 100840 | 1.82 | 3 |
| syn-risor-cand-c05-r1b2-3718460d | sel50 | 3/5 | kimi-r0 | kimi-r0:pass | 4/5 | 4/5 | 4/4 | 54.4 | 0 | 202873 | 4.9 | 4 |
| syn-vscode-sqltools-cand-c02-r1b0-fb4b293a | sel50 | 2/5 | kimi-r0 | kimi-r0:pass | 3/5 | 2/5 | 5/5 | 53.6 | 0 | 145540 | 1.74 | 3 |
| syn-wasmtime-cand-c03-353ec720 | sel50 | 3/5 | kimi-r1 | kimi-r1:pass | 4/5 | 2/5 | 4/4 | 51.6 | 5 | 253134 | 7.4 | 2 |
| syn-yq-cand-c02-r1b0-0af03743 | sel50 | 2/5 | kimi-r1 | kimi-r1:pass | 3/5 | 1/5 | 3/5 | 71.7 | 0 | 151484 | 3.08 | 3 |
| syn-aiohttp-cand-c03-r1b1-b706aac1 | sel50 | 3/5 | kimi-r1 kimi-r4 | kimi-r1:pass kimi-r4:pass | 5/5 | 3/5 | 5/5 | 53.2 | 0 | 109435 | 1.9 | 5 |
| syn-bitecs-cand-c05-r1b4-bc99c654 | sel50 | 5/5 | - | - | 5/5 | 5/5 | 3/5 | 51.1 | 0 | 112072 | 1.61 | 5 |
| syn-gomacro-cand-c01-c696c6ca | sel50 | 1/5 | - | - | 1/5 | 0/5 | 4/5 | 59.4 | 5 | 242844 | 9.74 | 0 |
| syn-ibis-cand-c03-r1b4-3d3b86b5 | sel50 | 2/5 | kimi-r1 | kimi-r1:fail | 2/5 | 1/5 | 3/5 | 53.5 | 0 | 197451 | 6.33 | 2 |
| syn-jest-cand-c01-r1b3-08931258 | sel50 | 3/5 | kimi-r0 | kimi-r0:fail | 3/5 | 0/5 | 3/5 | 51.8 | 0 | 240309 | 6.17 | 3 |
| syn-pylama-cand-c02-r1b0-r2b2-02688723 | sel50 | 1/5 | kimi-r3 | kimi-r3:pass | 2/5 | 1/5 | 3/5 | 59.8 | 0 | 108684 | 1.67 | 2 |
| syn-transitions-cand-c03-c910069f | sel50 | 4/5 | - | - | 4/5 | 2/5 | 4/5 | 62.0 | 0 | 225708 | 4.73 | 4 |
| syn-vscode-sqltools-cand-c02-r1b3-7119812a | sel50 | 4/5 | kimi-r3 | kimi-r3:pass | 5/5 | 2/5 | 3/5 | 53.7 | 0 | 139186 | 2.18 | 5 |
| syn-wasmtime-cand-c04-r1b1-bbb510ca | sel50 | 4/5 | - | - | 4/5 | 2/5 | 4/5 | 61.4 | 0 | 170989 | 4.09 | 3 |
| syn-yq-cand-c02-r1b1-2f81d284 | sel50 | 5/5 | - | - | 5/5 | 1/4 | 3/5 | 69.5 | 0 | 165649 | 3.64 | 5 |
| syn-aiohttp-cand-c04-7f733601 | sel50 | 1/5 | kimi-r0 | kimi-r0:fail | 1/5 | 0/5 | 3/5 | 57.7 | 0 | 191883 | 3.95 | 1 |
| syn-dacite-cand-c01-r1b3-84fe778b | sel50 | 4/5 | - | - | 4/5 | 2/5 | 4/5 | 51.6 | 0 | 106461 | 1.67 | 4 |
| syn-gomacro-cand-c02-r1b0-e8804dd9 | sel50 | 1/5 | - | - | 1/5 | 2/5 | 1/5 | 54.6 | 4 | 242121 | 8.11 | 1 |
| syn-ibis-cand-c04-86a17614 | sel50 | 4/5 | - | - | 4/5 | 1/5 | 4/5 | 63.0 | 0 | 192511 | 5.76 | 4 |
| syn-jest-cand-c04-r1b4-d8b1da89 | sel50 | 4/5 | kimi-r3 | kimi-r3:pass | 5/5 | 3/5 | 5/5 | 51.2 | 0 | 90895 | 1.62 | 5 |
| syn-pylama-cand-c03-r1b2-6182c7b3 | sel50 | 5/5 | - | - | 5/5 | 5/5 | 4/5 | 60.3 | 0 | 80126 | 1.09 | 5 |
| syn-transitions-cand-c04-r1b2-799317b1 | sel50 | 4/5 | - | - | 4/5 | 2/5 | 3/5 | 47.9 | 0 | 174314 | 3.73 | 4 |
| syn-vscode-sqltools-cand-c04-r1b3-0c604520 | sel50 | 4/5 | - | - | 4/5 | 2/5 | 4/5 | 53.0 | 0 | 163335 | 2.62 | 4 |
| syn-wasmtime-cand-c04-r1b4-786cced5 | sel50 | 5/5 | - | - | 5/5 | 1/5 | 5/5 | 59.4 | 0 | 200159 | 4.28 | 5 |
| syn-yq-cand-c02-r1b4-c30118dd | sel50 | 4/5 | - | - | 4/5 | 1/5 | 4/5 | 67.8 | 0 | 170642 | 3.3 | 3 |
| syn-aiohttp-cand-c05-r1b0-7d9ec1f5 | sel50 | 5/5 | - | - | 5/5 | 5/5 | 4/5 | 56.8 | 0 | 154051 | 2.73 | 5 |
| syn-dacite-cand-c02-r1b4-325c5f9d | sel50 | 3/5 | - | - | 3/5 | 3/5 | 4/5 | 48.4 | 0 | 116015 | 1.58 | 3 |
| syn-gomacro-cand-c02-r1b3-ed535f70 | sel50 | 4/5 | kimi-r0 | kimi-r0:fail | 4/5 | 1/5 | 2/5 | 56.4 | 5 | 245727 | 7.01 | 4 |
| syn-ibis-cand-c04-r1b0-r2b0-35888795 | sel50 | 2/5 | - | - | 2/5 | 4/5 | 3/5 | 59.3 | 0 | 172160 | 3.86 | 2 |
| syn-jest-cand-c05-9e1caaee | sel50 | 4/5 | kimi-r1 | kimi-r1:pass | 5/5 | 2/5 | 3/5 | 61.5 | 0 | 238476 | 7.5 | 5 |
| syn-pylama-cand-c03-r1b4-r2b4-58e91fc7 | sel50 | 1/5 | - | - | 1/5 | 2/5 | 3/5 | 56.1 | 0 | 101288 | 1.83 | 1 |
| syn-transitions-cand-c05-r1b2-6a1e0dca | sel50 | 5/5 | - | - | 5/5 | 4/5 | 5/5 | 51.3 | 0 | 165207 | 4.1 | 5 |
| syn-vscode-sqltools-cand-c05-r1b3-911416cb | sel50 | 1/5 | - | - | 1/5 | 3/5 | 4/5 | 49.0 | 0 | 155761 | 2.41 | 1 |
| syn-yj-cand-c01-7e8e179e | sel50 | 5/5 | - | - | 5/5 | 5/5 | 5/5 | 63.0 | 0 | 107886 | 1.68 | 5 |
| syn-yq-cand-c04-r1b4-cae9dfb2 | sel50 | 2/5 | - | - | 2/5 | 0/4 | 3/5 | 70.7 | 0 | 154764 | 3.01 | 2 |
| syn-argo-cd-cand-c05-cc3e1272 | sel50 | 1/5 | - | - | 1/5 | 4/5 | 3/5 | 55.6 | 0 | 236608 | 4.16 | 1 |
| syn-dacite-cand-c04-r1b0-eeb19c3a | sel50 | 3/5 | - | - | 3/5 | 5/5 | 5/5 | 47.5 | 0 | 121952 | 1.49 | 3 |
| syn-gomacro-cand-c04-r1b4-2d729ead | sel50 | 5/5 | - | - | 5/5 | 2/5 | 4/5 | 52.4 | 0 | 206615 | 4.42 | 5 |
| syn-ibis-cand-c05-r1b2-r2b3-269e70d9 | sel50 | 4/5 | - | - | 4/5 | 2/5 | 4/5 | 58.5 | 0 | 136485 | 3.37 | 4 |
| syn-jest-cand-c05-r1b2-d6a136fe | sel50 | 4/5 | kimi-r0 | kimi-r0:pass | 5/5 | 2/5 | 5/5 | 52.9 | 0 | 230337 | 4.9 | 4 |
| syn-pylama-cand-c05-r1b4-6eedf9a6 | sel50 | 2/5 | - | - | 2/5 | 4/5 | 3/5 | 49.5 | 0 | 90268 | 1.67 | 2 |
| syn-vscode-sqltools-cand-c01-r1b0-24c4a31b | sel50 | 4/5 | - | - | 4/5 | 3/5 | 5/5 | 53.6 | 0 | 135803 | 1.46 | 4 |
| syn-wasmtime-cand-c01-r1b1-82c9e518 | sel50 | 5/5 | - | - | 5/5 | 5/5 | 5/5 | 49.6 | 0 | 200090 | 5.1 | 5 |
| syn-yq-cand-c01-r1b4-c75b758f | sel50 | 5/5 | - | - | 5/5 | 5/5 | 5/5 | 68.3 | 0 | 132924 | 2.1 | 5 |
| syn-zsa-cand-c02-r1b1-ae1eb794 | sel50 | 3/5 | kimi-r0 kimi-r1 | kimi-r0:broken_again kimi-r1:pass | 4/5 | 2/5 | 4/5 | 51.9 | 0 | 205723 | 5.2 | 4 |
| syn-risor-cand-c03-b214824f | rest47 | 5/5 | - | - | 5/5 | - | - | 55.9 | 0 | 153391 | 3.08 | 4 |
| syn-pylama-cand-c04-r1b4-4d29f6ec | rest47 | 5/5 | - | - | 5/5 | - | - | 54.9 | 0 | 98676 | 1.62 | 5 |
| syn-dacite-cand-c04-r1b1-58e640ec | rest47 | 5/5 | - | - | 5/5 | - | - | 55.4 | 0 | 88740 | 1.24 | 5 |
| syn-litestar-cand-c03-r1b3-a019450a | rest47 | 5/5 | - | - | 5/5 | - | - | 50.3 | 0 | 193596 | 6.93 | 5 |
| syn-ibis-cand-c01-r1b4-r2b1-b677379a | rest47 | 5/5 | - | - | 5/5 | - | - | 55.3 | 0 | 158718 | 3.63 | 5 |
| syn-opengemini-cand-c01-r1b0-68a373fe | rest47 | 2/5 | - | - | 2/5 | - | - | 53.5 | 0 | 111420 | 1.74 | 2 |
| syn-gomacro-cand-c03-r1b4-67b6a828 | rest47 | 3/5 | - | - | 3/5 | - | - | 54.0 | 0 | 209614 | 6.22 | 3 |
| syn-transitions-cand-c02-r1b3-658108d1 | rest47 | 4/5 | - | - | 4/5 | - | - | 47.6 | 0 | 161133 | 3.46 | 4 |
| syn-jest-cand-c05-r1b0-5ec91bba | rest47 | 2/5 | kimi-r4 | kimi-r4:pass | 3/5 | - | - | 54.6 | 0 | 237120 | 6.75 | 3 |
| syn-yq-cand-c03-63063d88 | rest47 | 5/5 | - | - | 5/5 | - | - | 63.7 | 0 | 143830 | 2.89 | 5 |
| syn-risor-cand-c03-r1b2-e765ed15 | rest47 | 5/5 | - | - | 5/5 | - | - | 49.3 | 0 | 195714 | 4.82 | 5 |
| syn-pylama-cand-c05-r1b2-r2b2-a71ff500 | rest47 | 3/5 | - | - | 3/5 | - | - | 65.4 | 0 | 88713 | 1.32 | 3 |
| syn-zsa-cand-c02-r1b3-38afb932 | rest47 | 4/5 | - | - | 4/5 | - | - | 45.0 | 0 | 213495 | 5.22 | 4 |
| syn-litestar-cand-c04-r1b0-081c670c | rest47 | 5/5 | - | - | 5/5 | - | - | 49.3 | 0 | 227443 | 7.69 | 5 |
| syn-ibis-cand-c03-r1b0-r2b1-807ec5aa | rest47 | 5/5 | - | - | 5/5 | - | - | 55.7 | 0 | 149422 | 3.36 | 5 |
| syn-gomacro-cand-c02-c3c4135e | rest47 | 2/5 | - | - | 2/5 | - | - | 54.4 | 5 | 242475 | 10.82 | 2 |
| syn-gomacro-cand-c04-r1b2-b79686cb | rest47 | 4/5 | kimi-r0 | kimi-r0:pass | 5/5 | - | - | 54.4 | 5 | 242757 | 7.27 | 5 |
| syn-transitions-cand-c04-r1b3-272dacdb | rest47 | 3/5 | - | - | 3/5 | - | - | 46.6 | 0 | 177999 | 3.38 | 2 |
| syn-wasmtime-cand-c03-r1b2-03550fa4 | rest47 | 3/5 | kimi-r0 | kimi-r0:broken_again | 3/5 | - | - | 51.7 | 4 | 242252 | 5.69 | 3 |
| syn-yq-cand-c05-cbe61bd7 | rest47 | 5/5 | - | - | 5/5 | - | - | 59.2 | 0 | 212840 | 4.08 | 5 |
| syn-gpython-cand-c01-r1b1-765875d9 | rest47 | 3/5 | kimi-r1 | kimi-r1:pass | 4/5 | - | - | 58.3 | 0 | 231097 | 6.97 | 4 |
| syn-pylama-cand-c05-r1b2-r2b3-0e4a83d7 | rest47 | 4/5 | - | - | 4/5 | - | - | 58.3 | 0 | 80307 | 1.12 | 4 |
| syn-litestar-cand-c01-r1b1-8f1b686c | rest47 | 4/5 | kimi-r0 | kimi-r0:pass | 5/5 | - | - | 47.3 | 0 | 182207 | 4.12 | 5 |
| syn-litestar-cand-c05-r1b0-eaf6a5ae | rest47 | 4/5 | - | - | 4/5 | - | - | 44.4 | 0 | 219014 | 8.78 | 4 |
| syn-ibis-cand-c05-r1b2-r2b0-3eb64a03 | rest47 | 4/5 | - | - | 4/5 | - | - | 54.3 | 0 | 173935 | 4.59 | 4 |
| syn-gomacro-cand-c02-r1b2-002b4abf | rest47 | 3/5 | - | - | 3/5 | - | - | 54.3 | 2 | 246662 | 6.95 | 3 |
| syn-gomacro-cand-c05-r1b1-d80cf0a5 | rest47 | 2/5 | kimi-r1 | kimi-r1:pass | 3/5 | - | - | 55.9 | 1 | 242167 | 5.75 | 3 |
| syn-jest-cand-c05-r1b0-975858f4 | rest47 | 5/5 | - | - | 5/5 | - | - | 48.2 | 0 | 162922 | 4.27 | 5 |
| syn-wasmtime-cand-c03-r1b4-4d3d43a0 | rest47 | 3/5 | kimi-r4 | kimi-r4:pass | 4/5 | - | - | 54.5 | 0 | 212772 | 5.46 | 3 |
| syn-gpython-cand-c04-r1b1-e954ad27 | rest47 | 4/5 | kimi-r4 | kimi-r4:pass | 5/5 | - | - | 59.9 | 1 | 242504 | 5.44 | 5 |
| syn-dacite-cand-c01-r1b1-52fa75d9 | rest47 | 4/5 | - | - | 4/5 | - | - | 51.6 | 0 | 91197 | 1.12 | 4 |
| syn-litestar-cand-c02-r1b0-422e37a0 | rest47 | 4/5 | - | - | 4/5 | - | - | 49.7 | 0 | 205079 | 7.49 | 4 |
| syn-aiohttp-cand-c02-r1b0-3c976d3c | rest47 | 5/5 | - | - | 5/5 | - | - | 56.9 | 0 | 93201 | 1.99 | 5 |
| syn-cerbos-cand-c02-r1b4-9bbe0541 | rest47 | 4/5 | kimi-r4 | kimi-r4:pass | 5/5 | - | - | 52.1 | 0 | 138068 | 2.06 | 5 |
| syn-gomacro-cand-c03-f73e50a8 | rest47 | 3/5 | - | - | 3/5 | - | - | 57.2 | 5 | 242879 | 10.43 | 3 |
| syn-vscode-sqltools-cand-c01-r1b3-d7f6794b | rest47 | 5/5 | - | - | 5/5 | - | - | 54.5 | 0 | 159593 | 2.3 | 5 |
| syn-jest-cand-c05-r1b1-561067a1 | rest47 | 5/5 | - | - | 5/5 | - | - | 49.3 | 0 | 205467 | 5.13 | 5 |
| syn-wasmtime-cand-c05-r1b1-9fcf072e | rest47 | 5/5 | - | - | 5/5 | - | - | 53.3 | 0 | 180599 | 4.29 | 5 |
| syn-argo-cd-cand-c03-r1b0-18b78a50 | rest47 | 4/5 | - | - | 4/5 | - | - | 57.8 | 0 | 130621 | 1.86 | 4 |
| syn-dacite-cand-c02-r1b0-29dfe3b0 | rest47 | 5/5 | - | - | 5/5 | - | - | 48.5 | 0 | 97984 | 1.48 | 5 |
| syn-litestar-cand-c03-r1b0-a082f895 | rest47 | 4/5 | - | - | 4/5 | - | - | 52.2 | 0 | 195285 | 5.21 | 4 |
| syn-bitecs-cand-c05-r1b2-1b4c0811 | rest47 | 0/5 | - | - | 0/5 | - | - | 54.9 | 0 | 105594 | 1.34 | 0 |
| syn-cerbos-cand-c05-r1b1-b283c14b | rest47 | 5/5 | - | - | 5/5 | - | - | 50.4 | 0 | 136672 | 1.97 | 4 |
| syn-gomacro-cand-c03-r1b1-68058464 | rest47 | 2/5 | - | - | 2/5 | - | - | 51.0 | 0 | 240527 | 6.71 | 2 |
| syn-transitions-cand-c01-r1b1-24d9c1a8 | rest47 | 5/5 | - | - | 5/5 | - | - | 51.1 | 0 | 176593 | 4.18 | 5 |
| syn-jest-cand-c04-r1b0-9bf4fb26 | rest47 | 2/5 | - | - | 2/5 | - | - | 58.0 | 0 | 225026 | 5.86 | 2 |
| syn-yq-cand-c01-r1b3-b5f105ff | rest47 | 2/5 | kimi-r4 | kimi-r4:fail | 2/5 | - | - | 66.9 | 0 | 133527 | 2.2 | 2 |
| syn-badger-cand-c02-r1b3-91f6816d | baihe | 4/5 | - | - | 4/5 | - | - | 59.9 | 0 | 86256 | 0.84 | 4 |
| syn-dacite-cand-c01-r1b4-84845ff4 | baihe | 4/5 | - | - | 4/5 | - | - | 47.5 | 0 | 78531 | 1.16 | 4 |
| syn-go-jsonnet-cand-c05-r1b2-8838d109 | baihe | 4/5 | - | - | 4/5 | - | - | 63.5 | 0 | 157017 | 2.72 | 4 |
| syn-go-lua-cand-c03-r1b1-8bbc6eb9 | baihe | 4/5 | - | - | 4/5 | - | - | 60.9 | 0 | 113797 | 1.67 | 4 |
| syn-gval-cand-c05-r1b0-9669b6c7 | baihe | 3/5 | - | - | 3/5 | - | - | 45.1 | 0 | 98259 | 1.42 | 3 |
| syn-inquirer-js-cand-c05-r1b3-34d7fad4 | baihe | 1/5 | - | - | 1/5 | - | - | 51.2 | 0 | 184810 | 4.14 | 1 |
| syn-mathjax-src-cand-c05-r1b0-bf6190e1 | baihe | 3/5 | kimi-r0 kimi-r4 | kimi-r0:pass kimi-r4:broken_again | 4/5 | - | - | 56.9 | 0 | 179650 | 3.34 | 4 |
| syn-nova-cand-c04-r1b0-afe620cd | baihe | 4/5 | kimi-r0 | kimi-r0:pass | 5/5 | - | - | 44.0 | 5 | 243306 | 6.26 | 4 |
| syn-python-prompt-toolkit-cand-c02-r1b0-2293d236 | baihe | 4/5 | - | - | 4/5 | - | - | 51.0 | 0 | 123406 | 1.98 | 4 |
| syn-remark-lint-cand-c03-r1b3-ecc843e0 | baihe | 5/5 | - | - | 5/5 | - | - | 50.3 | 0 | 156330 | 2.9 | 5 |
| syn-risor-cand-c01-r1b4-bde52eba | baihe | 3/5 | kimi-r4 | kimi-r4:pass | 4/5 | - | - | 55.7 | 0 | 203354 | 4.85 | 4 |
| syn-risor-cand-c05-r1b2-e15a5178 | baihe | 2/5 | kimi-r0 kimi-r3 | kimi-r0:pass kimi-r3:fail | 3/5 | - | - | 53.3 | 4 | 242329 | 7.8 | 2 |
| syn-urllib3-cand-c05-r1b4-28cff21e | baihe | 4/5 | - | - | 4/5 | - | - | 52.6 | 0 | 81562 | 1.23 | 4 |
| syn-badger-cand-c03-r1b3-0993435b | baihe | 4/5 | - | - | 4/5 | - | - | 57.3 | 0 | 68139 | 0.67 | 4 |
| syn-dacite-cand-c02-r1b0-a325e28a | baihe | 2/5 | kimi-r1 kimi-r4 | kimi-r1:fail kimi-r4:pass | 3/5 | - | - | 52.5 | 0 | 104983 | 1.43 | 3 |
| syn-go-jsonnet-cand-c05-r1b3-69f0af17 | baihe | 1/5 | - | - | 1/5 | - | - | 55.2 | 0 | 184450 | 3.32 | 1 |
| syn-go-lua-cand-c03-r1b3-6e43e3c7 | baihe | 5/5 | - | - | 5/5 | - | - | 59.8 | 0 | 141496 | 2.36 | 5 |
| syn-gval-cand-c05-r1b1-56f73cd4 | baihe | 4/5 | - | - | 4/5 | - | - | 50.9 | 0 | 121266 | 1.8 | 4 |
| syn-lucid-cand-c04-r1b1-44b0d57a | baihe | 1/5 | kimi-r3 | kimi-r3:fail | 1/5 | - | - | 53.1 | 0 | 127688 | 2.36 | 1 |
| syn-mathjax-src-cand-c05-r1b1-68a272b0 | baihe | 5/5 | - | - | 5/5 | - | - | 54.2 | 1 | 224430 | 5.64 | 4 |
| syn-pyserde-cand-c01-aed9b543 | baihe | 4/5 | kimi-r2 | kimi-r2:pass | 5/5 | - | - | 56.8 | 0 | 88170 | 1.2 | 5 |
| syn-python-prompt-toolkit-cand-c04-63029829 | baihe | 3/5 | kimi-r3 | kimi-r3:pass | 4/5 | - | - | 49.5 | 0 | 115621 | 1.79 | 4 |
| syn-remark-lint-cand-c03-r1b4-a0f28434 | baihe | 2/5 | kimi-r1 | kimi-r1:broken_again | 2/5 | - | - | 59.7 | 0 | 201835 | 3.82 | 2 |
| syn-risor-cand-c02-r1b1-0ccf905c | baihe | 4/5 | kimi-r2 | kimi-r2:pass | 5/5 | - | - | 50.7 | 0 | 211559 | 5.26 | 5 |
| syn-risor-cand-c05-r1b3-e0a6aa76 | baihe | 4/5 | kimi-r0 | kimi-r0:pass | 5/5 | - | - | 55.0 | 0 | 212049 | 5.83 | 5 |
| syn-checkov-cand-c03-r1b4-69f4564c | baihe | 1/5 | - | - | 1/5 | - | - | 49.2 | 0 | 128194 | 2.08 | 1 |
| syn-dacite-cand-c02-r1b4-b4d8de8a | baihe | 4/5 | - | - | 4/5 | - | - | 44.6 | 0 | 96001 | 1.44 | 4 |
| syn-go-jsonnet-cand-c05-r1b4-6d66c2db | baihe | 4/5 | kimi-r1 | kimi-r1:broken_again | 4/5 | - | - | 59.0 | 0 | 194544 | 3.6 | 4 |
| syn-go-lua-cand-c04-r1b1-5f114064 | baihe | 4/5 | - | - | 4/5 | - | - | 60.9 | 0 | 124135 | 1.72 | 4 |
| syn-inquirer-js-cand-c01-r1b2-5d163477 | baihe | 4/5 | - | - | 4/5 | - | - | 52.9 | 0 | 192367 | 4.23 | 4 |
| syn-mathjax-src-cand-c01-r1b3-495b5657 | baihe | 5/5 | - | - | 5/5 | - | - | 54.7 | 0 | 203421 | 5.55 | 5 |
| syn-mathjax-src-cand-c05-r1b4-079eb9ad | baihe | 5/5 | - | - | 5/5 | - | - | 53.0 | 0 | 184081 | 4.17 | 5 |
| syn-pyserde-cand-c01-r1b0-ccf26101 | baihe | 4/5 | - | - | 4/5 | - | - | 54.0 | 0 | 176429 | 2.86 | 4 |
| syn-python-prompt-toolkit-cand-c05-r1b0-2bee84d9 | baihe | 4/5 | - | - | 4/5 | - | - | 47.2 | 0 | 114584 | 1.25 | 3 |
| syn-remark-lint-cand-c04-r1b3-93c239cc | baihe | 5/5 | - | - | 5/5 | - | - | 62.0 | 0 | 208682 | 4.31 | 5 |
| syn-risor-cand-c02-r1b2-8e2a3134 | baihe | 5/5 | - | - | 5/5 | - | - | 57.5 | 0 | 219820 | 6.97 | 5 |
| syn-risor-cand-c05-r1b4-1ee392ce | baihe | 3/5 | kimi-r1 | kimi-r1:pass | 4/5 | - | - | 53.5 | 4 | 242636 | 7.54 | 4 |
| syn-citty-cand-c01-r1b0-6fa08bef | baihe | 4/5 | - | - | 4/5 | - | - | 50.2 | 0 | 90550 | 1.38 | 4 |
| syn-dacite-cand-c03-r1b4-6d646048 | baihe | 5/5 | - | - | 5/5 | - | - | 53.6 | 0 | 104187 | 1.34 | 5 |
| syn-go-lua-cand-c01-r1b2-ede51ab6 | baihe | 4/5 | - | - | 4/5 | - | - | 50.1 | 0 | 159756 | 3.13 | 4 |
| syn-go-lua-cand-c05-r1b3-aa069d83 | baihe | 3/5 | - | - | 3/5 | - | - | 59.3 | 0 | 123039 | 1.54 | 3 |
| syn-inquirer-js-cand-c02-r1b0-9d85623f | baihe | 4/5 | - | - | 4/5 | - | - | 57.4 | 0 | 169375 | 3.78 | 4 |
| syn-mathjax-src-cand-c02-r1b0-3bf1ffa4 | baihe | 3/5 | kimi-r3 | kimi-r3:pass | 4/5 | - | - | 50.8 | 0 | 185250 | 3.76 | 4 |
| syn-nova-cand-c02-78764589 | baihe | 3/5 | - | - | 3/5 | - | - | 54.4 | 3 | 244757 | 7.04 | 2 |
| syn-pyserde-cand-c01-r1b4-f0120cd2 | baihe | 4/5 | - | - | 4/5 | - | - | 42.2 | 0 | 193046 | 4.14 | 4 |
| syn-remark-lint-cand-c01-r1b0-a55630d0 | baihe | 1/5 | kimi-r0 | kimi-r0:broken_again | 1/5 | - | - | 57.6 | 0 | 202244 | 3.63 | 1 |
| syn-remark-lint-cand-c05-496f733f | baihe | 5/5 | - | - | 5/5 | - | - | 61.0 | 0 | 185643 | 3.7 | 5 |
| syn-risor-cand-c02-r1b4-4f19eaca | baihe | 2/5 | - | - | 2/5 | - | - | 58.7 | 1 | 242102 | 7.58 | 2 |
| syn-urllib3-cand-c01-r1b3-c55145a7 | baihe | 4/5 | - | - | 4/5 | - | - | 55.5 | 0 | 177011 | 2.86 | 4 |
| syn-citty-cand-c01-r1b4-66b0e024 | baihe | 5/5 | - | - | 5/5 | - | - | 46.8 | 0 | 69606 | 0.97 | 5 |
| syn-django-recurrence-cand-c05-r1b3-0df4afde | baihe | 5/5 | - | - | 5/5 | - | - | 54.4 | 0 | 81199 | 0.97 | 5 |
| syn-go-lua-cand-c01-r1b3-c9a601af | baihe | 4/5 | kimi-r3 | kimi-r3:pass | 5/5 | - | - | 60.9 | 0 | 124009 | 1.49 | 5 |
| syn-gval-cand-c01-r1b0-b0a98783 | baihe | 4/5 | kimi-r3 | kimi-r3:pass | 5/5 | - | - | 52.1 | 0 | 107340 | 1.3 | 5 |
| syn-inquirer-js-cand-c03-r1b0-0548d9f2 | baihe | 5/5 | kimi-r1 | kimi-r1:fail | 4/5 | - | - | 57.6 | 0 | 201521 | 4.36 | 4 |
| syn-mathjax-src-cand-c02-r1b3-2cb17867 | baihe | 5/5 | - | - | 5/5 | - | - | 54.9 | 0 | 236153 | 5.68 | 5 |
| syn-nova-cand-c02-r1b0-40b314b3 | baihe | 4/5 | kimi-r2 | kimi-r2:broken_again | 4/5 | - | - | 43.4 | 0 | 180723 | 4.33 | 4 |
| syn-pyserde-cand-c02-r1b1-e4743457 | baihe | 4/5 | kimi-r0 | kimi-r0:pass | 5/5 | - | - | 51.5 | 0 | 190377 | 3.4 | 5 |
| syn-remark-lint-cand-c01-r1b1-dc064647 | baihe | 4/5 | - | - | 4/5 | - | - | 59.2 | 0 | 182744 | 3.85 | 4 |
| syn-remark-lint-cand-c05-r1b0-f0523255 | baihe | 4/5 | - | - | 4/5 | - | - | 63.4 | 0 | 155305 | 3.08 | 4 |
| syn-risor-cand-c03-r1b3-3380be16 | baihe | 2/5 | kimi-r0 | kimi-r0:fail | 2/5 | - | - | 56.2 | 0 | 234061 | 4.84 | 2 |
| syn-urllib3-cand-c02-5415bd10 | baihe | 5/5 | - | - | 5/5 | - | - | 54.3 | 0 | 66628 | 0.85 | 5 |
| syn-citty-cand-c02-r1b4-6fb61416 | baihe | 4/5 | - | - | 4/5 | - | - | 50.5 | 0 | 75795 | 1.04 | 4 |
| syn-go-jsonnet-cand-c03-r1b3-98110934 | baihe | 4/5 | - | - | 4/5 | - | - | 51.2 | 0 | 167398 | 2.33 | 4 |
| syn-go-lua-cand-c02-41f65269 | baihe | 4/5 | kimi-r3 | kimi-r3:pass | 5/5 | - | - | 61.5 | 0 | 182075 | 2.68 | 5 |
| syn-gval-cand-c03-r1b4-236f69c1 | baihe | 5/5 | - | - | 5/5 | - | - | 48.7 | 0 | 97187 | 1.22 | 5 |
| syn-inquirer-js-cand-c03-r1b3-0365aa3a | baihe | 3/5 | - | - | 3/5 | - | - | 47.8 | 0 | 169167 | 3.51 | 3 |
| syn-mathjax-src-cand-c02-r1b4-da3483cd | baihe | 4/5 | kimi-r0 | kimi-r0:pass | 5/5 | - | - | 55.8 | 0 | 182226 | 3.46 | 5 |
| syn-nova-cand-c03-7463f7ee | baihe | 5/5 | - | - | 5/5 | - | - | 57.6 | 0 | 230807 | 7.34 | 5 |
| syn-pytest-cand-c01-r1b0-954445d0 | baihe | 2/5 | kimi-r1 kimi-r3 | kimi-r1:fail kimi-r3:fail | 2/5 | - | - | 62.7 | 1 | 242326 | 5.37 | 2 |
| syn-remark-lint-cand-c01-r1b3-32f4b57c | baihe | 3/5 | - | - | 3/5 | - | - | 62.3 | 0 | 220185 | 5.82 | 3 |
| syn-risor-cand-c01-e8bef1af | baihe | 5/5 | - | - | 5/5 | - | - | 57.1 | 0 | 105594 | 1.96 | 5 |
| syn-risor-cand-c04-r1b1-eab3d6e4 | baihe | 3/5 | kimi-r2 | kimi-r2:broken_again | 3/5 | - | - | 51.4 | 0 | 241302 | 7.52 | 3 |
| syn-urllib3-cand-c02-r1b0-e164e517 | baihe | 5/5 | - | - | 5/5 | - | - | 48.4 | 0 | 121242 | 1.81 | 5 |
| syn-citty-cand-c04-a54df7c6 | baihe | 5/5 | - | - | 5/5 | - | - | 51.8 | 0 | 88229 | 1.06 | 5 |
| syn-go-jsonnet-cand-c03-r1b4-9e83e729 | baihe | 5/5 | - | - | 5/5 | - | - | 59.2 | 0 | 193099 | 3.54 | 5 |
| syn-go-lua-cand-c02-r1b3-c6a1bad0 | baihe | 5/5 | - | - | 5/5 | - | - | 55.5 | 0 | 126537 | 1.8 | 5 |
| syn-gval-cand-c04-r1b2-f217ee32 | baihe | 5/5 | - | - | 5/5 | - | - | 46.2 | 0 | 98159 | 1.21 | 5 |
| syn-inquirer-js-cand-c04-r1b0-53edb73c | baihe | 4/5 | - | - | 4/5 | - | - | 55.7 | 0 | 169943 | 3.43 | 4 |
| syn-mathjax-src-cand-c03-r1b0-4e89d622 | baihe | 4/5 | kimi-r2 | kimi-r2:pass | 5/5 | - | - | 53.6 | 0 | 196982 | 4.13 | 5 |
| syn-nova-cand-c03-r1b0-cb0355d9 | baihe | 4/5 | kimi-r4 | kimi-r4:pass | 5/5 | - | - | 48.5 | 4 | 241702 | 7.37 | 5 |
| syn-pytest-cand-c02-69e6073d | baihe | 3/5 | kimi-r2 kimi-r4 | kimi-r2:pass kimi-r4:pass | 5/5 | - | - | 59.3 | 0 | 223393 | 4.4 | 5 |
| syn-remark-lint-cand-c01-r1b4-b6f41331 | baihe | 2/5 | kimi-r4 | kimi-r4:pass | 3/5 | - | - | 55.2 | 0 | 158937 | 2.87 | 3 |
| syn-risor-cand-c01-r1b0-e54193d4 | baihe | 4/5 | kimi-r4 | kimi-r4:pass | 5/5 | - | - | 48.1 | 0 | 227380 | 5.39 | 5 |
| syn-risor-cand-c04-r1b3-767bbc8d | baihe | 5/5 | - | - | 5/5 | - | - | 54.4 | 0 | 221755 | 4.68 | 3 |
| syn-urllib3-cand-c03-r1b2-8d41fa44 | baihe | 5/5 | - | - | 5/5 | - | - | 48.1 | 0 | 142520 | 2.66 | 5 |
| syn-citty-cand-c04-r1b1-ec1c0299 | baihe | 4/5 | kimi-r0 | kimi-r0:pass | 5/5 | - | - | 48.8 | 0 | 138291 | 2.46 | 5 |
| syn-go-jsonnet-cand-c04-r1b2-b0655666 | baihe | 1/5 | - | - | 1/5 | - | - | 58.0 | 0 | 125801 | 1.62 | 1 |
| syn-go-lua-cand-c03-r1b0-856c42f7 | baihe | 5/5 | - | - | 5/5 | - | - | 56.6 | 0 | 145783 | 2.21 | 5 |
| syn-gval-cand-c04-r1b3-c8d55378 | baihe | 5/5 | - | - | 5/5 | - | - | 54.2 | 0 | 91291 | 1.1 | 5 |
| syn-inquirer-js-cand-c05-r1b0-a34e3650 | baihe | 2/5 | kimi-r1 | kimi-r1:pass | 3/5 | - | - | 52.8 | 0 | 227325 | 4.94 | 3 |
| syn-mathjax-src-cand-c04-r1b3-2d9ff806 | baihe | 5/5 | - | - | 5/5 | - | - | 53.1 | 0 | 198567 | 4.71 | 5 |
| syn-nova-cand-c04-73057cd6 | baihe | 4/5 | kimi-r4 | kimi-r4:pass | 5/5 | - | - | 54.6 | 5 | 242046 | 7.34 | 4 |
| syn-python-prompt-toolkit-cand-c01-r1b0-48067663 | baihe | 4/5 | - | - | 4/5 | - | - | 47.4 | 0 | 98715 | 1.31 | 4 |
| syn-remark-lint-cand-c03-5e615329 | baihe | 4/5 | kimi-r1 | kimi-r1:broken_again | 4/5 | - | - | 56.6 | 0 | 218534 | 4.0 | 4 |
| syn-risor-cand-c01-r1b1-ec426125 | baihe | 3/5 | - | - | 3/5 | - | - | 56.9 | 0 | 191701 | 4.8 | 3 |
| syn-risor-cand-c04-r1b4-275adcf4 | baihe | 5/5 | - | - | 5/5 | - | - | 51.4 | 0 | 223743 | 6.87 | 5 |
| syn-urllib3-cand-c04-r1b3-a525714c | baihe | 5/5 | - | - | 5/5 | - | - | 60.6 | 0 | 137749 | 2.2 | 5 |

Full SFT-usable list with attempt dirs and export paths: analysis/sft-usable-passes.jsonl.
