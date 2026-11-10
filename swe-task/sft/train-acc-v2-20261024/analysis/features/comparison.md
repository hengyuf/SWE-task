| metric | base (sc14) | v2 SFT 5 ep (iter38) | v2 SFT 10 ep (iter76) | v2 SFT 20 ep (iter152) | v2 empty-think-prefix 5 ep (iter38) | v2 empty-think-prefix 10 ep (iter76) | v2 empty-think-prefix 20 ep (iter152) | v3 empty-think-prefix 5 ep (iter71) | v3 empty-think-prefix 10 ep (iter142) |
|---|---|---|---|---|---|---|---|---|---|
| rollouts | 250 | 250 | 250 | 250 | 250 | 250 | 250 | 250 | 250 |
| graded | 250 | 249 | 250 | 250 | 249 | 249 | 250 | 250 | 245 |
| infra | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 5 |
| with_export | 250 | 250 | 250 | 250 | 249 | 250 | 250 | 250 | 248 |
| pass_at_1 | 17.2 | 18.1 | 36.0 | 44.0 | 26.9 | 39.4 | 52.8 | 44.0 | 56.7 |
| task_mean_pass | 17.2 | 18.0 | 36.0 | 44.0 | 26.8 | 39.2 | 52.8 | 44.0 | 56.5 |
| pass_at_5 | 46.0 | 52.0 | 74.0 | 74.0 | 68.0 | 80.0 | 84.0 | 82.0 | 90.0 |
| tasks | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 |
| turns_per_rollout_mean | 245.16 | 287.62 | 269.97 | 234.39 | 331.96 | 273.31 | 211.42 | 256.78 | 205.31 |
| turns_per_rollout_median | 207.5 | 243.5 | 241.5 | 190.0 | 299 | 230.5 | 175.0 | 207.5 | 171.5 |
| pct_turns_with_reasoning | 100.0 | 64.7 | 59.3 | 63.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| pct_action_turns_with_reasoning | 100.0 | 64.7 | 59.3 | 63.1 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| pct_rollouts_all_actions_reasoned | 90.8 | 0.4 | 0.4 | 0.0 | 89.2 | 99.2 | 99.2 | 98.4 | 98.4 |
| rollout_mean_pct_action_reasoned | 100.0 | 64.4 | 59.4 | 62.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| longest_unreasoned_action_run_median | 0.0 | 5.0 | 6.0 | 5.0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| longest_unreasoned_action_run_mean | 0.09 | 5.74 | 5.88 | 5.28 | 0.11 | 0.01 | 0.01 | 0.02 | 0.02 |
| longest_unreasoned_action_run_p90 | 0 | 8 | 9 | 8 | 1 | 0 | 0 | 0 | 0 |
| longest_unreasoned_action_run_max | 1 | 18 | 20 | 15 | 1 | 1 | 1 | 1 | 1 |
| rollout_pct_action_reasoned_p10 | 100.0 | 56.8 | 51.7 | 56.1 | 99.8 | 100.0 | 100.0 | 100.0 | 100.0 |
| rollout_pct_action_reasoned_p50 | 100.0 | 64.0 | 59.1 | 62.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| rollout_pct_action_reasoned_p90 | 100.0 | 72.0 | 66.5 | 70.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| rollout_pct_action_reasoned_max | 100.0 | 100.0 | 100.0 | 78.6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| pct_rollouts_action_reasoned_ge_90 | 100.0 | 0.4 | 0.8 | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| pct_first_action_reasoned | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| reasoning_tokens_per_reasoning_turn_median | 126.0 | 210 | 241.0 | 196.0 | 137.0 | 130 | 135.0 | 130.0 | 134 |
| reasoning_tokens_per_reasoning_turn_mean | 658.55 | 757.31 | 875.05 | 852.51 | 568.44 | 564.81 | 662.44 | 585.47 | 644.62 |
| reasoning_chars_per_reasoning_turn_median | 450.0 | 778 | 885.0 | 730.0 | 508.0 | 479 | 502.0 | 481.0 | 496 |
| reasoning_tokens_per_turn_mean_incl_zero | 658.2 | 489.9 | 518.9 | 537.5 | 568.2 | 564.7 | 662.4 | 585.4 | 644.5 |
| reasoning_tokens_per_rollout_mean | 161375.15 | 140905.21 | 140098.91 | 125987.35 | 188611.1 | 154344.81 | 140040.16 | 150316.94 | 132325.91 |
| reasoning_tokens_per_rollout_median | 125790.5 | 100577.5 | 101566.5 | 85847.5 | 138365 | 114755.0 | 99445.5 | 112863.0 | 95750.0 |
| pct_turns_with_visible_text | 5.5 | 32.0 | 32.5 | 28.8 | 39.1 | 40.1 | 37.1 | 39.0 | 41.7 |
| pct_action_turns_with_visible_text | 5.1 | 31.9 | 32.3 | 28.5 | 39.0 | 40.0 | 36.8 | 38.8 | 41.5 |
| pct_action_turns_bare | 0.0 | 35.0 | 40.3 | 36.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| visible_chars_median | 83 | 89.0 | 87.0 | 88.0 | 88 | 79 | 89 | 85 | 85.0 |
| tool_calls_per_turn | 1.08 | 1.09 | 1.09 | 1.09 | 1.11 | 1.08 | 1.1 | 1.12 | 1.14 |
| tool_calls_per_action_turn | 1.09 | 1.09 | 1.1 | 1.1 | 1.11 | 1.08 | 1.11 | 1.13 | 1.15 |
| pct_turns_parallel_calls | 7.3 | 7.4 | 7.6 | 8.2 | 8.6 | 6.8 | 8.4 | 9.8 | 11.6 |
| tool_calls_per_rollout_mean | 265.68 | 313.02 | 294.44 | 256.22 | 367.81 | 295.09 | 233.17 | 288.65 | 234.89 |
| pct_turns_notool | 0.4 | 0.4 | 0.4 | 0.5 | 0.3 | 0.4 | 0.5 | 0.4 | 0.5 |
| pct_turns_reasoning_only | 0.0 | 0.1 | 0.1 | 0.1 | 0.0 | 0.0 | 0.1 | 0.0 | 0.1 |
| pct_turns_text_notool | 0.4 | 0.2 | 0.3 | 0.4 | 0.3 | 0.3 | 0.4 | 0.4 | 0.5 |
| pct_turns_empty | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| pct_rollouts_git_commit | 93.6 | 49.2 | 62.0 | 73.6 | 75.9 | 82.0 | 91.2 | 89.6 | 88.7 |
| pct_rollouts_commit_in_last_3_calls | 90.4 | 48.0 | 60.4 | 69.6 | 75.1 | 80.4 | 90.8 | 88.8 | 87.9 |
| pct_rollouts_branch_created | 100.0 | 97.2 | 97.2 | 98.8 | 98.4 | 98.0 | 99.2 | 100.0 | 98.8 |
| pct_rollouts_final_notool_message | 100.0 | 99.6 | 99.6 | 99.2 | 98.0 | 98.4 | 99.6 | 98.8 | 99.2 |
| pct_rollouts_final_summary_200 | 96.4 | 65.2 | 86.8 | 88.4 | 85.9 | 90.4 | 94.8 | 94.0 | 91.9 |
| final_message_chars_median | 1901.5 | 1455 | 1749 | 1874.5 | 1594.5 | 1784.5 | 1756 | 1569 | 1602.0 |
| pct_rollouts_final_reasoning | 98.0 | 65.1 | 62.2 | 56.5 | 95.5 | 96.3 | 99.2 | 97.6 | 98.0 |
| pct_rollouts_todowrite | 78.8 | 79.2 | 82.8 | 76.4 | 83.1 | 71.6 | 80.0 | 84.0 | 83.5 |
| todowrite_calls_per_rollout_mean | 1.36 | 1.8 | 2.02 | 1.76 | 1.53 | 1.36 | 1.81 | 1.61 | 1.75 |
| pct_rollouts_task_subagent | 6.0 | 5.2 | 6.0 | 4.4 | 7.6 | 2.4 | 8.8 | 7.6 | 11.3 |
| task_calls_per_rollout_mean | 0.09 | 0.06 | 0.07 | 0.07 | 0.1 | 0.03 | 0.14 | 0.1 | 0.15 |
| pct_rollouts_webfetch | 18.8 | 13.2 | 14.0 | 11.2 | 16.1 | 13.2 | 10.0 | 11.6 | 9.7 |
| webfetch_calls_total | 80 | 56 | 56 | 41 | 71 | 44 | 31 | 39 | 33 |
| pct_rollouts_skill | 0.0 | 0.0 | 0.0 | 0.0 | 0.4 | 0.0 | 0.0 | 0.0 | 0.0 |
| compactions_per_rollout_mean | 0.82 | 0.91 | 0.89 | 0.66 | 1.24 | 0.95 | 0.71 | 0.87 | 0.7 |
| pct_rollouts_compacted | 58.0 | 58.0 | 62.4 | 48.4 | 74.7 | 63.2 | 50.8 | 59.6 | 52.8 |
| agent_wall_min_median | 52.9 | 60.9 | 54.8 | 43.8 | 74.0 | 66.7 | 51.1 | 54.6 | 52.5 |
| patch_empty_pct | 6.4 | 51.0 | 38.0 | 26.4 | 24.1 | 17.7 | 8.4 | 10.4 | 10.2 |
| pct_unreasoned_action_after_reasoned | 100.0 | 62.8 | 60.1 | 62.7 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| tool_calls_per_rollout_median | 230.0 | 270.5 | 267.0 | 211.0 | 327 | 253.5 | 195.5 | 236.0 | 205.0 |
| action_turns_per_rollout_mean | 244.16 | 286.58 | 268.9 | 233.3 | 330.92 | 272.27 | 210.34 | 255.75 | 204.26 |
| open_think_turns_total | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| reasoning_tokens_per_reasoning_turn_p10 | 20 | 33 | 40 | 35 | 26 | 26 | 26 | 26 | 26 |
| reasoning_tokens_per_reasoning_turn_p25 | 42 | 73 | 88 | 72 | 51 | 51 | 52 | 51 | 52 |
| reasoning_tokens_per_reasoning_turn_p75 | 496 | 643 | 692 | 596 | 441 | 406 | 437 | 414 | 415 |
| reasoning_tokens_per_reasoning_turn_p90 | 1479 | 1616 | 1776 | 1569 | 1235 | 1130 | 1302 | 1218 | 1259 |
| reasoning_tokens_per_reasoning_turn_p99 | 7793 | 8902 | 10110 | 9267 | 6469 | 6713 | 8180 | 7118 | 8267 |
| pct_reasoning_turns_lt_20_tokens | 9.7 | 4.0 | 2.6 | 3.3 | 5.7 | 5.6 | 5.5 | 5.5 | 5.5 |
| pct_reasoning_turns_lt_50_tokens | 28.7 | 16.8 | 13.4 | 16.7 | 24.3 | 24.3 | 23.7 | 24.1 | 23.6 |
| hub_output_tokens_per_rollout_mean | 208005.99 | 212668.95 | 202564.78 | 184066.8 | 273971.98 | 221975.64 | 195328.98 | 216909.49 | 186368.34 |
| agent_wall_min_mean | 61.4 | 69.3 | 62.2 | 53.5 | 85.2 | 77.2 | 61.4 | 65.8 | 58.4 |
| agent_wall_min_p90 | 100.4 | 138.6 | 107.8 | 102.0 | 164.6 | 136.3 | 106.1 | 108.7 | 99.1 |
| rollouts_deadline_timeout | 4 | 8 | 7 | 4 | 16 | 12 | 3 | 7 | 4 |
| turn_class_mix: action+reasoning | 99.5 | 64.5 | 59.0 | 62.8 | 99.7 | 99.6 | 99.5 | 99.6 | 99.5 |
| turn_class_mix: action, no reasoning | 0.0 | 35.2 | 40.6 | 36.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| turn_class_mix: no-tool reasoning-only | 0.0 | 0.1 | 0.1 | 0.1 | 0.0 | 0.0 | 0.1 | 0.0 | 0.1 |
| turn_class_mix: no-tool text | 0.4 | 0.2 | 0.3 | 0.4 | 0.3 | 0.3 | 0.4 | 0.4 | 0.5 |
| turn_class_mix: no-tool empty | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 95% CI (task bootstrap): pct_action_turns_with_reasoning | [99.9, 100.0] | [63.6, 65.8] | [58.2, 60.4] | [61.9, 64.3] | [100.0, 100.0] | [100.0, 100.0] | [100.0, 100.0] | [100.0, 100.0] | [100.0, 100.0] |
| 95% CI (task bootstrap): pct_action_turns_with_visible_text | [4.4, 5.8] | [30.5, 33.3] | [30.9, 33.8] | [27.4, 29.8] | [37.5, 40.6] | [38.6, 41.4] | [35.4, 38.2] | [37.7, 40.1] | [40.1, 42.9] |
| 95% CI (task bootstrap): pct_turns_parallel_calls | [6.3, 8.4] | [6.6, 8.3] | [6.4, 8.8] | [7.1, 9.6] | [7.8, 9.5] | [5.8, 8.0] | [7.4, 9.5] | [8.8, 11.1] | [10.1, 13.2] |
| diff_pct_action_reasoned_vs_sc14 (paired task bootstrap) | - | {'diff': -35.3, 'ci95': [-36.4, -34.2], 'tasks': 50} | {'diff': -40.7, 'ci95': [-41.8, -39.6], 'tasks': 50} | {'diff': -36.9, 'ci95': [-38.1, -35.7], 'tasks': 50} | {'diff': 0.0, 'ci95': [-0.0, 0.0], 'tasks': 50} | {'diff': 0.0, 'ci95': [0.0, 0.1], 'tasks': 50} | {'diff': 0.0, 'ci95': [0.0, 0.1], 'tasks': 50} | {'diff': 0.0, 'ci95': [0.0, 0.1], 'tasks': 50} | {'diff': 0.0, 'ci95': [0.0, 0.1], 'tasks': 50} |
| diff_pct_action_reasoned_vs_5ep (paired task bootstrap) | - | - | {'diff': -5.4, 'ci95': [-6.4, -4.3], 'tasks': 50} | {'diff': -1.6, 'ci95': [-3.0, -0.1], 'tasks': 50} | {'diff': 35.3, 'ci95': [34.2, 36.4], 'tasks': 50} | {'diff': 35.3, 'ci95': [34.2, 36.4], 'tasks': 50} | {'diff': 35.3, 'ci95': [34.2, 36.4], 'tasks': 50} | {'diff': 35.3, 'ci95': [34.2, 36.4], 'tasks': 50} | {'diff': 35.3, 'ci95': [34.2, 36.4], 'tasks': 50} |
| diff_pct_action_reasoned_vs_same_epoch_raw (paired task bootstrap) | - | - | - | - | {'diff': 35.3, 'ci95': [34.2, 36.4], 'tasks': 50} | {'diff': 40.7, 'ci95': [39.6, 41.8], 'tasks': 50} | {'diff': 36.9, 'ci95': [35.8, 38.1], 'tasks': 50} | {'diff': 35.3, 'ci95': [34.2, 36.4], 'tasks': 50} | {'diff': 40.7, 'ci95': [39.6, 41.8], 'tasks': 50} |
| distance to sc14: turn_class_tv | - | 0.353 | 0.406 | 0.368 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| distance to sc14: ks_reasoning_tokens_per_turn | - | 0.354 | 0.407 | 0.369 | 0.051 | 0.05 | 0.053 | 0.053 | 0.055 |
| distance to sc14: ks_tool_calls_per_turn | - | 0.002 | 0.003 | 0.009 | 0.013 | 0.005 | 0.011 | 0.025 | 0.043 |
| distance to sc14: ks_turns_per_rollout | - | 0.152 | 0.12 | 0.1 | 0.254 | 0.108 | 0.164 | 0.072 | 0.164 |
| distance to sc14: ks_reasoning_tokens_per_rollout | - | 0.26 | 0.24 | 0.364 | 0.137 | 0.12 | 0.252 | 0.192 | 0.296 |
| tool share %: bash | 62.6 | 60.8 | 61.5 | 64.2 | 59.9 | 59.5 | 62.2 | 59.4 | 57.7 |
| tool share %: read | 12.6 | 11.5 | 13.3 | 11.6 | 10.7 | 13.7 | 10.9 | 11.1 | 12.1 |
| tool share %: edit | 20.2 | 22.2 | 19.8 | 19.7 | 23.5 | 21.0 | 21.9 | 24.0 | 24.5 |
| tool share %: grep | 2.2 | 1.8 | 2.0 | 1.3 | 1.7 | 2.5 | 1.4 | 1.7 | 1.8 |
| tool share %: glob | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| tool share %: write | 1.7 | 2.9 | 2.5 | 2.4 | 3.6 | 2.8 | 2.7 | 3.2 | 2.9 |
| tool share %: todowrite | 0.5 | 0.6 | 0.7 | 0.7 | 0.4 | 0.5 | 0.8 | 0.6 | 0.7 |
| tool share %: task | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 | 0.1 |
| tool share %: webfetch | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| % action turns reasoned, rollout decile 1 | 100.0 | 64.6 | 60.0 | 64.3 | 99.8 | 100.0 | 100.0 | 100.0 | 100.0 |
| % action turns reasoned, rollout decile 2 | 100.0 | 64.5 | 59.2 | 63.9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| % action turns reasoned, rollout decile 3 | 100.0 | 63.7 | 59.0 | 61.9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| % action turns reasoned, rollout decile 4 | 100.0 | 63.9 | 58.2 | 61.9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| % action turns reasoned, rollout decile 5 | 100.0 | 63.1 | 57.6 | 61.9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| % action turns reasoned, rollout decile 6 | 100.0 | 63.5 | 58.3 | 63.1 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| % action turns reasoned, rollout decile 7 | 100.0 | 65.3 | 58.8 | 62.5 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| % action turns reasoned, rollout decile 8 | 100.0 | 65.2 | 59.2 | 62.7 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| % action turns reasoned, rollout decile 9 | 100.0 | 65.4 | 59.7 | 63.3 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| % action turns reasoned, rollout decile 10 | 99.9 | 67.5 | 62.7 | 65.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

base (sc14) top reasoning openers: [('Let me', 4008), ('Now let', 1617), ('This is', 809), ('Now I', 716), ("Wait there's", 483), ('Wait that', 414), ('Wait I', 399), ('Wait but', 397)]

v2 SFT 5 ep (iter38) top reasoning openers: [('Let me', 1571), ('Now let', 1197), ('Now I', 927), ('Wait I', 701), ("Wait there's", 342), ('Now the', 312), ('Wait the', 297), ('I see', 253)]

v2 SFT 10 ep (iter76) top reasoning openers: [('Now let', 1253), ('Let me', 1159), ('Now I', 879), ('Wait I', 534), ("Wait there's", 320), ('Wait the', 226), ('OK so', 217), ('Now the', 192)]

v2 SFT 20 ep (iter152) top reasoning openers: [('Let me', 1600), ('Now let', 1226), ("Wait there's", 1008), ('Now I', 800), ('Wait I', 780), ('OK so', 404), ('Wait the', 346), ('All tests', 198)]

v2 empty-think-prefix 5 ep (iter38) top reasoning openers: [('Let me', 3593), ('Now let', 1991), ('Wait I', 1848), ('Now I', 1282), ("Wait there's", 1107), ('Wait the', 1048), ('Hmm wait', 529), ('Wait but', 459)]

v2 empty-think-prefix 10 ep (iter76) top reasoning openers: [('Let me', 3021), ('Now let', 2012), ('Wait I', 1189), ('Now I', 928), ("Wait there's", 920), ('Wait the', 792), ('Wait but', 653), ('Hmm wait', 650)]

v2 empty-think-prefix 20 ep (iter152) top reasoning openers: [('Let me', 2515), ('Now let', 1713), ('Now I', 1335), ("Wait there's", 1278), ('Wait I', 1192), ('Wait the', 617), ('Now the', 565), ('OK so', 558)]

v3 empty-think-prefix 5 ep (iter71) top reasoning openers: [('Let me', 2745), ('Now let', 1775), ('Wait I', 1412), ('Now I', 1258), ('Wait the', 983), ("Wait there's", 899), ('Wait but', 587), ('Hmm wait', 380)]

v3 empty-think-prefix 10 ep (iter142) top reasoning openers: [('Let me', 2117), ('Now let', 1723), ('Wait I', 1052), ('Now I', 1048), ("Wait there's", 930), ('Wait the', 866), ('Wait but', 654), ('Now the', 291)]
