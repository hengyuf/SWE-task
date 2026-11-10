| metric | base Nemotron (sc14) | Kimi-K3 | GLM-5.3 | DeepSeek-V4-Pro | v2 SFT (sc14 + kimi SFT) |
|---|---|---|---|---|---|
| rollouts | 250 | 250 | 250 | 250 | 250 |
| graded | 250 | 248 | 249 | 250 | 249 |
| infra | 0 | 2 | 1 | 0 | 1 |
| with_export | 250 | 250 | 249 | 250 | 250 |
| pass_at_1 | 17.2 | 77.4 | 76.7 | 53.2 | 18.1 |
| task_mean_pass | 17.2 | 77.6 | 76.8 | 53.2 | 18.0 |
| pass_at_5 | 46.0 | 100.0 | 100.0 | 86.0 | 52.0 |
| tasks | 50 | 50 | 50 | 50 | 50 |
| turns_per_rollout_mean | 245.16 | 106.19 | 172.82 | 155.25 | 287.62 |
| turns_per_rollout_median | 207.5 | 93.5 | 154 | 127.0 | 243.5 |
| pct_turns_with_reasoning | 100.0 | 54.7 | 55.0 | 57.1 | 64.7 |
| pct_action_turns_with_reasoning | 100.0 | 54.9 | 55.0 | 57.2 | 64.7 |
| pct_rollouts_all_actions_reasoned | 90.8 | 0.0 | 0.0 | 0.0 | 0.4 |
| rollout_mean_pct_action_reasoned | 100.0 | 55.0 | 56.6 | 58.4 | 64.4 |
| longest_unreasoned_action_run_median | 0.0 | 5.0 | 6 | 5.0 | 5.0 |
| longest_unreasoned_action_run_mean | 0.09 | 5.05 | 6.09 | 5.59 | 5.74 |
| longest_unreasoned_action_run_p90 | 0 | 7 | 9 | 8 | 8 |
| longest_unreasoned_action_run_max | 1 | 30 | 23 | 16 | 18 |
| rollout_pct_action_reasoned_p10 | 100.0 | 45.2 | 46.5 | 49.3 | 56.8 |
| rollout_pct_action_reasoned_p50 | 100.0 | 54.1 | 55.8 | 58.4 | 64.0 |
| rollout_pct_action_reasoned_p90 | 100.0 | 64.7 | 67.8 | 68.1 | 72.0 |
| rollout_pct_action_reasoned_max | 100.0 | 76.6 | 83.8 | 78.2 | 100.0 |
| pct_rollouts_action_reasoned_ge_90 | 100.0 | 0.0 | 0.0 | 0.0 | 0.4 |
| pct_first_action_reasoned | 100.0 | 99.6 | 98.8 | 100.0 | 100.0 |
| reasoning_tokens_per_reasoning_turn_median | 126.0 | 148.0 | 215 | 167 | 210 |
| reasoning_tokens_per_reasoning_turn_mean | 658.55 | 799.6 | 1089.79 | 848.86 | 757.31 |
| reasoning_chars_per_reasoning_turn_median | 450.0 | 577.5 | 813 | 644 | 778 |
| reasoning_tokens_per_turn_mean_incl_zero | 658.2 | 437.7 | 599.1 | 484.8 | 489.9 |
| reasoning_tokens_per_rollout_mean | 161375.15 | 46479.37 | 103530.31 | 75273.76 | 140905.21 |
| reasoning_tokens_per_rollout_median | 125790.5 | 40762.5 | 97980 | 71431.5 | 100577.5 |
| pct_turns_with_visible_text | 5.5 | 32.8 | 23.7 | 12.5 | 32.0 |
| pct_action_turns_with_visible_text | 5.1 | 32.2 | 23.3 | 12.0 | 31.9 |
| pct_action_turns_bare | 0.0 | 43.0 | 42.1 | 41.6 | 35.0 |
| visible_chars_median | 83 | 78.0 | 84.0 | 75.0 | 89.0 |
| tool_calls_per_turn | 1.08 | 1.15 | 1.09 | 1.11 | 1.09 |
| tool_calls_per_action_turn | 1.09 | 1.16 | 1.09 | 1.12 | 1.09 |
| pct_turns_parallel_calls | 7.3 | 13.6 | 8.2 | 10.9 | 7.4 |
| tool_calls_per_rollout_mean | 265.68 | 122.08 | 187.51 | 172.81 | 313.02 |
| pct_turns_notool | 0.4 | 0.9 | 0.6 | 0.6 | 0.4 |
| pct_turns_reasoning_only | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 |
| pct_turns_text_notool | 0.4 | 0.9 | 0.6 | 0.6 | 0.2 |
| pct_turns_empty | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| pct_rollouts_git_commit | 93.6 | 97.6 | 97.2 | 99.6 | 49.2 |
| pct_rollouts_commit_in_last_3_calls | 90.4 | 96.0 | 90.0 | 90.4 | 48.0 |
| pct_rollouts_branch_created | 100.0 | 98.8 | 99.6 | 99.6 | 97.2 |
| pct_rollouts_final_notool_message | 100.0 | 99.6 | 99.2 | 100.0 | 99.6 |
| pct_rollouts_final_summary_200 | 96.4 | 97.6 | 98.8 | 99.6 | 65.2 |
| final_message_chars_median | 1901.5 | 1830 | 2343 | 1690.0 | 1455 |
| pct_rollouts_final_reasoning | 98.0 | 41.0 | 46.2 | 37.2 | 65.1 |
| pct_rollouts_todowrite | 78.8 | 80.4 | 54.2 | 68.4 | 79.2 |
| todowrite_calls_per_rollout_mean | 1.36 | 2.43 | 1.38 | 1.42 | 1.8 |
| pct_rollouts_task_subagent | 6.0 | 6.0 | 10.0 | 0.0 | 5.2 |
| task_calls_per_rollout_mean | 0.09 | 0.06 | 0.1 | 0.0 | 0.06 |
| pct_rollouts_webfetch | 18.8 | 10.8 | 14.5 | 42.0 | 13.2 |
| webfetch_calls_total | 80 | 40 | 54 | 142 | 56 |
| pct_rollouts_skill | 0.0 | 0.0 | 0.0 | 0.8 | 0.0 |
| compactions_per_rollout_mean | 0.82 | 0.0 | 0.0 | 0.0 | 0.91 |
| pct_rollouts_compacted | 58.0 | 0.0 | 0.0 | 0.0 | 58.0 |
| agent_wall_min_median | 52.9 | 35.5 | 35.8 | 26.0 | 60.9 |
| patch_empty_pct | 6.4 | 1.6 | 1.2 | 0.4 | 51.0 |
| pct_unreasoned_action_after_reasoned | 100.0 | 59.4 | 57.4 | 58.2 | 62.8 |
| turn_class_mix: action+reasoning | 99.5 | 54.4 | 54.7 | 56.9 | 64.5 |
| turn_class_mix: action, no reasoning | 0.0 | 44.7 | 44.7 | 42.5 | 35.2 |
| turn_class_mix: no-tool reasoning-only | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 |
| turn_class_mix: no-tool text | 0.4 | 0.9 | 0.6 | 0.6 | 0.2 |
| turn_class_mix: no-tool empty | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 95% CI (task bootstrap): pct_action_turns_with_reasoning | [99.9, 100.0] | [53.4, 56.4] | [53.5, 56.7] | [55.7, 59.0] | [63.6, 65.8] |
| 95% CI (task bootstrap): pct_action_turns_with_visible_text | [4.4, 5.8] | [31.0, 33.5] | [20.5, 26.5] | [10.3, 14.0] | [30.5, 33.3] |
| 95% CI (task bootstrap): pct_turns_parallel_calls | [6.3, 8.4] | [11.2, 16.3] | [6.8, 9.5] | [9.6, 12.2] | [6.6, 8.3] |
| distance to sc14: turn_class_tv | - | 0.452 | 0.449 | 0.427 | 0.353 |
| distance to sc14: ks_reasoning_tokens_per_turn | - | 0.452 | 0.45 | 0.428 | 0.354 |
| distance to sc14: ks_tool_calls_per_turn | - | 0.063 | 0.008 | 0.035 | 0.002 |
| distance to sc14: ks_turns_per_rollout | - | 0.616 | 0.282 | 0.404 | 0.152 |
| distance to sc14: ks_reasoning_tokens_per_rollout | - | 0.824 | 0.303 | 0.568 | 0.26 |
| tool share %: bash | 62.6 | 61.2 | 62.5 | 62.5 | 60.8 |
| tool share %: read | 12.6 | 12.7 | 14.5 | 15.4 | 11.5 |
| tool share %: edit | 20.2 | 19.1 | 16.7 | 16.6 | 22.2 |
| tool share %: grep | 2.2 | 2.1 | 2.3 | 2.3 | 1.8 |
| tool share %: glob | 0.1 | 0.1 | 0.4 | 0.2 | 0.1 |
| tool share %: write | 1.7 | 2.6 | 2.7 | 1.8 | 2.9 |
| tool share %: todowrite | 0.5 | 2.0 | 0.7 | 0.8 | 0.6 |
| tool share %: task | 0.0 | 0.1 | 0.1 | 0 | 0.0 |
| tool share %: webfetch | 0.1 | 0.1 | 0.1 | 0.3 | 0.1 |
| tool share %: skill | 0 | 0 | 0 | 0.0 | 0 |
| % action turns reasoned, first tool bash | 99.9 | 55.3 | 55.2 | 57.7 | 62.7 |
| % action turns reasoned, first tool read | 100.0 | 52.5 | 58.1 | 58.1 | 65.6 |
| % action turns reasoned, first tool edit | 100.0 | 50.9 | 45.8 | 50.0 | 66.1 |
| % action turns reasoned, first tool grep | 99.9 | 81.9 | 82.7 | 82.1 | 72.0 |
| % action turns reasoned, first tool write | 99.9 | 73.5 | 75.1 | 76.6 | 87.9 |
| % action turns reasoned, first tool todowrite | 100.0 | 45.0 | 58.8 | 60.5 | 66.8 |
| % action turns reasoned, first tool glob | 100.0 | 0.0 | 75.0 | 83.3 | 26.1 |
| % action turns reasoned, first tool task | 100.0 | 27.3 | 71.4 | - | 83.3 |
| % action turns reasoned, first tool webfetch | 100.0 | 69.0 | 80.0 | 87.5 | 90.9 |
| % action turns reasoned, prev_reasoned | 100.0 | 51.0 | 52.8 | 56.3 | 65.6 |
| % action turns reasoned, prev_unreasoned | 100.0 | 58.7 | 57.1 | 57.9 | 62.6 |
| % action turns reasoned, prev_ok | 100.0 | 52.5 | 52.8 | 55.3 | 63.5 |
| % action turns reasoned, prev_error | 100.0 | 68.4 | 67.5 | 68.1 | 69.7 |
| % action turns reasoned, rollout decile 1 | 100.0 | 57.9 | 65.2 | 65.3 | 64.6 |
| % action turns reasoned, rollout decile 2 | 100.0 | 60.2 | 61.4 | 65.0 | 64.5 |
| % action turns reasoned, rollout decile 3 | 100.0 | 60.0 | 56.1 | 58.2 | 63.7 |
| % action turns reasoned, rollout decile 4 | 100.0 | 56.7 | 54.4 | 57.0 | 63.9 |
| % action turns reasoned, rollout decile 5 | 100.0 | 54.3 | 52.2 | 56.0 | 63.1 |
| % action turns reasoned, rollout decile 6 | 100.0 | 52.9 | 49.1 | 52.5 | 63.5 |
| % action turns reasoned, rollout decile 7 | 100.0 | 50.8 | 51.5 | 50.6 | 65.3 |
| % action turns reasoned, rollout decile 8 | 100.0 | 48.4 | 49.8 | 50.9 | 65.2 |
| % action turns reasoned, rollout decile 9 | 100.0 | 53.8 | 54.2 | 56.2 | 65.4 |
| % action turns reasoned, rollout decile 10 | 99.9 | 53.9 | 56.5 | 60.7 | 67.5 |

base Nemotron (sc14) top reasoning openers: [('Let me', 4008), ('Now let', 1617), ('This is', 809), ('Now I', 716), ("Wait there's", 483), ('Wait that', 414), ('Wait I', 399), ('Wait but', 397)]

Kimi-K3 top reasoning openers: [('Now let', 961), ('Let me', 528), ('Now I', 391), ('All tests', 180), ("Wait there's", 153), ('This is', 113), ('Now the', 103), ('Good Now', 87)]

GLM-5.3 top reasoning openers: [('Now let', 1572), ('Let me', 764), ('Now I', 503), ('All tests', 315), ('Wait I', 219), ('I see', 206), ('This is', 164), ("Wait there's", 139)]

DeepSeek-V4-Pro top reasoning openers: [('Now let', 1647), ('Let me', 751), ('Now I', 522), ('This is', 285), ('All tests', 231), ('Now the', 191), ('Good Now', 156), ("Wait there's", 134)]

v2 SFT (sc14 + kimi SFT) top reasoning openers: [('Let me', 1571), ('Now let', 1197), ('Now I', 927), ('Wait I', 701), ("Wait there's", 342), ('Now the', 312), ('Wait the', 297), ('I see', 253)]

Kimi-K3 hub cost: {"slots": 50, "usd_total": 138.2, "usd_per_slot_mean": 2.764, "usd_per_slot_median": 2.214, "prompt_Mtok_per_slot": 11.52, "cached_pct": 96.6, "completion_ktok_per_slot": 75.3, "calls_per_slot": 108.1}

GLM-5.3 hub cost: {"slots": 250, "usd_total": 1043.66, "usd_per_slot_mean": 4.175, "usd_per_slot_median": 3.153, "prompt_Mtok_per_slot": 33.0, "cached_pct": 99.0, "completion_ktok_per_slot": 150.6, "calls_per_slot": 192.5, "usd_by_key": {"NVAPI_KEY_NEW_1": 1043.66}, "slots_by_key": {"NVAPI_KEY_NEW_1": 250}}

DeepSeek-V4-Pro hub cost: {"slots": 250, "usd_total": 496.16, "usd_per_slot_mean": 1.985, "usd_per_slot_median": 1.377, "prompt_Mtok_per_slot": 24.91, "cached_pct": 99.2, "completion_ktok_per_slot": 117.1, "calls_per_slot": 157.1, "usd_by_key": {"NVAPI_KEY_NEW_1": 483.55, "NVAPI_KEY_NEW_2": 12.6}, "slots_by_key": {"NVAPI_KEY_NEW_1": 240, "NVAPI_KEY_NEW_2": 10}}
