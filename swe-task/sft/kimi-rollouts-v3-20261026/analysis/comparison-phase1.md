| metric | base Nemotron (sc14) | Kimi-K3 previous (1M, no sentence) | Kimi-K3 v3 (256k + sentence) |
|---|---|---|---|
| rollouts | 250 | 250 | 250 |
| graded | 250 | 248 | 248 |
| infra | 0 | 2 | 2 |
| with_export | 250 | 250 | 248 |
| pass_at_1 | 17.2 | 77.4 | 48.8 |
| task_mean_pass | 17.2 | 77.6 | 48.5 |
| pass_at_5 | 46.0 | 100.0 | 90.0 |
| tasks | 50 | 50 | 50 |
| turns_per_rollout_mean | 245.16 | 106.19 | 79.94 |
| turns_per_rollout_median | 207.5 | 93.5 | 73.0 |
| pct_turns_with_reasoning | 100.0 | 54.7 | 57.9 |
| pct_action_turns_with_reasoning | 100.0 | 54.9 | 58.0 |
| pct_rollouts_all_actions_reasoned | 90.8 | 0.0 | 0.8 |
| rollout_mean_pct_action_reasoned | 100.0 | 55.0 | 58.1 |
| longest_unreasoned_action_run_median | 0.0 | 5.0 | 4.0 |
| longest_unreasoned_action_run_mean | 0.09 | 5.05 | 4.26 |
| longest_unreasoned_action_run_p90 | 0 | 7 | 7 |
| longest_unreasoned_action_run_max | 1 | 30 | 12 |
| rollout_pct_action_reasoned_p10 | 100.0 | 45.2 | 46.2 |
| rollout_pct_action_reasoned_p50 | 100.0 | 54.1 | 57.7 |
| rollout_pct_action_reasoned_p90 | 100.0 | 64.7 | 70.5 |
| rollout_pct_action_reasoned_max | 100.0 | 76.6 | 100.0 |
| pct_rollouts_action_reasoned_ge_90 | 100.0 | 0.0 | 0.8 |
| pct_first_action_reasoned | 100.0 | 99.6 | 99.6 |
| reasoning_tokens_per_reasoning_turn_median | 126.0 | 148.0 | 143 |
| reasoning_tokens_per_reasoning_turn_mean | 658.55 | 799.6 | 1248.75 |
| reasoning_chars_per_reasoning_turn_median | 450.0 | 577.5 | 562 |
| reasoning_tokens_per_turn_mean_incl_zero | 658.2 | 437.7 | 723.6 |
| reasoning_tokens_per_rollout_mean | 161375.15 | 46479.37 | 57840.4 |
| reasoning_tokens_per_rollout_median | 125790.5 | 40762.5 | 45168.5 |
| pct_turns_with_visible_text | 5.5 | 32.8 | 34.5 |
| pct_action_turns_with_visible_text | 5.1 | 32.2 | 34.1 |
| pct_action_turns_bare | 0.0 | 43.0 | 39.9 |
| visible_chars_median | 83 | 78.0 | 81 |
| tool_calls_per_turn | 1.08 | 1.15 | 1.17 |
| tool_calls_per_action_turn | 1.09 | 1.16 | 1.18 |
| pct_turns_parallel_calls | 7.3 | 13.6 | 15.8 |
| tool_calls_per_rollout_mean | 265.68 | 122.08 | 93.47 |
| pct_turns_notool | 0.4 | 0.9 | 1.2 |
| pct_turns_reasoning_only | 0.0 | 0.0 | 0.3 |
| pct_turns_text_notool | 0.4 | 0.9 | 0.8 |
| pct_turns_empty | 0.0 | 0.0 | 0.1 |
| pct_rollouts_git_commit | 93.6 | 97.6 | 66.5 |
| pct_rollouts_commit_in_last_3_calls | 90.4 | 96.0 | 66.1 |
| pct_rollouts_branch_created | 100.0 | 98.8 | 91.5 |
| pct_rollouts_final_notool_message | 100.0 | 99.6 | 99.6 |
| pct_rollouts_final_summary_200 | 96.4 | 97.6 | 66.9 |
| final_message_chars_median | 1901.5 | 1830 | 1529 |
| pct_rollouts_final_reasoning | 98.0 | 41.0 | 55.1 |
| pct_rollouts_todowrite | 78.8 | 80.4 | 70.6 |
| todowrite_calls_per_rollout_mean | 1.36 | 2.43 | 1.81 |
| pct_rollouts_task_subagent | 6.0 | 6.0 | 7.7 |
| task_calls_per_rollout_mean | 0.09 | 0.06 | 0.09 |
| pct_rollouts_webfetch | 18.8 | 10.8 | 8.5 |
| webfetch_calls_total | 80 | 40 | 26 |
| pct_rollouts_skill | 0.0 | 0.0 | 0.0 |
| compactions_per_rollout_mean | 0.82 | 0.0 | 0.04 |
| pct_rollouts_compacted | 58.0 | 0.0 | 3.6 |
| agent_wall_min_median | 52.9 | 35.5 | 55.1 |
| patch_empty_pct | 6.4 | 1.6 | 33.5 |
| pct_unreasoned_action_after_reasoned | 100.0 | 59.4 | 61.7 |
| turn_class_mix: action+reasoning | 99.5 | 54.4 | 57.3 |
| turn_class_mix: action, no reasoning | 0.0 | 44.7 | 41.5 |
| turn_class_mix: no-tool reasoning-only | 0.0 | 0.0 | 0.3 |
| turn_class_mix: no-tool text | 0.4 | 0.9 | 0.8 |
| turn_class_mix: no-tool empty | 0.0 | 0.0 | 0.1 |
| 95% CI (task bootstrap): pct_action_turns_with_reasoning | [99.9, 100.0] | [53.4, 56.4] | [56.2, 59.9] |
| 95% CI (task bootstrap): pct_action_turns_with_visible_text | [4.4, 5.8] | [31.0, 33.5] | [32.5, 35.8] |
| 95% CI (task bootstrap): pct_turns_parallel_calls | [6.3, 8.4] | [11.2, 16.3] | [13.8, 17.8] |
| distance to sc14: turn_class_tv | - | 0.452 | 0.423 |
| distance to sc14: ks_reasoning_tokens_per_turn | - | 0.452 | 0.42 |
| distance to sc14: ks_tool_calls_per_turn | - | 0.063 | 0.085 |
| distance to sc14: ks_turns_per_rollout | - | 0.616 | 0.739 |
| distance to sc14: ks_reasoning_tokens_per_rollout | - | 0.824 | 0.662 |
| tool share %: bash | 62.6 | 61.2 | 58.9 |
| tool share %: read | 12.6 | 12.7 | 15.2 |
| tool share %: edit | 20.2 | 19.1 | 18.0 |
| tool share %: grep | 2.2 | 2.1 | 2.8 |
| tool share %: glob | 0.1 | 0.1 | 0.1 |
| tool share %: write | 1.7 | 2.6 | 2.7 |
| tool share %: todowrite | 0.5 | 2.0 | 1.9 |
| tool share %: task | 0.0 | 0.1 | 0.1 |
| tool share %: webfetch | 0.1 | 0.1 | 0.1 |
| tool share %: skill | 0 | 0 | 0 |
| % action turns reasoned, first tool bash | 99.9 | 55.3 | 59.4 |
| % action turns reasoned, first tool read | 100.0 | 52.5 | 52.7 |
| % action turns reasoned, first tool edit | 100.0 | 50.9 | 53.1 |
| % action turns reasoned, first tool grep | 99.9 | 81.9 | 82.3 |
| % action turns reasoned, first tool write | 99.9 | 73.5 | 71.8 |
| % action turns reasoned, first tool todowrite | 100.0 | 45.0 | 49.1 |
| % action turns reasoned, first tool glob | 100.0 | 0.0 | 42.9 |
| % action turns reasoned, first tool task | 100.0 | 27.3 | 50.0 |
| % action turns reasoned, first tool webfetch | 100.0 | 69.0 | 83.3 |
| % action turns reasoned, prev_reasoned | 100.0 | 51.0 | 55.0 |
| % action turns reasoned, prev_unreasoned | 100.0 | 58.7 | 60.9 |
| % action turns reasoned, prev_ok | 100.0 | 52.5 | 55.6 |
| % action turns reasoned, prev_error | 100.0 | 68.4 | 71.5 |
| % action turns reasoned, rollout decile 1 | 100.0 | 57.9 | 60.5 |
| % action turns reasoned, rollout decile 2 | 100.0 | 60.2 | 65.1 |
| % action turns reasoned, rollout decile 3 | 100.0 | 60.0 | 59.9 |
| % action turns reasoned, rollout decile 4 | 100.0 | 56.7 | 59.6 |
| % action turns reasoned, rollout decile 5 | 100.0 | 54.3 | 57.7 |
| % action turns reasoned, rollout decile 6 | 100.0 | 52.9 | 56.0 |
| % action turns reasoned, rollout decile 7 | 100.0 | 50.8 | 54.9 |
| % action turns reasoned, rollout decile 8 | 100.0 | 48.4 | 54.1 |
| % action turns reasoned, rollout decile 9 | 100.0 | 53.8 | 56.7 |
| % action turns reasoned, rollout decile 10 | 99.9 | 53.9 | 55.4 |

base Nemotron (sc14) top reasoning openers: [('Let me', 4008), ('Now let', 1617), ('This is', 809), ('Now I', 716), ("Wait there's", 483), ('Wait that', 414), ('Wait I', 399), ('Wait but', 397)]

Kimi-K3 previous (1M, no sentence) top reasoning openers: [('Now let', 961), ('Let me', 528), ('Now I', 391), ('All tests', 180), ("Wait there's", 153), ('This is', 113), ('Now the', 103), ('Good Now', 87)]

Kimi-K3 v3 (256k + sentence) top reasoning openers: [('Now let', 880), ('Let me', 513), ('Now I', 375), ('This is', 135), ('All tests', 130), ("Wait there's", 95), ('Good Now', 68), ('Now the', 66)]

Kimi-K3 previous (1M, no sentence) hub cost: {"slots": 50, "usd_total": 138.2, "usd_per_slot_mean": 2.764, "usd_per_slot_median": 2.214, "prompt_Mtok_per_slot": 11.52, "cached_pct": 96.6, "completion_ktok_per_slot": 75.3, "calls_per_slot": 108.1}

Kimi-K3 v3 (256k + sentence) hub cost: {"slots": 257, "usd_total": 783.48, "usd_per_slot_mean": 3.049, "usd_per_slot_median": 2.259, "prompt_Mtok_per_slot": 7.6, "cached_pct": 84.1, "completion_ktok_per_slot": 68.8, "calls_per_slot": 84.3, "usd_by_key": {"NVAPI_KEY_NEW_2": 783.48}, "slots_by_key": {"NVAPI_KEY_NEW_2": 255}}
