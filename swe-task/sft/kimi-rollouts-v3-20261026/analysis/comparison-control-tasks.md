| metric | Kimi-K3 previous (1M, no sentence) | Kimi-K3 v3 (256k + sentence) | Kimi-K3 control (1M, no sentence, today) |
|---|---|---|---|
| rollouts | 20 | 20 | 20 |
| graded | 20 | 20 | 18 |
| infra | 0 | 0 | 2 |
| with_export | 20 | 20 | 19 |
| pass_at_1 | 85.0 | 80.0 | 66.7 |
| task_mean_pass | 85.0 | 80.0 | 65.0 |
| pass_at_5 | 100.0 | 100.0 | 100.0 |
| tasks | 4 | 4 | 4 |
| turns_per_rollout_mean | 63.05 | 55.3 | 50.79 |
| turns_per_rollout_median | 61.5 | 50.0 | 48 |
| pct_turns_with_reasoning | 55.1 | 61.9 | 50.8 |
| pct_action_turns_with_reasoning | 55.5 | 62.2 | 51.1 |
| pct_rollouts_all_actions_reasoned | 0.0 | 0.0 | 5.3 |
| rollout_mean_pct_action_reasoned | 54.7 | 60.6 | 53.6 |
| longest_unreasoned_action_run_median | 4.0 | 4.0 | 5 |
| longest_unreasoned_action_run_mean | 4.8 | 4.2 | 4.74 |
| longest_unreasoned_action_run_p90 | 6 | 6 | 7 |
| longest_unreasoned_action_run_max | 9 | 6 | 8 |
| rollout_pct_action_reasoned_p10 | 42.1 | 48.2 | 45.5 |
| rollout_pct_action_reasoned_p50 | 55.3 | 58.8 | 50.0 |
| rollout_pct_action_reasoned_p90 | 63.9 | 72.4 | 67.4 |
| rollout_pct_action_reasoned_max | 71.8 | 73.8 | 100.0 |
| pct_rollouts_action_reasoned_ge_90 | 0.0 | 0.0 | 5.3 |
| pct_first_action_reasoned | 100.0 | 100.0 | 100.0 |
| reasoning_tokens_per_reasoning_turn_median | 116 | 114 | 118.0 |
| reasoning_tokens_per_reasoning_turn_mean | 1289.29 | 1395.96 | 2178.0 |
| reasoning_chars_per_reasoning_turn_median | 432 | 448 | 461.5 |
| reasoning_tokens_per_turn_mean_incl_zero | 710.6 | 864.6 | 1105.9 |
| reasoning_tokens_per_rollout_mean | 44802.9 | 47811.6 | 56169.53 |
| reasoning_tokens_per_rollout_median | 40084.0 | 39709.0 | 39991 |
| pct_turns_with_visible_text | 36.1 | 44.2 | 35.6 |
| pct_action_turns_with_visible_text | 35.1 | 43.3 | 34.8 |
| pct_action_turns_bare | 41.3 | 35.1 | 44.5 |
| visible_chars_median | 78 | 78 | 78.0 |
| tool_calls_per_turn | 1.2 | 1.27 | 1.23 |
| tool_calls_per_action_turn | 1.22 | 1.29 | 1.26 |
| pct_turns_parallel_calls | 16.2 | 20.9 | 16.8 |
| tool_calls_per_rollout_mean | 75.4 | 70.05 | 62.53 |
| pct_turns_notool | 1.6 | 1.8 | 2.0 |
| pct_turns_reasoning_only | 0.0 | 0.1 | 0.4 |
| pct_turns_text_notool | 1.6 | 1.7 | 1.6 |
| pct_turns_empty | 0.0 | 0.0 | 0.0 |
| pct_rollouts_git_commit | 100.0 | 95.0 | 78.9 |
| pct_rollouts_commit_in_last_3_calls | 100.0 | 95.0 | 78.9 |
| pct_rollouts_branch_created | 100.0 | 95.0 | 94.7 |
| pct_rollouts_final_notool_message | 100.0 | 100.0 | 100.0 |
| pct_rollouts_final_summary_200 | 100.0 | 95.0 | 78.9 |
| final_message_chars_median | 1520.0 | 1645.0 | 1581 |
| pct_rollouts_final_reasoning | 30.0 | 45.0 | 36.8 |
| pct_rollouts_todowrite | 70.0 | 65.0 | 63.2 |
| todowrite_calls_per_rollout_mean | 1.85 | 1.85 | 1.74 |
| pct_rollouts_task_subagent | 0.0 | 0.0 | 0.0 |
| task_calls_per_rollout_mean | 0.0 | 0.0 | 0.0 |
| pct_rollouts_webfetch | 20.0 | 15.0 | 15.8 |
| webfetch_calls_total | 6 | 4 | 8 |
| pct_rollouts_skill | 0.0 | 0.0 | 0.0 |
| compactions_per_rollout_mean | 0.0 | 0.0 | 0.0 |
| pct_rollouts_compacted | 0.0 | 0.0 | 0.0 |
| agent_wall_min_median | 27.1 | 47.7 | 46.6 |
| patch_empty_pct | 0.0 | 5.0 | 22.2 |
| pct_unreasoned_action_after_reasoned | 54.0 | 62.2 | 52.5 |
| distance_to_sc14 | None | None | None |
| turn_class_mix: action+reasoning | 54.6 | 61.1 | 50.1 |
| turn_class_mix: action, no reasoning | 43.8 | 37.1 | 48.0 |
| turn_class_mix: no-tool reasoning-only | 0.0 | 0.1 | 0.4 |
| turn_class_mix: no-tool text | 1.6 | 1.7 | 1.6 |
| turn_class_mix: no-tool empty | 0.0 | 0.0 | 0.0 |
| 95% CI (task bootstrap): pct_action_turns_with_reasoning | [48.9, 59.8] | [53.4, 67.1] | [46.2, 55.9] |
| 95% CI (task bootstrap): pct_action_turns_with_visible_text | [31.4, 39.3] | [35.4, 47.2] | [31.0, 39.7] |
| 95% CI (task bootstrap): pct_turns_parallel_calls | [12.1, 21.7] | [17.4, 25.6] | [13.3, 20.3] |
| distance to sc14: turn_class_tv | - | - | - |
| distance to sc14: ks_reasoning_tokens_per_turn | - | - | - |
| distance to sc14: ks_tool_calls_per_turn | - | - | - |
| distance to sc14: ks_turns_per_rollout | - | - | - |
| distance to sc14: ks_reasoning_tokens_per_rollout | - | - | - |
| tool share %: bash | 57.2 | 56.0 | 50.6 |
| tool share %: read | 15.7 | 18.1 | 20.0 |
| tool share %: edit | 20.4 | 19.3 | 21.0 |
| tool share %: grep | 0.8 | 0.7 | 0.9 |
| tool share %: glob | 0 | 0 | 0.2 |
| tool share %: write | 3.1 | 3.0 | 3.7 |
| tool share %: todowrite | 2.5 | 2.6 | 2.8 |
| tool share %: task | 0 | 0 | 0 |
| tool share %: webfetch | 0.4 | 0.3 | 0.7 |
| tool share %: skill | 0 | 0 | 0 |
| % action turns reasoned, first tool bash | 60.6 | 66.9 | 55.6 |
| % action turns reasoned, first tool read | 40.7 | 47.5 | 43.4 |
| % action turns reasoned, first tool edit | 47.2 | 57.1 | 41.3 |
| % action turns reasoned, first tool grep | 100.0 | 100.0 | 87.5 |
| % action turns reasoned, first tool write | 70.2 | 70.7 | 64.3 |
| % action turns reasoned, first tool todowrite | 35.3 | 36.4 | 37.0 |
| % action turns reasoned, first tool glob | - | - | - |
| % action turns reasoned, first tool task | - | - | - |
| % action turns reasoned, first tool webfetch | 50.0 | 100.0 | 100.0 |
| % action turns reasoned, prev_reasoned | 56.6 | 61.9 | 49.3 |
| % action turns reasoned, prev_unreasoned | 52.5 | 61.0 | 50.9 |
| % action turns reasoned, prev_ok | 53.1 | 60.5 | 48.1 |
| % action turns reasoned, prev_error | 69.2 | 72.2 | 73.2 |
| % action turns reasoned, rollout decile 1 | 55.3 | 64.4 | 42.9 |
| % action turns reasoned, rollout decile 2 | 55.6 | 51.4 | 57.7 |
| % action turns reasoned, rollout decile 3 | 51.2 | 63.9 | 54.8 |
| % action turns reasoned, rollout decile 4 | 58.9 | 64.8 | 51.5 |
| % action turns reasoned, rollout decile 5 | 53.9 | 58.0 | 46.4 |
| % action turns reasoned, rollout decile 6 | 48.8 | 56.5 | 48.4 |
| % action turns reasoned, rollout decile 7 | 58.2 | 67.6 | 56.8 |
| % action turns reasoned, rollout decile 8 | 62.2 | 71.2 | 52.6 |
| % action turns reasoned, rollout decile 9 | 54.1 | 64.2 | 55.8 |
| % action turns reasoned, rollout decile 10 | 56.7 | 61.0 | 43.1 |

Kimi-K3 previous (1M, no sentence) top reasoning openers: [('Now let', 52), ('Let me', 40), ('All tests', 19), ('Now I', 18), ("Wait there's", 10), ('Now the', 9), ('All pass', 7), ('All good', 4)]

Kimi-K3 v3 (256k + sentence) top reasoning openers: [('Now let', 56), ('Let me', 44), ('Now I', 13), ('All tests', 11), ('Now the', 8), ('Now update', 7), ("Wait there's", 7), ('Wait one', 6)]

Kimi-K3 control (1M, no sentence, today) top reasoning openers: [('Now let', 40), ('Let me', 36), ('Now I', 18), ('All tests', 14), ('All pass', 8), ("Wait there's", 8), ('The THREAT', 5), ('Now the', 5)]

Kimi-K3 previous (1M, no sentence) hub cost: {"slots": 50, "usd_total": 138.2, "usd_per_slot_mean": 2.764, "usd_per_slot_median": 2.214, "prompt_Mtok_per_slot": 11.52, "cached_pct": 96.6, "completion_ktok_per_slot": 75.3, "calls_per_slot": 108.1}

Kimi-K3 v3 (256k + sentence) hub cost: {"slots": 257, "usd_total": 783.48, "usd_per_slot_mean": 3.049, "usd_per_slot_median": 2.259, "prompt_Mtok_per_slot": 7.6, "cached_pct": 84.1, "completion_ktok_per_slot": 68.8, "calls_per_slot": 84.3, "usd_by_key": {"NVAPI_KEY_NEW_2": 783.48}, "slots_by_key": {"NVAPI_KEY_NEW_2": 255}}

Kimi-K3 control (1M, no sentence, today) hub cost: {"slots": 20, "usd_total": 42.19, "usd_per_slot_mean": 2.11, "usd_per_slot_median": 1.865, "prompt_Mtok_per_slot": 4.59, "cached_pct": 82.7, "completion_ktok_per_slot": 76.3, "calls_per_slot": 67.0, "usd_by_key": {"NVAPI_KEY_NEW_2": 42.19}, "slots_by_key": {"NVAPI_KEY_NEW_2": 20}}
