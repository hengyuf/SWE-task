| metric | base Nemotron (sc14) | Kimi-K3 | v2 SFT (sc14 + kimi SFT) |
|---|---|---|---|
| rollouts | 250 | 250 | 250 |
| graded | 250 | 248 | 249 |
| infra | 0 | 2 | 1 |
| with_export | 250 | 250 | 250 |
| pass_at_1 | 17.2 | 77.4 | 18.1 |
| task_mean_pass | 17.2 | 77.6 | 18.0 |
| pass_at_5 | 46.0 | 100.0 | 52.0 |
| tasks | 50 | 50 | 50 |
| turns_per_rollout_mean | 245.16 | 106.19 | 287.62 |
| turns_per_rollout_median | 207.5 | 93.5 | 243.5 |
| pct_turns_with_reasoning | 100.0 | 54.7 | 64.7 |
| pct_action_turns_with_reasoning | 100.0 | 54.9 | 64.7 |
| pct_rollouts_all_actions_reasoned | 90.8 | 0.0 | 0.4 |
| rollout_mean_pct_action_reasoned | 100.0 | 55.0 | 64.4 |
| longest_unreasoned_action_run_median | 0.0 | 5.0 | 5.0 |
| pct_first_action_reasoned | 100.0 | 99.6 | 100.0 |
| reasoning_tokens_per_reasoning_turn_median | 126.0 | 148.0 | 210 |
| reasoning_tokens_per_reasoning_turn_mean | 658.55 | 799.6 | 757.31 |
| reasoning_chars_per_reasoning_turn_median | 450.0 | 577.5 | 778 |
| reasoning_tokens_per_turn_mean_incl_zero | 658.2 | 437.7 | 489.9 |
| reasoning_tokens_per_rollout_mean | 161375.15 | 46479.37 | 140905.21 |
| reasoning_tokens_per_rollout_median | 125790.5 | 40762.5 | 100577.5 |
| pct_turns_with_visible_text | 5.5 | 32.8 | 32.0 |
| pct_action_turns_with_visible_text | 5.1 | 32.2 | 31.9 |
| pct_action_turns_bare | 0.0 | 43.0 | 35.0 |
| visible_chars_median | 83 | 78.0 | 89.0 |
| tool_calls_per_turn | 1.08 | 1.15 | 1.09 |
| tool_calls_per_action_turn | 1.09 | 1.16 | 1.09 |
| pct_turns_parallel_calls | 7.3 | 13.6 | 7.4 |
| tool_calls_per_rollout_mean | 265.68 | 122.08 | 313.02 |
| pct_turns_notool | 0.4 | 0.9 | 0.4 |
| pct_turns_reasoning_only | 0.0 | 0.0 | 0.1 |
| pct_turns_text_notool | 0.4 | 0.9 | 0.2 |
| pct_turns_empty | 0.0 | 0.0 | 0.0 |
| pct_rollouts_git_commit | 93.6 | 97.6 | 49.2 |
| pct_rollouts_commit_in_last_3_calls | 90.4 | 96.0 | 48.0 |
| pct_rollouts_branch_created | 100.0 | 98.8 | 97.2 |
| pct_rollouts_final_notool_message | 100.0 | 99.6 | 99.6 |
| pct_rollouts_final_summary_200 | 96.4 | 97.6 | 65.2 |
| final_message_chars_median | 1901.5 | 1830 | 1455 |
| pct_rollouts_final_reasoning | 98.0 | 41.0 | 65.1 |
| pct_rollouts_todowrite | 78.8 | 80.4 | 79.2 |
| todowrite_calls_per_rollout_mean | 1.36 | 2.43 | 1.8 |
| pct_rollouts_task_subagent | 6.0 | 6.0 | 5.2 |
| task_calls_per_rollout_mean | 0.09 | 0.06 | 0.06 |
| pct_rollouts_webfetch | 18.8 | 10.8 | 13.2 |
| webfetch_calls_total | 80 | 40 | 56 |
| pct_rollouts_skill | 0.0 | 0.0 | 0.0 |
| compactions_per_rollout_mean | 0.82 | 0.0 | 0.91 |
| pct_rollouts_compacted | 58.0 | 0.0 | 58.0 |
| agent_wall_min_median | 52.9 | 35.5 | 60.9 |
| patch_empty_pct | 6.4 | 1.6 | 51.0 |
| pct_unreasoned_action_after_reasoned | 100.0 | 59.4 | 62.8 |
| turn_class_mix: action+reasoning | 99.5 | 54.4 | 64.5 |
| turn_class_mix: action, no reasoning | 0.0 | 44.7 | 35.2 |
| turn_class_mix: no-tool reasoning-only | 0.0 | 0.0 | 0.1 |
| turn_class_mix: no-tool text | 0.4 | 0.9 | 0.2 |
| turn_class_mix: no-tool empty | 0.0 | 0.0 | 0.0 |
| 95% CI (task bootstrap): pct_action_turns_with_reasoning | [99.9, 100.0] | [53.4, 56.4] | [63.6, 65.8] |
| 95% CI (task bootstrap): pct_action_turns_with_visible_text | [4.4, 5.8] | [31.0, 33.5] | [30.5, 33.3] |
| 95% CI (task bootstrap): pct_turns_parallel_calls | [6.3, 8.4] | [11.2, 16.3] | [6.6, 8.3] |
| distance to sc14: turn_class_tv | - | 0.452 | 0.353 |
| distance to sc14: ks_reasoning_tokens_per_turn | - | 0.452 | 0.354 |
| distance to sc14: ks_tool_calls_per_turn | - | 0.063 | 0.002 |
| distance to sc14: ks_turns_per_rollout | - | 0.616 | 0.152 |
| distance to sc14: ks_reasoning_tokens_per_rollout | - | 0.824 | 0.26 |
| tool share %: bash | 62.6 | 61.2 | 60.8 |
| tool share %: read | 12.6 | 12.7 | 11.5 |
| tool share %: edit | 20.2 | 19.1 | 22.2 |
| tool share %: grep | 2.2 | 2.1 | 1.8 |
| tool share %: glob | 0.1 | 0.1 | 0.1 |
| tool share %: write | 1.7 | 2.6 | 2.9 |
| tool share %: todowrite | 0.5 | 2.0 | 0.6 |
| tool share %: task | 0.0 | 0.1 | 0.0 |
| tool share %: webfetch | 0.1 | 0.1 | 0.1 |
| tool share %: skill | 0 | 0 | 0 |
| % action turns reasoned, first tool bash | 99.9 | 55.3 | 62.7 |
| % action turns reasoned, first tool read | 100.0 | 52.5 | 65.6 |
| % action turns reasoned, first tool edit | 100.0 | 50.9 | 66.1 |
| % action turns reasoned, first tool grep | 99.9 | 81.9 | 72.0 |
| % action turns reasoned, first tool write | 99.9 | 73.5 | 87.9 |
| % action turns reasoned, first tool todowrite | 100.0 | 45.0 | 66.8 |
| % action turns reasoned, first tool glob | 100.0 | 0.0 | 26.1 |
| % action turns reasoned, first tool task | 100.0 | 27.3 | 83.3 |
| % action turns reasoned, first tool webfetch | 100.0 | 69.0 | 90.9 |
| % action turns reasoned, prev_reasoned | 100.0 | 51.0 | 65.6 |
| % action turns reasoned, prev_unreasoned | 100.0 | 58.7 | 62.6 |
| % action turns reasoned, prev_ok | 100.0 | 52.5 | 63.5 |
| % action turns reasoned, prev_error | 100.0 | 68.4 | 69.7 |
| % action turns reasoned, rollout decile 1 | 100.0 | 57.9 | 64.6 |
| % action turns reasoned, rollout decile 2 | 100.0 | 60.2 | 64.5 |
| % action turns reasoned, rollout decile 3 | 100.0 | 60.0 | 63.7 |
| % action turns reasoned, rollout decile 4 | 100.0 | 56.7 | 63.9 |
| % action turns reasoned, rollout decile 5 | 100.0 | 54.3 | 63.1 |
| % action turns reasoned, rollout decile 6 | 100.0 | 52.9 | 63.5 |
| % action turns reasoned, rollout decile 7 | 100.0 | 50.8 | 65.3 |
| % action turns reasoned, rollout decile 8 | 100.0 | 48.4 | 65.2 |
| % action turns reasoned, rollout decile 9 | 100.0 | 53.8 | 65.4 |
| % action turns reasoned, rollout decile 10 | 99.9 | 53.9 | 67.5 |

base Nemotron (sc14) top reasoning openers: [('Let me', 4008), ('Now let', 1617), ('This is', 809), ('Now I', 716), ("Wait there's", 483), ('Wait that', 414), ('Wait I', 399), ('Wait but', 397)]

Kimi-K3 top reasoning openers: [('Now let', 961), ('Let me', 528), ('Now I', 391), ('All tests', 180), ("Wait there's", 153), ('This is', 113), ('Now the', 103), ('Good Now', 87)]

v2 SFT (sc14 + kimi SFT) top reasoning openers: [('Let me', 1571), ('Now let', 1197), ('Now I', 927), ('Wait I', 701), ("Wait there's", 342), ('Now the', 312), ('Wait the', 297), ('I see', 253)]

Kimi-K3 hub cost: {"slots": 50, "usd_total": 138.2, "usd_per_slot_mean": 2.764, "usd_per_slot_median": 2.214, "prompt_Mtok_per_slot": 11.52, "cached_pct": 96.6, "completion_ktok_per_slot": 75.3, "calls_per_slot": 108.1}
