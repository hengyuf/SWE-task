| metric | base Nemotron (sc14) | Kimi-K3 | GLM-5.3 | DeepSeek-V4-Pro | v2 SFT (sc14 + kimi SFT) |
|---|---|---|---|---|---|
| rollouts | 10 | 10 | 2 | 2 | 10 |
| graded | 10 | 10 | 2 | 2 | 10 |
| infra | 0 | 0 | 0 | 0 | 0 |
| with_export | 10 | 10 | 2 | 2 | 10 |
| pass_at_1 | 40.0 | 70.0 | 50.0 | 50.0 | 50.0 |
| task_mean_pass | 40.0 | 70.0 | 50.0 | 50.0 | 50.0 |
| pass_at_5 | 50.0 | 100.0 | 50.0 | 50.0 | 100.0 |
| tasks | 2 | 2 | 2 | 2 | 2 |
| turns_per_rollout_mean | 169.8 | 87.0 | 151.0 | 121.0 | 208.6 |
| turns_per_rollout_median | 182.5 | 80.0 | 151.0 | 121.0 | 203.0 |
| pct_turns_with_reasoning | 99.9 | 58.7 | 60.3 | 57.9 | 66.9 |
| pct_action_turns_with_reasoning | 99.9 | 59.1 | 60.7 | 57.9 | 67.0 |
| pct_rollouts_all_actions_reasoned | 90.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| rollout_mean_pct_action_reasoned | 99.9 | 58.8 | 62.6 | 58.6 | 65.1 |
| longest_unreasoned_action_run_median | 0.0 | 4.5 | 5.0 | 6.5 | 5.5 |
| pct_first_action_reasoned | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| reasoning_tokens_per_reasoning_turn_median | 104 | 131 | 242.5 | 159.0 | 209.5 |
| reasoning_tokens_per_reasoning_turn_mean | 544.74 | 831.64 | 990.95 | 1015.68 | 689.55 |
| reasoning_chars_per_reasoning_turn_median | 397 | 528 | 928.0 | 608.0 | 793.5 |
| reasoning_tokens_per_turn_mean_incl_zero | 544.4 | 488.5 | 597.2 | 587.6 | 461.5 |
| reasoning_tokens_per_rollout_mean | 92442.4 | 42496.7 | 90176.5 | 71097.5 | 96261.3 |
| reasoning_tokens_per_rollout_median | 92369.5 | 46508.0 | 90176.5 | 71097.5 | 78938.0 |
| pct_turns_with_visible_text | 5.2 | 33.2 | 29.5 | 11.2 | 32.2 |
| pct_action_turns_with_visible_text | 4.6 | 32.4 | 29.0 | 10.4 | 32.0 |
| pct_action_turns_bare | 0.1 | 38.5 | 36.7 | 39.6 | 32.4 |
| visible_chars_median | 78.5 | 82 | 84 | 66 | 92.0 |
| tool_calls_per_turn | 1.13 | 1.2 | 1.13 | 1.09 | 1.08 |
| tool_calls_per_action_turn | 1.14 | 1.21 | 1.14 | 1.1 | 1.09 |
| pct_turns_parallel_calls | 12.5 | 20.5 | 12.6 | 9.9 | 8.1 |
| tool_calls_per_rollout_mean | 192.2 | 104.0 | 171.0 | 132.0 | 225.4 |
| pct_turns_notool | 0.6 | 1.1 | 0.7 | 0.8 | 0.5 |
| pct_turns_reasoning_only | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| pct_turns_text_notool | 0.6 | 1.1 | 0.7 | 0.8 | 0.3 |
| pct_turns_empty | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 |
| pct_rollouts_git_commit | 90.0 | 100.0 | 100.0 | 100.0 | 70.0 |
| pct_rollouts_commit_in_last_3_calls | 80.0 | 100.0 | 50.0 | 100.0 | 70.0 |
| pct_rollouts_branch_created | 100.0 | 100.0 | 100.0 | 100.0 | 90.0 |
| pct_rollouts_final_notool_message | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| pct_rollouts_final_summary_200 | 100.0 | 100.0 | 100.0 | 100.0 | 70.0 |
| final_message_chars_median | 2454.5 | 2035.5 | 2697.5 | 2085.5 | 1853.5 |
| pct_rollouts_final_reasoning | 100.0 | 30.0 | 0.0 | 50.0 | 50.0 |
| pct_rollouts_todowrite | 80.0 | 100.0 | 100.0 | 50.0 | 80.0 |
| todowrite_calls_per_rollout_mean | 1.3 | 2.8 | 3.0 | 2.0 | 2.6 |
| pct_rollouts_task_subagent | 10.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| task_calls_per_rollout_mean | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| pct_rollouts_webfetch | 30.0 | 30.0 | 0.0 | 0.0 | 40.0 |
| webfetch_calls_total | 5 | 6 | 0 | 0 | 7 |
| pct_rollouts_skill | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| compactions_per_rollout_mean | 0.5 | 0.0 | 0.0 | 0.0 | 0.6 |
| pct_rollouts_compacted | 50.0 | 0.0 | 0.0 | 0.0 | 50.0 |
| agent_wall_min_median | 37.8 | 40.0 | 35.6 | 20.3 | 51.8 |
| patch_empty_pct | 10.0 | 0.0 | 0.0 | 0.0 | 30.0 |
| pct_unreasoned_action_after_reasoned | 100.0 | 61.4 | 63.6 | 58.4 | 62.9 |
| turn_class_mix: action+reasoning | 99.4 | 58.4 | 60.3 | 57.4 | 66.7 |
| turn_class_mix: action, no reasoning | 0.1 | 40.5 | 39.1 | 41.7 | 32.8 |
| turn_class_mix: no-tool reasoning-only | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| turn_class_mix: no-tool text | 0.6 | 1.1 | 0.7 | 0.8 | 0.3 |
| turn_class_mix: no-tool empty | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 |
| 95% CI (task bootstrap): pct_action_turns_with_reasoning | [99.9, 100.0] | [57.8, 60.2] | [50.6, 74.6] | [54.6, 62.6] | [63.1, 70.0] |
| 95% CI (task bootstrap): pct_action_turns_with_visible_text | [2.6, 6.4] | [29.9, 34.7] | [21.3, 39.7] | [7.1, 15.2] | [26.2, 36.6] |
| 95% CI (task bootstrap): pct_turns_parallel_calls | [12.4, 12.5] | [17.0, 24.4] | [11.4, 14.2] | [9.0, 10.6] | [6.8, 9.8] |
| distance to sc14: turn_class_tv | - | 0.41 | 0.391 | 0.419 | 0.329 |
| distance to sc14: ks_reasoning_tokens_per_turn | - | 0.412 | 0.397 | 0.421 | 0.331 |
| distance to sc14: ks_tool_calls_per_turn | - | 0.08 | 0.006 | 0.026 | 0.043 |
| distance to sc14: ks_turns_per_rollout | - | 0.9 | 0.6 | 0.7 | 0.4 |
| distance to sc14: ks_reasoning_tokens_per_rollout | - | 1.0 | 0.5 | 0.6 | 0.4 |
| tool share %: bash | 50.9 | 66.6 | 52.6 | 65.9 | 62.5 |
| tool share %: read | 16.2 | 9.2 | 17.5 | 11.4 | 11.3 |
| tool share %: edit | 22.8 | 14.2 | 17.0 | 12.9 | 20.7 |
| tool share %: grep | 8.1 | 5.3 | 9.6 | 7.6 | 1.6 |
| tool share %: glob | 0.3 | 0.4 | 0.9 | 0.4 | 0.2 |
| tool share %: write | 0.7 | 1.0 | 0.6 | 0.4 | 2.2 |
| tool share %: todowrite | 0.7 | 2.7 | 1.8 | 1.5 | 1.2 |
| tool share %: task | 0.1 | 0 | 0 | 0 | 0 |
| tool share %: webfetch | 0.3 | 0.6 | 0 | 0 | 0.3 |
| tool share %: skill | 0 | 0 | 0 | 0 | 0 |
| % action turns reasoned, first tool bash | 99.9 | 61.1 | 57.1 | 63.1 | 68.5 |
| % action turns reasoned, first tool read | 100.0 | 53.9 | 60.4 | 52.2 | 66.4 |
| % action turns reasoned, first tool edit | 100.0 | 51.0 | 62.5 | 35.3 | 60.0 |
| % action turns reasoned, first tool grep | 100.0 | 86.2 | 85.0 | 70.0 | 86.4 |
| % action turns reasoned, first tool write | 100.0 | 60.0 | 50.0 | 0.0 | 83.3 |
| % action turns reasoned, first tool todowrite | 100.0 | 38.5 | 66.7 | 50.0 | 72.0 |
| % action turns reasoned, first tool glob | 100.0 | - | 100.0 | - | 0.0 |
| % action turns reasoned, first tool task | 100.0 | - | - | - | - |
| % action turns reasoned, first tool webfetch | 100.0 | 100.0 | - | - | 100.0 |
| % action turns reasoned, prev_reasoned | 99.9 | 57.5 | 58.8 | 57.2 | 68.9 |
| % action turns reasoned, prev_unreasoned | 100.0 | 60.2 | 62.9 | 58.0 | 62.6 |
| % action turns reasoned, prev_ok | 99.9 | 57.4 | 58.4 | 56.8 | 66.3 |
| % action turns reasoned, prev_error | 100.0 | 67.7 | 75.0 | 72.7 | 69.6 |
| % action turns reasoned, rollout decile 1 | 100.0 | 67.9 | 82.8 | 87.0 | 68.0 |
| % action turns reasoned, rollout decile 2 | 100.0 | 62.8 | 63.3 | 62.5 | 66.8 |
| % action turns reasoned, rollout decile 3 | 100.0 | 61.6 | 70.0 | 58.3 | 67.1 |
| % action turns reasoned, rollout decile 4 | 100.0 | 65.5 | 70.0 | 50.0 | 60.9 |
| % action turns reasoned, rollout decile 5 | 100.0 | 59.8 | 48.4 | 58.3 | 57.1 |
| % action turns reasoned, rollout decile 6 | 100.0 | 48.8 | 62.1 | 54.2 | 63.6 |
| % action turns reasoned, rollout decile 7 | 100.0 | 58.6 | 46.7 | 37.5 | 68.6 |
| % action turns reasoned, rollout decile 8 | 100.0 | 62.8 | 53.3 | 66.7 | 70.5 |
| % action turns reasoned, rollout decile 9 | 100.0 | 51.2 | 63.3 | 58.3 | 73.6 |
| % action turns reasoned, rollout decile 10 | 99.4 | 52.2 | 48.4 | 48.0 | 73.7 |

base Nemotron (sc14) top reasoning openers: [('Let me', 138), ('Now let', 77), ('Now I', 36), ('Wait I', 15), ('Now add', 12), ("Wait there's", 11), ('OK so', 11), ('Now the', 11)]

Kimi-K3 top reasoning openers: [('Now let', 49), ('Let me', 31), ('Now I', 21), ('No network', 9), ('All pass', 6), ('Two failures', 3), ('The THREAT', 3), ('Good Now', 3)]

GLM-5.3 top reasoning openers: [('Now let', 14), ('Let me', 12), ('Now I', 4), ('Now the', 3), ('I have', 2), ('Now update', 2), ('Wait I', 2), ('No network', 2)]

DeepSeek-V4-Pro top reasoning openers: [('Now let', 7), ('Let me', 6), ('Now add', 4), ('Wait I', 2), ('Good Now', 2), ('No network', 2), ('This is', 2), ('security IsNamespaceEnabled', 2)]

v2 SFT (sc14 + kimi SFT) top reasoning openers: [('Let me', 65), ('Now let', 50), ('Wait I', 34), ('Now I', 29), ('Now the', 14), ('The file', 14), ('Wait the', 10), ('No network', 6)]

Kimi-K3 hub cost: {"slots": 50, "usd_total": 138.2, "usd_per_slot_mean": 2.764, "usd_per_slot_median": 2.214, "prompt_Mtok_per_slot": 11.52, "cached_pct": 96.6, "completion_ktok_per_slot": 75.3, "calls_per_slot": 108.1}

GLM-5.3 hub cost: {"slots": 2, "usd_total": 6.33, "usd_per_slot_mean": 3.166, "usd_per_slot_median": 3.166, "prompt_Mtok_per_slot": 24.97, "cached_pct": 99.1, "completion_ktok_per_slot": 117.8, "calls_per_slot": 151.0}

DeepSeek-V4-Pro hub cost: {"slots": 2, "usd_total": 2.59, "usd_per_slot_mean": 1.293, "usd_per_slot_median": 1.293, "prompt_Mtok_per_slot": 15.76, "cached_pct": 99.3, "completion_ktok_per_slot": 95.3, "calls_per_slot": 121.0}
