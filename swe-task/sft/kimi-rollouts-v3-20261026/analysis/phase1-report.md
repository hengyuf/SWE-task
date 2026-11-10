# phase1-report (sel50; 50 tasks)

| metric | base Nemotron sc14 | Kimi-K3 previous (1M, no sentence) | Kimi-K3 v3 (256k + sentence) |
|---|---|---|---|
| rollouts | 250 | 250 | 250 |
| graded | 250 | 248 | 248 |
| infra | 0 | 2 | 2 |
| with_export | 250 | 250 | 248 |
| pass_at_1 | 17.2 | 77.4 | 48.8 |
| pass | 43 | 192 | 121 |
| turns_per_rollout_median | 207.5 | 93.5 | 73.0 |
| action_turns | 61039 | 26299 | 19578 |
| pct_action_turns_with_reasoning | 100.0 | 54.9 | 58.0 |
| ci95_pct_action_turns_with_reasoning | [99.9, 100.0] | [53.4, 56.4] | [56.2, 59.9] |
| rollout_mean_pct_action_reasoned | 100.0 | 55.0 | 58.1 |
| pct_rollouts_all_actions_reasoned | 90.8 | 0.0 | 0.8 |
| longest_unreasoned_run_median | 0.0 | 5.0 | 4.0 |
| longest_unreasoned_run_mean | 0.09 | 5.05 | 4.26 |
| longest_unreasoned_run_p90 | 0 | 7 | 7 |
| longest_unreasoned_run_max | 1 | 30 | 12 |
| pct_first_action_reasoned | 100.0 | 99.6 | 99.6 |
| reasoning_tokens_per_reasoning_turn_median | 126.0 | 148.0 | 143 |
| reasoning_tokens_per_reasoning_turn_mean | 658.5 | 799.6 | 1248.8 |
| reasoning_tokens_per_turn_incl_zero | 658.2 | 437.7 | 723.6 |
| reasoning_tokens_per_rollout_median | 125790.5 | 40762.5 | 45168.5 |
| pct_action_turns_with_visible_text | 5.1 | 32.2 | 34.1 |
| ci95_pct_action_turns_with_visible_text | [4.4, 5.8] | [31.0, 33.5] | [32.5, 35.8] |
| pct_action_turns_visible_without_reasoning | 0.0 | 2.2 | 2.1 |
| pct_action_turns_bare | 0.0 | 43.0 | 39.9 |
| visible_chars_median | 83 | 78.0 | 81 |
| pct_rollouts_git_commit | 93.6 | 97.6 | 66.5 |
| pct_rollouts_compacted | 58.0 | 0.0 | 3.6 |
| compactions_per_rollout_mean | 0.82 | 0.0 | 0.04 |
| agent_wall_min_median | 52.9 | 35.5 | 55.1 |

paired kimi3 - kimi: {"pct_action_turns_with_reasoning": {"diff": 3.1, "ci95": [1.7, 4.5], "tasks": 50}, "pct_action_turns_with_visible_text": {"diff": 1.9, "ci95": [0.6, 3.3], "tasks": 50}}

kimi3 without degenerate rollouts: {"rollouts": 175, "graded": 173, "infra": 2, "with_export": 173, "pass_at_1": 67.1, "pass": 116, "turns_per_rollout_median": 84, "action_turns": 15237, "pct_action_turns_with_reasoning": 58.0, "ci95_pct_action_turns_with_reasoning": [56.0, 60.0], "rollout_mean_pct_action_reasoned": 57.4, "pct_rollouts_all_actions_reasoned": 0.0, "longest_unreasoned_run_median": 4, "longest_unreasoned_run_mean": 4.47, "longest_unreasoned_run_p90": 7, "longest_unreasoned_run_max": 12, "pct_first_action_reasoned": 100.0, "reasoning_tokens_per_reasoning_turn_median": 139.0, "reasoning_tokens_per_reasoning_turn_mean": 859.1, "reasoning_tokens_per_turn_incl_zero": 496.5, "reasoning_tokens_per_rollout_median": 39569, "pct_action_turns_with_visible_text": 35.2, "ci95_pct_action_turns_with_visible_text": [33.4, 37.0], "pct_action_turns_visible_without_reasoning": 2.3, "pct_action_turns_bare": 39.8, "visible_chars_median": 80, "pct_rollouts_git_commit": 92.5, "pct_rollouts_compacted": 4.0, "compactions_per_rollout_mean": 0.04, "agent_wall_min_median": 56.5}
paired kimi3(clean) - kimi: {"pct_action_turns_with_reasoning": {"diff": 3.1, "ci95": [1.5, 4.6], "tasks": 50}}

| window metric | kimi3 | kimi | ctrl-old |
|---|---|---|---|
| action_turns_after_last_compaction_median | 41 | None | None |
| agent_errors | {'TimeoutError': 5} | {'TimeoutError': 1} | {} |
| calls_cut_at_output_cap | 0 | 0 | 0 |
| calls_over_compaction_point | 2 | 1250 | 0 |
| compacted_early_stop | 0 | 0 | 0 |
| compacted_ended_notool | 9 | 0 | 0 |
| compacted_ended_on_compaction | 0 | 0 | 0 |
| compaction_trigger_prompt_median | 241301 | None | None |
| compactions_hist | {'0': 239, '1': 9} | {'0': 250} | {'0': 19} |
| compactions_per_compacted_rollout_mean | 1.0 | None | None |
| compactions_per_rollout_mean | 0.04 | 0.0 | 0.0 |
| cost_reported_usd_mean | 2.994 | 2.82 | 1.631 |
| degenerate_hub_replies | 77 | 0 | 5 |
| graded | 248 | 248 | 18 |
| graded_compacted | 9 | 0 | 0 |
| graded_not_compacted | 239 | 248 | 18 |
| hub_errors | {'408:other': 20, '500:other': 3, '429:rate_limit': 7, '502:other': 1} | {'500:other': 3, '408:other': 1, '422:budget': 2, '503:other': 1, '401:other': 2} | {'408:other': 1} |
| hub_replies | 20549 | 27163 | 1065 |
| infra | 2 | 2 | 2 |
| max_prompt_tokens | 243395 | 505801 | 144434 |
| pass_at_1 | 48.8 | 77.4 | 66.7 |
| pass_at_1_compacted | 44.4 | None | None |
| pass_at_1_excluding_degenerate_end | 68.0 | 77.4 | 85.7 |
| pass_at_1_not_compacted | 49.0 | 77.4 | 66.7 |
| pct_compacted_with_any_commit | 66.7 | None | None |
| pct_compacted_with_commit_after_last_compaction | 66.7 | None | None |
| pct_not_compacted_with_any_commit | 66.5 | 97.6 | 78.9 |
| pct_rollouts_compacted | 3.6 | 0.0 | 0.0 |
| pct_rollouts_max_prompt_over_242k | 0.8 | 4.8 | 0.0 |
| rollouts | 250 | 250 | 20 |
| rollouts_ended_by_degenerate_reply | 70 | 0 | 4 |
| rollouts_with_context_overflow | 0 | 0 | 0 |
| rollouts_with_cut_call | 0 | 0 | 0 |
| session_errors | {'UnknownError': 5} | {'UnknownError': 2} | {} |
| with_export | 248 | 250 | 19 |

degenerate hub replies (kimi3):
  all: {"rollouts": 250, "hub_replies": 20549, "degenerate_replies": 77, "degenerate_per_1000_replies": 3.75, "pct_rollouts_with_degenerate_reply": 30.0, "rollouts_ended_by_degenerate_reply": 70, "pct_rollouts_ended_by_degenerate_reply": 28.0, "pass_at_1_raw": 48.8, "graded": 248, "pass_at_1_excluding_degenerate_end": 68.0, "graded_excluding_degenerate_end": 178, "degenerate_endings_that_failed": 70}
  compacted: {"rollouts": 9, "hub_replies": 1894, "degenerate_replies": 2, "degenerate_per_1000_replies": 1.06, "pct_rollouts_with_degenerate_reply": 22.2, "rollouts_ended_by_degenerate_reply": 2, "pct_rollouts_ended_by_degenerate_reply": 22.2, "pass_at_1_raw": 44.4, "graded": 9, "pass_at_1_excluding_degenerate_end": 57.1, "graded_excluding_degenerate_end": 7, "degenerate_endings_that_failed": 2}
  not_compacted: {"rollouts": 239, "hub_replies": 18655, "degenerate_replies": 75, "degenerate_per_1000_replies": 4.02, "pct_rollouts_with_degenerate_reply": 30.5, "rollouts_ended_by_degenerate_reply": 68, "pct_rollouts_ended_by_degenerate_reply": 28.5, "pass_at_1_raw": 49.0, "graded": 239, "pass_at_1_excluding_degenerate_end": 68.4, "graded_excluding_degenerate_end": 171, "degenerate_endings_that_failed": 68}

degenerate hub replies (kimi):
  all: {"rollouts": 250, "hub_replies": 27163, "degenerate_replies": 0, "degenerate_per_1000_replies": 0.0, "pct_rollouts_with_degenerate_reply": 0.0, "rollouts_ended_by_degenerate_reply": 0, "pct_rollouts_ended_by_degenerate_reply": 0.0, "pass_at_1_raw": 77.4, "graded": 248, "pass_at_1_excluding_degenerate_end": 77.4, "graded_excluding_degenerate_end": 248, "degenerate_endings_that_failed": 0}
  compacted: {"rollouts": 0, "hub_replies": 0, "degenerate_replies": 0, "degenerate_per_1000_replies": null, "pct_rollouts_with_degenerate_reply": null, "rollouts_ended_by_degenerate_reply": 0, "pct_rollouts_ended_by_degenerate_reply": null, "pass_at_1_raw": null, "graded": 0, "pass_at_1_excluding_degenerate_end": null, "graded_excluding_degenerate_end": 0, "degenerate_endings_that_failed": 0}
  not_compacted: {"rollouts": 250, "hub_replies": 27163, "degenerate_replies": 0, "degenerate_per_1000_replies": 0.0, "pct_rollouts_with_degenerate_reply": 0.0, "rollouts_ended_by_degenerate_reply": 0, "pct_rollouts_ended_by_degenerate_reply": 0.0, "pass_at_1_raw": 77.4, "graded": 248, "pass_at_1_excluding_degenerate_end": 77.4, "graded_excluding_degenerate_end": 248, "degenerate_endings_that_failed": 0}

cost: {"kimi3": {"slots": 256, "usd_total": 781.48, "usd_per_slot_mean": 3.053, "usd_per_slot_median": 2.263, "prompt_Mtok_per_slot": 7.6, "cached_pct": 84.1, "completion_ktok_per_slot": 68.9, "calls_per_slot": 84.3}, "kimi_previous_r0": {"slots": 50, "usd_total": 138.2, "usd_per_slot_mean": 2.764, "usd_per_slot_median": 2.214, "prompt_Mtok_per_slot": 11.52, "cached_pct": 96.6, "completion_ktok_per_slot": 75.3, "calls_per_slot": 108.1}}
hub keys (leg starts): {"NVAPI_KEY_NEW_2": 12}
projection: {"remaining_tasks": 144, "hengyuf": 47, "baihe": 97, "rollouts": 720, "usd_per_rollout_measured": 3.053, "size_factor_previous_kimi_token_length": 1.064, "usd_total_unscaled": 2198, "usd_total_size_scaled": 2338, "usd_hengyuf_rest47": 717, "usd_baihe97": 1481}
