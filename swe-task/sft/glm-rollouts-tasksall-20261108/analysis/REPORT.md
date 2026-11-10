# glm-rollouts-tasksall: report

Self-hosted GLM-5.3 x 5 on the tasks of sft/tasks-all-20261106 without GLM-5.3 rollouts: 15/82 campaigns done; **1001/1622 = 61.7%** of graded rollouts pass (1,625 rollouts, 3 without a grade).

| tier | pass / graded |
|---|---|
| 1 accepted/strong | 260/395 = 65.8% |
| 2 partial | 397/727 = 54.6% |
| 3 weak | 236/375 = 62.9% |
| 4 none/unlabeled | 108/125 = 86.4% |

| inventory status | pass / graded |
|---|---|
| accepted | 226/295 = 76.6% |
| rejected | 760/1307 = 58.1% |
| withdrawn | 15/20 = 75.0% |

| owner | pass / graded |
|---|---|
| baihe | 111/130 = 85.4% |
| hengyuf | 890/1492 = 59.7% |

- Passes per task (of 5): {0: 69, 1: 21, 2: 21, 3: 38, 4: 56, 5: 120}.
- Same tasks, build probes (5 slots each): GLM-5.3 996/1617 = 61.6% vs Kimi-K3 954/1586 = 60.2% vs sc14 360/1618 = 22.2% (324 tasks).
- OpenCode compaction: 565 rollouts (249 pass); passes without compaction: 752.
- Replies >= 131,072 tokens: 0 rollouts; >= 32k: 331; max prompt 250219 tokens; median 144 LLM calls and 123483 generated tokens per rollout.
- Final attempt index: {'attempt-0': 1604, 'attempt-2': 8, 'attempt-1': 13}; sandbox-failure re-runs per campaign: none.
- Agent-timeout slots from the 2-server dip (2026-11-09 03:10-04:50) re-run once and applied: {'slots': 44, 'now_pass': 19, 'now_fail': 25} (results/glmlocal-dip1.json).
- Pending campaigns: t3-h03, t3-h04, t3-h05, t3-h06, t3-h07, t3-h08, t3-h09, t3-h10, t3-h11, t3-h12, t3-h13, t4-h00, t4-h01, t4-b01, t4-h02, t4-b02, t4-h03, t4-b03, t4-h04, t4-b04, t4-h05, t4-b05, t4-h06, t4-b06, t4-h07, t4-b07, t4-h08, t4-b08, t4-h09, t4-b09, t4-h10, t4-b10, t4-h11, t4-b11, t4-h12, t4-b12, t4-h13, t4-b13, t4-h14, t4-b14, t4-h15, t4-b15, t4-h16, t4-b16, t4-h17, t4-b17, t4-h18, t4-b18, t4-h19, t4-b19, t4-h20, t4-b20, t4-h21, t4-b21, t4-h22, t4-h23, t4-h24, t4-h25, t4-h26, t4-h27, t4-h28, t4-h29, t4-h30, t4-h31, t4-h32, t4-h33, t4-h34.
- Files: `analysis/summary.json`, `analysis/rollouts.jsonl` (one row per rollout, with the task key).
