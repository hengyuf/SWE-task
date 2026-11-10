# glm-rollouts-sftv3: final report

Self-hosted GLM-5.3 x 5 on the 220 SFT-v3 tasks: **885/1100 = 80.5%** of graded rollouts pass (1100 rollouts, 0 ended without a grade).

| campaign | pass | fail | infra | rate |
|---|---|---|---|---|
| hengyuf-g0 | 126 | 29 | 0 | 81.3% |
| hengyuf-g1 | 116 | 39 | 0 | 74.8% |
| hengyuf-g2 | 130 | 25 | 0 | 83.9% |
| hengyuf-g3 | 124 | 31 | 0 | 80.0% |
| baihe-g0 | 97 | 23 | 0 | 80.8% |
| baihe-g1 | 91 | 29 | 0 | 75.8% |
| baihe-g2 | 97 | 23 | 0 | 80.8% |
| baihe-g3 | 104 | 16 | 0 | 86.7% |

- Passes per task (of 5): {0: 6, 1: 8, 2: 16, 3: 26, 4: 53, 5: 111} (tasks with 0 passes: 6).
- By task source: v2-baihe 389/480 = 81.0%; v2-hengyuf 361/455 = 79.3%; v5-official2 135/165 = 81.8%.
- Same tasks, Kimi-K3 (v4/v5 final records): GLM 885/1100 = 80.5% vs Kimi 883/1103 = 80.1% on 220 tasks.
- Hosted GLM-5.3 (teacher-rollouts-v2, 1M window) on the 48 overlapping tasks: local 188/240 = 78.3% vs hosted 187/241 = 77.6%.
- OpenCode compaction: 398 rollouts (308 of them pass); passes without compaction: 577.
- Rollouts with a reply >= 131,072 tokens (above the old cap; allowed by the 256k limit): 0; with a reply >= 32k: 260; max prompt 261,966 tokens. Median LLM calls per rollout 141.0, median generated tokens 125,079.0.
- Final attempt index: {'attempt-0': 634, 'attempt-1': 460, 'attempt-2': 6} (attempt-1+ = ATP retried an interrupted attempt).
- Ungraded slots re-run (user 2026-11-08, `results/glmlocal-infra-rerun1.json`): hengyuf-g0 syn-litestar-cand-c01-r1b1-8f1b686c glmlocal-r1: infra -> pass; hengyuf-g0 syn-yq-cand-c05-cbe61bd7 glmlocal-r0: infra -> pass; hengyuf-g3 syn-goleveldb-cand-c01-r1b1-7523606b glmlocal-r1: infra -> pass.
- Records: `results/glmlocal-<campaign>.json`; flat list `analysis/final-rollouts.jsonl`; summary `analysis/final-summary.json`.
