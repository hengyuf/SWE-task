# Tier 1 + 2: GLM-5.3 vs Nemotron student (sc14), and acceptance with GLM-5.3 as the teacher

## Accuracy (same tasks)

| group | tasks | with GLM | GLM-5.3 | sc14 (Nemotron student) | Kimi-K3 (build probe) |
|---|---|---|---|---|---|
| tier 1, originally accepted | 267 | 267 | 1067/1335 = 79.9% | 103/1330 = 7.7% | 1119/1319 = 84.8% |
| tier 1, strong but not accepted | 34 | 34 | 83/170 = 48.8% | 31/170 = 18.2% | 73/169 = 43.2% |
| tier 2 (partial, none accepted) | 146 | 146 | 397/727 = 54.6% | 128/725 = 17.7% | 342/711 = 48.1% |
| all tier 1 + 2 | 447 | 447 | 1547/2232 = 69.3% | 262/2225 = 11.8% | 1534/2199 = 69.8% |

## Acceptance with GLM-5.3 as the teacher (original rule otherwise)

| group | tasks | accepted with GLM | rejected: GLM < 0.6 | rejected: student > 0.2 | similarity < 6 | rates OK, similarity never scored | leaky | inconclusive |
|---|---|---|---|---|---|---|---|---|
| tier 1, originally accepted | 267 | **231** | 36 | 0 | 0 | 0 | 0 | 0 |
| tier 1, strong but not accepted | 34 | **0** | 16 | 13 | 1 | 4 | 0 | 0 |
| tier 2 (partial, none accepted) | 146 | **0** | 62 | 42 | 25 | 17 | 0 | 0 |
| all tier 1 + 2 | 447 | **231** | 114 | 55 | 26 | 21 | 0 | 0 |

- Originally accepted (tier 1): 267; **still accepted with GLM-5.3 as the teacher: 231** (86.5%); lost: {'rejected: GLM rate < 0.6': 36}.
- Not accepted before, accepted with GLM-5.3: 0 ({} were their original reasons).
- GLM rollouts by source: {'self-hosted (SFT-v3 run, 2026-11-07)': 220, 'self-hosted (this campaign)': 225, 'hosted (teacher-rollouts-v2)': 2}.
- GLM's rate is its raw pass rate (its passes were not audited for cheating, unlike Kimi's clean rate), so 'accepted with GLM' is an upper bound on that check; similarity is the original score.
