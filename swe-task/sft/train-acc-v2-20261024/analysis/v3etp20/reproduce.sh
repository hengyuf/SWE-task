#!/bin/bash
# train-acc v3etp20 analysis (2026-11-04): every command, in order. Read-only on runs / results / probes / fleets / routers
# and on the other analysis dirs; writes only under this directory. Copied tools are marked "2026-11-04" in their docstrings;
# each copy reproduces the published columns of the earlier models (checks noted below).
# Python: /usr/bin/python3 -I -B; the two steps that import `tokenizers` / `orjson` / `scipy` (TR/features.py, the format
# detector) use the vLLM env python with the same flags, as the original tools did.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
V=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024/analysis/v3etp20
PY="/usr/bin/python3 -I -B"
VPY="/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/runtime/envs/vllm/bin/python -I -B"
CLS=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/eval/deepswe-v1.1/analysis/empty-patch-20261018/base-step0-rep5-out131k
EP=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024/analysis/empty-patch

# 1. per-call think analysis (http.jsonl of final attempts + counting-parser log lines) -> think/calls-v3etp20.jsonl
cd $V/think && $PY think_analysis.py --models v3etp20 --procs 12 > run-v3etp20.log
$PY split_table.py > split_table.log                 # earlier columns identical to analysis/think/split_table.json
$PY notool_kinds.py > notool_kinds.log               # recomputes all models; earlier ones identical
$PY reasoning_quality.py --models v3etp20 v3etp10 --procs 12 > reasoning_quality-v3etp20.log   # v3etp10 row reproduces
$PY runaway_units.py                                  # loop unit of every length-cut call (etp20, v3etp5/10/20)

# 2. DeepSWE empty-patch classifier on each slot's final attempt (v3etp10 re-run = reproduction check, 0 rows differ)
cd $V/empty-patch
for M in v3etp20 v3etp10; do
  $PY build_run.py $M
  $PY $CLS/analyze.py $M/run $M --procs 12 > $M/analyze.log
  $PY $CLS/summarize.py $M $M/run > $M/summary.txt
done
$PY table.py > table.log                              # earlier columns identical to analysis/empty-patch/table.json

# 3. teacher-rollout extractor (needs tokenizers) and comparison
cd $V/features
$VPY run_features.py --sources v3etp20 --workers 12 > run-features-v3etp20.log
$VPY run_features.py --sources v3etp10 --workers 12 --suffix=-recheck >> run-features-v3etp20.log   # byte-identical
$PY run_compare.py > run_compare.log

# 4. paired accuracy tests, cost
cd $V/accuracy && $PY paired.py > paired.log          # the 22 earlier pairs reproduce analysis/accuracy/paired.json
cd $V/cost && $PY cost.py > cost.log                  # earlier models and pairs reproduce analysis/cost/cost.json

# 5. comparison table, serving gaps, infra sensitivity, thinkx, per-slot join, rollout-level tests
cd $V/compare
args=""; for m in sc14 sftv2 sftv2e10 sftv2e20 etp5 etp10 etp20 v3etp5 v3etp10; do args="$args $m=$EP/$m/all.jsonl"; done
$PY thinkx.py --json thinkx_ta.json $args v3etp20=../empty-patch/v3etp20/all.jsonl > thinkx_ta.out
$PY serving_gaps.py > serving_gaps.out
$PY sensitivity_infra.py > sensitivity_infra.out
$PY compare_v3etp20.py > compare_v3etp20.out          # earlier columns identical to analysis/v3etp/comparison_v3etp.json

# 6. degenerate behaviors (degenerate-behaviors-20261031 lenses, copied; earlier models reproduce the published files)
cd $V/degenerate/endings
$PY extract_endings.py --procs 12 > logs/extract.log && $PY derive.py > logs/derive.log && $PY tables.py > logs/tables.log
cd $V/degenerate/text
$VPY scan_models.py --only trainacc --models sc14,etp20,v3etp5,v3etp10,v3etp20 --workers 10 > logs/scan_models.log
$PY aggregate.py > logs/aggregate.log && $PY tables.py > logs/tables.log
cd $V/degenerate/format
$VPY fmt_detect.py --out out --procs 8 ta-sc14=$EP/sc14/all.jsonl ta-etp20=$EP/etp20/all.jsonl ta-v3etp5=$EP/v3etp5/all.jsonl \
    ta-v3etp10=$EP/v3etp10/all.jsonl ta-v3etp20=../../empty-patch/v3etp20/all.jsonl > logs/fmt_detect.log
$VPY format_counts.py > /dev/null
cd $V/degenerate && $PY compaction_v3etp20.py > /dev/null
cd $V/degenerate/endings-ds-crosscheck
$PY extract_endings.py --procs 12 > logs/extract.log && $PY derive.py > logs/derive.log && $PY ds_crosscheck.py > /dev/null

# 7. joins that need the outputs above
cd $V/compare && $PY slots_v3etp20.py > /dev/null && $PY stats_tests.py > /dev/null
cd $V/coverage && $PY coverage_v3etp20.py > /dev/null
cd $V/live && $PY live_metrics_final.py > /dev/null
