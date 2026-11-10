#!/bin/bash
# verify-2 (2026-11-04): independent recomputation of the v3etp20 train-acc analysis from RAW data
# (results/*.json, probes/<model>/group-*/.../attempt-*/{http.jsonl,started.json,terminal.json,generation-validity-v2-r7.json},
#  fleets/<model>/logs/vllm-*.log, routers/<model>-group-*/student-router-requests.jsonl, logs/ta-*.out,
#  SFT corpus selection files, DeepSWE run dirs (rollouts.jsonl + OpenCode export.json)). Own code only; read-only elsewhere.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
V=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024/analysis/v3etp20/verify-2
PY="/usr/bin/python3 -I -B"
cd $V/acc && $PY acc_verify.py && $PY rate_tests.py                       # pass@1/5, CIs, 10 paired tests, empty-patch tests, infra sensitivity
cd $V/extract
for m in v3etp20 v3etp10 v3etp5 etp20 sc14 etp10; do $PY extract_calls.py $m --procs 16; done   # per-call journal extraction
for m in v3etp20 v3etp10 v3etp5 etp20 sc14 etp10; do $PY metrics.py $m > /dev/null; done        # out/metrics_<m>.txt, out/slots_<m>.jsonl
$PY compare.py                                                             # out/compare.txt (paired per-rollout tests, serving counterfactual)
for m in v3etp20 v3etp10 etp20 sc14 v3etp5; do $PY textscan.py $m --procs 8; done             # foreign-script runs, loop regions
$PY text_summary.py sc14 etp20 v3etp5 v3etp10 v3etp20 > out/text_summary.txt
$PY reasoning_len.py sc14 etp20 v3etp5 v3etp10 v3etp20 > out/reasoning_len.txt                # char-based proxy (no tokenizer)
cd $V/serving && $PY router_stats.py v3etp20 > router_v3etp20.txt && $PY engine_stats.py v3etp20 v3etp10 v3etp5 etp20 base > engine_stats.txt
cd $V/coverage && $PY coverage_verify.py > coverage_verify.txt
cd $V/deepswe && $PY ds_compaction.py                                      # ds_compaction.txt
cd $V && $PY adhoc/markup_turns.py extract/out/v3etp20.jsonl.gz            # reasoning turns with tool markup / special tokens
