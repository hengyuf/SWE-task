#!/bin/bash
# Builds the two ready replay sets (64 DeepSWE GLM sessions, 64 synthetic-task GLM sessions). Login node OK.
set -euo pipefail
cd /lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/tools/replay
export PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
$PY -I build_replay_set.py \
  --source deepswe=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-deepswe-20261030/dataset/glm.jsonl.gz \
  --n-sessions 64 --seed 0 --workers 8 --verify-every 20 \
  --cross-check-template /lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/hsg-ref/model_meta/chat_template.jinja \
  --out data/deepswe_glm_64.jsonl.gz
$PY -I build_replay_set.py \
  --source synth=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/teacher-rollouts-v2-20261025/features/glm.jsonl \
  --n-sessions 64 --seed 0 --workers 8 --verify-every 20 \
  --cross-check-template /lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve/hsg-ref/model_meta/chat_template.jinja \
  --out data/synth_v2_glm_64.jsonl.gz
echo BUILD_ALL_DONE
