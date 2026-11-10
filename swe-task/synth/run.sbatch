#!/bin/bash
# Run one synth command in a CPU job.   Usage: sbatch run.sbatch <script.py | -m module> [args...]
#   sbatch run.sbatch smoke.py --run smoke1 s1 s3 s4
#   sbatch run.sbatch failure_analyzer.py --run val1 --tasks abs-stepped-slices
#SBATCH --job-name=synth
#SBATCH --account=nemotron_agents_dev
#SBATCH --partition=cpu_datamover
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
# Not --exclusive (Baihe's setting): the CPU node only orchestrates (agents' commands and rollouts run in
# OpenSandbox, models over HTTP). Measured: val1 build 0.29 cores avg / 17.5 GB peak; smoke s2 probe 0.50 cores /
# 10.5 GB. 96 GB leaves room for official probes (up to ~250 concurrent rollouts; ATP itself uses 24 CPU / 96 GB).
#SBATCH --time=12:00:00
#SBATCH --export=NONE
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth/runs/slurm/%x-%j.out
set -euo pipefail
# The code reads every credential from a file when it needs it (hub key: NVAPI_KEY_NEW in hengyuf's .env; OpenSandbox:
# ../secrets/opensandbox.env), so drop anything secret-like or session-bound inherited from the submitting shell.
for name in $(compgen -e); do
  case "${name^^}" in
    *KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*CREDENTIAL*) unset "$name" 2>/dev/null || true ;;
  esac
done
unset SYNTH_STUDENT_BASE_URL  # every job starts (or finds) its own student router (student/client.py)
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
cd /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth
exec /lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/atp/runtime/controller/bin/python3 -B "$@"
