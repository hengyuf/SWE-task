#!/bin/bash
# Dev: start one vLLM engine spanning 2 nodes of an existing allocation (srun --overlap into JOB).
# Usage: dev_launch2.sh <name> <mode> -- <vllm args shared by both ranks>
#   mode mp : torch-distributed multi-node (TP/PP over 2 nodes): rank0 --nnodes 2 --node-rank 0 --master-addr IP0; rank1 adds --headless
#   mode dp : data-parallel over nodes (one DP rank group per node): --data-parallel-size-local and --data-parallel-address IP0; rank1 --headless --data-parallel-start-rank
# Env: JOB (Slurm job id), NODE0, NODE1, VENV, DP_LOCAL (dp mode: DP ranks per node, default 1), DP_SIZE (default 2*DP_LOCAL)
set -euo pipefail
G=/lustre/fsw/portfolios/nemotron/users/hengyuf/glm53-serve
IMG=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/runtime/images/nemo-rl-cuda13.2-base-20260903-x86_64.sqsh
name="$1"; mode="$2"; shift 2; [[ "${1:-}" == "--" ]] && shift
: "${JOB:?}" "${NODE0:?}" "${NODE1:?}"
IP0=$(getent hosts "$NODE0" | awk '{print $1; exit}')
VENV="${VENV:-$G/env/vllm-0.32.0}"
common_env=(VENV="$VENV" NCCL_SOCKET_IFNAME=enp90s0np0 GLOO_SOCKET_IFNAME=enp90s0np0 ${EXTRA_ENV:-})
run_rank() {  # node rank args...
  local node="$1" rank="$2"; shift 2
  cd /tmp && nohup srun --jobid="$JOB" --overlap -N1 -n1 -w "$node" --container-image="$IMG" --container-name=glm53 \
    --container-mounts=/lustre/fsw:/lustre/fsw,/raid/scratch:/raid/scratch --no-container-mount-home --container-workdir=/tmp \
    env "${common_env[@]}" LOGF="$G/logs/$name-rank$rank.log" bash "$G/scripts/dev_serve.sh" "$@" \
    > "$G/logs/$name-rank$rank.srun.log" 2>&1 &
  echo "rank $rank on $node: srun pid $!"
}
case "$mode" in
  mp)
    run_rank "$NODE1" 1 "$@" --nnodes 2 --node-rank 1 --master-addr "$IP0" --headless
    run_rank "$NODE0" 0 "$@" --nnodes 2 --node-rank 0 --master-addr "$IP0" ;;
  dp)
    L="${DP_LOCAL:-1}"; S="${DP_SIZE:-$((2*L))}"
    run_rank "$NODE1" 1 "$@" --data-parallel-size "$S" --data-parallel-size-local "$L" --data-parallel-start-rank "$L" \
      --data-parallel-address "$IP0" --data-parallel-rpc-port 13345 --headless
    run_rank "$NODE0" 0 "$@" --data-parallel-size "$S" --data-parallel-size-local "$L" \
      --data-parallel-address "$IP0" --data-parallel-rpc-port 13345 ;;
  *) echo "unknown mode $mode"; exit 2 ;;
esac
echo "endpoint: http://$IP0:8000/v1  logs: $G/logs/$name-rank{0,1}.log"
