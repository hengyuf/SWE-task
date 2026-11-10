#!/usr/bin/env bash
# train-acc-v2 (2026-10-24): submit everything through the schedule helper (babysit pairs: one candidate per account).
#   bash submit.sh workers [FLEET ...]  # per fleet 10 worker slots: leg 1 now, and leg 2 queued after leg 1 (afterany on
#                                       # both candidates), so serving continues past 4 h. Default fleets: sftv2 base
#   bash submit.sh drivers [MODEL ...]  # 8 probe campaigns per model (driver.sbatch on cpu_dataprocessing). Default
#                                       # models: sftv2 sc14
# Fleets / models: sftv2 = v2 SFT 5 ep (iter38), base = sc14 (driver model "sc14"); added 2026-10-25 (user): sftv2e10 =
# v2 SFT 10 ep (iter76), sftv2e20 = v2 SFT 20 ep (iter152). The 10ep/20ep fleets serve with the counting variant of the
# lenient-think parser (assets/nano_v3_lenient_think_counted_parser.py: same parse + TA_THINK_* log lines).
# Added 2026-10-27 (user, "fix the thinking pattern"): etp5 / etp10 = the v2 corpus with an empty think prefix, 5 ep
# (iter38) and 10 ep (iter76), 16 nodes / EP32; served like the 10/20 ep fleets (counting parser).
# Added 2026-10-30 (user): v3etp5 / v3etp10 = the SFT-v3 corpus (sft-v3-20261029: v2 conversations <= 262,144 tokens +
# Kimi-K3 v4/v5 passes, 1,594 conversations) with the empty think prefix, WSD 5 ep (iter71) and 10 ep (iter142),
# 16 nodes / EP32; served like the etp fleets (counting parser).
# Added 2026-11-04 (user): v3etp20 = the same SFT-v3 empty-think-prefix recipe, WSD 20 ep (iter284).
# Every pair has its own helper state dir (RESULTS_DIR) under babysit/. Prints "<what> <leg> <id0>,<id1>".
# Knobs (2026-10-25), for re-submitting part of a fleet or campaign:
#   workers: TA_SLOTS (default "00 .. 09"; a bridge uses new slots, e.g. "10 11"), TA_LEGS (2 = leg 1 + queued leg 2,
#            1 = one leg only), TA_PARTITION (batch), TA_TIME (worker.sbatch's 04:00:00 when unset), STUDENT_DRAIN_S
#            (passed through to serve_model.sh; use 900 for a 2 h bridge)
#   drivers: TA_GROUPS (default "0 .. 7"): rerun only these groups (probe() resumes a campaign); TA_DRIVER_PARTITION
#            (default cpu_dataprocessing; probe with sbatch --test-only and use the CPU partition that starts first:
#            cpu_datamover / cpu_interactive share another node pool); TA_DRIVER_DEPENDENCY (2026-10-30; sbatch
#            --dependency, e.g. "after:<w00 cand 0>?after:<w00 cand 1>?after:<w01 cand 0>?..." over the fleet's leg-1
#            worker candidates: the drivers start with the fleet's first worker, because driver.py waits <= 2 h for 8
#            workers, then <= 1 h for one, and fails after that; a pair's losing candidate is cancelled when its winner
#            starts)
set -euo pipefail
TA=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024
HELPER=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/recipes/scale-rebench-super-sc-length-formal-v1/fullscale-start-v1/overrides/r3-network-v1/launch/schedule_helper.sh
# 2026-10-25: + pool0-00064, pool0-00402, pool0-00989 (vLLM failed to start there: "CUDA driver error: file not found").
# 2026-10-25 14:55: + pool0-01239, pool0-01856 (same error in the 10ep fleet: inductor autotuning "CUDA driver error: file not found").
# 2026-10-27 15:34: + pool0-00525 (etp10 w07: vLLM not ready 1200 s after the job start; worker marked the node suspect).
# 2026-10-30: + pool0-00825, pool0-00826, pool0-00955 (etp5 DeepSWE shards failed vLLM startup there on 2026-10-27),
# pool0-01764, pool0-01884 (DeepSWE shards: node-specific "vLLM not ready in time" hangs); the list now equals the
# DeepSWE evals' 20-node exclude list.
# 2026-10-31: + pool0-01888 (v3etp10 leg-2 worker 21094326: vLLM exited during startup, "Engine core initialization failed").
# 2026-11-04: + pool0-01447, pool0-00965 (DeepSWE shards hung at vLLM startup there on 10-31 and 11-02).
EXCLUDE=pool0-00058,pool0-00484,pool0-00502,pool0-01460,pool0-00613,pool0-00196,pool0-01970,pool0-00128,pool0-00560,pool0-00064,pool0-00402,pool0-00989,pool0-01239,pool0-01856,pool0-00525,pool0-00825,pool0-00826,pool0-00955,pool0-01764,pool0-01884,pool0-01888,pool0-01447,pool0-00965
REAPER='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"inference_server","description":"1-node 8-GPU vLLM inference worker for the SFT training-accuracy test (4h job); GPUs idle between rollout bursts"}}'
SFT=/lustre/fsw/portfolios/nemotron/projects/nemotron_evals_research/users/hengyuf
declare -A MODEL=(
  [sftv2]=$SFT/super-kimi-raw-sft-v2-wsd-5ep-20261023/exports/iter_0000038/hf
  [base]=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/models/swonlysc-step0014/hf
  [sftv2e10]=$SFT/super-kimi-raw-sft-v2-wsd-10ep-20261024/exports/iter_0000076/hf
  [sftv2e20]=$SFT/super-kimi-raw-sft-v2-wsd-20ep-20261024/exports/iter_0000152/hf
  [etp5]=$SFT/super-kimi-v2-empty-think-prefix-wsd-5ep-16n-ep32-20261027/exports/iter_0000038/hf
  [etp10]=$SFT/super-kimi-v2-empty-think-prefix-wsd-10ep-16n-ep32-20261027/exports/iter_0000076/hf
  [etp20]=$SFT/super-kimi-v2-empty-think-prefix-wsd-20ep-16n-ep32-20261027/exports/iter_0000152/hf  # 2026-10-27 22:05 (user): the same, 20 ep
  [v3etp5]=$SFT/super-kimi-sft-v3-empty-think-prefix-wsd-5ep-16n-ep32-20261029/exports/iter_0000071/hf  # 2026-10-30 (user): SFT-v3, 5 ep
  [v3etp10]=$SFT/super-kimi-sft-v3-empty-think-prefix-wsd-10ep-16n-ep32-20261029/exports/iter_0000142/hf  # 2026-10-30 (user): SFT-v3, 10 ep
  [v3etp20]=$SFT/super-kimi-sft-v3-empty-think-prefix-wsd-20ep-16n-ep32-20261029/exports/iter_0000284/hf  # 2026-11-04 (user): SFT-v3, 20 ep
)
declare -A COUNTED=([sftv2e10]=1 [sftv2e20]=1 [etp5]=1 [etp10]=1 [etp20]=1 [v3etp5]=1 [v3etp10]=1 [v3etp20]=1)
DRIVER_MODELS=" sftv2 sc14 sftv2e10 sftv2e20 etp5 etp10 etp20 v3etp5 v3etp10 v3etp20 "
mkdir -p "$TA/logs"
case "${1:-}" in
  workers)
    shift; fleets=("$@"); [[ ${#fleets[@]} -gt 0 ]] || fleets=(sftv2 base)
    for fleet in "${fleets[@]}"; do
      [[ -n "${MODEL[$fleet]:-}" ]] || { echo "unknown fleet $fleet" >&2; exit 2; }
      [[ -f "${MODEL[$fleet]}/config.json" ]] || { echo "missing checkpoint for $fleet: ${MODEL[$fleet]}" >&2; exit 2; }
      mkdir -p "$TA/fleets/$fleet/registry" "$TA/fleets/$fleet/logs"
      if [[ -n "${COUNTED[$fleet]:-}" ]]; then
        export TA_REASONING_PARSER=nano_v3_lenient_think_counted
        export TA_REASONING_PARSER_PLUGIN="$TA/assets/nano_v3_lenient_think_counted_parser.py"
      else
        unset TA_REASONING_PARSER TA_REASONING_PARSER_PLUGIN
      fi
      for slot in ${TA_SLOTS:-00 01 02 03 04 05 06 07 08 09}; do
        export RESULTS_DIR="$TA/babysit/$fleet-w$slot" BATCH_SCRIPT="$TA/worker.sbatch" EXP_NAME="ta-$fleet-w$slot"
        export STUDENT_ROOT="$TA/fleets/$fleet" STUDENT_MODEL_PATH="${MODEL[$fleet]}"
        mkdir -p "$RESULTS_DIR"
        extra=(--partition="${TA_PARTITION:-batch}" --exclude="$EXCLUDE" --comment="$REAPER")
        [[ -n "${TA_TIME:-}" ]] && extra+=(--time="$TA_TIME")
        # 2026-10-27: TA_BEGIN (sbatch --begin, e.g. now+3hours) for tail workers that must be up before leg 1 drains
        [[ -n "${TA_BEGIN:-}" ]] && extra+=(--begin="$TA_BEGIN")
        leg1=$(source "$HELPER" && babysit_submit "${extra[@]}")
        echo "$EXP_NAME leg1 $leg1"
        [[ "${TA_LEGS:-2}" == 1 ]] && continue
        leg2=$(source "$HELPER" && BABYSIT_LEG=2 babysit_submit "${extra[@]}" --dependency="afterany:${leg1/,/:}")
        echo "$EXP_NAME leg2 $leg2"
      done
    done ;;
  drivers)
    shift; models=("$@"); [[ ${#models[@]} -gt 0 ]] || models=(sftv2 sc14)
    for model in "${models[@]}"; do
      [[ "$DRIVER_MODELS" == *" $model "* ]] || { echo "unknown driver model $model" >&2; exit 2; }
      for group in ${TA_GROUPS:-0 1 2 3 4 5 6 7}; do
        [[ "$group" =~ ^[0-7]$ ]] || { echo "bad group $group" >&2; exit 2; }
        export RESULTS_DIR="$TA/babysit/drv-$model-g$group" BATCH_SCRIPT="$TA/driver.sbatch" EXP_NAME="ta-drv-$model-g$group"
        export TA_MODEL="$model" TA_GROUP="$group"
        mkdir -p "$RESULTS_DIR"
        dextra=(--partition="${TA_DRIVER_PARTITION:-cpu_dataprocessing}")
        [[ -n "${TA_DRIVER_DEPENDENCY:-}" ]] && dextra+=(--dependency="$TA_DRIVER_DEPENDENCY")  # 2026-10-30, see above
        ids=$(source "$HELPER" && babysit_submit "${dextra[@]}")
        echo "$EXP_NAME leg1 $ids"
      done
    done ;;
  *) echo "usage: $0 workers [FLEET ...] | drivers [MODEL ...]" >&2; exit 2 ;;
esac
