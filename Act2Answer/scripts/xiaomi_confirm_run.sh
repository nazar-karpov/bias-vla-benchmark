#!/usr/bin/env bash
# Resumable full confirm run for Xiaomi on pairs_choice_vla_confirm.
# 1600 episodes x {noswap, swap}, chunked, skip-on-done markers.
# Requires the Xiaomi policy server already up on :10086 (xiaomi_server_h100.sh).
set -uo pipefail
source ~/conda/etc/profile.d/conda.sh
conda activate act2ans

R=/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer
export REPO_ROOT=$R
export PYTHONPATH=$R/SimplerEnv:$R/ManiSkill
export TOKENIZERS_PARALLELISM=false
export XIAOMI_TASK_ID=${XIAOMI_TASK_ID:-bridge_delta}

ASSETS=pairs_choice_vla_confirm
TOTAL=${TOTAL:-1600}
CHUNK=${CHUNK:-50}
# CRITICAL: Xiaomi client shares a single action_plans list sized to the FIRST
# sub-batch; if buffer_inferbatch < count the episodes past the first sub-batch
# get another episode's cached actions (episode mixing). So inferbatch MUST equal
# the chunk size (one sub-batch), exactly like the stock eval_xiaomi.sh.
INFERBATCH=$CHUNK
EVAL_GPU=${EVAL_GPU:-0}
STATE=~/logs/xiaomi_confirm_state
mkdir -p "$STATE"

cd "$R/SimplerEnv"

run_pol () {
  local pol="$1"; local swapflag="$2"
  for (( s=0; s<TOTAL; s+=CHUNK )); do
    local marker="$STATE/${pol}_s${s}.done"
    if [ -f "$marker" ]; then echo "SKIP ${pol} s${s} (done)"; continue; fi
    echo "RUN ${pol} s${s}/${TOTAL} $(date -u)"
    local extra=()
    [ "$swapflag" = "1" ] && extra=(--do-swap)
    if CUDA_VISIBLE_DEVICES=$EVAL_GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python3 -u -m simpler_env.eval \
          --vla xiaomi --start-id "$s" --count "$CHUNK" \
          --assets "$ASSETS" --obj-set test \
          --buffer-inferbatch "$INFERBATCH" "${extra[@]}"; then
      touch "$marker"
    else
      echo "FAILED ${pol} s${s}" ; touch "$STATE/${pol}_s${s}.failed"
    fi
  done
}

echo "START_XIAOMI_CONFIRM $(date -u)"
run_pol noswap 0
run_pol swap 1
echo "DONE_XIAOMI_CONFIRM $(date -u)"
touch "$STATE/ALL_DONE"
