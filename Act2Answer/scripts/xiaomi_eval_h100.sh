#!/usr/bin/env bash
# Run Act2Answer eval with the Xiaomi client (env act2ans -> TCP :10086 -> mibot server).
set -uo pipefail
source ~/conda/etc/profile.d/conda.sh
conda activate act2ans

R=/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer
export REPO_ROOT=$R
export PYTHONPATH=$R/SimplerEnv:$R/ManiSkill
export TOKENIZERS_PARALLELISM=false
export XIAOMI_TASK_ID=${XIAOMI_TASK_ID:-bridge_delta}

ASSETS=${ASSETS:-pairs_choice_vla_confirm}
COUNT=${COUNT:-3}
START_ID=${START_ID:-0}
EVAL_GPU=${EVAL_GPU:-0}
INFERBATCH=${INFERBATCH:-3}

cd "$R/SimplerEnv"   # overlay bg relative path
SWAP_ARGS=()
[ "${DO_SWAP:-0}" = "1" ] && SWAP_ARGS=(--do-swap)

CUDA_VISIBLE_DEVICES=$EVAL_GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python3 -u -m simpler_env.eval \
    --vla xiaomi --start-id "$START_ID" --count "$COUNT" \
    --assets "$ASSETS" --obj-set "${OBJ_SET:-test}" \
    --buffer-inferbatch "$INFERBATCH" "${SWAP_ARGS[@]}"
