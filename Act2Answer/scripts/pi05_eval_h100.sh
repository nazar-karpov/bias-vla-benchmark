#!/usr/bin/env bash
# Run Act2Answer eval with the pi05 client (env magma_act2answer -> zmq -> pi05 server).
set -uo pipefail
source ~/conda/etc/profile.d/conda.sh
conda activate magma_act2answer

R=/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer
export REPO_ROOT=$R
export PYTHONPATH=$R/SimplerEnv:$R/ManiSkill
export TOKENIZERS_PARALLELISM=false
export PI05_HOST=127.0.0.1
export PI05_PORT=${PI05_PORT:-20005}

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
    --vla pi05 --start-id "$START_ID" --count "$COUNT" \
    --assets "$ASSETS" --obj-set "${OBJ_SET:-test}" \
    --buffer-inferbatch "$INFERBATCH" "${SWAP_ARGS[@]}"
