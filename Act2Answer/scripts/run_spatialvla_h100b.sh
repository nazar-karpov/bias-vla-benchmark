#!/usr/bin/env bash
# SpatialVLA confirm-run worker for h100b.
# Params via env: START_ID, COUNT, SWAP(0/1), NAME, GPU(default 0), BUF(default 8), SHARD(default 50)
set -euo pipefail
export CONDA_ROOT="$HOME/conda"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ROOT/envs/spatialvla_act2answer"

REPO_ROOT="$HOME/bias_benchmark/nazar_folder/Act2Answer"
export PYTHONPATH="$REPO_ROOT/SimplerEnv:$REPO_ROOT/ManiSkill:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false

START_ID=${START_ID:-0}
COUNT=${COUNT:-2}
SWAP=${SWAP:-0}
GPU=${GPU:-0}
BUF=${BUF:-8}
SHARD=${SHARD:-50}
NAME=${NAME:-confirm-spatialvla}

extra=()
[ "$SWAP" = "1" ] && extra=(--do-swap)

cd "$REPO_ROOT/SimplerEnv"
echo "START_SVLA $(date -u) name=$NAME start=$START_ID count=$COUNT swap=$SWAP gpu=$GPU buf=$BUF shard=$SHARD"
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m simpler_env.eval \
  --vla spatialvla --start-id "$START_ID" --count "$COUNT" --shard-size "$SHARD" \
  --assets pairs_choice_vla_confirm --obj-set test --buffer-inferbatch "$BUF" \
  --name "$NAME" "${extra[@]}" < /dev/null
echo "DONE_SVLA $NAME $(date -u)"
