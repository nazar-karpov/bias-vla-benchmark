#!/bin/bash
# Magma VLA-eval (motor rollout) on pairs_choice_vla_confirm (mid, 1.3x tiles, crop).
# Runs BOTH layouts (noswap+swap). Chunked by --shard-size 50 so each 50-ep chunk
# writes its own -s<cs> dir (do NOT call per-chunk, give a RANGE — see confirm memory).
# Usage: run_magma_vla_mid.sh <start_id> <count> <name_prefix> [gpu] [inferbatch]
set -u
R=/workspace/moskalenko/bias-vla-benchmark-main
A=$R/Act2Answer
source /home/user/conda/etc/profile.d/conda.sh
conda activate /home/user/conda/envs/magma_act2answer
export REPO_ROOT=$A
export PYTHONPATH=$A/SimplerEnv:$A/ManiSkill
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
cd $A/SimplerEnv

START="${1:-0}"
COUNT="${2:-1600}"
NAME="${3:-confirm-mid-magma}"
GPU="${4:-0}"
IB="${5:-8}"

for swap_arg in noswap swap; do
  extra=""
  [ "$swap_arg" = swap ] && extra="--do-swap"
  echo "RUN_MAGMA_VLA ${swap_arg} start=$START count=$COUNT $(date -u)"
  CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python3 -u -m simpler_env.eval \
      --vla magma --start-id "$START" --count "$COUNT" --shard-size 50 \
      --assets pairs_choice_vla_confirm --obj-set test --buffer-inferbatch "$IB" \
      --name "${NAME}-${swap_arg}" $extra < /dev/null
done
echo "DONE_MAGMA_VLA start=$START count=$COUNT"
