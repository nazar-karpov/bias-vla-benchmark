#!/usr/bin/env bash
# Блочный раннер confirm_smart: 100-эп блоки × noswap/swap, скип по stats.
# usage: smart_blocks.sh <vla> [доп. CLI-аргументы eval]
set -uo pipefail
VLA="$1"; shift
R="/workspace/moskalenko/bias-vla-benchmark-main"
A="$R/Act2Answer"
LOG=~/logs/smart; mkdir -p $LOG
source "$HOME/conda/etc/profile.d/conda.sh"
conda activate "$HOME/conda/envs/magma_act2answer"
export REPO_ROOT="$A" PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$A/SimplerEnv"
for sw in "" "--do-swap"; do
  tag=noswap; [ -n "$sw" ] && tag=swap
  for s in 0 100 200 300; do
    name="smart-${VLA}-${tag}-s${s}"
    [ -f "$A/outputs/$name/glob/vis_0_test/stats.yaml" ] && { echo "SKIP $name"; continue; }
    echo "[$(date -u +%H:%M:%S)] EVAL $name"
    python3 -u -m simpler_env.eval --vla "$VLA" \
      --assets confirm_smart --start-id $s --count 100 --episode-len 80 \
      --buffer-inferbatch 10 --buffer-minibatch -1 --shard-size 100 \
      --name "$name" $sw "$@" < /dev/null >> "$LOG/${VLA}.log" 2>&1
    echo "[$(date -u +%H:%M:%S)] EVAL DONE $name rc=$?"
  done
done
echo "=== SMART $VLA DONE ==="
