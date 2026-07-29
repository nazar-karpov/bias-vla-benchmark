#!/bin/bash
set -uo pipefail
source ~/bias_benchmark/miniconda3/etc/profile.d/conda.sh
conda activate ~/bias_benchmark/miniconda3/envs/spatialvla_act2answer
export REPO_ROOT=~/bias_benchmark/nazar_folder/Act2Answer
export PYTHONPATH=$REPO_ROOT/SimplerEnv:$REPO_ROOT/ManiSkill
export TOKENIZERS_PARALLELISM=false XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=${GPU:-0}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $REPO_ROOT/SimplerEnv
OUT=$REPO_ROOT/outputs/vlm_mcq; LOG=$REPO_ROOT/logs/vlm_mcq
run() { local assets="$1" count="$2"
  echo ">>> paligemma $assets count=$count $(date -u)" | tee -a "$LOG/_progress.log"
  python -u $REPO_ROOT/scripts/paligemma_vlm_qa.py --assets "$assets" --count "$count" --render-chunk 55 \
    --out "$OUT/paligemma_${assets}.json" > "$LOG/paligemma_${assets}.log" 2>&1
  local rc=$?; local p=$(grep -a "parsed=" "$LOG/paligemma_${assets}.log" | tail -1)
  echo "<<< paligemma $assets rc=$rc $p $(date -u)" | tee -a "$LOG/_progress.log"
}
run pairs_bias 520
run safeeditbench 62
echo "PALI_OFFLINE_DONE $(date -u)" | tee -a "$LOG/_progress.log"
