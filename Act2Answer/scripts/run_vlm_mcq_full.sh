#!/bin/bash
# Полный VLM MCQ-прогон: Magma + PaliGemma x {pairs_bias(520), safeeditbench(62)} x {noswap,swap}.
# Новый промпт: "Put cube on <obj>\nA left\nB right\nAnswer with the option's letter...".
set -uo pipefail
source ~/bias_benchmark/miniconda3/etc/profile.d/conda.sh
export REPO_ROOT=~/bias_benchmark/nazar_folder/Act2Answer
export PYTHONPATH=$REPO_ROOT/SimplerEnv:$REPO_ROOT/ManiSkill
export TOKENIZERS_PARALLELISM=false XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=${GPU:-0}
cd $REPO_ROOT/SimplerEnv
OUT=$REPO_ROOT/outputs/vlm_mcq
LOG=$REPO_ROOT/logs/vlm_mcq
mkdir -p "$OUT" "$LOG"
STAMP=$(date -u +%Y%m%d_%H%M)
echo "VLM_MCQ_START $(date -u)" | tee -a "$LOG/_progress.log"

run() { # <env> <script> <assets> <count> <tag>
  local env="$1" script="$2" assets="$3" count="$4" tag="$5"
  conda activate ~/bias_benchmark/miniconda3/envs/$env
  echo ">>> $tag $assets count=$count $(date -u)" | tee -a "$LOG/_progress.log"
  python -u $REPO_ROOT/scripts/$script \
    --assets "$assets" --count "$count" --render-chunk 55 \
    --out "$OUT/${tag}_${assets}.json" > "$LOG/${tag}_${assets}.log" 2>&1
  local rc=$?
  local p=$(grep -a "parsed=" "$LOG/${tag}_${assets}.log" | tail -1)
  echo "<<< $tag $assets rc=$rc $p $(date -u)" | tee -a "$LOG/_progress.log"
  conda deactivate
}

run magma_act2answer   magma_vlm_qa.py     pairs_bias    520 magma
run magma_act2answer   magma_vlm_qa.py     safeeditbench 62  magma
run spatialvla_act2answer paligemma_vlm_qa.py pairs_bias    520 paligemma
run spatialvla_act2answer paligemma_vlm_qa.py safeeditbench 62  paligemma

echo "VLM_MCQ_ALL_DONE $(date -u)" | tee -a "$LOG/_progress.log"
