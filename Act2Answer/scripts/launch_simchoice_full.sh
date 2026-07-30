#!/bin/bash
# Полные прогоны vlm_sim_choice (26400 запросов/модель) — генерится локально, токен не в git
source ~/bias_benchmark/miniconda3/etc/profile.d/conda.sh
export REPO_ROOT=~/bias_benchmark/nazar_folder/Act2Answer
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=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
O=$REPO_ROOT/outputs
S=$REPO_ROOT/scripts/vlm_sim_choice.py
F=$O/simframes_choice

run_magma() {
  conda activate ~/bias_benchmark/miniconda3/envs/magma_act2answer
  CUDA_VISIBLE_DEVICES=1 python -u $S --model magma --frames-dir $F     --out $O/simchoice-all-magma.json < /dev/null
  conda deactivate
}
run_pali_qwen() {
  conda activate ~/bias_benchmark/miniconda3/envs/spatialvla_act2answer
  CUDA_VISIBLE_DEVICES=0 python -u $S --model paligemma --frames-dir $F     --out $O/simchoice-all-paligemma.json < /dev/null
  conda deactivate
  conda activate ~/bias_benchmark/miniconda3/envs/internvla
  CUDA_VISIBLE_DEVICES=0 python -u $S --model qwenbase --frames-dir $F     --out $O/simchoice-all-qwenbase.json < /dev/null
}
case "$1" in
  magma) run_magma ;;
  paliqwen) run_pali_qwen ;;
esac
