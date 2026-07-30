#!/bin/bash
# Ночные тесты усиления bias: A=кроп кадра, B=кропнутые плитки, D=tiles-промпт.
# Каждый: subset 4 вопроса, 3200 логит-запросов, без gen-check.
source ~/bias_benchmark/miniconda3/etc/profile.d/conda.sh
export REPO_ROOT=~/bias_benchmark/nazar_folder/Act2Answer
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=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
O=$REPO_ROOT/outputs
S=$REPO_ROOT/scripts/vlm_sim_choice.py
Q="boss,wealthy,suburbs,skier"

three() {  # $1=model $2=gpu
  CUDA_VISIBLE_DEVICES=$2 python -u $S --model $1 --frames-dir $O/simframes_choice_crop     --only-question "$Q" --out $O/cropframe-subset-$1.json < /dev/null
  CUDA_VISIBLE_DEVICES=$2 python -u $S --model $1 --frames-dir $O/simframes_choice_croptile     --only-question "$Q" --out $O/croptile-subset-$1.json < /dev/null
  CUDA_VISIBLE_DEVICES=$2 python -u $S --model $1 --frames-dir $O/simframes_choice     --only-question "$Q" --prompt-style tiles --out $O/tileprompt-subset-$1.json < /dev/null
}

case "$1" in
  magma)
    conda activate ~/bias_benchmark/miniconda3/envs/magma_act2answer
    three magma 1
    ;;
  paliqwen)
    conda activate ~/bias_benchmark/miniconda3/envs/spatialvla_act2answer
    three paligemma 0
    conda deactivate
    conda activate ~/bias_benchmark/miniconda3/envs/internvla
    three qwenbase 0
    ;;
esac
echo NIGHT_CHAIN_DONE_$1
