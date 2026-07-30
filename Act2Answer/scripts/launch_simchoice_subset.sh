#!/bin/bash
# Подмножество вопросов: boss/employee + расовые (wealthy/poor, suburbs/inner city,
# skier/mugger). 3200 логит-запросов + столько же ген-проверок на модель.
source ~/bias_benchmark/miniconda3/etc/profile.d/conda.sh
export REPO_ROOT=~/bias_benchmark/nazar_folder/Act2Answer
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=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
O=$REPO_ROOT/outputs
S=$REPO_ROOT/scripts/vlm_sim_choice.py
F=$O/simframes_choice
Q="boss,wealthy,suburbs,skier"

case "$1" in
  magma)
    conda activate ~/bias_benchmark/miniconda3/envs/magma_act2answer
    CUDA_VISIBLE_DEVICES=1 python -u $S --model magma --frames-dir $F       --only-question "$Q" --gen-check       --out $O/simchoice-subset-magma.json < /dev/null
    ;;
  paliqwen)
    conda activate ~/bias_benchmark/miniconda3/envs/spatialvla_act2answer
    CUDA_VISIBLE_DEVICES=0 python -u $S --model paligemma --frames-dir $F       --only-question "$Q" --gen-check       --out $O/simchoice-subset-paligemma.json < /dev/null
    conda deactivate
    conda activate ~/bias_benchmark/miniconda3/envs/internvla
    CUDA_VISIBLE_DEVICES=0 python -u $S --model qwenbase --frames-dir $F       --only-question "$Q" --gen-check       --out $O/simchoice-subset-qwenbase.json < /dev/null
    ;;
esac
