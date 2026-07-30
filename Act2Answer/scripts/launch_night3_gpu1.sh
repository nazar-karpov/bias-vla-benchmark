#!/bin/bash
# Третья очередь GPU1: большие плитки (scale 1.5, кропнутые текстуры) — big и bigcombo.
source ~/bias_benchmark/miniconda3/etc/profile.d/conda.sh
export REPO_ROOT=~/bias_benchmark/nazar_folder/Act2Answer
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=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
O=$REPO_ROOT/outputs
S=$REPO_ROOT/scripts/vlm_sim_choice.py
Q="boss,wealthy,suburbs,skier"

conda activate ~/bias_benchmark/miniconda3/envs/spatialvla_act2answer
CUDA_VISIBLE_DEVICES=1 python -u $S --model paligemma --frames-dir $O/simframes_choice_big   --only-question "$Q" --out $O/big-subset-paligemma.json < /dev/null
CUDA_VISIBLE_DEVICES=1 python -u $S --model paligemma --frames-dir $O/simframes_choice_bigcombo   --only-question "$Q" --out $O/bigcombo-subset-paligemma.json < /dev/null
conda deactivate
conda activate ~/bias_benchmark/miniconda3/envs/magma_act2answer
CUDA_VISIBLE_DEVICES=1 python -u $S --model magma --frames-dir $O/simframes_choice_big   --only-question "$Q" --out $O/big-subset-magma.json < /dev/null
CUDA_VISIBLE_DEVICES=1 python -u $S --model magma --frames-dir $O/simframes_choice_bigcombo   --only-question "$Q" --out $O/bigcombo-subset-magma.json < /dev/null
echo NIGHT3_GPU1_DONE
