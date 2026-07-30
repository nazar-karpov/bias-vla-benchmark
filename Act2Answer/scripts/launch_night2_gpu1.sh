#!/bin/bash
# Вторая ночная очередь на GPU1: combo (кроп кадра + кропнутые плитки), затем
# конкат на кропнутых фото (тест C). Baseline C = choice-all-*.json.
source ~/bias_benchmark/miniconda3/etc/profile.d/conda.sh
export REPO_ROOT=~/bias_benchmark/nazar_folder/Act2Answer
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=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
O=$REPO_ROOT/outputs
Q="boss,wealthy,suburbs,skier"
CROPIMGS=~/bias_benchmark/nazar_folder/pairs_bias_crop/imgs

conda activate ~/bias_benchmark/miniconda3/envs/magma_act2answer
CUDA_VISIBLE_DEVICES=1 python -u $REPO_ROOT/scripts/vlm_sim_choice.py --model magma   --frames-dir $O/simframes_choice_combo --only-question "$Q"   --out $O/combo-subset-magma.json < /dev/null
conda deactivate

conda activate ~/bias_benchmark/miniconda3/envs/spatialvla_act2answer
CUDA_VISIBLE_DEVICES=1 python -u $REPO_ROOT/scripts/vlm_sim_choice.py --model paligemma   --frames-dir $O/simframes_choice_combo --only-question "$Q"   --out $O/combo-subset-paligemma.json < /dev/null
CUDA_VISIBLE_DEVICES=1 python -u $REPO_ROOT/scripts/vlm_concat_choice_flat.py --model paligemma   --imgs $CROPIMGS --only-question "$Q"   --out $O/concatcrop-subset-paligemma.json < /dev/null
conda deactivate

conda activate ~/bias_benchmark/miniconda3/envs/magma_act2answer
CUDA_VISIBLE_DEVICES=1 python -u $REPO_ROOT/scripts/vlm_concat_choice_flat.py --model magma   --imgs $CROPIMGS --only-question "$Q"   --out $O/concatcrop-subset-magma.json < /dev/null
echo NIGHT2_GPU1_DONE
