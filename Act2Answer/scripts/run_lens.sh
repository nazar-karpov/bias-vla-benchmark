#!/bin/bash
set -e
source ~/bias_benchmark/miniconda3/etc/profile.d/conda.sh
conda activate ~/bias_benchmark/miniconda3/envs/magma_act2answer
export REPO_ROOT=~/bias_benchmark/nazar_folder/Act2Answer
export PYTHONPATH=$REPO_ROOT/scripts:$REPO_ROOT/SimplerEnv:$REPO_ROOT/ManiSkill
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export HF_TOKEN='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'
cd $REPO_ROOT/SimplerEnv
python -u $REPO_ROOT/scripts/magma_logit_lens.py "$@"
