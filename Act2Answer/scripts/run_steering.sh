#!/bin/bash
# Launcher: Magma activation-steering (causal race->judgement test).
# Args: <layer> <out_suffix> [extra python args...]
set -e
source ~/bias_benchmark/miniconda3/etc/profile.d/conda.sh
conda activate ~/bias_benchmark/miniconda3/envs/magma_act2answer
export REPO_ROOT=~/bias_benchmark/nazar_folder/Act2Answer
export PYTHONPATH=$REPO_ROOT/SimplerEnv:$REPO_ROOT/ManiSkill
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export HF_TOKEN='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'
cd $REPO_ROOT/SimplerEnv
LAYER=$1; SUF=$2; shift 2
python -u $REPO_ROOT/scripts/magma_steering.py \
  --imgs ~/bias_benchmark/nazar_folder/pairs_bias/imgs \
  --layer "$LAYER" \
  --out $REPO_ROOT/outputs/magma_steering_${SUF}.json "$@"
