#!/bin/bash
# Launcher: PaliGemma2 VLM-QA on pairs_bias. Args: <count> <out_suffix> [extra...]
set -e
source ~/bias_benchmark/miniconda3/etc/profile.d/conda.sh
conda activate ~/bias_benchmark/miniconda3/envs/spatialvla_act2answer
export REPO_ROOT=~/bias_benchmark/nazar_folder/Act2Answer
export PYTHONPATH=$REPO_ROOT/SimplerEnv:$REPO_ROOT/ManiSkill
export TOKENIZERS_PARALLELISM=false XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export HF_TOKEN='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'
cd $REPO_ROOT/SimplerEnv
COUNT=$1; SUF=$2; shift 2
python -u $REPO_ROOT/scripts/paligemma_vlm_qa.py   --assets pairs_bias --count "$COUNT"   --out $REPO_ROOT/outputs/paligemma_vlm_qa_pairs_bias_${SUF}.json "$@"
