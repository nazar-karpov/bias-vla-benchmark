#!/bin/bash
# Confirm-прогон magma на V100 GPU1: все NOSWAP (1600 эп.) chunked-режимом.
source ~/bias_benchmark/miniconda3/etc/profile.d/conda.sh
export REPO_ROOT=/home/User17/bias_benchmark/nazar_folder/Act2Answer
export PYTHONPATH=$REPO_ROOT/SimplerEnv:$REPO_ROOT/ManiSkill TOKENIZERS_PARALLELISM=false
conda activate ~/bias_benchmark/miniconda3/envs/magma_act2answer
cd $REPO_ROOT/SimplerEnv
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python3 -u -m simpler_env.eval --vla magma --start-id 0 --count 1600 --shard-size 50 \
    --assets pairs_choice_vla_confirm --obj-set test --buffer-inferbatch 14 \
    --name confirm-magma-noswap < /dev/null
echo CONFIRM_V100_DONE
