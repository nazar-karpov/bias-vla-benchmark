#!/usr/bin/env bash
set -uo pipefail
MODEL=$1; VLA=$2; GPU=$3
B=$HOME/bias_benchmark; A=$B/nazar_folder/Act2Answer; L=$B/nazar_folder/cropped_run/logs
source $B/miniconda3/etc/profile.d/conda.sh
conda activate $B/miniconda3/envs/$2_act2answer 2>/dev/null || conda activate $B/miniconda3/envs/spatialvla_act2answer
# magma env name is magma_act2answer, spatialvla -> spatialvla_act2answer
case $MODEL in magma) conda activate $B/miniconda3/envs/magma_act2answer;; spatialvla) conda activate $B/miniconda3/envs/spatialvla_act2answer;; esac
export REPO_ROOT=$A PYTHONPATH=$A/SimplerEnv:$A/ManiSkill TOKENIZERS_PARALLELISM=false XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU
cd $A/SimplerEnv
for swap in noswap swap; do
  extra=""; [ $swap = swap ] && extra="--do-swap"
  setsid python3 -u -m simpler_env.eval --vla $MODEL --start-id 0 --count 55 --assets pairs_bias_crop --obj-set test --episode-len 80 --buffer-inferbatch 5 --buffer-minibatch -1 --name softcrop-$MODEL-pairs_bias_crop-$swap $extra < /dev/null > $L/softcrop-$MODEL-$swap.log 2>&1
  echo "DONE $MODEL $swap rc=$? $(grep -a FINAL_STATS $L/softcrop-$MODEL-$swap.log|tail -1)"
done
