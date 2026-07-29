#!/usr/bin/env bash
set -uo pipefail
GPU=$1; IB=${2:-1}
B=$HOME/bias_benchmark; A=$B/nazar_folder/Act2Answer; L=$B/nazar_folder/cropped_run/logs
source $B/miniconda3/etc/profile.d/conda.sh; conda activate $B/miniconda3/envs/magma_act2answer
export REPO_ROOT=$A PYTHONPATH=$A/SimplerEnv:$A/ManiSkill TOKENIZERS_PARALLELISM=false XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU
cd $A/SimplerEnv
for swap in noswap swap; do
  extra=""; [ $swap = swap ] && extra="--do-swap"
  setsid python3 -u -m simpler_env.eval --vla magma --start-id 0 --count 55 --assets pairs_bias_crop --obj-set test --episode-len 80 --buffer-inferbatch $IB --buffer-minibatch -1 --name softcrop-magma-pairs_bias_crop-$swap $extra < /dev/null > $L/softcrop-magma-$swap.log 2>&1
  echo "DONE magma $swap rc=$? $(grep -a FINAL_STATS $L/softcrop-magma-$swap.log|tail -1)"
done
