#!/bin/bash
# spatialvla на половинном кардсете fast5 (660 эп.), шарды по 50, оба swap, bf16.
source ~/bias_benchmark/miniconda3/etc/profile.d/conda.sh
export REPO_ROOT=/home/User17/bias_benchmark/nazar_folder/Act2Answer
export PYTHONPATH=$REPO_ROOT/SimplerEnv:$REPO_ROOT/ManiSkill
export TOKENIZERS_PARALLELISM=false
conda activate ~/bias_benchmark/miniconda3/envs/spatialvla_act2answer
cd $REPO_ROOT/SimplerEnv
for swap in noswap swap; do
  extra=""
  [ "$swap" = swap ] && extra="--do-swap"
  for s in $(seq 0 50 650); do
    c=50
    [ $s -eq 650 ] && c=10
    name=fastvla-spatialvla-$swap-s$s
    if [ -f $REPO_ROOT/outputs/$name/glob/vis_0_test/stats.yaml ]; then
      echo "SKIP $name"; continue
    fi
    echo "SHARD $name $(date -u)"
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false       python3 -u -m simpler_env.eval --vla spatialvla --start-id $s --count $c         --assets pairs_choice_vla_fast5 --obj-set test --buffer-inferbatch 20         $extra --name $name < /dev/null
  done
done
echo FASTVLA_DONE_spatialvla
