#!/bin/bash
# Быстрый VLA-замер: pairs_choice_vla_fast (1320 эп.), шарды по 50, оба swap.
# $1 = magma|spatialvla, $2 = GPU, $3 = inferbatch, $4 = extra env (напр. SPATIALVLA_FP16=1)
source ~/bias_benchmark/miniconda3/etc/profile.d/conda.sh
export REPO_ROOT=/home/User17/bias_benchmark/nazar_folder/Act2Answer
export PYTHONPATH=$REPO_ROOT/SimplerEnv:$REPO_ROOT/ManiSkill
export TOKENIZERS_PARALLELISM=false
VLA=$1; GPU=$2; IB=$3
[ -n "$4" ] && export $4
case $VLA in
  magma) conda activate ~/bias_benchmark/miniconda3/envs/magma_act2answer ;;
  spatialvla) conda activate ~/bias_benchmark/miniconda3/envs/spatialvla_act2answer ;;
esac
cd $REPO_ROOT/SimplerEnv
for swap in noswap swap; do
  extra=""
  [ "$swap" = swap ] && extra="--do-swap"
  for s in $(seq 0 50 1300); do
    c=50
    [ $s -eq 1300 ] && c=20
    name=fastvla-$VLA-$swap-s$s
    if [ -f $REPO_ROOT/outputs/$name/glob/vis_0_test/stats.yaml ]; then
      echo "SKIP $name (готов)"; continue
    fi
    echo "SHARD $name $(date -u)"
    CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false       python3 -u -m simpler_env.eval --vla $VLA --start-id $s --count $c         --assets pairs_choice_vla_fast --obj-set test --buffer-inferbatch $IB         $extra --name $name < /dev/null
  done
done
echo FASTVLA_DONE_$VLA
