#!/bin/bash
# Точечный прогон 6 ключевых вопросов: boss(0),wealthy(40),suburbs(80),skier(120),
# peaceful(280),pilot(840) — по 40 эп. (pos+neg), оба swap. Готовые шарды скипаются
# по наличию stats.yaml (для V100 noswap 0-159/280-319 уже покрыт 50-шардами).
source ~/bias_benchmark/miniconda3/etc/profile.d/conda.sh
export REPO_ROOT=/home/User17/bias_benchmark/nazar_folder/Act2Answer
export PYTHONPATH=$REPO_ROOT/SimplerEnv:$REPO_ROOT/ManiSkill
export TOKENIZERS_PARALLELISM=false
conda activate ~/bias_benchmark/miniconda3/envs/magma_act2answer
cd $REPO_ROOT/SimplerEnv
covered() {  # $1=swap $2=start: покрыт ли диапазон старыми 50-шардами
  local sw=$1 st=$2
  for base in 0 40 80 120 280 840; do :; done
  # старые шарды кратны 50: диапазон [st, st+39] покрыт если шард floor(st/50)*50 и
  # floor((st+39)/50)*50 оба готовы
  local a=$(( st/50*50 )) b=$(( (st+39)/50*50 ))
  [ -f $REPO_ROOT/outputs/fastvla-magma-$sw-s$a/glob/vis_0_test/stats.yaml ] || return 1
  [ -f $REPO_ROOT/outputs/fastvla-magma-$sw-s$b/glob/vis_0_test/stats.yaml ] || return 1
  return 0
}
for swap in noswap swap; do
  extra=""
  [ "$swap" = swap ] && extra="--do-swap"
  for st in 0 40 80 120 280 840; do
    name=fastvla-magma-$swap-q$st
    if [ -f $REPO_ROOT/outputs/$name/glob/vis_0_test/stats.yaml ]; then echo "SKIP $name"; continue; fi
    if covered $swap $st; then echo "COVERED $swap $st (старые шарды)"; continue; fi
    echo "SHARD $name $(date -u)"
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      python3 -u -m simpler_env.eval --vla magma --start-id $st --count 40 \
        --assets pairs_choice_vla_fast --obj-set test --buffer-inferbatch 14 \
        $extra --name $name < /dev/null
  done
done
echo CORE6_MAGMA_DONE
