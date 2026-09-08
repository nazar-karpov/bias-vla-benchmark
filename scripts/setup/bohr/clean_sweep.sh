#!/usr/bin/env bash
# Чистый свип: строго по одному процессу за раз, ничего параллельно.
# Первый свип был испорчен фоновой нагрузкой (rsync на vlm8 + замеры на второй карте).
set -u
source $HOME/ws/env_bohr.sh
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BOARD_XY_SCALE=1.0 A2A_SAVE_VIDEO=0 CUDA_VISIBLE_DEVICES=0
cd $REPO_ROOT/SimplerEnv
for px in 512 256; do
  for n in 6 12 24 48; do
    echo "### base_img=$px count=$n ###"
    A2A_BASE_IMG=$px timeout 2400 python -u $REPO_ROOT/scripts/bench_vla_speed.py \
      --vla magma --assets pairs_choice_vla_confirm --count $n --episode-len 30 2>&1 \
      | grep -E "BENCH_RESULT|out of memory|Traceback"
  done
done
echo CLEAN_SWEEP_DONE
