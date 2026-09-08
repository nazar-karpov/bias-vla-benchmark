#!/usr/bin/env bash
# А/Б: одни и те же эпизоды при base_img_size 512 и 256, жадный режим (без сэмплирования).
# Полный эпизод 80 шагов, чтобы метрики были настоящие.
set -u
source $HOME/ws/env_bohr.sh
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BOARD_XY_SCALE=1.0 A2A_SAVE_VIDEO=0 A2A_GREEDY=1
cd $REPO_ROOT/SimplerEnv
for px in 512 256; do
  echo "########## base_img_size=$px ##########"
  CUDA_VISIBLE_DEVICES=${GPU:-0} A2A_BASE_IMG=$px timeout 3600 \
    python -u $REPO_ROOT/scripts/bench_vla_speed.py --vla magma \
      --assets pairs_choice_vla_confirm --count ${N:-48} --episode-len 80 2>&1 \
    | grep -E "BASE_IMG|GREEDY|STATS|на эпизод|BENCH_RESULT|Error|Traceback|out of memory"
  echo
done
echo AB_DONE
