#!/usr/bin/env bash
# Свип по числу параллельных сред: где насыщается пропускная способность.
set -u
source $HOME/ws/env_bohr.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0} TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BOARD_XY_SCALE=1.0 A2A_SAVE_VIDEO=0
cd $REPO_ROOT/SimplerEnv
for n in ${COUNTS:-12 24 48}; do
  echo "########## COUNT=$n ##########"
  timeout 2400 python -u $REPO_ROOT/scripts/bench_vla_speed.py --vla magma \
    --assets pairs_choice_vla_confirm --count $n --episode-len ${L:-30} 2>&1 \
    | grep -E "BENCH |загрузка|прогон|модель|симулятор|прочее|пик VRAM|на эпизод|BENCH_RESULT|Error|Traceback|out of memory"
  echo
done
echo SWEEP_DONE
