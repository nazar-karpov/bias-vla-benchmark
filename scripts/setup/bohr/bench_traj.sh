#!/usr/bin/env bash
# Накладные расходы логирования траекторий (нужно для непрерывного pull).
set -u
source $HOME/ws/env_bohr.sh
cd $REPO_ROOT/SimplerEnv
for t in 0 1; do
  echo "### A2A_TRAJ_LOG=$t ###"
  env -u A2A_BASE_IMG CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BOARD_XY_SCALE=1.0 \
    A2A_SAVE_VIDEO=0 A2A_TRAJ_LOG=$t timeout 1800 \
    python -u $REPO_ROOT/scripts/bench_vla_speed.py --vla magma \
      --assets pairs_choice_vla_confirm --count 48 --episode-len 30 2>&1 \
    | grep -E "модель |симулятор|прочее|на эпизод|BENCH_RESULT|Traceback|Error"
done
echo TRAJ_BENCH_DONE
