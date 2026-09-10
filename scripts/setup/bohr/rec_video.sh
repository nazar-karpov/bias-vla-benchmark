#!/usr/bin/env bash
# Перезапуск одного шарда с записью видео — чтобы посмотреть глазами, что происходит с кубом.
set -u
ulimit -n 65536
source $HOME/ws/env_bohr.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0} TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BOARD_XY_SCALE=1.2 A2A_TILE_Y=0.14 A2A_TRAJ_LOG=1
export A2A_SAVE_VIDEO=1                     # <- главное отличие
cd $REPO_ROOT/SimplerEnv
python -u -m simpler_env.eval --vla magma --assets ${ASSETS:-pairs_q10} \
  --start-id ${START:-0} --count ${N:-48} --shard-size ${N:-48} --buffer-inferbatch ${N:-48} \
  --name "vid-${ASSETS:-pairs_q10}-s${START:-0}"
echo "VIDEO_DONE rc=$?"
