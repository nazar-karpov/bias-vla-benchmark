#!/usr/bin/env bash
# Проверка типа тела плиток: те же эпизоды и seed, что у видеозаписи vid-pairs_q10-s0 (динамические).
set -u
ulimit -n 65536
source $HOME/ws/env_bohr.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0} TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BOARD_XY_SCALE=1.2 A2A_TILE_Y=0.14 A2A_TRAJ_LOG=1 A2A_SAVE_VIDEO=1
export A2A_TILE_BODY=${TILE_BODY:-kinematic}
unset A2A_BASE_IMG
cd $REPO_ROOT/SimplerEnv
python -u -m simpler_env.eval --vla magma --assets ${ASSETS:-pairs_q10} \
  --start-id ${START:-0} --count ${N:-48} --shard-size ${N:-48} --buffer-inferbatch ${N:-48} \
  --name "vid-${ASSETS:-pairs_q10}-s${START:-0}-${A2A_TILE_BODY}"
echo "VIDEO_DONE rc=$?"
