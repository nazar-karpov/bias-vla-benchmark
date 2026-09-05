#!/usr/bin/env bash
# Первые кадры FOCUS для одной раскладки. Параметры: CONFIG SCALE TILE_Y [GPU]
set -u
CONFIG=$1; SCALE=$2; TY=$3; GPU=${4:-0}
source $HOME/ws/env_bohr.sh
export CUDA_VISIBLE_DEVICES=$GPU BOARD_XY_SCALE=$SCALE A2A_TILE_Y=$TY
python -u $REPO_ROOT/scripts/render_focus_frames.py --config "$CONFIG" \
  --out $REPO_ROOT/outputs/focus_frames --chunk 50
echo "FOCUS_FRAMES_DONE $CONFIG rc=$?"
