#!/usr/bin/env bash
# Пересборка кадров FOCUS с АКТУАЛЬНЫХ pairs/*.tsv (6000 пар) — три раскладки.
set -u
source $HOME/ws/env_bohr.sh
S=$REPO_ROOT/scripts; OUT=$REPO_ROOT/outputs/focus_frames
render() { # config scale tile_y gpu
  CUDA_VISIBLE_DEVICES=$4 BOARD_XY_SCALE=$2 A2A_TILE_Y=$3 \
    python -u $S/render_focus_frames.py --assets focus_pairs --config "$1" --out "$OUT" --chunk 50 \
    > $HOME/ws/logs_focus_tsv_$1.log 2>&1
  echo "RENDER_DONE $1 rc=$?"
}
( render a2a_default_s1p0_y0p155 1.0 0.155 0; render confirm_s1p3_y0p155 1.3 0.155 0 ) &
( render andrey_s1p2_y0p14 1.2 0.14 1 ) &
wait
echo FOCUS_TSV_RENDER_ALL_DONE
