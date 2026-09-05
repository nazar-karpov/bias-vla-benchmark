#!/usr/bin/env bash
# Третий конфиг «как confirm-кардсет»: масштаб 1.3, слоты ±0.155 — для всех четырёх кардсетов.
set -u
source $HOME/ws/env_bohr.sh
S=$REPO_ROOT/scripts; C=confirm_s1p3_y0p155
render() { # assets outdir gpu
  CUDA_VISIBLE_DEVICES=$3 BOARD_XY_SCALE=1.3 A2A_TILE_Y=0.155 \
    python -u $S/render_focus_frames.py --assets "$1" --config $C --out "$REPO_ROOT/outputs/$2" --chunk 50 \
    > $HOME/ws/logs_${2}_confirm.log 2>&1
  echo "RENDER_DONE $1 rc=$?"
}
( render visbias_frames visbias_frames 1 ) &
( render focus_pairs focus_frames 0; render pairs_frames pairs_frames 0; render veri_frames veri_frames 0 ) &
wait
M=$HOME/ws/datasets/gdrive_meta; CFGS="a2a_default_s1p0_y0p155 andrey_s1p2_y0p14 $C"
python $S/build_focus_frames_manifest.py --frames-root $REPO_ROOT/outputs/focus_frames \
  --vla $M/focus_reflect/deprecated/vla_manifests/focus_two_image_selection.csv \
  --vlm $M/focus_reflect/deprecated/vlm_manifests/focus_vlm_parallel_two_image_selection.csv --configs $CFGS
CS=$REPO_ROOT/ManiSkill/mani_skill/assets/carrot
python $S/build_pair_frames_manifest.py --assets-dir $CS/pairs_frames --frames-root $REPO_ROOT/outputs/pairs_frames --configs $CFGS
python $S/build_pair_frames_manifest.py --assets-dir $CS/veri_frames --frames-root $REPO_ROOT/outputs/veri_frames --configs $CFGS \
  --deprecated-vla $M/veri_emergency/deprecated/vla_manifests/veri_two_image_selection.csv \
  --deprecated-vlm $M/veri_emergency/deprecated/vlm_manifests/veri_vlm_parallel_two_image_selection.csv
python $S/build_pair_frames_manifest.py --assets-dir $CS/visbias_frames --frames-root $REPO_ROOT/outputs/visbias_frames --configs $CFGS \
  --deprecated-vla $M/visbias/deprecated/vla_manifests/visbias_two_image_selection.csv \
  --deprecated-vlm $M/visbias/deprecated/vlm_manifests/visbias_vlm_parallel_two_image_selection.csv
echo CONFIRM_CONFIG_DONE
