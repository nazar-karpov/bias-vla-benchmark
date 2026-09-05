#!/usr/bin/env bash
# Дособрать VisBias: +65 пар из deprecated-манифеста (в tsv их нет), докат кадров, манифест.
while ! grep -q QUEUE_REST_DONE $HOME/ws/logs_queue_rest.log; do sleep 30; done
source $HOME/ws/env_bohr.sh
S=$REPO_ROOT/scripts; M=$HOME/ws/datasets/gdrive_meta/visbias
CS=$REPO_ROOT/ManiSkill/mani_skill/assets/carrot/visbias_frames; OUT=$REPO_ROOT/outputs/visbias_frames
python $S/gen_pairs_cardset.py --name visbias_frames --pairs $M/pairs/gender.tsv $M/pairs/ethnicity.tsv $M/pairs/profession.tsv \
  --deprecated $M/deprecated/vla_manifests/visbias_two_image_selection.csv --images-root $HOME/ws/datasets/visbias_square --out $CS
( export CUDA_VISIBLE_DEVICES=0 BOARD_XY_SCALE=1.2 A2A_TILE_Y=0.14
  python -u $S/render_focus_frames.py --assets visbias_frames --config andrey_s1p2_y0p14 --out $OUT --chunk 50 ) > $HOME/ws/logs_visbias_frames_andrey2.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1 BOARD_XY_SCALE=1.0 A2A_TILE_Y=0.155
  python -u $S/render_focus_frames.py --assets visbias_frames --config a2a_default_s1p0_y0p155 --out $OUT --chunk 50 ) > $HOME/ws/logs_visbias_frames_default2.log 2>&1 &
wait
python $S/build_pair_frames_manifest.py --assets-dir $CS --frames-root $OUT --configs a2a_default_s1p0_y0p155 andrey_s1p2_y0p14 \
  --deprecated-vla $M/deprecated/vla_manifests/visbias_two_image_selection.csv --deprecated-vlm $M/deprecated/vlm_manifests/visbias_vlm_parallel_two_image_selection.csv
echo AFTER_VISBIAS_DONE
