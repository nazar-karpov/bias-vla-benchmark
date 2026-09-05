#!/usr/bin/env bash
# Полный конвейер «датасет с Drive → квадраты → кардсет → кадры двух раскладок → манифест».
# Параметры: NAME MODE IMAGES_SRC PAIRS_TSV... (через запятую) [DEPRECATED_VLA] [DEPRECATED_VLM]
#   NAME        имя кардсета/папки кадров (pairs_frames, veri_frames, visbias_frames)
#   MODE        square_images: none|pad|face|center
#   IMAGES_SRC  ~/ws/datasets/<dataset>  (пути таблиц относительно него)
set -u
NAME=$1; MODE=$2; SRC=$3; TSV=$4; DVLA=${5:-}; DVLM=${6:-}
source $HOME/ws/env_bohr.sh
S=$REPO_ROOT/scripts
SQ=${SRC}_square
CS=$REPO_ROOT/ManiSkill/mani_skill/assets/carrot/$NAME
OUT=$REPO_ROOT/outputs/$NAME
python $S/square_images.py --src "$SRC" --out "$SQ" --mode "$MODE" || exit 1
python $S/gen_pairs_cardset.py --name "$NAME" --pairs ${TSV//,/ } ${DVLA:+--deprecated $DVLA} \
  --images-root "$SQ" --out "$CS" || exit 1
# две раскладки параллельно на двух картах
( export CUDA_VISIBLE_DEVICES=0 BOARD_XY_SCALE=1.2 A2A_TILE_Y=0.14
  python -u $S/render_focus_frames.py --assets "$NAME" --config andrey_s1p2_y0p14 --out "$OUT" --chunk 50 ) \
  > $HOME/ws/logs_${NAME}_andrey.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1 BOARD_XY_SCALE=1.0 A2A_TILE_Y=0.155
  python -u $S/render_focus_frames.py --assets "$NAME" --config a2a_default_s1p0_y0p155 --out "$OUT" --chunk 50 ) \
  > $HOME/ws/logs_${NAME}_default.log 2>&1 &
wait
python $S/build_pair_frames_manifest.py --assets-dir "$CS" --frames-root "$OUT" \
  --configs a2a_default_s1p0_y0p155 andrey_s1p2_y0p14 ${DVLA:+--deprecated-vla $DVLA} ${DVLM:+--deprecated-vlm $DVLM}
cp "$SQ/crops.csv" "$OUT/crops.csv" 2>/dev/null
echo "DATASET_FRAMES_DONE $NAME"
