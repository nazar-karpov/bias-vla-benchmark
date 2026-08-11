#!/usr/bin/env bash
# Первые кадры симуляции кардсета pairs_choice_vla_confirm — под линейный пробинг.
# Использует готовый render_sim_choice_frames.py (сим без модели, chunked по 50).
#   COUNT=6  bash scripts/render_linprobe_frames.sh   # проба
#   COUNT=0  bash scripts/render_linprobe_frames.sh   # все 1600
set -uo pipefail

COUNT="${COUNT:-6}"
ASSETS="${ASSETS:-pairs_choice_vla_confirm}"
OUT="${OUT:-}"

A="/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer"
[ -n "$OUT" ] || OUT="$A/outputs/linprobe_frames_${ASSETS}$( [ "$COUNT" != "0" ] && echo "_sample$COUNT" )"

source ~/conda/etc/profile.d/conda.sh
conda activate magma_act2answer
export REPO_ROOT="$A"
export PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
# model_db confirm-кардсета УЖЕ содержит mid-масштаб 1.3; дефолт BOARD_XY_SCALE=1.3
# (коммит 86596ca, для sohas) умножается ПОВЕРХ и даёт 1.69 — не как в confirm-прогонах.
export BOARD_XY_SCALE="${BOARD_XY_SCALE:-1.0}"
cd "$A/SimplerEnv" || exit 1

python -u "$A/scripts/render_sim_choice_frames.py" \
  --assets "$ASSETS" --count "$COUNT" --out-dir "$OUT" < /dev/null
rc=$?
# полный pairs.json рядом с кадрами — манифест скрипта теряет qkey/polarity
cp "$A/ManiSkill/mani_skill/assets/carrot/$ASSETS/pairs.json" "$OUT/pairs.json"
echo "DONE rc=$rc out=$OUT"
