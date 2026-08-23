#!/usr/bin/env bash
# SINGLE-CARD, ЦЕНТРАЛЬНЫЙ СЛОТ: одна карточка в фиксированной точке (y=0).
# Отличие от run_single_card_pilot.sh: карточка не переезжает лево-право, поэтому
#   * нет позиционного крена (в слотах он давал R−L ≈ −111 мм, на порядок больше
#     любого демографического эффекта);
#   * контрбаланс слотов не нужен -> ВДВОЕ меньше эпизодов на тот же вопрос.
# Плитка в центре видна целиком и на ~33% крупнее, но рука со схваченным кубом
# заслоняет её дальнюю четверть — куда именно попадает лицо, задаёт YAW
# (см. center_card_probe.py / center_card_readability.py).
# Имя шарда обязано содержать "-center-": по нему single_card_assent.py понимает,
# что слот один, и берёт координату карточки из --card-x/--card-y.
set -uo pipefail
R=/workspace/moskalenko/bias-vla-benchmark-main
A="$R/Act2Answer"
CONDA=/workspace/moskalenko/conda
export REPO_ROOT="$A" PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
export MS_ASSET_DIR=/workspace/moskalenko/maniskill_assets
export HF_HOME=/workspace/moskalenko/hf_cache
export TOKENIZERS_PARALLELISM=false
export A2A_SINGLE_TILE=1     # одиночный режим (см. put_on_in_scene_multi_v4)
export A2A_TRAJ_LOG=1        # траектории под непрерывную метрику
export A2A_SAVE_VIDEO=0
export BOARD_XY_SCALE=1.0    # масштаб 1.3 уже в model_db кардсета

export A2A_SINGLE_TILE_X="${TILE_X:--0.25}"
export A2A_SINGLE_TILE_Y="${TILE_Y:-0.0}"
export A2A_SINGLE_TILE_YAW="${TILE_YAW:-90}"

ASSET="${ASSET:-pairs_single_pilot}"
NAME="${NAME:-center-pilot}"
TOTAL="${TOTAL:-400}"
SHARD="${SHARD:-100}"
PAR="${PAR:-4}"              # magma: ~35 ГБ VRAM/процесс, 2×H100 -> 4 клиента
LOG=/workspace/moskalenko/logs_single_card
mkdir -p "$LOG"
source "$CONDA/etc/profile.d/conda.sh"
conda activate "$CONDA/envs/magma_act2answer"
cd "$A/SimplerEnv" || exit 1

echo "CENTER RUN: asset=$ASSET tile=($A2A_SINGLE_TILE_X,$A2A_SINGLE_TILE_Y) yaw=$A2A_SINGLE_TILE_YAW total=$TOTAL"
i=0
for ((s=0; s<TOTAL; s+=SHARD)); do
  name="${NAME}-magma-center-sh${s}"
  [ -f "$A/outputs/$name/glob/vis_0_test/stats.yaml" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 20; done
  gpu=$(( i % 2 ))
  echo "[$(date -u +%H:%M:%S)] START $name gpu=$gpu"
  CUDA_VISIBLE_DEVICES=$gpu \
    python -u -m simpler_env.eval --vla magma \
      --assets "$ASSET" --obj-set test --start-id "$s" --count "$SHARD" \
      --episode-len 80 --buffer-inferbatch 10 --buffer-minibatch -1 \
      --name "$name" < /dev/null > "$LOG/$name.log" 2>&1 &
  i=$(( i + 1 ))
  sleep 20
done
wait
echo "SINGLE_CARD_CENTER_DONE $(date -u)"
