#!/usr/bin/env bash
# SINGLE-CARD, ЦЕНТРАЛЬНЫЙ СЛОТ: одна карточка в фиксированной точке.
# Отличие от run_single_card_pilot.sh: карточка не переезжает лево-право, поэтому
#   * нет позиционного крена (в слотах он давал R−L ≈ −111 мм, на порядок больше
#     любого демографического эффекта, и глушил сам полярный гейт);
#   * контрбаланс слотов не нужен -> ВДВОЕ меньше эпизодов на тот же вопрос.
# Рабочая точка (эксп. 40): TILE_X=-0.25 TILE_Y=0.05 TILE_YAW=90 — лицо целиком
# видно, перекрытие рукой 5%, читаемость выше боковых слотов (center_card_probe.py,
# center_card_readability.py). TILE_Y=0.0 — контрольная «лицо под кубом».
# Имя шарда обязано содержать "-center-": по нему single_card_assent.py понимает,
# что слот один, и берёт координату карточки из --card-x/--card-y.
#
#   VLA=magma      — модель в клиенте, ~39 ГБ VRAM/процесс -> PAR=2 на 2×H100;
#   VLA=internvla  — политика живёт в отдельном zmq-сервере (поднимается тут же
#                    или переиспользуется уже запущенный), клиент = симулятор.
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
export A2A_SINGLE_TILE_Y="${TILE_Y:-0.05}"
export A2A_SINGLE_TILE_YAW="${TILE_YAW:-90}"

VLA="${VLA:-magma}"
ASSET="${ASSET:-pairs_single_pilot}"
NAME="${NAME:-center-pilot}"
TOTAL="${TOTAL:-400}"
SHARD="${SHARD:-100}"
SEED="${SEED:-0}"            # confirm-прогоны ОБЯЗАНЫ менять сид: torch.manual_seed
                             # зашит в run.py, тот же сид = те же сэмплы действий
BLOCKS="${BLOCKS:-}"         # выборочные старты шардов ("200 300 1200"); пусто = 0..TOTAL
PAR="${PAR:-2}"
PORT="${INTERNVLA_PORT:-10093}"
LOG=/workspace/moskalenko/logs_single_card
mkdir -p "$LOG"
source "$CONDA/etc/profile.d/conda.sh"

port_alive() {  # port
  python -c "import socket;s=socket.socket();s.settimeout(2);exit(0 if s.connect_ex(('127.0.0.1',$1))==0 else 1)" 2>/dev/null
}

start_server() {  # port gpu tag  (как в run_top8_cross.sh; уже поднятый переиспользуется)
  local p="$1" g="$2" tag="$3"
  port_alive "$p" && { echo "сервер $tag уже слушает :$p — переиспользую"; return 0; }
  echo "[$(date -u +%H:%M:%S)] поднимаю сервер $tag :$p (GPU $g)"
  ( conda activate "$CONDA/envs/internvla"
    export PYTHONPATH="$R/InternVLA-M1:$PYTHONPATH" CUDA_VISIBLE_DEVICES="$g"
    cd "$R/InternVLA-M1"
    exec python deployment/model_server/server_policy_M1.py \
      --ckpt_path "$(cat "$R/internvla_ckpt/ckpt_path.txt")" \
      --port "$p" --use_bf16 ) > "$LOG/_server_$tag.log" 2>&1 &
  for i in $(seq 1 120); do
    port_alive "$p" && { echo "  поднялся за ~$((i*10))с"; sleep 15; return 0; }
    sleep 10
  done
  echo "  СЕРВЕР $tag НЕ ПОДНЯЛСЯ"; return 1
}

VLA_ARG=""
if [ "$VLA" = internvla ]; then
  start_server "$PORT" 1 A || exit 1
  start_server "$(( PORT + 1 ))" 0 B || true
  # критично: без --vla-path eval.py берёт чужой дефолтный чекпоинт и падает
  VLA_ARG="--vla-path $(cat "$R/internvla_ckpt/ckpt_path.txt")"
fi

conda activate "$CONDA/envs/magma_act2answer"   # клиент-симулятор для обеих моделей
cd "$A/SimplerEnv" || exit 1

if [ -z "$BLOCKS" ]; then
  BLOCKS=$(seq 0 "$SHARD" $(( TOTAL - 1 )) | tr '\n' ' ')
fi
echo "CENTER RUN: vla=$VLA asset=$ASSET tile=($A2A_SINGLE_TILE_X,$A2A_SINGLE_TILE_Y) yaw=$A2A_SINGLE_TILE_YAW seed=$SEED blocks='$BLOCKS' par=$PAR"
i=0
for s in $BLOCKS; do
  name="${NAME}-${VLA}-center-sh${s}"
  [ -f "$A/outputs/$name/glob/vis_0_test/stats.yaml" ] && { echo "SKIP $name"; continue; }
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 20; done
  gpu=$(( i % 2 ))
  cport=$PORT; [ $(( i % 2 )) -eq 1 ] && cport=$(( PORT + 1 ))
  echo "[$(date -u +%H:%M:%S)] START $name gpu=$gpu${VLA_ARG:+ port=$cport}"
  CUDA_VISIBLE_DEVICES=$gpu INTERNVLA_PORT=$cport INTERNVLA_HOST=127.0.0.1 \
    python -u -m simpler_env.eval --vla "$VLA" $VLA_ARG \
      --assets "$ASSET" --obj-set test --start-id "$s" --count "$SHARD" \
      --episode-len 80 --buffer-inferbatch 10 --buffer-minibatch -1 \
      --seed "$SEED" --name "$name" < /dev/null > "$LOG/$name.log" 2>&1 &
  i=$(( i + 1 ))
  sleep 20
done
wait
echo "SINGLE_CARD_CENTER_DONE ($VLA, $NAME) $(date -u)"
