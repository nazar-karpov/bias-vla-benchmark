#!/usr/bin/env bash
# НОЧНОЙ ПРОГОН: 33 вопроса × 2 полярности × 10 сцен × 2 демо-пары = 1320 эп,
# noswap+swap, на magma и internvla. Масштаб 1.3 вшит в кардсет
# (metrics/neutral_position_bias.md: 1.3 лучше 1.5 по чистому крену).
#
# Magma  — standalone, две раскладки параллельно на GPU 0 и 1.
# InternVLA — сервер (env internvla, GPU 1) + клиенты (env magma_act2answer, GPU 0).
# Внутри процесса eval сам режет прогон на шарды (--shard-size) и
# резюмируется: уже посчитанные шарды пропускаются при перезапуске.
set -uo pipefail
R=/workspace/moskalenko/bias-vla-benchmark-main
A="$R/Act2Answer"
CONDA=/workspace/moskalenko/conda
export REPO_ROOT="$A" PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
export MS_ASSET_DIR=/workspace/moskalenko/maniskill_assets
export HF_HOME=/workspace/moskalenko/hf_cache
export TOKENIZERS_PARALLELISM=false
export A2A_TRAJ_LOG=1              # копим траектории под интегральные метрики
# видео не нужны для метрик и жрут CPU после симуляции (см. run.py)
export A2A_SAVE_VIDEO="${A2A_SAVE_VIDEO:-0}"
export BOARD_XY_SCALE=1.0          # масштаб (1.3) уже в model_db кардсета

ASSET="${ASSET:-pairs_q33_night}"
COUNT="${COUNT:-1320}"
SHARD="${SHARD:-110}"
EPLEN="${EPLEN:-80}"
IB="${IB:-10}"
MODELS="${MODELS:-magma internvla}"
PORT="${INTERNVLA_PORT:-10093}"
LOG=/workspace/moskalenko/logs_night
mkdir -p "$LOG"
source "$CONDA/etc/profile.d/conda.sh"

srv_pid=""
port_open() {
  python -c "import socket;s=socket.socket();s.settimeout(2);exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)" 2>/dev/null
}
start_internvla_server() {
  port_open && { echo "сервер InternVLA уже слушает :$PORT"; return 0; }
  echo "[$(date -u +%H:%M:%S)] поднимаю сервер InternVLA :$PORT (GPU 1)"
  ( conda activate "$CONDA/envs/internvla"
    export PYTHONPATH="$R/InternVLA-M1:$PYTHONPATH"
    export CUDA_VISIBLE_DEVICES=1
    cd "$R/InternVLA-M1"
    exec python deployment/model_server/server_policy_M1.py \
      --ckpt_path "$(cat "$R/internvla_ckpt/ckpt_path.txt")" \
      --port "$PORT" --use_bf16 ) > "$LOG/internvla_server.log" 2>&1 &
  srv_pid=$!
  for i in $(seq 1 120); do
    port_open && { echo "сервер поднялся за ~$((i*10))с"; sleep 15; return 0; }
    kill -0 "$srv_pid" 2>/dev/null || { echo "сервер УМЕР, см. $LOG/internvla_server.log"; return 1; }
    sleep 10
  done
  echo "СЕРВЕР НЕ ПОДНЯЛСЯ за 20 мин"; return 1
}

run_layout() {  # model layout gpu
  local mdl="$1" lay="$2" gpu="$3"
  local sw=""; [ "$lay" = swap ] && sw="--do-swap"
  local name="night-q33-${mdl}-${lay}"
  local extra=""
  [ "$mdl" = internvla ] && extra="--vla-path $(cat "$R/internvla_ckpt/ckpt_path.txt")"
  echo "[$(date -u +%H:%M:%S)] START $name gpu=$gpu"
  CUDA_VISIBLE_DEVICES=$gpu INTERNVLA_PORT=$PORT INTERNVLA_HOST=127.0.0.1 \
    python -u -m simpler_env.eval --vla "$mdl" $extra \
      --assets "$ASSET" --obj-set test --start-id 0 --count "$COUNT" \
      --episode-len "$EPLEN" --buffer-inferbatch "$IB" --buffer-minibatch -1 \
      --shard-size "$SHARD" --name "$name" $sw < /dev/null \
      > "$LOG/$name.log" 2>&1
  echo "[$(date -u +%H:%M:%S)] DONE  $name rc=$?"
}

for mdl in $MODELS; do
  echo "===== MODEL $mdl $(date -u) ====="
  if [ "$mdl" = internvla ]; then
    start_internvla_server || { echo "ПРОПУСК internvla"; continue; }
    # Сервер (Qwen2.5-VL-3B bf16) делит GPU1 с клиентом: клиент занимает ~37 ГБ
    # из 80, запаса хватает. Иначе оба клиента ютятся на GPU0 и прогон вдвое
    # дольше (6.4 ч против 3.2 ч).
    G1=0; G2=1
  else
    G1=0; G2=1
  fi
  conda activate "$CONDA/envs/magma_act2answer"   # клиент всегда здесь
  cd "$A/SimplerEnv" || exit 1
  run_layout "$mdl" noswap "$G1" &
  run_layout "$mdl" swap   "$G2" &
  wait
  if [ -n "$srv_pid" ]; then
    kill "$srv_pid" 2>/dev/null; pkill -f server_policy_M1 2>/dev/null
    srv_pid=""; sleep 20
  fi
  echo "===== MODEL $mdl FINISHED $(date -u) ====="
done
echo "NIGHT_ALL_DONE $(date -u)"
