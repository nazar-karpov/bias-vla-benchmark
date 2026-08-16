#!/usr/bin/env bash
# ПОЛНЫЙ КРЕСТ: 33 вопроса × 2 полярности × 50 сцен × 4 демо-пары = 13200 эп
# × noswap/swap на модель. Масштаб 1.3 вшит в кардсет pairs_q33_full.
#
# Зачем: ночной прогон на 1320 эп дал ~20 наблюдений на ячейку (вопрос×ось) —
# статистика распылилась по 66 ячейкам и ни один эффект не пережил FDR.
# Здесь 200 эп на ячейку, т.е. уровень старого confirm, где эффекты находились.
#
# Раскладка (из профилировки):
#   magma     — автономна, узкое место VRAM (35 ГБ/процесс) -> 4 клиента;
#   internvla — узкое место СЕРВЕР (ест ~17 ядер), поэтому серверов ДВА
#               (:10093 GPU1, :10094 GPU0), клиенты раскидываются поровну.
# Резюмируемость: готовые шарды пропускаются, после падения просто перезапустить.
set -uo pipefail
R=/workspace/moskalenko/bias-vla-benchmark-main
A="$R/Act2Answer"
CONDA=/workspace/moskalenko/conda
export REPO_ROOT="$A" PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
export MS_ASSET_DIR=/workspace/moskalenko/maniskill_assets
export HF_HOME=/workspace/moskalenko/hf_cache
export TOKENIZERS_PARALLELISM=false
export A2A_TRAJ_LOG=1        # траектории под непрерывные метрики (у magma AR 7%,
                             # дискретный канал слаб — опора на pull в мм)
export A2A_SAVE_VIDEO=0      # 264 шарда × 100 видео = часы CPU впустую
export BOARD_XY_SCALE=1.0    # масштаб 1.3 уже в model_db кардсета

ASSET="${ASSET:-pairs_q33_full}"
TOTAL="${TOTAL:-13200}"
SHARD="${SHARD:-100}"
MODELS="${MODELS:-internvla magma}"   # internvla первой: у неё AR 50% и результат надёжнее
PORT="${INTERNVLA_PORT:-10093}"
LOG=/workspace/moskalenko/logs_full_cross
mkdir -p "$LOG"
source "$CONDA/etc/profile.d/conda.sh"

start_server() {  # port gpu tag
  local p="$1" g="$2" tag="$3"
  python -c "import socket;s=socket.socket();s.settimeout(2);exit(0 if s.connect_ex(('127.0.0.1',$p))==0 else 1)" 2>/dev/null && {
    echo "сервер $tag уже слушает :$p"; return 0; }
  echo "[$(date -u +%H:%M:%S)] поднимаю сервер $tag :$p (GPU $g)"
  ( conda activate "$CONDA/envs/internvla"
    export PYTHONPATH="$R/InternVLA-M1:$PYTHONPATH" CUDA_VISIBLE_DEVICES="$g"
    cd "$R/InternVLA-M1"
    exec python deployment/model_server/server_policy_M1.py \
      --ckpt_path "$(cat "$R/internvla_ckpt/ckpt_path.txt")" \
      --port "$p" --use_bf16 ) > "$LOG/_server_$tag.log" 2>&1 &
  for i in $(seq 1 120); do
    python -c "import socket;s=socket.socket();s.settimeout(2);exit(0 if s.connect_ex(('127.0.0.1',$p))==0 else 1)" 2>/dev/null && {
      echo "  поднялся за ~$((i*10))с"; sleep 15; return 0; }
    sleep 10
  done
  echo "  СЕРВЕР $tag НЕ ПОДНЯЛСЯ"; return 1
}

for MODEL in $MODELS; do
  echo "===== MODEL $MODEL $(date -u) ====="
  if [ "$MODEL" = internvla ]; then
    start_server "$PORT" 1 A || continue
    start_server "$(( PORT + 1 ))" 0 B || true
    PAR=8
  else
    pkill -f server_policy_M1 2>/dev/null; sleep 10   # освободить VRAM под magma
    PAR=4
  fi
  conda activate magma_act2answer     # клиенты обеих моделей живут здесь
  cd "$A/SimplerEnv" || exit 1
  VLA_ARG=""
  [ "$MODEL" = internvla ] && VLA_ARG="--vla-path $(cat "$R/internvla_ckpt/ckpt_path.txt")"

  i=0
  for lay in noswap swap; do
    sw=""; [ "$lay" = swap ] && sw="--do-swap"
    for ((s=0; s<TOTAL; s+=SHARD)); do
      name="full33-${MODEL}-${lay}-sh${s}"
      [ -f "$A/outputs/$name/glob/vis_0_test/stats.yaml" ] && continue
      while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 20; done
      gpu=$(( i % 2 ))
      cport=$PORT
      [ "$MODEL" = internvla ] && [ $(( i % 2 )) -eq 1 ] && cport=$(( PORT + 1 ))
      echo "[$(date -u +%H:%M:%S)] START $name gpu=$gpu${cport:+ port=$cport}"
      CUDA_VISIBLE_DEVICES=$gpu INTERNVLA_PORT=$cport INTERNVLA_HOST=127.0.0.1 \
        python -u -m simpler_env.eval --vla "$MODEL" $VLA_ARG \
          --assets "$ASSET" --obj-set test --start-id "$s" --count "$SHARD" \
          --episode-len 80 --buffer-inferbatch 10 --buffer-minibatch -1 \
          --name "$name" $sw < /dev/null > "$LOG/$name.log" 2>&1 &
      i=$(( i + 1 ))
      sleep 20
    done
  done
  wait
  echo "===== MODEL $MODEL FINISHED $(date -u) ====="
done
pkill -f server_policy_M1 2>/dev/null
echo "FULL_CROSS_DONE $(date -u)"
