#!/usr/bin/env bash
# ТОП-8 ВОПРОСОВ полного креста на InternVLA (вместо всех 33: ~37ч -> ~9ч).
# Вопросы: boss, wealthy, athlete, dentist, professor, pilot, CEO, sysadmin —
# те, где у Magma были сильнейшие эффекты (см. эксп. 36/39). Кардсет и имена
# шардов ТЕ ЖЕ (pairs_q33_full, full33-internvla-*), поэтому уже готовые шарды
# этих блоков переиспользуются, а анализ full33_summary/integral_pull работает
# без изменений.
#
# ЧЕРЕДОВАНИЕ noswap/swap: шарды идут парами (noswap-shN, swap-shN, noswap-shN+100,
# ...), а не двумя волнами. Так обе раскладки всегда набраны примерно поровну и
# промежуточный pull (парная метрика) считается честно в любой момент прогона.
set -uo pipefail
R=/workspace/moskalenko/bias-vla-benchmark-main
A="$R/Act2Answer"
CONDA=/workspace/moskalenko/conda
export REPO_ROOT="$A" PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
export MS_ASSET_DIR=/workspace/moskalenko/maniskill_assets
export HF_HOME=/workspace/moskalenko/hf_cache
export TOKENIZERS_PARALLELISM=false
export A2A_TRAJ_LOG=1
export A2A_SAVE_VIDEO=0
export BOARD_XY_SCALE=1.0

ASSET="${ASSET:-pairs_q33_full}"
SHARD=100
PAR="${PAR:-6}"
PORT="${INTERNVLA_PORT:-10093}"
LOG=/workspace/moskalenko/logs_full_cross
mkdir -p "$LOG"

# начала блоков топ-8 (pos и neg по 200 эп = по 2 шарда на блок)
BLOCKS="0 200 400 600 4000 4200 6400 6600 7200 7400 8400 8600 9600 9800 10400 10600"

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

start_server "$PORT" 1 A || exit 1
start_server "$(( PORT + 1 ))" 0 B || true

conda activate magma_act2answer     # клиент
cd "$A/SimplerEnv" || exit 1
VLA_ARG="--vla-path $(cat "$R/internvla_ckpt/ckpt_path.txt")"

i=0
for s in $BLOCKS; do
  for lay in noswap swap; do      # <-- чередование внутри шарда: пара сразу
    sw=""; [ "$lay" = swap ] && sw="--do-swap"
    name="full33-internvla-${lay}-sh${s}"
    [ -f "$A/outputs/$name/glob/vis_0_test/stats.yaml" ] && { echo "SKIP $name"; continue; }
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 20; done
    gpu=$(( i % 2 ))
    cport=$PORT; [ $(( i % 2 )) -eq 1 ] && cport=$(( PORT + 1 ))
    echo "[$(date -u +%H:%M:%S)] START $name gpu=$gpu port=$cport"
    CUDA_VISIBLE_DEVICES=$gpu INTERNVLA_PORT=$cport INTERNVLA_HOST=127.0.0.1 \
      python -u -m simpler_env.eval --vla internvla $VLA_ARG \
        --assets "$ASSET" --obj-set test --start-id "$s" --count "$SHARD" \
        --episode-len 80 --buffer-inferbatch 10 --buffer-minibatch -1 \
        --name "$name" $sw < /dev/null > "$LOG/$name.log" 2>&1 &
    i=$(( i + 1 ))
    sleep 20
  done
done
wait
echo "TOP8_CROSS_DONE $(date -u)"
