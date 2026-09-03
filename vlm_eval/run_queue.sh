#!/usr/bin/env bash
# Очередь опроса VLM по готовым кадрам на ОБЩЕМ узле.
#
# Соседи занимают GPU волнами, поэтому: ждём окно по памяти, запускаем, а если
# нас выбило по OOM — докатываем с чекпойнта (vlm_ask.py умеет). Так одна модель
# может собраться за несколько окон.
#
#   bash run_queue.sh <frames_dir> <meta.json> <questions.json> <out_dir> [модели...]
set -u
# Код лежит рядом со скриптом, данные — отдельно и переопределяются через EXP.
E="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP=${EXP:-/workspace/moskalenko/exp_weapon}

FRAMES=${1:?frames_dir}
META=${2:?pairs_meta.json}
QUESTIONS=${3:?questions.json}
OUT=${4:?out_dir}
shift 4
MODELS=("$@")
[ ${#MODELS[@]} -gt 0 ] || MODELS=(Qwen/Qwen2.5-VL-3B-Instruct)

PY=${PY:-/home/jovyan/.mlspace/envs/jobs_demo/bin/python}
NEED_MB=${NEED_MB:-12000}      # окно памяти, с которым имеет смысл стартовать
WAIT_MAX=${WAIT_MAX:-3600}     # сколько ждать окно, секунд
ATTEMPTS=${ATTEMPTS:-20}       # попыток на модель (каждая — новое окно)
export HF_HOME=${HF_HOME:-/workspace/moskalenko/hf_cache}
# токен лежит в домашней папке, а HF_HOME мы уводим на /workspace — указываем путь явно,
# иначе gated-репозитории (PaliGemma, Cosmos) не откроются
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HUB_DISABLE_XET=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Доп. пакеты, которых нет в окружении (open_clip для Magma). Каталог отключаемый:
# у OpenVLA свои требования к timm, и общий prefix там только мешает.
PYLIBS=${PYLIBS-$EXP/pylibs}
[ -n "$PYLIBS" ] && export PYTHONPATH="$PYLIBS:${PYTHONPATH:-}"
mkdir -p "$OUT"

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '{free=$2-$3; print $1, free}' | sort -k2 -nr | head -1
}

is_complete() {  # $1 — json результата
  [ -f "$1" ] || return 1
  "$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('complete') else 1)" "$1"
}

for M in "${MODELS[@]}"; do
  TAG=$(echo "$M" | tr '/' '_')
  DST="$OUT/${TAG}.json"
  if is_complete "$DST"; then echo "SKIP $M (готово)"; continue; fi

  for try in $(seq 1 "$ATTEMPTS"); do
    waited=0
    while :; do
      read -r GPU FREE <<<"$(pick_gpu)"
      [ "$FREE" -ge "$NEED_MB" ] && break
      if [ "$waited" -ge "$WAIT_MAX" ]; then break; fi
      [ $((waited % 600)) -eq 0 ] && echo "WAIT $M: свободно ${FREE}MB (попытка $try, ${waited}s)"
      sleep 60; waited=$((waited + 60))
    done
    read -r GPU FREE <<<"$(pick_gpu)"
    if [ "$FREE" -lt "$NEED_MB" ]; then
      echo "NOWINDOW $M: попытка $try, максимум свободного ${FREE}MB"; continue
    fi
    echo "=== RUN $M попытка $try на GPU$GPU (свободно ${FREE}MB) $(date -u)"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u $E/vlm_ask.py \
        --frames-dir "$FRAMES" --meta "$META" --questions "$QUESTIONS" \
        --model "$M" --out "$DST" --device cuda:0 ${LIMIT:+--limit $LIMIT} \
        ${GENERATE:+--generate $GENERATE}
    if is_complete "$DST"; then echo "=== OK $M (попыток: $try)"; break; fi
    echo "=== ПРЕРВАНО $M, докачу на следующем окне"
    sleep 30
  done
  is_complete "$DST" || echo "=== НЕ СОБРАЛОСЬ $M"
done
echo "ОЧЕРЕДЬ ЗАВЕРШЕНА $(date -u)"
