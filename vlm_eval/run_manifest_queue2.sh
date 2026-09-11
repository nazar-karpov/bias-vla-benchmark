#!/usr/bin/env bash
# Прогон одной модели по трём датасетам в трёх постановках.
#
# Девять условий на модель:
#   <датасет> single       — одна картинка, ответ Yes/No
#   <датасет> pair ab      — склейка двух, ответ A/B (формулировка из манифеста)
#   <датасет> pair lr      — та же склейка, ответ left/right (наша прежняя)
#
# Узел общий: ждём окно по видеопамяти, при вылете докатываем с чекпойнта.
#
#   MODEL=... PY=... [PYLIBS=...] bash run_manifest_queue.sh <out_dir>
set -u

OUT=${1:?out_dir}
MODEL=${MODEL:?MODEL}
PY=${PY:-/home/jovyan/.mlspace/envs/jobs_demo/bin/python}
# Код лежит рядом со скриптом, данные — отдельно и переопределяются через EXP.
E="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP=${EXP:-/workspace/moskalenko/exp_weapon}
D=/workspace/moskalenko/datasets
NEED_MB=${NEED_MB:-14000}
WAIT_MAX=${WAIT_MAX:-3600}
ATTEMPTS=${ATTEMPTS:-8}

export HF_HOME=${HF_HOME:-/workspace/moskalenko/hf_cache}
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export HF_HUB_DISABLE_XET=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYLIBS=${PYLIBS-$EXP/pylibs}
[ -n "$PYLIBS" ] && export PYTHONPATH="$PYLIBS:${PYTHONPATH:-}"
mkdir -p "$OUT"

# датасет | манифест | режим | формулировка | сколько строк (0 = все)
CONDS=(
  "Veri_safety|veri_vlm_parallel_single_image_binary.csv|single|ab|0"
  "Veri_safety|veri_vlm_parallel_two_image_selection.csv|pair|ab|0"
  "Veri_safety|veri_vlm_parallel_two_image_selection.csv|pair|lr|0"
  "Focus_Reflect|focus_vlm_parallel_single_image_binary.csv|single|ab|300"
  "Focus_Reflect|focus_vlm_parallel_two_image_selection.csv|pair|ab|150"
  "Focus_Reflect|focus_vlm_parallel_two_image_selection.csv|pair|lr|150"
  "Visbias|visbias_vlm_parallel_single_image_binary.csv|single|ab|360"
  "Visbias|visbias_vlm_parallel_two_image_selection.csv|pair|ab|0"
  "Visbias|visbias_vlm_parallel_two_image_selection.csv|pair|lr|0"
)

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '{free=$2-$3; print $1, free}' | sort -k2 -nr | head -1
}
is_complete() {
  [ -f "$1" ] || return 1
  "$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('complete') else 1)" "$1"
}

TAG=$(echo "$MODEL" | tr '/' '_')
for cond in "${CONDS[@]}"; do
  IFS='|' read -r DS MAN MODE PHR N <<<"$cond"
  SUF=$([ "$MODE" = single ] && echo single || echo "pair_$PHR")
  DST="$OUT/${TAG}__${DS}__${SUF}.json"
  if is_complete "$DST"; then echo "SKIP $DS/$SUF"; continue; fi

  for try in $(seq 1 "$ATTEMPTS"); do
    waited=0
    while :; do
      read -r GPU FREE <<<"$(pick_gpu)"
      [ "$FREE" -ge "$NEED_MB" ] && break
      [ "$waited" -ge "$WAIT_MAX" ] && break
      [ $((waited % 600)) -eq 0 ] && echo "WAIT $MODEL $DS/$SUF: свободно ${FREE}MB"
      sleep 60; waited=$((waited + 60))
    done
    read -r GPU FREE <<<"$(pick_gpu)"
    if [ "$FREE" -lt "$NEED_MB" ]; then echo "NOWINDOW $MODEL $DS/$SUF"; continue; fi
    echo "=== RUN $MODEL | $DS | $SUF | попытка $try | GPU$GPU ${FREE}MB | $(date -u +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$E/vlm_ask_manifest.py" \
        --manifest "$D/$DS/vlm_manifests/$MAN" --images-root "$D/$DS" \
        --mode "$MODE" --phrasing "$PHR" --sample "$N" \
        --model "$MODEL" --out "$DST" --device cuda:0
    is_complete "$DST" && { echo "=== OK $DS/$SUF"; break; }
    echo "=== ПРЕРВАНО $DS/$SUF, докачу"; sleep 20
  done
done
echo "МОДЕЛЬ ЗАВЕРШЕНА: $MODEL $(date -u)"
