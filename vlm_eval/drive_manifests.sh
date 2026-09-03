#!/usr/bin/env bash
# Прогон всех моделей из основной таблицы по трём датасетам.
# У каждой модели своё окружение: Magma и Florence живут в transformers 4.40,
# SpatialVLA в 4.48, остальные в 5.0. PYLIBS отключается для OpenVLA — там свой timm.
set -u
# Код лежит рядом со скриптом, данные — отдельно и переопределяются через EXP.
E="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP=${EXP:-/workspace/moskalenko/exp_weapon}
OUT=${1:-$E/results_manifests}
JD=/home/jovyan/.mlspace/envs/jobs_demo/bin/python
PI0=/home/jovyan/.mlspace/envs/krapukhin_pi0/bin/python
OVL=/home/jovyan/.mlspace/envs/krapukhin_openvla/bin/python

# окружение | PYLIBS (- = выключить) | модель
SPECS=(
  "$JD|+|Qwen/Qwen2.5-VL-3B-Instruct"
  "$JD|+|Qwen/Qwen3-VL-4B-Instruct"
  "$JD|+|nvidia/Cosmos-Reason2-2B"
  "$JD|+|RLWRLD/RLDX-1-VLM"
  "$JD|+|google/paligemma-3b-mix-224"
  "$JD|+|google/paligemma-3b-pt-224"
  "$JD|+|google/paligemma2-3b-mix-224"
  "$JD|+|google/paligemma2-3b-pt-224"
  "$JD|+|vla:internvla"
  "$JD|+|vla:xiaomi_pretrain"
  "$JD|+|vla:pi05"
  "$JD|+|allenai/Molmo2-ER"
  "$JD|+|allenai/MolmoAct2-SO100_101"
  "$PI0|+|IPEC-COMMUNITY/spatialvla-4b-224-pt"
  "$OVL|+|microsoft/Magma-8B"
  "$OVL|+|microsoft/Florence-2-large"
  "$OVL|+|microsoft/Florence-2-large-ft"
  "$OVL|-|openvla/openvla-7b"
)

mkdir -p "$OUT"
for spec in "${SPECS[@]}"; do
  IFS='|' read -r PYBIN LIBS MODEL <<<"$spec"
  echo ""
  echo "############################## $MODEL  $(date -u +%H:%M:%S)"
  if [ "$LIBS" = "-" ]; then
    MODEL="$MODEL" PY="$PYBIN" PYLIBS= bash "$E/run_manifest_queue.sh" "$OUT" || echo "СБОЙ $MODEL"
  else
    MODEL="$MODEL" PY="$PYBIN" bash "$E/run_manifest_queue.sh" "$OUT" || echo "СБОЙ $MODEL"
  fi
done
echo "ВСЕ МОДЕЛИ ПРОЙДЕНЫ $(date -u)"
