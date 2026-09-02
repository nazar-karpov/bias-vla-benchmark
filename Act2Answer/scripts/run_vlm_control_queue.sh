#!/usr/bin/env bash
# Контроль валидности: тот же кадр, вопрос man/woman (ответ визуально определён).
# 200 сцен × 2 раскладки × 2 полярности = 800 запросов на модель.
set -uo pipefail
W=/workspace/moskalenko/bias-vla-benchmark-main
FRAMES=$W/Act2Answer/outputs/simframes_confirm13
OUT=$W/Act2Answer/outputs/vlm_confirm/control
mkdir -p "$OUT"
source ~/conda/etc/profile.d/conda.sh

SPECS=(
  "internvla-m1-qwen25vl3b|internvla|qwen|InternRobotics/InternVLA-M1|bfloat16"
  "cosmos-reason2-2b|mibot|qwen|nvidia/Cosmos-Reason2-2B|bfloat16"
  "paligemma-3b-pt-224|spatialvla_act2answer|paligemma|google/paligemma-3b-pt-224|bfloat16"
  "paligemma2-3b-pt-224|spatialvla_act2answer|paligemma|google/paligemma2-3b-pt-224|bfloat16"
  "qwen3vl-4b-instruct|mibot|qwen|Qwen/Qwen3-VL-4B-Instruct|bfloat16"
  "rldx1-vlm-qwen3vl8b|mibot|qwen|RLWRLD/RLDX-1-VLM|bfloat16"
  "magma-8b|magma_act2answer|magma|microsoft/Magma-8B|float16"
  "openvla-prismatic-7b|openvla_rl4vla|prismatic|prism-dinosiglip-224px+7b|bfloat16"
)

want=("$@")
selected() {
  [ ${#want[@]} -eq 0 ] && return 0
  for w in "${want[@]}"; do [ "$w" = "$1" ] && return 0; done
  return 1
}

for spec in "${SPECS[@]}"; do
  IFS='|' read -r tag env backend model dtype <<<"$spec"
  selected "$tag" || continue
  res="$OUT/$tag.json"
  n=$(python3 -c "import json;print(len(json.load(open('$res'))))" 2>/dev/null || echo 0)
  if [ "$n" = "800" ]; then echo "[skip] $tag контроль готов"; continue; fi
  echo "=== [$(date -u +%H:%M:%S)] CONTROL $tag env=$env"
  conda activate ~/conda/envs/"$env" || { echo "[fail] env $env"; continue; }
  export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1 PYTHONNOUSERSITE=1
  [ "$backend" = prismatic ] && export PYTHONPATH="$W/Act2Answer/openvla"
  python -u "$W/Act2Answer/scripts/vlm_confirm_choice.py" \
    --backend "$backend" --model "$model" --tag "$tag" --dtype "$dtype" \
    --questions control --frames-dir "$FRAMES" --out "$res" --gen-check 4 < /dev/null
  echo "=== [$(date -u +%H:%M:%S)] CONTROL $tag rc=$? -> $res"
  unset PYTHONPATH
  conda deactivate
done
echo "CONTROL QUEUE DONE $(date -u)"
