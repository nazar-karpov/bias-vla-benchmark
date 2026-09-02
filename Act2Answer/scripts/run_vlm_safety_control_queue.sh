#!/usr/bin/env bash
# Контроль к safety-прогону: те же 96 пар и вопросы, но на СКЛЕЙКЕ исходных картинок
# (без симулятора). 96 × 2 порядка × 2 полярности = 384 запроса на модель.
set -uo pipefail
W=/workspace/moskalenko/bias-vla-benchmark-main
FRAMES=$W/Act2Answer/outputs/simframes_safety13
IMGS=/home/user/bias_benchmark/roman_folder/bias-vla-benchmark-main/sohas100_run/images
OUT=$W/Act2Answer/outputs/vlm_safety/control_concat
NEED=384
mkdir -p "$OUT"
source ~/conda/etc/profile.d/conda.sh

SPECS=(
  "internvla-m1-qwen25vl3b|internvla|qwen|InternRobotics/InternVLA-M1|bfloat16|8"
  "cosmos-reason2-2b|mibot|qwen|nvidia/Cosmos-Reason2-2B|bfloat16|8"
  "paligemma-3b-pt-224|spatialvla_act2answer|paligemma|google/paligemma-3b-pt-224|bfloat16|8"
  "paligemma2-3b-pt-224|spatialvla_act2answer|paligemma|google/paligemma2-3b-pt-224|bfloat16|8"
  "diag-paligemma-3b-mix-224|spatialvla_act2answer|paligemma|google/paligemma-3b-mix-224|bfloat16|8"
  "diag-paligemma2-3b-mix-224|spatialvla_act2answer|paligemma|google/paligemma2-3b-mix-224|bfloat16|8"
  "qwen3vl-4b-instruct|mibot|qwen|Qwen/Qwen3-VL-4B-Instruct|bfloat16|8"
  "rldx1-vlm-qwen3vl8b|mibot|qwen|RLWRLD/RLDX-1-VLM|bfloat16|8"
  "magma-8b|magma_act2answer|magma|microsoft/Magma-8B|float16|8"
  "openvla-prismatic-7b|openvla_rl4vla|prismatic|prism-dinosiglip-224px+7b|bfloat16|0"
)

want=("$@")
selected() {
  [ ${#want[@]} -eq 0 ] && return 0
  for w in "${want[@]}"; do [ "$w" = "$1" ] && return 0; done
  return 1
}

for spec in "${SPECS[@]}"; do
  IFS='|' read -r tag env backend model dtype genchk <<<"$spec"
  selected "$tag" || continue
  res="$OUT/$tag.json"
  n=$(python3 -c "import json;print(len(json.load(open('$res'))))" 2>/dev/null || echo 0)
  if [ "$n" = "$NEED" ]; then echo "[skip] $tag готов"; continue; fi

  echo "=== [$(date -u +%H:%M:%S)] $tag  env=$env backend=$backend"
  conda activate ~/conda/envs/"$env" || { echo "[fail] env $env"; continue; }
  export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1 PYTHONNOUSERSITE=1
  [ "$backend" = prismatic ] && export PYTHONPATH="$W/Act2Answer/openvla"
  python -u "$W/Act2Answer/scripts/vlm_safety_concat_control.py" \
    --backend "$backend" --model "$model" --tag "$tag" --dtype "$dtype" \
    --frames-dir "$FRAMES" --images-dir "$IMGS" --out "$res" --gen-check "$genchk" < /dev/null
  rc=$?
  unset PYTHONPATH
  conda deactivate
  echo "=== [$(date -u +%H:%M:%S)] $tag rc=$rc"
done
echo "QUEUE DONE $(date -u)"
