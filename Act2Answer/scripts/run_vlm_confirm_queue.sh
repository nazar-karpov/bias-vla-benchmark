#!/usr/bin/env bash
# Последовательный прогон базовых VLM на кадрах confirm. Резюмируемо: если в
# outputs/vlm_confirm/<tag>.json уже 3200 записей — модель пропускается.
# Модели идут ПО ОДНОЙ: на узле живут чужие VLA-прогоны, память GPU делим.
#
#   bash run_vlm_confirm_queue.sh <tag1> <tag2> ...     # только указанные
#   bash run_vlm_confirm_queue.sh                        # все из таблицы
set -uo pipefail
W=/workspace/moskalenko/bias-vla-benchmark-main
FRAMES=$W/Act2Answer/outputs/simframes_confirm13
OUT=$W/Act2Answer/outputs/vlm_confirm
mkdir -p "$OUT"
source ~/conda/etc/profile.d/conda.sh

# tag|conda-env|backend|model|dtype   (порядок = от мелких к крупным)
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

free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

for spec in "${SPECS[@]}"; do
  IFS='|' read -r tag env backend model dtype <<<"$spec"
  selected "$tag" || continue
  res="$OUT/$tag.json"
  n=$(python3 -c "import json,sys;print(len(json.load(open('$res'))))" 2>/dev/null || echo 0)
  if [ "$n" = "3200" ]; then echo "[skip] $tag уже готов (3200)"; continue; fi

  echo "=== [$(date -u +%H:%M:%S)] $tag  env=$env backend=$backend  free=$(free_mb)MiB"
  conda activate ~/conda/envs/"$env" || { echo "[fail] env $env"; continue; }
  export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
  export PYTHONNOUSERSITE=1
  [ "$backend" = prismatic ] && export PYTHONPATH="$W/Act2Answer/openvla"
  python -u "$W/Act2Answer/scripts/vlm_confirm_choice.py" \
    --backend "$backend" --model "$model" --tag "$tag" --dtype "$dtype" \
    --frames-dir "$FRAMES" --out "$res" --gen-check 8 < /dev/null
  rc=$?
  unset PYTHONPATH
  conda deactivate
  echo "=== [$(date -u +%H:%M:%S)] $tag rc=$rc  ->  $res"
done
echo "QUEUE DONE $(date -u)"
