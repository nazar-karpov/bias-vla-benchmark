#!/usr/bin/env bash
# Сетка масштабов плиток на CEILING-задаче (правильный ответ известен) —
# объективный критерий выбора размера: где accuracy максимальна и ещё нет
# геометрических артефактов (см. metrics/tile_visibility.txt).
#
# Обе GPU: масштабы раскидываются по картам парами.
#   SCALES="1.0 1.3 1.5 1.7" MODEL=magma bash run_scale_sweep.sh
set -uo pipefail
R=/workspace/moskalenko/bias-vla-benchmark-main
A="$R/Act2Answer"
source /workspace/moskalenko/conda/etc/profile.d/conda.sh
conda activate magma_act2answer
export REPO_ROOT="$A" PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
export MS_ASSET_DIR=/workspace/moskalenko/maniskill_assets
export HF_HOME=/workspace/moskalenko/hf_cache   # веса на /workspace: ~/.cache умирает с нодой
export TOKENIZERS_PARALLELISM=false
export A2A_TRAJ_LOG=1          # заодно копим траектории для будущих метрик
# видео не нужны для метрик и жрут CPU после симуляции (см. run.py)
export A2A_SAVE_VIDEO="${A2A_SAVE_VIDEO:-0}"
cd "$A/SimplerEnv" || exit 1

SCALES="${SCALES:-1.0 1.3 1.5 1.7}"
MODEL="${MODEL:-magma}"
ASSET="${ASSET:-ceiling_color}"
COUNT="${COUNT:-56}"
IB="${IB:-8}"
LOG=/workspace/moskalenko/logs_scale_sweep
mkdir -p "$LOG"

i=0
for sc in $SCALES; do
  for sw in "" "--do-swap"; do
    tag=noswap; [ -n "$sw" ] && tag=swap
    name="scalesweep-${ASSET#ceiling_}-${MODEL}-s${sc}-${tag}"
    if [ -f "$A/outputs/$name/glob/vis_0_test/stats.yaml" ]; then
      echo "SKIP $name"; continue
    fi
    gpu=$(( i % 2 ))
    echo "[$(date -u +%H:%M:%S)] START $name gpu=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu BOARD_XY_SCALE=$sc \
      python -u -m simpler_env.eval --vla "$MODEL" \
        --assets "$ASSET" --count "$COUNT" --episode-len 80 \
        --buffer-inferbatch "$IB" --buffer-minibatch -1 \
        --name "$name" $sw < /dev/null > "$LOG/$name.log" 2>&1 &
    i=$(( i + 1 ))
    [ $(( i % 2 )) -eq 0 ] && wait     # по одному процессу на карту
  done
done
wait
echo "SWEEP_DONE"
