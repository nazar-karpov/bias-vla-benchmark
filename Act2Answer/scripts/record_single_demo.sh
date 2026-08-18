#!/usr/bin/env bash
# Перепрогон отдельных single-card эпизодов С ЗАПИСЬЮ ВИДЕО (демо для отчёта).
# Боевые прогоны идут с A2A_SAVE_VIDEO=0 (264 шарда × 100 роликов = часы CPU),
# поэтому наглядные примеры пишутся отдельно узкими диапазонами.
#
#   STARTS="261 359 40" bash record_single_demo.sh
set -uo pipefail
R=/workspace/moskalenko/bias-vla-benchmark-main
A="$R/Act2Answer"
CONDA=/workspace/moskalenko/conda
export REPO_ROOT="$A" PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
export MS_ASSET_DIR=/workspace/moskalenko/maniskill_assets
export HF_HOME=/workspace/moskalenko/hf_cache
export TOKENIZERS_PARALLELISM=false
export A2A_SINGLE_TILE=1
export A2A_TRAJ_LOG=1
export A2A_SAVE_VIDEO=1          # <-- главное отличие от боевого раннера
export BOARD_XY_SCALE=1.0

ASSET="${ASSET:-pairs_single_pilot}"
STARTS="${STARTS:-261 359 40 31}"
LOG=/workspace/moskalenko/logs_single_card
mkdir -p "$LOG"

source "$CONDA/etc/profile.d/conda.sh"
conda activate "$CONDA/envs/magma_act2answer"
cd "$A/SimplerEnv" || exit 1

for s in $STARTS; do
  name="demo-single-ep${s}"
  echo "[$(date -u +%H:%M:%S)] запись $name"
  CUDA_VISIBLE_DEVICES="${GPU:-1}" python -u -m simpler_env.eval --vla magma \
    --assets "$ASSET" --obj-set test --start-id "$s" --count 1 \
    --episode-len 80 --buffer-inferbatch 10 --buffer-minibatch -1 \
    --name "$name" < /dev/null > "$LOG/$name.log" 2>&1
done
echo "DEMO_RECORD_DONE $(date -u)"
find "$A/outputs" -path "*demo-single-*" -name "*.mp4" | sort
