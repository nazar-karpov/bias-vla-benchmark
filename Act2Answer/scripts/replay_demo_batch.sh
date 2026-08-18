#!/usr/bin/env bash
# Пакетная запись демо-видео из сохранённых траекторий (без прогона модели).
#   EPS="261 40 359 31" bash replay_demo_batch.sh
set -uo pipefail
R=/workspace/moskalenko/bias-vla-benchmark-main
A="$R/Act2Answer"
export REPO_ROOT="$A" PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
export MS_ASSET_DIR=/workspace/moskalenko/maniskill_assets
export BOARD_XY_SCALE=1.0 A2A_SINGLE_TILE=1
PY=/workspace/moskalenko/conda/envs/magma_act2answer/bin/python
EPS="${EPS:-261 40 359 31}"
ASSET="${ASSET:-pairs_single_pilot}"
RUNS="${RUNS:-../outputs/single-pilot-magma-noswap-*}"
cd "$A/SimplerEnv" || exit 1
for e in $EPS; do
  CUDA_VISIBLE_DEVICES=0 $PY ../scripts/replay_traj_video.py \
    --runs "$RUNS" --ep "$e" --assets "$ASSET" \
    --pairs "../ManiSkill/mani_skill/assets/carrot/$ASSET/pairs.json" \
    --out-dir ../outputs/demo_videos 2>&1 | grep -E "эпизод|карточка|->"
done
echo "BATCH_DONE"
