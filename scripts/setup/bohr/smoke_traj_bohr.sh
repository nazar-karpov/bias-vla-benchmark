#!/usr/bin/env bash
# Smoke: 4 эпизода без VLA-политики (случайные действия) — проверить, что
# traj.npz пишется симулятором и читается integral_pull.py.
set -uo pipefail
A=/home/moskalenko/ws/bias-vla-benchmark-main/Act2Answer
source /home/moskalenko/ws/conda/etc/profile.d/conda.sh
conda activate magma_act2answer
export REPO_ROOT="$A"
export PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export BOARD_XY_SCALE=1.0
export A2A_TRAJ_LOG=1
# Ассеты ManiSkill (bridge-сцена) — НА /workspace: дефолт ~/.maniskill умирает
# вместе с нодой. Все прогоны обязаны экспортировать эту переменную.
export MS_ASSET_DIR=/home/moskalenko/ws/maniskill_assets
cd "$A/SimplerEnv" || exit 1
python -u /home/moskalenko/ws/setup/smoke_traj_bohr.py < /dev/null
echo "SMOKE_DONE rc=$?"
