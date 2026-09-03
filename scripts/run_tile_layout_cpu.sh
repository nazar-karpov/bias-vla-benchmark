#!/usr/bin/env bash
# Проба расстановки плиток на CPU-ноде (без GPU): physx_cpu + SwiftShader
# (программный Vulkan из Chrome: /workspace/moskalenko/swiftshader). lavapipe
# не подходит — SAPIEN требует VK_KHR_external_semaphore_fd.
#   SCALE=1.0 LAYOUTS="0.155,0;0.13,0" OUT=.../tile_layout bash run_tile_layout_cpu.sh
# SCALE = BOARD_XY_SCALE (model_db уже 1.3 → эффективный масштаб 1.3×SCALE).
set -uo pipefail
A=/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer
source /workspace/moskalenko/conda/etc/profile.d/conda.sh
conda activate magma_act2answer
export REPO_ROOT="$A"
export PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
export MS_ASSET_DIR=/workspace/moskalenko/maniskill_assets
export LD_LIBRARY_PATH=/workspace/moskalenko/mesa_vk/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export VK_ICD_FILENAMES=/workspace/moskalenko/swiftshader/vk_swiftshader_icd.json
export VK_DRIVER_FILES=/workspace/moskalenko/swiftshader/vk_swiftshader_icd.json
export BOARD_XY_SCALE="${SCALE:-1.0}"
export CUDA_VISIBLE_DEVICES=""
IDS="${IDS:-0}"
SWAPFLAG=""; [ "${SWAP:-0}" = "1" ] && SWAPFLAG="--swap"
cd "$A/SimplerEnv" || exit 1
python -u "$A/scripts/tile_layout_probe.py" --cpu --assets "${ASSETS:-pairs_choice_vla_confirm}" \
  --ids $IDS "--layouts=${LAYOUTS}" --out-dir "${OUT}" $SWAPFLAG < /dev/null
echo "TILE_LAYOUT_DONE rc=$?"
