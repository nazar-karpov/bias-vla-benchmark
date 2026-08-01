#!/usr/bin/env bash
# Launch the lerobot pi05 policy server (env lerobot_pi05) on H100.
set -uo pipefail
source ~/conda/etc/profile.d/conda.sh
conda activate lerobot_pi05

export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PI05_CKPT=${PI05_CKPT:-qownscks/pi05_widowx}
export PI05_PORT=${PI05_PORT:-20005}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

R=/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer
exec python -u "$R/scripts/pi05_server.py" --ckpt "$PI05_CKPT" --port "$PI05_PORT"
