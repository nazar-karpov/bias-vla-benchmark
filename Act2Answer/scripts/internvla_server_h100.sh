#!/usr/bin/env bash
# H100 launcher for the InternVLA-M1 websocket policy server (:10093 by default).
# conda at ~/conda; env 'internvla'; repo + ckpt under /workspace/moskalenko.
set -euo pipefail
R="/workspace/moskalenko/bias-vla-benchmark-main"
CKPT="${1:-$(cat "$R/internvla_ckpt/ckpt_path.txt")}"
PORT="${2:-10093}"
GPU="${INTERNVLA_GPU:-0}"
REPO="$R/InternVLA-M1"

source "$HOME/conda/etc/profile.d/conda.sh"
conda activate "$HOME/conda/envs/internvla"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$REPO"
exec python deployment/model_server/server_policy_M1.py \
  --ckpt_path "$CKPT" --port "$PORT" --use_bf16
