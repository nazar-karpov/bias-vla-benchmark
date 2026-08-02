#!/usr/bin/env bash
# Launch the InternVLA-M1 websocket policy server (the process Act2Answer's
# M1Inference / WebsocketClientPolicy connects to on port 10093).
# Args: <ckpt_path.pt> <port>
set -euo pipefail
CKPT="${1:?usage: internvla_server.sh <ckpt.pt> <port>}"
PORT="${2:-10093}"
REPO="${INTERNVLA_REPO:-$HOME/bias_benchmark/nazar_folder/InternVLA-M1}"

CONDA_ROOT="${CONDA_ROOT:-$HOME/bias_benchmark/miniconda3}"
CONDA_ENVS_DIR="${CONDA_ENVS_DIR:-$CONDA_ROOT/envs}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENVS_DIR/internvla"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
cd "$REPO"
exec python deployment/model_server/server_policy_M1.py \
  --ckpt_path "$CKPT" --port "$PORT" --use_bf16
