#!/usr/bin/env bash
# H100 launcher for the RLDX-1 zmq policy server (:20000 by default).
# uv-based .venv in the RLDX-1 repo; H100 keeps flash-attn (Ampere+).
set -uo pipefail
R="/workspace/moskalenko/bias-vla-benchmark-main"
PORT="${1:-20000}"
GPU="${RLDX_GPU:-0}"
REPO="$R/RLDX-1"
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$REPO"
CKPT="$(cat "$REPO/.rldx_ckpt_path.txt" 2>/dev/null)"
[ -n "$CKPT" ] || { echo "no rldx ckpt path"; exit 1; }
exec uv run --no-sync python rldx/eval/run_rldx_server.py \
  --model-path "$CKPT" \
  --embodiment-tag OXE_BRIDGE_ORIG \
  --host 0.0.0.0 --port "$PORT"
