#!/usr/bin/env bash
# Launch the RLDX-1 zmq policy server (PolicyServer, REQ/REP + msgpack) that the
# Act2Answer rldx client policy connects to. Arg: <port>. Model path + embodiment
# come from setup_rldx.sh's output file.
set -uo pipefail
PORT="${1:-20000}"
REPO="${RLDX_REPO:-$HOME/bias_benchmark/nazar_folder/RLDX-1}"
export PATH="$HOME/.local/bin:$PATH"
cd "$REPO"

CKPT="$(cat "$REPO/.rldx_ckpt_path.txt" 2>/dev/null)"
[ -n "$CKPT" ] || { echo "no rldx ckpt path (run setup_rldx.sh first)"; exit 1; }

# Attention backend: flash-attn only builds/runs on Ampere+ (sm_80+). On Volta
# (V100, sm_70) the load succeeds but the first forward dies with
# "FlashAttention only supports Ampere GPUs or newer". Auto-fall-back to sdpa
# when the visible GPU's compute capability is < 8.0, unless RLDX_ATTN_IMPL is
# already set by the caller. Ampere+ boxes (H100) keep flash-attn for throughput.
# Nodes without a CUDA toolkit cannot build flash-attn at all, so it may simply be
# absent from the venv regardless of how new the GPU is -- check before the sm check.
if [ -z "${RLDX_ATTN_IMPL:-}" ]; then
  if ! uv run --no-sync python -c 'import flash_attn' >/dev/null 2>&1; then
    export RLDX_ATTN_IMPL=sdpa
    echo "RLDX_ATTN_IMPL=sdpa (flash-attn not installed)"
  fi
fi
if [ -z "${RLDX_ATTN_IMPL:-}" ]; then
  CC_MAJOR="$(uv run --no-sync python -c 'import torch;print(torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0)' 2>/dev/null || echo 0)"
  if [ "${CC_MAJOR:-0}" -lt 8 ] 2>/dev/null; then
    export RLDX_ATTN_IMPL=sdpa
    echo "RLDX_ATTN_IMPL=sdpa (GPU compute capability ${CC_MAJOR}.x < 8.0, flash-attn unsupported)"
  fi
fi

# widowx_bridge -> OXE_BRIDGE_ORIG (the released SIMPLER-WidowX default).
exec uv run --no-sync python rldx/eval/run_rldx_server.py \
  --model-path "$CKPT" \
  --embodiment-tag OXE_BRIDGE_ORIG \
  --host 0.0.0.0 --port "$PORT"
