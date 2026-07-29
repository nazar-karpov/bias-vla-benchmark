#!/usr/bin/env bash
# Launch the RLDX-1 zmq policy server (PolicyServer, REQ/REP + msgpack) that the
# Act2Answer rldx client policy connects to. Arg: <port>. Model path + embodiment
# come from setup_rldx.sh's output file.
set -uo pipefail
PORT="${1:-20000}"
REPO="$HOME/bias_benchmark/nazar_folder/RLDX-1"
export PATH="$HOME/.local/bin:$PATH"
cd "$REPO"

CKPT="$(cat "$REPO/.rldx_ckpt_path.txt" 2>/dev/null)"
[ -n "$CKPT" ] || { echo "no rldx ckpt path (run setup_rldx.sh first)"; exit 1; }

# widowx_bridge -> OXE_BRIDGE_ORIG (the released SIMPLER-WidowX default).
exec uv run python rldx/eval/run_rldx_server.py \
  --model-path "$CKPT" \
  --embodiment-tag OXE_BRIDGE_ORIG \
  --host 0.0.0.0 --port "$PORT"
