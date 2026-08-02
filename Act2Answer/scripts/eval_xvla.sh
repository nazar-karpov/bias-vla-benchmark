#!/usr/bin/env bash
# Act2Answer eval for X-VLA (2toinf/X-VLA). Like xiaomi/rldx, the model runs in its
# own process (X-VLA's FastAPI deploy.py, see scripts/servers/run_xvla_server.sh)
# and this drives the thin client from the shared eval env.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"

ASSETS=${ASSETS:-pairs}
COUNT=${COUNT:-50}
START_ID=${START_ID:-0}
EVAL_GPU=${EVAL_GPU:-3}
BUFFER_INFERBATCH=${BUFFER_INFERBATCH:-$COUNT}
XVLA_PORT=${XVLA_PORT:-8010}
LOG=${A2A_LOG_DIR}/xvla_${ASSETS}_eval.log

conda activate "${CONDA_ENVS_DIR}/act2ans"
export PYTHONPATH="${REPO_ROOT}/SimplerEnv:${REPO_ROOT}/ManiSkill:${PYTHONPATH:-}"
export XVLA_HOST="${XVLA_HOST:-127.0.0.1}"
export XVLA_PORT

portopen() {
  python3 - "$1" <<'PY' >/dev/null 2>&1
import socket, sys
s = socket.socket()
s.settimeout(3)
s.connect(("localhost", int(sys.argv[1])))
s.close()
PY
}

: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "START_XVLA_EVAL $(date -u) assets=$ASSETS count=$COUNT gpu=$EVAL_GPU port=$XVLA_PORT"
until portopen "$XVLA_PORT"; do
  echo "WAIT_XVLA_SERVER_${XVLA_PORT}"
  sleep 10
done
for swap_arg in noswap swap; do
  extra=()
  [ "$swap_arg" = swap ] && extra=(--do-swap)
  echo "RUN_XVLA ${swap_arg} $(date -u)"
  CUDA_VISIBLE_DEVICES=$EVAL_GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python3 -u -m simpler_env.eval \
      --vla xvla --start-id "$START_ID" --count "$COUNT" --assets "$ASSETS" \
      --obj-set "${OBJ_SET:-test}" --buffer-inferbatch "$BUFFER_INFERBATCH" \
      ${SHARD_SIZE:+--shard-size $SHARD_SIZE} "${extra[@]}"
done
echo "DONE_XVLA_EVAL $(date -u)"
