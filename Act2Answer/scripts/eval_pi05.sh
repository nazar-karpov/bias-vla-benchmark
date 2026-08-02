#!/usr/bin/env bash
# Act2Answer eval for pi0.5 (lerobot). Like rldx/xiaomi the model runs in its own
# process (scripts/pi05_server.py, launched by scripts/pi05_server_h100.sh) and this
# drives the thin client from the shared eval env.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"

ASSETS=${ASSETS:-pairs_choice_vla_confirm}
COUNT=${COUNT:-9}
START_ID=${START_ID:-0}
EVAL_GPU=${EVAL_GPU:-3}
BUFFER_INFERBATCH=${BUFFER_INFERBATCH:-$COUNT}
PI05_PORT=${PI05_PORT:-20005}
LOG=${A2A_LOG_DIR}/pi05_${ASSETS}_eval.log

conda activate "${CONDA_ENVS_DIR}/act2ans"
export PYTHONPATH="${REPO_ROOT}/SimplerEnv:${REPO_ROOT}/ManiSkill:${PYTHONPATH:-}"
export PI05_HOST="${PI05_HOST:-127.0.0.1}"
export PI05_PORT

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

echo "START_PI05_EVAL $(date -u) assets=$ASSETS count=$COUNT gpu=$EVAL_GPU port=$PI05_PORT"
until portopen "$PI05_PORT"; do
  echo "WAIT_PI05_SERVER_${PI05_PORT}"
  sleep 10
done
for swap_arg in noswap swap; do
  extra=()
  [ "$swap_arg" = swap ] && extra=(--do-swap)
  echo "RUN_PI05 ${swap_arg} $(date -u)"
  CUDA_VISIBLE_DEVICES=$EVAL_GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python3 -u -m simpler_env.eval \
      --vla pi05 --start-id "$START_ID" --count "$COUNT" --assets "$ASSETS" \
      --obj-set "${OBJ_SET:-test}" --buffer-inferbatch "$BUFFER_INFERBATCH" \
      ${SHARD_SIZE:+--shard-size $SHARD_SIZE} "${extra[@]}"
done
echo "DONE_PI05_EVAL $(date -u)"
