#!/usr/bin/env bash
# H100: полный confirm InternVLA, диапазон [START, START+COUNT), noswap+swap.
# Клиент (magma_act2answer env) к УЖЕ ПОДНЯТОМУ серверу на INTERNVLA_PORT (:10093).
set -uo pipefail
R="/workspace/moskalenko/bias-vla-benchmark-main"
A="$R/Act2Answer"
ASSET="${ASSET:-pairs_choice_vla_confirm}"
START="${START:-0}"
COUNT="${COUNT:-1600}"
SHARD="${SHARD:-100}"
EPLEN="${EPLEN:-80}"
INFERBATCH="${INFERBATCH:-8}"
GPU_A2A="${GPU_A2A:-0}"
PORT="${INTERNVLA_PORT:-10093}"
NAMEPFX="${NAMEPFX:-confirm-internvla-full-h100}"
EVAL_ENV="${EVAL_ENV:-magma_act2answer}"
LOGDIR="$R/cropped_run/logs"
mkdir -p "$LOGDIR"

source "$HOME/conda/etc/profile.d/conda.sh"
conda activate "$HOME/conda/envs/$EVAL_ENV"
export REPO_ROOT="$A"
export PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
export TOKENIZERS_PARALLELISM=false XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="$GPU_A2A"
export INTERNVLA_PORT="$PORT" INTERNVLA_HOST=127.0.0.1
export INTERNVLA_CKPT="$(cat "$R/internvla_ckpt/ckpt_path.txt")"
cd "$A/SimplerEnv"

run_layout() {
  local swap="$1" extra="$2"
  local name="${NAMEPFX}-${swap}"
  local logf="$LOGDIR/${name}.log"
  echo "[$(date -u +%H:%M:%S)] START $name (start=$START count=$COUNT port=$PORT gpu=$GPU_A2A) -> $logf"
  setsid python3 -u -m simpler_env.eval \
    --vla internvla --vla-path "$INTERNVLA_CKPT" \
    --start-id "$START" --count "$COUNT" \
    --assets "$ASSET" --obj-set test --episode-len "$EPLEN" \
    --buffer-inferbatch "$INFERBATCH" --buffer-minibatch -1 \
    --shard-size "$SHARD" --name "$name" $extra < /dev/null \
    >"$logf" 2>&1
  echo "[$(date -u +%H:%M:%S)] DONE  $name rc=$?"
}

run_layout noswap ""
run_layout swap   "--do-swap"
echo "[$(date -u +%H:%M:%S)] CONFIRM FULL DONE ($NAMEPFX)"
