#!/usr/bin/env bash
# V100: полный confirm RLDX, диапазон [START, START+COUNT), noswap+swap.
# Клиент (spatialvla_act2answer env) к УЖЕ ПОДНЯТОМУ RLDX-серверу на RLDX_PORT.
set -uo pipefail
BIAS="$HOME/bias_benchmark"
A="$BIAS/nazar_folder/Act2Answer"
CONDA_ROOT="$BIAS/miniconda3"
ASSET="${ASSET:-pairs_choice_vla_confirm}"
START="${START:-0}"
COUNT="${COUNT:-1600}"
SHARD="${SHARD:-100}"
EPLEN="${EPLEN:-80}"
INFERBATCH="${INFERBATCH:-8}"
GPU_A2A="${GPU_A2A:-1}"
NAMEPFX="${NAMEPFX:-confirm-rldx-full-v100}"
EVAL_ENV="${EVAL_ENV:-spatialvla_act2answer}"
LOGDIR="$BIAS/nazar_folder/cropped_run/logs"
mkdir -p "$LOGDIR"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ROOT/envs/$EVAL_ENV"
export REPO_ROOT="$A"
export PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
export TOKENIZERS_PARALLELISM=false XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="$GPU_A2A"
export RLDX_HOST="${RLDX_HOST:-127.0.0.1}" RLDX_PORT="${RLDX_PORT:-20000}"
cd "$A/SimplerEnv"

run_layout() {
  local swap="$1" extra="$2"
  local name="${NAMEPFX}-${swap}"
  local logf="$LOGDIR/${name}.log"
  echo "[$(date -u +%H:%M:%S)] START $name (start=$START count=$COUNT rldx_port=$RLDX_PORT gpu=$GPU_A2A) -> $logf"
  setsid python3 -u -m simpler_env.eval \
    --vla rldx \
    --start-id "$START" --count "$COUNT" \
    --assets "$ASSET" --obj-set test --episode-len "$EPLEN" \
    --buffer-inferbatch "$INFERBATCH" --buffer-minibatch -1 \
    --shard-size "$SHARD" --name "$name" $extra < /dev/null \
    >"$logf" 2>&1
  echo "[$(date -u +%H:%M:%S)] DONE  $name rc=$?"
}

run_layout noswap ""
run_layout swap   "--do-swap"
echo "[$(date -u +%H:%M:%S)] CONFIRM RLDX FULL DONE ($NAMEPFX)"
