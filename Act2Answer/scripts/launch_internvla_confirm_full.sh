#!/usr/bin/env bash
# Полный confirm InternVLA, диапазон эпизодов [START, START+COUNT) на одной карте,
# noswap+swap. Клиент к УЖЕ ПОДНЯТОМУ серверу на INTERNVLA_PORT.
# Env-параметры: START COUNT GPU_A2A INTERNVLA_PORT NAMEPFX EVAL_ENV
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
GPU_A2A="${GPU_A2A:-0}"
PORT="${INTERNVLA_PORT:-10093}"
NAMEPFX="${NAMEPFX:-confirm-internvla-full}"
EVAL_ENV="${EVAL_ENV:-spatialvla_act2answer}"
LOGDIR="$BIAS/nazar_folder/cropped_run/logs"
mkdir -p "$LOGDIR"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ROOT/envs/$EVAL_ENV"
export REPO_ROOT="$A"
export PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
export TOKENIZERS_PARALLELISM=false XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="$GPU_A2A"
export INTERNVLA_PORT="$PORT" INTERNVLA_HOST=127.0.0.1
export INTERNVLA_CKPT="$(cat "$BIAS/nazar_folder/internvla_ckpt/ckpt_path.txt")"
cd "$A/SimplerEnv"

run_layout() {
  local swap="$1" extra="$2"
  local name="${NAMEPFX}-${swap}"
  local logf="$LOGDIR/${name}.log"
  echo "[$(date -u +%H:%M:%S)] START $name (asset=$ASSET start=$START count=$COUNT port=$PORT gpu=$GPU_A2A) -> $logf"
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
