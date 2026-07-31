#!/usr/bin/env bash
# InternVLA на CONFIRM-подвыборке (pairs_choice_vla_confirm_probe40, 640 эп,
# 40/блок). shard=40 => один блок (вопрос x ось x полярность) на шард, метрики
# капают поблочно. Клиент к УЖЕ ПОДНЯТОМУ серверу :10093.
set -uo pipefail
BIAS="$HOME/bias_benchmark"
A="$BIAS/nazar_folder/Act2Answer"
CONDA_ROOT="$BIAS/miniconda3"
ASSET="${ASSET:-pairs_choice_vla_confirm_probe40}"
COUNT="${COUNT:-640}"
SHARD="${SHARD:-40}"
EPLEN="${EPLEN:-80}"
INFERBATCH="${INFERBATCH:-5}"
GPU_A2A="${GPU_A2A:-0}"
EVAL_ENV="${EVAL_ENV:-spatialvla_act2answer}"
LOGDIR="$BIAS/nazar_folder/cropped_run/logs"
mkdir -p "$LOGDIR"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ROOT/envs/$EVAL_ENV"
export REPO_ROOT="$A"
export PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="$GPU_A2A"
export INTERNVLA_CKPT="$(cat "$BIAS/nazar_folder/internvla_ckpt/ckpt_path.txt")"
cd "$A/SimplerEnv"

run_layout() {
  local swap="$1" extra="$2"
  local name="confirm-internvla-probe40-${swap}"
  local logf="$LOGDIR/${name}.log"
  echo "[$(date -u +%H:%M:%S)] START $name (asset=$ASSET shard=$SHARD) -> $logf"
  setsid python3 -u -m simpler_env.eval \
    --vla internvla --start-id 0 --count "$COUNT" \
    --assets "$ASSET" --obj-set test --episode-len "$EPLEN" \
    --buffer-inferbatch "$INFERBATCH" --buffer-minibatch -1 \
    --shard-size "$SHARD" --name "$name" $extra < /dev/null \
    >"$logf" 2>&1
  echo "[$(date -u +%H:%M:%S)] DONE  $name rc=$?"
}

run_layout noswap ""
run_layout swap   "--do-swap"
echo "[$(date -u +%H:%M:%S)] CONFIRM PROBE DONE"
