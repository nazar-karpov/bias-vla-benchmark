#!/usr/bin/env bash
# Шардированный InternVLA-прогон на pairs_bias_crop для ИНКРЕМЕНТАЛЬНЫХ метрик.
# Клиент подключается к УЖЕ ПОДНЯТОМУ серверу :10093 (веса не перегружаются).
# Каждый чанк по SHARD эпизодов пишет свой stats.yaml сразу по завершении ->
# watch_bias_incremental.py читает их накопительно.
#
# Env-настройка скопирована из run_cropped_benchmark.sh::eval_layout.
set -uo pipefail
BIAS="$HOME/bias_benchmark"
A="$BIAS/nazar_folder/Act2Answer"
CONDA_ROOT="$BIAS/miniconda3"
ASSET="pairs_bias_crop"
COUNT="${COUNT:-55}"
SHARD="${SHARD:-10}"
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
# критично для internvla-клиента: иначе eval.py берёт чужой дефолт-путь и падает
export INTERNVLA_CKPT="$(cat "$BIAS/nazar_folder/internvla_ckpt/ckpt_path.txt")"
cd "$A/SimplerEnv"

run_layout() {                            # run_layout <swaptag> <extra>
  local swap="$1" extra="$2"
  local name="crop-internvla-${ASSET}-${swap}"
  local logf="$LOGDIR/${name}-sharded.log"
  echo "[$(date -u +%H:%M:%S)] START $name (shard=$SHARD) -> $logf"
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
echo "[$(date -u +%H:%M:%S)] ALL LAYOUTS DONE"
