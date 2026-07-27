#!/usr/bin/env bash
# Ночной прогон Act2Answer: 2 модели x 2 датасета x 2 прохода (noswap/swap).
# Запускается ВНУТРИ conda-env соответствующей модели. Модель и GPU передаются аргументами.
# Использование (обычно из tmux):
#   MODEL=magma      GPU=0 bash night_run.sh
#   MODEL=spatialvla GPU=1 bash night_run.sh
set -uo pipefail

MODEL="${MODEL:?set MODEL=magma|spatialvla}"
GPU="${GPU:?set GPU=0|1}"
COUNT="${COUNT:-55}"          # эпизодов на прогон (в диапазоне 50-60)
INFERBATCH="${INFERBATCH:-5}" # 55 % 5 == 0
EPLEN="${EPLEN:-80}"          # дефолтная длина эпизода

A="$HOME/bias_benchmark/nazar_folder/Act2Answer"
export REPO_ROOT="$A"
export PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="$GPU"

LOGDIR="$A/logs/night"
mkdir -p "$LOGDIR"
cd "$A/SimplerEnv" || exit 1

STAMP=$(date -u +%Y%m%d)
echo "NIGHT_RUN_START model=$MODEL gpu=$GPU count=$COUNT eplen=$EPLEN $(date -u)" | tee -a "$LOGDIR/_progress_${MODEL}.log"

run_one() {
  local asset="$1" swap="$2"
  local extra=""; [ "$swap" = swap ] && extra="--do-swap"
  local name="night-${MODEL}-${asset}-${swap}"
  local log="$LOGDIR/${name}.log"
  echo ">>> START $name $(date -u)" | tee -a "$LOGDIR/_progress_${MODEL}.log"
  python3 -u -m simpler_env.eval \
    --vla "$MODEL" --start-id 0 --count "$COUNT" \
    --assets "$asset" --obj-set test \
    --episode-len "$EPLEN" \
    --buffer-inferbatch "$INFERBATCH" --buffer-minibatch -1 \
    --name "$name" $extra > "$log" 2>&1
  local rc=$?
  local final=$(grep -a "FINAL_STATS" "$log" | tail -1)
  echo "<<< DONE  $name rc=$rc $final $(date -u)" | tee -a "$LOGDIR/_progress_${MODEL}.log"
}

for asset in safeeditbench pairs_bias; do
  for swap in noswap swap; do
    run_one "$asset" "$swap"
  done
done

echo "NIGHT_RUN_ALL_DONE model=$MODEL $(date -u)" | tee -a "$LOGDIR/_progress_${MODEL}.log"
