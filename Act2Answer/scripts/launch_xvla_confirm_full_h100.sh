#!/usr/bin/env bash
# H100: Xiaomi confirm, диапазон [START, START+COUNT), noswap+swap.
# ВАЖНО: xiaomi-политика кэширует инструкцию (assert) — поэтому КАЖДЫЙ 100-блок
# гоняем ОТДЕЛЬНЫМ процессом (один вопрос на процесс), inferbatch=10 (100%10=0).
set -uo pipefail
R="/workspace/moskalenko/bias-vla-benchmark-main"
A="$R/Act2Answer"
ASSET="${ASSET:-pairs_choice_vla_confirm}"
START="${START:-0}"
COUNT="${COUNT:-1600}"
BLOCK=100
EPLEN="${EPLEN:-80}"
INFERBATCH=10
GPU_A2A="${GPU_A2A:-0}"
NAMEPFX="${NAMEPFX:-confirm-xvla}"
EVAL_ENV="${EVAL_ENV:-magma_act2answer}"
LOGDIR="$R/cropped_run/logs"
mkdir -p "$LOGDIR"

source "$HOME/conda/etc/profile.d/conda.sh"
conda activate "$HOME/conda/envs/$EVAL_ENV"
export REPO_ROOT="$A"
export PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
export TOKENIZERS_PARALLELISM=false XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="$GPU_A2A"
export XVLA_HOST=127.0.0.1 XVLA_PORT=8010
cd "$A/SimplerEnv"

run_layout() {
  local swap="$1" extra="$2"
  local name="${NAMEPFX}-${swap}"
  local logf="$LOGDIR/${name}.log"
  echo "[$(date -u +%H:%M:%S)] START $name (start=$START count=$COUNT по $BLOCK) -> $logf"
  local s
  for ((s=START; s<START+COUNT; s+=BLOCK)); do
    if [ -f "$A/outputs/${name}-s${s}/glob/vis_0_test/stats.yaml" ]; then
      echo "[$(date -u +%H:%M:%S)] BLOCK SKIP $name s$s (готов)"; continue
    fi
    echo "[$(date -u +%H:%M:%S)] BLOCK $name s$s" >> "$logf"
    python3 -u -m simpler_env.eval \
      --vla xvla \
      --start-id "$s" --count "$BLOCK" \
      --assets "$ASSET" --obj-set test --episode-len "$EPLEN" \
      --buffer-inferbatch "$INFERBATCH" --buffer-minibatch -1 \
      --shard-size "$BLOCK" --name "${name}-s${s}" $extra < /dev/null \
      >>"$logf" 2>&1
    echo "[$(date -u +%H:%M:%S)] BLOCK DONE $name s$s rc=$?"
  done
  echo "[$(date -u +%H:%M:%S)] DONE $name"
}

run_layout noswap ""
run_layout swap   "--do-swap"
echo "[$(date -u +%H:%M:%S)] CONFIRM XIAOMI FULL DONE ($NAMEPFX)"
