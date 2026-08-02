#!/usr/bin/env bash
# H100: GR00T N1.7 confirm, диапазон [START, START+COUNT), layout = $1 (noswap|swap).
# По образцу xiaomi/xvla: каждый 100-блок отдельным процессом, уникальные -s имена,
# skip готовых блоков (резюмируемо). Сервер: gr00t_server_h100.sh (:5555, sim-wrapper).
set -uo pipefail
R="/workspace/moskalenko/bias-vla-benchmark-main"
A="$R/Act2Answer"
LAYOUT="${1:?usage: launch_gr00t_confirm_full_h100.sh noswap|swap}"
ASSET="${ASSET:-pairs_choice_vla_confirm}"
START="${START:-0}"
COUNT="${COUNT:-1600}"
BLOCK=100
EPLEN="${EPLEN:-80}"
INFERBATCH=10
GPU_A2A="${GPU_A2A:-0}"
NAMEPFX="${NAMEPFX:-confirm-gr00t}"
EVAL_ENV="${EVAL_ENV:-magma_act2answer}"
LOGDIR="$R/cropped_run/logs"
mkdir -p "$LOGDIR"

source "$HOME/conda/etc/profile.d/conda.sh"
conda activate "$HOME/conda/envs/$EVAL_ENV"
export REPO_ROOT="$A"
export PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
export TOKENIZERS_PARALLELISM=false XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="$GPU_A2A"
export GR00T_HOST="${GR00T_HOST:-127.0.0.1}" GR00T_PORT="${GR00T_PORT:-5555}"
cd "$A/SimplerEnv"

extra=""
[ "$LAYOUT" = "swap" ] && extra="--do-swap"
name="${NAMEPFX}-${LAYOUT}"
logf="$LOGDIR/${name}.log"
echo "[$(date -u +%H:%M:%S)] START $name (start=$START count=$COUNT по $BLOCK) -> $logf"
for ((s=START; s<START+COUNT; s+=BLOCK)); do
  if [ -f "$A/outputs/${name}-s${s}/glob/vis_0_test/stats.yaml" ]; then
    echo "[$(date -u +%H:%M:%S)] BLOCK SKIP $name s$s (готов)"; continue
  fi
  echo "[$(date -u +%H:%M:%S)] BLOCK $name s$s" >> "$logf"
  python3 -u -m simpler_env.eval \
    --vla gr00t \
    --start-id "$s" --count "$BLOCK" \
    --assets "$ASSET" --obj-set test --episode-len "$EPLEN" \
    --buffer-inferbatch "$INFERBATCH" --buffer-minibatch -1 \
    --shard-size "$BLOCK" --name "${name}-s${s}" $extra < /dev/null \
    >>"$logf" 2>&1
  echo "[$(date -u +%H:%M:%S)] BLOCK DONE $name s$s rc=$?"
done
echo "[$(date -u +%H:%M:%S)] CONFIRM GR00T $LAYOUT DONE ($NAMEPFX)"
