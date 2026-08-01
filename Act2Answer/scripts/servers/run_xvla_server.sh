#!/usr/bin/env bash
# Launch X-VLA's FastAPI inference server (deploy.py), which the Act2Answer xvla
# client talks to over POST /act.
#
#   XVLA_REPO=<clone of 2toinf/X-VLA> XVLA_CKPT=<checkpoint dir> \
#     GPU=0 PORT=8010 bash run_xvla_server.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../env.sh"

LOG=${A2A_LOG_DIR}/xvla_server.log
REPO=${XVLA_REPO:?set XVLA_REPO to the X-VLA checkout}
CKPT=${XVLA_CKPT:?set XVLA_CKPT to the model checkpoint}
PORT=${PORT:-8010}
GPU=${GPU:-0}

exec > >(tee -a "$LOG") 2>&1

echo "START_XVLA_SERVER $(date -u) gpu=$GPU port=$PORT ckpt=$CKPT"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENVS_DIR}/xvla"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
cd "$REPO"

CUDA_VISIBLE_DEVICES="$GPU" exec python deploy.py \
  --model_path "$CKPT" \
  ${XVLA_PROCESSOR_PATH:+--processor_path "$XVLA_PROCESSOR_PATH"} \
  ${XVLA_LORA_PATH:+--LoRA_path "$XVLA_LORA_PATH"} \
  --host 0.0.0.0 --port "$PORT" --disable_slurm
