#!/usr/bin/env bash
# X-VLA policy-сервер (FastAPI :8010), env xvla, ckpt 2toINF/X-VLA-WidowX из HF-кэша.
set -uo pipefail
R="/workspace/moskalenko/bias-vla-benchmark-main"
SNAP=$(ls -d ~/.cache/huggingface/hub/models--2toINF--X-VLA-WidowX/snapshots/* | head -1)
source ~/conda/etc/profile.d/conda.sh
conda activate ~/conda/envs/xvla
export CUDA_VISIBLE_DEVICES="${XVLA_GPU:-0}"
cd "$R/X-VLA"
rm -f ./logs/info.json  # стейл-маркер прошлого сервера блокирует старт
exec python deploy.py --model_path "$SNAP" --port "${1:-8010}" --disable_slurm
