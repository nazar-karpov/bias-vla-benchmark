#!/usr/bin/env bash
# Launch the Xiaomi-Robotics-0 policy server (env mibot) on H100.
# server.py imports only torch + transformers (AutoModel trust_remote_code); the
# heavy mibot/lightning/mmengine training stack is NOT needed for inference.
set -uo pipefail
source ~/conda/etc/profile.d/conda.sh
conda activate mibot

export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

XR=/home/user/bias_benchmark/roman_folder/bias-vla-benchmark-main/Xiaomi-Robotics-0
MODEL=${MODEL:-XiaomiRobotics/Xiaomi-Robotics-0-SimplerEnv-WidowX}
PORT=${PORT:-10086}

cd "$XR"
exec python deploy/server.py --model "$MODEL" --host localhost --port "$PORT"
