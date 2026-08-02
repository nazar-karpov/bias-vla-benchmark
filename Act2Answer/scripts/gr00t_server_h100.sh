#!/usr/bin/env bash
# GR00T N1.7 policy-сервер (Isaac-GR00T, zmq :5555), embodiment simpler_env_widowx.
set -uo pipefail
G="/workspace/moskalenko/bias-vla-benchmark-main/Isaac-GR00T"
SNAP=$(ls -d ~/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-SimplerEnv-Bridge/snapshots/* | head -1)
PORT="${1:-5555}"
source ~/conda/etc/profile.d/conda.sh
conda activate ~/conda/envs/gr00t
export CUDA_VISIBLE_DEVICES="${GR00T_GPU:-0}" TOKENIZERS_PARALLELISM=false
cd "$G"
# --use-sim-policy-wrapper: Gr00tSimPolicyWrapper принимает ПЛОСКИЕ obs
# ('video.image_0', 'state.x', ...) — под них написан наш клиент gr00t.py.
exec python gr00t/eval/run_gr00t_server.py --model-path "$SNAP" --embodiment-tag simpler_env_widowx --port "$PORT" --host 0.0.0.0 --use-sim-policy-wrapper
