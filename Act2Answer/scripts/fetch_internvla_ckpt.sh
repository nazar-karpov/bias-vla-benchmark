#!/usr/bin/env bash
# Download the InternVLA-M1 pretrained RT-1/Bridge checkpoint from HuggingFace
# into the layout server_policy_M1.py + read_mode_config expect:
#   <run_dir>/checkpoints/steps_50000_pytorch_model.pt   (+ config.json, dataset_statistics.json)
# Writes the resolved .pt path to <dest>/ckpt_path.txt.
# Idempotent: skips download if the .pt already exists.
set -euo pipefail

DEST="${1:?usage: fetch_internvla_ckpt.sh <dest_dir>}"
HF_REPO="InternRobotics/InternVLA-M1-Pretrain-RT-1-Bridge"
RUN_DIR="$DEST/InternVLA-M1-Pretrain-RT-1-Bridge"
mkdir -p "$RUN_DIR"

# find an existing .pt (any steps_*.pt) first
existing="$(find "$RUN_DIR" -name '*.pt' | head -1 || true)"
if [ -n "$existing" ]; then
  echo "ckpt already present: $existing"
  echo "$existing" > "$DEST/ckpt_path.txt"
  exit 0
fi

# use whatever huggingface CLI / python is available in the internvla env
CONDA_ROOT="${CONDA_ROOT:-$HOME/bias_benchmark/miniconda3}"
CONDA_ENVS_DIR="${CONDA_ENVS_DIR:-$CONDA_ROOT/envs}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENVS_DIR/internvla"

export HF_HUB_ENABLE_HF_TRANSFER=0
python - "$HF_REPO" "$RUN_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dst = sys.argv[1], sys.argv[2]
p = snapshot_download(repo_id=repo, local_dir=dst, local_dir_use_symlinks=False)
print("downloaded to", p)
PY

pt="$(find "$RUN_DIR" -name '*.pt' | head -1)"
if [ -z "$pt" ]; then
  echo "ERROR: no .pt found under $RUN_DIR after download"; find "$RUN_DIR" -maxdepth 3 | head -40; exit 1
fi
# ensure config.json + dataset_statistics.json sit where read_mode_config looks
# (run_dir = <pt>/../.. ; it reads <run_dir>/config.json + dataset_statistics.json)
echo "ckpt: $pt"
echo "$pt" > "$DEST/ckpt_path.txt"
