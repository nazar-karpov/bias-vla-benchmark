#!/usr/bin/env bash
# Подготовка X-VLA на ноде h100: клон + env xvla + ckpt. CPU/сеть only.
set -uo pipefail
R="/workspace/moskalenko/bias-vla-benchmark-main"
XREPO="$R/X-VLA"
CONDA_ROOT="$HOME/conda"

echo "[$(date -u +%H:%M:%S)] STEP1 clone"
[ -d "$XREPO" ] || git clone --depth 1 https://github.com/2toinf/X-VLA.git "$XREPO" || exit 1

echo "[$(date -u +%H:%M:%S)] STEP2 env xvla"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
if [ ! -d "$CONDA_ROOT/envs/xvla" ]; then
  conda create --solver libmamba --override-channels -c defaults -p "$CONDA_ROOT/envs/xvla" -y python=3.10 pip || exit 1
fi
conda activate "$CONDA_ROOT/envs/xvla"
export PYTHONNOUSERSITE=1
pip install --upgrade pip wheel setuptools
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r "$XREPO/requirements.txt"
python - <<'PY'
import torch, transformers, fastapi, json_numpy
print("xvla_env_ok", torch.__version__, transformers.__version__, "cuda", torch.cuda.is_available())
PY

echo "[$(date -u +%H:%M:%S)] STEP3 ckpt"
export HF_HUB_ENABLE_HF_TRANSFER=0
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="2toINF/X-VLA-WidowX")
print("XVLA_CKPT_DONE", p)
PY
echo "[$(date -u +%H:%M:%S)] PREP_XVLA_DONE"
