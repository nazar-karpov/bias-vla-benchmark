#!/usr/bin/env bash
# Полная подготовка Xiaomi на ноде h100: клон + env mibot + патч + ckpt.
# CPU/сеть only — безопасно параллельно GPU-прогонам.
set -uo pipefail
R="/workspace/moskalenko/bias-vla-benchmark-main"
export CONDA_ROOT="$HOME/conda"
export CONDA_ENVS_DIR="$HOME/conda/envs"
export A2A_EXTERNAL_DIR="$R"
export XIAOMI_REPO="$R/Xiaomi-Robotics-0"

echo "[$(date -u +%H:%M:%S)] STEP1 clone"
if [ ! -d "$XIAOMI_REPO" ]; then
  git clone --depth 1 https://github.com/XiaomiRobotics/Xiaomi-Robotics-0.git "$XIAOMI_REPO" || exit 1
fi

echo "[$(date -u +%H:%M:%S)] STEP2 патч XR0_ATTN"
cd "$XIAOMI_REPO" && git apply --check "$R/Act2Answer/patches/xiaomi-robotics-0-attn.patch" 2>/dev/null \
  && git apply "$R/Act2Answer/patches/xiaomi-robotics-0-attn.patch" && echo "патч применён" \
  || echo "патч уже применён или не лёг (проверить!)"

echo "[$(date -u +%H:%M:%S)] STEP3 env mibot"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
if [ ! -d "$CONDA_ENVS_DIR/mibot" ]; then
  conda create --solver libmamba --override-channels -c defaults -p "$CONDA_ENVS_DIR/mibot" -y python=3.12.2 pip || exit 1
fi
conda activate "$CONDA_ENVS_DIR/mibot"
export PYTHONNOUSERSITE=1
pip install --upgrade pip wheel setuptools ninja packaging
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r "$R/Act2Answer/requirements/xiaomi_server.txt"
python - <<'PY'
import torch, transformers
print("mibot_env_ok", torch.__version__, transformers.__version__, "cuda", torch.cuda.is_available())
PY

echo "[$(date -u +%H:%M:%S)] STEP4 ckpt"
export HF_HUB_ENABLE_HF_TRANSFER=0
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="XiaomiRobotics/Xiaomi-Robotics-0-SimplerEnv-WidowX")
print("XIAOMI_CKPT_DONE", p)
PY
echo "[$(date -u +%H:%M:%S)] PREP_XIAOMI_DONE"
