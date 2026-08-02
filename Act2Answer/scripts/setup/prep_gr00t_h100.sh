#!/usr/bin/env bash
# Подготовка GR00T N1.7 на h100: клон Isaac-GR00T + env gr00t + ckpt. CPU/сеть.
set -uo pipefail
R="/workspace/moskalenko/bias-vla-benchmark-main"
G="$R/Isaac-GR00T"
CONDA_ROOT="$HOME/conda"

echo "[$(date -u +%H:%M:%S)] STEP1 clone"
[ -d "$G" ] || git clone --depth 1 https://github.com/NVIDIA/Isaac-GR00T.git "$G" || exit 1
ls "$G" | head -20

echo "[$(date -u +%H:%M:%S)] STEP2 env gr00t"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
if [ ! -d "$CONDA_ROOT/envs/gr00t" ]; then
  conda create --solver libmamba --override-channels -c defaults -p "$CONDA_ROOT/envs/gr00t" -y python=3.10 pip || exit 1
fi
conda activate "$CONDA_ROOT/envs/gr00t"
export PYTHONNOUSERSITE=1
pip install --upgrade pip wheel setuptools
# ставим пакет без flash-attn (доставим prebuilt отдельно, как для остальных)
pip install -e "$G" --no-build-isolation 2>&1 | tail -5 || pip install -e "$G" 2>&1 | tail -5
python - <<'PY'
try:
    import gr00t
    print("gr00t import OK", getattr(gr00t, "__version__", "?"))
except Exception as e:
    print("gr00t import FAIL:", repr(e)[:200])
PY

echo "[$(date -u +%H:%M:%S)] STEP3 ckpt"
export HF_HUB_ENABLE_HF_TRANSFER=0
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="nvidia/GR00T-N1.7-SimplerEnv-Bridge")
print("GR00T_CKPT_DONE", p)
PY
echo "[$(date -u +%H:%M:%S)] PREP_GR00T_DONE"
