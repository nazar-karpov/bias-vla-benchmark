#!/usr/bin/env bash
set -uo pipefail
B=$HOME/bias_benchmark
source $B/miniconda3/etc/profile.d/conda.sh
conda activate $B/miniconda3/envs/internvla
export PYTHONNOUSERSITE=1
R=$B/nazar_folder/Act2Answer
REPO=$B/nazar_folder/InternVLA-M1
echo RESUME_START $(date -u)
pip install --upgrade pip wheel setuptools ninja packaging
# torch 2.6.0 not on cu121 index -> use cu124 (server driver is CUDA 13, works)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r $R/requirements/internvla_server.txt
pip install -r $REPO/requirements.txt
pip install -e $REPO --no-deps
pip install 'setuptools<81'
python - <<'PY'
import torch, transformers
print('internvla_env_ok', torch.__version__, transformers.__version__, 'cuda', torch.cuda.is_available())
PY
echo RESUME_DONE $(date -u)
