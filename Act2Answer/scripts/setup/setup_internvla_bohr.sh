#!/usr/bin/env bash
# InternVLA-M1 policy-сервер на Bohr (2×4090, без sudo, conda create по defaults упирается в ToS).
# Вместо conda — uv venv (uv лежит в env magma_act2answer; тот же приём, что для Isaac-GR00T/.venv).
# Состав как у облачного env internvla (setup_internvla_env.sh): torch 2.6.0+cu124, требования сервера
# и репо InternVLA-M1, плюс rich (без него падает логгер overwatch) и готовый wheel flash-attn.
#
#   setsid nohup bash Act2Answer/scripts/setup/setup_internvla_bohr.sh > ~/ws/_internvla_bohr_setup.log 2>&1 &
set -euo pipefail
R=$HOME/ws/bias-vla-benchmark-main
A=$R/Act2Answer
IR=$R/InternVLA-M1
V=${INTERNVLA_VENV:-$HOME/ws/venvs/internvla}
UV=$HOME/ws/conda/envs/magma_act2answer/bin/uv
FA_WHL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
export UV_CACHE_DIR=$HOME/ws/.uv_cache UV_HTTP_TIMEOUT=600
echo "START_INTERNVLA_BOHR $(date -u)"
[ -x "$V/bin/python" ] || $UV venv "$V" --python 3.10 --seed
P=$V/bin/python
$UV pip install --python $P torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
$UV pip install --python $P -r $A/requirements/internvla_server.txt
$UV pip install --python $P -r $IR/requirements.txt
$UV pip install --python $P -e $IR --no-deps
$UV pip install --python $P rich
$UV pip install --python $P --no-deps "$FA_WHL" || echo "flash-attn wheel не встал — сервер уйдёт на sdpa"
$P - <<'PY'
import torch, transformers
try:
    import flash_attn; fa = flash_attn.__version__
except Exception as e:
    fa = f"нет ({type(e).__name__})"
print("internvla_env_ok torch", torch.__version__, "tf", transformers.__version__, "cuda", torch.cuda.is_available(), "flash_attn", fa)
PY
echo "DONE_INTERNVLA_BOHR $(date -u)"
