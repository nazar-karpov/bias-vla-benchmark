#!/usr/bin/env bash
# Set up a dedicated conda env for the lerobot pi0.5 (pi05) policy server.
# Server-based, mirrors the RLDX/InternVLA pattern: the heavy lerobot deps live in
# their own env; the Act2Answer eval client (magma_act2answer) talks to it over zmq.
set -uo pipefail

source ~/conda/etc/profile.d/conda.sh
ENV=lerobot_pi05
LOG=~/logs/setup_pi05.log
mkdir -p ~/logs
exec > >(tee -a "$LOG") 2>&1
echo "START_PI05_SETUP $(date -u)"

if ! conda env list | grep -q "/envs/$ENV\b"; then
  conda create -y -p ~/conda/envs/$ENV python=3.10 pip
fi
conda activate ~/conda/envs/$ENV
export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip wheel setuptools

# lerobot 0.4.4 with the pi (physical-intelligence) extra: pulls pi05 policy deps.
python -m pip install "lerobot[pi]==0.4.4"

# zmq/msgpack for our tiny policy server wire protocol.
python -m pip install pyzmq msgpack

python - <<'PY'
import torch, importlib.util
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "cc", torch.cuda.get_device_capability())
for m in ["lerobot", "zmq", "msgpack", "transformers"]:
    print(m, importlib.util.find_spec(m) is not None)
try:
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy  # noqa
    print("pi0 import OK")
except Exception as e:
    print("pi0 import path A failed:", e)
PY
echo "DONE_PI05_SETUP $(date -u)"
