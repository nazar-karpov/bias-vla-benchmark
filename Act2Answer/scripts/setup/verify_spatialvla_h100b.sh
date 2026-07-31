#!/usr/bin/env bash
set -euo pipefail
export CONDA_ROOT="$HOME/conda"
export CONDA_ENVS_DIR="$HOME/conda/envs"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
REPO_ROOT="$HOME/bias_benchmark/nazar_folder/Act2Answer"
conda activate "$CONDA_ENVS_DIR/spatialvla_act2answer"
export PYTHONPATH="$REPO_ROOT/SimplerEnv:$REPO_ROOT/ManiSkill:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
cd "$REPO_ROOT/SimplerEnv"

python3 - <<'PY'
import numpy, torch, transformers
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(),
      "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("transformers", transformers.__version__)
import mani_skill, simpler_env
print("mani_skill", mani_skill.__version__, "@", mani_skill.__file__)
print("simpler_env @", simpler_env.__file__)
# eval entrypoint importable + spatialvla policy discoverable
import importlib
m = importlib.import_module("simpler_env.eval")
print("simpler_env.eval OK ->", m.__file__)
PY

echo "=== confirm cardset assets ==="
ls "$REPO_ROOT/ManiSkill/mani_skill/assets/carrot/pairs_choice_vla_confirm" | head
echo "cards: $(find "$REPO_ROOT/ManiSkill/mani_skill/assets/carrot/pairs_choice_vla_confirm" -maxdepth 1 -type d | wc -l)"
echo "VERIFY_OK"
