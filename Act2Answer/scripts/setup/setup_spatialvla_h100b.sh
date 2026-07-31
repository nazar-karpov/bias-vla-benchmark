#!/usr/bin/env bash
# h100b-specific launcher for setup_spatialvla_env.sh.
# This node's conda lives at ~/conda (not /opt/conda), and we create the env
# under ~/conda/envs (not the shared /home/jovyan/.mlspace/envs).
set -euo pipefail
export CONDA_ROOT="$HOME/conda"
export CONDA_ENVS_DIR="$HOME/conda/envs"
export A2A_EXTERNAL_DIR="$HOME/bias_benchmark/nazar_folder"   # unused here but keep env.sh happy
source "$CONDA_ROOT/etc/profile.d/conda.sh"
exec bash "$(dirname "${BASH_SOURCE[0]}")/setup_spatialvla_env.sh"
