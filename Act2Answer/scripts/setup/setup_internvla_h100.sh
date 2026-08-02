#!/usr/bin/env bash
# h100-specific launcher for setup_internvla_env.sh.
# This node's conda lives at ~/conda; env created under ~/conda/envs.
# External repos live next to the top-level repo dir (A2A_EXTERNAL_DIR).
set -euo pipefail
export CONDA_ROOT="$HOME/conda"
export CONDA_ENVS_DIR="$HOME/conda/envs"
export A2A_EXTERNAL_DIR="/workspace/moskalenko/bias-vla-benchmark-main"
export INTERNVLA_REPO="$A2A_EXTERNAL_DIR/InternVLA-M1"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
exec bash "$(dirname "${BASH_SOURCE[0]}")/setup_internvla_env.sh"
