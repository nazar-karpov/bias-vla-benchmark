#!/usr/bin/env bash
# Set up the RLDX-1 repo env (uv-based, python 3.10) and download a SIMPLER-WidowX
# checkpoint. Idempotent: re-running is cheap once .venv + ckpt exist.
# Arg: <repo_dir>. Writes <repo>/.rldx_ckpt_path.txt with the resolved model path.
set -uo pipefail
REPO="${1:?usage: setup_rldx.sh <repo_dir>}"
export PATH="$HOME/.local/bin:$PATH"
cd "$REPO"

echo "RLDX_SETUP_START $(date -u)"
command -v uv >/dev/null || { echo "ERROR: uv not on PATH"; exit 1; }

# 1. env (.venv in repo). uv sync resolves from uv.lock; then editable install.
if [ ! -x "$REPO/.venv/bin/python" ]; then
  SYNC_FLAGS=()
  command -v nvcc >/dev/null 2>&1 || SYNC_FLAGS+=(--no-install-package flash-attn)
  uv sync --python 3.10 "${SYNC_FLAGS[@]}" || { echo "uv sync failed"; exit 1; }
fi
uv pip install -e . || { echo "uv pip install -e . failed"; exit 1; }

# 2. sanity import
uv run --no-sync python -c "import rldx; from rldx.policy.server_client import PolicyServer; print('rldx_import_ok', rldx.__version__)" \
  || { echo "rldx import failed"; exit 1; }

# 3. checkpoint: the SIMPLER-WidowX finetune matches the widowx_bridge setup we use.
CKPT_REPO="RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX"
CKPT_DIR="${RLDX_CKPT_ROOT:-$HOME/bias_benchmark/nazar_folder/rldx_ckpt}/RLDX-1-FT-SIMPLER-WIDOWX"
if [ ! -f "$CKPT_DIR/.download_done" ]; then
  mkdir -p "$CKPT_DIR"
  uv run --no-sync python - "$CKPT_REPO" "$CKPT_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dst = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo, local_dir=dst)
print("downloaded", dst)
PY
  touch "$CKPT_DIR/.download_done"
fi
echo "$CKPT_DIR" > "$REPO/.rldx_ckpt_path.txt"
echo "RLDX_SETUP_DONE $(date -u) ckpt=$CKPT_DIR"
