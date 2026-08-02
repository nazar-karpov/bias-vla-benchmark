#!/usr/bin/env bash
# Трек A (h100): magma (локальная, без сервера).
set -uo pipefail
R="/workspace/moskalenko/bias-vla-benchmark-main"
bash "$R/Act2Answer/scripts/smart_blocks.sh" magma
echo "=== TRACK A DONE ==="
