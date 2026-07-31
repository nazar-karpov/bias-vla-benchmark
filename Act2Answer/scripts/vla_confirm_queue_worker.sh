#!/bin/bash
# w3 range (1200-1599, 400 eps) waits for a free GPU slot, then runs.
# 3 concurrent magma VLA workers OOM the camera-buffer (~70GB); 2 is safe (~55GB).
# So this holds until GPU memory drops below THRESH (a worker finished) or only
# <=1 eval worker remains, then launches. Idempotent-ish: relies on chunk -s dirs.
set -u
THRESH=42000   # MiB; launch when used mem falls below this (i.e. a worker freed)
while true; do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  nw=$(pgrep -fc "simpler_env.eval")
  if [ "$mem" -lt "$THRESH" ] || [ "$nw" -le 1 ]; then
    echo "QUEUE_W3_LAUNCH mem=${mem} workers=${nw} $(date -u)"
    break
  fi
  sleep 60
done
exec bash /workspace/moskalenko/run_magma_vla_mid.sh 1200 400 confirm-mid-magma-w3 0 8
