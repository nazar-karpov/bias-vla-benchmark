#!/bin/bash
# Lightweight detector for the 2-worker (+queued w3) Magma VLA run.
# DETECT ONLY (no auto-relaunch — avoids the earlier relaunch race). Emits a line
# on: ECC event, a worker frozen 3 checks, a worker exiting, or all done. The
# supervising loop decides what to do. Progress = chunk -s dirs with stats.yaml.
set -u
A=/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer/outputs
declare -A STRIKE=()
ecc_base=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader|tr -d ' ')
echo "WD2_START ecc_base=$ecc_base $(date -u)"
prev_running=99
while true; do
  ecc=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader|tr -d ' ')
  [ "$ecc" != "$ecc_base" ] && { echo "ECC_EVENT now=$ecc was=$ecc_base $(date -u)"; ecc_base=$ecc; }
  # snapshot cpu per eval pid
  declare -A A0=(); for P in $(pgrep -f simpler_env.eval); do A0[$P]=$(awk '{print $14+$15}' /proc/$P/stat 2>/dev/null); done
  sleep 12
  running=0
  for P in $(pgrep -f simpler_env.eval); do
    running=$((running+1))
    b=$(awk '{print $14+$15}' /proc/$P/stat 2>/dev/null); a=${A0[$P]:-$b}; d=$((b-a))
    if [ "$d" -lt 24 ]; then
      STRIKE[$P]=$(( ${STRIKE[$P]:-0}+1 ))
      [ "${STRIKE[$P]}" -ge 3 ] && echo "WORKER_FROZEN pid=$P cpu/12s=$d $(date -u)"
    else STRIKE[$P]=0; fi
  done
  # note transitions in worker count (a worker finished/died)
  [ "$running" != "$prev_running" ] && { echo "WORKERS_NOW=$running $(date -u)"; prev_running=$running; }
  # all-done: no eval workers AND queue-runner gone
  if [ "$running" -eq 0 ] && ! pgrep -f queue_w3_vla_mid.sh >/dev/null; then
    echo "WD2_ALL_DONE $(date -u)"; break
  fi
  sleep 48
done
