#!/bin/bash
# Compact progress for Magma VLA-eval confirm-mid. Progress = chunk dirs with stats.
A=/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer/outputs
done_chunks=0
for d in $A/confirm-mid-magma-w*-noswap-s* $A/confirm-mid-magma-w*-swap-s*; do
  [ -d "$d" ] && [ -n "$(find "$d" -name 'stats.yaml' 2>/dev/null|head -1)" ] && done_chunks=$((done_chunks+1))
done
# total chunks expected: (600+600+400)/50 * 2 layouts = 32*2 = 64
alive=$(pgrep -fc "simpler_env.eval")
wd=$(pgrep -fc watchdog_vla_mid.sh)
ecc=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader|tr -d ' ')
tmp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader|tr -d ' ')
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader|tr -d ' ')
pct=$(( done_chunks*100/64 ))
echo "VLA-MID $(date -u +%H:%M:%S) chunks=$done_chunks/64 (${pct}%, ~$((done_chunks*50)) rollouts) workers=$alive wd=$wd ecc=$ecc temp=${tmp} mem=${mem}"
