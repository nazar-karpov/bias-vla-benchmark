#!/usr/bin/env bash
# Очередь проб расстановки на CPU-ноде (RAM 3 ГБ → строго по одной).
OUT=/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer/outputs/tile_layout
while pgrep -f tile_layout_probe.py > /dev/null; do sleep 10; done
# эффективный масштаб = 1.3 × SCALE
SCALE=0.8846 LAYOUTS="0.155,0;0.145,0;0.135,0" OUT=$OUT bash /workspace/moskalenko/run_tile_layout_cpu.sh > /workspace/moskalenko/logs_tile_layout_s115.log 2>&1
SCALE=0.9231 LAYOUTS="0.155,0;0.14,0" OUT=$OUT bash /workspace/moskalenko/run_tile_layout_cpu.sh > /workspace/moskalenko/logs_tile_layout_s12.log 2>&1
SCALE=0.7692 LAYOUTS="0.155,0" OUT=$OUT bash /workspace/moskalenko/run_tile_layout_cpu.sh > /workspace/moskalenko/logs_tile_layout_s10.log 2>&1
echo QUEUE_DONE
