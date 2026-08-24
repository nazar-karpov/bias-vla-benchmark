#!/usr/bin/env bash
# Подтверждающий прогон 4 пре-регистрированных ячеек (docs/CONFIRM_CENTER4.md).
# Последовательно: сид 1 и сид 2, в каждом — pilot/neg + top6-блоки
# (sysadmin/neg 200,300; athlete/pos 1200,1300; wealthy/neg 2200,2300).
set -uo pipefail
SC=/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer/scripts/run_single_card_center.sh
for seed in 1 2; do
  echo "=== SEED $seed: pilot/neg ==="
  VLA=magma SEED=$seed BLOCKS="200 300" ASSET=pairs_single_pilot \
    NAME="conf4pilot-s${seed}" PAR=2 bash "$SC"
  echo "=== SEED $seed: top6 blocks ==="
  VLA=magma SEED=$seed BLOCKS="200 300 1200 1300 2200 2300" ASSET=pairs_single_top6 \
    NAME="conf4top6-s${seed}" PAR=2 bash "$SC"
done
echo "CONFIRM4_ALL_DONE $(date -u)"
