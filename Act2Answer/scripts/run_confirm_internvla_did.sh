#!/usr/bin/env bash
# Confirm-прогон InternVLA DiD gender (docs/CONFIRM_INTERNVLA_DID.md).
# Сиды 1..6 × pilot обе полярности (блоки 0 100 200 300), PAR=4 на 2 серверах.
set -uo pipefail
SC=/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer/scripts/run_single_card_center.sh
for seed in ${SEEDS:-1 2 3 4 5 6}; do
  echo "=== SEED $seed ==="
  VLA=internvla SEED=$seed BLOCKS="0 100 200 300" ASSET=pairs_single_pilot \
    NAME="confdid-s${seed}" PAR=4 bash "$SC"
done
echo "CONFIRM_IVLA_DID_DONE seeds=[${SEEDS:-1 2 3 4 5 6}] $(date -u)"
