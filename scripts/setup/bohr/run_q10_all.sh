#!/usr/bin/env bash
# Вся серия q10 для одного порядка на одной карте. Порядки разводим по картам:
#   ORDER=noswap GPU=0 ./run_q10_all.sh &
#   ORDER=swap   GPU=1 ./run_q10_all.sh &
# Оба порядка нужны для непрерывного pull (притяжение из пар noswap/swap).
# Датасеты идут от маленьких к большим: если серию придётся оборвать, целые датасеты уже готовы.
set -u
ORDER=${ORDER:-noswap}
GPU=${GPU:-0}
S=$(dirname "${BASH_SOURCE[0]:-$0}")
for a in veri_q10 pairs_q10 visbias_q10 focus_q10; do
  echo "=== $(date -u +%H:%M) начинаю $a ($ORDER) ==="
  ASSETS=$a ORDER=$ORDER GPU=$GPU bash "$S/run_q10_magma.sh"
done
echo "Q10_ALL_DONE $ORDER $(date -u)"
