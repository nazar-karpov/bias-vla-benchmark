#!/usr/bin/env bash
# Дождаться освобождения GPU и записать демо-видео одноплиточных эпизодов.
# Нужно потому, что боевые прогоны идут с A2A_SAVE_VIDEO=0 и держат обе карты:
# демо приходится ставить в очередь, а не рядом.
#
# Ждём, пока не останется процессов simpler_env.eval (или пока не истечёт
# MAX_WAIT), затем запускаем record_single_demo.sh.
set -uo pipefail
R=/workspace/moskalenko/bias-vla-benchmark-main
LOG=/workspace/moskalenko/logs_single_card
MAX_WAIT="${MAX_WAIT:-14400}"     # 4 часа
STARTS="${STARTS:-261 40 359 209}"

waited=0
while [ "$waited" -lt "$MAX_WAIT" ]; do
  running=$(ps aux | grep -c "[s]impler_env.eval" || true)
  if [ "$running" -le 1 ]; then
    echo "[$(date -u +%H:%M:%S)] GPU свободны (процессов: $running), пишу демо"
    STARTS="$STARTS" GPU="${GPU:-0}" bash "$R/Act2Answer/scripts/record_single_demo.sh"
    echo "[$(date -u +%H:%M:%S)] демо готово"
    find "$R/Act2Answer/outputs" -path "*demo-single-*" -name "*.mp4" | sort
    exit 0
  fi
  sleep 120
  waited=$(( waited + 120 ))
done
echo "TIMEOUT: за $MAX_WAIT c GPU так и не освободились"
exit 1
