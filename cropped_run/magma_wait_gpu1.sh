#!/usr/bin/env bash
# ждём пока spatialvla добьёт и GPU1 освободится (<8GB used), потом magma на GPU1 ib=5
set -uo pipefail
B=$HOME/bias_benchmark; L=$B/nazar_folder/cropped_run/logs
for i in $(seq 1 120); do
  # spatialvla-процессы больше не бегут?
  if ! pgrep -f 'vla spatialvla' >/dev/null; then
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    [ "$used" -lt 8000 ] && break
  fi
  sleep 20
done
echo "GPU1 free after $((i*20))s, launching magma" >> $L/magma_wait.log
bash $B/nazar_folder/cropped_run/soft_rerun_magma.sh 1 5 >> $L/softrun_magma3.log 2>&1
