#!/usr/bin/env bash
# Предполётная проверка облачной ноды перед запуском VLA-прогона (после инцидента 11.09:
# карты с заклинившим Vulkan-рендером). Печатает PASS/FAIL по каждой карте.
#   bash /workspace/moskalenko/ws_h100/preflight_node.sh
set -u
WS=/workspace/moskalenko
PY=$WS/conda/envs/magma_act2answer/bin/python
T=$WS/bias-vla-benchmark-main/Act2Answer/scripts/test_render_dev.py
echo "== нода: $(hostname)"
# $HOME эфемерный: раннеры ждут $HOME/ws -> ws_h100
[ -e "$HOME/ws" ] || ln -s $WS/ws_h100 "$HOME/ws"
echo "HOME/ws -> $(readlink -f $HOME/ws)"
echo "GPU: $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | tr '\n' ';')"
echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)  nproc: $(nproc)  ulimit -n hard: $(ulimit -Hn)"
echo "/workspace: $(df -h $WS | tail -1 | awk '{print $4" свободно"}')  repo: $(git -C $WS/bias-vla-benchmark-main log --oneline -1 | cut -c1-60)"
[ -x "$PY" ] && echo "conda python: ok" || echo "conda python: НЕТ"
[ -d "$WS/maniskill_assets" ] && echo "MS_ASSET_DIR: ok" || echo "MS_ASSET_DIR: НЕТ"
[ -d "$WS/hf_cache" ] && echo "HF cache: ok" || echo "HF cache: НЕТ"
echo "чужих процессов simpler_env.eval: $(pgrep -cf simpler_env.eval)"
N=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
fail=0
for ((g=0; g<N; g++)); do
  r=$(CUDA_VISIBLE_DEVICES=$g timeout 90 $PY -u $T cuda 2>&1 | grep -E "RENDER_OK|resolved" | tr '\n' ' ')
  if echo "$r" | grep -q RENDER_OK; then rs="render PASS"; else rs="render FAIL/HANG"; fail=1; fi
  m=$(CUDA_VISIBLE_DEVICES=$g timeout 60 $PY -c "import torch;a=torch.randn(4096,4096,device='cuda',dtype=torch.float16);b=a@a;torch.cuda.synchronize();print('MATMUL_OK')" 2>&1 | grep -c MATMUL_OK)
  [ "$m" = 1 ] && ms="cuda PASS" || { ms="cuda FAIL"; fail=1; }
  echo "GPU$g: $rs, $ms  [$(echo "$r" | grep -o 'pci=[^ ]*')]"
done
[ $fail = 0 ] && echo "PREFLIGHT_PASS $(hostname)" || echo "PREFLIGHT_FAIL $(hostname)"
