#!/usr/bin/env bash
# Семантический потолок: ceiling_color (56) + ceiling_gender (200), noswap+swap.
# Последовательно: gr00t (server :5555) -> magma (standalone) -> xvla (:8010) -> rldx2 (:20000).
set -uo pipefail
R="/workspace/moskalenko/bias-vla-benchmark-main"
A="$R/Act2Answer"
LOG=~/logs/ceiling
mkdir -p $LOG

source "$HOME/conda/etc/profile.d/conda.sh"

wait_port() { # host port tries
  for i in $(seq 1 "$3"); do
    python3 -c "import socket;s=socket.socket();s.settimeout(2);exit(0 if s.connect_ex(('$1',$2))==0 else 1)" && return 0
    sleep 10
  done
  return 1
}

run_eval() { # vla name extra_env...
  local vla="$1"; shift
  conda activate "$HOME/conda/envs/magma_act2answer"
  export REPO_ROOT="$A" PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
  export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
  cd "$A/SimplerEnv"
  for asset in ceiling_color ceiling_gender; do
    local cnt=56 ib=8
    [ "$asset" = ceiling_gender ] && cnt=200 && ib=10
    for sw in "" "--do-swap"; do
      local tag=noswap; [ -n "$sw" ] && tag=swap
      local name="ceiling-${asset#ceiling_}-${vla}-${tag}"
      [ -f "$A/outputs/$name/glob/vis_0_test/stats.yaml" ] && { echo "SKIP $name"; continue; }
      echo "[$(date -u +%H:%M:%S)] EVAL $name"
      env "$@" python3 -u -m simpler_env.eval --vla "$vla" \
        --assets "$asset" --count "$cnt" --episode-len 80 \
        --buffer-inferbatch "$ib" --buffer-minibatch -1 \
        --name "$name" $sw < /dev/null >> "$LOG/${vla}.log" 2>&1
      echo "[$(date -u +%H:%M:%S)] EVAL DONE $name rc=$?"
    done
  done
}

# ---- 1. GR00T ----
echo "=== GR00T ==="
pkill -f run_gr00t_server.py; sleep 3
(cd "$R" && nohup bash Act2Answer/scripts/gr00t_server_h100.sh > $LOG/gr00t_server.log 2>&1 &)
wait_port 127.0.0.1 5555 30 && sleep 60 || { echo "GR00T server FAIL"; }
run_eval gr00t GR00T_HOST=127.0.0.1 GR00T_PORT=5555
pkill -f run_gr00t_server.py

# ---- 2. Magma (standalone) ----
echo "=== MAGMA ==="
run_eval magma _=1

# ---- 3. xVLA ----
echo "=== XVLA ==="
(cd "$R" && nohup bash Act2Answer/scripts/xvla_server_h100.sh > $LOG/xvla_server.log 2>&1 &)
wait_port 127.0.0.1 8010 60 && sleep 30 || { echo "XVLA server FAIL"; }
run_eval xvla XVLA_HOST=127.0.0.1 XVLA_PORT=8010
pkill -f "deploy.py\|xvla_server"

# ---- 4. RLDX2 ----
echo "=== RLDX2 ==="
(cd "$R" && nohup bash Act2Answer/scripts/rldx_server_h100.sh > $LOG/rldx_server.log 2>&1 &)
wait_port 127.0.0.1 20000 60 && sleep 30 || { echo "RLDX server FAIL"; }
run_eval rldx RLDX_HOST=127.0.0.1 RLDX_PORT=20000
pkill -f rldx

echo "=== CEILING ALL DONE ==="
