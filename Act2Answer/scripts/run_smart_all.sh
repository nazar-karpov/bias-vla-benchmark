#!/usr/bin/env bash
# confirm_smart (smart/stupid, 400 эп) на 6 моделях h100 последовательно.
# SVLA гоняется отдельно на h100b. Резюмируемо (skip по stats.yaml).
set -uo pipefail
R="/workspace/moskalenko/bias-vla-benchmark-main"
A="$R/Act2Answer"
LOG=~/logs/smart
mkdir -p $LOG
source "$HOME/conda/etc/profile.d/conda.sh"

wait_port() { for i in $(seq 1 "$3"); do python3 -c "import socket;s=socket.socket();s.settimeout(2);exit(0 if s.connect_ex(('$1',$2))==0 else 1)" && return 0; sleep 10; done; return 1; }

run_eval() { # vla extra_cli... (env через export до вызова)
  local vla="$1"; shift
  conda activate "$HOME/conda/envs/magma_act2answer"
  export REPO_ROOT="$A" PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
  export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
  cd "$A/SimplerEnv"
  for sw in "" "--do-swap"; do
    local tag=noswap; [ -n "$sw" ] && tag=swap
    local name="smart-${vla}-${tag}"
    [ -f "$A/outputs/$name/glob/vis_0_test/stats.yaml" ] && { echo "SKIP $name"; continue; }
    echo "[$(date -u +%H:%M:%S)] EVAL $name"
    python3 -u -m simpler_env.eval --vla "$vla" \
      --assets confirm_smart --count 400 --episode-len 80 \
      --buffer-inferbatch 10 --buffer-minibatch -1 \
      --name "$name" $sw "$@" < /dev/null >> "$LOG/${vla}.log" 2>&1
    echo "[$(date -u +%H:%M:%S)] EVAL DONE $name rc=$?"
  done
}

echo "=== MAGMA ==="
run_eval magma

echo "=== GR00T ==="
pkill -f run_gr00t_server.py; sleep 3
(cd "$R" && nohup bash Act2Answer/scripts/gr00t_server_h100.sh > $LOG/gr00t_server.log 2>&1 &)
wait_port 127.0.0.1 5555 40 && sleep 60 || echo "GR00T server FAIL"
GR00T_HOST=127.0.0.1 GR00T_PORT=5555 run_eval gr00t
pkill -f run_gr00t_server.py

echo "=== INTERNVLA ==="
(cd "$R" && nohup bash Act2Answer/scripts/internvla_server_h100.sh > $LOG/internvla_server.log 2>&1 &)
wait_port 127.0.0.1 10093 60 && sleep 30 || echo "INTERNVLA server FAIL"
INTERNVLA_HOST=127.0.0.1 INTERNVLA_PORT=10093 run_eval internvla --vla-path "$(cat $R/internvla_ckpt/ckpt_path.txt)"
pkill -f internvla || true

echo "=== XVLA ==="
(cd "$R" && nohup bash Act2Answer/scripts/xvla_server_h100.sh > $LOG/xvla_server.log 2>&1 &)
wait_port 127.0.0.1 8010 60 && sleep 30 || echo "XVLA server FAIL"
XVLA_HOST=127.0.0.1 XVLA_PORT=8010 run_eval xvla
pkill -f deploy.py || true

echo "=== RLDX2 ==="
(cd "$R" && nohup bash Act2Answer/scripts/rldx_server_h100.sh > $LOG/rldx_server.log 2>&1 &)
wait_port 127.0.0.1 20000 90 && sleep 30 || echo "RLDX server FAIL"
RLDX_HOST=127.0.0.1 RLDX_PORT=20000 run_eval rldx
pkill -f "rldx" || true

echo "=== XIAOMI ==="
(cd "$R" && nohup bash Act2Answer/scripts/xiaomi_server_h100.sh > $LOG/xiaomi_server.log 2>&1 &)
wait_port 127.0.0.1 10086 60 && sleep 30 || echo "XIAOMI server FAIL"
# xiaomi ассертит на смене инструкции -> два блока по 200 (один вопрос на процесс)
conda activate "$HOME/conda/envs/magma_act2answer"
export REPO_ROOT="$A" PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 XIAOMI_TASK_ID=bridge_delta
cd "$A/SimplerEnv"
for sw in "" "--do-swap"; do
  tag=noswap; [ -n "$sw" ] && tag=swap
  for s in 0 200; do
    name="smart-xiaomi-${tag}-s${s}"
    [ -f "$A/outputs/$name/glob/vis_0_test/stats.yaml" ] && { echo "SKIP $name"; continue; }
    echo "[$(date -u +%H:%M:%S)] EVAL $name"
    python3 -u -m simpler_env.eval --vla xiaomi \
      --assets confirm_smart --start-id $s --count 200 --episode-len 80 \
      --buffer-inferbatch 10 --buffer-minibatch -1 --shard-size 200 \
      --name "$name" $sw < /dev/null >> "$LOG/xiaomi.log" 2>&1
    echo "[$(date -u +%H:%M:%S)] EVAL DONE $name rc=$?"
  done
done
pkill -f "deploy/server.py" || true

echo "=== SMART ALL DONE (h100) ==="
