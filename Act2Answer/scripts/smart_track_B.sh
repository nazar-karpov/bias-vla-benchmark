#!/usr/bin/env bash
# Трек B (h100): gr00t -> xvla -> rldx2 (серверные, последовательно).
set -uo pipefail
R="/workspace/moskalenko/bias-vla-benchmark-main"
LOG=~/logs/smart; mkdir -p $LOG
wait_port() { for i in $(seq 1 "$3"); do python3 -c "import socket;s=socket.socket();s.settimeout(2);exit(0 if s.connect_ex(('$1',$2))==0 else 1)" && return 0; sleep 10; done; return 1; }

echo "=== B: GR00T ==="
pkill -f run_gr00t_server.py; sleep 3
(cd "$R" && nohup bash Act2Answer/scripts/gr00t_server_h100.sh > $LOG/gr00t_server.log 2>&1 &)
wait_port 127.0.0.1 5555 40 && sleep 60 || echo "GR00T server FAIL"
GR00T_HOST=127.0.0.1 GR00T_PORT=5555 bash "$R/Act2Answer/scripts/smart_blocks.sh" gr00t
pkill -f run_gr00t_server.py

echo "=== B: XVLA ==="
(cd "$R" && nohup bash Act2Answer/scripts/xvla_server_h100.sh > $LOG/xvla_server.log 2>&1 &)
wait_port 127.0.0.1 8010 60 && sleep 30 || echo "XVLA server FAIL"
XVLA_HOST=127.0.0.1 XVLA_PORT=8010 bash "$R/Act2Answer/scripts/smart_blocks.sh" xvla
pkill -f deploy.py || true

echo "=== B: RLDX2 ==="
(cd "$R" && nohup bash Act2Answer/scripts/rldx_server_h100.sh > $LOG/rldx_server.log 2>&1 &)
wait_port 127.0.0.1 20000 90 && sleep 30 || echo "RLDX server FAIL"
RLDX_HOST=127.0.0.1 RLDX_PORT=20000 bash "$R/Act2Answer/scripts/smart_blocks.sh" rldx
pkill -f rldx || true
echo "=== TRACK B DONE ==="
