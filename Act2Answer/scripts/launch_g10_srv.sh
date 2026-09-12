#!/usr/bin/env bash
# Серверные модели программы g10: GR00T-N1.7 (zmq, Isaac-GR00T) и Xiaomi-Robotics-0 (tcp/pickle).
# Сервер политики живёт в venv Isaac-GR00T/.venv (uv sync: py3.12, torch 2.9, transformers 4.57,
# flash-attn 2.8.3; см. scripts/patches/README.md) — Xiaomi-серверу нужны только torch+transformers,
# поэтому venv общий. Клиент-симулятор — env magma_act2answer (run_g10_magma.sh с VLA=gr00t|xiaomi).
#
# На каждой карте K серверов (K=${K:-1}) + K цепочек клиента; порт = BASE_PORT + GPU*10 + k.
# Зачем K>1: Xiaomi отвечает ~1.1 с на запрос (замер 12.09 на 4090: forward 1.10 с при 98 % util),
# клиент шлёт запросы по одному, и карта половину времени простаивает (рендер/физика) —
# два сервера на карте дали ×1.56 по эпизодам.
#
#   VLA=gr00t  NODE=A|C2|C3|C4          bash launch_g10_srv.sh      # 8 карт облака, K=1
#   VLA=xiaomi NODE=XA|XC2|XC3|XC4 K=3  bash launch_g10_srv.sh      # 8 карт облака, 3 сервера/карту
#   VLA=xiaomi NODE=XB K=2              bash launch_g10_srv.sh      # Bohr: PAIRS, 2 сервера/карту
#   VLA=... NODE=SMOKE                  bash launch_g10_srv.sh      # 4 эпизода на GPU0
#   DRY=1 — показать, откуда продолжит (серверы не поднимаются)
#   STOP=1 — убить серверы этой модели на ноде (сервер Xiaomi — mp.Process-ребёнок, убиваем по порту)
set -u
VLA=${VLA:?VLA=gr00t|xiaomi}
NODE=${NODE:-}
DRY=${DRY:-0}
K=${K:-1}
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)   # корень bias-vla-benchmark-main (h100q или Bohr)
A=$R/Act2Answer
VENV=$R/Isaac-GR00T/.venv/bin/python
RUN=$A/scripts/run_g10_magma.sh
LOGS=$HOME/ws
HF=${HF_HOME:-$([ -d /workspace/moskalenko/hf_cache ] && echo /workspace/moskalenko/hf_cache || echo $HOME/ws/hf_cache)}
case "$VLA" in
  gr00t)  BASE_PORT=${BASE_PORT:-5500};  PORT_VAR=GR00T_PORT;  HOST_VAR=GR00T_HOST
          SNAP=$(ls -d $HF/hub/models--nvidia--GR00T-N1.7-SimplerEnv-Bridge/snapshots/* | head -1) ;;
  xiaomi) BASE_PORT=${BASE_PORT:-10000}; PORT_VAR=XIAOMI_PORT; HOST_VAR=XIAOMI_HOST
          SNAP=$(ls -d $HF/hub/models--XiaomiRobotics--Xiaomi-Robotics-0-SimplerEnv-WidowX/snapshots/* | head -1) ;;
  *) echo "VLA=gr00t|xiaomi"; exit 1;;
esac
export VLA
# клиентские переменные (проходят в run_g10_magma.sh через окружение)
export XIAOMI_TASK_ID=bridge_delta INFERBATCH=${INFERBATCH:-48}
export XIAOMI_BATCH=${XIAOMI_BATCH:-1}   # клиент шлёт весь буфер одним запросом (нужен xiaomi_server_batch.py)
XIAOMI_SERVER=${XIAOMI_SERVER:-$A/scripts/xiaomi_server_batch.py}   # deploy/server.py — оригинал, по одному
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( K > 1 ? 4 : 10 ))}

port_of() { echo $((BASE_PORT + $1 * 10 + $2)); }   # gpu k
port_alive() { (exec 3<>/dev/tcp/127.0.0.1/$1) 2>/dev/null && exec 3>&- && return 0; return 1; }
kill_port() {  # порт → pid слушателя (ss); на облаке ss без pid, тогда по "--port P" в argv родителя и его mp-детям
  local pids=$(ss -ltnp 2>/dev/null | grep ":$1 " | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u)
  for par in $(pgrep -f -- "--port $1( |$)"); do pids="$pids $(pgrep -P $par) $par"; done
  for p in $pids; do kill -9 $p 2>/dev/null; done; }

if [ "${STOP:-0}" = 1 ]; then
  pkill -f "run_gr00t_server.py|deploy/server.py" 2>/dev/null
  for g in 0 1 2 3 4 5 6 7; do for k in 0 1 2 3 4 5; do p=$(port_of $g $k); port_alive $p && { kill_port $p; echo "убит сервер :$p"; }; done; done
  exit 0
fi

start_server() {  # gpu k
  local g=$1 k=$2
  local p=$(port_of $g $k)
  port_alive $p && { echo "сервер $VLA :$p (GPU$g/$k) уже слушает — переиспользую"; return 0; }
  local log="$LOGS/_srv_${VLA}_$(hostname | cut -c1-24)_gpu${g}_$k.log"
  ( export CUDA_VISIBLE_DEVICES="$g" HF_HOME=$HF HF_HUB_DISABLE_XET=1
    export OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS TOKENIZERS_PARALLELISM=false
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    export GR00T_COSMOS_PATH=$(ls -d $HF/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/* 2>/dev/null | head -1)
    export XR0_ATTN=${XR0_ATTN:-flash_attention_2}   # MiBoTForActionGeneration в tf 4.57.3 не проходит sdpa-проверку
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
    if [ "$VLA" = gr00t ]; then
      cd $R/Isaac-GR00T
      exec setsid nohup $VENV gr00t/eval/run_gr00t_server.py --model-path "$SNAP" \
        --embodiment-tag simpler_env_widowx --port "$p" --host 0.0.0.0 --use-sim-policy-wrapper
    else
      cd $R/Xiaomi-Robotics-0
      exec setsid nohup $VENV $XIAOMI_SERVER --model "$SNAP" --host localhost --port "$p"
    fi ) < /dev/null > "$log" 2>&1 &
  for i in $(seq 1 90); do sleep 10; port_alive $p && { echo "сервер $VLA :$p (GPU$g/$k) поднялся за ~$((i*10)) с"; sleep 5; return 0; }
    grep -q "Traceback" "$log" 2>/dev/null && { echo "СЕРВЕР $VLA :$p (GPU$g/$k) УПАЛ — см. $log"; return 1; }; done
  echo "СЕРВЕР $VLA :$p (GPU$g/$k) НЕ ПОДНЯЛСЯ за 15 мин — см. $log"; return 1
}

chain() {  # gpu k  "ASSETS:ORDER:START0:END" ...
  local gpu=$1 k=$2; shift 2
  local p=$(port_of $gpu $k)
  local cmds=""
  for spec in "$@"; do
    IFS=: read -r a o s e <<< "$spec"
    cmds+="$PORT_VAR=$p $HOST_VAR=127.0.0.1 VLA=$VLA ASSETS=$a ORDER=$o GPU=$gpu START0=$s END=$e DRY=$DRY bash $RUN; "
  done
  if [ "$DRY" = 1 ]; then bash -c "$cmds"; return; fi
  start_server $gpu $k || return 1
  setsid -f nohup bash -c "$cmds" < /dev/null > "$LOGS/chain_g10_${VLA}_${NODE}_gpu${gpu}_$k.out" 2>&1
  echo "GPU$gpu/$k :$p → $*"
}

# split3 GPU ASSETS ORDER  — диапазон [0,2496) или [2496,5000) тремя потоками (границы кратны 48)
split3_lo() { chain $1 0 $2:$3:0:816;    chain $1 1 $2:$3:816:1680;  chain $1 2 $2:$3:1680:2496; }
split3_hi() { chain $1 0 $2:$3:2496:3312; chain $1 1 $2:$3:3312:4176; chain $1 2 $2:$3:4176:5000; }
# split2 — двумя потоками (K=2: h100q с 48 ядрами не тянет 12 цепочек — троттлинг, ssh не отвечает)
split2_lo() { chain $1 0 $2:$3:0:1248;    chain $1 1 $2:$3:1248:2496; }
split2_hi() { chain $1 0 $2:$3:2496:3744; chain $1 1 $2:$3:3744:5000; }

case "$NODE" in
  SMOKE)
    start_server 0 0 || exit 1
    source $HOME/ws/env_bohr.sh
    export ${PORT_VAR}=$(port_of 0 0) ${HOST_VAR}=127.0.0.1 CUDA_VISIBLE_DEVICES=0 MKL_NUM_THREADS=$OMP_NUM_THREADS
    export BOARD_XY_SCALE=1.2 A2A_TILE_Y=0.14 A2A_TRAJ_LOG=1 A2A_SAVE_VIDEO=0
    rm -rf $A/outputs/smoke-$VLA-noswap
    python -u -m simpler_env.eval --vla $VLA --assets pairs_g10 \
      --start-id 0 --count 4 --shard-size 4 --buffer-inferbatch 4 --name smoke-$VLA-noswap \
      < /dev/null > $LOGS/_smoke_$VLA.log 2>&1
    echo "SMOKE rc=$?"; ls $A/outputs/smoke-$VLA-noswap/glob/vis_0_test/ 2>&1
    ;;
  # ---- K=1: один поток на карту (GR00T) ----
  A)
    chain 0 0 focus_g10:noswap:0:2496    pairs_g10:noswap:0:480
    chain 1 0 focus_g10:noswap:2496:5000 pairs_g10:noswap:480:1000
    chain 2 0 focus_g10:swap:0:2496      pairs_g10:swap:0:480
    chain 3 0 focus_g10:swap:2496:5000   pairs_g10:swap:480:1000
    ;;
  C2) chain 0 0 visbias_g10:noswap:0:2496; chain 1 0 visbias_g10:noswap:2496:5000 ;;
  C3) chain 0 0 visbias_g10:swap:0:2496 ;;
  C4) chain 0 0 visbias_g10:swap:2496:5000 ;;
  # ---- K=3: три потока на карту (Xiaomi); PAIRS уходит на Bohr (XB) ----
  XA)  split3_lo 0 focus_g10 noswap; split3_hi 1 focus_g10 noswap; split3_lo 2 focus_g10 swap; split3_hi 3 focus_g10 swap ;;
  XA2) split2_lo 0 focus_g10 noswap; split2_hi 1 focus_g10 noswap; split2_lo 2 focus_g10 swap; split2_hi 3 focus_g10 swap ;;
  XC2) split3_lo 0 visbias_g10 noswap; split3_hi 1 visbias_g10 noswap ;;
  XC3) split3_lo 0 visbias_g10 swap ;;
  XC4) split3_hi 0 visbias_g10 swap ;;
  XB)  chain 0 0 pairs_g10:noswap:0:480; chain 0 1 pairs_g10:noswap:480:1000
       chain 1 0 pairs_g10:swap:0:480;   chain 1 1 pairs_g10:swap:480:1000 ;;
  *) echo "NODE=A|C2|C3|C4|XA|XA2|XC2|XC3|XC4|XB|SMOKE"; exit 1;;
esac
[ "$DRY" = 1 ] || { sleep 3; echo "клиентов simpler_env.eval: $(pgrep -f 'python -u -m simpler_env.eval' | wc -l), серверов (портов): $(for g in 0 1 2 3; do for k in 0 1 2; do port_alive $(port_of $g $k) && echo x; done; done | wc -l)"; }
