#!/usr/bin/env bash
# Серверные модели программы g10: GR00T-N1.7 (zmq, Isaac-GR00T) и Xiaomi-Robotics-0 (tcp/pickle).
# Сервер политики живёт в venv Isaac-GR00T/.venv (uv sync: py3.12, torch 2.9, transformers 4.57,
# flash-attn 2.8.3) — Xiaomi-серверу нужны только torch+transformers, поэтому venv общий.
# Клиент-симулятор — env magma_act2answer (run_g10_magma.sh с VLA=gr00t|xiaomi).
# На каждой карте: один сервер (порт BASE+GPU) + одна цепочка клиента.
#   VLA=gr00t  NODE=A|C2|C3|C4|SMOKE bash launch_g10_srv.sh
#   VLA=xiaomi NODE=A|C2|C3|C4|SMOKE bash launch_g10_srv.sh
#   DRY=1 — показать, откуда продолжит (серверы не поднимаются)
set -u
VLA=${VLA:?VLA=gr00t|xiaomi}
NODE=${NODE:?NODE=A|C2|C3|C4|SMOKE}
DRY=${DRY:-0}
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)   # корень bias-vla-benchmark-main (h100q или Bohr)
A=$R/Act2Answer
VENV=$R/Isaac-GR00T/.venv/bin/python
RUN=$A/scripts/run_g10_magma.sh
LOGS=$HOME/ws
HF=${HF_HOME:-$([ -d /workspace/moskalenko/hf_cache ] && echo /workspace/moskalenko/hf_cache || echo $HOME/ws/hf_cache)}
case "$VLA" in
  gr00t)  BASE_PORT=${BASE_PORT:-5555};  PORT_VAR=GR00T_PORT;  HOST_VAR=GR00T_HOST
          SNAP=$(ls -d $HF/hub/models--nvidia--GR00T-N1.7-SimplerEnv-Bridge/snapshots/* | head -1) ;;
  xiaomi) BASE_PORT=${BASE_PORT:-10086}; PORT_VAR=XIAOMI_PORT; HOST_VAR=XIAOMI_HOST
          SNAP=$(ls -d $HF/hub/models--XiaomiRobotics--Xiaomi-Robotics-0-SimplerEnv-WidowX/snapshots/* | head -1) ;;
  *) echo "VLA=gr00t|xiaomi"; exit 1;;
esac
export VLA
# клиентские переменные (проходят в run_g10_magma.sh через окружение)
export XIAOMI_TASK_ID=bridge_delta INFERBATCH=${INFERBATCH:-48}

port_alive() { (exec 3<>/dev/tcp/127.0.0.1/$1) 2>/dev/null && exec 3>&- && return 0; return 1; }

start_server() {  # gpu
  local g=$1
  local p=$((BASE_PORT + g))
  port_alive $p && { echo "сервер $VLA :$p (GPU$g) уже слушает — переиспользую"; return 0; }
  local log="$LOGS/_srv_${VLA}_$(hostname | cut -c1-24)_gpu$g.log"
  ( export CUDA_VISIBLE_DEVICES="$g" HF_HOME=$HF HF_HUB_DISABLE_XET=1
    export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 TOKENIZERS_PARALLELISM=false
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    export GR00T_COSMOS_PATH=$(ls -d $HF/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/* | head -1)
    export XR0_ATTN=${XR0_ATTN:-flash_attention_2}   # MiBoTForActionGeneration в tf 4.57.3 не проходит sdpa-проверку
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
    if [ "$VLA" = gr00t ]; then
      cd $R/Isaac-GR00T
      exec setsid nohup $VENV gr00t/eval/run_gr00t_server.py --model-path "$SNAP" \
        --embodiment-tag simpler_env_widowx --port "$p" --host 0.0.0.0 --use-sim-policy-wrapper
    else
      cd $R/Xiaomi-Robotics-0
      exec setsid nohup $VENV deploy/server.py --model "$SNAP" --host localhost --port "$p"
    fi ) < /dev/null > "$log" 2>&1 &
  for i in $(seq 1 90); do sleep 10; port_alive $p && { echo "сервер $VLA :$p (GPU$g) поднялся за ~$((i*10)) с"; sleep 10; return 0; }
    grep -q "Traceback" "$log" 2>/dev/null && { echo "СЕРВЕР $VLA :$p (GPU$g) УПАЛ — см. $log"; return 1; }; done
  echo "СЕРВЕР $VLA :$p (GPU$g) НЕ ПОДНЯЛСЯ за 15 мин — см. $log"; return 1
}

chain() {  # gpu  "ASSETS:ORDER:START0:END" ...
  local gpu=$1; shift
  local cmds=""
  for spec in "$@"; do
    IFS=: read -r a o s e <<< "$spec"
    cmds+="$PORT_VAR=$((BASE_PORT + gpu)) $HOST_VAR=127.0.0.1 VLA=$VLA ASSETS=$a ORDER=$o GPU=$gpu START0=$s END=$e DRY=$DRY bash $RUN; "
  done
  if [ "$DRY" = 1 ]; then bash -c "$cmds"; return; fi
  start_server $gpu || return 1
  setsid -f nohup bash -c "$cmds" < /dev/null > "$LOGS/chain_g10_${VLA}_${NODE}_gpu${gpu}.out" 2>&1
  echo "GPU$gpu: $*"
}

case "$NODE" in
  SMOKE)
    start_server 0 || exit 1
    source $HOME/ws/env_bohr.sh
    export ${PORT_VAR}=$BASE_PORT ${HOST_VAR}=127.0.0.1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
    export BOARD_XY_SCALE=1.2 A2A_TILE_Y=0.14 A2A_TRAJ_LOG=1 A2A_SAVE_VIDEO=0
    rm -rf $A/outputs/smoke-$VLA-noswap
    python -u -m simpler_env.eval --vla $VLA --assets pairs_g10 \
      --start-id 0 --count 4 --shard-size 4 --buffer-inferbatch 4 --name smoke-$VLA-noswap \
      < /dev/null > $LOGS/_smoke_$VLA.log 2>&1
    echo "SMOKE rc=$?"; ls $A/outputs/smoke-$VLA-noswap/glob/vis_0_test/ 2>&1
    ;;
  A)
    chain 0 focus_g10:noswap:0:2496    pairs_g10:noswap:0:480
    chain 1 focus_g10:noswap:2496:5000 pairs_g10:noswap:480:1000
    chain 2 focus_g10:swap:0:2496      pairs_g10:swap:0:480
    chain 3 focus_g10:swap:2496:5000   pairs_g10:swap:480:1000
    ;;
  C2)
    chain 0 visbias_g10:noswap:0:2496
    chain 1 visbias_g10:noswap:2496:5000
    ;;
  C3) chain 0 visbias_g10:swap:0:2496 ;;
  C4) chain 0 visbias_g10:swap:2496:5000 ;;
  *) echo "NODE=A|C2|C3|C4|SMOKE"; exit 1;;
esac
[ "$DRY" = 1 ] || { sleep 3; echo "клиентов simpler_env.eval: $(pgrep -f 'python -u -m simpler_env.eval' | wc -l), серверов: $(pgrep -cf 'run_gr00t_server|deploy/server.py')"; }
