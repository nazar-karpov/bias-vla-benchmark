#!/usr/bin/env bash
# InternVLA-M1 в программе g10: политика живёт в zmq-сервере (env internvla), клиент-симулятор —
# в magma_act2answer (run_g10_magma.sh с VLA=internvla). На каждой карте: один сервер (порт
# 10093+GPU) + одна цепочка клиента. Раскладка по нодам (12.09.2026, 8×H100):
#   NODE=A  (h100q, 4 карты): FOCUS половины × порядки → PAIRS половины × порядки
#   NODE=C2 (h100q2, 2 карты): VisBias noswap [0,2496) / [2496,5000)
#   NODE=C3 (h100q3, 1 карта): VisBias swap [0,2496)
#   NODE=C4 (h100q4, 1 карта): VisBias swap [2496,5000)
#   NODE=SMOKE: сервер на GPU0 + 4 эпизода pairs_$CS (проверка перед боем)
#   NODE=A DRY=1 — только показать, откуда продолжит (серверы не поднимаются)
set -u
PROG=${PROG:-g10}; CS=${CS:-$PROG}; export PROG   # e10: PROG=e10 → кардсеты focus_e10/visbias_e10/pairs_e10, шарды e10-<vla>-...
NODE=${NODE:?NODE=A|C2|C3|C4|SMOKE}
DRY=${DRY:-0}
R=/workspace/moskalenko/bias-vla-benchmark-main
A=$R/Act2Answer
CONDA=/workspace/moskalenko/conda
RUN=$A/scripts/run_g10_magma.sh
LOGS=$HOME/ws
CKPT=$(cat $R/internvla_ckpt/ckpt_path.txt)
BASE_PORT=${BASE_PORT:-10093}
export VLA=internvla VLA_ARGS="--vla-path $CKPT" INTERNVLA_HOST=127.0.0.1

port_alive() { (exec 3<>/dev/tcp/127.0.0.1/$1) 2>/dev/null && exec 3>&- && return 0; return 1; }

start_server() {  # gpu -> порт BASE_PORT+gpu; уже поднятый переиспользуется
  local g=$1
  local p=$((BASE_PORT + g))
  port_alive $p && { echo "сервер :$p (GPU$g) уже слушает — переиспользую"; return 0; }
  ( source $CONDA/etc/profile.d/conda.sh; conda activate $CONDA/envs/internvla
    export PYTHONPATH="$R/InternVLA-M1:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="$g"
    export HF_HOME=/workspace/moskalenko/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 TOKENIZERS_PARALLELISM=false
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
    cd $R/InternVLA-M1
    exec setsid nohup python -u deployment/model_server/server_policy_M1.py \
      --ckpt_path "$CKPT" --port "$p" --use_bf16 ) < /dev/null > "$LOGS/_ivla_server_$(hostname | cut -c1-24)_gpu$g.log" 2>&1 &
  for i in $(seq 1 60); do sleep 10; port_alive $p && { echo "сервер :$p (GPU$g) поднялся за ~$((i*10)) с"; sleep 10; return 0; }; done
  echo "СЕРВЕР :$p (GPU$g) НЕ ПОДНЯЛСЯ за 10 мин — см. $LOGS/_ivla_server_*_gpu$g.log"; return 1
}

chain() {  # gpu  "ASSETS:ORDER:START0:END" ...
  local gpu=$1; shift
  local cmds=""
  for spec in "$@"; do
    IFS=: read -r a o s e <<< "$spec"
    cmds+="INTERNVLA_PORT=$((BASE_PORT + gpu)) VLA=internvla VLA_ARGS='--vla-path $CKPT' ASSETS=$a ORDER=$o GPU=$gpu START0=$s END=$e DRY=$DRY bash $RUN; "
  done
  if [ "$DRY" = 1 ]; then bash -c "$cmds"; return; fi
  start_server $gpu || return 1
  setsid -f nohup bash -c "$cmds" < /dev/null > "$LOGS/chain_${PROG}_internvla_${NODE}_gpu${gpu}.out" 2>&1
  echo "GPU$gpu: $*"
}

case "$NODE" in
  SMOKE)
    start_server 0 || exit 1
    source $HOME/ws/env_bohr.sh
    export INTERNVLA_PORT=$BASE_PORT CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
    export BOARD_XY_SCALE=1.2 A2A_TILE_Y=0.14 A2A_TRAJ_LOG=1 A2A_SAVE_VIDEO=0
    rm -rf $A/outputs/smoke-internvla-noswap
    python -u -m simpler_env.eval --vla internvla --vla-path "$CKPT" --assets pairs_$CS \
      --start-id 0 --count 4 --shard-size 4 --buffer-inferbatch 4 --name smoke-internvla-noswap \
      < /dev/null > $LOGS/_ivla_smoke.log 2>&1
    echo "SMOKE rc=$?"; ls $A/outputs/smoke-internvla-noswap/glob/vis_0_test/ 2>&1
    ;;
  A)
    chain 0 focus_$CS:noswap:0:2496    pairs_$CS:noswap:0:480
    chain 1 focus_$CS:noswap:2496:5000 pairs_$CS:noswap:480:1000
    chain 2 focus_$CS:swap:0:2496      pairs_$CS:swap:0:480
    chain 3 focus_$CS:swap:2496:5000   pairs_$CS:swap:480:1000
    ;;
  C2)
    chain 0 visbias_$CS:noswap:0:2496
    chain 1 visbias_$CS:noswap:2496:5000
    ;;
  C3) chain 0 visbias_$CS:swap:0:2496 ;;
  C4) chain 0 visbias_$CS:swap:2496:5000 ;;
  *) echo "NODE=A|C2|C3|C4|SMOKE"; exit 1;;
esac
[ "$DRY" = 1 ] || { sleep 3; echo "клиентов simpler_env.eval: $(pgrep -cf simpler_env.eval), серверов: $(pgrep -cf server_policy_M1)"; }
