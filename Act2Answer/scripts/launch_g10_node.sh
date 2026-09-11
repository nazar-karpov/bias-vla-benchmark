#!/usr/bin/env bash
# Раскладка g10 по 8 картам двух нод (общий /workspace):
#   NODE=A (h100q):  4 карты — FOCUS обе половины × оба порядка, следом PAIRS теми же картами
#   NODE=B (h100q2): 4 карты — VisBias обе половины × оба порядка
# Каждая карта = одна цепочка (setsid nohup), внутри run_g10_magma.sh последовательно.
# Половины кратны шарду 48: [0,2496) и [2496,5000); PAIRS [0,480) и [480,1000).
#
#   NODE=A VLA=magma ./launch_g10_node.sh          # DRY=1 — только показать, откуда продолжит
set -u
NODE=${NODE:?NODE=A|B}
VLA=${VLA:-magma}
DRY=${DRY:-0}
R=$(dirname "$(readlink -f "$0")")/run_g10_magma.sh
LOGS=$HOME/ws

chain() {  # gpu  "ASSETS:ORDER:START0:END" ...
  local gpu=$1; shift
  local cmds=""
  for spec in "$@"; do
    IFS=: read -r a o s e <<< "$spec"
    cmds+="VLA=$VLA ASSETS=$a ORDER=$o GPU=$gpu START0=$s END=$e DRY=$DRY bash $R; "
  done
  if [ "$DRY" = 1 ]; then
    bash -c "$cmds"
  else
    setsid -f nohup bash -c "$cmds" < /dev/null > "$LOGS/chain_g10_${VLA}_${NODE}_gpu${gpu}.out" 2>&1
    echo "GPU$gpu: $*"
  fi
}

case "$NODE" in
  A)
    chain 0 focus_g10:noswap:0:2496    pairs_g10:noswap:0:480
    chain 1 focus_g10:noswap:2496:5000 pairs_g10:noswap:480:1000
    chain 2 focus_g10:swap:0:2496      pairs_g10:swap:0:480
    chain 3 focus_g10:swap:2496:5000   pairs_g10:swap:480:1000
    ;;
  B)
    # h100q2 (11.09.2026): на картах 1 и 3 заклинил Vulkan-рендер (даже 64×64 виснет),
    # CUDA-счёт на них жив. Поэтому пары карт: процессу видны две, run.py кладёт
    # симулятор+рендер на первую видимую (здоровую 0/2), модель — на вторую (1/3).
    # По два процесса на пару: [0,2496) и [2496,5000) на каждый порядок.
    chain 0,1 visbias_g10:noswap:0:2496
    chain 0,1 visbias_g10:noswap:2496:5000
    chain 2,3 visbias_g10:swap:0:2496
    chain 2,3 visbias_g10:swap:2496:5000
    ;;
  *) echo "NODE=A|B"; exit 1;;
esac
[ "$DRY" = 1 ] || { sleep 3; echo "запущено процессов simpler_env.eval: $(pgrep -cf simpler_env.eval)"; }
