#!/usr/bin/env bash
# Оркестратор программы q193f (09.10.2026): банк 193 вопроса на ВСЕХ парах FOCUS/VisBias (по 579 000 эп. на
# условие, порядок раундами — см. gen_q193f_cardset.py). Основа — драйвер q193 v3 (учёт свободной памяти).
# Отличия: префикс q193f, N=579000, готовность по указателю конца сплошного префикса шардов (полный перебор
# 12 тыс. stats.yaml на каждой итерации тяжёл для NFS), модели по умолчанию — только Magma.
# Нода h100q5 (job 88d2b211, 4×H100) общая: студент из tmux 29i_kos учит модель на тех же картах (до 41 ГБ на карту).
#
# Карта g всегда гоняет своё условие COND[g] (pairs_q193g/e × noswap/swap, по 19 300 эп.). Порядок моделей —
# приоритет: magma → internvla → xiaomi → gr00t → spatialvla. Когда на карте ничего нашего не идёт:
#   * первая по приоритету модель с недоделанным условием, если хватает свободной памяти (NEED, ГБ) — полный
#     диапазон (run_g10_magma.sh сам продолжит с первого недостающего шарда);
#   * если ей памяти не хватает (чужая задача) — ПОДМЕНА: следующая по приоритету модель, которой хватает, но
#     кусками по CHUNK шардов (после куска планировщик решает заново → карта вернётся к старшей модели, как
#     только освободится память);
#   * наши простаивающие серверы других моделей на карте перед запуском гасятся (их память учитывается как свободная).
# Падение по OOM → пауза OOM_WAIT для этой пары (карта, модель), попыткой не считается. Прочие падения — попытки;
# MAX_TRIES подряд без прогресса → модель на этой карте пропускается (❌). Зависание (лог молчит HANG_MIN мин) →
# kill клиента по проверенному pid. Чужие процессы не трогаем никогда.
# Метрики модели — когда все 4 её условия готовы (_q193_metrics.sh, в фоне).
#
#   (setsid nohup bash ~/ws/_q193f_driver.sh >> ~/ws/_q193f_driver_h100q7.log 2>&1 < /dev/null &)
#   DRY=1 bash ~/ws/_q193f_driver.sh   — одна итерация: показать решения, ничего не запускать
#   touch ~/ws/_q193f_driver.stop         — выйти после текущей итерации (прогоны не трогает)
set -u
export TZ=UTC
DRY=${DRY:-0}
PIDF=/tmp/_q193f_driver.pid
if [ "$DRY" != 1 ]; then
  if [ -f $PIDF ]; then
    op=$(cat $PIDF)
    if [ -n "$op" ] && [ "$op" != $$ ] && tr '\0' ' ' 2>/dev/null < /proc/$op/cmdline | grep -q "_q193f_driver"; then
      echo "$(date) драйвер уже работает на этой ноде (pid $op) — выхожу"; exit 1
    fi
  fi
  echo $$ > $PIDF
fi

R=/workspace/moskalenko/bias-vla-benchmark-main; A=$R/Act2Answer; S=$A/scripts; OUT=$A/outputs
WS=$HOME/ws
MODELS=(${MODELS:-magma})
declare -A NEED=([magma]=50 [internvla]=28 [xiaomi]=32 [gr00t]=28 [spatialvla]=32)   # ГБ свободных для старта
declare -A SRV_GB=([internvla]=10 [xiaomi]=14 [gr00t]=10)                            # наш сервер на карте, ГБ
# Две ноды по 4 карты: каждой — свои 4 условия (assets:order), переменная COND_LIST (порядок = GPU0..3).
COND=(${COND_LIST:?нужен COND_LIST="focus_q193fa:noswap focus_q193fa:swap ..."})
NODE_TAG=${NODE_TAG:-$(hostname | cut -c9-16)}
SH=48
declare -A NA   # эпизодов в кардсете (длина pairs.json), считается один раз
for c in "${COND[@]}"; do a=${c%%:*}
  [ -n "${NA[$a]:-}" ] || NA[$a]=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$A/ManiSkill/mani_skill/assets/carrot/$a/pairs.json")
done
HANG_MIN=${HANG_MIN:-30}; MAX_TRIES=${MAX_TRIES:-6}; POLL=${POLL:-120}
OOM_WAIT=${OOM_WAIT:-600}; CHUNK=${CHUNK:-24}
STOPF=$WS/_q193f_driver_${NODE_TAG}.stop
[ "$DRY" = 1 ] || rm -f "$STOPF"
declare -A TRIES LASTMISS OOMWAIT OOMSEEN SKIP METRICS_STARTED FM

log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }

# Указатель конца сплошного префикса готовых шардов (FM) обновляется БЕЗ подоболочки: в $(…) присваивание
# потерялось бы, и каждый вызов перебирал бы 12 тыс. stats.yaml с нуля. fm_update кладёт результат в FMK,
# nmiss — число недоделанных шардов от первой дыры до конца кардсета.
fm_update() {  # v a o
  local key="$1:$2:$3" k
  k=${FM[$key]:-0}
  local n=${NA[$2]}
  while [ $k -lt $n ] && [ -f "$OUT/q193f-$1-$2-$3-s$k/glob/vis_0_test/stats.yaml" ]; do k=$((k+SH)); done
  FM[$key]=$k; FMK=$k; NMISS=$(( (n - k + SH - 1) / SH )); NSH=$(( (n + SH - 1) / SH ))
}
# цепочка из одного спека: `bash -c "VLA=… bash run_g10_magma.sh; "` делает exec — ищем run_g10_magma.sh по environ
chain_pid() {  # v a o g
  local p e
  for p in $(pgrep -f -- "[r]un_g10_magma.sh"); do
    e=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null) || continue
    grep -qx "VLA=$1" <<< "$e" && grep -qx "ASSETS=$2" <<< "$e" && grep -qx "ORDER=$3" <<< "$e" \
      && grep -qx "GPU=$4" <<< "$e" && { echo $p; return 0; }
  done
}
client_pid() { pgrep -f -- "[s]impler_env.eval --vla $1 --assets $2 .*--name q193f-$1-$2-$3" | head -1; }
logf()       { echo "$WS/logs_q193f_$1_$2_$3_$4.log"; }       # v a o start0
lastlog()    { ls -t $WS/logs_q193f_$1_$2_$3_*.log 2>/dev/null | head -1; }   # свежий лог (полный диапазон или кусок)

srv_port() { case $2 in internvla) echo $((10093 + $1));; gr00t) echo $((5500 + $1 * 10));; xiaomi) echo $((10000 + $1 * 10));; esac; }
srv_alive() { local p; p=$(srv_port $1 $2); [ -n "$p" ] && (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null; }
stop_srv() {  # g v — гасим сервер этой модели на этой карте (по порту в argv, с проверкой cmdline)
  local g=$1 v=$2 p; p=$(srv_port $g $v); [ -z "$p" ] && return 0
  for par in $(pgrep -f -- "--port $p( |$)"); do
    [ "$par" -gt 1 ] 2>/dev/null || continue
    if tr '\0' ' ' < /proc/$par/cmdline 2>/dev/null | grep -qE "server_policy_M1|run_gr00t_server|xiaomi_server_batch|deploy/server.py"; then
      for c in $(pgrep -P $par); do [ "$c" -gt 1 ] && kill $c 2>/dev/null; done
      kill $par 2>/dev/null; log "GPU$g: погашен сервер $v :$p (pid $par)"
    fi
  done
}
kill_client() {  # pid v a o — только если это наш клиент
  local pid=$1
  [ "$pid" -gt 1 ] 2>/dev/null || return 1
  tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -q "simpler_env.eval --vla $2 --assets $3 .*--name q193f-$2-$3-$4" || return 1
  kill $pid 2>/dev/null; sleep 5; kill -0 $pid 2>/dev/null && kill -9 $pid 2>/dev/null
  return 0
}
gpu_util() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $1 2>/dev/null | tr -d ' '; }
free_gb()  { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i $1 2>/dev/null \
               | awk -F', ' '{printf "%d", ($1 - $2) / 1024}'; }
oom_in_log() {  # файл лога — последний запуск (после последнего START_G10) упал по памяти?
  awk 'BEGIN{IGNORECASE=1} /START_G10/{f=0} /out of memory|OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY|DONE_G10.*rc=135/{f=1} END{exit !f}' "$1" 2>/dev/null
}

launch_gpu() {  # g v start end
  local g=$1 v=$2 s=$3 e=$4 a=${COND[$1]%%:*} o=${COND[$1]##*:}
  local spec="$a:$o:$s:$e" key="$1:$2"
  log "GPU$g: ЗАПУСК $v $spec (попытка $(( ${TRIES[$key]:-0} + 1 )), свободно $(free_gb $g) ГБ)"
  [ "$DRY" = 1 ] && return 0
  case $v in
    magma)       env PROG=q193f VLA=magma NODE=CUSTOM "CUSTOM$g=$spec" bash $S/launch_g10_node.sh ;;
    spatialvla)  env A2A_ENV=spatialvla_act2answer PROG=q193f VLA=spatialvla NODE=CUSTOM "CUSTOM$g=$spec" bash $S/launch_g10_node.sh ;;
    internvla)   env PROG=q193f NODE=CUSTOM "CUSTOM$g=$spec" bash $S/launch_g10_internvla.sh ;;
    gr00t|xiaomi) env PROG=q193f VLA=$v NODE=CUSTOM "CUSTOM$g=$spec" bash $S/launch_g10_srv.sh ;;
  esac 2>&1 | sed "s/^/    [launch GPU$g] /"
  TRIES[$key]=$(( ${TRIES[$key]:-0} + 1 ))
  # разнесённый старт (одновременный старт вешает клиентов, 11.09): ждём первых шагов симуляции в логе, ≤12 мин
  local i u lf; lf=$(logf $v $a $o $s)
  for i in $(seq 1 48); do
    sleep 15
    if awk '/START_G10/{f=0} /elapsed_steps/{f=1} END{exit !f}' "$lf" 2>/dev/null; then
      u=$(gpu_util $g); log "GPU$g: $v симулирует (util ${u}%) через ~$((i*15)) с"; return 0
    fi
    [ -z "$(chain_pid $v $a $o $g)" ] && { log "GPU$g: цепочка $v уже вышла (см. $lf)"; return 1; }
  done
  log "GPU$g: за 12 мин нет шагов симуляции — пусть решает детектор зависаний"
}

log "СТАРТ драйвера q193f (все пары FOCUS/VisBias, учёт памяти), нода $NODE_TAG, условия: ${COND[*]}, эпизодов: $(for c in "${COND[@]}"; do a=${c%%:*}; printf "%s " ${NA[$a]}; done), модели: ${MODELS[*]}, pid $$, DRY=$DRY"
while :; do
  [ -f "$STOPF" ] && { log "STOP-файл — выхожу (прогоны не трогаю)"; exit 0; }
  status=""; alldone=1
  for g in 0 1 2 3; do
    a=${COND[$g]%%:*}; o=${COND[$g]##*:}
    # 1) идёт ли на карте наша цепочка (любая модель)?
    run=""
    for v in "${MODELS[@]}"; do [ -n "$(chain_pid $v $a $o $g)" ] && { run=$v; break; }; done
    if [ -n "$run" ]; then
      alldone=0
      cl=$(client_pid $run $a $o); lf=$(lastlog $run $a $o)
      if [ -n "$cl" ] && [ -n "$lf" ]; then
        age=$(( ( $(date +%s) - $(stat -c %Y "$lf") ) / 60 ))
        if [ $age -ge $HANG_MIN ]; then
          log "GPU$g: ⚠ $run $a $o — лог молчит $age мин, убиваю клиента $cl (цепочка выйдет, перезапущу)"
          [ "$DRY" = 1 ] || kill_client $cl $run $a $o || log "GPU$g: pid $cl не прошёл проверку cmdline — не трогаю"
        fi
      fi
      fm_update $run $a $o; status+=" | GPU$g $run осталось $NMISS/$NSH"
      continue
    fi
    # 2) на карте ничего нашего не идёт — выбираем модель
    now=$(date +%s); top=""; pick=""; pick_full=0; why=""
    for v in "${MODELS[@]}"; do
      key="$g:$v"
      [ -n "${SKIP[$key]:-}" ] && continue
      fm_update $v $a $o; m=$NMISS
      if [ "$m" -eq 0 ]; then srv_alive $g $v && { [ "$DRY" = 1 ] || stop_srv $g $v; }; continue; fi
      alldone=0
      # OOM в последнем запуске этого драйвера → пауза (раз на запуск), попыткой не считается
      if [ "${LASTMISS[$key]:--1}" -ge 0 ] && [ -z "${OOMSEEN[$key]:-}" ] && oom_in_log "$(lastlog $v $a $o)"; then
        OOMWAIT[$key]=$((now + OOM_WAIT)); OOMSEEN[$key]=1
        log "GPU$g: ⚠ $v $a $o упал по памяти (OOM) — эту модель на карте не трогаю $((OOM_WAIT / 60)) мин, попытка не считается"
      fi
      [ -z "$top" ] && top=$v
      if [ "$now" -lt "${OOMWAIT[$key]:-0}" ]; then why+=" $v:ждёт-после-OOM"; continue; fi
      # память: свободно + наши простаивающие серверы ДРУГИХ моделей на этой карте (их погасим)
      fr=$(free_gb $g); rec=0
      for w in internvla xiaomi gr00t; do [ "$w" != "$v" ] && srv_alive $g $w && rec=$((rec + SRV_GB[$w])); done
      if [ $((fr + rec)) -lt "${NEED[$v]}" ]; then why+=" $v:мало-памяти(${fr}+${rec}<${NEED[$v]})"; continue; fi
      pick=$v; [ "$v" = "$top" ] && pick_full=1
      break
    done
    if [ -z "$pick" ]; then
      [ -n "$top" ] && status+=" | GPU$g ждёт:$why" || status+=" | GPU$g — всё"
      continue
    fi
    key="$g:$pick"; fm_update $pick $a $o; m=$NMISS
    # прогресс с прошлого запуска → счётчик попыток с нуля; OOM-повтор попыткой не считается
    [ "${LASTMISS[$key]:--1}" -ge 0 ] && [ "$m" -lt "${LASTMISS[$key]}" ] && TRIES[$key]=0
    if [ -n "${OOMSEEN[$key]:-}" ]; then TRIES[$key]=$(( ${TRIES[$key]:-0} > 0 ? ${TRIES[$key]} - 1 : 0 )); unset "OOMSEEN[$key]"; fi
    if [ "${TRIES[$key]:-0}" -ge $MAX_TRIES ]; then
      log "GPU$g: ❌ $pick $a $o — $MAX_TRIES запусков без прогресса, осталось $m шардов; ПРОПУСКАЮ модель на карте"
      tail -5 "$(lastlog $pick $a $o)" 2>/dev/null | sed "s/^/    /"
      SKIP[$key]=1; [ "$DRY" = 1 ] || stop_srv $g $pick
      continue
    fi
    # гасим наши простаивающие серверы других моделей на карте
    for w in internvla xiaomi gr00t; do [ "$w" != "$pick" ] && srv_alive $g $w && { [ "$DRY" = 1 ] || stop_srv $g $w; }; done
    LASTMISS[$key]=$m
    if [ $pick_full = 1 ]; then
      launch_gpu $g $pick 0 ${NA[$a]}
    else
      fm_update $pick $a $o; s=$FMK; e=$(( s + CHUNK * SH )); [ $e -gt ${NA[$a]} ] && e=${NA[$a]}
      log "GPU$g: подмена — $top не помещается ($why), даю карте $pick кусок [$s,$e)"
      launch_gpu $g $pick $s $e
    fi
    status+=" | GPU$g $pick осталось $m/$NSH"
  done
  # метрики: модель, у которой готовы все 4 условия
  for v in "${MODELS[@]}"; do
    [ -n "${METRICS_STARTED[$v]:-}" ] && continue
    done4=1
    for g in 0 1 2 3; do a=${COND[$g]%%:*}; o=${COND[$g]##*:}; fm_update $v $a $o; [ $NMISS -eq 0 ] || { done4=0; break; }; done
    [ $done4 = 1 ] || continue
    METRICS_STARTED[$v]=1
    log "ГОТОВЫ условия ноды $NODE_TAG для $v (метрики — вручную после обеих нод: ~/ws/_q193f_metrics.sh $v)"
  done
  log "статус$status"
  [ "$DRY" = 1 ] && exit 0
  [ $alldone = 1 ] && { log "ALL_DONE q193f: все модели на всех картах"; exit 0; }
  sleep $POLL
done
