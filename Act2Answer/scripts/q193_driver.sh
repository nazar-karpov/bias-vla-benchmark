#!/usr/bin/env bash
# Оркестратор программы q193 (29.09.2026, нода job 88d2b211 = h100q5, 4×H100, квота 48 ядер).
# Банк FairACT: 193 вопроса × PAIRS (100 gender + 100 skin_color) × 2 порядка = 77 200 эп. на модель.
# Карта g всегда гоняет своё условие COND[g] (по 19 300 эп.); модели идут по карте друг за другом
# (конвейер): как только условие g модели V готово, сервер V на карте g гасится и карта берёт
# следующую модель — хвоста с простоем нет. Упавшую цепочку перезапускает (run_g10_magma.sh
# продолжает с первого недостающего шарда), зависшего клиента (лог молчит > HANG_MIN мин) убивает
# по проверенному pid. Метрики модели — когда готовы все 4 её условия (_q193_metrics.sh, в фоне).
#
#   (setsid nohup bash ~/ws/_q193_driver.sh > ~/ws/_q193_driver_h100q5.log 2>&1 < /dev/null &)
#   touch ~/ws/_q193_driver.stop  — выйти после текущей итерации (прогоны не трогает)
set -u
export TZ=UTC
# защита от второго экземпляра (дубли сессий 13.09 убили PID 1): pid-файл на локальном /tmp +
# проверка cmdline. flock не годится — fd лока наследуют запущенные цепочки и держат его сутками.
PIDF=/tmp/_q193_driver.pid
if [ -f $PIDF ]; then
  op=$(cat $PIDF)
  if [ -n "$op" ] && [ "$op" != $$ ] && tr '\0' ' ' < /proc/$op/cmdline 2>/dev/null | grep -q "_q193_driver.sh"; then
    echo "$(date) драйвер уже работает на этой ноде (pid $op) — выхожу"; exit 1
  fi
fi
echo $$ > $PIDF

R=/workspace/moskalenko/bias-vla-benchmark-main; A=$R/Act2Answer; S=$A/scripts; OUT=$A/outputs
WS=$HOME/ws
MODELS=(${MODELS:-magma internvla xiaomi gr00t spatialvla})
COND=(pairs_q193g:noswap pairs_q193g:swap pairs_q193e:noswap pairs_q193e:swap)
N=19300; SH=48
HANG_MIN=${HANG_MIN:-30}; MAX_TRIES=${MAX_TRIES:-6}; POLL=${POLL:-120}
STOPF=$WS/_q193_driver.stop
rm -f "$STOPF"; echo $$ > $WS/_q193_driver.pid
declare -a MI TRIES LASTMISS
for g in 0 1 2 3; do MI[$g]=0; TRIES[$g]=0; LASTMISS[$g]=-1; done
declare -A METRICS_STARTED

log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }

missing() {  # vla assets order → число шардов без stats.yaml
  local v=$1 a=$2 o=$3 k=0 m=0
  while [ $k -lt $N ]; do
    [ -f "$OUT/q193-$v-$a-$o-s$k/glob/vis_0_test/stats.yaml" ] || m=$((m+1))
    k=$((k+SH))
  done
  echo $m
}
# Цепочка из одного спека: `bash -c "VLA=… ASSETS=… bash run_g10_magma.sh; "` делает exec, и присваивания
# из cmdline пропадают (29.09: первая версия по cmdline решила, что цепочки нет, и стартовала карты подряд).
# Поэтому ищем процесс run_g10_magma.sh и сверяем его окружение (/proc/<pid>/environ).
chain_pid() {  # v a o g
  local p e
  for p in $(pgrep -f -- "[r]un_g10_magma.sh"); do
    e=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null) || continue
    grep -qx "VLA=$1" <<< "$e" && grep -qx "ASSETS=$2" <<< "$e" && grep -qx "ORDER=$3" <<< "$e" \
      && grep -qx "GPU=$4" <<< "$e" && { echo $p; return 0; }
  done
}
client_pid() { pgrep -f -- "[s]impler_env.eval --vla $1 --assets $2 .*--name q193-$1-$2-$3" | head -1; }
logf()       { echo "$WS/logs_q193_$1_$2_$3_0.log"; }

srv_port() {  # g v → порт сервера политики (пусто у Magma/SpatialVLA)
  case $2 in internvla) echo $((10093 + $1));; gr00t) echo $((5500 + $1 * 10));; xiaomi) echo $((10000 + $1 * 10));; esac
}
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
  tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -q "simpler_env.eval --vla $2 --assets $3 .*--name q193-$2-$3-$4" || return 1
  kill $pid 2>/dev/null; sleep 5; kill -0 $pid 2>/dev/null && kill -9 $pid 2>/dev/null
  return 0
}
gpu_util() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $1 2>/dev/null | tr -d ' '; }

launch_gpu() {  # g v
  local g=$1 v=$2 a=${COND[$1]%%:*} o=${COND[$1]##*:}
  local spec="$a:$o:0:$N"
  log "GPU$g: ЗАПУСК $v $spec (попытка $((TRIES[$g]+1)))"
  case $v in
    magma)       env PROG=q193 VLA=magma NODE=CUSTOM "CUSTOM$g=$spec" bash $S/launch_g10_node.sh ;;
    spatialvla)  env A2A_ENV=spatialvla_act2answer PROG=q193 VLA=spatialvla NODE=CUSTOM "CUSTOM$g=$spec" bash $S/launch_g10_node.sh ;;
    internvla)   env PROG=q193 NODE=CUSTOM "CUSTOM$g=$spec" bash $S/launch_g10_internvla.sh ;;
    gr00t|xiaomi) env PROG=q193 VLA=$v NODE=CUSTOM "CUSTOM$g=$spec" bash $S/launch_g10_srv.sh ;;
  esac 2>&1 | sed "s/^/    [launch GPU$g] /"
  TRIES[$g]=$((TRIES[$g]+1))
  # разнесённый старт: одновременный старт 4 процессов вешает двоих намертво (11.09) —
  # ждём первых шагов симуляции (elapsed_steps после последнего START_G10 в логе; util карты
  # не годится — 100 % уже на загрузке весов), не дольше 12 мин, и только потом трогаем следующую
  local i u lf; lf=$(logf $v $a $o)
  for i in $(seq 1 48); do
    sleep 15
    if awk '/START_G10/{f=0} /elapsed_steps/{f=1} END{exit !f}' "$lf" 2>/dev/null; then
      u=$(gpu_util $g); log "GPU$g: $v симулирует (util ${u}%) через ~$((i*15)) с"; return 0
    fi
    [ -z "$(chain_pid $v $a $o $g)" ] && { log "GPU$g: цепочка $v уже вышла (см. $lf)"; return 1; }
  done
  log "GPU$g: за 12 мин нет шагов симуляции — пусть решает детектор зависаний"
}

log "СТАРТ драйвера q193, модели: ${MODELS[*]}, pid $$, узел $(hostname | cut -c1-24)"
while :; do
  [ -f "$STOPF" ] && { log "STOP-файл — выхожу (прогоны не трогаю)"; exit 0; }
  status=""
  for g in 0 1 2 3; do
    a=${COND[$g]%%:*}; o=${COND[$g]##*:}
    while [ ${MI[$g]} -lt ${#MODELS[@]} ]; do
      v=${MODELS[${MI[$g]}]}
      cp_=$(chain_pid $v $a $o $g)
      m=$(missing $v $a $o)
      if [ "$m" -eq 0 ] && [ -z "$cp_" ]; then          # условие готово → следующая модель
        log "GPU$g: ГОТОВО $v $a $o"
        stop_srv $g $v
        MI[$g]=$((MI[$g]+1)); TRIES[$g]=0; LASTMISS[$g]=-1
        continue
      fi
      if [ -z "$cp_" ]; then                              # цепочки нет, а шарды недоделаны → (пере)запуск
        [ "${LASTMISS[$g]}" -ge 0 ] && [ "$m" -lt "${LASTMISS[$g]}" ] && TRIES[$g]=0
        if [ ${TRIES[$g]} -ge $MAX_TRIES ]; then
          log "GPU$g: ❌ $v $a $o — $MAX_TRIES запусков без прогресса, осталось $m шардов; ПРОПУСКАЮ модель на карте"
          tail -5 "$(logf $v $a $o)" 2>/dev/null | sed "s/^/    /"
          stop_srv $g $v; MI[$g]=$((MI[$g]+1)); TRIES[$g]=0; LASTMISS[$g]=-1
          continue
        fi
        LASTMISS[$g]=$m
        launch_gpu $g $v
      else                                                # идёт: проверка зависания по логу
        cl=$(client_pid $v $a $o); lf=$(logf $v $a $o)
        if [ -n "$cl" ] && [ -f "$lf" ]; then
          age=$(( ( $(date +%s) - $(stat -c %Y "$lf") ) / 60 ))
          if [ $age -ge $HANG_MIN ]; then
            log "GPU$g: ⚠ $v $a $o — лог молчит $age мин, убиваю клиента $cl (цепочка выйдет, перезапущу)"
            kill_client $cl $v $a $o || log "GPU$g: pid $cl не прошёл проверку cmdline — не трогаю"
          fi
        fi
      fi
      status+=" | GPU$g $v осталось $m/$(( (N+SH-1)/SH ))"
      break
    done
    [ ${MI[$g]} -ge ${#MODELS[@]} ] && status+=" | GPU$g — всё"
  done
  # метрики: модель, которую прошли все 4 карты
  minmi=${MI[0]}; for g in 1 2 3; do [ ${MI[$g]} -lt $minmi ] && minmi=${MI[$g]}; done
  for ((j=0; j<minmi; j++)); do
    v=${MODELS[$j]}
    [ -n "${METRICS_STARTED[$v]:-}" ] && continue
    METRICS_STARTED[$v]=1
    log "МЕТРИКИ $v → $WS/_q193_metrics_$v.log"
    (setsid nohup bash $WS/_q193_metrics.sh $v > $WS/_q193_metrics_$v.log 2>&1 < /dev/null &)
  done
  log "статус$status"
  [ $minmi -ge ${#MODELS[@]} ] && { log "ALL_DONE q193: все модели на всех картах"; exit 0; }
  sleep $POLL
done
