# статус q193f с любой ноды (общий NFS): драйверы обеих нод, прогресс 8 условий, раунд, скорость, срок
# bash ~/ws/_q193f_status.sh [vla]   (по умолчанию magma)
V=${1:-magma}; O=/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer/outputs; now=$(date +%s)
echo "$(date -u '+%m-%d %H:%M UTC')"
for t in h100q7 h100q8; do
  L=$HOME/ws/_q193f_driver_$t.log
  echo "== $t: $(grep 'статус' $L 2>/dev/null | tail -1 | cut -c16-230)"
  grep -E "❌|⚠|ПРОПУСКАЮ|ГОТОВЫ|ALL_DONE" $L 2>/dev/null | tail -3 | sed 's/^/   /'
done
for c in focus_q193fa-noswap focus_q193fa-swap focus_q193fb-noswap focus_q193fb-swap \
         visbias_q193fa-noswap visbias_q193fa-swap visbias_q193fb-noswap visbias_q193fb-swap; do
  n=0; recent=0
  for f in $O/q193f-$V-$c-s*/glob/vis_0_test/stats.yaml; do
    [ -f "$f" ] || continue; n=$((n+1)); [ $((now - $(stat -c %Y "$f"))) -le 3600 ] && recent=$((recent+1))
  done
  case $c in *fa-*) tot=6063; pr=1940;; *) tot=6000; pr=1920;; esac
  ep=$((n*48))
  if [ $ep -le $((50*pr)) ]; then r=$((ep/pr)); g=$((10*r)); e=$((10*r))
  else r=$((50 + (ep-50*pr)/(pr/2))); g=500; e=$((10*r)); fi
  eta=$([ $recent -gt 0 ] && echo "$(( (tot-n)/recent/24 )) сут $(( ((tot-n)/recent) % 24 )) ч" || echo "?")
  echo "  $c: $n/$tot шардов, раунд $r (≈$g пар пола + $e пар этничности на вопрос), за час $recent (~$((recent*48)) эп./ч), до конца $eta"
done
echo "  GPU h100q7/h100q8 смотреть nvidia-smi на ноде; клиентов $V на этой ноде: $(pgrep -f "[s]imple""r_env.eval --vla $V" | wc -l)"
