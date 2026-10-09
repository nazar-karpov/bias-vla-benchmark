# $1 = vla — метрики программы q193f (банк 193 вопроса на всех парах FOCUS/VisBias: gender 500 + ethnicity 2500).
# Валидатор + all_metrics по вопросам (pull vs 0 → *_abs.csv, дискретный → *_discrete.csv) и по осям (Δ pos−neg
# для пар полюсов). Работает и на частично готовом прогоне (берёт то, что есть) — для промежуточных срезов.
V=$1; export TZ=UTC REPO_ROOT=/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer
P=/workspace/moskalenko/conda/envs/magma_act2answer/bin/python
R=/workspace/moskalenko/bias-vla-benchmark-main; A=$R/Act2Answer; mkdir -p $R/metrics
echo "START $V $(date)"
for a in focus_q193f visbias_q193f; do
  echo "#### валидатор $a"; nice -n 10 $P $A/scripts/check_traj_log.py --run q193f-$V-$a --sep s 2>&1 | grep "шардов\|схвачен\|структура\|❌" | head -8
done
for a in focus_q193f visbias_q193f; do
  echo "#### all_metrics по вопросам $a"
  nice -n 10 $P $A/scripts/all_metrics.py --run q193f-$V-$a --assets $a --sep s --topic-col question_id --csv $R/metrics/q193f_${V}_$a.csv 2>&1 | tail -20
  echo "#### all_metrics по осям $a"
  nice -n 10 $P $A/scripts/all_metrics.py --run q193f-$V-$a --assets $a --sep s --csv $R/metrics/q193fax_${V}_$a.csv 2>&1 | tail -20
done
echo "METRICS_DONE $(date)"
