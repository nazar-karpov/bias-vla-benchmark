# $1 = vla — метрики программы q193 (банк FairACT: 193 вопроса × PAIRS 100 gender + 100 skin_color).
# 1) по вопросам (--topic-col question_id): pull vs 0 → *_abs.csv, дискретный → *_discrete.csv
# 2) по осям (axis): Δ pos−neg для пар полюсов PAIRS (pole_a/pole_b) → q193ax_*.csv
V=$1; export TZ=UTC REPO_ROOT=/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer
P=/workspace/moskalenko/conda/envs/magma_act2answer/bin/python
R=/workspace/moskalenko/bias-vla-benchmark-main; A=$R/Act2Answer; mkdir -p $R/metrics
echo "START $V $(date)"
for a in pairs_q193g pairs_q193e; do
  echo "#### валидатор $a"; nice -n 10 $P $A/scripts/check_traj_log.py --run q193-$V-$a --sep s 2>&1 | grep "шардов\|схвачен\|❌" | head -6
done
for a in pairs_q193g pairs_q193e; do
  echo "#### all_metrics по вопросам $a"
  nice -n 10 $P $A/scripts/all_metrics.py --run q193-$V-$a --assets $a --sep s --topic-col question_id --csv $R/metrics/q193_${V}_$a.csv 2>&1 | tail -40
  echo "#### all_metrics по осям $a"
  nice -n 10 $P $A/scripts/all_metrics.py --run q193-$V-$a --assets $a --sep s --csv $R/metrics/q193ax_${V}_$a.csv 2>&1 | tail -40
done
echo "METRICS_DONE $(date)"
