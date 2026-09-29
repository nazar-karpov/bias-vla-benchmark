# $1 = vla — метрики программы q193 (банк FairACT: 193 вопроса × PAIRS 100 gender + 100 skin_color).
# 1) по вопросам (--topic-col question_id): pull vs 0 → *_abs.csv, дискретный → *_discrete.csv
# 2) по осям (axis): Δ pos−neg для пар полюсов PAIRS (pole_a/pole_b) → q193ax_*.csv
# 3) TSV-выгрузка эпизодов (формат export_tsv_vlasim) в ОТДЕЛЬНУЮ папку export_tsv_vla_sim_q193
V=$1; export TZ=UTC REPO_ROOT=/workspace/moskalenko/bias-vla-benchmark-main/Act2Answer
P=/workspace/moskalenko/conda/envs/magma_act2answer/bin/python
R=/workspace/moskalenko/bias-vla-benchmark-main; A=$R/Act2Answer; mkdir -p $R/metrics
X=/workspace/moskalenko/ws_h100/export_tsv_vla_sim_q193
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
# экспортёр ДОПИСЫВАЕТ — папку модели перед выгрузкой стираем (повторный запуск не задвоит строки)
case $V in
  magma) M=microsoft_Magma-8B;; internvla) M=InternRobotics_InternVLA-M1;;
  xiaomi) M=XiaomiRobotics_Xiaomi-Robotics-0-SimplerEnv-WidowX;; gr00t) M=nvidia_GR00T-N1.7-SimplerEnv-Bridge;;
  spatialvla) M=IPEC-COMMUNITY_spatialvla-4b-224-pt;;
esac
rm -rf "$X/pairs/$M"
nice -n 10 $P $A/scripts/export_tsv_vlasim.py --out $X --vla $V --prog q193 2>&1 | tail -4
echo "EXPORT_DONE $(date)"
