#!/usr/bin/env bash
# Гендерная программа g10 (12.09.2026): 10 вопросов (5 пар pos/neg) × 3 кардсета
# (focus_g10, visbias_g10, pairs_g10 — только tsv:gender) × 2 порядка, 5 моделей.
# Один процесс = один кардсет × один порядок × диапазон эпизодов [START0, END).
#
#   ASSETS=focus_g10 ORDER=noswap GPU=0 START0=0 END=2496 ./run_g10_magma.sh
#   DRY=1 ASSETS=visbias_g10 ORDER=swap START0=2496 ./run_g10_magma.sh   # откуда продолжит
#
# Имена шардов абсолютные: g10-<assets>-<order>-s<индекс эпизода>. ПРОДОЛЖЕНИЕ: старт = первый
# шард в [START0, END), у которого нет glob/vis_0_test/stats.yaml. Диапазон делить по границе
# шарда (кратно SHARD), остаток в конце кардсета eval.py режет сам (шард короче).
# Раскладка и лог как в q10: BOARD_XY_SCALE=1.2, A2A_TILE_Y=0.14, траектория пишется (в ней
# теперь action/action_t0/boardL_xy/gripper_q/qvel), видео нет, лимит генерации Magma 8.
set -u
ASSETS=${ASSETS:?нужен ASSETS}
ORDER=${ORDER:-noswap}
GPU=${GPU:-0}
SHARD=${SHARD:-48}
START0=${START0:-0}
DRY=${DRY:-0}
VLA=${VLA:-magma}

ulimit -n 65536 || echo "ВНИМАНИЕ: не удалось поднять ulimit -n"

source $HOME/ws/env_bohr.sh                    # REPO_ROOT, HF_HOME, MS_ASSET_DIR, conda, cd SimplerEnv
export CUDA_VISIBLE_DEVICES=$GPU
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BOARD_XY_SCALE=1.2 A2A_TILE_Y=0.14
export A2A_TRAJ_LOG=1 A2A_SAVE_VIDEO=0
export MAGMA_MAX_NEW_TOKENS=${MAGMA_MAX_NEW_TOKENS:-8}
unset A2A_BASE_IMG

extra=()
[ "$ORDER" = swap ] && extra=(--do-swap)

TOTAL=$(python - "$REPO_ROOT/ManiSkill/mani_skill/assets/carrot/$ASSETS/pairs.json" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1]))))
PY
)
END=${END:-$TOTAL}
NAME="g10-${ASSETS}-${ORDER}"
START=$(python - "$REPO_ROOT/outputs" "$NAME" "$SHARD" "$START0" "$END" <<'PY'
import os, sys
out, name, shard, s0, end = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
k = s0
while k < end and os.path.exists(os.path.join(out, f"{name}-s{k}", "glob", "vis_0_test", "stats.yaml")):
    k += shard
print(min(k, end))
PY
)
REMAIN=$(( END - START ))
LOG=$HOME/ws/logs_g10_${VLA}_${ASSETS}_${ORDER}_${START0}.log

if [ "$REMAIN" -le 0 ]; then
  MSG="SKIP_G10 $(date -u) vla=$VLA assets=$ASSETS order=$ORDER [$START0,$END) — готово"
  if [ "$DRY" = 1 ]; then echo "DRY $MSG"; else echo "$MSG" | tee -a "$LOG"; fi
  exit 0
fi
MSG="START_G10 $(date -u) vla=$VLA assets=$ASSETS order=$ORDER gpu=$GPU shard=$SHARD диапазон=[$START0,$END) старт=$START осталось=$REMAIN max_new_tokens=$MAGMA_MAX_NEW_TOKENS repo=$(git -C $REPO_ROOT rev-parse --short HEAD)"
if [ "$DRY" = 1 ]; then echo "DRY $MSG"; exit 0; fi
echo "$MSG" | tee -a "$LOG"

# Если осталось не больше одного шарда, eval.py уходит в нешардированный режим и назвал бы папку
# без суффикса -s<индекс>. Даём имя с суффиксом сами, чтобы ряд шардов остался сплошным.
if [ "$REMAIN" -gt "$SHARD" ]; then
  NM=(--name "$NAME")
else
  NM=(--name "${NAME}-s${START}")
fi

cd $REPO_ROOT/SimplerEnv
python -u -m simpler_env.eval --vla "$VLA" --assets "$ASSETS" \
  --start-id "$START" --count "$REMAIN" --shard-size "$SHARD" --buffer-inferbatch "$SHARD" \
  "${NM[@]}" "${extra[@]}" < /dev/null >> "$LOG" 2>&1
rc=$?
echo "DONE_G10 $(date -u) rc=$rc vla=$VLA assets=$ASSETS order=$ORDER [$START0,$END)" | tee -a "$LOG"
exit $rc
