#!/usr/bin/env bash
# Прогон Magma по десяти вопросам (5 осей × 2 полярности) на раскладке Андрея (1.2, ±0.14).
# Один процесс = один кардсет × один порядок. Порядки разводим по картам: noswap на 0, swap на 1.
#
#   ASSETS=focus_q10 ORDER=noswap GPU=0 ./run_q10_magma.sh
#   DRY=1 ASSETS=visbias_q10 ORDER=swap ./run_q10_magma.sh    # только показать, откуда продолжит
#
# Модель грузится ОДИН раз, среды пересоздаются шардами по SHARD (лимит GPU-камер SAPIEN ~50).
#
# ПРОДОЛЖЕНИЕ: старт вычисляется сам — по сплошному префиксу уже готовых шардов (у готового есть
# glob/vis_0_test/stats.yaml). Имена шардов абсолютные (q10-<assets>-<order>-s<индекс>), поэтому
# продолжение дописывает тот же ряд, а не начинает заново.
set -u
ASSETS=${ASSETS:?нужен ASSETS}
ORDER=${ORDER:-noswap}
GPU=${GPU:-0}
SHARD=${SHARD:-48}
DRY=${DRY:-0}

# Мягкий лимит дескрипторов на ноде 1024 — SAPIEN на 48 средах его выбирает и падает
# ("Too many open files", следом Vulkan ErrorOutOfHostMemory). Жёсткий лимит миллион.
ulimit -n 65536 || echo "ВНИМАНИЕ: не удалось поднять ulimit -n"

source $HOME/ws/env_bohr.sh
export CUDA_VISIBLE_DEVICES=$GPU
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BOARD_XY_SCALE=1.2 A2A_TILE_Y=0.14      # раскладка, выбранная Андреем
export A2A_TRAJ_LOG=1                          # нужен для непрерывного pull
export A2A_SAVE_VIDEO=0                        # видео на 126k эпизодов не нужно
export MAGMA_MAX_NEW_TOKENS=${MAGMA_MAX_NEW_TOKENS:-1000}   # 1000 = до EOS, исходное поведение
unset A2A_BASE_IMG                             # картинку не режем (решение Назара)

extra=()
[ "$ORDER" = swap ] && extra=(--do-swap)

# --count 0 ("все") ломает шардирование — проверка в eval.py написана как `shard_size < count`,
# при нуле она ложна, и поднимается разом весь кардсет. Поэтому число эпизодов считаем явно.
TOTAL=$(python - "$REPO_ROOT/ManiSkill/mani_skill/assets/carrot/$ASSETS/pairs.json" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1]))))
PY
)
NAME="q10-${ASSETS}-${ORDER}"
START=$(python - "$REPO_ROOT/outputs" "$NAME" "$SHARD" <<'PY'
import os, sys
out, name, shard = sys.argv[1], sys.argv[2], int(sys.argv[3])
k = 0
while os.path.exists(os.path.join(out, f"{name}-s{k * shard}", "glob", "vis_0_test", "stats.yaml")):
    k += 1
print(k * shard)
PY
)
END=${END:-$TOTAL}                        # точечный пересчёт: обработать только [START, END)
REMAIN=$(( END - START ))
LOG=$HOME/ws/logs_q10_${ASSETS}_${ORDER}.log

# В сухом режиме в боевой лог не пишем — иначе засорим историю прогона строками START без DONE.
if [ "$REMAIN" -le 0 ]; then
  MSG="SKIP_Q10 $(date -u) assets=$ASSETS order=$ORDER — уже готово ($TOTAL эпизодов)"
  if [ "$DRY" = 1 ]; then echo "DRY $MSG"; else echo "$MSG" | tee -a "$LOG"; fi
  exit 0
fi
MSG="START_Q10 $(date -u) assets=$ASSETS order=$ORDER gpu=$GPU shard=$SHARD эпизодов=$TOTAL старт=$START конец=$END осталось=$REMAIN max_new_tokens=$MAGMA_MAX_NEW_TOKENS"
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
python -u -m simpler_env.eval --vla magma --assets "$ASSETS" \
  --start-id "$START" --count "$REMAIN" --shard-size "$SHARD" --buffer-inferbatch "$SHARD" \
  "${NM[@]}" "${extra[@]}" >> "$LOG" 2>&1
rc=$?
echo "DONE_Q10 $(date -u) rc=$rc assets=$ASSETS order=$ORDER" | tee -a "$LOG"
