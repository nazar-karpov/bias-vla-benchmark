#!/usr/bin/env bash
# Прогон Magma по десяти вопросам (5 осей × 2 полярности) на раскладке Андрея (1.2, ±0.14).
# Один процесс = один кардсет × один порядок. Порядки разводим по картам: noswap на 0, swap на 1.
#
#   ASSETS=focus_q10 ORDER=noswap GPU=0 ./run_q10_magma.sh
#
# Модель грузится ОДИН раз, среды пересоздаются шардами по SHARD (лимит GPU-камер SAPIEN ~50).
set -u
ASSETS=${ASSETS:?нужен ASSETS}
ORDER=${ORDER:-noswap}
GPU=${GPU:-0}
SHARD=${SHARD:-48}

# Мягкий лимит дескрипторов на ноде 1024 — SAPIEN на 48 средах его выбирает и падает
# ("Too many open files", следом Vulkan ErrorOutOfHostMemory). Жёсткий лимит миллион.
ulimit -n 65536 || echo "ВНИМАНИЕ: не удалось поднять ulimit -n"

source $HOME/ws/env_bohr.sh
export CUDA_VISIBLE_DEVICES=$GPU
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BOARD_XY_SCALE=1.2 A2A_TILE_Y=0.14      # раскладка, выбранная Андреем
export A2A_TRAJ_LOG=1                          # нужен для непрерывного pull (первичная метрика)
export A2A_SAVE_VIDEO=0                        # видео на 126k эпизодов не нужно

extra=()
[ "$ORDER" = swap ] && extra=(--do-swap)

# ВАЖНО: --count 0 ("все") ломает шардирование — проверка в eval.py написана как
# `shard_size < count`, и при нуле она ложна, поэтому поднимается разом ВЕСЬ кардсет
# (тысячи сред -> тысячи камер -> "Unable to create GPU parallelized camera group"
# и "Too many open files"). Поэтому число эпизодов задаём явно.
TOTAL=$(python -c "import json;print(len(json.load(open('$REPO_ROOT/ManiSkill/mani_skill/assets/carrot/$ASSETS/pairs.json'))))")

LOG=$HOME/ws/logs_q10_${ASSETS}_${ORDER}.log
echo "START_Q10 $(date -u) assets=$ASSETS order=$ORDER gpu=$GPU shard=$SHARD эпизодов=$TOTAL" | tee -a "$LOG"
cd $REPO_ROOT/SimplerEnv
python -u -m simpler_env.eval --vla magma --assets "$ASSETS" \
  --start-id 0 --count "$TOTAL" --shard-size "$SHARD" --buffer-inferbatch "$SHARD" \
  --name "q10-${ASSETS}-${ORDER}" "${extra[@]}" >> "$LOG" 2>&1
echo "DONE_Q10 $(date -u) rc=$? assets=$ASSETS order=$ORDER" | tee -a "$LOG"
