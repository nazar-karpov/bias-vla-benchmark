#!/usr/bin/env bash
# Контроль к А/Б лимита генерации: НОРМАЛЬНЫЙ вопрос («нет образования», индексы 9000..9047).
# Ожидание: без «убегающей» генерации лимит 8 даёт бит-в-бит те же действия, что и 1000.
set -u
ulimit -n 65536
source $HOME/ws/env_bohr.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0} TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BOARD_XY_SCALE=1.2 A2A_TILE_Y=0.14 A2A_TRAJ_LOG=1 A2A_SAVE_VIDEO=0 MAGMA_SAMPLE=0
unset A2A_BASE_IMG
cd $REPO_ROOT/SimplerEnv
for mt in 1000 8; do
  t0=$(date +%s); echo "### ctrl MAGMA_MAX_NEW_TOKENS=$mt старт $(date -u +%H:%M:%S)"
  MAGMA_MAX_NEW_TOKENS=$mt python -u -m simpler_env.eval --vla magma --assets visbias_q10 \
    --start-id 9000 --count 48 --shard-size 48 --buffer-inferbatch 48 --name mnt${mt}-noedu 2>&1 \
    | grep -E "EVAL_DONE_SECONDS|FINAL_STATS|Traceback|Error|out of memory"
  echo "### ctrl MAGMA_MAX_NEW_TOKENS=$mt готово за $(( $(date +%s)-t0 )) с"
done
echo MNT_CTRL_DONE
