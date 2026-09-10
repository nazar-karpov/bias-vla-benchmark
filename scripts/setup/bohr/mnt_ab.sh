#!/usr/bin/env bash
# А/Б лимита генерации Magma на медленном вопросе (докторская): 1000 против 8 токенов,
# жадный режим, боевая раскладка q10. Одни и те же 48 эпизодов в обоих плечах.
set -u
ulimit -n 65536
source $HOME/ws/env_bohr.sh
export CUDA_VISIBLE_DEVICES=${GPU:-1} TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BOARD_XY_SCALE=1.2 A2A_TILE_Y=0.14 A2A_TRAJ_LOG=1 A2A_SAVE_VIDEO=0 MAGMA_SAMPLE=0
unset A2A_BASE_IMG
cd $REPO_ROOT/SimplerEnv
for mt in 1000 8; do
  t0=$(date +%s); echo "### MAGMA_MAX_NEW_TOKENS=$mt старт $(date -u +%H:%M:%S)"
  MAGMA_MAX_NEW_TOKENS=$mt python -u -m simpler_env.eval --vla magma --assets visbias_q10 \
    --start-id 6000 --count 48 --shard-size 48 --buffer-inferbatch 48 --name mnt${mt}-doctorate 2>&1 \
    | grep -E "EVAL_DONE_SECONDS|FINAL_STATS|Traceback|Error|out of memory"
  echo "### MAGMA_MAX_NEW_TOKENS=$mt готово за $(( $(date +%s)-t0 )) с"
done
echo MNT_AB_DONE
