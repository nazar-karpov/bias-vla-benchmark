#!/usr/bin/env bash
# ВНИМАНИЕ: A2A_CUBE_MASS и A2A_CUBE_MAX_DEPEN из среды убраны — проверка опровергла гипотезу
# о «микрограммовом кубе» (_mass игнорируется SAPIEN), см. docs/JOURNAL.md 2026-09-10.
# Проверка физики куба на тех же 48 эпизодах и seed, что у vid-pairs_q10-s0 (исходная физика).
set -u
ulimit -n 65536
source $HOME/ws/env_bohr.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0} TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BOARD_XY_SCALE=1.2 A2A_TILE_Y=0.14 A2A_TRAJ_LOG=1 A2A_SAVE_VIDEO=1
unset A2A_BASE_IMG
cd $REPO_ROOT/SimplerEnv
run() {
  tag=$1; shift
  echo "### $tag старт $(date -u +%H:%M:%S)"
  env "$@" python -u -m simpler_env.eval --vla magma --assets pairs_q10 \
    --start-id 0 --count 48 --shard-size 48 --buffer-inferbatch 48 --name "vid-pairs_q10-s0-$tag" \
    > $HOME/ws/logs_phys_$tag.log 2>&1
  echo "### $tag готово rc=$? $(date -u +%H:%M:%S)"; grep -E "FINAL_STATS|Traceback|Error" $HOME/ws/logs_phys_$tag.log | tail -3
}
run kin-mass0.02 A2A_TILE_BODY=kinematic A2A_CUBE_MASS=0.02
run kin-depen1 A2A_TILE_BODY=kinematic A2A_CUBE_MAX_DEPEN=1.0
echo PHYS_DONE
