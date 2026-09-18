#!/usr/bin/env bash
# Демо-ролики к видео статьи FairACT: перепрогон демо-кардсета gifdemo_<vla> (make_gif_demo_cardset.py)
# в обоих порядках с записью «чистого» видео (A2A_VIDEO_CLEAN=1: кадры камеры 640×480 без подписи и
# апскейла, качество 9 из 10) и полной траекторией (traj.npz). Условия сцены — как в программах g10/e10/x3:
# BOARD_XY_SCALE=1.2, A2A_TILE_Y=0.14, 80 шагов.
#
#   VLA=magma     GPU=1 BUF=20 bash Act2Answer/scripts/record_gif_demos.sh
#   VLA=internvla GPU=0 BUF=24 INTERNVLA_PORT=10093 bash Act2Answer/scripts/record_gif_demos.sh   # сервер уже поднят
#   ASSETS=gifdemo_magma_w4 START=36 GPU=1 VLA=magma BUF=20 bash …   # кусок кардсета [START, START+COUNT)
#
# Выход: Act2Answer/outputs/gif-<кардсет без «gifdemo_»>-<order>[-s<start>]/glob/vis_0_test/{video_<i>-s_<succ>.mp4,
# stats.yaml, traj.npz}; вторая волна: ASSETS=gifdemo_magma_w2 → gif-magma_w2-<order>-s*.
set -u
VLA=${VLA:?VLA=magma|internvla}
ASSETS=${ASSETS:-gifdemo_$VLA}
GPU=${GPU:-0}
BUF=${BUF:-16}
ORDERS=${ORDERS:-"noswap swap"}
R=$HOME/ws/bias-vla-benchmark-main
A=$R/Act2Answer
source $HOME/ws/env_bohr.sh    # conda magma_act2answer, REPO_ROOT, HF_HOME, MS_ASSET_DIR; cwd = SimplerEnv
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BOARD_XY_SCALE=1.2 A2A_TILE_Y=0.14 A2A_TRAJ_LOG=1 MAGMA_MAX_NEW_TOKENS=8
export A2A_SAVE_VIDEO=1 A2A_VIDEO_CLEAN=1 A2A_VIDEO_QUALITY=${A2A_VIDEO_QUALITY:-9}
unset A2A_BASE_IMG
N=$(python -c "import json; print(len(json.load(open('$A/ManiSkill/mani_skill/assets/carrot/$ASSETS/pairs.json'))))")
START=${START:-0}; COUNT=${COUNT:-$((N - START))}   # поддиапазон кардсета (две карты — два куска)
VLA_ARGS=()
if [ "$VLA" = internvla ]; then
  export INTERNVLA_HOST=127.0.0.1 INTERNVLA_PORT=${INTERNVLA_PORT:-10093}
  VLA_ARGS=(--vla-path "$R/internvla_ckpt/InternVLA-M1-Pretrain-RT-1-Bridge/checkpoints/steps_50000_pytorch_model.pt")
fi
for ORDER in $ORDERS; do
  extra=(); [ "$ORDER" = swap ] && extra=(--do-swap)
  LOG=$HOME/ws/logs_gif_${VLA}_${ORDER}.log
  echo "START_GIF $(date -u) vla=$VLA assets=$ASSETS order=$ORDER n=$N range=[$START,$((START + COUNT))) buf=$BUF gpu=$GPU repo=$(git -C $R rev-parse --short HEAD)" | tee -a "$LOG"
  # один шард eval называет без суффикса -s<start> — даём суффикс сами, чтобы имена были однородны
  NM="gif-${ASSETS#gifdemo_}-$ORDER"; [ "$COUNT" -le "$BUF" ] && NM="$NM-s$START"
  python -u -m simpler_env.eval --vla "$VLA" --assets "$ASSETS" --start-id "$START" --count "$COUNT" \
    --shard-size "$BUF" --buffer-inferbatch "$BUF" --name "$NM" "${extra[@]}" "${VLA_ARGS[@]}" \
    < /dev/null >> "$LOG" 2>&1
  echo "DONE_GIF $(date -u) rc=$? vla=$VLA order=$ORDER" | tee -a "$LOG"
done
