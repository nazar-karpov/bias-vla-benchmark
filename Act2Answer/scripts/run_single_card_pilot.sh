#!/usr/bin/env bash
# SINGLE-CARD ПИЛОТ: ОДНА карточка на столе (A2A_SINGLE_TILE=1).
# Кардсет pairs_single_pilot: qkey=pilot (сильнейший эффект полного креста,
# стюардесса −40мм) — 200 плиток × pos/neg = 400 эп; noswap/swap = слот
# карточки (лев/прав), контрбаланс асимметрии камеры.
# Метрика — прогресс куба к карточке из traj.npz (A2A_TRAJ_LOG=1);
# дискретный канал (is_answered) пишется побочно.
# Резюмируемость: готовые шарды пропускаются, после падения перезапустить.
set -uo pipefail
R=/workspace/moskalenko/bias-vla-benchmark-main
A="$R/Act2Answer"
CONDA=/workspace/moskalenko/conda
export REPO_ROOT="$A" PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
export MS_ASSET_DIR=/workspace/moskalenko/maniskill_assets
export HF_HOME=/workspace/moskalenko/hf_cache
export TOKENIZERS_PARALLELISM=false
export A2A_SINGLE_TILE=1     # одиночный режим (см. put_on_in_scene_multi_v4)
export A2A_TRAJ_LOG=1        # траектории под непрерывную метрику
export A2A_SAVE_VIDEO=0
export BOARD_XY_SCALE=1.0    # масштаб 1.3 уже в model_db кардсета

ASSET="${ASSET:-pairs_single_pilot}"
NAME="${NAME:-single-pilot}"   # префикс имён шардов (для плеча B: single-pilot-cond)
TOTAL="${TOTAL:-400}"
SHARD="${SHARD:-100}"
PAR="${PAR:-4}"              # magma: ~35 ГБ VRAM/процесс, 2×H100 -> 4 клиента
LOG=/workspace/moskalenko/logs_single_card
mkdir -p "$LOG"
source "$CONDA/etc/profile.d/conda.sh"
conda activate "$CONDA/envs/magma_act2answer"
cd "$A/SimplerEnv" || exit 1

i=0
for lay in noswap swap; do
  sw=""; [ "$lay" = swap ] && sw="--do-swap"
  for ((s=0; s<TOTAL; s+=SHARD)); do
    name="${NAME}-magma-${lay}-sh${s}"
    [ -f "$A/outputs/$name/glob/vis_0_test/stats.yaml" ] && continue
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 20; done
    gpu=$(( i % 2 ))
    echo "[$(date -u +%H:%M:%S)] START $name gpu=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu \
      python -u -m simpler_env.eval --vla magma \
        --assets "$ASSET" --obj-set test --start-id "$s" --count "$SHARD" \
        --episode-len 80 --buffer-inferbatch 10 --buffer-minibatch -1 \
        --name "$name" $sw < /dev/null > "$LOG/$name.log" 2>&1 &
    i=$(( i + 1 ))
    sleep 20
  done
done
wait
echo "SINGLE_CARD_PILOT_DONE $(date -u)"
