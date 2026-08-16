#!/usr/bin/env bash
# Сетка масштабов, ПАРАЛЛЕЛЬНАЯ версия.
#
# Профилировка исходного раннера: GPU 0% при 77 ГБ VRAM и 255% CPU из 12800
# доступных (128 ядер). Два узких места:
#   1) запись 400 видео после симуляции — снято A2A_SAVE_VIDEO=0;
#   2) num_envs=400 держит все среды разом и съедает 77 ГБ из 80 -> на карту
#      влезал ровно один процесс.
# Здесь прогон режется на шарды по SHARD эпизодов (VRAM ~ линейна по num_envs),
# поэтому на карту помещается несколько процессов и загружаются свободные ядра.
#
#   SCALES="0.8 1.0 1.7 1.9" PER_GPU=3 SHARD=100 bash run_scale_sweep_par.sh
set -uo pipefail
R=/workspace/moskalenko/bias-vla-benchmark-main
A="$R/Act2Answer"
source /workspace/moskalenko/conda/etc/profile.d/conda.sh
conda activate magma_act2answer
export REPO_ROOT="$A" PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
export MS_ASSET_DIR=/workspace/moskalenko/maniskill_assets
export HF_HOME=/workspace/moskalenko/hf_cache
export TOKENIZERS_PARALLELISM=false
export A2A_TRAJ_LOG=1
export A2A_SAVE_VIDEO=0            # главный резерв: не писать 400 роликов
cd "$A/SimplerEnv" || exit 1

SCALES="${SCALES:-0.8 1.0 1.7 1.9}"
MODEL="${MODEL:-magma}"
ASSET="${ASSET:-neutral_colors_big}"
TOTAL="${TOTAL:-400}"
SHARD="${SHARD:-100}"
PER_GPU="${PER_GPU:-3}"            # процессов на карту
IB="${IB:-10}"
LOG=/workspace/moskalenko/logs_scale_par
mkdir -p "$LOG"

MAXJOBS=$(( PER_GPU * 2 ))
i=0
for sc in $SCALES; do
  for lay in noswap swap; do
    sw=""; [ "$lay" = swap ] && sw="--do-swap"
    for ((s=0; s<TOTAL; s+=SHARD)); do
      name="scalepar-${ASSET}-${MODEL}-s${sc}-${lay}-sh${s}"
      [ -f "$A/outputs/$name/glob/vis_0_test/stats.yaml" ] && { echo "SKIP $name"; continue; }
      # ждём свободный слот
      while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 20; done
      gpu=$(( i % 2 ))
      echo "[$(date -u +%H:%M:%S)] START $name gpu=$gpu"
      CUDA_VISIBLE_DEVICES=$gpu BOARD_XY_SCALE=$sc \
        python -u -m simpler_env.eval --vla "$MODEL" \
          --assets "$ASSET" --obj-set test --start-id "$s" --count "$SHARD" \
          --episode-len 80 --buffer-inferbatch "$IB" --buffer-minibatch -1 \
          --name "$name" $sw < /dev/null > "$LOG/$name.log" 2>&1 &
      i=$(( i + 1 ))
      sleep 25   # разнести пики загрузки модели, иначе всплеск VRAM
    done
  done
done
wait
echo "SWEEP_PAR_DONE $(date -u)"
