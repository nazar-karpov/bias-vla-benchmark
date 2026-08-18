#!/usr/bin/env bash
# Добить шарды top8, упавшие на старте с "Unable to create GPU parallelized
# camera group" (6 клиентов одновременно исчерпали буферы GPU-камер).
# Идём ПОСЛЕДОВАТЕЛЬНО (PAR=1) — надёжнее, шардов всего несколько.
# Сервера InternVLA уже слушают 10093/10094 — переиспользуем, не поднимаем.
set -uo pipefail
R=/workspace/moskalenko/bias-vla-benchmark-main
A="$R/Act2Answer"
CONDA=/workspace/moskalenko/conda
export REPO_ROOT="$A" PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
export MS_ASSET_DIR=/workspace/moskalenko/maniskill_assets
export HF_HOME=/workspace/moskalenko/hf_cache
export TOKENIZERS_PARALLELISM=false
export A2A_TRAJ_LOG=1 A2A_SAVE_VIDEO=0 BOARD_XY_SCALE=1.0

LOG=/workspace/moskalenko/logs_full_cross
source "$CONDA/etc/profile.d/conda.sh"
conda activate magma_act2answer
cd "$A/SimplerEnv" || exit 1
VLA_ARG="--vla-path $(cat "$R/internvla_ckpt/ckpt_path.txt")"

# список "слот:шард"
JOBS="${JOBS:-swap:6400 swap:8600 noswap:10600 swap:10600}"
i=0
for j in $JOBS; do
  lay="${j%%:*}"; s="${j##*:}"
  sw=""; [ "$lay" = swap ] && sw="--do-swap"
  name="full33-internvla-${lay}-sh${s}"
  if [ -f "$A/outputs/$name/glob/vis_0_test/stats.yaml" ]; then
    echo "SKIP $name (уже готов)"; continue
  fi
  gpu=$(( i % 2 )); cport=$(( 10093 + i % 2 ))
  echo "[$(date -u +%H:%M:%S)] START $name gpu=$gpu port=$cport"
  CUDA_VISIBLE_DEVICES=$gpu INTERNVLA_PORT=$cport INTERNVLA_HOST=127.0.0.1 \
    python -u -m simpler_env.eval --vla internvla $VLA_ARG \
      --assets pairs_q33_full --obj-set test --start-id "$s" --count 100 \
      --episode-len 80 --buffer-inferbatch 10 --buffer-minibatch -1 \
      --name "$name" $sw < /dev/null > "$LOG/$name.log" 2>&1
  echo "[$(date -u +%H:%M:%S)] DONE $name (exit=$?)"
  i=$(( i + 1 ))
done
echo "FINISH_MISSING_DONE $(date -u)"
