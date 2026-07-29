#!/usr/bin/env bash
# Резюмируемый мастер: (1) soft-VLA magma+spatialvla на pairs_bias_crop,
# (2) 4 VLM-джоба (magma/paligemma x pairs_bias/pairs_bias_crop).
# ПОСЛЕДОВАТЕЛЬНО на одном GPU (GPU 1 чистый; GPU 0 ломает magma camera-group).
# Маркеры в state2/. Запускать повторно безопасно.
set -uo pipefail
B=$HOME/bias_benchmark; A=$B/nazar_folder/Act2Answer
CR=$B/nazar_folder/cropped_run; L=$CR/logs; ST=$CR/state2; mkdir -p $ST $L
GPU=${GPU:-1}
CONDA=$B/miniconda3
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L/_vlm_soft_master.log; }
done_(){ [ -f $ST/$1.done ]; }
mark(){ touch $ST/$1.done; }

act(){ source $CONDA/etc/profile.d/conda.sh; conda activate $CONDA/envs/$1; }

# ---- soft-VLA (in-process, env модели) ----
vla_soft(){ # model env swap
  local m=$1 e=$2 sw=$3 name=softcrop-$1-$3
  done_ $name && { log "SKIP $name"; return 0; }
  if grep -qa FINAL_STATS $L/$name.log 2>/dev/null; then mark $name; log "already $name"; return 0; fi
  local extra=""; [ $sw = swap ] && extra="--do-swap"
  log "START $name (GPU $GPU)"
  ( act $e; export REPO_ROOT=$A PYTHONPATH=$A/SimplerEnv:$A/ManiSkill TOKENIZERS_PARALLELISM=false XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU
    cd $A/SimplerEnv
    setsid python3 -u -m simpler_env.eval --vla $m --start-id 0 --count 55 --assets pairs_bias_crop --obj-set test --episode-len 80 --buffer-inferbatch 5 --buffer-minibatch -1 --name $name $extra < /dev/null
  ) > $L/$name.log 2>&1
  grep -qa FINAL_STATS $L/$name.log && { mark $name; log "OK $name $(grep -a FINAL_STATS $L/$name.log|tail -1)"; } || log "FAIL $name"
}

# ---- VLM (kadr из симулятора, обе раскладки за раз) ----
vlm(){ # tag script env assets
  local tag=$1 scr=$2 e=$3 assets=$4 name=vlm-$1-$4
  done_ $name && { log "SKIP $name"; return 0; }
  log "START $name (GPU $GPU)"
  ( act $e; export REPO_ROOT=$A PYTHONPATH=$A/SimplerEnv:$A/ManiSkill TOKENIZERS_PARALLELISM=false XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    cd $A/SimplerEnv
    setsid python3 -u $A/scripts/$scr --assets $assets --count 55 --device cuda:0 --out $A/outputs/$name.json < /dev/null
  ) > $L/$name.log 2>&1
  [ -f $A/outputs/$name.json ] && { mark $name; log "OK $name"; } || log "FAIL $name (see $L/$name.log)"
}

wait_idle(){ # ждём пока никакой simpler_env.eval или vlm_qa не бежит на нашем GPU
  for i in $(seq 1 240); do
    pgrep -f "simpler_env.eval|magma_vlm_qa|paligemma_vlm_qa" >/dev/null || return 0
    sleep 20
  done
}
log "=== MASTER START GPU=$GPU ==="
wait_idle; vla_soft spatialvla spatialvla_act2answer noswap
wait_idle; vla_soft spatialvla spatialvla_act2answer swap
wait_idle; vla_soft magma magma_act2answer noswap
wait_idle; vla_soft magma magma_act2answer swap
wait_idle; vlm magma    magma_vlm_qa.py     magma_act2answer      pairs_bias
wait_idle; vlm magma    magma_vlm_qa.py     magma_act2answer      pairs_bias_crop
wait_idle; vlm paligemma paligemma_vlm_qa.py spatialvla_act2answer     pairs_bias
wait_idle; vlm paligemma paligemma_vlm_qa.py spatialvla_act2answer     pairs_bias_crop
log "=== MASTER END ==="
