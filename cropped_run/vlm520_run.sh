#!/usr/bin/env bash
set -uo pipefail
B=$HOME/bias_benchmark; A=$B/nazar_folder/Act2Answer; L=$B/nazar_folder/cropped_run/logs; ST=$B/nazar_folder/cropped_run/state2
source $B/miniconda3/etc/profile.d/conda.sh; conda activate $B/miniconda3/envs/spatialvla_act2answer
export REPO_ROOT=$A PYTHONPATH=$A/SimplerEnv:$A/ManiSkill TOKENIZERS_PARALLELISM=false XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $A/SimplerEnv
# magma в своём env (magma_act2answer), paligemma в spatialvla_act2answer
run(){ # model script env assets
  local m=$1 scr=$2 e=$3 assets=$4 name=vlm520-$1-$4
  [ -f $ST/$name.done ] && { echo "SKIP $name"; return; }
  source $B/miniconda3/etc/profile.d/conda.sh; conda activate $B/miniconda3/envs/$e
  echo "START $name $(date -u)"
  python3 -u $A/scripts/$scr --assets $assets --count 520 --device cuda:0 --out $A/outputs/$name.json > $L/$name.log 2>&1
  [ -f $A/outputs/$name.json ] && { touch $ST/$name.done; echo "OK $name"; } || echo "FAIL $name"
}
run magma     magma_vlm_qa.py     magma_act2answer     pairs_bias
run magma     magma_vlm_qa.py     magma_act2answer     pairs_bias_crop
run paligemma paligemma_vlm_qa.py spatialvla_act2answer pairs_bias
run paligemma paligemma_vlm_qa.py spatialvla_act2answer pairs_bias_crop
echo ALL_DONE
