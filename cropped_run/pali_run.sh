#!/usr/bin/env bash
set -uo pipefail
B=$HOME/bias_benchmark; A=$B/nazar_folder/Act2Answer; L=$B/nazar_folder/cropped_run/logs; ST=$B/nazar_folder/cropped_run/state2
source $B/miniconda3/etc/profile.d/conda.sh; conda activate $B/miniconda3/envs/spatialvla_act2answer
export REPO_ROOT=$A PYTHONPATH=$A/SimplerEnv:$A/ManiSkill TOKENIZERS_PARALLELISM=false XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $A/SimplerEnv
for assets in pairs_bias pairs_bias_crop; do
  name=vlm-paligemma-$assets
  python3 -u $A/scripts/paligemma_vlm_qa.py --assets $assets --count 55 --device cuda:0 --out $A/outputs/$name.json < /dev/null > $L/$name.log 2>&1
  [ -f $A/outputs/$name.json ] && { touch $ST/$name.done; echo "OK $name"; } || echo "FAIL $name"
done
