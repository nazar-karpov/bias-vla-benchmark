#!/bin/bash
# Финал ночи: qwen big/bigcombo subset (GPU0); pali ПОЛНЫЙ 33-вопросный перекрёст
# на bigcombo (GPU1, стартует после доездки magma bigcombo).
source ~/bias_benchmark/miniconda3/etc/profile.d/conda.sh
export REPO_ROOT=~/bias_benchmark/nazar_folder/Act2Answer
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=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
O=$REPO_ROOT/outputs
S=$REPO_ROOT/scripts/vlm_sim_choice.py
Q="boss,wealthy,suburbs,skier"

case "$1" in
  qwen)
    conda activate ~/bias_benchmark/miniconda3/envs/internvla
    CUDA_VISIBLE_DEVICES=0 python -u $S --model qwenbase --frames-dir $O/simframes_choice_big       --only-question "$Q" --out $O/big-subset-qwenbase.json < /dev/null
    CUDA_VISIBLE_DEVICES=0 python -u $S --model qwenbase --frames-dir $O/simframes_choice_bigcombo       --only-question "$Q" --out $O/bigcombo-subset-qwenbase.json < /dev/null
    echo NIGHT4_QWEN_DONE
    ;;
  palifull)
    until grep -q NIGHT3_GPU1_DONE $O/night3-gpu1.log; do sleep 30; done
    conda activate ~/bias_benchmark/miniconda3/envs/spatialvla_act2answer
    CUDA_VISIBLE_DEVICES=1 python -u $S --model paligemma --frames-dir $O/simframes_choice_bigcombo       --out $O/bigcombo-all-paligemma.json < /dev/null
    echo NIGHT4_PALIFULL_DONE
    ;;
esac
