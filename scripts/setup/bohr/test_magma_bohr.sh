#!/usr/bin/env bash
# Bohr: приёмочный тест Magma-8B + симулятор на GPU — 2 эпизода VLM-опроса
# (magma_vlm_qa.py: рендер первых кадров + загрузка модели + ответы).
set -u
source $HOME/ws/env_bohr.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BOARD_XY_SCALE=1.0
OUT=$REPO_ROOT/outputs/_bohr_magma_test.json
python -u $REPO_ROOT/../scripts/magma_vlm_qa.py --assets pairs_choice_vla_confirm \
  --start-id 0 --count 2 --render-chunk 2 --out "$OUT" --device cuda:0 < /dev/null
echo "MAGMA_TEST_DONE rc=$?"
ls -la "$OUT" 2>/dev/null && python -c "import json;d=json.load(open('$OUT'));print(type(d).__name__, len(d))"
