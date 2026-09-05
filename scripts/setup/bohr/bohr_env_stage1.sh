#!/usr/bin/env bash
# Bohr, стадия 1: conda-env magma_act2answer из экспорта cloud.ru (python 3.10, torch 2.4.0+cu121,
# все pip-пакеты КРОМЕ editable-репо mani-skill/simpler-env — они ставятся стадией 2 после переноса репы).
set -u
source $HOME/ws/conda/etc/profile.d/conda.sh
E=$HOME/ws/env_exports/magma_act2answer.pip.txt
conda create -y -q -n magma_act2answer python=3.10 || exit 1
conda activate magma_act2answer
pip install -q "setuptools<81" wheel
pip install -q torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121 || exit 1
grep -v -i "^mani-skill\|^simpler-env\|^torch==\|^torchvision==\|@ file\|^-e \|^flash-attn\|^pip==\|^setuptools==\|^wheel==" $E > /tmp/req_stage1.txt
pip install -q -r /tmp/req_stage1.txt --extra-index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -20
python -c "import torch, sapien, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count()); print('sapien', sapien.__version__, 'transformers', transformers.__version__)"
echo STAGE1_DONE
