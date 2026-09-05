#!/usr/bin/env bash
# Bohr, стадия 2 (после переноса репы): requirements/magma.txt + editable ManiSkill/SimplerEnv,
# затем перебить абсолютные симлинки кардсетов /workspace/moskalenko → /home/moskalenko/ws.
set -u
W=$HOME/ws
A=$W/bias-vla-benchmark-main/Act2Answer
source $W/conda/etc/profile.d/conda.sh
conda activate magma_act2answer
python -m pip install -q -r "$A/requirements/magma.txt" 2>&1 | tail -5
echo "=== req rc=$?"
python -m pip install -q -e "$A/ManiSkill" -e "$A/SimplerEnv" 2>&1 | tail -5
echo "=== editable rc=$?"
python -m pip install -q "setuptools<81"
python -c "import sapien, gymnasium, mani_skill, simpler_env; print('sapien/gym/mani_skill/simpler_env OK')"
echo "=== import rc=$?"
# симлинки shapes: абсолютные пути cloud.ru → Bohr
n=0
while IFS= read -r l; do
  t=$(readlink "$l")
  case "$t" in /workspace/moskalenko/*) ln -sfn "${t/\/workspace\/moskalenko/$W}" "$l"; n=$((n+1));; esac
done < <(find "$A/ManiSkill/mani_skill/assets/carrot" -maxdepth 3 -type l)
echo "=== symlinks rewritten: $n; broken left: $(find "$A/ManiSkill/mani_skill/assets/carrot" -maxdepth 3 -xtype l | wc -l)"
echo STAGE2_DONE
