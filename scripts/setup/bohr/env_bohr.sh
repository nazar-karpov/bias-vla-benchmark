# Окружение Bohr (Selectel, 2×RTX 4090). Подключать: source ~/ws/env_bohr.sh
# Аналог /workspace/moskalenko с cloud.ru, но без root → всё в $HOME/ws.
export WS=$HOME/ws
export REPO_ROOT=$WS/bias-vla-benchmark-main/Act2Answer
export HF_HOME=$WS/hf_cache
export MS_ASSET_DIR=$WS/maniskill_assets
export PYTHONPATH=$REPO_ROOT/SimplerEnv:$REPO_ROOT/ManiSkill${PYTHONPATH:+:$PYTHONPATH}
source $WS/conda/etc/profile.d/conda.sh
conda activate "${A2A_ENV:-magma_act2answer}"
# рендер/евал запускать из SimplerEnv: overlay-фон ./bridge_real_eval_1.png ищется относительно cwd
cd $REPO_ROOT/SimplerEnv
