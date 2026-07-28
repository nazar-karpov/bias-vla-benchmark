#!/usr/bin/env bash
###############################################################################
# run_cropped_benchmark.sh
#
# One resumable driver that:
#   1. builds the CROPPED pairs_bias cardset (center-crop on the person),
#   2. ensures each model is installed (env/repo/checkpoint/server as needed),
#   3. runs magma, spatialvla, internvla, rldx on pairs_bias_crop (noswap+swap).
#
# RESUMABLE: every step writes a marker in $STATE_DIR. Re-running skips any step
# whose marker exists, so a crash resumes from the failed step (not the start).
# Delete a marker to force that step to re-run.
#
# SKIP-ON-FAIL: if a model's setup/server fails, it's logged FAILED (marker
# *.failed) and the driver moves to the next model -- one broken model never
# blocks the rest. Re-running retries failed models (their .failed marker is
# cleared at the start of each attempt).
#
# Usage (from anywhere, no conda needed to start):
#   bash run_cropped_benchmark.sh                 # all models
#   MODELS="magma spatialvla" bash run_cropped_benchmark.sh
#   COUNT=55 bash run_cropped_benchmark.sh        # episodes per layout
###############################################################################
set -uo pipefail

# ------------------------------------------------------------------ paths ----
BIAS="$HOME/bias_benchmark"
A="$BIAS/nazar_folder/Act2Answer"
CONDA_ROOT="$BIAS/miniconda3"
CONDA_ENVS="$CONDA_ROOT/envs"
CARROT="$A/ManiSkill/mani_skill/assets/carrot"
SCRIPTS="$A/scripts"                     # this file's own dir on the server

ASSET="pairs_bias_crop"                  # the cropped cardset name
SRC_IMGS="$BIAS/nazar_folder/pairs_bias/imgs"          # 200 uncropped tiles
SRC_CSV="$BIAS/nazar_folder/pairs_bias/questions.csv"  # same episodes as uncropped
CROP_IMGS="$BIAS/nazar_folder/pairs_bias_crop/imgs"

WORK="$BIAS/nazar_folder/cropped_run"
STATE_DIR="$WORK/state"
LOGDIR="$WORK/logs"
mkdir -p "$STATE_DIR" "$LOGDIR"

MODELS="${MODELS:-magma spatialvla internvla rldx}"
COUNT="${COUNT:-55}"                     # episodes per layout (55 % 5 == 0)
INFERBATCH="${INFERBATCH:-5}"
EPLEN="${EPLEN:-80}"
# GPU pool: 0 and 1 are the free-ish ones (2/3 usually busy with others).
GPU_A2A="${GPU_A2A:-0}"                   # SimplerEnv render + client model
GPU_SRV="${GPU_SRV:-1}"                   # heavy server-based model process

MASTER_LOG="$LOGDIR/_master.log"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }

step_done() { [ -f "$STATE_DIR/$1.done" ]; }
mark_done()  { touch "$STATE_DIR/$1.done"; }
mark_failed(){ touch "$STATE_DIR/$1.failed"; }
clear_failed(){ rm -f "$STATE_DIR/$1.failed"; }

# run a shell command as a guarded step; returns 0 if done/skipped, 1 on failure
guard() {                                # guard <marker> <logfile> <cmd...>
  local marker="$1" logf="$2"; shift 2
  if step_done "$marker"; then log "SKIP  $marker (done)"; return 0; fi
  log "START $marker -> $logf"
  if "$@" >"$logf" 2>&1; then
    mark_done "$marker"; log "OK    $marker"; return 0
  else
    local rc=$?; log "FAIL  $marker rc=$rc (see $logf)"; return 1
  fi
}

conda_run() {                            # conda_run <env> <cmd...>
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENVS/$1"
}

# ------------------------------------------------------- stage: cardset ------
build_cardset() {
  set -e
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  # any env with PIL+trimesh works; magma_act2answer has them.
  conda activate "$CONDA_ENVS/magma_act2answer"
  python3 "$SCRIPTS/crop_pairs_images.py" --in "$SRC_IMGS" --out "$CROP_IMGS" \
    --frac "${CROP_FRAC:-0.65}" --y-bias "${CROP_YBIAS:-0.35}"
  python3 "$SCRIPTS/make_cardset.py" \
    --images "$CROP_IMGS" --questions "$SRC_CSV" --out "$CARROT/$ASSET"
  test -f "$CARROT/$ASSET/pairs.json"
}

# --------------------------------------------------- eval one layout ---------
# Runs inside the model's conda env. server-based models must have a live server.
eval_layout() {                          # eval_layout <model> <env> <swap>
  local model="$1" env="$2" swap="$3"
  local extra=""; [ "$swap" = swap ] && extra="--do-swap"
  local name="crop-${model}-${ASSET}-${swap}"
  local logf="$LOGDIR/${name}.log"
  if step_done "$name"; then log "SKIP  $name (done)"; return 0; fi
  log "START $name -> $logf"
  (
    set -e
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENVS/$env"
    export REPO_ROOT="$A"
    export PYTHONPATH="$A/SimplerEnv:$A/ManiSkill"
    export TOKENIZERS_PARALLELISM=false
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export CUDA_VISIBLE_DEVICES="$GPU_A2A"
    cd "$A/SimplerEnv"
    setsid python3 -u -m simpler_env.eval \
      --vla "$model" --start-id 0 --count "$COUNT" \
      --assets "$ASSET" --obj-set test --episode-len "$EPLEN" \
      --buffer-inferbatch "$INFERBATCH" --buffer-minibatch -1 \
      --name "$name" $extra < /dev/null
  ) >"$logf" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && grep -qa "FINAL_STATS" "$logf"; then
    mark_done "$name"
    log "OK    $name $(grep -a FINAL_STATS "$logf" | tail -1)"
    return 0
  fi
  log "FAIL  $name rc=$rc (no FINAL_STATS)"
  return 1
}

run_both_layouts() {                     # run_both_layouts <model> <env>
  local model="$1" env="$2" ok=0
  eval_layout "$model" "$env" noswap || ok=1
  eval_layout "$model" "$env" swap   || ok=1
  return $ok
}

###############################################################################
# per-model setup functions (return 0 = ready to eval, 1 = failed)
###############################################################################

# magma / spatialvla: in-process, envs already built (see act2answer memory).
setup_inprocess() {                      # setup_inprocess <env>
  local env="$1"
  [ -d "$CONDA_ENVS/$env" ]
}

# InternVLA-M1: server-based. Needs external repo + env + HF ckpt + ws server:10093.
INTERNVLA_REPO="$BIAS/nazar_folder/InternVLA-M1"
INTERNVLA_CKPT_DIR="$BIAS/nazar_folder/internvla_ckpt"
INTERNVLA_PORT=10093

setup_internvla() {
  # 1. repo
  if [ ! -d "$INTERNVLA_REPO/.git" ]; then
    guard "internvla-clone" "$LOGDIR/internvla-clone.log" \
      git clone https://github.com/InternRobotics/InternVLA-M1.git "$INTERNVLA_REPO" || return 1
  fi
  # 2. env (uses repo's own setup script; env.sh paths overridden below)
  if [ ! -d "$CONDA_ENVS/internvla" ]; then
    guard "internvla-env" "$LOGDIR/internvla-env.log" \
      env CONDA_ROOT="$CONDA_ROOT" CONDA_ENVS_DIR="$CONDA_ENVS" \
          A2A_EXTERNAL_DIR="$BIAS/nazar_folder" INTERNVLA_REPO="$INTERNVLA_REPO" \
          bash "$A/scripts/setup/setup_internvla_env.sh" || return 1
  fi
  # 3. checkpoint (HF). resolved to a *.pt path exported as INTERNVLA_CKPT.
  guard "internvla-ckpt" "$LOGDIR/internvla-ckpt.log" \
    bash "$SCRIPTS/fetch_internvla_ckpt.sh" "$INTERNVLA_CKPT_DIR" || return 1
  export INTERNVLA_CKPT="$(cat "$INTERNVLA_CKPT_DIR/ckpt_path.txt" 2>/dev/null)"
  [ -f "$INTERNVLA_CKPT" ] || { log "internvla ckpt path missing"; return 1; }
  # 4. server on :10093 (background, GPU_SRV)
  start_internvla_server || return 1
}

start_internvla_server() {
  if curl -s "http://127.0.0.1:$INTERNVLA_PORT" >/dev/null 2>&1 \
     || (exec 3<>"/dev/tcp/127.0.0.1/$INTERNVLA_PORT") 2>/dev/null; then
    log "internvla server already up on :$INTERNVLA_PORT"; return 0
  fi
  local srvlog="$LOGDIR/internvla-server.log"
  log "starting internvla server -> $srvlog"
  ( source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENVS/internvla"
    export CUDA_VISIBLE_DEVICES="$GPU_SRV"
    cd "$INTERNVLA_REPO"
    setsid bash "$SCRIPTS/internvla_server.sh" "$INTERNVLA_CKPT" "$INTERNVLA_PORT" \
      >"$srvlog" 2>&1 < /dev/null &
  )
  # wait for port (up to 5 min: model load is slow)
  for i in $(seq 1 60); do
    (exec 3<>"/dev/tcp/127.0.0.1/$INTERNVLA_PORT") 2>/dev/null && { log "internvla server up"; return 0; }
    grep -qa "Traceback\|Error" "$srvlog" 2>/dev/null && { log "internvla server crashed"; return 1; }
    sleep 5
  done
  log "internvla server did not come up in time"; return 1
}

# RLDX-1: server-based (zmq), uv env, own repo + HF ckpt + Act2Answer adapter.
RLDX_REPO="$BIAS/nazar_folder/RLDX-1"
RLDX_PORT=20000

setup_rldx() {
  guard "rldx-setup" "$LOGDIR/rldx-setup.log" \
    bash "$SCRIPTS/setup_rldx.sh" "$RLDX_REPO" || return 1
  start_rldx_server || return 1
}

start_rldx_server() {
  (exec 3<>"/dev/tcp/127.0.0.1/$RLDX_PORT") 2>/dev/null && { log "rldx server already up"; return 0; }
  local srvlog="$LOGDIR/rldx-server.log"
  log "starting rldx server -> $srvlog"
  ( cd "$RLDX_REPO"
    export CUDA_VISIBLE_DEVICES="$GPU_SRV"
    setsid bash "$SCRIPTS/rldx_server.sh" "$RLDX_PORT" >"$srvlog" 2>&1 < /dev/null & )
  for i in $(seq 1 72); do               # up to 6 min (bigger model)
    (exec 3<>"/dev/tcp/127.0.0.1/$RLDX_PORT") 2>/dev/null && { log "rldx server up"; return 0; }
    grep -qa "Traceback\|Error" "$srvlog" 2>/dev/null && { log "rldx server crashed"; return 1; }
    sleep 5
  done
  log "rldx server did not come up in time"; return 1
}

###############################################################################
# main
###############################################################################
log "=== run_cropped_benchmark START models='$MODELS' count=$COUNT ==="

# stage 1: cardset (blocking -- everything needs it)
if ! step_done cardset; then
  if guard cardset "$LOGDIR/cardset.log" build_cardset; then :; else
    log "FATAL: cardset build failed -- cannot proceed"; exit 1
  fi
fi

# stage 2+: per model, skip-on-fail
for model in $MODELS; do
  # retry a previously-failed model on re-run
  clear_failed "$model"
  log "----- model: $model -----"
  # env = the conda env the Act2Answer eval loop runs in (must have SimplerEnv).
  # server-based models (internvla/rldx) run a heavy server in their OWN env, but
  # the eval loop is a thin websocket/zmq CLIENT -> run it in a SimplerEnv env.
  EVAL_ENV="${EVAL_ENV:-spatialvla_act2answer}"
  case "$model" in
    magma)      env=magma_act2answer;      setup_inprocess "$env"; ready=$? ;;
    spatialvla) env=spatialvla_act2answer; setup_inprocess "$env"; ready=$? ;;
    internvla)  env="$EVAL_ENV"; setup_internvla; ready=$? ;;
    rldx)       env="$EVAL_ENV"; setup_rldx;      ready=$? ;;
    *) log "unknown model '$model' -- skipping"; continue ;;
  esac
  if [ "$ready" -ne 0 ]; then
    log "SKIP model $model (setup failed)"; mark_failed "$model"; continue
  fi
  if run_both_layouts "$model" "$env"; then
    log "MODEL DONE $model"
  else
    log "MODEL PARTIAL/FAIL $model (some layouts missing FINAL_STATS)"; mark_failed "$model"
  fi
done

log "=== run_cropped_benchmark END ==="
# summary
echo "----- SUMMARY -----" | tee -a "$MASTER_LOG"
for model in $MODELS; do
  for swap in noswap swap; do
    n="crop-${model}-${ASSET}-${swap}"
    if step_done "$n"; then st="DONE"; else st="MISSING"; fi
    echo "  $n : $st" | tee -a "$MASTER_LOG"
  done
  [ -f "$STATE_DIR/$model.failed" ] && echo "  ** $model marked FAILED **" | tee -a "$MASTER_LOG"
done
