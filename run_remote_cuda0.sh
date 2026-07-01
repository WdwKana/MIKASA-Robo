#!/bin/bash
# Single-GPU (cuda:0) sequential runner for the second machine.
# One run at a time (~16-17 GB VRAM, ~8 h each on an L4). Two won't fit in 24 GB.
#
# PRINCIPLED PER-TASK CONFIG (auto-selected below):
#   CRES  (color residual)  ON  <=> the task's cue is COLOR       -> RememberColor*, RememberShapeAndColor*
#   CAPS  (action smoothing) ON <=> success needs the robot to STABILIZE (is_robot_static / is_stable)
#                                    -> all the above + RememberShape*, ShellGame*, Rotate*, TakeItBack
#   plain (no trick)            <=> pure dynamic reaction, no stabilization -> Intercept* (non-grab)
# => cres_caps entry / caps entry / plain entry, chosen automatically per task. Do NOT hand-mix.
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0          # <-- only GPU 0 is usable on this machine
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

# ---- edit me: the tasks to run (Goal 1 Tier A shown) --------------------------
TASKS=(
  ShellGameTouch-v0 ShellGamePush-v0 ShellGamePick-v0
  RotateStrictPos-v0 RotateStrictPosNeg-v0
  RotateLenientPos-v0 RotateLenientPosNeg-v0
  TakeItBack-v0
)
METHODS=(srbtr gru lstm mlp)           # drop 'mlp' if time-constrained
SEEDS=(33 42 99)
# ------------------------------------------------------------------------------

# task -> config tag
config_for() {
  case "$1" in
    RememberColor*|RememberShapeAndColor*)          echo cres_caps ;;
    RememberShape*|ShellGame*|Rotate*|TakeItBack*)  echo caps ;;
    Intercept*)                                     echo plain ;;   # pure Intercept; Grab: see note in SPEC
    *)                                              echo caps ;;    # safe default
  esac
}
# (method, config) -> entry file
entry_for() {
  local base
  case "$1" in
    srbtr) base=ppo_memtasks_ebm_srb_tr ;;
    gru)   base=ppo_memtasks_dinov2_gru ;;
    lstm)  base=ppo_memtasks_dinov2_lstm ;;
    mlp)   base=ppo_memtasks_dinov2_mlp ;;
  esac
  case "$2" in
    cres_caps) echo "baselines/ppo/${base}_cres_caps.py" ;;
    caps)      echo "baselines/ppo/${base}_caps.py" ;;
    plain)     echo "baselines/ppo/${base}.py" ;;
  esac
}

for ENV in "${TASKS[@]}"; do
  CFG=$(config_for "$ENV")
  # per-task schedule (ONLY thing that varies by task)
  if [[ "$ENV" == Intercept* ]]; then NUMSTEPS=90; EVALSTEPS=540; EVALFREQ=16
  else                                NUMSTEPS=60; EVALSTEPS=720; EVALFREQ=24; fi
  # CAPS flag only when config uses it
  if [[ "$CFG" == plain ]]; then CAPS_FLAG=""; else CAPS_FLAG="--caps-lambda-t=0.15"; fi
  for M in "${METHODS[@]}"; do
    ENTRY=$(entry_for "$M" "$CFG")
    for SEED in "${SEEDS[@]}"; do
      echo "=========================================================="
      echo "=== method=$M  env=$ENV  seed=$SEED  config=$CFG  (cuda:0) ==="
      echo "=========================================================="
      python3 "$ENTRY" \
        --env_id="$ENV" --exp-name="${M}-${CFG}-seed${SEED}" \
        --capture-video --save-model \
        --num-steps="$NUMSTEPS" --num-eval-steps="$EVALSTEPS" --eval-freq="$EVALFREQ" \
        --include-rgb --include-joints --seed="$SEED" \
        --total-timesteps=7_000_000 --no-finite-horizon-gae \
        --num-envs=256 --num-eval-envs=16 --num-minibatches=32 --update-epochs=2 \
        --gae-lambda=0.9 --gamma=0.99 --learning-rate=1e-4 --anneal-lr \
        --ent-coef=0.001 --target-kl=0.05 $CAPS_FLAG \
        || echo "!!! FAILED method=$M env=$ENV seed=$SEED (likely camera-OOM; rerun this one) !!!"
    done
  done
done
echo "ALL DONE."
