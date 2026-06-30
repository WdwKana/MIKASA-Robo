#!/bin/bash
# Single-GPU (cuda:0) sequential runner for the second machine.
# One run at a time (~16-17 GB VRAM, ~8 h each on an L4). Two won't fit in 24 GB.
# Edit TASKS / METHODS / SEEDS below, then: bash run_remote_cuda0.sh
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0          # <-- only GPU 0 is usable on this machine
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

# ---- edit me -----------------------------------------------------------------
# Tier A first (Goal 1). Add Tier B/C or other tasks as needed.
TASKS=(
  ShellGameTouch-v0 ShellGamePush-v0 ShellGamePick-v0
  RotateStrictPos-v0 RotateStrictPosNeg-v0
  RotateLenientPos-v0 RotateLenientPosNeg-v0
  TakeItBack-v0
)
METHODS=(srbtr gru lstm mlp)           # drop 'mlp' if time-constrained
SEEDS=(33 42 99)
# ------------------------------------------------------------------------------

declare -A ENTRY=(
  [srbtr]=baselines/ppo/ppo_memtasks_ebm_srb_tr_cres_caps.py
  [gru]=baselines/ppo/ppo_memtasks_dinov2_gru_cres_caps.py
  [lstm]=baselines/ppo/ppo_memtasks_dinov2_lstm_cres_caps.py
  [mlp]=baselines/ppo/ppo_memtasks_dinov2_mlp_cres_caps.py
)

for ENV in "${TASKS[@]}"; do
  # per-task schedule (ONLY thing that varies by task)
  if [[ "$ENV" == Intercept* ]]; then NUMSTEPS=90; EVALSTEPS=540; EVALFREQ=16
  else                                NUMSTEPS=60; EVALSTEPS=720; EVALFREQ=24; fi
  for M in "${METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      echo "=========================================================="
      echo "=== method=$M  env=$ENV  seed=$SEED  (CUDA_VISIBLE_DEVICES=0) ==="
      echo "=========================================================="
      python3 "${ENTRY[$M]}" \
        --env_id="$ENV" --exp-name="${M}-crescaps-seed${SEED}" \
        --capture-video --save-model \
        --num-steps="$NUMSTEPS" --num-eval-steps="$EVALSTEPS" --eval-freq="$EVALFREQ" \
        --include-rgb --include-joints --seed="$SEED" \
        --total-timesteps=7_000_000 --no-finite-horizon-gae \
        --num-envs=256 --num-eval-envs=16 --num-minibatches=32 --update-epochs=2 \
        --gae-lambda=0.9 --gamma=0.99 --learning-rate=1e-4 --anneal-lr \
        --ent-coef=0.001 --target-kl=0.05 --caps-lambda-t=0.15 \
        || echo "!!! FAILED method=$M env=$ENV seed=$SEED (likely camera-OOM; rerun this one) !!!"
    done
  done
done
echo "ALL DONE."
