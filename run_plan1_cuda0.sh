#!/bin/bash
# ============================================================================
# PLAN 1 (validation batch): ShellGameTouch-v0 x {srbtr,gru,lstm,mlp} x {33,42,99}
#   = 12 runs, GPU 0 ONLY, one at a time, ~8 h each  ->  ~4 days total.
# Full CANONICAL hyperparameters (identical to run_remote_cuda0.sh / SPECIFICATION).
# Runs inside the micromamba 'mikasa' env, isolated from the machine's foreign
# (yux) CONDA_* config. Never touches GPUs 1/2/3 (xiedc's jobs).
# Per-run logs -> ~/mikasa_plan1_logs/<run>.log   Resumable: skips finished runs.
# ============================================================================
set -uo pipefail
unset CONDA_EXE CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_PYTHON_EXE CONDA_DEFAULT_ENV
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
export CONDARC="$HOME/micromamba/.condarc_empty"
export CUDA_VISIBLE_DEVICES=0          # GPU 0 ONLY
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error
MM="$HOME/micromamba/bin/micromamba"
REPO="/home/wangdw/MIKASA-Robo"
LOGDIR="$HOME/mikasa_plan1_logs"
mkdir -p "$LOGDIR"
cd "$REPO"

TASKS=(ShellGameTouch-v0)
METHODS=(srbtr gru lstm mlp)           # drop 'mlp' here if you want 9 runs / ~3 days
SEEDS=(33 42 99)
declare -A ENTRY=(
  [srbtr]=baselines/ppo/ppo_memtasks_ebm_srb_tr_cres_caps.py
  [gru]=baselines/ppo/ppo_memtasks_dinov2_gru_cres_caps.py
  [lstm]=baselines/ppo/ppo_memtasks_dinov2_lstm_cres_caps.py
  [mlp]=baselines/ppo/ppo_memtasks_dinov2_mlp_cres_caps.py
)

for ENV in "${TASKS[@]}"; do
  if [[ "$ENV" == Intercept* ]]; then NUMSTEPS=90; EVALSTEPS=540; EVALFREQ=16
  else                                NUMSTEPS=60; EVALSTEPS=720; EVALFREQ=24; fi
  for M in "${METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      RUN="${M}-crescaps-seed${SEED}__${ENV}"
      LOG="$LOGDIR/${RUN}.log"
      # resume guard: a complete run logs "rc=0"; skip those so a restart continues.
      if [[ -f "$LOG" ]] && grep -q "DONE $RUN rc=0" "$LOG"; then
        echo "===== SKIP (already done) $RUN ====="; continue
      fi
      echo "===== START $RUN  $(date)  -> $LOG ====="
      "$MM" run -n mikasa python3 "${ENTRY[$M]}" \
        --env-id="$ENV" --exp-name="${M}-crescaps-seed${SEED}" \
        --capture-video --save-model \
        --num-steps="$NUMSTEPS" --num-eval-steps="$EVALSTEPS" --eval-freq="$EVALFREQ" \
        --include-rgb --include-joints --seed="$SEED" \
        --total-timesteps=7_000_000 --no-finite-horizon-gae \
        --num-envs=256 --num-eval-envs=16 --num-minibatches=32 --update-epochs=2 \
        --gae-lambda=0.9 --gamma=0.99 --learning-rate=1e-4 --anneal-lr \
        --ent-coef=0.001 --target-kl=0.05 --caps-lambda-t=0.15 \
        > "$LOG" 2>&1 \
        && echo "===== DONE $RUN rc=0  $(date) =====" \
        || echo "!!!!! FAILED $RUN rc=$?  $(date)  (rerun this one; often transient camera-OOM) !!!!!"
    done
  done
done
echo "ALL PLAN-1 DONE  $(date)"
