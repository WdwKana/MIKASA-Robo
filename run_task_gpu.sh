#!/bin/bash
# Usage: run_task_gpu.sh <GPU_ID> <ENV_ID>
# CAPS-only (no CRES) for non-color tasks, per the task-structure rule
# (CRES only where color is a discriminative cue). srbtr/gru/lstm x {33,42,99}.
# mlp OMITTED (no CAPS-only mlp entry yet; alice to provide + sync).
# Naming: <method>-caps-seed<SEED>. Resumable (skips runs already logged rc=0).
set -uo pipefail
GPU="$1"; ENVID="$2"
unset CONDA_EXE CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_PYTHON_EXE CONDA_DEFAULT_ENV
export MAMBA_ROOT_PREFIX="$HOME/micromamba"; export CONDARC="$HOME/micromamba/.condarc_empty"
export CUDA_VISIBLE_DEVICES="$GPU"; export PYTHONUNBUFFERED=1; export TRANSFORMERS_VERBOSITY=error
MM="$HOME/micromamba/bin/micromamba"; cd /home/wangdw/MIKASA-Robo
LOGDIR="$HOME/mikasa_plan1_logs"; mkdir -p "$LOGDIR"
METHODS=(srbtr gru lstm); SEEDS=(33 42 99)
declare -A ENTRY=(
  [srbtr]=baselines/ppo/ppo_memtasks_ebm_srb_tr_caps.py
  [gru]=baselines/ppo/ppo_memtasks_dinov2_gru_caps.py
  [lstm]=baselines/ppo/ppo_memtasks_dinov2_lstm_caps.py )
if [[ "$ENVID" == Intercept* ]]; then NUMSTEPS=90; EVALSTEPS=540; EVALFREQ=16
else                                NUMSTEPS=60; EVALSTEPS=720; EVALFREQ=24; fi
for M in "${METHODS[@]}"; do for SEED in "${SEEDS[@]}"; do
  RUN="${M}-caps-seed${SEED}__${ENVID}"; LOG="$LOGDIR/${RUN}.log"
  if [[ -f "$LOG" ]] && grep -q "DONE $RUN rc=0" "$LOG"; then echo "===== SKIP (done) $RUN ====="; continue; fi
  echo "===== START $RUN (GPU$GPU) $(date) ====="
  "$MM" run -n mikasa python3 "${ENTRY[$M]}" \
    --env-id="$ENVID" --exp-name="${M}-caps-seed${SEED}" --capture-video --save-model \
    --num-steps="$NUMSTEPS" --num-eval-steps="$EVALSTEPS" --eval-freq="$EVALFREQ" \
    --include-rgb --include-joints --seed="$SEED" \
    --total-timesteps=7_000_000 --no-finite-horizon-gae \
    --num-envs=256 --num-eval-envs=16 --num-minibatches=32 --update-epochs=2 \
    --gae-lambda=0.9 --gamma=0.99 --learning-rate=1e-4 --anneal-lr \
    --ent-coef=0.001 --target-kl=0.05 --caps-lambda-t=0.15 \
    > "$LOG" 2>&1 && echo "===== DONE $RUN rc=0 $(date) =====" \
    || echo "!!!!! FAILED $RUN rc=$? $(date) !!!!!"
done; done
echo "ALL DONE $ENVID (GPU$GPU) $(date)"
