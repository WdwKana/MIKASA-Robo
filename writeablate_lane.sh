#!/bin/bash
# writeablate_lane.sh <GPU> <RULE: random|fifo>
# Runs the write-rule ablation: {RememberColor5,RememberShape5,RememberShapeAndColor3x2}
# x {33,42,99} with the given rule. CRES+CAPS config identical to main table.
# exp-name = srbtr-w<RULE>-crescaps-seed<SEED> (EXACT pattern; never collides with
# main-table srbtr-crescaps-seed* globs). Resumable via DONE markers.
set -uo pipefail
GPU="$1"; RULE="$2"
unset CONDA_EXE CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_PYTHON_EXE CONDA_DEFAULT_ENV
export MAMBA_ROOT_PREFIX="$HOME/micromamba"; export CONDARC="$HOME/micromamba/.condarc_empty"
export CUDA_VISIBLE_DEVICES="$GPU"; export PYTHONUNBUFFERED=1; export TRANSFORMERS_VERBOSITY=error
MM="$HOME/micromamba/bin/micromamba"; cd /home/wangdw/MIKASA-Robo
LOGDIR="$HOME/mikasa_plan1_logs"; mkdir -p "$LOGDIR"
TASKS=(RememberColor5-v0 RememberShape5-v0 RememberShapeAndColor3x2-v0)
SEEDS=(33 42 99)
for T in "${TASKS[@]}"; do for SEED in "${SEEDS[@]}"; do
  RUN="srbtr-w${RULE}-crescaps-seed${SEED}__${T}"; LOG="$LOGDIR/${RUN}.log"
  if [[ -f "$LOG" ]] && grep -q "DONE $RUN rc=0" "$LOG"; then echo "===== SKIP (done) $RUN ====="; continue; fi
  echo "===== START $RUN (GPU$GPU) $(date) ====="
  "$MM" run -n mikasa python3 baselines/ppo/ppo_memtasks_ebm_srb_tr_cres_caps_writeablate.py \
    --env-id="$T" --exp-name="srbtr-w${RULE}-crescaps-seed${SEED}" \
    --write-rule="$RULE" \
    --capture-video --save-model \
    --num-steps=60 --num-eval-steps=720 --eval-freq=24 \
    --include-rgb --include-joints --seed="$SEED" \
    --total-timesteps=7_000_000 --no-finite-horizon-gae \
    --num-envs=256 --num-eval-envs=16 --num-minibatches=32 --update-epochs=2 \
    --gae-lambda=0.9 --gamma=0.99 --learning-rate=1e-4 --anneal-lr \
    --ent-coef=0.001 --target-kl=0.05 --caps-lambda-t=0.15 \
    > "$LOG" 2>&1 && echo "===== DONE $RUN rc=0 $(date) =====" \
    || echo "!!!!! FAILED $RUN rc=$? $(date) !!!!!"
done; done
echo "ALL DONE write-rule=$RULE (GPU$GPU) $(date)"
