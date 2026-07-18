#!/bin/bash
# run_task_cfg.sh <GPU> <ENV_ID> <CONFIG: crescaps|caps|plain> <TOTAL_STEPS>
# METHODS/SEEDS overridable via env (default: srbtr gru lstm / 33 42 99). Resumable.
# exp-name = <method>-<config>-seed<SEED>  (self-describing; align tags+steps with alice).
set -uo pipefail
GPU="$1"; ENVID="$2"; CFG="$3"; STEPS="$4"
METHODS=(${METHODS:-srbtr gru lstm}); SEEDS=(${SEEDS:-33 42 99})
unset CONDA_EXE CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_PYTHON_EXE CONDA_DEFAULT_ENV
export MAMBA_ROOT_PREFIX="$HOME/micromamba"; export CONDARC="$HOME/micromamba/.condarc_empty"
export CUDA_VISIBLE_DEVICES="$GPU"; export PYTHONUNBUFFERED=1; export TRANSFORMERS_VERBOSITY=error
MM="$HOME/micromamba/bin/micromamba"; cd /home/wangdw/MIKASA-Robo
LOGDIR="$HOME/mikasa_plan1_logs"; mkdir -p "$LOGDIR"
case "$CFG" in
  cres_caps|crescaps) declare -A E=([srbtr]=baselines/ppo/ppo_memtasks_ebm_srb_tr_cres_caps.py [gru]=baselines/ppo/ppo_memtasks_dinov2_gru_cres_caps.py [lstm]=baselines/ppo/ppo_memtasks_dinov2_lstm_cres_caps.py [mlp]=baselines/ppo/ppo_memtasks_dinov2_mlp_cres_caps.py [ffm]=baselines/ppo/ppo_memtasks_dinov2_ffm.py [shm]=baselines/ppo/ppo_memtasks_dinov2_shm.py [srbtr-nolstm]=baselines/ppo/ppo_memtasks_ebm_srb_tr_nolstm_cres_caps.py); CAPS="--caps-lambda-t=0.15";;
  caps)     declare -A E=([srbtr]=baselines/ppo/ppo_memtasks_ebm_srb_tr_caps.py [gru]=baselines/ppo/ppo_memtasks_dinov2_gru_caps.py [lstm]=baselines/ppo/ppo_memtasks_dinov2_lstm_caps.py [mlp]=baselines/ppo/ppo_memtasks_dinov2_mlp_caps.py [ffm]=baselines/ppo/ppo_memtasks_dinov2_ffm.py [shm]=baselines/ppo/ppo_memtasks_dinov2_shm.py); CAPS="--caps-lambda-t=0.15";;
  plain)    declare -A E=([srbtr]=baselines/ppo/ppo_memtasks_ebm_srb_tr.py [gru]=baselines/ppo/ppo_memtasks_dinov2_gru.py [lstm]=baselines/ppo/ppo_memtasks_dinov2_lstm.py [mlp]=baselines/ppo/ppo_memtasks_dinov2_mlp.py [srbtr-nolstm]=baselines/ppo/ppo_memtasks_ebm_srb_tr_nolstm.py [ffm]=baselines/ppo/ppo_memtasks_dinov2_ffm.py [shm]=baselines/ppo/ppo_memtasks_dinov2_shm.py); CAPS="";;
  *) echo "bad CFG $CFG"; exit 1;;
esac
if [[ "$ENVID" == Intercept* ]]; then NS=90;ES=540;EF=16; else NS=60;ES=720;EF=24; fi
for M in "${METHODS[@]}"; do for SEED in "${SEEDS[@]}"; do
  XTRA=""; if [[ ( "$M" == ffm || "$M" == shm ) && "$CFG" == cres_caps ]]; then XTRA="--color-aug"; fi
  RUN="${M}-${CFG}-seed${SEED}__${ENVID}"; LOG="$LOGDIR/${RUN}.log"
  if [[ -f "$LOG" ]] && grep -q "DONE $RUN rc=0" "$LOG"; then echo "===== SKIP (done) $RUN ====="; continue; fi
  echo "===== START $RUN (GPU$GPU, ${STEPS} steps) $(date) ====="
  "$MM" run -n mikasa python3 "${E[$M]}" \
    --env-id="$ENVID" --exp-name="${M}-${CFG}-seed${SEED}" --capture-video --save-model \
    --num-steps=$NS --num-eval-steps=$ES --eval-freq=$EF \
    --include-rgb --include-joints --seed=$SEED \
    --total-timesteps=$STEPS --no-finite-horizon-gae \
    --num-envs=256 --num-eval-envs=16 --num-minibatches=32 --update-epochs=2 \
    --gae-lambda=0.9 --gamma=0.99 --learning-rate=1e-4 --anneal-lr \
    --ent-coef=0.001 --target-kl=0.05 $CAPS $XTRA \
    > "$LOG" 2>&1 && echo "===== DONE $RUN rc=0 $(date) =====" || echo "!!!!! FAILED $RUN rc=$? $(date) !!!!!"
done; done
echo "ALL DONE $ENVID/$CFG (GPU$GPU) $(date)"
