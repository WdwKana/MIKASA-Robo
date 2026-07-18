#!/bin/bash
# remember_nolstm_lane.sh <GPU> <WAIT_SESSION> <TASK1> [TASK2 ...]
# Waits for tmux session to end, SMOKE-GATES the new nolstm_cres_caps entry
# (35k steps on the first task), then runs srbtr-nolstm cres_caps x {33,42,99}
# per task sequentially. Aborts loudly if the smoke fails.
set -uo pipefail
GPU="$1"; WAIT="$2"; shift 2; TASKS=("$@")
SOCK=/home/wangdw/.tmux_sockets/mikasa
unset CONDA_EXE CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_PYTHON_EXE CONDA_DEFAULT_ENV
export MAMBA_ROOT_PREFIX="$HOME/micromamba"; export CONDARC="$HOME/micromamba/.condarc_empty"
export PYTHONUNBUFFERED=1; export TRANSFORMERS_VERBOSITY=error
MM="$HOME/micromamba/bin/micromamba"; cd /home/wangdw/MIKASA-Robo

echo "[lane] waiting for '$WAIT' to finish... $(date)"
while tmux -S "$SOCK" ls -F '#{session_name}' 2>/dev/null | grep -qx "$WAIT"; do sleep 60; done
echo "[lane] '$WAIT' done. SMOKE GATE on GPU$GPU (${TASKS[0]}, 35k steps) $(date)"

SLOG=/home/wangdw/mikasa_plan1_logs/_smoke_nolstmcres_gpu${GPU}.log
CUDA_VISIBLE_DEVICES=$GPU "$MM" run -n mikasa python3 baselines/ppo/ppo_memtasks_ebm_srb_tr_nolstm_cres_caps.py \
  --env-id="${TASKS[0]}" --exp-name=smoke-nolstmcres --no-capture-video --no-save-model \
  --num-steps=60 --num-eval-steps=720 --eval-freq=2 --include-rgb --include-joints --seed=33 \
  --total-timesteps=35000 --no-finite-horizon-gae --num-envs=256 --num-eval-envs=16 \
  --num-minibatches=32 --update-epochs=2 --gae-lambda=0.9 --gamma=0.99 --learning-rate=1e-4 \
  --anneal-lr --ent-coef=0.001 --target-kl=0.05 --caps-lambda-t=0.15 > "$SLOG" 2>&1
if ! grep -q "Evaluation Metric" "$SLOG"; then
  echo "[lane] !!!!! SMOKE FAILED on GPU$GPU — ABORTING LANE (see $SLOG) !!!!!"
  grep -m2 -E "Traceback|Error" "$SLOG"; exit 1
fi
echo "[lane] smoke PASS on GPU$GPU. Cleaning smoke dirs, starting batch."
find checkpoints -maxdepth 6 -type d -name "smoke-nolstmcres*" -exec rm -rf {} + 2>/dev/null

for T in "${TASKS[@]}"; do
  METHODS='srbtr-nolstm' SEEDS='33 42 99' bash /home/wangdw/MIKASA-Robo/run_task_cfg.sh "$GPU" "$T" cres_caps 7000000
done
echo "[lane] ALL TASKS DONE on GPU$GPU $(date)"
