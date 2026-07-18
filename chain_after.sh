#!/bin/bash
# chain_after.sh <WAIT_SESSION> <GPU> <ENV_ID> <CFG> <STEPS>
# Waits for tmux session <WAIT_SESSION> (private socket) to end, then runs the batch.
set -uo pipefail
WAIT="$1"; GPU="$2"; ENVID="$3"; CFG="$4"; STEPS="$5"
SOCK=/home/wangdw/.tmux_sockets/mikasa
echo "[chain] waiting for session '$WAIT' to finish... $(date)"
while tmux -S "$SOCK" ls -F '#{session_name}' 2>/dev/null | grep -qx "$WAIT"; do sleep 60; done
echo "[chain] '$WAIT' finished -> starting $ENVID ($CFG) on GPU$GPU  $(date)"
METHODS="${METHODS:-srbtr gru lstm mlp}" SEEDS="${SEEDS:-33 42 99}" \
  bash /home/wangdw/MIKASA-Robo/run_task_cfg.sh "$GPU" "$ENVID" "$CFG" "$STEPS"
