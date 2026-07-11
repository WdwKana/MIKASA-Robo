#!/bin/bash
# Exp 3 — probe memory formation across training checkpoints.
# For each ckpt of one run: collect a few rollout batches, then probe h at all t.
# Usage: bash exp3_training_probe.sh <run_ts_dir> <method> <env_id> <out_root> [batches]
set -euo pipefail

RUN_DIR="$1"       # .../lstm-crescaps-seed99__.../20260626_022714
METHOD="$2"        # lstm | gru
ENV_ID="$3"        # RememberColor5-v0
OUT_ROOT="$4"
BATCHES="${5:-6}"
PY=/home/s4176650/.conda/envs/mikasa/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"

for CKPT in $(ls "$RUN_DIR"/ckpt_*.pt | sort -t_ -k3 -n); do
  TAG=$(basename "$CKPT" .pt)
  OUT="$OUT_ROOT/$TAG"
  if [[ -f "$OUT/probe/probe_curves.json" ]]; then
    echo "skip $TAG (done)"; continue
  fi
  echo "=== $TAG ==="
  $PY -u "$HERE/exp2_collect_rollouts.py" --method "$METHOD" --env-id "$ENV_ID" \
      --ckpt "$CKPT" --batches "$BATCHES" --out "$OUT/rollouts"
  $PY -u "$HERE/exp2b_probe.py" --data "$OUT/rollouts" --out "$OUT/probe" --targets h
done
echo "ALL CHECKPOINTS DONE"
