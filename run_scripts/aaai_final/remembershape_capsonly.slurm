#!/bin/bash
#SBATCH --job-name=rshape-caps
#SBATCH --partition=gpu-l4-24g
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-35
#SBATCH --output=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/rshape-caps-%A_%a.out
#SBATCH --error=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/rshape-caps-%A_%a.err
# RememberShape 3/5/9 x {srbtr,gru,lstm,mlp} x seeds{33,42,99}, CAPS-only (NO CRES).
# Shape is not a color cue -> CRES off (principled: CRES only for color tasks). CAPS on
# because success requires is_robot_static (terminal settling).
set -euo pipefail
cd /zfsstore/user/s4176650/MIKASA-Robo
export PYTHONUNBUFFERED=1
source /home/s4176650/.conda/envs/mikasa/etc/conda/activate.d/*.sh 2>/dev/null || true
export PATH="/home/s4176650/.conda/envs/mikasa/bin:$PATH"
export HF_HUB_CACHE=/zfsstore/user/s4176650/.cache/huggingface/hub
export TRANSFORMERS_VERBOSITY=error

METHODS=(srbtr gru lstm mlp)
declare -A ENTRY=(
  [srbtr]=baselines/ppo/ppo_memtasks_ebm_srb_tr_caps.py
  [gru]=baselines/ppo/ppo_memtasks_dinov2_gru_caps.py
  [lstm]=baselines/ppo/ppo_memtasks_dinov2_lstm_caps.py
  [mlp]=baselines/ppo/ppo_memtasks_dinov2_mlp_caps.py
)
TASKS=(RememberShape3-v0 RememberShape5-v0 RememberShape9-v0)
SEEDS=(33 42 99)

M=${METHODS[$(( SLURM_ARRAY_TASK_ID / 9 ))]}
REM=$(( SLURM_ARRAY_TASK_ID % 9 ))
ENV=${TASKS[$(( REM / 3 ))]}
SEED=${SEEDS[$(( REM % 3 ))]}
echo "=== job=$SLURM_JOB_ID method=$M env=$ENV seed=$SEED (CAPS-only, no CRES) ==="

python3 "${ENTRY[$M]}" \
    --env_id=${ENV} --exp-name=${M}-caps-seed${SEED} \
    --capture-video --save-model \
    --num-steps=60 --num-eval-steps=720 --eval-freq=24 \
    --include-rgb --include-joints --seed=${SEED} \
    --total-timesteps=7_000_000 --no-finite-horizon-gae \
    --num-envs=256 --num-eval-envs=16 --num-minibatches=32 --update-epochs=2 \
    --gae-lambda=0.9 --gamma=0.99 --learning-rate=1e-4 --anneal-lr \
    --ent-coef=0.001 --target-kl=0.05 --caps-lambda-t=0.15
