#!/bin/bash
#SBATCH --job-name=mlp-cc-l4main
#SBATCH --partition=gpu-l4-24g
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-17
#SBATCH --output=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/mlpcc-l4main-%A_%a.out
#SBATCH --error=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/mlpcc-l4main-%A_%a.err

set -euo pipefail
cd /zfsstore/user/s4176650/MIKASA-Robo
export PYTHONUNBUFFERED=1
source /home/s4176650/.conda/envs/mikasa/etc/conda/activate.d/*.sh 2>/dev/null || true
export PATH="/home/s4176650/.conda/envs/mikasa/bin:$PATH"
export HF_HUB_CACHE=/zfsstore/user/s4176650/.cache/huggingface/hub
export TRANSFORMERS_VERBOSITY=error

SEEDS=(33 42 99)
TASKS=(RememberColor5-v0 RememberColor9-v0 RememberShape5-v0 RememberShape9-v0 RememberShapeAndColor3x2-v0 RememberShapeAndColor3x3-v0)
ENV=${TASKS[$(( SLURM_ARRAY_TASK_ID / 3 ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % 3 ))]}
if [[ "$ENV" == Intercept* ]]; then NUMSTEPS=90; EVALSTEPS=540; EVALFREQ=16; else NUMSTEPS=60; EVALSTEPS=720; EVALFREQ=24; fi
echo "=== job=$SLURM_JOB_ID env=$ENV seed=$SEED part=gpu-l4-24g ==="

python3 baselines/ppo/ppo_memtasks_dinov2_mlp_cres_caps.py \
    --env_id=${ENV} --exp-name=mlp-crescaps-seed${SEED} \
    --capture-video --save-model \
    --num-steps=${NUMSTEPS} --num-eval-steps=${EVALSTEPS} \
    --include-rgb --include-joints --seed=${SEED} \
    --total-timesteps=7_000_000 --no-finite-horizon-gae --eval-freq=${EVALFREQ} \
    --num-envs=256 --num-eval-envs=16 --num-minibatches=32 --update-epochs=2 \
    --gae-lambda=0.9 --gamma=0.99 --learning-rate=1e-4 --anneal-lr \
    --ent-coef=0.001 --target-kl=0.05 --caps-lambda-t=0.15 
