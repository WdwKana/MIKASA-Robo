#!/bin/bash
#SBATCH --job-name=srbtr-rc9
#SBATCH --partition=gpu-l4-24g
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2
#SBATCH --output=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/srbtr-rc9-%A_%a.out
#SBATCH --error=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/srbtr-rc9-%A_%a.err

set -euo pipefail
cd /zfsstore/user/s4176650/MIKASA-Robo
export PYTHONUNBUFFERED=1
source /home/s4176650/.conda/envs/mikasa/etc/conda/activate.d/*.sh 2>/dev/null || true
export PATH="/home/s4176650/.conda/envs/mikasa/bin:$PATH"
export HF_HUB_CACHE=/zfsstore/user/s4176650/.cache/huggingface/hub
export TRANSFORMERS_VERBOSITY=error

SEEDS=(33 42 99)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "=== job=$SLURM_JOB_ID seed=$SEED node=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python3 baselines/ppo/ppo_memtasks_ebm_srb_tr.py \
    --env_id=RememberColor9-v0 \
    --exp-name=ppo-srbtr-rc9-seed${SEED} \
    --capture-video --save-model \
    --num-steps=60 \
    --num-eval-steps=720 \
    --include-rgb --include-joints \
    --seed=${SEED} \
    --total-timesteps=10_000_000 \
    --no-finite-horizon-gae \
    --eval-freq=48 \
    --num-envs=256 --num-eval-envs=16 \
    --num-minibatches=32 \
    --update-epochs=2 \
    --gae-lambda=0.9 --gamma=0.99 \
    --learning-rate=1e-4 --anneal-lr \
    --ent-coef=0.001 \
    --target-kl=0.05
