#!/bin/bash
#SBATCH --job-name=ebm-rc9-smoke
#SBATCH --partition=gpu-l4-24g
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/ebm-rc9-smoke-%j.out
#SBATCH --error=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/ebm-rc9-smoke-%j.err

set -euo pipefail

cd /zfsstore/user/s4176650/MIKASA-Robo
source /home/s4176650/.conda/envs/mikasa/etc/conda/activate.d/*.sh 2>/dev/null || true
export PATH="/home/s4176650/.conda/envs/mikasa/bin:$PATH"
export HF_HUB_CACHE=/zfsstore/user/s4176650/.cache/huggingface/hub
export TRANSFORMERS_VERBOSITY=error

echo "=== job=$SLURM_JOB_ID  node=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Smoke test goal: verify EBM PPO trains end-to-end on GPU for ~6 iterations.
# total-timesteps = 100K = 100_000 / (60 steps * 256 envs) ≈ 6.5 iterations.
# Single seed; not array. Catches OOM / NaN / dim mismatches early without
# burning compute on a full run.

python3 baselines/ppo/ppo_memtasks_ebm.py \
    --env_id=RememberColor9-v0 \
    --exp-name=ebm-rc9-smoke \
    --no-capture-video \
    --no-save-model \
    --num-steps=60 \
    --num-eval-steps=60 \
    --include-rgb \
    --include-joints \
    --seed=42 \
    --total-timesteps=100_000 \
    --no-finite-horizon-gae \
    --eval-freq=10 \
    --num-envs=256 \
    --num-minibatches=32 \
    --update-epochs=4 \
    --num-eval-envs=8 \
    --gae-lambda=0.9 \
    --gamma=0.8 \
    --learning-rate=3e-4 \
    --no-anneal-lr \
    --saliency-ckpt=analysis/ebm/path_a_head_v2.pt
