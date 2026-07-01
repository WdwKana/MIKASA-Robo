#!/bin/bash
#SBATCH --job-name=grabprobe-plain
#SBATCH --partition=gpu-l4-24g
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-5
#SBATCH --output=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/grabprobe-plain-%A_%a.out
#SBATCH --error=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/grabprobe-plain-%A_%a.err
# Head-to-head probe: plain SRB-TR (no CRES/CAPS) on InterceptGrab{Slow,Fast} x seeds{33,42,99}.
# We already have plain GrabMedium + crescaps for all 3 Grab; this fills Slow/Fast plain so we can
# decide the Grab config on the MEAN (not the lucky-seed ceiling).
set -euo pipefail
cd /zfsstore/user/s4176650/MIKASA-Robo
export PYTHONUNBUFFERED=1
source /home/s4176650/.conda/envs/mikasa/etc/conda/activate.d/*.sh 2>/dev/null || true
export PATH="/home/s4176650/.conda/envs/mikasa/bin:$PATH"
export HF_HUB_CACHE=/zfsstore/user/s4176650/.cache/huggingface/hub
export TRANSFORMERS_VERBOSITY=error

TASKS=(InterceptGrabSlow-v0 InterceptGrabFast-v0)
SEEDS=(33 42 99)
ENV=${TASKS[$(( SLURM_ARRAY_TASK_ID / 3 ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % 3 ))]}
echo "=== job=$SLURM_JOB_ID srbtr PLAIN env=$ENV seed=$SEED (no CRES, no CAPS) ==="

python3 baselines/ppo/ppo_memtasks_ebm_srb_tr.py \
    --env_id=${ENV} --exp-name=srbtr-plain-seed${SEED} \
    --capture-video --save-model \
    --num-steps=90 --num-eval-steps=540 --eval-freq=16 \
    --include-rgb --include-joints --seed=${SEED} \
    --total-timesteps=7_000_000 --no-finite-horizon-gae \
    --num-envs=256 --num-eval-envs=16 --num-minibatches=32 --update-epochs=2 \
    --gae-lambda=0.9 --gamma=0.99 --learning-rate=1e-4 --anneal-lr \
    --ent-coef=0.001 --target-kl=0.05
