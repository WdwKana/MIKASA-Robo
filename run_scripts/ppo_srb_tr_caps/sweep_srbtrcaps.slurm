#!/bin/bash
#SBATCH --job-name=srbtrcaps-sweep
#SBATCH --partition=gpu-l4-24g
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-7
#SBATCH --output=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/srbtrcaps-sweep-%A_%a.out
#SBATCH --error=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/srbtrcaps-sweep-%A_%a.err

set -euo pipefail
cd /zfsstore/user/s4176650/MIKASA-Robo
export PYTHONUNBUFFERED=1
source /home/s4176650/.conda/envs/mikasa/etc/conda/activate.d/*.sh 2>/dev/null || true
export PATH="/home/s4176650/.conda/envs/mikasa/bin:$PATH"
export HF_HUB_CACHE=/zfsstore/user/s4176650/.cache/huggingface/hub
export TRANSFORMERS_VERBOSITY=error

# Clean-base SRB-TR+CAPS λ ablation on Color5 + Shape5, seed 33.
# λ=0.0 cells are sanity refs: CAPS entry with caps_lambda_t=0 must == plain SRB-TR.
ENVS=(RememberColor5-v0 RememberColor5-v0 RememberColor5-v0 RememberColor5-v0 \
      RememberShape5-v0 RememberShape5-v0 RememberShape5-v0 RememberShape5-v0)
LAMS=(0.0 0.05 0.15 0.5 0.0 0.05 0.15 0.5)
ENV=${ENVS[$SLURM_ARRAY_TASK_ID]}
LAM=${LAMS[$SLURM_ARRAY_TASK_ID]}
SEED=33
SHORT=$(echo $ENV | sed 's/Remember//; s/-v0//')
echo "=== job=$SLURM_JOB_ID env=$ENV caps_lambda_t=$LAM seed=$SEED node=$(hostname) ==="

python3 baselines/ppo/ppo_memtasks_ebm_srb_tr_caps.py \
    --env_id=${ENV} \
    --exp-name=srbtrcaps-${SHORT}-lamT${LAM}-seed${SEED} \
    --capture-video --save-model \
    --num-steps=60 --num-eval-steps=720 \
    --include-rgb --include-joints \
    --seed=${SEED} \
    --total-timesteps=10_000_000 \
    --no-finite-horizon-gae --eval-freq=48 \
    --num-envs=256 --num-eval-envs=16 \
    --num-minibatches=32 --update-epochs=2 \
    --gae-lambda=0.9 --gamma=0.99 \
    --learning-rate=1e-4 --anneal-lr \
    --ent-coef=0.001 --target-kl=0.05 \
    --caps-lambda-t=${LAM}
