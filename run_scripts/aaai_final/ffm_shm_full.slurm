#!/bin/bash
#SBATCH --job-name=ffmshm-full
#SBATCH --partition=gpu-l4-24g
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-59
#SBATCH --output=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/ffmshm-full-%A_%a.out
#SBATCH --error=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/ffmshm-full-%A_%a.err
# FFM + SHM baselines (vendored author cells, perception-matched harness), same protocol
# as the main matrix: Remember tasks = CRES+CAPS (--color-aug --caps-lambda-t=0.15),
# pure-dynamic Fast tasks = plain. Seeds 33/42/99, 7M steps, same schedules.
# Cell-specific hyperparameters use the authors' experiment defaults (entry defaults):
# FFM m=64 c=4 out=512; SHM H=72 post=1024.
set -euo pipefail
cd /zfsstore/user/s4176650/MIKASA-Robo
export PYTHONUNBUFFERED=1
source /home/s4176650/.conda/envs/mikasa/etc/conda/activate.d/*.sh 2>/dev/null || true
export PATH="/home/s4176650/.conda/envs/mikasa/bin:$PATH"
export HF_HUB_CACHE=/zfsstore/user/s4176650/.cache/huggingface/hub
export TRANSFORMERS_VERBOSITY=error

METHODS=(ffm shm)
declare -A ENTRY=(
  [ffm]=baselines/ppo/ppo_memtasks_dinov2_ffm.py
  [shm]=baselines/ppo/ppo_memtasks_dinov2_shm.py
)
TASKS=(RememberColor3-v0 RememberColor5-v0 RememberColor9-v0
       RememberShape3-v0 RememberShape5-v0 RememberShape9-v0
       RememberShapeAndColor3x2-v0 RememberShapeAndColor3x3-v0
       InterceptFast-v0 InterceptGrabFast-v0)
SEEDS=(33 42 99)

M=${METHODS[$(( SLURM_ARRAY_TASK_ID / 30 ))]}
REM=$(( SLURM_ARRAY_TASK_ID % 30 ))
ENV=${TASKS[$(( REM / 3 ))]}
SEED=${SEEDS[$(( REM % 3 ))]}

if [[ "$ENV" == Intercept* ]]; then
  NUMSTEPS=90; EVALSTEPS=540; EVALFREQ=16
  CFG=plain; TRICKS=""
else
  NUMSTEPS=60; EVALSTEPS=720; EVALFREQ=24
  CFG=crescaps; TRICKS="--color-aug --caps-lambda-t=0.15"
fi
echo "=== job=$SLURM_JOB_ID method=$M env=$ENV seed=$SEED config=$CFG ==="

python3 "${ENTRY[$M]}" \
    --env_id=${ENV} --exp-name=${M}-${CFG}-seed${SEED} \
    --capture-video --save-model \
    --num-steps=${NUMSTEPS} --num-eval-steps=${EVALSTEPS} --eval-freq=${EVALFREQ} \
    --include-rgb --include-joints --seed=${SEED} \
    --total-timesteps=7_000_000 --no-finite-horizon-gae \
    --num-envs=256 --num-eval-envs=16 --num-minibatches=32 --update-epochs=2 \
    --gae-lambda=0.9 --gamma=0.99 --learning-rate=1e-4 --anneal-lr \
    --ent-coef=0.001 --target-kl=0.05 ${TRICKS}
