"""Generate SLURM files for the MemoryVLA-style baseline:
  Tasks: RC5, Shape5
  Method: ppo_memtasks_ebm_memvla.py (push-all + cosine-pair-merge bank,
          2-layer cross-attention, no saliency)
  Total: 2 tasks × 3 seeds = 2 array files, 6 runs.

Time limit is generous (~16h) because the push-all + iterative pair-merge
adds ~260ms/step vs ~30ms for our saliency-filtered K=8 push (measured at
B=256, L=64). Over 90 steps that adds ~20s per rollout iteration, so the
same total-timesteps budget takes substantially longer than ours.

Run: python run_scripts/_gen_memvla_slurm.py
"""
from pathlib import Path

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")

TASKS = [
    ("RememberColor5-v0",      "rc5",      60, 720, 48),
    ("RememberShape5-v0",      "shape5",   60, 720, 48),
    # Trajectory tasks — testing whether MemVLA-style K-V buffer fails on
    # tasks that need continuous-state integration (velocity, ball tracking).
    ("InterceptMedium-v0",     "imed",     90, 540, 25),
    ("InterceptSlow-v0",       "islow",    90, 540, 25),
    ("InterceptFast-v0",       "ifast",    90, 540, 25),
    ("InterceptGrabMedium-v0", "igrabmed", 90, 540, 25),
]

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=memvla-{short}
#SBATCH --partition=gpu-l4-24g
#SBATCH --time=16:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2
#SBATCH --output=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/memvla-{short}-%A_%a.out
#SBATCH --error=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/memvla-{short}-%A_%a.err

set -euo pipefail
cd /zfsstore/user/s4176650/MIKASA-Robo
export PYTHONUNBUFFERED=1
source /home/s4176650/.conda/envs/mikasa/etc/conda/activate.d/*.sh 2>/dev/null || true
export PATH="/home/s4176650/.conda/envs/mikasa/bin:$PATH"
export HF_HUB_CACHE=/zfsstore/user/s4176650/.cache/huggingface/hub
export TRANSFORMERS_VERBOSITY=error

SEEDS=(33 42 99)
SEED=${{SEEDS[$SLURM_ARRAY_TASK_ID]}}
echo "=== job=$SLURM_JOB_ID  array=$SLURM_ARRAY_TASK_ID  seed=$SEED  node=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python3 baselines/ppo/ppo_memtasks_ebm_memvla.py \\
    --env_id={env_id} \\
    --exp-name=ppo-memvla-{short}-seed${{SEED}} \\
    --capture-video --save-model \\
    --num-steps={num_steps} \\
    --num-eval-steps={num_eval_steps} \\
    --include-rgb --include-joints \\
    --seed=${{SEED}} \\
    --total-timesteps=10_000_000 \\
    --no-finite-horizon-gae \\
    --eval-freq={eval_freq} \\
    --num-envs=256 --num-eval-envs=16 \\
    --num-minibatches=32 \\
    --update-epochs=2 \\
    --gae-lambda=0.9 --gamma=0.99 \\
    --learning-rate=1e-4 --anneal-lr \\
    --ent-coef=0.001 \\
    --target-kl=0.05 \\
    --saliency-ckpt=analysis/ebm/path_a_head_v3.pt
"""


def main():
    out_dir = ROOT / "run_scripts" / "ppo_memvla"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for env_id, short, ns, nes, ef in TASKS:
        content = TEMPLATE.format(
            short=short, env_id=env_id,
            num_steps=ns, num_eval_steps=nes, eval_freq=ef,
        )
        path = out_dir / f"memvla_{short}.slurm"
        path.write_text(content)
        n += 1
        print(f"  wrote {path}")
    print(f"\ntotal: {n} array files in {out_dir} ({n*3} runs)")


if __name__ == "__main__":
    main()
