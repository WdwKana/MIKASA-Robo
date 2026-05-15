"""Generate SLURM for V2 EBM-Hybrid on trajectory tasks (imed, intercept-grab*)
+ regression on rc9 (must not lose episodic strength).

Tasks per user:
  - InterceptMedium (compare V2 vs V1 EBM 0.866 / LSTM 0.896)
  - InterceptGrabMedium (new — test grab task)
  - RememberColor9 (regression — ensure episodic strength preserved)
  - RememberShape5 (regression — current strongest ours-vs-baseline gap)
"""
from pathlib import Path

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
OUT  = ROOT / "run_scripts" / "ppo_v2_hybrid"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = [
    ("InterceptMedium-v0",     "imed",     90, 1080, 25, "12:00:00"),
    ("InterceptGrabMedium-v0", "igrabmed", 90, 1080, 25, "12:00:00"),
    ("RememberColor9-v0",      "rc9",      60,  720, 48, "10:00:00"),
    ("RememberShape5-v0",      "shape5",   60,  720, 48, "10:00:00"),
]

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=v2hyb-{short}
#SBATCH --partition=gpu-l4-24g
#SBATCH --time={time_limit}
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2
#SBATCH --output=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/v2hyb-{short}-%A_%a.out
#SBATCH --error=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/v2hyb-{short}-%A_%a.err

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

python3 baselines/ppo/ppo_memtasks_ebm_hybrid.py \\
    --env_id={env_id} \\
    --exp-name=ppo-ebm-hybrid-{short}-seed${{SEED}} \\
    --capture-video --save-model \\
    --num-steps={ns} \\
    --num-eval-steps={nes} \\
    --include-rgb --include-joints \\
    --seed=${{SEED}} \\
    --total-timesteps=10_000_000 \\
    --no-finite-horizon-gae \\
    --eval-freq={ef} \\
    --num-envs=256 --num-eval-envs=16 \\
    --num-minibatches=32 \\
    --update-epochs=2 \\
    --gae-lambda=0.9 --gamma=0.99 \\
    --learning-rate=1e-4 --anneal-lr \\
    --ent-coef=0.001 \\
    --target-kl=0.05 \\
    --saliency-ckpt=analysis/ebm/path_a_head_v3.pt
"""

n = 0
for env_id, short, ns, nes, ef, tl in TASKS:
    out_path = OUT / f"{short}.slurm"
    out_path.write_text(TEMPLATE.format(
        env_id=env_id, short=short, ns=ns, nes=nes, ef=ef, time_limit=tl,
    ))
    print(f"  wrote {out_path}")
    n += 1
print(f"total: {n} files")
