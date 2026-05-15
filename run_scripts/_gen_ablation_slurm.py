"""Generate ablation SLURM files for Tier 0:
  Tasks: RC5, Shape5
  Methods: A6 family (mlp/gru/lstm + dinov2+sal) + A3 (ebm --no-saliency)
           + A_backbone (ebm --vit-backbone clip)
  Total: 2 tasks × 5 methods = 10 array files (3 seeds each = 30 runs).

Run: python run_scripts/_gen_ablation_slurm.py
"""
from pathlib import Path

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")

# (env_id, short, num_steps, num_eval_steps, eval_freq)
TASKS = [
    ("RememberColor5-v0", "rc5",    60, 720, 48),
    ("RememberShape5-v0", "shape5", 60, 720, 48),
]

# (slug, python_entry, time_limit, extra_flags)
METHODS = [
    # A6 family: same encoder as ours, different memory mechanism
    ("gru_dinov2sal",  "ppo_memtasks_gru_dinov2sal.py",  "06:00:00",
        " \\\n    --saliency-ckpt=analysis/ebm/path_a_head_v3.pt"),
    ("lstm_dinov2sal", "ppo_memtasks_lstm_dinov2sal.py", "06:00:00",
        " \\\n    --saliency-ckpt=analysis/ebm/path_a_head_v3.pt \\\n"
        "    --lstm-hidden-size=512 --lstm-num-layers=1 --lstm-dropout=0.0"),
    ("mlp_dinov2sal",  "ppo_memtasks_mlp_dinov2sal.py",  "06:00:00",
        " \\\n    --saliency-ckpt=analysis/ebm/path_a_head_v3.pt"),
    # A3: ours but without saliency-based filter (writes all DINOv2 patches)
    ("ebm_nosal",      "ppo_memtasks_ebm.py",            "10:00:00",
        " \\\n    --saliency-ckpt=analysis/ebm/path_a_head_v3.pt \\\n"
        "    --no-saliency"),
    # A_backbone: ours but with CLIP-ViT-B/16 instead of DINOv2-S/14
    ("ebm_clip",       "ppo_memtasks_ebm.py",            "12:00:00",
        " \\\n    --saliency-ckpt=analysis/ebm/path_a_head_v3_clip.pt \\\n"
        "    --vit-backbone=clip"),
]

TEMPLATE = """#!/bin/bash
#SBATCH --job-name={slug_short}-{short}
#SBATCH --partition=gpu-l4-24g
#SBATCH --time={time_limit}
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2
#SBATCH --output=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/{slug_short}-{short}-%A_%a.out
#SBATCH --error=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/{slug_short}-{short}-%A_%a.err

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

python3 baselines/ppo/{python_entry} \\
    --env_id={env_id} \\
    --exp-name=ppo-{slug_under}-{short}-seed${{SEED}} \\
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
    --target-kl=0.05{extra_flags}
"""


def main():
    out_dir = ROOT / "run_scripts" / "ppo_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for env_id, short, ns, nes, ef in TASKS:
        for slug, entry, tl, extra in METHODS:
            # short slug for SLURM job name (max chars)
            slug_short = {
                "gru_dinov2sal":  "a6-gru",
                "lstm_dinov2sal": "a6-lstm",
                "mlp_dinov2sal":  "a6-mlp",
                "ebm_nosal":      "a3-nosal",
                "ebm_clip":       "ab-clip",
            }[slug]
            content = TEMPLATE.format(
                slug_short=slug_short,
                short=short, time_limit=tl,
                python_entry=entry,
                env_id=env_id,
                slug_under=slug.replace("_", "-"),
                num_steps=ns, num_eval_steps=nes, eval_freq=ef,
                extra_flags=extra,
            )
            path = out_dir / f"{slug}_{short}.slurm"
            path.write_text(content)
            n += 1
            print(f"  wrote {path}")
    print(f"\ntotal: {n} files in {out_dir}")


if __name__ == "__main__":
    main()
