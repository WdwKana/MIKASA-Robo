"""Generate SLURM files for V3/V4 hybrid variants.

Phase 1 — target tasks (per current research direction):
  - IMed   : primary trajectory task; we have V1=0.87 SR_once and V2=0.91
             SR_once. We need to know whether LSTM/Clean-GRU/Belief-state
             variants preserve or improve this, especially in SR_end.
  - RC9    : static-recall regression; V1=0.15, V2=0.086 (parallel GRU
             hurts). If V3b (clean GRU input) recovers V1's level, the
             dilution penalty was the GRU input, not the GRU.
  - Shape5 : sanity static-recall (V1=0.68 dominates); checks no variant
             tanks here.

Variants:
  V3a — LSTM-Hybrid       (parallel LSTM, same GRU input as V2)
  V3b — Clean-GRU Hybrid  (parallel GRU, input only [proprio, curr])
  V4a — Belief-GRU        (clean GRU input, GRU output drives both query and fuse)
  V4b — Belief-LSTM       (same as V4a with LSTM)

Three seeds each. Time limit generous since LSTM and belief variants are new.

Run: python run_scripts/_gen_v3v4_slurm.py
"""
from pathlib import Path

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")

TASKS = [
    ("InterceptMedium-v0", "imed",   90, 540, 25),
    ("RememberColor9-v0",  "rc9",    60, 720, 48),
    ("RememberShape5-v0",  "shape5", 60, 720, 48),
]

VARIANTS = [
    ("v3a-lstm",     "ppo_memtasks_ebm_hybrid_lstm.py"),
    ("v3b-cleangru", "ppo_memtasks_ebm_hybrid_clean.py"),
    ("v4a-belgru",   "ppo_memtasks_ebm_belief_gru.py"),
    ("v4b-bellstm",  "ppo_memtasks_ebm_belief_lstm.py"),
]

TEMPLATE = """#!/bin/bash
#SBATCH --job-name={slug}-{short}
#SBATCH --partition=gpu-l4-24g
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2
#SBATCH --output=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/{slug}-{short}-%A_%a.out
#SBATCH --error=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/{slug}-{short}-%A_%a.err

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

python3 baselines/ppo/{entry} \\
    --env_id={env_id} \\
    --exp-name=ppo-{slug}-{short}-seed${{SEED}} \\
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
    out_dir = ROOT / "run_scripts" / "ppo_v3v4"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for env_id, short, ns, nes, ef in TASKS:
        for slug, entry in VARIANTS:
            content = TEMPLATE.format(
                slug=slug, short=short, entry=entry,
                env_id=env_id, num_steps=ns,
                num_eval_steps=nes, eval_freq=ef,
            )
            path = out_dir / f"{slug}_{short}.slurm"
            path.write_text(content)
            n += 1
            print(f"  wrote {path}")
    print(f"\ntotal: {n} array files in {out_dir} ({n*3} runs)")


if __name__ == "__main__":
    main()
