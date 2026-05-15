"""Generate SLURM array files for {gru_dual, lstm_dual, mlp_dual, ebm} ×
{tasks}. Hyperparams are taken from _common.sh (the user's working set).
Run: python run_scripts/_gen_slurm.py
"""
from pathlib import Path

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")

TASKS = [
    # (env_id, short, num_steps, num_eval_steps, eval_freq, time_limit_baseline, time_limit_ebm)
    # MEASURED: baseline (lstm-dual-shape9) actually takes ~3:52h, not 1-1.5h I'd guessed.
    # Set baseline = 6h (1.5x safety margin), EBM = 10h (until first new EBM finishes
    # we don't know exact runtime, so keep safety margin > EBM).
    ("RememberColor5-v0",          "rc5",     60,  720, 48, "6:00:00", "10:00:00"),
    ("RememberColor9-v0",          "rc9",     60,  720, 48, "6:00:00", "10:00:00"),
    ("RememberShape5-v0",          "shape5",  60,  720, 48, "6:00:00", "10:00:00"),
    ("RememberShape9-v0",          "shape9",  60,  720, 48, "6:00:00", "10:00:00"),
    ("RememberShapeAndColor3x2-v0", "sac3x2", 60,  720, 48, "6:00:00", "10:00:00"),
    ("RememberShapeAndColor3x3-v0", "sac3x3", 60,  720, 48, "6:00:00", "10:00:00"),
    # ShellGameTouch removed — YCB asset SHA mismatch on HuggingFace, env creation fails.
    ("InterceptMedium-v0",         "imed",    90, 1080, 25, "7:00:00", "12:00:00"),
]

METHODS = [
    # (slug,  python_entry,                             extra_flags, time_index)
    ("gru_dual",  "ppo_memtasks_gru_dual.py",  "",                      0),
    ("lstm_dual", "ppo_memtasks_lstm_dual.py",
     " \\\n    --lstm-hidden-size=512 --lstm-num-layers=1 --lstm-dropout=0.0", 0),
    ("mlp_dual",  "ppo_memtasks_mlp_dual.py",  "",                      0),
    ("ebm",       "ppo_memtasks_ebm.py",
     " \\\n    --saliency-ckpt=analysis/ebm/path_a_head_v3.pt",         1),
]

TEMPLATE = """#!/bin/bash
#SBATCH --job-name={method_short}-{short}
#SBATCH --partition=gpu-l4-24g
#SBATCH --time={time_limit}
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2
#SBATCH --output=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/{method_short}-{short}-%A_%a.out
#SBATCH --error=/zfsstore/user/s4176650/MIKASA-Robo/slurm_logs/{method_short}-{short}-%A_%a.err

set -euo pipefail
cd /zfsstore/user/s4176650/MIKASA-Robo
export PYTHONUNBUFFERED=1
source /home/s4176650/.conda/envs/mikasa/etc/conda/activate.d/*.sh 2>/dev/null || true
export PATH="/home/s4176650/.conda/envs/mikasa/bin:$PATH"
{ebm_env}
SEEDS=(33 42 99)
SEED=${{SEEDS[$SLURM_ARRAY_TASK_ID]}}
echo "=== job=$SLURM_JOB_ID  array=$SLURM_ARRAY_TASK_ID  seed=$SEED  node=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python3 baselines/ppo/{python_entry} \\
    --env_id={env_id} \\
    --exp-name=ppo-{method_short_under}-{short}-seed${{SEED}} \\
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


EBM_ENV = """export HF_HUB_CACHE=/zfsstore/user/s4176650/.cache/huggingface/hub
export TRANSFORMERS_VERBOSITY=error
"""


def main():
    out_count = 0
    for env_id, short, ns, nes, ef, tlb, tle in TASKS:
        for slug, entry, extra, ti in METHODS:
            time_limit = tle if ti == 1 else tlb
            method_short = slug.replace("_", "-").replace("dual", "dual")  # leave dashes
            # cleaner: just use slug directly with dashes
            method_short = slug.replace("_dual", "-dual")
            method_short_under = slug
            # SLURM job name length limit 32 chars - keep short
            if method_short == "ebm":
                slurm_method_short = "ebm"
            else:
                slurm_method_short = method_short  # gru-dual / lstm-dual / mlp-dual
            ebm_env = EBM_ENV if slug == "ebm" else ""
            if env_id.startswith("RememberShapeAndColor"):
                folder_root = "remember_shape_and_color"
                file_short  = f"remember_shape_and_color_{env_id.split('Color')[-1].split('-')[0]}"
            elif env_id.startswith("RememberShape"):
                folder_root = "remember_shape"
                file_short  = f"remember_shape_{env_id.split('Shape')[-1].split('-')[0]}"
            elif env_id.startswith("RememberColor"):
                folder_root = "remember_color"
                file_short  = f"remember_color_{env_id.split('Color')[-1].split('-')[0]}"
            elif env_id.startswith("ShellGame"):
                folder_root = "shell_game"
                file_short  = f"shell_game_{env_id.split('Game')[-1].lower().split('-')[0]}"
            elif env_id.startswith("Intercept"):
                folder_root = "intercept"
                file_short  = f"intercept_{env_id.split('Intercept')[-1].lower().split('-')[0]}"
            else:
                folder_root = env_id.lower()
                file_short  = env_id.lower()

            run_dir = ROOT / "run_scripts" / "ppo" / f"ppo_{slug}" / "dense_reward" / folder_root / "rgb_joint"
            run_dir.mkdir(parents=True, exist_ok=True)
            content = TEMPLATE.format(
                method_short=slurm_method_short,
                short=short,
                time_limit=time_limit,
                ebm_env=ebm_env,
                python_entry=entry,
                env_id=env_id,
                method_short_under=method_short_under.replace("_dual", "-dual"),
                num_steps=ns, num_eval_steps=nes, eval_freq=ef,
                extra_flags=extra,
            )
            out_path = run_dir / f"{file_short}.slurm"
            out_path.write_text(content)
            out_count += 1
            print(f"  wrote {out_path}")
    print(f"total: {out_count} files")


if __name__ == "__main__":
    main()
