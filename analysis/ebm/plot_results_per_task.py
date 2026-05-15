"""
Per-task curves: success_once + success_at_end as SEPARATE figures.
Y-axis upper bound adaptive to data magnitude (so EBM lead is visible on
low-SR tasks like color9 / shape9).

Output:
  analysis/ebm/results/{task}_success_once.png
  analysis/ebm/results/{task}_success_at_end.png
  analysis/ebm/results/_summary_per_task.csv
"""

import csv
import math
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
CKPT = ROOT / "checkpoints/ppo_memtasks/rgb_joints/normalized_dense"
OUT  = ROOT / "analysis/ebm/results"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = [
    ("MLP (no memory)", "ppo-mlp-dual",  "#999999"),
    ("GRU",             "ppo-gru-dual",  "#1f77b4"),
    ("LSTM",            "ppo-lstm-dual", "#2ca02c"),
    ("EBM-Robo (ours)", "ppo-ebm",       "#d62728"),
]

# (env_id, short_token-in-exp-name, display-name)
TASKS = [
    ("RememberColor5-v0",          "rc5",     "RememberColor5"),
    ("RememberColor9-v0",          "rc9",     "RememberColor9"),
    ("RememberShape5-v0",          "shape5",  "RememberShape5"),
    ("RememberShape9-v0",          "shape9",  "RememberShape9"),
    ("RememberShapeAndColor3x2-v0", "sac3x2", "RememberShapeAndColor3x2"),
    ("RememberShapeAndColor3x3-v0", "sac3x3", "RememberShapeAndColor3x3"),
    ("InterceptMedium-v0",         "imed",    "InterceptMedium"),
]
SEEDS = [33, 42, 99]


def find_csv(env, exp_token, short, seed):
    # exp-name pattern matches: ppo-{method}-{short}-seed{seed} OR
    # legacy ppo-{method}-dense-{env-slug}-seed{seed}.
    patterns = [
        f"{exp_token}-{short}-seed{seed}__{seed}__rgb_joints__*",
        f"{exp_token}-dense-*-seed{seed}__{seed}__rgb_joints__*",
    ]
    candidates = []
    for pat in patterns:
        candidates += list((CKPT / env).glob(pat + "/*/training_metrics.csv"))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1]   # newest timestamp


def load_eval_curve(csv_path):
    steps, so, sae = [], [], []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("mode") != "eval":
                continue
            try:
                steps.append(float(row["total_env_steps"]))
                so.append(float(row["success_once"]))
                sae.append(float(row.get("success_at_end", 0.0)))
            except (KeyError, ValueError):
                continue
    return np.array(steps), np.array(so), np.array(sae)


def aggregate(curves):
    if not curves:
        return None
    n = min(len(c[0]) for c in curves)
    if n == 0:
        return None
    steps = curves[0][0][:n]
    so  = np.stack([c[1][:n] for c in curves])
    sae = np.stack([c[2][:n] for c in curves])
    return {
        "steps":    steps,
        "so_mean":  so.mean(0),  "so_std":  so.std(0),
        "sae_mean": sae.mean(0), "sae_std": sae.std(0),
        "n_seeds":  len(curves),
    }


def adaptive_ylim(observed_max: float) -> float:
    """Round observed_max + 30% margin up to nearest tier in
    {0.4, 0.6, 0.8, 1.05}. Floor at 0.4."""
    cap = observed_max * 1.3
    for t in (0.4, 0.6, 0.8):
        if cap <= t:
            return t
    return 1.05


def collect_task(env, exp_token, short):
    data = {}
    for name, tok, color in METHODS:
        curves = []
        for seed in SEEDS:
            p = find_csv(env, tok, short, seed)
            if p is None:
                continue
            steps, so, sae = load_eval_curve(p)
            if len(steps) > 0:
                curves.append((steps, so, sae))
        agg = aggregate(curves)
        if agg is not None:
            agg["color"] = color
            data[name] = agg
    return data


def plot_one(task_display, data, metric_key_mean, metric_key_std, metric_label, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    obs_max = 0.0
    for name, _, color in METHODS:
        if name not in data:
            continue
        d = data[name]
        ax.plot(d["steps"], d[metric_key_mean], label=f"{name} (n={d['n_seeds']})",
                color=color, lw=2.0)
        ax.fill_between(d["steps"],
                        d[metric_key_mean]-d[metric_key_std],
                        d[metric_key_mean]+d[metric_key_std],
                        color=color, alpha=0.18)
        obs_max = max(obs_max, float(d[metric_key_mean].max()))
    yhi = adaptive_ylim(obs_max)
    ax.set_ylim(-0.02, yhi)
    ax.set_xlabel("environment steps", fontsize=11)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(f"{task_display} — {metric_label} (mean ± std, 3 seeds)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    summary_rows = [["env", "method", "n_seeds", "final_step",
                     "final_so_mean", "final_so_std", "peak_so_mean",
                     "final_sae_mean", "final_sae_std", "peak_sae_mean"]]
    for env, short, display in TASKS:
        print(f"\n=== {env} ===")
        data = collect_task(env, exp_token=None, short=short)
        # find_csv expects exp_token but we passed None: fix by re-calling
        data = {}
        for name, tok, color in METHODS:
            curves = []
            for seed in SEEDS:
                p = find_csv(env, tok, short, seed)
                if p is None:
                    continue
                steps, so, sae = load_eval_curve(p)
                if len(steps) > 0:
                    curves.append((steps, so, sae))
            agg = aggregate(curves)
            if agg is None:
                continue
            agg["color"] = color
            data[name] = agg
            print(f"  {name}: {agg['n_seeds']} seeds, {len(agg['steps'])} evals, "
                  f"final SO={agg['so_mean'][-1]:.3f}±{agg['so_std'][-1]:.3f}, "
                  f"final SAE={agg['sae_mean'][-1]:.3f}±{agg['sae_std'][-1]:.3f}")
        if not data:
            print(f"  no data, skipped")
            continue
        # plot
        out_so  = OUT / f"{short}_success_once.png"
        out_sae = OUT / f"{short}_success_at_end.png"
        plot_one(display, data, "so_mean",  "so_std",  "success_once",   out_so)
        plot_one(display, data, "sae_mean", "sae_std", "success_at_end", out_sae)
        print(f"    -> {out_so.name}, {out_sae.name}")
        for name, _, _ in METHODS:
            if name in data:
                d = data[name]
                summary_rows.append([env, name, d["n_seeds"], int(d["steps"][-1]),
                                     f"{d['so_mean'][-1]:.4f}", f"{d['so_std'][-1]:.4f}",
                                     f"{d['so_mean'].max():.4f}",
                                     f"{d['sae_mean'][-1]:.4f}", f"{d['sae_std'][-1]:.4f}",
                                     f"{d['sae_mean'].max():.4f}"])
    csv_path = OUT / "_summary_per_task.csv"
    with open(csv_path, "w") as f:
        w = csv.writer(f); w.writerows(summary_rows)
    print(f"\nsummary -> {csv_path}")
    print(f"plots in -> {OUT}")


if __name__ == "__main__":
    main()
