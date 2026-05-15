"""
Plot success_once / return curves for all 4 methods on RC5 + RC9.

Reads training_metrics.csv from each run, picks eval rows, aggregates
mean ± std across 3 seeds, and plots side-by-side.

Output:
  analysis/ebm/results/main_table_rc5_rc9.png
  analysis/ebm/results/_summary.csv
"""

import csv
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
CKPT = ROOT / "checkpoints/ppo_memtasks/rgb_joints/normalized_dense"
OUT  = ROOT / "analysis/ebm/results"
OUT.mkdir(parents=True, exist_ok=True)

# (display_name, exp_name_glob_token, color)
METHODS = [
    ("MLP (no memory)", "ppo-mlp-dual",  "#999999"),
    ("GRU",             "ppo-gru-dual",  "#1f77b4"),
    ("LSTM",            "ppo-lstm-dual", "#2ca02c"),
    ("EBM-Robo (ours)", "ppo-ebm",       "#d62728"),
]
ENVS = ["RememberColor5-v0", "RememberColor9-v0"]
SEEDS = [33, 42, 99]


def find_csv(env, exp_token, seed):
    # env="RememberColor5-v0" -> "5"; "RememberColor9-v0" -> "9"
    rc_n = env[13]
    pattern = f"{exp_token}-rc{rc_n}-seed{seed}__{seed}__rgb_joints__*"
    matches = sorted((CKPT / env).glob(pattern + "/*/training_metrics.csv"))
    return matches[-1] if matches else None


def load_eval_curve(csv_path):
    """Returns (env_steps, success_once, success_at_end, return_) arrays for eval rows."""
    steps, sr, sae, ret = [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("mode") != "eval":
                continue
            try:
                steps.append(float(row["total_env_steps"]))
                sr.append(float(row["success_once"]))
                sae.append(float(row.get("success_at_end", 0.0)))
                ret.append(float(row["return"]))
            except (KeyError, ValueError):
                continue
    return np.array(steps), np.array(sr), np.array(sae), np.array(ret)


def aggregate_seeds(curves):
    """curves: list of (steps, sr, sae, ret) — align on shortest length."""
    if not curves:
        return None
    min_len = min(len(c[0]) for c in curves)
    if min_len == 0:
        return None
    steps = curves[0][0][:min_len]
    sr_stack  = np.stack([c[1][:min_len] for c in curves])
    sae_stack = np.stack([c[2][:min_len] for c in curves])
    ret_stack = np.stack([c[3][:min_len] for c in curves])
    return {
        "steps":     steps,
        "sr_mean":   sr_stack.mean(0),  "sr_std":   sr_stack.std(0),
        "sae_mean":  sae_stack.mean(0), "sae_std":  sae_stack.std(0),
        "rt_mean":   ret_stack.mean(0), "rt_std":   ret_stack.std(0),
        "n_seeds":   len(curves),
    }


def collect():
    out = defaultdict(dict)
    for env in ENVS:
        for name, tok, color in METHODS:
            curves = []
            for seed in SEEDS:
                csv_path = find_csv(env, tok, seed)
                if csv_path is None:
                    continue
                steps, sr, sae, ret = load_eval_curve(csv_path)
                if len(steps) > 0:
                    curves.append((steps, sr, sae, ret))
                    print(f"  [{env}][{name}] seed={seed}: {len(steps)} eval pts, "
                          f"final SO={sr[-1]:.3f} SE={sae[-1]:.3f} ret={ret[-1]:.3f}")
                else:
                    print(f"  [{env}][{name}] seed={seed}: NO eval rows yet")
            agg = aggregate_seeds(curves)
            if agg is not None:
                agg["color"] = color
                out[env][name] = agg
    return out


def plot(data):
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    for col, env in enumerate(ENVS):
        ax_so  = axes[0, col]
        ax_se  = axes[1, col]
        ax_ret = axes[2, col]
        for name, _, color in METHODS:
            if name not in data[env]:
                continue
            d = data[env][name]
            for ax, key_m, key_s, ylabel in [
                (ax_so,  "sr_mean",  "sr_std",  "success_once"),
                (ax_se,  "sae_mean", "sae_std", "success_at_end"),
                (ax_ret, "rt_mean",  "rt_std",  "episode return"),
            ]:
                ax.plot(d["steps"], d[key_m], label=f"{name} (n={d['n_seeds']})",
                        color=color, lw=1.6)
                ax.fill_between(d["steps"], d[key_m]-d[key_s], d[key_m]+d[key_s],
                                color=color, alpha=0.15)
        for row, ylabel in enumerate(["success_once", "success_at_end", "episode return"]):
            ax = axes[row, col]
            ax.set_title(f"{env}  —  {ylabel}")
            ax.set_xlabel("env steps");  ax.set_ylabel(ylabel)
            if row in (0, 1):
                ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("EBM-Robo vs baselines on RememberColor — dual-camera, working hypers (3 seeds)",
                 fontsize=12)
    fig.tight_layout()
    out_path = OUT / "main_table_rc5_rc9.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nplot -> {out_path}")
    return out_path


def write_summary_csv(data):
    rows = [["env", "method", "n_seeds", "final_step",
             "final_so_mean", "final_so_std", "peak_so_mean",
             "final_se_mean", "final_se_std", "peak_se_mean",
             "final_return_mean", "final_return_std"]]
    for env in ENVS:
        for name, _, _ in METHODS:
            if name not in data[env]:
                continue
            d = data[env][name]
            rows.append([
                env, name, d["n_seeds"], int(d["steps"][-1]),
                f"{d['sr_mean'][-1]:.4f}",  f"{d['sr_std'][-1]:.4f}",  f"{d['sr_mean'].max():.4f}",
                f"{d['sae_mean'][-1]:.4f}", f"{d['sae_std'][-1]:.4f}", f"{d['sae_mean'].max():.4f}",
                f"{d['rt_mean'][-1]:.4f}",  f"{d['rt_std'][-1]:.4f}",
            ])
    out_path = OUT / "_summary.csv"
    with open(out_path, "w") as f:
        w = csv.writer(f); w.writerows(rows)
    print(f"summary -> {out_path}")


def main():
    print("=== Collecting curves ===")
    data = collect()
    plot(data)
    write_summary_csv(data)
    # also print summary table
    print("\n=== Summary (final ± std | peak) ===")
    for env in ENVS:
        print(f"\n  {env}:")
        for name, _, _ in METHODS:
            if name in data[env]:
                d = data[env][name]
                print(f"    {name:24s}  "
                      f"SO: final={d['sr_mean'][-1]:.3f}±{d['sr_std'][-1]:.3f}  peak={d['sr_mean'].max():.3f}  | "
                      f"SE: final={d['sae_mean'][-1]:.3f}±{d['sae_std'][-1]:.3f}  peak={d['sae_mean'].max():.3f}")


if __name__ == "__main__":
    main()
