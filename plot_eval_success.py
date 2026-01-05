#!/usr/bin/env python3
"""
Plot eval success curves from MIKASA-Robo training_metrics.csv logs.

It produces two figures (PNG):
  1) eval success (success_once) vs steps (total_env_steps)
  2) eval success end (success_at_end) vs steps (total_env_steps)

By default it uses the file paths you listed in chat, but you can override them
via CLI flags.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _extract_seed_from_path(path: str) -> Optional[str]:
    # Examples:
    #   ...-v0__33__rgb_joints__.../training_metrics.csv
    #   ...-v0-belief__42__rgb_joints_belief__.../training_metrics.csv
    m = re.search(r"__([0-9]+)__", path)
    if m:
        return m.group(1)
    return None


def _load_eval_metrics(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = {"total_env_steps", "mode", "success_once", "success_at_end"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    df = df[df["mode"].astype(str).str.lower() == "eval"].copy()
    if df.empty:
        raise ValueError(f"{csv_path} has 0 eval rows after filtering mode==eval")

    # If there are repeated eval rows for the same step, average them.
    df = (
        df.sort_values("total_env_steps")
        .groupby("total_env_steps", as_index=False)[["success_once", "success_at_end"]]
        .mean()
    )
    return df


def _build_long_df(csv_paths: Iterable[str], group_name: str) -> pd.DataFrame:
    frames = []
    csv_paths = list(csv_paths)
    for i, p in enumerate(csv_paths):
        if not os.path.exists(p):
            raise FileNotFoundError(f"CSV not found: {p}")

        d = _load_eval_metrics(p)
        seed = _extract_seed_from_path(p) or str(i)
        d["seed"] = seed
        d["group"] = group_name
        frames.append(d)

    if not frames:
        raise ValueError(f"No CSVs provided for group={group_name}")
    return pd.concat(frames, ignore_index=True)


def _smooth_by_group(df: pd.DataFrame, value_col: str, window: int) -> pd.DataFrame:
    if window <= 1:
        return df

    out = df.copy()
    out[value_col] = (
        out.sort_values("total_env_steps")
        .groupby("group", as_index=False, group_keys=False)[value_col]
        .apply(lambda s: s.rolling(window=window, min_periods=1).mean())
    )
    return out


def _plot_metric(
    stats: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    out_path: str,
    show_std: bool,
    title: Optional[str] = None,
) -> None:
    plt.figure(figsize=(8, 4.8))

    for group_name, g in stats.groupby("group"):
        g = g.sort_values("total_env_steps")
        x = g["total_env_steps"].to_numpy()
        y = g[f"{metric_col}_mean"].to_numpy()

        plt.plot(x, y, linewidth=2.0, label=group_name)

        if show_std:
            y_std = g[f"{metric_col}_std"].fillna(0.0).to_numpy()
            plt.fill_between(x, y - y_std, y + y_std, alpha=0.18)

    plt.xlabel("Steps (total_env_steps)")
    plt.ylabel(ylabel)
    plt.ylim(0.0, 1.0)
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _prefix_algo_label(algo: str, label: str) -> str:
    algo = (algo or "").strip()
    label = (label or "").strip()

    if not algo:
        return label
    if not label:
        return algo

    # Avoid doubling like "PPO PPO rgb_joints"
    if label.lower().startswith(algo.lower()):
        return label
    return f"{algo} {label}"


def main() -> None:
    parser = argparse.ArgumentParser()

    default_belief = [
        "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberShapeAndColor3x2-v0/"
        "ppo-mlp-dense-remember-shape-and-color-3x2-v0-belief__33__rgb_joints_belief__20251222_195647/20251222_195647/training_metrics.csv",
        "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberShapeAndColor3x2-v0/"
        "ppo-mlp-dense-remember-shape-and-color-3x2-v0-belief__42__rgb_joints_belief__20251222_195659/20251222_195659/training_metrics.csv",
    ]
    default_rgb_joints = [
        "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints/normalized_dense/RememberShapeAndColor3x2-v0/"
        "ppo-mlp-dense-remember-shape-and-color-3x2-v0__33__rgb_joints__20251221_224825/20251221_224825/training_metrics.csv",
        "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints/normalized_dense/RememberShapeAndColor3x2-v0/"
        "ppo-mlp-dense-remember-shape-and-color-3x2-v0__42__rgb_joints__20251221_224831/20251221_224831/training_metrics.csv",
    ]

    parser.add_argument(
        "--belief",
        nargs="+",
        default=default_belief,
        help="One or more training_metrics.csv paths for the rgb_joints_belief group.",
    )
    parser.add_argument(
        "--rgb-joints",
        nargs="+",
        default=default_rgb_joints,
        help="One or more training_metrics.csv paths for the rgb_joints (non-belief) group.",
    )
    parser.add_argument(
        "--belief-label",
        default="rgb_joints_belief",
        help="Legend label for belief group.",
    )
    parser.add_argument(
        "--rgb-joints-label",
        default="rgb_joints",
        help="Legend label for non-belief group.",
    )
    parser.add_argument(
        "--algo",
        default="PPO",
        help="Prefix to add to group labels in legend (e.g., PPO). Use empty string to disable.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path.cwd() / "plots_eval"),
        help="Output directory for PNG files.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Rolling window size for smoothing the mean curve (1 disables).",
    )
    parser.add_argument(
        "--no-std",
        action="store_true",
        help="Disable std shading across seeds.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    belief_group = _prefix_algo_label(args.algo, args.belief_label)
    rgb_group = _prefix_algo_label(args.algo, args.rgb_joints_label)

    df_belief = _build_long_df(args.belief, belief_group)
    df_rgb = _build_long_df(args.rgb_joints, rgb_group)
    df = pd.concat([df_belief, df_rgb], ignore_index=True)

    # Long -> stats per group x step
    stats = (
        df.melt(
            id_vars=["group", "seed", "total_env_steps"],
            value_vars=["success_once", "success_at_end"],
            var_name="metric",
            value_name="value",
        )
        .groupby(["group", "total_env_steps", "metric"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Convert back into wide columns: success_once_mean/std, success_at_end_mean/std
    wide = stats.pivot_table(
        index=["group", "total_env_steps"],
        columns="metric",
        values=["mean", "std", "count"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{stat}" for stat, metric in wide.columns]
    wide = wide.reset_index().sort_values(["group", "total_env_steps"])

    wide = _smooth_by_group(wide, "success_once_mean", args.smooth)
    wide = _smooth_by_group(wide, "success_at_end_mean", args.smooth)

    _plot_metric(
        wide,
        metric_col="success_once",
        ylabel="Eval success (success_once)",
        out_path=str(out_dir / "eval_success.png"),
        show_std=not args.no_std,
        title=None,
    )
    _plot_metric(
        wide,
        metric_col="success_at_end",
        ylabel="Eval success end (success_at_end)",
        out_path=str(out_dir / "eval_success_end.png"),
        show_std=not args.no_std,
        title=None,
    )

    print(f"Saved: {out_dir / 'eval_success.png'}")
    print(f"Saved: {out_dir / 'eval_success_end.png'}")


if __name__ == "__main__":
    main()


