#!/usr/bin/env python3
"""
Plot PPO memtasks curves comparing two observation modes:
  - rgb_joints
  - rgb_joints_belief

For each task, it:
  - finds the latest run for each seed (e.g., 33 and 42) under each mode
  - loads training_metrics.csv
  - filters to eval (default) or train rows
  - averages curves across seeds (per mode)
  - saves one PNG per task (title = task name)

This script ONLY reads existing logs and writes new plot images. It does not modify
any existing training / evaluation code.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MODES_DEFAULT = ("rgb_joints", "rgb_joints_belief")
SEEDS_DEFAULT = (33, 42)
METRICS_DEFAULT = ("success_once", "success_at_end")


_TS_RE = re.compile(r"__(\d{8}_\d{6})(?:/|$)")


@dataclass(frozen=True)
class RunSelection:
    task: str
    mode: str
    seed: int
    csv_path: Path
    timestamp_tag: str


def _timestamp_tag_from_path(p: Path) -> Optional[str]:
    """
    Extract YYYYMMDD_HHMMSS tags from a path like:
      ...__20251226_170323/.../training_metrics.csv
    If multiple tags exist, returns the maximum (latest) tag.
    """
    s = p.as_posix()
    tags = _TS_RE.findall(s)
    if not tags:
        return None
    # Tags are lexicographically sortable due to fixed format YYYYMMDD_HHMMSS.
    return max(tags)


def _is_ppo_run_path(p: Path) -> bool:
    # Typical run directory names: "ppo-mlp-dense-...".
    return any(part.startswith("ppo-") for part in p.parts)


def _find_latest_training_metrics_csv(
    *,
    task_dir: Path,
    mode: str,
    seed: int,
    strict: bool,
) -> RunSelection:
    if not task_dir.exists():
        raise FileNotFoundError(f"Task dir not found: {task_dir}")

    needle = f"__{seed}__{mode}__"
    candidates: List[Tuple[str, Path]] = []
    for csv_path in task_dir.rglob("training_metrics.csv"):
        s = csv_path.as_posix()
        if needle not in s:
            continue
        if not _is_ppo_run_path(csv_path):
            continue

        tag = _timestamp_tag_from_path(csv_path)
        if tag is None:
            continue
        candidates.append((tag, csv_path))

    if not candidates:
        msg = (
            f"No training_metrics.csv found for task={task_dir.name}, mode={mode}, seed={seed} "
            f"under {task_dir}"
        )
        if strict:
            raise FileNotFoundError(msg)
        raise FileNotFoundError(msg)

    # pick latest by timestamp tag; tie-breaker: file mtime
    candidates.sort(key=lambda x: (x[0], x[1].stat().st_mtime))
    tag, csv_path = candidates[-1]

    return RunSelection(task=task_dir.name, mode=mode, seed=seed, csv_path=csv_path, timestamp_tag=tag)


def _load_metrics(
    csv_path: Path,
    *,
    mode_filter: str,
    metrics: Sequence[str],
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"total_env_steps", "mode", *metrics}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    mode_filter = mode_filter.strip().lower()
    df = df[df["mode"].astype(str).str.lower() == mode_filter].copy()
    if df.empty:
        raise ValueError(f"{csv_path} has 0 rows after filtering mode=={mode_filter}")

    # numeric + sort
    df["total_env_steps"] = pd.to_numeric(df["total_env_steps"], errors="coerce")
    df = df.dropna(subset=["total_env_steps"])
    df["total_env_steps"] = df["total_env_steps"].astype(np.int64)

    for c in metrics:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # average repeated rows at same step (if any)
    df = (
        df.sort_values("total_env_steps")
        .groupby("total_env_steps", as_index=False)[list(metrics)]
        .mean(numeric_only=True)
    )
    return df


def _mean_std_across_seeds(
    seed_dfs: Dict[int, pd.DataFrame],
    *,
    metrics: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      mean_df: columns = ["total_env_steps", *metrics]
      std_df:  columns = ["total_env_steps", *metrics]
    """
    frames = {}
    for seed, df in seed_dfs.items():
        frames[seed] = df.set_index("total_env_steps")[list(metrics)]

    wide = pd.concat(frames, axis=1).sort_index()

    means: Dict[str, pd.Series] = {}
    stds: Dict[str, pd.Series] = {}
    for metric in metrics:
        cols = [(seed, metric) for seed in frames.keys() if (seed, metric) in wide.columns]
        vals = wide[cols]
        means[metric] = vals.mean(axis=1)
        stds[metric] = vals.std(axis=1)

    mean_df = pd.DataFrame(means, index=wide.index).reset_index().rename(columns={"index": "total_env_steps"})
    std_df = pd.DataFrame(stds, index=wide.index).reset_index().rename(columns={"index": "total_env_steps"})
    return mean_df, std_df


def _rolling_smooth(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    if window <= 1:
        return df[col]
    return df[col].rolling(window=window, min_periods=1).mean()


def _infer_tasks(checkpoints_root: Path, *, modes: Sequence[str], variant: str) -> List[str]:
    task_sets: List[set] = []
    for mode in modes:
        d = checkpoints_root / mode / variant
        if not d.exists():
            raise FileNotFoundError(f"Mode directory not found: {d}")
        task_sets.append({p.name for p in d.iterdir() if p.is_dir()})

    common = set.intersection(*task_sets) if task_sets else set()
    return sorted(common)


def _pretty_metric_label(mode_filter: str, metric: str) -> str:
    mode_filter = mode_filter.lower()
    prefix = "Eval" if mode_filter == "eval" else "Train"
    return f"{prefix} {metric}"


def _plot_task(
    *,
    task: str,
    mode_filter: str,
    metrics: Sequence[str],
    mode_to_mean: Dict[str, pd.DataFrame],
    mode_to_std: Dict[str, pd.DataFrame],
    mode_to_label: Dict[str, str],
    out_path: Path,
    smooth: int,
    show_std: bool,
) -> None:
    n = len(metrics)
    if n <= 0:
        raise ValueError("metrics must be non-empty")

    fig_w = 10.0 if n > 1 else 7.5
    fig_h = 4.2
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[0]

    for ax, metric in zip(axes, metrics):
        for mode_name, mean_df in mode_to_mean.items():
            mean_df = mean_df.sort_values("total_env_steps")
            x = mean_df["total_env_steps"].to_numpy()
            y = _rolling_smooth(mean_df, metric, smooth).to_numpy()
            ax.plot(x, y, linewidth=2.0, label=mode_to_label.get(mode_name, mode_name))

            if show_std:
                std_df = mode_to_std[mode_name].sort_values("total_env_steps")
                y_std = _rolling_smooth(std_df, metric, smooth).fillna(0.0).to_numpy()
                ax.fill_between(x, y - y_std, y + y_std, alpha=0.18)

        ax.set_xlabel("Steps (total_env_steps)")
        ax.set_ylabel(_pretty_metric_label(mode_filter, metric))
        if metric.startswith("success"):
            ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(task)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints-root",
        type=str,
        default=str(Path(__file__).resolve().parent / "checkpoints" / "ppo_memtasks"),
        help="Root of PPO memtasks checkpoints (contains mode subdirs).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="normalized_dense",
        help="Subdirectory under each mode to use (e.g., normalized_dense).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(MODES_DEFAULT),
        help="Two or more modes to compare (default: rgb_joints rgb_joints_belief).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(SEEDS_DEFAULT),
        help="Seeds to average within each mode (default: 33 42).",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional explicit task names. If omitted, uses intersection across modes.",
    )
    parser.add_argument(
        "--mode-filter",
        type=str,
        default="eval",
        choices=["eval", "train"],
        help="Which rows to plot from training_metrics.csv (default: eval).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(METRICS_DEFAULT),
        help="Metric columns to plot (default: success_once success_at_end).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "plots_mode_compare"),
        help="Output directory for PNG images (one per task).",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Rolling window size for smoothing mean/std curves (1 disables).",
    )
    parser.add_argument(
        "--show-std",
        action="store_true",
        help="Show std shading across seeds for each mode.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any task/mode/seed is missing.",
    )
    parser.add_argument(
        "--algo-label",
        type=str,
        default="PPO",
        help="Prefix label shown in legend (set empty string to disable).",
    )
    args = parser.parse_args()

    checkpoints_root = Path(args.checkpoints_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modes: List[str] = list(args.modes)
    seeds: List[int] = list(args.seeds)
    metrics: List[str] = list(args.metrics)

    if args.tasks is None:
        tasks = _infer_tasks(checkpoints_root, modes=modes, variant=args.variant)
    else:
        tasks = list(args.tasks)

    if not tasks:
        raise ValueError("No tasks to plot (tasks list is empty).")
    if len(modes) < 2:
        raise ValueError("Need at least two modes to compare.")

    mode_to_label = {}
    for m in modes:
        if str(args.algo_label).strip():
            mode_to_label[m] = f"{args.algo_label.strip()} {m}"
        else:
            mode_to_label[m] = m

    print(f"[plot] checkpoints_root={checkpoints_root}")
    print(f"[plot] variant={args.variant}")
    print(f"[plot] mode_filter={args.mode_filter}")
    print(f"[plot] modes={modes}")
    print(f"[plot] seeds={seeds}")
    print(f"[plot] tasks={tasks}")
    print(f"[plot] metrics={metrics}")
    print(f"[plot] out_dir={out_dir}")

    for task in tasks:
        mode_to_mean: Dict[str, pd.DataFrame] = {}
        mode_to_std: Dict[str, pd.DataFrame] = {}

        for mode in modes:
            task_dir = checkpoints_root / mode / args.variant / task
            selected: List[RunSelection] = []
            seed_dfs: Dict[int, pd.DataFrame] = {}
            for seed in seeds:
                sel = _find_latest_training_metrics_csv(task_dir=task_dir, mode=mode, seed=seed, strict=args.strict)
                selected.append(sel)
                seed_dfs[seed] = _load_metrics(sel.csv_path, mode_filter=args.mode_filter, metrics=metrics)

            mean_df, std_df = _mean_std_across_seeds(seed_dfs, metrics=metrics)
            mode_to_mean[mode] = mean_df
            mode_to_std[mode] = std_df

            # Print selection per mode (useful for audit)
            for sel in selected:
                print(
                    f"[select] task={task} mode={mode} seed={sel.seed} "
                    f"ts={sel.timestamp_tag} csv={sel.csv_path}"
                )

        out_path = out_dir / f"{task}.png"
        _plot_task(
            task=task,
            mode_filter=args.mode_filter,
            metrics=metrics,
            mode_to_mean=mode_to_mean,
            mode_to_std=mode_to_std,
            mode_to_label=mode_to_label,
            out_path=out_path,
            smooth=int(args.smooth),
            show_std=bool(args.show_std),
        )
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()





