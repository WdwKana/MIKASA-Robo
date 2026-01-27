#!/usr/bin/env python3
"""
Plot comparison for memtasks (last5), aggregating across a fixed seed set using:
  - mean curve + min/max shading
  - `training_metrics.csv`, filtered by `mode`, x = `total_env_steps`

Originally used for 2 curves (baseline vs AIB v6). Extended to support an
optional 3rd curve (e.g., nonbelief/ppo) while keeping the same plotting style
and reproducibility reporting.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TS_RE = re.compile(r"__(\d{8}_\d{6})$")


def parse_ts(run_name: str) -> str:
    m = TS_RE.search(run_name)
    return m.group(1) if m else ""


def choose_latest(paths: List[Path]) -> Path:
    if len(paths) == 1:
        return paths[0]
    # .../<run_name>/<timestamp>/training_metrics.csv
    return sorted(paths, key=lambda p: parse_ts(p.parent.parent.name))[-1]


def load_eval(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    ev = df[df["mode"] == "eval"].copy().sort_values("total_env_steps")
    x = ev["total_env_steps"].to_numpy(float)
    return x, ev["success_once"].to_numpy(float), ev["success_at_end"].to_numpy(float)

def load_mode_col(csv_path: Path, *, mode: str, col: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    mdf = df[df["mode"] == mode].copy().sort_values("total_env_steps")
    x = mdf["total_env_steps"].to_numpy(float)
    y = mdf[col].to_numpy(float)
    return x, y


@dataclass(frozen=True)
class CurveSpec:
    key: str
    label: str
    color: str
    enabled: bool = True


@dataclass(frozen=True)
class MetricSpec:
    key: str
    mode: str
    col: str
    ylabel: str
    clamp_01: bool


DEFAULT_CURVES: List[CurveSpec] = [
    CurveSpec("baseline", "ppo+believer", "#1f77b4", True),
    CurveSpec("aib", "ppo+AIB (ours)", "#d62728", True),
    CurveSpec("nonbelief", "ppo", "#000000", False),
]


def describe_csv(p: Path) -> str:
    """
    Layout is usually: .../<run_name>/<timestamp>/training_metrics.csv
      - run_dir: the run_name folder (contains cmd/config/log)
      - ts_dir:  the timestamp folder
    """
    ts_dir = p.parent
    run_dir = p.parent.parent
    return f"csv: {p}\n    run_dir: {run_dir}\n    ts_dir: {ts_dir}\n"


def plot_mean_minmax(
    *,
    ax: plt.Axes,
    label: str,
    color: str,
    seed_curves: List[Tuple[np.ndarray, np.ndarray]],
):
    """
    Match `plot_last5_seedsets.py` behavior:
      - x grid is the union of all x points (no interpolation)
      - per-seed values missing at an x are treated as NaN
      - mean is nanmean; shading uses nanmin/nanmax
    """
    if not seed_curves:
        return
    xs = sorted(set(np.concatenate([x for x, _ in seed_curves]).tolist()))
    xs = np.array(xs, dtype=float)
    Ys: List[np.ndarray] = []
    for x, y in seed_curves:
        m = dict(zip(x.tolist(), y.tolist()))
        Ys.append(np.array([m.get(float(xx), np.nan) for xx in xs], dtype=float))
    Y = np.vstack(Ys)
    y_mean = np.nanmean(Y, axis=0)
    y_low = np.nanmin(Y, axis=0)
    y_high = np.nanmax(Y, axis=0)
    ax.fill_between(xs, y_low, y_high, color=color, alpha=0.15)
    ax.plot(xs, y_mean, color=color, linewidth=3.0, label=label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline-task-dir",
        type=Path,
        default=Path(
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberShapeAndColor3x2-v0"
        ),
        help="Task directory containing baseline (belief) run folders.",
    )
    ap.add_argument(
        "--aib-task-dir",
        type=Path,
        default=None,
        help="Task directory containing AIB run folders (defaults to --baseline-task-dir).",
    )
    ap.add_argument(
        "--nonbelief-task-dir",
        type=Path,
        default=None,
        help="Task directory containing nonbelief run folders (optional third curve).",
    )
    ap.add_argument("--task-name", type=str, default="RememberShapeAndColor3x2-v0")
    ap.add_argument("--seeds", type=str, default="33,42,99", help="Comma-separated seeds.")
    ap.add_argument(
        "--baseline-glob",
        type=str,
        default="**/ppo-last5-baseline-newhyper-lr1e-4-{seed}__{seed}__*/**/training_metrics.csv",
        help="Glob pattern (relative to --task-dir) for baseline training_metrics.csv; supports {seed}.",
    )
    ap.add_argument(
        "--aib-glob",
        type=str,
        default="**/ppo-last5-cvae-v6-ent0001-kl005-gamma095-lr1e4-{seed}__{seed}__*/**/training_metrics.csv",
        help="Glob pattern (relative to --task-dir) for AIB training_metrics.csv; supports {seed}.",
    )
    ap.add_argument(
        "--nonbelief-glob",
        type=str,
        default="**/ppo-nonbelief-baseline-newhyper-{seed}__{seed}__*/**/training_metrics.csv",
        help="Glob pattern (relative to --nonbelief-task-dir) for nonbelief training_metrics.csv; supports {seed}.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/local/s4176650/MIKASA-Robo/plots_single_runs/RememberShapeAndColor3x2-v0/baseline_newhyper_vs_aib_v6_seeds33_42_99"
        ),
    )
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--xlabel", type=str, default="steps")
    ap.add_argument("--no-title", action="store_true", default=True)
    ap.add_argument("--tight-y", action="store_true", default=True)
    ap.add_argument("--y-pad", type=float, default=0.03)
    ap.add_argument("--y-min-span", type=float, default=0.12)
    ap.add_argument("--plot-train-reward", action="store_true", help="Also plot train-mode reward vs steps.")
    args = ap.parse_args()

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("Empty --seeds")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility: exact invocation.
    cmdline = " ".join(shlex.quote(x) for x in [os.path.abspath(__file__), *os.sys.argv[1:]])

    baseline_task_dir = args.baseline_task_dir
    aib_task_dir = args.aib_task_dir or baseline_task_dir
    nonbelief_task_dir = args.nonbelief_task_dir

    # Discover CSVs (one per seed per curve; choose latest if multiple).
    baseline_csvs: Dict[str, Path] = {}
    aib_csvs: Dict[str, Path] = {}
    nonbelief_csvs: Dict[str, Path] = {}

    for seed in seeds:
        base_pat = args.baseline_glob.format(seed=seed)
        base_matches = list(baseline_task_dir.glob(base_pat))
        if base_matches:
            baseline_csvs[seed] = choose_latest(base_matches)

        aib_pat = args.aib_glob.format(seed=seed)
        aib_matches = list(aib_task_dir.glob(aib_pat))
        if aib_matches:
            aib_csvs[seed] = choose_latest(aib_matches)

        if nonbelief_task_dir is not None:
            nb_pat = args.nonbelief_glob.format(seed=seed)
            nb_matches = list(nonbelief_task_dir.glob(nb_pat))
            if nb_matches:
                nonbelief_csvs[seed] = choose_latest(nb_matches)

    base_seeds = sorted(baseline_csvs.keys(), key=int)
    aib_seeds = sorted(aib_csvs.keys(), key=int)
    nb_seeds = sorted(nonbelief_csvs.keys(), key=int)

    if not base_seeds or not aib_seeds:
        raise SystemExit(
            f"Missing curves. baseline_seeds={base_seeds}, aib_seeds={aib_seeds}, nonbelief_seeds={nb_seeds}"
        )

    cache: Dict[Path, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    cache_mode_col: Dict[Tuple[Path, str, str], Tuple[np.ndarray, np.ndarray]] = {}

    def load_cached(p: Path):
        if p not in cache:
            cache[p] = load_eval(p)
        return cache[p]

    def load_cached_mode_col(p: Path, mode: str, col: str):
        k = (p, mode, col)
        if k not in cache_mode_col:
            cache_mode_col[k] = load_mode_col(p, mode=mode, col=col)
        return cache_mode_col[k]

    metrics: List[MetricSpec] = [
        MetricSpec("success_once", "eval", "success_once", "success", True),
        MetricSpec("success_at_end", "eval", "success_at_end", "success", True),
    ]
    if args.plot_train_reward:
        metrics.append(MetricSpec("train_reward", "train", "reward", "reward", False))

    report: List[str] = []
    report.append("### baseline newhyper vs AIB v6 (+ optional nonbelief) (last5)\n")
    report.append(f"baseline_task_dir: {baseline_task_dir}\n")
    report.append(f"aib_task_dir: {aib_task_dir}\n")
    report.append(f"nonbelief_task_dir: {nonbelief_task_dir}\n")
    report.append(f"task_name: {args.task_name}\n")
    report.append(f"requested_seeds: {seeds}\n")
    report.append(f"baseline_glob: {args.baseline_glob}\n")
    report.append(f"aib_glob: {args.aib_glob}\n")
    report.append(f"nonbelief_glob: {args.nonbelief_glob}\n")
    report.append(f"baseline seeds used: {base_seeds}\n")
    report.append(f"AIB seeds used: {aib_seeds}\n")
    report.append(f"nonbelief seeds used: {nb_seeds}\n")
    report.append(f"out_dir: {args.out_dir}\n")
    report.append(f"plot_script: {Path(__file__).resolve()}\n")
    report.append(f"plot_command: {cmdline}\n")
    report.append("curve_labels:\n")
    curves: List[CurveSpec] = [
        CurveSpec("baseline", "ppo+believer", "#1f77b4", True),
        CurveSpec("aib", "ppo+AIB (ours)", "#d62728", True),
        CurveSpec("nonbelief", "ppo", "#000000", nonbelief_task_dir is not None),
    ]
    for c in curves:
        if c.enabled:
            report.append(f"  - {c.key}: {c.label}\n")
    report.append("\npaths:\n")
    for s in base_seeds:
        report.append(f"  baseline seed{s}:\n    {describe_csv(baseline_csvs[s])}")
    for s in aib_seeds:
        report.append(f"  AIB seed{s}:\n    {describe_csv(aib_csvs[s])}")
    for s in nb_seeds:
        report.append(f"  nonbelief seed{s}:\n    {describe_csv(nonbelief_csvs[s])}")
    report.append("\n")

    for ms in metrics:
        # Tight y-limits from plotted data
        if args.tight_y:
            all_y: List[np.ndarray] = []
            for s in base_seeds:
                if ms.mode == "eval":
                    _, y1, y2 = load_cached(baseline_csvs[s])
                    all_y.append(y1 if ms.col == "success_once" else y2)
                else:
                    _, y = load_cached_mode_col(baseline_csvs[s], ms.mode, ms.col)
                    all_y.append(y)
            for s in aib_seeds:
                if ms.mode == "eval":
                    _, y1, y2 = load_cached(aib_csvs[s])
                    all_y.append(y1 if ms.col == "success_once" else y2)
                else:
                    _, y = load_cached_mode_col(aib_csvs[s], ms.mode, ms.col)
                    all_y.append(y)
            for s in nb_seeds:
                if ms.mode == "eval":
                    _, y1, y2 = load_cached(nonbelief_csvs[s])
                    all_y.append(y1 if ms.col == "success_once" else y2)
                else:
                    _, y = load_cached_mode_col(nonbelief_csvs[s], ms.mode, ms.col)
                    all_y.append(y)
            yy = np.concatenate(all_y) if all_y else np.array([0.0, 1.0])
            yy = yy[~np.isnan(yy)]
            y_min = float(np.min(yy))
            y_max = float(np.max(yy))
            if (y_max - y_min) < args.y_min_span:
                mid = 0.5 * (y_min + y_max)
                y_min = mid - 0.5 * args.y_min_span
                y_max = mid + 0.5 * args.y_min_span
            if ms.clamp_01:
                y0 = max(0.0, y_min - args.y_pad)
                y1 = min(1.0, y_max + args.y_pad)
            else:
                y0 = y_min - args.y_pad
                y1 = y_max + args.y_pad
        else:
            y0, y1 = (0.0, 1.0) if ms.clamp_01 else (None, None)

        fig, ax = plt.subplots(figsize=(10.5, 4.2))

        base_seed_curves = []
        for s in base_seeds:
            if ms.mode == "eval":
                x, y_once, y_end = load_cached(baseline_csvs[s])
                y = y_once if ms.col == "success_once" else y_end
            else:
                x, y = load_cached_mode_col(baseline_csvs[s], ms.mode, ms.col)
            base_seed_curves.append((x, y))
        plot_mean_minmax(ax=ax, label=curves[0].label, color=curves[0].color, seed_curves=base_seed_curves)

        aib_seed_curves = []
        for s in aib_seeds:
            if ms.mode == "eval":
                x, y_once, y_end = load_cached(aib_csvs[s])
                y = y_once if ms.col == "success_once" else y_end
            else:
                x, y = load_cached_mode_col(aib_csvs[s], ms.mode, ms.col)
            aib_seed_curves.append((x, y))
        plot_mean_minmax(ax=ax, label=curves[1].label, color=curves[1].color, seed_curves=aib_seed_curves)

        if nonbelief_task_dir is not None and nb_seeds:
            nb_seed_curves = []
            for s in nb_seeds:
                if ms.mode == "eval":
                    x, y_once, y_end = load_cached(nonbelief_csvs[s])
                    y = y_once if ms.col == "success_once" else y_end
                else:
                    x, y = load_cached_mode_col(nonbelief_csvs[s], ms.mode, ms.col)
                nb_seed_curves.append((x, y))
            plot_mean_minmax(ax=ax, label=curves[2].label, color=curves[2].color, seed_curves=nb_seed_curves)

        if y0 is not None and y1 is not None:
            ax.set_ylim(y0, y1)
        ax.set_xlabel(args.xlabel)
        ax.set_ylabel(ms.ylabel)
        if not args.no_title:
            ax.set_title(args.task_name)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")
        fig.tight_layout(rect=(0, 0, 0.82, 1))

        out_path = args.out_dir / f"{args.task_name}__baseline_newhyper_vs_aib_v6__{ms.key}__seeds_{'_'.join(seeds)}.png"
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)

        report.append(f"out_{ms.key}: {out_path}\n")

    (args.out_dir / f"{args.task_name}__baseline_newhyper_vs_aib_v6__report.txt").write_text("".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

