#!/usr/bin/env python3
"""
RememberShape5-v0: compare baseline-newhyper vs multiple AIB(v6) variants.

Goal (per "group"):
  - Find baseline run-groups under a task directory (e.g. "" and "norm200")
  - For each group, plot 4 curves:
      baseline + v6-1 + v6-01 + v6-001
    aggregated across common seeds (default: 33/42/99) using mean + min/max shading.

Outputs success_once/success_at_end (eval) by default; optional train reward.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def plot_mean_minmax(
    *,
    ax: plt.Axes,
    label: str,
    color: str,
    seed_curves: List[Tuple[np.ndarray, np.ndarray]],
):
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


@dataclass(frozen=True)
class CurveSpec:
    key: str
    label: str
    color: str
    glob_fmt: str  # relative to task dir; supports {seed} and {group_part}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    mode: str
    col: str
    ylabel: str
    clamp_01: bool


BASELINE_GROUP_RE = re.compile(r"^baseline-last5-newhyper1e-4(?P<g>.*?)-(?P<seed>\d+)__")


def discover_groups(task_dir: Path) -> List[str]:
    """
    Discover group keys based on baseline run names:
      baseline-last5-newhyper1e-4-33__...              -> group ""
      baseline-last5-newhyper1e-4-norm200-33__...      -> group "norm200"
    """
    groups: set[str] = set()
    for p in task_dir.glob("**/baseline-last5-newhyper1e-4-*/*/training_metrics.csv"):
        run_name = p.parent.parent.name
        m = BASELINE_GROUP_RE.match(run_name)
        if not m:
            continue
        g = (m.group("g") or "").lstrip("-").strip()
        groups.add(g)
    return sorted(groups)


def group_part(group: str) -> str:
    return f"-{group}" if group else ""


def common_seeds(dicts: Iterable[Dict[str, Path]]) -> List[str]:
    seeds: Optional[set[str]] = None
    for d in dicts:
        s = set(d.keys())
        seeds = s if seeds is None else (seeds & s)
    return sorted(seeds or [], key=int)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task-dir",
        type=Path,
        default=Path(
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberShape5-v0"
        ),
    )
    ap.add_argument("--task-name", type=str, default="RememberShape5-v0")
    ap.add_argument("--seeds", type=str, default="33,42,99", help="Comma-separated seeds.")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "/local/s4176650/MIKASA-Robo/plots_single_runs/RememberShape5-v0/newhyper_compare_v6_1_01_001_groups"
        ),
    )
    ap.add_argument("--xlabel", type=str, default="steps")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--plot-train-reward", action="store_true")
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="If a group is missing some curves, still plot the subset of curves available (using common seeds across that subset).",
    )
    ap.add_argument("--tight-y", action="store_true", default=True)
    ap.add_argument("--y-pad", type=float, default=0.03)
    ap.add_argument("--y-min-span", type=float, default=0.12)
    ap.add_argument("--no-title", action="store_true", default=True)
    args = ap.parse_args()

    requested_seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    if not requested_seeds:
        raise SystemExit("Empty --seeds")

    args.out_root.mkdir(parents=True, exist_ok=True)

    cmdline = " ".join(shlex.quote(x) for x in [os.path.abspath(__file__), *os.sys.argv[1:]])

    curves: List[CurveSpec] = [
        CurveSpec(
            "baseline",
            "ppo+believer",
            "#1f77b4",
            "**/baseline-last5-newhyper1e-4{group_part}-{seed}__{seed}__*/**/training_metrics.csv",
        ),
        CurveSpec(
            "v6-1",
            "ppo+AIB(v6-1)",
            "#d62728",
            "**/ppo-cvae-last5-v6-1-newhyper1e-4{group_part}-{seed}__{seed}__*/**/training_metrics.csv",
        ),
        CurveSpec(
            "v6-01",
            "ppo+AIB(v6-01)",
            "#2ca02c",
            "**/ppo-cvae-last5-v6-01-newhyper1e-4{group_part}-{seed}__{seed}__*/**/training_metrics.csv",
        ),
        CurveSpec(
            "v6-001",
            "ppo+AIB(v6-001)",
            "#ff7f0e",
            "**/ppo-cvae-last5-v6-001-newhyper1e-4{group_part}-{seed}__{seed}__*/**/training_metrics.csv",
        ),
    ]

    metrics: List[MetricSpec] = [
        MetricSpec("success_once", "eval", "success_once", "success", True),
        MetricSpec("success_at_end", "eval", "success_at_end", "success", True),
    ]
    if args.plot_train_reward:
        metrics.append(MetricSpec("train_reward", "train", "reward", "reward", False))

    groups = discover_groups(args.task_dir)
    if not groups:
        raise SystemExit(f"No baseline groups discovered under {args.task_dir}")

    cache_eval: Dict[Path, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    cache_mode: Dict[Tuple[Path, str, str], Tuple[np.ndarray, np.ndarray]] = {}

    def load_cached_eval(p: Path):
        if p not in cache_eval:
            cache_eval[p] = load_eval(p)
        return cache_eval[p]

    def load_cached_mode(p: Path, mode: str, col: str):
        k = (p, mode, col)
        if k not in cache_mode:
            cache_mode[k] = load_mode_col(p, mode=mode, col=col)
        return cache_mode[k]

    for g in groups:
        gpart = group_part(g)
        out_dir = args.out_root / (g if g else "default")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Collect csvs per curve per seed (choose latest if multiple).
        curve_csvs: Dict[str, Dict[str, Path]] = {c.key: {} for c in curves}
        for seed in requested_seeds:
            for c in curves:
                pat = c.glob_fmt.format(seed=seed, group_part=gpart)
                matches = list(args.task_dir.glob(pat))
                if matches:
                    curve_csvs[c.key][seed] = choose_latest(matches)

        used_seeds = common_seeds(curve_csvs.values())
        curves_to_plot = curves
        partial_note: Optional[str] = None
        if not used_seeds and args.allow_partial:
            present = [c for c in curves if curve_csvs[c.key]]
            used_seeds = common_seeds(curve_csvs[c.key] for c in present)
            if present and used_seeds:
                curves_to_plot = present
                missing = [c.key for c in curves if c not in present]
                partial_note = f"PARTIAL: missing_curves={missing}"
        if not used_seeds:
            # Nothing to plot for this group.
            report_path = out_dir / "report.txt"
            report_path.write_text(
                "\n".join(
                    [
                        "No common seeds across all curves for this group.",
                        f"group: {g!r}",
                        f"task_dir: {args.task_dir}",
                        f"requested_seeds: {requested_seeds}",
                        f"plot_command: {cmdline}",
                        "",
                        "curve_seed_counts:",
                        *[f"  {c.key}: {sorted(curve_csvs[c.key].keys(), key=int)}" for c in curves],
                    ]
                )
                + "\n"
            )
            continue

        # y-limits derived from plotted data (per metric) across used seeds/curves.
        for ms in metrics:
            if args.tight_y:
                all_y: List[np.ndarray] = []
                for c in curves_to_plot:
                    for s in used_seeds:
                        p = curve_csvs[c.key][s]
                        if ms.mode == "eval":
                            _, y_once, y_end = load_cached_eval(p)
                            all_y.append(y_once if ms.col == "success_once" else y_end)
                        else:
                            _, y = load_cached_mode(p, ms.mode, ms.col)
                            all_y.append(y)
                yy = np.concatenate(all_y) if all_y else np.array([0.0, 1.0], dtype=float)
                yy = np.asarray(yy, dtype=float).reshape(-1)
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
            for c in curves_to_plot:
                seed_curves: List[Tuple[np.ndarray, np.ndarray]] = []
                for s in used_seeds:
                    p = curve_csvs[c.key][s]
                    if ms.mode == "eval":
                        x, y_once, y_end = load_cached_eval(p)
                        y = y_once if ms.col == "success_once" else y_end
                    else:
                        x, y = load_cached_mode(p, ms.mode, ms.col)
                    seed_curves.append((x, y))
                plot_mean_minmax(ax=ax, label=c.label, color=c.color, seed_curves=seed_curves)

            if y0 is not None and y1 is not None:
                ax.set_ylim(y0, y1)
            ax.set_xlabel(args.xlabel)
            ax.set_ylabel(ms.ylabel)
            if not args.no_title:
                ax.set_title(args.task_name)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
            fig.tight_layout(rect=(0, 0, 0.82, 1))

            seed_tag = "_".join(used_seeds)
            group_tag = g if g else "default"
            curve_tag = f"{len(curves_to_plot)}curves" if len(curves_to_plot) != 4 else "4curves"
            out_path = out_dir / f"{args.task_name}__group_{group_tag}__{ms.key}__{curve_tag}__seeds_{seed_tag}.png"
            fig.savefig(out_path, dpi=args.dpi)
            plt.close(fig)

        # Report
        report_lines: List[str] = []
        report_lines.append("### RememberShape5 newhyper: 4-curve comparison\n")
        report_lines.append(f"task_dir: {args.task_dir}\n")
        report_lines.append(f"task_name: {args.task_name}\n")
        report_lines.append(f"group: {g!r}\n")
        report_lines.append(f"requested_seeds: {requested_seeds}\n")
        report_lines.append(f"used_common_seeds: {used_seeds}\n")
        if partial_note:
            report_lines.append(f"{partial_note}\n")
        report_lines.append(f"out_dir: {out_dir}\n")
        report_lines.append(f"plot_script: {Path(__file__).resolve()}\n")
        report_lines.append(f"plot_command: {cmdline}\n")
        report_lines.append("curves:\n")
        for c in curves_to_plot:
            report_lines.append(f"  - {c.key}: {c.label}\n")
        if len(curves_to_plot) != 4:
            missing = [c.key for c in curves if c not in curves_to_plot]
            report_lines.append(f"missing_curves: {missing}\n")
        report_lines.append("\npaths:\n")
        for c in curves_to_plot:
            for s in used_seeds:
                report_lines.append(f"  {c.key} seed{s}: {curve_csvs[c.key][s]}\n")
        (out_dir / "report.txt").write_text("".join(report_lines))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

