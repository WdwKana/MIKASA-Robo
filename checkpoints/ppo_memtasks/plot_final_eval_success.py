#!/usr/bin/env python3
"""
Plot eval success curves for the 4 memory tasks, comparing CVAE (ours) vs baseline.

Scope: ONLY reads data under:
  /local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/

It searches for final runs (folders containing "final") for:
  - RememberColor5-v0
  - RememberColor9-v0
  - RememberShapeAndColor3x2-v0
  - RememberShapeAndColor3x3-v0

For each task, it produces 2 plots (two curves: cvae vs baseline), averaging over seeds 33 and 42:
  - x-axis: total_env_steps (eval)
  - y-axis: success_once
  - y-axis: success_at_end
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


TASK_SUFFIXES = ("5-v0", "9-v0", "3x2-v0", "3x3-v0")
TARGET_SEEDS = (33, 42)

# Curves to draw on each plot:
# - ppo: runs under rgb_joints/ (no belief)
# - ppo+belief: baseline runs under rgb_joints_belief/ (final)
# - ppo+belief_with_action (ours): cvae runs under rgb_joints_belief/ (final)
CURVES = ("ppo", "ppo+belief", "ppo+belief_with_action")
CURVE_LABELS = {
    "ppo": "ppo",
    "ppo+belief": "ppo+belief",
    "ppo+belief_with_action": "ppo+belief_with_action (ours)",
}


@dataclass(frozen=True)
class RunKey:
    task: str
    curve: str
    seed: int


def _extract_timestamp_token(path_str: str) -> Optional[str]:
    # Typical token in your paths: 20260106_165220
    tokens = re.findall(r"(20\d{6}_\d{6})", path_str)
    return tokens[-1] if tokens else None


def _timestamp_sort_key(p: Path) -> Tuple[int, float, str]:
    """
    Prefer parsing YYYYMMDD_HHMMSS from path; fallback to mtime; then path string.
    Returns a tuple suitable for max().
    """
    ps = p.as_posix()
    token = _extract_timestamp_token(ps)
    if token:
        try:
            dt = datetime.strptime(token, "%Y%m%d_%H%M%S")
            return (1, dt.timestamp(), ps)
        except ValueError:
            pass
    try:
        return (0, p.stat().st_mtime, ps)
    except OSError:
        return (0, 0.0, ps)


def _discover_final_csvs(base_dir: Path) -> Dict[RunKey, Path]:
    """
    Returns mapping (task, curve, seed) -> training_metrics.csv (chosen as latest if multiple).
    """
    base_dir = base_dir.resolve()
    found: Dict[RunKey, Path] = {}

    for csv_path in base_dir.rglob("training_metrics.csv"):
        ps = csv_path.as_posix()
        if "/.git/" in ps:
            continue

        # Task detection: only include exact suffixes requested
        task = None
        for suf in TASK_SUFFIXES:
            # tasks look like ".../RememberColor5-v0/..." etc; suffix must match full folder name
            # so check for "/<something><suffix>/" and capture that folder name.
            m = re.search(rf"/([^/]+{re.escape(suf)})/", ps)
            if m:
                task = m.group(1)
                break
        if task is None:
            continue

        # Curve classification
        curve = None
        if "/rgb_joints/normalized_dense/" in ps and "/rgb_joints_belief/" not in ps:
            # plain PPO (no belief); choose latest run per task/seed
            if "/ppo" in ps:
                curve = "ppo"
        elif "/rgb_joints_belief/normalized_dense/" in ps:
            # belief runs: only use final ones
            if "final" in ps and "baseline" in ps:
                curve = "ppo+belief"
            elif "final" in ps and "cvae" in ps:
                curve = "ppo+belief_with_action"

        if curve is None:
            continue

        # Seed detection (your folder names include "__33__" / "__42__")
        seed = None
        for s in TARGET_SEEDS:
            if f"__{s}__" in ps:
                seed = s
                break
        if seed is None:
            continue

        key = RunKey(task=task, curve=curve, seed=seed)
        if key not in found:
            found[key] = csv_path
            continue

        # If duplicates exist, keep the latest
        if _timestamp_sort_key(csv_path) > _timestamp_sort_key(found[key]):
            found[key] = csv_path

    return found


def _load_eval_series(csv_path: Path, metric_col: str) -> pd.Series:
    df = pd.read_csv(csv_path)

    if "mode" in df.columns:
        df = df[df["mode"] == "eval"]

    if "total_env_steps" not in df.columns:
        raise KeyError(f"Missing column 'total_env_steps' in {csv_path}")
    if metric_col not in df.columns:
        raise KeyError(f"Missing column '{metric_col}' in {csv_path}")

    sdf = df[["total_env_steps", metric_col]].copy()
    # Guard against duplicates; take mean per step
    sdf = sdf.groupby("total_env_steps", as_index=True)[metric_col].mean().sort_index()
    sdf.index = sdf.index.astype(int)
    return sdf


def _mean_over_seeds(
    by_seed: Dict[int, pd.Series],
    required_seeds: Iterable[int],
) -> pd.Series:
    seeds = list(required_seeds)
    cols = []
    for s in seeds:
        if s not in by_seed:
            raise KeyError(f"Missing seed {s} series")
        cols.append(by_seed[s].rename(str(s)))

    joined = pd.concat(cols, axis=1).sort_index()
    # Only keep points where BOTH seeds are present
    mean = joined.mean(axis=1, skipna=False).dropna()
    return mean


def _plot_task(
    *,
    out_dir: Path,
    task: str,
    metric_col: str,
    metric_title: str,
    algo_to_series: Dict[str, pd.Series],
    x_label: str,
    y_label: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task}__{metric_title}.png"

    plt.figure(figsize=(8.5, 5.0))
    for curve in CURVES:
        series = algo_to_series.get(curve)
        if series is None:
            continue
        plt.plot(series.index.values, series.values, linewidth=2.0, label=CURVE_LABELS.get(curve, curve))

    plt.title(f"{task} - {metric_title} (eval, mean over seeds 33 & 42)")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def _write_paths_manifest(
    *,
    out_dir: Path,
    tasks: List[str],
    found: Dict[RunKey, Path],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "data_paths_used.txt"

    lines: List[str] = []
    lines.append("Eval plots - CSV paths actually used")
    lines.append(f"Output dir: {out_dir}")
    lines.append(f"Base dir: {out_dir.parent}")
    lines.append(f"Seeds averaged: {', '.join(map(str, TARGET_SEEDS))}")
    lines.append("")

    for task in tasks:
        lines.append(f"Task: {task}")
        for curve in CURVES:
            lines.append(f"  Curve: {CURVE_LABELS.get(curve, curve)}")
            for seed in TARGET_SEEDS:
                key = RunKey(task=task, curve=curve, seed=seed)
                p = found.get(key)
                lines.append(f"    seed {seed}: {p.as_posix() if p else 'MISSING'}")
        lines.append("")

    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks",
        help="Base directory to search under (default: the required checkpoints directory).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory for plots. Default: <base-dir>/plots_final_eval_success",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (base_dir / "plots_final_eval_success")

    found = _discover_final_csvs(base_dir)

    # Build task list from what we found (but keep a stable ordering matching your suffix list)
    tasks_found = sorted({k.task for k in found.keys()})
    # If multiple tasks share suffixes, stable sort by our suffix ordering.
    def _task_rank(t: str) -> Tuple[int, str]:
        for i, suf in enumerate(TASK_SUFFIXES):
            if t.endswith(suf):
                return (i, t)
        return (999, t)

    tasks = sorted(tasks_found, key=_task_rank)

    missing: List[str] = []
    for task in tasks:
        for curve in CURVES:
            for seed in TARGET_SEEDS:
                if RunKey(task=task, curve=curve, seed=seed) not in found:
                    missing.append(f"{task} / {curve} / seed {seed}")

    if missing:
        print("WARNING: Missing some required runs:")
        for m in missing:
            print("  -", m)
        print("Will plot only tasks with all required (algo × seed) present.\n")

    plotted: List[Path] = []
    for task in tasks:
        # Ensure all required present
        ok = True
        for curve in CURVES:
            for seed in TARGET_SEEDS:
                if RunKey(task=task, curve=curve, seed=seed) not in found:
                    ok = False
        if not ok:
            continue

        for metric_col, metric_title, y_label in [
            ("success_once", "eval_success_once", "Eval Success Once"),
            ("success_at_end", "eval_success_end", "Eval Success End"),
        ]:
            curve_to_mean: Dict[str, pd.Series] = {}
            for curve in CURVES:
                by_seed: Dict[int, pd.Series] = {}
                for seed in TARGET_SEEDS:
                    csv_path = found[RunKey(task=task, curve=curve, seed=seed)]
                    by_seed[seed] = _load_eval_series(csv_path, metric_col=metric_col)
                curve_to_mean[curve] = _mean_over_seeds(by_seed, required_seeds=TARGET_SEEDS)

            out_path = _plot_task(
                out_dir=out_dir,
                task=task,
                metric_col=metric_col,
                metric_title=metric_title,
                algo_to_series=curve_to_mean,
                x_label="Steps (total_env_steps, eval)",
                y_label=y_label,
            )
            plotted.append(out_path)

    manifest_path = _write_paths_manifest(out_dir=out_dir, tasks=tasks, found=found)
    print(f"Saved {len(plotted)} plot(s) to: {out_dir}")
    for p in plotted:
        print(" -", p)
    print(f"Wrote CSV path manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

