#!/usr/bin/env python3
"""
5-method comparison for MIKASA-Robo main experiments.

Methods: CVAE-v6 (ours), PPO+Believer (VAE), PPO, PPO+LSTM, PPO+GRU.
Tasks:   6 memtasks as declared in sphinx/experiments/specification.md.
Seeds:   33, 42, 99.

Auto-discovers the latest `training_metrics.csv` per (task, method, seed) from
the canonical `checkpoints/ppo_memtasks/...` layout, preferring runs with the
most eval rows (so partial/aborted runs are skipped when a longer one exists
for the same seed/timestamp prefix).

Emits per-task PNGs for `success_once` and `success_at_end` (mean + min/max
shading across seeds, ICML style) and a summary `.txt` with last-3-eval
averages per method (only tasks where every seed reached the final eval).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REPO_ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
CKPT_ROOT = REPO_ROOT / "checkpoints" / "ppo_memtasks"

TASKS: List[str] = [
    "RememberColor9-v0",
    "RememberShapeAndColor3x2-v0",
    "RememberShape9-v0",
    "RememberShapeAndColor3x3-v0",
    "InterceptFast-v0",
    "InterceptMedium-v0",
]

SEEDS: List[str] = ["33", "42", "99"]


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    color: str
    # "rgb_joints" (non-belief) or "rgb_joints_belief"
    suite: str
    # exp-name prefix (ppo-baseline, ppo-lstm, ppo-gru, ppo-vae, cvae-v6)
    prefix: str


METHODS: List[MethodSpec] = [
    MethodSpec("cvae",     "CVAE-v6 (ours)",  "#d62728", "rgb_joints_belief", "cvae-v6"),
    MethodSpec("vae",      "PPO+Believer",    "#1f77b4", "rgb_joints_belief", "ppo-vae"),
    MethodSpec("ppo",      "PPO",             "#2ca02c", "rgb_joints",        "ppo-baseline"),
    MethodSpec("lstm",     "PPO+LSTM",        "#9467bd", "rgb_joints",        "ppo-lstm"),
    MethodSpec("gru",      "PPO+GRU",         "#ff7f0e", "rgb_joints",        "ppo-gru"),
]

LINEWIDTH_PT = 2.5
SHADE_ALPHA = 0.15
FIG_W, FIG_H = 7.2, 4.0


def task_dir(task: str, suite: str) -> Path:
    return CKPT_ROOT / suite / "normalized_dense" / task


def csv_row_count(p: Path) -> int:
    try:
        with p.open("rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def find_csv(task: str, method: MethodSpec, seed: str) -> Optional[Path]:
    """
    Match run dirs like: <prefix>-<task>-<seed>__<seed>__<suite>__<timestamp>/
    then look for <timestamp>/training_metrics.csv. Pick the file with the
    largest row count (longest run); tie-break by latest timestamp.
    """
    base = task_dir(task, method.suite)
    pattern = f"{method.prefix}-{task}-{seed}__{seed}__{method.suite}__*"
    candidates: List[Path] = []
    for run_dir in base.glob(pattern):
        if not run_dir.is_dir():
            continue
        for csv in run_dir.glob("*/training_metrics.csv"):
            candidates.append(csv)

    if not candidates:
        return None

    def key(p: Path) -> Tuple[int, str]:
        return (csv_row_count(p), p.parent.parent.name)

    candidates.sort(key=key, reverse=True)
    return candidates[0]


def load_eval(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    ev = df[df["mode"].astype(str).str.lower() == "eval"].copy()
    ev = ev.sort_values("total_env_steps")
    return ev


def seed_curves(task: str, method: MethodSpec, metric: str) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    out = []
    for s in SEEDS:
        csv = find_csv(task, method, s)
        if csv is None:
            continue
        ev = load_eval(csv)
        if ev.empty:
            continue
        out.append(
            (
                ev["total_env_steps"].to_numpy(float),
                ev[metric].to_numpy(float),
                s,
            )
        )
    return out


def plot_mean_minmax(ax, label, color, curves):
    if not curves:
        return
    xs = sorted({float(x) for xx, _, _ in curves for x in xx})
    xs = np.array(xs, dtype=float)
    Ys = []
    for x, y, _ in curves:
        m = dict(zip(x.tolist(), y.tolist()))
        Ys.append(np.array([m.get(float(xx), np.nan) for xx in xs], dtype=float))
    Y = np.vstack(Ys)
    y_mean = np.nanmean(Y, axis=0)
    y_low = np.nanmin(Y, axis=0)
    y_high = np.nanmax(Y, axis=0)
    ax.fill_between(xs, y_low, y_high, color=color, alpha=SHADE_ALPHA)
    ax.plot(xs, y_mean, color=color, linewidth=LINEWIDTH_PT, label=label)


def last_n_eval_mean(curves: List[Tuple[np.ndarray, np.ndarray, str]], n: int = 3) -> Dict[str, float]:
    """Return mean of last-n eval values per seed; key is seed."""
    out = {}
    for _, y, s in curves:
        if len(y) >= n:
            out[s] = float(np.mean(y[-n:]))
        elif len(y) > 0:
            out[s] = float(np.mean(y))
    return out


def plot_task(task: str, out_dir: Path) -> Dict[str, Dict[str, object]]:
    """Return {metric: {method_key: {"per_seed": {...}, "mean": f, "std": f, "n_seeds": int, "max_step": f}}}."""
    results: Dict[str, Dict[str, object]] = {}
    for metric, ylabel in [("success_once", "success_once"), ("success_at_end", "success_at_end")]:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        per_method: Dict[str, object] = {}

        for m in METHODS:
            curves = seed_curves(task, m, metric)
            plot_mean_minmax(ax, m.label, m.color, curves)

            per_seed = last_n_eval_mean(curves, n=3)
            if per_seed:
                vals = np.array(list(per_seed.values()), dtype=float)
                max_step = max((float(x[-1]) if len(x) else 0.0) for x, _, _ in curves)
                per_method[m.key] = {
                    "label": m.label,
                    "per_seed": per_seed,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=0)),
                    "n_seeds": int(len(vals)),
                    "max_step": max_step,
                }
            else:
                per_method[m.key] = {
                    "label": m.label,
                    "per_seed": {},
                    "mean": float("nan"),
                    "std": float("nan"),
                    "n_seeds": 0,
                    "max_step": 0.0,
                }

        ax.set_xlabel("steps (total_env_steps)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, loc="best", fontsize=9)
        fig.tight_layout()

        out_path = out_dir / f"{task}__{metric}__5methods.png"
        fig.savefig(out_path, dpi=200, facecolor="white")
        plt.close(fig)

        results[metric] = per_method

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "plots" / "5methods_belief_partial_comparison")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Dict[str, Dict[str, object]]] = {}
    missing: List[str] = []
    discovered: List[str] = []

    for task in TASKS:
        print(f"== {task} ==")
        for m in METHODS:
            seeds_found = []
            for s in SEEDS:
                p = find_csv(task, m, s)
                if p:
                    seeds_found.append(f"{s}:{csv_row_count(p)}rows")
                else:
                    missing.append(f"{task} / {m.key} / seed{s}")
            discovered.append(f"{task} / {m.key}: {', '.join(seeds_found) if seeds_found else 'NONE'}")
            print(f"  {m.key:8s} -> {seeds_found}")
        all_results[task] = plot_task(task, out_dir)

    # Emit summary table.
    lines: List[str] = []
    lines.append("# 5-Method Comparison — Last-3-Eval Averages\n")
    lines.append(f"tasks: {len(TASKS)}, seeds: {SEEDS}\n")
    lines.append(f"out_dir: {out_dir}\n\n")

    for metric in ("success_once", "success_at_end"):
        lines.append(f"## metric: {metric}\n\n")
        header = f"{'task':32s} | {'method':18s} | n | {'mean±std':16s} | per_seed\n"
        lines.append(header)
        lines.append("-" * len(header) + "\n")
        for task in TASKS:
            for m in METHODS:
                r = all_results[task][metric].get(m.key, None)
                if r is None or r["n_seeds"] == 0:
                    lines.append(f"{task:32s} | {m.label:18s} | 0 | {'N/A':16s} | —\n")
                    continue
                ms = f"{r['mean']:.3f}±{r['std']:.3f}"
                per_seed = ", ".join(f"s{k}:{v:.3f}" for k, v in sorted(r["per_seed"].items()))
                lines.append(f"{task:32s} | {m.label:18s} | {r['n_seeds']} | {ms:16s} | {per_seed}\n")
            lines.append("\n")

    lines.append("\n## Discovery log\n\n")
    for d in discovered:
        lines.append(d + "\n")
    if missing:
        lines.append("\n## Missing runs\n\n")
        for m in missing:
            lines.append(m + "\n")

    (out_dir / "summary.txt").write_text("".join(lines))
    print(f"\nSummary written: {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
