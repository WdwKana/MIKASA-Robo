#!/usr/bin/env python3
"""
Plot 2 runs: x-axis = steps, y-axis = eval success.

This is useful when runs are still ongoing and you want to compare
learning curves directly, but only using eval points.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
from pathlib import Path
from typing import List, Tuple

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
    return sorted(paths, key=lambda p: parse_ts(p.parent.parent.name))[-1]


def find_training_metrics_csv(run_dir: Path) -> Path:
    matches = list(run_dir.glob("**/training_metrics.csv"))
    if not matches:
        raise FileNotFoundError(f"No training_metrics.csv found under {run_dir}")
    return choose_latest(matches)


def load_eval_xy(csv_path: Path, metric_col: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    ev = df[df["mode"] == "eval"].copy().sort_values("total_env_steps")
    x = ev["total_env_steps"].to_numpy(float)
    y = ev[metric_col].to_numpy(float)
    return x, y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-name", type=str, required=True)
    ap.add_argument("--baseline-run", type=Path, required=True)
    ap.add_argument("--aib-run", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--xlabel", type=str, default="steps")
    ap.add_argument("--ylabel", type=str, default="eval success")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    baseline_csv = find_training_metrics_csv(args.baseline_run)
    aib_csv = find_training_metrics_csv(args.aib_run)

    cmdline = " ".join(shlex.quote(x) for x in [os.path.abspath(__file__), *os.sys.argv[1:]])

    curves = [
        ("ppo+believer", "#1f77b4", baseline_csv),
        ("ppo+AIB (ours)", "#d62728", aib_csv),
    ]

    metrics = [
        ("success_once", "success_once"),
        ("success_at_end", "success_at_end"),
    ]

    report_lines: List[str] = []
    report_lines.append("### 2-run plot: x=steps, y=eval success\n")
    report_lines.append(f"task_name: {args.task_name}\n")
    report_lines.append(f"out_dir: {args.out_dir}\n")
    report_lines.append(f"plot_script: {Path(__file__).resolve()}\n")
    report_lines.append(f"plot_command: {cmdline}\n")
    report_lines.append("\npaths:\n")
    report_lines.append(f"  baseline_run: {args.baseline_run}\n")
    report_lines.append(f"  baseline_csv: {baseline_csv}\n")
    report_lines.append(f"  aib_run: {args.aib_run}\n")
    report_lines.append(f"  aib_csv: {aib_csv}\n")
    report_lines.append("\n")

    for metric_key, col in metrics:
        fig, ax = plt.subplots(figsize=(6.6, 5.2))
        for label, color, csv_path in curves:
            x, y = load_eval_xy(csv_path, col)
            ax.plot(x, y, color=color, linewidth=2.5, label=label)
        ax.set_xlabel(args.xlabel)
        ax.set_ylabel(args.ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()

        out_path = args.out_dir / f"{args.task_name}__steps_vs_evalsuccess__{metric_key}.png"
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)

        report_lines.append(f"out_{metric_key}: {out_path}\n")

    (args.out_dir / f"{args.task_name}__steps_vs_evalsuccess__report.txt").write_text("".join(report_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

