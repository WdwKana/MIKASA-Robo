#!/usr/bin/env python3
"""
Plot 2 CSVs: x-axis = steps (total_env_steps), y-axis = eval success.

Use-case: Compare two runs/variants given explicit `training_metrics.csv` paths.

Output:
  - Two plots (success_once, success_at_end)
  - A report including exact CSV paths and last-eval values
"""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_eval_xy(csv_path: Path, metric_col: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    ev = df[df["mode"] == "eval"].copy().sort_values("total_env_steps")
    x = ev["total_env_steps"].to_numpy(float)
    y = ev[metric_col].to_numpy(float)
    return x, y


def last_eval_vals(csv_path: Path) -> Tuple[float, float, float]:
    df = pd.read_csv(csv_path)
    ev = df[df["mode"] == "eval"].copy().sort_values("total_env_steps")
    if ev.empty:
        return float("nan"), float("nan"), float("nan")
    row = ev.iloc[-1]
    return float(row["total_env_steps"]), float(row["success_once"]), float(row["success_at_end"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-name", type=str, required=True)
    ap.add_argument("--csv-a", type=Path, required=True)
    ap.add_argument("--csv-b", type=Path, required=True)
    ap.add_argument("--label-a", type=str, required=True)
    ap.add_argument("--label-b", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--xlabel", type=str, default="steps")
    ap.add_argument("--ylabel", type=str, default="eval success")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--no-title", action="store_true", default=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cmdline = " ".join(shlex.quote(x) for x in [os.path.abspath(__file__), *os.sys.argv[1:]])

    curves = [
        (args.label_a, "#1f77b4", args.csv_a),
        (args.label_b, "#d62728", args.csv_b),
    ]

    metrics: List[Tuple[str, str]] = [
        ("success_once", "success_once"),
        ("success_at_end", "success_at_end"),
    ]

    report_lines: List[str] = []
    report_lines.append("### 2-csv plot: x=steps, y=eval success\n")
    report_lines.append(f"task_name: {args.task_name}\n")
    report_lines.append(f"out_dir: {args.out_dir}\n")
    report_lines.append(f"plot_script: {Path(__file__).resolve()}\n")
    report_lines.append(f"plot_command: {cmdline}\n\n")
    report_lines.append("paths:\n")
    report_lines.append(f"  csv_a: {args.csv_a}\n")
    report_lines.append(f"  label_a: {args.label_a}\n")
    la = last_eval_vals(args.csv_a)
    report_lines.append(f"  last_eval_a: steps={la[0]}, success_once={la[1]}, success_at_end={la[2]}\n")
    report_lines.append(f"  csv_b: {args.csv_b}\n")
    report_lines.append(f"  label_b: {args.label_b}\n")
    lb = last_eval_vals(args.csv_b)
    report_lines.append(f"  last_eval_b: steps={lb[0]}, success_once={lb[1]}, success_at_end={lb[2]}\n\n")

    for metric_key, col in metrics:
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        for label, color, csv_path in curves:
            x, y = load_eval_xy(csv_path, col)
            ax.plot(x, y, color=color, linewidth=2.5, label=label)
        ax.set_xlabel(args.xlabel)
        ax.set_ylabel(args.ylabel)
        if not args.no_title:
            ax.set_title(args.task_name)
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

