#!/usr/bin/env python3
"""
Main-experiment plots per sphinx/experiments/specification.md §6.

Produces TWO parallel plot sets, one per belief-model configuration:
  - ep6000:  Believer + CVAE-v6 use the original 6000-epoch Stage1 models
             (exp-name prefix: cvae-v6-<env>-<seed>, ppo-vae-<env>-<seed>)
  - ep10000: Believer + CVAE-v6 use the re-trained 10000-epoch Stage1 models
             (exp-name prefix: cvae-v6-ep10000-<env>-<seed>, ppo-vae-ep10000-<env>-<seed>)

PPO / PPO+LSTM / PPO+GRU are shared across both sets (they do not depend on
Stage1 belief-model training).

Per spec:
  - All methods on the same figure for each task
  - Mean line with shaded std region across seeds
  - Title includes task name
  - X-axis: environment steps; Y-axis: success rate [0, 1]
  - Quantitative summary: last-3-eval mean, per seed and as mean±std.
"""

from __future__ import annotations

import argparse
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
    suite: str      # "rgb_joints" or "rgb_joints_belief"
    prefix: str     # exp-name prefix; may be formatted with {suffix} placeholder


# Order controls legend and per-task last-3 table.
METHODS_TEMPLATE: List[MethodSpec] = [
    MethodSpec("cvae", "CVAE-v6 (Ours)",  "#d62728", "rgb_joints_belief", "cvae-v6{suffix}"),
    MethodSpec("vae",  "PPO+Believer",    "#1f77b4", "rgb_joints_belief", "ppo-vae{suffix}"),
    MethodSpec("ppo",  "PPO",             "#2ca02c", "rgb_joints",        "ppo-baseline"),
    MethodSpec("lstm", "PPO+LSTM",        "#9467bd", "rgb_joints",        "ppo-lstm"),
    MethodSpec("gru",  "PPO+GRU",         "#ff7f0e", "rgb_joints",        "ppo-gru"),
]

LINEWIDTH_PT = 2.5
SHADE_ALPHA = 0.20
FIG_W, FIG_H = 7.5, 4.4


def methods_for_config(belief_suffix: str) -> List[MethodSpec]:
    """Apply the belief-suffix (e.g. '' or '-ep10000') to belief methods only."""
    out = []
    for m in METHODS_TEMPLATE:
        prefix = m.prefix.format(suffix=belief_suffix) if "{suffix}" in m.prefix else m.prefix
        out.append(MethodSpec(m.key, m.label, m.color, m.suite, prefix))
    return out


def csv_row_count(p: Path) -> int:
    try:
        with p.open("rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def task_dir(task: str, suite: str) -> Path:
    return CKPT_ROOT / suite / "normalized_dense" / task


def find_csv(task: str, method: MethodSpec, seed: str) -> Optional[Path]:
    base = task_dir(task, method.suite)
    pattern = f"{method.prefix}-{task}-{seed}__{seed}__{method.suite}__*"
    cands = []
    for run_dir in base.glob(pattern):
        if not run_dir.is_dir():
            continue
        for csv in run_dir.glob("*/training_metrics.csv"):
            cands.append(csv)
    if not cands:
        return None
    # Prefer longest run; tie-break on latest timestamp folder name.
    cands.sort(key=lambda p: (csv_row_count(p), p.parent.name), reverse=True)
    return cands[0]


def load_eval(csv_path: Path, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    ev = df[df["mode"].astype(str).str.lower() == "eval"].sort_values("total_env_steps")
    return ev["total_env_steps"].to_numpy(float), ev[metric].to_numpy(float)


def aggregate(curves: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align seeds on the union x-grid, return (xs, mean, std). Uses nan-aware stats."""
    xs = sorted({float(x) for xx, _ in curves for x in xx})
    xs = np.array(xs, dtype=float)
    Ys = []
    for x, y in curves:
        m = dict(zip(x.tolist(), y.tolist()))
        Ys.append(np.array([m.get(float(xx), np.nan) for xx in xs], dtype=float))
    Y = np.vstack(Ys)
    # Per spec §6.3: mean + std across seeds.
    return xs, np.nanmean(Y, axis=0), np.nanstd(Y, axis=0, ddof=0)


def plot_task(
    task: str,
    methods: List[MethodSpec],
    metric: str,
    ylabel: str,
    title_suffix: str,
    out_path: Path,
) -> Dict[str, Dict[str, object]]:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    per_method: Dict[str, Dict[str, object]] = {}
    for m in methods:
        curves: List[Tuple[np.ndarray, np.ndarray]] = []
        per_seed_last3: Dict[str, float] = {}
        for s in SEEDS:
            csv = find_csv(task, m, s)
            if csv is None:
                continue
            x, y = load_eval(csv, metric)
            if x.size == 0:
                continue
            curves.append((x, y))
            if len(y) >= 3:
                per_seed_last3[s] = float(np.mean(y[-3:]))
            elif len(y) > 0:
                per_seed_last3[s] = float(np.mean(y))

        if curves:
            xs, mean, std = aggregate(curves)
            ax.fill_between(xs, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                            color=m.color, alpha=SHADE_ALPHA)
            ax.plot(xs, mean, color=m.color, linewidth=LINEWIDTH_PT, label=m.label)

        vals = np.array(list(per_seed_last3.values()), dtype=float) if per_seed_last3 else np.array([])
        per_method[m.key] = {
            "label": m.label,
            "per_seed": per_seed_last3,
            "mean": float(np.mean(vals)) if vals.size else float("nan"),
            "std":  float(np.std(vals, ddof=0)) if vals.size else float("nan"),
            "n_seeds": int(vals.size),
        }

    ax.set_xlabel("environment steps")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{task} — {title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return per_method


def write_summary(
    all_results: Dict[str, Dict[str, Dict[str, Dict[str, object]]]],
    config_name: str,
    out_path: Path,
) -> None:
    lines: List[str] = []
    lines.append(f"# Main Experiments — last-3-eval summary ({config_name})\n\n")
    lines.append(f"tasks: {len(TASKS)}, seeds: {SEEDS}\n")
    lines.append("Per §6.2: average of last 3 eval checkpoints per seed; mean±std across seeds.\n\n")

    for metric in ("success_once", "success_at_end"):
        lines.append(f"## {metric}\n\n")
        header = f"| {'task':30s} | {'method':20s} | n | {'mean±std':15s} | per_seed |\n"
        sep = "|" + "-" * 32 + "|" + "-" * 22 + "|---|" + "-" * 17 + "|" + "-" * 40 + "|\n"
        lines.append(header)
        lines.append(sep)
        for task in TASKS:
            for m in METHODS_TEMPLATE:
                r = all_results[task][metric].get(m.key)
                if r is None or r["n_seeds"] == 0:
                    lines.append(f"| {task:30s} | {m.label:20s} | 0 | {'N/A':15s} | — |\n")
                    continue
                ms = f"{r['mean']:.3f}±{r['std']:.3f}"
                per_seed = ", ".join(f"s{k}:{v:.3f}" for k, v in sorted(r["per_seed"].items()))
                lines.append(f"| {task:30s} | {m.label:20s} | {r['n_seeds']} | {ms:15s} | {per_seed} |\n")
            lines.append("|" + " " * 32 + "|" + " " * 22 + "|   |" + " " * 17 + "|" + " " * 40 + "|\n")
        lines.append("\n")

    out_path.write_text("".join(lines))


def run_one_config(belief_suffix: str, config_name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = methods_for_config(belief_suffix)

    print(f"\n{'=' * 60}\n  CONFIG: {config_name} (belief_suffix={belief_suffix!r})\n{'=' * 60}")
    for m in methods:
        print(f"  - {m.key:4s} -> prefix={m.prefix}")

    all_results: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {}

    for task in TASKS:
        print(f"\n  task: {task}")
        per_task: Dict[str, Dict[str, Dict[str, object]]] = {}
        for metric, ylabel in [
            ("success_once",   "success_once"),
            ("success_at_end", "success_at_end"),
        ]:
            out_path = out_dir / f"{task}__{metric}.png"
            per_method = plot_task(task, methods, metric, ylabel, config_name, out_path)
            per_task[metric] = per_method
            n_found = sum(1 for r in per_method.values() if r["n_seeds"] > 0)
            print(f"    {metric:15s} -> {n_found}/5 methods plotted -> {out_path.name}")
        all_results[task] = per_task

    summary = out_dir / "summary.md"
    write_summary(all_results, config_name, summary)
    print(f"\n  summary: {summary}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "plots")
    args = ap.parse_args()

    run_one_config(belief_suffix="",          config_name="ep6000",
                   out_dir=args.out_root / "main_ep6000")
    run_one_config(belief_suffix="-ep10000",  config_name="ep10000",
                   out_dir=args.out_root / "main_ep10000")


if __name__ == "__main__":
    main()
