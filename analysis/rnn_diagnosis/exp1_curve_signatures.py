"""Exp 1 — training-curve signatures across methods & difficulty (CPU only).

For each (task, method, seed) run in checkpoints/aaai_final we extract the eval
success_once curve and compute takeoff statistics:

  - takeoff_step : first env step where eval success_once exceeds TAKEOFF_THR
                   and never falls below it for the next PERSIST evals
  - half_step    : first env step reaching 50% of the run's final performance
  - final        : mean success_once over the last 5 evals
  - plateau_frac : fraction of training spent below TAKEOFF_THR

Hypothesis being probed: if write-selection is learned via the slow
value-bootstrap loop, recurrent methods should show long plateaus with
takeoff times that (a) come late, (b) vary wildly across seeds, and
(c) get worse with distractor count (RC3 -> RC5 -> RC9). SRB-TR, whose
writes need no learning, should take off early and consistently.

Outputs: summary CSV + per-task curve grid PNG under analysis/rnn_diagnosis/out/.
"""
from __future__ import annotations

import csv
import glob
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/zfsstore/user/s4176650/MIKASA-Robo/checkpoints/aaai_final/rgb_joints/normalized_dense"
OUT = "/zfsstore/user/s4176650/MIKASA-Robo/analysis/rnn_diagnosis/out"
TASKS = [
    "RememberColor3-v0",
    "RememberColor5-v0",
    "RememberColor9-v0",
    "RememberShape3-v0",
    "RememberShape5-v0",
    "RememberShape9-v0",
    "RememberShapeAndColor3x2-v0",
    "RememberShapeAndColor3x3-v0",
]
TAKEOFF_THR = 0.10
PERSIST = 3  # evals that must stay above threshold to count as takeoff

METHOD_ORDER = ["srbtr", "lstm", "gru", "ffm", "shm", "mlp"]
COLORS = {
    "srbtr": "#d62728",
    "lstm": "#1f77b4",
    "gru": "#2ca02c",
    "ffm": "#9467bd",
    "shm": "#8c564b",
    "mlp": "#7f7f7f",
}


def parse_run_dir(name: str):
    m = re.match(r"([a-z]+)-[a-z_]*-?seed(\d+)__", name)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def load_eval_curve(csv_path: str):
    steps, once, at_end = [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("mode") != "eval":
                continue
            try:
                steps.append(float(row["total_env_steps"]))
                once.append(float(row["success_once"]))
                at_end.append(float(row["success_at_end"]))
            except (ValueError, KeyError):
                continue
    return np.array(steps), np.array(once), np.array(at_end)


def takeoff_stats(steps, curve):
    n = len(curve)
    final = float(np.mean(curve[-5:])) if n >= 5 else float(np.mean(curve))
    takeoff = np.nan
    for i in range(n - PERSIST + 1):
        if np.all(curve[i : i + PERSIST] > TAKEOFF_THR):
            takeoff = steps[i]
            break
    half = np.nan
    if final > 0.02:
        for i in range(n):
            if curve[i] >= 0.5 * final:
                half = steps[i]
                break
    plateau = float(np.mean(curve <= TAKEOFF_THR))
    return takeoff, half, final, plateau


def main():
    os.makedirs(OUT, exist_ok=True)
    # runs[task][method][seed] = (steps, once, at_end); keep latest timestamp per seed
    runs = defaultdict(lambda: defaultdict(dict))
    seen_ts = defaultdict(dict)
    for task in TASKS:
        for run_dir in sorted(glob.glob(os.path.join(ROOT, task, "*__*"))):
            name = os.path.basename(run_dir)
            method, seed = parse_run_dir(name)
            if method is None:
                continue
            csvs = glob.glob(os.path.join(run_dir, "**", "training_metrics.csv"), recursive=True)
            if not csvs:
                continue
            ts = name.split("__")[-1]
            key = (method, seed)
            if key in seen_ts[task] and ts <= seen_ts[task][key]:
                continue
            seen_ts[task][key] = ts
            curve = load_eval_curve(sorted(csvs)[-1])
            if len(curve[0]) >= 10:
                runs[task][method][seed] = curve

    rows = []
    for task in TASKS:
        for method, per_seed in runs[task].items():
            for seed, (steps, once, at_end) in sorted(per_seed.items()):
                tko, half, final, plat = takeoff_stats(steps, once)
                rows.append(
                    dict(task=task, method=method, seed=seed,
                         takeoff_Msteps=round(tko / 1e6, 2) if np.isfinite(tko) else None,
                         half_Msteps=round(half / 1e6, 2) if np.isfinite(half) else None,
                         final_once=round(final, 3),
                         final_at_end=round(float(np.mean(at_end[-5:])), 3),
                         plateau_frac=round(plat, 2)))

    with open(os.path.join(OUT, "exp1_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ── curve grid ────────────────────────────────────────────────────────
    present_tasks = [t for t in TASKS if runs[t]]
    ncols = min(4, len(present_tasks))
    nrows = (len(present_tasks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    for ax_i, task in enumerate(present_tasks):
        ax = axes[ax_i // ncols][ax_i % ncols]
        for method in METHOD_ORDER:
            if method not in runs[task]:
                continue
            for seed, (steps, once, _) in sorted(runs[task][method].items()):
                ax.plot(steps / 1e6, once, color=COLORS[method], alpha=0.55, lw=1.2,
                        label=method if seed == sorted(runs[task][method])[0] else None)
        ax.set_title(task.replace("-v0", ""), fontsize=10)
        ax.set_xlabel("env steps (M)")
        ax.set_ylabel("eval success_once")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7)
    for j in range(len(present_tasks), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp1_curves.png"), dpi=140)

    # ── console digest ────────────────────────────────────────────────────
    print(f"{'task':<28} {'method':<7} {'final_once (per seed)':<26} {'takeoff M-steps (per seed)'}")
    for task in present_tasks:
        for method in METHOD_ORDER:
            if method not in runs[task]:
                continue
            fin, tko = [], []
            for seed, (steps, once, _) in sorted(runs[task][method].items()):
                t, h, f, p = takeoff_stats(steps, once)
                fin.append(f"{f:.2f}")
                tko.append(f"{t/1e6:.1f}" if np.isfinite(t) else "never")
            print(f"{task:<28} {method:<7} {'/'.join(fin):<26} {'/'.join(tko)}")


if __name__ == "__main__":
    main()
