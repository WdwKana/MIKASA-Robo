#!/usr/bin/env python3
"""
Reproducible plotting for last5 ablation comparisons (belief runs).

Curves (3 algos, user-facing names / ICML figure style):
  - AIB+PPO                          (red)
  - AIB+PPO (Without Action Head)
  - AIB+PPO (With Mse Inverse Dynamic)

Data:
  - reads `training_metrics.csv`
  - filters `mode == "eval"`
  - x-axis: `total_env_steps` (label configurable, default: "steps")
  - y-axis: success_once / success_at_end
  - plots mean + min/max band across the chosen seeds per curve

Seed handling:
  - You pass a seedset; each curve uses the subset of that seedset that exists for that curve.
  - (So different curves may use different counts if some runs are missing.)

Run selection:
  - Excludes runs whose run_dir contains token v1/v2/expert/test (tokenized by -_/ boundaries).
  - Uses canonical run_dir prefixes per algorithm for robust matching.
  - If multiple candidates exist for the same (algo, seed), chooses the latest timestamp suffix `__YYYYMMDD_HHMMSS`
    when present, else last in sort order.
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
IGNORE_TOKEN_RE = re.compile(r"(^|[-_/])(?:v1|v2|expert|test)($|[-_/])", re.IGNORECASE)

# ICML figure checklist:
#   - No title inside figure: default behavior is **no title** (caption should provide it).
#   - Axis names: always set x/y labels.
#   - Legend: always include legend for each curve.
#   - Legibility / reproduction: thick, dark lines (>=0.5 pt); we use 2.8 pt.
#   - Background: white; shading is translucent.
LINEWIDTH_PT = 2.8
SHADE_ALPHA = 0.15
# Figure sizing (match plot_baseline_newhyper_vs_aib_v6_last5.py):
# ICML does not mandate a specific aspect ratio; the practical requirement is readability and fitting paper layout.
# The old default (10.5, 4.2) is very wide; use a more typical 2-column-friendly size by default.
FIG_WIDTH_IN = 6.8
FIG_HEIGHT_IN = 3.8


def place_legend_avoid_overlap(fig: plt.Figure, ax: plt.Axes):
    """
    Put legend *inside* the axes, choosing a corner that overlaps the plotted data the least.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    candidates = ["upper right", "upper left", "lower left", "lower right"]

    pts_axes = []
    for ln in ax.get_lines():
        xy = ln.get_xydata()
        if xy.size == 0:
            continue
        disp = ax.transData.transform(xy)
        axes_xy = ax.transAxes.inverted().transform(disp)
        pts_axes.append(axes_xy)
    if pts_axes:
        pts_axes = np.vstack(pts_axes)
    else:
        pts_axes = np.zeros((0, 2), dtype=float)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    best_loc = None
    best_score = None
    for loc in candidates:
        leg = ax.legend(handles, labels, frameon=False, loc=loc)
        fig.canvas.draw()
        bbox_disp = leg.get_window_extent(renderer=renderer)
        (x0, y0) = ax.transAxes.inverted().transform((bbox_disp.x0, bbox_disp.y0))
        (x1, y1) = ax.transAxes.inverted().transform((bbox_disp.x1, bbox_disp.y1))
        xmin, xmax = (min(x0, x1), max(x0, x1))
        ymin, ymax = (min(y0, y1), max(y0, y1))
        if pts_axes.shape[0] == 0:
            score = 0
        else:
            inside = (
                (pts_axes[:, 0] >= xmin)
                & (pts_axes[:, 0] <= xmax)
                & (pts_axes[:, 1] >= ymin)
                & (pts_axes[:, 1] <= ymax)
            )
            score = int(np.sum(inside))
        leg.remove()
        if best_score is None or score < best_score:
            best_score = score
            best_loc = loc
    ax.legend(handles, labels, frameon=False, loc=(best_loc or "lower right"))


def resolve_training_metrics_csv(pathish: Path) -> Path:
    """
    Accept either:
      - .../training_metrics.csv
      - ts_dir: .../<timestamp>/   (contains training_metrics.csv)
      - run_dir: .../<run_name>/   (contains <timestamp>/training_metrics.csv)
    """
    p = Path(pathish)
    if p.is_file() and p.name == "training_metrics.csv":
        return p
    if p.is_dir():
        direct = p / "training_metrics.csv"
        if direct.exists():
            return direct
        cands = list(p.glob("*/training_metrics.csv"))
        if cands:
            return choose_latest(cands)
    raise FileNotFoundError(f"Could not resolve training_metrics.csv from: {p}")

def parse_ts(s: str) -> str:
    m = TS_RE.search(s)
    return m.group(1) if m else ""


def load_eval(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    ev = df[df["mode"] == "eval"].copy().sort_values("total_env_steps")
    x = ev["total_env_steps"].to_numpy(float)
    return x, ev["success_once"].to_numpy(float), ev["success_at_end"].to_numpy(float)


def choose_latest(paths: List[Path]) -> Path:
    if len(paths) == 1:
        return paths[0]
    return sorted(paths, key=lambda p: parse_ts(p.parent.parent.name))[-1]


def parse_seed_from_run_dir(run_dir: str) -> Optional[str]:
    m = re.search(r"__(\d+)__", run_dir)
    return m.group(1) if m else None


def seedset_from_arg(s: str) -> List[str]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"Empty seedset: {s}")
    return parts


@dataclass(frozen=True)
class Curve:
    key: str
    label: str
    color: str


DEFAULT_CURVES = [
    # Legend order: show our method first (red), then ablations.
    Curve("aib", "AIB+PPO", "#d62728"),
    Curve("no_action", "AIB+PPO (Without Action Head)", "#1f77b4"),
    Curve("mse_inv_dyn", "AIB+PPO (With Mse Inverse Dynamic)", "#2ca02c"),
]


def build_curves(label_overrides: Optional[Dict[str, str]] = None) -> List[Curve]:
    overrides = label_overrides or {}
    return [Curve(c.key, str(overrides.get(c.key, c.label)), c.color) for c in DEFAULT_CURVES]

PRESETS: Dict[str, Dict[str, object]] = {
    # RememberShapeAndColor3x2-v0, 3 algorithms x 3 seeds (33/42/99), user-provided run folders.
    "rsac3x2_ablation_newhyper_seedset_33_42_99": {
        "task_name": "RememberShapeAndColor3x2-v0",
        "seeds": ["33", "42", "99"],
        # AIB+PPO (ours)
        "aib_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberShapeAndColor3x2-v0/ppo-last5-cvae-v6-ent0001-kl005-gamma095-lr1e4-33__33__rgb_joints_belief__20260124_025525",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberShapeAndColor3x2-v0/ppo-last5-cvae-v6-ent0001-kl005-gamma095-lr1e4-42__42__rgb_joints_belief__20260123_224544",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberShapeAndColor3x2-v0/ppo-last5-cvae-v6-ent0001-kl005-gamma095-lr1e4-99__99__rgb_joints_belief__20260124_051828",
        ],
        # AIB+PPO (Without Action Head)
        "no_action_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberShapeAndColor3x2-v0/ppo-last5-cvae-newhyper-none-33__33__rgb_joints_belief__20260128_051617",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberShapeAndColor3x2-v0/ppo-last5-cvae-newhyper-none-42__42__rgb_joints_belief__20260128_073841",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberShapeAndColor3x2-v0/ppo-last5-cvae-newhyper-none-99__99__rgb_joints_belief__20260128_100648",
        ],
        # AIB+PPO (With Mse Inverse Dynamic)
        "mse_inv_dyn_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberShapeAndColor3x2-v0/ppo-last5-cvae-newhyper-ablation-mse-33__33__rgb_joints_belief__20260128_051721",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberShapeAndColor3x2-v0/ppo-last5-cvae-newhyper-ablation-mse-42__42__rgb_joints_belief__20260128_074208",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberShapeAndColor3x2-v0/ppo-last5-cvae-newhyper-ablation-mse-99__99__rgb_joints_belief__20260128_100732",
        ],
    },
    # InterceptMedium-v0, 3 algorithms x 3 seeds (33/42/99), user-provided run folders.
    # Order matches CURVES keys: aib / no_action / mse_inv_dyn
    "interceptmedium_ablation_seedset_33_42_99_v2_none_v7": {
        "task_name": "InterceptMedium-v0",
        "seeds": ["33", "42", "99"],
        # AIB+PPO (v2)
        "aib_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptMedium-v0/ppo-cvae-v2-interceptmedium-33__33__rgb_joints_belief__20260126_212815",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptMedium-v0/ppo-cvae-v2-interceptmedium-42__42__rgb_joints_belief__20260126_231857",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptMedium-v0/ppo-cvae-v2-interceptmedium-99__99__rgb_joints_belief__20260127_010912",
        ],
        # AIB+PPO (Without Action Head)
        "no_action_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptMedium-v0/ppo-cvae-ablation-none-interceptmedium-33__33__rgb_joints_belief__20260128_112324",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptMedium-v0/ppo-cvae-ablation-none-interceptmedium-42__42__rgb_joints_belief__20260128_131312",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptMedium-v0/ppo-cvae-ablation-none-interceptmedium-99__99__rgb_joints_belief__20260128_150206",
        ],
        # AIB+PPO (With Mse Inverse Dynamic) (v7)
        "mse_inv_dyn_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptMedium-v0/ppo-cvae-v7-interceptmedium-33__33__rgb_joints_belief__20260128_123637",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptMedium-v0/ppo-cvae-v7-interceptmedium-42__42__rgb_joints_belief__20260128_142750",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptMedium-v0/ppo-cvae-v7-interceptmedium-99__99__rgb_joints_belief__20260128_161938",
        ],
    },
    # InterceptFast-v0, 3 algorithms x 3 seeds (33/42/99), user-provided run folders.
    # Order matches CURVES keys: aib / no_action / mse_inv_dyn
    "interceptfast_ablation_seedset_33_42_99_v2_none_mse": {
        "task_name": "InterceptFast-v0",
        "seeds": ["33", "42", "99"],
        # AIB+PPO (v2)
        "aib_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-cvae-v2-interceptfast-33__33__rgb_joints_belief__20260127_213816/20260127_213816",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-cvae-v2-interceptfast-42__42__rgb_joints_belief__20260127_232947",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-cvae-v2-interceptfast-99__99__rgb_joints_belief__20260128_011917",
        ],
        # AIB+PPO (Without Action Head) (none)
        "no_action_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-cvae-none-interceptfast-33__33__rgb_joints_belief__20260128_171251",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-cvae-none-interceptfast-42__42__rgb_joints_belief__20260128_190119",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-cvae-none-interceptfast-99__99__rgb_joints_belief__20260128_205231",
        ],
        # AIB+PPO (With Mse Inverse Dynamic) (mse)
        "mse_inv_dyn_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-cvae-mse2-interceptfast-33__33__rgb_joints_belief__20260129_061237",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-cvae-mse2-interceptfast-42__42__rgb_joints_belief__20260129_061237",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-cvae-mse2-interceptfast-99__99__rgb_joints_belief__20260129_061237",
        ],
    },
    # InterceptFast-v0, AIB+PPO parameter sweep (3 params x 3 seeds).
    # Internal slots still use aib / no_action / mse_inv_dyn, but user-facing labels
    # are overridden so the figure legend shows the compared parameter values.
    "interceptfast_param_sweep_seedset_33_42_99": {
        "task_name": "InterceptFast-v0",
        "seeds": ["33", "42", "99"],
        "curve_labels": {
            "aib": "PPO+AIB (λ=1)",
            "no_action": "PPO+AIB (λ=0.1)",
            "mse_inv_dyn": "PPO+AIB (λ=0.01)",
        },
        # The v2/v6 token in these run names is a historical naming mismatch and
        # does not denote different algorithm versions for this comparison.
        "aib_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-cvae-v2-interceptfast-33__33__rgb_joints_belief__20260127_213816",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-cvae-v2-interceptfast-42__42__rgb_joints_belief__20260127_232947",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-cvae-v2-interceptfast-99__99__rgb_joints_belief__20260128_011917",
        ],
        "no_action_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-cvae-v6-01-interceptfast-33__33__rgb_joints_belief__20260318_234003",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-cvae-v6-01-interceptfast-42__42__rgb_joints_belief__20260319_013109",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-cvae-v6-01-interceptfast-99__99__rgb_joints_belief__20260319_032128",
        ],
        "mse_inv_dyn_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-cvae-v6-001-interceptfast-33__33__rgb_joints_belief__20260318_234003",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-cvae-v6-001-interceptfast-42__42__rgb_joints_belief__20260319_013109",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-cvae-v6-001-interceptfast-99__99__rgb_joints_belief__20260319_032007",
        ],
    },
    # RememberColor9-v0, 3 algorithms x 3 seeds (33/42/99).
    # Order matches CURVES keys: aib / no_action / mse_inv_dyn
    "color9_ablation_seedset_33_42_99_v6_none_mse": {
        "task_name": "RememberColor9-v0",
        "seeds": ["33", "42", "99"],
        # AIB+PPO (v6)
        "aib_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberColor9-v0/ppo-cvae-last5-v6-33__33__rgb_joints_belief__20260127_044744",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberColor9-v0/ppo-cvae-last5-v6-42__42__rgb_joints_belief__20260127_071517",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberColor9-v0/ppo-cvae-last5-v6-99__99__rgb_joints_belief__20260127_094253",
        ],
        # AIB+PPO (Without Action Head) (none)
        "no_action_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberColor9-v0/ppo-cvae-last5-none-33__33__rgb_joints_belief__20260128_171839",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberColor9-v0/ppo-cvae-last5-none-42__42__rgb_joints_belief__20260128_194931",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberColor9-v0/ppo-cvae-last5-none-99__99__rgb_joints_belief__20260128_221959",
        ],
        # AIB+PPO (With Mse Inverse Dynamic) (mse)
        "mse_inv_dyn_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberColor9-v0/ppo-cvae-last5-mse-33__33__rgb_joints_belief__20260128_181033",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberColor9-v0/ppo-cvae-last5-mse-42__42__rgb_joints_belief__20260128_203832",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberColor9-v0/ppo-cvae-last5-mse-99__99__rgb_joints_belief__20260128_230627",
        ],
    },
}


def canonical_prefixes(task: str, curve_key: str) -> List[str]:
    raise RuntimeError("Deprecated: use canonical_run_dir_patterns() instead")


def canonical_run_dir_patterns(task: str, curve_key: str, seed: str) -> List[re.Pattern]:
    """
    Return strict, *canonical* run_dir matchers.
    Important: this intentionally does NOT match variants like `ppo-last5-cvae-v7-1-33__...`.
    """
    # (task isn't currently used but kept for future flexibility)
    if curve_key == "aib":
        return [
            re.compile(rf"^ppo-last5-cvae-{seed}__{seed}__"),
            re.compile(rf"^ppo-cvae-last5-{seed}__{seed}__"),
            re.compile(rf"^cvae-last5-{seed}__{seed}__"),
        ]
    if curve_key == "no_action":
        return [
            re.compile(rf"^ppo-last5-ablation-none-{seed}__{seed}__"),
            re.compile(rf"^ppo-last5-cvae-newhyper-none-{seed}__{seed}__"),
        ]
    if curve_key == "mse_inv_dyn":
        return [
            re.compile(rf"^ppo-last5-cvae-newhyper-ablation-mse-{seed}__{seed}__"),
        ]
    raise ValueError(curve_key)

def plot_mean_minmax(*, ax: plt.Axes, label: str, color: str, seed_curves: List[Tuple[np.ndarray, np.ndarray]]):
    """
    Union-x grid (no interpolation), NaN-masked mean + min/max band.
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
    ax.fill_between(xs, y_low, y_high, color=color, alpha=SHADE_ALPHA)
    ax.plot(xs, y_mean, color=color, linewidth=LINEWIDTH_PT, label=label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--belief-root", type=Path, default=Path("/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense"))
    ap.add_argument("--out-root", type=Path, default=Path("/local/s4176650/MIKASA-Robo/plots_ablation_last5"))
    ap.add_argument("--out-dir", type=Path, default=None, help="Optional exact output directory. Overrides --out-root/task/last5 layout.")
    ap.add_argument("--task", required=False, default="", help="Env name, e.g. RememberShapeAndColor3x2-v0 (optional if using --preset)")
    ap.add_argument("--seedset", required=False, default="33,42,99", help="Comma-separated seeds, e.g. 33,42,99 (optional if using --preset)")
    ap.add_argument(
        "--preset",
        type=str,
        default="",
        help=f"Optional built-in preset name. Available: {', '.join(sorted(PRESETS.keys()))}",
    )
    ap.add_argument("--aib-paths", type=str, default="", help="Comma-separated run/ts/csv paths aligned with seed order.")
    ap.add_argument("--no-action-paths", type=str, default="", help="Comma-separated run/ts/csv paths aligned with seed order.")
    ap.add_argument("--mse-inv-dyn-paths", type=str, default="", help="Comma-separated run/ts/csv paths aligned with seed order.")
    ap.add_argument("--aib-label", type=str, default="", help="Optional legend label override for the first curve.")
    ap.add_argument("--no-action-label", type=str, default="", help="Optional legend label override for the second curve.")
    ap.add_argument("--mse-inv-dyn-label", type=str, default="", help="Optional legend label override for the third curve.")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--xlabel", type=str, default="steps")
    ap.add_argument("--figwidth-in", type=float, default=FIG_WIDTH_IN, help="Figure width in inches.")
    ap.add_argument("--figheight-in", type=float, default=FIG_HEIGHT_IN, help="Figure height in inches.")
    ap.add_argument(
        "--with-title",
        action="store_true",
        default=False,
        help="If set, include a title inside the figure (NOT recommended for ICML; use caption instead).",
    )
    ap.add_argument("--tight-y", action="store_true")
    ap.add_argument("--y-pad", type=float, default=0.03)
    ap.add_argument("--y-min-span", type=float, default=0.12)
    ap.add_argument("--tag", type=str, default="", help="Optional suffix tag for filenames, e.g. seedset_33_42_123")
    args = ap.parse_args()

    def _parse_list(s: str) -> List[str]:
        return [x.strip() for x in s.split(",") if x.strip()]

    # Apply preset (preferred for reproducible plotting).
    if args.preset:
        if args.preset not in PRESETS:
            raise SystemExit(f"Unknown --preset={args.preset}. Available: {', '.join(sorted(PRESETS.keys()))}")
        ps = PRESETS[args.preset]
        task = str(ps["task_name"])
        seedset = [str(x) for x in ps["seeds"]]
        aib_paths = [str(x) for x in ps.get("aib_paths", [])]
        no_action_paths = [str(x) for x in ps.get("no_action_paths", [])]
        mse_inv_dyn_paths = [str(x) for x in ps.get("mse_inv_dyn_paths", [])]
        label_overrides = {str(k): str(v) for k, v in dict(ps.get("curve_labels", {})).items()}
    else:
        if not args.task:
            raise SystemExit("--task is required unless using --preset")
        task = args.task
        seedset = seedset_from_arg(args.seedset)
        aib_paths = _parse_list(args.aib_paths)
        no_action_paths = _parse_list(args.no_action_paths)
        mse_inv_dyn_paths = _parse_list(args.mse_inv_dyn_paths)
        label_overrides = {}

    if args.aib_label:
        label_overrides["aib"] = args.aib_label
    if args.no_action_label:
        label_overrides["no_action"] = args.no_action_label
    if args.mse_inv_dyn_label:
        label_overrides["mse_inv_dyn"] = args.mse_inv_dyn_label

    curves = build_curves(label_overrides)

    task_dir = args.belief_root / task
    out_dir = args.out_dir if args.out_dir is not None else (args.out_root / task / "last5")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmdline = " ".join(shlex.quote(x) for x in [os.path.abspath(__file__), *os.sys.argv[1:]])

    # Gather CSVs once (used only in auto-discovery mode).
    all_csvs = list(task_dir.rglob("training_metrics.csv")) if (task_dir.exists()) else []

    # Select CSV per (curve, seed). Two modes:
    #   - Explicit paths (from preset or CLI) -> resolve training_metrics.csv from run/ts/csv paths
    #   - Auto-discovery (legacy) -> search by canonical patterns
    used: Dict[str, Dict[str, Path]] = {c.key: {} for c in curves}
    ambiguities: List[str] = []

    explicit_mode = bool(aib_paths or no_action_paths or mse_inv_dyn_paths or args.preset)
    if explicit_mode:
        # Enforce full triplets per curve (aligned with seedset order).
        if len(aib_paths) != len(seedset):
            raise SystemExit("Need exactly one path per seed for --aib-paths (or preset).")
        if len(no_action_paths) != len(seedset):
            raise SystemExit("Need exactly one path per seed for --no-action-paths (or preset).")
        if len(mse_inv_dyn_paths) != len(seedset):
            raise SystemExit("Need exactly one path per seed for --mse-inv-dyn-paths (or preset).")
        for i, seed in enumerate(seedset):
            used["aib"][seed] = resolve_training_metrics_csv(Path(aib_paths[i]))
            used["no_action"][seed] = resolve_training_metrics_csv(Path(no_action_paths[i]))
            used["mse_inv_dyn"][seed] = resolve_training_metrics_csv(Path(mse_inv_dyn_paths[i]))
    else:
        for seed in seedset:
            for c in curves:
                cands = []
                patterns = canonical_run_dir_patterns(task, c.key, seed)
                for p in all_csvs:
                    run_dir = p.relative_to(task_dir).parts[0]
                    if IGNORE_TOKEN_RE.search(run_dir):
                        continue
                    if "last5" not in run_dir.lower():
                        continue
                    if any(pat.match(run_dir) for pat in patterns):
                        cands.append(p)
                if not cands:
                    continue
                chosen = choose_latest(cands)
                used[c.key][seed] = chosen
                if len(cands) > 1:
                    ambiguities.append(f"GROUP ({c.key}, seed={seed}) chose {chosen} from {len(cands)} candidates")

    cache: Dict[Path, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def load_cached(p: Path):
        if p not in cache:
            cache[p] = load_eval(p)
        return cache[p]

    def tight_ylim(y_arrays: List[np.ndarray]) -> Tuple[float, float]:
        yy = np.concatenate(y_arrays) if y_arrays else np.array([0.0, 1.0])
        yy = yy[~np.isnan(yy)]
        y_min = float(np.min(yy))
        y_max = float(np.max(yy))
        if (y_max - y_min) < args.y_min_span:
            mid = 0.5 * (y_min + y_max)
            y_min = mid - 0.5 * args.y_min_span
            y_max = mid + 0.5 * args.y_min_span
        y0 = max(0.0, y_min - args.y_pad)
        y1 = min(1.0, y_max + args.y_pad)
        return y0, y1

    def plot_metric(metric: str):
        # Collect y for tight-y across whatever will be plotted
        all_y: List[np.ndarray] = []
        if args.tight_y:
            for c in curves:
                for s, p in used[c.key].items():
                    x, y_once, y_end = load_cached(p)
                    all_y.append(y_once if metric == "success_once" else y_end)
            y0, y1 = tight_ylim(all_y)
        else:
            y0, y1 = 0.0, 1.0

        fig, ax = plt.subplots(figsize=(args.figwidth_in, args.figheight_in))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        for c in curves:
            seeds = sorted(used[c.key].keys(), key=int)
            if not seeds:
                continue
            seed_curves: List[Tuple[np.ndarray, np.ndarray]] = []
            for s in seeds:
                x, y_once, y_end = load_cached(used[c.key][s])
                y = y_once if metric == "success_once" else y_end
                seed_curves.append((x, y))
            plot_mean_minmax(ax=ax, label=c.label, color=c.color, seed_curves=seed_curves)

        ax.set_ylim(y0, y1)
        ax.set_xlabel(args.xlabel)
        ax.set_ylabel("success")
        if args.with_title:
            ax.set_title(task)
        ax.grid(True, alpha=0.3)
        place_legend_avoid_overlap(fig, ax)
        fig.tight_layout()

        suffix = f"__{args.tag}" if args.tag else ""
        out_path = out_dir / f"{task}__last5__{metric}{suffix}__3algos.png"
        fig.savefig(out_path, dpi=args.dpi, facecolor="white")
        plt.close(fig)

    plot_metric("success_once")
    plot_metric("success_at_end")

    # report
    rep_lines: List[str] = []
    rep_lines.append(f"### Used paths for {task} last5 3-way comparison plot\n")
    rep_lines.append(f"belief_root: {args.belief_root}\n")
    rep_lines.append(f"out_dir: {out_dir}\n")
    rep_lines.append(f"seedset_requested: {seedset}\n")
    rep_lines.append(f"preset: {args.preset or '(none)'}\n")
    rep_lines.append(f"tag: {args.tag}\n")
    rep_lines.append(f"plot_script: {Path(__file__).resolve()}\n")
    rep_lines.append(f"plot_command: {cmdline}\n")
    rep_lines.append(f"icml_linewidth_pt: {LINEWIDTH_PT}\n")
    rep_lines.append("icml_no_title_default: True (use caption in LaTeX)\n")
    rep_lines.append("curve_labels:\n")
    for c in curves:
        rep_lines.append(f"  - {c.key}: {c.label}\n")
    rep_lines.append("\n## Used CSVs\n\n")
    for c in curves:
        seeds = sorted(used[c.key].keys(), key=int)
        rep_lines.append(f"[{c.key}] seeds={seeds}\n")
        for s in seeds:
            rep_lines.append(f"  seed{s}: {used[c.key][s]}\n")
        rep_lines.append("\n")

    rep_lines.append("## Ambiguities / multiple candidates\n")
    if ambiguities:
        rep_lines.extend([f"- {a}\n" for a in ambiguities])
    else:
        rep_lines.append("None\n")

    rep_name = f"{task}__last5__report{('__' + args.tag) if args.tag else ''}__3algos.txt"
    (out_dir / rep_name).write_text("".join(rep_lines))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

