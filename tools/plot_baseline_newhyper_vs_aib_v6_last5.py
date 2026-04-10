#!/usr/bin/env python3
"""
Plot comparison for memtasks (last5), aggregating across a fixed seed set using:
  - mean curve + min/max shading
  - `training_metrics.csv`, filtered by `mode`, x = `total_env_steps`

Originally used for 2 curves (baseline vs AIB v6). Extended to support an
optional 3rd curve (e.g., nonbelief/ppo) while keeping the same plotting style
and reproducibility reporting.

ICML figure checklist (see ICML template "Figures" subsection):
  - No title inside figure: default behavior is **no title** (caption should provide it).
  - Axis names: we always set x/y labels.
  - Legend: we always add a legend naming each curve.
  - Legibility / reproduction: curve lines are dark and >= 0.5 pt thick
    (we use LINEWIDTH_PT >= 2.5 by default).
  - Background: keep white background (no gray text background); shading is translucent.
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

LINEWIDTH_PT = 2.8  # ICML: >= 0.5 pt for reproduction
SHADE_ALPHA = 0.15
# Figure sizing:
# ICML does not mandate a specific aspect ratio; the practical requirement is readability and fitting paper layout.
# The old default (10.5, 4.2) is very wide; use a more typical 2-column-friendly size by default.
FIG_WIDTH_IN = 6.8
FIG_HEIGHT_IN = 3.8


def place_legend_avoid_overlap(fig: plt.Figure, ax: plt.Axes):
    """
    Put legend *inside* the axes, choosing a corner that overlaps the plotted data the least.

    Matplotlib has loc="best", but it can still overlap with dense curves/shading.
    This function tests a small set of common corners and picks the one with minimal overlap
    with line points in axes coordinates.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    # Candidate corners (inside axes)
    candidates = ["upper right", "upper left", "lower left", "lower right"]

    # Collect all line points (already plotted) in axes coordinates.
    pts_axes = []
    for ln in ax.get_lines():
        xy = ln.get_xydata()
        if xy.size == 0:
            continue
        # transform from data -> display -> axes coords
        disp = ax.transData.transform(xy)
        axes_xy = ax.transAxes.inverted().transform(disp)
        pts_axes.append(axes_xy)
    if pts_axes:
        pts_axes = np.vstack(pts_axes)
    else:
        pts_axes = np.zeros((0, 2), dtype=float)

    # Force a draw so we can measure legend extents.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    best_loc = None
    best_score = None

    for loc in candidates:
        # Create a temporary legend to measure its bbox.
        leg = ax.legend(handles, labels, frameon=False, loc=loc)
        fig.canvas.draw()
        bbox_disp = leg.get_window_extent(renderer=renderer)
        # Convert legend bbox into axes coordinates
        (x0, y0) = ax.transAxes.inverted().transform((bbox_disp.x0, bbox_disp.y0))
        (x1, y1) = ax.transAxes.inverted().transform((bbox_disp.x1, bbox_disp.y1))
        xmin, xmax = (min(x0, x1), max(x0, x1))
        ymin, ymax = (min(y0, y1), max(y0, y1))

        # Score: number of curve points that fall inside legend bbox (lower is better).
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

        # Remove temp legend before next candidate.
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

    This lets you feed run folders directly (more convenient than exact CSV paths).
    """
    p = Path(pathish)
    if p.is_file() and p.name == "training_metrics.csv":
        return p
    if p.is_dir():
        direct = p / "training_metrics.csv"
        if direct.exists():
            return direct
        # Common layouts put CSV one level below (run_dir -> ts_dir -> csv).
        cands = list(p.glob("*/training_metrics.csv"))
        if cands:
            return choose_latest(cands)
    raise FileNotFoundError(f"Could not resolve training_metrics.csv from: {p}")


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
    # Legend order preference: show our method first (red), then baseline (blue), then nonbelief (green).
    CurveSpec("aib", "PPO+AIB (ours)", "#d62728", True),
    CurveSpec("baseline", "PPO+Believer", "#1f77b4", True),
    # ICML: prefer distinct, dark colors. Use green for the 3rd curve (red/blue/green).
    CurveSpec("nonbelief", "PPO", "#2ca02c", False),
]

PRESETS: Dict[str, Dict[str, object]] = {
    # RememberShapeAndColor3x2-v0 (seedset 33/42/99) using user-provided run folders.
    "rsac3x2_seedset_33_42_99_v6_newhyper": {
        "task_name": "RememberShapeAndColor3x2-v0",
        "seeds": ["33", "42", "99"],
        "baseline_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberShapeAndColor3x2-v0/ppo-last5-baseline-newhyper-lr1e-4-33__33__rgb_joints_belief__20260124_025315/20260124_025315",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberShapeAndColor3x2-v0/ppo-last5-baseline-newhyper-lr1e-4-42__42__rgb_joints_belief__20260124_051543",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberShapeAndColor3x2-v0/ppo-last5-baseline-newhyper-lr1e-4-99__99__rgb_joints_belief__20260124_073705",
        ],
        "aib_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberShapeAndColor3x2-v0/ppo-last5-cvae-v6-ent0001-kl005-gamma095-lr1e4-33__33__rgb_joints_belief__20260124_025525",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberShapeAndColor3x2-v0/ppo-last5-cvae-v6-ent0001-kl005-gamma095-lr1e4-42__42__rgb_joints_belief__20260123_224544",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberShapeAndColor3x2-v0/ppo-last5-cvae-v6-ent0001-kl005-gamma095-lr1e4-99__99__rgb_joints_belief__20260124_051828",
        ],
        "nonbelief_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints/normalized_dense/RememberShapeAndColor3x2-v0/ppo-nonbelief-newhyper-lr1e-4-33__33__rgb_joints__20260124_095903",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints/normalized_dense/RememberShapeAndColor3x2-v0/ppo-nonbelief-newhyper-lr1e-4-42__42__rgb_joints__20260124_114704",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints/normalized_dense/RememberShapeAndColor3x2-v0/ppo-nonbelief-newhyper-lr1e-4-99__99__rgb_joints__20260124_133641",
        ],
    },
    # RememberColor5-v0 (seedset 33/42/99) using user-provided run folders.
    "color5_seedset_33_42_99_v6_newhyper": {
        "task_name": "RememberColor5-v0",
        "seeds": ["33", "42", "99"],
        "baseline_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberColor5-v0/ppo-baseline-last5-newhyper1e-4-33__33__rgb_joints_belief__20260124_074349",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberColor5-v0/ppo-baseline-last5-newhyper1e-4-42__42__rgb_joints_belief__20260124_095524",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberColor5-v0/ppo-baseline-last5-newhyper1e-4-99__99__rgb_joints_belief__20260124_120724",
        ],
        "aib_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberColor5-v0/ppo-cvae-last5-v6-1-newhyper1e-4-33__33__rgb_joints_belief__20260124_025624",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberColor5-v0/ppo-cvae-last5-v6-1-newhyper1e-4-42__42__rgb_joints_belief__20260124_051344",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberColor5-v0/ppo-cvae-last5-v6-1-newhyper1e-4-99__99__rgb_joints_belief__20260124_073240",
        ],
        "nonbelief_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints/normalized_dense/RememberColor5-v0/ppo-nonbelief-newhyper1e-4-baseline-33__33__rgb_joints__20260124_103240",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints/normalized_dense/RememberColor5-v0/ppo-nonbelief-newhyper1e-4-baseline-42__42__rgb_joints__20260124_122153",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints/normalized_dense/RememberColor5-v0/ppo-nonbelief-newhyper1e-4-baseline-99__99__rgb_joints__20260124_141200",
        ],
    },
    # RememberColor9-v0 (seedset 33/42/99) using user-provided run folders.
    "color9_seedset_33_42_99_v6_newhyper": {
        "task_name": "RememberColor9-v0",
        "seeds": ["33", "42", "99"],
        "baseline_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberColor9-v0/ppo-baseline-last5-newhyper-33__33__rgb_joints_belief__20260127_045539",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberColor9-v0/ppo-baseline-last5-newhyper-42__42__rgb_joints_belief__20260127_072000",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberColor9-v0/ppo-baseline-last5-newhyper-99__99__rgb_joints_belief__20260127_094326",
        ],
        "aib_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberColor9-v0/ppo-cvae-last5-v6-33__33__rgb_joints_belief__20260127_044744",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberColor9-v0/ppo-cvae-last5-v6-42__42__rgb_joints_belief__20260127_071517",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/RememberColor9-v0/ppo-cvae-last5-v6-99__99__rgb_joints_belief__20260127_094253",
        ],
        "nonbelief_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints/normalized_dense/RememberColor9-v0/ppo-nonbelief-baseline-newhyper-33__33__rgb_joints__20260127_050829",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints/normalized_dense/RememberColor9-v0/ppo-nonbelief-baseline-newhyper-42__42__rgb_joints__20260127_070646",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints/normalized_dense/RememberColor9-v0/ppo-nonbelief-baseline-newhyper-99__99__rgb_joints__20260127_090617",
        ],
    },
    # InterceptMedium-v0 (seedset 33/42/99) using user-provided run folders.
    "interceptmedium_seedset_33_42_99_belief_cvae_v2_nonbelief": {
        "task_name": "InterceptMedium-v0",
        "seeds": ["33", "42", "99"],
        # baseline curve == belief PPO (blue label: PPO+Believer)
        "baseline_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptMedium-v0/ppo-belief-interceptmedium-33__33__rgb_joints_belief__20260127_035552",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptMedium-v0/ppo-belief-interceptmedium-42__42__rgb_joints_belief__20260127_054208",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptMedium-v0/ppo-belief-interceptmedium-99__99__rgb_joints_belief__20260127_072753",
        ],
        # aib curve == cvae-v2 (red label: PPO+AIB (ours))
        "aib_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptMedium-v0/ppo-cvae-v2-interceptmedium-33__33__rgb_joints_belief__20260126_212815",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptMedium-v0/ppo-cvae-v2-interceptmedium-42__42__rgb_joints_belief__20260126_231857",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptMedium-v0/ppo-cvae-v2-interceptmedium-99__99__rgb_joints_belief__20260127_010912",
        ],
        # nonbelief curve == PPO (green label: PPO)
        "nonbelief_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints/normalized_dense/InterceptMedium-v0/ppo-nonbelief-newhyper-lr1e-4-33__33__rgb_joints__20260127_005114",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints/normalized_dense/InterceptMedium-v0/ppo-nonbelief-newhyper-lr1e-4-42__42__rgb_joints__20260127_021819",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints/normalized_dense/InterceptMedium-v0/ppo-nonbelief-newhyper-lr1e-4-99__99__rgb_joints__20260127_034421",
        ],
    },
    # InterceptFast-v0 (seedset 33/42/99) using user-provided run folders.
    "interceptfast_seedset_33_42_99_belief_cvae_v2_nonbelief": {
        "task_name": "InterceptFast-v0",
        "seeds": ["33", "42", "99"],
        # baseline curve == belief PPO (blue label: PPO+Believer)
        "baseline_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-belief-interceptfast-33__33__rgb_joints_belief__20260127_215614",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-belief-interceptfast-42__42__rgb_joints_belief__20260127_234338",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-belief-interceptfast-99__99__rgb_joints_belief__20260128_013222",
        ],
        # aib curve == cvae-v2 (red label: PPO+AIB (ours))
        "aib_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-cvae-v2-interceptfast-33__33__rgb_joints_belief__20260127_213816",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-cvae-v2-interceptfast-42__42__rgb_joints_belief__20260127_232947",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense/InterceptFast-v0/ppo-cvae-v2-interceptfast-99__99__rgb_joints_belief__20260128_011917",
        ],
        # nonbelief curve == PPO (green label: PPO)
        "nonbelief_paths": [
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints/normalized_dense/InterceptFast-v0/ppo-nonbelief-newhyper-lr1e-4-33__33__rgb_joints__20260127_202735",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints/normalized_dense/InterceptFast-v0/ppo-nonbelief-newhyper-lr1e-4-42__42__rgb_joints__20260127_214546",
            "/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints/normalized_dense/InterceptFast-v0/ppo-nonbelief-newhyper-lr1e-4-99__99__rgb_joints__20260127_230712",
        ],
    },
}


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
    ax.fill_between(xs, y_low, y_high, color=color, alpha=SHADE_ALPHA)
    ax.plot(xs, y_mean, color=color, linewidth=LINEWIDTH_PT, label=label)


def _parse_csv_list_arg(s: str) -> List[Path]:
    items = [x.strip() for x in s.split(",") if x.strip()]
    return [Path(x) for x in items]


def _apply_preset_to_args(args: argparse.Namespace, preset_name: str) -> List[str]:
    if preset_name not in PRESETS:
        raise SystemExit(f"Unknown preset {preset_name}. Available: {', '.join(sorted(PRESETS.keys()))}")
    ps = PRESETS[preset_name]
    args.task_name = str(ps["task_name"])
    seeds = [str(x) for x in ps["seeds"]]
    args.baseline_csvs = ",".join([str(x) for x in ps.get("baseline_paths", [])])
    args.aib_csvs = ",".join([str(x) for x in ps.get("aib_paths", [])])
    args.nonbelief_csvs = ",".join([str(x) for x in ps.get("nonbelief_paths", [])])
    return seeds


def _run_one_job(
    *,
    args: argparse.Namespace,
    seeds: List[str],
    final_dir: Path,
    cmdline: str,
    preset_name: str,
) -> None:
    """
    Execute one plot job (one task) into a shared final_dir.
    """
    baseline_task_dir = args.baseline_task_dir
    aib_task_dir = args.aib_task_dir or baseline_task_dir
    nonbelief_task_dir = args.nonbelief_task_dir

    baseline_csvs: Dict[str, Path] = {}
    aib_csvs: Dict[str, Path] = {}
    nonbelief_csvs: Dict[str, Path] = {}

    explicit_baseline = _parse_csv_list_arg(args.baseline_csvs)
    explicit_aib = _parse_csv_list_arg(args.aib_csvs)
    explicit_nonbelief = _parse_csv_list_arg(args.nonbelief_csvs)

    if explicit_baseline or explicit_aib or explicit_nonbelief:
        # Preferred path-fed mode (user supplies exact files).
        if explicit_baseline and len(explicit_baseline) != len(seeds):
            raise SystemExit("--baseline-csvs must have same length as --seeds (or be empty).")
        if explicit_aib and len(explicit_aib) != len(seeds):
            raise SystemExit("--aib-csvs must have same length as --seeds (or be empty).")
        if explicit_nonbelief and len(explicit_nonbelief) != len(seeds):
            raise SystemExit("--nonbelief-csvs must have same length as --seeds (or be empty).")

        for i, seed in enumerate(seeds):
            if explicit_baseline:
                baseline_csvs[seed] = resolve_training_metrics_csv(explicit_baseline[i])
            if explicit_aib:
                aib_csvs[seed] = resolve_training_metrics_csv(explicit_aib[i])
            if explicit_nonbelief:
                nonbelief_csvs[seed] = resolve_training_metrics_csv(explicit_nonbelief[i])
        input_mode = "explicit_csvs"
    else:
        input_mode = "glob_search"
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
    has_nonbelief = bool(nb_seeds)

    if not base_seeds or not aib_seeds:
        raise SystemExit(
            f"Missing curves for task={args.task_name}. baseline_seeds={base_seeds}, aib_seeds={aib_seeds}, nonbelief_seeds={nb_seeds}"
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
    report.append(f"task_name: {args.task_name}\n")
    report.append(f"preset: {preset_name or '(none)'}\n")
    report.append(f"requested_seeds: {seeds}\n")
    report.append(f"input_mode: {input_mode}\n")
    report.append(f"baseline_task_dir: {baseline_task_dir}\n")
    report.append(f"aib_task_dir: {aib_task_dir}\n")
    report.append(f"nonbelief_task_dir: {nonbelief_task_dir}\n")
    report.append(f"baseline_glob: {args.baseline_glob}\n")
    report.append(f"aib_glob: {args.aib_glob}\n")
    report.append(f"nonbelief_glob: {args.nonbelief_glob}\n")
    report.append(f"baseline seeds used: {base_seeds}\n")
    report.append(f"AIB seeds used: {aib_seeds}\n")
    report.append(f"nonbelief seeds used: {nb_seeds}\n")
    report.append(f"final_results_dir: {final_dir}\n")
    report.append(f"plot_script: {Path(__file__).resolve()}\n")
    report.append(f"plot_command: {cmdline}\n")
    report.append(f"icml_linewidth_pt: {LINEWIDTH_PT}\n")
    report.append("icml_no_title_default: True (use caption in LaTeX)\n")
    report.append("curve_labels:\n")
    curves: List[CurveSpec] = [
        CurveSpec("aib", "PPO+AIB (ours)", "#d62728", True),
        CurveSpec("baseline", "PPO+Believer", "#1f77b4", True),
        CurveSpec("nonbelief", "PPO", "#2ca02c", (nonbelief_task_dir is not None) or has_nonbelief),
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

        fig, ax = plt.subplots(figsize=(args.figwidth_in, args.figheight_in))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        aib_seed_curves = []
        for s in aib_seeds:
            if ms.mode == "eval":
                x, y_once, y_end = load_cached(aib_csvs[s])
                y = y_once if ms.col == "success_once" else y_end
            else:
                x, y = load_cached_mode_col(aib_csvs[s], ms.mode, ms.col)
            aib_seed_curves.append((x, y))
        plot_mean_minmax(ax=ax, label=curves[0].label, color=curves[0].color, seed_curves=aib_seed_curves)

        base_seed_curves = []
        for s in base_seeds:
            if ms.mode == "eval":
                x, y_once, y_end = load_cached(baseline_csvs[s])
                y = y_once if ms.col == "success_once" else y_end
            else:
                x, y = load_cached_mode_col(baseline_csvs[s], ms.mode, ms.col)
            base_seed_curves.append((x, y))
        plot_mean_minmax(ax=ax, label=curves[1].label, color=curves[1].color, seed_curves=base_seed_curves)

        if ((nonbelief_task_dir is not None) or has_nonbelief) and nb_seeds:
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
        if args.with_title:
            ax.set_title(args.task_name)
        ax.grid(True, alpha=0.3)
        place_legend_avoid_overlap(fig, ax)
        fig.tight_layout()

        out_path = final_dir / f"{args.task_name}_{ms.key}__seeds_{'_'.join(seeds)}.png"
        fig.savefig(out_path, dpi=args.dpi, facecolor="white")
        plt.close(fig)

        report.append(f"out_{ms.key}: {out_path}\n")

    # Write per-task report into the shared final_dir.
    (final_dir / f"{args.task_name}__report.txt").write_text("".join(report))


def main() -> int:
    ap = argparse.ArgumentParser()
    # Input modes:
    #   1) Explicit CSVs (preferred): provide --*-csvs as comma-separated paths aligned with --seeds.
    #   2) Auto-discovery (legacy/compat): provide task dirs + globs (kept for convenience).
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
        "--preset",
        type=str,
        default="",
        help=f"Optional built-in preset name. Available: {', '.join(sorted(PRESETS.keys()))}",
    )
    ap.add_argument(
        "--presets",
        type=str,
        default="",
        help="Run multiple built-in presets in one command (comma-separated preset names).",
    )
    ap.add_argument(
        "--baseline-csvs",
        type=str,
        default="",
        help="Explicit baseline CSVs (comma-separated), aligned with --seeds order. If set, overrides glob search.",
    )
    ap.add_argument(
        "--aib-csvs",
        type=str,
        default="",
        help="Explicit AIB CSVs (comma-separated), aligned with --seeds order. If set, overrides glob search.",
    )
    ap.add_argument(
        "--nonbelief-csvs",
        type=str,
        default="",
        help="Explicit nonbelief CSVs (comma-separated), aligned with --seeds order. If set, overrides glob search.",
    )
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
    ap.add_argument(
        "--final-results-dir",
        type=Path,
        default=None,
        help="If set, write ALL generated images/reports into this single folder (no per-task subfolders).",
    )
    ap.add_argument("--dpi", type=int, default=300, help="Higher dpi improves legibility in papers.")
    ap.add_argument("--xlabel", type=str, default="steps")
    ap.add_argument("--figwidth-in", type=float, default=FIG_WIDTH_IN, help="Figure width in inches.")
    ap.add_argument("--figheight-in", type=float, default=FIG_HEIGHT_IN, help="Figure height in inches.")
    ap.add_argument(
        "--with-title",
        action="store_true",
        default=False,
        help="If set, include a title inside the figure (NOT recommended for ICML; use caption instead).",
    )
    ap.add_argument("--tight-y", action="store_true", default=True)
    ap.add_argument("--y-pad", type=float, default=0.03)
    ap.add_argument("--y-min-span", type=float, default=0.12)
    ap.add_argument("--plot-train-reward", action="store_true", help="Also plot train-mode reward vs steps.")
    args = ap.parse_args()

    # Reproducibility: exact invocation.
    cmdline = " ".join(shlex.quote(x) for x in [os.path.abspath(__file__), *os.sys.argv[1:]])

    # If --final-results-dir is provided, use it; else preserve legacy behavior by creating <out_dir>/final_results.
    if args.final_results_dir is not None:
        final_dir = args.final_results_dir
        final_dir.mkdir(parents=True, exist_ok=True)
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        final_dir = args.out_dir / "final_results"
        final_dir.mkdir(parents=True, exist_ok=True)

    # Multi-preset batch mode.
    if args.presets.strip():
        preset_names = [p.strip() for p in args.presets.split(",") if p.strip()]
        if not preset_names:
            raise SystemExit("Empty --presets")
        (final_dir / "_batch_command.txt").write_text(cmdline + "\n")
        (final_dir / "_batch_presets.txt").write_text("\n".join(preset_names) + "\n")
        for pn in preset_names:
            seeds = _apply_preset_to_args(args, pn)
            _run_one_job(args=args, seeds=seeds, final_dir=final_dir, cmdline=cmdline, preset_name=pn)
        return 0

    # Single-job mode (existing behavior).
    if args.preset:
        seeds = _apply_preset_to_args(args, args.preset)
    else:
        seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
        if not seeds:
            raise SystemExit("Empty --seeds")
    _run_one_job(args=args, seeds=seeds, final_dir=final_dir, cmdline=cmdline, preset_name=(args.preset or ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

