#!/usr/bin/env python3
"""
Reproducible plotting for MIKASA-Robo memtasks.

This script generates last5 seedset comparison plots for three curves:
  - baseline belief  -> label "ppo+believer"
  - cVAE belief      -> label "ppo+AIB (ours)"
  - nonbelief PPO    -> label "ppo"

It reads `training_metrics.csv`, filters `mode == "eval"`, uses `total_env_steps` as x,
and plots mean curve with min/max shading across the selected seeds for each curve.

Selection rules (by default):
  - belief baseline/cvae: only "canonical" run directory names (no extra tags like mix/v*/ablation)
  - nonbelief: any run with matching seed; if multiple candidates, prefer run_dir containing "nonbelief",
    then choose latest timestamp suffix `__YYYYMMDD_HHMMSS` when present.

It writes per-seedset `report_paths.txt` including:
  - used seeds per curve
  - exact CSV paths used
  - best-effort pointers to run metadata files (command/config/log) if present
  - script path and invocation command line for reproducibility
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SEED_RE = re.compile(r"__(\d+)__")
TS_RE = re.compile(r"__(\d{8}_\d{6})$")


def parse_ts(s: str) -> str:
    m = TS_RE.search(s)
    return m.group(1) if m else ""


def seed_from_str(s: str) -> Optional[str]:
    m = SEED_RE.search(s)
    return m.group(1) if m else None


def canonical_baseline_re(seed: str) -> re.Pattern:
    return re.compile(rf"^(?:ppo-last5-baseline|ppo-baseline-last5|baseline-last5)-{seed}__{seed}__")


def canonical_cvae_re(seed: str) -> re.Pattern:
    return re.compile(rf"^(?:ppo-last5-cvae|ppo-cvae-last5|cvae-last5)-{seed}__{seed}__")


def load_eval(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    ev = df[df["mode"] == "eval"].copy().sort_values("total_env_steps")
    x = ev["total_env_steps"].to_numpy(float)
    return x, ev["success_once"].to_numpy(float), ev["success_at_end"].to_numpy(float)


def best_effort_metadata_files(run_root: Path) -> List[Path]:
    """
    Try to locate typical metadata files to reproduce the training run.
    We don't parse them; we just point to them if they exist.
    """
    candidates = [
        "command.txt",
        "cmd.txt",
        "args.txt",
        "run_args.txt",
        "config.json",
        "config.yaml",
        "config.yml",
        "params.json",
        "params.yaml",
        "log.txt",
        "stdout.txt",
        "stderr.txt",
    ]
    found: List[Path] = []
    for name in candidates:
        p = run_root / name
        if p.exists():
            found.append(p)
    return found


def choose_latest(paths: List[Path]) -> Path:
    if len(paths) == 1:
        return paths[0]
    return sorted(paths, key=lambda p: parse_ts(p.parent.parent.name))[-1]


def choose_nonbelief(paths: List[Path]) -> Path:
    # Prefer run_dir containing 'nonbelief', then latest timestamp.
    # NOTE: Some runs may be incomplete (e.g., only step=0 logged). Prefer the run
    # that actually progressed farthest, using max total_env_steps among eval rows.
    def key(p: Path):
        run_dir = p.parent.parent.name
        has_nb = 1 if "nonbelief" in run_dir.lower() else 0
        ts = parse_ts(run_dir)
        try:
            df = pd.read_csv(p, usecols=["mode", "total_env_steps"])
            ev = df[df["mode"] == "eval"]
            max_steps = float(ev["total_env_steps"].max()) if len(ev) else -1.0
            n_eval = int(len(ev))
        except Exception:
            max_steps = -1.0
            n_eval = -1
        return (has_nb, max_steps, n_eval, ts)

    return max(paths, key=key)


@dataclass(frozen=True)
class CurveSpec:
    key: str
    label: str
    color: str


CURVES: List[CurveSpec] = [
    CurveSpec("baseline", "ppo+believer", "#1f77b4"),
    CurveSpec("aib", "ppo+AIB (ours)", "#d62728"),
    CurveSpec("nonbelief", "ppo", "#2ca02c"),
]


def parse_seedset_list(seedsets: List[str]) -> List[List[str]]:
    out: List[List[str]] = []
    for s in seedsets:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if not parts:
            raise ValueError(f"Empty seedset: {s}")
        out.append(parts)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--belief-root", type=Path, default=Path("/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense"))
    ap.add_argument("--nonbelief-root", type=Path, default=Path("/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints/normalized_dense"))
    ap.add_argument("--out-root", type=Path, default=Path("/local/s4176650/MIKASA-Robo/plots_last5_seedsets"))
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--seedset", action="append", required=True, help="Comma-separated seeds, e.g. 33,42,99,100,123. Can be provided multiple times.")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--xlabel", type=str, default="steps")
    ap.add_argument("--no-title", action="store_true", help="Do not set a plot title.")
    ap.add_argument("--tight-y", action="store_true", help="Use tight y-limits computed from plotted data.")
    ap.add_argument("--y-pad", type=float, default=0.03)
    ap.add_argument("--y-min-span", type=float, default=0.12)
    args = ap.parse_args()

    seedsets = parse_seedset_list(args.seedset)
    args.out_root.mkdir(parents=True, exist_ok=True)

    # For reproducibility: store the exact command line used.
    cmdline = " ".join(shlex.quote(x) for x in [os.path.abspath(__file__), *os.sys.argv[1:]])

    metrics = [("success_once", "success_once"), ("success_at_end", "success_at_end")]

    cache: Dict[Path, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def load_cached(p: Path):
        if p not in cache:
            cache[p] = load_eval(p)
        return cache[p]

    for seedset in seedsets:
        seed_tag = "_".join(seedset)
        out_dir = args.out_root / seed_tag
        out_dir.mkdir(parents=True, exist_ok=True)

        report: List[str] = []
        report.append(f"### last5 plots seed_set={seedset}\n")
        report.append(f"belief_root: {args.belief_root}\n")
        report.append(f"nonbelief_root: {args.nonbelief_root}\n")
        report.append(f"out_dir: {out_dir}\n")
        report.append(f"plot_script: {Path(__file__).resolve()}\n")
        report.append(f"plot_command: {cmdline}\n")
        report.append("curve_labels:\n")
        for c in CURVES:
            report.append(f"  - {c.key}: {c.label}\n")
        report.append("\n")

        for task in args.tasks:
            report.append(f"## TASK {task}\n")
            btask = args.belief_root / task
            ntask = args.nonbelief_root / task

            # Collect candidate CSVs per curve per seed
            base_paths: Dict[str, Path] = {}
            aib_paths: Dict[str, Path] = {}
            nb_paths: Dict[str, Path] = {}

            # Pre-list all training_metrics.csv for speed
            belief_csvs = list(btask.rglob("training_metrics.csv")) if btask.exists() else []
            nb_csvs = list(ntask.rglob("training_metrics.csv")) if ntask.exists() else []

            for seed in seedset:
                # baseline belief (canonical naming only)
                base_cands: List[Path] = []
                base_pat = canonical_baseline_re(seed)
                for p in belief_csvs:
                    run_dir = p.relative_to(btask).parts[0]
                    if base_pat.match(run_dir):
                        base_cands.append(p)
                if base_cands:
                    base_paths[seed] = choose_latest(base_cands)

                # AIB belief (canonical cvae naming only)
                aib_cands: List[Path] = []
                aib_pat = canonical_cvae_re(seed)
                for p in belief_csvs:
                    run_dir = p.relative_to(btask).parts[0]
                    if aib_pat.match(run_dir):
                        aib_cands.append(p)
                if aib_cands:
                    aib_paths[seed] = choose_latest(aib_cands)

                # nonbelief: any matching seed; choose best
                nb_cands: List[Path] = []
                for p in nb_csvs:
                    s = seed_from_str(str(p))
                    if s == seed:
                        nb_cands.append(p)
                if nb_cands:
                    nb_paths[seed] = choose_nonbelief(nb_cands)

            base_seeds = sorted(base_paths.keys(), key=int)
            aib_seeds = sorted(aib_paths.keys(), key=int)
            nb_seeds = sorted(nb_paths.keys(), key=int)

            report.append(f"  baseline seeds used: {base_seeds}\n")
            report.append(f"  AIB seeds used: {aib_seeds}\n")
            report.append(f"  nonbelief seeds used: {nb_seeds}\n")

            def dump_paths(name: str, mp: Dict[str, Path]):
                for s in sorted(mp.keys(), key=int):
                    csvp = mp[s]
                    report.append(f"    {name} seed{s}: {csvp}\n")
                    # Attempt to locate run metadata files
                    ts_dir = csvp.parent  # .../<timestamp>/
                    run_dir = csvp.parents[1]  # .../<run_name>/
                    report.append(f"      run_dir: {run_dir}\n")
                    report.append(f"      ts_dir: {ts_dir}\n")
                    # Prefer metadata in run_dir, but also check ts_dir (some setups dump logs there)
                    md = best_effort_metadata_files(run_dir) or best_effort_metadata_files(ts_dir)
                    if md:
                        report.append("      metadata_files:\n")
                        for m in md:
                            report.append(f"        - {m}\n")

            dump_paths("baseline", base_paths)
            dump_paths("AIB", aib_paths)
            dump_paths("nonbelief", nb_paths)

            if not base_seeds or not aib_seeds:
                report.append("  WARNING: baseline or AIB missing entirely; skipped plotting this task.\n\n")
                continue

            for metric, ycol in metrics:
                # Compute tight y-limits from available plotted data
                if args.tight_y:
                    all_y: List[np.ndarray] = []
                    for s in base_seeds:
                        x, y1, y2 = load_cached(base_paths[s])
                        all_y.append(y1 if ycol == "success_once" else y2)
                    for s in aib_seeds:
                        x, y1, y2 = load_cached(aib_paths[s])
                        all_y.append(y1 if ycol == "success_once" else y2)
                    for s in nb_seeds:
                        x, y1, y2 = load_cached(nb_paths[s])
                        all_y.append(y1 if ycol == "success_once" else y2)
                    yy = np.concatenate(all_y) if all_y else np.array([0.0, 1.0])
                    yy = yy[~np.isnan(yy)]
                    y_min = float(np.min(yy))
                    y_max = float(np.max(yy))
                    if (y_max - y_min) < args.y_min_span:
                        mid = 0.5 * (y_min + y_max)
                        y_min = mid - 0.5 * args.y_min_span
                        y_max = mid + 0.5 * args.y_min_span
                    y0 = max(0.0, y_min - args.y_pad)
                    y1 = min(1.0, y_max + args.y_pad)
                else:
                    y0, y1 = 0.0, 1.0

                plt.figure(figsize=(10.5, 4.2))

                def plot_curve(label: str, color: str, seeds: List[str], paths: Dict[str, Path]):
                    if not seeds:
                        return
                    seed_curves = []
                    for s in seeds:
                        x, y_once, y_end = load_cached(paths[s])
                        y = y_once if ycol == "success_once" else y_end
                        seed_curves.append((x, y))
                    xs = sorted(set(np.concatenate([x for x, _ in seed_curves]).tolist()))
                    xs = np.array(xs, dtype=float)
                    Ys = []
                    for x, y in seed_curves:
                        m = dict(zip(x.tolist(), y.tolist()))
                        Ys.append(np.array([m.get(float(xx), np.nan) for xx in xs], dtype=float))
                    Y = np.vstack(Ys)
                    y_mean = np.nanmean(Y, axis=0)
                    y_low = np.nanmin(Y, axis=0)
                    y_high = np.nanmax(Y, axis=0)
                    plt.fill_between(xs, y_low, y_high, color=color, alpha=0.15)
                    plt.plot(xs, y_mean, color=color, linewidth=3.0, label=label)

                # baseline + AIB must use requested seeds (they exist if present in base_paths/aib_paths)
                plot_curve(CURVES[0].label, CURVES[0].color, base_seeds, base_paths)
                plot_curve(CURVES[1].label, CURVES[1].color, aib_seeds, aib_paths)
                plot_curve(CURVES[2].label, CURVES[2].color, nb_seeds, nb_paths)

                plt.ylim(y0, y1)
                plt.xlabel(args.xlabel)
                plt.ylabel("success")
                if not args.no_title:
                    plt.title(task)
                plt.grid(True, alpha=0.3)
                plt.legend(frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")
                plt.tight_layout(rect=(0, 0, 0.82, 1))

                out_path = out_dir / f"{task}__last5__{ycol}__seedset_{seed_tag}.png"
                plt.savefig(out_path, dpi=args.dpi)
                plt.close()

            report.append("\n")

        (out_dir / "report_paths.txt").write_text("".join(report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

