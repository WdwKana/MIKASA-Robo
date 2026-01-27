#!/usr/bin/env python3
"""
Reproducible plotting for Intercept* memtasks (3 curves, fixed seedset).

Curves (3 algos):
  - believer baseline -> label "ppo+believer"  (run_dir contains "ppo-belief")
  - AIB (ours)       -> label "ppo+AIB (ours)" (run_dir contains "ppo-cvae" OR "ppo-cave")
  - nonbelief PPO    -> label "ppo"            (run_dir contains "ppo-nonbelief")

Data:
  - reads `training_metrics.csv`
  - filters `mode == "eval"`
  - x-axis: `total_env_steps` (label configurable, default: "steps")
  - y-axis: success_once / success_at_end
  - plots mean curve with min/max shading across the selected seeds for each curve

Seed handling:
  - You pass a seedset (e.g. 33 42 99)
  - Each curve uses the subset of that seedset that exists for that curve

Run selection:
  - If multiple candidates exist for the same (curve, seed), prefers latest timestamp suffix
    `__YYYYMMDD_HHMMSS` when present.
  - For nonbelief, additionally prefers the run that progressed farthest (max eval total_env_steps),
    to avoid incomplete runs that only logged step=0.

Outputs:
  - Plots written under: <out-root>/<task>/intercept_3curves_seedset_<seeds>/
  - A per-run report listing exact CSV paths used, plus best-effort run metadata pointers.
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


SEED_RE = re.compile(r"__(\d+)__")
TS_RE = re.compile(r"__(\d{8}_\d{6})$")


def parse_ts(s: str) -> str:
    m = TS_RE.search(s)
    return m.group(1) if m else ""


def seed_from_str(s: str) -> Optional[str]:
    m = SEED_RE.search(s)
    return m.group(1) if m else None


def choose_latest(paths: List[Path]) -> Path:
    # training_metrics.csv is .../<run_dir>/<ts_dir>/training_metrics.csv
    # Prefer latest run_dir timestamp.
    return sorted(paths, key=lambda p: parse_ts(p.parent.parent.name))[-1]


def choose_nonbelief(paths: List[Path]) -> Path:
    # Prefer run_dir containing 'nonbelief', then the run that progressed farthest (eval max steps),
    # then latest timestamp.
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


def load_eval(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    df = df[df["mode"] == "eval"].copy()
    df = df.sort_values("total_env_steps")
    x = df["total_env_steps"].to_numpy(dtype=float)
    y1 = df["success_once"].to_numpy(dtype=float)
    y2 = df["success_at_end"].to_numpy(dtype=float)
    return x, y1, y2


def best_effort_metadata_files(base_dir: Path) -> List[Path]:
    if not base_dir.exists():
        return []
    cands: List[Path] = []
    for pat in [
        "config*.yaml",
        "config*.yml",
        "hydra*.yaml",
        "hydra*.yml",
        "command*.txt",
        "cmd*.txt",
        "*.log",
        "train.log",
        "stdout*.txt",
        "stderr*.txt",
    ]:
        cands.extend(sorted(base_dir.glob(pat)))
    # de-dup while preserving order
    out: List[Path] = []
    seen = set()
    for p in cands:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out[:30]


@dataclass(frozen=True)
class CurveSpec:
    key: str
    label: str
    color: str


CURVES = [
    CurveSpec("baseline", "ppo+believer", "#1f77b4"),
    CurveSpec("aib", "ppo+AIB (ours)", "#d62728"),
    CurveSpec("nonbelief", "ppo", "#2ca02c"),
]


def tight_ylim(
    y_arrays: List[np.ndarray],
    *,
    y_pad: float,
    y_min_span: float,
    clamp_01: bool,
) -> Tuple[float, float]:
    yy = np.concatenate(y_arrays) if y_arrays else np.array([0.0, 1.0])
    yy = np.asarray(yy, dtype=float).reshape(-1)
    yy = yy[~np.isnan(yy)]
    y_min = float(np.min(yy)) if len(yy) else 0.0
    y_max = float(np.max(yy)) if len(yy) else 1.0
    if (y_max - y_min) < y_min_span:
        mid = 0.5 * (y_min + y_max)
        y_min = mid - 0.5 * y_min_span
        y_max = mid + 0.5 * y_min_span
    y0 = y_min - y_pad
    y1 = y_max + y_pad
    if clamp_01:
        y0 = max(0.0, y0)
        y1 = min(1.0, y1)
    return float(y0), float(y1)


def is_baseline_run(run_dir: str) -> bool:
    s = run_dir.lower()
    return "ppo-belief" in s


def is_aib_run(run_dir: str) -> bool:
    s = run_dir.lower()
    return ("ppo-cvae" in s) or ("ppo-cave" in s)


def is_nonbelief_run(run_dir: str) -> bool:
    s = run_dir.lower()
    return "ppo-nonbelief" in s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=["InterceptMedium-v0", "InterceptSlow-v0"],
        help="Tasks to plot (directories under belief/nonbelief roots).",
    )
    ap.add_argument("--seedset", nargs="+", default=["33", "42", "99"])
    ap.add_argument(
        "--belief-root",
        type=Path,
        default=Path("/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense"),
    )
    ap.add_argument(
        "--nonbelief-root",
        type=Path,
        default=Path("/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints/normalized_dense"),
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("/local/s4176650/MIKASA-Robo/plots_single_runs"),
    )
    ap.add_argument("--xlabel", default="steps")
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--tight-y", action="store_true", default=True)
    ap.add_argument("--y-pad", type=float, default=0.03)
    ap.add_argument("--y-min-span", type=float, default=0.10)
    ap.add_argument("--no-title", action="store_true", default=False)
    args = ap.parse_args()

    seedset = [str(s) for s in args.seedset]
    seed_tag = "_".join(seedset)

    script_path = Path(__file__).resolve()
    cmdline = " ".join([shlex.quote(x) for x in [os.fspath(script_path), *os.sys.argv[1:]]])

    for task in args.tasks:
        btask = args.belief_root / task
        ntask = args.nonbelief_root / task
        out_dir = args.out_root / task / f"intercept_3curves_seedset_{seed_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)

        report: List[str] = []
        report.append(f"task: {task}\n")
        report.append(f"seedset requested: {seedset}\n")
        report.append(f"belief_root: {btask}\n")
        report.append(f"nonbelief_root: {ntask}\n")
        report.append(f"out_dir: {out_dir}\n")
        report.append(f"script: {script_path}\n")
        report.append(f"command: {cmdline}\n")
        report.append("\n")

        belief_csvs = list(btask.rglob("training_metrics.csv")) if btask.exists() else []
        nb_csvs = list(ntask.rglob("training_metrics.csv")) if ntask.exists() else []

        used: Dict[str, Dict[str, Path]] = {c.key: {} for c in CURVES}
        ambiguities: List[str] = []

        for seed in seedset:
            # baseline (belief)
            base_cands: List[Path] = []
            for p in belief_csvs:
                run_dir = p.relative_to(btask).parts[0]
                if seed_from_str(str(p)) != seed:
                    continue
                if is_baseline_run(run_dir):
                    base_cands.append(p)
            if base_cands:
                chosen = choose_latest(base_cands)
                used["baseline"][seed] = chosen
                if len(base_cands) > 1:
                    ambiguities.append(f"(baseline, seed={seed}) chose {chosen} from {len(base_cands)} candidates")

            # AIB (belief)
            aib_cands: List[Path] = []
            for p in belief_csvs:
                run_dir = p.relative_to(btask).parts[0]
                if seed_from_str(str(p)) != seed:
                    continue
                if is_aib_run(run_dir):
                    aib_cands.append(p)
            if aib_cands:
                chosen = choose_latest(aib_cands)
                used["aib"][seed] = chosen
                if len(aib_cands) > 1:
                    ambiguities.append(f"(aib, seed={seed}) chose {chosen} from {len(aib_cands)} candidates")

            # nonbelief
            nb_cands: List[Path] = []
            for p in nb_csvs:
                run_dir = p.relative_to(ntask).parts[0]
                if seed_from_str(str(p)) != seed:
                    continue
                if is_nonbelief_run(run_dir):
                    nb_cands.append(p)
            if nb_cands:
                chosen = choose_nonbelief(nb_cands)
                used["nonbelief"][seed] = chosen
                if len(nb_cands) > 1:
                    ambiguities.append(
                        f"(nonbelief, seed={seed}) chose {chosen} from {len(nb_cands)} candidates"
                    )

        # Report exact paths used
        def dump_curve(name: str):
            mp = used[name]
            seeds = sorted(mp.keys(), key=int)
            report.append(f"{name} seeds used: {seeds}\n")
            for s in seeds:
                csvp = mp[s]
                report.append(f"  seed{s}: {csvp}\n")
                ts_dir = csvp.parent
                run_dir = csvp.parents[1]
                report.append(f"    run_dir: {run_dir}\n")
                report.append(f"    ts_dir: {ts_dir}\n")
                md = best_effort_metadata_files(run_dir) or best_effort_metadata_files(ts_dir)
                if md:
                    report.append("    metadata_files:\n")
                    for m in md:
                        report.append(f"      - {m}\n")
            report.append("\n")

        dump_curve("baseline")
        dump_curve("aib")
        dump_curve("nonbelief")

        if ambiguities:
            report.append("ambiguities:\n")
            for a in ambiguities:
                report.append(f"  - {a}\n")
            report.append("\n")

        # Plotting
        metrics = [("success_once", "success_once"), ("success_at_end", "success_at_end")]
        cache: Dict[Path, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        def load_cached(p: Path):
            if p not in cache:
                cache[p] = load_eval(p)
            return cache[p]

        for _, ycol in metrics:
            all_y: List[np.ndarray] = []
            for c in CURVES:
                for s in sorted(used[c.key].keys(), key=int):
                    x, y_once, y_end = load_cached(used[c.key][s])
                    all_y.append(y_once if ycol == "success_once" else y_end)

            if args.tight_y:
                y0, y1 = tight_ylim(all_y, y_pad=args.y_pad, y_min_span=args.y_min_span, clamp_01=True)
            else:
                y0, y1 = 0.0, 1.0

            plt.figure(figsize=(10.5, 4.2))

            def plot_curve(label: str, color: str, paths: Dict[str, Path]):
                seeds = sorted(paths.keys(), key=int)
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

            plot_curve(CURVES[0].label, CURVES[0].color, used["baseline"])
            plot_curve(CURVES[1].label, CURVES[1].color, used["aib"])
            plot_curve(CURVES[2].label, CURVES[2].color, used["nonbelief"])

            plt.ylim(y0, y1)
            plt.xlabel(args.xlabel)
            plt.ylabel("success")
            if not args.no_title:
                plt.title(task)
            plt.grid(True, alpha=0.3)
            plt.legend(frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")
            plt.tight_layout(rect=(0, 0, 0.82, 1))

            out_path = out_dir / f"{task}__intercept_3curves__{ycol}__seedset_{seed_tag}.png"
            plt.savefig(out_path, dpi=args.dpi)
            plt.close()

        (out_dir / f"{task}__intercept_3curves__report.txt").write_text("".join(report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

