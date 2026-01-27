#!/usr/bin/env python3
"""
Plot 2 belief algorithms with mean + min/max band across a fixed seedset.

Typical use:
  - baseline (ppo+believer)
  - AIB (ppo+AIB (ours))

The script discovers runs under:
  <belief_root>/<task>/**/training_metrics.csv

Selection:
  - Filter by seed (parsed from `__{seed}__` in path string)
  - Filter by run_dir substring for each curve (provided via args)
  - If multiple candidates exist for the same (curve, seed), choose latest timestamp suffix
    `__YYYYMMDD_HHMMSS` on run_dir name.

Plot:
  - x: total_env_steps (label configurable, default: "steps")
  - y: success_once / success_at_end (eval mode only)
  - mean curve with min/max shading across the seeds available for that curve
  - optional tight y-limits (default on) clamped to [0, 1]

Report:
  - writes a report listing exact CSV paths used and last-eval values per seed.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED_RE = re.compile(r"__(\d+)__")
TS_RE = re.compile(r"__(\d{8}_\d{6})$")


def seed_from_str(s: str) -> Optional[str]:
    m = SEED_RE.search(s)
    return m.group(1) if m else None


def parse_ts(s: str) -> str:
    m = TS_RE.search(s)
    return m.group(1) if m else ""


def choose_latest(paths: List[Path]) -> Path:
    return sorted(paths, key=lambda p: parse_ts(p.parent.parent.name))[-1]


def load_eval(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    ev = df[df["mode"] == "eval"].copy().sort_values("total_env_steps")
    return ev


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


@dataclass(frozen=True)
class CurveSpec:
    key: str
    label: str
    color: str
    run_substr: str


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, required=True)
    ap.add_argument(
        "--seedset",
        nargs="+",
        required=True,
        help="Seeds, e.g. 33 42 99",
    )
    ap.add_argument(
        "--belief-root",
        type=Path,
        default=Path("/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense"),
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--baseline-substr", type=str, required=True)
    ap.add_argument("--aib-substr", type=str, required=True)
    ap.add_argument("--xlabel", type=str, default="steps")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--tight-y", action="store_true", default=True)
    ap.add_argument("--y-pad", type=float, default=0.03)
    ap.add_argument("--y-min-span", type=float, default=0.10)
    ap.add_argument("--no-title", action="store_true", default=True)
    args = ap.parse_args()

    seedset = [str(s) for s in args.seedset]
    seed_tag = "_".join(seedset)

    task_dir = args.belief_root / args.task
    all_csvs = list(task_dir.rglob("training_metrics.csv")) if task_dir.exists() else []

    curves = [
        CurveSpec("baseline", "ppo+believer", "#1f77b4", args.baseline_substr.lower()),
        CurveSpec("aib", "ppo+AIB (ours)", "#d62728", args.aib_substr.lower()),
    ]

    used: Dict[str, Dict[str, Path]] = {c.key: {} for c in curves}
    ambiguities: List[str] = []

    for seed in seedset:
        for c in curves:
            cands: List[Path] = []
            for p in all_csvs:
                if seed_from_str(str(p)) != seed:
                    continue
                run_dir = p.relative_to(task_dir).parts[0]
                if c.run_substr not in run_dir.lower():
                    continue
                cands.append(p)
            if not cands:
                continue
            chosen = choose_latest(cands)
            used[c.key][seed] = chosen
            if len(cands) > 1:
                ambiguities.append(f"({c.key}, seed={seed}) chose {chosen} from {len(cands)} candidates")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()
    cmdline = " ".join([shlex.quote(x) for x in [os.fspath(script_path), *os.sys.argv[1:]]])

    report: List[str] = []
    report.append(f"task: {args.task}\n")
    report.append(f"seedset requested: {seedset}\n")
    report.append(f"belief_root: {task_dir}\n")
    report.append(f"out_dir: {args.out_dir}\n")
    report.append(f"script: {script_path}\n")
    report.append(f"command: {cmdline}\n\n")

    cache: Dict[Path, pd.DataFrame] = {}

    def load_cached(p: Path) -> pd.DataFrame:
        if p not in cache:
            cache[p] = load_eval(p)
        return cache[p]

    def dump_curve(c: CurveSpec):
        mp = used[c.key]
        seeds = sorted(mp.keys(), key=int)
        report.append(f"{c.key} ({c.label}) seeds used: {seeds}\n")
        for s in seeds:
            csvp = mp[s]
            ev = load_cached(csvp)
            if len(ev):
                last = ev.iloc[-1]
                last_steps = float(last["total_env_steps"])
                last_once = float(last["success_once"])
                last_end = float(last["success_at_end"])
            else:
                last_steps = float("nan")
                last_once = float("nan")
                last_end = float("nan")
            report.append(f"  seed{s}: {csvp}\n")
            report.append(f"    last_eval: steps={last_steps}, success_once={last_once}, success_at_end={last_end}\n")
        report.append("\n")

    for c in curves:
        dump_curve(c)

    if ambiguities:
        report.append("ambiguities:\n")
        for a in ambiguities:
            report.append(f"  - {a}\n")
        report.append("\n")

    metrics = [("success_once", "success_once"), ("success_at_end", "success_at_end")]

    for metric_key, col in metrics:
        # collect for tight y across plotted data
        all_y: List[np.ndarray] = []
        for c in curves:
            for s in sorted(used[c.key].keys(), key=int):
                ev = load_cached(used[c.key][s])
                all_y.append(ev[col].to_numpy(dtype=float))
        if args.tight_y:
            y0, y1 = tight_ylim(all_y, y_pad=args.y_pad, y_min_span=args.y_min_span, clamp_01=True)
        else:
            y0, y1 = 0.0, 1.0

        plt.figure(figsize=(10.5, 4.2))

        def plot_curve(c: CurveSpec):
            mp = used[c.key]
            seeds = sorted(mp.keys(), key=int)
            if not seeds:
                return
            seed_curves = []
            for s in seeds:
                ev = load_cached(mp[s])
                x = ev["total_env_steps"].to_numpy(dtype=float)
                y = ev[col].to_numpy(dtype=float)
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
            plt.fill_between(xs, y_low, y_high, color=c.color, alpha=0.15)
            plt.plot(xs, y_mean, color=c.color, linewidth=3.0, label=c.label)

        for c in curves:
            plot_curve(c)

        plt.ylim(y0, y1)
        plt.xlabel(args.xlabel)
        plt.ylabel("success")
        if not args.no_title:
            plt.title(args.task)
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")
        plt.tight_layout(rect=(0, 0, 0.82, 1))

        out_path = args.out_dir / f"{args.task}__2curves__{metric_key}__seedset_{seed_tag}.png"
        plt.savefig(out_path, dpi=args.dpi)
        plt.close()

    (args.out_dir / f"{args.task}__2curves__report.txt").write_text("".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

