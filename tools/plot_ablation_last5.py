#!/usr/bin/env python3
"""
Reproducible plotting for last5 ablation comparisons (belief runs).

Curves (4 algos):
  - baseline                 -> "ppo+believer"
  - cvae (ours)              -> "ppo+AIB"
  - ablation-nonaction/none  -> "ppo+AIB(without action head)"
  - ablation-bc              -> "ppo+AIB (with behavior cloning)"

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


CURVES = [
    Curve("baseline", "ppo+believer", "#1f77b4"),
    Curve("aib", "ppo+AIB", "#d62728"),
    Curve("nonaction", "ppo+AIB(without action head)", "#ff7f0e"),
    Curve("bc", "ppo+AIB (with behavior cloning)", "#9467bd"),
]


def canonical_prefixes(task: str, curve_key: str) -> List[str]:
    raise RuntimeError("Deprecated: use canonical_run_dir_patterns() instead")


def canonical_run_dir_patterns(task: str, curve_key: str, seed: str) -> List[re.Pattern]:
    """
    Return strict, *canonical* run_dir matchers.
    Important: this intentionally does NOT match variants like `ppo-last5-cvae-v7-1-33__...`.
    """
    # (task isn't currently used but kept for future flexibility)
    if curve_key == "baseline":
        return [
            re.compile(rf"^ppo-last5-baseline-{seed}__{seed}__"),
            re.compile(rf"^ppo-baseline-last5-{seed}__{seed}__"),
            re.compile(rf"^baseline-last5-{seed}__{seed}__"),
        ]
    if curve_key == "aib":
        return [
            re.compile(rf"^ppo-last5-cvae-{seed}__{seed}__"),
            re.compile(rf"^ppo-cvae-last5-{seed}__{seed}__"),
            re.compile(rf"^cvae-last5-{seed}__{seed}__"),
        ]
    if curve_key == "nonaction":
        # RememberColor5 uses ablation-none naming; treat as nonaction.
        return [
            re.compile(rf"^ppo-last5-ablation-nonaction-{seed}__{seed}__"),
            re.compile(rf"^ppo-last5-ablation-none-{seed}__{seed}__"),
            re.compile(rf"^ppo-cvae-last5-ablation-nonaction-{seed}__{seed}__"),
            re.compile(rf"^ppo-cvae-last5-ablation-none-{seed}__{seed}__"),
        ]
    if curve_key == "bc":
        return [
            re.compile(rf"^ppo-last5-ablation-bc-{seed}__{seed}__"),
            re.compile(rf"^ppo-cvae-last5-ablation-bc-{seed}__{seed}__"),
        ]
    raise ValueError(curve_key)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--belief-root", type=Path, default=Path("/local/s4176650/MIKASA-Robo/checkpoints/ppo_memtasks/rgb_joints_belief/normalized_dense"))
    ap.add_argument("--out-root", type=Path, default=Path("/local/s4176650/MIKASA-Robo/plots_ablation_last5"))
    ap.add_argument("--task", required=True, help="Env name, e.g. RememberShapeAndColor3x2-v0")
    ap.add_argument("--seedset", required=True, help="Comma-separated seeds, e.g. 33,42,99,100,123")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--xlabel", type=str, default="steps")
    ap.add_argument("--no-title", action="store_true")
    ap.add_argument("--tight-y", action="store_true")
    ap.add_argument("--y-pad", type=float, default=0.03)
    ap.add_argument("--y-min-span", type=float, default=0.12)
    ap.add_argument("--tag", type=str, default="", help="Optional suffix tag for filenames, e.g. seedset_33_42_123")
    args = ap.parse_args()

    task = args.task
    seedset = seedset_from_arg(args.seedset)

    task_dir = args.belief_root / task
    out_dir = args.out_root / task / "last5"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmdline = " ".join(shlex.quote(x) for x in [os.path.abspath(__file__), *os.sys.argv[1:]])

    # Gather CSVs once
    all_csvs = list(task_dir.rglob("training_metrics.csv")) if task_dir.exists() else []

    # Select CSV per (curve, seed)
    used: Dict[str, Dict[str, Path]] = {c.key: {} for c in CURVES}
    ambiguities: List[str] = []

    for seed in seedset:
        for c in CURVES:
            cands: List[Path] = []
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
            for c in CURVES:
                for s, p in used[c.key].items():
                    x, y_once, y_end = load_cached(p)
                    all_y.append(y_once if metric == "success_once" else y_end)
            y0, y1 = tight_ylim(all_y)
        else:
            y0, y1 = 0.0, 1.0

        plt.figure(figsize=(10.5, 4.2))

        for c in CURVES:
            seeds = sorted(used[c.key].keys(), key=int)
            if not seeds:
                continue
            seed_curves = []
            for s in seeds:
                x, y_once, y_end = load_cached(used[c.key][s])
                y = y_once if metric == "success_once" else y_end
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

        plt.ylim(y0, y1)
        plt.xlabel(args.xlabel)
        plt.ylabel("success")
        if not args.no_title:
            plt.title(task)
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")
        plt.tight_layout(rect=(0, 0, 0.82, 1))

        suffix = f"__{args.tag}" if args.tag else ""
        out_path = out_dir / f"{task}__last5__{metric}{suffix}__4algos.png"
        plt.savefig(out_path, dpi=args.dpi)
        plt.close()

    plot_metric("success_once")
    plot_metric("success_at_end")

    # report
    rep_lines: List[str] = []
    rep_lines.append(f"### Used paths for {task} last5 4-way ablation plot\n")
    rep_lines.append(f"belief_root: {args.belief_root}\n")
    rep_lines.append(f"seedset_requested: {seedset}\n")
    rep_lines.append(f"tag: {args.tag}\n")
    rep_lines.append(f"plot_script: {Path(__file__).resolve()}\n")
    rep_lines.append(f"plot_command: {cmdline}\n")
    rep_lines.append("curve_labels:\n")
    for c in CURVES:
        rep_lines.append(f"  - {c.key}: {c.label}\n")
    rep_lines.append("\n## Used CSVs\n\n")
    for c in CURVES:
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

    rep_name = f"report_paths_and_ambiguities{('__' + args.tag) if args.tag else ''}.txt"
    (out_dir / rep_name).write_text("".join(rep_lines))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

