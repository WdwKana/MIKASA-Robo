#!/usr/bin/env python3
"""
Plot eval success curves from explicitly provided training_metrics.csv paths.

This is intended for quick one-off plotting when you already have the exact CSVs.

Example:
  python plot_from_explicit_csvs.py --out out.png \\
    /abs/path/to/baseline_seed33/training_metrics.csv \\
    /abs/path/to/baseline_seed42/training_metrics.csv \\
    /abs/path/to/cvae_seed33/training_metrics.csv \\
    /abs/path/to/cvae_seed42/training_metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _infer_seed(p: Path) -> int:
    s = p.as_posix()
    for seed in (33, 42):
        if f"__{seed}__" in s:
            return seed
    raise ValueError(f"Could not infer seed (expected __33__ or __42__) from path: {p}")


def _infer_group(p: Path) -> str:
    s = p.as_posix().lower()
    if "cvae" in s:
        return "cvae (ours)"
    if "baseline" in s:
        return "baseline"
    # fallback: if user passes unknown names, still try to plot
    return "unknown"


def _load_eval(df_path: Path) -> pd.DataFrame:
    df = pd.read_csv(df_path)
    if "mode" in df.columns:
        df = df[df["mode"] == "eval"]
    required = {"total_env_steps", "success_once", "success_at_end"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns {sorted(missing)} in {df_path}")
    # de-dup per step
    out = (
        df.groupby("total_env_steps", as_index=False)[["success_once", "success_at_end"]]
        .mean()
        .sort_values("total_env_steps")
    )
    out["total_env_steps"] = out["total_env_steps"].astype(int)
    return out


def _mean_over_seeds(frames: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    # inner join on steps so we only average when both seeds have the point
    merged = None
    for seed, df in sorted(frames.items(), key=lambda kv: kv[0]):
        sdf = df.rename(
            columns={
                "success_once": f"success_once__{seed}",
                "success_at_end": f"success_at_end__{seed}",
            }
        )
        if merged is None:
            merged = sdf
        else:
            merged = merged.merge(sdf, on="total_env_steps", how="inner")
    assert merged is not None
    merged["success_once_mean"] = merged[[c for c in merged.columns if c.startswith("success_once__")]].mean(axis=1)
    merged["success_at_end_mean"] = merged[[c for c in merged.columns if c.startswith("success_at_end__")]].mean(axis=1)
    return merged[["total_env_steps", "success_once_mean", "success_at_end_mean"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Output png path.")
    ap.add_argument("csvs", nargs="+", help="One or more training_metrics.csv paths.")
    args = ap.parse_args()

    csv_paths = [Path(p).resolve() for p in args.csvs]
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # group -> seed -> df
    grouped: Dict[str, Dict[int, pd.DataFrame]] = {}
    used: List[Tuple[str, int, Path]] = []
    for p in csv_paths:
        group = _infer_group(p)
        seed = _infer_seed(p)
        grouped.setdefault(group, {})[seed] = _load_eval(p)
        used.append((group, seed, p))

    # We expect baseline and cvae (ours), each with both seeds
    series: Dict[str, pd.DataFrame] = {}
    for group, by_seed in grouped.items():
        if 33 in by_seed and 42 in by_seed:
            series[group] = _mean_over_seeds(by_seed)

    if not series:
        raise RuntimeError("No plottable series found (need at least one group with both seeds 33 and 42).")

    # plot: one figure, two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=False)

    for label in ["baseline", "cvae (ours)"]:
        if label not in series:
            continue
        df = series[label]
        ax1.plot(df["total_env_steps"], df["success_once_mean"], linewidth=2.0, label=label)
        ax2.plot(df["total_env_steps"], df["success_at_end_mean"], linewidth=2.0, label=label)

    # plot any other labels too (rare)
    for label, df in series.items():
        if label in ("baseline", "cvae (ours)"):
            continue
        ax1.plot(df["total_env_steps"], df["success_once_mean"], linewidth=2.0, label=label)
        ax2.plot(df["total_env_steps"], df["success_at_end_mean"], linewidth=2.0, label=label)

    ax1.set_title("eval success once (mean over seeds 33 & 42)")
    ax2.set_title("eval success end (mean over seeds 33 & 42)")
    for ax in (ax1, ax2):
        ax.set_xlabel("Steps (total_env_steps, eval)")
        ax.grid(True, alpha=0.25)
    ax1.set_ylabel("Success")

    # shared legend
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    # Write a sidecar manifest so you can verify exactly what was used
    manifest = out_path.with_suffix(".paths.txt")
    lines = ["CSV paths used:"]
    for group, seed, p in sorted(used, key=lambda t: (t[0], t[1])):
        lines.append(f"- {group} seed {seed}: {p.as_posix()}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved plot: {out_path}")
    print(f"Wrote manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

