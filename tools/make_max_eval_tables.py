#!/usr/bin/env python3
"""
Generate ICML-style LaTeX tables for:
  - max eval success_once
  - max eval success_at_end

Computation (per user request):
  - consider the 4 presets stored in `tools/plot_baseline_newhyper_vs_aib_v6_last5.py`,
    excluding color5 by default
  - for each (task, algo, seed): read training_metrics.csv, filter mode=eval,
    compute the maximum value over the entire eval history for:
        - success_once
        - success_at_end
  - per (task, algo): report mean±std over seeds (std uses ddof=0)
  - Total row: pool all (task, seed) maxima across selected tasks (12 values / algo),
    then compute mean±std (ddof=0)
  - LaTeX formatting matches the user's template: abbreviated task names, 2 decimals,
    and bold the best-mean method per row.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _load_plot_module():
    here = Path(__file__).resolve().parent
    plot_path = here / "plot_baseline_newhyper_vs_aib_v6_last5.py"
    spec = importlib.util.spec_from_file_location("plot_baseline_newhyper_vs_aib_v6_last5", str(plot_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {plot_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_PLOT = _load_plot_module()
PRESETS = _PLOT.PRESETS
resolve_training_metrics_csv = _PLOT.resolve_training_metrics_csv


@dataclass(frozen=True)
class AlgoSpec:
    key: str
    col_label: str
    preset_paths_key: str


ALGOS: List[AlgoSpec] = [
    AlgoSpec("aib", "PPO+AIB", "aib_paths"),
    AlgoSpec("baseline", "PPO+Bel", "baseline_paths"),
    AlgoSpec("nonbelief", "PPO", "nonbelief_paths"),
]


TASK_ABBR: Dict[str, str] = {
    "RememberShapeAndColor3x2-v0": "RSC3x2",
    "RememberColor9-v0": "RC9",
    "InterceptMedium-v0": "IMed",
    "InterceptFast-v0": "IFast",
}


def _read_eval_max(csv_path: Path) -> Tuple[float, float]:
    df = pd.read_csv(csv_path)
    ev = df[df["mode"] == "eval"].copy()
    if ev.empty:
        raise RuntimeError(f"No eval rows in {csv_path}")
    max_once = float(np.max(ev["success_once"].to_numpy(float)))
    max_end = float(np.max(ev["success_at_end"].to_numpy(float)))
    return max_once, max_end


def _mean_std(x: List[float]) -> Tuple[float, float]:
    arr = np.asarray(x, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.sqrt(np.var(arr, ddof=0)))
    return mean, std


def _fmt(mean: float, std: float, *, digits: int, bold: bool) -> str:
    f = f"{{:.{digits}f}}"
    core = f"{f.format(mean)}$\\pm${f.format(std)}"
    return f"\\textbf{{{core}}}" if bold else core


def _latex_table(
    *,
    caption: str,
    label: str,
    rows: List[Tuple[str, Dict[str, Tuple[float, float]]]],
    digits: int,
) -> str:
    # rows: (task_abbr, {algo_key: (mean,std)})
    header = (
        "\\begin{table}[t]\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        "  \\begin{center}\n"
        "    \\begin{footnotesize}\n"
        "      \\begin{sc}\n"
        "        \\setlength{\\tabcolsep}{5pt}\n"
        "        \\begin{tabular}{lccc}\n"
        "          \\toprule\n"
        "          Task & PPO+AIB & PPO+Bel & PPO \\\\\n"
        "          \\midrule\n"
    )
    body: List[str] = []
    for task_abbr, stats in rows:
        means = {k: stats[k][0] for k in ["aib", "baseline", "nonbelief"]}
        best = max(means.values())
        body.append(
            "          "
            + f"{task_abbr} & "
            + _fmt(*stats["aib"], digits=digits, bold=(means["aib"] == best))
            + " & "
            + _fmt(*stats["baseline"], digits=digits, bold=(means["baseline"] == best))
            + " & "
            + _fmt(*stats["nonbelief"], digits=digits, bold=(means["nonbelief"] == best))
            + " \\\\\n"
        )
    footer = (
        "          \\midrule\n"
        "          Total  & {TOTAL_AIB} & {TOTAL_BEL} & {TOTAL_PPO} \\\\\n"
        "          \\bottomrule\n"
        "        \\end{tabular}\n"
        "      \\end{sc}\n"
        "    \\end{footnotesize}\n"
        "  \\end{center}\n"
        "  \\vskip -0.1in\n"
        "\\end{table}\n"
    )
    return header + "".join(body) + footer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exclude-presets",
        type=str,
        default="color5_seedset_33_42_99_v6_newhyper",
        help="Comma-separated preset names to exclude.",
    )
    ap.add_argument("--digits", type=int, default=2, help="Decimal digits in LaTeX cells.")
    args = ap.parse_args()

    excluded = {x.strip() for x in args.exclude_presets.split(",") if x.strip()}
    preset_names = [k for k in PRESETS.keys() if k not in excluded]
    if not preset_names:
        raise SystemExit("No presets selected after exclusions.")

    # details[preset][algo_key][seed] = {'csv':..., 'max_once':..., 'max_end':...}
    details: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {}
    # aggregated rows per metric
    rows_once: List[Tuple[str, Dict[str, Tuple[float, float]]]] = []
    rows_end: List[Tuple[str, Dict[str, Tuple[float, float]]]] = []

    pooled_once: Dict[str, List[float]] = {a.key: [] for a in ALGOS}
    pooled_end: Dict[str, List[float]] = {a.key: [] for a in ALGOS}

    for pn in preset_names:
        ps = PRESETS[pn]
        task_name = str(ps["task_name"])
        task_abbr = TASK_ABBR.get(task_name, task_name)
        seeds = [str(s) for s in ps["seeds"]]
        details[pn] = {}

        per_algo_once: Dict[str, Tuple[float, float]] = {}
        per_algo_end: Dict[str, Tuple[float, float]] = {}

        for algo in ALGOS:
            paths = [Path(x) for x in ps.get(algo.preset_paths_key, [])]
            if len(paths) != len(seeds):
                raise SystemExit(f"{pn}: {algo.preset_paths_key} has {len(paths)} paths, expected {len(seeds)}")
            details[pn][algo.key] = {}

            seed_max_once: List[float] = []
            seed_max_end: List[float] = []

            for i, seed in enumerate(seeds):
                csv_path = resolve_training_metrics_csv(paths[i])
                m_once, m_end = _read_eval_max(csv_path)
                details[pn][algo.key][seed] = {
                    "csv": str(csv_path),
                    "task_name": task_name,
                    "task_abbr": task_abbr,
                    "max_success_once": m_once,
                    "max_success_at_end": m_end,
                }
                seed_max_once.append(m_once)
                seed_max_end.append(m_end)
                pooled_once[algo.key].append(m_once)
                pooled_end[algo.key].append(m_end)

            per_algo_once[algo.key] = _mean_std(seed_max_once)
            per_algo_end[algo.key] = _mean_std(seed_max_end)

        rows_once.append((task_abbr, per_algo_once))
        rows_end.append((task_abbr, per_algo_end))

    # Totals
    total_once = {k: _mean_std(pooled_once[k]) for k in pooled_once}
    total_end = {k: _mean_std(pooled_end[k]) for k in pooled_end}
    best_total_once = max(total_once[k][0] for k in ["aib", "baseline", "nonbelief"])
    best_total_end = max(total_end[k][0] for k in ["aib", "baseline", "nonbelief"])

    # Build LaTeX and inject Total row formatting.
    latex_once = _latex_table(
        caption="Eval success (success\\_once). Task names are abbreviated (see Appendix). Results are mean$\\pm$std over seeds.",
        label="tab:eval_success_once_max",
        rows=rows_once,
        digits=args.digits,
    )
    latex_end = _latex_table(
        caption="Eval success (success\\_at\\_end). Task names are abbreviated (see Appendix). Results are mean$\\pm$std over seeds.",
        label="tab:eval_success_end_max",
        rows=rows_end,
        digits=args.digits,
    )

    latex_once = latex_once.replace(
        "{TOTAL_AIB}",
        _fmt(*total_once["aib"], digits=args.digits, bold=(total_once["aib"][0] == best_total_once)),
    ).replace(
        "{TOTAL_BEL}",
        _fmt(*total_once["baseline"], digits=args.digits, bold=(total_once["baseline"][0] == best_total_once)),
    ).replace(
        "{TOTAL_PPO}",
        _fmt(*total_once["nonbelief"], digits=args.digits, bold=(total_once["nonbelief"][0] == best_total_once)),
    )

    latex_end = latex_end.replace(
        "{TOTAL_AIB}",
        _fmt(*total_end["aib"], digits=args.digits, bold=(total_end["aib"][0] == best_total_end)),
    ).replace(
        "{TOTAL_BEL}",
        _fmt(*total_end["baseline"], digits=args.digits, bold=(total_end["baseline"][0] == best_total_end)),
    ).replace(
        "{TOTAL_PPO}",
        _fmt(*total_end["nonbelief"], digits=args.digits, bold=(total_end["nonbelief"][0] == best_total_end)),
    )

    print(latex_once)
    print()
    print(latex_end)
    print()

    # Verification dump (compact)
    print("### VERIFICATION VALUES (per seed max over eval)\n")
    for pn in preset_names:
        task = str(PRESETS[pn]["task_name"])
        abbr = TASK_ABBR.get(task, task)
        print(f"preset={pn} task={task} abbr={abbr}")
        for algo in ALGOS:
            print(f"  algo={algo.key}")
            for seed in sorted(details[pn][algo.key].keys(), key=int):
                rec = details[pn][algo.key][seed]
                print(
                    f"    seed={seed} "
                    f"max_once={rec['max_success_once']:.6f} "
                    f"max_end={rec['max_success_at_end']:.6f} "
                    f"csv={rec['csv']}"
                )
        print()

    def _dump_total(metric: str, total: Dict[str, Tuple[float, float]]):
        print(f"pooled_total metric={metric} (12 values per algo)")
        for k in ["aib", "baseline", "nonbelief"]:
            m, s = total[k]
            print(f"  algo={k} mean={m:.6f} std={s:.6f}")
        print()

    _dump_total("success_once", total_once)
    _dump_total("success_at_end", total_end)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

