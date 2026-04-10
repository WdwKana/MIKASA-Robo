#!/usr/bin/env python3
"""
Build ICML-style LaTeX tables from the run paths stored in
`tools/plot_baseline_newhyper_vs_aib_v6_last5.py`.

Requested behavior (user):
  - focus on 4 tasks (exclude color5 preset)
  - for each (task, algo, seed), read training_metrics.csv, filter mode=eval,
    take the last 3 eval points of:
        - success_once
        - success_at_end
  - per seed: compute mean/var over the last-3 points (record these)
  - per (task, algo): aggregate across 3 seeds:
        - mean across seeds of (seed_mean_last3)
        - variance across seeds of (seed_mean_last3)
    We also report std (=sqrt(var)) for the "mean ± std" formatting in tables.
  - produce two tables: one for success_once, one for success_at_end
  - compute gain of AIB over the best non-AIB method for each task and overall.

Notes:
  - "variance" uses population variance (ddof=0), since user asked for 方差.
  - "overall" pools all (task, seed) points (12 values) per algorithm.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import importlib.util
import sys


def _load_plot_module():
    """
    `tools/` is not necessarily a Python package, so avoid `from tools.xxx import ...`.
    Load `plot_baseline_newhyper_vs_aib_v6_last5.py` by path.
    """
    here = Path(__file__).resolve().parent
    plot_path = here / "plot_baseline_newhyper_vs_aib_v6_last5.py"
    if not plot_path.exists():
        raise FileNotFoundError(f"Expected plot script at: {plot_path}")
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
    label: str
    preset_paths_key: str


ALGOS: List[AlgoSpec] = [
    AlgoSpec("aib", "PPO+AIB (ours)", "aib_paths"),
    AlgoSpec("baseline", "PPO+Believer", "baseline_paths"),
    AlgoSpec("nonbelief", "PPO", "nonbelief_paths"),
]


def _read_last3_eval(csv_path: Path) -> Tuple[List[float], List[float]]:
    df = pd.read_csv(csv_path)
    ev = df[df["mode"] == "eval"].copy().sort_values("total_env_steps")
    tail = ev.tail(3)
    if len(tail) < 3:
        raise RuntimeError(f"Need >=3 eval rows in {csv_path}, got {len(tail)}")
    once = [float(x) for x in tail["success_once"].to_list()]
    end = [float(x) for x in tail["success_at_end"].to_list()]
    return once, end


def _mean_var(x: List[float]) -> Tuple[float, float]:
    arr = np.asarray(x, dtype=float)
    return float(np.mean(arr)), float(np.var(arr, ddof=0))


def _fmt_mean_pm_std(mean: float, var: float, *, digits: int) -> str:
    std = math.sqrt(max(0.0, var))
    f = f"{{:.{digits}f}}"
    return f"{f.format(mean)}$\\pm$ {f.format(std)}"


def _latex_table(
    *,
    caption: str,
    label: str,
    rows: List[Tuple[str, Dict[str, Tuple[float, float]], float]],
    digits: int,
) -> str:
    # rows: (task_name, {algo_key: (mean,var)}, gain_abs)
    header = (
        "\\begin{table}[t]\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        "  \\begin{center}\n"
        "    \\begin{small}\n"
        "      \\begin{sc}\n"
        "        \\begin{tabular}{lcccc}\n"
        "          \\toprule\n"
        "          Task & PPO+AIB (ours) & PPO+Believer & PPO & $\\Delta$ vs Best \\\\\n"
        "          \\midrule\n"
    )
    body_lines: List[str] = []
    for task, stats_by_algo, gain_abs in rows:
        aib = _fmt_mean_pm_std(*stats_by_algo["aib"], digits=digits)
        base = _fmt_mean_pm_std(*stats_by_algo["baseline"], digits=digits)
        ppo = _fmt_mean_pm_std(*stats_by_algo["nonbelief"], digits=digits)
        body_lines.append(
            f"          {task} & {aib} & {base} & {ppo} & {gain_abs:.{digits}f} \\\\\n"
        )
    footer = (
        "          \\bottomrule\n"
        "        \\end{tabular}\n"
        "      \\end{sc}\n"
        "    \\end{small}\n"
        "  \\end{center}\n"
        "  \\vskip -0.1in\n"
        "\\end{table}\n"
    )
    return header + "".join(body_lines) + footer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exclude-presets",
        type=str,
        default="color5_seedset_33_42_99_v6_newhyper",
        help="Comma-separated preset names to exclude.",
    )
    ap.add_argument(
        "--digits",
        type=int,
        default=3,
        help="Decimal digits for table and deltas.",
    )
    ap.add_argument(
        "--print-json",
        action="store_true",
        help="Also print full intermediate values as JSON for verification.",
    )
    args = ap.parse_args()

    excluded = {x.strip() for x in args.exclude_presets.split(",") if x.strip()}
    preset_names = [k for k in PRESETS.keys() if k not in excluded]
    if not preset_names:
        raise SystemExit("No presets selected after exclusions.")

    # Intermediate details:
    # details[preset][algo_key][seed] = {...}
    details: Dict[str, Dict[str, Dict[str, object]]] = {}

    # Aggregated:
    # agg[preset][metric_key][algo_key] = {mean_seedmeans, var_seedmeans}
    agg: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    # Pooled totals (across all selected presets): per algo gather seed_mean_last3 values
    pooled_seed_means: Dict[str, Dict[str, List[float]]] = {
        "success_once": {a.key: [] for a in ALGOS},
        "success_at_end": {a.key: [] for a in ALGOS},
    }

    for pn in preset_names:
        ps = PRESETS[pn]
        task_name = str(ps["task_name"])
        seeds = [str(s) for s in ps["seeds"]]
        details[pn] = {}
        agg[pn] = {"success_once": {}, "success_at_end": {}}

        for algo in ALGOS:
            paths = [Path(x) for x in ps.get(algo.preset_paths_key, [])]
            if len(paths) != len(seeds):
                raise SystemExit(f"{pn}: {algo.preset_paths_key} has {len(paths)} paths, expected {len(seeds)}")

            details[pn][algo.key] = {}

            seed_means_once: List[float] = []
            seed_means_end: List[float] = []

            for i, seed in enumerate(seeds):
                csv_path = resolve_training_metrics_csv(paths[i])
                last3_once, last3_end = _read_last3_eval(csv_path)
                m_once, v_once = _mean_var(last3_once)
                m_end, v_end = _mean_var(last3_end)

                details[pn][algo.key][seed] = {
                    "task_name": task_name,
                    "csv": str(csv_path),
                    "last3": {
                        "success_once": last3_once,
                        "success_at_end": last3_end,
                    },
                    "seed_mean_last3": {
                        "success_once": m_once,
                        "success_at_end": m_end,
                    },
                    "seed_var_last3": {
                        "success_once": v_once,
                        "success_at_end": v_end,
                    },
                }

                seed_means_once.append(m_once)
                seed_means_end.append(m_end)
                pooled_seed_means["success_once"][algo.key].append(m_once)
                pooled_seed_means["success_at_end"][algo.key].append(m_end)

            # Aggregate across seeds: mean/var of seed_mean_last3
            mean_once, var_once = _mean_var(seed_means_once)
            mean_end, var_end = _mean_var(seed_means_end)
            agg[pn]["success_once"][algo.key] = {"mean": mean_once, "var": var_once}
            agg[pn]["success_at_end"][algo.key] = {"mean": mean_end, "var": var_end}

    def best_other_mean(pn: str, metric: str) -> float:
        m_base = agg[pn][metric]["baseline"]["mean"]
        m_ppo = agg[pn][metric]["nonbelief"]["mean"]
        return max(m_base, m_ppo)

    def gain_abs(pn: str, metric: str) -> float:
        return agg[pn][metric]["aib"]["mean"] - best_other_mean(pn, metric)

    # Build rows for each table.
    once_rows: List[Tuple[str, Dict[str, Tuple[float, float]], float]] = []
    end_rows: List[Tuple[str, Dict[str, Tuple[float, float]], float]] = []

    for pn in preset_names:
        task = str(PRESETS[pn]["task_name"])
        once_stats = {k: (agg[pn]["success_once"][k]["mean"], agg[pn]["success_once"][k]["var"]) for k in ["aib", "baseline", "nonbelief"]}
        end_stats = {k: (agg[pn]["success_at_end"][k]["mean"], agg[pn]["success_at_end"][k]["var"]) for k in ["aib", "baseline", "nonbelief"]}
        once_rows.append((task, once_stats, gain_abs(pn, "success_once")))
        end_rows.append((task, end_stats, gain_abs(pn, "success_at_end")))

    # Pooled overall row (across tasks and seeds).
    pooled_agg: Dict[str, Dict[str, Tuple[float, float]]] = {"success_once": {}, "success_at_end": {}}
    for metric in ["success_once", "success_at_end"]:
        for algo in ["aib", "baseline", "nonbelief"]:
            m, v = _mean_var(pooled_seed_means[metric][algo])
            pooled_agg[metric][algo] = (m, v)

    pooled_best_other_once = max(pooled_agg["success_once"]["baseline"][0], pooled_agg["success_once"]["nonbelief"][0])
    pooled_best_other_end = max(pooled_agg["success_at_end"]["baseline"][0], pooled_agg["success_at_end"]["nonbelief"][0])
    pooled_gain_once = pooled_agg["success_once"]["aib"][0] - pooled_best_other_once
    pooled_gain_end = pooled_agg["success_at_end"]["aib"][0] - pooled_best_other_end

    once_rows.append(("Total", pooled_agg["success_once"], pooled_gain_once))
    end_rows.append(("Total", pooled_agg["success_at_end"], pooled_gain_end))

    latex_once = _latex_table(
        caption="Eval success (success\\_once), mean over last-3 eval points per seed; reported as mean$\\pm$std across seeds.",
        label="tab:eval_success_once_last3",
        rows=once_rows,
        digits=args.digits,
    )
    latex_end = _latex_table(
        caption="Eval success (success\\_at\\_end), mean over last-3 eval points per seed; reported as mean$\\pm$std across seeds.",
        label="tab:eval_success_end_last3",
        rows=end_rows,
        digits=args.digits,
    )

    print(latex_once)
    print()
    print(latex_end)
    print()

    # Print numeric summary in a compact, grep-able form.
    print("### NUMERIC SUMMARY (means/vars over seeds of seed_mean_last3; var uses ddof=0)\n")
    for pn in preset_names:
        task = str(PRESETS[pn]["task_name"])
        print(f"preset={pn} task={task}")
        for metric in ["success_once", "success_at_end"]:
            ga = gain_abs(pn, metric)
            bo = best_other_mean(pn, metric)
            print(f"  metric={metric} gain_abs={ga:.{args.digits}f} best_other_mean={bo:.{args.digits}f}")
            for algo in ["aib", "baseline", "nonbelief"]:
                m = agg[pn][metric][algo]["mean"]
                v = agg[pn][metric][algo]["var"]
                print(f"    algo={algo} mean={m:.{args.digits}f} var={v:.{args.digits}f}")
        print()

    print("pooled_total (across all selected tasks and seeds; 12 values per algo)")
    for metric in ["success_once", "success_at_end"]:
        if metric == "success_once":
            gain = pooled_gain_once
            best_other = pooled_best_other_once
        else:
            gain = pooled_gain_end
            best_other = pooled_best_other_end
        print(f"  metric={metric} gain_abs={gain:.{args.digits}f} best_other_mean={best_other:.{args.digits}f}")
        for algo in ["aib", "baseline", "nonbelief"]:
            m, v = pooled_agg[metric][algo]
            print(f"    algo={algo} mean={m:.{args.digits}f} var={v:.{args.digits}f}")
    print()

    if args.print_json:
        print("### FULL DETAILS JSON\n")
        print(json.dumps({"selected_presets": preset_names, "details": details, "agg": agg, "pooled_seed_means": pooled_seed_means}, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

