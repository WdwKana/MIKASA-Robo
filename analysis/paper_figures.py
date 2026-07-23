"""Final paper figures + tables: training curves (mean±std over seeds) and
eval-last3 / max tables for all 14 tasks x {srbtr,gru,lstm,mlp,ffm,shm}.

Config per task family (the paper's final protocol):
  Remember*            -> cres_caps   (aaai_final,  {m}-crescaps-seed*)
  Intercept* (all)     -> plain       (both trees,  {m}-plain-seed*;
                          GrabMedium srbtr = legacy ppo-srbtr-igrabmed-seed*)

Style follows the scientific-figure-making skill (publication rcParams,
semantic palette, dedicated legend panel, pdf+png export).
"""
from __future__ import annotations

import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AF = os.path.join(ROOT, "checkpoints/aaai_final/rgb_joints/normalized_dense")
PM = os.path.join(ROOT, "checkpoints/ppo_memtasks/rgb_joints/normalized_dense")
OUT = os.path.join(ROOT, "final_results/paper_figures")
os.makedirs(OUT, exist_ok=True)

# ── skill palette (semantic: blue=ours, reds/teal=RNN baselines, violet/green=new) ──
PALETTE = {
    "blue_main": "#0F4D92", "red_strong": "#B64342", "teal": "#42949E",
    "neutral": "#8F8E8E", "violet": "#9A4D8E", "green_3": "#8BCF8B",
}
METHODS = ["srbtr", "gru", "lstm", "mlp", "ffm", "shm"]
MCOLOR = {"srbtr": PALETTE["blue_main"], "gru": PALETTE["teal"],
          "lstm": PALETTE["red_strong"], "mlp": PALETTE["neutral"],
          "ffm": PALETTE["violet"], "shm": PALETTE["green_3"]}
MLABEL = {"srbtr": "SRB-TR (ours)", "gru": "GRU", "lstm": "LSTM",
          "mlp": "MLP (no memory)", "ffm": "FFM", "shm": "SHM"}
MLW = {"srbtr": 2.6}          # ours thicker; default 1.8

TASKS = [  # (env_id, short title, config)
    ("RememberColor3-v0", "RememberColor3", "crescaps"),
    ("RememberColor5-v0", "RememberColor5", "crescaps"),
    ("RememberColor9-v0", "RememberColor9", "crescaps"),
    ("RememberShape3-v0", "RememberShape3", "crescaps"),
    ("RememberShape5-v0", "RememberShape5", "crescaps"),
    ("RememberShape9-v0", "RememberShape9", "crescaps"),
    ("RememberShapeAndColor3x2-v0", "ShapeAndColor3x2", "crescaps"),
    ("RememberShapeAndColor3x3-v0", "ShapeAndColor3x3", "crescaps"),
    ("ShellGameTouch-v0", "ShellGameTouch", "cres_caps"),
    ("ShellGamePush-v0", "ShellGamePush", "cres_caps"),
    ("InterceptSlow-v0", "InterceptSlow", "plain"),
    ("InterceptMedium-v0", "InterceptMedium", "plain"),
    ("InterceptFast-v0", "InterceptFast", "plain"),
    ("InterceptGrabSlow-v0", "InterceptGrabSlow", "plain"),
    ("InterceptGrabMedium-v0", "InterceptGrabMedium", "plain"),
    ("InterceptGrabFast-v0", "InterceptGrabFast", "plain"),
]
METRICS = [("success_once", "Success (once)"),
           ("success_at_end", "Success (at end)"),
           ("return", "Return")]


def run_csvs(task: str, method: str, config: str) -> list[str]:
    pats = [f"{AF}/{task}/{method}-{config}-seed*/*/training_metrics.csv",
            f"{PM}/{task}/{method}-{config}-seed*/*/training_metrics.csv"]
    if task == "InterceptGrabMedium-v0" and method == "srbtr" and config == "plain":
        pats.append(f"{PM}/{task}/ppo-srbtr-igrabmed-seed*/*/training_metrics.csv")
    out = []
    for p in pats:
        out += glob.glob(p)
    # dedupe by seed (e.g. Machine-B's NaN-repro duplicate dirs): keep the csv
    # with the most eval rows per seed.
    by_seed: dict[str, str] = {}
    for p in sorted(out):
        seed = p.split("seed")[1].split("_")[0]
        if seed not in by_seed:
            by_seed[seed] = p
        else:
            n_old = sum(1 for r in csv.DictReader(open(by_seed[seed])) if r["mode"] == "eval")
            n_new = sum(1 for r in csv.DictReader(open(p)) if r["mode"] == "eval")
            if n_new > n_old:
                by_seed[seed] = p
    return [by_seed[s] for s in sorted(by_seed)]


def load_eval(path: str):
    rows = [r for r in csv.DictReader(open(path)) if r["mode"] == "eval"]
    if not rows:
        return None
    steps = np.array([float(r["total_env_steps"]) for r in rows])
    data = {k: np.array([float(r[k]) for r in rows]) for k, _ in METRICS}
    return steps, data


def collect(task: str, method: str, config: str):
    """-> dict(metric -> (steps, mean, std)) truncated to min seed length, plus
    per-seed eval3/max stats; None if no complete-ish runs."""
    series = [s for s in (load_eval(p) for p in run_csvs(task, method, config)) if s]
    series = [s for s in series if len(s[0]) >= 4]           # drop crashed stubs
    if not series:
        return None
    L = min(len(s[0]) for s in series)
    curves, stats = {}, {}
    for k, _ in METRICS:
        arr = np.stack([s[1][k][:L] for s in series])         # (n_seeds, L)
        curves[k] = (series[0][0][:L], arr.mean(0), arr.std(0))
        e3 = np.array([s[1][k][-3:].mean() for s in series])  # full-length last3
        mx = np.array([s[1][k].max() for s in series])
        stats[k] = dict(e3_m=e3.mean(), e3_s=e3.std(), mx_m=mx.mean(),
                        mx_s=mx.std(), n=len(series))
    return curves, stats


# ── gather everything ──────────────────────────────────────────────────────
DATA = {}
for env, short, cfg in TASKS:
    for m in METHODS:
        got = collect(env, m, cfg)
        if got:
            DATA[(env, m)] = got

# ── publication style (per skill: spines off, frameless legend) ────────────
plt.rcParams.update({
    "font.family": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
    "font.size": 15, "axes.linewidth": 2.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})

N_PANELS = len(TASKS)                      # 16 -> 3x6 grid + legend + blank
for key, ylab in METRICS:
    fig, axes = plt.subplots(3, 6, figsize=(28, 12.5))
    axes = axes.ravel()
    for i, (env, short, cfg) in enumerate(TASKS):
        ax = axes[i]
        for m in METHODS:
            if (env, m) not in DATA:
                continue
            steps, mean, std = DATA[(env, m)][0][key]
            x = steps / 1e6
            ax.plot(x, mean, color=MCOLOR[m], lw=MLW.get(m, 1.8),
                    label=MLABEL[m], zorder=3 if m == "srbtr" else 2)
            ax.fill_between(x, mean - std, mean + std, color=MCOLOR[m],
                            alpha=0.15, lw=0)
        ax.set_title(short, fontsize=15)
        if i >= N_PANELS - 6:
            ax.set_xlabel("Env steps (M)")
        if i % 6 == 0:
            ax.set_ylabel(ylab)
        if key != "return":
            ax.set_ylim(-0.02, None)
    # dedicated legend panel (skill pattern); blank any remaining slots
    for j in range(N_PANELS, len(axes)):
        axes[j].set_axis_off()
    handles = [Line2D([], [], color=MCOLOR[m], lw=MLW.get(m, 1.8) + 0.6,
                      label=MLABEL[m]) for m in METHODS]
    axes[N_PANELS].legend(handles=handles, loc="center", fontsize=16,
                          handlelength=2.4)
    fig.tight_layout(pad=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/curves_{key}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved curves_{key}.png/.pdf")

# ── tables (eval-last3 and max), markdown ──────────────────────────────────
def fmt(v, s, ret=False):
    return (f"{v:.1f}±{s:.1f}" if ret else f"{v:.3f}±{s:.3f}")

for stat in ("e3", "mx"):
    name = "eval_last3" if stat == "e3" else "max_over_evals"
    lines = [f"# Results table — {name} (seed mean±std)\n"]
    for key, ylab in METRICS:
        lines.append(f"\n## {ylab}\n")
        lines.append("| Task | " + " | ".join(MLABEL[m] for m in METHODS) + " |")
        lines.append("|---|" + "---|" * len(METHODS))
        for env, short, cfg in TASKS:
            cells, vals = [], []
            for m in METHODS:
                if (env, m) not in DATA:
                    cells.append("—"); vals.append(-1); continue
                st = DATA[(env, m)][1][key]
                v, s = st[f"{stat}_m"], st[f"{stat}_s"]
                cells.append(fmt(v, s, ret=(key == "return")))
                vals.append(v)
            best = int(np.argmax(vals))
            if vals[best] >= 0:
                cells[best] = f"**{cells[best]}**"
            lines.append(f"| {short} | " + " | ".join(cells) + " |")
    open(f"{OUT}/table_{name}.md", "w").write("\n".join(lines) + "\n")
    print(f"saved table_{name}.md")

print("\nDONE ->", OUT)
