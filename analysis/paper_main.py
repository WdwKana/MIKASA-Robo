"""Camera-ready main figure + table: 12-task final results.

Figure: 4x3 panel grid of Success training curves (mean±std over seeds),
single top legend, sized for a full-width (figure*) AAAI placement.
Table:  booktabs LaTeX (+ markdown preview), Success as percent, family
blocks, per-family and overall Average rows, ours = last column.

Metric = "Success" := success_once, final value = mean of last-3 evals.
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

PALETTE = {"blue_main": "#0F4D92", "red_strong": "#B64342", "teal": "#42949E",
           "neutral": "#8F8E8E", "violet": "#9A4D8E", "green_3": "#8BCF8B"}
# baselines first (simple -> specialised memory), ours last
METHODS = ["mlp", "gru", "lstm", "ffm", "shm", "srbtr"]
MCOLOR = {"srbtr": PALETTE["blue_main"], "gru": PALETTE["teal"],
          "lstm": PALETTE["red_strong"], "mlp": PALETTE["neutral"],
          "ffm": PALETTE["violet"], "shm": PALETTE["green_3"]}
MLABEL = {"srbtr": "PPO+STRM (ours)", "gru": "PPO+GRU", "lstm": "PPO+LSTM",
          "mlp": "PPO", "ffm": "PPO+FFM", "shm": "PPO+SHM"}
MLW = {"srbtr": 2.8}

# 12 final tasks. Panel/row order = INCREASING MEMORY DEMAND (stated in the
# caption): occlusion tracking -> 3-way -> 5-way -> compositional (6/9 combos)
# -> 9-way discrimination -> dynamic interception. Rows of the 4x3 grid map
# onto that progression, so the ordering is a difficulty axis, not a margin
# ranking.
TASKS = [
    # row 1: occlusion tracking + easiest (3-way) cue recall
    ("ShellGameTouch-v0", "ShellGameTouch", "cres_caps", "man"),
    ("ShellGamePush-v0", "ShellGamePush", "cres_caps", "man"),
    ("RememberColor3-v0", "RememberColor3", "crescaps", "rem"),
    ("RememberShape3-v0", "RememberShape3", "crescaps", "rem"),
    # row 2: 5-way cue recall + compositional recall
    ("RememberColor5-v0", "RememberColor5", "crescaps", "rem"),
    ("RememberShape5-v0", "RememberShape5", "crescaps", "rem"),
    ("RememberShapeAndColor3x2-v0", "ShapeAndColor3x2", "crescaps", "rem"),
    ("RememberShapeAndColor3x3-v0", "ShapeAndColor3x3", "crescaps", "rem"),
    # row 3: hardest (9-way) discrimination + dynamic interception
    ("RememberColor9-v0", "RememberColor9", "crescaps", "rem"),
    ("RememberShape9-v0", "RememberShape9", "crescaps", "rem"),
    ("InterceptGrabFast-v0", "InterceptGrabFast", "plain", "dyn"),
    ("InterceptFast-v0", "InterceptFast", "plain", "dyn"),
]
DIVERGED = {("ShellGameTouch-v0", "shm"), ("ShellGamePush-v0", "shm")}


SEEDS = ("33", "42", "99")   # uniform across every (task, method) cell


def run_csvs(task, method, config):
    pats = [f"{AF}/{task}/{method}-{config}-seed*/*/training_metrics.csv",
            f"{PM}/{task}/{method}-{config}-seed*/*/training_metrics.csv"]
    by_seed = {}
    for p in sorted(sum((glob.glob(x) for x in pats), [])):
        seed = p.split("seed")[1].split("_")[0]
        if seed not in SEEDS:      # extra seeds (e.g. Push 100/123) excluded
            continue
        n = sum(1 for r in csv.DictReader(open(p)) if r["mode"] == "eval")
        if seed not in by_seed or n > by_seed[seed][0]:
            by_seed[seed] = (n, p)
    return [v[1] for _, v in sorted(by_seed.items())]


def load(task, method, config):
    """-> (steps, per-seed success arrays truncated), per-seed eval3 list."""
    seqs = []
    for p in run_csvs(task, method, config):
        rows = [r for r in csv.DictReader(open(p)) if r["mode"] == "eval"]
        if len(rows) < 4:
            continue
        steps = np.array([float(r["total_env_steps"]) for r in rows])
        y = np.array([float(r["success_once"]) for r in rows])
        seqs.append((steps, y))
    if not seqs:
        return None
    L = min(len(s[0]) for s in seqs)
    arr = np.stack([s[1][:L] for s in seqs])
    e3 = np.array([s[1][-3:].mean() for s in seqs])
    return seqs[0][0][:L], arr, e3


DATA = {}
for env, short, cfg, fam in TASKS:
    for m in METHODS:
        got = load(env, m, cfg)
        if got:
            DATA[(env, m)] = got

# ── figure: 4x3, top legend, print-sized fonts ─────────────────────────────
plt.rcParams.update({
    "font.family": ["DejaVu Sans", "Helvetica", "sans-serif"],
    "font.size": 17, "axes.titlesize": 18, "axes.labelsize": 17,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "axes.linewidth": 1.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})
fig, axes = plt.subplots(3, 4, figsize=(16, 10.2), sharex=True)
axes = axes.ravel()
for i, (env, short, cfg, fam) in enumerate(TASKS):
    ax = axes[i]
    for m in METHODS:
        if (env, m) not in DATA:
            continue
        steps, arr, _ = DATA[(env, m)]
        x = steps / 1e6
        mu, sd = arr.mean(0), arr.std(0)
        ax.plot(x, mu, color=MCOLOR[m], lw=MLW.get(m, 1.7),
                zorder=3 if m == "srbtr" else 2)
        ax.fill_between(x, mu - sd, mu + sd, color=MCOLOR[m], alpha=0.13, lw=0)
    ax.set_title(short, pad=5)
    ax.set_xlim(0, 7)
    ax.set_xticks([0, 3.5, 7])
    ax.set_ylim(-0.02, None)
    if i >= 8:
        ax.set_xlabel(r"Env steps ($\times 10^6$)")
    if i % 4 == 0:
        ax.set_ylabel("Success")
handles = [Line2D([], [], color=MCOLOR[m], lw=MLW.get(m, 1.7) + 0.8,
                  label=MLABEL[m]) for m in METHODS]
fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=17,
           handlelength=2.2, bbox_to_anchor=(0.5, 1.045), columnspacing=1.6)
fig.tight_layout(pad=1.0, h_pad=1.4)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/main_success_12.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved main_success_12.png/.pdf")

# ── table: Success (%) mean±std, family blocks, Avg rows, LaTeX + md ───────
def cell_stats(env, m):
    if (env, m) not in DATA:
        return None
    e3 = DATA[(env, m)][2]
    return 100 * e3.mean(), 100 * e3.std(), len(e3)

def fmt_cell(env, m, bold):
    st = cell_stats(env, m)
    if st is None:
        return ("---", "—")
    v, s, _ = st
    dag = r"$^\dagger$" if (env, m) in DIVERGED else ""
    tex = f"{v:.1f}\\,{{\\scriptsize$\\pm${s:.1f}}}{dag}"
    md = f"{v:.1f}±{s:.1f}" + ("†" if (env, m) in DIVERGED else "")
    if bold:
        tex, md = f"\\textbf{{{tex}}}", f"**{md}**"
    return tex, md

def avg_row(tasks_sub, m):
    vals = [cell_stats(env, m) for env, *_ in tasks_sub]
    if any(v is None for v in vals):
        return None
    return float(np.mean([v[0] for v in vals]))

# table lists all 12 tasks in the same order as the figure panels, no blocks.
tex = [r"\begin{table*}[t]", r"\centering", r"\small",
       r"\setlength{\tabcolsep}{5pt}",
       r"\begin{tabular}{l" + "c" * len(METHODS) + "}", r"\toprule",
       "Task & " + " & ".join(("\\textbf{%s}" % MLABEL[m]) if m == "srbtr"
                              else MLABEL[m] for m in METHODS) + r" \\"]
md = ["| Task | " + " | ".join(MLABEL[m] for m in METHODS) + " |",
      "|---|" + "---|" * len(METHODS)]
tex.append(r"\midrule")
for env, short, cfg, _ in TASKS:
    stats = {m: cell_stats(env, m) for m in METHODS}
    best = max((m for m in METHODS if stats[m]), key=lambda m: stats[m][0])
    tcells, mcells = zip(*(fmt_cell(env, m, m == best) for m in METHODS))
    tex.append(f"{short} & " + " & ".join(tcells) + r" \\")
    md.append(f"| {short} | " + " | ".join(mcells) + " |")
tex.append(r"\midrule")
for label, sub in (("Average", TASKS),):
    avgs = {m: avg_row(sub, m) for m in METHODS}
    ok = {m: v for m, v in avgs.items() if v is not None}
    best = max(ok, key=ok.get)
    tcells, mcells = [], []
    for m in METHODS:
        if avgs[m] is None:
            tcells.append("---"); mcells.append("—"); continue
        t, d = f"{avgs[m]:.1f}", f"{avgs[m]:.1f}"
        if m == best:
            t, d = f"\\textbf{{{t}}}", f"**{d}**"
        tcells.append(t); mcells.append(d)
    tex.append(f"\\textit{{{label}}} & " + " & ".join(tcells) + r" \\")
    md.append(f"| *{label}* | " + " | ".join(mcells) + " |")
tex += [r"\bottomrule", r"\end{tabular}",
        r"\caption{...}", r"\label{tab:main}", r"\end{table*}"]
open(f"{OUT}/main_table_12.tex", "w").write("\n".join(tex) + "\n")
open(f"{OUT}/main_table_12.md", "w").write("\n".join(md) + "\n")
print("saved main_table_12.tex / .md")
