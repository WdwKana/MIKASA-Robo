"""Combined memory-survival figure: recurrent washout vs SRB-TR buffer.

Overlays the per-timestep cue-decodability curves of LSTM h, GRU h, and
SRB-TR (buffer mean + reader output) on RememberColor5, with task phases
shaded. This is the mechanism figure: same perception, same task, same RL —
only the memory substrate differs.

Usage: python exp2c_combined_figure.py --root out --out out/memory_survival_rc5.png
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURVES = [
    ("probe_lstm_rc5_s99", "h", "LSTM hidden state", "#1f77b4", "-"),
    ("probe_gru_rc5_s33", "h", "GRU hidden state", "#2ca02c", "-"),
    ("probe_ffm_rc5_s99", "h", "FFM memory matrix (fixed decay)", "#9467bd", "-"),
    ("probe_srbtr_rc5_s99", "bufmean", "SRB-TR episodic buffer", "#d62728", "-"),
    ("probe_srbtr_rc5_s99", "s", "SRB-TR reader output", "#d62728", "--"),
    ("probe_lstm_rc5_s99", "emb", "instantaneous perception (ceiling/leakage)", "#7f7f7f", ":"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(__file__), "out"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "memory_survival_rc5.png")

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.axvspan(0, 5, color="gold", alpha=0.18)
    ax.axvspan(5, 10, color="gray", alpha=0.15)
    ax.text(2.4, 1.06, "cue", ha="center", fontsize=8)
    ax.text(7.5, 1.06, "occluded", ha="center", fontsize=8)
    ax.text(35, 1.06, "all candidate cubes visible (distractor phase)", ha="center", fontsize=8)

    chance = None
    for dirname, key, label, color, ls in CURVES:
        pj = os.path.join(a.root, dirname, "probe_curves.json")
        if not os.path.exists(pj):
            print("skip (missing):", pj)
            continue
        with open(pj) as f:
            r = json.load(f)
        if key not in r:
            print(f"skip (no key {key}):", pj)
            continue
        chance = r["chance"]
        ax.plot(np.arange(r["T"]), r[key], ls, color=color, lw=2, label=label)

    if chance:
        ax.axhline(chance, color="k", ls=":", lw=1, label=f"chance = {chance:.2f}")
    ax.set_xlabel("episode step t")
    ax.set_ylabel("cue decodability (linear probe, CV acc)")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
