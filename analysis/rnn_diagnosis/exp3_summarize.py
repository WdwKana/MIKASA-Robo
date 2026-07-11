"""Exp 3 summary — memory formation vs success takeoff across training.

Reads the per-checkpoint probe outputs produced by exp3_training_probe.sh and
the run's training_metrics.csv, and plots, against training iteration:
    - cue decodability from h during occlusion (t in [5,10), pure memory)
    - cue decodability from h late in the episode (t in [40,60))
    - eval success_once / success_at_end
If memory is trained via the slow bootstrap loop, decodability should rise
late and track (or lag) success. If decodability rises early while success
stays flat, the bottleneck is readout/control, not memory formation.

Usage: python exp3_summarize.py --root out/exp3_lstm_rc5_s99 --metrics <training_metrics.csv> --out out/exp3_lstm_rc5_s99/summary.png
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    iters, occ, late, s_once, s_end = [], [], [], [], []
    for pj in sorted(glob.glob(os.path.join(a.root, "ckpt_*", "probe", "probe_curves.json"))):
        m = re.search(r"ckpt_\d+_(\d+)", pj)
        it = int(m.group(1))
        with open(pj) as f:
            r = json.load(f)
        h = np.array(r["h"])
        iters.append(it)
        occ.append(h[5:10].mean())
        late.append(h[40:60].mean())
        s_once.append(r["succ_once"])
        s_end.append(r["succ_end"])
    order = np.argsort(iters)
    iters = np.array(iters)[order]
    occ, late = np.array(occ)[order], np.array(late)[order]
    s_once, s_end = np.array(s_once)[order], np.array(s_end)[order]

    ev_it, ev_once = [], []
    with open(a.metrics) as f:
        for row in csv.DictReader(f):
            if row.get("mode") == "eval":
                ev_it.append(float(row["iteration"]))
                ev_once.append(float(row["success_once"]))

    n_cls_chance = json.load(open(sorted(glob.glob(
        os.path.join(a.root, "ckpt_*", "probe", "probe_curves.json")))[0]))["chance"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(iters, occ, "o-", color="#d62728", lw=2, label="decode from h, occlusion t∈[5,10)")
    ax.plot(iters, late, "s--", color="#ff9896", lw=1.6, label="decode from h, late t∈[40,60)")
    ax.axhline(n_cls_chance, color="k", ls=":", lw=1, label="chance")
    ax.plot(ev_it, ev_once, color="#1f77b4", alpha=0.4, lw=1.2, label="eval success_once (train log)")
    ax.plot(iters, s_once, "^-", color="#1f77b4", lw=1.6, ms=5, label="success_once (probe rollouts)")
    ax.plot(iters, s_end, "v-", color="#17becf", lw=1.2, ms=4, label="success_at_end (probe rollouts)")
    ax.set_xlabel("training iteration")
    ax.set_ylabel("accuracy / success")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7.5)
    ax.set_title(os.path.basename(a.root))
    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    print("saved", a.out)


if __name__ == "__main__":
    main()
