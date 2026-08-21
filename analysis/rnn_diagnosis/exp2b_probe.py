"""Exp 2b — per-timestep linear probes over recorded rollouts.

For every timestep t, fits a multinomial logistic probe X_t -> cue color with
k-fold CV over episodes and reports test accuracy. Probing h (and c) tells us
how much cue information the memory carries at time t; probing emb (the
instantaneous encoder output) gives the perception ceiling and exposes
behavioural leakage (late-episode emb decodability = the agent's own pose
betrays the answer, not memory).

Phases for RememberColor: cue visible t in [0,5), occluded [5,10), all cubes
revealed t >= 10.

Usage:
    python exp2b_probe.py --data out/rollouts_lstm_rc5_seed99 --out out/probe_lstm_rc5_seed99
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SKIP_KEYS = {"color", "surp", "npush", "buf_feats", "buf_mask"}


def load_rollouts(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "batch*.npz")))
    assert files, f"no rollout batches under {data_dir}"
    parts = {}
    for f in files:
        d = np.load(f)
        for k in d.files:
            parts.setdefault(k, []).append(d[k])
    out = {}
    for k, xs in parts.items():
        if k == "color":
            out[k] = np.concatenate(xs)          # (E,)
        elif k in SKIP_KEYS:
            continue
        else:
            out[k] = np.concatenate(xs, axis=1)  # episodes on axis 1
    return out


def probe_acc(X: np.ndarray, y: np.ndarray, folds: int = 5, epochs: int = 300,
              lr: float = 0.05, wd: float = 1e-3, device: str = "cuda") -> float:
    """k-fold CV accuracy of a linear softmax probe."""
    E = X.shape[0]
    classes = np.unique(y)
    n_cls = len(classes)
    y_idx = np.searchsorted(classes, y)
    rng = np.random.RandomState(0)
    perm = rng.permutation(E)
    accs = []
    for k in range(folds):
        test = perm[k::folds]
        train = np.setdiff1d(perm, test)
        Xtr = torch.tensor(X[train], dtype=torch.float32, device=device)
        Xte = torch.tensor(X[test], dtype=torch.float32, device=device)
        mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        ytr = torch.tensor(y_idx[train], device=device)
        yte = torch.tensor(y_idx[test], device=device)
        W = torch.zeros(X.shape[1], n_cls, device=device, requires_grad=True)
        b = torch.zeros(n_cls, device=device, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(Xtr @ W + b, ytr)
            loss.backward()
            opt.step()
        with torch.no_grad():
            accs.append(((Xte @ W + b).argmax(-1) == yte).float().mean().item())
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", nargs="+", default=None,
                    help="which arrays to probe (default: all of h, c, emb)")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_rollouts(args.data)
    y = data["color"]
    n_cls = len(np.unique(y))
    chance = 1.0 / n_cls
    first_arr = next(k for k in ("h", "s") if k in data)
    T = data[first_arr].shape[0]
    succ_once = data["succ"].any(axis=0)
    print(f"episodes={len(y)}  classes={n_cls}  T={T}  "
          f"succ_once={succ_once.mean():.2f}  succ_end={data['succ'][-1].mean():.2f}")

    targets = args.targets or [k for k in ("h", "c", "emb", "s", "bufmean", "reader", "y") if k in data]
    results = {"chance": chance, "T": T, "succ_once": float(succ_once.mean()),
               "succ_end": float(data["succ"][-1].mean()), "episodes": int(len(y))}
    for tgt in targets:
        curve = []
        for t in range(T):
            acc = probe_acc(data[tgt][t].astype(np.float32), y,
                            folds=args.folds, device=device)
            curve.append(acc)
            if t % 10 == 0:
                print(f"{tgt} t={t:02d} acc={acc:.3f}", flush=True)
        results[tgt] = curve

    with open(os.path.join(args.out, "probe_curves.json"), "w") as f:
        json.dump(results, f, indent=1)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axvspan(0, 5, color="gold", alpha=0.18, label="cue visible")
    ax.axvspan(5, 10, color="gray", alpha=0.15, label="occluded")
    styles = {"h": ("-", "#d62728"), "c": ("--", "#ff9896"), "emb": ("-", "#1f77b4"),
              "s": ("-", "#d62728"), "bufmean": ("-", "#2ca02c"), "reader": ("--", "#e377c2"),
              "y": ("--", "#9467bd")}
    for tgt in targets:
        ls, col = styles.get(tgt, ("-", None))
        ax.plot(range(T), results[tgt], ls, color=col, lw=2, label=f"probe on {tgt}")
    ax.axhline(chance, color="k", lw=1, ls=":", label=f"chance={chance:.2f}")
    ax.set_xlabel("episode step t")
    ax.set_ylabel("cue-color decode accuracy (CV)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.set_title(os.path.basename(args.data))
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "probe_curves.png"), dpi=140)
    print("saved", os.path.join(args.out, "probe_curves.png"))


if __name__ == "__main__":
    main()
