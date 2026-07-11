"""Exp 5 — methodological controls for the memory-survival probes.

For each recorded rollout set, at representative timesteps:
  1. SHUFFLE CONTROL (Hewitt-Liang style): probe trained on permuted labels
     must sit at chance, else the CV pipeline leaks.
  2. FAILURE-ONLY probe: decodability restricted to episodes with NO success
     — kills the behavioural-leakage explanation (the agent's own reaching
     cannot betray the answer if it never reaches the right cube).
  3. BOOTSTRAP 95% CIs over episodes for the reported accuracies.
  4. MLP probe (2-layer) alongside the linear one — a chance-level LINEAR
     result could hide non-linearly coded information; chance-level MLP
     makes the absence claim robust.

Usage: python exp5_probe_controls.py --data out/roll_lstm_rc5_s99 --target h --ts 7 20 40
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from exp2b_probe import load_rollouts


def fit_probe(Xtr, ytr, Xte, n_cls, mlp=False, epochs=400, lr=0.05, wd=1e-3):
    device = Xtr.device
    if mlp:
        model = torch.nn.Sequential(
            torch.nn.Linear(Xtr.shape[1], 128), torch.nn.ReLU(),
            torch.nn.Linear(128, n_cls)).to(device)
        params = model.parameters()
    else:
        model = torch.nn.Linear(Xtr.shape[1], n_cls).to(device)
        params = model.parameters()
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(Xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        return model(Xte).argmax(-1)


def cv_acc(X, y, folds=5, mlp=False, seed=0, device="cuda"):
    E = X.shape[0]
    classes = np.unique(y)
    y_idx = np.searchsorted(classes, y)
    perm = np.random.RandomState(seed).permutation(E)
    correct = np.zeros(E, dtype=bool)
    for k in range(folds):
        te = perm[k::folds]
        tr = np.setdiff1d(perm, te)
        Xtr = torch.tensor(X[tr], dtype=torch.float32, device=device)
        Xte = torch.tensor(X[te], dtype=torch.float32, device=device)
        mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
        pred = fit_probe((Xtr - mu) / sd, torch.tensor(y_idx[tr], device=device),
                         (Xte - mu) / sd, len(classes), mlp=mlp)
        correct[te] = pred.cpu().numpy() == y_idx[te]
    return correct  # per-episode correctness -> bootstrap outside


def boot_ci(correct, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    E = len(correct)
    accs = [correct[rng.randint(0, E, E)].mean() for _ in range(n)]
    return float(np.mean(correct)), float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", default="h")
    ap.add_argument("--ts", type=int, nargs="+", default=[7, 20, 40])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = load_rollouts(a.data)
    y = d["color"]
    fail = ~d["succ"].any(axis=0)
    print(f"{a.data}: episodes={len(y)} failures={fail.sum()} chance={1/len(np.unique(y)):.2f}")

    results = {}
    for t in a.ts:
        X = d[a.target][t].astype(np.float32)
        row = {}
        m, lo, hi = boot_ci(cv_acc(X, y, device=device))
        row["linear"] = (round(m, 3), round(lo, 3), round(hi, 3))
        m, lo, hi = boot_ci(cv_acc(X, y, mlp=True, device=device))
        row["mlp"] = (round(m, 3), round(lo, 3), round(hi, 3))
        y_shuf = np.random.RandomState(1).permutation(y)
        m, lo, hi = boot_ci(cv_acc(X, y_shuf, device=device))
        row["shuffle_ctrl"] = (round(m, 3), round(lo, 3), round(hi, 3))
        if fail.sum() >= 40:
            m, lo, hi = boot_ci(cv_acc(X[fail], y[fail], device=device))
            row["failure_only"] = (round(m, 3), round(lo, 3), round(hi, 3))
        results[t] = row
        print(f"t={t:2d} " + "  ".join(f"{k}={v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]" for k, v in row.items()))

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=1)


if __name__ == "__main__":
    main()
