"""Exp 4 — tracking probe: does the recurrent state integrate recent frames?

Two ridge-regression probes on Intercept rollouts (ball task):

  1. VELOCITY: decode ball linear velocity at time t from h_t vs from emb_t.
     A single frame carries (almost) no velocity; only temporal integration
     does. R2(h) >> R2(emb) = direct evidence the recurrence tracks.
  2. LAG POSITION: decode ball position at time t-k from h_t, for k = 0..12,
     with emb_t as the autocorrelation control. The h-over-emb excess R2 as a
     function of k is the memory horizon for dynamic content — expected to
     decay within a few steps, which for tracking is the CORRECT behaviour
     (stale positions of a moving object are wrong, not useful).

Usage: python exp4_tracking_probe.py --data out/roll_gru_int_s33 --out out/track_gru_int_s33
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


def load(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "batch*.npz")))
    assert files, f"no batches under {data_dir}"
    parts = {}
    for f in files:
        d = np.load(f)
        for k in ("h", "emb", "ball_pos", "ball_vel"):
            parts.setdefault(k, []).append(d[k])
    return {k: np.concatenate(v, axis=1) for k, v in parts.items()}  # (T,E,·)


def ridge_r2(X, Y, folds=5, lam=1e2, device="cuda"):
    """k-fold CV R^2 of ridge regression X -> Y (multi-output, averaged)."""
    E = X.shape[0]
    rng = np.random.RandomState(0)
    perm = rng.permutation(E)
    r2s = []
    for k in range(folds):
        te = perm[k::folds]
        tr = np.setdiff1d(perm, te)
        Xtr = torch.tensor(X[tr], dtype=torch.float32, device=device)
        Xte = torch.tensor(X[te], dtype=torch.float32, device=device)
        Ytr = torch.tensor(Y[tr], dtype=torch.float32, device=device)
        Yte = torch.tensor(Y[te], dtype=torch.float32, device=device)
        mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        ym = Ytr.mean(0, keepdim=True)
        G = Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1], device=device)
        W = torch.linalg.solve(G, Xtr.T @ (Ytr - ym))
        pred = Xte @ W + ym
        ss_res = ((Yte - pred) ** 2).sum(0)
        ss_tot = ((Yte - Yte.mean(0, keepdim=True)) ** 2).sum(0) + 1e-9
        r2s.append((1 - ss_res / ss_tot).mean().item())
    return float(np.mean(r2s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-lag", type=int, default=12)
    ap.add_argument("--t-min", type=int, default=15,
                    help="probe timesteps t >= t_min (skip launch transient)")
    ap.add_argument("--t-max", type=int, default=70)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = load(a.data)
    T, E = d["h"].shape[:2]
    speed = np.linalg.norm(d["ball_vel"], axis=-1)
    print(f"T={T} episodes={E}  mean ball speed t∈[{a.t_min},{min(a.t_max,T)}): "
          f"{speed[a.t_min:a.t_max].mean():.3f} m/s")

    # pool timesteps: each (t, episode) sample is one row
    ts = range(a.t_min, min(a.t_max, T))
    results = {"t_min": a.t_min, "t_max": min(a.t_max, T), "episodes": E}

    # 1) instantaneous velocity from h vs emb
    for src in ("h", "emb"):
        X = np.concatenate([d[src][t].astype(np.float32) for t in ts], axis=0)
        Y = np.concatenate([d["ball_vel"][t] for t in ts], axis=0)
        results[f"vel_r2_{src}"] = ridge_r2(X, Y, device=device)
        print(f"velocity R2 from {src}: {results[f'vel_r2_{src}']:.3f}")

    # 2) lag-position memory curve
    for src in ("h", "emb"):
        curve = []
        for k in range(a.max_lag + 1):
            X = np.concatenate([d[src][t].astype(np.float32) for t in ts if t - k >= 0], axis=0)
            Y = np.concatenate([d["ball_pos"][t - k] for t in ts if t - k >= 0], axis=0)
            curve.append(ridge_r2(X, Y, device=device))
        results[f"lagpos_r2_{src}"] = curve
        print(f"lag-position R2 from {src}: {[round(v, 3) for v in curve]}")

    with open(os.path.join(a.out, "tracking_probe.json"), "w") as f:
        json.dump(results, f, indent=1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.bar(["from h\n(recurrent state)", "from emb\n(single frame)"],
           [results["vel_r2_h"], results["vel_r2_emb"]],
           color=["#d62728", "#1f77b4"])
    ax.set_ylabel("ball velocity decode R²")
    ax.set_title("velocity needs temporal integration")
    ax = axes[1]
    ks = np.arange(a.max_lag + 1)
    ax.plot(ks, results["lagpos_r2_h"], "o-", color="#d62728", label="from h")
    ax.plot(ks, results["lagpos_r2_emb"], "s-", color="#1f77b4", label="from emb (autocorr control)")
    ax.set_xlabel("lag k (steps into the past)")
    ax.set_ylabel("ball position at t−k, decode R²")
    ax.set_title("memory horizon for dynamic content")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "tracking_probe.png"), dpi=140)
    print("saved", os.path.join(a.out, "tracking_probe.png"))


if __name__ == "__main__":
    main()
