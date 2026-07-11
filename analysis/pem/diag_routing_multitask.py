"""
Multi-task Stage 0 — does the Token Routing actually do the right thing?

Two tasks, three methods, two destinations:

  tasks    : RememberColor9-v0  (sparse-event task — cube reveals)
             InterceptMedium-v0 (continuous-motion task — ball trajectory)
  methods  : SRB-MS, SRB-TR (new), V1-saliency-head (reference)
  routes   : episodic buffer (write candidates),
             LSTM-input pool (working-memory candidates)

We check:
  • Buffer-side  : does the buffer hold the task-relevant target across time?
                    (cube for RC9, ball for IMed) — paper claim: routing should
                    NOT degrade Remember-task buffer composition.
  • LSTM-side    : is the LSTM input pool populated by what the task needs to
                    track? cube for RC9 (no change vs MS), BALL for IMed
                    (the whole point of TR: motion → working memory).

Metrics per (task, method):
  buffer_cube_pres(t) — fraction of episodes with cube/ball in buffer at t
  lstm_target_frac(t) — fraction of LSTM pool that is cube/ball patches
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
sys.path.insert(0, str(ROOT))

from baselines.ppo.modules.frozen_vit import FrozenDualDinoV2
from baselines.ppo.modules.saliency_head import load_saliency_head
from analysis.pem.run_stage0p import (
    set_seed, color_mask, extract_episodes,
)


def patch_gt(pix_mask, grid, frac=0.05):
    """Adjustable threshold per task (cube is large, ball is small)."""
    H, W = pix_mask.shape
    Hp, Wp = grid
    ph, pw = H // Hp, W // Wp
    out = np.zeros((Hp, Wp), dtype=bool)
    for i in range(Hp):
        for j in range(Wp):
            blk = pix_mask[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            out[i, j] = blk.mean() > frac
    return out.reshape(-1)
from analysis.pem.diag_buffer_dynamics import (
    SimBuffer, score_srb, score_saliency, score_random,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────── method-specific scoring (buffer + lstm routes) ────────────────

@torch.no_grad()
def scores_srb_ms(p_t, p_prev, ema_change, sim_buf, alpha=0.1, lam=1.0):
    """Returns (buffer_score, lstm_score)."""
    if p_prev is not None:
        cur_change = ((p_t - p_prev) ** 2).sum(-1)
        ema_change.mul_(1 - alpha).add_(alpha * cur_change)
    srb = score_srb(p_t, sim_buf, p_t.shape[-1])
    d_norm = float(p_t.shape[-1])
    buf_s = srb / (1.0 + lam * ema_change / d_norm)
    # SRB-MS's LSTM pool is currently the SAME top-K as buffer (its v1 behavior)
    lstm_s = buf_s.clone()
    return buf_s, lstm_s


@torch.no_grad()
def scores_srb_tr(p_t, p_prev, ema_change, sim_buf, alpha=0.1, lam=1.0):
    """TR routing: buffer = SRB-MS score; LSTM pool = ema_change (motion)."""
    if p_prev is not None:
        cur_change = ((p_t - p_prev) ** 2).sum(-1)
        ema_change.mul_(1 - alpha).add_(alpha * cur_change)
    srb = score_srb(p_t, sim_buf, p_t.shape[-1])
    d_norm = float(p_t.shape[-1])
    buf_s = srb / (1.0 + lam * ema_change / d_norm)
    lstm_s = ema_change.clone()                     # high-motion → LSTM
    return buf_s, lstm_s


@torch.no_grad()
def scores_v1(p_t, sal_head, xy_concat):
    """V1: saliency head for both buffer and LSTM (V3a-style pool top-K saliency)."""
    s = score_saliency(p_t, sal_head, xy_concat)
    return s, s.clone()


# ──────────────── simulate ────────────────

def simulate(method, episodes, vit, sal_head, xy_concat, K=8, L=64, tau=30.0,
             patch_frac=0.05):
    """Returns per-step arrays:
      buf_target  : (E, T) prob target patch in buffer at t (eviction-tracked)
      lstm_target_frac : (E, T) fraction of LSTM top-K that is target patch
      buf_target_frac  : (E, T) fraction of buffer that is target patch
      spatial_buf : (N_total,) write counts to buffer
      spatial_lstm: (N_total,) routing counts to LSTM
    """
    Hp, Wp = vit.grid; N_v = Hp * Wp
    N_total = 2 * N_v
    E, T = len(episodes), episodes[0]["feat"].shape[0]
    buf_target = np.zeros((E, T), dtype=np.float32)
    buf_target_frac = np.zeros((E, T), dtype=np.float32)
    lstm_target_frac = np.zeros((E, T), dtype=np.float32)
    spatial_buf = np.zeros(N_total, dtype=np.float32)
    spatial_lstm = np.zeros(N_total, dtype=np.float32)

    for e, ep in enumerate(episodes):
        sim = SimBuffer(L=L, tau=tau)
        # precompute target GT per frame (cube for RC, ball for IMed)
        gt_per_t = []
        for t in range(T):
            gt = patch_gt(color_mask(ep["base"][t], ep["color"]), (Hp, Wp),
                          frac=patch_frac)
            gt_per_t.append(gt)

        ema_change = torch.zeros(N_total, device=DEVICE)
        p_prev = None
        for t in range(T):
            p_t = ep["feat"][t].to(DEVICE)
            if method == "srb_ms":
                buf_s, lstm_s = scores_srb_ms(p_t, p_prev, ema_change, sim)
            elif method == "srb_tr":
                buf_s, lstm_s = scores_srb_tr(p_t, p_prev, ema_change, sim)
            elif method == "v1_saliency":
                buf_s, lstm_s = scores_v1(p_t, sal_head, xy_concat)
            else:
                raise ValueError(method)

            buf_top = torch.topk(buf_s, K).indices.cpu().numpy()
            lstm_top = torch.topk(lstm_s, K).indices.cpu().numpy()

            # buffer simulation (priority eviction)
            for n in buf_top:
                is_tgt = (n < N_v) and bool(gt_per_t[t][n])
                sim.push(p_t[n].clone(), priority=float(buf_s[n]), t_now=t,
                         prov=(t, int(n), is_tgt))
                spatial_buf[n] += 1
            # lstm routing stats
            lstm_target_count = sum(1 for n in lstm_top if n < N_v and gt_per_t[t][n])
            lstm_target_frac[e, t] = lstm_target_count / K
            for n in lstm_top:
                spatial_lstm[n] += 1

            # buffer composition
            if sim.used() == 0:
                buf_target[e, t] = 0; buf_target_frac[e, t] = 0
            else:
                cnt = sum(1 for prov in sim.provenance if prov[2])
                buf_target[e, t] = 1.0 if cnt > 0 else 0.0
                buf_target_frac[e, t] = cnt / sim.used()

            p_prev = p_t.clone()
    return {
        "buf_target": buf_target,
        "buf_target_frac": buf_target_frac,
        "lstm_target_frac": lstm_target_frac,
        "spatial_buf": spatial_buf,
        "spatial_lstm": spatial_lstm,
    }


# ──────────────── plotting + reporting ────────────────

def report_task(task_name, results, vit, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Fig: buffer preservation
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for name, r in results.items():
        axes[0].plot(r["buf_target"].mean(0), label=name)
        axes[1].plot(r["lstm_target_frac"].mean(0), label=name)
    axes[0].set_xlabel("step"); axes[0].set_ylabel("P(target in buffer)")
    axes[0].set_title(f"{task_name}: buffer preservation (target = cube/ball)")
    axes[0].set_ylim(0, 1.05); axes[0].grid(True, alpha=0.3); axes[0].legend()
    axes[1].set_xlabel("step"); axes[1].set_ylabel("fraction of LSTM top-K that is target")
    axes[1].set_title(f"{task_name}: LSTM input pool composition")
    axes[1].set_ylim(0, 1.05); axes[1].grid(True, alpha=0.3); axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"routing_{task_name}.png", dpi=110)
    plt.close(fig)

    # Spatial heatmaps (base view)
    Hp, Wp = vit.grid
    n_methods = len(results)
    fig, axes = plt.subplots(2, n_methods, figsize=(4*n_methods, 8))
    for col, (name, r) in enumerate(results.items()):
        for row, key in enumerate(["spatial_buf", "spatial_lstm"]):
            ax = axes[row, col] if n_methods > 1 else axes[row]
            data = r[key][:Hp*Wp].reshape(Hp, Wp)
            im = ax.imshow(data, cmap="hot")
            label = "BUFFER" if row == 0 else "LSTM pool"
            ax.set_title(f"{name}\n{label} writes")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / f"spatial_{task_name}.png", dpi=110)
    plt.close(fig)

    # summary
    print(f"\n=== {task_name} ===")
    print(f"{'method':<14} {'buf_tgt_early':>14} {'buf_tgt_late':>14} "
          f"{'lstm_tgt_mean':>14} {'buf_tgt_frac':>14}")
    for name, r in results.items():
        be = r["buf_target"][:, :10].mean()
        bl = r["buf_target"][:, 40:].mean() if r["buf_target"].shape[1] > 40 else float("nan")
        lt = r["lstm_target_frac"].mean()
        bf = r["buf_target_frac"].mean()
        print(f"{name:<14} {be:>14.3f} {bl:>14.3f} {lt:>14.3f} {bf:>14.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--saliency-ckpt", default="analysis/ebm/path_a_head_v3.pt")
    args = p.parse_args()
    set_seed(0)

    vit = FrozenDualDinoV2().to(DEVICE).eval()
    sal_head, xy_concat = load_saliency_head(
        str(ROOT / args.saliency_ckpt), device=DEVICE, freeze=True)

    task_specs = [
        ("RememberColor9-v0", 0.05),    # cube is large, default threshold
        ("InterceptMedium-v0", 0.005),  # ball is small (4-5 pixels), tighter
    ]
    for task, frac in task_specs:
        data_dir = ROOT / "analysis/ebm/path_a_data" / task
        if not data_dir.exists():
            print(f"[skip] {task} — no cached frames"); continue
        print(f"\n[diag] task={task} patch_frac={frac}; extracting features...")
        episodes = extract_episodes(vit, data_dir)
        print(f"  {len(episodes)} episodes, T={episodes[0]['feat'].shape[0]}")

        results = {}
        for method in ["v1_saliency", "srb_ms", "srb_tr"]:
            print(f"[diag] method={method}...")
            results[method] = simulate(method, episodes, vit, sal_head, xy_concat,
                                       K=args.K, L=args.L, patch_frac=frac)

        out_dir = ROOT / "analysis/pem/diag_routing" / task
        report_task(task.replace("-v0", ""), results, vit, out_dir)
        print(f"plots → {out_dir}")


if __name__ == "__main__":
    main()
