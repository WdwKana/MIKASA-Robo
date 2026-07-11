"""
Stage-0 diagnostic for Multi-View Self-Referential Surprise (MV-SRS).

Tests whether augmenting SRB's L2-on-DINOv2 surprise with an OR-combined
L2-on-raw-RGB-statistics signal recovers cube-specific buffer preservation
on tasks where DINOv2 features are color-blind.

For each cached task (RC5, RC9, Shape5):
  • SRB-MS  (DINOv2-only baseline, our existing method)
  • MV-SRS  (max-of-normalized {DINOv2 surprise, RGB surprise})

Measure per-step P(target-color cube in buffer) and per-target write counts.

The RGB view is a parameter-free 3-d mean-RGB per patch (14×14 pixel block).
No labels are used; this is purely "novelty in any complementary view."
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
sys.path.insert(0, str(ROOT))

from baselines.ppo.modules.frozen_vit import FrozenDualDinoV2
from analysis.pem.run_stage0p import (
    set_seed, color_mask, patch_gt, extract_episodes,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────── Lightweight buffer with provenance ──────────────────

class SimBuffer:
    def __init__(self, L=64, tau=30.0, novelty_thresh=0.95):
        self.L, self.tau, self.novelty_thresh = L, tau, novelty_thresh
        self.feats = []          # DINOv2 features per slot (d_vit,)
        self.rgbs = []           # RGB stats per slot (3,)
        self.prios = []
        self.ts = []
        self.prov = []           # (write_t, patch_n, is_target_color)

    def used(self): return len(self.feats)

    def push(self, feat, rgb, prio, t_now, prov):
        if self.used() > 0:
            existing = torch.stack(self.feats)
            sims = F.cosine_similarity(feat.unsqueeze(0), existing, dim=-1)
            if sims.max() > self.novelty_thresh:
                return False
        if self.used() < self.L:
            self.feats.append(feat); self.rgbs.append(rgb)
            self.prios.append(prio); self.ts.append(t_now); self.prov.append(prov)
            return True
        ages = torch.tensor([t_now - ts for ts in self.ts], dtype=torch.float32)
        prios = torch.tensor(self.prios, dtype=torch.float32)
        eff = prios * torch.exp(-ages / self.tau)
        idx = int(eff.argmin().item())
        if float(prio) <= float(eff[idx]):
            return False
        self.feats[idx] = feat; self.rgbs[idx] = rgb
        self.prios[idx] = prio; self.ts[idx] = t_now; self.prov[idx] = prov
        return True


# ──────────────────── RGB-per-patch helper ────────────────────

def rgb_per_patch_pair(base_rgb_t: np.ndarray, hand_rgb_t: np.ndarray,
                       grid=(9, 9), input_size=126):
    """Return (N_total=162, 3) mean RGB per patch for both views.
    Resize each view to input_size, compute per-14x14 block mean."""
    Hp, Wp = grid
    ph = pw = input_size // Hp
    out = np.zeros(2 * Hp * Wp * 3, dtype=np.float32).reshape(2 * Hp * Wp, 3)
    for v_idx, view in enumerate([base_rgb_t, hand_rgb_t]):
        # resize 128->126 by simple crop (close enough for stage-0)
        view_r = view[1:127, 1:127, :]                            # (126, 126, 3)
        for i in range(Hp):
            for j in range(Wp):
                blk = view_r[i*ph:(i+1)*ph, j*pw:(j+1)*pw]        # (14, 14, 3)
                out[v_idx * Hp * Wp + i * Wp + j] = blk.mean(axis=(0, 1))
    return out                                                     # (162, 3)


# ──────────────────── scoring functions ────────────────────

def score_srb_ms(p_t, p_prev, ema_change, sim_buf, ms_alpha=0.1, ms_lambda=1.0):
    """Return per-patch surprise; mutates ema_change in place."""
    d = p_t.shape[-1]
    if p_prev is not None:
        cur_change = ((p_t - p_prev) ** 2).sum(-1)
        ema_change.mul_(1 - ms_alpha).add_(ms_alpha * cur_change)
    # SRB raw: min dist to buffer
    if sim_buf.used() == 0:
        srb_raw = (p_t * p_t).sum(-1)
    else:
        buf = torch.stack(sim_buf.feats)
        diff = p_t.unsqueeze(1) - buf.unsqueeze(0)
        srb_raw = (diff * diff).sum(-1).min(dim=-1).values
    suppress = 1.0 + ms_lambda * ema_change / d
    return srb_raw / suppress


def score_rgb_ms(rgb_stats, rgb_prev, ema_rgb_change, sim_buf,
                 ms_alpha=0.1, ms_lambda=1.0):
    """Motion-suppressed RGB-space surprise. Same ms gate as DINOv2 view."""
    d_rgb = rgb_stats.shape[-1]                       # 3
    if rgb_prev is not None:
        cur = ((rgb_stats - rgb_prev) ** 2).sum(-1)
        ema_rgb_change.mul_(1 - ms_alpha).add_(ms_alpha * cur)
    if sim_buf.used() == 0:
        raw = (rgb_stats * rgb_stats).sum(-1)
    else:
        buf_rgb = torch.stack(sim_buf.rgbs)
        diff = rgb_stats.unsqueeze(1) - buf_rgb.unsqueeze(0)
        raw = (diff * diff).sum(-1).min(dim=-1).values
    suppress = 1.0 + ms_lambda * ema_rgb_change / d_rgb
    return raw / suppress


def normalize(x):
    """Min-max normalize per frame."""
    mn = x.min(); mx = x.max()
    return (x - mn) / (mx - mn + 1e-8)


def score_mv_concat(p_t, rgb_t, p_prev, rgb_prev, ema_change, ema_rgb_change,
                    sim_buf, ms_alpha=0.1, ms_lambda=1.0):
    """
    Token-level concat [DINOv2, RGB] -> diagonal Mahalanobis surprise.

    Uses per-dim running variance from buffer entries to whiten each dimension
    independently, so DINOv2 (high-variance / 768-d) and RGB (low-variance / 3-d)
    contribute proportionally. Implicitly amplifies DINOv2's low-variance
    color-correlated subspace too (since whitening lifts low-var dims).

    Single L2 surprise, single top-K. No combiner.
    """
    d_vit = p_t.shape[-1]
    d_rgb = rgb_t.shape[-1]
    # motion suppression: track separately per side (different scales)
    if p_prev is not None:
        cur_v = ((p_t - p_prev) ** 2).sum(-1)
        ema_change.mul_(1 - ms_alpha).add_(ms_alpha * cur_v)
        cur_r = ((rgb_t - rgb_prev) ** 2).sum(-1)
        ema_rgb_change.mul_(1 - ms_alpha).add_(ms_alpha * cur_r)
    # joint surprise: per-dim whitening using buffer per-dim std (diag Mahalanobis)
    combined_t = torch.cat([p_t, rgb_t], dim=-1)                # (N, d_vit + d_rgb)
    if sim_buf.used() < 4:
        raw = (combined_t * combined_t).sum(-1)
    else:
        buf_feat = torch.stack(sim_buf.feats)
        buf_rgb = torch.stack(sim_buf.rgbs)
        buf_combined = torch.cat([buf_feat, buf_rgb], dim=-1)   # (L, d_vit + d_rgb)
        std = buf_combined.std(dim=0).clamp(min=1e-3)          # (d_vit + d_rgb,)
        w_t = combined_t / std
        w_b = buf_combined / std
        diff = w_t.unsqueeze(1) - w_b.unsqueeze(0)              # (N, L, d)
        raw = (diff * diff).sum(-1).min(dim=-1).values          # (N,)
    # joint motion suppression: scale-aware
    suppress = 1.0 + ms_lambda * (ema_change / d_vit + ema_rgb_change / d_rgb) * 0.5
    return raw / suppress


# ──────────────────── simulation ────────────────────

def simulate(method, episodes, vit, K=8, L=64, tau=30.0,
             patch_frac=0.05, mv_weight=1.0):
    Hp, Wp = vit.grid; N_v = Hp * Wp; N_total = 2 * N_v
    E, T = len(episodes), episodes[0]["feat"].shape[0]
    buf_target = np.zeros((E, T), dtype=np.float32)
    buf_target_frac = np.zeros((E, T), dtype=np.float32)
    n_used = np.zeros((E, T), dtype=np.int32)

    for e, ep in enumerate(episodes):
        sim = SimBuffer(L=L, tau=tau)
        # Precompute color GT per frame
        gt_per_t = [patch_gt(color_mask(ep["base"][t], ep["color"]), (Hp, Wp),
                              ) if patch_frac is None else
                    _patch_gt_thr(color_mask(ep["base"][t], ep["color"]), (Hp, Wp), patch_frac)
                    for t in range(T)]
        # Precompute per-patch RGB stats for all frames (cheap)
        rgb_all = np.stack([rgb_per_patch_pair(ep["base"][t], ep["hand"][t])
                             for t in range(T)])         # (T, 162, 3)
        ema_change = torch.zeros(N_total, device=DEVICE)
        ema_rgb_change = torch.zeros(N_total, device=DEVICE)
        p_prev = None
        rgb_prev = None
        for t in range(T):
            p_t = ep["feat"][t].to(DEVICE)
            rgb_t = torch.from_numpy(rgb_all[t]).float().to(DEVICE)

            if method == "srb_ms":
                sur = score_srb_ms(p_t, p_prev, ema_change, sim)
                topk = torch.topk(sur, K).indices.cpu().numpy()
                prios = [float(sur[n]) for n in topk]
            elif method == "mv_srs":
                srb_s = score_srb_ms(p_t, p_prev, ema_change, sim)
                rgb_s = score_rgb_ms(rgb_t, rgb_prev, ema_rgb_change, sim)
                sur = torch.maximum(normalize(srb_s),
                                     mv_weight * normalize(rgb_s))
                topk = torch.topk(sur, K).indices.cpu().numpy()
                prios = [float(sur[n]) for n in topk]
            elif method == "mv_split":
                # Each view picks K/2 independently. Guarantees both views write.
                srb_s = score_srb_ms(p_t, p_prev, ema_change, sim)
                rgb_s = score_rgb_ms(rgb_t, rgb_prev, ema_rgb_change, sim)
                k_each = max(1, K // 2)
                t_dino = torch.topk(srb_s, k_each).indices.cpu().numpy()
                t_rgb  = torch.topk(rgb_s, k_each).indices.cpu().numpy()
                # de-dup while preserving order
                seen = set(); topk_list = []
                for n in list(t_dino) + list(t_rgb):
                    if int(n) in seen: continue
                    seen.add(int(n)); topk_list.append(int(n))
                topk = np.asarray(topk_list, dtype=np.int64)
                # use srb_s for buffer priority (we still want DINOv2 to dominate eviction)
                prios = [float(srb_s[n]) for n in topk]
            elif method == "mv_concat":
                sur = score_mv_concat(p_t, rgb_t, p_prev, rgb_prev,
                                       ema_change, ema_rgb_change, sim)
                topk = torch.topk(sur, K).indices.cpu().numpy()
                prios = [float(sur[n]) for n in topk]
            else:
                raise ValueError(method)

            for n, prio in zip(topk, prios):
                is_target = (n < N_v) and bool(gt_per_t[t][n])
                sim.push(p_t[n].clone(), rgb_t[n].clone(),
                         prio=prio, t_now=t,
                         prov=(t, int(n), is_target))

            n_used[e, t] = sim.used()
            if sim.used() > 0:
                cnt = sum(1 for prov in sim.prov if prov[2])
                buf_target[e, t] = 1.0 if cnt > 0 else 0.0
                buf_target_frac[e, t] = cnt / sim.used()
            p_prev = p_t.clone()
            rgb_prev = rgb_t.clone()
    return {"buf_target": buf_target, "buf_target_frac": buf_target_frac,
            "n_used": n_used}


def _patch_gt_thr(pix, grid, frac):
    Hp, Wp = grid; H, W = pix.shape
    ph, pw = H // Hp, W // Wp
    out = np.zeros((Hp, Wp), dtype=bool)
    for i in range(Hp):
        for j in range(Wp):
            out[i, j] = pix[i*ph:(i+1)*ph, j*pw:(j+1)*pw].mean() > frac
    return out.reshape(-1)


def report(name, results):
    bt = results["buf_target"]
    bf = results["buf_target_frac"]
    early = bt[:, :10].mean()
    late = bt[:, 40:].mean() if bt.shape[1] > 40 else float("nan")
    return early, late, bf.mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+",
                   default=["RememberColor5-v0", "RememberColor9-v0", "RememberShape5-v0"])
    args = p.parse_args()
    set_seed(0)

    vit = FrozenDualDinoV2().to(DEVICE).eval()

    print(f"\n{'task':<24} {'method':<10} {'P(tgt) early':<14} {'P(tgt) late':<14} {'tgt frac':<10}")
    print("="*86)
    for task in args.tasks:
        d = ROOT / "analysis/ebm/path_a_data" / task
        if not d.exists(): print(f"[skip] {task}"); continue
        eps = extract_episodes(vit, d)
        print(f"--- {task} ({len(eps)} eps) ---")
        method_results = {}
        for method in ["srb_ms", "mv_split", "mv_concat"]:
            r = simulate(method, eps, vit, K=8, L=64, patch_frac=0.05)
            method_results[method] = r
            early, late, frac = report(method, r)
            tag = {"srb_ms": "SRB-MS", "mv_split": "MV-SPLIT",
                   "mv_concat": "MV-CONCAT"}[method]
            print(f"{task:<24} {tag:<10} {early:.3f}         {late:.3f}         {frac:.4f}")

        # save fig: cube preservation over time
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for method, label in [("srb_ms",    "SRB-MS (DINOv2 only)"),
                              ("mv_split",  "MV-SPLIT (K/2 each)"),
                              ("mv_concat", "MV-CONCAT (Mahalanobis)")]:
            r = method_results[method]
            ax.plot(r["buf_target"].mean(0), label=label)
        ax.set_xlabel("step"); ax.set_ylabel("P(target cube in buffer)")
        ax.set_title(f"{task}: Stage-0 cube preservation")
        ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3); ax.legend()
        out = ROOT / "analysis/pem/diag_mvsrs" / f"{task}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
        print(f"  fig -> {out}")


if __name__ == "__main__":
    main()
