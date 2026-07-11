"""
Stage 0 variant explorer — try 5 surprise mechanisms in parallel.

Hypothesis space (all use frozen DINOv2 patches; all replace saliency head):
  V1 FFP  — Frame-to-Frame Predictor  : p̂_t = f(p_{t-1}); MSE; surprise = L2 residual
                                         [baseline, Stage 0']
  V2 HN   — History Novelty (non-param): surprise[n] = min over k∈[1,K] of
                                         ‖p_t[n] − p_{t-k}[n]‖²
                                         (the buffer's own history IS the predictor)
  V3 PP   — Per-Position Prior         : learned background p̄[n] per patch position;
                                         surprise[n] = ‖p_t[n] − p̄[n]‖²
                                         (spatial prior; works at t=0 by construction)
  V4 DP   — Delta Predictor            : f predicts Δ_t = p_t − p_{t-1} from p_{t-1};
                                         surprise = ‖Δ_t − Δ̂_t‖² (higher-order dynamics)
  V5 HC   — History + PP combined      : max(V2, V3), captures both spatial outliers
                                         and temporal novelty against history

Eval: per-timestep IoU/recall(top-K=8 surprise patches, color-derived cube GT),
on cached RC9 frames (20 episodes × 60 steps). The interesting timesteps:
  t=0  : cold-start (no temporal context)         — only V3, V5 should work here
  t=10 : cube reveal after occlusion at t=5-9      — V1, V2, V4, V5 should fire
  t>>0 stable: cube static                          — nothing should fire (sparse buffer)
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
sys.path.insert(0, str(ROOT))

from baselines.ppo.modules.frozen_vit import FrozenDualDinoV2
from analysis.pem.predictor import PEMPredictor
from analysis.pem.run_stage0p import (
    set_seed, color_mask, patch_gt, extract_episodes,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────── modules for trainable variants ────────────────

class PerPositionPrior(nn.Module):
    """Learned per-patch-position background prior. Surprise = ‖p_t − p̄‖²."""
    def __init__(self, n_patches: int, d_vit: int):
        super().__init__()
        self.prior = nn.Parameter(torch.zeros(n_patches, d_vit))

    def surprise(self, p_t):
        # p_t: (B, N, d); prior: (N, d)
        return ((p_t - self.prior.unsqueeze(0)) ** 2).sum(-1)   # (B, N)

    def loss(self, p_t):
        return self.surprise(p_t).mean()


class DeltaPredictor(nn.Module):
    """Predict Δ_t = p_t − p_{t-1} from p_{t-1}; surprise = ‖Δ_t − Δ̂_t‖²."""
    def __init__(self, d_vit, n_patches, hidden=256, pos_dim=32):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, n_patches, pos_dim) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(d_vit + pos_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_vit),
        )

    def forward(self, p_prev):
        B, N, _ = p_prev.shape
        pos = self.pos.expand(B, N, -1)
        return self.mlp(torch.cat([p_prev, pos], dim=-1))    # predicted Δ

    def surprise(self, p_t, p_prev):
        delta_true = p_t - p_prev                              # (B, N, d)
        delta_pred = self.forward(p_prev)                      # (B, N, d)
        return ((delta_true - delta_pred) ** 2).sum(-1)


# ──────────────── training helpers ────────────────

def train_ffp(model, episodes, epochs=30, lr=2e-4, bs=128):
    """FFP and DP have the same training signature (predict from p_{t-1})."""
    prev_list, curr_list = [], []
    for ep in episodes:
        f = ep["feat"]
        prev_list.append(f[:-1]); curr_list.append(f[1:])
    P_prev = torch.cat(prev_list, 0); P_curr = torch.cat(curr_list, 0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Np = P_prev.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(Np); tot = 0
        for s in range(0, Np, bs):
            idx = perm[s:s+bs]
            pp = P_prev[idx].to(DEVICE); pc = P_curr[idx].to(DEVICE)
            if isinstance(model, PEMPredictor):
                ph = model(pp)
                loss = ((ph - pc) ** 2).mean()
            else:   # DeltaPredictor
                d_pred = model(pp)
                loss = ((d_pred - (pc - pp)) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * pp.size(0)


def train_pp(model, episodes, epochs=30, lr=2e-3, bs=128):
    """PerPositionPrior — train to minimize average ‖p_t − p̄‖² (i.e., p̄ = mean of p_t)."""
    all_feats = torch.cat([ep["feat"] for ep in episodes], 0)   # (M*T, N, d)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Np = all_feats.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(Np)
        for s in range(0, Np, bs):
            x = all_feats[perm[s:s+bs]].to(DEVICE)
            loss = model.loss(x)
            opt.zero_grad(); loss.backward(); opt.step()


# ──────────────── compute per-timestep surprise tensors ────────────────

@torch.no_grad()
def surprise_FFP(model, episodes):
    """Returns (E, T, N) surprise tensor. t=0 surprise undefined → set to 0."""
    feats = torch.stack([ep["feat"] for ep in episodes]).to(DEVICE)  # (E,T,N,d)
    E, T, N, d = feats.shape
    out = torch.zeros(E, T, N, device=DEVICE)
    for t in range(1, T):
        p_hat = model(feats[:, t-1])
        out[:, t] = ((feats[:, t] - p_hat) ** 2).sum(-1)
    return out.cpu().numpy()


@torch.no_grad()
def surprise_DP(model, episodes):
    feats = torch.stack([ep["feat"] for ep in episodes]).to(DEVICE)
    E, T, N, d = feats.shape
    out = torch.zeros(E, T, N, device=DEVICE)
    for t in range(1, T):
        out[:, t] = model.surprise(feats[:, t], feats[:, t-1])
    return out.cpu().numpy()


@torch.no_grad()
def surprise_PP(model, episodes):
    feats = torch.stack([ep["feat"] for ep in episodes]).to(DEVICE)
    E, T, N, d = feats.shape
    out = torch.zeros(E, T, N, device=DEVICE)
    for t in range(T):
        out[:, t] = model.surprise(feats[:, t])     # works at t=0
    return out.cpu().numpy()


@torch.no_grad()
def surprise_HN(episodes, K=5):
    """Non-parametric: for each patch position n, surprise[t,n] =
    min over k∈[1,K] of ‖p_t[n] − p_{t-k}[n]‖². No training needed."""
    feats = torch.stack([ep["feat"] for ep in episodes]).to(DEVICE)  # (E,T,N,d)
    E, T, N, d = feats.shape
    out = torch.full((E, T, N), 0.0, device=DEVICE)
    for t in range(T):
        if t == 0:
            continue                # undefined; leave 0 (we'll combine with PP later)
        ks = list(range(1, min(K, t) + 1))
        # stack history (e, K_t, N, d); compute per-position L2 to current
        hist = torch.stack([feats[:, t - k] for k in ks], dim=1)    # (E, K_t, N, d)
        dist = ((feats[:, t].unsqueeze(1) - hist) ** 2).sum(-1)     # (E, K_t, N)
        out[:, t] = dist.min(dim=1).values
    return out.cpu().numpy()


@torch.no_grad()
def surprise_SRB(episodes, K_topk=8, L_buf=64, novelty_thresh=0.0):
    """Self-Referential Buffer: surprise[t,n] = min distance from p_t[n] to any
    current buffer entry. Top-K-by-surprise above novelty_thresh get written.
    Buffer = FIFO of past writes (eviction by age, oldest out at capacity L).

    This is a closed-loop simulation: the buffer's own state determines what
    gets written next. No external predictor, no history cache, no trained
    module. The buffer IS the surprise oracle.

    Returns surprise array (E, T, N) for evaluation against color GT.
    """
    feats = torch.stack([ep["feat"] for ep in episodes]).to(DEVICE)
    E, T, N, d = feats.shape
    surprise_out = torch.zeros(E, T, N, device=DEVICE)

    for e in range(E):
        # per-env buffer simulation
        buf = torch.zeros(L_buf, d, device=DEVICE)
        used = 0
        for t in range(T):
            p_t = feats[e, t]                                # (N, d)
            if used == 0:
                # buffer empty: everything is "novel" → use raw feature norm
                # so top-K picks the most distinctive patches at t=0
                sur = (p_t ** 2).sum(-1)
            else:
                # distance from each patch to nearest buffer entry
                dist = ((p_t.unsqueeze(1) - buf[:used].unsqueeze(0)) ** 2).sum(-1)
                # (N, used) — min over buffer dim
                sur = dist.min(dim=1).values
            surprise_out[e, t] = sur

            # write top-K patches into buffer (FIFO eviction)
            topk_val, topk_idx = sur.topk(K_topk)
            for k in range(K_topk):
                if topk_val[k] <= novelty_thresh and used > 0:
                    continue
                if used < L_buf:
                    buf[used] = p_t[topk_idx[k]]
                    used += 1
                else:
                    # FIFO: shift out oldest, push new
                    buf = torch.roll(buf, shifts=-1, dims=0)
                    buf[-1] = p_t[topk_idx[k]]
    return surprise_out.cpu().numpy()


# ──────────────── eval harness ────────────────

def evaluate(surprise_array, episodes, vit, K=8):
    """surprise_array: (E, T, N_total=162). Use base view (first N_v=81 patches)."""
    Hp, Wp = vit.grid; N_v = Hp * Wp
    per_t = {}                    # t -> list of (iou, recall)
    for e, ep in enumerate(episodes):
        for t in range(surprise_array.shape[1]):
            gt = patch_gt(color_mask(ep["base"][t], ep["color"]), (Hp, Wp))
            if gt.sum() == 0:
                continue
            sur = surprise_array[e, t, :N_v]
            topk = np.argsort(-sur)[:K]
            sel = np.zeros(N_v, dtype=bool); sel[topk] = True
            inter = (sel & gt).sum(); union = (sel | gt).sum()
            per_t.setdefault(t, []).append(
                (inter / max(union, 1), inter / max(gt.sum(), 1))
            )
    return per_t


def summarize(per_t, name):
    print(f"\n=== {name} ===")
    print(f"{'t':>3} {'n':>4} {'mean_IoU':>9} {'mean_recall':>11} {'hit_recall>=0.5':>16}")
    for t in [0, 1, 2, 3, 4, 10, 11, 12, 15, 20, 30, 40, 50, 59]:
        if t not in per_t: continue
        vals = per_t[t]
        if not vals: continue
        iou = np.mean([v[0] for v in vals]); rec = np.mean([v[1] for v in vals])
        hit = np.mean([v[1] >= 0.5 for v in vals])
        print(f"{t:>3} {len(vals):>4} {iou:>9.3f} {rec:>11.3f} {hit:>16.3f}")
    # overall
    all_ious = [v[0] for vals in per_t.values() for v in vals]
    all_recs = [v[1] for vals in per_t.values() for v in vals]
    hit = np.mean([r >= 0.5 for r in all_recs]) if all_recs else float("nan")
    print(f"OVERALL mean_IoU={np.mean(all_ious):.3f}  mean_recall={np.mean(all_recs):.3f}  hit_recall>=0.5={hit:.3f}")
    return {
        "mean_iou": float(np.mean(all_ious)),
        "mean_recall": float(np.mean(all_recs)),
        "hit_recall50": float(hit),
        "per_t": {t: {"mean_iou": float(np.mean([v[0] for v in vals])),
                      "mean_recall": float(np.mean([v[1] for v in vals]))}
                  for t, vals in per_t.items()}
    }


# ──────────────── main ────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="RememberColor9-v0")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--K-topk", type=int, default=8)
    p.add_argument("--K-history", type=int, default=5)
    args = p.parse_args()
    set_seed(0)

    vit = FrozenDualDinoV2().to(DEVICE).eval()
    print(f"[explore] task={args.task} grid={vit.grid}")
    print("[explore] extracting features...")
    episodes = extract_episodes(vit, ROOT / "analysis/ebm/path_a_data" / args.task)
    print(f"  {len(episodes)} episodes; T=60, N=162, d=384")
    N_total = 2 * vit.num_patches_per_view

    results = {}

    # V1 FFP
    print("\n[V1 FFP] training frame-to-frame predictor...")
    ffp = PEMPredictor(d_vit=vit.dim, n_patches=N_total).to(DEVICE)
    train_ffp(ffp, episodes, epochs=args.epochs)
    sur_ffp = surprise_FFP(ffp, episodes)
    results["V1_FFP"] = summarize(evaluate(sur_ffp, episodes, vit, K=args.K_topk), "V1 FFP (frame-to-frame)")

    # V2 HN
    print("\n[V2 HN] computing history novelty (no training)...")
    sur_hn = surprise_HN(episodes, K=args.K_history)
    results["V2_HN"] = summarize(evaluate(sur_hn, episodes, vit, K=args.K_topk), f"V2 HN (history K={args.K_history})")

    # V3 PP
    print("\n[V3 PP] training per-position prior...")
    pp = PerPositionPrior(n_patches=N_total, d_vit=vit.dim).to(DEVICE)
    train_pp(pp, episodes, epochs=args.epochs)
    sur_pp = surprise_PP(pp, episodes)
    results["V3_PP"] = summarize(evaluate(sur_pp, episodes, vit, K=args.K_topk), "V3 PP (per-position prior)")

    # V4 DP
    print("\n[V4 DP] training delta predictor...")
    dp = DeltaPredictor(d_vit=vit.dim, n_patches=N_total).to(DEVICE)
    train_ffp(dp, episodes, epochs=args.epochs)
    sur_dp = surprise_DP(dp, episodes)
    results["V4_DP"] = summarize(evaluate(sur_dp, episodes, vit, K=args.K_topk), "V4 DP (delta predictor)")

    # V5 HC: max(HN-normalized, PP-normalized)
    print("\n[V5 HC] combining V2+V3 (max of normalized scores)...")
    def norm(x): return (x - x.mean()) / (x.std() + 1e-8)
    sur_hc = np.maximum(norm(sur_hn), norm(sur_pp))
    results["V5_HC"] = summarize(evaluate(sur_hc, episodes, vit, K=args.K_topk), "V5 HC (HN ⊕ PP)")

    # V6 SRB: self-referential buffer
    print("\n[V6 SRB] simulating self-referential buffer (no module)...")
    sur_srb = surprise_SRB(episodes, K_topk=args.K_topk, L_buf=64)
    results["V6_SRB"] = summarize(evaluate(sur_srb, episodes, vit, K=args.K_topk),
                                  "V6 SRB (buffer = own surprise oracle)")

    # final comparison table
    print("\n" + "=" * 92)
    print(f"{'variant':<26} {'overall_iou':>11} {'overall_recall':>14} "
          f"{'t=0_recall':>11} {'t=10_recall':>12} {'trained?':>10}")
    print("=" * 92)
    trained_map = {"V1_FFP": "yes", "V2_HN": "NO", "V3_PP": "yes",
                    "V4_DP": "yes", "V5_HC": "partial", "V6_SRB": "NO"}
    for name, m in results.items():
        t0 = m["per_t"].get(0, {}).get("mean_recall", float("nan"))
        t10 = m["per_t"].get(10, {}).get("mean_recall", float("nan"))
        print(f"{name:<26} {m['mean_iou']:>11.3f} {m['mean_recall']:>14.3f} "
              f"{t0:>11.3f} {t10:>12.3f} {trained_map.get(name,'?'):>10}")

    out_json = ROOT / "analysis/pem" / f"explore_{args.task}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[explore] -> {out_json}")


if __name__ == "__main__":
    main()
