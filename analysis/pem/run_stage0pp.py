"""
Stage 0'' — MPEB sanity (memory-driven surprise).

Replaces frame-to-frame surprise (Stage 0', which missed t=0) with
predictions from a recurrent working memory state h_{t-1}. At episode start
h_init is a learned token, so t=0 also gets a meaningful prediction → any
genuinely-new visual content (the colored cube) shows up as surprise.

Pipeline:
  1. Load cached MIKASA frames; extract frozen DINOv2 features (162 patches/frame).
  2. Build episode tensor (B_ep, T=60, 162, d_vit).
  3. Train MPEB end-to-end (GRU + MemoryPredictor) with MSE on episodes,
     unrolling through time.
  4. Evaluate per-step surprise → top-K-by-surprise IoU/recall vs color GT.
  5. Save overlays AND a per-timestep curve so we can see whether t=0
     captures the cube (the failure mode of Stage 0').
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
sys.path.insert(0, str(ROOT))

from baselines.ppo.modules.frozen_vit import FrozenDualDinoV2
from analysis.pem.mpeb_module import MPEB
from analysis.pem.run_stage0p import (
    set_seed, color_mask, patch_gt, extract_episodes, MIKASA_COLORS,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_mpeb(model, episodes, epochs=80, lr=3e-4):
    """Train on full-episode unrolls. episodes: list of dicts with .feat (T,N,d)."""
    # Stack into batch: (B_ep, T, N, d)
    feats = torch.stack([ep["feat"] for ep in episodes]).to(DEVICE)  # (E, T, N, d)
    B, T, N, d = feats.shape
    print(f"[train] full-batch unroll: B_ep={B}, T={T}, N={N}, d={d}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        preds, _ = model.unroll(feats)               # (B, T, N, d) — p̂_t from h_{t-1}
        loss = ((preds - feats) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            with torch.no_grad():
                # also report MSE just on t=0 (the cold-start case)
                t0_mse = ((preds[:, 0] - feats[:, 0]) ** 2).mean().item()
            print(f"  ep{ep:02d} mse={loss.item():.4f}  t0_mse={t0_mse:.4f}")


@torch.no_grad()
def evaluate(model, episodes, vit, K=8, iou_thresh=0.25):
    """For each (episode, t), compute surprise top-K vs color GT (base view only)."""
    Hp, Wp = vit.grid
    N_v = Hp * Wp
    feats = torch.stack([ep["feat"] for ep in episodes]).to(DEVICE)  # (E,T,162,d)
    surprise = model.surprise(feats).cpu().numpy()                   # (E,T,162)

    per_t = {}                # t -> list of (iou, recall)
    per_frame = []            # for overlay selection
    for e, ep in enumerate(episodes):
        for t in range(feats.shape[1]):
            gt = patch_gt(color_mask(ep["base"][t], ep["color"]), (Hp, Wp))
            if gt.sum() == 0:
                continue
            sur = surprise[e, t, :N_v]
            topk = np.argsort(-sur)[:K]
            sel = np.zeros(N_v, dtype=bool); sel[topk] = True
            inter = (sel & gt).sum(); union = (sel | gt).sum()
            iou = inter / max(union, 1)
            recall = inter / max(gt.sum(), 1)
            per_t.setdefault(t, []).append((iou, recall))
            per_frame.append((ep, t, sur, gt, iou))

    # aggregate per timestep
    summary = {}
    for t, vals in sorted(per_t.items()):
        ious = [v[0] for v in vals]; recs = [v[1] for v in vals]
        summary[t] = {
            "n": len(vals),
            "mean_iou": float(np.mean(ious)),
            "mean_recall": float(np.mean(recs)),
            "hit_rate_iou": float(np.mean([i > iou_thresh for i in ious])),
            "hit_rate_recall50": float(np.mean([r >= 0.5 for r in recs])),
        }
    # overall (frames-flat avg)
    all_ious = [v[0] for vals in per_t.values() for v in vals]
    all_recs = [v[1] for vals in per_t.values() for v in vals]
    overall = {
        "n_eval_frames": len(all_ious),
        "mean_iou": float(np.mean(all_ious)) if all_ious else float("nan"),
        "mean_recall": float(np.mean(all_recs)) if all_recs else float("nan"),
        "hit_rate_iou": float(np.mean([i > iou_thresh for i in all_ious])) if all_ious else float("nan"),
        "hit_rate_recall50": float(np.mean([r >= 0.5 for r in all_recs])) if all_recs else float("nan"),
    }
    return summary, overall, per_frame


def print_per_t(summary):
    print(f"\n{'t':>3} {'n':>4} {'mean_iou':>9} {'mean_recall':>11} {'hit_iou>0.25':>14} {'hit_rec>=0.5':>14}")
    for t in sorted(summary.keys()):
        s = summary[t]
        if t in (0, 1, 2, 3, 4, 10, 11, 12, 15, 20, 30, 40, 50, 59):
            print(f"{t:>3} {s['n']:>4} {s['mean_iou']:>9.3f} {s['mean_recall']:>11.3f} "
                  f"{s['hit_rate_iou']:>14.3f} {s['hit_rate_recall50']:>14.3f}")


def save_overlays(per_frame, vit, out_dir, n=12, focus_t=None):
    Hp, Wp = vit.grid
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if focus_t is not None:
        chosen = [f for f in per_frame if f[1] == focus_t][:n]
        if not chosen:
            chosen = sorted(per_frame, key=lambda x: -x[4])[:n]
    else:
        chosen = sorted(per_frame, key=lambda x: -x[4])[:n]
    for i, (ep, t, sur, gt, iou) in enumerate(chosen):
        base = ep["base"][t]
        sur_map = sur[:Hp*Wp].reshape(Hp, Wp)
        sur_norm = (sur_map - sur_map.min()) / max(sur_map.max() - sur_map.min(), 1e-8)
        sur_up = np.array(Image.fromarray((sur_norm*255).astype(np.uint8))
                          .resize((128, 128), Image.BILINEAR)) / 255.0
        gt_up = np.array(Image.fromarray((gt.reshape(Hp, Wp)*255).astype(np.uint8))
                         .resize((128, 128), Image.NEAREST)) / 255.0
        fig, ax = plt.subplots(1, 3, figsize=(11, 4))
        ax[0].imshow(base); ax[0].set_title(f"t={t} color={ep['color']}"); ax[0].axis("off")
        ax[1].imshow(base); ax[1].imshow(gt_up, cmap="Greens", alpha=0.5)
        ax[1].set_title("GT object"); ax[1].axis("off")
        ax[2].imshow(base); ax[2].imshow(sur_up, cmap="hot", alpha=0.6, vmin=0, vmax=1)
        ax[2].set_title(f"surprise (MPEB)  IoU={iou:.2f}"); ax[2].axis("off")
        fig.tight_layout()
        fname = f"overlay_t{t:02d}_{i:02d}_iou{iou:.2f}.png"
        fig.savefig(out_dir / fname, dpi=100, bbox_inches="tight")
        plt.close(fig)
    print(f"  saved {len(chosen)} overlays -> {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="RememberColor9-v0")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--mem-dim", type=int, default=128)
    p.add_argument("--iou-thresh", type=float, default=0.25)
    args = p.parse_args()
    set_seed(0)

    data_dir = ROOT / "analysis/ebm/path_a_data" / args.task
    vit = FrozenDualDinoV2().to(DEVICE).eval()
    print(f"[stage0''] task={args.task} grid={vit.grid} d_vit={vit.dim}")

    print("[stage0''] extracting DINOv2 features for cached episodes...")
    episodes = extract_episodes(vit, data_dir)
    print(f"  {len(episodes)} episodes, T=60, 162 patches each")

    model = MPEB(d_vit=vit.dim, n_patches=2*vit.num_patches_per_view,
                 mem_dim=args.mem_dim).to(DEVICE)
    print(f"[stage0''] MPEB params: {sum(p.numel() for p in model.parameters()):,}")
    print("[stage0''] training MPEB (working memory + memory predictor)...")
    train_mpeb(model, episodes, epochs=args.epochs)

    print("[stage0''] evaluating surprise top-K vs color GT...")
    per_t, overall, per_frame = evaluate(model, episodes, vit, K=args.K,
                                         iou_thresh=args.iou_thresh)
    print("\n=== Overall (frames-flat) ===")
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"  {k}={v:.3f}")
        else:
            print(f"  {k}={v}")
    print("\n=== Per-timestep IoU / recall ===")
    print_per_t(per_t)

    out_vis = ROOT / "analysis/pem/stage0pp_vis" / args.task
    print("\n[stage0''] saving overlays...")
    save_overlays(per_frame, vit, out_vis, n=8)
    # also save t=0 overlays specifically (the cold-start test)
    save_overlays(per_frame, vit, out_vis / "t0_focus", n=8, focus_t=0)
    save_overlays(per_frame, vit, out_vis / "t10_focus", n=8, focus_t=10)

    out_json = ROOT / "analysis/pem" / f"stage0pp_{args.task}.json"
    with open(out_json, "w") as f:
        json.dump({"task": args.task, "overall": overall, "per_t": per_t}, f, indent=2)
    print(f"[stage0''] metrics -> {out_json}")


if __name__ == "__main__":
    main()
