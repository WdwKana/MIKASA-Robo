"""
PEM Stage 0' — offline sanity check.

Pipeline:
  1. Load cached MIKASA frames (analysis/ebm/path_a_data/<task>/ep*.npz).
  2. Extract frozen DINOv2 patch features for base + hand views (162 patches).
  3. Build (p_{t-1}, p_t) pairs across episodes; train PEMPredictor with
     MSE for ~30 epochs.  (Cached frames have no actions; we condition on
     `p_{t-1}` only — see predictor.py.)
  4. Compute per-patch surprise s_t = ‖p_t − p̂_t‖² on held-out episodes.
  5. Compare top-K-by-surprise against the color-derived ground-truth
     object patches (color is used ONLY for scoring, never for training).
  6. Save overlay visualizations on representative cube-reveal frames.

Pass bar: surprise heatmap visually lights up on the colored cube at the
moments it is shown, AND top-K-by-surprise hit-rate clearly exceeds chance.
This is the same evaluation harness used for the slot-attention Stage 0
(analysis/slot/run_stage0.py), so the two methods are directly comparable.
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
from analysis.pem.predictor import PEMPredictor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIKASA_COLORS = {
    0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255),
    3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255),
    6: (128, 0, 0), 7: (128, 128, 0), 8: (0, 128, 128),
}
COLOR_THR = 60
PATCH_FRAC = 0.05


def set_seed(s=0):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True


def color_mask(rgb_uint8, color_idx):
    target = np.array(MIKASA_COLORS[color_idx], dtype=np.float32)
    return np.linalg.norm(rgb_uint8.astype(np.float32) - target, axis=-1) < COLOR_THR


def patch_gt(pix_mask, grid):
    H, W = pix_mask.shape
    Hp, Wp = grid
    ph, pw = H // Hp, W // Wp
    out = np.zeros((Hp, Wp), dtype=bool)
    for i in range(Hp):
        for j in range(Wp):
            blk = pix_mask[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            out[i, j] = blk.mean() > PATCH_FRAC
    return out.reshape(-1)


@torch.no_grad()
def extract_episodes(vit, data_dir):
    """Return list of dicts: feat (T, 162, d), base_rgb (T, 128, 128, 3), color."""
    files = sorted(Path(data_dir).glob("ep*.npz"))
    eps = []
    for fp in files:
        d = np.load(fp)
        base = d["base_rgb"]
        hand = d["hand_rgb"]
        color = int(d["color_idx"])
        rgb6 = torch.from_numpy(np.concatenate([base, hand], axis=-1)).to(DEVICE)
        tok_b, tok_h, _, _ = vit(rgb6)                    # each (T, 81, d)
        feat = torch.cat([tok_b, tok_h], dim=1).cpu()     # (T, 162, d)
        eps.append({"feat": feat, "base": base, "hand": hand, "color": color})
    return eps


def train_predictor(predictor, episodes, epochs=30, bs=128, lr=2e-4):
    """Train on (p_{t-1}, p_t) pairs from all episodes."""
    prev_list, curr_list = [], []
    for ep in episodes:
        f = ep["feat"]            # (T, N, d)
        prev_list.append(f[:-1])
        curr_list.append(f[1:])
    P_prev = torch.cat(prev_list, 0)
    P_curr = torch.cat(curr_list, 0)
    print(f"[train] {P_prev.shape[0]} (t-1, t) pairs, feat dim {P_prev.shape}")
    opt = torch.optim.Adam(predictor.parameters(), lr=lr)
    N = P_prev.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(N)
        tot = 0.0
        for s in range(0, N, bs):
            idx = perm[s:s+bs]
            pp = P_prev[idx].to(DEVICE)
            pc = P_curr[idx].to(DEVICE)
            p_hat = predictor(pp)
            loss = ((p_hat - pc) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            opt.step()
            tot += loss.item() * pp.size(0)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  ep{ep:02d} mse={tot/N:.4f}")


@torch.no_grad()
def evaluate(predictor, episodes, vit, K=8, iou_thresh=0.25):
    """For each frame t>=1, compute surprise; compare top-K-surprise patches
    against the color-GT patches of the base view.

    Saliency / GT was defined over both views (162 patches); here for the
    "object capture" check we project surprise into the base-view half (first
    81 patches) so we can compare to the base-camera color GT mask.
    """
    Hp, Wp = vit.grid
    N_v = Hp * Wp                   # 81
    best_ious = []
    best_recalls = []
    hits = []
    n_eval = 0
    per_frame = []
    for ep in episodes:
        f = ep["feat"]              # (T, 162, d)
        T = f.shape[0]
        for t in range(1, T):
            base = ep["base"][t]
            gt = patch_gt(color_mask(base, ep["color"]), (Hp, Wp))   # (N_v,) bool
            if gt.sum() == 0:
                continue
            p_prev = f[t-1].unsqueeze(0).to(DEVICE)
            p_t = f[t].unsqueeze(0).to(DEVICE)
            sur = predictor.surprise(p_t, p_prev)[0].cpu().numpy()    # (162,)
            sur_base = sur[:N_v]                                       # base view
            # top-K-by-surprise in base view
            topk = np.argsort(-sur_base)[:K]
            sel = np.zeros(N_v, dtype=bool); sel[topk] = True
            inter = (sel & gt).sum(); union = (sel | gt).sum()
            iou = inter / max(union, 1)
            recall = inter / max(gt.sum(), 1)
            best_ious.append(iou); best_recalls.append(recall)
            hits.append(1.0 if recall >= 0.5 else 0.0)
            n_eval += 1
            per_frame.append((ep, t, sur_base, gt, iou))
    return {
        "n_eval_frames": n_eval,
        "mean_iou": float(np.mean(best_ious)) if n_eval else float("nan"),
        "mean_recall": float(np.mean(best_recalls)) if n_eval else float("nan"),
        "hit_rate_iou": float(np.mean([x > iou_thresh for x in best_ious])) if n_eval else float("nan"),
        "hit_rate_recall50": float(np.mean(hits)) if n_eval else float("nan"),
        "iou_thresh": iou_thresh,
        "K": K,
    }, per_frame


def save_overlays(per_frame, vit, out_dir, n=12):
    Hp, Wp = vit.grid
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # Pick frames with highest IoU AND best recall to show capability
    per_sorted = sorted(per_frame, key=lambda x: -x[4])[:n]
    for i, (ep, t, sur, gt, iou) in enumerate(per_sorted):
        base = ep["base"][t]
        sur_map = sur.reshape(Hp, Wp)
        sur_norm = (sur_map - sur_map.min()) / max(sur_map.max() - sur_map.min(), 1e-8)
        sur_up = np.array(Image.fromarray((sur_norm*255).astype(np.uint8))
                          .resize((128,128), Image.BILINEAR)) / 255.0
        gt_up = np.array(Image.fromarray((gt.reshape(Hp, Wp)*255).astype(np.uint8))
                         .resize((128,128), Image.NEAREST)) / 255.0
        fig, ax = plt.subplots(1, 3, figsize=(11, 4))
        ax[0].imshow(base); ax[0].set_title(f"frame t={t} (color {ep['color']})"); ax[0].axis("off")
        ax[1].imshow(base); ax[1].imshow(gt_up, cmap="Greens", alpha=0.5)
        ax[1].set_title("GT object"); ax[1].axis("off")
        ax[2].imshow(base); ax[2].imshow(sur_up, cmap="hot", alpha=0.6, vmin=0, vmax=1)
        ax[2].set_title(f"surprise (top-K IoU={iou:.2f})"); ax[2].axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"overlay_{i:02d}_iou{iou:.2f}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
    print(f"  saved {len(per_sorted)} overlays -> {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="RememberColor9-v0")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--iou-thresh", type=float, default=0.25)
    args = p.parse_args()
    set_seed(0)

    data_dir = ROOT / "analysis/ebm/path_a_data" / args.task
    vit = FrozenDualDinoV2().to(DEVICE).eval()
    print(f"[stage0'] task={args.task} grid={vit.grid} N_patches_per_view={vit.num_patches_per_view} d_vit={vit.dim}")

    print("[stage0'] extracting DINOv2 features for cached episodes...")
    episodes = extract_episodes(vit, data_dir)
    print(f"  {len(episodes)} episodes, each (T=60, 162 patches, d={vit.dim})")

    predictor = PEMPredictor(d_vit=vit.dim, n_patches=2*vit.num_patches_per_view,
                             action_dim=None, hidden=256, pos_dim=32).to(DEVICE)
    print(f"[stage0'] predictor params: {sum(p.numel() for p in predictor.parameters()):,}")
    print("[stage0'] training predictor (MSE, no labels)...")
    train_predictor(predictor, episodes, epochs=args.epochs)

    print("[stage0'] evaluating surprise top-K vs color GT...")
    metrics, per_frame = evaluate(predictor, episodes, vit, K=args.K, iou_thresh=args.iou_thresh)
    print(f"  n_eval_frames={metrics['n_eval_frames']}")
    print(f"  mean IoU(top-K, GT) = {metrics['mean_iou']:.3f}")
    print(f"  hit_rate(IoU>{args.iou_thresh}) = {metrics['hit_rate_iou']:.3f}")
    print(f"  mean recall(top-K covers GT) = {metrics['mean_recall']:.3f}")
    print(f"  hit_rate(recall>=0.5) = {metrics['hit_rate_recall50']:.3f}")

    out_vis = ROOT / "analysis/pem/stage0p_vis" / args.task
    save_overlays(per_frame, vit, out_vis)
    out_json = ROOT / "analysis/pem" / f"stage0p_{args.task}.json"
    with open(out_json, "w") as f:
        json.dump({**metrics, "task": args.task}, f, indent=2)
    print(f"[stage0'] metrics -> {out_json}")


if __name__ == "__main__":
    main()
