"""
Stage 0 — perception validation for object-centric slots (NO RL).

1. Load cached MIKASA frames (analysis/ebm/path_a_data/<task>/ep*.npz).
2. Extract frozen DINOv2 patch features (base camera).
3. Train DINOSAUR slot attention UNSUPERVISED (reconstruct DINOv2 features).
4. Evaluate whether a slot lands on the task-relevant object:
   - ground-truth object region = pixels matching the target color
     (color is privileged info used ONLY for scoring, never for training).
   - downsample to the 9x9 patch grid -> GT patch mask.
   - hard-assign each patch to its argmax slot -> per-slot patch masks.
   - best-slot IoU with GT; hit-rate = fraction of frames best-IoU > thresh.
5. Save overlay visualizations.

Pass bar: hit-rate clearly above chance, slots qualitatively on the object.
Precision may be below the color saliency head; we only need to show the
capability exists label-free.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
sys.path.insert(0, str(ROOT))
from baselines.ppo.modules.frozen_vit import FrozenDualDinoV2
from analysis.slot.slot_dinosaur import DinosaurSlots

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
    torch.backends.cudnn.benchmark = False


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
def extract_features(vit, data_dir, grid, input_size):
    """Returns list of dicts with feats (N,d), base_rgb, color_idx per frame."""
    files = sorted(Path(data_dir).glob("ep*.npz"))
    frames = []
    for fp in files:
        d = np.load(fp)
        base = d["base_rgb"]            # (T,128,128,3)
        hand = d["hand_rgb"]
        color = int(d["color_idx"])
        rgb6 = torch.from_numpy(np.concatenate([base, hand], axis=-1)).to(DEVICE)
        tok_b, tok_h, _, _ = vit(rgb6)   # (T, N, d)
        T = base.shape[0]
        for t in range(T):
            frames.append({
                "feat": tok_b[t].cpu(),      # base camera only
                "base": base[t],
                "color": color,
            })
    return frames


def train_slots(model, feats_tensor, epochs=60, bs=128, lr=2e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = feats_tensor.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(N)
        tot = 0.0
        for s in range(0, N, bs):
            idx = perm[s:s+bs]
            x = feats_tensor[idx].to(DEVICE)
            loss = model.loss(x)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * x.size(0)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  ep{ep:02d} recon_mse={tot/N:.4f}")


@torch.no_grad()
def evaluate(model, vit, frames, grid, iou_thresh=0.25):
    """For each frame with a visible target, compute best-slot IoU vs GT patches."""
    Hp, Wp = grid
    best_ious = []
    best_recalls = []
    hits = []
    n_eval = 0
    per_frame = []
    for fr in frames:
        gt = patch_gt(color_mask(fr["base"], fr["color"]), grid)  # (N,)
        if gt.sum() == 0:
            continue  # target not visible in this frame
        n_eval += 1
        feat = fr["feat"].unsqueeze(0).to(DEVICE)
        out = model(feat)
        masks = out["masks"][0].cpu().numpy()    # (K, N)
        hard = masks.argmax(axis=0)              # (N,) slot id per patch
        K = masks.shape[0]
        # best slot by IoU
        best = 0.0; best_k = -1; best_recall = 0.0
        for k in range(K):
            slot_patches = (hard == k)
            inter = (slot_patches & gt).sum()
            union = (slot_patches | gt).sum()
            iou = inter / max(union, 1)
            recall = inter / max(gt.sum(), 1)   # fraction of GT covered by this slot
            if iou > best:
                best = iou; best_k = k
            if recall > best_recall:
                best_recall = recall
        best_ious.append(best)
        best_recalls.append(best_recall)
        # "hit" = some single slot covers >=50% of the target patches
        hits.append(1.0 if best_recall >= 0.5 else 0.0)
        per_frame.append((fr, hard, best_k, best))
    best_ious = np.array(best_ious)
    best_recalls = np.array(best_recalls)
    hits = np.array(hits)
    return {
        "n_eval_frames": n_eval,
        "mean_best_iou": float(best_ious.mean()) if n_eval else float("nan"),
        "hit_rate_iou": float((best_ious > iou_thresh).mean()) if n_eval else float("nan"),
        "mean_best_recall": float(best_recalls.mean()) if n_eval else float("nan"),
        "hit_rate_recall50": float(hits.mean()) if n_eval else float("nan"),
        "iou_thresh": iou_thresh,
    }, per_frame


def save_overlays(per_frame, grid, out_dir, n=12):
    Hp, Wp = grid
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # pick frames with highest IoU to show the capability
    per_frame_sorted = sorted(per_frame, key=lambda x: -x[3])[:n]
    for i, (fr, hard, best_k, iou) in enumerate(per_frame_sorted):
        base = fr["base"]
        gt = patch_gt(color_mask(fr["base"], fr["color"]), grid).reshape(Hp, Wp)
        best_slot_mask = (hard == best_k).reshape(Hp, Wp)
        fig, ax = plt.subplots(1, 3, figsize=(11, 4))
        ax[0].imshow(base); ax[0].set_title("frame"); ax[0].axis("off")
        ax[1].imshow(base)
        gt_up = np.array(Image.fromarray((gt*255).astype(np.uint8)).resize((128,128), Image.NEAREST))/255.
        ax[1].imshow(gt_up, cmap="Greens", alpha=0.5)
        ax[1].set_title(f"GT object (color {fr['color']})"); ax[1].axis("off")
        ax[2].imshow(base)
        bs_up = np.array(Image.fromarray((best_slot_mask*255).astype(np.uint8)).resize((128,128), Image.NEAREST))/255.
        ax[2].imshow(bs_up, cmap="Reds", alpha=0.5)
        ax[2].set_title(f"best slot #{best_k}  IoU={iou:.2f}"); ax[2].axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"overlay_{i:02d}_iou{iou:.2f}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
    print(f"  saved {len(per_frame_sorted)} overlays -> {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="RememberColor9-v0")
    p.add_argument("--num-slots", type=int, default=6)
    p.add_argument("--slot-dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--iou-thresh", type=float, default=0.25)
    p.add_argument("--input-size", type=int, default=126,
                   help="DINOv2 input size (multiple of 14). 126->9x9, 224->16x16, 322->23x23")
    args = p.parse_args()
    set_seed(0)

    data_dir = ROOT / "analysis/ebm/path_a_data" / args.task
    vit = FrozenDualDinoV2(input_size=args.input_size).to(DEVICE).eval()
    grid = vit.grid
    N_patch = vit.num_patches_per_view
    print(f"[stage0] task={args.task} grid={grid} N_patch={N_patch} d_vit={vit.dim}")

    print("[stage0] extracting frozen DINOv2 features...")
    frames = extract_features(vit, data_dir, grid, vit.input_size)
    feats = torch.stack([f["feat"] for f in frames])  # (M, N, d)
    print(f"  {len(frames)} frames, feats {tuple(feats.shape)}")

    model = DinosaurSlots(feat_dim=vit.dim, slot_dim=args.slot_dim,
                          num_slots=args.num_slots, num_patches=N_patch).to(DEVICE)
    print(f"[stage0] training slots (K={args.num_slots}) unsupervised...")
    train_slots(model, feats, epochs=args.epochs)

    print("[stage0] evaluating object capture vs color GT...")
    metrics, per_frame = evaluate(model, vit, frames, grid, args.iou_thresh)
    print(f"  n_eval_frames={metrics['n_eval_frames']}")
    print(f"  mean_best_iou={metrics['mean_best_iou']:.3f}")
    print(f"  hit_rate(IoU>{args.iou_thresh})={metrics['hit_rate_iou']:.3f}")
    print(f"  mean_best_recall={metrics['mean_best_recall']:.3f}")
    print(f"  hit_rate(recall>=0.5)={metrics['hit_rate_recall50']:.3f}")

    out_dir = ROOT / "analysis/slot/stage0_vis" / args.task
    save_overlays(per_frame, grid, out_dir)

    import json
    res_path = ROOT / "analysis/slot" / f"stage0_{args.task}.json"
    with open(res_path, "w") as f:
        json.dump({**metrics, "task": args.task, "num_slots": args.num_slots}, f, indent=2)
    print(f"[stage0] metrics saved -> {res_path}")


if __name__ == "__main__":
    main()
