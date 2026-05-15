"""
Path A v3 — retrain saliency head for the OPTIMIZED FrozenDualDinoV2
(fp16, 126×126 input, 9×9=81 patches per view).

Same MIL-free per-patch supervision as v2 (color matching), but
features come from the new fast ViT module so the head's input
distribution matches what PPO will see at inference.

Reuses cached rollout data in analysis/ebm/path_a_data/.

Output: analysis/ebm/path_a_head_v3.pt (used by ppo_memtasks_ebm.py via
        --saliency-ckpt flag).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baselines.ppo.modules.frozen_clip_v2 import FrozenDualClipV2 as FrozenDualDinoV2  # alias to keep diff small
from baselines.ppo.modules.saliency_head import SaliencyHead

ROOT      = Path("/zfsstore/user/s4176650/MIKASA-Robo")
DATA_DIR  = ROOT / "analysis/ebm/path_a_data/RememberColor9-v0"
HEAD_PATH = ROOT / "analysis/ebm/path_a_head_v3_clip_v2.pt"
OUT_VIS   = ROOT / "analysis/ebm/path_a_vis_v3_clip_v2"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K     = 8


def set_seed(seed: int = 0):
    """Set seeds for full reproducibility of head training."""
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

MIKASA_COLORS = {
    0: (255, 0,   0),   1: (0,   255, 0),   2: (0,   0,   255),
    3: (255, 255, 0),   4: (255, 0,   255), 5: (0,   255, 255),
    6: (128, 0,   0),   7: (128, 128, 0),   8: (0,   128, 128),
}
COLOR_THR  = 60
PATCH_FRAC = 0.05


def color_match_mask(rgb_uint8: np.ndarray, target_color: int) -> np.ndarray:
    target = np.array(MIKASA_COLORS[target_color], dtype=np.float32)
    diff = rgb_uint8.astype(np.float32) - target
    return np.linalg.norm(diff, axis=-1) < COLOR_THR


def patch_label_from_pixels(pix_mask: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    H, W = pix_mask.shape
    Hp, Wp = grid
    ph, pw = H // Hp, W // Wp
    labels = np.zeros(grid, dtype=np.float32)
    for i in range(Hp):
        for j in range(Wp):
            blk = pix_mask[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            labels[i, j] = blk.mean() > PATCH_FRAC
    return labels


@torch.no_grad()
def extract_and_label(vit: FrozenDualDinoV2):
    """Extract DINOv2 features and per-patch color labels for all episodes."""
    Hp, Wp = vit.grid
    Nv = Hp * Wp
    S = vit.input_size

    files = sorted(DATA_DIR.glob("ep*.npz"))
    print(f"[feat] {len(files)} episodes, grid={vit.grid}, input={S}x{S}")

    eps = []
    for fp in files:
        d = np.load(fp)
        base, hand = d["base_rgb"], d["hand_rgb"]
        color = int(d["color_idx"])
        T = base.shape[0]

        # ViT features (uses our optimized fp16 path; output float32)
        rgb6 = torch.from_numpy(np.concatenate([base, hand], axis=-1)).to(DEVICE)
        tok_b, tok_h, cls_b, cls_h = vit(rgb6)
        # tokens: (T, Nv, d) -- already fp32

        # per-patch labels (for both views, against the SAME target color)
        labels = np.zeros((T, 2 * Nv), dtype=np.float32)
        for t in range(T):
            for v_idx, img in enumerate([base[t], hand[t]]):
                pil = Image.fromarray(img).resize((S, S), Image.BILINEAR)
                pix = color_match_mask(np.array(pil), color)
                lbl = patch_label_from_pixels(pix, (Hp, Wp)).reshape(-1)
                labels[t, v_idx*Nv:(v_idx+1)*Nv] = lbl

        eps.append({
            "ep_id":   fp.stem,
            "color":   color,
            "tok_b":   tok_b.cpu(),
            "tok_h":   tok_h.cpu(),
            "labels":  torch.from_numpy(labels),
            "T":       T,
        })
        pos_frac = labels.sum() / labels.size
        print(f"  {fp.stem}: T={T} color={color} pos_frac={pos_frac:.4f}")
    return eps, vit.grid, vit.dim


def train(eps, grid, patch_dim):
    Hp, Wp = grid
    Nv = Hp * Wp

    rng = np.random.default_rng(0)
    idx = np.arange(len(eps)); rng.shuffle(idx)
    n_train = int(0.8 * len(eps))
    tr_idx, va_idx = idx[:n_train], idx[n_train:]

    head = SaliencyHead(patch_dim=patch_dim).to(DEVICE)
    base_xy = head.make_xy_embed(grid, view_id=0, device=DEVICE)
    hand_xy = head.make_xy_embed(grid, view_id=1, device=DEVICE)
    xy_concat = torch.cat([base_xy, hand_xy], dim=0)

    def stack(idxs):
        toks, lbls = [], []
        for i in idxs:
            cat = torch.cat([eps[i]["tok_b"], eps[i]["tok_h"]], dim=1)  # (T, 2Nv, d)
            toks.append(cat); lbls.append(eps[i]["labels"])
        return torch.cat(toks, 0), torch.cat(lbls, 0)

    tr_x, tr_y = stack(tr_idx);  va_x, va_y = stack(va_idx)
    print(f"[train] tr_frames={tr_x.shape[0]} va_frames={va_x.shape[0]} "
          f"tr_pos_frac={tr_y.mean():.4f}")

    pos_weight = torch.tensor(max(1.0, (1 - tr_y.mean()) / max(1e-6, tr_y.mean())))
    opt = torch.optim.Adam(head.parameters(), lr=3e-4)
    bs, epochs = 64, 30

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(tr_x.shape[0])
        loss_sum, n = 0.0, 0
        for s in range(0, len(perm), bs):
            i = perm[s:s+bs]
            x = tr_x[i].to(DEVICE); y = tr_y[i].to(DEVICE)
            logits = head(x, xy_concat)
            loss = F.binary_cross_entropy_with_logits(logits, y,
                                                      pos_weight=pos_weight.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * x.size(0); n += x.size(0)
        head.eval()
        with torch.no_grad():
            x = va_x.to(DEVICE); y = va_y.to(DEVICE)
            probs = torch.sigmoid(head(x, xy_concat))
            acc = ((probs > 0.5) == (y > 0.5)).float().mean().item()
            rec = (((probs > 0.5) & (y > 0.5)).float().sum()
                   / max(1.0, (y > 0.5).float().sum())).item()
        print(f"  ep{ep:02d}  tr_loss={loss_sum/n:.4f}  va_acc={acc:.3f}  va_pos_recall={rec:.3f}")

    torch.save({
        "state_dict": head.state_dict(),
        "patch_dim":  patch_dim,
        "grid":       grid,
        "xy_dim":     head.xy_dim,
    }, HEAD_PATH)
    print(f"[train] saved -> {HEAD_PATH}")
    return head, xy_concat, grid


@torch.no_grad()
def visualize(vit, head, xy_concat, grid):
    """Run trained head on the demo frames; produce overlays for sanity."""
    Hp, Wp = grid
    Nv = Hp * Wp
    OUT_VIS.mkdir(parents=True, exist_ok=True)

    tasks_steps = [
        ("RememberColor9-v0",          ["000", "005", "010", "020", "030", "040", "050"]),
        ("InterceptFast-v0",           ["000", "005", "010", "020", "030", "040", "050"]),
        ("RememberShapeAndColor3x2-v0", ["000", "005", "010", "020", "030", "040", "050"]),
    ]
    CROPS = {"base_cam": (512, 0, 640, 128), "hand_cam": (512, 128, 640, 256)}

    for task, steps in tasks_steps:
        for step in steps:
            src = ROOT / f"{task}_frames" / f"render_env0_step{step}.png"
            if not src.exists():
                continue
            full = Image.open(src).convert("RGB")
            base = np.array(full.crop(CROPS["base_cam"]))
            hand = np.array(full.crop(CROPS["hand_cam"]))
            rgb6 = torch.from_numpy(np.concatenate([base[None], hand[None]], axis=-1)).to(DEVICE)
            tok_b, tok_h, _, _ = vit(rgb6)
            cat = torch.cat([tok_b, tok_h], dim=1)
            sal = head(cat, xy_concat)[0].cpu().numpy()
            base_sal = sal[:Nv].reshape(grid)
            hand_sal = sal[Nv:].reshape(grid)

            order = np.argsort(-sal)
            joint_top = []
            for ix in order[:TOP_K]:
                if ix < Nv:
                    joint_top.append(("base_cam", int(ix // Wp), int(ix % Wp)))
                else:
                    j = ix - Nv
                    joint_top.append(("hand_cam", int(j // Wp), int(j % Wp)))

            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            base_pil = Image.fromarray(base); hand_pil = Image.fromarray(hand)
            ax[0, 0].imshow(base_pil); ax[0, 0].set_title("base raw"); ax[0, 0].axis("off")
            ax[0, 1].imshow(hand_pil); ax[0, 1].set_title("hand raw"); ax[0, 1].axis("off")
            for cidx, (raw_pil, smap) in enumerate([(base_pil, base_sal), (hand_pil, hand_sal)]):
                Wpx, Hpx = raw_pil.size
                ph, pw = Hpx // Hp, Wpx // Wp
                snorm = (smap - smap.min()) / (smap.max() - smap.min() + 1e-8)
                up = np.array(Image.fromarray((snorm*255).astype(np.uint8)).resize(raw_pil.size, Image.BILINEAR)) / 255.0
                ax[1, cidx].imshow(raw_pil); ax[1, cidx].imshow(up, cmap="hot", alpha=0.55, vmin=0, vmax=1)
                cam = "base_cam" if cidx == 0 else "hand_cam"
                for c, i, j in joint_top:
                    if c == cam:
                        rect = mpatches.Rectangle((j*pw, i*ph), pw, ph,
                                                  linewidth=1.6, edgecolor="lime", facecolor="none")
                        ax[1, cidx].add_patch(rect)
                ax[1, cidx].set_title(f"{cam} HEAD-v3 saliency + joint top-{TOP_K}"); ax[1, cidx].axis("off")

            fig.suptitle(f"[Path A v3] {task} step{step}", fontsize=10)
            fig.tight_layout()
            out = OUT_VIS / task / f"step{step}_overlay.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"[vis] -> {OUT_VIS}")


def main():
    set_seed(0)
    vit = FrozenDualDinoV2().to(DEVICE).eval()
    eps, grid, patch_dim = extract_and_label(vit)
    head, xy_concat, grid = train(eps, grid, patch_dim)
    visualize(vit, head, xy_concat, grid)


if __name__ == "__main__":
    main()
