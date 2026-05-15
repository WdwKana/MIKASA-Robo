"""
Path A v2 — Upper-bound feasibility test:
  Use direct per-patch color-matching labels (not MIL) for RememberColor9.
  If the head can't produce clean saliency even with this clean signal,
  DINOv2 features themselves are insufficient for the task.

Per-patch label: a patch is "positive" iff >= 5% of its pixels match the
target color RGB (within Euclidean distance threshold in 0-255 space).

Trains a 2-layer MLP head, evaluates on demo frames.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
DATA_DIR = ROOT / "analysis/ebm/path_a_data/RememberColor9-v0"
FEAT_PATH = ROOT / "analysis/ebm/path_a_features.pt"
HEAD_PATH = ROOT / "analysis/ebm/path_a_head_v2.pt"
OUT_VIS = ROOT / "analysis/ebm/path_a_vis_v2"
MODEL_NAME = "facebook/dinov2-small"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 8


def set_seed(seed: int = 0):
    """Set seeds for full reproducibility of head training."""
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# MIKASA RememberColor RGB targets (matches RememberColorInfoWrapper.colors_names)
MIKASA_COLORS = {
    0: (255, 0,   0),   # Red
    1: (0,   255, 0),   # Lime
    2: (0,   0,   255), # Blue
    3: (255, 255, 0),   # Yellow
    4: (255, 0,   255), # Magenta
    5: (0,   255, 255), # Cyan
    6: (128, 0,   0),   # Maroon
    7: (128, 128, 0),   # Olive
    8: (0,   128, 128), # Teal
}
COLOR_THR = 60        # pixel L2 distance threshold (0..441)
PATCH_FRAC = 0.05     # patch is positive if >= 5% of its pixels match


# ─── Per-patch color label ──────────────────────────────────────────────────

def color_match_mask(rgb_uint8: np.ndarray, target_color: int) -> np.ndarray:
    """rgb_uint8 (H,W,3) -> bool (H,W) where pixel matches target color."""
    target = np.array(MIKASA_COLORS[target_color], dtype=np.float32)
    diff = rgb_uint8.astype(np.float32) - target
    return np.linalg.norm(diff, axis=-1) < COLOR_THR


def patch_label_from_pixels(pix_mask: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Aggregate pixel mask to grid: patch positive if mean(pixel_mask) > PATCH_FRAC."""
    H, W = pix_mask.shape
    Hp, Wp = grid
    ph, pw = H // Hp, W // Wp
    labels = np.zeros(grid, dtype=np.float32)
    for i in range(Hp):
        for j in range(Wp):
            blk = pix_mask[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            labels[i, j] = blk.mean() > PATCH_FRAC
    return labels


# ─── Build per-patch labels for all collected frames ────────────────────────

def build_patch_labels(features):
    """For every saved (frame, view), compute the per-patch positive mask using
    color-matching against the episode's target color.

    Returns:
      labels: (total_frames, 2*N_v) float32 in {0, 1}
      positive_frac per episode/step printed for sanity.
    """
    eps = features["episodes"]
    grid = features["grid"]
    Hp, Wp = grid
    Nv = Hp * Wp

    # we need to read raw rgb again — just resize to whatever DINOv2 processor used.
    # The DINOv2-S/14 default resize is 224. Use processor for consistency.
    proc = AutoImageProcessor.from_pretrained(MODEL_NAME)

    all_labels = []
    pos_count = 0; total_count = 0
    for ep in eps:
        ep_id = ep["ep_id"]
        npz = np.load(DATA_DIR / f"{ep_id}.npz")
        base, hand = npz["base_rgb"], npz["hand_rgb"]   # (T,128,128,3) uint8
        color = int(ep["color"])
        T = base.shape[0]

        # match the size used by the DINOv2 processor (224x224, patch=14 -> grid=16x16)
        target_size = (Hp * 14, Wp * 14)  # e.g. 224x224
        ep_labels = np.zeros((T, 2 * Nv), dtype=np.float32)
        for t in range(T):
            for v_idx, img in enumerate([base[t], hand[t]]):
                pil = Image.fromarray(img).resize(target_size, Image.BILINEAR)
                arr = np.array(pil)
                pix_mask = color_match_mask(arr, color)
                lbl = patch_label_from_pixels(pix_mask, grid).reshape(-1)
                ep_labels[t, v_idx*Nv:(v_idx+1)*Nv] = lbl
        all_labels.append(torch.from_numpy(ep_labels))
        ep_pos = ep_labels.sum(); ep_total = ep_labels.size
        pos_count += ep_pos; total_count += ep_total
        print(f"  {ep_id}: color={color} T={T}  positive_patch_frac={ep_pos/ep_total:.4f}")

    print(f"[labels] global positive fraction: {pos_count/total_count:.4f}")
    return all_labels


# ─── Head (binary per-patch classifier) ─────────────────────────────────────

class SaliencyHead(nn.Module):
    def __init__(self, patch_dim, xy_dim=32, hidden=(256, 128)):
        super().__init__()
        self.xy_dim = xy_dim
        self.mlp = nn.Sequential(
            nn.Linear(patch_dim + xy_dim, hidden[0]), nn.GELU(),
            nn.Linear(hidden[0], hidden[1]), nn.GELU(),
            nn.Linear(hidden[1], 1),
        )

    def make_xy_embed(self, grid, view_id, device):
        Hp, Wp = grid
        ys = torch.linspace(-1, 1, Hp, device=device)
        xs = torch.linspace(-1, 1, Wp, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        feats = []
        for k in range(self.xy_dim // 6):
            f = 2 ** k
            feats += [torch.sin(f*gy), torch.cos(f*gy), torch.sin(f*gx), torch.cos(f*gx)]
        feats += [torch.full_like(gy, float(view_id)),
                  torch.full_like(gy, 1 - float(view_id))]
        emb = torch.stack(feats, dim=-1).view(Hp*Wp, -1)
        if emb.shape[-1] < self.xy_dim:
            pad = torch.zeros(emb.shape[0], self.xy_dim - emb.shape[-1], device=device)
            emb = torch.cat([emb, pad], dim=-1)
        return emb[:, :self.xy_dim]

    def forward(self, tokens, xy_embed):
        N, d = tokens.shape[-2:]
        xy = xy_embed.unsqueeze(0).expand(*tokens.shape[:-2], N, -1)
        x = torch.cat([tokens, xy], dim=-1)
        return self.mlp(x).squeeze(-1)


# ─── Train ──────────────────────────────────────────────────────────────────

def train(features, labels):
    eps = features["episodes"]
    patch_dim = features["patch_dim"]
    grid = features["grid"]
    Hp, Wp = grid; Nv = Hp * Wp

    rng = np.random.default_rng(0)
    idx = np.arange(len(eps)); rng.shuffle(idx)
    n_train = int(0.8 * len(eps))
    tr_idx = idx[:n_train]; va_idx = idx[n_train:]

    head = SaliencyHead(patch_dim).to(DEVICE)
    base_xy = head.make_xy_embed(grid, view_id=0, device=DEVICE)
    hand_xy = head.make_xy_embed(grid, view_id=1, device=DEVICE)
    xy_concat = torch.cat([base_xy, hand_xy], dim=0)

    def episode_to_frames(idx_list):
        toks, lbls = [], []
        for i in idx_list:
            base = eps[i]["base_feat"]; hand = eps[i]["hand_feat"]
            cat = torch.cat([base, hand], dim=1)
            toks.append(cat); lbls.append(labels[i])
        return torch.cat(toks, dim=0), torch.cat(lbls, dim=0)

    tr_x, tr_y = episode_to_frames(tr_idx)
    va_x, va_y = episode_to_frames(va_idx)
    print(f"[train] frames: tr={tr_x.shape[0]} va={va_x.shape[0]}  "
          f"tr_pos_frac={tr_y.mean():.4f}  va_pos_frac={va_y.mean():.4f}")

    pos_weight = torch.tensor(max(1.0, (1 - tr_y.mean()) / max(1e-6, tr_y.mean())))
    print(f"[train] pos_weight={pos_weight.item():.2f}")

    opt = torch.optim.Adam(head.parameters(), lr=3e-4)
    bs = 64; epochs = 30
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
            logits = head(x, xy_concat)
            probs = torch.sigmoid(logits)
            ap = ((probs > 0.5) == (y > 0.5)).float().mean().item()
            # patch-level recall on positive-only patches
            if (y > 0.5).any():
                rec = ((probs > 0.5) & (y > 0.5)).float().sum() / (y > 0.5).float().sum()
                rec = rec.item()
            else:
                rec = 0.0
        print(f"  ep{ep:02d}  tr_loss={loss_sum/n:.4f}  va_acc={ap:.3f}  va_pos_recall={rec:.3f}")

    torch.save({"state_dict": head.state_dict(),
                "patch_dim": patch_dim, "grid": grid, "xy_dim": head.xy_dim},
               HEAD_PATH)
    return head, xy_concat, grid


# ─── Visualize ──────────────────────────────────────────────────────────────

@torch.no_grad()
def visualize(head, xy_concat, grid):
    proc = AutoImageProcessor.from_pretrained(MODEL_NAME)
    backbone = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    Hp, Wp = grid; Nv = Hp * Wp
    OUT_VIS.mkdir(parents=True, exist_ok=True)

    tasks_steps = [
        ("RememberColor9-v0",          ["000", "005", "010", "020", "030", "040", "050"]),
        ("InterceptFast-v0",           ["000", "005", "010", "020", "030", "040", "050"]),
        ("RememberShapeAndColor3x2-v0", ["000", "005", "010", "020", "030", "040", "050"]),
    ]
    CROPS = {"base_cam": (512, 0, 640, 128), "hand_cam": (512, 128, 640, 256)}

    summary = []
    for task, steps in tasks_steps:
        for step in steps:
            src = ROOT / f"{task}_frames" / f"render_env0_step{step}.png"
            if not src.exists():
                continue
            full = Image.open(src).convert("RGB")
            base = full.crop(CROPS["base_cam"])
            hand = full.crop(CROPS["hand_cam"])
            base_inp = proc(images=base, return_tensors="pt").to(DEVICE)
            hand_inp = proc(images=hand, return_tensors="pt").to(DEVICE)
            base_feat = backbone(pixel_values=base_inp.pixel_values).last_hidden_state[:, 1:, :]
            hand_feat = backbone(pixel_values=hand_inp.pixel_values).last_hidden_state[:, 1:, :]
            cat = torch.cat([base_feat, hand_feat], dim=1)

            sal_logits = head(cat, xy_concat)[0].cpu().numpy()
            base_sal = sal_logits[:Nv].reshape(grid)
            hand_sal = sal_logits[Nv:].reshape(grid)

            order = np.argsort(-sal_logits)
            joint_top = []
            for ix in order[:TOP_K]:
                if ix < Nv:
                    joint_top.append(("base_cam", int(ix // Wp), int(ix % Wp)))
                else:
                    j = ix - Nv
                    joint_top.append(("hand_cam", int(j // Wp), int(j % Wp)))
            n_hand = sum(1 for c, _, _ in joint_top if c == "hand_cam")

            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            ax[0, 0].imshow(base); ax[0, 0].set_title("base raw"); ax[0, 0].axis("off")
            ax[0, 1].imshow(hand); ax[0, 1].set_title("hand raw"); ax[0, 1].axis("off")
            for cidx, (cam_name, raw, smap) in enumerate([("base", base, base_sal),
                                                          ("hand", hand, hand_sal)]):
                Wpx, Hpx = raw.size
                ph, pw = Hpx // Hp, Wpx // Wp
                snorm = (smap - smap.min()) / (smap.max() - smap.min() + 1e-8)
                up = np.array(Image.fromarray((snorm*255).astype(np.uint8)).resize(raw.size, Image.BILINEAR)) / 255.0
                ax[1, cidx].imshow(raw); ax[1, cidx].imshow(up, cmap="hot", alpha=0.55, vmin=0, vmax=1)
                for c, i, j in joint_top:
                    if c == ("base_cam" if cidx == 0 else "hand_cam"):
                        rect = mpatches.Rectangle((j*pw, i*ph), pw, ph,
                                                  linewidth=1.6, edgecolor="lime", facecolor="none")
                        ax[1, cidx].add_patch(rect)
                ax[1, cidx].set_title(f"{cam_name} HEADv2 + joint top-{TOP_K}")
                ax[1, cidx].axis("off")

            fig.suptitle(f"[Path A v2] {task} step{step}  hand={n_hand} base={TOP_K - n_hand}",
                         fontsize=10)
            fig.tight_layout()
            out = OUT_VIS / task / f"step{step}_overlay.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
            summary.append({"task": task, "step": int(step),
                            "n_hand": n_hand, "n_base": TOP_K - n_hand,
                            "sal_max": float(sal_logits.max()),
                            "sal_mean": float(sal_logits.mean())})
            print(f"  {task} step{step}: max={sal_logits.max():.2f} mean={sal_logits.mean():.2f} "
                  f"hand_in_top={n_hand}")

    (OUT_VIS / "_summary.json").write_text(json.dumps(summary, indent=2))


def main():
    set_seed(0)
    features = torch.load(FEAT_PATH, weights_only=False)
    labels = build_patch_labels(features)
    head, xy_concat, grid = train(features, labels)
    visualize(head, xy_concat, grid)


if __name__ == "__main__":
    main()
