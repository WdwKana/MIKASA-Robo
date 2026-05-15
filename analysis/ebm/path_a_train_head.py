"""
Path A — Step 2-4 in one go:
  (a) extract frozen DINOv2 features for collected RememberColor9 frames,
  (b) train a 2-layer MLP saliency head via MIL: per-patch logit -> attention
      pool -> predict 9-way target color,
  (c) visualize the learned per-patch saliency on the demo frames.

Goal: this is a feasibility test, not the final head.

If the trained head's per-patch logits clearly highlight the colored cube on
test frames, Path A is feasible.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
DATA_DIR = ROOT / "analysis/ebm/path_a_data/RememberColor9-v0"
FEAT_PATH = ROOT / "analysis/ebm/path_a_features.pt"
HEAD_PATH = ROOT / "analysis/ebm/path_a_head.pt"
OUT_VIS = ROOT / "analysis/ebm/path_a_vis"
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


# ─── (a) feature extraction ─────────────────────────────────────────────────

@torch.no_grad()
def extract_features():
    """For every saved frame, run DINOv2 on base_cam and hand_cam separately,
    save patch tokens + grid info to FEAT_PATH."""
    if FEAT_PATH.exists():
        print(f"[feat] existing -> {FEAT_PATH}")
        return torch.load(FEAT_PATH, weights_only=False)

    proc = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    files = sorted(DATA_DIR.glob("ep*.npz"))
    print(f"[feat] {len(files)} episodes -> extracting DINOv2 features")

    eps_data = []
    for fp in files:
        d = np.load(fp)
        base, hand = d["base_rgb"], d["hand_rgb"]   # (T, 128, 128, 3) uint8
        color = int(d["color_idx"])
        T = base.shape[0]

        feats_per_view = {}
        for view, imgs in [("base", base), ("hand", hand)]:
            pil_list = [Image.fromarray(imgs[t]) for t in range(T)]
            inp = proc(images=pil_list, return_tensors="pt").to(DEVICE)
            out = model(pixel_values=inp.pixel_values)
            tokens = out.last_hidden_state[:, 1:, :].cpu()      # (T, N_v, d)
            feats_per_view[view] = tokens
            grid_h = inp.pixel_values.shape[2] // model.config.patch_size
            grid_w = inp.pixel_values.shape[3] // model.config.patch_size
        eps_data.append({
            "color": color,
            "base_feat": feats_per_view["base"],   # (T, N_v, d)
            "hand_feat": feats_per_view["hand"],   # (T, N_v, d)
            "grid":     (grid_h, grid_w),
            "T":        T,
            "ep_id":    fp.stem,
        })
        print(f"  {fp.stem}: T={T}, color={color}, base={tuple(feats_per_view['base'].shape)}, hand={tuple(feats_per_view['hand'].shape)}")

    torch.save({
        "model": MODEL_NAME,
        "patch_dim": eps_data[0]["base_feat"].shape[-1],
        "grid": eps_data[0]["grid"],
        "episodes": eps_data,
    }, FEAT_PATH)
    print(f"[feat] saved -> {FEAT_PATH}")
    return torch.load(FEAT_PATH, weights_only=False)


# ─── (b) MIL training ───────────────────────────────────────────────────────

class SaliencyHead(nn.Module):
    """2-layer MLP from (patch_token, xy_embed) -> per-patch saliency logit."""
    def __init__(self, patch_dim: int, xy_dim: int = 32, hidden=(256, 128)):
        super().__init__()
        self.xy_dim = xy_dim
        self.mlp = nn.Sequential(
            nn.Linear(patch_dim + xy_dim, hidden[0]), nn.GELU(),
            nn.Linear(hidden[0], hidden[1]), nn.GELU(),
            nn.Linear(hidden[1], 1),
        )

    def make_xy_embed(self, grid: tuple[int, int], view_id: int, device):
        Hp, Wp = grid
        ys = torch.linspace(-1, 1, Hp, device=device)
        xs = torch.linspace(-1, 1, Wp, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        # encode (y, x, view_id) with sin/cos at multiple frequencies
        feats = []
        for k in range(self.xy_dim // 6):
            f = 2 ** k
            feats += [torch.sin(f*gy), torch.cos(f*gy), torch.sin(f*gx), torch.cos(f*gx)]
        feats += [torch.full_like(gy, float(view_id)),
                  torch.full_like(gy, 1 - float(view_id))]
        emb = torch.stack(feats, dim=-1).view(Hp*Wp, -1)
        # pad/trim to xy_dim
        if emb.shape[-1] < self.xy_dim:
            pad = torch.zeros(emb.shape[0], self.xy_dim - emb.shape[-1], device=device)
            emb = torch.cat([emb, pad], dim=-1)
        return emb[:, :self.xy_dim]

    def forward(self, tokens: torch.Tensor, xy_embed: torch.Tensor) -> torch.Tensor:
        """tokens: (..., N, d), xy_embed: (N, xy_dim) -> logits (..., N)"""
        N, d = tokens.shape[-2:]
        xy = xy_embed.unsqueeze(0).expand(*tokens.shape[:-2], N, -1)
        x = torch.cat([tokens, xy], dim=-1)
        return self.mlp(x).squeeze(-1)


class MILClassifier(nn.Module):
    """Saliency head + 9-way color classifier from attention-pooled patches."""
    def __init__(self, patch_dim: int, num_classes: int = 9, xy_dim: int = 32):
        super().__init__()
        self.saliency = SaliencyHead(patch_dim, xy_dim=xy_dim)
        self.cls = nn.Linear(patch_dim, num_classes)

    def forward(self, tokens, xy_embed):
        """tokens: (B, N, d), xy_embed: (N, xy_dim)
        returns: cls_logits (B, num_classes), saliency_logits (B, N)"""
        sal = self.saliency(tokens, xy_embed)   # (B, N)
        attn = sal.softmax(dim=-1).unsqueeze(-1)  # (B, N, 1)
        pooled = (attn * tokens).sum(dim=-2)      # (B, d)
        return self.cls(pooled), sal


def train(features):
    """Train per-frame: attention-pool patches across base+hand views to
    predict the episode's target color."""
    eps = features["episodes"]
    patch_dim = features["patch_dim"]
    grid = features["grid"]
    Hp, Wp = grid
    Nv = Hp * Wp

    # split episodes 80/20
    rng = np.random.default_rng(0)
    idx = np.arange(len(eps)); rng.shuffle(idx)
    n_train = int(0.8 * len(eps))
    train_eps = [eps[i] for i in idx[:n_train]]
    val_eps   = [eps[i] for i in idx[n_train:]]
    print(f"[train] {len(train_eps)} train eps, {len(val_eps)} val eps")

    head = MILClassifier(patch_dim).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=3e-4)

    # build xy_embed for both views and concat order
    base_xy = head.saliency.make_xy_embed(grid, view_id=0, device=DEVICE)
    hand_xy = head.saliency.make_xy_embed(grid, view_id=1, device=DEVICE)
    xy_concat = torch.cat([base_xy, hand_xy], dim=0)   # (2*N_v, xy_dim)

    def episode_to_frames(ep_list):
        all_tokens, all_labels = [], []
        for ep in ep_list:
            base_feat = ep["base_feat"]   # (T, N_v, d)
            hand_feat = ep["hand_feat"]
            cat = torch.cat([base_feat, hand_feat], dim=1)    # (T, 2*N_v, d)
            all_tokens.append(cat)
            all_labels.append(torch.full((cat.shape[0],), ep["color"], dtype=torch.long))
        return torch.cat(all_tokens, dim=0), torch.cat(all_labels, dim=0)

    train_tokens, train_labels = episode_to_frames(train_eps)
    val_tokens,   val_labels   = episode_to_frames(val_eps)
    print(f"[train] frames: train={train_tokens.shape[0]}, val={val_tokens.shape[0]}")

    bs = 64
    epochs = 30
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(train_tokens.shape[0])
        loss_sum = 0.0; n = 0
        for s in range(0, len(perm), bs):
            i = perm[s:s+bs]
            x = train_tokens[i].to(DEVICE)
            y = train_labels[i].to(DEVICE)
            logits, sal = head(x, xy_concat)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * x.size(0); n += x.size(0)
        head.eval()
        with torch.no_grad():
            x = val_tokens.to(DEVICE); y = val_labels.to(DEVICE)
            logits, _ = head(x, xy_concat)
            acc = (logits.argmax(-1) == y).float().mean().item()
        print(f"  ep{ep:02d}  train_loss={loss_sum/n:.4f}  val_acc={acc:.3f}")

    torch.save({
        "state_dict": head.state_dict(),
        "patch_dim":  patch_dim,
        "grid":       grid,
        "xy_dim":     head.saliency.xy_dim,
    }, HEAD_PATH)
    print(f"[train] saved -> {HEAD_PATH}")
    return head, xy_concat, grid


# ─── (c) visualize on demo frames ───────────────────────────────────────────

@torch.no_grad()
def visualize(head, xy_concat, grid):
    """Run trained head on the demo frames and visualize per-patch saliency."""
    proc = AutoImageProcessor.from_pretrained(MODEL_NAME)
    backbone = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    Hp, Wp = grid
    Nv = Hp * Wp
    OUT_VIS.mkdir(parents=True, exist_ok=True)

    # read each demo frame, crop base + hand, run features + head saliency
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
            cat = torch.cat([base_feat, hand_feat], dim=1)   # (1, 2N_v, d)

            sal_logits = head.saliency(cat, xy_concat)       # (1, 2N_v)
            sal = sal_logits[0].cpu().numpy()

            base_sal = sal[:Nv].reshape(grid)
            hand_sal = sal[Nv:].reshape(grid)

            # joint top-K across both views
            order = np.argsort(-sal)
            joint_top = []
            for idx in order[:TOP_K]:
                if idx < Nv:
                    joint_top.append(("base_cam", int(idx // Wp), int(idx % Wp)))
                else:
                    j = idx - Nv
                    joint_top.append(("hand_cam", int(j // Wp), int(j % Wp)))

            n_hand = sum(1 for c, _, _ in joint_top if c == "hand_cam")
            n_base = TOP_K - n_hand

            # render
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            ax[0, 0].imshow(base); ax[0, 0].set_title("base_cam raw"); ax[0, 0].axis("off")
            ax[0, 1].imshow(hand); ax[0, 1].set_title("hand_cam raw"); ax[0, 1].axis("off")

            for cidx, (cam_name, raw, smap) in enumerate([("base", base, base_sal), ("hand", hand, hand_sal)]):
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
                ax[1, cidx].set_title(f"{cam_name}_cam HEAD saliency + joint top-{TOP_K}")
                ax[1, cidx].axis("off")

            fig.suptitle(f"[Path A head] {task} step{step}  joint top-{TOP_K}: hand={n_hand} base={n_base}",
                         fontsize=10)
            fig.tight_layout()
            out = OUT_VIS / task / f"step{step}_overlay.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
            summary.append({"task": task, "step": int(step), "n_hand": n_hand, "n_base": n_base,
                            "sal_max": float(sal.max()), "sal_mean": float(sal.mean())})
            print(f"  {task} step{step}: max={sal.max():.2f}  mean={sal.mean():.2f}  hand_in_top={n_hand}")

    (OUT_VIS / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[vis] -> {OUT_VIS}")


def main():
    set_seed(0)
    features = extract_features()
    head, xy_concat, grid = train(features)
    visualize(head, xy_concat, grid)


if __name__ == "__main__":
    main()
