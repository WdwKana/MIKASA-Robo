"""
Inverse-Jitter SSL training for a color-aware adapter on frozen DINOv2.

Motivation: large frozen ViT backbones (DINOv2/CLIP/SAM measured at sep_ratio
≈ 1.0/1.07/1.04 on MIKASA colors) suppress color information by construction
— DINO/iBOT pretraining uses color-jitter augmentation as a positive pair,
explicitly forcing color invariance into the L2 geometry of feature space.

We *invert* this objective with a tiny frozen adapter:
  - geometric augmentation (h-flip) → positive pair  (keep DINOv2's invariance)
  - color jitter (hue/sat/bri/con)  → negative pair  (RESTORE color sensitivity)

No env labels are used. The adapter is task-agnostic and architecture-agnostic;
all downstream agents (LSTM/GRU/MLP/SRB-TR) consume the same augmented features
without modification.

Trained on raw RGB frames cached from MIKASA tasks (color labels in the .npz
are NOT touched). Adapter is per-patch MLP applied to DINOv2 patch tokens.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
sys.path.insert(0, str(ROOT))

from baselines.ppo.modules.frozen_vit import FrozenDualDinoV2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────── adapter architecture ────────────────────────

class ColorAwareAdapter(nn.Module):
    """Per-patch MLP. Input: DINOv2 patch feat (d_in). Output: color-aware
    projection (d_proj), L2-normalized for cosine/L2 inner consistency.
    """
    def __init__(self, d_in: int = 384, d_proj: int = 128, d_hidden: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_proj),
        )
        self.d_proj = d_proj

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, N, d_in) or (..., d_in). Returns L2-normalized projection."""
        x = self.head(feat)
        return F.normalize(x, dim=-1)


# ──────────────────────── dataset ────────────────────────

class CachedFramesDataset(torch.utils.data.Dataset):
    """Sample random RGB frames from cached eval rollouts. No labels used."""
    def __init__(self, data_dirs: list[Path]):
        self.frames = []
        for d in data_dirs:
            for fp in sorted(d.glob("ep*.npz")):
                ep = np.load(fp)
                base = ep["base_rgb"]                       # (T, 128, 128, 3)
                hand = ep["hand_rgb"]
                # subsample every-2 frames to keep dataset compact
                self.frames.extend(base[::2])
                self.frames.extend(hand[::2])
        self.frames = np.stack(self.frames)                 # (N, 128, 128, 3) uint8
        print(f"[dataset] {len(self.frames)} frames loaded")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        x = self.frames[i]                                  # (128, 128, 3) uint8
        return torch.from_numpy(x)


# ──────────────────────── augmentation ────────────────────────

def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """rgb: (B, 3, H, W) in [0,1]. Returns hsv in [0,1]."""
    r, g, b = rgb.unbind(dim=1)
    cmax, _ = rgb.max(dim=1)
    cmin, _ = rgb.min(dim=1)
    diff = cmax - cmin
    h = torch.zeros_like(cmax)
    mask_r = (cmax == r) & (diff > 0)
    mask_g = (cmax == g) & (diff > 0)
    mask_b = (cmax == b) & (diff > 0)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r].clamp(min=1e-8)) % 6
    h[mask_g] = ((b[mask_g] - r[mask_g]) / diff[mask_g].clamp(min=1e-8)) + 2
    h[mask_b] = ((r[mask_b] - g[mask_b]) / diff[mask_b].clamp(min=1e-8)) + 4
    h = h / 6.0
    s = torch.where(cmax > 0, diff / cmax.clamp(min=1e-8), torch.zeros_like(cmax))
    v = cmax
    return torch.stack([h, s, v], dim=1)


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    h, s, v = hsv.unbind(dim=1)
    h6 = (h * 6.0) % 6.0
    i = h6.floor().to(torch.long)
    f = h6 - h6.floor()
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r = torch.empty_like(v); g = torch.empty_like(v); b = torch.empty_like(v)
    for k, (rk, gk, bk) in enumerate([(v, t, p), (q, v, p), (p, v, t),
                                       (p, q, v), (t, p, v), (v, p, q)]):
        mask = i == k
        r = torch.where(mask, rk, r)
        g = torch.where(mask, gk, g)
        b = torch.where(mask, bk, b)
    return torch.stack([r, g, b], dim=1).clamp(0, 1)


class JitterPair:
    """Generate (geom_pair, color_neg) for inverse-jitter SSL.

    geom: horizontal flip with prob 0.5 (alignment via known patch permutation).
    color: heavy hue rotation + saturation + brightness + contrast, no geometry.
           Implemented in-place to avoid torchvision dependency.
    """
    def __init__(self,
                 brightness=0.5, contrast=0.5, saturation=0.8, hue_range=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue_range = hue_range

    def _color_jitter(self, x_chw: torch.Tensor) -> torch.Tensor:
        """x_chw: (B, 3, H, W) float in [0,1]. Returns jittered."""
        B = x_chw.shape[0]
        device = x_chw.device
        # Brightness: multiply
        b_factor = 1.0 + (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * self.brightness
        x = (x_chw * b_factor).clamp(0, 1)
        # Contrast: scale around per-image mean
        mean = x.mean(dim=(2, 3), keepdim=True)
        c_factor = 1.0 + (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * self.contrast
        x = ((x - mean) * c_factor + mean).clamp(0, 1)
        # Hue + saturation in HSV
        hsv = _rgb_to_hsv(x)                                        # (B, 3, H, W)
        h_shift = (torch.rand(B, 1, 1, device=device) * 2 - 1) * self.hue_range
        s_factor = 1.0 + (torch.rand(B, 1, 1, device=device) * 2 - 1) * self.saturation
        hsv[:, 0] = (hsv[:, 0] + h_shift) % 1.0
        hsv[:, 1] = (hsv[:, 1] * s_factor).clamp(0, 1)
        x = _hsv_to_rgb(hsv)
        return x

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, 128, 128, 3) uint8.
        Returns (x_clean, x_geom, x_color, flip_mask).
        """
        B = x.shape[0]
        x_clean = x.clone()
        # geometric branch: independent per-sample h-flip
        flip_mask = torch.rand(B) < 0.5
        x_geom = x.clone()
        x_geom[flip_mask] = torch.flip(x_geom[flip_mask], dims=[-2])    # flip W axis

        # color branch
        x_color_chw = x.float().permute(0, 3, 1, 2) / 255.0
        x_color_chw = self._color_jitter(x_color_chw)
        x_color = (x_color_chw * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()

        return x_clean, x_geom, x_color, flip_mask


# ──────────────────────── DINOv2 patch features ────────────────────────

@torch.no_grad()
def patches_from_view(vit: FrozenDualDinoV2, x_hwc: torch.Tensor) -> torch.Tensor:
    """Forward a single-camera RGB image (B, H, W, 3) uint8 → patches (B, N, d).

    FrozenDualDinoV2 expects 6-channel dual-cam input. We duplicate the single
    view into both base & hand slots, then only use the base-tokens output.
    """
    rgb6 = torch.cat([x_hwc, x_hwc], dim=-1)                # (B, H, W, 6)
    tok_b, _, _, _ = vit(rgb6)                              # (B, N, d)
    return tok_b


# ──────────────────────── flip-permutation for patch alignment ────────────────

def horizontal_flip_perm(Hp: int, Wp: int, device) -> torch.Tensor:
    """For a patch grid of size (Hp, Wp), return a permutation of N=Hp*Wp
    indices that re-aligns patches after a horizontal image flip.
    """
    idx = torch.arange(Hp * Wp, device=device).reshape(Hp, Wp)
    return idx.flip(-1).reshape(-1)                         # (N,)


# ──────────────────────── InfoNCE loss ─────────────────────────────────

def infonce_per_patch(anchor: torch.Tensor,
                      positive: torch.Tensor,
                      color_neg: torch.Tensor,
                      tau: float = 0.1) -> torch.Tensor:
    """
    Per-patch InfoNCE.
      anchor    : (B, N, d) clean features
      positive  : (B, N, d) geom-augmented features (flip-realigned)
      color_neg : (B, N, d) color-jittered features at SAME patch position

    For each (b, n):
      pos = sim(anchor[b,n], positive[b,n])
      neg_color = sim(anchor[b,n], color_neg[b,n])
      neg_cross = sim(anchor[b,n], anchor[b', n']) for all (b',n') != (b,n) in batch
                   (uses random subsample for tractability)
      Loss = -log( exp(pos/τ) / (exp(pos/τ) + exp(neg_color/τ) + Σ exp(neg_cross/τ)) )
    """
    B, N, d = anchor.shape
    # all features pre-normalized (adapter does it)
    pos_sim = (anchor * positive).sum(-1)                   # (B, N)
    color_neg_sim = (anchor * color_neg).sum(-1)            # (B, N)

    # cross-sample negatives: flatten anchor as bank, sample K patches per anchor
    K = 64
    bank = anchor.reshape(B * N, d)                          # (B*N, d)
    rand_idx = torch.randint(0, B * N, (B, N, K), device=anchor.device)
    cross_neg = bank[rand_idx]                               # (B, N, K, d)
    cross_neg_sim = (anchor.unsqueeze(2) * cross_neg).sum(-1)  # (B, N, K)

    # logits: [pos, color_neg, cross_neg×K]
    logits = torch.cat([pos_sim.unsqueeze(-1),
                         color_neg_sim.unsqueeze(-1),
                         cross_neg_sim], dim=-1) / tau       # (B, N, 2+K)
    labels = torch.zeros(B * N, dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits.reshape(B * N, -1), labels)


# ──────────────────────── training loop ────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dirs", nargs="+", default=[
        "analysis/ebm/path_a_data/RememberColor9-v0",
        "analysis/ebm/path_a_data/RememberColor5-v0",
        "analysis/ebm/path_a_data/RememberShape5-v0",
    ])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--d-proj", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="analysis/color_head/adapter.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # frozen backbone + trainable adapter
    vit = FrozenDualDinoV2().to(DEVICE).eval()
    adapter = ColorAwareAdapter(d_in=vit.dim, d_proj=args.d_proj).to(DEVICE)
    print(f"[model] DINOv2-{vit.dim}d frozen + adapter ({sum(p.numel() for p in adapter.parameters())} params)")

    Hp, Wp = vit.grid
    flip_perm = horizontal_flip_perm(Hp, Wp, DEVICE)
    N = Hp * Wp

    # data
    ds = CachedFramesDataset([ROOT / d for d in args.data_dirs])
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                          shuffle=True, num_workers=2,
                                          drop_last=True)
    jitter = JitterPair()

    optim = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    for ep in range(args.epochs):
        running = 0.0
        n_steps = 0
        for x in loader:
            x = x.to(DEVICE)                                # (B, H, W, 3) uint8
            x_clean, x_geom, x_color, flip_mask = jitter(x)

            # frozen DINOv2 → patches
            with torch.no_grad():
                f_clean = patches_from_view(vit, x_clean)   # (B, N, d_vit)
                f_geom  = patches_from_view(vit, x_geom)
                f_color = patches_from_view(vit, x_color)

            # adapter projection (trainable)
            p_clean = adapter(f_clean)                      # (B, N, d_proj)
            p_geom  = adapter(f_geom)
            p_color = adapter(f_color)

            # re-align flipped samples: for samples with flip_mask=True, the
            # geom-patches at position n correspond to clean-patches at flip_perm[n]
            p_geom_aligned = p_geom.clone()
            flip_mask_dev = flip_mask.to(DEVICE)
            if flip_mask_dev.any():
                # gather along patch axis with flip_perm for flipped samples
                idx = flip_perm.unsqueeze(0).expand(flip_mask_dev.sum(), -1)  # (n_flipped, N)
                idx_exp = idx.unsqueeze(-1).expand(-1, -1, args.d_proj)
                p_geom_aligned[flip_mask_dev] = p_geom[flip_mask_dev].gather(1, idx_exp)

            loss = infonce_per_patch(p_clean, p_geom_aligned, p_color, tau=args.tau)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item()
            n_steps += 1

        sched.step()
        print(f"[ep {ep+1:02d}/{args.epochs}] loss={running/n_steps:.4f} lr={sched.get_last_lr()[0]:.2e}")

    # save
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "adapter_state_dict": adapter.state_dict(),
        "d_in": vit.dim, "d_proj": args.d_proj,
        "trained_on": args.data_dirs,
        "tau": args.tau, "epochs": args.epochs, "lr": args.lr,
    }, out)
    print(f"[save] adapter → {out}")


if __name__ == "__main__":
    main()
