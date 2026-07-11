"""
Supervised Contrastive training of a color-aware adapter on frozen DINOv2.

Adapter architecture: takes concat[DINOv2 patch feature, mean RGB of patch]
as input. The explicit RGB channel sidesteps the information bottleneck of
inverse-jitter SSL on frozen backbones (Plan B negative result: sep_ratio
stayed at 1.07 after 30 epochs).

Labels: derived per-patch from raw pixels via HSV-hue quantization, NOT from
env state. 16 hue bins around the color wheel + 1 "neutral" bin for low-
saturation patches. The label is a universal scene-color attribute, not a
task-specific target identity — works on any MIKASA frame (table, gripper,
cube, ball, ...). Trained over all four cached task datasets.

This generalizes beyond RC9's 9 colors: training never sees the task label
"which color is the target"; the held-out RC9 sep_ratio evaluation is a
zero-shot generalization test.

SupCon (Khosla et al. 2020): same-bin patches pull together, different-bin
push apart in the projected space.
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
N_HUE_BINS = 16
N_VAL_BINS = 3                                              # dim / mid / bright
NEUTRAL_BIN = N_HUE_BINS * N_VAL_BINS                       # = 48; total classes = 49
SAT_THRESH = 0.20


# ──────────────────────── adapter (takes [feat, rgb]) ───────────────────

class ColorAwareAdapter(nn.Module):
    """Per-patch MLP. Input = concat[DINOv2_feat (d_in), mean_RGB (3)].
    Output = L2-normalized projection (d_proj).
    """
    def __init__(self, d_in: int = 384, d_proj: int = 128, d_hidden: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_in + 3, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_proj),
        )
        self.d_in = d_in
        self.d_proj = d_proj
        self.d_hidden = d_hidden

    def forward(self, feat: torch.Tensor, mean_rgb: torch.Tensor) -> torch.Tensor:
        """feat: (B, N, d_in). mean_rgb: (B, N, 3) in [0,1]. → (B, N, d_proj) normalized."""
        x = torch.cat([feat, mean_rgb], dim=-1)
        return F.normalize(self.head(x), dim=-1)


# ──────────────────────── HSV → bin label ────────────────────────

def rgb_to_hue_bin(mean_rgb: torch.Tensor,
                   n_hue_bins: int = N_HUE_BINS,
                   n_val_bins: int = N_VAL_BINS,
                   sat_thresh: float = SAT_THRESH) -> torch.Tensor:
    """mean_rgb: (..., 3) in [0,1]. Returns long labels in [0, n_hue*n_val],
    with the last id reserved for low-saturation (neutral) patches.

    Label encoding: id = val_bin * n_hue + hue_bin
    This separates e.g. (255,0,0) and (128,0,0) which share hue but differ in V.
    """
    r, g, b = mean_rgb.unbind(dim=-1)
    cmax, _ = mean_rgb.max(dim=-1)
    cmin, _ = mean_rgb.min(dim=-1)
    diff = cmax - cmin
    val = cmax                                              # V in HSV
    sat = torch.where(cmax > 1e-6, diff / cmax.clamp(min=1e-6), torch.zeros_like(cmax))
    # hue computation (0..1)
    h = torch.zeros_like(cmax)
    mask_r = (cmax == r) & (diff > 1e-6)
    mask_g = (cmax == g) & (diff > 1e-6)
    mask_b = (cmax == b) & (diff > 1e-6)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r].clamp(min=1e-8)) % 6
    h[mask_g] = ((b[mask_g] - r[mask_g]) / diff[mask_g].clamp(min=1e-8)) + 2
    h[mask_b] = ((r[mask_b] - g[mask_b]) / diff[mask_b].clamp(min=1e-8)) + 4
    h = h / 6.0
    # 2D bin assignment: (hue, value)
    hue_idx = (h * n_hue_bins).long().clamp(0, n_hue_bins - 1)
    val_idx = (val * n_val_bins).long().clamp(0, n_val_bins - 1)
    bin_idx = val_idx * n_hue_bins + hue_idx                # 0..(n_hue*n_val - 1)
    # low-saturation → neutral bin (= n_hue * n_val)
    neutral = n_hue_bins * n_val_bins
    bin_idx = torch.where(sat < sat_thresh,
                          torch.full_like(bin_idx, neutral),
                          bin_idx)
    return bin_idx


# ──────────────────────── per-patch mean RGB (matches DINOv2 grid) ─────────

def mean_rgb_per_patch(frame_uint8: torch.Tensor,
                       grid: tuple, input_size: int, patch_size: int) -> torch.Tensor:
    """frame_uint8: (B, H_img, W_img, 3) uint8. Returns (B, N_patches, 3) in [0,1]."""
    Hp, Wp = grid
    x = frame_uint8.float() / 255.0
    x = x.permute(0, 3, 1, 2).contiguous()                  # (B, 3, H, W)
    if x.shape[-1] != input_size:
        x = F.interpolate(x, size=(input_size, input_size), mode="bilinear", align_corners=False)
    B = x.shape[0]
    x = x.reshape(B, 3, Hp, patch_size, Wp, patch_size).mean(dim=(3, 5))   # (B, 3, Hp, Wp)
    return x.permute(0, 2, 3, 1).reshape(B, Hp * Wp, 3)


# ──────────────────────── dataset ───────────────────────────────────────

class MultiTaskFrames(torch.utils.data.Dataset):
    """Pool of single-camera RGB frames from all cached task directories. No labels used."""
    def __init__(self, data_dirs: list[Path], stride: int = 2):
        self.frames = []
        for d in data_dirs:
            for fp in sorted(d.glob("ep*.npz")):
                ep = np.load(fp)
                self.frames.extend(ep["base_rgb"][::stride])
                self.frames.extend(ep["hand_rgb"][::stride])
        self.frames = np.stack(self.frames)
        print(f"[dataset] {len(self.frames)} frames pooled from {len(data_dirs)} tasks")

    def __len__(self): return len(self.frames)
    def __getitem__(self, i): return torch.from_numpy(self.frames[i])


@torch.no_grad()
def patches_from_view(vit: FrozenDualDinoV2, x_hwc: torch.Tensor) -> torch.Tensor:
    rgb6 = torch.cat([x_hwc, x_hwc], dim=-1)
    tok_b, _, _, _ = vit(rgb6)
    return tok_b


# ──────────────────────── SupCon loss ──────────────────────────────────

def supcon_loss(projections: torch.Tensor, labels: torch.Tensor,
                tau: float = 0.1, ignore_bin: int | None = None) -> torch.Tensor:
    """SupCon (Khosla 2020) over a flat set of projections.

    projections: (M, d), L2-normalized
    labels:      (M,) long
    ignore_bin:  if set, mask out anchors with this label (e.g., neutral patches)

    For each anchor i, positives = j s.t. labels[j]==labels[i] (i != j),
    negatives = all other j (j != i). Returns mean loss over anchors that
    have at least one positive.
    """
    M, d = projections.shape
    # similarity matrix
    sim = projections @ projections.T / tau                 # (M, M)
    sim.fill_diagonal_(-1e9)                                # no self-positive
    # positive mask: same label, not self, not ignored
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(M, dtype=torch.bool, device=sim.device)
    pos_mask = same & ~eye
    if ignore_bin is not None:
        valid_anchor = labels != ignore_bin
        pos_mask &= valid_anchor.unsqueeze(1)               # ignore neutral anchors
    # for each anchor, has positives?
    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return torch.zeros((), device=sim.device, requires_grad=True)
    # log-softmax over all non-self entries
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # mean log-prob of positives
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_mask.float().sum(dim=1).clamp(min=1)
    return -mean_log_prob_pos[has_pos].mean()


# ──────────────────────── training loop ────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dirs", nargs="+", default=[
        "analysis/ebm/path_a_data/RememberColor9-v0",
        "analysis/ebm/path_a_data/RememberColor5-v0",
        "analysis/ebm/path_a_data/RememberShape5-v0",
        "analysis/ebm/path_a_data/InterceptMedium-v0",
    ])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--d-proj", type=int, default=128)
    p.add_argument("--d-hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="analysis/color_head/adapter_supcon.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    vit = FrozenDualDinoV2().to(DEVICE).eval()
    Hp, Wp = vit.grid
    adapter = ColorAwareAdapter(d_in=vit.dim, d_proj=args.d_proj, d_hidden=args.d_hidden).to(DEVICE)
    print(f"[model] DINOv2-{vit.dim}d frozen + adapter ({sum(q.numel() for q in adapter.parameters())} params)")
    print(f"[label] {N_HUE_BINS} hue × {N_VAL_BINS} val bins + 1 neutral (sat<{SAT_THRESH}) = {NEUTRAL_BIN+1} classes")

    ds = MultiTaskFrames([ROOT / d for d in args.data_dirs])
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                          shuffle=True, num_workers=2,
                                          drop_last=True)

    optim = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    n_total_bins = N_HUE_BINS * N_VAL_BINS + 1
    for ep in range(args.epochs):
        running = 0.0; n_batches = 0; bin_counts = torch.zeros(n_total_bins, dtype=torch.long)
        for x in loader:
            x = x.to(DEVICE)                                # (B, H, W, 3) uint8
            with torch.no_grad():
                feat = patches_from_view(vit, x)            # (B, N, d_vit)
                rgb = mean_rgb_per_patch(x, (Hp, Wp), vit.input_size, vit.patch_size)
                labels = rgb_to_hue_bin(rgb)                # (B, N) long
            proj = adapter(feat, rgb)                       # (B, N, d_proj)
            # flatten batch+patch for SupCon
            B, N, d = proj.shape
            flat_proj = proj.reshape(B * N, d)
            flat_lbl = labels.reshape(B * N)
            # subsample to control compute (M=512 is enough for SupCon)
            M = min(B * N, 768)
            sel = torch.randperm(B * N, device=DEVICE)[:M]
            loss = supcon_loss(flat_proj[sel], flat_lbl[sel],
                                tau=args.tau, ignore_bin=NEUTRAL_BIN)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item(); n_batches += 1
            bin_counts += torch.bincount(flat_lbl.cpu(), minlength=n_total_bins)
        sched.step()
        # bin distribution
        bc = bin_counts.tolist()
        hue_total = sum(bc[:N_HUE_BINS * N_VAL_BINS]); neutral = bc[NEUTRAL_BIN]
        print(f"[ep {ep+1:02d}/{args.epochs}] loss={running/n_batches:.4f} "
              f"hue/neutral={hue_total}/{neutral} lr={sched.get_last_lr()[0]:.2e}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "adapter_state_dict": adapter.state_dict(),
        "d_in": vit.dim, "d_proj": args.d_proj, "d_hidden": args.d_hidden,
        "trained_on": args.data_dirs,
        "n_hue_bins": N_HUE_BINS, "n_val_bins": N_VAL_BINS,
        "sat_thresh": SAT_THRESH,
        "tau": args.tau, "epochs": args.epochs, "lr": args.lr,
    }, out)
    print(f"[save] adapter → {out}")


if __name__ == "__main__":
    main()
