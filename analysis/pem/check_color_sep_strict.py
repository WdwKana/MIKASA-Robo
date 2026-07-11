"""Re-examine the DINOv2 'color-blind' claim with STRICT patch purity filters.

The original check_color_sep.py used blk.mean() > 0.01 (1% pixels match) which
allowed patches that are 99% gray table + 1% red cube. DINOv2 features of such
patches are dominated by table content, not cube color.

This script tries 3 progressively stricter purity thresholds:
  >1% (original, kept for comparison)
  >30% (mostly object)
  >60% (very pure)
  + synthetic uniform-color patches as ceiling

If sep_ratio rises sharply with purity, then DINOv2 DOES encode color — our
original test was confounded by patch composition, not a backbone limitation.
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/zfsstore/user/s4176650/MIKASA-Robo")

from baselines.ppo.modules.frozen_vit import FrozenDualDinoV2
from analysis.pem.run_stage0p import color_mask, MIKASA_COLORS

DEVICE = torch.device("cuda")
vit = FrozenDualDinoV2().to(DEVICE).eval()
Hp, Wp = vit.grid

def mean_sep(features_by_color):
    sep_ratios = []
    counts = []
    for c in range(9):
        if len(features_by_color[c]) < 2: continue
        feats = np.stack(features_by_color[c])
        counts.append(len(feats))
        diff_self = feats[:, None, :] - feats[None, :, :]
        d_self = np.sqrt((diff_self**2).sum(-1))
        d_self = d_self[~np.eye(len(feats), dtype=bool)]
        mean_self = d_self.mean()
        others = []
        for c2 in range(9):
            if c2 == c or len(features_by_color[c2]) < 1: continue
            feats2 = np.stack(features_by_color[c2])
            diff = feats[:, None, :] - feats2[None, :, :]
            others.append(np.sqrt((diff**2).sum(-1)).mean())
        mean_other = np.mean(others) if others else 0
        if mean_self > 0:
            sep_ratios.append(mean_other / mean_self)
    return float(np.mean(sep_ratios)) if sep_ratios else 0, counts

# ── Real cached frames at multiple purity thresholds ─────────────────────────

data_dir = Path("/zfsstore/user/s4176650/MIKASA-Robo/analysis/ebm/path_a_data/RememberColor9-v0")
ep_files = sorted(data_dir.glob("ep*.npz"))[:10]

for purity in [0.01, 0.30, 0.60]:
    color_feats = {c: [] for c in range(9)}
    with torch.no_grad():
        for fp in ep_files:
            d = np.load(fp)
            base = d["base_rgb"]
            for t in [10, 12, 15, 20]:
                rgb6 = torch.from_numpy(np.concatenate([base[t:t+1], base[t:t+1]], axis=-1)).to(DEVICE)
                tok_b, _, _, _ = vit(rgb6)
                patches = tok_b[0]
                H, W = 128, 128
                ph, pw = H // Hp, W // Wp
                for c in range(9):
                    mask_pix = color_mask(base[t], c)
                    if mask_pix.sum() < 4: continue
                    for i in range(Hp):
                        for j in range(Wp):
                            blk = mask_pix[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                            if blk.mean() > purity:
                                color_feats[c].append(patches[i*Wp + j].cpu().numpy())
    ratio, counts = mean_sep(color_feats)
    n_total = sum(counts)
    print(f"purity > {purity*100:>2.0f}%   n_patches per color: {counts}   total {n_total:>5}   sep_ratio = {ratio:.3f}")

# ── Synthetic uniform-color patches (ceiling: pure color input) ──────────────

print("\n── synthetic uniform-color images (ceiling test) ──")
synth_feats = {c: [] for c in range(9)}
with torch.no_grad():
    for c, rgb in MIKASA_COLORS.items():
        # 128×128 image, single solid color, fed as dual-cam pair
        img = torch.tensor(rgb, dtype=torch.uint8, device=DEVICE).view(1, 1, 1, 3).expand(1, 128, 128, 3).contiguous()
        rgb6 = torch.cat([img, img], dim=-1)
        tok_b, _, _, _ = vit(rgb6)
        # all 81 patches see the same color → take all
        for n in range(81):
            synth_feats[c].append(tok_b[0, n].cpu().numpy())
ratio_synth, counts_synth = mean_sep(synth_feats)
print(f"synthetic uniform colors  n per color: {counts_synth[0]}   sep_ratio = {ratio_synth:.3f}")
