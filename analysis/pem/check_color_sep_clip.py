"""Same color-separability test as check_color_sep.py but for CLIP backbone.

Question: does CLIP's per-patch feature space actually separate MIKASA colors
under L2? If sep_ratio > 1.3 (vs DINOv2's ~1.0), then v1's CLIP failure was
about RL integration, not about whether perception could see color.
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/zfsstore/user/s4176650/MIKASA-Robo")

from baselines.ppo.modules.frozen_clip import FrozenDualClip
from analysis.pem.run_stage0p import color_mask

DEVICE = torch.device("cuda")
vit = FrozenDualClip().to(DEVICE).eval()
Hp, Wp = vit.grid                                      # (8, 8) for CLIP
d_vit = vit.dim                                        # 768

data_dir = Path("/zfsstore/user/s4176650/MIKASA-Robo/analysis/ebm/path_a_data/RememberColor9-v0")
ep_files = sorted(data_dir.glob("ep*.npz"))[:10]

color_features = {c: [] for c in range(9)}

with torch.no_grad():
    for fp in ep_files:
        d = np.load(fp)
        base = d["base_rgb"]
        for t in [10, 12, 15, 20]:
            # CLIP backbone preprocesses internally; feed dual-cam (B, H, W, 6) uint8
            rgb6 = torch.from_numpy(
                np.concatenate([base[t:t+1], base[t:t+1]], axis=-1)
            ).to(DEVICE)
            tok_b, _, _, _ = vit(rgb6)                  # (1, 64, 768)
            patches = tok_b[0]                          # (64, 768)
            for c in range(9):
                mask_pix = color_mask(base[t], c)
                if mask_pix.sum() < 4: continue
                H, W = mask_pix.shape
                ph, pw = H // Hp, W // Wp
                for i in range(Hp):
                    for j in range(Wp):
                        blk = mask_pix[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                        if blk.mean() > 0.01:
                            color_features[c].append(patches[i*Wp + j].cpu().float().numpy())

print("\n=== CLIP patch features: how distinguishable are colors? ===")
print(f"{'color':<10}{'n_samples':<12}{'mean L2 to self':<20}{'mean L2 to others':<20}{'sep ratio':<10}")
sep_ratios = []
for c in range(9):
    if not color_features[c]: continue
    feats = np.stack(color_features[c])
    if len(feats) > 1:
        diff_self = feats[:, None, :] - feats[None, :, :]
        d_self = np.sqrt((diff_self**2).sum(-1))
        d_self = d_self[~np.eye(len(feats), dtype=bool)]
        mean_self = d_self.mean()
    else:
        mean_self = 0
    others = []
    for c2 in range(9):
        if c2 == c: continue
        if not color_features[c2]: continue
        feats2 = np.stack(color_features[c2])
        diff = feats[:, None, :] - feats2[None, :, :]
        others.append(np.sqrt((diff**2).sum(-1)).mean())
    mean_other = np.mean(others) if others else 0
    sep_ratio = mean_other / (mean_self + 1e-8)
    if mean_self > 0: sep_ratios.append(sep_ratio)
    print(f"{c:<10}{len(feats):<12}{mean_self:<20.3f}{mean_other:<20.3f}{sep_ratio:<10.3f}")
print(f"\nMean sep_ratio (CLIP) = {np.mean(sep_ratios):.3f}")
print(f"(DINOv2 baseline was ~1.00 → at-or-below random)")
