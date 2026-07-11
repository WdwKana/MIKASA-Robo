"""Check: in DINOv2 feature space, how similar are patches of DIFFERENT colors
vs DIFFERENT shapes? If colors are highly similar, SRB can't distinguish them."""
import sys
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, "/zfsstore/user/s4176650/MIKASA-Robo")

from baselines.ppo.modules.frozen_vit import FrozenDualDinoV2
from analysis.pem.run_stage0p import color_mask, MIKASA_COLORS

DEVICE = torch.device("cuda")
vit = FrozenDualDinoV2().to(DEVICE).eval()

# Load RC9 cached frames (which have multiple colored cubes per frame)
data_dir = Path("/zfsstore/user/s4176650/MIKASA-Robo/analysis/ebm/path_a_data/RememberColor9-v0")
ep_files = sorted(data_dir.glob("ep*.npz"))[:10]

# For each frame, extract patch features at known color positions
# Compare patch features of DIFFERENT colors to see how similar they are
color_features = {c: [] for c in range(9)}

with torch.no_grad():
    for fp in ep_files:
        d = np.load(fp)
        base = d["base_rgb"]
        # Use a mid-episode frame where all colors should be visible
        # Check colors at t=10 (likely all cubes visible)
        for t in [10, 12, 15, 20]:
            rgb6 = torch.from_numpy(np.concatenate([base[t:t+1], base[t:t+1]], axis=-1)).to(DEVICE)
            tok_b, _, _, _ = vit(rgb6)  # (1, 81, 384)
            patches = tok_b[0]  # (81, 384)
            # For each color, find which patches are that color
            for c in range(9):
                mask_pix = color_mask(base[t], c)  # (128, 128)
                if mask_pix.sum() < 4: continue
                # Convert to patch grid
                H, W = mask_pix.shape
                Hp = Wp = 9
                ph, pw = H // Hp, W // Wp
                for i in range(Hp):
                    for j in range(Wp):
                        blk = mask_pix[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                        if blk.mean() > 0.01:
                            color_features[c].append(patches[i*Wp + j].cpu().numpy())

# Compute pairwise distances
print("\n=== DINOv2 patch features: how distinguishable are colors? ===")
print(f"{'color':<10}{'n_samples':<12}{'mean L2 to self':<20}{'mean L2 to others':<20}{'sep ratio':<10}")
for c in range(9):
    if not color_features[c]: continue
    feats = np.stack(color_features[c])
    # self-distance: pairwise within same color
    if len(feats) > 1:
        diff_self = feats[:, None, :] - feats[None, :, :]
        d_self = np.sqrt((diff_self**2).sum(-1))
        # exclude diagonal
        d_self = d_self[~np.eye(len(feats), dtype=bool)]
        mean_self = d_self.mean()
    else:
        mean_self = 0
    # cross-distance: to OTHER colors
    others = []
    for c2 in range(9):
        if c2 == c: continue
        if not color_features[c2]: continue
        feats2 = np.stack(color_features[c2])
        diff = feats[:, None, :] - feats2[None, :, :]
        others.append(np.sqrt((diff**2).sum(-1)).mean())
    mean_other = np.mean(others) if others else 0
    sep_ratio = mean_other / (mean_self + 1e-8)
    print(f"{c:<10}{len(feats):<12}{mean_self:<20.3f}{mean_other:<20.3f}{sep_ratio:<10.3f}")
