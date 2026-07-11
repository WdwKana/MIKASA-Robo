"""Color-separability probe for SAM image encoder.

SAM is trained on segmentation, which requires color discriminability at the
encoder. If SAM patch features show sep_ratio >> 1 on MIKASA colors, that
identifies a backbone that *natively* solves what MV-SPLIT bolts on.
If SAM is also ~1.0, then per-patch-L2 is the wrong metric for ALL SSL/large-
scale-trained ViTs and MV-SPLIT is the right answer.

Downloads facebook/sam-vit-base if not cached (~360MB).
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/zfsstore/user/s4176650/MIKASA-Robo")

from transformers import SamModel, SamProcessor
from analysis.pem.run_stage0p import color_mask

DEVICE = torch.device("cuda")

print("Loading SAM (facebook/sam-vit-base)...")
model = SamModel.from_pretrained("facebook/sam-vit-base").to(DEVICE).eval()
# Use only the vision encoder
image_encoder = model.vision_encoder

# SAM's encoder takes (B, 3, 1024, 1024), outputs (B, 256, 64, 64) feature grid.
# We'll feed 128x128 → upsample to 1024 → encoder → 64x64 grid.
# For our purpose, we'll just take a smaller central feature region.

SAM_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
SAM_STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

@torch.no_grad()
def sam_patches(base_rgb_uint8):
    """base_rgb_uint8: (128, 128, 3) uint8 → (Hp*Wp, d) patch features.

    SAM encoder produces a 64x64x256 feature map for 1024x1024 input.
    We resize 128->1024 (8x upsample), get 64x64 features (each = ~16px in
    upsampled space = 2 original px). Downsample feature grid to 9x9 to
    match DINOv2 grid for comparability.
    """
    x = torch.from_numpy(base_rgb_uint8).to(DEVICE).float() / 255.0   # (128,128,3)
    x = x.permute(2, 0, 1).unsqueeze(0)                                # (1,3,128,128)
    x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
    x = (x - SAM_MEAN) / SAM_STD
    out = image_encoder(pixel_values=x)
    fmap = out.last_hidden_state                                        # (1, 256, 64, 64)
    # downsample to 9x9 grid via adaptive avg pool
    fmap = F.adaptive_avg_pool2d(fmap, (9, 9))                          # (1, 256, 9, 9)
    return fmap.squeeze(0).permute(1, 2, 0).reshape(-1, 256)            # (81, 256)


data_dir = Path("/zfsstore/user/s4176650/MIKASA-Robo/analysis/ebm/path_a_data/RememberColor9-v0")
ep_files = sorted(data_dir.glob("ep*.npz"))[:10]
color_features = {c: [] for c in range(9)}

for fp in ep_files:
    d = np.load(fp)
    base = d["base_rgb"]
    for t in [10, 12, 15, 20]:
        feats = sam_patches(base[t]).cpu().float().numpy()             # (81, 256)
        for c in range(9):
            mask_pix = color_mask(base[t], c)
            if mask_pix.sum() < 4: continue
            H, W = mask_pix.shape
            Hp = Wp = 9
            ph, pw = H // Hp, W // Wp
            for i in range(Hp):
                for j in range(Wp):
                    blk = mask_pix[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                    if blk.mean() > 0.01:
                        color_features[c].append(feats[i*Wp + j])

print("\n=== SAM patch features: how distinguishable are colors? ===")
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
print(f"\nMean sep_ratio (SAM) = {np.mean(sep_ratios):.3f}")
print("(DINOv2 ≈ 1.00, CLIP ≈ 1.07 — both color-blind under L2)")
