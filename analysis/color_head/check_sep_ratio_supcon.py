"""sep_ratio check for SupCon-hue-trained adapter.

The adapter takes [DINOv2_feat, mean_RGB] as input. We test L2 separability
on RC9's 9 task colors (a zero-shot generalization slice — training labels
were 16 hue bins derived from ALL pixels, not these specific 9 colors).
"""
import sys, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
sys.path.insert(0, str(ROOT))

from baselines.ppo.modules.frozen_vit import FrozenDualDinoV2
from analysis.pem.run_stage0p import color_mask
from analysis.color_head.train_supcon_hue import ColorAwareAdapter, mean_rgb_per_patch

DEVICE = torch.device("cuda")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="analysis/color_head/adapter_supcon.pt")
    ap.add_argument("--mode", choices=["adapter_only", "concat"], default="adapter_only")
    args = ap.parse_args()

    vit = FrozenDualDinoV2().to(DEVICE).eval()
    ck = torch.load(ROOT / args.ckpt, map_location=DEVICE)
    adapter = ColorAwareAdapter(d_in=ck["d_in"], d_proj=ck["d_proj"],
                                 d_hidden=ck.get("d_hidden", 256)).to(DEVICE).eval()
    adapter.load_state_dict(ck["adapter_state_dict"])
    for p in adapter.parameters(): p.requires_grad_(False)

    Hp, Wp = vit.grid
    data_dir = ROOT / "analysis/ebm/path_a_data/RememberColor9-v0"
    ep_files = sorted(data_dir.glob("ep*.npz"))[:10]
    color_features = {c: [] for c in range(9)}

    with torch.no_grad():
        for fp in ep_files:
            d = np.load(fp)
            base = d["base_rgb"]
            for t in [10, 12, 15, 20]:
                rgb6 = torch.from_numpy(np.concatenate([base[t:t+1], base[t:t+1]], axis=-1)).to(DEVICE)
                tok_b, _, _, _ = vit(rgb6)                  # (1, 81, 384)
                feat = tok_b                                 # (1, 81, 384)
                rgb_patches = mean_rgb_per_patch(
                    torch.from_numpy(base[t:t+1]).to(DEVICE),
                    (Hp, Wp), vit.input_size, vit.patch_size)
                proj = adapter(feat, rgb_patches)[0]        # (81, d_proj)
                if args.mode == "adapter_only":
                    out = proj
                else:                                         # concat
                    out = torch.cat([F.normalize(feat[0], dim=-1), proj], dim=-1)

                for c in range(9):
                    mask_pix = color_mask(base[t], c)
                    if mask_pix.sum() < 4: continue
                    H, W = mask_pix.shape
                    ph, pw = H // Hp, W // Wp
                    for i in range(Hp):
                        for j in range(Wp):
                            blk = mask_pix[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                            if blk.mean() > 0.01:
                                color_features[c].append(out[i*Wp + j].cpu().float().numpy())

    print(f"\n=== SupCon adapter ({args.mode}) — RC9 color sep_ratio ===")
    print(f"{'color':<10}{'n':<8}{'L2 self':<14}{'L2 others':<14}{'sep ratio':<10}")
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
            if c2 == c or not color_features[c2]: continue
            feats2 = np.stack(color_features[c2])
            diff = feats[:, None, :] - feats2[None, :, :]
            others.append(np.sqrt((diff**2).sum(-1)).mean())
        mean_other = np.mean(others) if others else 0
        sep_ratio = mean_other / (mean_self + 1e-8)
        if mean_self > 0: sep_ratios.append(sep_ratio)
        print(f"{c:<10}{len(feats):<8}{mean_self:<14.3f}{mean_other:<14.3f}{sep_ratio:<10.3f}")
    print(f"\nMean sep_ratio ({args.mode}) = {np.mean(sep_ratios):.3f}")
    print(f"  baselines: DINOv2 1.00 | CLIP 1.07 | SAM 1.04 | Inverse-Jitter 1.07")

if __name__ == "__main__":
    main()
