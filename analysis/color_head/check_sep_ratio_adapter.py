"""Color-separability test on adapter-augmented DINOv2 features.

Same probe as analysis/pem/check_color_sep.py but features go through the
trained inverse-jitter adapter. Expectation: sep_ratio rises from ~1.00
(raw DINOv2) to >1.5 — meaning patches of different colors are now further
apart in L2 than patches of the same color.
"""
import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
sys.path.insert(0, str(ROOT))

from baselines.ppo.modules.frozen_vit import FrozenDualDinoV2
from analysis.pem.run_stage0p import color_mask
from analysis.color_head.train_inverse_jitter import ColorAwareAdapter

DEVICE = torch.device("cuda")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="analysis/color_head/adapter.pt")
    ap.add_argument("--mode", choices=["adapter_only", "concat", "raw"], default="adapter_only",
                    help="adapter_only: test in projected space. concat: [feat,proj]. raw: DINOv2 only (sanity)")
    args = ap.parse_args()

    vit = FrozenDualDinoV2().to(DEVICE).eval()
    ck = torch.load(ROOT / args.ckpt, map_location=DEVICE)
    adapter = ColorAwareAdapter(d_in=ck["d_in"], d_proj=ck["d_proj"]).to(DEVICE).eval()
    adapter.load_state_dict(ck["adapter_state_dict"])
    for p in adapter.parameters(): p.requires_grad_(False)

    data_dir = ROOT / "analysis/ebm/path_a_data/RememberColor9-v0"
    ep_files = sorted(data_dir.glob("ep*.npz"))[:10]
    color_features = {c: [] for c in range(9)}

    Hp, Wp = vit.grid

    with torch.no_grad():
        for fp in ep_files:
            d = np.load(fp)
            base = d["base_rgb"]
            for t in [10, 12, 15, 20]:
                rgb6 = torch.from_numpy(
                    np.concatenate([base[t:t+1], base[t:t+1]], axis=-1)
                ).to(DEVICE)
                tok_b, _, _, _ = vit(rgb6)                   # (1, 81, 384)
                feat = tok_b[0]                              # (81, 384)
                proj = adapter(feat.unsqueeze(0))[0]         # (81, 128)

                if args.mode == "raw":
                    out = feat
                elif args.mode == "adapter_only":
                    out = proj
                else:                                         # concat
                    out = torch.cat([F.normalize(feat, dim=-1), proj], dim=-1)

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

    print(f"\n=== Adapter mode: {args.mode} — color separability on RC9 ===")
    print(f"{'color':<10}{'n_samples':<12}{'L2 to self':<15}{'L2 to others':<15}{'sep ratio':<10}")
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
        print(f"{c:<10}{len(feats):<12}{mean_self:<15.3f}{mean_other:<15.3f}{sep_ratio:<10.3f}")
    print(f"\nMean sep_ratio ({args.mode}) = {np.mean(sep_ratios):.3f}")
    print(f"  reference: raw DINOv2 = 1.00, CLIP = 1.07, SAM = 1.04")

if __name__ == "__main__":
    main()
