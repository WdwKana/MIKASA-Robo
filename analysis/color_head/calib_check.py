"""Measure the color-scale calibration reference on REAL RC5 frames.

The module currently calibrates s on the mean per-patch color norm over ALL
patches — but >90% of patches are gray table (centered ≈ 0), so that mean is
tiny and s overshoots. We want s referenced to the CUBE-patch color scale (what
the probe used). This script reports the all-patch vs cube-patch reference and
the percentile of the all-patch distribution that matches the cube scale, so we
can pick a robust, label-free runtime statistic.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
sys.path.insert(0, str(ROOT))
from baselines.ppo.modules.frozen_vit import FrozenDualDinoV2
from analysis.pem.run_stage0p import set_seed, color_mask, patch_gt, extract_episodes

DEV = "cuda"


@torch.no_grad()
def per_patch_centered_rgb(base, hand, S=126, P=14):
    Hp = S // P
    out = []
    for v in (base, hand):
        x = v.astype(np.float32) / 255.0
        x = x[1:127, 1:127, :].reshape(Hp, P, Hp, P, 3).mean(axis=(1, 3))
        out.append(x.reshape(Hp * Hp, 3))
    return np.concatenate(out, 0) - 0.5            # (162,3) centered


@torch.no_grad()
def main():
    set_seed(0)
    vit = FrozenDualDinoV2().to(DEV).eval()
    eps = extract_episodes(vit, ROOT / "analysis/ebm/path_a_data/RememberColor5-v0")
    Hp = vit.grid[0]; N_v = Hp * Hp
    R = (torch.randn(3, 384, generator=torch.Generator().manual_seed(1234)) / (384 ** 0.5))

    all_proj, cube_proj, dino_norms = [], [], []
    for ep in eps:
        for t in range(ep["feat"].shape[0]):
            f = ep["feat"][t]                                   # (162,384)
            dino_norms.append(f.norm(dim=-1))
            c = torch.from_numpy(per_patch_centered_rgb(ep["base"][t], ep["hand"][t])).float()
            proj = (c @ R).norm(dim=-1)                         # (162,)
            all_proj.append(proj)
            gt = patch_gt(color_mask(ep["base"][t], ep["color"]), (Hp, Hp))  # base view cubes
            cube_idx = np.where(gt)[0]
            if len(cube_idx):
                cube_proj.append(proj[cube_idx])

    allp = torch.cat(all_proj); cubep = torch.cat(cube_proj)
    mean_dino = torch.cat(dino_norms).mean().item()
    print(f"mean||dino|| = {mean_dino:.2f}")
    print(f"all-patch  ||c@R||: mean={allp.mean():.4f}  "
          f"p90={allp.quantile(.9):.4f} p95={allp.quantile(.95):.4f} "
          f"p98={allp.quantile(.98):.4f} p99={allp.quantile(.99):.4f} max={allp.max():.4f}")
    print(f"cube-patch ||c@R||: mean={cubep.mean():.4f}  (this is the probe reference)")
    cube_ref = cubep.mean().item()
    # which all-patch percentile matches cube mean?
    pct = (allp < cube_ref).float().mean().item() * 100
    print(f"cube-mean ({cube_ref:.4f}) ≈ all-patch p{pct:.1f}")
    print(f"\ncolor_scale for frac=0.4:")
    print(f"  using all-patch mean  : {0.4*mean_dino/allp.mean():.1f}   (WRONG, overshoots)")
    print(f"  using cube-patch mean : {0.4*mean_dino/cube_ref:.1f}   (probe-correct)")
    print(f"  using all-patch p98   : {0.4*mean_dino/allp.quantile(.98):.1f}")
    print(f"  using all-patch p99   : {0.4*mean_dino/allp.quantile(.99):.1f}")
    print(f"\nRecommend COLOR_REF (cube scale) = {cube_ref:.4f}")


if __name__ == "__main__":
    main()
