"""
Quick probe for color-fusion scaling design (CRES / CCAT variants).

Measures, on cached RC5 frames:
  1. typical per-patch ||DINOv2|| (so we know the scale to match a color residual to)
  2. cosine(DINOv2 feat) between SAME-color vs DIFFERENT-color cube patches
     → tells us how far below the 0.95 novelty thresh different-color cubes
       already sit (if already <0.95, no fix needed; if ~0.99, we must inject)
  3. with a color residual z = dino + lambda * R(meanRGB) added, how the
     same/diff-color cosine moves as a function of the residual-norm fraction.

This calibrates `color_frac` (residual norm as a fraction of ||dino||) so that
different-color cube patches drop below novelty_thresh while same-color stay above.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
sys.path.insert(0, str(ROOT))
from baselines.ppo.modules.frozen_vit import FrozenDualDinoV2
from analysis.pem.run_stage0p import set_seed, color_mask, extract_episodes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _per_patch_meanrgb(base_rgb_t, hand_rgb_t, S=126, P=14):
    """Return (N_total, 3) mean RGB in [0,1], matching DINOv2 9x9 grid per view."""
    Hp = S // P
    out = []
    for view in (base_rgb_t, hand_rgb_t):
        v = view.astype(np.float32) / 255.0
        v = v[1:127, 1:127, :]                      # 128->126 crop
        v = v.reshape(Hp, P, Hp, P, 3).mean(axis=(1, 3))   # (9,9,3)
        out.append(v.reshape(Hp * Hp, 3))
    return np.concatenate(out, axis=0)              # (162, 3)


@torch.no_grad()
def main():
    set_seed(0)
    vit = FrozenDualDinoV2().to(DEVICE).eval()
    eps = extract_episodes(vit, ROOT / "analysis/ebm/path_a_data/RememberColor5-v0")
    Hp = vit.grid[0]; N_v = Hp * Hp

    dino_norms = []
    cube_feats = []     # list of (feat[384], color_idx) for cube patches (base view)
    cube_rgb = []
    for ep in eps:
        feat = ep["feat"]               # (T, 162, 384)
        color = ep["color"]
        T = feat.shape[0]
        for t in range(T):
            f = feat[t]                 # (162, 384)
            dino_norms.append(f.norm(dim=-1))   # (162,)
            # cube patches in base view, by color mask
            from analysis.pem.run_stage0p import patch_gt
            gt = patch_gt(color_mask(ep["base"][t], color), (Hp, Hp))   # (81,) bool
            rgb = _per_patch_meanrgb(ep["base"][t], ep["hand"][t])      # (162,3)
            for n in np.where(gt)[0]:
                cube_feats.append((f[n].numpy(), color))
                cube_rgb.append(rgb[n])

    dino_norms = torch.cat(dino_norms)
    print(f"[norm] per-patch ||dino||: mean={dino_norms.mean():.2f} "
          f"median={dino_norms.median():.2f} p10={dino_norms.quantile(0.1):.2f} "
          f"p90={dino_norms.quantile(0.9):.2f}")
    print(f"[data] {len(cube_feats)} cube patches across {len(eps)} eps")

    feats = np.stack([f for f, _ in cube_feats])        # (M, 384)
    cols  = np.array([c for _, c in cube_feats])        # (M,)
    rgbs  = np.stack(cube_rgb)                           # (M, 3)
    feats_t = torch.from_numpy(feats).float()
    rgbs_t = torch.from_numpy(rgbs).float()

    # sample pairs
    rng = np.random.default_rng(0)
    M = len(feats)
    n_pairs = 4000
    ii = rng.integers(0, M, n_pairs); jj = rng.integers(0, M, n_pairs)
    keep = ii != jj
    ii, jj = ii[keep], jj[keep]
    same = cols[ii] == cols[jj]

    def cos_stats(z):
        zi = F.normalize(z[ii], dim=-1); zj = F.normalize(z[jj], dim=-1)
        c = (zi * zj).sum(-1).numpy()
        return c[same], c[~same]

    def ncc_acc(z):
        """Leave-one-out nearest-centroid color-ID accuracy on cube patches.
        Proxy for 'how well a readout (the reader) can tell colors apart'."""
        zt = F.normalize(z, dim=-1)
        uniq = np.unique(cols)
        # per-color centroid, then classify each patch to nearest centroid
        cents = torch.stack([zt[cols == c].mean(0) for c in uniq])     # (C, d)
        cents = F.normalize(cents, dim=-1)
        sims = zt @ cents.t()                                          # (M, C)
        pred = uniq[sims.argmax(-1).numpy()]
        return float(np.mean(pred == cols))

    # baseline DINOv2
    cs, cd = cos_stats(feats_t)
    base_acc = ncc_acc(feats_t)
    chance = 1.0 / len(np.unique(cols))
    print("\n[baseline DINOv2] cube patches (chance acc = %.3f):" % chance)
    print(f"  same-color cos: mean={cs.mean():.4f}  frac>0.95={np.mean(cs>0.95):.3f}")
    print(f"  diff-color cos: mean={cd.mean():.4f}  frac>0.95={np.mean(cd>0.95):.3f}")
    print(f"  gap(same-diff)={cs.mean()-cd.mean():.4f}   color-ID acc={base_acc:.3f}")

    g = torch.Generator().manual_seed(1234)
    R = torch.randn(3, 384, generator=g) / (384 ** 0.5)   # var 1/d -> ||rgb@R||~||rgb||
    mean_dino = dino_norms.mean().item()

    for desc_name, desc in [("raw RGB", rgbs_t), ("centered RGB-0.5", rgbs_t - 0.5)]:
        rgb_proj = desc @ R
        base_proj_norm = rgb_proj.norm(dim=-1).mean().item()
        print(f"\n[CRES  z=dino+s*(c@R)]   color desc = {desc_name}")
        print(f"  {'frac':>6} {'s':>7} {'same>.95':>9} {'diff>.95':>9} {'gap':>8} {'colACC':>8}")
        for cf in [0.0, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
            s = cf * mean_dino / (base_proj_norm + 1e-6)
            z = feats_t + s * rgb_proj
            cs, cd = cos_stats(z)
            print(f"  {cf:>6.2f} {s:>7.2f} {np.mean(cs>0.95):>9.3f} "
                  f"{np.mean(cd>0.95):>9.3f} {cs.mean()-cd.mean():>8.4f} {ncc_acc(z):>8.3f}")

    for desc_name, desc in [("raw RGB", rgbs_t), ("centered RGB-0.5", rgbs_t - 0.5)]:
        cnorm = desc.norm(dim=-1).mean().item()
        print(f"\n[CCAT  z=[dino, s*c]]    color desc = {desc_name}")
        print(f"  {'frac':>6} {'s':>7} {'same>.95':>9} {'diff>.95':>9} {'gap':>8} {'colACC':>8}")
        for cf in [0.0, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
            s = cf * mean_dino / (cnorm + 1e-6)
            z = torch.cat([feats_t, s * desc], dim=-1)
            cs, cd = cos_stats(z)
            print(f"  {cf:>6.2f} {s:>7.2f} {np.mean(cs>0.95):>9.3f} "
                  f"{np.mean(cd>0.95):>9.3f} {cs.mean()-cd.mean():>8.4f} {ncc_acc(z):>8.3f}")


if __name__ == "__main__":
    main()
