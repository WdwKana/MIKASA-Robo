"""Test concat at multiple alpha weights + diagnose which MIKASA colors are
covered by the training hue bins.

The concat output is:
    out = concat[ F.normalize(DINOv2_feat), alpha * adapter_proj ]
L2² distance decomposes as ||DINOv2_self|| + alpha² · ||adapter_self||, so
alpha balances semantic vs color contribution.
"""
import sys, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
sys.path.insert(0, str(ROOT))

from baselines.ppo.modules.frozen_vit import FrozenDualDinoV2
from analysis.pem.run_stage0p import color_mask, MIKASA_COLORS
from analysis.color_head.train_supcon_hue import (
    ColorAwareAdapter, mean_rgb_per_patch, rgb_to_hue_bin, N_HUE_BINS, SAT_THRESH,
)

DEVICE = torch.device("cuda")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="analysis/color_head/adapter_supcon.pt")
ap.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0, 8.0])
args = ap.parse_args()

vit = FrozenDualDinoV2().to(DEVICE).eval()
ck = torch.load(ROOT / args.ckpt, map_location=DEVICE)
adapter = ColorAwareAdapter(d_in=ck["d_in"], d_proj=ck["d_proj"],
                             d_hidden=ck.get("d_hidden", 256)).to(DEVICE).eval()
adapter.load_state_dict(ck["adapter_state_dict"])
for p in adapter.parameters(): p.requires_grad_(False)
Hp, Wp = vit.grid

# diagnose MIKASA colors: where do their RGBs fall in our hue bin scheme?
print("=== MIKASA color → training bin mapping ===")
print(f"  (n_bins={N_HUE_BINS}, sat<{SAT_THRESH} → neutral bin {N_HUE_BINS})")
print(f"{'color':<8}{'RGB':<20}{'bin':<8}{'comment'}")
for c, rgb_uint in MIKASA_COLORS.items():
    rgb_t = torch.tensor(rgb_uint, dtype=torch.float32).unsqueeze(0) / 255.0
    bin_id = rgb_to_hue_bin(rgb_t).item()
    flag = "NEUTRAL (excluded!)" if bin_id == N_HUE_BINS else ""
    print(f"{c:<8}{str(rgb_uint):<20}{bin_id:<8}{flag}")

# load probe data
data_dir = ROOT / "analysis/ebm/path_a_data/RememberColor9-v0"
ep_files = sorted(data_dir.glob("ep*.npz"))[:10]

all_feats = {}                     # alpha -> {color: list of vectors}
for a in args.alphas:
    all_feats[a] = {c: [] for c in range(9)}
all_feats["adapter_only"] = {c: [] for c in range(9)}
all_feats["raw_dinov2"] = {c: [] for c in range(9)}

with torch.no_grad():
    for fp in ep_files:
        d = np.load(fp)
        base = d["base_rgb"]
        for t in [10, 12, 15, 20]:
            rgb6 = torch.from_numpy(np.concatenate([base[t:t+1], base[t:t+1]], axis=-1)).to(DEVICE)
            tok_b, _, _, _ = vit(rgb6)                       # (1, 81, 384)
            feat = tok_b[0]                                   # (81, 384)
            rgb_p = mean_rgb_per_patch(
                torch.from_numpy(base[t:t+1]).to(DEVICE),
                (Hp, Wp), vit.input_size, vit.patch_size)
            proj = adapter(feat.unsqueeze(0), rgb_p)[0]      # (81, 128)
            feat_normed = F.normalize(feat, dim=-1)

            for c in range(9):
                mask_pix = color_mask(base[t], c)
                if mask_pix.sum() < 4: continue
                H, W = mask_pix.shape
                ph, pw = H // Hp, W // Wp
                for i in range(Hp):
                    for j in range(Wp):
                        blk = mask_pix[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                        if blk.mean() > 0.01:
                            n = i * Wp + j
                            all_feats["adapter_only"][c].append(proj[n].cpu().float().numpy())
                            all_feats["raw_dinov2"][c].append(feat_normed[n].cpu().float().numpy())
                            for a in args.alphas:
                                cat = torch.cat([feat_normed[n], a * proj[n]], dim=-1)
                                all_feats[a][c].append(cat.cpu().float().numpy())

def mean_sep_ratio(feats_by_color):
    sep_ratios = []
    for c in range(9):
        if not feats_by_color[c]: continue
        feats = np.stack(feats_by_color[c])
        if len(feats) < 2: continue
        diff_self = feats[:, None, :] - feats[None, :, :]
        d_self = np.sqrt((diff_self**2).sum(-1))
        d_self = d_self[~np.eye(len(feats), dtype=bool)]
        mean_self = d_self.mean()
        others = []
        for c2 in range(9):
            if c2 == c or not feats_by_color[c2]: continue
            feats2 = np.stack(feats_by_color[c2])
            diff = feats[:, None, :] - feats2[None, :, :]
            others.append(np.sqrt((diff**2).sum(-1)).mean())
        mean_other = np.mean(others) if others else 0
        if mean_self > 0: sep_ratios.append(mean_other / mean_self)
    return float(np.mean(sep_ratios))

print(f"\n=== Mean sep_ratio (RC9, 9 task colors) ===")
print(f"  raw DINOv2 (unit-norm)   : {mean_sep_ratio(all_feats['raw_dinov2']):.3f}   (reference: should be ~1.00)")
print(f"  adapter_only (128-d)     : {mean_sep_ratio(all_feats['adapter_only']):.3f}")
for a in args.alphas:
    print(f"  concat alpha={a:>4.1f} (512-d) : {mean_sep_ratio(all_feats[a]):.3f}")
