"""Diagnose: when is the cube visible in cached frames, and where does surprise
land at the cube-reveal moment vs later?"""
import sys, numpy as np, torch
sys.path.insert(0, "/zfsstore/user/s4176650/MIKASA-Robo")
from baselines.ppo.modules.frozen_vit import FrozenDualDinoV2
from analysis.pem.predictor import PEMPredictor
from analysis.pem.run_stage0p import (
    set_seed, color_mask, patch_gt, extract_episodes, train_predictor)

DEVICE = torch.device("cuda")
set_seed(0)

vit = FrozenDualDinoV2().to(DEVICE).eval()
episodes = extract_episodes(vit, "/zfsstore/user/s4176650/MIKASA-Robo/analysis/ebm/path_a_data/RememberColor9-v0")

# 1. Cube visibility timeline (averaged across episodes)
print("=== Cube visibility (GT patches > 0) across timesteps ===")
visibility = np.zeros(60)
for ep in episodes:
    for t in range(60):
        gt = patch_gt(color_mask(ep["base"][t], ep["color"]), vit.grid)
        if gt.sum() > 0: visibility[t] += 1
print(f"  t=0: {int(visibility[0])}/{len(episodes)} episodes have cube visible")
print(f"  t=1: {int(visibility[1])}")
print(f"  t=5: {int(visibility[5])}")
print(f"  t=10: {int(visibility[10])}")
print(f"  t=20: {int(visibility[20])}")
print(f"  t=40: {int(visibility[40])}")
print(f"  t=59: {int(visibility[59])}")

# 2. Train predictor briefly + measure surprise→cube IoU PER TIMESTEP
predictor = PEMPredictor(d_vit=vit.dim, n_patches=2*vit.num_patches_per_view).to(DEVICE)
print("\n=== Training predictor ===")
train_predictor(predictor, episodes, epochs=30)

print("\n=== Per-timestep IoU(top-K=8 surprise, cube GT) — base view only ===")
print(f"{'t':>3} {'N_with_cube':>11} {'mean_IoU':>9} {'mean_recall':>11}")
N_v = vit.num_patches_per_view
predictor.eval()
with torch.no_grad():
    for t in range(1, 60):
        ious, recalls, sur_means = [], [], []
        for ep in episodes:
            gt = patch_gt(color_mask(ep["base"][t], ep["color"]), vit.grid)
            if gt.sum() == 0: continue
            p_prev = ep["feat"][t-1].unsqueeze(0).to(DEVICE)
            p_t = ep["feat"][t].unsqueeze(0).to(DEVICE)
            sur = predictor.surprise(p_t, p_prev)[0].cpu().numpy()[:N_v]
            topk = np.argsort(-sur)[:8]
            sel = np.zeros(N_v, dtype=bool); sel[topk] = True
            inter = (sel & gt).sum(); union = (sel | gt).sum()
            ious.append(inter / max(union, 1))
            recalls.append(inter / max(gt.sum(), 1))
            sur_means.append(sur.mean())
        if not ious: continue
        if t in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 59):
            print(f"{t:>3} {len(ious):>11} {np.mean(ious):>9.3f} {np.mean(recalls):>11.3f}    mean|sur|={np.mean(sur_means):.3f}")
