"""
Stage 0 buffer-preservation test with adapter-augmented features.

Three feature modes for the buffer L2 surprise:
  raw       — raw DINOv2 patch features (baseline)
  proj      — adapter projection only  (color-only, drops shape info)
  concat    — concat[normalize(DINOv2), alpha * adapter_proj] (deployment shape)

For each task (RC5, RC9, Shape5), runs SRB-MS simulation and reports
P(target in buffer) early/late. Pareto target:
  RC5/RC9    : adapter mode beats raw (color helps)
  Shape5     : adapter mode ≈ raw (shape preserved)
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
sys.path.insert(0, str(ROOT))

from baselines.ppo.modules.frozen_vit import FrozenDualDinoV2
from analysis.pem.run_stage0p import color_mask, set_seed
from analysis.pem.diag_mvsrs import SimBuffer, _patch_gt_thr
from analysis.color_head.train_supcon_hue import (
    ColorAwareAdapter, mean_rgb_per_patch,
)

DEVICE = torch.device("cuda")


def extract_episodes_with_adapter(vit, adapter, data_dir, mode: str, alpha: float):
    """Returns list of {feat (T, N, d_out), base, hand, color}."""
    Hp, Wp = vit.grid
    files = sorted(Path(data_dir).glob("ep*.npz"))
    eps = []
    for fp in files:
        d = np.load(fp)
        base = d["base_rgb"]                                 # (T, 128, 128, 3)
        hand = d["hand_rgb"]
        color = int(d["color_idx"])
        T = base.shape[0]
        rgb6 = torch.from_numpy(np.concatenate([base, hand], axis=-1)).to(DEVICE)
        with torch.no_grad():
            tok_b, tok_h, _, _ = vit(rgb6)                   # (T, 81, d_vit)
            feat_raw = torch.cat([tok_b, tok_h], dim=1)      # (T, 162, d_vit)
            if mode == "raw":
                out = feat_raw
            else:
                # per-patch mean RGB for both views
                base_t = torch.from_numpy(base).to(DEVICE)
                hand_t = torch.from_numpy(hand).to(DEVICE)
                rgb_b = mean_rgb_per_patch(base_t, (Hp, Wp), vit.input_size, vit.patch_size)
                rgb_h = mean_rgb_per_patch(hand_t, (Hp, Wp), vit.input_size, vit.patch_size)
                rgb_p = torch.cat([rgb_b, rgb_h], dim=1)     # (T, 162, 3)
                proj = adapter(feat_raw, rgb_p)              # (T, 162, d_proj)
                if mode == "proj":
                    out = proj
                else:  # concat
                    feat_norm = F.normalize(feat_raw, dim=-1)
                    out = torch.cat([feat_norm, alpha * proj], dim=-1)
        eps.append({"feat": out.cpu(), "base": base, "hand": hand, "color": color})
    return eps


def simulate_srbms(episodes, K=8, L=64, tau=30.0, patch_frac=0.05,
                   Hp=9, Wp=9, ms_alpha=0.1, ms_lambda=1.0,
                   novelty_thresh=0.95):
    """Standard SRB-MS simulation. Returns per-step buffer composition."""
    N_v = Hp * Wp
    N_total = 2 * N_v
    E, T = len(episodes), episodes[0]["feat"].shape[0]
    buf_target = np.zeros((E, T), dtype=np.float32)
    buf_target_frac = np.zeros((E, T), dtype=np.float32)
    n_used_avg = np.zeros(T, dtype=np.float32)
    for e, ep in enumerate(episodes):
        sim = SimBuffer(L=L, tau=tau, novelty_thresh=novelty_thresh)
        # target color mask per frame
        gt_per_t = [_patch_gt_thr(color_mask(ep["base"][t], ep["color"]), (Hp, Wp), patch_frac)
                    for t in range(T)]
        # we don't need rgb for sim buffer (SimBuffer.push takes it but we pass dummy)
        d_out = ep["feat"].shape[-1]
        ema_change = torch.zeros(N_total, device=DEVICE)
        p_prev = None
        for t in range(T):
            p_t = ep["feat"][t].to(DEVICE)                   # (N, d_out)
            # SRB-MS surprise
            if p_prev is not None:
                cur_change = ((p_t - p_prev) ** 2).sum(-1)
                ema_change.mul_(1 - ms_alpha).add_(ms_alpha * cur_change)
            if sim.used() == 0:
                s_raw = (p_t * p_t).sum(-1)
            else:
                buf = torch.stack(sim.feats)
                diff = p_t.unsqueeze(1) - buf.unsqueeze(0)
                s_raw = (diff * diff).sum(-1).min(dim=-1).values
            suppress = 1.0 + ms_lambda * ema_change / d_out
            sur = s_raw / suppress
            topk = torch.topk(sur, K).indices.cpu().numpy()
            for n in topk:
                is_tgt = (n < N_v) and bool(gt_per_t[t][n])
                # SimBuffer.push expects rgb; pass zero
                sim.push(p_t[n].clone(), torch.zeros(3, device=DEVICE),
                         prio=float(sur[n]), t_now=t,
                         prov=(t, int(n), is_tgt))
            if sim.used() > 0:
                cnt = sum(1 for prov in sim.prov if prov[2])
                buf_target[e, t] = 1.0 if cnt > 0 else 0.0
                buf_target_frac[e, t] = cnt / sim.used()
            n_used_avg[t] += sim.used()
            p_prev = p_t.clone()
    n_used_avg /= len(episodes)
    return buf_target, buf_target_frac, n_used_avg


def report(buf_target, buf_target_frac):
    early = buf_target[:, :10].mean()
    late = buf_target[:, 40:].mean() if buf_target.shape[1] > 40 else float("nan")
    return early, late, buf_target_frac.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="analysis/color_head/adapter_supcon_v3.pt")
    ap.add_argument("--tasks", nargs="+",
                    default=["RememberColor5-v0", "RememberColor9-v0", "RememberShape5-v0"])
    ap.add_argument("--alpha", type=float, default=4.0,
                    help="adapter weight in concat mode")
    args = ap.parse_args()
    set_seed(0)

    vit = FrozenDualDinoV2().to(DEVICE).eval()
    ck = torch.load(ROOT / args.ckpt, map_location=DEVICE)
    adapter = ColorAwareAdapter(d_in=ck["d_in"], d_proj=ck["d_proj"],
                                 d_hidden=ck.get("d_hidden", 256)).to(DEVICE).eval()
    adapter.load_state_dict(ck["adapter_state_dict"])
    for p in adapter.parameters(): p.requires_grad_(False)
    print(f"[adapter] loaded {args.ckpt} (d_proj={ck['d_proj']}, alpha={args.alpha})")

    print(f"\n{'task':<22} {'mode':<18} {'P_early':<9} {'P_late':<9} {'buf_late':<10}")
    print("=" * 75)
    for task in args.tasks:
        data_dir = ROOT / "analysis/ebm/path_a_data" / task
        if not data_dir.exists(): continue
        # raw baseline
        eps_raw = extract_episodes_with_adapter(vit, adapter, data_dir, "raw", 0)
        bt, bf, nu = simulate_srbms(eps_raw, K=8, L=64, patch_frac=0.05, novelty_thresh=0.95)
        e, l, f = report(bt, bf); buf_late = nu[40:].mean() if len(nu) > 40 else nu[-1]
        print(f"{task:<22} {'DINOv2 baseline':<18} {e:<9.3f} {l:<9.3f} {buf_late:<10.1f}")
        # concat at alpha sweep
        for alpha in [0.5, 1.0, 2.0, 4.0]:
            eps = extract_episodes_with_adapter(vit, adapter, data_dir, "concat", alpha)
            bt, bf, nu = simulate_srbms(eps, K=8, L=64, patch_frac=0.05, novelty_thresh=0.95)
            e, l, f = report(bt, bf)
            buf_late = nu[40:].mean() if len(nu) > 40 else nu[-1]
            print(f"{task:<22} CONCAT_a={alpha:<10.1f} {e:<9.3f} {l:<9.3f} {buf_late:<10.1f}")
        print()


if __name__ == "__main__":
    main()
