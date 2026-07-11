"""Smoke test for CRES / CCAT modules: forward, calibration, buffer growth,
snapshot/restore round-trip, and the differentiable replay path (mirrors the
exact call the PPO update makes). Catches dimension bugs before SLURM submit."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
sys.path.insert(0, str(ROOT))
from baselines.ppo.modules import (
    EBMSRBTRCRESMemoryModule, EBMSRBTRCCATMemoryModule, EBMSRBTRMVMemoryModule)

DEV = "cuda"
B, P, T = 4, 25, 6
H = 128

# Load REAL RC5 frames so the color-scale calibration is meaningful (random
# noise averages to 0.5/gray per patch and pathologically inflates the scale).
import numpy as np
_eps = sorted((ROOT / "analysis/ebm/path_a_data/RememberColor5-v0").glob("ep*.npz"))
_frames = []
for fp in _eps[:B]:
    d = np.load(fp)
    _frames.append(np.concatenate([d["base_rgb"], d["hand_rgb"]], axis=-1))  # (T,128,128,6)
_REAL = torch.from_numpy(np.stack(_frames)).to(DEV)   # (B, T, 128,128,6) uint8


def real_rgb6(t):
    return _REAL[:, t % _REAL.shape[1]].contiguous()


def run(name, Mod, **kw):
    print(f"\n===== {name} =====")
    torch.manual_seed(0)
    m = Mod(num_envs=B, proprio_dim=P, device=DEV, **kw).to(DEV)
    d_vit = m.vit.dim
    d_buf = m.buffer.d
    print(f"d_vit={d_vit}  buffer.d={d_buf}  reader.W_K.in={m.reader.W_K.in_features}")
    assert m.reader.W_K.in_features == d_buf, "reader/buffer dim mismatch!"

    # cache stores like the PPO loop
    caches = []
    m.reset(torch.ones(B, dtype=torch.bool, device=DEV))
    for t in range(T):
        rgb6 = real_rgb6(t)
        proprio = torch.randn(B, P, device=DEV)
        gru_pre = m.gru_state.squeeze(0).detach().clone()      # (B, 2H)
        s_t, info = m.step(rgb6, proprio, t=t)
        assert s_t.shape == (B, m.d_state), s_t.shape
        assert torch.isfinite(s_t).all(), "NaN/inf in s_t"
        caches.append({
            "features": m.buffer.features.detach().clone(),
            "used": m.buffer.used.clone(),
            "timestamps": m.buffer.timestamps.clone(),
            "saliency": m.buffer.saliency.clone(),
            "cls_base": info["cls_base"], "cls_hand": info["cls_hand"],
            "proprio": proprio, "gru_pre": gru_pre, "gru_input": info["gru_input"],
        })
        cs = info.get("color_scale")
        cs_str = f"color_scale={cs.item():.2f}  " if cs is not None else ""
        print(f"  t={t}  used={m.buffer.used.tolist()}  n_pushed={info['n_pushed'].tolist()}"
              f"  {cs_str}|s_t|={s_t.norm(dim=-1).mean():.2f}")
    if info.get("color_scale") is not None:
        assert info["color_scale"].item() > 0, "color_scale not calibrated!"

    # snapshot / restore round-trip
    snap = m.snapshot()
    feats_before = m.buffer.features.clone()
    m.step(torch.randint(0, 256, (B, H, H, 6), dtype=torch.uint8, device=DEV),
           torch.randn(B, P, device=DEV), t=T)          # perturb
    m.restore(snap)
    assert torch.allclose(m.buffer.features, feats_before), "restore mismatch!"
    print("  snapshot/restore OK")

    # differentiable replay (exactly as PPO update calls it)
    c = caches[3]
    cached = {"features": c["features"], "used": c["used"],
              "timestamps": c["timestamps"], "saliency": c["saliency"]}
    m.train()
    s_rep = m.replay(cached, c["cls_base"], c["cls_hand"], c["proprio"],
                     c["gru_pre"].unsqueeze(0), c["gru_input"])
    assert s_rep.shape == (B, m.d_state), s_rep.shape
    loss = s_rep.sum()
    loss.backward()
    g = m.reader.W_V.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, "no grad to reader!"
    print(f"  replay OK  s_rep|={s_rep.norm(dim=-1).mean():.2f}  reader.W_V grad_norm={g.norm():.4f}")
    print(f"  {name}: ALL CHECKS PASSED")


if __name__ == "__main__":
    run("CRES (residual, dim-preserving)", EBMSRBTRCRESMemoryModule, color_frac=0.4)
    run("CCAT (concat, dim+3)", EBMSRBTRCCATMemoryModule, color_frac=0.4)
    # sanity: baseline MV still constructs/forwards (no regression from __init__ edit)
    run("MV (baseline, regression check)", EBMSRBTRMVMemoryModule)
    print("\nALL MODULES OK")
