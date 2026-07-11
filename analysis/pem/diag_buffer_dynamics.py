"""
SRB Stage-0 diagnostic — does the buffer actually preserve cube info?

Simulates write-side dynamics for three methods on the same cached RC9 frames:
  - SRB         : top-K by min L2 distance to current buffer entries
  - V1-saliency : top-K by trained saliency-head output (path_a_head_v3.pt)
  - A3-random   : top-K uniform random per step

Each method shares the same buffer infrastructure (size L=64, FIFO+priority
eviction, novelty filter). For every (episode, t), we count:

  cube_in_buffer(t)  — does any buffer slot hold a patch whose source frame
                       had the colored cube at that patch position?
  cube_frac(t)       — fraction of valid buffer slots that are cube patches.

Plots:
  fig1_cube_preservation.png — mean cube_in_buffer over t, across 20 episodes
  fig2_cube_frac.png         — mean cube_frac over t
  fig3_spatial_writes.png    — heatmap of which (base) patch positions each
                                method writes to (averaged over episode + ep)

This pinpoints whether SRB's underperformance is:
  (a) buffer never has cube (write filter wrong) → fix write rule
  (b) buffer has cube but at low fraction (signal-to-noise) → fix scoring
  (c) buffer has cube as much as V1 (reader / LSTM is bottleneck) → look elsewhere
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path("/zfsstore/user/s4176650/MIKASA-Robo")
sys.path.insert(0, str(ROOT))

from baselines.ppo.modules.frozen_vit import FrozenDualDinoV2
from baselines.ppo.modules.saliency_head import load_saliency_head
from analysis.pem.run_stage0p import (
    set_seed, color_mask, patch_gt, extract_episodes,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ────────────────── lightweight buffer simulator ──────────────────

class SimBuffer:
    """Standalone buffer with provenance tracking (which patch each slot came
    from). FIFO + priority eviction with priority * exp(-age/tau)."""

    def __init__(self, L=64, tau=30.0, novelty_thresh=0.95):
        self.L = L; self.tau = tau; self.novelty_thresh = novelty_thresh
        self.features = []        # list of (d,) tensors
        self.priorities = []      # write-time priority
        self.timestamps = []      # write-time
        self.provenance = []      # (episode_t, patch_n, is_cube_when_written)

    def used(self):
        return len(self.features)

    def push(self, feat, priority, t_now, prov):
        """Add an entry; if full, evict lowest-effective-priority entry."""
        # novelty filter against existing entries
        if self.used() > 0:
            existing = torch.stack(self.features)
            sims = torch.nn.functional.cosine_similarity(
                feat.unsqueeze(0), existing, dim=-1)
            if sims.max() > self.novelty_thresh:
                return False
        if self.used() < self.L:
            self.features.append(feat); self.priorities.append(priority)
            self.timestamps.append(t_now); self.provenance.append(prov)
            return True
        # evict lowest effective priority
        ages = torch.tensor([t_now - ts for ts in self.timestamps], dtype=torch.float32)
        prios = torch.tensor(self.priorities, dtype=torch.float32)
        eff = prios * torch.exp(-ages / self.tau)
        idx = int(eff.argmin().item())
        # also check if new entry would have higher effective priority
        new_eff = float(priority)  # age = 0 for new entry → exp(0)=1
        if new_eff <= float(eff[idx]):
            return False
        self.features[idx] = feat; self.priorities[idx] = priority
        self.timestamps[idx] = t_now; self.provenance[idx] = prov
        return True

    def reset(self):
        self.__init__(L=self.L, tau=self.tau, novelty_thresh=self.novelty_thresh)


# ────────────────── method-specific scoring ──────────────────

@torch.no_grad()
def score_srb(p_t, sim_buf, d_vit):
    """Min L2 distance to existing buffer entries. Empty buffer → ||p_t||²."""
    if sim_buf.used() == 0:
        return (p_t * p_t).sum(-1)
    buf = torch.stack(sim_buf.features)                 # (M, d)
    diff = p_t.unsqueeze(1) - buf.unsqueeze(0)          # (N, M, d)
    dist = (diff * diff).sum(-1)                        # (N, M)
    return dist.min(dim=-1).values                      # (N,)


@torch.no_grad()
def score_saliency(p_t, sal_head, xy_concat):
    """V1 saliency head scores."""
    logits = sal_head(p_t.unsqueeze(0), xy_concat)[0]
    return torch.sigmoid(logits)


@torch.no_grad()
def score_random(p_t, rng):
    """Uniform random per patch."""
    return torch.from_numpy(rng.random(p_t.shape[0])).to(p_t.device).float()


@torch.no_grad()
def score_wfa(p_t):
    """Within-frame anomaly: ‖p_t[n] − mean_t‖² where mean_t is across all
    patches of this frame. Highlights spatially distinctive patches."""
    mean = p_t.mean(dim=0, keepdim=True)              # (1, d)
    return ((p_t - mean) ** 2).sum(-1)                # (N,)


@torch.no_grad()
def score_srb_wfa(p_t, sim_buf):
    """Multiplicative: SRB × WFA, normalized. Both novel temporally AND
    spatially distinctive."""
    srb = score_srb(p_t, sim_buf, p_t.shape[-1])
    wfa = score_wfa(p_t)
    # normalize each to comparable scale
    srb_n = srb / (srb.max() + 1e-8)
    wfa_n = wfa / (wfa.max() + 1e-8)
    return srb_n * wfa_n


@torch.no_grad()
def score_srb_ms(p_t, p_prev, ema_change, sim_buf, alpha=0.1):
    """Motion-suppressed SRB. EMA of frame-to-frame change is used to
    downweight patches that change every frame (gripper / arm artifacts).
    Mutates ema_change in place.

      surprise = SRB / (1 + λ · ema_change[n])
    """
    if p_prev is not None:
        cur_change = ((p_t - p_prev) ** 2).sum(-1)
        ema_change.mul_(1 - alpha).add_(alpha * cur_change)
    srb = score_srb(p_t, sim_buf, p_t.shape[-1])
    return srb / (1.0 + ema_change)


# ────────────────── per-episode simulation ──────────────────

def simulate_method(method, episodes, vit, sal_head=None, xy_concat=None,
                    K=8, L=64, tau=30.0):
    """Returns per-step (E, T) arrays:
      cube_present : bool, any cube entry in buffer
      cube_frac    : fraction of buffer entries that are cube
      n_used       : buffer fill count
    Also returns spatial_writes (n_episodes × n_patches) for fig 3.
    """
    Hp, Wp = vit.grid
    N_v = Hp * Wp           # 81 base, 81 hand
    N_total = 2 * N_v

    E = len(episodes); T = episodes[0]["feat"].shape[0]
    cube_present = np.zeros((E, T), dtype=np.float32)
    cube_frac    = np.zeros((E, T), dtype=np.float32)
    n_used       = np.zeros((E, T), dtype=np.int32)
    spatial_writes = np.zeros(N_total, dtype=np.float32)

    rng = np.random.default_rng(0)
    for e, ep in enumerate(episodes):
        sim = SimBuffer(L=L, tau=tau)
        # precompute per-frame GT (cube patches in base view)
        gt_per_t = []
        for t in range(T):
            gt = patch_gt(color_mask(ep["base"][t], ep["color"]), (Hp, Wp))
            gt_per_t.append(gt)

        # state for motion-suppressed variant
        ema_change = torch.zeros(N_total, device=DEVICE)
        p_prev = None

        for t in range(T):
            p_t = ep["feat"][t].to(DEVICE)              # (162, d)
            # compute score per patch
            if method == "srb":
                scores = score_srb(p_t, sim, vit.dim)
            elif method == "saliency":
                scores = score_saliency(p_t, sal_head, xy_concat)
            elif method == "random":
                scores = score_random(p_t, rng)
            elif method == "wfa":
                scores = score_wfa(p_t)
            elif method == "srb_wfa":
                scores = score_srb_wfa(p_t, sim)
            elif method == "srb_ms":
                scores = score_srb_ms(p_t, p_prev, ema_change, sim, alpha=0.1)
            else:
                raise ValueError(method)
            p_prev = p_t.clone()
            # top-K
            topk = torch.topk(scores, K).indices.cpu().numpy()
            for n in topk:
                # is this patch a cube at time t?
                is_cube = (n < N_v) and bool(gt_per_t[t][n])
                sim.push(p_t[n].clone(), priority=float(scores[n]),
                         t_now=t, prov=(t, int(n), is_cube))
                spatial_writes[n] += 1

            # measure
            n_used[e, t] = sim.used()
            if sim.used() == 0:
                cube_present[e, t] = 0; cube_frac[e, t] = 0
            else:
                cube_count = sum(1 for prov in sim.provenance if prov[2])
                cube_present[e, t] = 1.0 if cube_count > 0 else 0.0
                cube_frac[e, t] = cube_count / sim.used()

    return cube_present, cube_frac, n_used, spatial_writes


# ────────────────── plots + reporting ──────────────────

def plot_results(results, vit, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # FIG 1: cube preservation
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (cp, cf, nu, sw) in results.items():
        ax.plot(np.mean(cp, axis=0), label=f"{name}  (mean across 20 eps)")
    ax.set_xlabel("episode timestep")
    ax.set_ylabel("P(cube in buffer)")
    ax.set_title("Does the buffer preserve cube info across the episode?")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_cube_preservation.png", dpi=110)
    plt.close(fig)

    # FIG 2: cube fraction
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (cp, cf, nu, sw) in results.items():
        ax.plot(np.mean(cf, axis=0), label=name)
    ax.set_xlabel("episode timestep")
    ax.set_ylabel("fraction of buffer that is cube patches")
    ax.set_title("Signal-to-noise: what fraction of buffer is task-relevant?")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_cube_frac.png", dpi=110)
    plt.close(fig)

    # FIG 3: spatial heatmaps (base view, 9x9)
    Hp, Wp = vit.grid
    fig, axes = plt.subplots(1, len(results), figsize=(4*len(results), 4.5))
    if len(results) == 1: axes = [axes]
    for ax, (name, (cp, cf, nu, sw)) in zip(axes, results.items()):
        base_sw = sw[:Hp*Wp].reshape(Hp, Wp)
        im = ax.imshow(base_sw, cmap="hot")
        ax.set_title(f"{name}\nwrite counts (base view)")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_spatial_writes.png", dpi=110)
    plt.close(fig)

    # summary numbers
    print("\n" + "="*84)
    print(f"{'method':<14} {'mean P(cube∈buf) early':>22} {'late (t≥40)':>14} {'mean cube%':>12}")
    print("="*84)
    for name, (cp, cf, nu, sw) in results.items():
        early = float(cp[:, :10].mean())
        late = float(cp[:, 40:].mean())
        frac = float(cf.mean())
        print(f"{name:<14} {early:>22.3f} {late:>14.3f} {frac:>12.3f}")
    print(f"\nplots → {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="RememberColor9-v0")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--tau", type=float, default=30.0)
    p.add_argument("--saliency-ckpt", default="analysis/ebm/path_a_head_v3.pt")
    args = p.parse_args()
    set_seed(0)

    vit = FrozenDualDinoV2().to(DEVICE).eval()
    print(f"[diag] task={args.task} grid={vit.grid}")
    episodes = extract_episodes(vit, ROOT / "analysis/ebm/path_a_data" / args.task)
    print(f"  {len(episodes)} episodes, T=60")

    print("[diag] loading saliency head for V1 simulation...")
    sal_head, xy_concat = load_saliency_head(
        str(ROOT / args.saliency_ckpt), device=DEVICE, freeze=True)

    results = {}
    for method in ["srb", "saliency", "random", "wfa", "srb_wfa", "srb_ms"]:
        print(f"[diag] simulating method={method}...")
        results[method] = simulate_method(
            method, episodes, vit,
            sal_head=sal_head, xy_concat=xy_concat,
            K=args.K, L=args.L, tau=args.tau)

    out_dir = ROOT / "analysis/pem/diag_buffer" / args.task
    plot_results(results, vit, out_dir)


if __name__ == "__main__":
    main()
