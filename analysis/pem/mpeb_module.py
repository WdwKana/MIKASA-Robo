"""
Memory-Predictive Episodic Buffer (MPEB) — Stage 0'' architecture.

Replaces frame-to-frame surprise with **memory-driven surprise**:

  Working memory (small GRU) accumulates context across the episode.
  Predictor maps memory state h_{t-1} -> predicted patch features p̂_t.
  Surprise s_t = ||p_t - p̂_t||² gates writes to the episodic buffer.

The learned init token h_init handles t=0: at episode start the predictor
output is the model's prior over initial scenes, and any genuinely-new
content (the colored cue) deviates from that prior.

This module is the architecture; training + evaluation is in run_stage0pp.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class WorkingMemory(nn.Module):
    """Small recurrent state that accumulates per-frame patch summaries."""

    def __init__(self, d_vit: int = 384, mem_dim: int = 128):
        super().__init__()
        self.mem_dim = mem_dim
        self.summary_proj = nn.Linear(d_vit, mem_dim)
        self.gru = nn.GRUCell(mem_dim, mem_dim)
        # learned initial state — h_init is what the predictor sees at t=0
        self.h_init = nn.Parameter(torch.zeros(1, mem_dim))

    def init_state(self, batch_size: int, device) -> torch.Tensor:
        return self.h_init.expand(batch_size, -1).to(device)

    def step(self, p_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """p_t: (B, N, d_vit); h_prev: (B, mem_dim) -> h_t (B, mem_dim)."""
        summary = self.summary_proj(p_t.mean(dim=1))   # (B, mem_dim)
        return self.gru(summary, h_prev)


class MemoryPredictor(nn.Module):
    """Predicts per-patch features from the working memory state.

    Per-patch shared MLP conditioned on (h_{t-1}, learned positional embed).
    """

    def __init__(self, mem_dim: int = 128, d_vit: int = 384,
                 n_patches: int = 162, pos_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.n_patches = n_patches
        self.pos = nn.Parameter(torch.randn(1, n_patches, pos_dim) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(mem_dim + pos_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_vit),
        )

    def forward(self, h_prev: torch.Tensor) -> torch.Tensor:
        """h_prev: (B, mem_dim) -> p̂_t (B, n_patches, d_vit)."""
        B = h_prev.shape[0]
        h_b = h_prev.unsqueeze(1).expand(B, self.n_patches, -1)
        pos = self.pos.expand(B, self.n_patches, -1)
        x = torch.cat([h_b, pos], dim=-1)
        return self.mlp(x)


class MPEB(nn.Module):
    """Working memory + memory-driven predictor, jointly trained."""

    def __init__(self, d_vit: int = 384, n_patches: int = 162,
                 mem_dim: int = 128, pos_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.wm = WorkingMemory(d_vit=d_vit, mem_dim=mem_dim)
        self.predictor = MemoryPredictor(
            mem_dim=mem_dim, d_vit=d_vit, n_patches=n_patches,
            pos_dim=pos_dim, hidden=hidden,
        )

    def unroll(self, features: torch.Tensor):
        """Process a batch of episodes (B, T, N, d).

        Returns:
          predictions (B, T, N, d) — p̂_t computed from h_{t-1}
          hidden_states (B, T, mem_dim) — h_t after step t
        """
        B, T, N, d = features.shape
        h = self.wm.init_state(B, features.device)   # (B, mem_dim)
        preds = []
        hids = []
        for t in range(T):
            p_hat_t = self.predictor(h)              # predict from h_{t-1}
            preds.append(p_hat_t)
            h = self.wm.step(features[:, t], h)      # update memory with p_t
            hids.append(h)
        return torch.stack(preds, dim=1), torch.stack(hids, dim=1)

    def surprise(self, features: torch.Tensor) -> torch.Tensor:
        """Return per-step per-patch surprise (B, T, N)."""
        preds, _ = self.unroll(features)
        return ((features - preds) ** 2).sum(-1)
