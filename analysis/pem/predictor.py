"""
PEM 1-step forward predictor for frozen DINOv2 patch features.

Per-patch shared-weight MLP with residual form:
  p̂_t = p_{t-1} + delta(p_{t-1}, pos_embed[, action])

Action conditioning is optional (Stage 0' cached frames have no actions; we
omit it and only condition on the previous patch features. In Stage 1' RL we
will add `action_{t-1}` to the per-patch input).

Trained with MSE against p_t. Output: per-patch surprise s_t[n] =
‖p_t[n] − p̂_t[n]‖² → top-K-by-surprise replaces top-K-by-saliency in V1.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PEMPredictor(nn.Module):
    def __init__(self,
                 d_vit: int = 384,
                 n_patches: int = 162,
                 action_dim: int | None = None,
                 hidden: int = 256,
                 pos_dim: int = 32):
        super().__init__()
        self.d_vit = d_vit
        self.n_patches = n_patches
        self.action_dim = action_dim
        self.pos = nn.Parameter(torch.randn(1, n_patches, pos_dim) * 0.02)
        in_dim = d_vit + pos_dim
        if action_dim is not None:
            self.act_proj = nn.Linear(action_dim, pos_dim)
            in_dim += pos_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_vit),
        )

    def forward(self, p_prev: torch.Tensor, a_prev: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = p_prev.shape
        pos = self.pos.expand(B, N, -1)
        feats = [p_prev, pos]
        if self.action_dim is not None:
            if a_prev is None:
                raise ValueError("action_dim set but a_prev is None")
            act = self.act_proj(a_prev).unsqueeze(1).expand(B, N, -1)
            feats.append(act)
        x = torch.cat(feats, dim=-1)
        delta = self.mlp(x)
        return p_prev + delta

    def surprise(self, p_t: torch.Tensor, p_prev: torch.Tensor,
                 a_prev: torch.Tensor | None = None) -> torch.Tensor:
        """Per-patch L2 prediction error, shape (B, N)."""
        p_hat = self.forward(p_prev, a_prev)
        return ((p_t - p_hat) ** 2).sum(-1)
