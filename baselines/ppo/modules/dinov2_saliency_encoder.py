"""
Shared perception encoder for A6 family (drop-in replacement for NatureCNN).

Pipeline:
  rgb6 (B, 128, 128, 6)  joints (B, 25)
       │                       │
       ▼                       │
  [Frozen DINOv2-S (fp16, 126x126)]
       │ tok_b, tok_h, cls_b, cls_h
       ▼
  [Frozen saliency head v3]
       │ per-patch logit
       ▼
  top-K (K=8 by default), saliency-weighted attention pool
       │ pooled (B, d_vit=384)  + cls_summary (B, 128)
       └─────┬───────────────────┘
             ▼
  output features = concat([cls_summary, pool_proj(pooled), joints])
                    = (B, 128 + 256 + 25 = 409)

This is the EXACT perception stack used by ours (M4), packaged as a single
nn.Module so we can drop it into ppo_memtasks_{mlp,gru,lstm}.py just like
NatureCNN. The downstream memory module (none / GRU / LSTM / our buffer) sees
the SAME input information.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .frozen_vit import FrozenDualDinoV2
from .saliency_head import load_saliency_head


class DinoV2SaliencyEncoder(nn.Module):
    """A6-family perception. Produces a flat feature vector per step
    suitable for plugging into MLP / GRU / LSTM agents."""

    def __init__(self, sample_obs: dict, saliency_ckpt: str,
                 K: int = 8, pool_dim: int = 256, summary_dim: int = 128,
                 device: str = "cuda"):
        super().__init__()
        self.K = K
        self.device = device

        self.vit = FrozenDualDinoV2()
        self.head, xy_concat = load_saliency_head(saliency_ckpt, device=device)
        self.register_buffer("xy_concat", xy_concat)
        d_vit = self.vit.dim

        self.curr_summary = nn.Sequential(
            nn.Linear(2 * d_vit, summary_dim * 2), nn.GELU(),
            nn.Linear(summary_dim * 2, summary_dim),
        )
        self.pool_proj = nn.Linear(d_vit, pool_dim)

        # determine joints dim from sample_obs (default 25 for MIKASA Panda)
        joints_dim = (sample_obs["joints"].shape[-1]
                      if "joints" in sample_obs else 25)
        self.joints_dim = joints_dim
        self.out_features = summary_dim + pool_dim + joints_dim

    def forward(self, observations: dict) -> torch.Tensor:
        """observations: dict with 'rgb' (B, H, W, 6) and 'joints' (B, J).
        Returns: (B, out_features)
        """
        rgb = observations["rgb"]
        joints = observations["joints"].float() if observations["joints"].dtype != torch.float32 else observations["joints"]

        # Frozen ViT — already no_grad inside FrozenDualDinoV2.forward
        tok_b, tok_h, cls_b, cls_h = self.vit(rgb)
        all_tok = torch.cat([tok_b, tok_h], dim=1)         # (B, 2*N_v, d_vit)

        # Frozen saliency head — no grad path (head is in eval, params requires_grad=False)
        with torch.no_grad():
            sal_logits = self.head(all_tok, self.xy_concat)   # (B, 2*N_v)
            sal_probs = torch.sigmoid(sal_logits)
            topk_val, topk_idx = sal_probs.topk(self.K, dim=-1)        # (B, K)
            cand = torch.gather(
                all_tok, 1,
                topk_idx.unsqueeze(-1).expand(-1, -1, all_tok.size(-1))
            )                                                          # (B, K, d_vit)

        # Saliency-weighted pool over top-K (mirror what cross-attn would do)
        weights = topk_val.softmax(dim=-1).unsqueeze(-1)               # (B, K, 1)
        pooled = (weights * cand).sum(dim=1)                            # (B, d_vit)
        pooled = self.pool_proj(pooled)                                 # (B, pool_dim)

        cls_sum = self.curr_summary(torch.cat([cls_b, cls_h], dim=-1))  # (B, summary_dim)

        return torch.cat([cls_sum, pooled, joints], dim=-1)             # (B, out_features)
