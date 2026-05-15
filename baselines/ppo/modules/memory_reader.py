"""
Memory Reader: single-layer cross-attention over the episodic buffer.

  Q  = MLP_Q( [proprio_t, curr_summary_t, learned_task_token] )
  K  = W_K @ M  + ts_embed(M.timestamps)
  V  = W_V @ M
  logit_i = (Q · K_i) / sqrt(d) + alpha * log(saliency_i)
  retrieved_t = softmax(logit) @ V

Padding entries (mask = False) get logit -> -inf.

This is the only learned module besides the actor/critic heads — saliency
head and ViT are frozen.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestampEmbedding(nn.Module):
    """Sinusoidal positional encoding indexed by step number."""
    def __init__(self, d: int, max_period: int = 100):
        super().__init__()
        self.d = d
        self.max_period = max_period
        # precomputable freqs (we'll compute on demand for arbitrary t)
        half = d // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (...) long -> (..., d) float."""
        t = t.float().unsqueeze(-1)
        a = t * self.freqs
        emb = torch.cat([torch.sin(a), torch.cos(a)], dim=-1)
        if emb.shape[-1] < self.d:
            pad = torch.zeros(*emb.shape[:-1], self.d - emb.shape[-1],
                              device=emb.device, dtype=emb.dtype)
            emb = torch.cat([emb, pad], dim=-1)
        return emb


class MemoryReader(nn.Module):
    def __init__(self, d_query: int, d_buffer: int, d_proj: int = 256,
                 alpha: float = 0.5):
        """
        d_query : dim of query input ([proprio, curr_summary, task_token])
        d_buffer: dim of buffer entries (= ViT patch dim, e.g. 384)
        d_proj  : projected K/V dim
        alpha   : weight on log(saliency) bias
        """
        super().__init__()
        self.d_proj = d_proj
        self.alpha = alpha
        self.W_Q = nn.Linear(d_query, d_proj)
        self.W_K = nn.Linear(d_buffer, d_proj)
        self.W_V = nn.Linear(d_buffer, d_proj)
        self.task_token = nn.Parameter(torch.randn(d_proj) * 0.02)
        self.ts_embed = TimestampEmbedding(d_proj)
        self.out_norm = nn.LayerNorm(d_proj)

    def forward(self,
                query_in: torch.Tensor,
                buffer_features: torch.Tensor,
                buffer_mask: torch.Tensor,
                buffer_timestamps: torch.Tensor,
                buffer_saliency: torch.Tensor) -> torch.Tensor:
        """
        query_in:          (B, d_query)
        buffer_features:   (B, L, d_buffer)
        buffer_mask:       (B, L) bool, True for valid
        buffer_timestamps: (B, L) long
        buffer_saliency:   (B, L) float
        Returns retrieved: (B, d_proj)
        """
        B, L, _ = buffer_features.shape

        Q = self.W_Q(query_in) + self.task_token         # (B, d_proj)
        K = self.W_K(buffer_features) + self.ts_embed(buffer_timestamps)  # (B, L, d_proj)
        V = self.W_V(buffer_features)                    # (B, L, d_proj)

        # attention logits
        logits = torch.einsum("bd,bld->bl", Q, K) / math.sqrt(self.d_proj)
        # saliency bias (clip to avoid log(<=0))
        sal_pos = buffer_saliency.clamp(min=1e-3)
        logits = logits + self.alpha * torch.log(sal_pos)
        # mask padding
        logits = logits.masked_fill(~buffer_mask, -1e9)

        # if entire buffer is empty for an env, softmax produces uniform; we
        # detect this and zero out the retrieved vector instead.
        any_valid = buffer_mask.any(dim=-1, keepdim=True)   # (B, 1)
        attn = logits.softmax(dim=-1)                       # (B, L)
        retrieved = torch.einsum("bl,bld->bd", attn, V)     # (B, d_proj)
        retrieved = retrieved * any_valid.float()
        return self.out_norm(retrieved)
