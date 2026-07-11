"""
DINOSAUR-style object-centric slot attention over frozen DINOv2 features.

Stage-0 perception module (no RL). Pipeline:
  frozen DINOv2 patch features (N, d_vit)  [we already have this]
    -> input projection to slot_dim
    -> Slot Attention (K slots, iterative competition)        [Locatello 2020]
    -> Spatial Broadcast Decoder reconstructs the DINOv2 features  [DINOSAUR, ICLR23]
    -> loss = MSE(recon, features)   (fully unsupervised, no color labels)

The decoder's per-slot alpha masks (B, K, N) are the object segmentation we
evaluate in Stage 0: does any slot land on the task-relevant object?

Reference: Seitzer et al., "Bridging the Gap to Real-World Object-Centric
Learning" (DINOSAUR), ICLR 2023.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    def __init__(self, num_slots: int, dim: int, iters: int = 3,
                 hidden_dim: int = 128, eps: float = 1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs: torch.Tensor, num_slots: int | None = None):
        """inputs: (B, N, dim). Returns slots (B,K,dim), attn (B,K,N)."""
        B, N, D = inputs.shape
        K = num_slots or self.num_slots
        mu = self.slots_mu.expand(B, K, -1)
        sigma = self.slots_logsigma.exp().expand(B, K, -1)
        slots = mu + sigma * torch.randn(B, K, D, device=inputs.device)

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        attn = None
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale   # (B,K,N)
            attn = dots.softmax(dim=1) + self.eps                    # compete over slots
            attn_norm = attn / attn.sum(dim=-1, keepdim=True)        # normalize over patches
            updates = torch.einsum("bjd,bij->bid", v, attn_norm)     # (B,K,D)
            slots = self.gru(updates.reshape(-1, D), slots_prev.reshape(-1, D))
            slots = slots.reshape(B, K, D)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        return slots, attn


class SpatialBroadcastDecoder(nn.Module):
    def __init__(self, slot_dim: int, feat_dim: int, num_patches: int, hidden: int = 256):
        super().__init__()
        self.num_patches = num_patches
        self.pos = nn.Parameter(torch.randn(1, 1, num_patches, slot_dim) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_dim + 1),
        )

    def forward(self, slots: torch.Tensor):
        """slots: (B,K,slot_dim). Returns recon (B,N,feat_dim), masks (B,K,N)."""
        B, K, D = slots.shape
        x = slots.unsqueeze(2).expand(B, K, self.num_patches, D)
        x = x + self.pos                              # (B,K,N,slot_dim)
        out = self.mlp(x)                             # (B,K,N,feat+1)
        feat, alpha = out[..., :-1], out[..., -1:]
        alpha = alpha.softmax(dim=1)                  # compete over slots
        recon = (feat * alpha).sum(dim=1)             # (B,N,feat)
        masks = alpha.squeeze(-1)                     # (B,K,N)
        return recon, masks


class DinosaurSlots(nn.Module):
    """Slot attention + broadcast decoder operating on frozen DINOv2 features."""
    def __init__(self, feat_dim: int = 384, slot_dim: int = 128,
                 num_slots: int = 6, num_patches: int = 81, iters: int = 3):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, slot_dim),
        )
        self.slot_attn = SlotAttention(num_slots, slot_dim, iters=iters)
        self.decoder = SpatialBroadcastDecoder(slot_dim, feat_dim, num_patches)
        self.num_slots = num_slots
        self.num_patches = num_patches

    def forward(self, feats: torch.Tensor):
        """feats: (B, N, feat_dim) frozen DINOv2 patch features.
        Returns dict with recon, masks (B,K,N), slots, attn."""
        x = self.in_proj(feats)
        slots, attn = self.slot_attn(x)
        recon, masks = self.decoder(slots)
        return {"recon": recon, "masks": masks, "slots": slots, "attn": attn}

    def loss(self, feats: torch.Tensor) -> torch.Tensor:
        out = self.forward(feats)
        return F.mse_loss(out["recon"], feats)
