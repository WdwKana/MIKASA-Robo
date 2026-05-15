"""
Saliency head: 2-layer MLP from frozen DINOv2 patch token + xy/view embedding
to a per-patch importance logit. Architecture matches Path A v2.

The head is trained offline (per task) by `analysis/ebm/path_a_train_head_v2.py`
and FROZEN at PPO time.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class SaliencyHead(nn.Module):
    def __init__(self, patch_dim: int, xy_dim: int = 32, hidden=(256, 128)):
        super().__init__()
        self.patch_dim = patch_dim
        self.xy_dim = xy_dim
        self.mlp = nn.Sequential(
            nn.Linear(patch_dim + xy_dim, hidden[0]), nn.GELU(),
            nn.Linear(hidden[0], hidden[1]), nn.GELU(),
            nn.Linear(hidden[1], 1),
        )

    def make_xy_embed(self, grid: tuple[int, int], view_id: int, device) -> torch.Tensor:
        Hp, Wp = grid
        ys = torch.linspace(-1, 1, Hp, device=device)
        xs = torch.linspace(-1, 1, Wp, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        feats = []
        for k in range(self.xy_dim // 6):
            f = 2 ** k
            feats += [torch.sin(f*gy), torch.cos(f*gy), torch.sin(f*gx), torch.cos(f*gx)]
        feats += [torch.full_like(gy, float(view_id)),
                  torch.full_like(gy, 1 - float(view_id))]
        emb = torch.stack(feats, dim=-1).view(Hp*Wp, -1)
        if emb.shape[-1] < self.xy_dim:
            pad = torch.zeros(emb.shape[0], self.xy_dim - emb.shape[-1], device=device)
            emb = torch.cat([emb, pad], dim=-1)
        return emb[:, :self.xy_dim]

    def forward(self, tokens: torch.Tensor, xy_embed: torch.Tensor) -> torch.Tensor:
        """tokens: (..., N, d), xy_embed: (N, xy_dim) -> logits (..., N)"""
        N = tokens.shape[-2]
        xy = xy_embed.unsqueeze(0).expand(*tokens.shape[:-2], N, -1)
        x = torch.cat([tokens, xy], dim=-1)
        return self.mlp(x).squeeze(-1)


def load_saliency_head(checkpoint_path: str | Path, device=None,
                       freeze: bool = True) -> tuple[SaliencyHead, torch.Tensor]:
    """Load a Path A v2 saliency head + return its (concatenated base+hand) xy embed.

    Returns:
      head: SaliencyHead in eval mode (frozen unless freeze=False).
      xy_concat: (2*N_v, xy_dim) ready to feed to head.forward together with
                 the dual-cam concatenated patch tokens.
    """
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    patch_dim = ckpt["patch_dim"]
    grid = ckpt["grid"]
    xy_dim = ckpt["xy_dim"]
    head = SaliencyHead(patch_dim=patch_dim, xy_dim=xy_dim)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    if freeze:
        for p in head.parameters():
            p.requires_grad_(False)
    if device is not None:
        head.to(device)
    base_xy = head.make_xy_embed(grid, view_id=0, device=device or "cpu")
    hand_xy = head.make_xy_embed(grid, view_id=1, device=device or "cpu")
    xy_concat = torch.cat([base_xy, hand_xy], dim=0)
    return head, xy_concat
