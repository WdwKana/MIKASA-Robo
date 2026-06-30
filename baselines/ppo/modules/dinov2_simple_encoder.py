"""DINOv2 (no saliency) feature extractor for fair perception-matched baselines.

Produces a flat feature vector per step (no buffer, no attention).
Forward path: frozen DINOv2 → [cls_base, cls_hand] → summary MLP → concat joints.
Suitable for plugging into MLP / GRU / LSTM agents to test "does DINOv2 alone
match V1's saliency-buffer pipeline on Remember tasks."

`color_aug=True` is the baseline-side analogue of SRB-TR-CRES: the GLOBAL
per-patch centered-mean-RGB color map (162×3=486) is CONCATENATED to the
feature — the canonical way to add an auxiliary observation modality (exactly
how `joints` is concatenated). This hands the recurrent baselines the SAME
color information SRB-TR's per-patch CRES gives the buffer, with NO bottleneck:
the full color map reaches the RNN, so the baseline is never color-handicapped,
and the learned downstream first layer is free to down-weight color on shape
tasks (effective weight ≠ input-dim fraction). Centering by 0.5 mirrors CRES
(neutral table ≈ 0, saturated cubes signed).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frozen_vit import FrozenDualDinoV2


class DinoV2SimpleEncoder(nn.Module):
    def __init__(self, sample_obs: dict, summary_dim: int = 256, device: str = "cuda",
                 color_aug: bool = False):
        super().__init__()
        self.vit = FrozenDualDinoV2()
        d_vit = self.vit.dim
        self.curr_summary = nn.Sequential(
            nn.Linear(2 * d_vit, summary_dim * 2), nn.GELU(),
            nn.Linear(summary_dim * 2, summary_dim),
        )
        joints_dim = (sample_obs["joints"].shape[-1]
                      if "joints" in sample_obs else 25)
        self.joints_dim = joints_dim
        self.color_aug = bool(color_aug)
        self.n_color = 2 * (self.vit.input_size // self.vit.patch_size) ** 2 * 3  # 162*3=486
        self.out_features = summary_dim + joints_dim + (self.n_color if self.color_aug else 0)

    @torch.no_grad()
    def _rgb_per_patch_flat(self, rgb6: torch.Tensor) -> torch.Tensor:
        """(B, H, W, 6) -> (B, 2*Hp*Hp*3) flattened per-patch centered mean RGB."""
        x = rgb6
        if x.dtype == torch.uint8:
            x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()               # (B, 6, H, W)
        base, hand = x[:, :3], x[:, 3:]
        S = self.vit.input_size
        P = self.vit.patch_size
        Hp = S // P
        if base.shape[-1] != S:
            base = F.interpolate(base, size=(S, S), mode="bilinear", align_corners=False)
            hand = F.interpolate(hand, size=(S, S), mode="bilinear", align_corners=False)
        B = base.shape[0]
        base = base.reshape(B, 3, Hp, P, Hp, P).mean(dim=(3, 5))   # (B,3,Hp,Hp)
        hand = hand.reshape(B, 3, Hp, P, Hp, P).mean(dim=(3, 5))
        base = base.permute(0, 2, 3, 1).reshape(B, Hp * Hp * 3)
        hand = hand.permute(0, 2, 3, 1).reshape(B, Hp * Hp * 3)
        return torch.cat([base, hand], dim=1) - 0.5               # (B, 486) centered

    def forward(self, observations: dict) -> torch.Tensor:
        rgb = observations["rgb"]
        joints = observations["joints"].float() if observations["joints"].dtype != torch.float32 else observations["joints"]
        _, _, cls_b, cls_h = self.vit(rgb)
        cls_sum = self.curr_summary(torch.cat([cls_b, cls_h], dim=-1))
        feats = [cls_sum, joints]
        if self.color_aug:
            feats.append(self._rgb_per_patch_flat(rgb).to(cls_sum.dtype))   # concat color map
        return torch.cat(feats, dim=-1)
