"""
Frozen DINOv2-S/14 wrapper for dual-camera batched inference.

Optimized for PPO rollout speed:
  (1) fp16 inference (DINOv2-S 22M params, no numerical issues at fp16)
  (2) base + hand cameras concatenated along the batch dim -> single ViT call
      per step (was 2 separate calls)
  (3) native input size near 128×128 via interpolate_pos_encoding=True
      (input rounded to nearest multiple of patch_size=14 -> 126×126 -> 9×9
      patch grid = 81 tokens per view, vs 16×16=256 at 224)

Expected combined speedup vs original: ~5-7×.

Output dtype is float32 (cast back from fp16 after the backbone) so downstream
heads keep their original numerical behavior.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD  = (0.229, 0.224, 0.225)
INPUT_SIZE = 126   # 9 × 14, nearest multiple of patch_size below 128


class FrozenDualDinoV2(nn.Module):
    """Frozen DINOv2 encoder for two cameras (fp16, batched).

    Input:
        rgb: (B, H, W, 6) uint8 or float, 6 channels = base[3] || hand[3]
    Output (all fp32):
        base_tokens: (B, N_v, d)
        hand_tokens: (B, N_v, d)
        cls_base, cls_hand: (B, d)
    where N_v = (INPUT_SIZE/14)^2 = 81, d = 384 (DINOv2-S/14).
    """

    def __init__(self, model_name: str = "facebook/dinov2-small",
                 input_size: int = INPUT_SIZE,
                 dtype: torch.dtype = torch.float16):
        super().__init__()
        backbone = AutoModel.from_pretrained(model_name)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone = backbone.to(dtype=dtype)
        self.backbone = backbone
        self.dtype = dtype
        self.patch_size = backbone.config.patch_size
        self.dim = backbone.config.hidden_size
        self.input_size = input_size
        assert input_size % self.patch_size == 0, (
            f"input_size {input_size} must be multiple of patch_size {self.patch_size}")
        # mean/std stored in fp32; cast on use
        self.register_buffer("mean", torch.tensor(DINOV2_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(DINOV2_STD).view(1, 3, 1, 1))

    def _preprocess_pair(self, rgb6: torch.Tensor) -> torch.Tensor:
        """rgb6: (B, H, W, 6) uint8/float -> (2B, 3, S, S) fp16, normalized.
        Order along batch: first B = base, last B = hand.
        """
        x = rgb6
        if x.dtype == torch.uint8:
            x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()        # (B, 6, H, W)
        base = x[:, 0:3]
        hand = x[:, 3:6]
        x = torch.cat([base, hand], dim=0)            # (2B, 3, H, W)
        S = self.input_size
        if x.shape[-2:] != (S, S):
            x = F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return x.to(dtype=self.dtype)

    @torch.no_grad()
    def forward(self, rgb6: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = rgb6.shape[0]
        x = self._preprocess_pair(rgb6)               # (2B, 3, S, S) fp16
        out = self.backbone(pixel_values=x, interpolate_pos_encoding=True)
        h = out.last_hidden_state                     # (2B, 1+N, d)
        cls = h[:, 0, :].float()                      # (2B, d) fp32
        tok = h[:, 1:, :].float()                     # (2B, N, d) fp32
        cls_b, cls_h = cls[:B], cls[B:]
        tok_b, tok_h = tok[:B], tok[B:]
        return tok_b, tok_h, cls_b, cls_h

    @property
    def num_patches_per_view(self) -> int:
        return (self.input_size // self.patch_size) ** 2

    @property
    def grid(self) -> tuple[int, int]:
        g = self.input_size // self.patch_size
        return (g, g)
