"""
Frozen CLIP-ViT-B/16 wrapper for dual-camera batched inference (A_backbone
ablation).

Mirrors FrozenDualDinoV2's interface (forward signature, output shapes by
view, num_patches_per_view, grid). Differences:
  - hidden dim 768 (vs DINOv2-S's 384)
  - patch_size 16 (vs DINOv2-S's 14)
  - native 224x224 input; we use 128x128 with `interpolate_pos_encoding=True`
    so that 128/16 = 8 → 8x8 = 64 patches per view (matches DINOv2's order
    of magnitude, ~81). fp16 inference for speed.

The saliency head consuming this backbone must be re-trained at the new
patch dim (768) and grid (8x8); see analysis/ebm/path_a_train_head_v3_clip.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel

CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
INPUT_SIZE = 128   # 8 × 16, multiple of CLIP patch_size 16


class FrozenDualClip(nn.Module):
    """Frozen CLIP-ViT-B/16 encoder for two cameras (fp16, batched)."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch16",
                 input_size: int = INPUT_SIZE,
                 dtype: torch.dtype = torch.float16):
        super().__init__()
        # Force safetensors (avoids CVE-2025-32434 torch.load issue on torch <2.6).
        backbone = CLIPVisionModel.from_pretrained(model_name, use_safetensors=True)
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
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(CLIP_STD).view(1, 3, 1, 1))

    def _preprocess_pair(self, rgb6: torch.Tensor) -> torch.Tensor:
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
        x = self._preprocess_pair(rgb6)
        out = self.backbone(pixel_values=x, interpolate_pos_encoding=True)
        h = out.last_hidden_state                     # (2B, 1+N, d)
        cls = h[:, 0, :].float()
        tok = h[:, 1:, :].float()
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
