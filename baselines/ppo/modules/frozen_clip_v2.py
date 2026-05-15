"""
Frozen CLIP-ViT-B/16 wrapper V2 (higher input resolution).

Difference from V1 (frozen_clip.py):
  - input_size: 128 -> 192 (12x16 = 192). Patch grid 12×12 = 144 per view,
    closer to DINOv2-S's 9×9=81 in spatial granularity (smaller object
    visibility improves).
  - Saliency head must be re-trained at this resolution; see
    analysis/ebm/path_a_train_head_v3_clip_v2.py.
"""
from __future__ import annotations
import torch
from .frozen_clip import FrozenDualClip


class FrozenDualClipV2(FrozenDualClip):
    """Identical to FrozenDualClip but defaults to input_size=192."""
    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch16",
                 input_size: int = 192,
                 dtype: torch.dtype = torch.float16):
        super().__init__(model_name=model_name, input_size=input_size, dtype=dtype)
