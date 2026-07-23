"""DINOv2 CLS + mean-pooled-patch encoder (bandwidth-control baseline).

Purpose: isolate representational BANDWIDTH from the memory MECHANISM. STRM's
buffer consumes the flat, view-untagged set of 162 patch tokens; this encoder
hands a recurrent baseline a pooled summary of exactly that same set, on top of
everything the standard baseline already receives.

Output = [ CLS summary (256) | joints (25) | color map (486, if color_aug)
           | mean over ALL 162 patch tokens (384) ]

Design choices (mirroring the fairness rules used elsewhere):
  * POOL ACROSS BOTH CAMERAS (one mean over the flat 162-token set), matching
    the view-untagged token set STRM's write/route path operates on — the
    ablation stays single-variable (pooled vs routed). Per-view GLOBAL
    information is already present via the CLS features, for the baseline and
    for STRM alike (both keep cls_base/cls_hand in fixed slots).
  * RAW CONCAT, no learned projection on the pooled vector — same no-bottleneck
    principle as the color map: nothing we train may throttle the baseline.
Canonical `dinov2_simple_encoder.py` is untouched (variant file per repo
convention).
"""
from __future__ import annotations

import torch

from .dinov2_simple_encoder import DinoV2SimpleEncoder


class DinoV2PoolCLSEncoder(DinoV2SimpleEncoder):
    """DinoV2SimpleEncoder + mean-pooled patch tokens appended (no bottleneck)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_features += self.vit.dim   # + pooled-patch vector (384)

    def forward(self, observations: dict) -> torch.Tensor:
        rgb = observations["rgb"]
        joints = (observations["joints"].float()
                  if observations["joints"].dtype != torch.float32
                  else observations["joints"])
        tok_b, tok_h, cls_b, cls_h = self.vit(rgb)
        cls_sum = self.curr_summary(torch.cat([cls_b, cls_h], dim=-1))
        # one mean over the flat 162-token set (both views, no camera tag) —
        # the same untagged token pool STRM's buffer/routing consumes.
        pooled = torch.cat([tok_b, tok_h], dim=1).mean(dim=1)
        feats = [cls_sum, joints]
        if self.color_aug:
            feats.append(self._rgb_per_patch_flat(rgb).to(cls_sum.dtype))
        feats.append(pooled.to(cls_sum.dtype))
        return torch.cat(feats, dim=-1)
