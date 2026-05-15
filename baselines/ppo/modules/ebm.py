"""
EBMMemoryModule: composition of frozen DINOv2 + frozen saliency head +
episodic buffer + learned cross-attention reader.

Public interface (one PPO env step):

  s_t, info = ebm.step(rgb6, proprio, t)
      rgb6:    (B, H, W, 6) uint8/float, dual-camera channel-concat
      proprio: (B, d_proprio)
      t:       int, current global step (used for timestamp + age decay)
      Returns:
        s_t :   (B, d_state)  fused state for actor/critic
        info:   dict with debug tensors (saliency, n_pushed, ...)

  ebm.reset(env_done_mask)         called by PPO at episode boundaries.

  ebm.snapshot() / ebm.restore()   for buffer caching across PPO update.

Architecturally:
  curr_summary = MLP_curr(concat(cls_base, cls_hand))
  query_in     = concat(proprio, curr_summary)
  buffer       <- top-K patches by saliency_head(patches) (sigmoid > 0.5)
                  filtered by novelty against existing buffer
  retrieved    = reader(query_in, buffer.get())
  s_t          = MLP_fuse(concat(proprio, curr_summary, retrieved))
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .frozen_vit import FrozenDualDinoV2
from .frozen_clip import FrozenDualClip
from .frozen_clip_v2 import FrozenDualClipV2
from .saliency_head import load_saliency_head
from .episodic_buffer import EpisodicBuffer
from .memory_reader import MemoryReader


class EBMMemoryModule(nn.Module):
    def __init__(
        self,
        num_envs: int,
        proprio_dim: int = 25,
        saliency_ckpt: str | Path | None = None,
        vit_name: str = "facebook/dinov2-small",
        vit_backbone: str = "dinov2",   # "dinov2" or "clip"
        L: int = 64,
        K: int = 8,
        d_proj: int = 256,
        d_state: int = 256,
        novelty_thresh: float = 0.95,
        tau_age: float = 30.0,
        device: str | torch.device = "cuda",
        no_saliency: bool = False,
    ):
        super().__init__()
        self.num_envs = num_envs
        self.K = K
        self.L = L
        self.d_state = d_state
        self.device = torch.device(device)
        self.no_saliency = no_saliency
        self.vit_backbone = vit_backbone

        # --- frozen perception ----------------------------------------------
        if vit_backbone == "dinov2":
            self.vit = FrozenDualDinoV2(vit_name)
        elif vit_backbone == "clip":
            self.vit = FrozenDualClip()
        elif vit_backbone == "clip_v2":
            self.vit = FrozenDualClipV2()
        else:
            raise ValueError(f"unknown vit_backbone: {vit_backbone}")
        self.vit.to(self.device)
        d_vit = self.vit.dim
        n_patch_per_view = self.vit.num_patches_per_view

        # --- frozen saliency head ------------------------------------------
        if saliency_ckpt is None:
            raise ValueError("saliency_ckpt is required (Path A v2 head); "
                             "if running ablation 'no head', construct EBMMemoryModule "
                             "with NoSaliencyHead manually.")
        self.saliency_head, xy_concat = load_saliency_head(
            saliency_ckpt, device=self.device, freeze=True)
        self.register_buffer("xy_concat", xy_concat)
        # head outputs logits over 2*N_v patches (base then hand)
        self.n_total_patches = 2 * n_patch_per_view

        # --- learned: current-frame summary head ---------------------------
        self.curr_summary = nn.Sequential(
            nn.Linear(2 * d_vit, 256), nn.GELU(),
            nn.Linear(256, 128),
        )

        # --- buffer (per-env, non-trainable state) -------------------------
        self.buffer = EpisodicBuffer(
            num_envs=num_envs, L=L, d=d_vit, device=self.device,
            tau_age=tau_age, novelty_thresh=novelty_thresh,
        )

        # --- learned: memory reader (cross-attn) ---------------------------
        self.reader = MemoryReader(
            d_query=proprio_dim + 128, d_buffer=d_vit, d_proj=d_proj,
        )

        # --- learned: fuse layer -------------------------------------------
        self.fuse = nn.Sequential(
            nn.Linear(proprio_dim + 128 + d_proj, 256), nn.GELU(),
            nn.Linear(256, d_state),
        )

    # ─── per-step forward ───────────────────────────────────────────────────

    def step(self, rgb6: torch.Tensor, proprio: torch.Tensor, t: int) -> tuple[torch.Tensor, dict]:
        """One env step. Returns fused state s_t and debug info."""
        # 1. ViT (no_grad)
        tok_b, tok_h, cls_b, cls_h = self.vit(rgb6)
        tokens_all = torch.cat([tok_b, tok_h], dim=1)        # (B, 2*N_v, d_vit)

        # 2. Saliency head (frozen, no_grad) OR ablation: skip head entirely
        with torch.no_grad():
            B = tokens_all.size(0)
            if self.no_saliency:
                # A3 ablation: every patch is a candidate with uniform "saliency"
                # of 1.0; novelty filter + L-cap will decide what survives.
                # We still cap to K patches per push to keep buffer write cost bounded;
                # randomly sample K out of 2*N_v each step.
                # NOTE: keeping uniform 1.0 means the saliency-bias term in reader
                # gets log(1) = 0, effectively neutralizing it.
                idx = torch.randint(0, tokens_all.size(1), (B, self.K), device=tokens_all.device)
                cand_feats = tokens_all.gather(
                    1, idx.unsqueeze(-1).expand(B, self.K, tokens_all.size(-1)))
                cand_sal = torch.ones(B, self.K, device=tokens_all.device)
                probs = torch.zeros(B, tokens_all.size(1), device=tokens_all.device)  # for info dict
            else:
                logits = self.saliency_head(tokens_all, self.xy_concat)  # (B, 2*N_v)
                probs = torch.sigmoid(logits)
                topk_val, topk_idx = probs.topk(self.K, dim=-1)            # (B, K)
                cand_feats = tokens_all.gather(
                    1, topk_idx.unsqueeze(-1).expand(B, self.K, tokens_all.size(-1))
                )                                                          # (B, K, d_vit)
                cand_sal = topk_val * (topk_val > 0.5).float()              # zero-out below thresh

        # 3. Push to buffer (under no_grad — buffer push is not differentiable)
        with torch.no_grad():
            self.buffer.push_batch(cand_feats, cand_sal, t_now=t)
        n_pushed = (cand_sal > 0).sum(dim=-1)                           # (B,)

        # 4. Build query
        curr = self.curr_summary(torch.cat([cls_b, cls_h], dim=-1))    # (B, 128)
        query_in = torch.cat([proprio, curr], dim=-1)                  # (B, p+128)

        # 5. Cross-attn read
        feats, mask, ts, sal = self.buffer.get()
        retrieved = self.reader(query_in, feats, mask, ts, sal)        # (B, d_proj)

        # 6. Fuse
        s_t = self.fuse(torch.cat([proprio, curr, retrieved], dim=-1))  # (B, d_state)

        return s_t, {
            "n_pushed":    n_pushed,
            "buffer_used": self.buffer.used.clone(),
            "saliency_max": probs.max(-1).values,
            "cls_base":    cls_b.detach(),    # (B, d_vit)
            "cls_hand":    cls_h.detach(),    # (B, d_vit)
        }

    # ─── replay (used during PPO update) ───────────────────────────────────

    def replay(self, cached_buffer: dict, cls_base: torch.Tensor,
               cls_hand: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """
        Differentiable replay through curr_summary + reader + fuse using the
        cached buffer state. ViT and saliency head are NOT invoked here —
        their outputs are already in cached_buffer.features and cls_*.

        Inputs:
          cached_buffer: dict with keys
              features   (B, L, d_vit)
              used       (B,)
              timestamps (B, L)
              saliency   (B, L)
          cls_base, cls_hand: (B, d_vit)  -- detached cls tokens from rollout
          proprio:           (B, p)

        Returns: s_t (B, d_state) — graph kept through curr_summary, reader, fuse.
        """
        L = cached_buffer["features"].shape[1]
        idx = torch.arange(L, device=cached_buffer["features"].device).unsqueeze(0)
        mask = idx < cached_buffer["used"].unsqueeze(1)

        curr = self.curr_summary(torch.cat([cls_base, cls_hand], dim=-1))
        query_in = torch.cat([proprio, curr], dim=-1)
        retrieved = self.reader(
            query_in,
            cached_buffer["features"],
            mask,
            cached_buffer["timestamps"],
            cached_buffer["saliency"],
        )
        s_t = self.fuse(torch.cat([proprio, curr, retrieved], dim=-1))
        return s_t

    # ─── episode boundary --------------------------------------------------

    def reset(self, env_done_mask: torch.Tensor) -> None:
        self.buffer.reset(env_done_mask)

    # ─── PPO update support: snapshot/restore buffer state ─────────────────

    def snapshot(self) -> dict:
        return self.buffer.state_dict_buffer()

    def restore(self, sd: dict) -> None:
        self.buffer.load_state_dict_buffer(sd)
